//! Splat rows on the host, in the trainer's raw layout, for parking the splats of an area on disk and restoring them.

use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::tensor::{Device, Tensor, TensorData};
use glam::Vec3;

/// Splat rows on the host in the trainer's raw layout (the Mip floor separate, not baked).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplatRows {
    /// `[n·3]`
    pub means: Vec<f32>,
    /// `[n·4]` w x y z as trained (unnormalised).
    pub rotations: Vec<f32>,
    /// `[n·3]`
    pub log_scales: Vec<f32>,
    /// `[n·coeffs·3]`, layout `[n, coeffs, 3]` like `Splats::sh_coeffs`.
    pub sh_coeffs: Vec<f32>,
    /// `[n]` logits.
    pub raw_opacities: Vec<f32>,
    /// `[n]` Mip floor; empty without one.
    pub min_scale: Vec<f32>,
    /// `[n]` seeding view slot.
    pub origin: Vec<Option<u32>>,
    /// `[n]` refines survived (0 without eviction state).
    pub age: Vec<f32>,
    /// `[n]` eviction importance (`+inf` unscored or without eviction state).
    pub importance: Vec<f32>,
    pub sh_coeffs_per_channel: u32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `rows` of `data`, each `width` values wide, in the order given.
fn pick<T: Copy>(data: &[T], width: usize, rows: &[usize]) -> Vec<T> {
    rows.iter()
        .flat_map(|&r| &data[r * width..(r + 1) * width])
        .copied()
        .collect()
}

impl SplatRows {
    pub fn len(&self) -> usize {
        self.raw_opacities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw_opacities.is_empty()
    }

    pub fn mean(&self, i: usize) -> Vec3 {
        Vec3::from_slice(&self.means[i * 3..i * 3 + 3])
    }

    fn coeff_width(&self) -> usize {
        self.sh_coeffs_per_channel as usize * 3
    }

    /// The rows at `rows`, in that order.
    pub fn select(&self, rows: &[usize]) -> SplatRows {
        SplatRows {
            means: pick(&self.means, 3, rows),
            rotations: pick(&self.rotations, 4, rows),
            log_scales: pick(&self.log_scales, 3, rows),
            sh_coeffs: pick(&self.sh_coeffs, self.coeff_width(), rows),
            raw_opacities: pick(&self.raw_opacities, 1, rows),
            min_scale: if self.min_scale.is_empty() {
                Vec::new()
            } else {
                pick(&self.min_scale, 1, rows)
            },
            origin: pick(&self.origin, 1, rows),
            age: pick(&self.age, 1, rows),
            importance: pick(&self.importance, 1, rows),
            sh_coeffs_per_channel: self.sh_coeffs_per_channel,
        }
    }

    /// The rows of every part, in order. When some parts carry a floor and others do not, those get a zero floor,
    /// which leaves a splat as it is. Panics when non-empty parts differ in SH degree.
    pub fn concat(parts: &[SplatRows]) -> SplatRows {
        let parts: Vec<&SplatRows> = parts.iter().filter(|p| !p.is_empty()).collect();
        let Some(first) = parts.first() else {
            return SplatRows::default();
        };
        let coeffs = first.sh_coeffs_per_channel;
        assert!(
            parts.iter().all(|p| p.sh_coeffs_per_channel == coeffs),
            "concatenated rows must share the SH degree"
        );
        let any_floor = parts.iter().any(|p| !p.min_scale.is_empty());
        let mut out = SplatRows {
            sh_coeffs_per_channel: coeffs,
            ..SplatRows::default()
        };
        for p in parts {
            out.means.extend_from_slice(&p.means);
            out.rotations.extend_from_slice(&p.rotations);
            out.log_scales.extend_from_slice(&p.log_scales);
            out.sh_coeffs.extend_from_slice(&p.sh_coeffs);
            out.raw_opacities.extend_from_slice(&p.raw_opacities);
            if any_floor {
                if p.min_scale.is_empty() {
                    out.min_scale.extend(std::iter::repeat_n(0.0, p.len()));
                } else {
                    out.min_scale.extend_from_slice(&p.min_scale);
                }
            }
            out.origin.extend_from_slice(&p.origin);
            out.age.extend_from_slice(&p.age);
            out.importance.extend_from_slice(&p.importance);
        }
        out
    }

    /// The rows with the Mip floor folded in, as `Splats::bake_min_scale` does on the GPU: scales become
    /// `sqrt(s² + f²)` and opacity is multiplied by `Π s / s'`, clamped to `[1e-6, 1 − 1e-6]`.
    pub fn baked(&self) -> SplatRows {
        if self.min_scale.is_empty() {
            return self.clone();
        }
        let mut out = self.clone();
        for i in 0..self.len() {
            let f2 = self.min_scale[i] * self.min_scale[i];
            let mut log_coef = 0.0;
            for a in 0..3 {
                let l = self.log_scales[i * 3 + a];
                let new = 0.5 * ((2.0 * l).exp() + f2).ln();
                log_coef += l - new;
                out.log_scales[i * 3 + a] = new;
            }
            let o = (sigmoid(self.raw_opacities[i]) * log_coef.exp()).clamp(1e-6, 1.0 - 1e-6);
            out.raw_opacities[i] = (o / (1.0 - o)).ln();
        }
        out.min_scale.clear();
        out
    }

    /// The rows as trainable splats on `device`, raw params without the floor.
    pub fn to_splats_raw(&self, device: &Device) -> Splats {
        let n = self.len();
        let t = |data: &[f32], width: usize| {
            Tensor::<2>::from_data(TensorData::new(data.to_vec(), [n, width]), device)
        };
        let coeffs = self.sh_coeffs_per_channel as usize;
        Splats::from_tensor_data(
            t(&self.means, 3),
            t(&self.rotations, 4),
            t(&self.log_scales, 3),
            Tensor::from_data(
                TensorData::new(self.sh_coeffs.clone(), [n, coeffs, 3]),
                device,
            ),
            Tensor::from_data(TensorData::new(self.raw_opacities.clone(), [n]), device),
            SplatRenderMode::Default,
        )
    }

    /// The rows as trainable splats on `device`, with the floor attached when they carry one.
    pub fn to_splats(&self, device: &Device) -> Splats {
        let splats = self.to_splats_raw(device);
        if self.min_scale.is_empty() {
            splats
        } else {
            splats.with_min_scale(Tensor::from_data(
                TensorData::new(self.min_scale.clone(), [self.len()]),
                device,
            ))
        }
    }
}
