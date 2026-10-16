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
    /// `[n]` whether a training step ever updated the row; empty when not known (the trainer then counts the rows as
    /// trained).
    pub trained: Vec<bool>,
    /// The rows' Adam moments; `None` when not known (the trainer then starts them at zero).
    pub moments: Option<RowMoments>,
}

/// The Adam moments of splat rows on the host, per parameter group in the trainer's layout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowMoments {
    /// `[n·10]` first moment of the transforms row (means 3, rotations 4, log-scales 3).
    pub transforms_m1: Vec<f32>,
    /// `[n·10]` second moment of the transforms row.
    pub transforms_m2: Vec<f32>,
    /// `[n·coeffs·3]` first moment of the SH coefficients, layout like `SplatRows::sh_coeffs`.
    pub sh_m1: Vec<f32>,
    /// `[n]` second moment of the SH coefficients, one per row (the trainer reduces it over the row).
    pub sh_m2: Vec<f32>,
    /// `[n]` first moment of the raw opacity.
    pub opacity_m1: Vec<f32>,
    /// `[n]` second moment of the raw opacity.
    pub opacity_m2: Vec<f32>,
}

impl RowMoments {
    /// Zero moments for `n` rows with `coeffs` SH coefficients per channel.
    pub fn zeros(n: usize, coeffs: usize) -> Self {
        Self {
            transforms_m1: vec![0.0; n * 10],
            transforms_m2: vec![0.0; n * 10],
            sh_m1: vec![0.0; n * coeffs * 3],
            sh_m2: vec![0.0; n],
            opacity_m1: vec![0.0; n],
            opacity_m2: vec![0.0; n],
        }
    }

    /// Per column: name, values and values per row for `coeffs` SH coefficients per channel, in field order.
    pub fn columns(&self, coeffs: usize) -> [(&'static str, &[f32], usize); 6] {
        [
            ("transforms_m1", &self.transforms_m1, 10),
            ("transforms_m2", &self.transforms_m2, 10),
            ("sh_m1", &self.sh_m1, coeffs * 3),
            ("sh_m2", &self.sh_m2, 1),
            ("opacity_m1", &self.opacity_m1, 1),
            ("opacity_m2", &self.opacity_m2, 1),
        ]
    }

    fn select(&self, coeffs: usize, rows: &[usize]) -> Self {
        Self {
            transforms_m1: pick(&self.transforms_m1, 10, rows),
            transforms_m2: pick(&self.transforms_m2, 10, rows),
            sh_m1: pick(&self.sh_m1, coeffs * 3, rows),
            sh_m2: pick(&self.sh_m2, 1, rows),
            opacity_m1: pick(&self.opacity_m1, 1, rows),
            opacity_m2: pick(&self.opacity_m2, 1, rows),
        }
    }

    fn extend(&mut self, other: &RowMoments) {
        self.transforms_m1.extend_from_slice(&other.transforms_m1);
        self.transforms_m2.extend_from_slice(&other.transforms_m2);
        self.sh_m1.extend_from_slice(&other.sh_m1);
        self.sh_m2.extend_from_slice(&other.sh_m2);
        self.opacity_m1.extend_from_slice(&other.opacity_m1);
        self.opacity_m2.extend_from_slice(&other.opacity_m2);
    }
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

    /// `Err` naming the first column (in field order) whose length does not fit `len()` rows: `len·3` means and
    /// log-scales, `len·4` rotations, `len·coeffs·3` SH, a floor that is empty or `len`, `len` origins, ages and
    /// importances, trained flags that are empty or `len`, and with moments their columns
    /// ([`RowMoments::columns`]).
    pub fn check(&self) -> Result<(), String> {
        let n = self.len();
        let or_empty = |len: usize| if len == 0 { 0 } else { n };
        let mut columns = vec![
            ("means", self.means.len(), n * 3),
            ("rotations", self.rotations.len(), n * 4),
            ("log_scales", self.log_scales.len(), n * 3),
            ("sh_coeffs", self.sh_coeffs.len(), n * self.coeff_width()),
            ("min_scale", self.min_scale.len(), or_empty(self.min_scale.len())),
            ("origin", self.origin.len(), n),
            ("age", self.age.len(), n),
            ("importance", self.importance.len(), n),
            ("trained", self.trained.len(), or_empty(self.trained.len())),
        ];
        if let Some(m) = &self.moments {
            let coeffs = self.sh_coeffs_per_channel as usize;
            columns.extend(m.columns(coeffs).map(|(name, v, width)| (name, v.len(), n * width)));
        }
        match columns.iter().find(|(_, got, want)| got != want) {
            Some((name, got, want)) => Err(format!(
                "splat rows column {name} has {got} values, {n} rows need {want}"
            )),
            None => Ok(()),
        }
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
            trained: if self.trained.is_empty() {
                Vec::new()
            } else {
                pick(&self.trained, 1, rows)
            },
            moments: self
                .moments
                .as_ref()
                .map(|m| m.select(self.sh_coeffs_per_channel as usize, rows)),
        }
    }

    /// The rows of every part, in order. When some parts carry a floor and others do not, those get a zero floor,
    /// which leaves a splat as it is; likewise parts without trained flags count as trained and parts without moments
    /// get zero ones. Panics when non-empty parts differ in SH degree.
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
        let any_trained = parts.iter().any(|p| !p.trained.is_empty());
        let any_moments = parts.iter().any(|p| p.moments.is_some());
        let mut out = SplatRows {
            sh_coeffs_per_channel: coeffs,
            moments: any_moments.then(RowMoments::default),
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
            if any_trained {
                if p.trained.is_empty() {
                    out.trained.extend(std::iter::repeat_n(true, p.len()));
                } else {
                    out.trained.extend_from_slice(&p.trained);
                }
            }
            if let Some(m) = out.moments.as_mut() {
                match &p.moments {
                    Some(pm) => m.extend(pm),
                    None => m.extend(&RowMoments::zeros(p.len(), coeffs as usize)),
                }
            }
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

#[cfg(test)]
#[path = "rows_tests.rs"]
mod tests;
