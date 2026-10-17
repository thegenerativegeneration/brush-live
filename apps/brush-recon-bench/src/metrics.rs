use brush_recon::{BackendKind, MetricDepth, ReconError};
use brush_recon::{ReconFrame, ReconstructionBackend};
use brush_triangle_render::TriangleSeed;
use glam::UVec2;
use image::RgbImage;
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RgbMetric {
    pub mae: f64,
    pub mse: f64,
    pub psnr: Option<f64>,
    pub exact_match: bool,
    pub channels: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RgbSums {
    pub absolute: f64,
    pub squared: f64,
    pub channels: u64,
}
impl RgbSums {
    pub fn add(
        &mut self,
        prediction: &[f32],
        target: &[u8],
        roi: Option<&[u8]>,
    ) -> Result<(), ReconError> {
        if prediction.len() != target.len()
            || prediction.len() % 3 != 0
            || prediction.iter().any(|v| !v.is_finite())
            || roi.is_some_and(|m| {
                m.len() != target.len() / 3 || m.iter().any(|v| *v != 0 && *v != 255)
            })
        {
            return Err(ReconError::input(
                "invalid RGB evaluation shape, mask or nonfinite prediction",
            ));
        }
        for (i, (&p, &t)) in prediction.iter().zip(target).enumerate() {
            if roi.is_some_and(|m| m[i / 3] == 0) {
                continue;
            }
            let error = f64::from(p) - f64::from(t as f32 / 255.);
            self.absolute += error.abs();
            self.squared += error * error;
            self.channels += 1;
        }
        Ok(())
    }
    pub fn merge(&mut self, other: Self) {
        self.absolute += other.absolute;
        self.squared += other.squared;
        self.channels += other.channels;
    }
    pub fn finish(&self) -> Option<RgbMetric> {
        if self.channels == 0 {
            return None;
        }
        let mse = self.squared / self.channels as f64;
        Some(RgbMetric {
            mae: self.absolute / self.channels as f64,
            mse,
            psnr: if mse == 0. {
                None
            } else {
                Some(-10. * mse.log10())
            },
            exact_match: mse == 0.,
            channels: self.channels,
        })
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DepthMetric {
    pub mae_m: f64,
    pub valid_targets: u64,
    pub available_predictions: u64,
    pub high_alpha_predictions: u64,
    pub available_fraction: f64,
    pub high_alpha_fraction: f64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthSums {
    pub error_m: f64,
    pub targets: u64,
    pub available: u64,
    pub high_alpha: u64,
}
impl DepthSums {
    pub fn add(
        &mut self,
        prediction: &[f32],
        alpha: &[f32],
        target: &MetricDepth,
        roi: Option<&[u8]>,
    ) -> Result<(), ReconError> {
        target.validate()?;
        let n = target.values_m.len();
        if prediction.len() != n
            || alpha.len() != n
            || roi.is_some_and(|m| m.len() != n || m.iter().any(|v| *v != 0 && *v != 255))
        {
            return Err(ReconError::input(
                "native depth evaluation shape/mask mismatch",
            ));
        }
        for i in 0..n {
            let z = target.values_m[i];
            let confidence = target.confidence.as_ref().map_or(0, |c| c[i]);
            if confidence != 2 || !(0.1..=8.).contains(&z) || roi.is_some_and(|m| m[i] == 0) {
                continue;
            }
            self.targets += 1;
            let available = alpha[i].is_finite()
                && alpha[i] > 1e-6
                && prediction[i].is_finite()
                && prediction[i] > 0.;
            if available {
                self.available += 1;
                self.error_m += f64::from((prediction[i] - z).abs());
            } else {
                self.error_m += 8.;
            }
            if alpha[i].is_finite() && alpha[i] >= 0.9 {
                self.high_alpha += 1;
            }
        }
        Ok(())
    }
    pub fn merge(&mut self, other: Self) {
        self.error_m += other.error_m;
        self.targets += other.targets;
        self.available += other.available;
        self.high_alpha += other.high_alpha;
    }
    pub fn finish(&self) -> Option<DepthMetric> {
        if self.targets == 0 {
            None
        } else {
            Some(DepthMetric {
                mae_m: self.error_m / self.targets as f64,
                valid_targets: self.targets,
                available_predictions: self.available,
                high_alpha_predictions: self.high_alpha,
                available_fraction: self.available as f64 / self.targets as f64,
                high_alpha_fraction: self.high_alpha as f64 / self.targets as f64,
            })
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct AggregateMetrics {
    pub full_rgb: Option<RgbMetric>,
    pub roi_rgb: Option<RgbMetric>,
    pub depth: Option<DepthMetric>,
    pub roi_depth: Option<DepthMetric>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ResearchGate {
    pub positive: Option<bool>,
    pub roi_rgb_improved: bool,
    pub depth_not_worse: Option<bool>,
    pub notes: Vec<String>,
}
pub fn research_gate(
    kind: BackendKind,
    initial: &AggregateMetrics,
    final_metrics: &AggregateMetrics,
) -> ResearchGate {
    let rgb = initial
        .roi_rgb
        .zip(final_metrics.roi_rgb)
        .is_some_and(|(a, b)| {
            if a.mae <= 1e-8 {
                b.mae <= 1e-8
            } else {
                b.mae <= a.mae * 0.9
            }
        });
    let depth = initial.depth.zip(final_metrics.depth).map(|(a, b)| {
        if a.mae_m <= 1e-6 {
            b.mae_m <= 1e-6
        } else {
            b.mae_m <= a.mae_m * 1.1
        }
    });
    let mut notes = vec![];
    if kind == BackendKind::Gaussian {
        notes.push(
            "Controlled RGB-only baseline: no native-depth output or triangle research gate".into(),
        );
        return ResearchGate {
            positive: None,
            roi_rgb_improved: rgb,
            depth_not_worse: None,
            notes,
        };
    }
    if !rgb {
        notes.push(
            "Held-out immutable-ROI RGB MAE failed the 10% improvement/near-zero tolerance gate"
                .into(),
        );
    }
    if depth != Some(true) {
        notes.push("Full native-grid fixed-denominator depth error failed the 10% non-worsening gate or had no valid targets".into());
    }
    ResearchGate {
        positive: Some(rgb && depth == Some(true)),
        roi_rgb_improved: rgb,
        depth_not_worse: depth,
        notes,
    }
}

#[derive(Debug, Serialize)]
pub struct ViewEvaluation {
    pub frame_id: u64,
    pub full_rgb: Option<RgbMetric>,
    pub roi_rgb: Option<RgbMetric>,
    pub depth: Option<DepthMetric>,
    pub roi_depth: Option<DepthMetric>,
}
pub struct DepthImages {
    pub size: UVec2,
    pub prediction: Vec<f32>,
    pub alpha: Vec<f32>,
    pub target: Arc<MetricDepth>,
    pub roi: Vec<u8>,
}
pub struct EvalImages {
    pub frame_id: u64,
    pub size: UVec2,
    pub rgb: Vec<f32>,
    pub target: Arc<RgbImage>,
    pub roi: Arc<Vec<u8>>,
    pub depth: Option<DepthImages>,
}
#[derive(Default, Serialize)]
pub struct Evaluation {
    pub aggregate: AggregateMetrics,
    pub views: Vec<ViewEvaluation>,
    #[serde(skip)]
    pub images: Vec<EvalImages>,
}
async fn read<const D: usize>(t: burn::tensor::Tensor<D>) -> Result<Vec<f32>, ReconError> {
    t.into_data_async()
        .await
        .map_err(|e| ReconError::Gpu(format!("evaluation readback: {e:?}")))?
        .try_to_vec()
        .map_err(|e| ReconError::Gpu(format!("evaluation values: {e:?}")))
}

pub async fn evaluate(
    backend: &impl ReconstructionBackend,
    views: &[Arc<ReconFrame>],
    initial_seed: &TriangleSeed,
) -> Result<Evaluation, ReconError> {
    let mut result = Evaluation::default();
    let mut full = RgbSums::default();
    let mut roi = RgbSums::default();
    let mut depth_all = DepthSums::default();
    let mut depth_roi = DepthSums::default();
    let mut seen = std::collections::BTreeSet::new();
    for frame in views {
        frame.validate()?;
        if !seen.insert(frame.id) {
            return Err(ReconError::input("duplicate evaluation frame ID"));
        }
        let size = UVec2::new(frame.rgb.width(), frame.rgb.height());
        let request = brush_recon::RenderRequest {
            camera: frame.camera,
            size,
            background: glam::Vec3::ZERO,
            depth: false,
            normals: false,
        };
        let output = backend.render(&request).await?;
        output.validate(&request)?;
        let rgb = read(output.rgb).await?;
        let alpha = read(output.alpha).await?;
        if alpha.iter().any(|v| !v.is_finite()) {
            return Err(ReconError::ModelState("nonfinite evaluated alpha".into()));
        }
        let mut f = RgbSums::default();
        let mut r = RgbSums::default();
        f.add(&rgb, frame.rgb.as_raw(), None)?;
        r.add(&rgb, frame.rgb.as_raw(), Some(&frame.roi))?;
        full.merge(f);
        roi.merge(r);
        let mut native = None;
        let mut d = DepthSums::default();
        let mut dr = DepthSums::default();
        if backend.capabilities().depth {
            if let Some(target) = &frame.lidar {
                let request = brush_recon::RenderRequest {
                    camera: frame.camera,
                    size: target.size,
                    background: glam::Vec3::ZERO,
                    depth: true,
                    normals: false,
                };
                let output = backend.render(&request).await?;
                output.validate(&request)?;
                let prediction = read(
                    output
                        .depth_m
                        .ok_or(ReconError::UnsupportedOutput("native evaluation depth"))?,
                )
                .await?;
                let alpha = read(output.alpha).await?;
                if prediction.iter().chain(&alpha).any(|v| !v.is_finite()) {
                    return Err(ReconError::ModelState(
                        "nonfinite native-depth render".into(),
                    ));
                }
                let mask = brush_recon::capture::mask::rasterize_roi(
                    &initial_seed.positions,
                    &initial_seed.faces,
                    &frame.camera,
                    target.size,
                    0.05,
                )?;
                d.add(&prediction, &alpha, target, None)?;
                dr.add(&prediction, &alpha, target, Some(&mask))?;
                depth_all.merge(d);
                depth_roi.merge(dr);
                native = Some(DepthImages {
                    size: target.size,
                    prediction,
                    alpha,
                    target: target.clone(),
                    roi: mask,
                });
            }
        }
        backend.synchronize().await?;
        result.views.push(ViewEvaluation {
            frame_id: frame.id,
            full_rgb: f.finish(),
            roi_rgb: r.finish(),
            depth: d.finish(),
            roi_depth: dr.finish(),
        });
        result.images.push(EvalImages {
            frame_id: frame.id,
            size,
            rgb,
            target: frame.rgb.clone(),
            roi: frame.roi.clone(),
            depth: native,
        });
    }
    result.aggregate = AggregateMetrics {
        full_rgb: full.finish(),
        roi_rgb: roi.finish(),
        depth: depth_all.finish(),
        roi_depth: depth_roi.finish(),
    };
    Ok(result)
}
