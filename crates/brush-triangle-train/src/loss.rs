use brush_recon::{ReconError, ReconFrame, RenderRequest, RenderResult};
use burn::tensor::{Tensor, TensorData};
use glam::{UVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossConfig {
    pub depth_weight: f32,
    pub seed_weight: f32,
    pub missing_depth_m: f32,
    pub min_confidence: u8,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
}
impl Default for LossConfig {
    fn default() -> Self {
        Self {
            depth_weight: 0.05,
            seed_weight: 0.001,
            missing_depth_m: 8.,
            min_confidence: 2,
            min_depth_m: 0.1,
            max_depth_m: 8.,
        }
    }
}
pub struct LossTerms {
    pub total: Tensor<1>,
    pub rgb: Tensor<1>,
    pub depth_m: Option<Tensor<1>>,
    pub seed: Tensor<1>,
}

pub fn triangle_loss(
    render: &RenderResult,
    frame: &ReconFrame,
    seed_positions: &Tensor<2>,
    positions: &Tensor<2>,
    cfg: &LossConfig,
) -> Result<LossTerms, ReconError> {
    frame.validate()?;
    let w = frame.rgb.width() as usize;
    let h = frame.rgb.height() as usize;
    render.validate(&RenderRequest {
        camera: frame.camera,
        size: UVec2::new(w as u32, h as u32),
        background: Vec3::ZERO,
        depth: render.depth_m.is_some(),
        normals: render.normals.is_some(),
    })?;
    if positions.dims() != seed_positions.dims()
        || positions.dims()[0] == 0
        || positions.dims()[1] != 3
        || !cfg.depth_weight.is_finite()
        || cfg.depth_weight < 0.
        || !cfg.seed_weight.is_finite()
        || cfg.seed_weight < 0.
        || !cfg.missing_depth_m.is_finite()
        || cfg.missing_depth_m < 0.
        || cfg.min_confidence > 2
        || !cfg.min_depth_m.is_finite()
        || cfg.min_depth_m <= 0.
        || !cfg.max_depth_m.is_finite()
        || cfg.max_depth_m < cfg.min_depth_m
    {
        return Err(ReconError::input(
            "invalid loss configuration or seed/model shape",
        ));
    }
    if !frame.roi.iter().any(|v| *v == 255) {
        return Err(ReconError::input(format!(
            "frame {} has empty immutable training ROI",
            frame.id
        )));
    }
    let d = render.rgb.device();
    let target = Tensor::<3>::from_data(
        TensorData::new(
            frame
                .rgb
                .as_raw()
                .iter()
                .map(|v| *v as f32 / 255.)
                .collect::<Vec<_>>(),
            [h, w, 3],
        ),
        &d,
    );
    let roi = Tensor::<3>::from_data(
        TensorData::new(
            frame
                .roi
                .iter()
                .map(|v| *v as f32 / 255.)
                .collect::<Vec<_>>(),
            [h, w, 1],
        ),
        &d,
    );
    let rgb = ((render.rgb.clone() - target).abs() * roi).mean();
    let mut depth_m = None;
    if let Some(depth) = &frame.lidar {
        let mut targets = vec![0.; h * w];
        let mut valid = vec![0.; h * w];
        let mut count = 0_usize;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if frame.roi[i] == 0 {
                    continue;
                }
                let uv = Vec2::new((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                if let Some(z) = depth
                    .sample_uv(uv, cfg.min_confidence)
                    .filter(|v| *v >= cfg.min_depth_m && *v <= cfg.max_depth_m)
                {
                    targets[i] = z;
                    valid[i] = 1.;
                    count += 1;
                }
            }
        }
        if count > 0 {
            let prediction = render
                .depth_m
                .as_ref()
                .ok_or(ReconError::UnsupportedOutput(
                    "depth for supervised triangle loss",
                ))?
                .clone();
            let available = render
                .alpha
                .clone()
                .greater_elem(1e-6)
                .bool_and(prediction.clone().greater_elem(0.))
                .bool_and(prediction.clone().is_nan().bool_not())
                .bool_and(prediction.clone().is_inf().bool_not());
            let invalid = available.bool_not();
            let target = Tensor::<2>::from_data(TensorData::new(targets, [h, w]), &d);
            let mask = Tensor::<2>::from_data(TensorData::new(valid, [h, w]), &d);
            let safe = prediction.mask_fill(invalid.clone(), 0.);
            depth_m = Some(
                ((safe - target)
                    .abs()
                    .mask_fill(invalid, cfg.missing_depth_m)
                    * mask)
                    .sum()
                    / count as f32,
            );
        }
    }
    let origin = if positions.device().is_autodiff() {
        seed_positions.clone().autodiff()
    } else {
        seed_positions.clone().without_autodiff()
    };
    let seed = (positions.clone() - origin).powi_scalar(2).mean();
    let mut total = rgb.clone() + seed.clone() * cfg.seed_weight;
    if let Some(depth) = &depth_m {
        total = total + depth.clone() * cfg.depth_weight;
    }
    Ok(LossTerms {
        total,
        rgb,
        depth_m,
        seed,
    })
}
