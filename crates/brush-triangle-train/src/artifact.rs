//! Versioned experimental parameter snapshot, not a resumable production checkpoint.
use crate::{TrainConfig, TriangleTrainer, schedule::ScheduleState};
use brush_recon::ReconError;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleState {
    pub format_version: u32,
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub rgb_logits: Vec<[f32; 3]>,
    pub weight_logits: Vec<f32>,
    pub config: TrainConfig,
    pub schedule: ScheduleState,
    pub iterations: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renderer_last_render: Option<brush_triangle_render::model::RendererStats>,
}

impl TriangleState {
    pub fn validate(&self) -> Result<(), ReconError> {
        self.config.validate()?;
        if self.format_version != 1
            || self.positions.is_empty()
            || self.faces.is_empty()
            || self.positions.len() != self.rgb_logits.len()
            || self.positions.len() != self.weight_logits.len()
            || self
                .rgb_logits
                .iter()
                .flatten()
                .chain(self.weight_logits.iter())
                .any(|v| !v.is_finite())
            || !self.schedule.progress.is_finite()
            || !(0.0..=1.0).contains(&self.schedule.progress)
        {
            return Err(ReconError::input(
                "invalid experimental triangle state version, shape or values",
            ));
        }
        brush_triangle_render::reference::ReferenceConfig {
            sigma: self.schedule.sigma,
            weight_floor: self.schedule.weight_floor,
            near_m: self.config.render.near_m,
        }
        .validate()?;
        self.seed().validate()
    }
    pub fn seed(&self) -> brush_triangle_render::TriangleSeed {
        brush_triangle_render::TriangleSeed {
            positions: self.positions.clone(),
            faces: self.faces.clone(),
            colors: self
                .rgb_logits
                .iter()
                .map(|c| c.map(|v| 1. / (1. + (-v).exp())))
                .collect(),
        }
    }
}
pub fn decode_state(bytes: &[u8]) -> Result<TriangleState, ReconError> {
    let state: TriangleState = serde_json::from_slice(bytes)
        .map_err(|e| ReconError::input(format!("experimental triangle state: {e}")))?;
    state.validate()?;
    Ok(state)
}
pub fn mesh_ply(state: &TriangleState) -> Result<Vec<u8>, ReconError> {
    state.validate()?;
    let seed = state.seed();
    let mut text = format!(
        "ply\nformat ascii 1.0\ncomment Experimental RGB mesh: training coverage and weights are not encoded\ncomment Vertical axis: y\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        seed.positions.len(),
        seed.faces.len()
    );
    for (p, c) in seed.positions.iter().zip(&seed.colors) {
        let rgb = c.map(|v| (v.clamp(0., 1.) * 255.).round() as u8);
        text.push_str(&format!(
            "{} {} {} {} {} {}\n",
            p[0], p[1], p[2], rgb[0], rgb[1], rgb[2]
        ));
    }
    for f in &seed.faces {
        text.push_str(&format!("3 {} {} {}\n", f[0], f[1], f[2]));
    }
    Ok(text.into_bytes())
}
pub fn restore_model(
    state: &TriangleState,
    device: &burn::tensor::Device,
) -> Result<brush_triangle_render::TriangleModel, ReconError> {
    state.validate()?;
    let mut model = brush_triangle_render::TriangleModel::from_seed(&state.seed(), device)?;
    let n = state.positions.len();
    model.rgb_logits = Tensor::from_data(
        burn::tensor::TensorData::new(
            state
                .rgb_logits
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            [n, 3],
        ),
        device,
    );
    model.weight_logits = Tensor::from_data(
        burn::tensor::TensorData::new(state.weight_logits.clone(), [n]),
        device,
    );
    Ok(model)
}
async fn read<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, ReconError> {
    t.without_autodiff()
        .into_data_async()
        .await
        .map_err(|e| ReconError::Gpu(format!("snapshot readback: {e:?}")))?
        .try_to_vec()
        .map_err(|e| ReconError::Gpu(format!("snapshot values: {e:?}")))
}
pub async fn snapshot(trainer: &TriangleTrainer) -> Result<TriangleState, ReconError> {
    let positions = read(trainer.model.positions.clone())
        .await?
        .chunks_exact(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let rgb_logits = read(trainer.model.rgb_logits.clone())
        .await?
        .chunks_exact(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let weight_logits = read(trainer.model.weight_logits.clone()).await?;
    let faces = trainer
        .model
        .faces()
        .without_autodiff()
        .into_data_async()
        .await
        .map_err(|e| ReconError::Gpu(format!("snapshot indices: {e:?}")))?
        .try_to_vec::<i32>()
        .map_err(|e| ReconError::Gpu(format!("snapshot indices: {e:?}")))?;
    let faces = faces
        .chunks_exact(3)
        .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
        .collect();
    let state = TriangleState {
        format_version: 1,
        positions,
        faces,
        rgb_logits,
        weight_logits,
        config: trainer.config.clone(),
        schedule: trainer.last_schedule,
        iterations: trainer.iterations,
        renderer_last_render: Some(trainer.model.renderer_stats()),
    };
    state.validate()?;
    Ok(state)
}
