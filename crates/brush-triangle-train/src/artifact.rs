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
    Ok(TriangleState {
        format_version: 1,
        positions,
        faces,
        rgb_logits,
        weight_logits,
        config: trainer.config.clone(),
        schedule: trainer.last_schedule,
        iterations: trainer.iterations,
    })
}
