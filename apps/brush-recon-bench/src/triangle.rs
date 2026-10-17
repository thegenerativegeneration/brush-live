use brush_recon::*;
use brush_triangle_render::{TriangleModel, TriangleSeed};
use brush_triangle_train::{TrainConfig, TriangleTrainer, artifact::snapshot};
use burn::tensor::Device;
use std::{collections::BTreeSet, sync::Arc};

pub struct TriangleBackend {
    seed: TriangleSeed,
    device: Device,
    config: TrainConfig,
    ids: BTreeSet<u64>,
    pub trainer: Option<TriangleTrainer>,
}
impl TriangleBackend {
    pub fn new(seed: TriangleSeed, device: Device, config: TrainConfig) -> Self {
        Self {
            seed,
            device,
            config,
            ids: BTreeSet::new(),
            trainer: None,
        }
    }
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }
}
impl ReconstructionBackend for TriangleBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Triangle
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            depth: true,
            normals: true,
        }
    }
    async fn initialize(&mut self, train: &[Arc<ReconFrame>]) -> Result<(), ReconError> {
        if self.trainer.is_some() {
            return Err(ReconError::input("triangle backend is already initialized"));
        }
        let ids = crate::training_ids(train)?;
        let model = TriangleModel::from_seed(&self.seed, &self.device)?;
        self.trainer = Some(TriangleTrainer::new(
            model,
            self.seed.clone(),
            self.config.clone(),
        )?);
        self.ids = ids;
        Ok(())
    }
    async fn train_step(
        &mut self,
        frame: &ReconFrame,
        context: StepContext,
    ) -> Result<StepStats, ReconError> {
        if !self.ids.contains(&frame.id) {
            return Err(ReconError::input(format!(
                "frame {} is not an initialized training ID",
                frame.id
            )));
        }
        self.trainer
            .as_mut()
            .ok_or_else(|| ReconError::ModelState("triangle backend is not initialized".into()))?
            .step(frame, context)
            .await
    }
    async fn render(&self, request: &RenderRequest) -> Result<RenderResult, ReconError> {
        request.validate(self.capabilities())?;
        self.trainer
            .as_ref()
            .ok_or_else(|| ReconError::ModelState("triangle backend is not initialized".into()))?
            .render(request)
            .await
    }
    async fn synchronize(&self) -> Result<(), ReconError> {
        self.device
            .sync()
            .map_err(|e| ReconError::Gpu(format!("device synchronization: {e:?}")))
    }
    async fn export_state(&self) -> Result<ExperimentalState, ReconError> {
        let state =
            snapshot(self.trainer.as_ref().ok_or_else(|| {
                ReconError::ModelState("triangle backend is not initialized".into())
            })?)
            .await?;
        let payload = serde_json::to_vec(&state)
            .map_err(|e| ReconError::ModelState(format!("triangle snapshot: {e}")))?;
        Ok(ExperimentalState {
            kind: BackendKind::Triangle,
            format_version: 1,
            payload,
        })
    }
    fn stats(&self) -> ModelStats {
        self.trainer
            .as_ref()
            .map_or_else(ModelStats::default, |t| ModelStats {
                vertices: t.model.num_vertices() as u32,
                triangles: t.model.num_faces() as u32,
                gaussians: 0,
                renderer_peak_bytes: t.model.renderer_stats().reserved_peak_bytes,
            })
    }
}
