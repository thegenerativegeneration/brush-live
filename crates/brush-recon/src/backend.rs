use crate::{ReconError, ReconFrame};
use brush_render::camera::Camera;
use burn::tensor::Tensor;
use glam::{UVec2, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Gaussian,
    Triangle,
}

#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    pub depth: bool,
    pub normals: bool,
}

#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub camera: Camera,
    pub size: UVec2,
    pub background: Vec3,
    pub depth: bool,
    pub normals: bool,
}

impl RenderRequest {
    pub fn validate(&self, capabilities: Capabilities) -> Result<(), ReconError> {
        crate::frame::validate_camera(&self.camera)?;
        crate::frame::pixel_count(self.size)?;
        if !self.background.is_finite() {
            return Err(ReconError::input("background must be finite"));
        }
        if self.depth && !capabilities.depth {
            return Err(ReconError::UnsupportedOutput("depth"));
        }
        if self.normals && !capabilities.normals {
            return Err(ReconError::UnsupportedOutput("normals"));
        }
        Ok(())
    }
}

pub struct RenderResult {
    pub rgb: Tensor<3>,
    pub alpha: Tensor<2>,
    pub depth_m: Option<Tensor<2>>,
    pub normals: Option<Tensor<3>>,
}

impl RenderResult {
    pub fn validate(&self, request: &RenderRequest) -> Result<(), ReconError> {
        request.validate(Capabilities {
            depth: true,
            normals: true,
        })?;
        let [h, w] = [request.size.y as usize, request.size.x as usize];
        if self.rgb.dims() != [h, w, 3]
            || self.alpha.dims() != [h, w]
            || self.depth_m.is_some() != request.depth
            || self.normals.is_some() != request.normals
            || self.depth_m.as_ref().is_some_and(|d| d.dims() != [h, w])
            || self.normals.as_ref().is_some_and(|n| n.dims() != [h, w, 3])
        {
            return Err(ReconError::input(
                "render channel presence/shapes do not match request",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StepStats {
    pub iteration: u64,
    /// Masked training RGB absolute error divided by 3*H*W (not ROI count).
    pub rgb_l1: f32,
    pub depth_l1_m: Option<f32>,
    pub total_loss: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct StepContext {
    pub iteration: u64,
    pub progress: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ModelStats {
    pub vertices: u32,
    pub triangles: u32,
    pub gaussians: u32,
    pub renderer_peak_bytes: u64,
}

pub struct ExperimentalState {
    pub kind: BackendKind,
    pub format_version: u32,
    pub payload: Vec<u8>,
}

// Generic async runner, intentionally not dyn-compatible or Send-constrained.
#[allow(async_fn_in_trait)]
pub trait ReconstructionBackend {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> Capabilities;
    async fn initialize(&mut self, train: &[Arc<ReconFrame>]) -> Result<(), ReconError>;
    async fn train_step(
        &mut self,
        frame: &ReconFrame,
        context: StepContext,
    ) -> Result<StepStats, ReconError>;
    async fn render(&self, request: &RenderRequest) -> Result<RenderResult, ReconError>;
    async fn synchronize(&self) -> Result<(), ReconError>;
    async fn export_state(&self) -> Result<ExperimentalState, ReconError>;
    fn stats(&self) -> ModelStats;
}
