use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_recon::*;
use brush_render::{
    AlphaMode,
    gaussian_splats::{SplatRenderMode, Splats, TextureMode, render_splats},
    sh::rgb_to_sh,
};
use brush_serde::SplatData;
use brush_train::{
    config::TrainConfig,
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_triangle_render::TriangleSeed;
use burn::tensor::Device;
use burn::{module::Module, tensor::s};
use clap::Parser;
use glam::Vec3;
use image::{DynamicImage, Rgba, RgbaImage};
use std::{collections::BTreeSet, sync::Arc};

pub fn mask_scene_batch(frame: &ReconFrame) -> Result<SceneBatch, ReconError> {
    frame.validate()?;
    let rgba = RgbaImage::from_fn(frame.rgb.width(), frame.rgb.height(), |x, y| {
        let c = frame.rgb.get_pixel(x, y).0;
        let a = frame.roi[(y * frame.rgb.width() + x) as usize];
        Rgba([c[0], c[1], c[2], a])
    });
    let (img_packed, has_alpha) =
        view_to_packed_data(DynamicImage::ImageRgba8(rgba), AlphaMode::Masked);
    Ok(SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::Masked,
        camera: frame.camera,
    })
}

pub struct GaussianBackend {
    seed: TriangleSeed,
    device: Device,
    rng_seed: u64,
    config: TrainConfig,
    ids: BTreeSet<u64>,
    splats: Option<Splats>,
    trainer: Option<SplatTrainer>,
    iterations: u64,
}
pub fn controlled_config() -> TrainConfig {
    let mut config = TrainConfig::parse_from(["brush-recon-bench"]);
    config.total_train_iters = u32::MAX;
    config.lr_mean_end = config.lr_mean;
    config.ssim_weight = 0.;
    config.ssim_every = 0;
    config.lpips_loss_weight = 0.;
    config.background_noise_strength = 0.;
    config.background_color = vec![0.; 3];
    config.sh_background = false;
    config.match_alpha_weight = 0.;
    config.mean_noise_weight = 0.;
    config.opac_decay = 0.;
    config.min_scale_factor = 0.;
    config.growth_stop_iter = 0;
    config.refine_every = u32::MAX;
    config
}
impl GaussianBackend {
    pub fn new(seed: TriangleSeed, device: Device, rng_seed: u64) -> Self {
        let config = controlled_config();
        Self {
            seed,
            device,
            rng_seed,
            config,
            ids: BTreeSet::new(),
            splats: None,
            trainer: None,
            iterations: 0,
        }
    }
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }
}
impl ReconstructionBackend for GaussianBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Gaussian
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            depth: false,
            normals: false,
        }
    }
    async fn initialize(&mut self, train: &[Arc<ReconFrame>]) -> Result<(), ReconError> {
        if self.splats.is_some() {
            return Err(ReconError::input("Gaussian backend is already initialized"));
        }
        self.seed.validate()?;
        if self.seed.positions.is_empty() {
            return Err(ReconError::input("Gaussian seed is empty"));
        }
        let ids = crate::training_ids(train)?;
        let data = SplatData {
            means: self.seed.positions.iter().flatten().copied().collect(),
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(
                self.seed
                    .colors
                    .iter()
                    .flat_map(|c| rgb_to_sh(Vec3::from(*c)).to_array())
                    .collect(),
            ),
            raw_opacities: None,
        };
        let splats = to_init_splats(data, SplatRenderMode::Default, &self.device.clone().inner())
            .with_sh_degree(0);
        let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        self.trainer = Some(SplatTrainer::new_seeded(
            &self.config,
            &self.device,
            bounds,
            self.rng_seed,
        ));
        self.splats = Some(splats);
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
        if context.iteration != self.iterations
            || !context.progress.is_finite()
            || !(0.0..=1.0).contains(&context.progress)
        {
            return Err(ReconError::input("invalid Gaussian training context"));
        }
        if !frame.roi.iter().any(|v| *v == 255) {
            return Err(ReconError::input("empty immutable training ROI"));
        }
        let batch = mask_scene_batch(frame)?;
        let current = self
            .splats
            .as_ref()
            .ok_or_else(|| ReconError::ModelState("Gaussian backend is not initialized".into()))?
            .clone();
        let trainer = self
            .trainer
            .as_mut()
            .ok_or_else(|| ReconError::ModelState("Gaussian trainer is not initialized".into()))?;
        let (next, stats) = trainer.step(batch, current.train()).await;
        let loss = stats
            .loss
            .into_data_async()
            .await
            .map_err(|e| ReconError::Gpu(format!("Gaussian loss: {e:?}")))?
            .try_to_vec::<f32>()
            .map_err(|e| ReconError::Gpu(format!("Gaussian loss value: {e:?}")))?[0];
        if !loss.is_finite() {
            return Err(ReconError::ModelState(
                "nonfinite Gaussian training loss".into(),
            ));
        }
        self.splats = Some(next.valid());
        self.iterations += 1;
        Ok(StepStats {
            iteration: self.iterations,
            rgb_l1: loss,
            depth_l1_m: None,
            total_loss: loss,
        })
    }
    async fn render(&self, request: &RenderRequest) -> Result<RenderResult, ReconError> {
        request.validate(self.capabilities())?;
        let splats = self
            .splats
            .as_ref()
            .ok_or_else(|| ReconError::ModelState("Gaussian backend is not initialized".into()))?;
        let (rgba, _) = render_splats(
            splats.clone(),
            &request.camera,
            request.size,
            request.background,
            None,
            TextureMode::Float,
        )
        .await;
        Ok(RenderResult {
            rgb: rgba.clone().slice(s![.., .., 0..3]),
            alpha: rgba
                .slice(s![.., .., 3..4])
                .reshape([request.size.y as usize, request.size.x as usize]),
            depth_m: None,
            normals: None,
        })
    }
    async fn synchronize(&self) -> Result<(), ReconError> {
        self.device
            .sync()
            .map_err(|e| ReconError::Gpu(format!("device synchronization: {e:?}")))
    }
    async fn export_state(&self) -> Result<ExperimentalState, ReconError> {
        let splats = self
            .splats
            .as_ref()
            .ok_or_else(|| ReconError::ModelState("Gaussian backend is not initialized".into()))?;
        let payload = brush_serde::splat_to_ply(splats.clone(), None)
            .await
            .map_err(|e| ReconError::ModelState(format!("Gaussian PLY export: {e:?}")))?;
        Ok(ExperimentalState {
            kind: BackendKind::Gaussian,
            format_version: 1,
            payload,
        })
    }
    fn stats(&self) -> ModelStats {
        ModelStats {
            gaussians: self.splats.as_ref().map_or(0, Splats::num_splats),
            ..Default::default()
        }
    }
}
