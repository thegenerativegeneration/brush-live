use crate::{
    loss::{LossConfig, triangle_loss},
    optimizer::{DeviceAdam, cap_vertex_displacement},
    schedule::{ScheduleConfig, ScheduleState, schedule},
};
use brush_recon::{ReconError, ReconFrame, RenderRequest, RenderResult, StepContext, StepStats};
use brush_triangle_render::{
    RenderConfig, TriangleModel, TriangleSeed, reference::ReferenceConfig, render_triangles,
};
use burn::tensor::{Tensor, TensorData};
use glam::{UVec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub vertex_lr: f32,
    pub rgb_lr: f32,
    pub weight_lr: f32,
    pub max_vertex_step_m: f32,
    pub loss: LossConfig,
    pub schedule: ScheduleConfig,
    pub render: RenderConfig,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vertex_lr: 1e-3,
            rgb_lr: 1e-2,
            weight_lr: 1e-2,
            max_vertex_step_m: 0.01,
            loss: LossConfig::default(),
            schedule: ScheduleConfig::default(),
            render: RenderConfig::default(),
        }
    }
}
impl TrainConfig {
    pub fn validate(&self) -> Result<(), ReconError> {
        self.loss.validate()?;
        schedule(0., &self.schedule)?;
        ReferenceConfig {
            sigma: self.render.sigma,
            weight_floor: self.render.weight_floor,
            near_m: self.render.near_m,
        }
        .validate()?;
        if [
            self.vertex_lr,
            self.rgb_lr,
            self.weight_lr,
            self.max_vertex_step_m,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
            || self.render.buffer_budget_bytes == 0
        {
            return Err(ReconError::input(
                "invalid optimizer learning rate, vertex step cap or renderer budget",
            ));
        }
        Ok(())
    }
}
pub struct TriangleTrainer {
    pub model: TriangleModel,
    pub config: TrainConfig,
    pub last_schedule: ScheduleState,
    pub iterations: u64,
    seed_positions: Tensor<2>,
    vertex_adam: DeviceAdam<2>,
    rgb_adam: DeviceAdam<2>,
    weight_adam: DeviceAdam<1>,
}
impl TriangleTrainer {
    pub fn new(
        mut model: TriangleModel,
        seed: TriangleSeed,
        config: TrainConfig,
    ) -> Result<Self, ReconError> {
        config.validate()?;
        seed.validate()?;
        if seed.faces.is_empty()
            || model.num_vertices() != seed.positions.len()
            || model.num_faces() != seed.faces.len()
            || [
                config.vertex_lr,
                config.rgb_lr,
                config.weight_lr,
                config.max_vertex_step_m,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(ReconError::input(
                "training needs a nonempty fixed seed/model and finite positive learning rates/step limit",
            ));
        }
        let last_schedule = schedule(0., &config.schedule)?;
        ReferenceConfig {
            sigma: config.render.sigma,
            weight_floor: config.render.weight_floor,
            near_m: config.render.near_m,
        }
        .validate()?;
        model.positions = model.positions.detach().without_autodiff();
        model.rgb_logits = model.rgb_logits.detach().without_autodiff();
        model.weight_logits = model.weight_logits.detach().without_autodiff();
        let seed_positions = Tensor::from_data(
            TensorData::new(
                seed.positions.into_iter().flatten().collect::<Vec<_>>(),
                [model.num_vertices(), 3],
            ),
            &model.positions.device(),
        );
        let vertex_adam = DeviceAdam::new(&model.positions);
        let rgb_adam = DeviceAdam::new(&model.rgb_logits);
        let weight_adam = DeviceAdam::new(&model.weight_logits);
        Ok(Self {
            model,
            config,
            last_schedule,
            iterations: 0,
            seed_positions,
            vertex_adam,
            rgb_adam,
            weight_adam,
        })
    }
    pub async fn step(
        &mut self,
        frame: &ReconFrame,
        context: StepContext,
    ) -> Result<StepStats, ReconError> {
        if context.iteration != self.iterations {
            return Err(ReconError::input(
                "training context iteration is out of sequence",
            ));
        }
        let state = schedule(context.progress, &self.config.schedule)?;
        let cfg = RenderConfig {
            sigma: state.sigma,
            weight_floor: state.weight_floor,
            ..self.config.render
        };
        let mut tracked = self.model.clone();
        tracked.positions = tracked.positions.detach().autodiff().require_grad();
        tracked.rgb_logits = tracked.rgb_logits.detach().autodiff().require_grad();
        tracked.weight_logits = tracked.weight_logits.detach().autodiff().require_grad();
        let request = RenderRequest {
            camera: frame.camera,
            size: UVec2::new(frame.rgb.width(), frame.rgb.height()),
            background: Vec3::ZERO,
            depth: frame.lidar.is_some(),
            normals: false,
        };
        let rendered = render_triangles(&tracked, &request, &cfg).await?;
        let loss = triangle_loss(
            &rendered,
            frame,
            &self.seed_positions,
            &tracked.positions,
            &self.config.loss,
        )?;
        let metrics = Tensor::cat(
            vec![
                loss.rgb.clone().without_autodiff(),
                loss.depth_m
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([1], &loss.rgb.device()))
                    .without_autodiff(),
                loss.total.clone().without_autodiff(),
            ],
            0,
        );
        let has_depth = loss.depth_m.is_some();
        let mut grads = loss.total.backward();
        let p = tracked
            .positions
            .grad_remove(&mut grads)
            .ok_or_else(|| ReconError::ModelState("missing position gradient".into()))?;
        let c = tracked
            .rgb_logits
            .grad_remove(&mut grads)
            .ok_or_else(|| ReconError::ModelState("missing RGB gradient".into()))?;
        let w = tracked
            .weight_logits
            .grad_remove(&mut grads)
            .ok_or_else(|| ReconError::ModelState("missing weight gradient".into()))?;
        let proposed = self
            .vertex_adam
            .update(&self.model.positions, p, self.config.vertex_lr)
            .await?;
        let positions = cap_vertex_displacement(
            self.model.positions.clone(),
            proposed,
            self.config.max_vertex_step_m,
        )?;
        let rgb = self
            .rgb_adam
            .update(&self.model.rgb_logits, c, self.config.rgb_lr)
            .await?;
        let weights = self
            .weight_adam
            .update(&self.model.weight_logits, w, self.config.weight_lr)
            .await?;
        let numbers = metrics
            .into_data_async()
            .await
            .map_err(|e| ReconError::Gpu(format!("training metric readback: {e:?}")))?
            .try_to_vec::<f32>()
            .map_err(|e| ReconError::Gpu(format!("training metrics: {e:?}")))?;
        if numbers.iter().any(|v| !v.is_finite()) {
            return Err(ReconError::ModelState("nonfinite training loss".into()));
        }
        self.model.positions = positions.detach();
        self.model.rgb_logits = rgb;
        self.model.weight_logits = weights;
        self.iterations += 1;
        self.last_schedule = state;
        Ok(StepStats {
            iteration: self.iterations,
            rgb_l1: numbers[0],
            depth_l1_m: has_depth.then_some(numbers[1]),
            total_loss: numbers[2],
        })
    }
    pub async fn render(&self, request: &RenderRequest) -> Result<RenderResult, ReconError> {
        render_triangles(
            &self.model,
            request,
            &RenderConfig {
                sigma: self.last_schedule.sigma,
                weight_floor: self.last_schedule.weight_floor,
                ..self.config.render
            },
        )
        .await
    }
    pub fn optimizer_scalar_readbacks(&self) -> u64 {
        self.vertex_adam.scalar_readbacks()
            + self.rgb_adam.scalar_readbacks()
            + self.weight_adam.scalar_readbacks()
    }
}
