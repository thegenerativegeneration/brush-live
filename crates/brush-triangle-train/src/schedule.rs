use brush_recon::{ReconError, StepContext};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ScheduleConfig {
    pub sigma_start: f32,
    pub sigma_end: f32,
    pub weight_floor_start: f32,
    pub weight_floor_end: f32,
}
impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            sigma_start: 1.,
            sigma_end: 0.1,
            weight_floor_start: 0.,
            weight_floor_end: 0.99,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ScheduleState {
    pub progress: f32,
    pub sigma: f32,
    pub weight_floor: f32,
}
pub fn schedule(progress: f32, cfg: &ScheduleConfig) -> Result<ScheduleState, ReconError> {
    if !progress.is_finite()
        || !cfg.sigma_start.is_finite()
        || cfg.sigma_start <= 0.
        || !cfg.sigma_end.is_finite()
        || cfg.sigma_end <= 0.
        || !cfg.weight_floor_start.is_finite()
        || !(0.0..=1.0).contains(&cfg.weight_floor_start)
        || !cfg.weight_floor_end.is_finite()
        || !(0.0..=1.0).contains(&cfg.weight_floor_end)
    {
        return Err(ReconError::input(
            "invalid sigma/floor schedule or progress",
        ));
    }
    let p = progress.clamp(0., 1.);
    Ok(ScheduleState {
        progress: p,
        sigma: cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * p,
        weight_floor: cfg.weight_floor_start + (cfg.weight_floor_end - cfg.weight_floor_start) * p,
    })
}
pub fn step_context(iteration: u64, steps: u64) -> Result<StepContext, ReconError> {
    if steps < 2 || iteration >= steps {
        return Err(ReconError::input(
            "fixed-step run requires N>=2 and iteration<N",
        ));
    }
    Ok(StepContext {
        iteration,
        progress: (iteration as f64 / (steps - 1) as f64) as f32,
    })
}
pub fn budget_context(
    iteration: u64,
    elapsed: Duration,
    budget: Duration,
) -> Result<StepContext, ReconError> {
    if budget.is_zero() {
        return Err(ReconError::input("training time budget must be positive"));
    }
    Ok(StepContext {
        iteration,
        progress: (elapsed.as_secs_f64() / budget.as_secs_f64()).clamp(0., 1.) as f32,
    })
}
