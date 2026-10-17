use brush_recon::StepContext;
use brush_recon::{
    BackendKind, ExperimentalState, ModelStats, ReconError, ReconstructionBackend, StepStats,
    capture::CaptureSplit,
};
use brush_triangle_train::schedule::{budget_context, step_context};
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RunBudget {
    Steps { steps: u64 },
    Time { seconds: f64 },
}
impl RunBudget {
    pub fn validate(&self) -> Result<(), ReconError> {
        match self {
            Self::Steps { steps } if *steps >= 2 => Ok(()),
            Self::Time { seconds }
                if *seconds > 0.
                    && seconds.is_finite()
                    && Duration::try_from_secs_f64(*seconds).is_ok() =>
            {
                Ok(())
            }
            _ => Err(ReconError::input(
                "budget needs steps>=2 or finite positive representable seconds",
            )),
        }
    }
    fn more(&self, iteration: u64, elapsed: Duration) -> bool {
        match self {
            Self::Steps { steps } => iteration < *steps,
            Self::Time { seconds } => elapsed.as_secs_f64() < *seconds,
        }
    }
    fn context(&self, iteration: u64, elapsed: Duration) -> Result<StepContext, ReconError> {
        match self {
            Self::Steps { steps } => step_context(iteration, *steps),
            Self::Time { seconds } => budget_context(
                iteration,
                elapsed,
                Duration::try_from_secs_f64(*seconds)
                    .map_err(|_| ReconError::input("time budget is not representable"))?,
            ),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct RunConfig {
    pub budget: RunBudget,
    pub warmup_steps: u64,
}
#[derive(Debug, Serialize)]
pub struct CurvePoint {
    pub frame_id: u64,
    pub progress: f32,
    pub optimization_seconds: f64,
    pub stats: StepStats,
}
#[derive(Serialize)]
pub struct RunRecord {
    pub backend: BackendKind,
    pub config: RunConfig,
    pub curve: Vec<CurvePoint>,
    pub stats: ModelStats,
    pub optimization_seconds: f64,
    pub initialization_seconds: f64,
    pub warmup_seconds: f64,
    pub wall_seconds: f64,
    #[serde(skip)]
    pub initial_state: ExperimentalState,
    #[serde(skip)]
    pub final_state: ExperimentalState,
}
pub async fn run_backend<B: ReconstructionBackend, F: FnMut() -> B>(
    mut factory: F,
    split: &CaptureSplit,
    config: &RunConfig,
) -> Result<RunRecord, ReconError> {
    config.budget.validate()?;
    if split.train.is_empty() {
        return Err(ReconError::input("runner requires training frames"));
    }
    let wall = Instant::now();
    let warm = Instant::now();
    if config.warmup_steps > 0 {
        let mut warmup = factory();
        warmup.initialize(&split.train).await?;
        for i in 0..config.warmup_steps {
            warmup
                .train_step(
                    &split.train[i as usize % split.train.len()],
                    StepContext {
                        iteration: i,
                        progress: 0.,
                    },
                )
                .await?;
        }
        warmup.synchronize().await?; // Drop every warmed model/optimizer before actual initialization.
    }
    let warmup_seconds = warm.elapsed().as_secs_f64();
    let mut backend = factory();
    let init = Instant::now();
    backend.initialize(&split.train).await?;
    backend.synchronize().await?;
    let initialization_seconds = init.elapsed().as_secs_f64();
    let initial_state = backend.export_state().await?;
    let initial_stats = backend.stats();
    let mut elapsed = Duration::ZERO;
    let mut iteration = 0_u64;
    let mut curve = vec![];
    while config.budget.more(iteration, elapsed) {
        let context = config.budget.context(iteration, elapsed)?;
        let frame = &split.train[iteration as usize % split.train.len()];
        let started = Instant::now();
        let stats = backend.train_step(frame, context).await?;
        backend.synchronize().await?;
        elapsed += started.elapsed();
        curve.push(CurvePoint {
            frame_id: frame.id,
            progress: context.progress,
            optimization_seconds: elapsed.as_secs_f64(),
            stats,
        });
        iteration = iteration
            .checked_add(1)
            .ok_or_else(|| ReconError::input("training iteration overflow"))?;
    }
    let stats = backend.stats();
    if stats.vertices != initial_stats.vertices
        || stats.triangles != initial_stats.triangles
        || stats.gaussians != initial_stats.gaussians
    {
        return Err(ReconError::ModelState(
            "fixed-topology/control run changed its model count".into(),
        ));
    }
    let final_state = backend.export_state().await?;
    Ok(RunRecord {
        backend: backend.kind(),
        config: config.clone(),
        curve,
        stats,
        optimization_seconds: elapsed.as_secs_f64(),
        initialization_seconds,
        warmup_seconds,
        wall_seconds: wall.elapsed().as_secs_f64(),
        initial_state,
        final_state,
    })
}
