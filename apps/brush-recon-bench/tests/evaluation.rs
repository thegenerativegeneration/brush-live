use brush_recon::capture::{CaptureManifest, CaptureSelection, CaptureSplit};
use brush_recon::*;
use brush_recon_bench::{
    gaussian::GaussianBackend,
    metrics::evaluate,
    run::{RunBudget, RunConfig, run_backend},
    triangle::TriangleBackend,
};
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use brush_triangle_render::TriangleSeed;
use brush_triangle_train::TrainConfig;
use burn::tensor::Device;
use glam::{Quat, UVec2, Vec2, Vec3};
use image::{Rgb, RgbImage};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![[-2., -2., 2.], [6., -2., 2.], [-2., 6., 2.]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0.4, 0.6, 0.8]; 3],
    }
}
fn frame(id: u64) -> Arc<ReconFrame> {
    Arc::new(ReconFrame {
        id,
        camera: Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            1.,
            1.,
            Vec2::splat(0.5),
            CameraModel::Pinhole,
        ),
        rgb: Arc::new(RgbImage::from_pixel(4, 4, Rgb([100, 150, 200]))),
        lidar: Some(Arc::new(MetricDepth {
            size: UVec2::splat(2),
            values_m: vec![2.; 4],
            confidence: Some(vec![2; 4]),
        })),
        roi: Arc::new(vec![255; 16]),
    })
}
fn split() -> CaptureSplit {
    CaptureSplit {
        train: vec![frame(174)],
        holdout: vec![frame(178)],
        manifest: CaptureManifest {
            format_version: 1,
            source: ".".into(),
            source_kind: "generated fixture".into(),
            pose_convention: "Brush".into(),
            selection: CaptureSelection::default(),
            train_ids: vec![174],
            holdout_ids: vec![178],
            source_sha256: Default::default(),
            selection_note: "fixture".into(),
            frozen: None,
        },
    }
}

struct Observed<B> {
    inner: B,
    requests: Arc<Mutex<Vec<(UVec2, bool)>>>,
    pending: Arc<AtomicBool>,
    slow: bool,
}
impl<B: ReconstructionBackend> ReconstructionBackend for Observed<B> {
    fn kind(&self) -> BackendKind {
        self.inner.kind()
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    async fn initialize(&mut self, f: &[Arc<ReconFrame>]) -> Result<(), ReconError> {
        self.pending.store(true, Ordering::SeqCst);
        self.inner.initialize(f).await
    }
    async fn train_step(
        &mut self,
        f: &ReconFrame,
        c: StepContext,
    ) -> Result<StepStats, ReconError> {
        self.pending.store(true, Ordering::SeqCst);
        self.inner.train_step(f, c).await
    }
    async fn render(&self, r: &RenderRequest) -> Result<RenderResult, ReconError> {
        self.requests.lock().unwrap().push((r.size, r.depth));
        self.pending.store(true, Ordering::SeqCst);
        if self.slow {
            std::thread::sleep(Duration::from_millis(30));
        }
        self.inner.render(r).await
    }
    async fn synchronize(&self) -> Result<(), ReconError> {
        if self.slow {
            std::thread::sleep(Duration::from_millis(20));
        }
        self.inner.synchronize().await?;
        self.pending.store(false, Ordering::SeqCst);
        Ok(())
    }
    async fn export_state(&self) -> Result<ExperimentalState, ReconError> {
        assert!(
            !self.pending.load(Ordering::SeqCst),
            "must explicitly synchronize before snapshots"
        );
        self.inner.export_state().await
    }
    fn stats(&self) -> ModelStats {
        self.inner.stats()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn evaluator_requests_native_depth_grid_and_counts_targets_once() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let requests = Arc::new(Mutex::new(vec![]));
    let mut backend = Observed {
        inner: TriangleBackend::new(seed(), d, TrainConfig::default()),
        requests: requests.clone(),
        pending: Arc::new(AtomicBool::new(false)),
        slow: false,
    };
    backend.initialize(&[frame(174)]).await.unwrap();
    backend.synchronize().await.unwrap();
    let evaluation = evaluate(&backend, &[frame(178)], &seed()).await.unwrap();
    assert_eq!(
        *requests.lock().unwrap(),
        [(UVec2::splat(4), false), (UVec2::splat(2), true)]
    );
    assert_eq!(evaluation.aggregate.depth.unwrap().valid_targets, 4);
    assert_eq!(evaluation.aggregate.roi_depth.unwrap().valid_targets, 4);
}

#[tokio::test(flavor = "current_thread")]
async fn gaussian_depth_metrics_are_unsupported_not_zero_error() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut backend = GaussianBackend::new(seed(), d, 0);
    backend.initialize(&[frame(174)]).await.unwrap();
    let evaluation = evaluate(&backend, &[frame(178)], &seed()).await.unwrap();
    assert!(evaluation.aggregate.depth.is_none() && evaluation.aggregate.roi_depth.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn runner_times_synchronization_but_excludes_slow_evaluation_and_reports_budget_overshoot() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let requests = Arc::new(Mutex::new(vec![]));
    let pending = Arc::new(AtomicBool::new(false));
    let data = split();
    let config = RunConfig {
        budget: RunBudget::Time { seconds: 0.001 },
        warmup_steps: 1,
    };
    let record = run_backend(
        || Observed {
            inner: TriangleBackend::new(seed(), d.clone(), TrainConfig::default()),
            requests: requests.clone(),
            pending: pending.clone(),
            slow: true,
        },
        &data,
        &config,
        &seed(),
    )
    .await
    .unwrap();
    assert_eq!(record.curve.len(), 1);
    assert!(record.optimization_seconds >= 0.020);
    assert!(record.evaluation_seconds >= 0.120);
    assert!(
        record.optimization_seconds < record.evaluation_seconds,
        "evaluation must not consume optimization budget"
    );
    assert!(record.wall_seconds >= record.optimization_seconds + record.evaluation_seconds);
}

#[tokio::test(flavor = "current_thread")]
async fn saved_artifacts_include_native_layout_images_states_and_refuse_overwrites() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let data = split();
    let mut record = run_backend(
        || TriangleBackend::new(seed(), d.clone(), TrainConfig::default()),
        &data,
        &RunConfig {
            budget: RunBudget::Steps { steps: 2 },
            warmup_steps: 1,
        },
        &seed(),
    )
    .await
    .unwrap();
    let output = tempfile::tempdir().unwrap();
    brush_recon_bench::report::save_artifacts(&record, output.path()).unwrap();
    for name in [
        "metrics.json",
        "curve.json",
        "initial.state.json",
        "final.state.json",
        "initial.ply",
        "final.ply",
        "images/initial-178.png",
        "images/final-178.png",
        "images/target-178.png",
        "images/roi-178.png",
        "images/final-178-depth.png",
        "images/target-178-depth.png",
    ] {
        assert!(output.path().join(name).is_file(), "{name}");
    }
    let raw = std::fs::read(output.path().join("images/final-178-depth.f32")).unwrap();
    assert_eq!(raw.len(), 2 * 2 * 4);
    let layout: serde_json::Value = serde_json::from_slice(
        &std::fs::read(output.path().join("images/native-layout.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(layout[0]["width"], 2);
    assert_eq!(layout[0]["height"], 2);
    let before = std::fs::read(output.path().join("final.state.json")).unwrap();
    assert!(brush_recon_bench::report::save_artifacts(&record, output.path()).is_err());
    assert_eq!(
        before,
        std::fs::read(output.path().join("final.state.json")).unwrap()
    );
    let footprint = brush_recon_bench::report::process_footprint();
    if let Some(current) = footprint.current_bytes {
        assert!(current > 0);
        if let Some(peak) = footprint.lifetime_peak_bytes {
            assert!(peak >= current);
        }
    }
    record.final_evaluation.images[0].roi = Arc::new(vec![0; 16]);
    let second = tempfile::tempdir().unwrap();
    assert!(brush_recon_bench::report::save_artifacts(&record, second.path()).is_err());
    assert!(
        !second.path().join("curve.json").exists(),
        "changed support must fail before artifact writes"
    );
}
