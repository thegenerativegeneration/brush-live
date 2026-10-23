//! GPU warm-up at server start. Kernels compile the first time they run at a
//! new size bucket (powers of two of the splat count, from 128, so small seed
//! sets find theirs compiled). Running the session's GPU work once per
//! bucket on dummy splats compiles them before a phone connects. Where
//! autotune is enabled (host builds), the same run also tunes them and fills
//! the cache in memory and on disk under the server's working directory.
//! Training and the voxel round's readback are warmed, and with a holdout
//! the evals' extras (LPIPS, seam render); the optional finish-time Fisher
//! pass runs on first use.

use crate::config::GuideConfig;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::kernels::camera_model::CameraModel;
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::module::Module as _;
use burn::tensor::Device;
use clap::Parser as _;
use glam::{UVec2, Vec3};
use rand::{RngExt as _, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::watch;
use web_time::Instant;

/// A warm-up running on its own GPU thread; `ready` turns true when it is
/// done. If it panics, the sender is dropped, so waiters stop waiting too.
pub struct Warmup {
    _actor: brush_async::Actor,
    ready: watch::Receiver<bool>,
    parked: watch::Receiver<bool>,
}

/// Pause control of a running warm-up: the flag to watch, and the signal
/// that the warm-up is waiting on it.
struct Gate {
    pause: watch::Receiver<bool>,
    parked: watch::Sender<bool>,
}

impl Gate {
    /// Returns once the pause flag is clear, parking in the meantime.
    async fn pass(&mut self) {
        if !*self.pause.borrow() {
            return;
        }
        let _ = self.parked.send(true);
        let _ = self.pause.wait_for(|paused| !paused).await;
        let _ = self.parked.send(false);
    }
}

impl Warmup {
    /// Starts [`warm_up`]; keep the returned value alive until it is done.
    pub fn spawn(config: GuideConfig, device: Device) -> Self {
        Self::spawn_pausable(config, device, watch::channel(false).1)
    }

    /// Like [`Self::spawn`], but the warm-up stops between splat counts while
    /// `pause` is true; the count in progress finishes first. [`Self::parked`]
    /// tells when it waits.
    pub fn spawn_pausable(
        config: GuideConfig,
        device: Device,
        pause: watch::Receiver<bool>,
    ) -> Self {
        let (tx, ready) = watch::channel(false);
        let (parked_tx, parked) = watch::channel(false);
        let mut gate = Gate {
            pause,
            parked: parked_tx,
        };
        let actor = brush_async::Actor::new("brush-guide-warmup");
        actor
            .run(move || async move {
                let sizes = warmup_sizes(config.max_splats);
                log::info!(
                    "warm-up: {} splat counts up to {}",
                    sizes.len(),
                    config.max_splats
                );
                let secs = warm_up_gated(&config, &device, Some(&mut gate)).await;
                log::info!("warm-up done in {secs:.1} s; sessions start now");
                let _ = tx.send(true);
            })
            .detach();
        Self {
            _actor: actor,
            ready,
            parked,
        }
    }

    /// True while the warm-up waits for its pause flag to clear.
    pub fn parked(&self) -> watch::Receiver<bool> {
        self.parked.clone()
    }

    pub fn ready(&self) -> watch::Receiver<bool> {
        self.ready.clone()
    }
}

/// Smallest splat count warmed; a capture can start from a few hundred seeds.
const MIN_SPLATS: u32 = 1 << 7;

static PROGRESS_DONE: AtomicU32 = AtomicU32::new(0);
static PROGRESS_TOTAL: AtomicU32 = AtomicU32::new(0);

/// Sizes done and sizes in total of the warm-up running now (or the last one).
pub fn progress() -> (u32, u32) {
    (
        PROGRESS_DONE.load(Ordering::Relaxed),
        PROGRESS_TOTAL.load(Ordering::Relaxed),
    )
}

fn reset_progress(total: u32) {
    PROGRESS_DONE.store(0, Ordering::Relaxed);
    PROGRESS_TOTAL.store(total, Ordering::Relaxed);
}

fn step_progress() {
    let total = PROGRESS_TOTAL.load(Ordering::Relaxed);
    let _ = PROGRESS_DONE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |d| {
        (d < total).then_some(d + 1)
    });
}

/// Keyframe size warmed: landscape 4:3 at `long_side`, as ARKit images are.
pub fn warmup_image(long_side: u32) -> UVec2 {
    UVec2::new(long_side, long_side * 3 / 4)
}

/// Splat counts to warm: one per size bucket from `MIN_SPLATS` up to
/// the bucket holding `max_splats`, the last one at `max_splats` itself.
pub fn warmup_sizes(max_splats: u32) -> Vec<u32> {
    let max_splats = max_splats.max(MIN_SPLATS);
    let mut sizes: Vec<u32> = std::iter::successors(Some(MIN_SPLATS), |n| n.checked_mul(2))
        .take_while(|&n| n < max_splats)
        .collect();
    sizes.push(max_splats);
    sizes
}

/// Runs, at each of `warmup_sizes(config.max_splats)`: two training steps
/// and a refine; then, with a holdout, the evals' extras once on the largest
/// splat set. `device` is the autodiff device. Returns the seconds it took.
pub async fn warm_up(config: &GuideConfig, device: &Device) -> f64 {
    warm_up_gated(config, device, None).await
}

async fn warm_up_gated(config: &GuideConfig, device: &Device, mut gate: Option<&mut Gate>) -> f64 {
    let start = Instant::now();
    let image = warmup_image(config.keyframe_long_side);
    let fov = 2.0 * (f64::from(image.x) / 2.0 / 700.0).atan();
    let fov_y = 2.0 * (f64::from(image.y) / 2.0 / 700.0).atan();
    let camera = Camera::new(
        Vec3::ZERO,
        glam::Quat::IDENTITY,
        fov,
        fov_y,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let sizes = warmup_sizes(config.max_splats);
    reset_progress(sizes.len() as u32);
    let mut last = None;
    for n in sizes {
        if let Some(gate) = gate.as_deref_mut() {
            gate.pass().await;
        }
        let t = Instant::now();
        let splats = dummy_splats(n, &camera, &mut rng, &device.clone().inner())
            .with_sh_degree(config.sh_degree);
        let trained = train_steps(config, device, splats, &camera, image, &mut rng).await;
        // The voxel round reads the splat parameters back each round.
        let _ = crate::session::splat_read::SplatRead::new(&trained).await;
        step_progress();
        log::info!(
            "warm-up: {n} splats in {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        last = Some(trained);
    }
    if let Some(splats) = last.filter(|_| config.holdout.every >= 2) {
        let t = Instant::now();
        crate::session::warm_up_eval(config, &splats, &camera, image).await;
        log::info!("warm-up: eval extras in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    start.elapsed().as_secs_f64()
}

/// `n` small Gaussians spread through the camera's view between 1 and 5 m.
fn dummy_splats(n: u32, camera: &Camera, rng: &mut impl rand::Rng, device: &Device) -> Splats {
    let (tx, ty) = (
        (camera.fov_x / 2.0).tan() as f32,
        (camera.fov_y / 2.0).tan() as f32,
    );
    let n = n as usize;
    let means: Vec<f32> = (0..n)
        .flat_map(|_| {
            let z = rng.random_range(1.0f32..5.0);
            [
                rng.random_range(-tx..tx) * z,
                rng.random_range(-ty..ty) * z,
                z,
            ]
        })
        .collect();
    let colours: Vec<f32> = (0..n * 3).map(|_| rng.random_range(0.0..1.0)).collect();
    Splats::from_raw(
        means,
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![(0.01f32).ln(); n * 3],
        colours,
        vec![0.0; n],
        SplatRenderMode::Default,
        device,
    )
}

/// The two warm-up steps cover both loss variants: 1 and 0 stay as they are,
/// any k >= 2 becomes 2 (step 1 L1-only, step 2 with SSIM).
fn warmup_ssim_every(ssim_every: u32) -> u32 {
    ssim_every.min(2)
}

/// The training options of the warm-up steps: the session's, where they choose kernels (the SH background's globe,
/// the training radius's masked step).
fn warmup_train_config(config: &GuideConfig) -> TrainConfig {
    let mut train_config = TrainConfig::parse_from(["brush-guide-warmup"]);
    train_config.max_splats = config.max_splats;
    train_config.ssim_every = warmup_ssim_every(config.ssim_every);
    train_config.sh_background = config.sh_background;
    train_config.sh_background_alpha_weight = config.sh_background_alpha_weight;
    train_config.train_radius_m = config.train_radius_m;
    train_config
}

async fn train_steps(
    config: &GuideConfig,
    device: &Device,
    splats: Splats,
    camera: &Camera,
    image: UVec2,
    rng: &mut impl rand::Rng,
) -> Splats {
    let train_config = warmup_train_config(config);
    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new_seeded(&train_config, device, bounds, config.seed);
    let pixels: Vec<u8> = (0..image.x * image.y * 3).map(|_| rng.random()).collect();
    let img = image::RgbImage::from_raw(image.x, image.y, pixels).expect("image size");
    let (img_packed, has_alpha) =
        view_to_packed_data(image::DynamicImage::ImageRgb8(img), AlphaMode::default());
    let batch = SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::default(),
        camera: *camera,
    };
    let mut splats = splats;
    for _ in 0..2 {
        let (stepped, _) = trainer.step(batch.clone(), splats.train()).await;
        splats = stepped.valid();
    }
    // Warms the refine kernels too, not just the training step's.
    let (refined, _) = trainer.refine(train_config.refine_every, splats).await;
    refined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_trains_with_the_sessions_kernel_choices() {
        let config = GuideConfig {
            sh_background: true,
            sh_background_alpha_weight: 0.3,
            train_radius_m: 25.0,
            ..GuideConfig::default()
        };
        let t = warmup_train_config(&config);
        assert!(t.sh_background);
        assert_eq!((t.sh_background_alpha_weight, t.train_radius_m), (0.3, 25.0));
        let t = warmup_train_config(&GuideConfig::default());
        assert!(!t.sh_background);
        assert_eq!(t.train_radius_m, 0.0);
    }

    #[test]
    fn warmup_ssim_every_covers_both_loss_variants() {
        assert_eq!(warmup_ssim_every(0), 0);
        assert_eq!(warmup_ssim_every(1), 1);
        assert_eq!(warmup_ssim_every(2), 2);
        assert_eq!(warmup_ssim_every(7), 2);
    }

    #[test]
    fn warmup_image_is_four_by_three_at_the_long_side() {
        assert_eq!(warmup_image(960), UVec2::new(960, 720));
        assert_eq!(warmup_image(500), UVec2::new(500, 375));
    }

    #[test]
    fn sizes_start_small_and_end_at_the_budget() {
        assert_eq!(
            warmup_sizes(25_000),
            vec![128, 256, 512, 1024, 2048, 4096, 8192, 16384, 25_000]
        );
        assert_eq!(*warmup_sizes(50_000).last().unwrap(), 50_000);
        assert_eq!(warmup_sizes(1 << 14).last(), Some(&16384));
        assert_eq!(warmup_sizes(100), vec![128]);
    }

    #[test]
    fn progress_counts_sizes_and_never_passes_the_total() {
        reset_progress(3);
        assert_eq!(progress(), (0, 3));
        step_progress();
        step_progress();
        step_progress();
        step_progress();
        assert_eq!(progress(), (3, 3));
    }
}
