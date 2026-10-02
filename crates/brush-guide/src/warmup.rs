//! GPU warm-up at server start. cubecl autotunes a kernel the first time it
//! runs at a new size bucket (powers of two of the splat count), which costs
//! seconds per bucket on a cold cache. Running the session's GPU work once
//! per bucket on dummy splats fills the cache (in memory, and on disk under
//! the server's working directory) before a phone connects.

use crate::config::GuideConfig;
use crate::geometry::depth::{render_colour, render_expected_depth};
use crate::scores::pass::{PassView, score_pass};
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::module::Module as _;
use burn::tensor::Device;
use clap::Parser as _;
use glam::{UVec2, Vec3};
use rand::{RngExt as _, SeedableRng};
use tokio::sync::watch;
use web_time::Instant;

/// A warm-up running on its own GPU thread; `ready` turns true when it is
/// done. If it panics, the sender is dropped, so waiters stop waiting too.
pub struct Warmup {
    _actor: brush_async::Actor,
    ready: watch::Receiver<bool>,
}

impl Warmup {
    /// Starts [`warm_up`]; keep the returned value alive until it is done.
    pub fn spawn(config: GuideConfig, device: Device) -> Self {
        let (tx, ready) = watch::channel(false);
        let actor = brush_async::Actor::new("brush-guide-warmup");
        actor
            .run(move || async move {
                let sizes = warmup_sizes(config.max_splats);
                log::info!("warm-up: {} splat counts up to {}", sizes.len(), config.max_splats);
                let secs = warm_up(&config, &device).await;
                log::info!("warm-up done in {secs:.1} s; sessions start now");
                let _ = tx.send(true);
            })
            .detach();
        Self {
            _actor: actor,
            ready,
        }
    }

    pub fn ready(&self) -> watch::Receiver<bool> {
        self.ready.clone()
    }
}

/// Smallest splat count warmed; sessions start from a few thousand seeds.
const MIN_SPLATS: u32 = 1 << 12;

/// Keyframe size warmed: landscape 4:3 at `long_side`, as ARKit images are.
pub fn warmup_image(long_side: u32) -> UVec2 {
    UVec2::new(long_side, long_side * 3 / 4)
}

/// Splat counts to warm: one per autotune bucket from `MIN_SPLATS` up to
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
/// and a refine, a one-view Fisher pass (render and backward), and the expected-depth and
/// colour renders of TSDF fusion. `device` is the autodiff device. Returns
/// the seconds it took.
pub async fn warm_up(config: &GuideConfig, device: &Device) -> f64 {
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
    for n in warmup_sizes(config.max_splats) {
        let t = Instant::now();
        let splats = dummy_splats(n, &camera, &mut rng, &device.clone().inner())
            .with_sh_degree(config.sh_degree);
        let splats = train_steps(config, device, splats, &camera, image, &mut rng).await;
        let view = PassView {
            camera,
            img_size: image,
            weight: 1.0,
        };
        let _ = score_pass(&splats, &[view], &config.pass).await;
        let small = image / 4;
        let _ = render_expected_depth(&splats, &camera, small).await;
        let _ = render_colour(&splats, &camera, small).await;
        log::info!(
            "warm-up: {n} splats in {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    start.elapsed().as_secs_f64()
}

/// `n` small Gaussians spread through the camera's view between 1 and 5 m.
fn dummy_splats(n: u32, camera: &Camera, rng: &mut impl rand::Rng, device: &Device) -> Splats {
    let (tx, ty) = ((camera.fov_x / 2.0).tan() as f32, (camera.fov_y / 2.0).tan() as f32);
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

async fn train_steps(
    config: &GuideConfig,
    device: &Device,
    splats: Splats,
    camera: &Camera,
    image: UVec2,
    rng: &mut impl rand::Rng,
) -> Splats {
    let mut train_config = TrainConfig::parse_from(["brush-guide-warmup"]);
    train_config.max_splats = config.max_splats;
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
    fn warmup_image_is_four_by_three_at_the_long_side() {
        assert_eq!(warmup_image(960), UVec2::new(960, 720));
        assert_eq!(warmup_image(500), UVec2::new(500, 375));
    }

    #[test]
    fn sizes_cover_each_bucket_up_to_the_budget() {
        assert_eq!(
            warmup_sizes(750_000),
            vec![4096, 8192, 16384, 32768, 65536, 131_072, 262_144, 524_288, 750_000]
        );
        assert_eq!(warmup_sizes(1 << 14), vec![4096, 8192, 16384]);
        assert_eq!(warmup_sizes(100), vec![4096]);
    }
}
