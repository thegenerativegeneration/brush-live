//! Profiling only adds device syncs: a profiled step must give the same
//! parameters as an unprofiled one from the same state.

use super::*;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

fn splats(n: usize, device: &Device) -> Splats {
    Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, 2.0]).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![-3.0; n * 3],
        vec![0.5; n * 3],
        vec![inverse_sigmoid(0.5); n],
        SplatRenderMode::Default,
        device,
    )
}

fn batch() -> SceneBatch {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        32,
        32,
        image::Rgb([200, 60, 20]),
    ));
    let (img_packed, has_alpha) = view_to_packed_data(img, AlphaMode::Transparent);
    let fov = 60f64.to_radians();
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::Transparent,
        camera: Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            fov,
            fov,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        ),
    }
}

async fn run(profiling: bool) -> Vec<f32> {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let mut s = splats(60, &device);
    let bounds = get_splat_bounds(s.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);
    trainer.set_profiling(profiling);
    for _ in 0..3 {
        let (stepped, _) = trainer.step(batch(), s.train()).await;
        s = stepped.valid();
    }
    s.means()
        .into_data_async()
        .await
        .unwrap()
        .try_into_vec::<f32>()
        .unwrap()
}

#[tokio::test]
async fn profiling_does_not_change_the_step() {
    let plain = run(false).await;
    let profiled = run(true).await;
    let max_diff = plain
        .iter()
        .zip(&profiled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= 1e-6, "means differ by up to {max_diff}");
}
