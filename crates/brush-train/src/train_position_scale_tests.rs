//! The position scale (`median_scale` of the mean LR and the position-noise clamp) with and without
//! `position_scale_cap_m`.

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
        image::Rgb([255, 0, 0]),
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

/// `lr_mean` of one step on bounds with median size 40 m, with no LR decay.
async fn lr_mean_with_cap(cap: f32) -> (f64, f32, f64) {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.lr_mean_end = config.lr_mean;
    config.position_scale_cap_m = cap;
    let bounds = BoundingBox::from_min_max(glam::Vec3::ZERO, glam::Vec3::splat(40.0));
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let scale = trainer.position_scale();
    let (_, stats) = trainer.step(batch(), splats(20, &device).train()).await;
    (stats.lr_mean, scale, config.lr_mean)
}

#[tokio::test]
async fn cap_bounds_the_position_scale_when_bounds_exceed_it() {
    let (lr, scale, base) = lr_mean_with_cap(10.0).await;
    assert_eq!(scale, 10.0);
    assert!((lr - base * 10.0).abs() < base * 1e-6, "lr_mean {lr}");
}

#[tokio::test]
async fn cap_off_or_above_bounds_keeps_the_median_size() {
    for cap in [0.0, 100.0] {
        let (lr, scale, base) = lr_mean_with_cap(cap).await;
        assert_eq!(scale, 40.0, "cap {cap}");
        assert!(
            (lr - base * 40.0).abs() < base * 1e-6,
            "cap {cap}: lr_mean {lr}"
        );
    }
}
