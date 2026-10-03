//! Flag-on convergence (gradient flow + world-space direction convention)
//! and flag-off purity (no basis cache, no background module) for the SH
//! environment globe (`crate::sh_background`).

use super::*;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

const SIZE: u32 = 32;

/// A single near-transparent splat, so the render is almost entirely
/// background and the globe has to do the work of matching the GT colour.
fn near_invisible_splat(device: &Device) -> Splats {
    Splats::from_raw(
        vec![0.0, 0.0, 2.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![-3.0; 3],
        vec![0.2; 3],
        vec![inverse_sigmoid(1e-4)],
        SplatRenderMode::Default,
        device,
    )
}

fn solid_batch(rgb: [u8; 3], rotation: glam::Quat) -> SceneBatch {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        SIZE,
        SIZE,
        image::Rgb(rgb),
    ));
    let (img_packed, has_alpha) = view_to_packed_data(img, AlphaMode::Transparent);
    let fov = 60f64.to_radians();
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::Transparent,
        camera: Camera::new(
            glam::Vec3::ZERO,
            rotation,
            fov,
            fov,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        ),
    }
}

async fn globe_rgb_at(trainer: &SplatTrainer, dir: glam::Vec3) -> [f32; 3] {
    let device = trainer.sh_background.as_ref().expect("globe on").coeffs().device();
    let dirs = Tensor::<1>::from_floats([dir.x, dir.y, dir.z], &device).reshape([1, 3]);
    let basis = sh_basis(dirs);
    let image = trainer.sh_background.as_ref().expect("globe on").image(basis);
    let data: Vec<f32> = image
        .into_data_async()
        .await
        .expect("readback")
        .try_into_vec()
        .expect("f32");
    [data[0], data[1], data[2]]
}

#[tokio::test]
async fn globe_learns_opposed_camera_colours_in_world_space() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.sh_background = true;
    config.sh_background_lr = 0.1;
    config.sh_background_rest_lr = 0.1;
    config.match_alpha_weight = 0.0;
    let base = near_invisible_splat(&device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);

    // Camera A faces Brush's forward axis (+z, identity rotation); camera B
    // faces -z (180 degree yaw). A camera-local direction bug can't tell
    // these apart (both have local-forward (0,0,1)) and would mix the
    // colours instead of separating them by world direction.
    let batch_a = solid_batch([220, 20, 20], glam::Quat::IDENTITY);
    let batch_b = solid_batch([20, 20, 220], glam::Quat::from_rotation_y(std::f32::consts::PI));

    let mut splats = base.train();
    for _ in 0..300 {
        let (s, _) = trainer.step(batch_a.clone(), splats).await;
        splats = s.train();
        let (s, _) = trainer.step(batch_b.clone(), splats).await;
        splats = s.train();
    }

    let red_dir_rgb = globe_rgb_at(&trainer, glam::Vec3::Z).await;
    let blue_dir_rgb = globe_rgb_at(&trainer, glam::Vec3::NEG_Z).await;

    assert!(
        red_dir_rgb[0] > red_dir_rgb[2] + 0.1,
        "+z dir should be red-dominant, got {red_dir_rgb:?}"
    );
    assert!(
        blue_dir_rgb[2] > blue_dir_rgb[0] + 0.1,
        "-z dir should be blue-dominant, got {blue_dir_rgb:?}"
    );
}

#[tokio::test]
async fn flag_off_builds_no_background_module_or_basis_cache() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    assert!(!config.sh_background, "default is off");
    let base = near_invisible_splat(&device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);

    let batch_a = solid_batch([220, 20, 20], glam::Quat::IDENTITY);
    let (s, _) = trainer.step(batch_a, base.train()).await;
    let _ = s;

    assert!(
        trainer.sh_background.is_none(),
        "flag off must never construct the globe module"
    );
    assert!(
        trainer.sh_pixel_centres.is_none(),
        "flag off must never build the pixel-centre grid"
    );
}
