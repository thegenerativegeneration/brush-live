use super::*;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

fn splats(n: usize, z: f32, device: &Device) -> Splats {
    Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, z]).collect(),
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

#[tokio::test]
async fn append_then_step_and_refine() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let base = splats(50, 2.0, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);

    let (s, _) = trainer.step(batch(), base.train()).await;
    let s = s.valid();
    let s = trainer.append_splats(s, splats(20, 2.5, &device).valid());
    assert_eq!(s.num_splats(), 70);

    let (s, _) = trainer.step(batch(), s.train()).await;
    let (s, _) = trainer.refine(1, s.valid()).await;
    assert!(s.num_splats() >= 1);
}

#[tokio::test]
async fn append_before_first_step() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let base = splats(10, 2.0, &device).valid();
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let s = trainer.append_splats(base, splats(5, 2.0, &device).valid());
    assert_eq!(s.num_splats(), 15);
    let (s, _) = trainer.step(batch(), s.train()).await;
    assert_eq!(s.num_splats(), 15);
}

#[tokio::test]
async fn append_keeps_mip_floor() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let base = splats(50, 2.0, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.set_view_cams(vec![(glam::Vec3::ZERO, 30.0)]);

    let (s, _) = trainer.step(batch(), base.train()).await;
    let (s, _) = trainer.refine(1, s.valid()).await;
    let pre_count = s.num_splats() as usize;
    let pre_floor = s
        .min_scale
        .clone()
        .expect("refine with view cams should attach a Mip floor")
        .into_data_async()
        .await
        .expect("floor readback")
        .try_into_vec::<f32>()
        .expect("floor readback");

    let s = trainer.append_splats(s, splats(20, 2.5, &device).valid());
    assert_eq!(s.num_splats() as usize, pre_count + 20);

    let floor = s
        .min_scale
        .clone()
        .expect("Mip floor should survive append_splats");
    assert_eq!(floor.dims()[0], pre_count + 20);
    let floor_vals = floor
        .into_data_async()
        .await
        .expect("floor readback")
        .try_into_vec::<f32>()
        .expect("floor readback");
    assert_eq!(&floor_vals[..pre_count], &pre_floor[..]);

    let (s, _) = trainer.step(batch(), s.train()).await;
    assert_eq!(s.num_splats() as usize, pre_count + 20);
}
