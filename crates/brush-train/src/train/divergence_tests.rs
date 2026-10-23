//! Robustness against corrupted optimizer state (an aborted GPU command buffer leaves an output unwritten) and
//! against diverged colours: a zeroed Adam second moment must not blow the colours up, and refine prunes splats whose
//! colour left any sensible range.

use super::*;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

/// `n` splats in a row at z = 2, in front of a camera at the origin looking down +z, with DC colour `sh0[i]` on the
/// red channel (others 0.5, cycling through `sh0`).
fn splats_with_sh0(n: usize, sh0: &[f32], device: &Device) -> Splats {
    Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, 2.0]).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![-3.0; n * 3],
        (0..n)
            .flat_map(|i| [sh0[i % sh0.len()], 0.5, 0.5])
            .collect(),
        vec![inverse_sigmoid(0.5); n],
        SplatRenderMode::Default,
        device,
    )
}

fn batch(rotation: glam::Quat) -> SceneBatch {
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
            rotation,
            fov,
            fov,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        ),
    }
}

/// Looks at the splats.
fn seeing() -> SceneBatch {
    batch(glam::Quat::IDENTITY)
}

/// Looks away from every splat: all gradients are zero.
fn blind() -> SceneBatch {
    batch(glam::Quat::from_rotation_y(std::f32::consts::PI))
}

async fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("f32")
}

/// The capture 2026-10-08T23-22-07Z failure: the SH second moment went to zero while the first moment was not, so
/// the next steps of splats no view sees (zero gradient, the second moment stays zero) divided by Adam's 1e-15
/// epsilon. The colours must move at most `MAX_NORMALIZED_STEP · lr` per step.
#[tokio::test]
async fn zeroed_second_moment_keeps_colour_steps_bounded() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let base = splats_with_sh0(50, &[0.5], &device).with_sh_degree(1);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);

    let mut s = base.train();
    for _ in 0..3 {
        let (next, _) = trainer.step(seeing(), s).await;
        s = next;
    }
    let optim = trainer.optim.as_mut().expect("optimizer after a step");
    let m = optim
        .sh_coeffs
        .momentum
        .as_mut()
        .expect("momentum after a step");
    assert!(
        floats(m.moment_1.clone().abs().max()).await[0] > 0.0,
        "the test needs a nonzero first moment"
    );
    m.moment_2 = m.moment_2.clone().zeros_like();

    let before = floats(s.sh_coeffs.val().inner()).await;
    let steps = 5;
    for _ in 0..steps {
        let (next, _) = trainer.step(blind(), s).await;
        s = next;
    }
    let after = floats(s.sh_coeffs.val().inner()).await;
    let moved = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let bound = steps as f32 * crate::adam_scaled::MAX_NORMALIZED_STEP * config.lr_coeffs_dc as f32;
    assert!(
        moved <= bound * 1.001,
        "colour moved {moved}, bound {bound}"
    );
}

/// Refine prunes a splat whose DC colour left the sensible range (|sh0| > `MAX_SH0`) and keeps an ordinary one.
#[tokio::test]
async fn refine_prunes_diverged_colours() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    // Only pruning changes the count: no growth, no force-splits.
    config.growth_grad_threshold = f32::MAX;
    config.split_at_screen_size = 0.0;
    let base = splats_with_sh0(2, &[50.0, 2.0], &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let (s, _) = trainer.step(seeing(), base.train()).await;

    let (s, _) = trainer.refine(1, s.valid()).await;
    let sh = floats(s.sh_coeffs.val()).await;
    let reds: Vec<f32> = sh.chunks(3).map(|c| c[0]).collect();
    assert!(
        reds.iter().all(|r| r.abs() <= 20.0),
        "a diverged colour survived refine: {reds:?}"
    );
    assert!(
        reds.iter().any(|r| (r - 2.0).abs() < 0.1),
        "the ordinary splat was pruned: {reds:?}"
    );
    assert_eq!(trainer.sh_pruned(), 1);
}
