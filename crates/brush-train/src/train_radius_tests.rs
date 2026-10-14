//! `train_radius_m`: splats farther than the radius from the training camera get no update in a step, neither their
//! parameters nor their Adam moments; splats within it, and every splat with the radius off, step as before.

use super::*;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

const NEAR: usize = 10;
const FAR: usize = 10;

/// `NEAR` splats 2 m and `FAR` splats 20 m in front of a camera at the origin looking down +z, all in view. Opacity
/// 0.9 keeps the position noise at zero.
fn splats(device: &Device) -> Splats {
    let near = (0..NEAR).flat_map(|i| [i as f32 * 0.1 - 0.45, 0.0, 2.0]);
    let far = (0..FAR).flat_map(|i| [i as f32 * 1.0 - 4.5, 0.0, 20.0]);
    Splats::from_raw(
        near.chain(far).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(NEAR + FAR),
        [vec![-2.0; NEAR * 3], vec![0.0; FAR * 3]].concat(),
        vec![0.5; (NEAR + FAR) * 3],
        vec![inverse_sigmoid(0.9); NEAR + FAR],
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

async fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("f32")
}

/// Parameters and Adam moments of one step's outcome, each as (name, values, values per splat).
struct State(Vec<(&'static str, Vec<f32>, usize)>);

impl State {
    async fn of(trainer: &SplatTrainer, splats: &Splats) -> Self {
        let n = splats.num_splats() as usize;
        let optim = trainer.optim.as_ref().expect("stepped");
        let mut out = Vec::new();
        let mut add = |name, v: Vec<f32>| {
            let per = v.len() / n;
            out.push((name, v, per));
        };
        add("transforms", floats(splats.transforms.val().inner()).await);
        add("sh", floats(splats.sh_coeffs.val().inner()).await);
        add("opacity", floats(splats.raw_opacities.val().inner()).await);
        let t = optim.transforms.momentum.as_ref().expect("momentum");
        add("transforms m1", floats(t.moment_1.clone()).await);
        add("transforms m2", floats(t.moment_2.clone()).await);
        let s = optim.sh_coeffs.momentum.as_ref().expect("momentum");
        add("sh m1", floats(s.moment_1.clone()).await);
        add("sh m2", floats(s.moment_2.clone()).await);
        let o = optim.opacities.momentum.as_ref().expect("momentum");
        add("opacity m1", floats(o.moment_1.clone()).await);
        add("opacity m2", floats(o.moment_2.clone()).await);
        Self(out)
    }

    /// Bit patterns of `rows` of every tensor.
    fn rows(&self, rows: std::ops::Range<usize>) -> Vec<(&'static str, Vec<u32>)> {
        self.0
            .iter()
            .map(|(name, v, per)| {
                (
                    *name,
                    v[rows.start * per..rows.end * per]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect(),
                )
            })
            .collect()
    }
}

/// A trainer with `radius` after one step with the radius off (so every splat carries Adam moments).
async fn trainer_after_one_step(radius: f32) -> (SplatTrainer, Splats) {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(["test"]);
    let bounds = BoundingBox::from_min_max(glam::Vec3::splat(-5.0), glam::Vec3::splat(20.0));
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let (splats, _) = trainer.step(batch(), splats(&device).train()).await;
    trainer.config.train_radius_m = radius;
    (trainer, splats)
}

#[tokio::test]
async fn splats_beyond_the_radius_keep_params_and_moments() {
    let (mut trainer, splats) = trainer_after_one_step(5.0).await;
    let before = State::of(&trainer, &splats).await;
    let (splats, _) = trainer.step(batch(), splats).await;
    let after = State::of(&trainer, &splats).await;
    let far = NEAR..NEAR + FAR;
    assert_eq!(
        before.rows(far.clone()),
        after.rows(far),
        "far splats moved"
    );
    for ((name, b), (_, a)) in before.rows(0..NEAR).iter().zip(after.rows(0..NEAR)) {
        assert_ne!(b, &a, "near splats' {name} did not update");
    }
    let masked = trainer.masked_by_radius().expect("radius on");
    let masked: Vec<i64> = masked
        .into_data_async()
        .await
        .expect("readback")
        .convert::<i64>()
        .try_to_vec()
        .expect("i64");
    assert_eq!(masked, vec![FAR as i64]);
}

/// The far splats' step-2 gradient is not zero: with the radius off their first moment is not just the decayed
/// one (m1 = β1 · m1 + (1 − β1) · g).
#[tokio::test]
async fn far_splats_have_a_gradient_in_the_step() {
    let (mut trainer, splats) = trainer_after_one_step(0.0).await;
    let before = State::of(&trainer, &splats).await;
    let (splats, _) = trainer.step(batch(), splats).await;
    let after = State::of(&trainer, &splats).await;
    let (name, m1_before, per) = &before.0[5];
    let (_, m1_after, _) = &after.0[5];
    assert_eq!(*name, "sh m1");
    let fresh = (NEAR * per..(NEAR + FAR) * per)
        .map(|i| (m1_after[i] - 0.9 * m1_before[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(fresh > 1e-9, "far splats got no gradient ({fresh})");
}

/// Radius 0 builds no mask (the step is the unmasked one, see `adam_scaled`'s `frozen_rows_*` tests for the bit
/// identity of an empty mask); a radius that holds every splat masks none.
#[tokio::test]
async fn radius_off_masks_nothing_and_a_radius_holding_everything_masks_none() {
    let (mut off, s_off) = trainer_after_one_step(0.0).await;
    let (mut all, s_all) = trainer_after_one_step(1e4).await;
    off.step(batch(), s_off).await;
    all.step(batch(), s_all).await;
    assert!(off.masked_by_radius().is_none());
    let masked = all.masked_by_radius().expect("radius on");
    let masked = masked
        .into_data_async()
        .await
        .expect("readback")
        .convert::<i64>();
    assert_eq!(masked.try_to_vec::<i64>().expect("i64"), vec![0]);
}
