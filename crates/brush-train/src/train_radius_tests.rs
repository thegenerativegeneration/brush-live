//! `train_radius_m`: splats farther than the radius from the training camera get no update in a step, neither their
//! parameters nor their Adam moments nor position noise, and until a step trains them again refine neither decays
//! their opacity nor densifies them; splats within it, and every splat with the radius off, step and refine as before.

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
const FAR_ROWS: std::ops::Range<usize> = NEAR..NEAR + FAR;
/// One near splat below `MIN_OPACITY`, last so pruning it moves no other row: refine replaces it (`num_dead` > 0).
const DEAD: usize = NEAR + FAR;

/// The training camera: far from the world origin and turned, so the radius is measured from a real pose (every
/// splat lies more than 5 m from the origin).
fn pose() -> (glam::Vec3, glam::Quat) {
    (
        glam::vec3(30.0, 2.0, -20.0),
        glam::Quat::from_euler(glam::EulerRot::YXZ, 0.5, 0.15, 0.05),
    )
}

/// A point `local` in the camera's frame (+z ahead) in world space.
fn world(local: glam::Vec3) -> [f32; 3] {
    let (position, rotation) = pose();
    (position + rotation * local).to_array()
}

/// `NEAR` splats 2 m and `FAR` splats 20 m ahead of the camera, all in view, and the `DEAD` one 2 m ahead; SH
/// degree 1 (rest coefficients nonzero). `far_opacity` 0.05 lets the position noise act on the far splats; 0.9 keeps
/// it at zero.
fn splats(far_opacity: f32, device: &Device) -> Splats {
    let near = (0..NEAR).flat_map(|i| world(glam::vec3(i as f32 * 0.1 - 0.45, 0.0, 2.0)));
    let far = (0..FAR).flat_map(|i| world(glam::vec3(i as f32 * 1.0 - 4.5, 0.0, 20.0)));
    let dead = world(glam::vec3(0.0, 0.2, 2.0));
    let n = NEAR + FAR + 1;
    let coeffs = sh_coeffs_for_degree(1) as usize;
    let sh = (0..n * coeffs)
        .flat_map(|k| {
            if k % coeffs == 0 {
                [0.5, 0.4, 0.3]
            } else {
                [0.05, -0.03, 0.02]
            }
        })
        .collect();
    Splats::from_raw(
        near.chain(far).chain(dead).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        [vec![-2.0; NEAR * 3], vec![0.0; FAR * 3], vec![-2.0; 3]].concat(),
        sh,
        [
            vec![inverse_sigmoid(0.9); NEAR],
            vec![inverse_sigmoid(far_opacity); FAR],
            vec![inverse_sigmoid(0.001)],
        ]
        .concat(),
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
    let (position, rotation) = pose();
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::Transparent,
        camera: Camera::new(
            position,
            rotation,
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

    /// The far rows' SH rest coefficients (bands 1+), as bits.
    fn far_sh_rest(&self) -> Vec<u32> {
        let (_, sh, per) = &self.0[1];
        FAR_ROWS
            .flat_map(|row| {
                sh[row * per + 3..(row + 1) * per]
                    .iter()
                    .map(|x| x.to_bits())
            })
            .collect()
    }
}

fn trainer(radius: f32, device: &Device) -> SplatTrainer {
    let mut config = TrainConfig::parse_from(["test"]);
    config.train_radius_m = radius;
    let (position, _) = pose();
    let bounds = BoundingBox::from_min_max(position - 25.0, position + 25.0);
    SplatTrainer::new(&config, device, bounds)
}

async fn device() -> Device {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    device.autodiff()
}

/// A trainer with `radius` after one step with the radius off (so every splat carries Adam moments).
async fn trainer_after_one_step(radius: f32, far_opacity: f32) -> (SplatTrainer, Splats) {
    let device = device().await;
    let mut trainer = trainer(0.0, &device);
    let (splats, _) = trainer
        .step(batch(), splats(far_opacity, &device).train())
        .await;
    trainer.config.train_radius_m = radius;
    (trainer, splats)
}

/// One more step of `trainer`: the state before and after.
async fn step(trainer: &mut SplatTrainer, splats: Splats) -> (State, State, Splats) {
    let before = State::of(trainer, &splats).await;
    let (splats, _) = trainer.step(batch(), splats).await;
    let after = State::of(trainer, &splats).await;
    (before, after, splats)
}

async fn masked_count(trainer: &SplatTrainer) -> Option<i64> {
    let masked = trainer.masked_by_radius()?;
    let data = masked.into_data_async().await.expect("readback");
    Some(data.convert::<i64>().try_to_vec::<i64>().expect("i64")[0])
}

#[tokio::test]
async fn splats_beyond_the_radius_keep_params_and_moments() {
    let (mut on, s_on) = trainer_after_one_step(5.0, 0.9).await;
    let (mut off, s_off) = trainer_after_one_step(0.0, 0.9).await;
    let (before, after, _) = step(&mut on, s_on).await;
    let (twin_before, twin_after, _) = step(&mut off, s_off).await;
    assert_eq!(
        before.rows(FAR_ROWS),
        after.rows(FAR_ROWS),
        "far splats moved"
    );
    assert_eq!(before.far_sh_rest(), after.far_sh_rest());
    for ((name, b), (_, a)) in before.rows(0..NEAR).iter().zip(after.rows(0..NEAR)) {
        assert_ne!(b, &a, "near splats' {name} did not update");
    }
    // Without the radius every far-row tensor changes in this step, so each check above can fail.
    for ((name, b), (_, a)) in twin_before
        .rows(FAR_ROWS)
        .iter()
        .zip(twin_after.rows(FAR_ROWS))
    {
        assert_ne!(b, &a, "radius off: far splats' {name} did not change");
    }
    assert_ne!(twin_before.far_sh_rest(), twin_after.far_sh_rest());
    assert_eq!(masked_count(&on).await, Some(FAR as i64));
}

/// The far splats' step-2 gradient is not zero: with the radius off their first moment is not just the decayed
/// one (m1 = β1 · m1 + (1 − β1) · g).
#[tokio::test]
async fn far_splats_have_a_gradient_in_the_step() {
    let (mut trainer, splats) = trainer_after_one_step(0.0, 0.9).await;
    let (before, after, _) = step(&mut trainer, splats).await;
    let (name, m1_before, per) = &before.0[5];
    let (_, m1_after, _) = &after.0[5];
    assert_eq!(*name, "sh m1");
    let fresh = (NEAR * per..(NEAR + FAR) * per)
        .map(|i| (m1_after[i] - 0.9 * m1_before[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(fresh > 1e-9, "far splats got no gradient ({fresh})");
}

/// Low-opacity far splats get position noise with the radius off; beyond the radius their means keep their bits.
#[tokio::test]
async fn splats_beyond_the_radius_get_no_position_noise() {
    let (mut on, s_on) = trainer_after_one_step(5.0, 0.05).await;
    let (mut off, s_off) = trainer_after_one_step(0.0, 0.05).await;
    let (before, after, _) = step(&mut on, s_on).await;
    let (twin_before, twin_after, _) = step(&mut off, s_off).await;
    let means = |s: &State| -> Vec<u32> {
        let (_, t, per) = &s.0[0];
        FAR_ROWS
            .flat_map(|row| t[row * per..row * per + 3].iter().map(|x| x.to_bits()))
            .collect()
    };
    assert_eq!(means(&before), means(&after), "far means moved");
    assert_ne!(
        means(&twin_before),
        means(&twin_after),
        "radius off: far means did not move"
    );
}

/// Radius 0 builds no mask (the step is the unmasked one, see `adam_scaled`'s `frozen_rows_*` tests for an empty
/// mask); a radius that holds every splat masks none.
#[tokio::test]
async fn radius_off_masks_nothing_and_a_radius_holding_everything_masks_none() {
    let (mut off, s_off) = trainer_after_one_step(0.0, 0.9).await;
    let (mut all, s_all) = trainer_after_one_step(1e4, 0.9).await;
    off.step(batch(), s_off).await;
    all.step(batch(), s_all).await;
    assert!(off.masked_by_radius().is_none());
    assert_eq!(masked_count(&all).await, Some(0));
}

/// The raw opacities and transforms around one refine, and whether the refine record carried the trained column.
struct Refined {
    opac: Vec<f32>,
    opac_after: Vec<f32>,
    transforms: Vec<f32>,
    transforms_after: Vec<f32>,
    trained_column: bool,
    stats: RefineStats,
}

const REFINE_STEPS: u32 = 3;

/// `REFINE_STEPS` steps with `radius` from the start, then every splat's refine weight set far above the growth
/// threshold (as if each had a large gradient) and every on-screen size above `split_at_screen_size`, and one refine
/// growing every candidate. The `DEAD` row is pruned, so the refine also picks a replacement parent.
async fn refine_after_steps(radius: f32) -> Refined {
    let device = device().await;
    let mut trainer = trainer(radius, &device);
    trainer.config.growth_select_fraction = 1.0;
    let mut splats = splats(0.9, &device).train();
    for _ in 0..REFINE_STEPS {
        splats = trainer.step(batch(), splats).await.0;
    }
    let record = trainer.refine_record.as_mut().expect("stepped");
    record.refine_weight_norm = record.refine_weight_norm.clone().ones_like();
    record.max_screen_size = record.max_screen_size.clone().ones_like();
    let trained_column = record.trained.is_some();
    let opac = floats(splats.raw_opacities.val().inner()).await;
    let transforms = floats(splats.transforms.val().inner()).await;
    let (refined, stats) = trainer.refine(REFINE_STEPS, splats.valid()).await;
    Refined {
        opac,
        opac_after: floats(refined.raw_opacities.val()).await,
        transforms,
        transforms_after: floats(refined.transforms.val()).await,
        trained_column,
        stats,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A split parent's opacity after refine as today: the split's `1 − (1 − o)^(1/√2)`, then the opacity decay.
fn split_and_decayed(raw: f32) -> f32 {
    let config = TrainConfig::parse_from(["test"]);
    let split = 1.0 - (1.0 - sigmoid(raw)).powf(FRAC_1_SQRT_2);
    split - config.opac_decay * (1.0 - REFINE_STEPS as f32 / config.total_train_iters as f32)
}

fn assert_split_and_decayed(r: &Refined, rows: std::ops::Range<usize>) {
    for row in rows {
        let (got, want) = (sigmoid(r.opac_after[row]), split_and_decayed(r.opac[row]));
        assert!(
            (got - want).abs() < 1e-4,
            "row {row}: opacity {got}, today's refine gives {want}"
        );
    }
}

/// Rows no step trained since the last refine keep their opacity and transforms and are no split parent (high
/// gradient, oversized or dead-splat replacement), however large their refine weight and screen size; trained rows
/// split and decay as without the radius.
#[tokio::test]
async fn untrained_rows_are_not_decayed_or_densified_at_refine() {
    let r = refine_after_steps(5.0).await;
    assert!(r.trained_column);
    assert!(
        sigmoid(r.opac[DEAD]) < MIN_OPACITY,
        "the dead splat stays below the opacity floor"
    );
    assert_eq!(
        r.stats.num_pruned, 1,
        "the dead splat is pruned and replaced"
    );
    assert!(r.stats.num_split_oversized > 0);
    assert_eq!(
        r.opac_after.len(),
        2 * NEAR + FAR,
        "only the near splats split"
    );
    assert_eq!(
        bits(&r.opac[FAR_ROWS]),
        bits(&r.opac_after[FAR_ROWS]),
        "far opacity changed"
    );
    let far_t = NEAR * 10..(NEAR + FAR) * 10;
    assert_eq!(
        bits(&r.transforms[far_t.clone()]),
        bits(&r.transforms_after[far_t]),
        "far splats split"
    );
    assert_split_and_decayed(&r, 0..NEAR);
}

/// Radius 0: no trained column is built and every splat refines as today (all split, all decay).
#[tokio::test]
async fn radius_off_refines_every_row_as_today() {
    let r = refine_after_steps(0.0).await;
    assert!(!r.trained_column);
    assert_eq!(r.opac_after.len(), 2 * (NEAR + FAR), "every splat splits");
    assert_split_and_decayed(&r, 0..NEAR + FAR);
}

/// Two refines with the Mip-Splatting floor on: the first attaches a floor to every splat; across the second, rows no
/// step trained keep their raw (unbaked) params, their opacity and their floor values bit for bit.
#[tokio::test]
async fn untrained_rows_keep_raw_params_and_floor_across_refine() {
    let device = device().await;
    let mut trainer = trainer(5.0, &device);
    let (position, _) = pose();
    trainer.set_view_cams(vec![(position, 27.7)]);
    let mut splats = trainer.step(batch(), splats(0.9, &device).train()).await.0;
    splats = trainer.refine(1, splats.valid()).await.0.train();
    for _ in 0..REFINE_STEPS {
        splats = trainer.step(batch(), splats).await.0;
    }
    let record = trainer.refine_record.as_mut().expect("stepped");
    record.refine_weight_norm = record.refine_weight_norm.clone().ones_like();
    let splats = splats.valid();
    let floor = |s: &Splats| s.min_scale.clone().expect("floor attached");
    let before = (
        floats(splats.transforms.val()).await,
        floats(splats.raw_opacities.val()).await,
        floats(floor(&splats)).await,
    );
    let (refined, _) = trainer.refine(2, splats).await;
    let after = (
        floats(refined.transforms.val()).await,
        floats(refined.raw_opacities.val()).await,
        floats(floor(&refined)).await,
    );
    let far_t = NEAR * 10..(NEAR + FAR) * 10;
    assert_eq!(
        bits(&before.0[far_t.clone()]),
        bits(&after.0[far_t]),
        "far raw params changed"
    );
    assert_eq!(
        bits(&before.1[FAR_ROWS]),
        bits(&after.1[FAR_ROWS]),
        "far opacity changed"
    );
    assert_eq!(
        bits(&before.2[FAR_ROWS]),
        bits(&after.2[FAR_ROWS]),
        "far floor changed"
    );
    assert!(
        before.2[FAR_ROWS].iter().all(|f| *f > 0.0),
        "far splats carry a floor"
    );
}

/// Refine statistics count what a step trained: a frozen row the camera sees gathers no refine weight, visibility or
/// screen size; with the radius off it does.
#[tokio::test]
async fn frozen_rows_gather_no_densification_stats() {
    for (radius, far_counts) in [(5.0, false), (0.0, true)] {
        let device = device().await;
        let mut trainer = trainer(radius, &device);
        trainer.step(batch(), splats(0.9, &device).train()).await;
        let record = trainer.refine_record.as_ref().expect("stepped");
        let weight = floats(record.refine_weight_norm.clone()).await;
        let vis = floats(record.vis_weight.clone()).await;
        let size = floats(record.max_screen_size.clone()).await;
        for row in FAR_ROWS {
            let counted = weight[row] > 0.0 && vis[row] > 0.0 && size[row] > 0.0;
            let zero = weight[row] == 0.0 && vis[row] == 0.0 && size[row] == 0.0;
            assert!(
                if far_counts { counted } else { zero },
                "radius {radius}, row {row}: {} {} {}",
                weight[row],
                vis[row],
                size[row]
            );
        }
        assert!(
            (0..NEAR).all(|row| vis[row] > 0.0),
            "radius {radius}: near rows unseen"
        );
    }
}
