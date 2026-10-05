//! Eviction through the refine/prune path: counts, which splats go, and
//! that optimizer and per-splat state stay row-aligned with the splats.

use super::*;
use crate::evict::{EvictConfig, ProtectCone};
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::module::Module;
use clap::Parser;

const N: usize = 60;

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

/// A trainer at a 60-splat budget with growth and force-splits off, so a
/// refine changes the count only by pruning; one step taken.
async fn trainer_at_budget(min_age: u32, external: bool) -> (SplatTrainer, Splats) {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.max_splats = N as u32;
    config.growth_grad_threshold = f32::MAX;
    config.split_at_screen_size = 0.0;
    let base = splats(N, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age,
        max_cell_fraction: 1.0,
        recent_refines: 2,
        external_importance: external,
    });
    let (s, _) = trainer.step(batch(), base.train()).await;
    (trainer, s.valid())
}

async fn rows<const D: usize>(t: Tensor<D>) -> Vec<Vec<f32>> {
    let n = t.dims()[0];
    let v = t
        .into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("f32");
    let w = v.len() / n.max(1);
    v.chunks(w).map(<[f32]>::to_vec).collect()
}

/// A permutation of 0..N, so the lowest scores are not the first rows.
fn importance() -> Vec<f32> {
    (0..N).map(|i| ((i * 37) % N) as f32).collect()
}

#[tokio::test]
async fn eviction_removes_lowest_and_keeps_adam_rows_aligned() {
    let (mut trainer, s) = trainer_at_budget(0, true).await;
    trainer.set_importance(&importance());
    trainer.note_seed_shortfall(1);

    let optim = trainer.optim.as_ref().expect("optimizer after a step");
    let m = optim.transforms.momentum.as_ref().expect("momentum");
    let before_m1 = rows(m.moment_1.clone()).await;
    let before_m2 = rows(m.moment_2.clone()).await;
    let om = optim.opacities.momentum.as_ref().expect("momentum");
    let before_om1 = rows(om.moment_1.clone().unsqueeze_dim::<2>(1)).await;
    let before_means = rows(s.means()).await;

    let (s, stats) = trainer.refine(1, s).await;

    // Budget 60, headroom 10 %: evict down to 54.
    assert_eq!(stats.num_evicted, 6);
    assert_eq!(stats.num_pruned, 6);
    assert_eq!(stats.num_added, 0);
    assert_eq!(s.num_splats() as usize, N - 6);

    let imp = importance();
    let kept: Vec<usize> = (0..N).filter(|&i| imp[i] >= 6.0).collect();
    assert_eq!(kept.len(), N - 6);

    let optim = trainer.optim.as_ref().expect("optimizer");
    let m = optim.transforms.momentum.as_ref().expect("momentum");
    let after_m1 = rows(m.moment_1.clone()).await;
    let after_m2 = rows(m.moment_2.clone()).await;
    let om = optim.opacities.momentum.as_ref().expect("momentum");
    let after_om1 = rows(om.moment_1.clone().unsqueeze_dim::<2>(1)).await;
    let sh = optim.sh_coeffs.momentum.as_ref().expect("momentum");
    assert_eq!(sh.moment_1.dims()[0], N - 6);
    assert_eq!(sh.moment_2.dims()[0], N - 6);
    let after_means = rows(s.means()).await;
    for (row, &src) in kept.iter().enumerate() {
        assert_eq!(after_m1[row], before_m1[src], "transform m1 row {row}");
        assert_eq!(after_m2[row], before_m2[src], "transform m2 row {row}");
        assert_eq!(after_om1[row], before_om1[src], "opacity m1 row {row}");
        assert_eq!(after_means[row], before_means[src], "mean row {row}");
    }

    let life = trainer
        .evict
        .as_ref()
        .and_then(|e| e.life.as_ref())
        .expect("life");
    assert_eq!(life.len(), N - 6);
    let after_imp = rows(life.importance.clone().unsqueeze_dim::<2>(1)).await;
    let expect: Vec<Vec<f32>> = kept.iter().map(|&i| vec![imp[i]]).collect();
    assert_eq!(after_imp, expect);

    // Training continues on the smaller model.
    let (s, _) = trainer.step(batch(), s.train()).await;
    assert_eq!(s.num_splats() as usize, N - 6);
}

/// No eviction: no demand, blocked demand without a new keyframe, and a keyframe whose seeds land elsewhere.
#[tokio::test]
async fn no_eviction_without_demand() {
    let (mut trainer, s) = trainer_at_budget(0, true).await;
    trainer.set_importance(&importance());
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 0);
    assert_eq!(s.num_splats() as usize, N);

    let (mut trainer, s) = split_trainer_at_budget().await;
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(
        stats.num_evicted, 0,
        "blocked demand without a new keyframe"
    );
    assert_eq!(stats.num_split_oversized, 0);
    assert_eq!(s.num_splats() as usize, N);

    let (mut trainer, s) = split_trainer_at_budget().await;
    // Seeds in a far cell: the blocked demand is all in the old cell.
    trainer.note_keyframe_seeds(&[5.5, 5.5, 5.5].repeat(16));
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 0, "keyframe elsewhere");
    assert_eq!(s.num_splats() as usize, N);
}

#[tokio::test]
async fn young_and_unscored_splats_are_protected() {
    // Age 1 after this refine, below min_age 2.
    let (mut trainer, s) = trainer_at_budget(2, true).await;
    trainer.set_importance(&importance());
    trainer.note_seed_shortfall(1);
    let (_, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 0);

    // Never scored: importance stays +inf.
    let (mut trainer, s) = trainer_at_budget(0, true).await;
    trainer.note_seed_shortfall(1);
    let (_, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 0);
}

#[tokio::test]
async fn splats_in_the_protect_cone_are_kept() {
    let (mut trainer, s) = trainer_at_budget(0, true).await;
    // Ascending importance: without the cone, splats 0..6 would go.
    trainer.set_importance(&(0..N).map(|i| i as f32).collect::<Vec<_>>());
    // Splats sit at x = 0.01·i, z = 2. A cone from the origin along +z with
    // half angle atan(0.195 / 2) covers i < 20.
    trainer.set_protect_cone(Some(ProtectCone {
        position: glam::Vec3::ZERO,
        forward: glam::Vec3::Z,
        cos_half_fov: (0.195f32 / 2.0).atan().cos(),
    }));
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 6);
    let xs: Vec<f32> = rows(s.means()).await.iter().map(|r| r[0]).collect();
    let expect: Vec<f32> = (0..N)
        .filter(|i| !(20..26).contains(i))
        .map(|i| i as f32 * 0.01)
        .collect();
    assert_eq!(xs, expect);
}

#[tokio::test]
async fn eviction_then_split_fills_to_growth_limit() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.max_splats = N as u32;
    config.growth_grad_threshold = f32::MAX;
    // Every visible splat counts as oversized, so splitting wants all of them.
    config.split_at_screen_size = 1e-6;
    let base = splats(N, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
        external_importance: true,
    });
    let (s, _) = trainer.step(batch(), base.train()).await;
    trainer.set_importance(&importance());
    // A keyframe seeded the splats' cell, so their split demand is new.
    trainer.note_keyframe_seeds(&rows(s.means()).await.concat());

    let (s, stats) = trainer.refine(1, s.valid()).await;
    // 60 → evict to 54 → split up to the growth limit 57.
    assert_eq!(stats.num_evicted, 6);
    assert_eq!(stats.num_split_oversized, 3);
    assert_eq!(s.num_splats(), 57);

    let optim = trainer.optim.as_ref().expect("optimizer");
    let m = optim.transforms.momentum.as_ref().expect("momentum");
    assert_eq!(m.moment_1.dims()[0], 57);
    let life = trainer
        .evict
        .as_ref()
        .and_then(|e| e.life.as_ref())
        .expect("life");
    assert_eq!(life.len(), 57);
    let imp = rows(life.importance.clone().unsqueeze_dim::<2>(1)).await;
    let age = rows(life.age.clone().unsqueeze_dim::<2>(1)).await;
    // Children are appended young and inherit a surviving parent's importance.
    for row in 54..57 {
        assert_eq!(age[row], vec![0.0]);
        assert!(imp[row][0] >= 6.0 && imp[..54].contains(&imp[row]));
    }
    assert!(age[..54].iter().all(|a| a[0] == 1.0));

    let (s, _) = trainer.step(batch(), s.train()).await;
    assert_eq!(s.num_splats(), 57);
}

#[tokio::test]
async fn evicts_once_per_importance_set_and_nan_keeps_scores() {
    let (mut trainer, s) = trainer_at_budget(0, true).await;
    trainer.set_importance(&importance());
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 6);

    // Still blocked, but no new scores since that eviction: wait.
    let (s, _) = trainer.step(batch(), s.train()).await;
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(2, s.valid()).await;
    assert_eq!(stats.num_evicted, 0);
    assert_eq!(s.num_splats() as usize, N - 6);

    // A set that observed nothing keeps the previous scores and re-arms.
    trainer.set_importance(&[f32::NAN; N - 6]);
    let life = trainer
        .evict
        .as_ref()
        .and_then(|e| e.life.as_ref())
        .expect("life");
    let kept = rows(life.importance.clone().unsqueeze_dim::<2>(1)).await;
    let imp = importance();
    let expect: Vec<Vec<f32>> = (0..N)
        .filter(|&i| imp[i] >= 6.0)
        .map(|i| vec![imp[i]])
        .collect();
    assert_eq!(kept, expect);
    let (s, _) = trainer.step(batch(), s.train()).await;
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(3, s.valid()).await;
    // 54 is the target already: nothing to evict, even when blocked.
    assert_eq!(stats.num_evicted, 0);
    assert_eq!(s.num_splats() as usize, N - 6);
}

/// Like [`eviction_then_split_fills_to_growth_limit`]: every splat wants a
/// force-split, the model is at its 60-splat budget and scored.
async fn split_trainer_at_budget() -> (SplatTrainer, Splats) {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.max_splats = N as u32;
    config.growth_grad_threshold = f32::MAX;
    config.split_at_screen_size = 1e-6;
    let base = splats(N, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
        external_importance: true,
    });
    let (s, _) = trainer.step(batch(), base.train()).await;
    trainer.set_importance(&importance());
    (trainer, s.valid())
}

#[tokio::test]
async fn evictions_stop_once_the_keyframe_window_passes() {
    // Cells stay recent for 2 refines; every refine has fresh scores.
    let (mut trainer, s) = split_trainer_at_budget().await;
    trainer.note_keyframe_seeds(&rows(s.means()).await.concat());
    let (mut s, stats) = trainer.refine(1, s).await;
    // 60 → 54, split back to the growth limit 57.
    assert_eq!(stats.num_evicted, 6);
    for iter in 2..5 {
        let (stepped, _) = trainer.step(batch(), s.train()).await;
        let n = stepped.num_splats() as usize;
        trainer.set_importance(&vec![1.0; n]);
        let (next, stats) = trainer.refine(iter, stepped.valid()).await;
        // Still in the window: 57 → 54 → 57. Then steady.
        let expect = if iter == 2 { 3 } else { 0 };
        assert_eq!(stats.num_evicted, expect, "refine {iter}");
        assert_eq!(next.num_splats(), 57);
        s = next;
    }
}

/// Overwrites the record's importance totals as if every splat was seen once with score `imp[i]`
/// (`f32::NAN` -> not seen this window).
fn record_importance(trainer: &mut SplatTrainer, imp: &[f32]) {
    let rec = trainer.refine_record.as_mut().expect("record after a step");
    let device = rec.vis_weight.device();
    let seen: Vec<f32> = imp
        .iter()
        .map(|x| if x.is_nan() { 0.0 } else { 1.0 })
        .collect();
    let sum: Vec<f32> = imp
        .iter()
        .map(|x| if x.is_nan() { 0.0 } else { *x })
        .collect();
    rec.importance_sum = Tensor::from_data(TensorData::new(sum, [imp.len()]), &device);
    rec.importance_views = Tensor::from_data(TensorData::new(seen, [imp.len()]), &device);
}

async fn life_importance(trainer: &SplatTrainer) -> Vec<f32> {
    trainer.importance().await.expect("eviction state")
}

#[tokio::test]
async fn a_step_records_importance_for_visible_splats() {
    let (trainer, _) = trainer_at_budget(0, false).await;
    let rec = trainer.refine_record.as_ref().expect("record");
    let views = rows(rec.importance_views.clone().unsqueeze_dim::<2>(1)).await;
    let sum = rows(rec.importance_sum.clone().unsqueeze_dim::<2>(1)).await;
    assert!(
        views.iter().any(|v| v[0] == 1.0),
        "some splat seen in the one step"
    );
    for (v, s) in views.iter().zip(&sum) {
        assert_eq!(
            v[0] > 0.0,
            s[0] > 0.0,
            "views count exactly the steps with a score"
        );
    }
}

#[tokio::test]
async fn recorded_importance_drives_eviction_without_any_external_call() {
    let (mut trainer, s) = trainer_at_budget(0, false).await;
    record_importance(&mut trainer, &importance());
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 6);
    let imp = importance();
    let kept: Vec<f32> = (0..N).filter(|&i| imp[i] >= 6.0).map(|i| imp[i]).collect();
    assert_eq!(life_importance(&trainer).await, kept);
    assert_eq!(s.num_splats() as usize, N - 6);
}

#[tokio::test]
async fn importance_is_the_mean_over_seeing_views_and_unseen_splats_keep_theirs() {
    let (mut trainer, s) = trainer_at_budget(0, false).await;
    {
        let rec = trainer.refine_record.as_mut().unwrap();
        let device = rec.vis_weight.device();
        let sum: Vec<f32> = (0..N)
            .map(|i| if i == 0 { 0.0 } else { 2.0 * i as f32 })
            .collect();
        let views: Vec<f32> = (0..N).map(|i| if i == 0 { 0.0 } else { 2.0 }).collect();
        rec.importance_sum = Tensor::from_data(TensorData::new(sum, [N]), &device);
        rec.importance_views = Tensor::from_data(TensorData::new(views, [N]), &device);
    }
    let (s, _) = trainer.refine(1, s).await;
    let first = life_importance(&trainer).await;
    assert_eq!(first[5], 5.0, "sum / views");
    assert!(first[0].is_infinite(), "never seen -> still unscored");
    let (s, _) = trainer.step(batch(), s.train()).await;
    let mut second = vec![f32::NAN; N];
    second[7] = 1.0;
    record_importance(&mut trainer, &second);
    let _ = trainer.refine(2, s.valid()).await;
    let after = life_importance(&trainer).await;
    assert_eq!(after[5], 5.0, "unseen this window -> keeps its score");
    assert_eq!(after[7], 1.0);
}

#[tokio::test]
async fn never_scored_splats_are_not_evicted() {
    let (mut trainer, s) = trainer_at_budget(0, false).await;
    record_importance(&mut trainer, &[f32::NAN; N]);
    trainer.note_seed_shortfall(1);
    let (_, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 0);
}

#[tokio::test]
async fn record_rows_stay_aligned_through_pad_and_keep() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let mut rec = RefineRecord::new(3, &device);
    rec.importance_sum = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    rec.importance_views = Tensor::from_floats([1.0, 1.0, 2.0], &device);
    let rec = rec.pad(2).keep(Tensor::from_ints([4, 2, 0], &device));
    assert_eq!(
        rows(rec.importance_sum.unsqueeze_dim::<2>(1)).await,
        vec![vec![0.0], vec![3.0], vec![1.0]]
    );
    assert_eq!(
        rows(rec.importance_views.unsqueeze_dim::<2>(1)).await,
        vec![vec![0.0], vec![2.0], vec![1.0]]
    );
}
