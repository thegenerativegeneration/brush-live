use super::super::*;
use crate::evict::EvictConfig;
use crate::rows::SplatRows;
use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, fold_min_scale, inverse_sigmoid};
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

pub(super) fn batch() -> SceneBatch {
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

/// A trainer built from `args` (after the program name) with eviction, origins and, when `view_cams`, a Mip floor,
/// after one step and one refine on `n` splats 2 m ahead of the camera.
pub(super) async fn trained_with(n: usize, args: &[&str], view_cams: bool) -> (SplatTrainer, Splats) {
    trained_on(|device| splats(n, 2.0, device), args, view_cams).await
}

/// [`trained_with`] on the splats `base` builds.
pub(super) async fn trained_on(
    base: impl FnOnce(&Device) -> Splats,
    args: &[&str],
    view_cams: bool,
) -> (SplatTrainer, Splats) {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let config = TrainConfig::parse_from(std::iter::once("test").chain(args.iter().copied()));
    let base = base(&device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    if view_cams {
        trainer.set_view_cams(vec![(glam::Vec3::ZERO, 30.0)]);
    }
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age: 3,
        max_cell_fraction: 0.3,
        recent_refines: 10,
    });
    trainer.track_origins(&base, 0);
    let (s, _) = trainer.step(batch(), base.train()).await;
    let (s, _) = trainer.refine(1, s.valid()).await;
    (trainer, s)
}

/// A trainer with eviction, origins and a Mip floor after one step and one refine.
async fn trained(n: usize) -> (SplatTrainer, Splats) {
    trained_with(n, &[], true).await
}

#[tokio::test]
async fn take_then_append_keeps_every_row_and_training_runs() {
    let (mut trainer, s) = trained(60).await;
    let before = trainer.read_rows(&s).await;
    assert_eq!(
        before.min_scale.len(),
        before.len(),
        "refine attached a floor"
    );
    // A step before the take, so the Adam moments are nonzero.
    let (s, _) = trainer.step(batch(), s.train()).await;
    let s = s.valid();
    let before = trainer.read_rows(&s).await;
    let moment_before = opacity_moment(&trainer).await;
    let n = before.len();
    let take: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let (s, taken) = trainer.take_rows(s, &take).await;
    let taken = taken.expect("a third of the rows");
    let (idx, kept): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| take[i]);
    assert_eq!(taken, before.select(&idx));
    assert_eq!(s.num_splats() as usize, kept.len());
    assert_eq!(
        s.min_scale.as_ref().map(|f| f.dims()[0]),
        Some(s.num_splats() as usize)
    );
    let s = trainer.append_rows(s, &taken);
    let after = trainer.read_rows(&s).await;
    let k = kept.len();
    assert_eq!(
        after.select(&(0..k).collect::<Vec<_>>()),
        before.select(&kept)
    );
    assert_eq!(after.select(&(k..n).collect::<Vec<_>>()), taken);
    let moment_after = opacity_moment(&trainer).await;
    let kept_moments: Vec<f32> = kept.iter().map(|&i| moment_before[i]).collect();
    assert!(
        kept_moments.iter().any(|&m| m != 0.0),
        "the step left moments to keep"
    );
    assert_eq!(&moment_after[..k], &kept_moments[..]);
    let taken_moments: Vec<f32> = idx.iter().map(|&i| moment_before[i]).collect();
    assert_eq!(&moment_after[k..], &taken_moments[..], "appended rows keep their moments");
    // Moments, refine record, life and origin tags stay row-aligned: a step and a refine run on the result.
    let (s, _) = trainer.step(batch(), s.train()).await;
    let (s, _) = trainer.refine(2, s.valid()).await;
    assert!(s.num_splats() > 0);
}

/// Taking every row is refused (the trainer keeps them all); taking none returns nothing.
#[tokio::test]
async fn taking_every_row_is_refused() {
    let (mut trainer, s) = trained(20).await;
    let n = s.num_splats() as usize;
    let (s, taken) = trainer.take_rows(s, &vec![true; n]).await;
    assert!(taken.is_none());
    assert_eq!(s.num_splats() as usize, n);
    let (s, taken) = trainer.take_rows(s, &vec![false; n]).await;
    assert!(taken.is_none() && s.num_splats() as usize == n);
}

/// Between refines the refine record exists; taking rows must shrink it with the splats.
#[tokio::test]
async fn take_between_steps_keeps_the_refine_record_aligned() {
    let (mut trainer, s) = trained(40).await;
    let (s, _) = trainer.step(batch(), s.train()).await;
    let n = s.num_splats() as usize;
    let (s, taken) = trainer
        .take_rows(s.valid(), &(0..n).map(|i| i < 10).collect::<Vec<_>>())
        .await;
    assert_eq!(taken.unwrap().len(), 10);
    assert_eq!(trainer.seen_since_refine().unwrap().dims()[0], n - 10);
    let (s, _) = trainer.step(batch(), s.train()).await;
    let (s, _) = trainer.refine(2, s.valid()).await;
    assert!(s.num_splats() > 0);
}

/// The visibility signal area training reads: splats in front of the camera contributed to the step's render, and
/// splats behind it did not.
#[tokio::test]
async fn seen_counts_splats_that_contributed_to_the_render() {
    let both_sides = |device: &Device| {
        let n = 40;
        Splats::from_raw(
            (0..n)
                .flat_map(|i| [i as f32 * 0.01, 0.0, if i % 2 == 0 { 2.0 } else { -2.0 }])
                .collect(),
            [1.0, 0.0, 0.0, 0.0].repeat(n),
            vec![-3.0; n * 3],
            vec![0.5; n * 3],
            vec![inverse_sigmoid(0.5); n],
            SplatRenderMode::Default,
            device,
        )
    };
    let (mut trainer, s) = trained_on(both_sides, &[], true).await;
    let (s, _) = trainer.step(batch(), s.train()).await;
    let rows = trainer.read_rows(&s.valid()).await;
    let seen: Vec<f32> = trainer
        .seen_since_refine()
        .unwrap()
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec()
        .unwrap();
    assert_eq!(seen.len(), rows.len());
    let (ahead, behind): (Vec<usize>, Vec<usize>) =
        (0..rows.len()).partition(|&i| rows.mean(i).z > 0.0);
    assert!(!ahead.is_empty() && !behind.is_empty());
    assert!(
        ahead.iter().any(|&i| seen[i] > 0.0),
        "importance_views must count visible splats"
    );
    assert!(
        behind.iter().all(|&i| seen[i] == 0.0),
        "splats behind the camera are never seen"
    );
}

#[tokio::test]
async fn appended_rows_keep_their_origins_age_and_importance() {
    let (mut trainer, s) = trained(30).await;
    let mut rows = trainer.read_rows(&s).await.select(&[0, 1, 2]);
    rows.origin = vec![Some(7), None, Some(2)];
    rows.age = vec![5.0, 6.0, 7.0];
    rows.importance = vec![0.25, f32::INFINITY, 1.5];
    let s = trainer.append_rows(s, &rows);
    let all = trainer.read_rows(&s).await;
    let n = all.len();
    assert_eq!(&all.origin[n - 3..], &rows.origin[..]);
    assert_eq!(&all.age[n - 3..], &rows.age[..]);
    assert_eq!(&all.importance[n - 3..], &rows.importance[..]);
}

/// Restored rows keep the floor they were parked with, not one recomputed from the current views; rows parked
/// without a floor get a zero one, which leaves them as they were.
#[tokio::test]
async fn appended_rows_keep_their_stored_floor() {
    let (mut trainer, s) = trained(30).await;
    let mut rows = trainer.read_rows(&s).await.select(&[0, 1, 2]);
    rows.min_scale = vec![0.003, 0.02, 0.0];
    let s = trainer.append_rows(s, &rows);
    let all = trainer.read_rows(&s).await;
    let n = all.len();
    assert_eq!(&all.min_scale[n - 3..], &rows.min_scale[..]);
    assert_eq!(&all.log_scales[(n - 3) * 3..], &rows.log_scales[..]);

    rows.min_scale.clear();
    let s = trainer.append_rows(s, &rows);
    let all = trainer.read_rows(&s).await;
    assert_eq!(&all.min_scale[n..], &[0.0; 3]);
}

/// Splats without a floor take restored rows with their floor folded in, as the bake on the host gives it.
#[tokio::test]
async fn appended_rows_are_baked_into_splats_without_a_floor() {
    let (mut trainer, s) = trained_with(30, &[], false).await;
    assert!(s.min_scale.is_none(), "no view cameras, no floor");
    let mut rows = trainer.read_rows(&s).await.select(&[0, 1, 2]);
    rows.min_scale = vec![0.01, 0.002, 0.05];
    let s = trainer.append_rows(s, &rows);
    let all = trainer.read_rows(&s).await;
    assert!(all.min_scale.is_empty());
    let n = all.len();
    let tail = all.select(&[n - 3, n - 2, n - 1]);
    let host = rows.baked();
    for (a, b) in tail.log_scales.iter().zip(&host.log_scales) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    for (a, b) in tail.raw_opacities.iter().zip(&host.raw_opacities) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

/// With the training radius on the refine record carries the trained column and the trainer the last step's frozen count;
/// taking and appending rows keeps every per-row tensor as long as the splats, and training runs on.
#[tokio::test]
async fn take_and_append_keep_rows_aligned_with_the_radius_on() {
    // Splats lie 2.0–2.07 m from the camera: rows past x = 0.45 are beyond the radius.
    let (mut trainer, s) = trained_with(60, &["--train-radius-m", "2.05"], true).await;
    let (s, _) = trainer.step(batch(), s.train()).await;
    let s = s.valid();
    assert!(
        trainer.refine_record.as_ref().unwrap().trained.is_some(),
        "the radius builds the trained column"
    );
    let n = s.num_splats() as usize;
    let take: Vec<bool> = (0..n).map(|i| i % 4 == 1).collect();
    let (s, taken) = trainer.take_rows(s, &take).await;
    let taken = taken.expect("a quarter of the rows");
    assert_rows_aligned(&trainer, &s);
    let s = trainer.append_rows(s, &taken);
    assert_eq!(s.num_splats() as usize, n);
    assert_rows_aligned(&trainer, &s);
    let trained: Vec<f32> = trainer
        .refine_record
        .as_ref()
        .unwrap()
        .trained
        .clone()
        .unwrap()
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec()
        .unwrap();
    assert!(
        trained[n - taken.len()..].iter().all(|&t| t == 0.0),
        "appended rows start untrained"
    );
    let (s, _) = trainer.step(batch(), s.train()).await;
    let (s, _) = trainer.refine(2, s.valid()).await;
    assert!(s.num_splats() > 0);
    assert_rows_aligned(&trainer, &s);
}

/// `masked_by_radius` is the last step's count; taking and appending rows afterwards leaves it as it was.
#[tokio::test]
async fn masked_count_survives_take_and_append() {
    // Splats lie 2.0-2.07 m from the camera: rows past x = 0.45 (i > 45) are beyond the radius.
    let (mut trainer, s) = trained_with(60, &["--train-radius-m", "2.05"], true).await;
    let (s, _) = trainer.step(batch(), s.train()).await;
    let s = s.valid();
    let count = |t: &SplatTrainer| {
        let c = t.masked_by_radius().expect("radius on, a step ran");
        async move {
            let data = c.into_data_async().await.unwrap();
            data.convert::<i64>().try_to_vec::<i64>().unwrap()[0]
        }
    };
    let c = count(&trainer).await;
    assert!(c > 4, "rows beyond the radius: {c}");
    let n = s.num_splats() as usize;
    // Ten rows, four of them beyond the radius.
    let take: Vec<bool> = (0..n).map(|i| (40..50).contains(&i)).collect();
    let (s, taken) = trainer.take_rows(s, &take).await;
    assert_eq!(count(&trainer).await, c, "after take");
    let s = trainer.append_rows(s, &taken.expect("ten rows"));
    assert_eq!(count(&trainer).await, c, "after append");
    let take: Vec<bool> = (0..n).map(|i| i < 5).collect();
    let (_, taken) = trainer.take_rows(s, &take).await;
    assert!(taken.is_some());
    assert_eq!(count(&trainer).await, c, "after the second take");
}

/// The opacity parameter's first Adam moment, per splat.
async fn opacity_moment(trainer: &SplatTrainer) -> Vec<f32> {
    let optim = trainer.optim.as_ref().expect("a step ran");
    let m = optim.opacities.momentum.as_ref().expect("a step ran");
    m.moment_1
        .clone()
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec()
        .unwrap()
}

/// Every per-row tensor the trainer holds is as long as `splats`.
fn assert_rows_aligned(trainer: &SplatTrainer, splats: &Splats) {
    let n = splats.num_splats() as usize;
    if let Some(f) = &splats.min_scale {
        assert_eq!(f.dims()[0], n, "floor");
    }
    if let Some(r) = &trainer.refine_record {
        for (name, t) in [
            ("refine_weight_norm", Some(&r.refine_weight_norm)),
            ("vis_weight", Some(&r.vis_weight)),
            ("max_screen_size", Some(&r.max_screen_size)),
            ("importance_sum", Some(&r.importance_sum)),
            ("importance_views", Some(&r.importance_views)),
            ("trained", r.trained.as_ref()),
            ("kept_floor", r.kept_floor.as_ref()),
        ] {
            if let Some(t) = t {
                assert_eq!(t.dims()[0], n, "record {name}");
            }
        }
    }
    let optim = trainer.optim.as_ref().expect("a step ran");
    let m = optim.transforms.momentum.as_ref().unwrap();
    assert_eq!(m.moment_1.dims()[0], n, "transforms moment 1");
    assert_eq!(m.moment_2.dims()[0], n, "transforms moment 2");
    let m = optim.sh_coeffs.momentum.as_ref().unwrap();
    assert_eq!(m.moment_1.dims()[0], n, "sh moment 1");
    assert_eq!(m.moment_2.dims()[0], n, "sh moment 2");
    let m = optim.opacities.momentum.as_ref().unwrap();
    assert_eq!(m.moment_1.dims()[0], n, "opacity moment 1");
    assert_eq!(m.moment_2.dims()[0], n, "opacity moment 2");
    let life = trainer.rows.life.as_ref().unwrap();
    assert_eq!(life.len(), n, "life");
    assert_eq!((life.importance.dims()[0], life.trained.dims()[0]), (n, n), "life importance, trained");
    assert_eq!(trainer.rows.origin.as_ref().unwrap().len(), n, "origin");
}

#[tokio::test]
async fn host_bake_matches_the_gpu_fold() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let mut rows = SplatRows {
        means: vec![0.0; 9],
        rotations: [1.0, 0.0, 0.0, 0.0].repeat(3),
        log_scales: vec![-3.0, -2.0, -6.0, -4.0, -4.0, -4.0, 0.0, -1.0, -8.0],
        sh_coeffs: vec![0.1; 9],
        raw_opacities: vec![0.3, -1.0, 4.0],
        min_scale: vec![0.01, 0.002, 0.05],
        origin: vec![None; 3],
        age: vec![0.0; 3],
        importance: vec![f32::INFINITY; 3],
        sh_coeffs_per_channel: 1,
        ..SplatRows::default()
    };
    let host = rows.baked();
    let s = rows.to_splats(&device);
    let (t, o) = fold_min_scale(
        s.transforms.val(),
        s.raw_opacities.val(),
        s.min_scale.clone().unwrap(),
    );
    let t: Vec<f32> = t.into_data_async().await.unwrap().try_to_vec().unwrap();
    let o: Vec<f32> = o.into_data_async().await.unwrap().try_to_vec().unwrap();
    for i in 0..3 {
        for a in 0..3 {
            assert!((host.log_scales[i * 3 + a] - t[i * 10 + 7 + a]).abs() < 1e-4);
        }
        assert!(
            (host.raw_opacities[i] - o[i]).abs() < 1e-3,
            "{} vs {}",
            host.raw_opacities[i],
            o[i]
        );
    }
    assert!(host.min_scale.is_empty());
    rows.min_scale.clear();
    assert_eq!(rows.baked(), rows, "no floor: unchanged");
}

#[tokio::test]
#[should_panic(expected = "splat rows column importance")]
async fn append_rows_refuses_misshapen_rows() {
    let (mut trainer, s) = trained(10).await;
    let mut rows = trainer.read_rows(&s).await.select(&[0, 1]);
    rows.importance.pop();
    let _ = trainer.append_rows(s, &rows);
}
