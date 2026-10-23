//! Rows taken out of the trainer carry their Adam moments and trained flags, and rows appended back resume
//! with them.

use super::super::*;
use super::tests::{batch, trained_on, trained_with};
use crate::rows::{RowMoments, SplatRows};
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use burn::module::Module;
use clap::Parser;
use rand::SeedableRng;

/// A copy of `t` that steps on independently. Its RNG starts afresh (`StdRng` cannot be copied), which a step with
/// background noise off never draws from.
fn twin(t: &SplatTrainer) -> SplatTrainer {
    SplatTrainer {
        config: t.config.clone(),
        lr_mean_decay: t.lr_mean_decay,
        lr_scaling_fixed: t.lr_scaling_fixed.clone(),
        lr_mean_columns: t.lr_mean_columns.clone(),
        refine_record: t.refine_record.clone(),
        optim: t.optim.clone(),
        bounds: t.bounds,
        step_count: t.step_count,
        max_sh_degree: t.max_sh_degree,
        rng: rand::rngs::StdRng::seed_from_u64(0),
        view_cams: t.view_cams.clone(),
        evict: t.evict.clone(),
        rows: t.rows.clone(),
        profile: t.profile,
        lpips: t.lpips.clone(),
        sh_background: t.sh_background.clone(),
        adam_clamped: t.adam_clamped,
        sh_pruned: t.sh_pruned,
        last_frozen: t.last_frozen.clone(),
        budget_evicted: t.budget_evicted.clone(),
        eviction_fell_back: t.eviction_fell_back,
    }
}

/// `x` rounded to bfloat16 (nearest, ties to even), as a tile file stores moments.
fn as_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn stored(m: &RowMoments) -> RowMoments {
    let q = |v: &[f32]| v.iter().copied().map(as_bf16).collect();
    RowMoments {
        transforms_m1: q(&m.transforms_m1),
        transforms_m2: q(&m.transforms_m2),
        sh_m1: q(&m.sh_m1),
        sh_m2: q(&m.sh_m2),
        opacity_m1: q(&m.opacity_m1),
        opacity_m2: q(&m.opacity_m2),
    }
}

/// `n` splats 2 m ahead of the camera, each a little deeper than the last, so the render order does not depend on
/// the row order.
fn staggered(n: usize, device: &Device) -> Splats {
    Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, 2.0 + i as f32 * 0.003]).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![-3.0; n * 3],
        vec![0.5; n * 3],
        vec![inverse_sigmoid(0.5); n],
        SplatRenderMode::Default,
        device,
    )
}

/// Per parameter column, `after - before` of the rows `at` of `after` against the rows of `before`.
fn updates(before: &SplatRows, after: &SplatRows) -> Vec<(&'static str, Vec<f32>)> {
    let d = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x - y).collect();
    vec![
        ("means", d(&after.means, &before.means)),
        ("rotations", d(&after.rotations, &before.rotations)),
        ("log_scales", d(&after.log_scales, &before.log_scales)),
        ("sh_coeffs", d(&after.sh_coeffs, &before.sh_coeffs)),
        ("raw_opacities", d(&after.raw_opacities, &before.raw_opacities)),
    ]
}

/// Per column, the largest difference between two updates relative to the largest update of `want`.
fn deviation(got: &[(&str, Vec<f32>)], want: &[(&str, Vec<f32>)]) -> Vec<(String, f32)> {
    got.iter()
        .zip(want)
        .map(|((name, g), (_, w))| {
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let off = g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            (name.to_string(), off / scale.max(f32::MIN_POSITIVE))
        })
        .collect()
}

/// Steps `s` once on the test batch and reads every row back.
async fn step_once(trainer: &mut SplatTrainer, s: Splats) -> SplatRows {
    let (s, _) = trainer.step(batch(), s.train()).await;
    trainer.read_rows(&s.valid()).await
}

/// Taken rows whose moments went through 16-bit storage and came back take the same step as their twin in a copy of
/// the trainer that never took them, within 16-bit tolerance; with zero moments (a version 1 tile) they do not.
#[tokio::test]
async fn restored_rows_step_like_their_unparked_twin() {
    // No noise; a mean learning rate 100× the default makes the means' updates thousands of f32 steps of the means
    // rather than a few hundred, so rounding of the means themselves stays far below the tolerance.
    let args = ["--mean-noise-weight", "0", "--background-noise-strength", "0", "--lr-mean", "2e-3"];
    let (mut trainer, mut s) = trained_on(|d| staggered(60, d), &args, true).await;
    for _ in 0..5 {
        let (next, _) = trainer.step(batch(), s.train()).await;
        s = next.valid();
    }
    let mut unparked = twin(&trainer);
    let mut control = twin(&trainer);
    let before = trainer.read_rows(&s).await;
    let n = before.len();
    let take: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let idx: Vec<usize> = (0..n).filter(|&i| take[i]).collect();
    let tail: Vec<usize> = (n - idx.len()..n).collect();

    let (rest, taken) = trainer.take_rows(s.clone(), &take).await;
    let mut parked = taken.expect("a third of the rows");
    parked.moments = Some(stored(parked.moments.as_ref().expect("the steps left moments")));
    let restored = trainer.append_rows(rest, &parked);
    let (rest, taken) = control.take_rows(s.clone(), &take).await;
    let zeroed = SplatRows { moments: None, ..taken.unwrap() };
    let zero_restored = control.append_rows(rest, &zeroed);

    let want = updates(&before.select(&idx), &step_once(&mut unparked, s).await.select(&idx));
    let got = updates(&parked, &step_once(&mut trainer, restored).await.select(&tail));
    let zero = updates(&zeroed, &step_once(&mut control, zero_restored).await.select(&tail));
    let (got, zero) = (deviation(&got, &want), deviation(&zero, &want));
    eprintln!("restored {got:?}\nzero moments {zero:?}");
    assert!(got.iter().all(|(_, d)| *d < 0.01), "restored rows stepped unlike their twin: {got:?}");
    assert!(zero.iter().any(|(_, d)| *d > 0.2), "zero moments step like the twin, the test proves nothing: {zero:?}");
}

/// Rows read with their moments and trained flags come back with them: the trainer's moment rows are the taken
/// ones, and a row the host marks untrained counts as untrained again. Rows without moments start at zero, rows
/// without flags count as trained.
#[tokio::test]
async fn appended_rows_keep_their_moments_and_trained_flags() {
    let (mut trainer, s) = trained_with(30, &[], true).await;
    let (s, _) = trainer.step(batch(), s.train()).await;
    let s = s.valid();
    let mut rows = trainer.read_rows(&s).await.select(&[0, 1, 2]);
    let m = rows.moments.clone().expect("a step ran: rows carry moments");
    assert!(m.opacity_m1.iter().any(|&v| v != 0.0) && m.sh_m2.iter().all(|&v| v > 0.0), "{m:?}");
    assert_eq!(rows.trained, vec![true; 3], "the steps trained every row");
    rows.trained = vec![false, true, false];
    let s = trainer.append_rows(s, &rows);
    let all = trainer.read_rows(&s).await;
    let n = all.len();
    let back = all.select(&[n - 3, n - 2, n - 1]);
    assert_eq!(back.moments.as_ref(), Some(&m));
    assert_eq!(back.trained, rows.trained);

    let bare = SplatRows { moments: None, trained: Vec::new(), ..rows.clone() };
    let s = trainer.append_rows(s, &bare);
    let all = trainer.read_rows(&s).await;
    let back = all.select(&[n, n + 1, n + 2]);
    assert_eq!(back.moments, Some(RowMoments::zeros(3, rows.sh_coeffs_per_channel as usize)));
    assert_eq!(back.trained, vec![true; 3]);
}

/// The engine's SH second moment is reduced on the compact rows and gathered back to the splats every step
/// (`burn_glue`): after the first step, which leaves m1 = 0.1·g and m2 = 0.001·mean(g²), every row's second moment is
/// 0.1·mean(m1²) of its own first moment, not a neighbour's.
#[tokio::test]
async fn the_first_step_reduces_each_rows_own_sh_gradient() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let n = 60;
    let base = Splats::from_raw(
        (0..n).flat_map(|i| [i as f32 * 0.01, (i % 7) as f32 * 0.01, 2.0 + i as f32 * 0.003]).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![-3.0; n * 3],
        (0..n * 12).map(|i| ((i * 7919) % 13) as f32 * 0.05 - 0.3).collect(),
        (0..n).map(|i| inverse_sigmoid(0.2 + 0.01 * i as f32)).collect(),
        SplatRenderMode::Default,
        &device,
    );
    let config = TrainConfig::parse_from(["test"]);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let (s, _) = trainer.step(batch(), base.train()).await;
    let rows = trainer.read_rows(&s.valid()).await;
    assert_eq!(rows.sh_coeffs_per_channel, 4);
    let m = rows.moments.expect("a step ran");
    let mut seen = 0;
    for i in 0..n {
        let m1 = &m.sh_m1[i * 12..(i + 1) * 12];
        let want = 0.1 * m1.iter().map(|v| v * v).sum::<f32>() / 12.0;
        let got = m.sh_m2[i];
        assert!((got - want).abs() <= 1e-4 * want.max(got) + 1e-37, "row {i}: m2 {got}, 0.1·mean(m1²) {want}");
        seen += usize::from(want > 0.0);
    }
    assert!(seen > n / 2, "{seen} of {n} rows had an SH gradient: the test proves little");
}
