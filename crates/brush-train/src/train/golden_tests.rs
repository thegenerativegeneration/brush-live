//! Characterization of `step` and `refine` with every fork-only path on at once: SH background with the alpha
//! penalty, training radius, held rows, a two-group row budget, eviction, origin tracking and the min-scale floor.
//! A change that keeps `step` and `refine` behaviour reproduces the golden numbers.

use super::radius_tests::{NEAR, batch, device, floats, splats};
use super::*;
use crate::budget::{GroupBudget, RowBudget};
use crate::evict::EvictConfig;
use brush_render::gaussian_splats::SplatRenderMode;
use burn::module::Module;
use clap::Parser;

const STEPS: u32 = 30;
const REFINE_EVERY: u32 = 10;
const MAX_SPLATS: u32 = 24;
/// The scenario camera, as in the radius tests.
const CAMERA: glam::Vec3 = glam::vec3(30.0, 2.0, -20.0);
const RADIUS: f32 = 5.0;
/// A near row whose first SH coefficient is set past `MAX_SH0`, so the first refine prunes it.
const DIVERGED_ROW: usize = 4;
/// A near row whose Adam moments are corrupted after step 5.
const CLAMPED_ROW: usize = 0;
/// Relative tolerance of the float sums; the GPU path drifts by up to about 1.5e-4 between runs.
const REL_TOL: f64 = 1e-3;

/// Which fork paths the scenario reached.
#[derive(Default, Debug)]
struct Scenario {
    masked_by_radius: i64,
    held_at_refine: bool,
    floor_seen: bool,
    budget_evictions: u64,
    origin_rows: Option<usize>,
    sh_background: bool,
}

/// What one refine reported.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RefineFingerprint {
    added: u32,
    split_oversized: u32,
    split_high_grad: u32,
    pruned: u32,
    pruned_non_finite: u32,
    evicted: u32,
    evicted_for_total: u32,
    total: u32,
    /// Sum of `seen_since_refine` right before the refine (steps in which rows contributed to the render).
    seen: i64,
}

#[derive(Debug, Clone)]
struct Golden {
    count: u32,
    refines: Vec<RefineFingerprint>,
    adam_clamped: u64,
    sh_pruned: u64,
    /// Sums of means, log scales, SH coefficients and raw opacities, for the rows within the training radius of the
    /// camera and for those beyond it, so one row's error cannot cancel against another's.
    sums: [[f64; 4]; 2],
}

/// Near rows in group 0, the rest in group 1; both capped so the cap bites on the grown scene.
fn two_groups(n: usize) -> RowBudget {
    let group = (0..n).map(|i| u32::from(i >= NEAR)).collect();
    let g = |max_rows| GroupBudget {
        max_rows,
        full: false,
        evictable: u32::MAX,
    };
    RowBudget {
        group,
        groups: vec![g(14), g(14)],
        limit_eviction: false,
        evict_held: false,
        fallback: None,
    }
}

/// Three far rows held, the others free.
fn held_rows(n: usize, device: &Device) -> Tensor<1, Bool> {
    let flags: Vec<bool> = (0..n).map(|i| (NEAR + 2..NEAR + 5).contains(&i)).collect();
    Tensor::from_data(TensorData::new(flags, [n]), &device.clone().inner())
}

/// Sums of `splats`' parameters over the rows within `RADIUS` of the camera and over the rest.
async fn group_sums(splats: &Splats) -> [[f64; 4]; 2] {
    let means = floats(splats.means()).await;
    let per_row = [
        (3, means.clone()),
        (3, floats(splats.log_scales()).await),
        (
            splats.sh_coeffs.dims()[1] * 3,
            floats(splats.sh_coeffs.val()).await,
        ),
        (1, floats(splats.raw_opacities.val()).await),
    ];
    let mut sums = [[0.0; 4]; 2];
    for (row, m) in means.chunks(3).enumerate() {
        let beyond = usize::from((glam::Vec3::new(m[0], m[1], m[2]) - CAMERA).length() > RADIUS);
        for (k, (width, values)) in per_row.iter().enumerate() {
            sums[beyond][k] += values[row * width..(row + 1) * width]
                .iter()
                .map(|&x| f64::from(x))
                .sum::<f64>();
        }
    }
    sums
}

/// The radius scene with one near row's DC colour diverged.
async fn diverged_scene(device: &Device) -> Splats {
    let base = splats(0.05, device);
    let mut sh = floats(base.sh_coeffs.val()).await;
    let width = sh.len() / base.num_splats() as usize;
    sh[DIVERGED_ROW * width] = 50.0;
    let transforms = floats(base.transforms.val()).await;
    let rows = transforms.chunks(10);
    Splats::from_raw(
        rows.clone().flat_map(|r| r[..3].to_vec()).collect(),
        rows.clone().flat_map(|r| r[3..7].to_vec()).collect(),
        rows.flat_map(|r| r[7..10].to_vec()).collect(),
        sh,
        floats(base.raw_opacities.val()).await,
        SplatRenderMode::Default,
        device,
    )
}

/// Leaves transform row `row` with a large first moment and no second moment, as an aborted GPU command buffer
/// would, so its next normalized step exceeds the clamp.
fn corrupt_moments(trainer: &mut SplatTrainer, row: usize) {
    let moments = trainer
        .optim
        .as_mut()
        .and_then(|o| o.transforms.momentum.as_mut())
        .expect("stepped");
    let [n, width] = moments.moment_2.dims();
    let device = moments.moment_2.device();
    let flags = |on: f32, off: f32| {
        let v: Vec<f32> = (0..n)
            .flat_map(|i| vec![if i == row { on } else { off }; width])
            .collect();
        Tensor::<2>::from_data(TensorData::new(v, [n, width]), &device)
    };
    moments.moment_2 = moments.moment_2.clone() * flags(0.0, 1.0);
    moments.moment_1 = moments.moment_1.clone() + flags(1000.0, 0.0);
}

async fn run_scenario() -> (Golden, Scenario) {
    let device = device().await;
    let mut config = TrainConfig::parse_from(["test"]);
    config.sh_background = true;
    config.sh_background_alpha_weight = 0.15;
    config.train_radius_m = RADIUS;
    config.growth_select_fraction = 1.0;
    config.split_at_screen_size = 0.05;
    config.max_splats = MAX_SPLATS;
    let base = diverged_scene(&device).await;
    let bounds = BoundingBox::from_min_max(CAMERA - 25.0, CAMERA + 25.0);
    let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
    });
    trainer.set_view_cams(vec![(CAMERA, 27.7)]);
    trainer.track_origins(&base, 3);
    let mut splats = base.train();
    let mut stats = Vec::new();
    let mut seen = Vec::new();
    let mut scenario = Scenario::default();
    for step in 1..=STEPS {
        if step % REFINE_EVERY == 1 {
            let n = splats.num_splats() as usize;
            trainer.set_frozen_rows(Some(held_rows(n, &device)));
        }
        splats = trainer.step(batch(), splats).await.0;
        if step == 5 {
            corrupt_moments(&mut trainer, CLAMPED_ROW);
        }
        if let Some(m) = trainer.masked_by_radius() {
            let data = m.into_data_async().await.expect("readback");
            scenario.masked_by_radius += data.convert::<i64>().try_to_vec::<i64>().expect("i64")[0];
        }
        if step % REFINE_EVERY == 0 {
            let n = splats.num_splats() as usize;
            trainer.set_row_budget(Some(two_groups(n)));
            trainer.note_seed_shortfall(1);
            scenario.held_at_refine |= trainer.frozen_rows().is_some();
            let rows = floats(trainer.seen_since_refine().expect("stepped")).await;
            seen.push(rows.iter().map(|&x| f64::from(x)).sum::<f64>().round() as i64);
            let (valid, s) = trainer.refine(step, splats.valid()).await;
            scenario.floor_seen |= valid.min_scale.is_some();
            scenario.budget_evictions += trainer
                .budget_evictions()
                .iter()
                .map(|&e| u64::from(e))
                .sum::<u64>();
            stats.push(s);
            splats = valid.train();
        }
    }
    scenario.origin_rows = trainer.origin_slots().await.map(|o| o.len());
    scenario.sh_background = trainer.sh_background().is_some();
    let valid = splats.valid();
    let golden = Golden {
        count: valid.num_splats(),
        refines: stats
            .iter()
            .zip(seen)
            .map(|(s, seen)| RefineFingerprint {
                added: s.num_added,
                split_oversized: s.num_split_oversized,
                split_high_grad: s.num_split_high_grad,
                pruned: s.num_pruned,
                pruned_non_finite: s.num_pruned_non_finite,
                evicted: s.num_evicted,
                evicted_for_total: s.evicted_for_total,
                total: s.total_splats,
                seen,
            })
            .collect(),
        adam_clamped: trainer.adam_clamped(),
        sh_pruned: trainer.sh_pruned(),
        sums: group_sums(&valid).await,
    };
    (golden, scenario)
}

#[allow(clippy::too_many_arguments)]
fn refine(
    added: u32,
    split_oversized: u32,
    split_high_grad: u32,
    pruned: u32,
    evicted: u32,
    total: u32,
    seen: i64,
) -> RefineFingerprint {
    RefineFingerprint {
        added,
        split_oversized,
        split_high_grad,
        pruned,
        pruned_non_finite: 0,
        evicted,
        evicted_for_total: evicted,
        total,
        seen,
    }
}

#[tokio::test]
async fn refine_and_step_match_golden() {
    let (got, scenario) = run_scenario().await;
    // Printed to re-record the golden below after an intended change.
    println!("GOLDEN {got:?}\nSCENARIO {scenario:?}");

    // Every fork path ran, so a silently skipped one fails here.
    assert!(scenario.sh_background, "SH background off");
    assert!(
        scenario.masked_by_radius > 0,
        "no row beyond the training radius"
    );
    assert!(scenario.held_at_refine, "no rows held at a refine");
    assert!(scenario.floor_seen, "no min-scale floor after a refine");
    assert!(scenario.budget_evictions > 0, "no budget eviction");
    assert!(
        got.refines.iter().any(|r| r.evicted > 0),
        "no eviction in any refine"
    );
    assert!(
        got.refines.iter().any(|r| r.split_oversized > 0),
        "no oversized split"
    );
    assert!(
        got.refines.iter().all(|r| r.seen > 0),
        "no row seen before a refine"
    );
    assert!(got.adam_clamped > 0, "no clamped Adam step");
    assert!(got.sh_pruned > 0, "no diverged colour pruned");
    assert_eq!(
        scenario.origin_rows,
        Some(got.count as usize),
        "origin tags out of step"
    );

    assert_eq!(got.count, 23);
    assert_eq!(
        got.refines,
        vec![
            refine(4, 2, 0, 2, 0, 23, 200),
            refine(0, 0, 0, 1, 1, 22, 230),
            refine(1, 1, 0, 0, 0, 23, 220)
        ]
    );
    assert_eq!((got.adam_clamped, got.sh_pruned), (50, 1));
    let want = [
        [
            199.30684745311737,
            -119.2168641090393,
            16.018418228253722,
            -10.391566753387451,
        ],
        [
            323.30559730529785,
            0.0,
            11.880000187084079,
            -26.499950408935547,
        ],
    ];
    let names = ["means", "log_scales", "sh_coeffs", "raw_opacities"];
    for (group, (got, want)) in ["within the radius", "beyond the radius"]
        .iter()
        .zip(got.sums.iter().zip(want))
    {
        for (name, (g, w)) in names.iter().zip(got.iter().zip(want)) {
            assert!(
                (g - w).abs() <= REL_TOL * w.abs().max(1.0),
                "{name} sum {group}: {g} vs golden {w}"
            );
        }
    }
}
