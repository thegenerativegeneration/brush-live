//! Characterization of `step` and `refine` with every fork-only path on at once: SH background with the alpha
//! penalty, training radius, held rows, a two-group row budget, eviction, origin tracking and the min-scale floor.
//! The golden numbers were recorded before `train.rs` was split; a behaviour-preserving change reproduces them.

use super::radius_tests::{NEAR, batch, device, floats, splats};
use super::*;
use crate::budget::{GroupBudget, RowBudget};
use crate::evict::EvictConfig;
use burn::module::Module;
use clap::Parser;

const STEPS: u32 = 30;
const REFINE_EVERY: u32 = 10;
const MAX_SPLATS: u32 = 24;
/// Relative tolerance of the float sums.
const REL_TOL: f64 = 1e-4;

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
}

#[derive(Debug, Clone)]
struct Golden {
    count: u32,
    refines: Vec<RefineFingerprint>,
    adam_clamped: u64,
    sh_pruned: u64,
    /// Sums of means, log scales, SH coefficients and raw opacities.
    sums: [f64; 4],
}

async fn fingerprint(splats: &Splats, trainer: &SplatTrainer, stats: &[RefineStats]) -> Golden {
    let total = |v: Vec<f32>| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    let sums = [
        total(floats(splats.means()).await),
        total(floats(splats.log_scales()).await),
        total(floats(splats.sh_coeffs.val()).await),
        total(floats(splats.raw_opacities.val()).await),
    ];
    Golden {
        count: splats.num_splats(),
        refines: stats
            .iter()
            .map(|s| RefineFingerprint {
                added: s.num_added,
                split_oversized: s.num_split_oversized,
                split_high_grad: s.num_split_high_grad,
                pruned: s.num_pruned,
                pruned_non_finite: s.num_pruned_non_finite,
                evicted: s.num_evicted,
                evicted_for_total: s.evicted_for_total,
                total: s.total_splats,
            })
            .collect(),
        adam_clamped: trainer.adam_clamped(),
        sh_pruned: trainer.sh_pruned(),
        sums,
    }
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

async fn run_scenario() -> (Golden, Scenario) {
    let device = device().await;
    let mut config = TrainConfig::parse_from(["test"]);
    config.sh_background = true;
    config.sh_background_alpha_weight = 0.15;
    config.train_radius_m = 5.0;
    config.growth_select_fraction = 1.0;
    config.max_splats = MAX_SPLATS;
    let base = splats(0.05, &device);
    let position = glam::vec3(30.0, 2.0, -20.0);
    let bounds = BoundingBox::from_min_max(position - 25.0, position + 25.0);
    let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);
    trainer.enable_eviction(EvictConfig {
        headroom: 0.1,
        min_age: 0,
        max_cell_fraction: 1.0,
        recent_refines: 2,
    });
    trainer.set_view_cams(vec![(position, 27.7)]);
    trainer.track_origins(&base, 3);
    let mut splats = base.train();
    let mut stats = Vec::new();
    let mut scenario = Scenario::default();
    for step in 1..=STEPS {
        if step % REFINE_EVERY == 1 {
            let n = splats.num_splats() as usize;
            trainer.set_frozen_rows(Some(held_rows(n, &device)));
        }
        splats = trainer.step(batch(), splats).await.0;
        if let Some(m) = trainer.masked_by_radius() {
            let data = m.into_data_async().await.expect("readback");
            scenario.masked_by_radius += data.convert::<i64>().try_to_vec::<i64>().expect("i64")[0];
        }
        if step % REFINE_EVERY == 0 {
            let n = splats.num_splats() as usize;
            trainer.set_row_budget(Some(two_groups(n)));
            trainer.note_seed_shortfall(1);
            scenario.held_at_refine |= trainer.frozen_rows().is_some();
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
    (fingerprint(&valid, &trainer, &stats).await, scenario)
}

#[derive(Default, Debug)]
struct Scenario {
    masked_by_radius: i64,
    held_at_refine: bool,
    floor_seen: bool,
    budget_evictions: u64,
    origin_rows: Option<usize>,
    sh_background: bool,
}

fn refine(
    added: u32,
    split_high_grad: u32,
    pruned: u32,
    evicted: u32,
    total: u32,
) -> RefineFingerprint {
    RefineFingerprint {
        added,
        split_oversized: 0,
        split_high_grad,
        pruned,
        pruned_non_finite: 0,
        evicted,
        evicted_for_total: evicted,
        total,
    }
}

#[tokio::test]
async fn refine_and_step_match_golden() {
    let (got, scenario) = run_scenario().await;
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
    assert_eq!(
        scenario.origin_rows,
        Some(got.count as usize),
        "origin tags out of step"
    );

    assert_eq!(got.count, 23);
    assert_eq!(
        got.refines,
        vec![
            refine(3, 2, 1, 0, 23),
            refine(0, 0, 1, 1, 22),
            refine(1, 1, 0, 0, 23)
        ]
    );
    assert_eq!((got.adam_clamped, got.sh_pruned), (0, 0));
    let want = [
        524.5225839614868,
        -97.50360774993896,
        27.763855570927262,
        -17.24403840303421,
    ];
    for (name, (g, w)) in ["means", "log_scales", "sh_coeffs", "raw_opacities"]
        .iter()
        .zip(got.sums.iter().zip(want))
    {
        assert!(
            (g - w).abs() <= REL_TOL * w.abs().max(1.0),
            "{name} sum {g} vs golden {w}"
        );
    }
}
