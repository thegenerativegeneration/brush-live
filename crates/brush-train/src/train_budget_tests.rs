//! Row budgets through refine ([`SplatTrainer::set_row_budget`]): a full group does not densify, a group over its cap
//! shrinks to it, a limited eviction keeps each group's quota, held rows no step ever trained are eviction candidates,
//! and the budget follows take and append and is dropped by refine.

use super::evict_tests::{N, batch, importance, record_importance, rows, splats, trainer_at_budget};
use super::radius_tests::{FAR, NEAR, bits, floats};
use super::*;
use crate::budget::{EvictionRelax, GroupBudget, NO_GROUP, RowBudget};
use crate::evict::EvictConfig;
use burn::module::Module;
use clap::Parser;

fn group(max_rows: u32, full: bool, evictable: u32) -> GroupBudget {
    GroupBudget { max_rows, full, evictable }
}

/// Rows 0..30 in group 0 and 30..60 in group 1 (of the 60-splat eviction model).
fn halves(g0: GroupBudget, g1: GroupBudget, limit_eviction: bool) -> RowBudget {
    let group = (0..N).map(|i| u32::from(i >= 30)).collect();
    RowBudget { group, groups: vec![g0, g1], limit_eviction, evict_held: false }
}

/// The importance of the splats left after a refine, in order.
async fn kept_importance(trainer: &SplatTrainer) -> Vec<f32> {
    let life = trainer.evict.as_ref().unwrap().life.as_ref().unwrap();
    rows(life.importance.clone().unsqueeze_dim::<2>(1)).await.concat()
}

/// The near/far model of the radius tests after three steps, every splat a growth and force-split candidate, one
/// refine with `budget` (near rows group 0, far and dead rows group 1): (transforms before, after, refine stats).
async fn refine_with(budget: Option<RowBudget>) -> (Vec<f32>, Vec<f32>, RefineStats) {
    let device = super::radius_tests::device().await;
    let mut trainer = super::radius_tests::trainer(0.0, &device);
    trainer.config.growth_select_fraction = 1.0;
    let mut s = super::radius_tests::splats(0.9, &device).train();
    for _ in 0..3 {
        s = trainer.step(super::radius_tests::batch(), s).await.0;
    }
    let record = trainer.refine_record.as_mut().unwrap();
    record.refine_weight_norm = record.refine_weight_norm.clone().ones_like();
    record.max_screen_size = record.max_screen_size.clone().ones_like();
    record.vis_weight = record.vis_weight.clone().ones_like();
    trainer.set_row_budget(budget);
    let before = floats(s.transforms.val().inner()).await;
    let (refined, stats) = trainer.refine(3, s.valid()).await;
    (before, floats(refined.transforms.val()).await, stats)
}

/// Rows of a full group neither split nor grow, nor stand in for the pruned row; the other group's rows do.
#[tokio::test]
async fn a_full_group_does_not_grow() {
    let (_, _, free) = refine_with(None).await;
    assert_eq!(free.num_added as usize, NEAR + FAR, "without a budget every row splits");
    let budget = RowBudget {
        group: (0..NEAR + FAR + 1).map(|i| u32::from(i >= NEAR)).collect(),
        groups: vec![group(10_000, true, 0), group(10_000, false, 0)],
        limit_eviction: false,
        evict_held: false,
    };
    let (before, after, stats) = refine_with(Some(budget)).await;
    assert_eq!(stats.num_pruned, 1);
    assert_eq!(stats.num_added as usize, FAR, "only the far rows split");
    assert_eq!(after.len() / 10, NEAR + 2 * FAR);
    assert_eq!(bits(&before[..NEAR * 10]), bits(&after[..NEAR * 10]), "a full group's row split");
}

/// A group over its cap gives its least important rows down to the cap at the refine, without a seed shortfall or
/// growth demand; the evicted rows are not replaced, and the evictions are told per group.
#[tokio::test]
async fn an_over_cap_group_shrinks_to_its_cap() {
    let (mut trainer, s) = trainer_at_budget(0).await;
    let imp = importance();
    record_importance(&mut trainer, &imp);
    trainer.set_row_budget(Some(halves(group(25, true, 0), group(100, false, 0), false)));
    let (s, stats) = trainer.refine(1, s).await;
    assert_eq!((stats.num_evicted, stats.num_pruned, stats.num_added), (5, 5, 0));
    assert_eq!(stats.evicted_for_total, 0, "a group's own cap is no eviction for the total");
    assert_eq!(s.num_splats() as usize, N - 5);
    let mut lowest: Vec<f32> = imp[..30].to_vec();
    lowest.sort_by(f32::total_cmp);
    let gone = &lowest[..5];
    let want: Vec<f32> = imp.iter().enumerate().filter(|&(i, v)| i >= 30 || !gone.contains(v)).map(|(_, v)| *v).collect();
    assert_eq!(kept_importance(&trainer).await, want);
    assert_eq!(trainer.budget_evictions(), &[5, 0]);
}

/// A limited eviction takes the least important rows only while their group may give one, and takes fewer than it
/// wants when the quotas run out.
#[tokio::test]
async fn a_limited_eviction_keeps_each_groups_quota() {
    let (mut trainer, s) = trainer_at_budget(0).await;
    let imp: Vec<f32> = (0..N).map(|i| i as f32).collect();
    record_importance(&mut trainer, &imp);
    trainer.note_seed_shortfall(1);
    trainer.set_row_budget(Some(halves(group(u32::MAX, false, 2), group(u32::MAX, false, 100), true)));
    let (_, stats) = trainer.refine(1, s).await;
    // Budget 60, headroom 10 %: six go, two from group 0 (its lowest), four from group 1.
    assert_eq!((stats.num_evicted, stats.evicted_for_total), (6, 6));
    let want: Vec<f32> = imp.iter().copied().filter(|&v| !(v < 2.0 || (30.0..34.0).contains(&v))).collect();
    assert_eq!(kept_importance(&trainer).await, want);
    assert_eq!(trainer.budget_evictions(), &[2, 4]);

    let (mut trainer, s) = trainer_at_budget(0).await;
    record_importance(&mut trainer, &imp);
    trainer.note_seed_shortfall(1);
    trainer.set_row_budget(Some(halves(group(u32::MAX, false, 1), group(u32::MAX, false, 0), true)));
    let (_, stats) = trainer.refine(1, s).await;
    assert_eq!(stats.num_evicted, 1, "quotas allow one");
}

/// Held rows no step ever trained are eviction candidates (ranked first, scored or not); held rows a step trained
/// stay no victims.
#[tokio::test]
async fn held_rows_no_step_trained_are_evicted_at_the_cap() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();
    let mut config = TrainConfig::parse_from(["test"]);
    config.max_splats = N as u32;
    config.growth_grad_threshold = f32::MAX;
    config.split_at_screen_size = 0.0;
    let base = splats(N, &device);
    let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.enable_eviction(EvictConfig { headroom: 0.1, min_age: 0, max_cell_fraction: 1.0, recent_refines: 2 });
    let mask = |rows: &dyn Fn(usize) -> bool| {
        let flags: Vec<bool> = (0..N).map(rows).collect();
        Tensor::<1, Bool>::from_data(TensorData::new(flags, [N]), &device.clone().inner())
    };
    // Rows 54..60 held from the start: never trained.
    trainer.set_frozen_rows(Some(mask(&|i| i >= 54)));
    let (s, _) = trainer.step(batch(), base.train()).await;
    // Rows 0..6 (the least important) held now, after a step trained them.
    trainer.set_frozen_rows(Some(mask(&|i| i < 6 || i >= 54)));
    // Ascending importance; row 59 unscored.
    let imp: Vec<f32> = (0..N).map(|i| if i == 59 { f32::NAN } else { i as f32 }).collect();
    record_importance(&mut trainer, &imp);
    trainer.note_seed_shortfall(1);
    let (s, stats) = trainer.refine(1, s.valid()).await;
    assert_eq!(stats.num_evicted, 6);
    assert_eq!(s.num_splats() as usize, N - 6);
    let want: Vec<f32> = (0..54).map(|i| i as f32).collect();
    assert_eq!(kept_importance(&trainer).await, want, "the six never-trained held rows go");
}

/// The budget follows take and append (appended rows have no group), a budget of another length is dropped, and
/// refine drops the budget.
#[tokio::test]
async fn the_budget_follows_take_and_append() {
    let (mut trainer, s) = trainer_at_budget(0).await;
    trainer.set_row_budget(Some(halves(group(100, false, 0), group(100, false, 0), false)));
    let take: Vec<bool> = (0..N).map(|i| i == 0 || i == 40).collect();
    let (s, taken) = trainer.take_rows(s, &take).await;
    let s = trainer.append_rows(s, &taken.unwrap());
    let mut want: Vec<u32> = (1..N).filter(|&i| i != 40).map(|i| u32::from(i >= 30)).collect();
    want.extend([NO_GROUP, NO_GROUP]);
    assert_eq!(trainer.row_budget.as_ref().unwrap().group, want);
    trainer.set_row_budget(Some(halves(group(1, true, 0), group(1, true, 0), false)).map(|mut b| {
        b.group.pop();
        b
    }));
    assert!(trainer.row_budget.is_none(), "a short budget was kept");
    trainer.set_row_budget(Some(RowBudget { group: want, ..halves(group(100, false, 0), group(100, false, 0), false) }));
    let _ = trainer.refine(1, s).await;
    assert!(trainer.row_budget.is_none(), "refine kept the budget");
}

/// A limited eviction whose quotas give nothing takes, once relaxed for a group, that group's rows by lowest importance,
/// held rows included; relaxed to unlimited it takes the least important rows that are not held, as without a budget.
#[tokio::test]
async fn a_relaxed_eviction_takes_held_rows_of_the_opened_groups() {
    let imp: Vec<f32> = (0..N).map(|i| i as f32).collect();
    // Rows 30..60 (group 1) held after a step trained them.
    let held = |s: &Splats| {
        let flags: Vec<bool> = (0..N).map(|i| i >= 30).collect();
        Tensor::<1, Bool>::from_data(TensorData::new(flags, [N]), &s.device().inner())
    };
    let run = async |relax: Option<EvictionRelax>| {
        let (mut trainer, s) = trainer_at_budget(0).await;
        record_importance(&mut trainer, &imp);
        trainer.set_frozen_rows(Some(held(&s)));
        trainer.note_seed_shortfall(1);
        trainer.set_row_budget(Some(halves(group(u32::MAX, false, 0), group(u32::MAX, false, 0), true)));
        if let Some(relax) = relax {
            trainer.relax_eviction(relax);
        }
        let (_, stats) = trainer.refine(1, s).await;
        (stats.num_evicted, kept_importance(&trainer).await)
    };
    assert_eq!(run(None).await.0, 0, "quotas of 0 give nothing");
    let (n, kept) = run(Some(EvictionRelax::Groups(vec![1]))).await;
    assert_eq!(n, 6);
    assert_eq!(kept, imp.iter().copied().filter(|&v| !(30.0..36.0).contains(&v)).collect::<Vec<_>>());
    let (n, kept) = run(Some(EvictionRelax::Unlimited)).await;
    assert_eq!(n, 6);
    assert_eq!(kept, imp.iter().copied().filter(|&v| v >= 6.0).collect::<Vec<_>>(), "held rows stay");
}
