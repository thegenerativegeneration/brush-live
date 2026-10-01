//! Unit tests for cell quotas, recent-cell demand and eviction counts.

use super::*;

#[test]
fn picks_lowest_within_the_cell_quota() {
    // Cell (0,0,0): 10 splats with the lowest scores 0..10. Cell
    // (1,0,0): 10 splats with scores 100..110. Two ineligible (+inf).
    let mut scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
    scores.extend((0..10).map(|i| 100.0 + i as f32));
    scores.extend([f32::INFINITY; 2]);
    let mut means = Vec::new();
    for i in 0..22 {
        let x = if (10..20).contains(&i) { 1.5 } else { 0.5 };
        means.extend([x, 0.5, 0.5]);
    }
    // Without a cap the six lowest all come from the first cell.
    let mut all = pick_evictions(&scores, &means, 6, 1.0);
    all.sort_unstable();
    assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
    // At 30 % the first cell (12 splats incl. the 2 ineligible) gives 3,
    // the rest comes from the second cell's lowest.
    let mut capped = pick_evictions(&scores, &means, 6, 0.3);
    capped.sort_unstable();
    assert_eq!(capped, vec![0, 1, 2, 10, 11, 12]);
    // Ineligible splats are never picked, even if the count is not met.
    let few = pick_evictions(&scores, &means, 50, 1.0);
    assert_eq!(few.len(), 20);
    assert!(few.iter().all(|&i| i < 20));
}

#[test]
fn target_and_limit_split_the_headroom() {
    assert_eq!(evict_target(100_000, 0.1), 90_000);
    assert_eq!(growth_limit(100_000, 0.1), 95_000);
    assert_eq!(evict_target(300_000, 0.1), 270_000);
    assert_eq!(growth_limit(300_000, 0.1), 285_000);
    assert_eq!(evict_target(15, 0.1), 13);
    assert_eq!(growth_limit(15, 0.1), 14);
}

#[test]
fn evicts_blocked_recent_demand_up_to_the_target() {
    // Room left for the demand: nothing.
    assert_eq!(evict_count(90_000, 5_000, 0, 100_000, 0.1), 0);
    // What the free room below the growth limit cannot hold.
    assert_eq!(evict_count(94_000, 2_000, 0, 100_000, 0.1), 1_000);
    assert_eq!(evict_count(95_000, 300, 0, 100_000, 0.1), 300);
    // Capped at the target.
    assert_eq!(evict_count(100_000, 50_000, 0, 100_000, 0.1), 10_000);
    // Dropped seeds evict down to the target, even without demand.
    assert_eq!(evict_count(99_000, 0, 10, 100_000, 0.1), 9_000);
    // Over the growth limit but nothing new wants to grow: nothing.
    assert_eq!(evict_count(100_000, 0, 0, 100_000, 0.1), 0);
    // Already below the target: nothing to evict.
    assert_eq!(evict_count(80_000, 50_000, 7, 100_000, 0.1), 0);
    // Above the budget (e.g. loaded over cap) without overflow.
    assert_eq!(evict_count(120_000, u32::MAX, 0, 100_000, 0.1), 30_000);
}

fn seeds_at(p: [f32; 3], n: usize) -> Vec<f32> {
    p.repeat(n)
}

#[test]
fn recent_cells_need_enough_seeds_and_expire() {
    let mut cells = RecentCells::default();
    assert!(!cells.active(3));
    // A keyframe without seeds (revisiting a covered area) marks nothing.
    cells.note_keyframe(&[]);
    assert!(!cells.active(3));
    // A few seeds in a hole do not count either.
    cells.note_keyframe(&seeds_at([0.5, 0.5, 0.5], MIN_CELL_SEEDS - 1));
    assert!(!cells.active(3));
    assert!(!cells.contains(glam::vec3(0.5, 0.5, 0.5), 3));

    cells.note_keyframe(&seeds_at([-4.5, 0.2, 3.1], MIN_CELL_SEEDS));
    assert!(cells.active(3));
    assert!(cells.contains(glam::vec3(-4.1, 0.9, 3.9), 3));
    assert!(!cells.contains(glam::vec3(-3.9, 0.9, 3.9), 3));
    // Recent for the window's refines, then not.
    cells.tick();
    cells.tick();
    assert!(cells.active(3));
    assert!(cells.contains(glam::vec3(-4.5, 0.2, 3.1), 3));
    cells.tick();
    assert!(!cells.active(3));
    assert!(!cells.contains(glam::vec3(-4.5, 0.2, 3.1), 3));
}

#[test]
fn demand_counts_only_candidates_in_recent_cells() {
    let in_recent = [true, true, false, false, true];
    let oversized = [true, false, true, false, false];
    let above = [true, true, true, true, true];
    // 1 oversized + round(3 · 0.5) high-gradient.
    assert_eq!(recent_demand(&in_recent, &oversized, &above, 0.5), 3);
    // Growth off (empty mask) leaves the force-splits.
    assert_eq!(recent_demand(&in_recent, &oversized, &[], 0.5), 1);
    assert_eq!(recent_demand(&[false; 5], &oversized, &above, 0.5), 0);
}

/// A toy run at a 1000-splat budget: every refine every cell wants 50
/// more splats (Brush's growth demand never reaches zero). Keyframes
/// seed a new cell during refines 0..10 only; cells stay recent for 5
/// refines; each refine has fresh scores.
#[test]
fn evictions_settle_once_keyframes_stop() {
    let (max, headroom, window) = (1_000, 0.1, 5);
    let mut cells = RecentCells::default();
    let mut current = 950u32;
    let mut evicted = Vec::new();
    for refine in 0..40 {
        if refine < 10 {
            cells.note_keyframe(&seeds_at([refine as f32 + 0.5, 0.0, 0.0], 20));
        }
        let mut n = 0;
        if cells.active(window) {
            let in_recent: Vec<bool> = (0..10)
                .map(|c| cells.contains(glam::vec3(c as f32 + 0.5, 0.0, 0.0), window))
                .collect();
            cells.backlog += recent_demand(&in_recent, &[true; 10], &[], 1.0) * 50;
            n = evict_count(current, cells.backlog, 0, max, headroom);
            cells.backlog = 0;
        }
        evicted.push(n);
        cells.tick();
        // Growth refills up to the growth limit.
        current = growth_limit(max, headroom);
    }
    assert!(evicted[..14].iter().all(|&n| n > 0), "{evicted:?}");
    assert!(evicted[14..].iter().all(|&n| n == 0), "{evicted:?}");
}
