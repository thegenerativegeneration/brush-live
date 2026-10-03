use super::*;

/// One SplatGeom per voxel along +x starting at `x0`, opacity 1.
fn row(x0: i32, n: i32) -> Vec<SplatGeom> {
    (0..n)
        .map(|i| g([(x0 + i) as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, 0.1).geom())
        .collect()
}

/// A voxel absent longer than the prune window comes back uninformed, with its age restarted.
#[test]
fn pruned_voxel_restarts_uninformed() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    // Two voxels so the pass has a range; V at x=0 gets informed bytes.
    agg.aggregate(
        &[g([0.5; 3], 1.0, 0.8, 0.1), g([5.5, 0.5, 0.5], 1.0, 0.2, 0.4)],
        &[],
        0.0,
    );
    for i in 0..=PRUNE_AFTER_ROUNDS {
        agg.cells(&row(5, 1), &[], 1.0 + i as f64);
    }
    let cells = agg.cells(&[row(0, 1), row(5, 1)].concat(), &[], 100.0);
    let v = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
    assert!(v.uninformed);
    assert_eq!(v.coverage, UNINFORMED_COVERAGE);
    assert_eq!(v.uncertainty, UNINFORMED_UNCERTAINTY);
    assert_eq!(v.age, 0, "first_seen restarts with the voxel");
}

/// A gap shorter than the window changes nothing: bytes and age survive.
#[test]
fn short_absence_keeps_bytes() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    agg.aggregate(
        &[g([0.5; 3], 1.0, 0.8, 0.1), g([5.5, 0.5, 0.5], 1.0, 0.2, 0.4)],
        &[],
        0.0,
    );
    for i in 0..PRUNE_AFTER_ROUNDS - 1 {
        agg.cells(&row(5, 1), &[], 1.0 + i as f64);
    }
    let cells = agg.cells(&[row(0, 1), row(5, 1)].concat(), &[], 10.0);
    let v = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
    assert!(!v.uninformed);
    assert_eq!(v.coverage, 204, "0.8 · 255 rounded, from the first round");
    assert_eq!(v.age, 10, "age still counts from the first appearance");
}

/// State of voxels nothing occupies any more is dropped; reset drops everything.
#[test]
fn absent_voxels_are_pruned_from_memory() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    agg.cells(&row(0, 10), &[], 0.0);
    assert_eq!(agg.tracked_voxels(), 10);
    for i in 0..=PRUNE_AFTER_ROUNDS {
        agg.cells(&row(100, 10), &[], 1.0 + i as f64);
    }
    assert_eq!(agg.tracked_voxels(), 10, "only the live set is retained");
    agg.reset();
    assert_eq!(agg.tracked_voxels(), 0);
}

/// A voxel the Fisher pass keeps scoring is alive even if cells() never emits it.
#[test]
fn fisher_scored_voxel_survives_pruning() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let k = [g([0.5; 3], 1.0, 0.6, 0.1), g([5.5, 0.5, 0.5], 1.0, 0.2, 0.4)];
    for i in 0..=PRUNE_AFTER_ROUNDS + 1 {
        agg.update_fisher(&k);
        agg.cells(&row(5, 1), &[], i as f64);
    }
    let cells = agg.cells(&[row(0, 1), row(5, 1)].concat(), &[], 50.0);
    let v = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
    assert!(!v.uninformed, "still scored every round, so still tracked");
    assert_eq!(v.coverage, 153, "0.6 · 255 rounded");
}
