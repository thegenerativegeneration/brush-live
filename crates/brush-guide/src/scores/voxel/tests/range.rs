use super::*;

/// A voxel whose σ did not change keeps a high byte when a later pass's
/// population collapses around it. Per-pass normalisation would re-rank it
/// mid-range (byte ≈ 129, EMA → 217); against the smoothed range it stays
/// near the top (byte 253, EMA → 254).
#[test]
fn unchanged_sigma_survives_a_collapsing_pass_population() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let at = |cells: &[Cell]| {
        cells
            .iter()
            .find(|c| c.center[0] < 1.0)
            .unwrap()
            .uncertainty
    };
    // Pass 1: V (σ 0.4) tops a wide range against A (σ 0.1).
    let first = agg.aggregate(
        &[g([0.5; 3], 1.0, 0.0, 0.4), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.1)],
        &[],
        0.0,
    );
    assert_eq!(at(&first), 255);
    // Pass 2: A is gone; the pass range collapses to (0.39, 0.41) around V.
    let second = agg.aggregate(
        &[
            g([0.5; 3], 1.0, 0.0, 0.4),
            g([6.5, 0.5, 0.5], 1.0, 0.0, 0.39),
            g([7.5, 0.5, 0.5], 1.0, 0.0, 0.41),
        ],
        &[],
        1.0,
    );
    assert!(at(&second) >= 250, "got {}", at(&second));
}

/// A pass with no finite σ leaves the smoothed range untouched: a later
/// identical pass maps exactly as if the NaN pass never happened.
#[test]
fn nan_pass_leaves_range_unchanged() {
    let round = [g([0.5; 3], 1.0, 0.0, 0.4), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.1)];
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    agg.aggregate(&round, &[], 0.0);
    agg.aggregate(&[none(0.5)], &[], 1.0);
    let cells = agg.aggregate(&round, &[], 2.0);
    let v = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
    assert_eq!(v.uncertainty, 255, "same pass, same range, same byte");
}

/// reset() forgets the smoothed range: the next pass is its own reference.
#[test]
fn reset_clears_the_smoothed_range() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    agg.aggregate(
        &[g([0.5; 3], 1.0, 0.0, 0.1), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.4)],
        &[],
        0.0,
    );
    agg.reset();
    let cells = agg.aggregate(
        &[g([0.5; 3], 1.0, 0.0, 0.19), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.21)],
        &[],
        1.0,
    );
    let top = cells.iter().find(|c| c.center[0] > 1.0).unwrap();
    assert_eq!(top.uncertainty, 255, "fresh range from the first pass after reset");
}
