use super::*;
use crate::scores::metrics::FisherRidge;

/// No ridge and unit pixel noise: a voxel's σ is `1/sqrt(Σ opacity · info)`.
const SCALE: UncertaintyScale = UncertaintyScale {
    ridge: FisherRidge { abs: 0.0, rel: 0.0 },
    sigma_pix: 1.0,
};

/// Isotropic position Fisher of a Gaussian that alone, at opacity 1, has `sigma`.
fn info(sigma: f32) -> [f32; 9] {
    let v = 1.0 / (sigma * sigma);
    [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]
}

fn g(pos: [f32; 3], opacity: f32, coverage: f32, sigma: f32) -> GaussianScore {
    GaussianScore {
        pos: Vec3::from(pos),
        opacity,
        coverage,
        fisher_pos: info(sigma),
        axis: Vec3::Z,
        flatness: 1.0,
    }
}

fn ga(pos: [f32; 3], opacity: f32, axis: [f32; 3]) -> GaussianScore {
    GaussianScore {
        pos: Vec3::from(pos),
        opacity,
        coverage: 0.5,
        fisher_pos: info(0.1),
        axis: Vec3::from(axis),
        flatness: 1.0,
    }
}

/// A Gaussian with no usable information: infinite σ.
fn none(x: f32) -> GaussianScore {
    g([x, 0.5, 0.5], 1.0, 0.0, f32::INFINITY)
}

fn cam(pos: [f32; 3], fwd: [f32; 3]) -> ViewCone {
    ViewCone {
        position: Vec3::from(pos),
        forward: Vec3::from(fwd).normalize(),
        cos_half_fov: 0.5,
    }
}

/// Degenerate rounds: no input, a single voxel, a uniform round, and a round where the p5/p95 range collapses but
/// one voxel is still well above it.
#[test]
fn degenerate_rounds() {
    let mut agg = VoxelAggregator::new(0.1, 0.1, SCALE);
    assert!(agg.aggregate(&[], &[], 0.0).is_empty());

    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.3)], &[], 0.0);
    assert_eq!(cells[0].uncertainty, 0, "a single voxel round maps to zero");

    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(
        &[
            g([0.5; 3], 1.0, 0.0, 0.1),
            g([1.5, 0.5, 0.5], 1.0, 0.0, 0.1),
        ],
        &[],
        0.0,
    );
    assert!(
        cells.iter().all(|c| c.uncertainty == 0),
        "a uniform round maps to zero"
    );

    // 19 equal voxels put p5 and p95 on the same value; the 20th is higher.
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let mut gs: Vec<_> = (0..19)
        .map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, 0.1))
        .collect();
    gs.push(g([19.5, 0.5, 0.5], 1.0, 0.0, 0.5));
    let cells = agg.aggregate(&gs, &[], 0.0);
    assert!((0..19).all(|x| byte_at(&cells, x) == 0));
    assert_eq!(byte_at(&cells, 19), 255, "above the collapsed range");
}

#[test]
fn groups_by_voxel_and_weights_by_opacity() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let cells = agg.aggregate(
        &[
            g([0.2, 0.2, 0.2], 0.9, 1.0, 1.0),
            g([0.8, 0.1, 0.5], 0.3, 0.0, 1.0),
            g([5.5, 0.5, 0.5], 0.05, 1.0, 1.0),
        ],
        &[],
        0.0,
    );
    assert_eq!(cells.len(), 1, "low-opacity voxel dropped");
    // (0.9·(0.2, 0.2, 0.2) + 0.3·(0.8, 0.1, 0.5)) / 1.2
    let expected = [0.35, 0.175, 0.275];
    for (got, want) in cells[0].center.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    assert_eq!(cells[0].coverage, (0.75f32 * 255.0).round() as u8);
}

/// One voxel per unit along x, voxel `i` with σ = 0.01 · 1.05^i · `factor`.
fn ladder(factor: f32) -> Vec<Cell> {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let gs: Vec<_> = (0..100)
        .map(|i| {
            let s = 0.01 * 1.05f32.powi(i) * factor;
            g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, s)
        })
        .collect();
    let mut cells = agg.aggregate(&gs, &[], 0.0);
    cells.sort_by(|a, b| a.center[0].total_cmp(&b.center[0]));
    cells
}

#[test]
fn log_sigma_maps_round_p5_to_0_and_p95_to_255() {
    let cells = ladder(1.0);
    let bytes: Vec<u8> = cells.iter().map(|c| c.uncertainty).collect();
    // p5 is voxel round(99 · 0.05) = 5, p95 voxel round(99 · 0.95) = 94.
    assert!(
        bytes[..=5].iter().all(|&b| b == 0),
        "at and below p5: {bytes:?}"
    );
    assert!(
        bytes[94..].iter().all(|&b| b == 255),
        "at and above p95: {bytes:?}"
    );
    // Log σ is linear in the index: voxel 50 sits at (50 − 5) / (94 − 5).
    assert_eq!(bytes[50], (45.0f32 / 89.0 * 255.0).round() as u8);
    assert!(bytes.windows(2).all(|w| w[0] <= w[1]), "monotone in σ");
}

#[test]
fn scaling_every_sigma_leaves_the_bytes_unchanged() {
    let bytes = |f: f32| ladder(f).iter().map(|c| c.uncertainty).collect::<Vec<_>>();
    assert_eq!(bytes(1.0), bytes(3.7));
    assert_eq!(bytes(1.0), bytes(0.02));
}

/// Byte of the voxel at unit `x` in `cells`.
fn byte_at(cells: &[Cell], x: usize) -> u8 {
    cells
        .iter()
        .find(|c| c.center[0].floor() as usize == x)
        .unwrap()
        .uncertainty
}

/// Uninformed table: a round with only infinite sigma maps every voxel to 255 and uninformed, and a voxel with no
/// information at all (infinite sigma, full coverage) gets the uninformed coverage placeholder too.
#[test]
fn rounds_without_information_are_uninformed() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(&[none(0.5), none(1.5)], &[], 0.0);
    assert!(cells.iter().all(|c| c.uncertainty == 255 && c.uninformed));

    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(&[g([0.5; 3], 1.0, 1.0, f32::INFINITY)], &[], 0.0);
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].coverage, UNINFORMED_COVERAGE);
    assert_eq!(cells[0].uncertainty, 255);
    assert!(cells[0].uninformed);
}

#[test]
fn infinite_sigma_does_not_shift_the_percentiles() {
    let mut plain = VoxelAggregator::new(1.0, 0.0, SCALE);
    let mut mixed = VoxelAggregator::new(1.0, 0.0, SCALE);
    let finite: Vec<_> = (0..20)
        .map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, 0.01 * 1.2f32.powi(i)))
        .collect();
    let mut with_inf = finite.clone();
    with_inf.extend((20..30).map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, f32::INFINITY)));
    let a = plain.aggregate(&finite, &[], 0.0);
    let b = mixed.aggregate(&with_inf, &[], 0.0);
    for x in 0..20 {
        assert_eq!(byte_at(&a, x), byte_at(&b, x), "voxel {x}");
    }
    assert!((20..30).all(|x| byte_at(&b, x) == 255));
    assert!(b.iter().filter(|c| c.uninformed).count() == 10);
}

#[test]
fn uninformed_voxel_is_infinite_under_the_production_ridge() {
    let mut agg = VoxelAggregator::new(
        1.0,
        0.0,
        crate::config::GuideConfig::default().uncertainty_scale(),
    );
    agg.record_raw(true);
    let blank = |x: f32| GaussianScore {
        fisher_pos: [0.0; 9],
        ..g([x, 0.5, 0.5], 1.0, 0.0, 1.0)
    };
    let nan = GaussianScore {
        fisher_pos: [f32::NAN; 9],
        ..g([1.5, 0.5, 0.5], 1.0, 0.0, 1.0)
    };
    let mut gs = vec![blank(0.5), nan];
    gs.extend((2..12).map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, 0.01 * 1.3f32.powi(i))));
    let cells = agg.aggregate(&gs, &[], 0.0);
    let raw = |x: i32| agg.raw_round().iter().find(|r| r.key.x == x).unwrap().sigma;
    assert_eq!(raw(0), f32::INFINITY, "all-zero information");
    assert_eq!(raw(1), f32::INFINITY, "only non-finite blocks");
    assert_eq!(byte_at(&cells, 0), 255);
    assert_eq!(byte_at(&cells, 1), 255);
    assert_eq!(byte_at(&cells, 2), 0, "lowest finite σ is the round's p5");
    assert_eq!(
        byte_at(&cells, 11),
        255,
        "highest finite σ is the round's p95"
    );
}

#[test]
fn splitting_a_gaussian_keeps_the_voxel_sigma() {
    let whole = GaussianScore {
        fisher_pos: [200.0, 20.0, 0.0, 20.0, 50.0, 0.0, 0.0, 0.0, 80.0],
        ..g([0.5; 3], 0.8, 0.5, 1.0)
    };
    let half = GaussianScore {
        fisher_pos: whole.fisher_pos.map(|v| v / 2.0),
        ..whole
    };
    let sigma = |gs: &[GaussianScore]| {
        let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
        agg.record_raw(true);
        agg.aggregate(gs, &[], 0.0);
        agg.raw_round()[0].sigma
    };
    let one = sigma(&[whole]);
    let two = sigma(&[
        half,
        GaussianScore {
            pos: Vec3::splat(0.6),
            ..half
        },
    ]);
    assert!(one.is_finite() && one > 0.0);
    assert!((one - two).abs() < 1e-6 * one, "{one} vs {two}");
}

#[test]
fn uncertainty_is_smoothed_across_rounds_per_voxel() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    // Two voxels so the round has a range; the other one is fixed at 0.1.
    let round = |s: f32| [g([0.5; 3], 1.0, 0.0, s), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.1)];
    let at = |cells: &[Cell]| {
        cells
            .iter()
            .find(|c| c.center[0] < 1.0)
            .unwrap()
            .uncertainty
    };
    let first = agg.aggregate(&round(0.4), &[], 0.0);
    assert_eq!(
        at(&first),
        255,
        "highest in its round; a new voxel starts at its value"
    );
    let second = agg.aggregate(&round(0.025), &[], 1.0);
    // Round value 0 (now the lowest); sent value = round(0.3·0 + 0.7·255) = 179.
    assert_eq!(at(&second), 179);
}

#[test]
fn reset_clears_uncertainty_history() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let round = |s: f32| [g([0.5; 3], 1.0, 0.0, s), g([5.5, 0.5, 0.5], 1.0, 0.0, 0.1)];
    agg.aggregate(&round(0.4), &[], 0.0);
    agg.reset();
    let c = agg.aggregate(&round(0.025), &[], 1.0);
    assert_eq!(c.iter().find(|c| c.center[0] < 1.0).unwrap().uncertainty, 0);
}

#[test]
fn raw_round_records_sigma_and_coverage_per_voxel() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    agg.record_raw(true);
    agg.aggregate(
        &[
            g([0.5; 3], 1.0, 1.0, 0.1),
            g([0.6; 3], 3.0, 0.0, 0.1),
            g([2.5, 0.5, 0.5], 1.0, 0.5, f32::NAN),
        ],
        &[],
        0.0,
    );
    let mut raw = agg.raw_round().to_vec();
    raw.sort_by_key(|r| r.key.x);
    assert_eq!(raw.len(), 2);
    assert_eq!(raw[0].key, IVec3::ZERO);
    // Σ opacity · info = 4 · 100 → σ = 0.05.
    assert!((raw[0].sigma - 0.05).abs() < 1e-6);
    assert!((raw[0].coverage - 0.25).abs() < 1e-6);
    assert_eq!(raw[1].key, IVec3::new(2, 0, 0));
    assert_eq!(raw[1].sigma, f32::INFINITY);
    agg.aggregate(&[g([0.5; 3], 1.0, 1.0, 0.1)], &[], 1.0);
    assert_eq!(agg.raw_round().len(), 1, "only the latest round");
}

#[test]
fn age_counts_from_first_appearance_and_saturates() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], &[], 10.0);
    let cells = agg.aggregate(
        &[
            g([0.5; 3], 1.0, 0.0, 0.0),
            g([2.5, 0.5, 0.5], 1.0, 0.0, 0.0),
        ],
        &[],
        13.4,
    );
    let age = |x: f32| cells.iter().find(|c| c.center[0] == x).unwrap().age;
    assert_eq!(age(0.5), 3);
    assert_eq!(age(2.5), 0);
    let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], &[], 1000.0);
    assert_eq!(cells[0].age, 255);
}

#[test]
fn nan_scores_count_as_uncovered_and_uninformative() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    agg.record_raw(true);
    let nan = GaussianScore {
        fisher_pos: [f32::NAN; 9],
        ..g([0.5; 3], 1.0, f32::NAN, 1.0)
    };
    let cells = agg.aggregate(
        &[
            nan,
            g([0.6; 3], 1.0, 1.0, 0.1),
            g([1.5, 0.5, 0.5], 1.0, 1.0, 0.1),
            GaussianScore {
                pos: Vec3::new(2.5, 0.5, 0.5),
                ..nan
            },
        ],
        &[],
        0.0,
    );
    assert_eq!(cells.len(), 3);
    let first = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
    assert!((first.center[0] - 0.55).abs() < 1e-6);
    assert_eq!(first.coverage, 128);
    let raw = agg
        .raw_round()
        .iter()
        .find(|r| r.key == IVec3::ZERO)
        .unwrap();
    assert!(
        (raw.sigma - 0.1).abs() < 1e-6,
        "the NaN Gaussian adds no information"
    );
    let only_nan = cells.iter().find(|c| c.center[0] > 2.0).unwrap();
    assert_eq!(only_nan.uncertainty, 255, "non-finite sigma maps to 255");
    assert!(only_nan.uninformed && !first.uninformed);
}

#[test]
fn non_finite_position_or_opacity_is_skipped() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(
        &[
            g([f32::NAN, 0.5, 0.5], 1.0, 1.0, 1.0),
            g([0.5; 3], f32::NAN, 0.0, 0.0),
            g([0.6; 3], 1.0, 1.0, 1.0),
        ],
        &[],
        0.0,
    );
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].coverage, 255);
}

mod legacy;
mod lifecycle;
mod normals;
mod range;
