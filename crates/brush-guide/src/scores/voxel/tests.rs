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

fn cam(pos: [f32; 3], fwd: [f32; 3]) -> ViewCone {
    ViewCone {
        position: Vec3::from(pos),
        forward: Vec3::from(fwd).normalize(),
        cos_half_fov: 0.5,
    }
}

#[test]
fn empty_input_gives_no_cells() {
    let mut agg = VoxelAggregator::new(0.1, 0.1, SCALE);
    assert!(agg.aggregate(&[], &[], 0.0).is_empty());
}

#[test]
fn groups_by_voxel_and_weights_by_opacity() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let cells = agg.aggregate(
        &[
            g([0.2, 0.2, 0.2], 0.9, 1.0, 0.0),
            g([0.8, 0.1, 0.5], 0.3, 0.0, 0.0),
            g([5.5, 0.5, 0.5], 0.05, 1.0, 0.0),
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

#[test]
fn uniform_round_maps_to_zero() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(
        &[
            g([0.5; 3], 1.0, 0.0, 0.1),
            g([1.5, 0.5, 0.5], 1.0, 0.0, 0.1),
        ],
        &[],
        0.0,
    );
    assert!(cells.iter().all(|c| c.uncertainty == 0));
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
fn more_gaussians_observing_a_voxel_lower_its_sigma() {
    let sigma = |n: usize| {
        let gs: Vec<_> = (0..n).map(|_| g([0.5; 3], 1.0, 0.0, 0.2)).collect();
        let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
        agg.aggregate(&gs, &[], 0.0);
        agg.raw_round()[0].sigma
    };
    assert!((sigma(1) - 0.2).abs() < 1e-6);
    assert!((sigma(4) - 0.1).abs() < 1e-6);
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
fn voxel_without_information_keeps_coverage_and_is_maximally_uncertain() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
    let cells = agg.aggregate(&[g([0.5; 3], 1.0, 1.0, f32::INFINITY)], &[], 0.0);
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].coverage, 255);
    assert_eq!(cells[0].uncertainty, 255);
}

#[test]
fn nan_scores_count_as_uncovered_and_uninformative() {
    let mut agg = VoxelAggregator::new(1.0, 0.0, SCALE);
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

#[test]
fn flat_patch_gets_normal_facing_the_cameras() {
    // Patch at z = 0.5 inside voxel [0,1)^3; Gaussian axes point ±z (unsigned).
    let gs: Vec<_> = (0..20)
        .map(|i| {
            let x = 0.1 + 0.04 * i as f32;
            ga(
                [x, 0.9 - 0.04 * i as f32, 0.5],
                0.8,
                if i % 2 == 0 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 0.0, -1.0]
                },
            )
        })
        .collect();
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let above = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
    assert!(Vec3::from(above[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
    let below = agg.aggregate(&gs, &[cam([0.5, 0.5, -3.0], [0.0, 0.0, 1.0])], 0.0);
    assert!(Vec3::from(below[0].normal.unwrap()).dot(Vec3::NEG_Z) > 0.99);
}

#[test]
fn crease_voxel_has_no_normal() {
    let gs: Vec<_> = (0..10)
        .map(|i| {
            ga(
                [0.5, 0.5, 0.5],
                0.8,
                if i < 5 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                },
            )
        })
        .collect();
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    assert_eq!(agg.aggregate(&gs, &[], 0.0)[0].normal, None);
}

#[test]
fn light_voxel_has_no_normal() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    assert_eq!(
        agg.aggregate(&[ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])], &[], 0.0)[0].normal,
        None
    );
}

#[test]
fn center_is_opacity_weighted_mean() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let c = agg.aggregate(
        &[
            ga([0.2, 0.5, 0.5], 0.9, [0.0, 0.0, 1.0]),
            ga([0.8, 0.5, 0.5], 0.1, [0.0, 0.0, 1.0]),
        ],
        &[],
        0.0,
    );
    assert!((c[0].center[0] - 0.26).abs() < 1e-5);
}

#[test]
fn density_sums_opacity_and_saturates() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let three: Vec<_> = (0..3)
        .map(|_| ga([0.5, 0.5, 0.5], 0.5, [0.0, 0.0, 1.0]))
        .collect();
    assert_eq!(agg.aggregate(&three, &[], 0.0)[0].density, 48);
    let many: Vec<_> = (0..10)
        .map(|_| ga([0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 1.0]))
        .collect();
    assert_eq!(agg.aggregate(&many, &[], 0.0)[0].density, 255);
}

#[test]
fn unseen_voxel_orients_toward_nearest_camera() {
    let gs: Vec<_> = (0..5)
        .map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0]))
        .collect();
    // Camera looks away (+x) from the voxel, so it does not "see" it; nearest-camera fallback applies.
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, -4.0], [1.0, 0.0, 0.0])], 0.0);
    assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::NEG_Z) > 0.99);
}

#[test]
fn round_gaussians_do_not_vote_on_the_normal() {
    // 20 round Gaussians whose arbitrary shortest axes point along x, plus
    // 3 flat ones along z: the normal comes from the flat ones.
    let mut gs: Vec<_> = (0..20)
        .map(|_| GaussianScore {
            flatness: 0.02,
            ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
        })
        .collect();
    gs.extend((0..3).map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])));
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
    assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
}

#[test]
fn only_round_gaussians_give_no_normal() {
    let gs: Vec<_> = (0..20)
        .map(|_| GaussianScore {
            flatness: 0.0,
            ..ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])
        })
        .collect();
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let c = agg.aggregate(&gs, &[], 0.0);
    assert_eq!(c[0].normal, None);
    assert_eq!(c[0].density, 255, "density stays opacity-weighted");
}

#[test]
fn weight_rule_uses_opacity_mass_and_a_smaller_flat_mass() {
    let round = |n: usize| {
        (0..n).map(|_| GaussianScore {
            flatness: 0.0,
            ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
        })
    };
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    // Opacity mass 1.0, flat mass 0.2: enough for a normal.
    let mut gs: Vec<_> = round(1).collect();
    gs.push(ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0]));
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
    assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
    // Opacity mass 1.8, flat mass 0.05: too little orientation.
    let mut gs: Vec<_> = round(2).collect();
    gs.push(GaussianScore {
        flatness: 0.25,
        ..ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])
    });
    assert_eq!(agg.aggregate(&gs, &[], 0.0)[0].normal, None);
}
