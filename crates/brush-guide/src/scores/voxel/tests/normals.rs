//! Voxel normals, centres and density.

use super::*;

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
