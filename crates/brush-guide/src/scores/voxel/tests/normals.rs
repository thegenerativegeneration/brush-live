//! Voxel normals, centres and density.

use super::*;

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

/// A flat patch orients toward whichever camera sees it; when no camera sees it, the nearest camera wins instead.
#[test]
fn normal_orients_toward_a_seeing_or_the_nearest_camera() {
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

    let gs: Vec<_> = (0..5)
        .map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0]))
        .collect();
    // Camera looks away (+x) from the voxel, so it does not "see" it; nearest-camera fallback applies.
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, -4.0], [1.0, 0.0, 0.0])], 0.0);
    assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::NEG_Z) > 0.99);
}

/// When a voxel gets no normal: a crease (axes cancel), too little opacity mass, only round Gaussians, or not
/// enough flat-weighted mass even with some opacity. Round Gaussians never vote, and the opacity/flatness weight
/// rule is what decides the boundary.
#[test]
fn a_voxel_gets_a_normal_only_from_enough_flat_weighted_mass() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);

    // Crease: axes cancel.
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
    assert_eq!(agg.aggregate(&gs, &[], 0.0)[0].normal, None, "crease");

    // Too little opacity mass.
    assert_eq!(
        agg.aggregate(&[ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])], &[], 0.0)[0].normal,
        None,
        "light voxel"
    );

    // 20 round Gaussians whose arbitrary shortest axes point along x, plus 3 flat ones along z: the normal comes
    // from the flat ones, the round ones do not vote.
    let mut gs: Vec<_> = (0..20)
        .map(|_| GaussianScore {
            flatness: 0.02,
            ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
        })
        .collect();
    gs.extend((0..3).map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])));
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
    assert!(
        Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99,
        "round Gaussians do not vote"
    );

    // Only round Gaussians: no normal, but density stays opacity-weighted.
    let gs: Vec<_> = (0..20)
        .map(|_| GaussianScore {
            flatness: 0.0,
            ..ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])
        })
        .collect();
    let c = agg.aggregate(&gs, &[], 0.0);
    assert_eq!(c[0].normal, None, "only round Gaussians");
    assert_eq!(c[0].density, 255, "density stays opacity-weighted");

    // Weight rule: opacity mass 1.0, flat mass 0.2 is enough; opacity mass 1.8, flat mass 0.05 is too little.
    let round = |n: usize| {
        (0..n).map(|_| GaussianScore {
            flatness: 0.0,
            ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
        })
    };
    let mut gs: Vec<_> = round(1).collect();
    gs.push(ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0]));
    let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
    assert!(
        Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99,
        "enough flat mass"
    );
    let mut gs: Vec<_> = round(2).collect();
    gs.push(GaussianScore {
        flatness: 0.25,
        ..ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])
    });
    assert_eq!(
        agg.aggregate(&gs, &[], 0.0)[0].normal,
        None,
        "too little flat mass"
    );
}
