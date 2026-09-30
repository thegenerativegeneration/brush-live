use brush_render::kernels::camera_model::CameraModel;
use glam::{UVec2, Vec2, vec3};

use super::fixtures::{
    ROD_CENTRES, aabb, depth_image, depth_image_sized, integrate, look_at, look_at_with,
    plane_setup, plane_z, rod_tsdf, thin_sheet_tsdf, union,
};
use super::*;

/// Sign changes of the sdf sampled every 5 mm from `a` to `b`, counting
/// only samples whose eight surrounding voxels are all observed (the
/// cells a mesher keeps when it drops cells touching weight 0).
fn zero_crossings(tsdf: &Tsdf, a: Vec3, b: Vec3) -> Vec<Vec3> {
    let n = ((b - a).length() / 0.005).round() as usize;
    let samples: Vec<(Vec3, Option<f32>)> = (0..=n)
        .map(|i| {
            let p = a.lerp(b, i as f32 / n as f32);
            (p, tsdf.sdf(p))
        })
        .collect();
    samples
        .windows(2)
        .filter_map(|w| match (w[0], w[1]) {
            ((p0, Some(s0)), (p1, Some(s1))) if (s0 > 0.0) != (s1 > 0.0) => {
                Some(p0.lerp(p1, s0 / (s0 - s1)))
            }
            _ => None,
        })
        .collect()
}

fn voxel_weight(tsdf: &Tsdf, world: Vec3) -> f32 {
    let g = (world / VOXEL).floor().as_ivec3();
    tsdf.voxel(g).map_or(0.0, |(_, w)| w)
}

#[test]
fn plane_from_three_poses() {
    let target = vec3(0.0, 0.0, 2.0);
    let cams = [
        look_at(Vec3::ZERO, target),
        look_at(vec3(0.5, 0.0, 0.2), target),
        look_at(vec3(-0.3, 0.4, 0.1), target),
    ];
    let mut tsdf = Tsdf::new();
    integrate(&mut tsdf, &cams, &plane_z(2.0));

    for x in [-0.5, -0.23, 0.0, 0.17, 0.5] {
        for y in [-0.5, -0.11, 0.0, 0.31, 0.5] {
            let on = tsdf.sdf(vec3(x, y, 2.0)).expect("plane observed");
            assert!(on.abs() < 0.01, "({x},{y}): sdf {on}");
            let front = tsdf.sdf(vec3(x, y, 1.95)).expect("front observed");
            assert!(front > 0.02, "({x},{y}): front sdf {front}");
            let behind = tsdf.sdf(vec3(x, y, 2.05)).expect("behind observed");
            assert!(behind < -0.02, "({x},{y}): behind sdf {behind}");
        }
    }
    assert_eq!(
        tsdf.sdf(vec3(0.0, 0.0, 2.5)),
        None,
        "beyond truncation is unobserved"
    );
}

#[test]
fn box_faces_from_four_sides() {
    let center = vec3(0.0, 0.0, 2.0);
    let cams = [
        look_at(vec3(0.0, 0.0, 0.7), center),
        look_at(vec3(1.3, 0.0, 2.0), center),
        look_at(vec3(0.0, 0.0, 3.3), center),
        look_at(vec3(-1.3, 0.0, 2.0), center),
    ];
    let mut tsdf = Tsdf::new();
    integrate(&mut tsdf, &cams, &aabb(center - 0.2, center + 0.2));

    for normal in [Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_X] {
        let face = center + 0.2 * normal;
        let on = tsdf.sdf(face).expect("face observed");
        assert!(on.abs() < 0.01, "face {normal}: sdf {on}");
        let outside = tsdf.sdf(face + 0.05 * normal).expect("outside observed");
        assert!(outside > 0.02, "face {normal}: outside sdf {outside}");
        let inside = tsdf.sdf(face - 0.05 * normal).expect("inside observed");
        assert!(inside < -0.02, "face {normal}: inside sdf {inside}");
    }
}

#[test]
fn thin_sheet_seen_from_both_sides_keeps_a_zero_crossing() {
    let tsdf = thin_sheet_tsdf();

    for (x, y) in [(0.0, 0.0), (0.21, -0.13), (-0.4, 0.33)] {
        let crossings = zero_crossings(
            &tsdf,
            vec3(x, y, 2.0 - 2.0 * TRUNC),
            vec3(x, y, 2.0 + 2.0 * TRUNC),
        );
        assert!(!crossings.is_empty(), "({x},{y}): no zero crossing");
        for c in &crossings {
            assert!(
                (c.z - 2.0).abs() <= VOXEL,
                "({x},{y}): crossing at z = {}",
                c.z
            );
        }
    }
}

/// Zero crossings of the rod of `rod_tsdf` along x and z at three heights.
fn rod_crossings(center: Vec3, width: f32) -> Vec<(Vec3, Vec<Vec3>)> {
    let tsdf = rod_tsdf(center, width);
    let mut all = Vec::new();
    for y in [0.0, 0.12, -0.27] {
        for dir in [Vec3::X, Vec3::Z] {
            let c = center + vec3(0.0, y, 0.0);
            let crossings = zero_crossings(&tsdf, c - 2.0 * TRUNC * dir, c + 2.0 * TRUNC * dir);
            all.push((dir, crossings));
        }
    }
    all
}

#[test]
fn rod_seen_from_four_sides_keeps_a_zero_crossing() {
    for center in ROD_CENTRES {
        for (dir, crossings) in rod_crossings(center, 0.05) {
            assert!(
                !crossings.is_empty(),
                "rod at {center} along {dir}: no zero crossing"
            );
            for c in crossings {
                let off_axis = ((c - center) * dir).length();
                assert!(
                    (off_axis - 0.025).abs() <= VOXEL,
                    "rod at {center} along {dir}: crossing {off_axis} m off the axis"
                );
            }
        }
    }
}

#[test]
#[ignore = "below the 5 cm voxel resolution"]
fn thin_rod_seen_from_four_sides_keeps_a_zero_crossing() {
    for center in ROD_CENTRES {
        for (dir, crossings) in rod_crossings(center, 0.02) {
            assert!(
                !crossings.is_empty(),
                "rod at {center} along {dir}: no zero crossing"
            );
        }
    }
}

#[test]
fn decay_scales_weights_down_to_the_floor() {
    let (mut tsdf, cam) = plane_setup();
    let p = vec3(0.025, 0.025, 2.475);
    let depth = depth_image(&cam, plane_z(2.5));
    for _ in 0..9 {
        tsdf.integrate(&depth, &cam);
    }
    let before = tsdf.sdf(p).unwrap();
    assert_eq!(voxel_weight(&tsdf, p), 10.0);
    tsdf.decay(0.5);
    assert_eq!(voxel_weight(&tsdf, p), 5.0);
    assert_eq!(tsdf.sdf(p), Some(before), "decay keeps distances");
    tsdf.decay(0.5);
    assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR + 0.5);
    tsdf.decay(0.5);
    assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR, "clamped to the floor");
    tsdf.decay(0.5);
    assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR);

    let (mut single, _) = plane_setup();
    single.decay(0.5);
    assert_eq!(voxel_weight(&single, p), 1.0, "below the floor: untouched");
}

#[test]
fn weights_are_capped() {
    let (mut tsdf, cam) = plane_setup();
    let depth = depth_image(&cam, plane_z(2.5));
    for _ in 0..30 {
        tsdf.integrate(&depth, &cam);
    }
    assert_eq!(voxel_weight(&tsdf, vec3(0.025, 0.025, 2.475)), MAX_WEIGHT);
}

fn sample_index(x: i32, y: i32, z: i32) -> usize {
    (x + 1) as usize + PADDED * ((y + 1) as usize + PADDED * (z + 1) as usize)
}

#[test]
fn brick_samples_are_padded_x_fastest() {
    let (tsdf, _) = plane_setup();
    let key = BrickKey(IVec3::new(0, 0, 2));
    let samples = tsdf.brick_samples(key).expect("allocated brick");
    let n = PADDED * PADDED * PADDED;
    assert_eq!((samples.sdf.len(), samples.weight.len()), (n, n));
    assert!(
        samples
            .sdf
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
    );

    let voxel = |x: i32, y: i32, z: i32| IVec3::new(x, y, 2 * BRICK + z);
    // Interior voxels and padding taken from the neighbours at -x, -y, +x+y.
    for (x, y, z) in [
        (3, 7, 9),
        (0, 0, 10),
        (19, 5, 8),
        (-1, 4, 9),
        (6, -1, 11),
        (20, 20, 10),
    ] {
        let (t, w) = tsdf.voxel(voxel(x, y, z)).expect("allocated");
        assert!(w > 0.0, "({x},{y},{z}) unobserved");
        let i = sample_index(x, y, z);
        assert_eq!((samples.sdf[i], samples.weight[i]), (t, w), "({x},{y},{z})");
    }
    // Plane at 2.5 m: z index 9 is 2.475 m (in front), 10 is 2.525 m (behind).
    assert!(samples.sdf[sample_index(5, 5, 9)] > 0.0);
    assert!(samples.sdf[sample_index(5, 5, 10)] < 0.0);

    assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(9, 9, 9))), None);
}

/// Seen from one side only, everything more than TRUNC behind the plane
/// stays unobserved: weight 0 and distance +1.
#[test]
fn one_sided_plane_leaves_samples_beyond_truncation_unobserved() {
    let (tsdf, _) = plane_setup();
    let samples = tsdf.brick_samples(BrickKey(IVec3::new(0, 0, 2))).unwrap();
    for x in -1..=BRICK {
        for y in -1..=BRICK {
            // z index 13 is 2.675 m, 17.5 cm behind the plane at 2.5 m.
            for z in 13..=BRICK {
                let i = sample_index(x, y, z);
                assert_eq!(samples.weight[i], 0.0, "({x},{y},{z})");
                assert_eq!(samples.sdf[i], 1.0, "({x},{y},{z})");
            }
            // Up to one voxel behind the plane observations count fully.
            assert_eq!(samples.weight[sample_index(x, y, 10)], 1.0, "({x},{y},10)");
        }
    }
}

/// A voxel becomes observed at `MIN_WEIGHT` and stays observed down to
/// `UNOBSERVED_WEIGHT`.
#[test]
fn observed_status_has_hysteresis() {
    let (mut tsdf, _) = plane_setup();
    let key = BrickKey(IVec3::new(0, 0, 2));
    let p = vec3(0.025, 0.025, 2.475);
    let g = voxel_of(p);
    let i = sample_index(0, 0, 9);
    let (t, _) = tsdf.voxel(g).unwrap();
    tsdf.take_changed();

    tsdf.set_voxel(g, t, 0.07);
    assert!(
        tsdf.sdf(p).is_some(),
        "observed voxel stays observed at 0.07"
    );
    assert_eq!(tsdf.brick_samples(key).unwrap().weight[i], 0.07);

    tsdf.set_voxel(g, t, 0.04);
    assert_eq!(tsdf.sdf(p), None, "below UNOBSERVED_WEIGHT");
    let samples = tsdf.brick_samples(key).unwrap();
    assert_eq!((samples.sdf[i], samples.weight[i]), (1.0, 0.0));
    assert!(
        tsdf.take_changed().contains(&key),
        "near-surface voxel lost is a change"
    );

    tsdf.set_voxel(g, t, 0.07);
    assert_eq!(
        tsdf.sdf(p),
        None,
        "unobserved voxel stays unobserved at 0.07"
    );
    tsdf.set_voxel(g, t, MIN_WEIGHT);
    assert!(tsdf.sdf(p).is_some(), "observed again at MIN_WEIGHT");
}

/// A 160×120 image with the principal point off centre and different
/// horizontal and vertical fov: a box in front of a wall, seen from one
/// pose, lands where it is.
#[test]
fn non_square_image_with_offset_principal_point() {
    let size = UVec2::new(160, 120);
    let cam = look_at_with(
        Vec3::ZERO,
        Vec3::Z,
        70f64.to_radians(),
        50f64.to_radians(),
        Vec2::new(0.4, 0.6),
    );
    let (lo, hi) = (vec3(0.2, -0.4, 1.5), vec3(0.6, -0.1, 1.9));
    let scene = union(aabb(lo, hi), plane_z(2.5));
    let mut tsdf = Tsdf::new();
    tsdf.integrate(&depth_image_sized(&cam, size, &scene), &cam);

    for (x, y) in [(0.25, -0.35), (0.55, -0.15), (0.4, -0.25)] {
        let on = tsdf.sdf(vec3(x, y, 1.5)).expect("box face observed");
        assert!(on.abs() < 0.01, "box face ({x},{y}): sdf {on}");
    }
    // Beside the box the rays reach the wall: free space at the face depth.
    for (x, y) in [(0.1, -0.25), (0.7, -0.25), (0.4, -0.5), (0.4, 0.0)] {
        let beside = tsdf.sdf(vec3(x, y, 1.5)).expect("free space observed");
        assert!(beside > 0.1, "beside the box ({x},{y}): sdf {beside}");
    }
    for (x, y) in [(-0.3, 0.3), (1.0, 0.5), (-0.6, -0.9), (0.1, -0.25)] {
        let wall = tsdf.sdf(vec3(x, y, 2.5)).expect("wall observed");
        assert!(wall.abs() < 0.01, "wall ({x},{y}): sdf {wall}");
    }
}

#[test]
fn reset_empties() {
    let (mut tsdf, _) = plane_setup();
    tsdf.reset();
    assert_eq!(tsdf.sdf(vec3(0.0, 0.0, 2.5)), None);
    assert!(tsdf.take_changed().is_empty());
    assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(0, 0, 2))), None);
}

#[test]
fn depth_beyond_max_depth_is_ignored() {
    let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 1.0));
    let mut tsdf = Tsdf::new();
    tsdf.integrate(&depth_image(&cam, plane_z(MAX_DEPTH + 0.5)), &cam);
    assert!(tsdf.bricks.is_empty(), "no bricks beyond MAX_DEPTH");

    let near = MAX_DEPTH - 0.5;
    tsdf.integrate(&depth_image(&cam, plane_z(near)), &cam);
    let sdf = tsdf.sdf(vec3(0.0, 0.0, near)).expect("observed");
    assert!(sdf.abs() < 0.01, "{sdf}");
}

#[test]
#[should_panic(expected = "pinhole")]
#[cfg(debug_assertions)]
fn integrate_rejects_non_pinhole_cameras() {
    let mut cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 1.0));
    let depth = depth_image(&cam, plane_z(2.0));
    cam.camera_model = CameraModel::KannalaBrandt4(Default::default());
    Tsdf::new().integrate(&depth, &cam);
}
