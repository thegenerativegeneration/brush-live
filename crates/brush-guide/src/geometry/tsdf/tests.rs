use glam::{UVec2, Vec2, vec3};

use super::fixtures::{
    aabb, depth_image, depth_image_sized, integrate, look_at, look_at_with, plane_setup, plane_z,
    union,
};
use super::*;

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
fn decay_scales_weights_down_to_the_floor() {
    let (mut tsdf, cam) = plane_setup();
    let p = vec3(0.025, 0.025, 2.475);
    let depth = depth_image(&cam, plane_z(2.5));
    for _ in 0..9 {
        tsdf.integrate(&depth, None, &cam);
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
        tsdf.integrate(&depth, None, &cam);
    }
    assert_eq!(voxel_weight(&tsdf, vec3(0.025, 0.025, 2.475)), MAX_WEIGHT);
}

fn sample_index(x: i32, y: i32, z: i32) -> usize {
    (x + 1) as usize + PADDED * ((y + 1) as usize + PADDED * (z + 1) as usize)
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
    tsdf.integrate(&depth_image_sized(&cam, size, &scene), None, &cam);

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
    tsdf.integrate(&depth_image(&cam, plane_z(MAX_DEPTH + 0.5)), None, &cam);
    assert!(tsdf.bricks.is_empty(), "no bricks beyond MAX_DEPTH");

    let near = MAX_DEPTH - 0.5;
    tsdf.integrate(&depth_image(&cam, plane_z(near)), None, &cam);
    let sdf = tsdf.sdf(vec3(0.0, 0.0, near)).expect("observed");
    assert!(sdf.abs() < 0.01, "{sdf}");
}

