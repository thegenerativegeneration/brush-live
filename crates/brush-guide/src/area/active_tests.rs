use super::*;
use crate::area::AreaParams;
use brush_render::kernels::camera_model::CameraModel;
use glam::{Quat, Vec2};

fn params() -> AreaParams {
    AreaParams {
        tile_size: 10.0,
        radius: 10.0,
        hysteresis: 5.0,
        seen_window: 1000,
        min_resident_keyframes: 20,
        far_seed: 10.0,
    }
}

/// A 90° camera at `pos` looking along +z.
fn camera_at(pos: Vec3) -> Camera {
    let fov = 90f64.to_radians();
    Camera::new(pos, Quat::IDENTITY, fov, fov, Vec2::new(0.5, 0.5), CameraModel::Pinhole)
}

/// A registry with one row at 0 m and one at 3 m height in the middle of each of `tiles`.
fn registry_with(tiles: &[(i32, i32)]) -> TileRegistry {
    let mut r = TileRegistry::new(params());
    let means: Vec<f32> = tiles
        .iter()
        .flat_map(|&(x, z)| {
            let (cx, cz) = (x as f32 * 10.0 + 5.0, z as f32 * 10.0 + 5.0);
            [cx, 0.0, cz, cx, 3.0, cz]
        })
        .collect();
    r.note_resident(&means, &[], 0);
    r
}

fn key(x: i32, z: i32) -> TileKey {
    TileKey { x, z }
}

/// The camera stands in tile (0, 0) at x = 6 looking along +z. Ahead (0, 1) fills most of the view, ahead-right
/// (1, 1) is nearer the axis than ahead-left (-1, 1); (0, 2) starts 18 m deep and (0, -1) lies behind.
fn scene() -> (TileRegistry, Camera) {
    let r = registry_with(&[(0, 0), (0, 1), (1, 1), (-1, 1), (0, 2), (0, -1)]);
    (r, camera_at(Vec3::new(6.0, 1.5, 2.0)))
}

#[test]
fn the_active_set_is_the_tile_underfoot_then_the_largest_in_view() {
    let (r, cam) = scene();
    let boxes = candidate_boxes(&r, &BTreeMap::new(), cam.position);
    assert_eq!(rank_active(&cam, &boxes, 15.0, 3, 10.0), vec![key(0, 0), key(0, 1), key(1, 1)]);
    let wide = rank_active(&cam, &boxes, 15.0, 10, 10.0);
    assert!(wide.contains(&key(-1, 1)), "{wide:?}");
    assert!(!wide.contains(&key(0, 2)), "18 m deep is beyond the 15 m view: {wide:?}");
    assert!(!wide.contains(&key(0, -1)), "behind the camera: {wide:?}");
    assert!(rank_active(&cam, &boxes, 30.0, 10, 10.0).contains(&key(0, 2)), "within a 30 m view");
}

#[test]
fn the_tile_underfoot_is_active_without_rows_and_counts_toward_the_limit() {
    let r = registry_with(&[(0, 1), (1, 1), (-1, 1)]);
    let cam = camera_at(Vec3::new(6.0, 1.5, 2.0));
    let boxes = candidate_boxes(&r, &BTreeMap::new(), cam.position);
    assert_eq!(rank_active(&cam, &boxes, 15.0, 3, 10.0), vec![key(0, 0), key(0, 1), key(1, 1)]);
    assert_eq!(rank_active(&cam, &boxes, 15.0, 1, 10.0), vec![key(0, 0)]);
    assert_eq!(rank_active(&cam, &boxes, 15.0, 0, 10.0), vec![key(0, 0)], "0 counts as 1");
}

#[test]
fn parked_and_left_tiles_are_no_candidates_and_fresh_seeds_add_boxes() {
    let mut r = registry_with(&[(0, 0), (0, 1), (5, 0)]);
    r.mark_parked(key(0, 1), 2, 0);
    // The walker stood at tile (5, 0) and is now 40 m from it: left.
    r.note_walker(Vec3::new(55.0, 0.0, 5.0));
    let centre = Vec3::new(6.0, 1.5, 2.0);
    let fresh = BTreeMap::from([(key(1, 1), (0.0, 2.0)), (key(0, 0), (-1.0, 0.5))]);
    let boxes = candidate_boxes(&r, &fresh, centre);
    assert_eq!(boxes.keys().copied().collect::<Vec<_>>(), vec![key(0, 0), key(1, 1)]);
    // A fresh range widens the registry's (0 to 3 m) and a fresh tile gets its own.
    assert_eq!(boxes[&key(0, 0)], (Vec3::new(0.0, -1.0, 0.0), Vec3::new(10.0, 3.0, 10.0)));
    assert_eq!(boxes[&key(1, 1)], (Vec3::new(10.0, 0.0, 10.0), Vec3::new(20.0, 2.0, 20.0)));
}

#[test]
fn the_views_that_see_an_active_box_within_the_view_depth() {
    let boxes = [(Vec3::new(0.0, 0.0, 10.0), Vec3::new(10.0, 3.0, 20.0))];
    let cams = [
        camera_at(Vec3::new(5.0, 1.5, 2.0)),
        camera_at(Vec3::new(5.0, 1.5, -20.0)),
        Camera { rotation: Quat::from_rotation_y(std::f32::consts::PI), ..camera_at(Vec3::new(5.0, 1.5, 2.0)) },
    ];
    assert_eq!(views_seeing(&cams, &boxes, 15.0), vec![0], "30 m back is too deep; turned around sees nothing");
    assert_eq!(views_seeing(&cams, &boxes, 40.0), vec![0, 1]);
}

#[test]
fn seeds_land_in_active_tiles_and_on_new_ground_only() {
    let r = registry_with(&[(0, 0), (0, 1), (1, 1)]);
    let fresh = BTreeMap::from([(key(2, 2), (0.0, 1.0))]);
    let active = [key(0, 0), key(0, 1)];
    let seeds = [
        [5.0, 0.0, 5.0],  // active
        [15.0, 0.0, 15.0], // resident, not active
        [25.0, 0.0, 25.0], // seeded since the readback, not active
        [35.0, 0.0, 5.0],  // new ground
        [5.0, 0.0, 15.0],  // active, but the triage dropped it already
    ];
    let means: Vec<f32> = seeds.iter().flatten().copied().collect();
    let mut keep = vec![true, true, true, true, false];
    let outside = keep_active_seeds(&r, &fresh, &active, &means, &mut keep);
    assert_eq!(keep, vec![true, false, false, true, false]);
    assert_eq!(outside, SeedSort { dropped: 2, new_ground: 1 });
}

#[test]
fn seed_heights_per_tile() {
    let means = [5.0, 1.0, 5.0, 6.0, -2.0, 6.0, 15.0, 4.0, 5.0];
    let h = seed_heights(&means, 10.0);
    assert_eq!(h, BTreeMap::from([(key(0, 0), (-2.0, 1.0)), (key(1, 0), (4.0, 4.0))]));
}
