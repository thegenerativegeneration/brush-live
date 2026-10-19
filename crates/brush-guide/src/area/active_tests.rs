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

fn rule(view_m: f32, max: u32) -> ActiveRule {
    ActiveRule { view_m, max, min_rows: 1, tile_size: 10.0 }
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

fn candidates(r: &TileRegistry, centre: Vec3) -> BTreeMap<TileKey, Candidate> {
    candidate_boxes(r, &BTreeMap::new(), centre)
}

/// The camera stands in tile (0, 0) at x = 6, 2 m from tile (0, -1), looking along +z. Ahead (0, 1) fills most of
/// the view, ahead-right (1, 1) is nearer the axis than ahead-left (-1, 1); (0, 2) starts 18 m deep and (0, -2) lies
/// 12 m behind.
fn scene() -> (TileRegistry, Camera) {
    let r = registry_with(&[(0, 0), (0, 1), (1, 1), (-1, 1), (0, 2), (0, -2)]);
    (r, camera_at(Vec3::new(6.0, 1.5, 2.0)))
}

#[test]
fn the_active_set_is_the_near_tiles_then_the_largest_in_view() {
    let (r, cam) = scene();
    let c = candidates(&r, cam.position);
    // (0, 0) underfoot and (0, -1) 2 m away are near (rows or not) and do not count toward the two ranked tiles.
    assert_eq!(rank_active(&cam, &c, rule(15.0, 2), &[]), vec![key(0, 0), key(0, -1), key(0, 1), key(1, 1)]);
    let wide = rank_active(&cam, &c, rule(15.0, 10), &[]);
    assert!(wide.contains(&key(-1, 1)), "{wide:?}");
    assert!(!wide.contains(&key(0, 2)), "18 m deep is beyond the 15 m view: {wide:?}");
    assert!(!wide.contains(&key(0, -2)), "behind the camera: {wide:?}");
    assert!(rank_active(&cam, &c, rule(30.0, 10), &[]).contains(&key(0, 2)), "within a 30 m view");
    assert_eq!(rank_active(&cam, &c, rule(15.0, 0), &[]), vec![key(0, 0), key(0, -1)], "no ranked tile");
}

/// A room at the corner of four tiles: the camera 1 m from the corner has all four near and active, besides the
/// ranked ones.
#[test]
fn the_tiles_of_a_corner_are_all_active() {
    let r = registry_with(&[(0, 0), (-1, 0), (0, -1), (-1, -1), (0, 1)]);
    let cam = camera_at(Vec3::new(0.7, 1.5, 0.7));
    let active = rank_active(&cam, &candidates(&r, cam.position), rule(15.0, 1), &[]);
    assert_eq!(active[0], key(0, 0), "underfoot first");
    for k in [key(-1, 0), key(0, -1), key(-1, -1), key(0, 1)] {
        assert!(active.contains(&k), "{k:?} missing from {active:?}");
    }
    assert_eq!(active.len(), 5);
    assert_eq!(near_tiles(Vec3::new(5.0, 0.0, 5.0), 10.0), vec![key(0, 0)], "2.5 m from no border");
}

/// The near radius is 2.5 m, at most half a tile: with 1 m tiles only the tiles within 0.5 m are near, never a tile
/// beyond the ring around the camera's own.
#[test]
fn the_near_radius_is_at_most_half_a_tile() {
    assert_eq!((near_radius(10.0), near_radius(4.0), near_radius(1.0)), (2.5, 2.0, 0.5));
    let near = near_tiles(Vec3::new(0.3, 0.0, 0.3), 1.0);
    assert_eq!(near, vec![key(0, 0), key(-1, 0), key(0, -1), key(-1, -1)]);
}

/// Two tiles that cover about the same share of the image, the camera wobbling between favouring one and the other:
/// without hysteresis the ranked tile flips at every wobble, with it the first one stays.
#[test]
fn a_wobbling_camera_does_not_flip_the_active_set() {
    let r = registry_with(&[(-1, 1), (0, 1)]);
    let c = candidates(&r, Vec3::new(0.0, 1.5, -5.0));
    let wobble = |i: usize| {
        let x = if i % 2 == 0 { 0.4 } else { -0.4 };
        Camera { rotation: Quat::from_rotation_y(0.02 * x), ..camera_at(Vec3::new(x, 1.5, -5.0)) }
    };
    let ranked = |active: &[TileKey]| active.iter().copied().filter(|k| k.z == 1).collect::<Vec<_>>();
    let flipping: Vec<Vec<TileKey>> =
        (0..6).map(|i| ranked(&rank_active(&wobble(i), &c, rule(30.0, 1), &[]))).collect();
    assert!(flipping.windows(2).any(|w| w[0] != w[1]), "no flip even without hysteresis: {flipping:?}");
    let mut active = rank_active(&wobble(0), &c, rule(30.0, 1), &[]);
    let first = ranked(&active);
    for i in 1..12 {
        active = rank_active(&wobble(i), &c, rule(30.0, 1), &active);
        assert_eq!(ranked(&active), first, "flipped at wobble {i}");
    }
}

/// An active tile stays while its box is in view up to 2 m beyond the view depth; a tile covering 1.3 times its share
/// takes its place.
#[test]
fn an_active_tile_stays_until_out_of_reach_or_clearly_beaten() {
    // (0, 1) starts 8 m ahead of the camera at z = 2.
    let r = registry_with(&[(0, 1), (1, 1)]);
    let cam = camera_at(Vec3::new(5.0, 1.5, 2.0));
    let c = candidates(&r, cam.position);
    let inc = [key(1, 1)];
    // From 7 m deep (z = 3): (0, 1) at 7 m is out of a 6.5 m view, (1, 1) stays within 6.5 + 2 m.
    let back = Camera { position: Vec3::new(5.0, 1.5, 3.0), ..cam };
    assert_eq!(rank_active(&back, &c, rule(6.5, 1), &[]).last(), Some(&key(0, 0)), "nothing within 6.5 m fresh");
    assert_eq!(rank_active(&back, &c, rule(6.5, 1), &inc).last(), Some(&key(1, 1)), "kept within 8.5 m");
    // Straight on, (0, 1) covers far more than 1.3 times (1, 1)'s share: it wins although (1, 1) was active.
    assert_eq!(rank_active(&cam, &c, rule(15.0, 1), &inc).last(), Some(&key(0, 1)));
}

#[test]
fn tiles_with_too_few_rows_are_no_candidates_unless_near() {
    let r = registry_with(&[(0, 0), (0, 1), (1, 1)]);
    let cam = camera_at(Vec3::new(6.0, 1.5, 4.0));
    let c = candidates(&r, cam.position);
    let strict = ActiveRule { min_rows: 3, ..rule(15.0, 3) };
    assert_eq!(rank_active(&cam, &c, strict, &[]), vec![key(0, 0)], "two rows each: only the near tile");
    assert_eq!(rank_active(&cam, &c, rule(15.0, 3), &[]).len(), 3);
}

#[test]
fn parked_and_left_tiles_are_no_candidates_and_heights_set_the_boxes() {
    let mut r = registry_with(&[(0, 0), (0, 1), (5, 0)]);
    r.mark_parked(key(0, 1), 2, 0);
    // The walker stood at tile (5, 0) and is now 40 m from it: left.
    r.note_walker(Vec3::new(55.0, 0.0, 5.0));
    let centre = Vec3::new(6.0, 1.5, 2.0);
    let heights = BTreeMap::from([
        (key(1, 1), Heights { lo: 0.0, hi: 2.0, rows: 7 }),
        (key(0, 1), Heights { lo: 0.0, hi: 2.0, rows: 7 }),
    ]);
    let c = candidate_boxes(&r, &heights, centre);
    assert_eq!(c.keys().copied().collect::<Vec<_>>(), vec![key(0, 0), key(1, 1)]);
    // Without a band the registry's range and count; with one, the band.
    assert_eq!(c[&key(0, 0)], Candidate { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(10.0, 3.0, 10.0), rows: 2 });
    assert_eq!(c[&key(1, 1)], Candidate { min: Vec3::new(10.0, 0.0, 10.0), max: Vec3::new(20.0, 2.0, 20.0), rows: 7 });
}

/// The band is the 5th to 95th percentile: a stray row 1 km up does not stretch it; bands widen together.
#[test]
fn heights_are_the_5_to_95_percent_band() {
    let mut means: Vec<f32> = (0..99).flat_map(|i| [5.0, i as f32, 5.0]).collect();
    means.extend([5.0, 1000.0, 5.0, 15.0, 4.0, 5.0]);
    let h = tile_heights(&means, 10.0);
    assert_eq!(h[&key(0, 0)], Heights { lo: 5.0, hi: 94.0, rows: 100 });
    assert_eq!(h[&key(1, 0)], Heights { lo: 4.0, hi: 4.0, rows: 1 });
    let mut all = h.clone();
    widen_all(&mut all, &BTreeMap::from([(key(1, 0), Heights { lo: -1.0, hi: 2.0, rows: 3 })]));
    assert_eq!(all[&key(1, 0)], Heights { lo: -1.0, hi: 4.0, rows: 4 });
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
    let fresh = BTreeMap::from([(key(2, 2), Heights { lo: 0.0, hi: 1.0, rows: 1 })]);
    let active = [key(0, 0), key(0, 1)];
    let seeds = [
        [5.0, 0.0, 5.0],   // active
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

/// Every tile holding resident rows has a box, left ones included: a tile's band where it has one, else the
/// registry's height range; parked tiles have none.
#[test]
fn resident_boxes_cover_left_and_banded_tiles_but_not_parked_ones() {
    let mut r = registry_with(&[(0, 0), (0, 1), (5, 0)]);
    r.mark_parked(key(0, 1), 2, 0);
    r.note_walker(Vec3::new(55.0, 0.0, 5.0));
    let heights = BTreeMap::from([
        (key(0, 0), Heights { lo: -1.0, hi: 1.0, rows: 4 }),
        (key(2, 2), Heights { lo: 0.5, hi: 2.0, rows: 3 }),
    ]);
    let boxes = resident_boxes(&r, &heights);
    assert_eq!(
        boxes,
        vec![
            (Vec3::new(0.0, -1.0, 0.0), Vec3::new(10.0, 1.0, 10.0)),
            (Vec3::new(20.0, 0.5, 20.0), Vec3::new(30.0, 2.0, 30.0)),
            (Vec3::new(50.0, 0.0, 0.0), Vec3::new(60.0, 3.0, 10.0)),
        ]
    );
    assert!(!candidate_boxes(&r, &heights, Vec3::new(6.0, 1.5, 2.0)).contains_key(&key(5, 0)), "(5, 0) is left");
}
