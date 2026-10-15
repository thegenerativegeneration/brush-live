use super::*;

fn params() -> AreaParams {
    AreaParams { tile_size: 10.0, radius: 25.0, hysteresis: 5.0, seen_window: 1000, min_resident_keyframes: 20, far_seed: 10.0 }
}

fn flat(points: &[[f32; 3]]) -> Vec<f32> {
    points.iter().flatten().copied().collect()
}

#[test]
fn keys_floor_negative_coordinates_and_borders() {
    let k = |x, z| TileKey::of(Vec3::new(x, 3.0, z), 10.0);
    assert_eq!(k(-0.01, 9.99), TileKey { x: -1, z: 0 });
    assert_eq!(k(0.0, -10.0), TileKey { x: 0, z: -1 });
    assert_eq!(k(10.0, -10.001), TileKey { x: 1, z: -2 });
    assert_eq!(k(-0.0, 0.0), TileKey { x: 0, z: 0 });
    assert_eq!(TileKey { x: -1, z: 3 }.stem(), "-1_3");
}

#[test]
fn footprint_distance_is_zero_inside_and_horizontal_outside() {
    let t = TileKey { x: 0, z: 0 };
    assert_eq!(t.footprint_distance(Vec3::new(5.0, 100.0, 5.0), 10.0), 0.0);
    assert!((t.footprint_distance(Vec3::new(15.0, 0.0, 5.0), 10.0) - 5.0).abs() < 1e-6);
    assert!((t.footprint_distance(Vec3::new(13.0, -7.0, 14.0), 10.0) - 5.0).abs() < 1e-6);
    assert!((TileKey { x: -2, z: 0 }.footprint_distance(Vec3::new(0.0, 0.0, 5.0), 10.0) - 10.0).abs() < 1e-6);
}

#[test]
fn note_resident_counts_rows_tracks_height_and_seen() {
    let mut r = TileRegistry::new(params());
    let means = flat(&[[1.0, 0.5, 1.0], [2.0, 4.0, 3.0], [-3.0, 1.0, 1.0]]);
    assert!(r.note_resident(&means, &[false, false, false], 100).is_empty());
    let t = r.get(TileKey { x: 0, z: 0 }).unwrap();
    assert_eq!((t.splats, t.y_min, t.y_max, t.last_seen_step), (2, 0.5, 4.0, 100));
    r.note_resident(&means, &[false, true, false], 300);
    assert_eq!(r.get(TileKey { x: 0, z: 0 }).unwrap().last_seen_step, 300);
    assert_eq!(r.get(TileKey { x: -1, z: 0 }).unwrap().last_seen_step, 100);
    // A resident tile without rows disappears.
    r.note_resident(&flat(&[[1.0, 0.5, 1.0]]), &[], 400);
    assert!(r.get(TileKey { x: -1, z: 0 }).is_none());
}

#[test]
fn park_needs_distance_beyond_hysteresis_and_an_unseen_window() {
    let mut r = TileRegistry::new(params());
    // Tile (0, 0); centres along x: footprint distance = centre.x − 10.
    r.note_resident(&flat(&[[5.0, 0.0, 5.0]]), &[], 0);
    assert!(r.to_park(Vec3::new(38.0, 0.0, 5.0), 2000, 50).is_empty(), "28 m: inside radius + hysteresis");
    assert_eq!(r.to_park(Vec3::new(41.0, 0.0, 5.0), 2000, 50), vec![TileKey { x: 0, z: 0 }]);
    assert!(r.to_park(Vec3::new(41.0, 0.0, 5.0), 1000, 50).is_empty(), "seen 1000 steps ago: still in window");
}

#[test]
fn a_restored_tile_stays_for_min_resident_keyframes() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0]]), &[], 0);
    let key = TileKey { x: 0, z: 0 };
    r.mark_parked(key, 1, 0);
    r.mark_restored(key, 100, 0);
    let far = Vec3::new(60.0, 0.0, 5.0);
    assert!(r.to_park(far, 5000, 119).is_empty());
    assert_eq!(r.to_park(far, 5000, 120), vec![key]);
}

#[test]
fn parked_tiles_come_back_by_radius() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [55.0, 0.0, 5.0]]), &[], 0);
    let near = TileKey { x: 0, z: 0 };
    r.mark_parked(near, 1, 0);
    assert_eq!(r.parked_within_radius(Vec3::new(30.0, 0.0, 5.0)), vec![near]);
    assert!(r.parked_within_radius(Vec3::new(36.0, 0.0, 5.0)).is_empty());
}

/// User decision D2 and Review Focus 2: far seeds and sparse seeds in parked tiles are dropped, dense seeds in a
/// parked tile within radius + 10 m restore it; seeds elsewhere are kept.
#[test]
fn seed_triage_drops_far_and_sparse_seeds_and_restores_dense_near_ones() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [55.0, 0.0, 5.0]]), &[], 0);
    let parked = TileKey { x: 0, z: 0 };
    r.mark_parked(parked, 1, 0);
    let seeds = |n: usize, x: f32| flat(&vec![[x, 1.0, 5.0]; n]);
    // Tile (0, 0) is 30 m from (40, 0, 5): within 25 + 10 m.
    let near = Vec3::new(40.0, 0.0, 5.0);
    let t = r.triage_seeds(&seeds(8, 3.0), near, 8);
    assert_eq!((t.restore, t.keep), (vec![parked], vec![false; 8]));
    let t = r.triage_seeds(&seeds(7, 3.0), near, 8);
    assert_eq!((t.restore.len(), t.keep), (0, vec![false; 7]), "too few: dropped, not appended");
    // 40 m away: beyond 25 + 10 m, so even many seeds stay out and the tile stays parked.
    let t = r.triage_seeds(&seeds(20, 3.0), Vec3::new(50.0, 0.0, 5.0), 8);
    assert_eq!((t.restore.len(), t.keep), (0, vec![false; 20]));
    let mut mixed = seeds(2, 55.0);
    mixed.extend(seeds(1, 200.0));
    assert_eq!(r.triage_seeds(&mixed, near, 8).keep, vec![true; 3], "resident tiles and new ground keep seeds");
}

/// Review Focus 4: room for a restore comes from unseen tiles beyond the radius, farthest first; tiles within
/// the radius or seen recently are never chosen.
#[test]
fn room_plan_parks_unseen_far_tiles_or_defers() {
    let mut r = TileRegistry::new(params());
    let mut pts = vec![[5.0, 0.0, 5.0]; 400]; // (0, 0): within the radius of (5, 0, 5)
    pts.extend(vec![[55.0, 0.0, 5.0]; 100]); // (5, 0): 45 m, unseen
    pts.extend(vec![[65.0, 0.0, 5.0]; 200]); // (6, 0): 55 m, seen at step 4900
    pts.extend(vec![[85.0, 0.0, 5.0]; 150]); // (8, 0): 75 m, unseen
    let seen: Vec<bool> = pts.iter().map(|p| p[0] == 65.0).collect();
    r.note_resident(&flat(&pts), &vec![false; pts.len()], 0);
    r.note_resident(&flat(&pts), &seen, 4900);
    let c = Vec3::new(5.0, 0.0, 5.0);
    assert_eq!(r.plan_room(100, 850, 1000, c, 5000, 100), RoomPlan { park_first: vec![], restore: true, shortfall: 0 });
    // Room 150: the farthest unseen tile (150) is enough for 300.
    assert_eq!(
        r.plan_room(300, 850, 1000, c, 5000, 100),
        RoomPlan { park_first: vec![TileKey { x: 8, z: 0 }], restore: true, shortfall: 0 }
    );
    assert_eq!(
        r.plan_room(350, 850, 1000, c, 5000, 100),
        RoomPlan { park_first: vec![TileKey { x: 8, z: 0 }, TileKey { x: 5, z: 0 }], restore: true, shortfall: 0 }
    );
    assert_eq!(
        r.plan_room(600, 850, 1000, c, 5000, 100),
        RoomPlan { park_first: vec![TileKey { x: 8, z: 0 }, TileKey { x: 5, z: 0 }], restore: false, shortfall: 200 }
    );
}

#[test]
fn tile_states_follow_distance_and_parking() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [45.0, 0.0, 5.0], [95.0, 0.0, 5.0]]), &[], 0);
    r.mark_parked(TileKey { x: 9, z: 0 }, 1, 0);
    assert_eq!(r.tile_states(Vec3::new(5.0, 0.0, 5.0)), vec![[0, 0, 0], [4, 0, 1], [9, 0, 2]]);
}

/// Review Focus 2: resident rows in a parked tile's column are reported, the parked state is kept.
#[test]
fn rows_in_a_parked_tile_are_reported_not_adopted() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [15.0, 0.0, 5.0]]), &[], 0);
    let key = TileKey { x: 0, z: 0 };
    r.mark_parked(key, 1, 0);
    let conflicts = r.note_resident(&flat(&[[4.0, 0.0, 4.0], [15.0, 0.0, 5.0]]), &[true, true], 100);
    assert_eq!(conflicts, vec![key]);
    let t = r.get(key).unwrap();
    assert!(!t.resident && t.splats == 1 && t.version == 1);
}

#[test]
fn rows_group_by_tile() {
    let means = flat(&[[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]);
    let g = group_rows(&means, 10.0);
    assert_eq!(g[&TileKey { x: 0, z: 0 }], vec![0, 2]);
    assert_eq!(rows_in(&[TileKey { x: -1, z: 0 }], &means, 10.0), vec![false, true, false]);
}

#[test]
fn summary_splits_resident_tiles_by_distance_and_sight() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [45.0, 0.0, 5.0], [45.0, 0.0, 6.0], [95.0, 0.0, 5.0]]), &[], 0);
    r.mark_parked(TileKey { x: 9, z: 0 }, 1, 0);
    let s = r.summary(Vec3::new(5.0, 0.0, 5.0));
    assert_eq!(s, AreaSummary { resident_tiles: 2, sight_tiles: 1, parked_tiles: 1, resident_splats: 3,
        sight_splats: 2, parked_splats: 1 });
}

#[test]
fn area_is_on_by_radius() {
    let mut c = GuideConfig::default();
    assert!(AreaParams::from_config(&c).is_some_and(|p| p.radius == 50.0 && p.tile_size == 10.0));
    c.area_radius_m = 0.0;
    assert!(AreaParams::from_config(&c).is_none());
}

#[test]
fn plan_room_skips_recently_restored_tiles() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [85.0, 0.0, 5.0]]), &[], 0);
    let far = TileKey { x: 8, z: 0 };
    r.mark_parked(far, 1, 0);
    r.mark_restored(far, 100, 0);
    let c = Vec3::new(5.0, 0.0, 5.0);
    assert_eq!(r.plan_room(2000, 2, 1000, c, 5000, 119).park_first, vec![]);
    assert_eq!(r.plan_room(2000, 2, 1000, c, 5000, 120).park_first, vec![far]);
}

#[test]
fn plan_room_resolves_equal_distance_by_key() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[5.0, 0.0, 5.0], [75.0, 0.0, 5.0], [75.0, 0.0, -15.0], [75.0, 0.0, 25.0]]), &[], 0);
    let c = Vec3::new(5.0, 0.0, 5.0);
    let plan = r.plan_room(10, 4, 4, c, 5000, 100);
    // (7, -2) and (7, 2) are farther than (7, 0); of those two the lower key comes first.
    assert_eq!(plan.park_first, vec![TileKey { x: 7, z: -2 }, TileKey { x: 7, z: 2 }, TileKey { x: 7, z: 0 }]);
}

#[test]
fn non_finite_points_map_to_defined_tiles() {
    let k = |x, z| TileKey::of(Vec3::new(x, 0.0, z), 10.0);
    assert_eq!(k(f32::NAN, f32::NAN), TileKey { x: 0, z: 0 });
    assert_eq!(k(f32::INFINITY, f32::NEG_INFINITY), TileKey { x: i32::MAX, z: i32::MIN });
}

#[test]
fn z_exactly_on_a_tile_edge_belongs_to_the_upper_tile() {
    assert_eq!(TileKey::of(Vec3::new(0.0, 0.0, 10.0), 10.0), TileKey { x: 0, z: 1 });
    assert_eq!(TileKey::of(Vec3::new(0.0, 0.0, 9.999), 10.0), TileKey { x: 0, z: 0 });
}

/// Rows merged into a parked tile's file raise its count and version; it stays parked from the same step.
#[test]
fn a_merge_keeps_a_tile_parked_with_more_rows() {
    let mut r = TileRegistry::new(params());
    let key = TileKey { x: 0, z: 0 };
    r.note_resident(&flat(&[[1.0, 0.0, 1.0]]), &[], 0);
    r.mark_parked(key, 10, 7);
    r.note_merged(key, 13);
    let t = r.get(key).unwrap();
    assert_eq!((t.resident, t.splats, t.version, t.parked_at_step), (false, 13, 2, 7));
}

/// Task 10b: the walker visits every tile whose footprint lies within the radius, rows or not; a visited tile beyond
/// radius + hysteresis is left, a tile never visited is not.
#[test]
fn walked_tiles_are_left_beyond_radius_plus_hysteresis() {
    let mut r = TileRegistry::new(params());
    r.note_walker(Vec3::new(5.0, 0.0, 5.0));
    // (-3, 0) spans x in [-30, -20]: exactly 25 m from x = 5, so visited; (-4, 0) and (4, 0) at 35 m are not.
    for (key, visited) in [((0, 0), true), ((-3, 0), true), ((3, 0), true), ((-4, 0), false), ((4, 0), false)] {
        let key = TileKey { x: key.0, z: key.1 };
        assert_eq!(r.left_keys(Vec3::new(5.0, 0.0, 5000.0)).contains(&key), visited, "{key:?}");
    }
    let far = Vec3::new(5.0 + 500.0, 0.0, 5.0);
    assert!(r.left_keys(Vec3::new(5.0, 0.0, 5.0)).is_empty(), "the walker is still there");
    // From x = 55: (0, 0) lies 45 m away (left), (3, 0) 15 m (not beyond 30 m), (2, 0) 25 m.
    let left = r.left_keys(Vec3::new(55.0, 0.0, 5.0));
    assert!(left.contains(&TileKey { x: 0, z: 0 }) && !left.contains(&TileKey { x: 3, z: 0 }));
    assert!(!left.contains(&TileKey { x: 2, z: 0 }), "25 m lies within radius + hysteresis");
    assert_eq!(r.left_keys(far).len(), r.left_keys(Vec3::new(5.0, 0.0, 5000.0)).len());
    r.note_walker(Vec3::new(f32::NAN, 0.0, 0.0));
    assert_eq!(r.left_keys(far).len(), r.left_keys(Vec3::new(5.0, 0.0, 5000.0)).len(), "a NaN walker visits nothing");
}

/// Review of Task 10: room for a restore may also come from left tiles that views still see, after the unseen far
/// tiles and farthest first; a seen tile the walker never visited is never parked for room.
#[test]
fn room_plan_parks_seen_left_tiles_after_unseen_ones() {
    let mut r = TileRegistry::new(params());
    r.note_walker(Vec3::new(5.0, 0.0, 5.0));
    r.note_walker(Vec3::new(5.0, 0.0, 45.0));
    let mut pts = vec![[5.0, 0.0, 5.0]; 100]; // (0, 0): visited, seen
    pts.extend(vec![[5.0, 0.0, 45.0]; 40]); // (0, 4): visited, seen
    pts.extend(vec![[125.0, 0.0, 5.0]; 200]); // (12, 0): the walker's tile
    pts.extend(vec![[205.0, 0.0, 5.0]; 70]); // (20, 0): never visited, seen
    pts.extend(vec![[255.0, 0.0, 5.0]; 30]); // (25, 0): never visited, unseen
    let seen: Vec<bool> = pts.iter().map(|p| p[0] != 255.0).collect();
    r.note_resident(&flat(&pts), &vec![false; pts.len()], 0);
    r.note_resident(&flat(&pts), &seen, 4900);
    let c = Vec3::new(125.0, 0.0, 5.0);
    let (unseen, near_left, far_left) = (TileKey { x: 25, z: 0 }, TileKey { x: 0, z: 0 }, TileKey { x: 0, z: 4 });
    assert_eq!(r.plan_room(20, 440, 440, c, 5000, 100).park_first, vec![unseen]);
    // (0, 4) lies farther from the walker than (0, 0).
    assert_eq!(
        r.plan_room(100, 440, 440, c, 5000, 100),
        RoomPlan { park_first: vec![unseen, far_left, near_left], restore: true, shortfall: 0 }
    );
    assert_eq!(
        r.plan_room(240, 440, 440, c, 5000, 100),
        RoomPlan { park_first: vec![unseen, far_left, near_left], restore: false, shortfall: 70 }
    );
}

/// Task 10b: a plan never parks every resident splat (`park_tiles` would skip that park); a smaller tile still goes.
#[test]
fn room_plan_never_parks_every_resident_splat() {
    let mut r = TileRegistry::new(params());
    r.note_resident(&flat(&[[85.0, 0.0, 5.0], [85.0, 0.0, 6.0], [95.0, 0.0, 5.0]]), &[], 0);
    let c = Vec3::new(5.0, 0.0, 5.0);
    // (9, 0) holds 1 splat and (8, 0) 2, all 3 resident: parking both would take every one.
    let plan = RoomPlan { park_first: vec![TileKey { x: 9, z: 0 }], restore: false, shortfall: 9 };
    assert_eq!(r.plan_room(10, 3, 3, c, 5000, 100), plan);
}
