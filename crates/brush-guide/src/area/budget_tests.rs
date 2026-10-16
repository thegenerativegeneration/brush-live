use super::*;
use crate::area::AreaParams;

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

fn key(x: i32, z: i32) -> TileKey {
    TileKey { x, z }
}

/// A point in the middle of tile (x, z) of 10 m tiles.
fn mid(x: i32, z: i32) -> [f32; 3] {
    [x as f32 * 10.0 + 5.0, 0.0, z as f32 * 10.0 + 5.0]
}

/// A registry with `rows` rows in the middle of each tile.
fn registry_with(tiles: &[((i32, i32), usize)]) -> TileRegistry {
    let mut r = TileRegistry::new(params());
    let means: Vec<f32> = tiles.iter().flat_map(|&((x, z), n)| mid(x, z).repeat(n)).collect();
    r.note_resident(&means, &[], 0);
    r
}

#[test]
fn counts_are_the_registrys_rows_plus_the_rows_seeded_since() {
    let r = registry_with(&[((0, 0), 3), ((1, 0), 2)]);
    let fresh = BTreeMap::from([
        (key(1, 0), Heights { lo: 0.0, hi: 1.0, rows: 4 }),
        (key(5, 5), Heights { lo: 0.0, hi: 1.0, rows: 1 }),
    ]);
    assert_eq!(tile_counts(&r, &fresh), BTreeMap::from([(key(0, 0), 3), (key(1, 0), 6), (key(5, 5), 1)]));
}

/// A tile at its cap takes no seed, one with room for fewer takes that many evenly spread, others take all; seeds
/// already dropped neither count nor come back. A cap of 0 drops nothing.
#[test]
fn seeds_into_a_full_tile_are_dropped() {
    let counts = BTreeMap::from([(key(0, 0), 10), (key(1, 0), 7)]);
    let seeds: Vec<f32> = [mid(0, 0).repeat(3), mid(1, 0).repeat(6), mid(2, 0).repeat(2), mid(1, 0).to_vec()].concat();
    let mut keep = vec![true; 12];
    keep[11] = false;
    let cleared = keep_under_tile_cap(&counts, 10, &seeds, 10.0, &mut keep);
    assert_eq!(cleared, 6);
    let want = [[false; 3].as_slice(), &[false, true, false, true, false, true], &[true, true], &[false]].concat();
    assert_eq!(keep, want);
    let mut keep = vec![true; 12];
    assert_eq!(keep_under_tile_cap(&counts, 0, &seeds, 10.0, &mut keep), 0);
    assert!(keep.iter().all(|&k| k));
}

/// After the left tiles, the tiles that are neither active nor near, lie beyond radius + max(hysteresis, far seed)
/// (20 m) and are not seen park, farthest first, until they hold the need; a nearer tile, a seen one, an active one
/// and a recently restored one never do, and neither does the last of the resident rows.
#[test]
fn far_inactive_tiles_park_farthest_first() {
    let mut r =
        registry_with(&[((0, 0), 5), ((1, 0), 5), ((2, 0), 4), ((3, 0), 4), ((4, 0), 3), ((5, 0), 2), ((-4, 0), 6)]);
    let centre = Vec3::new(5.0, 0.0, 5.0);
    let unseen = |_: TileKey| false;
    // (5, 0) is active; (1, 0) and (2, 0) lie within 20 m (5 and 15 m).
    let keep = [key(0, 0), key(5, 0)];
    assert_eq!(far_inactive_to_park(&r, 5, 25, centre, 30, &keep, unseen), vec![key(-4, 0)]);
    assert_eq!(far_inactive_to_park(&r, 7, 25, centre, 30, &keep, unseen), vec![key(-4, 0), key(4, 0)]);
    let all = far_inactive_to_park(&r, 100, 25, centre, 30, &keep, unseen);
    assert_eq!(all, vec![key(-4, 0), key(4, 0), key(3, 0)]);
    let seen = |k: TileKey| k == key(-4, 0);
    assert_eq!(far_inactive_to_park(&r, 5, 25, centre, 30, &keep, seen), vec![key(4, 0), key(3, 0)], "seen");
    r.mark_parked(key(-4, 0), 6, 0);
    r.mark_restored(key(-4, 0), 25, 0);
    let fresh = far_inactive_to_park(&r, 5, 25, centre, 30, &keep, unseen);
    assert_eq!(fresh, vec![key(4, 0), key(3, 0)], "restored 5 kf ago");
    // Of 7 resident rows, (-4, 0) takes 6: adding (4, 0) or (3, 0) would take them all.
    assert_eq!(far_inactive_to_park(&r, 100, 7, centre, 50, &keep, unseen), vec![key(-4, 0)]);
}

/// One group per tile: a tile at the cap is full and keeps the cap; with a fair share an active tile gives its rows
/// above `max_splats` / active tiles and the others none. Without a cap and a fair share there is no budget.
#[test]
fn the_row_budget_marks_full_tiles_and_fair_shares() {
    let means: Vec<f32> = [mid(0, 0).repeat(5), mid(1, 0).repeat(3), mid(2, 0).to_vec(), mid(0, 0).to_vec()].concat();
    let active = [key(0, 0), key(1, 0)];
    let (keys, b) = row_budget(&means, 10.0, 4, Some(FairShare { active: &active, max_splats: 4 })).unwrap();
    assert_eq!(keys, vec![key(0, 0), key(1, 0), key(2, 0)]);
    assert_eq!(b.group, vec![0, 0, 0, 0, 0, 1, 1, 1, 2, 0]);
    let g = |max_rows, full, evictable| GroupBudget { max_rows, full, evictable };
    assert_eq!(b.groups, vec![g(4, true, 4), g(4, false, 1), g(4, false, 0)]);
    assert!(b.limit_eviction);
    let (_, b) = row_budget(&means, 10.0, 0, Some(FairShare { active: &active, max_splats: 4 })).unwrap();
    assert_eq!(b.groups, vec![g(u32::MAX, false, 4), g(u32::MAX, false, 1), g(u32::MAX, false, 0)]);
    let (_, b) = row_budget(&means, 10.0, 4, None).unwrap();
    assert!(!b.limit_eviction && b.groups.iter().all(|g| g.evictable == 0));
    assert!(row_budget(&means, 10.0, 0, None).is_none());
}
