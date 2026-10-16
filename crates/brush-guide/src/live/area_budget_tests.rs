//! Task 17c: the per-tile cap (seeds into a full tile dropped, a full tile does not grow, a tile over the cap shrinks
//! to it), room at the total cap (left tiles, then far tiles neither active nor near, park before an eviction; the
//! eviction keeps each active tile's fair share), and `area_freeze` inactive_at_cap.

use super::*;
use crate::area::{TileKey, rows_in};
use crate::config::{AreaFreeze, GuideConfig};
use crate::live::area::tests::{active_config, add, area_config, device, tmp};
use crate::live::area_freeze::Held;
use crate::live::area_freeze::tests::{by_row, rows_where};
use crate::live::area_restore::tests::train;
use glam::Vec3;
use std::collections::{BTreeMap, BTreeSet};

/// The resident rows per tile of `tile_size`.
async fn counts(live: &LiveModel, tile_size: f32) -> BTreeMap<TileKey, usize> {
    let rows = rows_where(live, |_| true).await;
    let mut out = BTreeMap::new();
    for i in 0..rows.len() {
        *out.entry(TileKey::of(rows.mean(i), tile_size)).or_default() += 1;
    }
    out
}

/// The `area_tile_max_fraction` that makes the per-tile cap `cap` rows at `live`'s `max_splats`.
fn cap_at(live: &mut LiveModel, cap: u32) {
    live.config.area_tile_max_fraction = cap as f32 / live.config.max_splats as f32;
    assert_eq!(live.config.area_tile_cap(), cap);
}

/// Every key point projects into one keyframe's view and seeds (alpha below 2 everywhere).
fn seeding_everywhere(config: GuideConfig) -> GuideConfig {
    GuideConfig { seed_alpha_threshold: 2.0, ..config }
}

/// Keyframe 0 at the origin, a refine, then a cap of 1 row per tile: keyframe 1 seeds onto the same ground and no
/// seed lands in a tile that held rows (all full), the drops counted; without a per-tile cap they land, and with
/// area training off the cap does nothing.
#[tokio::test]
async fn seeds_into_a_full_tile_are_dropped() {
    for (cap, radius) in [(1, 2.0), (0, 2.0), (1, 0.0)] {
        let dir = tmp(&format!("budget-seeds-{cap}-{radius}"));
        let mut live = LiveModel::new(seeding_everywhere(area_config(radius)), device().await);
        live.enable_area(dir.join("tiles"));
        add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 5).await;
        cap_at(&mut live, cap);
        let before = counts(&live, 1.0).await;
        let n = live.splats().unwrap().num_splats();
        add(&mut live, &dir, 1, Vec3::new(0.1, 0.0, 2.0), Vec3::new(0.1, 0.0, 0.0)).await;
        let after = counts(&live, 1.0).await;
        let grew = |k: &TileKey| after.get(k).copied().unwrap_or(0) > before[k];
        let (into_held, added) = (before.keys().filter(|k| grew(k)).count(), live.splats().unwrap().num_splats() - n);
        let dropped = live.area_report().map_or(0, |r| r.counters.seeds_over_tile_cap);
        if cap == 1 && radius > 0.0 {
            assert_eq!(into_held, 0, "a full tile took seeds: {before:?} -> {after:?}");
            assert!(dropped > 0);
        } else {
            assert!(into_held > 0 && added > 0, "seeds did not land ({cap}, {radius}): {before:?} -> {after:?}");
            assert_eq!(dropped, 0);
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// 10 m tiles, keyframe 0 looking at the middle of tile (0, 0), so all of its rows lie in that tile; every row a
/// growth candidate (gradient threshold 0, all selected) or none (`grow` false: no growth, no force-split).
async fn one_tile(name: &str, grow: bool) -> (LiveModel, std::path::PathBuf) {
    let dir = tmp(name);
    let config = GuideConfig { area_tile_size_m: 10.0, area_radius_m: 20.0, ..area_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    if grow {
        live.train_config.growth_grad_threshold = 0.0;
        live.train_config.growth_select_fraction = 1.0;
    } else {
        live.train_config.growth_grad_threshold = f32::MAX;
        live.train_config.split_at_screen_size = 0.0;
    }
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(5.0, 0.0, 7.0), Vec3::new(5.0, 0.0, 5.0)).await;
    train(&mut live, 5).await;
    (live, dir)
}

/// A tile at its cap does not grow at the next refine, where without a cap it does.
#[tokio::test]
async fn a_full_tile_does_not_grow() {
    for cap_it in [false, true] {
        let (mut live, dir) = one_tile(&format!("budget-full-{cap_it}"), true).await;
        let tile = TileKey { x: 0, z: 0 };
        let n = counts(&live, 10.0).await[&tile];
        assert_eq!(counts(&live, 10.0).await.len(), 1);
        cap_at(&mut live, if cap_it { n as u32 } else { 0 });
        train(&mut live, 5).await;
        let after = counts(&live, 10.0).await[&tile];
        if cap_it {
            assert!(after <= n, "a full tile grew: {n} -> {after}");
        } else {
            assert!(after > n, "without a cap the tile did not grow: {n} -> {after}");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A tile over its cap evicts its least important rows down to the cap at the next refine (growth off: nothing
/// else changes the count); the evictions are reported per tile.
#[tokio::test]
async fn an_over_cap_tile_shrinks_to_its_cap() {
    let (mut live, dir) = one_tile("budget-over", false).await;
    let tile = TileKey { x: 0, z: 0 };
    let n = counts(&live, 10.0).await[&tile];
    assert!(n > 10, "{n} rows");
    let cap = n as u32 - 5;
    cap_at(&mut live, cap);
    train(&mut live, 5).await;
    let after = counts(&live, 10.0).await[&tile];
    assert!(after <= cap as usize, "{n} -> {after} over the cap {cap}");
    let r = live.area_report().unwrap();
    assert!(r.tile_evictions.len() == 1 && r.tile_evictions[0][..2] == [0, 0], "{:?}", r.tile_evictions);
    let evicted = r.tile_evictions[0][2] as usize;
    assert!((1..=5).contains(&evicted) && evicted <= n - after, "{evicted} evicted, {n} -> {after}");
    assert!(r.tile_splats.iter().any(|t| t[..2] == [0, 0]), "{:?}", r.tile_splats);
    std::fs::remove_dir_all(dir).ok();
}

/// Rows outside the active tiles (a 1 m tile wall seen from 5 m, two tiles active).
async fn wall(name: &str, freeze: AreaFreeze) -> (LiveModel, std::path::PathBuf) {
    let dir = tmp(name);
    let mut live = LiveModel::new(GuideConfig { area_freeze: freeze, ..active_config(2.0) }, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 5.0), Vec3::new(-0.3, 0.0, 0.0)).await;
    (live, dir)
}

/// With inactive_at_cap the rows outside the active tiles train while the resident rows stay below 0.9 of
/// `max_splats`; from 0.9 on they are held, still held between 0.85 and 0.9, and train again below 0.85.
#[tokio::test]
async fn inactive_at_cap_holds_inactive_rows_at_the_cap_only() {
    let (mut live, dir) = wall("budget-at-cap", AreaFreeze::InactiveAtCap).await;
    let active: BTreeSet<TileKey> = live.area.as_ref().unwrap().active.iter().copied().collect();
    let inactive = |p: Vec3| !active.contains(&TileKey::of(p, 1.0));
    let before = by_row(&rows_where(&live, inactive).await);
    assert!(!before.is_empty());
    assert_eq!(live.area_report().unwrap().frozen_rows, 0);
    train(&mut live, 5).await;
    let trained = by_row(&rows_where(&live, inactive).await);
    assert_ne!(before, trained, "below the cap the inactive rows did not train");
    let n = live.splats().unwrap().num_splats() as f32;
    let step_at = async |live: &mut LiveModel, share: f32| {
        live.config.max_splats = (n / share).floor() as u32;
        let before = by_row(&rows_where(live, inactive).await);
        train(live, 1).await;
        // Held: every inactive row is still there bit for bit (an active row may have moved into an inactive tile).
        let after = by_row(&rows_where(live, inactive).await);
        let held = before.iter().all(|row| after.contains(row));
        let r = live.area_report().unwrap();
        assert_eq!(r.at_cap, held, "{share}: {r:?}");
        assert_eq!(r.frozen_rows > 0, held, "{share}: {r:?}");
        held
    };
    assert!(!step_at(&mut live, 0.88).await, "held at 0.88, below the 0.9 hold");
    assert!(step_at(&mut live, 0.9).await, "not held at 0.9");
    assert!(step_at(&mut live, 0.86).await, "released at 0.86, above the 0.85 release");
    assert!(!step_at(&mut live, 0.84).await, "still held at 0.84, below the 0.85 release");
    std::fs::remove_dir_all(dir).ok();
}

/// Seeds in a tile with rows outside the active set: inactive drops them, inactive_at_cap below the cap keeps them
/// (those rows train there).
#[tokio::test]
async fn below_the_cap_inactive_at_cap_seeds_where_inactive_does_not() {
    for freeze in [AreaFreeze::Inactive, AreaFreeze::InactiveAtCap] {
        let (mut live, dir) = wall(&format!("budget-seed-mode-{freeze:?}"), freeze).await;
        let rows = counts(&live, 1.0).await;
        let active = live.area.as_ref().unwrap().active.clone();
        let tile = *rows.keys().find(|k| !active.contains(k)).expect("a tile with rows outside the active set");
        let seed = [tile.x as f32 + 0.5, 0.0, tile.z as f32 + 0.5];
        let mut keep = vec![true];
        live.keep_active(&seed, &mut keep);
        assert_eq!(keep[0], freeze == AreaFreeze::InactiveAtCap, "{freeze:?}");
        std::fs::remove_dir_all(dir).ok();
    }
}

/// The walk: keyframe 0 at the origin (its tiles left later), keyframe 1 at x = 8 looking 16 m ahead (rows at
/// z = -14: never visited, beyond radius + far seed, not in view within 8 m, so neither left nor active), keyframe 2
/// at x = 8 looking 4 m along +x (active; the newest camera does not see z = -14). Returns the left, far and active
/// tiles holding rows.
async fn walked(name: &str) -> (LiveModel, std::path::PathBuf, [BTreeSet<TileKey>; 3]) {
    let dir = tmp(name);
    // A seen window of 1000 steps: no tile parks after the refine for being unseen.
    let config = GuideConfig { area_seen_window_steps: 1000, ..active_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(8.0, 0.0, 2.0), Vec3::new(8.0, 0.0, -14.0)).await;
    add(&mut live, &dir, 2, Vec3::new(8.0, 0.0, 2.0), Vec3::new(12.0, 0.0, 2.0)).await;
    train(&mut live, 4).await;
    let held = counts(&live, 1.0).await;
    let area = live.area.as_ref().unwrap();
    let centre = Vec3::new(8.0, 0.0, 2.0);
    let left: BTreeSet<TileKey> = area.registry.left_keys(centre).into_iter().filter(|k| held.contains_key(k)).collect();
    let far: BTreeSet<TileKey> = held.keys().copied().filter(|k| k.z <= -10).collect();
    let active: BTreeSet<TileKey> = area.active.iter().copied().filter(|k| held.contains_key(k)).collect();
    assert!(!left.is_empty() && !far.is_empty() && !active.is_empty(), "{left:?} {far:?} {active:?}");
    assert!(far.iter().all(|k| !area.registry.is_left(*k, centre) && !area.active.contains(k)));
    (live, dir, [left, far, active])
}

/// At the total cap the left tiles park first, then far tiles neither active nor near (farthest first), before the
/// refine; active tiles never park, and nothing is evicted.
#[tokio::test]
async fn at_the_cap_left_then_far_inactive_tiles_park_before_any_eviction() {
    let (mut live, dir, [left, far, active]) = walked("budget-room").await;
    let rows = counts(&live, 1.0).await;
    let sum = |keys: &BTreeSet<TileKey>| keys.iter().map(|k| rows[k] as u32).sum::<u32>();
    let room = |live: &LiveModel| live.config.max_splats - live.splats().unwrap().num_splats();
    // The previous refine's evictions beyond the room: the left rows plus one row more.
    live.last_refine_evicted = room(&live) + sum(&left) + 1;
    let evicted = live.num_evicted();
    train(&mut live, 1).await;
    let r = live.area_report().unwrap();
    let registry = &live.area.as_ref().unwrap().registry;
    let parked = |k: &TileKey| registry.get(*k).is_some_and(|t| !t.resident);
    assert!(left.iter().all(parked), "a left tile stayed: {r:?}");
    assert!(far.iter().any(parked), "no far tile parked: {r:?}");
    assert!(!active.iter().any(parked), "an active tile parked");
    assert_eq!(r.counters.parks_for_room, r.counters.parks, "{r:?}");
    assert_eq!(r.counters.eviction_fallbacks, 0, "the parks covered the need: {r:?}");
    assert_eq!(live.num_evicted(), evicted, "the refine still evicted");
    std::fs::remove_dir_all(dir).ok();

    // Asked for every row: the left and far tiles park first, then the eviction falls back (nothing inactive is left
    // resident, so it is unlimited); the active and near tiles stay, and so does at least one row.
    let (mut live, dir, [left, far, active]) = walked("budget-room-all").await;
    live.last_refine_evicted = u32::MAX;
    train(&mut live, 1).await;
    let r = live.area_report().unwrap();
    let registry = &live.area.as_ref().unwrap().registry;
    let parked = |k: &TileKey| registry.get(*k).is_some_and(|t| !t.resident);
    assert!(left.iter().chain(&far).all(parked), "a left or far tile stayed: {r:?}");
    assert_eq!(r.counters.eviction_fallbacks, 1, "{r:?}");
    assert!(active.iter().all(|k| registry.get(*k).is_some_and(|t| t.resident)), "an active tile parked");
    assert!(live.splats().unwrap().num_splats() > 0);
    std::fs::remove_dir_all(dir).ok();
}

/// The stall: inactive_at_cap at the cap, every tile within the radius (nothing parks), the wall held behind the
/// newest keyframe and the active tiles at their fair share. The eviction falls back to the held wall rows (outside
/// the newest keyframe's cone), and the next keyframe's seeds find room.
#[tokio::test]
async fn at_the_cap_with_nothing_to_park_the_eviction_falls_back_and_growth_goes_on() {
    let dir = tmp("budget-fallback");
    let config = GuideConfig {
        area_freeze: AreaFreeze::InactiveAtCap,
        area_radius_m: 50.0,
        area_seen_window_steps: 1000,
        area_tile_max_fraction: 0.0,
        evict_min_age: 0,
        evict_headroom: 0.5,
        ..active_config(2.0)
    };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 5.0), Vec3::new(-0.3, 0.0, 0.0)).await;
    train(&mut live, 5).await;
    // Turned around: the wall (z = 0) is behind the newest keyframe and no longer active.
    add(&mut live, &dir, 1, Vec3::new(-0.3, 0.0, 5.0), Vec3::new(-0.3, 0.0, 10.0)).await;
    train(&mut live, 4).await;
    let wall = |p: Vec3| p.z < 2.0;
    let n = live.splats().unwrap().num_splats();
    let wall_rows = rows_where(&live, wall).await.len();
    live.config.max_splats = n;
    live.trainer.as_mut().unwrap().note_seed_shortfall(1);
    let evicted = live.num_evicted();
    train(&mut live, 1).await;
    let r = live.area_report().unwrap();
    assert!(r.at_cap, "{r:?}");
    assert_eq!(r.counters.parks, 0, "{r:?}");
    assert_eq!(r.counters.eviction_fallbacks, 1, "{r:?}");
    assert!(live.num_evicted() > evicted, "nothing evicted: {r:?}");
    assert!(rows_where(&live, wall).await.len() < wall_rows, "no held wall row went");
    let after = live.splats().unwrap().num_splats();
    assert!(after < n);
    add(&mut live, &dir, 2, Vec3::new(-0.3, 0.0, 5.0), Vec3::new(3.0, 0.0, 9.0)).await;
    assert!(live.splats().unwrap().num_splats() > after, "the next keyframe found no room");
    std::fs::remove_dir_all(dir).ok();
}

/// The fallback's last tier: with every resident tile active or near the limit goes, as without area training; with one
/// tile outside the active set that tile may give rows, held ones included, and the limit stays for the others.
#[tokio::test]
async fn an_all_active_capture_evicts_without_a_limit() {
    for all_active in [true, false] {
        let dir = tmp(&format!("budget-all-active-{all_active}"));
        let config = GuideConfig { area_freeze: AreaFreeze::InactiveAtCap, ..active_config(50.0) };
        let mut live = LiveModel::new(config, device().await);
        live.enable_area(dir.join("tiles"));
        add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 5.0), Vec3::new(-0.3, 0.0, 0.0)).await;
        train(&mut live, 5).await;
        let rows = counts(&live, 1.0).await;
        let mut active: Vec<TileKey> = rows.keys().copied().collect();
        let outside = active.pop().unwrap();
        if all_active {
            active.push(outside);
        }
        live.area.as_mut().unwrap().active = active;
        let n = live.splats().unwrap().num_splats();
        live.config.max_splats = n;
        live.note_cap(n);
        let all = rows_where(&live, |_| true).await;
        live.set_row_budget(&all.means);
        assert!(live.trainer.as_ref().unwrap().row_budget().unwrap().limit_eviction);
        live.last_refine_evicted = n / 2;
        let splats = live.splats.take().unwrap();
        let splats = live.park_for_room(splats).await;
        live.splats = Some(splats);
        let budget = live.trainer.as_ref().unwrap().row_budget().unwrap().clone();
        let keys = live.area.as_ref().unwrap().budget_keys.clone();
        assert_eq!(live.area_report().unwrap().counters.eviction_fallbacks, 1);
        if all_active {
            assert!(!budget.limit_eviction, "the limit stayed");
        } else {
            let g = keys.iter().position(|k| *k == outside).unwrap();
            assert!(budget.limit_eviction && budget.evict_held);
            assert_eq!(budget.groups[g].evictable, u32::MAX, "the tile outside the active set was not opened");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// The row budget the refine gets: while the rows outside the active tiles are held an eviction may take each active
/// tile's rows above `max_splats` / active tiles and nothing of other tiles; with inactive_at_cap below the cap or
/// `area_freeze` left it is not limited, and left without a per-tile cap there is no budget.
#[tokio::test]
async fn the_eviction_keeps_each_active_tiles_fair_share() {
    let (mut live, dir, [_, _, active]) = walked("budget-fair").await;
    let all = rows_where(&live, |_| true).await;
    let rows = counts(&live, 1.0).await;
    let n_active = live.area.as_ref().unwrap().active.len() as u32;
    live.config.max_splats = 2 * n_active;
    live.set_row_budget(&all.means);
    let budget = live.trainer.as_ref().unwrap().row_budget().unwrap().clone();
    let keys = live.area.as_ref().unwrap().budget_keys.clone();
    assert!(budget.limit_eviction);
    for (key, g) in keys.iter().zip(&budget.groups) {
        let want = if active.contains(key) { rows[key] as u32 - 2.min(rows[key] as u32) } else { 0 };
        assert_eq!(g.evictable, want, "{key:?}");
    }
    let in_tile: Vec<bool> = rows_in(&keys[..1], &all.means, 1.0);
    assert!(budget.group.iter().zip(&in_tile).all(|(&g, &t)| (g == 0) == t), "groups follow the tiles");
    live.config.area_freeze = AreaFreeze::InactiveAtCap;
    live.set_row_budget(&all.means);
    assert!(!live.trainer.as_ref().unwrap().row_budget().unwrap().limit_eviction, "inactive_at_cap below the cap");
    live.config.area_freeze = AreaFreeze::Left;
    live.set_row_budget(&all.means);
    assert!(!live.trainer.as_ref().unwrap().row_budget().unwrap().limit_eviction);
    live.config.area_tile_max_fraction = 0.0;
    live.set_row_budget(&all.means);
    assert!(live.trainer.as_ref().unwrap().row_budget().is_none());
    std::fs::remove_dir_all(dir).ok();
}

/// Holding the rows outside no tile (nothing active) holds every row.
#[tokio::test]
async fn nothing_active_holds_every_row() {
    let (mut live, dir) = wall("budget-hold-all", AreaFreeze::Inactive).await;
    let splats = live.splats().unwrap().clone();
    live.hold_rows(&splats, Held::OutsideOf(BTreeSet::new())).await;
    assert_eq!(live.area_report().unwrap().frozen_rows, splats.num_splats());
    std::fs::remove_dir_all(dir).ok();
}
