//! The rows held still in left tiles keep at most `area_held_fraction` of `max_splats`; after a refine the
//! farthest held tiles park, even in view, until the rest fit, and they park as they were held.

use super::tests::{by_row, rows_where};
use super::*;
use crate::area::{ParkedTile, decode_tile, encode_tile, read_tile};
use brush_train::rows::SplatRows;
use crate::config::{AreaFreeze, GuideConfig};
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::train;
use std::path::PathBuf;

/// Keyframe 0 at the origin, then keyframe 1 6 m on looking back at it: the origin tiles are left, held and in view
/// (a seen window of 1000 steps keeps them from parking unseen). The budget is 20 000 splats.
async fn looking_back(name: &str, fraction: f32) -> (LiveModel, PathBuf) {
    looking_back_with(name, GuideConfig { area_held_fraction: fraction, ..area_config(2.0) }).await
}

/// [`looking_back`] with `config` (its seen window set to 1000 steps).
async fn looking_back_with(name: &str, config: GuideConfig) -> (LiveModel, PathBuf) {
    let dir = tmp(name);
    let config = GuideConfig { area_seen_window_steps: 1000, ..config };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    (live, dir)
}

/// `rows` as a tile file stores them (f16 columns, normalised rotations).
fn as_filed(rows: SplatRows) -> SplatRows {
    let tile = ParkedTile { key: TileKey { x: 0, z: 0 }, version: 1, slot_poses: Vec::new(), rows };
    decode_tile(&encode_tile(&tile)).unwrap().rows
}

/// Without a cap (fraction 1.0) the held tiles stay resident; with a cap of half the held rows the farthest held
/// tiles park at the next refine, though in view, until the held rows fit, and each parked file holds its rows as
/// they were held (bit for bit as the file stores them).
#[tokio::test]
async fn held_rows_over_the_cap_park_farthest_first_even_in_view() {
    let (mut live, dir) = looking_back("cap-off", 1.0).await;
    let held = live.area_report().unwrap().frozen_rows;
    assert!(held > 10, "{held} held");
    train(&mut live, 10).await;
    let r = live.area_report().unwrap();
    assert_eq!((r.counters.parks, r.counters.parks_for_cap), (0, 0), "{r:?}");
    assert!(r.frozen_rows >= held, "{r:?}");
    std::fs::remove_dir_all(dir).ok();

    let cap = held / 2;
    let (mut live, dir) = looking_back("cap-half", (cap as f32 + 0.5) / 20_000.0).await;
    let keys = live.area.as_ref().unwrap().frozen_keys.clone();
    let before = rows_where(&live, |p| keys.contains(&TileKey::of(p, 1.0))).await;
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.parks_for_cap >= 1 && r.counters.parks == r.counters.parks_for_cap, "{r:?}");
    assert!(r.frozen_rows <= cap, "{} held over a cap of {cap}", r.frozen_rows);
    let area = live.area.as_ref().unwrap();
    let centre = Vec3::new(6.0, 0.0, 2.0);
    let dist = |k: &TileKey| k.footprint_distance(centre, 1.0);
    let parked = area.registry.parked_keys();
    let nearest_parked = parked.iter().map(dist).fold(f32::INFINITY, f32::min);
    assert!(area.frozen_keys.iter().all(|k| dist(k) <= nearest_parked), "a nearer held tile parked first");
    for k in &parked {
        assert_eq!(area.registry.get(*k).unwrap().last_seen_step, live.iter, "parked tile {k:?} was not in view");
    }
    let in_parked = |i: &usize| parked.contains(&TileKey::of(before.mean(*i), 1.0));
    let (gone, stayed): (Vec<usize>, Vec<usize>) = (0..before.len()).partition(in_parked);
    let mut filed = Vec::new();
    for &k in &parked {
        filed.extend(by_row(&read_tile(&dir.join("tiles"), k).unwrap().rows));
    }
    filed.sort();
    assert_eq!(filed, by_row(&as_filed(before.select(&gone))), "a held row changed on its way to the tile file");
    // Every held row that stayed is still there, bit for bit (a row trained elsewhere may have moved in).
    let resident = by_row(&rows_where(&live, |p| keys.contains(&TileKey::of(p, 1.0))).await);
    assert!(by_row(&before.select(&stayed)).iter().all(|row| resident.contains(row)), "a held row changed");
    std::fs::remove_dir_all(dir).ok();
}

/// Area training off: the cap parks nothing and nothing is held.
#[tokio::test]
async fn area_off_has_no_cap() {
    let dir = tmp("cap-area-off");
    let config = GuideConfig { area_held_fraction: 0.0, ..area_config(0.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 10).await;
    assert!(live.area_report().is_none());
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
    assert!(!dir.join("tiles").exists(), "a tile parked");
    std::fs::remove_dir_all(dir).ok();
}

/// With `area_freeze` none, the left tiles' rows keep training while resident, nothing is held, and the cap still
/// parks the farthest left tiles once their rows exceed it.
#[tokio::test]
async fn hold_off_left_rows_train_and_the_cap_parks_the_farthest() {
    let hold_off =
        |fraction| GuideConfig { area_freeze: AreaFreeze::Off, area_held_fraction: fraction, ..area_config(2.0) };
    let (mut live, dir) = looking_back_with("cap-hold-off", hold_off(1.0)).await;
    let left = live.area.as_ref().unwrap().registry.left_keys(Vec3::new(6.0, 0.0, 2.0));
    let in_left = |p: Vec3| left.contains(&TileKey::of(p, 1.0));
    let before = rows_where(&live, in_left).await;
    assert!(before.len() > 10, "{} rows in left tiles", before.len());
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
    train(&mut live, 10).await;
    let after = rows_where(&live, in_left).await;
    let sum = |r: &SplatRows| r.means.iter().map(|&v| f64::from(v)).sum::<f64>();
    assert_ne!(sum(&before), sum(&after), "the left rows did not train");
    let r = live.area_report().unwrap();
    assert_eq!((r.frozen_rows, r.counters.parks), (0, 0), "{r:?}");
    assert!(r.left_splats as usize >= before.len() / 2, "{r:?}");
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
    std::fs::remove_dir_all(dir).ok();

    let cap = before.len() as u32 / 2;
    let (mut live, dir) = looking_back_with("cap-hold-off-half", hold_off((cap as f32 + 0.5) / 20_000.0)).await;
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.parks_for_cap >= 1 && r.counters.parks == r.counters.parks_for_cap, "{r:?}");
    assert!(r.left_splats <= cap, "{} left rows over a cap of {cap}", r.left_splats);
    assert_eq!(r.frozen_rows, 0);
    let area = live.area.as_ref().unwrap();
    let centre = Vec3::new(6.0, 0.0, 2.0);
    let dist = |k: &TileKey| k.footprint_distance(centre, 1.0);
    let nearest_parked = area.registry.parked_keys().iter().map(dist).fold(f32::INFINITY, f32::min);
    let resident_left: Vec<TileKey> =
        left.iter().copied().filter(|&k| area.registry.get(k).is_some_and(|t| t.resident)).collect();
    assert!(resident_left.iter().all(|k| dist(k) <= nearest_parked), "a nearer left tile parked first");
    std::fs::remove_dir_all(dir).ok();
}
