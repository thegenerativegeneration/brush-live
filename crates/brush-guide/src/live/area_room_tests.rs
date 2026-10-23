//! Restores: left tiles make room for a restore, a deferral streak notes its shortfall once and ends
//! with a keyframe that does not defer the tile again; the seeds re-sort after a seeds-pass restore drops a tile; a
//! new restore empties the junction's batch cache.

use super::*;
use crate::config::GuideConfig;
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{drift_rows, not_left_from_6m, parked_origin, parked_origin_with, train};

/// Keeps a few resident rows at (1.5, 0, 1.5), next to where the walker comes back, so parking for room never has
/// to take every resident row.
async fn rows_by_the_walker(live: &mut LiveModel) {
    drift_rows(live, (0..5).flat_map(|i| [1.2 + 0.1 * i as f32, 0.0, 1.5]).collect()).await;
}

/// A restore at the cap that would otherwise defer parks the left tiles at x = 20, which keyframe 1's views still
/// see, and restores.
#[tokio::test]
async fn a_restore_at_the_cap_parks_a_seen_left_tile() {
    let (mut live, dir) = parked_origin("room-left").await;
    let left: Vec<TileKey> =
        live.area.as_ref().unwrap().registry.resident_keys().into_iter().filter(|k| k.x >= 10).collect();
    assert!(!left.is_empty());
    rows_by_the_walker(&mut live).await;
    let seen = live.iter;
    assert!(left.iter().all(|&k| live.area.as_ref().unwrap().registry.get(k).unwrap().last_seen_step == seen));
    live.config.max_splats = live.splats().unwrap().num_splats();
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.restores >= 1, "{r:?}");
    let registry = &live.area.as_ref().unwrap().registry;
    assert!(left.iter().any(|&k| registry.get(k).is_some_and(|t| !t.resident)), "no left tile parked for room");
    assert!(live.splats().unwrap().num_splats() <= live.config.max_splats, "{r:?}");
    std::fs::remove_dir_all(dir).ok();
}

/// A restore deferred while the seen tiles at x = 16 lie within radius + hysteresis comes back at the next keyframe,
/// once the walker has left them and they can be parked for it.
#[tokio::test]
async fn a_deferred_restore_comes_back_once_a_seen_left_tile_parks() {
    let dir = tmp("room-deferred");
    // Radius 2 + hysteresis 10: tiles park, and are left, beyond 12 m. The seed spacing follows max_splats ×
    // seed_view_fraction; a fraction of 1 keeps the seeds dense once the cap is lowered below.
    let config = GuideConfig { area_hysteresis_m: 10.0, seed_view_fraction: 1.0, ..area_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(16.0, 0.0, 2.0), Vec3::new(16.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    assert!(live.area_report().unwrap().summary.parked_tiles >= 1, "the origin parked");
    let far: Vec<TileKey> =
        live.area.as_ref().unwrap().registry.resident_keys().into_iter().filter(|k| k.x >= 10).collect();
    assert!(!far.is_empty());
    rows_by_the_walker(&mut live).await;
    live.config.max_splats = live.splats().unwrap().num_splats();
    // At x = 6 the origin's seeds bring it back, but x = 16 lies 10 m away: not left, and seen at the last refine
    // (no training follows).
    add(&mut live, &dir, 2, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.deferrals >= 1 && r.counters.restores == 0, "{r:?}");
    // At x = 3, x = 16 lies 13 m away: left.
    add(&mut live, &dir, 3, Vec3::new(3.0, 0.0, 2.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.restores >= 1, "{r:?}");
    let registry = &live.area.as_ref().unwrap().registry;
    assert!(far.iter().any(|&k| registry.get(k).is_some_and(|t| !t.resident)), "no left tile parked for room");
    std::fs::remove_dir_all(dir).ok();
}

/// A tile that stays deferred notes its shortfall once per streak; the streak ends with a keyframe that does not
/// defer it again, and the next deferral notes it again.
#[tokio::test]
async fn a_deferral_streak_notes_its_shortfall_once() {
    let (mut live, dir) = parked_origin("defer-streak").await;
    let keys = live.area.as_ref().unwrap().registry.parked_keys();
    let (a, b) = (keys[0], keys[1]);
    let pending = |live: &LiveModel| live.trainer.as_ref().unwrap().pending_seed_shortfall();
    let base = pending(&live);
    live.defer(a, 100);
    live.defer(a, 100);
    assert_eq!(pending(&live), base + 100, "a second deferral in the streak noted its shortfall");
    live.defer(b, 7);
    assert_eq!(pending(&live), base + 107);
    live.end_deferral_streaks();
    live.defer(a, 100);
    assert_eq!(pending(&live), base + 107, "both were deferred at the last keyframe: the streaks go on");
    live.end_deferral_streaks();
    let streaks: Vec<TileKey> = live.area.as_ref().unwrap().defer.streaks.keys().copied().collect();
    assert_eq!(streaks, vec![a], "b was not deferred again");
    // A keyframe at x = 20 defers nothing: a's streak ends with it.
    add(&mut live, &dir, 2, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    assert!(live.area.as_ref().unwrap().defer.streaks.is_empty());
    let base = pending(&live);
    live.defer(a, 5);
    assert_eq!(pending(&live), base + 5, "a new streak notes its shortfall");
    assert_eq!(live.area_report().unwrap().counters.deferrals, 5, "every deferral counts");
    std::fs::remove_dir_all(dir).ok();
}

/// When a seeds-pass restore drops the tile the seeds hit (its file is unreadable), the seeds
/// are sorted again against the registry as it is now, so the keyframe's rows land in the dropped tile's ground (a
/// tile the walker has not left).
#[tokio::test]
async fn seeds_land_where_a_seeds_pass_restore_dropped_the_tile() {
    let (mut live, dir) = parked_origin_with("seeds-resort", not_left_from_6m()).await;
    let tiles = dir.join("tiles");
    let parked = live.area.as_ref().unwrap().registry.parked_keys();
    for &key in &parked {
        std::fs::write(crate::area::park_file::tile_path(&tiles, key), b"not a tile").unwrap();
    }
    // 6 m away: beyond the radius, so the seeds pass tries them.
    add(&mut live, &dir, 2, Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.drops >= 1 && r.counters.restores == 0, "{r:?}");
    let dropped: Vec<TileKey> =
        parked.into_iter().filter(|&k| live.area.as_ref().unwrap().registry.get(k).is_none()).collect();
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let landed = (0..rows.len())
        .filter(|&i| rows.origin[i] == Some(2) && dropped.contains(&TileKey::of(rows.mean(i), 1.0)))
        .count();
    assert!(landed > 0, "no seed of keyframe 2 in the dropped tiles {dropped:?}");
    std::fs::remove_dir_all(dir).ok();
}

/// A restore while a junction runs starts it over with an empty batch cache.
#[tokio::test]
async fn a_new_restore_empties_the_junction_cache() {
    let (mut live, dir) = parked_origin("junction-restart").await;
    let pos = Vec3::new(0.3, 0.0, 2.0);
    add(&mut live, &dir, 2, pos, Vec3::ZERO).await;
    train(&mut live, 4).await;
    assert!(!live.area.as_ref().unwrap().junction.cache.is_empty());
    let key = live.area.as_ref().unwrap().registry.resident_keys().into_iter().find(|k| k.x <= 0).unwrap();
    let splats = live.splats.take().unwrap();
    let splats = live.park_tiles(splats, &[key]).await;
    let splats = live.restore_tiles(splats, &[key], RestoreReason::Distance, pos).await;
    live.splats = Some(splats);
    let junction = &live.area.as_ref().unwrap().junction;
    assert_eq!(junction.steps_left, 10);
    assert!(junction.cache.is_empty(), "the restarted junction kept {} decoded batches", junction.cache.len());
    std::fs::remove_dir_all(dir).ok();
}

/// Fix round 1: a tile that stays deferred notes its shortfall at the first deferral and every
/// `DEFER_RENOTE_EVERY` after, so a large tile still gets its room after a bounded number of retries.
#[tokio::test]
async fn a_long_deferral_streak_notes_its_shortfall_every_ten_deferrals() {
    let (mut live, dir) = parked_origin("defer-renote").await;
    let key = live.area.as_ref().unwrap().registry.parked_keys()[0];
    let pending = |live: &LiveModel| live.trainer.as_ref().unwrap().pending_seed_shortfall();
    let base = pending(&live);
    for n in 1..=25u32 {
        live.defer(key, 100);
        let noted = 1 + (n - 1) / DEFER_RENOTE_EVERY;
        assert_eq!(pending(&live), base + 100 * noted, "after {n} deferrals");
    }
    std::fs::remove_dir_all(dir).ok();
}
