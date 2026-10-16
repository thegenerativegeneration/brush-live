//! Task 10b: the rows of a tile the walker has left stay as they were while resident; tiles never visited and
//! restored tiles train; the trainer's mask stays row-aligned through parks, restores, appends and refines.

use super::*;
use crate::area::TileKey;
use crate::config::GuideConfig;
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{add_seeing, drift_rows, parked_origin, train};
use crate::live::area::tests::add_at;
use brush_train::rows::SplatRows;

/// The trainer's rows whose mean satisfies `keep`, in trainer order.
pub(in crate::live) async fn rows_where(live: &LiveModel, keep: impl Fn(Vec3) -> bool) -> SplatRows {
    let all = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let idx: Vec<usize> = (0..all.len()).filter(|&i| keep(all.mean(i))).collect();
    all.select(&idx)
}

/// The rows' parameters and Mip floor as bits (age and eviction importance are bookkeeping, not the model).
fn params(r: &SplatRows) -> Vec<Vec<u32>> {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect();
    vec![bits(&r.means), bits(&r.rotations), bits(&r.log_scales), bits(&r.sh_coeffs), bits(&r.raw_opacities),
        bits(&r.min_scale)]
}

/// [`params`] per row, sorted.
pub(in crate::live) fn by_row(r: &SplatRows) -> Vec<Vec<u32>> {
    let cols = params(r);
    let mut rows: Vec<Vec<u32>> = (0..r.len())
        .map(|i| cols.iter().flat_map(|c| { let w = c.len() / r.len(); c[i * w..(i + 1) * w].to_vec() }).collect())
        .collect();
    rows.sort();
    rows
}

/// Near the origin, where keyframe 0 seeds (tiles x, z in -1..=0 at 1 m tiles).
fn at_origin(p: Vec3) -> bool {
    p.x.abs() < 1.5 && p.z.abs() < 1.5
}

/// Per row, whether the trainer holds it (all false without a mask), and per row whether it lies in a tile the mask
/// was built from.
async fn held_and_left(live: &LiveModel) -> (Vec<bool>, Vec<bool>) {
    let splats = live.splats.as_ref().unwrap();
    let n = splats.num_splats() as usize;
    let held = match live.trainer.as_ref().unwrap().frozen_rows() {
        Some(m) => m
            .clone()
            .int()
            .into_data_async()
            .await
            .unwrap()
            .convert::<i32>()
            .try_into_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|v| v != 0)
            .collect(),
        None => vec![false; n],
    };
    assert_eq!(held.len(), n, "the mask is out of step with the splats");
    let means: Vec<f32> = splats.means().into_data_async().await.unwrap().try_to_vec().unwrap();
    let keys: Vec<TileKey> = live.area.as_ref().unwrap().frozen_keys.iter().copied().collect();
    (held, crate::area::rows_in(&keys, &means, 1.0))
}

/// After a refine the mask holds exactly the rows in the left resident tiles.
async fn assert_holds_left_rows(live: &LiveModel) {
    let (held, left) = held_and_left(live).await;
    assert_eq!(held, left);
    assert_eq!(live.area_report().unwrap().frozen_rows as usize, held.iter().filter(|&&h| h).count());
}

/// After a keyframe the mask holds rows in left tiles only; rows appended since are not held.
async fn assert_holds_only_left_rows(live: &LiveModel) {
    let (held, left) = held_and_left(live).await;
    assert!(held.iter().zip(&left).all(|(&h, &l)| !h || l), "a held row lies outside the left tiles");
}

/// (a) The user requirement: keyframe 1 stands 6 m away (the origin tiles are left) and looks back at them, so its
/// training views draw them and they stay resident; their rows keep every parameter and their floor bit for bit
/// across steps and two refines.
#[tokio::test]
async fn a_left_tile_in_view_stays_as_the_walker_left_it() {
    let dir = tmp("freeze-left");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    // Keyframe 1 seeds into the left tiles too; its rows are not held until the next refine.
    let keys = live.area.as_ref().unwrap().frozen_keys.clone();
    let kf0_rows = async |live: &LiveModel| {
        let rows = rows_where(live, |p| keys.contains(&TileKey::of(p, 1.0))).await;
        rows.select(&(0..rows.len()).filter(|&i| rows.origin[i] == Some(0)).collect::<Vec<_>>())
    };
    let before = kf0_rows(&live).await;
    let held = live.area_report().unwrap().frozen_rows;
    assert!(before.len() > 10 && held as usize == before.len(), "{held} held, {} of keyframe 0", before.len());
    train(&mut live, 10).await;
    // Every row is still there, bit for bit (a row trained elsewhere may have moved in).
    let after = by_row(&kf0_rows(&live).await);
    assert!(by_row(&before).iter().all(|row| after.contains(row)), "a left tile changed");
    let r = live.area_report().unwrap();
    assert!(r.frozen_rows >= held, "{r:?}");
    let area = live.area.as_ref().unwrap();
    for &k in &keys {
        let t = area.registry.get(k).unwrap();
        assert!(t.resident && t.last_seen_step == live.iter, "tile {k:?} was not drawn in the last window: {t:?}");
    }
    assert_holds_left_rows(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// (b) Ground the walker never stood near keeps training, however far: keyframe 0 looks 10 m ahead.
#[tokio::test]
async fn an_unvisited_far_tile_still_trains() {
    let dir = tmp("freeze-unvisited");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    let far = |p: Vec3| p.z < -5.0;
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -8.0)).await;
    let before = rows_where(&live, far).await;
    assert!(before.len() > 10);
    train(&mut live, 10).await;
    let after = rows_where(&live, far).await;
    assert_ne!(params(&after), params(&before), "far rows did not train");
    assert_eq!(live.area_report().unwrap().frozen_rows, 0);
    assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
    std::fs::remove_dir_all(dir).ok();
}

/// (c) A tile the walker returns to comes back and trains; only the tiles it left on the way are held.
#[tokio::test]
async fn a_restored_tile_trains_again() {
    let (mut live, dir) = parked_origin("freeze-restored").await;
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO).await;
    assert!(live.area_report().unwrap().counters.restores >= 1);
    assert_holds_only_left_rows(&live).await;
    let (held, _) = held_and_left(&live).await;
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    assert!((0..rows.len()).all(|i| !held[i] || rows.mean(i).x > 15.0), "a restored row is held");
    let before = rows_where(&live, at_origin).await;
    train(&mut live, 10).await;
    let after = rows_where(&live, at_origin).await;
    let sum = |r: &SplatRows| r.means.iter().map(|&v| f64::from(v)).sum::<f64>();
    assert_ne!(sum(&before), sum(&after), "the restored tile did not train");
    std::fs::remove_dir_all(dir).ok();
}

/// (d) Eviction at the cap takes no held row: keyframe 1 turns away from the left origin tiles and its seeds overrun
/// the budget; copies of origin rows (same scores) in a tile nobody visited go instead.
#[tokio::test]
async fn eviction_at_the_cap_never_takes_held_rows() {
    let dir = tmp("freeze-evict");
    let config = GuideConfig {
        max_splats: 60,
        seed_view_fraction: 1.0,
        evict_min_age: 0,
        evict_max_cell_fraction: 1.0,
        area_seen_window_steps: 1000,
        ..area_config(2.0)
    };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    // With parking on, the refine would park the left tiles first (`left_tiles_park_before_an_eviction_bites`).
    live.area.as_mut().unwrap().parking_off = true;
    let (pos, away) = (Vec3::new(6.0, 0.0, 2.0), Vec3::new(6.0, 0.0, 10.0));
    add_seeing(&mut live, &dir, 1, pos, away, &[away]).await;
    let held = live.area_report().unwrap().frozen_rows;
    let keys = live.area.as_ref().unwrap().frozen_keys.clone();
    let in_left = |p: Vec3| keys.contains(&TileKey::of(p, 1.0));
    let before = rows_where(&live, in_left).await;
    assert!(held as usize == before.len() && held > 10, "{held} held, {} in the left tiles", before.len());
    // 20 copies behind keyframe 1 (out of its protected cone), 8 m from it: never visited.
    drift_rows(&mut live, (0..20).flat_map(|i| [6.1 + 0.04 * i as f32, 0.0, -5.5]).collect()).await;
    let evicted = live.num_evicted();
    train(&mut live, 5).await;
    assert!(live.num_evicted() > evicted, "nothing was evicted");
    let after = rows_where(&live, in_left).await;
    assert_eq!(params(&after), params(&before), "a held row was evicted or changed");
    std::fs::remove_dir_all(dir).ok();
}

/// (e) Area training off: no mask is ever built, whatever the walker does.
#[tokio::test]
async fn area_off_holds_nothing() {
    for enable in [true, false] {
        let dir = tmp(&format!("freeze-off-{enable}"));
        let radius = if enable { 0.0 } else { 2.0 };
        let mut live = LiveModel::new(area_config(radius), device().await);
        if enable {
            live.enable_area(dir.join("tiles"));
        }
        add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 5).await;
        add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
        assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
        train(&mut live, 10).await;
        assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
        assert!(live.area_report().is_none());
        std::fs::remove_dir_all(dir).ok();
    }
}

/// (f) The mask holds the rows of the left resident tiles after every refine, and only such rows in between, through
/// keyframes that seed, parks, a restore and the appends they make (every step checks its length).
#[tokio::test]
async fn the_mask_follows_parks_restores_appends_and_refines() {
    let dir = tmp("freeze-aligned");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    assert_holds_only_left_rows(&live).await;
    assert!(live.area_report().unwrap().frozen_rows > 0);
    train(&mut live, 5).await;
    assert_holds_left_rows(&live).await;
    // Turned away at x = 20, the walker leaves the origin and x = 6 unseen: both park.
    add(&mut live, &dir, 2, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    assert_holds_only_left_rows(&live).await;
    train(&mut live, 5).await;
    assert!(live.area_report().unwrap().counters.parks >= 1);
    assert_holds_left_rows(&live).await;
    // Back at the origin: x = 20 is left now, the origin comes back.
    add(&mut live, &dir, 3, Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO).await;
    assert!(live.area_report().unwrap().counters.restores >= 1);
    assert_holds_only_left_rows(&live).await;
    train(&mut live, 5).await;
    assert_holds_left_rows(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// Fix round 1: the seeds of a keyframe 6 m from the left origin tiles neither bring them back nor land in them.
#[tokio::test]
async fn seeds_on_a_left_tile_neither_restore_it_nor_land() {
    let (mut live, dir) = parked_origin("freeze-seeds-left").await;
    let pos = Vec3::new(0.0, 0.0, 6.0);
    add(&mut live, &dir, 2, pos, Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert_eq!(r.counters.restores, 0, "{r:?}");
    let left = live.area.as_ref().unwrap().registry.left_keys(pos);
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let kf2: Vec<usize> = (0..rows.len()).filter(|&i| rows.origin[i] == Some(2)).collect();
    assert!(kf2.iter().all(|&i| !left.contains(&TileKey::of(rows.mean(i), 1.0))), "a seed landed in a left tile");
    std::fs::remove_dir_all(dir).ok();
}

/// Fix round 1: seeds on ground the walker never stood near still land, however far.
#[tokio::test]
async fn seeds_on_unvisited_far_ground_still_land() {
    let dir = tmp("freeze-seeds-far");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add_at(&mut live, &dir, 0, &[0.0]).await;
    train(&mut live, 5).await;
    // Turned sideways, away from the origin rows, toward ground 8 m off.
    add(&mut live, &dir, 1, Vec3::new(0.3, 0.0, 2.0), Vec3::new(-8.0, 0.0, 2.0)).await;
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let far = (0..rows.len()).filter(|&i| rows.origin[i] == Some(1) && rows.mean(i).x < -5.0).count();
    assert!(far > 0, "no seed of keyframe 1 landed 8 m away");
    std::fs::remove_dir_all(dir).ok();
}

/// Fix round 1: at the cap, with the left origin tiles held and in view, a refine about to evict parks them first
/// instead of evicting rows of the live area.
#[tokio::test]
async fn left_tiles_park_before_an_eviction_bites() {
    let dir = tmp("freeze-park-before-evict");
    let config = GuideConfig {
        max_splats: 60,
        seed_view_fraction: 1.0,
        evict_min_age: 0,
        evict_max_cell_fraction: 1.0,
        area_seen_window_steps: 1000,
        ..area_config(2.0)
    };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    // Keyframe 1 looks back at the origin (left, held, in view); its seeds overrun the budget.
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    assert!(live.area_report().unwrap().frozen_rows > 10);
    // 20 scored rows of the live area, behind keyframe 1 (out of its protected cone): what an eviction would take.
    let live_rows = |p: Vec3| p.x > 6.0 && p.z > 2.5;
    drift_rows(&mut live, (0..20).flat_map(|i| [6.1 + 0.04 * i as f32, 0.0, 2.8]).collect()).await;
    let before = rows_where(&live, live_rows).await;
    assert_eq!(before.len(), 20);
    assert!(live.trainer.as_ref().unwrap().pending_evictions(live.splats().unwrap().num_splats()) > 0);
    let evicted = live.num_evicted();
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.parks >= 1, "{r:?}");
    assert_eq!(live.num_evicted(), evicted, "the eviction still took splats: {r:?}");
    assert_eq!(rows_where(&live, live_rows).await.len(), 20, "live-area rows went");
    std::fs::remove_dir_all(dir).ok();
}

/// Steady state: with no eviction pending, the splats the previous refine evicted (its own growth demand, which
/// `pending_evictions` cannot see) beyond the room left below the cap still park the held left tiles before the next
/// refine, so they make the room instead of the live area; evictions the room absorbs park nothing.
#[tokio::test]
async fn the_last_refines_evictions_park_left_tiles_before_the_next() {
    let dir = tmp("freeze-park-last-evicted");
    let config = GuideConfig { area_seen_window_steps: 1000, ..area_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    // Keyframe 1 looks back at the origin: left, held, in view, so no park after a refine takes it.
    add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
    let held = live.area_report().unwrap().frozen_rows;
    assert!(held > 10);
    assert_eq!(live.trainer.as_ref().unwrap().pending_evictions(live.splats().unwrap().num_splats()), 0);
    // The 5 steps before the next refine change no count, so the room now is the room at that refine.
    let room = |live: &LiveModel| 20_000 - live.splats().unwrap().num_splats();
    live.last_refine_evicted = room(&live);
    train(&mut live, 5).await;
    assert_eq!(live.area_report().unwrap().counters.parks, 0, "evictions the room absorbs park nothing");
    live.last_refine_evicted = room(&live) + held;
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.parks >= 1, "{r:?}");
    std::fs::remove_dir_all(dir).ok();
}

/// `refine` records its own evictions as the next refine's floor: at the cap, with parking off, one refine evicts and
/// `last_refine_evicted` is what it evicted.
#[tokio::test]
async fn a_refine_records_what_it_evicted() {
    let dir = tmp("freeze-last-evicted");
    let config = GuideConfig {
        max_splats: 60,
        seed_view_fraction: 1.0,
        evict_min_age: 0,
        evict_max_cell_fraction: 1.0,
        ..area_config(2.0)
    };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    live.area.as_mut().unwrap().parking_off = true;
    add(&mut live, &dir, 1, Vec3::new(0.3, 0.0, 2.0), Vec3::new(0.3, 0.0, 0.0)).await;
    let before = live.num_evicted();
    train(&mut live, 5).await;
    let evicted = live.num_evicted() - before;
    assert!(evicted > 0, "the refine evicted nothing");
    assert_eq!(u64::from(live.last_refine_evicted), evicted);
    std::fs::remove_dir_all(dir).ok();
}
