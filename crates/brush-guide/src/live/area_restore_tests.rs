use super::*;
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::area::tests::{add, area_config, device, keyframe_toward, tmp};
use brush_train::rows::SplatRows;
use glam::Vec3;

pub(in crate::live) async fn train(live: &mut LiveModel, steps: u32) {
    for _ in 0..steps {
        live.train_step().await;
    }
}

pub(in crate::live) async fn parked_origin_with(name: &str, config: GuideConfig) -> (LiveModel, std::path::PathBuf) {
    let dir = tmp(name);
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    assert!(live.area_report().unwrap().summary.parked_tiles >= 1);
    (live, dir)
}

pub(in crate::live) async fn parked_origin(name: &str) -> (LiveModel, std::path::PathBuf) {
    parked_origin_with(name, area_config(2.0)).await
}

/// No resident row lies in a parked tile (Review Focus 2).
pub(in crate::live) async fn assert_no_resident_row_in_parked_tiles(live: &LiveModel) {
    let parked = live.area.as_ref().unwrap().registry.parked_keys();
    let means: Vec<f32> = live.splats().unwrap().means().into_data_async().await.unwrap().try_to_vec().unwrap();
    for p in means.chunks(3) {
        let key = TileKey::of(Vec3::new(p[0], p[1], p[2]), 1.0);
        assert!(!parked.contains(&key), "a resident splat at {p:?} sits in parked tile {key:?}");
    }
}

pub(in crate::live) fn correction(id: u64, pos: Vec3, target: Vec3) -> crate::live::ViewPose {
    crate::live::ViewPose::from_arkit(id, &keyframe_toward(id, pos, target).0.pose).unwrap()
}

#[tokio::test]
async fn coming_back_restores_by_distance_and_starts_a_junction() {
    let (mut live, dir) = parked_origin("restore-distance").await;
    let parked = live.area_report().unwrap().summary.parked_splats;
    let before = live.splats().unwrap().num_splats();
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    // Tiles within 2 m come back by distance, the rest the keyframe sees by seed hits.
    assert!(r.counters.restores >= 1 && r.summary.parked_splats < parked, "{r:?}");
    assert!(r.counters.restore_ms > 0.0 && r.counters.restore_ms >= r.counters.max_restore_ms, "{r:?}");
    assert!(live.splats().unwrap().num_splats() > before);
    assert!(live.junction_steps_left() > 0);
    assert!(!live.area.as_ref().unwrap().junction.views.is_empty(), "a view sees the restored tiles");
    train(&mut live, 4).await;
    assert!(live.junction_steps_left() < 10, "junction views were drawn");
    assert!(!live.area.as_ref().unwrap().junction.cache.is_empty(), "drawn junction views stay decoded");
    train(&mut live, 30).await;
    assert_eq!(live.junction_steps_left(), 0);
    let junction = &live.area.as_ref().unwrap().junction;
    assert!(junction.boxes.is_empty() && junction.cache.is_empty(), "cleared when the junction ends");
    std::fs::remove_dir_all(dir).ok();
}

/// Radius 2 m and hysteresis 5 m: from 6 m the origin tiles are beyond the radius but not left (Task 10b: seeds on
/// left tiles are dropped).
pub(in crate::live) fn not_left_from_6m() -> GuideConfig {
    GuideConfig { area_hysteresis_m: 5.0, ..area_config(2.0) }
}

/// A parked tile within radius + far_seed that the keyframe's seeds hit comes back before seeding, and the keyframe
/// seeds again against an alpha mask that holds the restored splats.
#[tokio::test]
async fn seeds_in_a_near_parked_tile_restore_it_before_seeding() {
    let (mut live, dir) = parked_origin_with("restore-seeds", not_left_from_6m()).await;
    let parked = live.area_report().unwrap().summary.parked_splats;
    let before = live.splats().unwrap().num_splats();
    // 6 m away: beyond the 2 m radius, within 2 + 10 m, and its seeds land on the parked origin tiles.
    add(&mut live, &dir, 2, Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.restores >= 1, "{r:?}");
    assert!(live.splats().unwrap().num_splats() - before >= parked);
    let (first, second) = live.last_reseed().expect("the keyframe reseeded after the restore");
    assert!(second < first, "seeds before the restore {first}, after {second}");
    assert_no_resident_row_in_parked_tiles(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// User decision D2: a parked tile seen from beyond radius + far_seed stays parked and gets no seeds.
#[tokio::test]
async fn far_seeds_leave_a_parked_tile_parked() {
    let config = GuideConfig { area_far_seed_m: 1.0, ..not_left_from_6m() };
    let (mut live, dir) = parked_origin_with("far-seeds", config).await;
    let parked_tiles = live.area_report().unwrap().summary.parked_tiles;
    add(&mut live, &dir, 2, Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert_eq!((r.counters.restores, r.summary.parked_tiles), (0, parked_tiles), "{r:?}");
    assert_no_resident_row_in_parked_tiles(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// Review Focus 2 / M3: seeds below the restore threshold that land in a parked tile are dropped, not appended.
#[tokio::test]
async fn few_seeds_in_a_parked_tile_are_dropped_not_appended() {
    let config = GuideConfig { area_restore_min_seeds: 1_000_000, ..not_left_from_6m() };
    let (mut live, dir) = parked_origin_with("few-seeds", config).await;
    add(&mut live, &dir, 2, Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO).await;
    assert_eq!(live.area_report().unwrap().counters.restores, 0);
    assert_no_resident_row_in_parked_tiles(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// Appends `n` copies of resident rows at `means` (flat xyz), as rows drifting across a border would land.
pub(in crate::live) async fn drift_rows(live: &mut LiveModel, means: Vec<f32>) {
    let n = means.len() / 3;
    let all = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let mut rows = all.select(&(0..n).collect::<Vec<_>>());
    rows.means = means;
    let s = live.splats.take().unwrap();
    live.splats = Some(live.trainer.as_mut().unwrap().append_rows(s, &rows));
}

/// Review Focus 2: resident rows that drift into a parked tile within radius + hysteresis bring the tile back once,
/// not a second copy. (Beyond that they join its file: `area_park_tests`.)
#[tokio::test]
async fn resident_rows_in_a_near_parked_tile_restore_it_once() {
    let config = GuideConfig { area_hysteresis_m: 5.0, ..area_config(2.0) };
    let (mut live, dir) = parked_origin_with("restore-rows", config).await;
    // 5 m from the origin tiles: beyond the radius (no restore by distance), within radius + hysteresis.
    let centre = Vec3::new(5.0, 0.0, 2.0);
    add(&mut live, &dir, 2, centre, Vec3::new(5.0, 0.0, 0.0)).await;
    assert_eq!(live.area_report().unwrap().counters.restores, 0);
    let key = live
        .area
        .as_ref()
        .unwrap()
        .registry
        .parked_keys()
        .into_iter()
        .min_by(|a, b| a.footprint_distance(centre, 1.0).total_cmp(&b.footprint_distance(centre, 1.0)))
        .unwrap();
    assert!(key.footprint_distance(centre, 1.0) <= 7.0);
    let parked_here = live.area.as_ref().unwrap().registry.get(key).unwrap().splats;
    let corner = key.min_xz(1.0);
    drift_rows(&mut live, (0..3).flat_map(|i| [corner.x + 0.2 + 0.1 * i as f32, 0.0, corner.y + 0.5]).collect()).await;
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.restores >= 1, "{r:?}");
    assert!(!live.area.as_ref().unwrap().registry.parked_keys().contains(&key));
    assert!(!crate::area::park_file::tile_path(&dir.join("tiles"), key).exists());
    // The tile came back once: it holds its parked rows and the drifted ones (which no view sees, so the refine
    // neither grew nor moved them), not a second copy.
    let means: Vec<f32> = live.splats().unwrap().means().into_data_async().await.unwrap().try_to_vec().unwrap();
    let here = means.chunks(3).filter(|p| TileKey::of(Vec3::new(p[0], p[1], p[2]), 1.0) == key).count() as u32;
    assert!((parked_here..=parked_here + 3).contains(&here), "{here} rows in the tile, {parked_here} parked: {r:?}");
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn a_pose_correction_while_parked_moves_the_restored_splats() {
    let (mut live, dir) = parked_origin("restore-pose").await;
    let key = live.area.as_ref().unwrap().registry.parked_keys()[0];
    let parked = crate::area::park_file::read_tile(&dir.join("tiles"), key).unwrap().rows;
    live.update_poses(&[correction(0, Vec3::new(0.1, 0.0, 2.0), Vec3::new(0.1, 0.0, 0.0))]);
    let moved = live.load_parked(key).unwrap();
    for i in 0..parked.len() {
        if parked.origin[i] == Some(0) {
            assert!((moved.mean(i).x - parked.mean(i).x - 0.1).abs() < 1e-4);
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// User decision D1 and Review Focus 5: a loop closure that moves the parked tile's only view by 2 m is applied
/// on restore, not treated as stale. Two corrections while parked compose: the restored rows move once, by both.
#[tokio::test]
async fn a_large_consistent_correction_is_applied_on_restore() {
    let (mut live, dir) = parked_origin("restore-large").await;
    let tiles = dir.join("tiles");
    let mut parked: Vec<Vec3> = Vec::new();
    for key in live.area.as_ref().unwrap().registry.parked_keys() {
        let rows = crate::area::park_file::read_tile(&tiles, key).unwrap().rows;
        parked.extend((0..rows.len()).filter(|&i| rows.origin[i] == Some(0)).map(|i| rows.mean(i)));
    }
    live.update_poses(&[correction(0, Vec3::new(2.0, 0.0, 2.0), Vec3::new(2.0, 0.0, 0.0))]);
    live.update_poses(&[correction(0, Vec3::new(2.5, 0.0, 2.0), Vec3::new(2.5, 0.0, 0.0))]);
    add(&mut live, &dir, 2, Vec3::new(2.3, 0.0, 2.0), Vec3::new(2.5, 0.0, 0.0)).await;
    let r = live.area_report().unwrap();
    assert_eq!(r.counters.drops, 0, "{r:?}");
    assert!(r.counters.restores >= 1, "{r:?}");
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let restored: Vec<Vec3> = (0..rows.len()).filter(|&i| rows.origin[i] == Some(0)).map(|i| rows.mean(i)).collect();
    assert!(!restored.is_empty());
    for m in restored {
        let from = m - Vec3::new(2.5, 0.0, 0.0);
        assert!(parked.iter().any(|p| p.distance(from) < 1e-4), "{m} is no parked row moved by 2.5 m in x");
    }
    std::fs::remove_dir_all(dir).ok();
}

/// User decision D1: views that disagree by more than the limit about where the tile is drop it.
#[tokio::test]
async fn views_disagreeing_about_a_parked_tile_drop_it() {
    use crate::area::park_file::{SlotPose, read_tile, write_tile};
    let (mut live, dir) = parked_origin("restore-inconsistent").await;
    let tiles = dir.join("tiles");
    let centre = Vec3::new(0.3, 0.0, 2.0);
    let key = live
        .area
        .as_ref()
        .unwrap()
        .registry
        .parked_keys()
        .into_iter()
        .min_by(|a, b| a.footprint_distance(centre, 1.0).total_cmp(&b.footprint_distance(centre, 1.0)))
        .unwrap();
    assert!(key.footprint_distance(centre, 1.0) <= 2.0, "the next keyframe brings this tile back by distance");
    // Give half the tile's rows to view 1 (at x = 20), as if both views had seeded it.
    let mut tile = read_tile(&tiles, key).unwrap();
    let half = tile.rows.len() / 2;
    for o in &mut tile.rows.origin[..half] {
        *o = Some(1);
    }
    let cam1 = live.views()[1].camera;
    tile.slot_poses.push(SlotPose { slot: 1, position: cam1.position, rotation: cam1.rotation });
    write_tile(&tiles, &tile).unwrap();
    // A correction moves view 0 by 1.5 m and leaves view 1: the views disagree by 1.5 m > 1 m.
    live.update_poses(&[correction(0, Vec3::new(1.5, 0.0, 2.0), Vec3::new(1.5, 0.0, 0.0))]);
    add(&mut live, &dir, 2, centre, Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.drops >= 1, "{r:?}");
    assert!(!live.area.as_ref().unwrap().registry.parked_keys().contains(&key));
    assert!(!crate::area::park_file::tile_path(&tiles, key).exists());
    std::fs::remove_dir_all(dir).ok();
}

/// Review Focus 4: a restore at the cap keeps the trainer within `max_splats` and loses no row to eviction:
/// the restore is deferred (or room is made by parking unseen far tiles), never paid for by evicting.
#[tokio::test]
async fn a_restore_at_the_cap_stays_within_the_budget() {
    let (mut live, dir) = parked_origin("restore-cap").await;
    let report = live.area_report().unwrap();
    let total_before = live.splats().unwrap().num_splats() + report.summary.parked_splats;
    // Leave room for half the parked splats only.
    live.config.max_splats = live.splats().unwrap().num_splats() + report.summary.parked_splats / 2;
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::ZERO).await;
    let r = live.area_report().unwrap();
    assert!(live.splats().unwrap().num_splats() <= live.config.max_splats, "{r:?}");
    assert!(r.counters.deferrals + r.counters.restores >= 1, "{r:?}");
    let total_after = live.splats().unwrap().num_splats() + r.summary.parked_splats;
    assert!(total_after >= total_before, "no splat was evicted to make room: {total_before} -> {total_after}");
    std::fs::remove_dir_all(dir).ok();
}

/// One keyframe at the origin trained for 5 steps (one refine, so the registry knows its tiles), then one of its
/// tiles parked directly: the model, its directory, the tile and the rows the park took.
async fn one_tile_parked(name: &str, config: GuideConfig) -> (LiveModel, std::path::PathBuf, TileKey, SplatRows) {
    let dir = tmp(name);
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    let keys = live.area.as_ref().unwrap().registry.resident_keys();
    assert!(keys.len() >= 2, "the park must leave rows behind: {keys:?}");
    let key = keys[0];
    let all = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let idx: Vec<usize> = (0..all.len()).filter(|&i| TileKey::of(all.mean(i), 1.0) == key).collect();
    let splats = live.splats.take().unwrap();
    live.splats = Some(live.park_tiles(splats, &[key]).await);
    assert_eq!(live.area.as_ref().unwrap().registry.parked_keys(), vec![key]);
    (live, dir, key, all.select(&idx))
}

/// The restored rows: the last `n` rows of the trainer after restoring `key`.
async fn restore_one(live: &mut LiveModel, key: TileKey, n: usize) -> SplatRows {
    let splats = live.splats.take().unwrap();
    let splats = live.restore_tiles(splats, &[key], RestoreReason::Distance, Vec3::ZERO).await;
    let back = live.trainer.as_ref().unwrap().read_rows(&splats).await;
    live.splats = Some(splats);
    back.select(&((back.len() - n)..back.len()).collect::<Vec<_>>())
}

/// The user requirement, read before any refine touches them: a parked and restored tile comes back as it was
/// taken, means exact, rotations normalised and the other columns within fp16, with its origins, ages, importance,
/// Mip floor and trained flags, and its Adam moments within bf16 (Task 17b).
#[tokio::test]
async fn a_restored_tile_equals_the_parked_one() {
    let (mut live, dir, key, taken) = one_tile_parked("restore-equal", area_config(2.0)).await;
    let back = restore_one(&mut live, key, taken.len()).await;
    assert_eq!(back.means, taken.means, "means are exact");
    assert_eq!((&back.origin, &back.age), (&taken.origin, &taken.age));
    assert_eq!(back.importance, taken.importance, "a tile parked for no step keeps its importance");
    let close = |a: &[f32], b: &[f32], what: &str| {
        assert_eq!(a.len(), b.len(), "{what}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 2e-3 * y.abs().max(1.0), "{what}: {x} vs {y}");
        }
    };
    let unit = |q: &[f32]| -> Vec<f32> {
        q.chunks(4)
            .flat_map(|r| {
                let l = r.iter().map(|v| v * v).sum::<f32>().sqrt();
                r.iter().map(move |v| v / l)
            })
            .collect()
    };
    close(&back.rotations, &unit(&taken.rotations), "rotations");
    close(&back.log_scales, &taken.log_scales, "log-scales");
    close(&back.raw_opacities, &taken.raw_opacities, "opacities");
    close(&back.sh_coeffs, &taken.sh_coeffs, "SH");
    close(&back.min_scale, &taken.min_scale, "floor");
    assert_eq!(back.trained, taken.trained);
    let (b, t) = (back.moments.expect("restored moments"), taken.moments.expect("parked moments"));
    assert!(t.opacity_m1.iter().any(|&v| v != 0.0), "the steps left moments to park");
    for ((name, got, _), (_, want, _)) in b.columns(4).iter().zip(t.columns(4)) {
        assert_eq!(got.len(), want.len(), "{name}");
        for (x, y) in got.iter().zip(want) {
            assert!((x - y).abs() <= y.abs() / 256.0, "{name}: {x} vs {y}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Task 17b: a tile restored k steps after its park comes back with its first moments scaled by β1^k (what k steps
/// without a gradient leave of them), its second moments as stored.
#[tokio::test]
async fn a_restore_decays_the_first_moments_by_the_steps_parked() {
    let (mut live, dir, key, taken) = one_tile_parked("restore-decay", area_config(2.0)).await;
    live.iter += 7;
    let b = restore_one(&mut live, key, taken.len()).await.moments.expect("restored moments");
    let t = taken.moments.expect("parked moments");
    let f = 0.9f32.powi(7);
    for ((name, got, _), (_, was, _)) in b.columns(4).iter().zip(t.columns(4)) {
        let scale = if name.ends_with("m1") { f } else { 1.0 };
        for (x, y) in got.iter().zip(was) {
            assert!((x - y * scale).abs() <= (y * scale).abs() / 128.0, "{name}: {x} vs {y}·{scale}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A tile parked for more than 10 seen windows comes back unscored (importance `+inf`); a shorter park keeps it.
#[tokio::test]
async fn a_long_park_resets_the_importance() {
    let config = GuideConfig { area_seen_window_steps: 1, ..area_config(2.0) };
    let (mut live, dir, key, taken) = one_tile_parked("restore-stale", config.clone()).await;
    assert!(taken.importance.iter().any(|i| i.is_finite()), "the refine scored the rows");
    live.iter += 10;
    assert_eq!(restore_one(&mut live, key, taken.len()).await.importance, taken.importance, "10 windows: kept");
    std::fs::remove_dir_all(dir).ok();
    let (mut live, dir, key, taken) = one_tile_parked("restore-stale", config).await;
    live.iter += 11;
    let back = restore_one(&mut live, key, taken.len()).await;
    assert!(back.importance.iter().all(|i| i.is_infinite()), "11 windows: unscored");
    std::fs::remove_dir_all(dir).ok();
}

/// A tile file with another SH degree than the model's is dropped (and retrains from its keyframes), not restored.
#[tokio::test]
async fn a_tile_of_another_sh_degree_is_dropped() {
    use crate::area::park_file::{read_tile, tile_path, write_tile};
    let (mut live, dir) = parked_origin("restore-sh").await;
    let tiles = dir.join("tiles");
    let keys = live.area.as_ref().unwrap().registry.parked_keys();
    for &key in &keys {
        let mut tile = read_tile(&tiles, key).unwrap();
        let coeffs = tile.rows.sh_coeffs_per_channel as usize;
        let band_0 = |v: &[f32]| v.chunks(coeffs * 3).flat_map(|c| c[..3].to_vec()).collect();
        tile.rows.sh_coeffs = band_0(&tile.rows.sh_coeffs);
        if let Some(m) = tile.rows.moments.as_mut() {
            m.sh_m1 = band_0(&m.sh_m1);
        }
        tile.rows.sh_coeffs_per_channel = 1;
        write_tile(&tiles, &tile).unwrap();
    }
    let splats = live.splats.take().unwrap();
    let before = splats.num_splats();
    let splats = live.restore_tiles(splats, &keys, RestoreReason::Distance, Vec3::ZERO).await;
    assert_eq!(splats.num_splats(), before);
    let r = live.area_report().unwrap();
    assert_eq!((r.counters.restores, r.counters.drops, r.summary.parked_tiles), (0, keys.len() as u32, 0), "{r:?}");
    assert!(keys.iter().all(|&k| !tile_path(&tiles, k).exists()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn restore_reasons_print_as_the_log_names_them() {
    let names: Vec<String> =
        [RestoreReason::Distance, RestoreReason::Seeds, RestoreReason::Rows].iter().map(ToString::to_string).collect();
    assert_eq!(names, ["distance", "seeds", "rows"]);
}

/// `keyframe_toward`'s camera with its feature points replaced by 5×5 grids 0.15 m apart (x and y) around `centres`.
pub(in crate::live) async fn add_seeing(
    live: &mut LiveModel,
    dir: &std::path::Path,
    id: u64,
    pos: Vec3,
    target: Vec3,
    centres: &[Vec3],
) {
    let (mut h, p) = keyframe_toward(id, pos, target);
    let mut payload = p[..h.jpeg_len as usize].to_vec();
    let points: Vec<Vec3> = centres
        .iter()
        .flat_map(|&c| (-2..=2).flat_map(move |i| (-2..=2).map(move |j| c + Vec3::new(i as f32, j as f32, 0.0) * 0.15)))
        .collect();
    payload.extend(points.iter().flat_map(|q| q.to_array()).flat_map(f32::to_le_bytes));
    h.num_points = points.len() as u32;
    assert!(live.add_keyframe(decode_keyframe(&h, &payload, dir).await.unwrap()).await);
}

/// Review I1: a seeds-pass restore at the cap that cannot fit parks nothing for room, so no seed lands in a tile
/// parked after the seeds were sorted.
#[tokio::test]
async fn seeds_never_land_in_a_tile_parked_for_room() {
    // The seed spacing follows max_splats × seed_view_fraction; a fraction of 1 keeps it at its minimum once the cap
    // is lowered below.
    // Hysteresis 10 m: from x = 20 the origin parks; from 10 m back it is not left, so its seeds still count.
    let config = GuideConfig { area_hysteresis_m: 10.0, seed_view_fraction: 1.0, ..area_config(2.0) };
    let (mut live, dir) = parked_origin_with("room-seeds", config).await;
    // The walker stands at x = 4 looking away from the origin. Behind it, within radius + hysteresis (so it does
    // not park) and seen by no view, a resident tile at x = 3 holds 5 rows.
    add_seeing(&mut live, &dir, 2, Vec3::new(4.0, 0.0, 2.0), Vec3::new(4.0, 0.0, 12.0), &[Vec3::new(4.0, 0.0, 12.0)])
        .await;
    let room_tile = TileKey { x: 3, z: 0 };
    drift_rows(&mut live, (0..5).flat_map(|i| [3.1 + 0.2 * i as f32, 3.0, 0.5]).collect()).await;
    train(&mut live, 10).await;
    assert!(live.area.as_ref().unwrap().registry.get(room_tile).is_some_and(|t| t.resident));
    live.config.max_splats = live.splats().unwrap().num_splats();
    // From 10 m back the keyframe sees the parked origin tiles and that tile; its points lie in both, and its
    // seeds hit the origin tiles beyond the radius, so they come back by seeds, which the cap defers.
    let (pos, target) = (Vec3::new(0.5, 0.0, 10.0), Vec3::new(0.5, 0.0, 0.0));
    add_seeing(&mut live, &dir, 3, pos, target, &[Vec3::ZERO, Vec3::new(3.5, 0.0, 0.5)]).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.deferrals >= 1 && r.counters.restores == 0, "{r:?}");
    assert!(live.area.as_ref().unwrap().registry.get(room_tile).is_some_and(|t| t.resident), "not parked for room");
    assert_no_resident_row_in_parked_tiles(&live).await;
    std::fs::remove_dir_all(dir).ok();
}

/// Review I3: a restore at the cap is deferred without reading the tile file, logged once per streak, and comes
/// back within a few keyframes once a refine makes room (here by parking the area the walker left). The area at x = 20
/// counts as restored at keyframe 2, so for one keyframe it cannot be parked for room (left tiles could be, Task 10b).
#[tokio::test]
async fn a_deferred_restore_comes_back_once_room_is_made() {
    let config = GuideConfig { area_min_resident_keyframes: 2, ..area_config(2.0) };
    let (mut live, dir) = parked_origin_with("restore-retry", config).await;
    let area = live.area.as_mut().unwrap();
    for key in area.registry.resident_keys() {
        area.registry.mark_restored(key, 2, live.iter);
    }
    let tiles = dir.join("tiles");
    let (pos, away) = (Vec3::new(0.3, 0.0, 2.0), Vec3::new(0.3, 0.0, -3.0));
    let registry = &live.area.as_ref().unwrap().registry;
    let near = registry.parked_within_radius(pos);
    assert!(near.len() >= 2, "{near:?}");
    let smallest = near.iter().map(|&k| registry.get(k).unwrap().splats).min().unwrap();
    assert!(smallest >= 4, "room for seeds but for no tile: {smallest}");
    // A deferral that read this file would drop the tile.
    std::fs::write(crate::area::park_file::tile_path(&tiles, near[0]), b"not a tile").unwrap();
    live.config.max_splats = live.splats().unwrap().num_splats() + smallest - 1;
    // The keyframe looks past the parked tiles, so its seeds (3 m beyond them) stay resident.
    add_seeing(&mut live, &dir, 2, pos, away, &[away]).await;
    let r = live.area_report().unwrap();
    assert_eq!((r.counters.restores, r.counters.drops), (0, 0), "{r:?}");
    assert!(r.counters.deferrals as usize >= near.len(), "{r:?}");
    assert_eq!(live.area.as_ref().unwrap().deferring.len(), near.len());
    let mut retries = 0;
    while live.area_report().unwrap().counters.restores == 0 {
        retries += 1;
        assert!(retries <= 3, "still deferred: {:?}", live.area_report().unwrap());
        train(&mut live, 5).await;
        add_seeing(&mut live, &dir, 2 + retries, pos, away, &[away]).await;
    }
    let area = live.area.as_ref().unwrap();
    assert!(area.deferring.keys().all(|k| area.registry.get(*k).is_some_and(|t| !t.resident)), "streaks end");
    std::fs::remove_dir_all(dir).ok();
}
