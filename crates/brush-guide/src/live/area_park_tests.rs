use super::*;
use crate::area::{TileKey, read_tile};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use crate::live::ViewPose;
use crate::live::area::tests::{add, area_config, device, keyframe_toward, tmp};
use glam::{Mat4, Vec3};

async fn train(live: &mut LiveModel, steps: u32) {
    for _ in 0..steps {
        live.train_step().await;
    }
}

/// The tile files in `dir`, by name, with their bytes.
fn tile_files(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap())
        })
        .collect();
    files.sort();
    files
}

/// Origin tiles park once training moves 20 m away; their splats leave the trainer, their file exists, and the
/// eval model still holds them.
#[tokio::test]
async fn tiles_left_behind_park_and_stay_in_the_eval_model() {
    let dir = tmp("park");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    // Refines at steps 10 and 15: the origin tiles (last seen at step 5) are unseen by keyframe 1's views.
    train(&mut live, 10).await;
    let report = live.area_report().unwrap();
    assert!(report.counters.parks >= 1 && report.summary.parked_splats > 0, "{report:?}");
    let c = report.counters;
    assert!(c.max_park_ms > 0.0 && c.park_ms >= c.max_park_ms && c.tile_bytes > 0, "{c:?}");
    assert!(!report.parking_off);
    let resident = live.splats().unwrap().clone();
    let means: Vec<f32> = resident.means().into_data_async().await.unwrap().try_to_vec().unwrap();
    assert!(means.chunks(3).all(|p| p[0] > 10.0), "origin splats left the trainer, keyframe 1's stayed");
    assert!(std::fs::read_dir(dir.join("tiles")).unwrap().count() >= 1);
    let eval = live.eval_splats().await.unwrap();
    assert_eq!(eval.num_splats(), resident.num_splats() + report.summary.parked_splats);
    std::fs::remove_dir_all(dir).ok();
}

/// Review Focus 3: when every tile is out of range, the tile under the newest keyframe's seeds still holds rows,
/// and a park that would take every row is skipped.
#[tokio::test]
async fn parking_never_empties_the_trainer() {
    let dir = tmp("park-empty");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    let before = live.splats().unwrap().num_splats();
    // The newest keyframe stands 30 m away and looks at empty space (no feature points, so it seeds nothing):
    // every resident tile is out of range and unseen, so the park would take every row.
    let (mut h, p) = keyframe_toward(1, Vec3::new(30.0, 0.0, 2.0), Vec3::new(30.0, 0.0, -10.0));
    h.num_points = 0;
    let p = p[..h.jpeg_len as usize].to_vec();
    assert!(live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap()).await);
    assert_eq!(live.splats().unwrap().num_splats(), before, "the empty view seeded nothing");
    train(&mut live, 10).await;
    let r = live.area_report().unwrap();
    assert!(r.counters.parks_skipped >= 1, "the guard was reached: {r:?}");
    assert_eq!(r.counters.parks, 0);
    assert!(live.splats().unwrap().num_splats() > 0);
    std::fs::remove_dir_all(dir).ok();
}

/// The user requirement: an area the walker has left does not change until the walker returns. Once parked, a
/// tile's file and the rows it loads stay exactly as they were while training goes on elsewhere, and each file
/// records the pose of every view that seeded its rows.
#[tokio::test]
async fn parked_tiles_stay_unchanged_while_training_goes_on() {
    let dir = tmp("park-frozen");
    let tiles = dir.join("tiles");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(tiles.clone());
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    let parked = live.area.as_ref().unwrap().registry.parked_keys();
    assert!(!parked.is_empty());
    let files = tile_files(&tiles);
    let rows: Vec<_> = parked.iter().map(|&k| live.load_parked(k).unwrap()).collect();
    for &key in &parked {
        let tile = read_tile(&tiles, key).unwrap();
        assert!(tile.rows.origin.iter().all(|o| o.is_some_and(|s| tile.slot_poses.iter().any(|p| p.slot == s))));
        assert_eq!(tile.slot_poses[0].position, live.views[0].camera.position);
    }
    train(&mut live, 20).await;
    assert_eq!(live.area.as_ref().unwrap().registry.parked_keys(), parked);
    assert_eq!(tile_files(&tiles), files, "a parked tile's file changed");
    let again: Vec<_> = parked.iter().map(|&k| live.load_parked(k).unwrap()).collect();
    assert_eq!(again, rows, "a parked tile's rows changed");
    std::fs::remove_dir_all(dir).ok();
}

/// A parked tile loads where its splats would be had they stayed resident: a pose correction of its seeding view
/// while it is parked moves its rows by the same rigid step.
#[tokio::test]
async fn parked_tiles_load_with_current_pose_corrections() {
    let dir = tmp("park-poses");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    let key: TileKey = live.area.as_ref().unwrap().registry.parked_keys()[0];
    let before = live.load_parked(key).unwrap();
    // View 0 shifts 0.5 m along x, its view direction unchanged.
    let pose = Mat4::look_at_rh(Vec3::new(0.5, 0.0, 2.0), Vec3::new(0.5, 0.0, 0.0), Vec3::Y).inverse();
    assert_eq!(live.update_poses(&[ViewPose::from_arkit(0, &pose.to_cols_array()).unwrap()]).moved, 1);
    let after = live.load_parked(key).unwrap();
    assert_eq!(after.len(), before.len());
    for i in 0..before.len() {
        let d = after.mean(i) - before.mean(i);
        assert!((d - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-4, "row {i} moved by {d}");
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Carry-over from Task 8: a park rebuilds the all-views loader when it still holds views that left the area, so
/// no stale view keeps training toward a tile that just parked.
#[tokio::test]
async fn a_park_drops_departed_views_from_the_all_views_loader() {
    let dir = tmp("park-loader");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    // View 1 looks away from the origin tiles, out of view 0's area: the all-views loader holds view 1 only.
    add(&mut live, &dir, 1, Vec3::new(2.5, 0.0, 2.0), Vec3::new(2.5, 0.0, 12.0)).await;
    assert_eq!(live.all_members, vec![1]);
    // From here views leave without the count rule rebuilding: view 1 leaves view 2's area but stays in the loader.
    live.config.all_loader_rebuild_every = 10;
    add(&mut live, &dir, 2, Vec3::new(5.0, 0.0, 2.0), Vec3::new(5.0, 0.0, 12.0)).await;
    assert_eq!((live.training_views(), live.all_members.clone()), (vec![2], vec![1]));
    train(&mut live, 10).await;
    assert!(live.area_report().unwrap().counters.parks >= 1);
    assert_eq!(live.all_members, live.training_views());
    std::fs::remove_dir_all(dir).ok();
}

/// A tile whose file cannot be written stays resident: its rows go back to the trainer instead of being lost, and
/// parking stays off until a reset, even once the directory is back.
#[tokio::test]
async fn a_tile_that_cannot_be_written_stays_resident() {
    let dir = tmp("park-unwritable");
    let mut live = LiveModel::new(area_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    std::fs::remove_dir_all(dir.join("tiles")).unwrap();
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    let r = live.area_report().unwrap();
    assert_eq!((r.counters.parks, r.summary.parked_tiles), (0, 0), "{r:?}");
    assert!(r.parking_off, "the eval record shows that parking is off");
    let means: Vec<f32> = live.splats().unwrap().means().into_data_async().await.unwrap().try_to_vec().unwrap();
    assert!(means.chunks(3).any(|p| p[0] < 10.0), "the origin rows were lost");
    std::fs::create_dir_all(dir.join("tiles")).unwrap();
    train(&mut live, 10).await;
    assert_eq!(live.area_report().unwrap().counters.parks, 0, "parking went on after a failed write");
    std::fs::remove_dir_all(dir).ok();
}

/// Without area training a refine reads nothing back and parks nothing; the eval model is the trainer's.
#[tokio::test]
async fn area_off_parks_nothing() {
    let dir = tmp("park-off");
    let mut live = LiveModel::new(GuideConfig { refine_every: 5, ..area_config(0.0) }, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    assert!(live.area_report().is_none());
    assert_eq!(live.eval_splats().await.unwrap().num_splats(), live.splats().unwrap().num_splats());
    std::fs::remove_dir_all(dir).ok();
}

/// Review I4: rows that drift into a parked tile beyond radius + hysteresis join its file instead of bringing it
/// back; the tile stays parked. Its old rows take the pose correction made while it was parked once, in the file,
/// whose slots then record the current poses, so a later load moves nothing again.
#[tokio::test]
async fn resident_rows_in_a_far_parked_tile_join_its_file() {
    use crate::live::area_restore::tests::{assert_no_resident_row_in_parked_tiles, correction, drift_rows, parked_origin};
    let (mut live, dir) = parked_origin("merge-far").await;
    let tiles = dir.join("tiles");
    let key = live.area.as_ref().unwrap().registry.parked_keys()[0];
    let old = read_tile(&tiles, key).unwrap();
    live.update_poses(&[correction(0, Vec3::new(0.1, 0.0, 2.0), Vec3::new(0.1, 0.0, 0.0))]);
    let corner = key.min_xz(1.0);
    drift_rows(&mut live, (0..3).flat_map(|i| [corner.x + 0.2 + 0.1 * i as f32, 0.0, corner.y + 0.5]).collect()).await;
    train(&mut live, 5).await;
    let r = live.area_report().unwrap();
    assert_eq!(r.counters.restores, 0, "{r:?}");
    assert!(live.area.as_ref().unwrap().registry.parked_keys().contains(&key));
    let file = read_tile(&tiles, key).unwrap();
    let n = old.rows.len();
    assert_eq!(file.rows.len(), n + 3);
    assert_eq!(file.version, old.version + 1);
    assert_eq!(live.area.as_ref().unwrap().registry.get(key).unwrap().splats as usize, n + 3);
    for i in (0..n).filter(|&i| old.rows.origin[i] == Some(0)) {
        let d = file.rows.mean(i) - old.rows.mean(i);
        assert!((d - Vec3::new(0.1, 0.0, 0.0)).length() < 1e-4, "row {i} moved by {d}");
    }
    assert!((n..n + 3).all(|i| TileKey::of(file.rows.mean(i), 1.0) == key));
    assert_eq!(live.load_parked(key).unwrap(), file.rows, "the file's slots record the current poses");
    assert_no_resident_row_in_parked_tiles(&live).await;
    std::fs::remove_dir_all(dir).ok();
}
