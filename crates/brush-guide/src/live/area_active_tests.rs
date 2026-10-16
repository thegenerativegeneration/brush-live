//! Task 17a: with `area_freeze` inactive only the active tiles (the tiles the newest keyframe looks at) train, the
//! training views are the keyframes that see one, and keyframes seed only into active tiles and new ground.

use super::*;
use crate::area::TileKey;
use crate::config::{AreaFreeze, GuideConfig};
use crate::live::area::tests::{active_config, add, device, tmp};
use crate::live::area_freeze::tests::{by_row, rows_where};
use crate::live::area_restore::tests::{parked_origin_with, train};
use glam::Vec3;

fn key(x: i32, z: i32) -> TileKey {
    TileKey { x, z }
}

fn tile_of(p: Vec3) -> TileKey {
    TileKey::of(p, 1.0)
}

/// The rows without their Mip floor.
fn without_floor(mut rows: brush_train::rows::SplatRows) -> brush_train::rows::SplatRows {
    rows.min_scale.clear();
    rows
}

fn active(live: &LiveModel) -> Vec<TileKey> {
    live.area.as_ref().unwrap().active.clone()
}

/// Keyframe 0 at x = -0.3 seeds a wall at z = 0 across tiles (-2, 0), (-1, 0) (in the middle of its image) and
/// (0, 0). With two active tiles the tile underfoot and (-1, 0) train; the other rows keep every parameter and their
/// floor bit for bit across steps and refines.
#[tokio::test]
async fn only_the_active_tiles_train() {
    let dir = tmp("active-train");
    let mut live = LiveModel::new(active_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 2.0), Vec3::new(-0.3, 0.0, 0.0)).await;
    assert_eq!(active(&live), vec![key(-1, 2), key(-1, 0)]);
    // The seeds span tiles (-2, 0), (-1, 0) and (0, 0).
    let inactive = |p: Vec3| tile_of(p) != key(-1, 0);
    let trained = |p: Vec3| tile_of(p) == key(-1, 0);
    let (before, before_active) = (rows_where(&live, inactive).await, rows_where(&live, trained).await);
    assert!(!before.is_empty() && !before_active.is_empty());
    let r = live.area_report().unwrap();
    assert_eq!(r.frozen_rows as usize, before.len(), "{r:?}");
    assert_eq!((r.active_tiles.clone(), r.active_splats as usize), (vec![[-1, 2], [-1, 0]], before_active.len()));
    // The first refine gives the first keyframe's rows their first Mip floor (they had none); their raw parameters
    // stay. From then on every held row is still there, floor and all, bit for bit (a split child of an active row
    // may have moved in).
    train(&mut live, 5).await;
    let floored = rows_where(&live, inactive).await;
    let after = by_row(&without_floor(floored.clone()));
    assert!(by_row(&without_floor(before)).iter().all(|row| after.contains(row)), "an inactive row trained");
    let floored = by_row(&floored);
    train(&mut live, 10).await;
    let after = by_row(&rows_where(&live, inactive).await);
    assert!(floored.iter().all(|row| after.contains(row)), "an inactive row changed");
    assert_ne!(by_row(&rows_where(&live, trained).await), by_row(&before_active), "the active rows did not train");
    assert!(live.area_report().unwrap().frozen_rows as usize >= floored.len());
    std::fs::remove_dir_all(dir).ok();
}

/// The newest keyframe turns toward (0, 0): it becomes active and trains, and (-1, 0) is held from then on.
#[tokio::test]
async fn the_mask_follows_the_active_set() {
    let dir = tmp("active-follow");
    let mut live = LiveModel::new(active_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 2.0), Vec3::new(-0.3, 0.0, 0.0)).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(0.7, 0.0, 2.0), Vec3::new(0.7, 0.0, 0.0)).await;
    assert_eq!(active(&live), vec![key(0, 2), key(0, 0)]);
    let left_behind = |p: Vec3| tile_of(p) == key(-1, 0);
    let now_active = |p: Vec3| tile_of(p) == key(0, 0);
    let (held, trains) = (rows_where(&live, left_behind).await, rows_where(&live, now_active).await);
    train(&mut live, 10).await;
    let after = by_row(&rows_where(&live, left_behind).await);
    assert!(by_row(&held).iter().all(|row| after.contains(row)), "a row of the inactive tile changed");
    assert_ne!(by_row(&rows_where(&live, now_active).await), by_row(&trains), "the newly active tile did not train");
    std::fs::remove_dir_all(dir).ok();
}

/// Keyframes 0 (looks at the origin tiles), 1 (same spot, turned around), 2 (5 m off, looks away) and 3 (8 m back,
/// beyond the 4 m view) are followed by 4, which looks at the origin again: the training views are 0 and 4. With
/// `area_freeze` left or none the views within the 2 m radius train as before (0, 1 and 4) and nothing is active.
#[tokio::test]
async fn the_training_views_see_an_active_tile() {
    for freeze in [AreaFreeze::Inactive, AreaFreeze::Left, AreaFreeze::Off] {
        let dir = tmp(&format!("active-views-{freeze:?}"));
        let mut live = LiveModel::new(GuideConfig { area_freeze: freeze, ..active_config(2.0) }, device().await);
        live.enable_area(dir.join("tiles"));
        add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
        add(&mut live, &dir, 1, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 6.0)).await;
        add(&mut live, &dir, 2, Vec3::new(5.0, 0.0, 2.0), Vec3::new(5.0, 0.0, 0.0)).await;
        add(&mut live, &dir, 3, Vec3::new(0.1, 0.0, 8.0), Vec3::ZERO).await;
        add(&mut live, &dir, 4, Vec3::new(0.2, 0.0, 2.0), Vec3::new(0.2, 0.0, 0.0)).await;
        if freeze == AreaFreeze::Inactive {
            assert_eq!(live.training_views(), vec![0, 4]);
            assert_eq!(active(&live)[0], key(0, 2));
            assert!(active(&live).contains(&key(0, 0)), "{:?}", active(&live));
        } else {
            assert_eq!(live.training_views(), vec![0, 1, 4], "{freeze:?}");
            assert!(active(&live).is_empty());
            assert!(live.area_report().unwrap().active_tiles.is_empty());
        }
        for _ in 0..16 {
            for loader in [live.recent.as_mut().unwrap(), live.all.as_mut().unwrap()] {
                let p = loader.next_batch().await.camera.position;
                // Views 0, 1 and 4 stand at z = 2 with x 0 or 0.2; views 2 and 3 do not.
                assert!(p.x < 0.3 && p.z < 3.0, "{freeze:?} served {p}");
            }
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// When no active tile has a box (none holds rows), the views within the radius train.
#[tokio::test]
async fn without_an_active_box_the_radius_views_train() {
    let dir = tmp("active-fallback");
    let mut live = LiveModel::new(active_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    add(&mut live, &dir, 1, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 4.0)).await;
    assert_eq!(live.training_views(), vec![1], "keyframe 0 does not see the wall keyframe 1 seeded behind it");
    live.area.as_mut().unwrap().active_boxes.clear();
    live.refresh_area_views();
    assert_eq!(live.training_views(), vec![0, 1]);
    std::fs::remove_dir_all(dir).ok();
}

/// Keyframe 1 looks at the right end of keyframe 0's wall and the ground beyond it; only the tile underfoot is
/// active. Its seeds in tiles that hold rows are dropped, those on new ground land. With `area_freeze` left (Task
/// 10b) its seeds land in tiles with rows too, so the scene does seed there.
#[tokio::test]
async fn seeds_outside_the_active_tiles_are_dropped_but_new_ground_seeds() {
    for freeze in [AreaFreeze::Inactive, AreaFreeze::Left] {
        let dir = tmp(&format!("active-seeds-{freeze:?}"));
        // Radius 3 m: from keyframe 1 the origin tiles are not left (Task 10b drops seeds on left tiles).
        let config = GuideConfig { area_freeze: freeze, area_active_tiles: 1, ..active_config(3.0) };
        let mut live = LiveModel::new(config, device().await);
        live.enable_area(dir.join("tiles"));
        add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 5).await;
        let known: Vec<TileKey> = live.area.as_ref().unwrap().registry.resident_keys();
        add(&mut live, &dir, 1, Vec3::new(0.8, 0.0, 3.0), Vec3::new(0.8, 0.0, 0.8)).await;
        let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
        let kf1: Vec<TileKey> =
            (0..rows.len()).filter(|&i| rows.origin[i] == Some(1)).map(|i| tile_of(rows.mean(i))).collect();
        let on_known = kf1.iter().filter(|k| known.contains(k)).count();
        assert!(kf1.len() > on_known, "{freeze:?}: no seed on new ground");
        if freeze == AreaFreeze::Inactive {
            assert_eq!(on_known, 0, "a seed landed in an inactive tile with rows");
        } else {
            assert!(on_known > 0, "the scene seeds no row in a tile with rows: the test proves nothing");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A pose update ranks the active set again from the newest view's corrected camera.
#[tokio::test]
async fn a_pose_update_ranks_the_active_set_again() {
    let dir = tmp("active-poses");
    let config = GuideConfig { move_anchored_splats: false, ..active_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 2.0), Vec3::new(-0.3, 0.0, 0.0)).await;
    assert_eq!(active(&live), vec![key(-1, 2), key(-1, 0)]);
    let (h, _) = crate::live::area::tests::keyframe_toward(0, Vec3::new(0.7, 0.0, 2.0), Vec3::new(0.7, 0.0, 0.0));
    live.update_poses(&[crate::live::ViewPose::from_arkit(0, &h.pose).unwrap()]);
    assert_eq!(active(&live), vec![key(0, 2), key(0, 0)]);
    std::fs::remove_dir_all(dir).ok();
}

/// Parking and restoring work as before: the walker leaves the origin (it parks) and comes back (it is restored and
/// active again).
#[tokio::test]
async fn parked_tiles_come_back_and_become_active() {
    let (mut live, dir) = parked_origin_with("active-restore", active_config(2.0)).await;
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::new(0.3, 0.0, 0.0)).await;
    assert!(live.area_report().unwrap().counters.restores >= 1);
    assert!(active(&live).contains(&key(0, 0)), "{:?}", active(&live));
    std::fs::remove_dir_all(dir).ok();
}

/// Area training off: nothing is active, held or reported, and every view trains.
#[tokio::test]
async fn area_off_is_unchanged() {
    for enable in [true, false] {
        let dir = tmp(&format!("active-off-{enable}"));
        let radius = if enable { 0.0 } else { 2.0 };
        let mut live = LiveModel::new(active_config(radius), device().await);
        if enable {
            live.enable_area(dir.join("tiles"));
        }
        add(&mut live, &dir, 0, Vec3::new(-0.3, 0.0, 2.0), Vec3::new(-0.3, 0.0, 0.0)).await;
        add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 6).await;
        assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
        assert!(live.area_report().is_none());
        assert_eq!(live.training_views(), vec![0, 1]);
        std::fs::remove_dir_all(dir).ok();
    }
}
