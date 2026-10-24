//! With `area_freeze` inactive only the active tiles (the tiles near the newest keyframe and those it
//! looks at) train, the training views are the keyframes that see one, and keyframes seed only into active tiles and
//! new ground. The scenes put the camera 5 m from a wall at z = 0 to 1 that spans 1 m tiles x = -4 to 2, so no wall
//! tile is near (2.5 m) and the two ranked tiles decide.

use super::*;
use crate::area::TileKey;
use crate::config::{AreaFreeze, GuideConfig};
use crate::live::area::tests::{active_config, add, device, tmp};
use crate::live::area_freeze::tests::{by_row, rows_where};
use crate::live::area_restore::tests::{parked_origin_with, train};
use glam::Vec3;
use std::collections::BTreeSet;

fn tile_of(p: Vec3) -> TileKey {
    TileKey::of(p, 1.0)
}

/// The rows without their Mip floor.
fn without_floor(mut rows: brush_train::rows::SplatRows) -> brush_train::rows::SplatRows {
    rows.min_scale.clear();
    rows
}

fn active(live: &LiveModel) -> Vec<TileKey> {
    live.area.as_ref().unwrap().active.tiles.clone()
}

/// The active wall tiles (z = 0).
fn active_wall(live: &LiveModel) -> BTreeSet<TileKey> {
    active(live).into_iter().filter(|k| k.z == 0).collect()
}

/// Keyframe looking straight at the wall from 5 m at x.
async fn facing_wall(live: &mut LiveModel, dir: &std::path::Path, id: u64, x: f32) {
    add(live, dir, id, Vec3::new(x, 0.0, 5.0), Vec3::new(x, 0.0, 0.0)).await;
}

/// Per row, whether the trainer holds it.
async fn held_flags(live: &LiveModel) -> Vec<bool> {
    let n = live.splats().unwrap().num_splats() as usize;
    match live.trainer.as_ref().unwrap().frozen_rows() {
        Some(m) => m.clone().int().into_data_async().await.unwrap().convert::<i32>().try_into_vec::<i32>().unwrap()
            .into_iter().map(|v| v != 0).collect(),
        None => vec![false; n],
    }
}

/// Two wall tiles train, the other wall tiles keep every parameter and their floor bit for bit across steps and
/// refines; the report counts the held rows, the active rows, and the held rows no step ever trained.
#[tokio::test]
async fn only_the_active_tiles_train() {
    let dir = tmp("active-train");
    let mut live = LiveModel::new(active_config(2.0), device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, -0.3).await;
    let wall = active_wall(&live);
    assert_eq!(wall.len(), 2, "{:?}", active(&live));
    let inactive = |p: Vec3| !wall.contains(&tile_of(p));
    let trained = |p: Vec3| wall.contains(&tile_of(p));
    let (before, before_active) = (rows_where(&live, inactive).await, rows_where(&live, trained).await);
    assert!(!before.is_empty() && !before_active.is_empty());
    let r = live.area_report().unwrap();
    assert_eq!(r.frozen_rows as usize, before.len(), "{r:?}");
    assert_eq!(r.active_splats as usize, before_active.len(), "{r:?}");
    assert_eq!(r.held_untrained_rows, r.frozen_rows, "no step has run: every held row is untrained");
    // The first refine gives the first keyframe's rows their first Mip floor (they had none); their raw parameters
    // stay. From then on every held row is still there, floor and all, bit for bit (a split child of an active row
    // may have moved in).
    let held_from_start = before.len();
    train(&mut live, 5).await;
    let floored = rows_where(&live, inactive).await;
    let after = by_row(&without_floor(floored.clone()));
    assert!(by_row(&without_floor(before)).iter().all(|row| after.contains(row)), "an inactive row trained");
    let floored = by_row(&floored);
    train(&mut live, 10).await;
    let after = by_row(&rows_where(&live, inactive).await);
    assert!(floored.iter().all(|row| after.contains(row)), "an inactive row changed");
    assert_ne!(by_row(&rows_where(&live, trained).await), by_row(&before_active), "the active rows did not train");
    let r = live.area_report().unwrap();
    // Rows of active tiles that moved into inactive ones are held too, but they trained.
    assert!(r.frozen_rows as usize >= floored.len() && r.held_untrained_rows as usize == held_from_start, "{r:?}");
    std::fs::remove_dir_all(dir).ok();
}

/// The newest keyframe faces the other end of the wall: its tiles become active and train, the tiles active before
/// are held from then on. The 10 m radius keeps every tile resident: at 2 m the far end of the wall parks whenever no
/// view the loaders drew in a refine window saw it, which takes its held rows out of the trainer.
#[tokio::test]
async fn the_mask_follows_the_active_set() {
    let dir = tmp("active-follow");
    let mut live = LiveModel::new(active_config(10.0), device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, -2.3).await;
    let first = active_wall(&live);
    train(&mut live, 5).await;
    facing_wall(&mut live, &dir, 1, 1.7).await;
    let second = active_wall(&live);
    let (gone, new): (BTreeSet<TileKey>, BTreeSet<TileKey>) =
        (first.difference(&second).copied().collect(), second.difference(&first).copied().collect());
    assert!(!gone.is_empty() && !new.is_empty(), "{first:?} then {second:?}");
    let was_active = |p: Vec3| gone.contains(&tile_of(p));
    let now_active = |p: Vec3| new.contains(&tile_of(p));
    let (held, trains) = (rows_where(&live, was_active).await, rows_where(&live, now_active).await);
    assert!(!held.is_empty() && !trains.is_empty());
    train(&mut live, 10).await;
    let after = by_row(&rows_where(&live, was_active).await);
    assert!(by_row(&held).iter().all(|row| after.contains(row)), "a row of a tile no longer active changed");
    assert_ne!(by_row(&rows_where(&live, now_active).await), by_row(&trains), "the newly active tiles did not train");
    let c = live.area_report().unwrap().counters;
    assert!(c.active_changes >= 2 && c.active_reversals == 0 && c.parks == 0, "{c:?}");
    std::fs::remove_dir_all(dir).ok();
}

/// Keyframes 0 (faces the wall), 1 (same spot, turned around), 2 (8 m along, faces its own wall), 3 (14 m back,
/// beyond the 8 m view) are followed by 4, which faces the wall again: the training views are 0 and 4. With
/// `area_freeze` left or none the views within the 2 m radius train (0, 1 and 4) and nothing is active.
#[tokio::test]
async fn the_training_views_see_an_active_tile() {
    for freeze in [AreaFreeze::Inactive, AreaFreeze::Left, AreaFreeze::Off] {
        let dir = tmp(&format!("active-views-{freeze:?}"));
        let mut live = LiveModel::new({ let mut c = active_config(2.0); c.area.freeze = freeze; c }, device().await);
        live.enable_area(dir.join("tiles"));
        facing_wall(&mut live, &dir, 0, 0.0).await;
        add(&mut live, &dir, 1, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 10.0)).await;
        facing_wall(&mut live, &dir, 2, 8.0).await;
        add(&mut live, &dir, 3, Vec3::new(0.1, 0.0, 14.0), Vec3::ZERO).await;
        facing_wall(&mut live, &dir, 4, 0.2).await;
        if freeze == AreaFreeze::Inactive {
            assert_eq!(live.training_views(), vec![0, 4]);
            assert_eq!(active_wall(&live).len(), 2, "{:?}", active(&live));
        } else {
            assert_eq!(live.training_views(), vec![0, 1, 4], "{freeze:?}");
            assert!(active(&live).is_empty());
            assert!(live.area_report().unwrap().active_tiles.is_empty());
        }
        for _ in 0..16 {
            for loader in [live.recent.as_mut().unwrap(), live.all.as_mut().unwrap()] {
                // Views 0, 1 and 4 stand at z = 5 with x 0 or 0.2; views 2 and 3 do not.
                let p = loader.next_batch().await.camera.position;
                assert!(p.x < 0.3 && p.z < 6.0, "{freeze:?} served {p}");
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
    facing_wall(&mut live, &dir, 0, 0.0).await;
    add(&mut live, &dir, 1, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 8.0)).await;
    assert_eq!(live.training_views(), vec![1], "keyframe 0 does not see the wall keyframe 1 seeded behind it");
    live.area.as_mut().unwrap().active.boxes.clear();
    live.refresh_area_views();
    assert_eq!(live.training_views(), vec![0, 1]);
    std::fs::remove_dir_all(dir).ok();
}

/// Keyframe 1 faces the wall's right end and the ground beyond it with one ranked tile: its seeds in tiles that hold
/// rows outside the active set are dropped, those on new ground land. With `area_freeze` left seeds land
/// in tiles with rows too, so the scene does seed there.
#[tokio::test]
async fn seeds_outside_the_active_tiles_are_dropped_but_new_ground_seeds() {
    for freeze in [AreaFreeze::Inactive, AreaFreeze::Left] {
        let dir = tmp(&format!("active-seeds-{freeze:?}"));
        let config = { let mut c = active_config(3.0); c.area.freeze = freeze; c.area.active_tiles = 1; c };
        let mut live = LiveModel::new(config, device().await);
        live.enable_area(dir.join("tiles"));
        facing_wall(&mut live, &dir, 0, 0.0).await;
        train(&mut live, 5).await;
        let known: Vec<TileKey> = live.area.as_ref().unwrap().registry.resident_keys();
        facing_wall(&mut live, &dir, 1, 4.5).await;
        let active = active(&live);
        let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
        let kf1: Vec<TileKey> =
            (0..rows.len()).filter(|&i| rows.origin[i] == Some(1)).map(|i| tile_of(rows.mean(i))).collect();
        let on_known = kf1.iter().filter(|k| known.contains(k) && !active.contains(k)).count();
        assert!(kf1.iter().any(|k| !known.contains(k)), "{freeze:?}: no seed on new ground");
        if freeze == AreaFreeze::Inactive {
            assert_eq!(on_known, 0, "a seed landed in an inactive tile with rows");
        } else {
            assert!(on_known > 0, "the scene seeds no row in a tile with rows: the test proves nothing");
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A pose update ranks the active set again from the newest view's corrected camera, and the next training step
/// holds the rows outside the new set.
#[tokio::test]
async fn a_pose_update_ranks_the_active_set_again() {
    let dir = tmp("active-poses");
    let config = GuideConfig { move_anchored_splats: false, ..active_config(2.0) };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, -2.3).await;
    let before = active_wall(&live);
    let (h, _) = crate::live::area::tests::keyframe_toward(0, Vec3::new(1.7, 0.0, 5.0), Vec3::new(1.7, 0.0, 0.0));
    live.update_poses(&[crate::live::ViewPose::from_arkit(0, &h.pose).unwrap()]);
    let after = active_wall(&live);
    assert!(before != after, "{before:?} stayed");
    assert!(live.area.as_ref().unwrap().mask.stale);
    // The rows as the step finds them (a step may move an active row across a border).
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    train(&mut live, 1).await;
    assert!(!live.area.as_ref().unwrap().mask.stale);
    let held = held_flags(&live).await;
    let now = active(&live);
    assert!((0..rows.len()).all(|i| held[i] != now.contains(&tile_of(rows.mean(i)))), "the mask is the old set's");
    std::fs::remove_dir_all(dir).ok();
}

/// A keyframe's distance restore brings back a tile that is not active while the active set stays: the restored
/// rows are held at the end of that keyframe (the restore marks the mask stale).
#[tokio::test]
async fn a_restored_tile_outside_the_active_set_is_held_at_once() {
    let dir = tmp("active-restore-held");
    let mut live = LiveModel::new(active_config(10.0), device().await);
    live.enable_area(dir.join("tiles"));
    facing_wall(&mut live, &dir, 0, -0.3).await;
    train(&mut live, 5).await;
    let set = active(&live);
    let wall = active_wall(&live);
    let x = live.area.as_ref().unwrap().registry.resident_keys().into_iter().find(|k| k.z == 0 && !wall.contains(k));
    let x = x.expect("a wall tile outside the active set");
    let splats = live.splats.take().unwrap();
    let splats = live.park_tiles(splats, &[x]).await;
    live.splats = Some(splats);
    assert_eq!(live.area_report().unwrap().counters.parks, 1);
    // The same view again, seeding nothing: only the restore appends rows.
    live.config.seeding.alpha_threshold = -1.0;
    facing_wall(&mut live, &dir, 1, -0.3).await;
    assert_eq!(live.area_report().unwrap().counters.restores, 1);
    assert_eq!(active(&live), set, "the active set changed");
    let held = held_flags(&live).await;
    let rows = live.trainer.as_ref().unwrap().read_rows(live.splats.as_ref().unwrap()).await;
    let in_x: Vec<usize> = (0..rows.len()).filter(|&i| tile_of(rows.mean(i)) == x).collect();
    assert!(!in_x.is_empty() && in_x.iter().all(|&i| held[i]), "a restored row of {x:?} trains");
    std::fs::remove_dir_all(dir).ok();
}

/// Parking and restoring work with active tiles: the walker leaves the origin (it parks) and comes back (it is restored and,
/// near the camera, active again).
#[tokio::test]
async fn parked_tiles_come_back_and_become_active() {
    let (mut live, dir) = parked_origin_with("active-restore", active_config(2.0)).await;
    add(&mut live, &dir, 2, Vec3::new(0.3, 0.0, 2.0), Vec3::new(0.3, 0.0, 0.0)).await;
    assert!(live.area_report().unwrap().counters.restores >= 1);
    assert!(active(&live).contains(&TileKey { x: 0, z: 0 }), "{:?}", active(&live));
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
        facing_wall(&mut live, &dir, 0, -0.3).await;
        add(&mut live, &dir, 1, Vec3::new(6.0, 0.0, 2.0), Vec3::ZERO).await;
        train(&mut live, 6).await;
        assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none());
        assert!(live.area_report().is_none());
        assert_eq!(live.training_views(), vec![0, 1]);
        std::fs::remove_dir_all(dir).ok();
    }
}

/// The scene of [`the_training_views_see_an_active_tile`]: keyframes 0 and 4 face the wall, 1 turns away, 2 faces its
/// own wall 8 m along, 3 stands 14 m back.
async fn five_views(live: &mut LiveModel, dir: &std::path::Path) {
    facing_wall(live, dir, 0, 0.0).await;
    add(live, dir, 1, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 10.0)).await;
    facing_wall(live, dir, 2, 8.0).await;
    add(live, dir, 3, Vec3::new(0.1, 0.0, 14.0), Vec3::ZERO).await;
    facing_wall(live, dir, 4, 0.2).await;
}

/// With inactive_at_cap below the cap the active set is ranked as with inactive, yet every view trains with the
/// loader members of area training off, and nothing is held.
#[tokio::test]
async fn below_the_cap_every_view_trains_as_with_area_off() {
    let mut members = Vec::new();
    for radius in [2.0, 0.0] {
        let dir = tmp(&format!("at-cap-views-{radius}"));
        let config = { let mut c = active_config(radius); c.area.freeze = AreaFreeze::InactiveAtCap; c };
        let mut live = LiveModel::new(config, device().await);
        live.enable_area(dir.join("tiles"));
        five_views(&mut live, &dir).await;
        train(&mut live, 6).await;
        assert!(live.trainer.as_ref().unwrap().frozen_rows().is_none(), "{radius}: rows held below the cap");
        if radius > 0.0 {
            assert_eq!(active_wall(&live).len(), 2, "{:?}", active(&live));
            let r = live.area_report().unwrap();
            assert!(!r.at_cap && r.frozen_rows == 0, "{r:?}");
        }
        let every: Vec<usize> = (0..5).collect();
        assert_eq!((live.training_views(), live.all_members.clone()), (every.clone(), every), "{radius}");
        members.push((live.recent_members.clone(), live.pending_views(), live.all_loader_builds));
        std::fs::remove_dir_all(dir).ok();
    }
    assert_eq!(members[0], members[1], "area on below the cap differs from area off");
}

/// inactive_at_cap: from 0.9 of `max_splats` on the views that see an active tile train (as with inactive) and the
/// rows outside the active tiles are held; between 0.85 and 0.9 that stays, below 0.85 every view trains again. Each
/// flip rebuilds the all-views loader once, the steps between flips not at all.
#[tokio::test]
async fn at_the_cap_the_active_views_train_and_each_flip_rebuilds_the_loaders_once() {
    let dir = tmp("at-cap-flip");
    let config = {
        let mut c = active_config(10.0);
        c.area.freeze = AreaFreeze::InactiveAtCap;
        c.area.seen_window_steps = 1000;
        c
    };
    let mut live = LiveModel::new(config, device().await);
    live.enable_area(dir.join("tiles"));
    five_views(&mut live, &dir).await;
    let every: Vec<usize> = (0..5).collect();
    assert_eq!(live.training_views(), every);
    let step_at = async |live: &mut LiveModel, share: f32| {
        let n = live.splats().unwrap().num_splats() as f32;
        live.config.max_splats = (n / share).floor() as u32;
        let builds = live.all_loader_builds;
        train(live, 1).await;
        let held = live.trainer.as_ref().unwrap().frozen_rows().is_some();
        assert_eq!(held, live.area_report().unwrap().at_cap, "{share}");
        assert_eq!(live.all_members, live.training_views(), "{share}");
        (live.training_views(), held, live.all_loader_builds - builds)
    };
    assert_eq!(step_at(&mut live, 0.88).await, (every.clone(), false, 0));
    assert_eq!(step_at(&mut live, 0.9).await, (vec![0, 4], true, 1));
    assert_eq!(step_at(&mut live, 0.86).await, (vec![0, 4], true, 0));
    assert_eq!(step_at(&mut live, 0.84).await, (every, false, 1));
    // A keyframe that reaches the cap switches before its view joins: the loaders are rebuilt once, with it.
    let n = live.splats().unwrap().num_splats() as f32;
    live.config.max_splats = (n / 0.9).floor() as u32;
    let builds = live.all_loader_builds;
    facing_wall(&mut live, &dir, 5, 0.1).await;
    assert!(live.area_report().unwrap().at_cap);
    let views = live.training_views();
    assert_eq!((views.first(), views.last()), (Some(&0), Some(&5)), "{views:?}");
    assert!(views.len() < 6 && live.all_members == views, "{views:?} {:?}", live.all_members);
    assert!(live.recent_members.iter().chain(&live.pending_views()).all(|v| views.contains(v)));
    assert_eq!(live.all_loader_builds - builds, 1);
    assert_eq!(live.area_report().unwrap().counters.parks, 0);
    std::fs::remove_dir_all(dir).ok();
}
