use super::*;
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{correction, drift_rows, parked_origin, train};
use glam::Vec3;

fn floor_count(live: &LiveModel) -> usize {
    live.trainer.as_ref().unwrap().view_cams().len()
}

/// With area training on, the Mip floor and the score round take only the views that see a resident tile: once the
/// origin tiles park, keyframe 0 (which sees only them) drops out of both.
#[tokio::test]
async fn the_floor_and_the_score_round_take_the_views_that_see_resident_tiles() {
    let (mut live, dir) = parked_origin("sight-floor").await;
    assert_eq!(live.views().len(), 2);
    train(&mut live, 5).await;
    assert_eq!(floor_count(&live), 1, "keyframe 0 sees only parked tiles");
    let cams = live.sight_cameras();
    assert_eq!(cams.len(), 1);
    assert!((cams[0].position.x - 20.0).abs() < 1e-3);
    assert_eq!(live.sight_view_count(), Some(1));
    std::fs::remove_dir_all(dir).ok();
}

/// A keyframe added while tiles are parked hands its own camera to the floor of its seeds, besides the views that see
/// resident tiles.
#[tokio::test]
async fn a_new_keyframe_joins_the_floor_cameras() {
    let (mut live, dir) = parked_origin("sight-new").await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 2, Vec3::new(20.5, 0.0, 2.0), Vec3::new(20.5, 0.0, 0.0)).await;
    assert_eq!(floor_count(&live), 2, "keyframes 1 and 2");
    assert_eq!(live.sight_cameras().len(), 2);
    std::fs::remove_dir_all(dir).ok();
}

/// With area training off every view's camera floors the splats and orients the score round, as before.
#[tokio::test]
async fn area_off_takes_every_view() {
    let dir = tmp("sight-off");
    let mut live = LiveModel::new(area_config(0.0), device().await);
    live.enable_area(dir.join("tiles"));
    add(&mut live, &dir, 0, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO).await;
    train(&mut live, 5).await;
    add(&mut live, &dir, 1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    train(&mut live, 10).await;
    assert_eq!(floor_count(&live), 2);
    assert_eq!(live.sight_cameras().len(), 2);
    assert_eq!(live.sight_view_count(), None);
    std::fs::remove_dir_all(dir).ok();
}

/// Rows resident next to keyframe 0's corrected camera, which sees no resident tile: their floor comes from keyframe
/// 1's camera alone (`sqrt(factor) · |p − cam1| / f1`), not from keyframe 0's nearer one.
#[tokio::test]
async fn a_dropped_camera_plays_no_part_in_the_floor() {
    let (mut live, dir) = parked_origin("sight-floor-exact").await;
    // Keyframe 0 moves next to the rows below (1 m from them, 1.6 m from keyframe 1), looking away from every
    // resident tile, and 2.2 m from keyframe 1: outside the area's 2 m radius.
    let near = Vec3::new(21.6, 0.0, 3.5);
    live.update_poses(&[correction(0, near, Vec3::new(40.0, 0.0, 3.5))]);
    drift_rows(&mut live, vec![20.5, 0.0, 3.5, 20.6, 0.0, 3.5]).await;
    // Steps 16 to 20; the refine at 20 floors the splats last.
    train(&mut live, 5).await;
    assert_eq!(live.trainer.as_ref().unwrap().view_cams().len(), 1, "keyframe 0 sees no resident tile");
    let splats = live.splats().unwrap();
    let means: Vec<f32> = splats.means().into_data_async().await.unwrap().try_to_vec().unwrap();
    let floor: Vec<f32> = splats.min_scale.clone().unwrap().into_data_async().await.unwrap().try_to_vec().unwrap();
    let (cam1, f1) = live.view_cams[1];
    let factor = live.train_config.min_scale_factor.sqrt();
    let mut drifted = 0;
    for (p, f) in means.chunks(3).zip(&floor) {
        let p = Vec3::new(p[0], p[1], p[2]);
        if p.z < 3.0 {
            continue;
        }
        drifted += 1;
        let want = factor * p.distance(cam1) / f1;
        assert!(
            (f - want).abs() <= 1e-3 * want,
            "floor {f} at {p}, want {want} (cam 0 would give {})",
            factor * p.distance(near) / f1
        );
    }
    assert!(drifted >= 2, "{drifted} drifted rows");
    std::fs::remove_dir_all(dir).ok();
}

/// The newest view always floors the splats, even when no view sees a resident tile and the area holds no view.
#[tokio::test]
async fn the_newest_view_always_floors() {
    let (mut live, dir) = parked_origin("sight-guard").await;
    live.update_poses(&[correction(1, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 30.0))]);
    live.area.as_mut().unwrap().views.clear();
    live.refresh_sight();
    live.push_floor_cams();
    let cams = live.trainer.as_ref().unwrap().view_cams().to_vec();
    assert_eq!(cams, vec![live.view_cams[1]]);
    std::fs::remove_dir_all(dir).ok();
}

/// Views in one pose bucket vote once in the score round, while every one of them floors the splats.
#[tokio::test]
async fn a_bucket_votes_once_and_floors_with_all_its_views() {
    let (mut live, dir) = parked_origin("sight-buckets").await;
    train(&mut live, 5).await;
    // Keyframe 2 repeats keyframe 1's pose; keyframe 3 stands in the next cell.
    add(&mut live, &dir, 2, Vec3::new(20.0, 0.0, 2.0), Vec3::new(20.0, 0.0, 0.0)).await;
    add(&mut live, &dir, 3, Vec3::new(20.6, 0.0, 2.0), Vec3::new(20.6, 0.0, 0.0)).await;
    assert_eq!(live.trainer.as_ref().unwrap().view_cams().len(), 3, "keyframes 1 to 3");
    assert_eq!(live.sight_view_count(), Some(3));
    let xs: Vec<f32> = live.sight_cameras().iter().map(|c| c.position.x).collect();
    assert_eq!(xs.len(), 2, "keyframe 2 votes through keyframe 1: {xs:?}");
    std::fs::remove_dir_all(dir).ok();
}
