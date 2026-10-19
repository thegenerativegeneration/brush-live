use super::*;
use crate::live::area::tests::{add, area_config, device, tmp};
use crate::live::area_restore::tests::{drift_rows, parked_origin, train};
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

/// A resident row behind every floor camera (in a tile no view sees) still gets a finite, positive floor at refine.
#[tokio::test]
async fn a_row_no_floor_camera_sees_keeps_a_finite_floor() {
    let (mut live, dir) = parked_origin("sight-finite").await;
    drift_rows(&mut live, vec![20.5, 0.0, 3.5, 20.6, 0.0, 3.5]).await;
    train(&mut live, 5).await;
    assert_eq!(floor_count(&live), 1);
    let splats = live.splats().unwrap();
    let means: Vec<f32> = splats.means().into_data_async().await.unwrap().try_to_vec().unwrap();
    assert!(means.chunks(3).any(|p| p[2] > 3.0), "the drifted rows stay resident");
    let floor: Vec<f32> = splats.min_scale.clone().unwrap().into_data_async().await.unwrap().try_to_vec().unwrap();
    assert!(floor.iter().all(|f| f.is_finite() && *f > 0.0), "{floor:?}");
    std::fs::remove_dir_all(dir).ok();
}
