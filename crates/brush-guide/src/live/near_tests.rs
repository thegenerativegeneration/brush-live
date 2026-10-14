//! `train_views_radius_m` on a `LiveModel`: which views count as near, the all loader, and the near cache's budget.

use super::poses::tests::{keyframe, pose_at};
use super::{LiveModel, ViewPose};
use crate::config::GuideConfig;
use crate::keyframe::decode_keyframe;
use glam::Vec3;

/// Packed bytes of one 16x12 test keyframe (RGBA8).
const VIEW_BYTES: u64 = 16 * 12 * 4;

async fn model(radius: f32, cache_bytes: u64, xs: &[f32], dir: &std::path::Path) -> LiveModel {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let config = GuideConfig {
        train_views_radius_m: radius,
        loader_cache_bytes: cache_bytes,
        ..GuideConfig::default()
    };
    let mut live = LiveModel::new(config, device);
    for (i, x) in xs.iter().enumerate() {
        add(&mut live, i as u64, *x, dir).await;
    }
    live
}

async fn add(live: &mut LiveModel, id: u64, x: f32, dir: &std::path::Path) {
    let (h, p) = keyframe(id, Vec3::new(x, 0.0, 2.0));
    live.add_keyframe(decode_keyframe(&h, &p, dir).await.unwrap())
        .await;
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("live-near-{name}-{}", std::process::id()))
}

/// Membership follows the views' current cameras: a pose update moves a view into or out of the radius.
#[tokio::test]
async fn pose_updates_change_near_membership() {
    let dir = tmp("poses");
    let live_xs = [0.0f32, 0.5, 1.0];
    let mut live = model(0.8, 1 << 30, &live_xs, &dir).await;
    assert_eq!(live.near_view_indices(), Some(vec![1, 2]));
    live.update_poses(&[ViewPose::from_arkit(0, &pose_at(Vec3::new(0.6, 0.0, 2.0))).unwrap()]);
    assert_eq!(live.near_view_indices(), Some(vec![0, 1, 2]));
    live.update_poses(&[ViewPose::from_arkit(1, &pose_at(Vec3::new(5.0, 0.0, 2.0))).unwrap()]);
    assert_eq!(live.near_view_indices(), Some(vec![0, 2]));
    // The newest view is the walker: a new keyframe far away leaves only itself near.
    add(&mut live, 3, 20.0, &dir).await;
    assert_eq!(live.near_view_indices(), Some(vec![3]));
    std::fs::remove_dir_all(dir).ok();
}

/// With the radius on there is no all-views loader, steps report the near count, and the near cache keeps within
/// its budget, giving up views the walker left for near ones.
#[tokio::test]
async fn near_cache_stays_in_budget_and_follows_the_walker() {
    let dir = tmp("cache");
    let mut live = model(0.8, 2 * VIEW_BYTES, &[0.0, 0.3, 0.6], &dir).await;
    assert!(
        live.all.is_none(),
        "all-views loader built with the radius on"
    );
    for _ in 0..60 {
        live.train_step().await;
    }
    assert_eq!(live.near_views_at_last_draw(), Some(3));
    assert!(
        live.near_cache.bytes() <= 2 * VIEW_BYTES,
        "{}",
        live.near_cache.bytes()
    );
    assert_eq!(live.near_cache.cached_views().len(), 2);
    add(&mut live, 3, 10.0, &dir).await;
    add(&mut live, 4, 10.3, &dir).await;
    for _ in 0..60 {
        live.train_step().await;
    }
    assert_eq!(live.near_views_at_last_draw(), Some(2));
    assert_eq!(live.near_cache.cached_views(), vec![3, 4]);
    assert!(live.near_cache.bytes() <= 2 * VIEW_BYTES);
    assert!(
        live.all.is_none(),
        "all-views loader built on a later keyframe"
    );
    std::fs::remove_dir_all(dir).ok();
}

/// A miss drops the cached views the walker left, though the budget has room for them.
#[tokio::test]
async fn near_cache_drops_views_the_walker_left() {
    let dir = tmp("left");
    let mut live = model(0.8, 10 * VIEW_BYTES, &[0.0, 0.3, 0.6], &dir).await;
    for _ in 0..60 {
        live.train_step().await;
    }
    assert_eq!(live.near_cache.cached_views(), vec![0, 1, 2]);
    add(&mut live, 3, 10.0, &dir).await;
    for _ in 0..30 {
        live.train_step().await;
    }
    assert_eq!(live.near_cache.cached_views(), vec![3]);
    assert_eq!(live.near_cache.bytes(), VIEW_BYTES);
    std::fs::remove_dir_all(dir).ok();
}

/// With the radius off nothing changes: the all loader exists, no near count is reported, the near cache stays empty.
#[tokio::test]
async fn radius_off_keeps_the_all_loader() {
    let dir = tmp("off");
    let mut live = model(0.0, 1 << 30, &[0.0, 0.3, 0.6], &dir).await;
    for _ in 0..20 {
        live.train_step().await;
    }
    assert!(live.all.is_some());
    assert_eq!(live.near_views_at_last_draw(), None);
    assert_eq!(live.near_view_indices(), None);
    assert_eq!(live.near_cache.bytes(), 0);
    std::fs::remove_dir_all(dir).ok();
}
