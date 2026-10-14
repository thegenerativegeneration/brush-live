use crate::test_scene;

use brush_guide::config::GuideConfig;
use brush_guide::keyframe::decode_keyframe;
use brush_guide::live::{LiveModel, PoseUpdateStats, ViewPose};
use glam::Vec3;

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("brush-guide-live-{name}-{}", std::process::id()))
}

#[tokio::test]
async fn keyframe_without_points_still_initialises() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("nopoints");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    let (mut h, p) = test_scene::keyframe(1, Vec3::new(0.0, 0.0, 2.0));
    h.num_points = 0;
    let p = p[..h.jpeg_len as usize].to_vec();
    assert!(
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await
    );
    assert!(
        live.splats().unwrap().num_splats() > 0,
        "random init fallback"
    );
    live.train_step().await;
    std::fs::remove_dir_all(dir).ok();
}

/// Keyframes initialise the model from nothing, grow it as new views add splats, and training continues.
#[tokio::test]
async fn keyframes_initialise_grow_and_train() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("grow");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    live.train_step().await; // no views: no-op, no panic
    assert!(live.splats().is_none());

    for (i, a) in [0.0f32, 1.2, 2.4].iter().enumerate() {
        let (h, p) = test_scene::keyframe(i as u64, Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()));
        let first = live
            .add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await;
        if i == 0 {
            assert!(first);
            assert!(live.splats().unwrap().num_splats() > 0);
        }
        for _ in 0..10 {
            live.train_step().await;
        }
    }
    assert_eq!(live.iter(), 30);
    assert_eq!(live.views().len(), 3);
    assert_eq!(live.last_keyframe_id(), Some(2));
    assert!(live.splats().unwrap().num_splats() > 0);
    std::fs::remove_dir_all(dir).ok();
}

/// The last step's mean LR is the base LR times its position scale, the bounds' median size.
#[tokio::test]
async fn last_step_lr_is_scaled_by_the_bounds_median_size() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("steplr");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    assert_eq!(live.last_step_lr(), (0.0, 0.0));
    let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
        .await;
    live.train_step().await;
    let (lr, scale) = live.last_step_lr();
    let base = brush_train::config::TrainConfig::default().lr_mean;
    assert!(scale > 0.0, "position scale {scale}");
    assert!((lr - base * f64::from(scale)).abs() < base * 1e-6, "lr_mean {lr}");
    std::fs::remove_dir_all(dir).ok();
}

/// `train_radius_m` reaches the trainer: a step with the radius on reports its masked count, off reports none.
#[tokio::test]
async fn train_radius_reaches_the_trainer() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("radius");
    for (radius, want_some) in [(0.0, false), (0.01, true)] {
        let config = GuideConfig {
            train_radius_m: radius,
            ..GuideConfig::default()
        };
        let mut live = LiveModel::new(config, device.clone());
        let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await;
        live.train_step().await;
        let masked = live.masked_by_radius().await;
        assert_eq!(masked.is_some(), want_some, "radius {radius}: {masked:?}");
        if let Some(n) = masked {
            assert!(n > 0, "radius {radius}: nothing beyond 1 cm of the camera");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn seeding_respects_max_splats() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("cap");
    let config = GuideConfig {
        max_splats: 20,
        ..GuideConfig::default()
    };
    let mut live = LiveModel::new(config, device);
    for (i, a) in [0.0f32, 1.2, 2.4].iter().enumerate() {
        let (h, p) = test_scene::keyframe(i as u64, Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()));
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await;
        let n = live.splats().unwrap().num_splats();
        assert!((1..=20).contains(&n), "keyframe {i}: {n} splats");
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A corrected pose replaces the view's camera, intrinsics kept; unknown and unchanged ids are counted, not applied;
/// training goes on afterwards.
#[tokio::test]
async fn poses_for_unknown_ids_are_counted_not_applied() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("poses");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    let positions = [
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 2.0),
        Vec3::new(-1.0, 0.0, 2.0),
    ];
    for (i, pos) in positions.iter().enumerate() {
        let (h, p) = test_scene::keyframe(i as u64, *pos);
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await;
    }
    let fov = live.views()[1].camera.fov_x;
    let moved = Vec3::new(1.2, 0.1, 2.0);
    let stats = live.update_poses(&[
        ViewPose::from_arkit(1, &test_scene::arkit_pose_at(moved)).unwrap(),
        ViewPose::from_arkit(2, &test_scene::arkit_pose_at(positions[2])).unwrap(),
        ViewPose::from_arkit(99, &test_scene::arkit_pose_at(moved)).unwrap(),
    ]);
    assert_eq!(
        stats,
        PoseUpdateStats {
            moved: 1,
            unchanged: 1,
            unknown: 1
        }
    );
    assert!((live.views()[1].camera.position - moved).length() < 1e-5);
    assert_eq!(live.views()[1].camera.fov_x, fov);
    assert!((live.views()[0].camera.position - positions[0]).length() < 1e-5);
    for _ in 0..5 {
        live.train_step().await;
    }
    assert_eq!(live.iter(), 5);
    std::fs::remove_dir_all(dir).ok();
}

/// The loaders cache decoded views and report the bytes they hold.
#[tokio::test]
async fn loader_cache_bytes_follow_the_cached_views() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("cache-bytes");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    assert_eq!(live.loader_cache_bytes(), 0);
    for (i, a) in [0.0f32, 1.2, 2.4].iter().enumerate() {
        let (h, p) = test_scene::keyframe(i as u64, Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()));
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap()).await;
    }
    for _ in 0..40 {
        live.train_step().await;
    }
    let bytes = live.loader_cache_bytes();
    // 64x48 RGBA8 per cached view, at most 3 views in each of the two loaders.
    assert!(bytes > 0 && bytes <= 2 * 3 * 64 * 48 * 4, "{bytes}");
    std::fs::remove_dir_all(dir).ok();
}
