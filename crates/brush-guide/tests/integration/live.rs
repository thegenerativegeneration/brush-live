use crate::test_scene;

use brush_guide::config::GuideConfig;
use brush_guide::keyframe::decode_keyframe;
use brush_guide::live::LiveModel;
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
