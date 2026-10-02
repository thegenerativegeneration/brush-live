use crate::test_scene;

use brush_guide::config::GuideConfig;
use brush_guide::protocol::{
    MeshBrick, ServerHeader, decode_cells, decode_frame, decode_mesh_bricks,
};
use brush_guide::session::GuideSession;
use glam::Vec3;
use std::time::Duration;

/// Score sets decode, mesh bricks follow with a version at least the score set's, and colours cover every vertex.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scores_arrive_after_keyframes_with_mesh_bricks_following() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let mut scores = session.scores();
    let meshes = session.meshes();

    for (i, a) in [0.0f32, 1.0, 2.0].iter().enumerate() {
        let (h, p) = test_scene::keyframe(i as u64, Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()));
        session.push_keyframe(h, p).await.unwrap();
    }
    // Duplicate resend is accepted without error.
    let (h, p) = test_scene::keyframe(2, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h, p).await.unwrap();

    let set = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            scores.changed().await.unwrap();
            if let Some(s) = scores.borrow().clone() {
                return s;
            }
        }
    })
    .await
    .expect("a ScoreSet within 60 s");

    let frame = set.to_frame();
    let (header, payload): (ServerHeader, &[u8]) = decode_frame(&frame).unwrap();
    let ServerHeader::ScoreSet {
        num_cells,
        voxel_size,
        ..
    } = header
    else {
        panic!("{header:?}")
    };
    assert_eq!(decode_cells(payload).unwrap().len(), num_cells as usize);
    assert!((voxel_size - 0.1).abs() < 1e-6);

    assert_eq!(session.status().borrow().num_keyframes, 3);
    let ply = session.export_splat().await.unwrap();
    assert!(ply.starts_with(b"ply"));

    let msg = meshes
        .borrow()
        .since(0)
        .expect("mesh bricks recorded with the score set");
    // Rounds after the score set may already be recorded: the version is
    // at least the score set's.
    assert!(
        msg.version >= set.version,
        "{} < {}",
        msg.version,
        set.version
    );
    assert!(!msg.bricks.is_empty(), "at least one brick");

    let frame = msg.to_frame();
    let (header, payload): (ServerHeader, &[u8]) = decode_frame(&frame).unwrap();
    let ServerHeader::MeshBricks {
        version,
        num_bricks,
        ..
    } = header
    else {
        panic!("{header:?}")
    };
    assert_eq!(version, msg.version);
    let decoded = decode_mesh_bricks(payload, num_bricks).unwrap();
    assert_eq!(decoded.len(), msg.bricks.len());
    for brick in &decoded {
        if let MeshBrick::Mesh(m) = brick {
            assert_eq!(m.colours.len(), m.positions.len(), "brick {:?}", m.key);
        }
    }

    session.reset().await;
    assert_eq!(session.status().borrow().num_keyframes, 0);
    assert!(session.meshes().borrow().since(0).is_none());
    std::fs::remove_dir_all(dir).ok();
}

/// A NaN pose is rejected without killing the session, and a resend with undecodable content is recognised as a
/// resend before decoding (the stored image is untouched).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_keyframe_reports_error_and_a_resend_is_acked_without_touching_the_stored_image() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-bad-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let (mut h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    h.pose[0] = f32::NAN;
    assert!(session.push_keyframe(h, p).await.is_err());
    let (h, p) = test_scene::keyframe(1, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h.clone(), p).await.unwrap();
    let stored = std::fs::read(dir.join("images/1.jpg")).unwrap();

    let garbage = vec![0u8; h.jpeg_len as usize];
    let h = brush_guide::protocol::KeyframeHeader { num_points: 0, ..h };
    session.push_keyframe(h, garbage).await.unwrap();
    assert_eq!(std::fs::read(dir.join("images/1.jpg")).unwrap(), stored);
    assert_eq!(session.status().borrow().num_keyframes, 1);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finish_writes_ply_and_pauses_training_until_next_keyframe() {
    let device = test_scene::device().await.autodiff();
    let dir =
        std::env::temp_dir().join(format!("brush-guide-session-finish-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h, p).await.unwrap();

    let path = dir.join("splat.ply");
    let len = session.finish(&path).await.unwrap();
    assert!(len > 0);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
    assert!(std::fs::read(&path).unwrap().starts_with(b"ply"));

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while session.status().borrow().train_iters_per_s != 0.0 && tokio::time::Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        session.status().borrow().train_iters_per_s,
        0.0,
        "idle after finish"
    );

    let (h, p) = test_scene::keyframe(1, Vec3::new(1.0, 0.0, 2.0));
    session.push_keyframe(h, p).await.unwrap();
    // A scoring round under GPU contention can delay the first published
    // rate by several seconds, so poll against a generous deadline.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while session.status().borrow().train_iters_per_s <= 0.0
        && tokio::time::Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        session.status().borrow().train_iters_per_s > 0.0,
        "training resumed within 20 s"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyframes_queue_until_the_warm_up_is_done() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-ready-{}", std::process::id()));
    let (ready_tx, ready) = tokio::sync::watch::channel(false);
    let session = GuideSession::start_when(GuideConfig::default(), device, dir.clone(), Some(ready));
    let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    let pusher = session.clone();
    let push = tokio::spawn(async move { pusher.push_keyframe(h, p).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!push.is_finished(), "the keyframe waits for the warm-up");
    assert_eq!(session.status().borrow().num_keyframes, 0);
    ready_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(60), push)
        .await
        .expect("acked once warm")
        .unwrap()
        .unwrap();
    assert_eq!(session.status().borrow().num_keyframes, 1);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn warm_up_runs_the_session_kernels_on_dummy_splats() {
    let device = test_scene::device().await.autodiff();
    let config = GuideConfig {
        max_splats: 4096,
        ..GuideConfig::default()
    };
    let secs = brush_guide::warmup::warm_up(&config, &device).await;
    assert!(secs > 0.0 && secs.is_finite());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn paused_session_holds_keyframes_and_stops_training_until_resumed() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-pause-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h, p).await.unwrap();
    // Let training publish a rate once, so a running worker would publish again within the window below.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while session.status().borrow().train_iters == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    session.set_paused(true).await;
    let iters = session.status().borrow().train_iters;
    let (h, p) = test_scene::keyframe(1, Vec3::new(1.0, 0.0, 2.0));
    let pusher = session.clone();
    let push = tokio::spawn(async move { pusher.push_keyframe(h, p).await });
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(!push.is_finished(), "a keyframe waits while paused");
    assert_eq!(session.status().borrow().train_iters, iters, "no training while paused");
    assert_eq!(session.status().borrow().num_keyframes, 1);

    session.set_paused(false).await;
    tokio::time::timeout(Duration::from_secs(10), push)
        .await
        .expect("acked after resume")
        .unwrap()
        .unwrap();
    assert_eq!(session.status().borrow().num_keyframes, 2);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn warm_up_parks_while_paused_and_finishes_once_released() {
    let device = test_scene::device().await.autodiff();
    let config = GuideConfig {
        max_splats: 4096,
        ..GuideConfig::default()
    };
    let (pause, pause_rx) = tokio::sync::watch::channel(true);
    let warmup = brush_guide::warmup::Warmup::spawn_pausable(config, device, pause_rx);
    let (mut parked, mut ready) = (warmup.parked(), warmup.ready());
    tokio::time::timeout(Duration::from_secs(10), parked.wait_for(|p| *p))
        .await
        .expect("parks while paused")
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!*ready.borrow(), "no warm-up work while parked");
    pause.send(false).unwrap();
    tokio::time::timeout(Duration::from_secs(60), ready.wait_for(|r| *r))
        .await
        .expect("done once released")
        .unwrap();
}
