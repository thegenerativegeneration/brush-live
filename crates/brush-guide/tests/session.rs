mod test_scene;

use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ServerHeader, decode_cells, decode_frame};
use brush_guide::session::GuideSession;
use glam::Vec3;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scores_arrive_after_keyframes() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let mut scores = session.scores();

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

    session.reset().await;
    assert_eq!(session.status().borrow().num_keyframes, 0);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_keyframe_reports_error_and_session_survives() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-bad-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let (mut h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    h.pose[0] = f32::NAN;
    assert!(session.push_keyframe(h, p).await.is_err());
    let (h, p) = test_scene::keyframe(1, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h, p).await.unwrap();
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resent_id_is_acked_without_touching_the_stored_image() {
    let device = test_scene::device().await.autodiff();
    let dir = std::env::temp_dir().join(format!("brush-guide-session-dup-{}", std::process::id()));
    let session = GuideSession::start(GuideConfig::default(), device, dir.clone());
    let (h, p) = test_scene::keyframe(0, Vec3::new(0.0, 0.0, 2.0));
    session.push_keyframe(h.clone(), p).await.unwrap();
    let stored = std::fs::read(dir.join("images/0.jpg")).unwrap();

    // Same id, undecodable content: recognised as a resend before decoding.
    let garbage = vec![0u8; h.jpeg_len as usize];
    let h = brush_guide::protocol::KeyframeHeader { num_points: 0, ..h };
    session.push_keyframe(h, garbage).await.unwrap();
    assert_eq!(std::fs::read(dir.join("images/0.jpg")).unwrap(), stored);
    assert_eq!(session.status().borrow().num_keyframes, 1);
    std::fs::remove_dir_all(dir).ok();
}
