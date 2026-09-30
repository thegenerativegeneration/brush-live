use glam::Vec3;

use super::worker::{flatness, shortest_axis};
use super::*;

/// Exercises the channel-closed check directly, without spinning up the
/// GPU-backed worker: a real "panicked worker" is covered by the
/// server's Hello-reuse behaviour instead.
#[tokio::test]
async fn is_alive_reflects_worker_channel() {
    let (tx, rx) = mpsc::channel(1);
    let (_scores_tx, scores) = watch::channel(None);
    let (_meshes_tx, meshes) = watch::channel(MeshLog::default());
    let (_status_tx, status) = watch::channel(StatusMsg::default());
    let session = GuideSession {
        tx,
        scores,
        meshes,
        status,
        _actor: Actor::new("test"),
    };
    assert!(session.is_alive());
    drop(rx);
    assert!(!session.is_alive());
}

#[test]
fn shortest_axis_follows_rotation() {
    let h = std::f32::consts::FRAC_1_SQRT_2;
    // 90° about x in [w, x, y, z]; the flat local z axis maps to ±y.
    let a = shortest_axis(&[h, h, 0.0, 0.0], &[1.0, 1.0, 0.01]);
    assert!(a.dot(Vec3::Y).abs() > 0.999, "{a}");
}

#[test]
fn flatness_compares_smallest_to_middle_scale() {
    assert!((flatness(&[1.0, 0.25, 0.5]) - 0.5).abs() < 1e-6);
    assert_eq!(flatness(&[2.0, 2.0, 2.0]), 0.0);
    assert!(flatness(&[1.0, 1.0, 0.001]) > 0.99);
    assert_eq!(flatness(&[0.0, 0.0, 1.0]), 0.0);
}
