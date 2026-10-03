use glam::Vec3;

use super::preview::{PREVIEW_FLOATS, PreviewClock, pack};
use super::splat_read::{flatness, shortest_axis};
use super::*;
use std::time::{Duration, Instant};

/// Exercises the channel-closed check directly, without spinning up the
/// GPU-backed worker: a real "panicked worker" is covered by the
/// server's Hello-reuse behaviour instead.
#[tokio::test]
async fn is_alive_reflects_worker_channel() {
    let (tx, rx) = mpsc::channel(1);
    let (_scores_tx, scores) = watch::channel(None);
    let (_status_tx, status) = watch::channel(StatusMsg::default());
    let (_preview_tx, preview) = watch::channel(None);
    let session = GuideSession {
        tx,
        scores,
        status,
        preview,
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

#[test]
fn pack_lays_out_one_splat_and_normalises_rotation() {
    let out = pack(
        &[1.0, 2.0, 3.0],
        &[2.0, 0.0, 0.0, 0.0],
        &[0.1, 0.2, 0.3],
        &[0.5],
        &[0.7, 0.8, 0.9],
    );
    assert_eq!(out.len(), PREVIEW_FLOATS);
    assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&out[3..7], &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(&out[7..10], &[0.1, 0.2, 0.3]);
    assert_eq!(out[10], 0.5);
    assert_eq!(&out[11..14], &[0.7, 0.8, 0.9]);
}

#[test]
fn pack_replaces_degenerate_rotation_with_identity() {
    let out = pack(&[0.0; 3], &[0.0; 4], &[1.0; 3], &[1.0], &[0.0; 3]);
    assert_eq!(&out[3..7], &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn preview_clock_follows_interval() {
    let t0 = Instant::now();
    let mut c = PreviewClock::default();
    assert!(!c.due(t0), "off by default");
    c.set(Some(Duration::ZERO));
    assert!(c.due(t0));
    c.taken(t0);
    assert!(c.due(t0), "zero interval: every step");
    c.set(Some(Duration::from_millis(100)));
    c.taken(t0);
    assert!(!c.due(t0 + Duration::from_millis(50)));
    assert!(c.due(t0 + Duration::from_millis(100)));
    c.set(None);
    assert!(!c.due(t0 + Duration::from_secs(1)));
}
