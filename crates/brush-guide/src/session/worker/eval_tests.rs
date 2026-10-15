use super::*;
use crate::protocol::KeyframeHeader;
use glam::{Mat4, Vec3};

/// A 16×12 keyframe at `(0, 0, 2)` looking at the origin, without feature points.
fn keyframe(id: u64) -> (KeyframeHeader, Vec<u8>) {
    let (w, h) = (16u32, 12u32);
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 16) as u8, (y * 20) as u8, 90]));
    let mut jpeg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
        .unwrap();
    let pose = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO, Vec3::Y).inverse();
    let header = KeyframeHeader {
        id,
        timestamp: 0.0,
        pose: pose.to_cols_array(),
        fx: 14.0,
        fy: 14.0,
        cx: 8.0,
        cy: 6.0,
        width: w,
        height: h,
        jpeg_len: jpeg.len() as u32,
        depth_size: None,
        depth_confidence: false,
        num_points: 0,
        mono_depth_size: None,
    };
    (header, jpeg)
}

/// Without held-out views an eval (the finish one included) returns before assembling the full model: no parked
/// tile is loaded and the step-rate window does not move.
#[tokio::test]
async fn an_eval_without_held_out_views_does_no_work() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let dir = std::env::temp_dir().join(format!("worker-eval-none-{}", std::process::id()));
    let (scores, _) = watch::channel(None);
    let (status, _) = watch::channel(StatusMsg::default());
    let (preview, _) = watch::channel(None);
    let channels = Channels { scores, status, preview };
    let mut w = Worker::new(GuideConfig::default(), device, dir.clone(), channels);
    let (h, p) = keyframe(0);
    assert!(w.live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap()).await);
    assert!(w.live.splats().is_some() && !w.holdout.has_views());
    let window = w.rate_window;
    assert!(!w.eval_holdout(true, true).await);
    assert_eq!(w.rate_window, window);
    assert!(!dir.join("eval.jsonl").exists());
    std::fs::remove_dir_all(dir).ok();
}
