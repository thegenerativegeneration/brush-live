mod test_scene;

use brush_guide::config::GuideConfig;
use brush_guide::keyframe::decode_keyframe;
use brush_guide::live::LiveModel;
use brush_guide::protocol::KeyframeHeader;
use glam::{Mat4, Vec3};

/// ARKit-style keyframe looking at the origin from `pos`, textured image, a few feature points.
fn keyframe(id: u64, pos: Vec3) -> (KeyframeHeader, Vec<u8>) {
    let (w, h) = (64u32, 48u32);
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 8) as u8])
    });
    let mut jpeg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let pose = Mat4::look_at_rh(pos, Vec3::ZERO, Vec3::Y).inverse();
    let points = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.2, 0.1, 0.0),
        Vec3::new(-0.2, -0.1, 0.1),
    ];
    let header = KeyframeHeader {
        id,
        timestamp: id as f64,
        pose: pose.to_cols_array(),
        fx: 50.0,
        fy: 50.0,
        cx: 32.0,
        cy: 24.0,
        width: w,
        height: h,
        jpeg_len: jpeg.len() as u32,
        depth_size: None,
        num_points: points.len() as u32,
    };
    let mut payload = jpeg;
    for p in points {
        for v in p.to_array() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    (header, payload)
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("brush-guide-live-{name}-{}", std::process::id()))
}

#[tokio::test]
async fn first_keyframe_initialises_and_trains() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("first");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    live.train_step().await; // no views: no-op, no panic
    assert!(live.splats().is_none());

    let (h, p) = keyframe(1, Vec3::new(0.0, 0.0, 2.0));
    assert!(
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await
    );
    assert!(live.splats().unwrap().num_splats() > 0);
    for _ in 0..20 {
        live.train_step().await;
    }
    assert_eq!(live.iter(), 20);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn duplicate_ids_are_ignored() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("dup");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    let (h, p) = keyframe(1, Vec3::new(0.0, 0.0, 2.0));
    assert!(
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await
    );
    assert!(
        !live
            .add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await
    );
    assert_eq!(live.views().len(), 1);
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn keyframe_without_points_still_initialises() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("nopoints");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    let (mut h, p) = keyframe(1, Vec3::new(0.0, 0.0, 2.0));
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

#[tokio::test]
async fn new_views_add_splats_and_training_continues() {
    let device = test_scene::device().await.autodiff();
    let dir = tmp("grow");
    let mut live = LiveModel::new(GuideConfig::default(), device);
    for (i, a) in [0.0f32, 1.2, 2.4].iter().enumerate() {
        let (h, p) = keyframe(i as u64, Vec3::new(2.0 * a.sin(), 0.0, 2.0 * a.cos()));
        live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
            .await;
        for _ in 0..30 {
            live.train_step().await;
        }
    }
    assert_eq!(live.views().len(), 3);
    assert_eq!(live.last_keyframe_id(), Some(2));
    assert!(live.splats().unwrap().num_splats() > 0);
    std::fs::remove_dir_all(dir).ok();
}
