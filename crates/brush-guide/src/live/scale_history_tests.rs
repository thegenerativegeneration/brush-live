//! The mono scale history across `LiveModel::reset`, which the session worker
//! calls for every new segment.

use super::*;
use crate::keyframe::decode_keyframe;
use crate::mono::{ScaleFit, ScaleSource};
use crate::protocol::KeyframeHeader;
use glam::Mat4;

const W: u32 = 32;
const H: u32 = 24;

/// A keyframe at the origin with a mono map of 2 m and, with `lidar`, a LiDAR
/// map of that depth (768 samples, enough for a LiDAR fit); no feature points.
async fn keyframe(id: u64, lidar: Option<f32>, dir: &std::path::Path) -> DecodedKeyframe {
    let img =
        image::RgbImage::from_fn(W, H, |x, y| image::Rgb([(x * 8) as u8, (y * 10) as u8, 90]));
    let mut payload = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut payload),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let jpeg_len = payload.len() as u32;
    let f16_map = |payload: &mut Vec<u8>, v: f32| {
        for _ in 0..W * H {
            payload.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    };
    if let Some(d) = lidar {
        f16_map(&mut payload, d);
    }
    f16_map(&mut payload, 2.0);
    let header = KeyframeHeader {
        id,
        timestamp: id as f64,
        pose: Mat4::IDENTITY.to_cols_array(),
        fx: 30.0,
        fy: 30.0,
        cx: W as f32 / 2.0,
        cy: H as f32 / 2.0,
        width: W,
        height: H,
        jpeg_len,
        depth_size: lidar.map(|_| [W, H]),
        depth_confidence: false,
        num_points: 0,
        mono_depth_size: Some([W, H]),
    };
    decode_keyframe(&header, &payload, dir).await.unwrap()
}

/// A LiDAR fit before the reset is the recent scale of a keyframe with too
/// few samples after it.
#[tokio::test]
async fn reset_keeps_the_scale_history() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let dir = std::env::temp_dir().join(format!("live-scale-history-{}", std::process::id()));
    let mut live = LiveModel::new(GuideConfig::default(), device);
    assert_eq!(live.scale_median(), None);
    assert!(live.add_keyframe(keyframe(0, Some(3.0), &dir).await).await);
    assert_eq!(live.scale_median(), Some(1.5));

    live.reset();
    assert!(live.views().is_empty());
    assert_eq!(live.scale_median(), Some(1.5));
    let few = keyframe(1, None, &dir).await;
    let (seed, fit) = live.mono_input(&few, UVec2::new(W, H));
    assert!(
        matches!(fit, ScaleFit::Fitted { source: ScaleSource::Recent, scale, .. } if scale == 1.5),
        "{fit}"
    );
    assert_eq!(seed.map(|s| s.scale), Some(1.5));
    std::fs::remove_dir_all(dir).ok();
}
