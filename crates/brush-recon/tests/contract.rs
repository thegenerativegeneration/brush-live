use brush_recon::*;
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use burn::tensor::{Device, Tensor};
use glam::{Quat, UVec2, Vec2, Vec3};
use image::RgbImage;
use std::sync::Arc;

fn camera() -> Camera {
    Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        1.0,
        Vec2::splat(0.5),
        CameraModel::Pinhole,
    )
}

fn frame() -> ReconFrame {
    ReconFrame {
        id: 174,
        camera: camera(),
        rgb: Arc::new(RgbImage::new(4, 3)),
        lidar: None,
        roi: Arc::new(vec![255; 12]),
    }
}

fn request() -> RenderRequest {
    RenderRequest {
        camera: camera(),
        size: UVec2::new(4, 3),
        background: Vec3::ZERO,
        depth: false,
        normals: false,
    }
}

#[test]
fn frame_rejects_mismatched_roi() {
    let mut f = frame();
    assert!(f.validate().is_ok());
    f.roi = Arc::new(vec![255; 11]);
    assert!(f.validate().is_err());
}

#[test]
fn invalid_camera_and_mask_values_are_rejected() {
    let mut f = frame();
    f.camera.position.x = f32::NAN;
    assert!(f.validate().is_err());
    f.camera = camera();
    f.roi = Arc::new(vec![7; 12]);
    assert!(f.validate().is_err());
    let mut r = request();
    r.size = UVec2::ZERO;
    assert!(
        r.validate(Capabilities {
            depth: true,
            normals: true
        })
        .is_err()
    );
}

#[test]
fn depth_rejects_invalid_shape() {
    let mut d = MetricDepth {
        size: UVec2::new(2, 2),
        values_m: vec![1.0; 4],
        confidence: Some(vec![2; 4]),
    };
    assert!(d.validate().is_ok());
    d.values_m.pop();
    assert!(d.validate().is_err());
    d.values_m.push(1.0);
    d.confidence = Some(vec![2; 3]);
    assert!(d.validate().is_err());
}

#[test]
fn depth_rejects_nonfinite_values_and_unknown_confidence() {
    let mut d = MetricDepth {
        size: UVec2::new(1, 1),
        values_m: vec![f32::NAN],
        confidence: None,
    };
    assert!(d.validate().is_err());
    d.values_m[0] = 1.0;
    d.confidence = Some(vec![3]);
    assert!(d.validate().is_err());
}

#[test]
fn requested_unsupported_channel_is_error() {
    let mut r = request();
    r.depth = true;
    assert!(matches!(
        r.validate(Capabilities {
            depth: false,
            normals: false
        }),
        Err(ReconError::UnsupportedOutput("depth"))
    ));
    r.depth = false;
    r.normals = true;
    assert!(matches!(
        r.validate(Capabilities {
            depth: false,
            normals: false
        }),
        Err(ReconError::UnsupportedOutput("normals"))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn result_channels_match_requested_size() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let r = request();
    let mut out = RenderResult {
        rgb: Tensor::zeros([3, 4, 3], &d),
        alpha: Tensor::zeros([3, 4], &d),
        depth_m: None,
        normals: None,
    };
    assert!(out.validate(&r).is_ok());
    out.alpha = Tensor::zeros([4, 3], &d);
    assert!(out.validate(&r).is_err());
    out.alpha = Tensor::zeros([3, 4], &d);
    out.depth_m = Some(Tensor::zeros([3, 4], &d));
    assert!(
        out.validate(&r).is_err(),
        "unrequested channel must be None"
    );
    let mut depth_request = r.clone();
    depth_request.depth = true;
    assert!(out.validate(&depth_request).is_ok());
    out.depth_m = None;
    assert!(
        out.validate(&depth_request).is_err(),
        "requested channel must be present"
    );
}
