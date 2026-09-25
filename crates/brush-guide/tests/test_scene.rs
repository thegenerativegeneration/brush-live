#![allow(dead_code)]
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::Device;
use glam::{Mat4, Vec3, vec2};

pub async fn device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await)
}

pub fn splats_from(means: &[[f32; 3]], log_scale: f32, opacity: f32, device: &Device) -> Splats {
    let n = means.len();
    Splats::from_raw(
        means.iter().flatten().copied().collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(n),
        vec![log_scale; n * 3],
        vec![0.5; n * 3],
        vec![inverse_sigmoid(opacity); n],
        SplatRenderMode::Default,
        device,
    )
}

/// Brush (`OpenCV`) camera at `pos` looking at `look_at`, 60° fov.
pub fn camera_at(pos: Vec3, look_at: Vec3) -> Camera {
    // Build an OpenGL-style view, then flip y/z like arkit_to_camera does.
    let up = if (look_at - pos).normalize().abs().dot(Vec3::Y) > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let mut c2w = Mat4::look_at_rh(pos, look_at, up).inverse();
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    let fov = 60f64.to_radians();
    Camera::new(
        translation,
        rotation,
        fov,
        fov,
        vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}
