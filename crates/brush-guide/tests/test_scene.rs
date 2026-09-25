#![allow(dead_code)]
use brush_guide::protocol::KeyframeHeader;
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

/// ARKit-style keyframe looking at the origin from `pos`, textured image, a few feature points.
pub fn keyframe(id: u64, pos: Vec3) -> (KeyframeHeader, Vec<u8>) {
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
