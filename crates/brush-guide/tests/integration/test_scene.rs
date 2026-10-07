#![allow(dead_code)]
use brush_guide::protocol::KeyframeHeader;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{
    SplatRenderMode, Splats, TextureMode, inverse_sigmoid, render_splats,
};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::sh::rgb_to_sh;
use burn::tensor::Device;
use glam::{Mat4, UVec2, Vec3, vec2};

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

/// ARKit camera-to-world pose (column-major, as `KeyframeHeader::pose`) at `pos` looking at the origin, y up.
pub fn arkit_pose_at(pos: Vec3) -> [f32; 16] {
    Mat4::look_at_rh(pos, Vec3::ZERO, Vec3::Y)
        .inverse()
        .to_cols_array()
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
    let points = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.2, 0.1, 0.0),
        Vec3::new(-0.2, -0.1, 0.1),
    ];
    let header = KeyframeHeader {
        id,
        timestamp: id as f64,
        pose: arkit_pose_at(pos),
        fx: 50.0,
        fy: 50.0,
        cx: 32.0,
        cy: 24.0,
        width: w,
        height: h,
        jpeg_len: jpeg.len() as u32,
        depth_size: None,
        depth_confidence: false,
        num_points: points.len() as u32,
        mono_depth_size: None,
    };
    let mut payload = jpeg;
    for p in points {
        for v in p.to_array() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    (header, payload)
}

/// Intrinsics of the wall keyframes: 64×48 px, f = 50 px, centred.
pub const WALL_W: u32 = 64;
pub const WALL_H: u32 = 48;
pub const WALL_F: f32 = 50.0;

/// A textured wall: `n`×`n` opaque splats on the plane z = 0 over `[-half, half]²`, in 0.5 m colour blocks.
pub fn wall(n: usize, half: f32, device: &Device) -> Splats {
    let step = 2.0 * half / (n - 1) as f32;
    let (mut means, mut sh) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (-half + i as f32 * step, -half + j as f32 * step);
            means.extend([x, y, 0.0]);
            let block = ((x + half) / 0.5) as i32 * 7 + ((y + half) / 0.5) as i32 * 3;
            let c = Vec3::new(
                (block % 5) as f32 / 4.0,
                (block % 3) as f32 / 2.0,
                (block % 7) as f32 / 6.0,
            );
            sh.extend(rgb_to_sh(c).to_array());
        }
    }
    let count = n * n;
    Splats::from_raw(
        means,
        [1.0, 0.0, 0.0, 0.0].repeat(count),
        vec![(step * 0.8).ln(); count * 3],
        sh,
        vec![inverse_sigmoid(0.99); count],
        SplatRenderMode::Default,
        device,
    )
}

fn wall_header(id: u64, pose: Mat4) -> KeyframeHeader {
    KeyframeHeader {
        id,
        timestamp: id as f64,
        pose: pose.to_cols_array(),
        fx: WALL_F,
        fy: WALL_F,
        cx: WALL_W as f32 / 2.0,
        cy: WALL_H as f32 / 2.0,
        width: WALL_W,
        height: WALL_H,
        jpeg_len: 0,
        depth_size: Some([WALL_W, WALL_H]),
        depth_confidence: false,
        num_points: 0,
        mono_depth_size: None,
    }
}

/// The Brush camera of an ARKit camera-to-world pose with the wall intrinsics.
pub fn wall_camera(pose: Mat4) -> Camera {
    brush_guide::keyframe::arkit_to_camera(&wall_header(0, pose)).unwrap()
}

/// `splats` seen from `camera` at the wall size on black: RGB in 0..1, row-major.
pub async fn render_rgb(splats: &Splats, camera: &Camera) -> Vec<f32> {
    let (img, _) = render_splats(
        splats.clone(),
        camera,
        UVec2::new(WALL_W, WALL_H),
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    let rgba = img
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    rgba.chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

/// Mean absolute RGB difference between `splats` seen from `camera` and `reference` (from `render_rgb`).
pub async fn rgb_error(splats: &Splats, camera: &Camera, reference: &[f32]) -> f32 {
    let rgb = render_rgb(splats, camera).await;
    rgb.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / rgb.len() as f32
}

/// Depth along the optical axis to the plane z = 0 per pixel, row-major, for an ARKit camera-to-world pose with the
/// wall intrinsics; 0 where the ray misses the plane.
pub fn wall_depth(pose: Mat4) -> Vec<f32> {
    let o = pose.w_axis.truncate();
    let (cx, cy) = (WALL_W as f32 / 2.0, WALL_H as f32 / 2.0);
    (0..WALL_H)
        .flat_map(|v| (0..WALL_W).map(move |u| (u, v)))
        .map(|(u, v)| {
            // ARKit camera: x right, y up, looking down -z; image rows go down. |z| = 1, so t is the depth.
            let ray = Vec3::new(
                (u as f32 + 0.5 - cx) / WALL_F,
                -(v as f32 + 0.5 - cy) / WALL_F,
                -1.0,
            );
            let d = pose.transform_vector3(ray);
            let t = -o.z / d.z;
            if t.is_finite() && t > 0.0 { t } else { 0.0 }
        })
        .collect()
}

/// A keyframe of `wall` whose image and LiDAR-like depth come from `true_pose` but whose header reports
/// `sent_pose`, as the phone's keyframes do while ORB-SLAM3 drifts.
pub async fn wall_keyframe(
    id: u64,
    wall: &Splats,
    true_pose: Mat4,
    sent_pose: Mat4,
) -> (KeyframeHeader, Vec<u8>) {
    let rgb = render_rgb(wall, &wall_camera(true_pose)).await;
    let img = image::RgbImage::from_fn(WALL_W, WALL_H, |x, y| {
        let i = ((y * WALL_W + x) * 3) as usize;
        image::Rgb([0, 1, 2].map(|c| (rgb[i + c].clamp(0.0, 1.0) * 255.0).round() as u8))
    });
    let mut jpeg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let mut header = wall_header(id, sent_pose);
    header.jpeg_len = jpeg.len() as u32;
    let mut payload = jpeg;
    for d in wall_depth(true_pose) {
        payload.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    (header, payload)
}
