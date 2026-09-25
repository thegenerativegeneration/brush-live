use crate::protocol::{KeyframeHeader, ProtocolError, split_keyframe_payload};
use brush_dataset::{load_image::LoadImage, scene::SceneView};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_vfs::BrushVfs;
use glam::{Mat3, Mat4, Vec3, vec2};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum KeyframeError {
    #[error("pose is not a finite rigid transform")]
    InvalidPose,
    #[error("invalid intrinsics")]
    InvalidIntrinsics,
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("jpeg decode failed: {0}")]
    Image(#[from] image::ImageError),
    #[error("jpeg size {actual:?} does not match header {expected:?}")]
    SizeMismatch { expected: (u32, u32), actual: (u32, u32) },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl DepthMap {
    pub fn sample_uv(&self, u: f32, v: f32) -> Option<f32> {
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        let d = self.values[(y * self.width + x) as usize];
        (d.is_finite() && d > 0.0).then_some(d)
    }
}

pub struct DecodedKeyframe {
    pub id: u64,
    pub camera: Camera,
    pub image: image::RgbImage,
    pub depth: Option<DepthMap>,
    pub points: Vec<Vec3>,
    pub view: SceneView,
}

pub fn arkit_to_camera(h: &KeyframeHeader) -> Result<Camera, KeyframeError> {
    let c2w = Mat4::from_cols_array(&h.pose);
    if !c2w.is_finite() {
        return Err(KeyframeError::InvalidPose);
    }
    let r = Mat3::from_mat4(c2w);
    let orthonormal = (r.transpose() * r - Mat3::IDENTITY).to_cols_array().iter().all(|v| v.abs() < 1e-3);
    if !orthonormal || (r.determinant() - 1.0).abs() > 1e-3 {
        return Err(KeyframeError::InvalidPose);
    }
    if !(h.fx > 0.0 && h.fy > 0.0 && h.width > 0 && h.height > 0) {
        return Err(KeyframeError::InvalidIntrinsics);
    }
    // ARKit cameras look down -z with y up; Brush uses +z forward, y down.
    let mut m = c2w;
    m.y_axis *= -1.0;
    m.z_axis *= -1.0;
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    let model = CameraModel::Pinhole;
    let fov_x = focal_to_fov(h.fx as f64, h.width, &model);
    let fov_y = focal_to_fov(h.fy as f64, h.height, &model);
    let center_uv = vec2(h.cx / h.width as f32, h.cy / h.height as f32);
    Ok(Camera::new(translation, rotation, fov_x, fov_y, center_uv, model))
}

pub async fn decode_keyframe(
    h: &KeyframeHeader,
    payload: &[u8],
    session_dir: &Path,
) -> Result<DecodedKeyframe, KeyframeError> {
    let camera = arkit_to_camera(h)?;
    let parts = split_keyframe_payload(h, payload)?;
    let image = image::load_from_memory_with_format(parts.jpeg, image::ImageFormat::Jpeg)?.into_rgb8();
    if image.dimensions() != (h.width, h.height) {
        return Err(KeyframeError::SizeMismatch { expected: (h.width, h.height), actual: image.dimensions() });
    }

    let rel = PathBuf::from(format!("images/{}.jpg", h.id));
    tokio::fs::create_dir_all(session_dir.join("images")).await?;
    tokio::fs::write(session_dir.join(&rel), parts.jpeg).await?;
    let vfs = Arc::new(BrushVfs::from_directory_files(session_dir, vec![rel.clone()]));
    let load = LoadImage::new(vfs, rel, None, h.width.max(h.height), None, false);

    let depth = h.depth_size.zip(parts.depth).map(|([width, height], values)| DepthMap { width, height, values });
    let points = parts.points.into_iter().map(Vec3::from).collect();
    Ok(DecodedKeyframe {
        id: h.id,
        camera,
        image,
        depth,
        points,
        view: SceneView { image: load, camera },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(pose: Mat4, width: u32, height: u32) -> KeyframeHeader {
        KeyframeHeader {
            id: 1,
            timestamp: 0.0,
            pose: pose.to_cols_array(),
            fx: 700.0,
            fy: 710.0,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
            jpeg_len: 0,
            depth_size: None,
            num_points: 0,
        }
    }

    #[test]
    fn identity_arkit_camera_looks_down_negative_z() {
        let cam = arkit_to_camera(&header(Mat4::IDENTITY, 960, 720)).unwrap();
        let forward = cam.local_to_world().transform_vector3(Vec3::Z);
        assert!((forward - Vec3::NEG_Z).length() < 1e-5, "{forward}");
        let down = cam.local_to_world().transform_vector3(Vec3::Y);
        assert!((down - Vec3::NEG_Y).length() < 1e-5, "{down}");
    }

    #[test]
    fn translation_and_focal_survive() {
        let pose = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let cam = arkit_to_camera(&header(pose, 960, 720)).unwrap();
        assert!((cam.position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        let f = cam.focal(glam::uvec2(960, 720));
        assert!((f.x - 700.0).abs() < 1e-2 && (f.y - 710.0).abs() < 1e-2, "{f}");
    }

    #[test]
    fn different_sizes_keep_their_own_intrinsics() {
        let a = arkit_to_camera(&header(Mat4::IDENTITY, 960, 720)).unwrap();
        let b = arkit_to_camera(&header(Mat4::IDENTITY, 720, 960)).unwrap();
        assert!((a.focal(glam::uvec2(960, 720)).x - 700.0).abs() < 1e-2);
        assert!((b.focal(glam::uvec2(720, 960)).x - 700.0).abs() < 1e-2);
        assert!((a.fov_x - b.fov_x).abs() > 1e-3);
    }

    #[test]
    fn invalid_poses_are_rejected() {
        let mut nan = Mat4::IDENTITY;
        nan.w_axis.x = f32::NAN;
        assert!(arkit_to_camera(&header(nan, 960, 720)).is_err());
        let scaled = Mat4::from_scale(Vec3::splat(2.0));
        assert!(arkit_to_camera(&header(scaled, 960, 720)).is_err());
    }

    #[test]
    fn depth_sampling() {
        let d = DepthMap { width: 2, height: 1, values: vec![1.0, 0.0] };
        assert_eq!(d.sample_uv(0.1, 0.5), Some(1.0));
        assert_eq!(d.sample_uv(0.9, 0.5), None);
    }

    #[tokio::test]
    async fn decode_writes_image_and_builds_view() {
        let dir = std::env::temp_dir().join(format!("brush-guide-kf-{}", std::process::id()));
        let img = image::RgbImage::from_pixel(8, 6, image::Rgb([200, 10, 10]));
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        let mut h = header(Mat4::IDENTITY, 8, 6);
        h.jpeg_len = jpeg.len() as u32;

        let kf = decode_keyframe(&h, &jpeg, &dir).await.unwrap();
        assert_eq!(kf.image.dimensions(), (8, 6));
        assert!(dir.join("images/1.jpg").exists());
        assert_eq!(kf.view.image.load().await.unwrap().width(), 8);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
