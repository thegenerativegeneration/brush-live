use crate::ReconError;
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use glam::UVec2;
use image::RgbImage;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MetricDepth {
    pub size: UVec2,
    pub values_m: Vec<f32>,
    pub confidence: Option<Vec<u8>>,
}

impl MetricDepth {
    pub fn validate(&self) -> Result<(), ReconError> {
        let n = pixel_count(self.size)?;
        if self.values_m.len() != n || self.confidence.as_ref().is_some_and(|c| c.len() != n) {
            return Err(ReconError::input(
                "depth/confidence dimensions do not match",
            ));
        }
        if self.values_m.iter().any(|d| !d.is_finite() || *d < 0.0)
            || self
                .confidence
                .as_ref()
                .is_some_and(|c| c.iter().any(|v| *v > 2))
        {
            return Err(ReconError::input(
                "depth must be finite/nonnegative; confidence must be 0..2",
            ));
        }
        Ok(())
    }
}

pub fn pixel_count(size: UVec2) -> Result<usize, ReconError> {
    if size.x == 0 || size.y == 0 {
        return Err(ReconError::input("image dimensions must be positive"));
    }
    (size.x as usize)
        .checked_mul(size.y as usize)
        .ok_or_else(|| ReconError::input("image size overflows"))
}

pub fn validate_camera(camera: &Camera) -> Result<(), ReconError> {
    if camera.camera_model != CameraModel::Pinhole {
        return Err(ReconError::UnsupportedCamera);
    }
    if !camera.is_valid()
        || (camera.rotation.length_squared() - 1.0).abs() > 1e-3
        || !(0.0..std::f64::consts::PI).contains(&camera.fov_x)
        || camera.fov_x == 0.0
        || !(0.0..std::f64::consts::PI).contains(&camera.fov_y)
        || camera.fov_y == 0.0
    {
        return Err(ReconError::input(
            "camera must have a finite rigid pose and positive pinhole field of view",
        ));
    }
    Ok(())
}

#[derive(Clone)]
pub struct ReconFrame {
    pub id: u64,
    pub camera: Camera,
    pub rgb: Arc<RgbImage>,
    pub lidar: Option<Arc<MetricDepth>>,
    pub roi: Arc<Vec<u8>>,
}

impl ReconFrame {
    pub fn validate(&self) -> Result<(), ReconError> {
        validate_camera(&self.camera)?;
        let n = pixel_count(UVec2::new(self.rgb.width(), self.rgb.height()))?;
        if self.roi.len() != n || self.roi.iter().any(|&v| v != 0 && v != 255) {
            return Err(ReconError::input(format!(
                "frame {}: ROI must be W*H binary 0/255 bytes",
                self.id
            )));
        }
        if let Some(d) = &self.lidar {
            d.validate()?;
        }
        Ok(())
    }
}
