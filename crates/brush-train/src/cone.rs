//! A camera's view approximated as a cone, for "does this camera see that point" tests.

use brush_render::camera::Camera;
use glam::Vec3;

/// A camera approximated as a cone: apex, axis, cos of the half angle.
#[derive(Clone, Copy, Debug)]
pub struct ViewCone {
    pub position: Vec3,
    pub forward: Vec3,
    pub cos_half_fov: f32,
}

impl ViewCone {
    /// `forward = rotation · Z`, half angle from the wider of the two FOVs.
    pub fn of_camera(c: &Camera) -> Self {
        Self {
            position: c.position,
            forward: c.rotation * Vec3::Z,
            cos_half_fov: (0.5 * c.fov_x.max(c.fov_y) as f32).cos(),
        }
    }

    /// Whether `p` lies in the cone, farther than 1e-4 and at most `far` from the apex.
    pub fn sees(&self, p: Vec3, far: f32) -> bool {
        let d = p - self.position;
        let len = d.length();
        len > 1e-4 && len <= far && d.dot(self.forward) >= self.cos_half_fov * len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::{Quat, Vec2};

    #[test]
    fn of_camera_uses_rotation_z_and_the_wider_fov() {
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let c = Camera::new(Vec3::new(1.0, 2.0, 3.0), rotation, 1.0, 0.5, Vec2::splat(0.5), CameraModel::Pinhole);
        let cone = ViewCone::of_camera(&c);
        assert_eq!(cone.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cone.forward, rotation * Vec3::Z);
        assert_eq!(cone.cos_half_fov, (0.5f32).cos());
    }

    #[test]
    fn sees_limits_distance_and_angle() {
        let cone = ViewCone {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            cos_half_fov: 0.5,
        };
        let far = 10.0;
        assert!(!cone.sees(Vec3::ZERO, far));
        assert!(cone.sees(Vec3::new(0.0, 0.0, far), far));
        assert!(!cone.sees(Vec3::new(0.0, 0.0, far + 0.01), far));
        assert!(!cone.sees(Vec3::new(5.0, 0.0, 1.0), far));
    }
}
