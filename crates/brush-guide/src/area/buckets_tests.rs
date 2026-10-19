use super::*;
use brush_render::kernels::camera_model::CameraModel;
use glam::{Quat, Vec2, Vec3};

/// A 60° camera at `pos` turned `yaw` radians about +y from looking along +z.
fn cam(pos: Vec3, yaw: f32) -> Camera {
    let fov = 60f64.to_radians();
    Camera::new(pos, Quat::from_rotation_y(yaw), fov, fov, Vec2::new(0.5, 0.5), CameraModel::Pinhole)
}

#[test]
fn one_view_per_cell_and_direction_with_the_largest_focal_ties_to_the_lowest_index() {
    let cams = [
        cam(Vec3::new(0.1, 0.0, 0.1), 0.0),
        cam(Vec3::new(0.3, 0.2, 0.4), 0.1),
        cam(Vec3::new(0.2, 0.1, 0.2), 0.0),
        cam(Vec3::new(0.6, 0.0, 0.1), 0.0),
        cam(Vec3::new(0.1, 0.0, 0.1), std::f32::consts::PI),
        cam(Vec3::new(0.2, 0.0, 0.2), 0.0),
    ];
    let focals = [50.0, 60.0, 60.0, 50.0, 50.0, 50.0];
    // 0, 1, 2, 5 share cell (0, 0, 0) and a yaw bin: 1 and 2 tie on the largest focal, 1 is lower. 3 lies in the next
    // cell, 4 looks the other way.
    assert_eq!(representatives(&cams, &focals), vec![1, 1, 1, 3, 4, 1]);
}

#[test]
fn pitch_splits_a_bucket_and_a_pose_that_is_not_finite_represents_itself() {
    let down = Camera::new(Vec3::ZERO, Quat::from_rotation_x(1.2), 1.0, 1.0, Vec2::new(0.5, 0.5), CameraModel::Pinhole);
    let cams = [cam(Vec3::ZERO, 0.0), down, cam(Vec3::NAN, 0.0), cam(Vec3::new(0.1, 0.1, 0.1), 0.0)];
    assert_eq!(representatives(&cams, &[50.0; 4]), vec![0, 1, 2, 0]);
}
