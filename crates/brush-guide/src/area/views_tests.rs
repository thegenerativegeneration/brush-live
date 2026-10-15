//! Tests of the area views and the frustum test on tile boxes.

use super::*;
use glam::Quat;

#[test]
fn area_views_use_horizontal_distance() {
    let p = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 40.0, 4.0), Vec3::new(6.0, 0.0, 0.0)];
    assert_eq!(area_views(&p, Vec3::ZERO, 5.0), vec![0, 1]);
}

#[test]
fn membership_additions_count_new_views_only() {
    assert_eq!(membership_additions(&[1, 2, 3], &[2, 3, 4, 5]), 2);
    assert_eq!(membership_additions(&[], &[0, 1]), 2);
    assert_eq!(membership_additions(&[1, 2, 3], &[2]), 0, "views leaving the area do not count");
}

fn camera() -> Camera {
    let fov = 60f64.to_radians();
    Camera::new(Vec3::ZERO, Quat::IDENTITY, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole)
}

#[test]
fn frustum_box_test_is_conservative_and_rejects_clear_misses() {
    let c = camera();
    assert!(camera_sees_box(&c, Vec3::new(-1.0, -1.0, 4.0), Vec3::new(1.0, 1.0, 6.0), 100.0));
    assert!(!camera_sees_box(&c, Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0), 100.0), "behind");
    assert!(!camera_sees_box(&c, Vec3::new(10.0, -1.0, 1.0), Vec3::new(12.0, 1.0, 2.0), 100.0), "outside the fov");
    assert!(!camera_sees_box(&c, Vec3::new(-1.0, -1.0, 150.0), Vec3::new(1.0, 1.0, 160.0), 100.0), "beyond far");
    assert!(camera_sees_box(&c, Vec3::splat(-5.0), Vec3::splat(5.0), 100.0), "camera inside the box");
}

#[test]
fn junction_views_are_the_candidates_that_see_the_tile() {
    let far_away = Camera { position: Vec3::new(0.0, 0.0, 500.0), ..camera() };
    let cams = [(4, camera()), (9, far_away)];
    assert_eq!(junction_views(&cams, Vec3::new(-1.0, -1.0, 4.0), Vec3::new(1.0, 1.0, 6.0), 100.0), vec![4]);
}

/// The planes are those of a pinhole: any other camera model (a fisheye bends rays past them) sees every box.
#[test]
fn frustum_box_test_keeps_every_box_for_other_camera_models() {
    use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
    let c = Camera { camera_model: CameraModel::KannalaBrandt4(KannalaBrandt4Params::default()), ..camera() };
    assert!(camera_sees_box(&c, Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0), 100.0), "behind");
    assert!(camera_sees_box(&c, Vec3::new(10.0, -1.0, 1.0), Vec3::new(12.0, 1.0, 2.0), 100.0), "outside the fov");
}

/// An off-centre principal point widens the frustum on one side: a box only that side sees is kept.
#[test]
fn frustum_box_test_follows_the_principal_point() {
    // tan(30°) ≈ 0.577; with the centre at u = 0.8 the left edge reaches x/z = −0.92 and the right +0.23.
    let c = Camera { center_uv: glam::vec2(0.8, 0.5), ..camera() };
    assert!(camera_sees_box(&c, Vec3::new(-8.5, -0.5, 9.9), Vec3::new(-8.0, 0.5, 10.0), 100.0), "left, wide side");
    assert!(!camera_sees_box(&c, Vec3::new(4.0, -0.5, 9.9), Vec3::new(4.5, 0.5, 10.0), 100.0), "right, narrow side");
}
