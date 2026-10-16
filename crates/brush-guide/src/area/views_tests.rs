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

/// Share of the image a box covers: tan(30°) ≈ 0.577, so the image spans ±5.77 m at 10 m depth.
#[test]
fn projected_area_is_the_image_share_the_box_covers() {
    let c = camera();
    let w = 2.0 * (30f32.to_radians()).tan() * 10.0;
    let thin = projected_area(&c, Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 10.001), 100.0);
    assert!((thin - 4.0 / (w * w)).abs() < 1e-3, "{thin}");
    // A deeper box shows its near face: the far face projects inside it.
    let deep = projected_area(&c, Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 12.0), 100.0);
    assert!((deep - thin).abs() < 1e-3, "{deep} vs {thin}");
    // Half of it beyond the image's right edge counts half.
    let edge = w / 2.0;
    let half = projected_area(&c, Vec3::new(edge - 1.0, -1.0, 10.0), Vec3::new(edge + 1.0, 1.0, 10.001), 100.0);
    assert!((half - thin / 2.0).abs() < 1e-3, "{half}");
    // An off-axis box seen from the side shows more than its front face.
    let side = projected_area(&c, Vec3::new(1.0, -1.0, 10.0), Vec3::new(3.0, 1.0, 12.0), 100.0);
    let front = projected_area(&c, Vec3::new(1.0, -1.0, 10.0), Vec3::new(3.0, 1.0, 10.001), 100.0);
    assert!(side > front * 1.05, "{side} vs {front}");
}

#[test]
fn projected_area_clips_to_the_image_and_the_depth_range() {
    let c = camera();
    // Inside a slim box the near plane cuts it: its corners alone (±0.2 at 5 m) would cover a fraction.
    let inside = projected_area(&c, Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, 5.0), 100.0);
    assert!((inside - 1.0).abs() < 1e-5, "camera inside the box: {inside}");
    let wall = projected_area(&c, Vec3::new(-50.0, -50.0, 2.0), Vec3::new(50.0, 50.0, 3.0), 100.0);
    assert!((wall - 1.0).abs() < 1e-5, "a wall filling the view: {wall}");
    assert_eq!(projected_area(&c, Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0), 100.0), 0.0, "behind");
    assert_eq!(projected_area(&c, Vec3::new(10.0, -1.0, 1.0), Vec3::new(12.0, 1.0, 2.0), 100.0), 0.0, "beside");
    assert_eq!(projected_area(&c, Vec3::new(-1.0, -1.0, 16.0), Vec3::new(1.0, 1.0, 20.0), 15.0), 0.0, "too deep");
    // A box reaching from 10 m to 20 m deep counts up to the far plane: its near face, the same as when it ends there.
    let cut = projected_area(&c, Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 20.0), 15.0);
    let ends = projected_area(&c, Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 15.0), 100.0);
    assert!(cut > 0.0 && (cut - ends).abs() < 1e-4, "{cut} vs {ends}");
    // A box beyond the far plane but for a 1 m sliver: the hull of its face at 14 m and its cut at 15 m.
    let sliver = projected_area(&c, Vec3::new(2.0, 2.0, 14.0), Vec3::new(4.0, 4.0, 20.0), 15.0);
    assert!((sliver - 0.017_279).abs() < 1e-5, "{sliver}");
}
