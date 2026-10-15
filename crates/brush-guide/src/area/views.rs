//! Which views train in area training, and which cameras see a tile.

use brush_render::camera::Camera;
use glam::Vec3;

/// Distance from a camera below which a box counts as behind it.
const NEAR: f32 = 0.01;

/// Indices of the `positions` within `radius` of `centre` horizontally (x and z; y is up), ascending.
pub fn area_views(positions: &[Vec3], centre: Vec3, radius: f32) -> Vec<usize> {
    positions
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.x - centre.x).hypot(p.z - centre.z) <= radius)
        .map(|(i, _)| i)
        .collect()
}

/// How many entries of `new` are missing from `old`; both ascending. Swapping the arguments counts the entries that
/// left.
pub fn membership_additions(old: &[usize], new: &[usize]) -> usize {
    new.iter().filter(|i| old.binary_search(i).is_err()).count()
}

/// Whether any part of the box `min`..`max` may lie in the camera's view frustum up to `far`. Conservative: false
/// only when all eight corners lie beyond one frustum plane, so a box that misses the frustum near a frustum edge can
/// still count as seen. The side planes follow the principal point; a field of view of 180° or more sees everything.
pub fn camera_sees_box(camera: &Camera, min: Vec3, max: Vec3, far: f32) -> bool {
    if camera.fov_x >= std::f64::consts::PI || camera.fov_y >= std::f64::consts::PI {
        return true;
    }
    // Image extents as x/z and y/z: the full width spans 2·tan(fov/2), split at the principal point.
    let (wx, wy) = (2.0 * (camera.fov_x / 2.0).tan() as f32, 2.0 * (camera.fov_y / 2.0).tan() as f32);
    let (left, right) = (-camera.center_uv.x * wx, (1.0 - camera.center_uv.x) * wx);
    let (top, bottom) = (-camera.center_uv.y * wy, (1.0 - camera.center_uv.y) * wy);
    let to_local = camera.world_to_local();
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            let c = Vec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            );
            to_local.transform_point3(c)
        })
        .collect();
    let outside: [&dyn Fn(Vec3) -> bool; 6] = [
        &|p| p.z < NEAR,
        &|p| p.z > far,
        &|p| p.x > p.z * right,
        &|p| p.x < p.z * left,
        &|p| p.y > p.z * bottom,
        &|p| p.y < p.z * top,
    ];
    !outside.iter().any(|beyond| corners.iter().all(|&p| beyond(p)))
}

/// The indices of the `cameras` that see the tile box `tile_min`..`tile_max` up to `far` (see [`camera_sees_box`]),
/// in the given order.
pub fn junction_views(cameras: &[(usize, Camera)], tile_min: Vec3, tile_max: Vec3, far: f32) -> Vec<usize> {
    cameras
        .iter()
        .filter(|(_, c)| camera_sees_box(c, tile_min, tile_max, far))
        .map(|(i, _)| *i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
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

    /// An off-centre principal point widens the frustum on one side: a box only that side sees is kept.
    #[test]
    fn frustum_box_test_follows_the_principal_point() {
        // tan(30°) ≈ 0.577; with the centre at u = 0.8 the left edge reaches x/z = −0.92 and the right +0.23.
        let c = Camera { center_uv: glam::vec2(0.8, 0.5), ..camera() };
        assert!(camera_sees_box(&c, Vec3::new(-8.5, -0.5, 9.9), Vec3::new(-8.0, 0.5, 10.0), 100.0), "left, wide side");
        assert!(!camera_sees_box(&c, Vec3::new(4.0, -0.5, 9.9), Vec3::new(4.5, 0.5, 10.0), 100.0), "right, narrow side");
    }
}
