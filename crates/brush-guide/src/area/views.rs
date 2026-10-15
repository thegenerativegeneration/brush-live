//! Which views train in area training, and which cameras see a tile.

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
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
/// still count as seen. The side planes follow the principal point. The planes are a pinhole's: a camera of any other
/// model (whose distortion can bend rays past them) or with a field of view of 180° or more sees every box.
pub fn camera_sees_box(camera: &Camera, min: Vec3, max: Vec3, far: f32) -> bool {
    if camera.camera_model != CameraModel::Pinhole
        || camera.fov_x >= std::f64::consts::PI
        || camera.fov_y >= std::f64::consts::PI
    {
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
#[path = "views_tests.rs"]
mod tests;
