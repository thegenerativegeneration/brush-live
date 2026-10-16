//! Which views train in area training, and which cameras see a tile.

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use glam::{Vec2, Vec3};

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
    let Some(Extents { left, right, top, bottom }) = pinhole_extents(camera) else {
        return true;
    };
    let corners = local_corners(camera, min, max);
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

/// The share of the camera's image (0 to 1) that the part of the box `min`..`max` between the near plane and `far`
/// (camera depth) covers: the convex hull of that part's projected vertices (the box corners between the planes and
/// the points where box edges cross them), clipped to the image. Exact for a pinhole; a camera of another model is
/// taken as the pinhole of its field of view, and one with a field of view of 180° or more gets 1 for a box it sees.
pub fn projected_area(camera: &Camera, min: Vec3, max: Vec3, far: f32) -> f32 {
    let extents = if camera.fov_x >= std::f64::consts::PI || camera.fov_y >= std::f64::consts::PI {
        None
    } else {
        Some(extents_of(camera))
    };
    let Some(Extents { left, right, top, bottom }) = extents else {
        return if camera_sees_box(camera, min, max, far) { 1.0 } else { 0.0 };
    };
    let corners = local_corners(camera, min, max);
    let mut points: Vec<Vec3> = corners.iter().copied().filter(|p| (NEAR..=far).contains(&p.z)).collect();
    for (i, &a) in corners.iter().enumerate() {
        for bit in [1, 2, 4] {
            if i & bit != 0 {
                continue;
            }
            let b = corners[i | bit];
            for plane in [NEAR, far] {
                if (a.z - plane) * (b.z - plane) < 0.0 {
                    points.push(a.lerp(b, (plane - a.z) / (b.z - a.z)));
                }
            }
        }
    }
    let image: Vec<Vec2> = points
        .iter()
        .map(|p| Vec2::new((p.x / p.z - left) / (right - left), (p.y / p.z - top) / (bottom - top)))
        .collect();
    let clipped = clip_to_unit_square(convex_hull(image));
    polygon_area(&clipped).clamp(0.0, 1.0)
}

/// Image extents as x/z and y/z at depth 1: left, right, top, bottom.
struct Extents {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}

/// The pinhole extents of `camera`; `None` for another camera model or a field of view of 180° or more.
fn pinhole_extents(camera: &Camera) -> Option<Extents> {
    let pinhole = camera.camera_model == CameraModel::Pinhole
        && camera.fov_x < std::f64::consts::PI
        && camera.fov_y < std::f64::consts::PI;
    pinhole.then(|| extents_of(camera))
}

/// The full width spans 2·tan(fov/2), split at the principal point.
fn extents_of(camera: &Camera) -> Extents {
    let (wx, wy) = (2.0 * (camera.fov_x / 2.0).tan() as f32, 2.0 * (camera.fov_y / 2.0).tan() as f32);
    Extents {
        left: -camera.center_uv.x * wx,
        right: (1.0 - camera.center_uv.x) * wx,
        top: -camera.center_uv.y * wy,
        bottom: (1.0 - camera.center_uv.y) * wy,
    }
}

/// The box's eight corners in camera space; corner `i` takes max x for bit 1, max y for bit 2 and max z for bit 4.
fn local_corners(camera: &Camera, min: Vec3, max: Vec3) -> [Vec3; 8] {
    let to_local = camera.world_to_local();
    std::array::from_fn(|i| {
        let c = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        to_local.transform_point3(c)
    })
}

/// Andrew's monotone chain: the hull's vertices counter-clockwise (in x right, y up terms), collinear ones dropped.
fn convex_hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<Vec2> = Vec::with_capacity(2 * points.len());
    for pass in [points.clone(), points.into_iter().rev().collect()] {
        let start = hull.len();
        for p in pass {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Sutherland–Hodgman clipping of a convex polygon to [0, 1]².
fn clip_to_unit_square(polygon: Vec<Vec2>) -> Vec<Vec2> {
    let edges: [(usize, f32, bool); 4] = [(0, 0.0, true), (0, 1.0, false), (1, 0.0, true), (1, 1.0, false)];
    edges.iter().fold(polygon, |poly, &(axis, bound, keep_above)| {
        let inside = |p: Vec2| if keep_above { p[axis] >= bound } else { p[axis] <= bound };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for (i, &cur) in poly.iter().enumerate() {
            let prev = poly[(i + poly.len() - 1) % poly.len()];
            if inside(cur) != inside(prev) {
                out.push(prev.lerp(cur, (bound - prev[axis]) / (cur[axis] - prev[axis])));
            }
            if inside(cur) {
                out.push(cur);
            }
        }
        out
    })
}

/// Shoelace area of a simple polygon.
fn polygon_area(polygon: &[Vec2]) -> f32 {
    let n = polygon.len();
    (0..n).map(|i| polygon[i].perp_dot(polygon[(i + 1) % n])).sum::<f32>().abs() / 2.0
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
