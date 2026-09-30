//! Synthetic depth scenes for tests: pinhole cameras, ray-cast depth
//! images and volumes fused from them.

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use glam::{Mat3, Quat, UVec2, Vec2, Vec3, vec3};

use super::Tsdf;
use crate::geometry::depth::DepthImage;

const SIZE: UVec2 = UVec2::new(128, 128);

/// Pinhole camera at `pos` looking at `target`, 60° fov; local +Z forward, +Y down.
pub(crate) fn look_at(pos: Vec3, target: Vec3) -> Camera {
    let fov = 60f64.to_radians();
    look_at_with(pos, target, fov, fov, Vec2::splat(0.5))
}

pub(super) fn look_at_with(
    pos: Vec3,
    target: Vec3,
    fov_x: f64,
    fov_y: f64,
    center_uv: Vec2,
) -> Camera {
    let forward = (target - pos).normalize();
    let right = Vec3::Y.cross(forward).normalize();
    let down = forward.cross(right);
    let rotation = Quat::from_mat3(&Mat3::from_cols(right, down, forward));
    Camera::new(pos, rotation, fov_x, fov_y, center_uv, CameraModel::Pinhole)
}

pub(crate) fn depth_image(camera: &Camera, hit: impl Fn(Vec3, Vec3) -> Option<f32>) -> DepthImage {
    depth_image_sized(camera, SIZE, hit)
}

/// Ray-casts `hit` per pixel centre. `hit(origin, dir)` returns the ray
/// parameter of the first hit; `dir` has unit camera-space z, so that
/// parameter is the depth along the camera's forward axis.
pub(super) fn depth_image_sized(
    camera: &Camera,
    size: UVec2,
    hit: impl Fn(Vec3, Vec3) -> Option<f32>,
) -> DepthImage {
    let (f, c) = (camera.focal(size), camera.center(size));
    let mut depth = Vec::new();
    for y in 0..size.y {
        for x in 0..size.x {
            let local = vec3(
                (x as f32 + 0.5 - c.x) / f.x,
                (y as f32 + 0.5 - c.y) / f.y,
                1.0,
            );
            let d = hit(camera.position, camera.rotation * local).unwrap_or(f32::NAN);
            depth.push(d);
        }
    }
    let alpha = depth
        .iter()
        .map(|d| if d.is_nan() { 0.0 } else { 1.0 })
        .collect();
    DepthImage {
        width: size.x,
        height: size.y,
        depth,
        alpha,
    }
}

pub(crate) fn plane_z(z: f32) -> impl Fn(Vec3, Vec3) -> Option<f32> {
    move |o, d| {
        let t = (z - o.z) / d.z;
        (t > 0.0).then_some(t)
    }
}

/// Axis-aligned box, entry point of the slab test.
pub(crate) fn aabb(min: Vec3, max: Vec3) -> impl Fn(Vec3, Vec3) -> Option<f32> {
    move |o, d| {
        let inv = d.recip();
        let (t0, t1) = ((min - o) * inv, (max - o) * inv);
        let near = t0.min(t1).max_element();
        let far = t0.max(t1).min_element();
        (near <= far && near > 0.0).then_some(near)
    }
}

/// Inside of an axis-aligned room: where a ray from inside leaves it.
fn room(min: Vec3, max: Vec3) -> impl Fn(Vec3, Vec3) -> Option<f32> {
    move |o, d| {
        let inv = d.recip();
        let (t0, t1) = ((min - o) * inv, (max - o) * inv);
        Some(t0.max(t1).min_element()).filter(|t| *t > 0.0)
    }
}

/// Nearest hit of two scenes.
pub(super) fn union(
    a: impl Fn(Vec3, Vec3) -> Option<f32>,
    b: impl Fn(Vec3, Vec3) -> Option<f32>,
) -> impl Fn(Vec3, Vec3) -> Option<f32> {
    move |o, d| match (a(o, d), b(o, d)) {
        (Some(s), Some(t)) => Some(s.min(t)),
        (s, t) => s.or(t),
    }
}

pub(crate) fn integrate(
    tsdf: &mut Tsdf,
    cameras: &[Camera],
    hit: &impl Fn(Vec3, Vec3) -> Option<f32>,
) {
    for cam in cameras {
        tsdf.integrate(&depth_image(cam, hit), cam);
    }
}

/// A 5 mm sheet at z = 2 m, 1.2 m square, fused from both sides.
pub(crate) fn thin_sheet_tsdf() -> Tsdf {
    let target = vec3(0.0, 0.0, 2.0);
    let cams = [
        look_at(Vec3::ZERO, target),
        look_at(vec3(0.0, 0.0, 4.0), target),
    ];
    let mut tsdf = Tsdf::new();
    let sheet = aabb(vec3(-0.6, -0.6, 1.9975), vec3(0.6, 0.6, 2.0025));
    integrate(&mut tsdf, &cams, &sheet);
    tsdf
}

/// A square rod of side `width` along y through `center`, 1 m long,
/// inside a 5 m room, fused from four sides at 1.5 m. The room walls give
/// every pixel a depth, so free space around the rod is observed as it
/// would be in a real scene.
pub(crate) fn rod_tsdf(center: Vec3, width: f32) -> Tsdf {
    let cams = [
        look_at(center - 1.5 * Vec3::Z, center),
        look_at(center + 1.5 * Vec3::X, center),
        look_at(center + 1.5 * Vec3::Z, center),
        look_at(center - 1.5 * Vec3::X, center),
    ];
    let mut tsdf = Tsdf::new();
    let half = vec3(0.5 * width, 0.5, 0.5 * width);
    let scene = union(
        aabb(center - half, center + half),
        room(center - 2.5, center + 2.5),
    );
    integrate(&mut tsdf, &cams, &scene);
    tsdf
}

/// Rod axes on a line of voxel centres and midway between them.
pub(crate) const ROD_CENTRES: [Vec3; 2] = [Vec3::new(0.025, 0.0, 2.025), Vec3::new(0.0, 0.0, 2.0)];

pub(super) fn plane_setup() -> (Tsdf, Camera) {
    let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
    let mut tsdf = Tsdf::new();
    tsdf.integrate(&depth_image(&cam, plane_z(2.5)), &cam);
    (tsdf, cam)
}
