use crate::ReconError;
use brush_render::camera::Camera;
use glam::{UVec2, Vec2, Vec3};

pub fn rasterize_roi(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    camera: &Camera,
    size: UVec2,
    near_m: f32,
) -> Result<Vec<u8>, ReconError> {
    crate::frame::validate_camera(camera)?;
    let n = crate::frame::pixel_count(size)?;
    if !near_m.is_finite()
        || near_m <= 0.
        || positions.iter().flatten().any(|v| !v.is_finite())
        || faces
            .iter()
            .flatten()
            .any(|i| *i as usize >= positions.len())
    {
        return Err(ReconError::input(
            "invalid hard-mask geometry or near plane",
        ));
    }
    if n as u64 > 512 * 1024 * 1024 {
        return Err(ReconError::BufferBudget {
            requested: n as u64,
            budget: 512 * 1024 * 1024,
        });
    }
    let mut mask = vec![0; n];
    let focal = camera.focal(size);
    let center = camera.center(size);
    for face in faces {
        let world = face.map(|i| Vec3::from(positions[i as usize]));
        if (world[1] - world[0]).cross(world[2] - world[0]).length() * 0.5 < 1e-8 {
            continue;
        }
        let local = world.map(|p| camera.world_to_local().transform_point3(p));
        if local.iter().any(|p| !p.is_finite() || p.z <= near_m) {
            continue;
        }
        let p = local.map(|v| v.truncate() / v.z * focal + center);
        if p.iter().any(|p| !p.is_finite()) {
            continue;
        }
        let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if area.abs() * 0.5 < 1e-6 {
            continue;
        }
        let min = p
            .iter()
            .fold(Vec2::splat(f32::INFINITY), |a, b| a.min(*b))
            .floor()
            .max(Vec2::ZERO)
            .min(size.as_vec2())
            .as_uvec2();
        let max = p
            .iter()
            .fold(Vec2::splat(f32::NEG_INFINITY), |a, b| a.max(*b))
            .ceil()
            .max(Vec2::ZERO)
            .min(size.as_vec2())
            .as_uvec2();
        for y in min.y..max.y {
            for x in min.x..max.x {
                let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                if (0..3)
                    .all(|i| (p[(i + 1) % 3] - p[i]).perp_dot(point - p[i]) * area.signum() >= 0.)
                {
                    mask[(y * size.x + x) as usize] = 255;
                }
            }
        }
    }
    Ok(mask)
}
