use crate::keyframe::DepthMap;
use brush_render::camera::Camera;
use glam::{UVec2, Vec2, Vec3};

pub struct SeedInput<'a> {
    pub camera: &'a Camera,
    /// Rendered accumulated opacity, row-major, `alpha_size`.
    pub alpha: &'a [f32],
    pub alpha_size: UVec2,
    pub rgb: &'a image::RgbImage,
    pub depth: Option<&'a DepthMap>,
    /// ARKit feature points, world frame.
    pub points: &'a [Vec3],
    pub stride: u32,
    pub alpha_threshold: f32,
}

pub struct Seeds {
    pub means: Vec<f32>,
    pub colors: Vec<Vec3>,
}

const FEATURE_RADIUS_PX: f32 = 24.0;

pub fn project(camera: &Camera, size: UVec2, world: Vec3) -> Option<(Vec2, f32)> {
    let local = camera.world_to_local().transform_point3(world);
    if local.z <= 1e-4 {
        return None;
    }
    let f = camera.focal(size);
    let c = camera.center(size);
    let px = Vec2::new(local.x / local.z * f.x + c.x, local.y / local.z * f.y + c.y);
    let inside = px.x >= 0.0 && px.y >= 0.0 && px.x < size.x as f32 && px.y < size.y as f32;
    inside.then_some((px, local.z))
}

fn unproject(camera: &Camera, size: UVec2, px: Vec2, depth: f32) -> Vec3 {
    let f = camera.focal(size);
    let c = camera.center(size);
    let local = Vec3::new(
        (px.x - c.x) / f.x * depth,
        (px.y - c.y) / f.y * depth,
        depth,
    );
    camera.local_to_world().transform_point3(local)
}

pub fn seed_points(input: &SeedInput) -> Seeds {
    let size = input.alpha_size;
    let img_size = UVec2::new(input.rgb.width(), input.rgb.height());
    let to_alpha_px = size.as_vec2() / img_size.as_vec2();
    let projected: Vec<(Vec2, f32)> = input
        .points
        .iter()
        .filter_map(|p| project(input.camera, size, *p))
        .collect();
    let median = {
        let mut d: Vec<f32> = projected.iter().map(|p| p.1).collect();
        d.sort_by(f32::total_cmp);
        d.get(d.len() / 2).copied()
    };
    let radius = FEATURE_RADIUS_PX * to_alpha_px.x;

    let mut seeds = Seeds {
        means: Vec::new(),
        colors: Vec::new(),
    };
    let half = input.stride as f32 / 2.0;
    for y in (0..size.y).step_by(input.stride as usize) {
        for x in (0..size.x).step_by(input.stride as usize) {
            if input.alpha[(y * size.x + x) as usize] >= input.alpha_threshold {
                continue;
            }
            let px = Vec2::new(x as f32 + half, y as f32 + half).min(size.as_vec2() - 0.5);
            let uv = px / size.as_vec2();
            let depth = input
                .depth
                .and_then(|d| d.sample_uv(uv.x, uv.y))
                .or_else(|| {
                    projected
                        .iter()
                        .map(|(p, d)| (p.distance(px), *d))
                        .filter(|(dist, _)| *dist <= radius)
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|(_, d)| d)
                })
                .or(median);
            let Some(depth) = depth else { continue };
            seeds
                .means
                .extend(unproject(input.camera, size, px, depth).to_array());
            let ix = ((uv.x * img_size.x as f32) as u32).min(img_size.x - 1);
            let iy = ((uv.y * img_size.y as f32) as u32).min(img_size.y - 1);
            let c = input.rgb.get_pixel(ix, iy).0;
            seeds
                .colors
                .push(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0);
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;

    fn cam() -> Camera {
        let fov = 90f64.to_radians();
        Camera::new(
            Vec3::ZERO,
            glam::Quat::IDENTITY,
            fov,
            fov,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    }

    fn input<'a>(
        alpha: &'a [f32],
        rgb: &'a image::RgbImage,
        depth: Option<&'a DepthMap>,
        points: &'a [Vec3],
    ) -> SeedInput<'a> {
        SeedInput {
            camera: Box::leak(Box::new(cam())),
            alpha,
            alpha_size: UVec2::new(4, 4),
            rgb,
            depth,
            points,
            stride: 2,
            alpha_threshold: 0.5,
        }
    }

    #[test]
    fn project_roundtrip() {
        let (px, d) = project(&cam(), UVec2::new(100, 100), Vec3::new(0.0, 0.0, 2.0)).unwrap();
        assert!((px - Vec2::new(50.0, 50.0)).length() < 1e-3);
        assert!((d - 2.0).abs() < 1e-5);
        assert!(project(&cam(), UVec2::new(100, 100), Vec3::new(0.0, 0.0, -2.0)).is_none());
    }

    #[test]
    fn covered_pixels_get_no_seeds() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let depth = DepthMap {
            width: 4,
            height: 4,
            values: vec![2.0; 16],
            confidence: None,
        };
        let seeds = seed_points(&input(&[1.0; 16], &rgb, Some(&depth), &[]));
        assert!(seeds.means.is_empty());
    }

    #[test]
    fn lidar_depth_unprojects_on_grid() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([255, 0, 0]));
        let depth = DepthMap {
            width: 4,
            height: 4,
            values: vec![2.0; 16],
            confidence: None,
        };
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &[]));
        assert_eq!(seeds.means.len(), 4 * 3, "2x2 grid at stride 2");
        for p in seeds.means.chunks_exact(3) {
            assert!((p[2] - 2.0).abs() < 1e-4, "depth along +z: {p:?}");
        }
        assert!((seeds.colors[0] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn falls_back_to_feature_points_then_skips() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let pts = [Vec3::new(0.0, 0.0, 3.0)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, None, &pts));
        assert_eq!(seeds.means.len(), 4 * 3);
        assert!(
            seeds
                .means
                .chunks_exact(3)
                .all(|p| p[2] > 2.0 && p[2] < 3.5)
        );
        let none = seed_points(&input(&[0.0; 16], &rgb, None, &[]));
        assert!(none.means.is_empty());
    }
}
