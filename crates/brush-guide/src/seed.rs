use crate::keyframe::DepthMap;
use crate::mono::MonoSeed;
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
    /// Phone mono depth with its fitted scale: fills grid pixels that LiDAR
    /// and feature points leave empty.
    pub mono: Option<MonoSeed<'a>>,
}

pub struct Seeds {
    pub means: Vec<f32>,
    pub colors: Vec<Vec3>,
    /// How many of the seeds came from mono depth.
    pub mono: usize,
}

const FEATURE_RADIUS_PX: f32 = 24.0;

/// A masked depth pixel is backfilled from a feature point only beyond
/// this depth: nearer masked pixels are usually reflective or dark
/// surfaces LiDAR flagged, where a feature 24 px away carries the wrong
/// depth, while beyond LiDAR's ~5 m working range a triangulated point
/// is the only signal there is.
const FEATURE_BACKFILL_MIN_DEPTH_M: f32 = 4.5;

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
    let radius = FEATURE_RADIUS_PX * to_alpha_px.x;

    // Nearest projected feature point within the radius, as depth.
    let feature_depth = |px: Vec2| {
        projected
            .iter()
            .map(|(p, d)| (p.distance(px), *d))
            .filter(|(dist, _)| *dist <= radius)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, d)| d)
    };

    let mut seeds = Seeds {
        means: Vec::new(),
        colors: Vec::new(),
        mono: 0,
    };
    let half = input.stride as f32 / 2.0;
    for y in (0..size.y).step_by(input.stride as usize) {
        for x in (0..size.x).step_by(input.stride as usize) {
            if input.alpha[(y * size.x + x) as usize] >= input.alpha_threshold {
                continue;
            }
            let px = Vec2::new(x as f32 + half, y as f32 + half).min(size.as_vec2() - 0.5);
            let uv = px / size.as_vec2();
            let measured = if let Some(d) = input.depth {
                // Trustworthy depth wins. Where the map has no value
                // (masked low-confidence, or beyond LiDAR range) a
                // projected feature point fills in, but only if it is
                // itself beyond FEATURE_BACKFILL_MIN_DEPTH_M: nearer
                // masked pixels are usually reflective/dark surfaces
                // LiDAR flagged, where a nearby feature carries the
                // wrong depth, so those are still skipped.
                d.sample_uv(uv.x, uv.y)
                    .or_else(|| feature_depth(px).filter(|fd| *fd > FEATURE_BACKFILL_MIN_DEPTH_M))
            } else {
                // No depth map at all: feature points first.
                feature_depth(px)
            };
            // Mono depth only where no measured source has a value. A pixel
            // with no source at all is skipped rather than guessed.
            let (depth, from_mono) = match measured {
                Some(d) => (d, false),
                None => match input.mono.as_ref().and_then(|m| m.depth_at(uv)) {
                    Some(d) => (d, true),
                    None => continue,
                },
            };
            seeds.mono += usize::from(from_mono);
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
    use crate::mono::MonoSeed;
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
            mono: None,
        }
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

    #[test]
    fn masked_depth_pixel_backfills_from_a_nearby_feature_point() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        // Pixel (1,1) (grid point at stride 2) has an invalid (0.0) depth
        // reading; the other grid points have valid LiDAR at 2.0. A feature
        // point at z=6 (beyond FEATURE_BACKFILL_MIN_DEPTH_M) projects
        // nearby: the masked pixel takes its depth, the valid ones keep
        // LiDAR's.
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = DepthMap {
            width: 4,
            height: 4,
            values,
            confidence: None,
        };
        let pts = [Vec3::new(0.0, 0.0, 6.0)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &pts));
        assert_eq!(seeds.means.len(), 4 * 3, "masked pixel is backfilled");
        let depths: Vec<f32> = seeds.means.chunks_exact(3).map(|p| p[2]).collect();
        assert_eq!(
            depths.iter().filter(|z| (**z - 2.0).abs() < 1e-4).count(),
            3,
            "LiDAR wins where it has a value: {depths:?}"
        );
        assert_eq!(
            depths.iter().filter(|z| (**z - 6.0).abs() < 1e-4).count(),
            1,
            "the masked pixel takes the feature depth: {depths:?}"
        );
    }

    #[test]
    fn masked_depth_pixel_backfill_is_gated_strictly_beyond_the_min_depth() {
        // The filter is a strict `>`: a feature point exactly at
        // FEATURE_BACKFILL_MIN_DEPTH_M is rejected, one just beyond it
        // backfills.
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = DepthMap {
            width: 4,
            height: 4,
            values,
            confidence: None,
        };

        let at_min = [Vec3::new(0.0, 0.0, FEATURE_BACKFILL_MIN_DEPTH_M)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &at_min));
        assert_eq!(
            seeds.means.len(),
            3 * 3,
            "exactly at the minimum is rejected"
        );

        let just_beyond = [Vec3::new(0.0, 0.0, FEATURE_BACKFILL_MIN_DEPTH_M + 0.1)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &just_beyond));
        assert_eq!(
            seeds.means.len(),
            4 * 3,
            "just beyond the minimum backfills"
        );
    }

    #[test]
    fn masked_depth_pixel_ignores_near_feature_points() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = DepthMap {
            width: 4,
            height: 4,
            values,
            confidence: None,
        };
        // Nearby feature (3 m < FEATURE_BACKFILL_MIN_DEPTH_M): indoors this
        // is the floater case the gate exists for; the masked pixel stays
        // unseeded.
        let pts = [Vec3::new(0.0, 0.0, 3.0)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &pts));
        assert_eq!(seeds.means.len(), 3 * 3);
        assert!(
            seeds
                .means
                .chunks_exact(3)
                .all(|p| (p[2] - 2.0).abs() < 1e-4)
        );
    }

    #[test]
    fn masked_depth_pixel_without_nearby_feature_point_is_skipped() {
        // Large rgb image scales FEATURE_RADIUS_PX down to ~0.24 alpha px
        // (same trick as the no-depth radius test), so the projected point
        // at alpha (0,0) is outside every grid centre's radius.
        let rgb = image::RgbImage::from_pixel(400, 400, image::Rgb([0, 0, 0]));
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = DepthMap {
            width: 4,
            height: 4,
            values,
            confidence: None,
        };
        let pts = [Vec3::new(-3.0, -3.0, 3.0)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &pts));
        assert_eq!(seeds.means.len(), 3 * 3, "no backfill outside the radius");
        assert!(
            seeds
                .means
                .chunks_exact(3)
                .all(|p| (p[2] - 2.0).abs() < 1e-4),
            "{:?}",
            seeds.means
        );
    }

    #[test]
    fn no_depth_feature_point_outside_radius_is_not_seeded() {
        // Use a large rgb image relative to the 4x4 alpha grid so
        // `FEATURE_RADIUS_PX` (defined in rgb-pixel units) maps down to a
        // small radius in alpha-grid space, letting a feature point that's
        // clearly still in view land well outside it.
        let rgb = image::RgbImage::from_pixel(400, 400, image::Rgb([0, 0, 0]));
        // Projects to alpha-space pixel (0, 0); nearest grid center is
        // (1, 1), a distance of sqrt(2) alpha px — far past the ~0.24 px
        // radius (24 rgb px scaled by 4/400). Previously this would have
        // fallen through to the median-depth fallback and seeded every
        // uncovered pixel anyway.
        let pts = [Vec3::new(-3.0, -3.0, 3.0)];
        let seeds = seed_points(&input(&[0.0; 16], &rgb, None, &pts));
        assert!(
            seeds.means.is_empty(),
            "no median fallback: {:?}",
            seeds.means
        );
    }

    fn mono_map(values: Vec<f32>) -> DepthMap {
        DepthMap {
            width: 4,
            height: 4,
            values,
            confidence: None,
        }
    }

    fn with_mono<'a>(
        base: SeedInput<'a>,
        map: &'a DepthMap,
        scale: f32,
        min_depth_m: f32,
    ) -> SeedInput<'a> {
        SeedInput {
            mono: Some(MonoSeed {
                depth: map,
                scale,
                min_depth_m,
            }),
            ..base
        }
    }

    fn depths(seeds: &Seeds) -> Vec<f32> {
        seeds.means.chunks_exact(3).map(|p| p[2]).collect()
    }

    #[test]
    fn lidar_wins_over_mono() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let depth = mono_map(vec![2.0; 16]);
        let mono = mono_map(vec![10.0; 16]);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &[]),
            &mono,
            1.0,
            0.0,
        ));
        assert_eq!(depths(&seeds), vec![2.0; 4]);
        assert_eq!(seeds.mono, 0);
    }

    #[test]
    fn feature_backfill_comes_before_mono() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = mono_map(values);
        let mono = mono_map(vec![10.0; 16]);
        let pts = [Vec3::new(0.0, 0.0, 6.0)];
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &pts),
            &mono,
            1.0,
            4.5,
        ));
        assert_eq!(seeds.mono, 0);
        assert_eq!(
            depths(&seeds)
                .iter()
                .filter(|z| (**z - 6.0).abs() < 1e-4)
                .count(),
            1
        );
    }

    #[test]
    fn with_lidar_mono_fills_masked_pixels_only_from_the_min_depth_on() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut values = vec![2.0; 16];
        values[4 + 1] = 0.0;
        let depth = mono_map(values);
        let far = mono_map(vec![10.0; 16]);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &[]),
            &far,
            1.0,
            4.5,
        ));
        assert_eq!(seeds.mono, 1);
        assert_eq!(
            depths(&seeds)
                .iter()
                .filter(|z| (**z - 10.0).abs() < 1e-4)
                .count(),
            1
        );

        let near = mono_map(vec![3.0; 16]);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &[]),
            &near,
            1.0,
            4.5,
        ));
        assert_eq!(
            (seeds.mono, seeds.colors.len()),
            (0, 3),
            "3 m is inside the LiDAR gate"
        );

        let at_gate = mono_map(vec![4.5; 16]);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &[]),
            &at_gate,
            1.0,
            4.5,
        ));
        assert_eq!(seeds.mono, 1, "the gate is inclusive");
    }

    #[test]
    fn without_lidar_mono_seeds_at_all_distances_after_feature_points() {
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mono = mono_map(vec![1.0; 16]);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, None, &[]),
            &mono,
            2.0,
            0.0,
        ));
        assert_eq!(depths(&seeds), vec![2.0; 4]);
        assert_eq!(seeds.mono, 4);

        let pts = [Vec3::new(0.0, 0.0, 3.0)];
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, None, &pts),
            &mono,
            2.0,
            0.0,
        ));
        assert!(
            depths(&seeds).iter().all(|z| (z - 3.0).abs() < 1e-4),
            "{:?}",
            depths(&seeds)
        );
        assert_eq!(seeds.mono, 0);
    }

    #[test]
    fn mono_beyond_100_m_and_invalid_mono_values_are_not_seeded() {
        // Grid centres sample map indices 5, 7, 13 and 15.
        let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut values = vec![0.0; 16];
        values[5] = 50.0;
        values[7] = 60.0;
        values[13] = f32::NAN;
        values[15] = -5.0;
        let mono = mono_map(values);
        let seeds = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, None, &[]),
            &mono,
            2.0,
            0.0,
        ));
        assert_eq!(
            depths(&seeds),
            vec![100.0],
            "100 m is the inclusive cap, 120 m is out"
        );
        assert_eq!(seeds.mono, 1);
    }

    /// Mono depth only adds seeds: dropping them leaves exactly the seeds of the same frame without mono.
    #[test]
    fn mono_only_adds_seeds_where_nothing_else_did() {
        // Large image: the feature radius shrinks to ~0.24 alpha px, so the point at alpha (1, 1)
        // backfills grid pixel (1, 1) only; (3, 3) is masked with no feature and takes mono.
        let rgb = image::RgbImage::from_fn(400, 400, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, 7])
        });
        let mut values = vec![2.0; 16];
        values[5] = 0.0;
        values[15] = 0.0;
        let depth = mono_map(values);
        let pts = [Vec3::new(-3.0, -3.0, 6.0)];
        let mono = mono_map(vec![20.0; 16]);
        let without = seed_points(&input(&[0.0; 16], &rgb, Some(&depth), &pts));
        let with = seed_points(&with_mono(
            input(&[0.0; 16], &rgb, Some(&depth), &pts),
            &mono,
            1.0,
            4.5,
        ));
        assert_eq!(
            (without.colors.len(), with.colors.len(), with.mono),
            (3, 4, 1)
        );
        let pairs = |s: &Seeds, skip_z: Option<f32>| -> Vec<(Vec<f32>, Vec3)> {
            s.means
                .chunks_exact(3)
                .zip(&s.colors)
                .filter(|(p, _)| skip_z.is_none_or(|z| (p[2] - z).abs() > 1e-4))
                .map(|(p, c)| (p.to_vec(), *c))
                .collect()
        };
        assert_eq!(pairs(&with, Some(20.0)), pairs(&without, None));
    }
}
