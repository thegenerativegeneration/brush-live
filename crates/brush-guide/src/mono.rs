//! Monocular depth from the phone (MoGe-2): a per-keyframe scale fit to
//! LiDAR or feature points, and the seeding input built from it.

use crate::config::GuideConfig;
use crate::keyframe::DepthMap;
use crate::seed::project;
use brush_render::camera::Camera;
use glam::{UVec2, Vec2, Vec3};
use std::fmt;

/// Fewest confident LiDAR pixels with a mono value for a LiDAR scale fit.
pub const MIN_LIDAR_SAMPLES: usize = 200;
/// Fewest in-view feature points with a mono value for a feature-point fit.
pub const MIN_FEATURE_SAMPLES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleSource {
    Lidar,
    Features,
    /// Too few samples for a fit: the network's own metric scale (1.0).
    Metric,
}

/// Outcome of a keyframe's mono scale fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleFit {
    /// Mono seeding is off or the keyframe has no mono block.
    Absent,
    /// A median ratio outside `mono_scale_range`.
    Rejected {
        source: ScaleSource,
        scale: f32,
        samples: usize,
    },
    Fitted {
        source: ScaleSource,
        scale: f32,
        samples: usize,
    },
}

impl ScaleFit {
    pub fn scale(&self) -> Option<f32> {
        match self {
            Self::Fitted { scale, .. } => Some(*scale),
            _ => None,
        }
    }
}

impl fmt::Display for ScaleFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |s: &ScaleSource| match s {
            ScaleSource::Lidar => "lidar",
            ScaleSource::Features => "features",
            ScaleSource::Metric => "metric",
        };
        match self {
            Self::Absent => write!(f, "absent"),
            Self::Rejected {
                source,
                scale,
                samples,
            } => write!(
                f,
                "rejected {} {scale:.3} ({samples} samples)",
                name(source)
            ),
            Self::Fitted {
                source,
                scale,
                samples,
            } => write!(f, "{} {scale:.3} ({samples} samples)", name(source)),
        }
    }
}

/// Upper median.
fn median(mut values: Vec<f32>) -> f32 {
    let mid = values.len() / 2;
    *values.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// LiDAR / mono at every LiDAR pixel with a positive finite value (and, with
/// a confidence map, confidence ≥ `min_confidence`) whose mono value at the
/// same image fraction is valid.
fn lidar_ratios(mono: &DepthMap, lidar: &DepthMap, min_confidence: u8) -> Vec<f32> {
    let mut ratios = Vec::new();
    for y in 0..lidar.height {
        for x in 0..lidar.width {
            let i = (y * lidar.width + x) as usize;
            if lidar
                .confidence
                .as_ref()
                .is_some_and(|c| c[i] < min_confidence)
            {
                continue;
            }
            let l = lidar.values[i];
            if !(l.is_finite() && l > 0.0) {
                continue;
            }
            let u = (x as f32 + 0.5) / lidar.width as f32;
            let v = (y as f32 + 0.5) / lidar.height as f32;
            if let Some(m) = mono.sample_uv(u, v) {
                ratios.push(l / m);
            }
        }
    }
    ratios
}

/// Camera depth / mono at every in-view feature point with a valid mono value.
fn feature_ratios(
    mono: &DepthMap,
    camera: &Camera,
    image_size: UVec2,
    points: &[Vec3],
) -> Vec<f32> {
    points
        .iter()
        .filter_map(|p| {
            let (px, z) = project(camera, image_size, *p)?;
            let uv = px / image_size.as_vec2();
            mono.sample_uv(uv.x, uv.y).map(|m| z / m)
        })
        .collect()
}

/// One scale for a keyframe's mono depth: the median LiDAR / mono ratio if
/// at least `MIN_LIDAR_SAMPLES` LiDAR pixels qualify, else the median
/// feature-point depth / mono ratio if at least `MIN_FEATURE_SAMPLES` points
/// do. With fewer of both, the network's own metric scale (1.0, samples of
/// the larger source) is trusted. A median outside `range` (inclusive) is
/// rejected, with no fallback.
pub fn fit_scale(
    mono: &DepthMap,
    lidar: Option<&DepthMap>,
    min_confidence: u8,
    camera: &Camera,
    image_size: UVec2,
    points: &[Vec3],
    range: (f32, f32),
) -> ScaleFit {
    let from_lidar = lidar
        .map(|l| lidar_ratios(mono, l, min_confidence))
        .unwrap_or_default();
    let (source, ratios) = if from_lidar.len() >= MIN_LIDAR_SAMPLES {
        (ScaleSource::Lidar, from_lidar)
    } else {
        let from_features = feature_ratios(mono, camera, image_size, points);
        if from_features.len() < MIN_FEATURE_SAMPLES {
            return ScaleFit::Fitted {
                source: ScaleSource::Metric,
                scale: 1.0,
                samples: from_lidar.len().max(from_features.len()),
            };
        }
        (ScaleSource::Features, from_features)
    };
    let samples = ratios.len();
    let scale = median(ratios);
    if scale.is_finite() && scale >= range.0 && scale <= range.1 {
        ScaleFit::Fitted {
            source,
            scale,
            samples,
        }
    } else {
        ScaleFit::Rejected {
            source,
            scale,
            samples,
        }
    }
}

/// Scaled mono depth beyond this is not seeded.
pub const MONO_MAX_DEPTH_M: f32 = 100.0;

/// Mono depth ready for seeding: the map, its fitted scale and the nearest
/// depth it may seed.
pub struct MonoSeed<'a> {
    pub depth: &'a DepthMap,
    pub scale: f32,
    /// `mono_min_depth_with_lidar_m` when the keyframe has a LiDAR map, else 0.
    pub min_depth_m: f32,
}

impl MonoSeed<'_> {
    /// Scaled mono depth at `uv` (fractions of the image), if valid and within
    /// `[min_depth_m, MONO_MAX_DEPTH_M]`.
    pub fn depth_at(&self, uv: Vec2) -> Option<f32> {
        let d = self.depth.sample_uv(uv.x, uv.y)? * self.scale;
        (d.is_finite() && d > 0.0 && d >= self.min_depth_m && d <= MONO_MAX_DEPTH_M).then_some(d)
    }
}

/// The keyframe's mono seeding input and the fit behind it. `lidar` is the
/// keyframe's LiDAR map (masked or not; confidence is checked here too).
pub fn mono_seed_for<'a>(
    mono: Option<&'a DepthMap>,
    lidar: Option<&DepthMap>,
    camera: &Camera,
    image_size: UVec2,
    points: &[Vec3],
    config: &GuideConfig,
) -> (Option<MonoSeed<'a>>, ScaleFit) {
    let Some(mono) = mono.filter(|_| config.mono_seeding) else {
        return (None, ScaleFit::Absent);
    };
    let fit = fit_scale(
        mono,
        lidar,
        config.min_depth_confidence,
        camera,
        image_size,
        points,
        config.mono_scale_range,
    );
    let seed = fit.scale().map(|scale| MonoSeed {
        depth: mono,
        scale,
        min_depth_m: if lidar.is_some() {
            config.mono_min_depth_with_lidar_m
        } else {
            0.0
        },
    });
    (seed, fit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GuideConfig;
    use brush_render::kernels::camera_model::CameraModel;

    const RANGE: (f32, f32) = (0.3, 3.0);

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

    fn size() -> UVec2 {
        UVec2::new(40, 30)
    }

    fn map(width: u32, height: u32, f: impl Fn(u32, u32) -> f32) -> DepthMap {
        let values = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        DepthMap {
            width,
            height,
            values,
            confidence: None,
        }
    }

    /// `n` (≤ 25) points at depth `z`, all inside the 90° view.
    fn features(n: usize, z: f32) -> Vec<Vec3> {
        (0..n)
            .map(|i| {
                let gx = (i % 5) as f32 - 2.0;
                let gy = (i / 5) as f32 - 2.0;
                Vec3::new(gx * 0.1 * z, gy * 0.1 * z, z)
            })
            .collect()
    }

    #[test]
    fn lidar_fit_is_the_median_ratio() {
        let mono = map(16, 12, |_, _| 2.0);
        let lidar = map(20, 15, |x, _| if x < 2 { 100.0 } else { 3.0 });
        let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE);
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Lidar,
                scale: 1.5,
                samples: 300
            }
        );
        assert_eq!(fit.scale(), Some(1.5));
    }

    /// Mono grid 10×5 against LiDAR 20×15 (different aspect): every LiDAR
    /// pixel is exactly twice the mono value at the same image fraction, so
    /// any other mapping moves the median off 2.
    #[test]
    fn lidar_fit_maps_pixels_by_image_fraction() {
        let mono = map(10, 5, |x, y| (x + 1) as f32 + 10.0 * y as f32);
        let lidar = map(20, 15, |x, y| {
            2.0 * ((x / 2 + 1) as f32 + 10.0 * (y / 3) as f32)
        });
        let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE);
        assert_eq!(fit.scale(), Some(2.0), "{fit:?}");
    }

    #[test]
    fn low_confidence_lidar_is_left_out() {
        let mono = map(16, 12, |_, _| 2.0);
        let mut lidar = map(20, 15, |x, _| if x < 14 { 3.0 } else { 30.0 });
        lidar.confidence = Some((0..300).map(|i| if i % 20 < 14 { 2 } else { 1 }).collect());
        let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE);
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Lidar,
                scale: 1.5,
                samples: 210
            }
        );
    }

    #[test]
    fn too_few_lidar_pixels_fall_back_to_features() {
        let mono = map(16, 12, |_, _| 2.0);
        let lidar = map(20, 15, |x, y| {
            if ((y * 20 + x) as usize) < MIN_LIDAR_SAMPLES - 1 {
                3.0
            } else {
                0.0
            }
        });
        let fit = fit_scale(
            &mono,
            Some(&lidar),
            2,
            &cam(),
            size(),
            &features(25, 5.0),
            RANGE,
        );
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Features,
                scale: 2.5,
                samples: 25
            }
        );
    }

    #[test]
    fn too_few_samples_trust_the_metric_scale() {
        let mono = map(16, 12, |_, _| 2.0);
        let few = features(MIN_FEATURE_SAMPLES - 1, 5.0);
        let fit = fit_scale(&mono, None, 2, &cam(), size(), &few, RANGE);
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Metric,
                scale: 1.0,
                samples: MIN_FEATURE_SAMPLES - 1
            }
        );
        assert_eq!(fit.scale(), Some(1.0));
        assert_eq!(fit.to_string(), "metric 1.000 (19 samples)");
        let fit = fit_scale(&mono, None, 2, &cam(), size(), &[], RANGE);
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Metric,
                scale: 1.0,
                samples: 0
            }
        );
        let invalid_mono = map(16, 12, |_, _| 0.0);
        assert_eq!(
            fit_scale(
                &invalid_mono,
                None,
                2,
                &cam(),
                size(),
                &features(25, 5.0),
                RANGE
            ),
            ScaleFit::Fitted {
                source: ScaleSource::Metric,
                scale: 1.0,
                samples: 0
            },
            "points on invalid mono pixels do not count"
        );
    }

    #[test]
    fn metric_sample_count_is_the_larger_source() {
        let mono = map(16, 12, |_, _| 2.0);
        let lidar = map(20, 15, |x, y| {
            if ((y * 20 + x) as usize) < 50 {
                3.0
            } else {
                0.0
            }
        });
        let fit = fit_scale(
            &mono,
            Some(&lidar),
            2,
            &cam(),
            size(),
            &features(10, 5.0),
            RANGE,
        );
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Metric,
                scale: 1.0,
                samples: 50
            }
        );
    }

    #[test]
    fn metric_seeds_keep_the_lidar_gate_and_the_cap() {
        let config = GuideConfig::default();
        let mono = map(16, 12, |x, _| if x < 8 { 3.0 } else { 150.0 });
        let lidar = map(20, 15, |_, _| 0.0);
        let (seed, fit) = mono_seed_for(Some(&mono), Some(&lidar), &cam(), size(), &[], &config);
        assert!(matches!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Metric,
                ..
            }
        ));
        let seed = seed.expect("fitted");
        assert_eq!(seed.scale, 1.0);
        assert_eq!(
            seed.depth_at(Vec2::new(0.1, 0.5)),
            None,
            "3 m < 4.5 m with a LiDAR map"
        );
        assert_eq!(seed.depth_at(Vec2::new(0.9, 0.5)), None, "150 m > cap");
        let mono = map(16, 12, |_, _| 6.0);
        let (seed, _) = mono_seed_for(Some(&mono), Some(&lidar), &cam(), size(), &[], &config);
        assert_eq!(seed.unwrap().depth_at(Vec2::new(0.5, 0.5)), Some(6.0));
        let mono = map(16, 12, |_, _| 3.0);
        let (seed, _) = mono_seed_for(Some(&mono), None, &cam(), size(), &[], &config);
        assert_eq!(
            seed.unwrap().depth_at(Vec2::new(0.5, 0.5)),
            Some(3.0),
            "no LiDAR map: all distances"
        );
    }

    #[test]
    fn scales_outside_the_range_are_rejected_and_the_bounds_accepted() {
        let mono = map(16, 12, |_, _| 1.0);
        let fit_for = |l: f32| {
            fit_scale(
                &mono,
                Some(&map(20, 15, |_, _| l)),
                2,
                &cam(),
                size(),
                &[],
                RANGE,
            )
        };
        assert_eq!(
            fit_for(5.0),
            ScaleFit::Rejected {
                source: ScaleSource::Lidar,
                scale: 5.0,
                samples: 300
            }
        );
        assert!(matches!(fit_for(0.2), ScaleFit::Rejected { .. }));
        assert_eq!(fit_for(3.0).scale(), Some(3.0));
        assert_eq!(fit_for(0.3).scale(), Some(0.3));
    }

    #[test]
    fn mostly_masked_lidar_fits_on_features_but_keeps_the_lidar_min_depth() {
        let config = GuideConfig::default();
        let mono = map(16, 12, |_, _| 2.0);
        let lidar = map(20, 15, |x, y| if x == 0 && y < 5 { 3.0 } else { 0.0 });
        let pts = features(25, 5.0);
        let (seed, fit) = mono_seed_for(Some(&mono), Some(&lidar), &cam(), size(), &pts, &config);
        assert_eq!(
            fit,
            ScaleFit::Fitted {
                source: ScaleSource::Features,
                scale: 2.5,
                samples: 25
            }
        );
        let seed = seed.expect("fitted");
        assert_eq!(seed.scale, 2.5);
        assert_eq!(seed.min_depth_m, config.mono_min_depth_with_lidar_m);

        let (seed, _) = mono_seed_for(Some(&mono), None, &cam(), size(), &pts, &config);
        assert_eq!(
            seed.expect("fitted").min_depth_m,
            0.0,
            "no LiDAR map: all distances"
        );
    }

    #[test]
    fn switched_off_absent_or_rejected_mono_gives_no_seed() {
        let mono = map(16, 12, |_, _| 2.0);
        let pts = features(25, 5.0);
        let off = GuideConfig {
            mono_seeding: false,
            ..GuideConfig::default()
        };
        let (seed, fit) = mono_seed_for(Some(&mono), None, &cam(), size(), &pts, &off);
        assert!(seed.is_none());
        assert_eq!(fit, ScaleFit::Absent);

        let on = GuideConfig::default();
        let (seed, fit) = mono_seed_for(None, None, &cam(), size(), &pts, &on);
        assert!(seed.is_none());
        assert_eq!(fit, ScaleFit::Absent);

        let far = features(25, 50.0);
        let (seed, fit) = mono_seed_for(Some(&mono), None, &cam(), size(), &far, &on);
        assert!(seed.is_none());
        assert!(matches!(fit, ScaleFit::Rejected { scale, .. } if scale == 25.0));
    }

    fn cam_at(position: Vec3, rotation: glam::Quat) -> Camera {
        let fov = 90f64.to_radians();
        Camera::new(position, rotation, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole)
    }

    /// LiDAR map at 40×30 whose pixel values are the mono value at the same image fraction times `factor`.
    fn lidar_from(mono: &DepthMap, factor: f32, confidence: Option<u8>) -> DepthMap {
        let values = (0..1200)
            .map(|i| {
                let (x, y) = (i % 40, i / 40);
                mono.sample_uv((x as f32 + 0.5) / 40.0, (y as f32 + 0.5) / 30.0).unwrap_or(1.0) * factor
            })
            .collect();
        DepthMap { width: 40, height: 30, values, confidence: confidence.map(|c| vec![c; 1200]) }
    }

    fn constant_mono(value: f32) -> DepthMap {
        DepthMap { width: 8, height: 6, values: vec![value; 48], confidence: None }
    }

    /// Mono map whose columns carry `columns`, repeated down 6 rows.
    fn column_mono(columns: [f32; 8]) -> DepthMap {
        DepthMap { width: 8, height: 6, values: (0..48).map(|i| columns[i % 8]).collect(), confidence: None }
    }

    /// Image-fraction samples at the centre of each of the eight mono columns, on the middle row.
    fn column_uvs() -> Vec<[f32; 2]> {
        (0..8).map(|c| [(c as f32 + 0.5) / 8.0, 0.5]).collect()
    }

    /// Feature points straight ahead of an identity camera whose depth over the (constant) mono value is `ratio`.
    fn feature_points(mono_value: f32, ratio: f32) -> Vec<Vec3> {
        (0..24).map(|i| Vec3::new((i % 5) as f32 * 0.01, (i / 5) as f32 * 0.01, mono_value * ratio)).collect()
    }

    struct Golden<'a> {
        mono: &'a DepthMap,
        lidar: Option<&'a DepthMap>,
        camera: &'a Camera,
        points: &'a [Vec3],
        uvs: &'a [[f32; 2]],
    }

    fn golden_case(g: &Golden, config: &GuideConfig) -> serde_json::Value {
        let (seed, fit) = mono_seed_for(Some(g.mono), g.lidar, g.camera, size(), g.points, config);
        let depth_at: Vec<Option<f32>> =
            g.uvs.iter().map(|uv| seed.as_ref().and_then(|s| s.depth_at(Vec2::new(uv[0], uv[1])))).collect();
        let (kind, source, scale, samples) = match fit {
            ScaleFit::Fitted { source, scale, samples } => ("fitted", source, scale, samples),
            ScaleFit::Rejected { source, scale, samples } => ("rejected", source, scale, samples),
            ScaleFit::Absent => unreachable!("mono is present and seeding is on"),
        };
        let source = match source {
            ScaleSource::Lidar => "lidar",
            ScaleSource::Features => "features",
            ScaleSource::Metric => "metric",
        };
        // The phone's camera looks down -z with y up: its camera-to-world has the engine's y and z axes negated.
        let rotation = glam::Mat3::from_quat(g.camera.rotation);
        let columns = [rotation.x_axis, -rotation.y_axis, -rotation.z_axis, g.camera.position];
        let camera_to_world: Vec<f32> =
            columns.iter().flat_map(|c| [c.x, c.y, c.z, if *c == g.camera.position { 1.0 } else { 0.0 }]).collect();
        serde_json::json!({
            "mono": { "width": g.mono.width, "height": g.mono.height, "values": g.mono.values },
            "lidar": g.lidar.map(|l| serde_json::json!({
                "width": l.width, "height": l.height, "values": l.values, "confidence": l.confidence
            })),
            "camera_to_world_phone_column_major": camera_to_world,
            "points_world": g.points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
            "fit": { "kind": kind, "source": source, "scale": scale, "samples": samples },
            "min_depth_m": seed.as_ref().map(|s| s.min_depth_m),
            "uvs": g.uvs,
            "depth_at": depth_at,
        })
    }

    /// Deterministic cases at the fit's and the seeding gate's boundaries.
    fn edge_cases(config: &GuideConfig) -> Vec<serde_json::Value> {
        let camera = cam();
        let min_depth = config.mono_min_depth_with_lidar_m;
        let (low, high) = config.mono_scale_range;
        let mut cases = Vec::new();
        let mut push = |mono: &DepthMap, lidar: Option<&DepthMap>, points: &[Vec3], uvs: &[[f32; 2]]| {
            cases.push(golden_case(&Golden { mono, lidar, camera: &camera, points, uvs }, config));
        };
        // Feature fits just outside the range, and a LiDAR fit exactly on each bound (inclusive).
        let flat = constant_mono(2.0);
        push(&flat, None, &feature_points(2.0, low * 0.9997), &column_uvs());
        push(&flat, None, &feature_points(2.0, high * 1.0001), &column_uvs());
        push(&flat, None, &feature_points(2.0, low * 1.0003), &column_uvs());
        let one = constant_mono(1.0);
        push(&one, Some(&lidar_from(&one, low, Some(2))), &[], &column_uvs());
        push(&one, Some(&lidar_from(&one, high, Some(2))), &[], &column_uvs());
        // Scale 2 over large values: 60 and 52 pass the 100 m cap on the wrong side, 50 sits on it.
        let large = column_mono([60.0, 52.0, 50.0, 49.9, 40.0, 2.0, 1.0, 0.1]);
        push(&large, Some(&lidar_from(&large, 2.0, Some(2))), &[], &column_uvs());
        // Scale 1 with depths around the LiDAR keyframe's minimum depth.
        let near = column_mono([
            min_depth * 0.9,
            min_depth * 0.999,
            min_depth,
            min_depth * 1.001,
            min_depth * 1.1,
            0.0,
            f32::MIN_POSITIVE,
            min_depth * 2.0,
        ]);
        push(&near, Some(&lidar_from(&near, 1.0, Some(2))), &[], &column_uvs());
        // The same depths without a LiDAR map have no minimum depth: the features fit scale 1.
        push(&near, None, &feature_points(min_depth * 2.0, 1.0), &column_uvs());
        // A LiDAR map without confidence counts every pixel.
        push(&near, Some(&lidar_from(&near, 1.5, None)), &[], &column_uvs());
        cases
    }

    /// Writes the golden cases of the phone's Swift port (`MonoScaleFit`) to the path in `SWIFT_FIXTURE_OUT`
    /// (`tools/fixtures/mono_scale_fit.json` of the app repo). Run with
    /// `SWIFT_FIXTURE_OUT=… tools/cargo-host.sh test --crate brush-guide -- write_swift_fixture --ignored`.
    #[test]
    #[ignore]
    fn write_swift_fixture() {
        let mut state = 0x2545_f491u32;
        let mut rnd = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        };
        let config = GuideConfig::default();
        let focal = cam().focal(size());
        let center = cam().center(size());
        let mut cases = Vec::new();
        for case in 0..36usize {
            let (mw, mh) = [(16u32, 12u32), (10, 5), (32, 24)][case % 3];
            let mono_values: Vec<f32> = (0..mw * mh)
                .map(|_| if rnd() < 0.1 { 0.0 } else { 0.5 + 3.0 * rnd() })
                .collect();
            let mono = DepthMap { width: mw, height: mh, values: mono_values, confidence: None };
            let ratio = 0.2 + 3.2 * rnd();
            let lidar_share = [1.0f32, 0.5, 0.1, 0.0][case % 4];
            // 40×30 LiDAR (1200 pixels): with `lidar_share` 1 and confidence 2 on 80 % of them, well over the 200
            // samples a LiDAR fit needs; the smaller shares fall back to features or the metric scale.
            let lidar = (case % 6 != 5).then(|| {
                let values: Vec<f32> = (0..1200)
                    .map(|i| {
                        let (x, y) = (i % 40, i / 40);
                        let m = mono.sample_uv((x as f32 + 0.5) / 40.0, (y as f32 + 0.5) / 30.0).unwrap_or(1.0);
                        if rnd() < lidar_share { m * ratio * (0.9 + 0.2 * rnd()) } else { 0.0 }
                    })
                    .collect();
                let confidence: Vec<u8> =
                    (0..1200).map(|_| if rnd() < 0.8 { 2 } else { (rnd() * 2.0) as u8 }).collect();
                DepthMap { width: 40, height: 30, values, confidence: Some(confidence) }
            });
            // Every third camera is the identity; the others are rotated and translated.
            let camera = if case % 3 == 0 {
                cam()
            } else {
                let axis = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
                let rotation = glam::Quat::from_axis_angle(axis, 6.0 * rnd() - 3.0);
                cam_at(Vec3::new(10.0 * rnd() - 5.0, 10.0 * rnd() - 5.0, 10.0 * rnd() - 5.0), rotation)
            };
            // Points in front of the camera, a fifth of them behind it, placed in world space.
            let n_points = [0usize, 10, 40, 60][(case / 4) % 4];
            let points: Vec<Vec3> = (0..n_points)
                .map(|_| {
                    let z = if rnd() < 0.2 { -1.0 - 3.0 * rnd() } else { 1.0 + 7.0 * rnd() };
                    let local = Vec3::new((rnd() - 0.5) * 1.6 * z, (rnd() - 0.5) * 1.6 * z, z);
                    camera.local_to_world().transform_point3(local)
                })
                .collect();
            let uvs: Vec<[f32; 2]> = (0..12).map(|_| [rnd(), rnd()]).collect();
            let g = Golden { mono: &mono, lidar: lidar.as_ref(), camera: &camera, points: &points, uvs: &uvs };
            cases.push(golden_case(&g, &config));
        }
        cases.extend(edge_cases(&config));
        let fixture = serde_json::json!({
            "image": { "width": size().x, "height": size().y, "fx": focal.x, "fy": focal.y, "cx": center.x, "cy": center.y },
            "min_confidence": config.min_depth_confidence,
            "range": [config.mono_scale_range.0, config.mono_scale_range.1],
            "min_depth_with_lidar_m": config.mono_min_depth_with_lidar_m,
            "case_count": cases.len(),
            "cases": cases,
        });
        let path = std::env::var("SWIFT_FIXTURE_OUT").expect("set SWIFT_FIXTURE_OUT to the fixture's path");
        std::fs::write(path, serde_json::to_string(&fixture).unwrap()).unwrap();
    }
}
