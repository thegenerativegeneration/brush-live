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
}

/// Outcome of a keyframe's mono scale fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleFit {
    /// Mono seeding is off or the keyframe has no mono block.
    Absent,
    /// Too few samples from LiDAR and from feature points.
    TooFewSamples,
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
        };
        match self {
            Self::Absent => write!(f, "absent"),
            Self::TooFewSamples => write!(f, "none"),
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
/// do. A median outside `range` (inclusive) is rejected.
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
            return ScaleFit::TooFewSamples;
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
    fn too_few_samples_give_no_scale() {
        let mono = map(16, 12, |_, _| 2.0);
        let few = features(MIN_FEATURE_SAMPLES - 1, 5.0);
        assert_eq!(
            fit_scale(&mono, None, 2, &cam(), size(), &few, RANGE),
            ScaleFit::TooFewSamples
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
            ScaleFit::TooFewSamples,
            "points on invalid mono pixels do not count"
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
}
