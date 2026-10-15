//! Monocular depth from the phone (MoGe-2): a per-keyframe scale fit to
//! LiDAR or feature points, and the seeding input built from it.

use crate::config::GuideConfig;
use crate::keyframe::DepthMap;
use crate::seed::project;
use brush_render::camera::Camera;
use glam::{UVec2, Vec2, Vec3};
use std::collections::VecDeque;
use std::fmt;

/// Fewest confident LiDAR pixels with a mono value for a LiDAR scale fit.
pub const MIN_LIDAR_SAMPLES: usize = 200;
/// Fewest in-view feature points with a mono value for a feature-point fit.
pub const MIN_FEATURE_SAMPLES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleSource {
    Lidar,
    Features,
    /// Too few samples for a fit and no scale fitted yet: the network's own
    /// metric scale (1.0).
    Metric,
    /// Too few samples for a fit: the median of the last fitted scales
    /// (`ScaleHistory`).
    Recent,
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
            ScaleSource::Recent => "recent",
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

/// The scales of the last `LEN` LiDAR or feature fits, for keyframes whose
/// own samples are too few. A scale relates the network's depth to camera
/// depth and does not depend on the segment's world, so `LiveModel::reset`
/// keeps its history: one history per engine, and the phone makes one
/// engine per capture.
#[derive(Clone, Debug, Default)]
pub struct ScaleHistory {
    scales: VecDeque<f32>,
}

impl ScaleHistory {
    pub const LEN: usize = 20;

    /// Records `fit` if it is an accepted LiDAR or feature-point fit; any
    /// other outcome (recent, metric, rejected, absent) leaves the history
    /// unchanged. Call once per keyframe.
    pub fn note(&mut self, fit: &ScaleFit) {
        if let ScaleFit::Fitted {
            source: ScaleSource::Lidar | ScaleSource::Features,
            scale,
            ..
        } = fit
        {
            if self.scales.len() == Self::LEN {
                self.scales.pop_front();
            }
            self.scales.push_back(*scale);
        }
    }

    /// Upper median of the kept scales, `None` while there are none.
    pub fn median(&self) -> Option<f32> {
        (!self.scales.is_empty()).then(|| median(self.scales.iter().copied().collect()))
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
/// do. With fewer of both, `recent` (the median of the last fitted scales) is
/// used as given, else the network's own metric scale (1.0); the samples are
/// those of the larger source. A median outside `range` (inclusive) is
/// rejected, with no fallback.
pub fn fit_scale(
    mono: &DepthMap,
    lidar: Option<&DepthMap>,
    min_confidence: u8,
    camera: &Camera,
    image_size: UVec2,
    points: &[Vec3],
    range: (f32, f32),
    recent: Option<f32>,
) -> ScaleFit {
    let from_lidar = lidar
        .map(|l| lidar_ratios(mono, l, min_confidence))
        .unwrap_or_default();
    let (source, ratios) = if from_lidar.len() >= MIN_LIDAR_SAMPLES {
        (ScaleSource::Lidar, from_lidar)
    } else {
        let from_features = feature_ratios(mono, camera, image_size, points);
        if from_features.len() < MIN_FEATURE_SAMPLES {
            let samples = from_lidar.len().max(from_features.len());
            return match recent {
                Some(scale) => ScaleFit::Fitted {
                    source: ScaleSource::Recent,
                    scale,
                    samples,
                },
                None => ScaleFit::Fitted {
                    source: ScaleSource::Metric,
                    scale: 1.0,
                    samples,
                },
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
#[derive(Clone, Copy)]
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
/// keyframe's LiDAR map (masked or not; confidence is checked here too);
/// `recent` is `ScaleHistory::median` for `fit_scale`.
pub fn mono_seed_for<'a>(
    mono: Option<&'a DepthMap>,
    lidar: Option<&DepthMap>,
    camera: &Camera,
    image_size: UVec2,
    points: &[Vec3],
    config: &GuideConfig,
    recent: Option<f32>,
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
        recent,
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
mod swift_fixture;

#[cfg(test)]
mod tests;
