use super::*;
use crate::config::GuideConfig;
use brush_render::kernels::camera_model::CameraModel;

const RANGE: (f32, f32) = (0.3, 3.0);

pub(super) fn cam() -> Camera {
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

pub(super) fn size() -> UVec2 {
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
    let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE, None);
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
    let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE, None);
    assert_eq!(fit.scale(), Some(2.0), "{fit:?}");
}

#[test]
fn low_confidence_lidar_is_left_out() {
    let mono = map(16, 12, |_, _| 2.0);
    let mut lidar = map(20, 15, |x, _| if x < 14 { 3.0 } else { 30.0 });
    lidar.confidence = Some((0..300).map(|i| if i % 20 < 14 { 2 } else { 1 }).collect());
    let fit = fit_scale(&mono, Some(&lidar), 2, &cam(), size(), &[], RANGE, None);
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
        None,
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
    let fit = fit_scale(&mono, None, 2, &cam(), size(), &few, RANGE, None);
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
    let fit = fit_scale(&mono, None, 2, &cam(), size(), &[], RANGE, None);
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
            RANGE,
            None
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
        None,
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
    let (seed, fit) = mono_seed_for(
        Some(&mono),
        Some(&lidar),
        &cam(),
        size(),
        &[],
        &config,
        None,
    );
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
    let (seed, _) = mono_seed_for(
        Some(&mono),
        Some(&lidar),
        &cam(),
        size(),
        &[],
        &config,
        None,
    );
    assert_eq!(seed.unwrap().depth_at(Vec2::new(0.5, 0.5)), Some(6.0));
    let mono = map(16, 12, |_, _| 3.0);
    let (seed, _) = mono_seed_for(Some(&mono), None, &cam(), size(), &[], &config, None);
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
            None,
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
    let (seed, fit) = mono_seed_for(
        Some(&mono),
        Some(&lidar),
        &cam(),
        size(),
        &pts,
        &config,
        None,
    );
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

    let (seed, _) = mono_seed_for(Some(&mono), None, &cam(), size(), &pts, &config, None);
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
    let (seed, fit) = mono_seed_for(Some(&mono), None, &cam(), size(), &pts, &off, None);
    assert!(seed.is_none());
    assert_eq!(fit, ScaleFit::Absent);

    let on = GuideConfig::default();
    let (seed, fit) = mono_seed_for(None, None, &cam(), size(), &pts, &on, None);
    assert!(seed.is_none());
    assert_eq!(fit, ScaleFit::Absent);

    let far = features(25, 50.0);
    let (seed, fit) = mono_seed_for(Some(&mono), None, &cam(), size(), &far, &on, None);
    assert!(seed.is_none());
    assert!(matches!(fit, ScaleFit::Rejected { scale, .. } if scale == 25.0));
}

fn fitted(source: ScaleSource, scale: f32) -> ScaleFit {
    ScaleFit::Fitted {
        source,
        scale,
        samples: 30,
    }
}

#[test]
fn too_few_samples_use_the_recent_scale() {
    let mono = map(16, 12, |_, _| 2.0);
    let few = features(MIN_FEATURE_SAMPLES - 1, 5.0);
    let fit = fit_scale(&mono, None, 2, &cam(), size(), &few, RANGE, Some(0.7));
    assert_eq!(
        fit,
        ScaleFit::Fitted {
            source: ScaleSource::Recent,
            scale: 0.7,
            samples: MIN_FEATURE_SAMPLES - 1
        }
    );
    assert_eq!(fit.to_string(), "recent 0.700 (19 samples)");
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
        &few,
        RANGE,
        Some(0.7),
    );
    assert!(matches!(
        fit,
        ScaleFit::Fitted {
            source: ScaleSource::Recent,
            samples: 50,
            ..
        }
    ));
    let fit = fit_scale(&mono, None, 2, &cam(), size(), &few, RANGE, None);
    assert!(
        matches!(fit, ScaleFit::Fitted { source: ScaleSource::Metric, scale, .. } if scale == 1.0)
    );
}

#[test]
fn enough_samples_ignore_the_recent_scale() {
    let mono = map(16, 12, |_, _| 2.0);
    let fit = fit_scale(
        &mono,
        None,
        2,
        &cam(),
        size(),
        &features(25, 5.0),
        RANGE,
        Some(0.7),
    );
    assert_eq!(
        fit,
        ScaleFit::Fitted {
            source: ScaleSource::Features,
            scale: 2.5,
            samples: 25
        }
    );
    let lidar = map(20, 15, |_, _| 3.0);
    let fit = fit_scale(
        &mono,
        Some(&lidar),
        2,
        &cam(),
        size(),
        &[],
        RANGE,
        Some(0.7),
    );
    assert_eq!(
        fit,
        ScaleFit::Fitted {
            source: ScaleSource::Lidar,
            scale: 1.5,
            samples: 300
        }
    );
}

#[test]
fn a_recent_scale_outside_the_range_is_used_as_given() {
    let mono = map(16, 12, |_, _| 2.0);
    let fit = fit_scale(&mono, None, 2, &cam(), size(), &[], RANGE, Some(5.0));
    assert_eq!(fit.scale(), Some(5.0));
    let config = GuideConfig::default();
    let (seed, fit) = mono_seed_for(Some(&mono), None, &cam(), size(), &[], &config, Some(0.1));
    assert!(matches!(
        fit,
        ScaleFit::Fitted {
            source: ScaleSource::Recent,
            ..
        }
    ));
    assert_eq!(seed.expect("fitted").scale, 0.1);
}

#[test]
fn history_is_empty_until_a_lidar_or_feature_fit() {
    let mut h = ScaleHistory::default();
    assert_eq!(h.median(), None);
    h.note(&fitted(ScaleSource::Recent, 0.5));
    h.note(&fitted(ScaleSource::Metric, 1.0));
    h.note(&ScaleFit::Rejected {
        source: ScaleSource::Lidar,
        scale: 9.0,
        samples: 300,
    });
    h.note(&ScaleFit::Absent);
    assert_eq!(h.median(), None);
    h.note(&fitted(ScaleSource::Lidar, 0.8));
    assert_eq!(h.median(), Some(0.8));
    h.note(&fitted(ScaleSource::Recent, 0.1));
    h.note(&fitted(ScaleSource::Metric, 1.0));
    assert_eq!(h.median(), Some(0.8));
    h.note(&fitted(ScaleSource::Features, 1.2));
    assert_eq!(h.median(), Some(1.2), "upper median of two");
}

#[test]
fn history_keeps_the_last_twenty_and_takes_the_upper_median() {
    let mut h = ScaleHistory::default();
    for i in 1..=25 {
        h.note(&fitted(ScaleSource::Features, i as f32));
    }
    // Kept: 6..=25; sorted[10] = 16.
    assert_eq!(ScaleHistory::LEN, 20);
    assert_eq!(h.median(), Some(16.0));
    let mut h = ScaleHistory::default();
    for v in [3.0, 1.0, 2.0] {
        h.note(&fitted(ScaleSource::Lidar, v));
    }
    assert_eq!(h.median(), Some(2.0));
}
