use brush_recon::{BackendKind, MetricDepth};
use brush_recon_bench::metrics::*;
use glam::UVec2;

#[test]
fn rgb_mae_psnr_and_micro_aggregation_use_real_channel_counts() {
    let mut a = RgbSums::default();
    a.add(&[1.; 3], &[0; 3], None).unwrap();
    let mut b = RgbSums::default();
    b.add(&[0.; 9], &[0; 9], None).unwrap();
    a.merge(b);
    let x = a.finish().unwrap();
    assert_eq!(x.channels, 12);
    assert_eq!(x.mae, 0.25);
    assert_eq!(x.mse, 0.25);
    assert!((x.psnr.unwrap() - 6.020599913279624).abs() < 1e-10);
    assert!(!x.exact_match);
}

#[test]
fn immutable_roi_and_full_image_are_separate_and_exact_match_psnr_is_json_safe() {
    let mut roi = RgbSums::default();
    roi.add(&[0., 0., 0., 1., 1., 1.], &[0; 6], Some(&[255, 0]))
        .unwrap();
    let exact = roi.finish().unwrap();
    assert!(exact.exact_match);
    assert_eq!(exact.psnr, None);
    assert_eq!(exact.channels, 3);
    assert_eq!(
        serde_json::to_value(exact).unwrap()["psnr"],
        serde_json::Value::Null
    );
    let mut full = RgbSums::default();
    full.add(&[0., 0., 0., 1., 1., 1.], &[0; 6], None).unwrap();
    assert_eq!(full.finish().unwrap().mae, 0.5);
    assert!(RgbSums::default().finish().is_none());
}

#[test]
fn depth_counts_every_valid_native_target_and_penalizes_missing_predictions() {
    let target = MetricDepth {
        size: UVec2::new(2, 2),
        values_m: vec![2., 3., 4., 5.],
        confidence: Some(vec![2, 2, 0, 2]),
    };
    let mut s = DepthSums::default();
    s.add(&[3., 0., 100., 8.], &[1., 0., 1., 0.95], &target, None)
        .unwrap();
    let x = s.finish().unwrap();
    assert_eq!(x.valid_targets, 3);
    assert_eq!(x.available_predictions, 2);
    assert_eq!(x.high_alpha_predictions, 2);
    assert_eq!(x.mae_m, 4.);
    let mut roi = DepthSums::default();
    roi.add(
        &[3., 0., 100., 8.],
        &[1., 0., 1., 0.95],
        &target,
        Some(&[255, 255, 255, 0]),
    )
    .unwrap();
    assert_eq!(roi.finish().unwrap().mae_m, 4.5);
    let mut missing = DepthSums::default();
    missing
        .add(&[f32::NAN; 4], &[0.; 4], &target, None)
        .unwrap();
    assert_eq!(missing.finish().unwrap().mae_m, 8.);
}

#[test]
fn metrics_reject_shape_mismatch_and_nonfinite_rgb_instead_of_hiding_it() {
    assert!(
        RgbSums::default()
            .add(&[f32::NAN; 3], &[0; 3], None)
            .is_err()
    );
    assert!(RgbSums::default().add(&[0.; 3], &[0; 6], None).is_err());
    let t = MetricDepth {
        size: UVec2::ONE,
        values_m: vec![2.],
        confidence: Some(vec![2]),
    };
    assert!(DepthSums::default().add(&[], &[], &t, None).is_err());
}

fn aggregate(rgb: f64, depth: f64) -> AggregateMetrics {
    AggregateMetrics {
        roi_rgb: Some(RgbMetric {
            mae: rgb,
            mse: rgb * rgb,
            psnr: None,
            exact_match: rgb == 0.,
            channels: 3,
        }),
        depth: Some(DepthMetric {
            mae_m: depth,
            valid_targets: 1,
            available_predictions: 1,
            high_alpha_predictions: 1,
            available_fraction: 1.,
            high_alpha_fraction: 1.,
        }),
        ..Default::default()
    }
}

#[test]
fn research_gate_handles_zero_baseline_and_unsupported_gaussian_depth() {
    assert_eq!(
        research_gate(
            BackendKind::Triangle,
            &aggregate(0., 0.),
            &aggregate(1e-9, 1e-7)
        )
        .positive,
        Some(true)
    );
    assert_eq!(
        research_gate(
            BackendKind::Triangle,
            &aggregate(0., 0.),
            &aggregate(1e-6, 0.)
        )
        .positive,
        Some(false)
    );
    assert_eq!(
        research_gate(
            BackendKind::Triangle,
            &aggregate(1., 1.),
            &aggregate(0.9, 1.1)
        )
        .positive,
        Some(true)
    );
    assert_eq!(
        research_gate(
            BackendKind::Triangle,
            &aggregate(1., 1.),
            &aggregate(0.8, 1.11)
        )
        .positive,
        Some(false)
    );
    assert_eq!(
        research_gate(
            BackendKind::Gaussian,
            &aggregate(1., 1.),
            &aggregate(0.8, 1.)
        )
        .positive,
        None
    );
}
