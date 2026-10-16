use brush_recon::{MetricDepth, ReconFrame, RenderResult};
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use brush_triangle_train::loss::{LossConfig, triangle_loss};
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, UVec2, Vec2, Vec3};
use image::{Rgb, RgbImage};
use std::sync::Arc;

fn frame() -> ReconFrame {
    ReconFrame {
        id: 174,
        camera: Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            1.,
            1.,
            Vec2::splat(0.5),
            CameraModel::Pinhole,
        ),
        rgb: Arc::new(RgbImage::from_pixel(2, 2, Rgb([255, 0, 0]))),
        lidar: None,
        roi: Arc::new(vec![255, 0, 0, 0]),
    }
}
fn output(d: &Device) -> RenderResult {
    RenderResult {
        rgb: Tensor::full([2, 2, 3], 0.5, d),
        alpha: Tensor::zeros([2, 2], d),
        depth_m: None,
        normals: None,
    }
}
async fn scalar(t: Tensor<1>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()[0]
}

#[tokio::test(flavor = "current_thread")]
async fn roi_rgb_uses_three_hw_denominator_and_does_not_shrink_with_alpha() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let f = frame();
    let p = Tensor::zeros([3, 3], &d);
    let mut out = output(&d);
    let a = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    assert!((scalar(a.rgb).await - 0.125).abs() < 1e-6);
    assert!(a.depth_m.is_none());
    out.alpha = Tensor::ones([2, 2], &d);
    let b = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    assert!((scalar(b.total).await - 0.125).abs() < 1e-6);
}

#[tokio::test(flavor = "current_thread")]
async fn missing_prediction_pays_eight_metres_on_fixed_valid_target_denominator() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut f = frame();
    f.roi = Arc::new(vec![255; 4]);
    f.lidar = Some(Arc::new(MetricDepth {
        size: UVec2::new(2, 2),
        values_m: vec![2., 3., 4., 5.],
        confidence: Some(vec![2, 2, 0, 0]),
    }));
    let mut out = output(&d);
    out.depth_m = Some(Tensor::from_data(
        TensorData::new(vec![3., 0., 99., 99.], [2, 2]),
        &d,
    ));
    out.alpha = Tensor::from_data(TensorData::new(vec![1., 0., 1., 1.], [2, 2]), &d);
    let p = Tensor::zeros([3, 3], &d);
    let loss = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    assert!(
        (scalar(loss.depth_m.unwrap()).await - 4.5).abs() < 1e-6,
        "(1m error + 8m missing)/2 valid targets"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn empty_roi_rejects_but_no_valid_depth_remains_rgb_only() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut f = frame();
    let p = Tensor::zeros([3, 3], &d);
    let mut out = output(&d);
    f.roi = Arc::new(vec![0; 4]);
    assert!(triangle_loss(&out, &f, &p, &p, &LossConfig::default()).is_err());
    f.roi = Arc::new(vec![255; 4]);
    f.lidar = Some(Arc::new(MetricDepth {
        size: UVec2::new(2, 2),
        values_m: vec![2.; 4],
        confidence: Some(vec![0; 4]),
    }));
    out.depth_m = Some(Tensor::zeros([2, 2], &d));
    let l = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    assert!(l.depth_m.is_none());
    assert!((scalar(l.total).await - 0.5).abs() < 1e-6);
}

#[tokio::test(flavor = "current_thread")]
async fn seed_prior_and_depth_target_sampling_use_metric_units() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let f = frame();
    let p = Tensor::ones([1, 3], &d);
    let origin = Tensor::zeros([1, 3], &d);
    let l = triangle_loss(&output(&d), &f, &origin, &p, &LossConfig::default()).unwrap();
    assert!((scalar(l.seed).await - 1.).abs() < 1e-6);
    assert!((scalar(l.total).await - 0.126).abs() < 1e-6);
}

#[tokio::test(flavor = "current_thread")]
async fn nonfinite_missing_depth_is_finite_penalty_with_zero_depth_gradient() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut f = frame();
    f.lidar = Some(Arc::new(MetricDepth {
        size: UVec2::splat(2),
        values_m: vec![2.; 4],
        confidence: Some(vec![2; 4]),
    }));
    let mut out = output(&d);
    out.alpha = Tensor::ones([2, 2], &d);
    let prediction =
        Tensor::<2>::from_data(TensorData::new(vec![f32::NAN, 2., 2., 2.], [2, 2]), &d)
            .require_grad();
    out.depth_m = Some(prediction.clone());
    let p = Tensor::zeros([1, 3], &d);
    let terms = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    assert!((scalar(terms.depth_m.unwrap()).await - 8.).abs() < 1e-6);
    let g = terms.total.backward();
    let v = prediction
        .grad(&g)
        .unwrap()
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite() && *x == 0.));
}
