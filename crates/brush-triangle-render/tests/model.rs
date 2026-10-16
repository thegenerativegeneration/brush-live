use brush_triangle_render::{TriangleModel, TriangleSeed};
use burn::tensor::Device;

fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0., 1., 0.5]; 3],
    }
}

#[test]
fn invalid_shapes_and_indices_are_rejected() {
    let mut s = seed();
    s.faces[0][0] = 3;
    assert!(s.validate().is_err());
    let mut s = seed();
    s.colors.pop();
    assert!(s.validate().is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn model_has_finite_logits_and_expected_tensor_layout() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let m = TriangleModel::from_seed(&seed(), &d).unwrap();
    assert_eq!(m.positions.dims(), [3, 3]);
    assert_eq!(m.faces().dims(), [1, 3]);
    assert_eq!(m.rgb_logits.dims(), [3, 3]);
    assert_eq!(m.weight_logits.dims(), [3]);
    let c = m
        .rgb_logits
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(c.iter().all(|v| v.is_finite()));
    let activated = 1. / (1. + (-c[0]).exp());
    assert!((activated - 1e-4).abs() < 1e-7);
    let w = m
        .weight_logits
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert!((1. / (1. + (-w[0]).exp()) - 0.1).abs() < 1e-7);
}
