use brush_triangle_train::optimizer::{DeviceAdam, cap_vertex_displacement};
use burn::tensor::{Device, Tensor, TensorData};

async fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().try_to_vec().unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn device_adam_matches_scalar_reference_and_retains_only_detached_state() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut x = Tensor::<1>::from_floats([1., -2., 3.], &d).require_grad();
    let mut adam = DeviceAdam::new(&x);
    let mut expected = [1.0_f64, -2., 3.];
    let mut m = [0.0_f64; 3];
    let mut v = [0.0_f64; 3];
    for (step, gradient) in [[0.1_f32, -0.2, 0.3], [-0.4, 0.1, 0.], [0., 0.25, -0.2]]
        .into_iter()
        .enumerate()
    {
        let g = Tensor::from_floats(gradient, &d);
        x = adam.update(&x, g, 0.03).await.unwrap();
        assert!(!x.is_autodiff());
        assert!(!x.is_require_grad());
        for i in 0..3 {
            m[i] = 0.9 * m[i] + 0.1 * gradient[i] as f64;
            v[i] = 0.999 * v[i] + 0.001 * (gradient[i] as f64).powi(2);
            expected[i] -= 0.03 * (m[i] / (1. - 0.9_f64.powi(step as i32 + 1)))
                / ((v[i] / (1. - 0.999_f64.powi(step as i32 + 1))).sqrt() + 1e-8);
        }
        for (a, b) in values(x.clone()).await.iter().zip(expected) {
            assert!((*a as f64 - b).abs() < 2e-5);
        }
    }
    assert_eq!(adam.step_count(), 3);
    assert_eq!(
        adam.scalar_readbacks(),
        3,
        "one finite-state scalar, no parameter/gradient readback"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn nonfinite_gradient_does_not_advance_or_poison_adam() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let x = Tensor::<1>::from_floats([1.], &d);
    let mut adam = DeviceAdam::new(&x);
    assert!(
        adam.update(&x, Tensor::from_floats([f32::NAN], &d), 0.01)
            .await
            .is_err()
    );
    assert_eq!(adam.step_count(), 0);
    let next = adam
        .update(&x, Tensor::from_floats([1.], &d), 0.01)
        .await
        .unwrap();
    assert!((values(next).await[0] - 0.99).abs() < 1e-6);
}

#[tokio::test(flavor = "current_thread")]
async fn vertex_cap_limits_vector_length_not_individual_coordinates() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let old = Tensor::<2>::zeros([2, 3], &d);
    let proposed = Tensor::from_data(TensorData::new(vec![3., 4., 0., 0., 0., 0.], [2, 3]), &d);
    let out = values(cap_vertex_displacement(old.clone(), proposed, 0.01).unwrap()).await;
    assert!((out[0] - 0.006).abs() < 1e-7 && (out[1] - 0.008).abs() < 1e-7);
    assert_eq!(out[3..], [0.; 3]);
    assert!(cap_vertex_displacement(old.clone(), old, 0.).is_err());
}
