use brush_recon::{RenderRequest, RenderResult};
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use brush_triangle_render::{RenderConfig, TriangleModel, TriangleSeed, render_triangles};
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, UVec2, Vec2, Vec3};

fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![
            [-0.71, -0.63, 2.0],
            [0.82, -0.47, 2.3],
            [-0.54, 0.88, 2.1],
            [0.91, 0.79, 2.6],
        ],
        faces: vec![[0, 1, 2], [1, 3, 2]],
        colors: vec![
            [0.2, 0.3, 0.7],
            [0.8, 0.2, 0.4],
            [0.3, 0.7, 0.2],
            [0.6, 0.4, 0.8],
        ],
    }
}
fn request(shift: f32) -> RenderRequest {
    RenderRequest {
        camera: Camera::new(
            Vec3::new(shift, 0.031, 0.),
            Quat::IDENTITY,
            1.,
            1.,
            Vec2::new(0.49, 0.51),
            CameraModel::Pinhole,
        ),
        size: UVec2::splat(8),
        background: Vec3::new(0.11, 0.07, 0.13),
        depth: true,
        normals: true,
    }
}
fn parameters(s: &TriangleSeed) -> [Vec<f32>; 3] {
    [
        s.positions.iter().flatten().copied().collect(),
        s.colors
            .iter()
            .flatten()
            .map(|c| (c / (1. - c)).ln())
            .collect(),
        (0..s.positions.len())
            .map(|i| {
                let p = 0.2 + 0.15 * i as f32;
                (p / (1. - p)).ln()
            })
            .collect(),
    ]
}
fn model(s: &TriangleSeed, p: &[Vec<f32>; 3], d: &Device, tracked: bool) -> TriangleModel {
    let mut m = TriangleModel::from_seed(s, d).unwrap();
    let n = s.positions.len();
    m.positions = Tensor::from_data(TensorData::new(p[0].clone(), [n, 3]), d);
    m.rgb_logits = Tensor::from_data(TensorData::new(p[1].clone(), [n, 3]), d);
    m.weight_logits = Tensor::from_data(TensorData::new(p[2].clone(), [n]), d);
    if tracked {
        m.positions = m.positions.require_grad();
        m.rgb_logits = m.rgb_logits.require_grad();
        m.weight_logits = m.weight_logits.require_grad();
    }
    m
}
async fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().try_to_vec().unwrap()
}
fn objective(out: RenderResult, kind: usize) -> Tensor<1> {
    let d = out.rgb.device();
    let [h, w, _] = out.rgb.dims();
    let spatial = Tensor::<2>::from_data(
        TensorData::new(
            (0..h * w)
                .map(|i| 0.2 + (i * 17 % 53) as f32 / 53.)
                .collect::<Vec<_>>(),
            [h, w],
        ),
        &d,
    );
    match kind {
        0 => (out.rgb
            * spatial.unsqueeze_dim(2)
            * Tensor::<1>::from_floats([0.7, 0.3, 0.9], &d).reshape([1, 1, 3]))
        .mean(),
        1 => (out.depth_m.unwrap() * spatial).mean(),
        2 => (out.normals.unwrap()
            * spatial.unsqueeze_dim(2)
            * Tensor::<1>::from_floats([0.4, 0.6, 0.2], &d).reshape([1, 1, 3]))
        .mean(),
        _ => (out.alpha * spatial).mean(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn all_parameters_match_finite_differences_for_rgb_depth_normals_and_alpha() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    for overlap in [false, true] {
        let mut s = seed();
        if overlap {
            let front = s.positions[..3].to_vec();
            s.positions = front
                .iter()
                .copied()
                .chain(front.iter().map(|p| p.map(|x| x * 1.4)))
                .collect();
            s.faces = vec![[0, 1, 2], [3, 4, 5]];
            s.colors = vec![
                [0.2, 0.3, 0.7],
                [0.8, 0.2, 0.4],
                [0.3, 0.7, 0.2],
                [0.6, 0.4, 0.8],
                [0.25, 0.55, 0.35],
                [0.4, 0.65, 0.2],
            ];
        }
        let p = parameters(&s);
        let cfg = RenderConfig {
            sigma: 0.7,
            weight_floor: 0.2,
            ..Default::default()
        };
        for shift in [0., 0.073] {
            for kind in 0..4 {
                let r = request(shift);
                let m = model(&s, &p, &d, true);
                let output = render_triangles(&m, &r, &cfg).await.unwrap();
                assert!(
                    output.rgb.device().is_autodiff(),
                    "render must retain the parameter autodiff graph"
                );
                let grads = objective(output, kind).backward();
                let analytic = [
                    values(m.positions.grad(&grads).expect("position graph")).await,
                    values(m.rgb_logits.grad(&grads).expect("color graph")).await,
                    values(m.weight_logits.grad(&grads).expect("weight graph")).await,
                ];
                let mut nonzero = [0; 3];
                for group in 0..3 {
                    for i in 0..p[group].len() {
                        let mut losses = [0.; 2];
                        for (side, delta) in [-0.001, 0.001].into_iter().enumerate() {
                            let mut q = p.clone();
                            q[group][i] += delta;
                            let m = model(&s, &q, &d, false);
                            losses[side] = values(objective(
                                render_triangles(&m, &r, &cfg).await.unwrap(),
                                kind,
                            ))
                            .await[0];
                        }
                        let numeric = (losses[1] - losses[0]) / 0.002;
                        let a = analytic[group][i];
                        assert!(a.is_finite() && numeric.is_finite());
                        assert!(
                            (a - numeric).abs() <= 2e-4 + 0.03 * a.abs().max(numeric.abs()),
                            "view{shift} objective{kind} group{group} parameter{i}: analytic {a}, numeric {numeric}"
                        );
                        if a.abs() > 1e-6 {
                            nonzero[group] += 1;
                        }
                    }
                }
                assert!(nonzero[0] >= 8, "geometry must be exercised: {nonzero:?}");
                if kind == 0 {
                    assert_eq!(nonzero[1], s.positions.len() * 3);
                    assert_eq!(nonzero[2], 2, "two distinct min-weight winners");
                }
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn shared_vertex_gradients_equal_sum_of_unshared_corners() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let s = seed();
    let p = parameters(&s);
    let map = [0_usize, 1, 2, 1, 3, 2];
    let u = TriangleSeed {
        positions: map.map(|i| s.positions[i]).to_vec(),
        colors: map.map(|i| s.colors[i]).to_vec(),
        faces: vec![[0, 1, 2], [3, 4, 5]],
    };
    let q = [
        map.iter()
            .flat_map(|i| p[0][i * 3..i * 3 + 3].to_vec())
            .collect(),
        map.iter()
            .flat_map(|i| p[1][i * 3..i * 3 + 3].to_vec())
            .collect(),
        map.map(|i| p[2][i]).to_vec(),
    ];
    let a = model(&s, &p, &d, true);
    let b = model(&u, &q, &d, true);
    let r = request(0.);
    let cfg = RenderConfig::default();
    let ga = objective(render_triangles(&a, &r, &cfg).await.unwrap(), 0).backward();
    let gb = objective(render_triangles(&b, &r, &cfg).await.unwrap(), 0).backward();
    let a = [
        values(a.positions.grad(&ga).unwrap()).await,
        values(a.rgb_logits.grad(&ga).unwrap()).await,
        values(a.weight_logits.grad(&ga).unwrap()).await,
    ];
    let b = [
        values(b.positions.grad(&gb).unwrap()).await,
        values(b.rgb_logits.grad(&gb).unwrap()).await,
        values(b.weight_logits.grad(&gb).unwrap()).await,
    ];
    for group in 0..3 {
        let width = if group == 2 { 1 } else { 3 };
        let mut expected = vec![0.; 4 * width];
        for (corner, &vertex) in map.iter().enumerate() {
            for c in 0..width {
                expected[vertex * width + c] += b[group][corner * width + c];
            }
        }
        for (x, y) in a[group].iter().zip(expected) {
            assert!((x - y).abs() < 2e-5);
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn all_culled_has_defined_zero_parameter_gradients() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut s = seed();
    for v in &mut s.positions {
        v[2] = -2.;
    }
    let m = model(&s, &parameters(&s), &d, true);
    let g = objective(
        render_triangles(&m, &request(0.), &RenderConfig::default())
            .await
            .unwrap(),
        0,
    )
    .backward();
    for v in [
        values(m.positions.grad(&g).expect("zero position gradient")).await,
        values(m.rgb_logits.grad(&g).expect("zero color gradient")).await,
        values(m.weight_logits.grad(&g).expect("zero weight gradient")).await,
    ] {
        assert!(v.iter().all(|x| x.is_finite() && *x == 0.));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn capped_alpha_has_zero_geometry_and_weight_derivatives() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let a = 1.5 + 2_f32.sqrt();
    let s = TriangleSeed {
        positions: vec![[-1., -1., 2.], [2. * a, -1., 2.], [-1., 2. * a, 2.]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0.2, 0.4, 0.7]; 3],
    };
    let m = model(&s, &parameters(&s), &d, true);
    let mut r = request(0.);
    let fov = focal_to_fov(1., 1, &CameraModel::Pinhole);
    r.size = UVec2::ONE;
    r.camera = Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        fov,
        fov,
        Vec2::ZERO,
        CameraModel::Pinhole,
    );
    let out = render_triangles(
        &m,
        &r,
        &RenderConfig {
            sigma: 0.1,
            weight_floor: 0.999,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!((values(out.alpha.clone()).await[0] - 0.99).abs() < 1e-6);
    let g = out.alpha.sum().backward();
    for t in [
        values(m.positions.grad(&g).unwrap()).await,
        values(m.rgb_logits.grad(&g).unwrap()).await,
        values(m.weight_logits.grad(&g).unwrap()).await,
    ] {
        assert!(t.iter().all(|v| v.is_finite() && v.abs() < 1e-8));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn equal_minimum_weights_choose_lowest_global_vertex_id_not_corner_order() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut s = seed();
    s.faces = vec![[2, 1, 0], [3, 2, 1]];
    let mut p = parameters(&s);
    p[2] = vec![0.; 4];
    let m = model(&s, &p, &d, true);
    let g = render_triangles(&m, &request(0.), &RenderConfig::default())
        .await
        .unwrap()
        .alpha
        .sum()
        .backward();
    let w = values(m.weight_logits.grad(&g).unwrap()).await;
    assert!(w[0] > 1e-6 && w[1] > 1e-6);
    assert_eq!(w[2], 0.);
    assert_eq!(w[3], 0.);
}
