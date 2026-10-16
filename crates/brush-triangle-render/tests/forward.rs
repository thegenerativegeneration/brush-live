use brush_recon::{ReconError, RenderRequest};
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use brush_triangle_render::{
    RenderConfig, TriangleModel, TriangleSeed,
    forward::required_buffer_bytes,
    reference::{ReferenceConfig, render_reference},
    render_triangles,
};
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, UVec2, Vec2, Vec3};

fn request(size: u32) -> RenderRequest {
    let fov = focal_to_fov(1., size, &CameraModel::Pinhole);
    RenderRequest {
        camera: Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            fov,
            fov,
            Vec2::ZERO,
            CameraModel::Pinhole,
        ),
        size: UVec2::splat(size),
        background: Vec3::new(0.2, 0.3, 0.4),
        depth: true,
        normals: true,
    }
}
fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![[0., 0., 1.], [8., 0., 2.], [0., 16., 4.]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
    }
}
async fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().try_to_vec().unwrap()
}

#[test]
fn counts_use_checked_arithmetic() {
    assert!(required_buffer_bytes(u64::MAX, 1, 1, 1, 1).is_err());
    assert!(required_buffer_bytes(1, u64::MAX, 1, 1, 1).is_err());
    assert!(required_buffer_bytes(1, 1, 1, 1, u64::MAX).is_err());
    assert!(required_buffer_bytes(3, 1, 16, 1, 1).unwrap() >= 16 * 8 * 4 + 32 * 4);
}

#[tokio::test(flavor = "current_thread")]
async fn tiled_forward_matches_cpu_for_tilt_winding_culling_and_partial_tiles() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    for case in 0..10 {
        let mut s = seed();
        match case {
            1 => s.faces[0] = [0, 2, 1],
            2 => s.positions[0][2] = 0.05,
            3 => s.faces[0] = [0, 0, 2],
            4 => s.faces.clear(),
            5 => s.positions.iter_mut().for_each(|p| p[0] += 100.),
            7 => {
                let a = 1.5 + 2_f32.sqrt();
                s.positions = vec![[-1., -1., 2.], [2. * a, -1., 2.], [-1., 2. * a, 2.]];
            }
            8 => s.positions = vec![[1., 0., 2.], [8., 0., 2.], [1., 8., 2.]],
            _ => {}
        }
        let r = request(if case == 6 {
            17
        } else if case == 7 {
            1
        } else {
            4
        });
        let mut m = TriangleModel::from_seed(&s, &d).unwrap();
        if case == 9 {
            m.weight_logits = Tensor::from_data(
                TensorData::new(vec![(0.2_f32 / 0.8).ln(), 0., (0.8_f32 / 0.2).ln()], [3]),
                &d,
            );
        }
        let colors: Vec<[f32; 3]> = values(m.rgb_logits.clone())
            .await
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let weights = values(m.weight_logits.clone()).await;
        let cfg = RenderConfig {
            sigma: 0.35,
            weight_floor: if case == 7 { 1. } else { 0.3 },
            ..Default::default()
        };
        let cpu = render_reference(
            &s,
            &colors,
            &weights,
            &r,
            &ReferenceConfig {
                sigma: cfg.sigma,
                weight_floor: cfg.weight_floor,
                near_m: cfg.near_m,
            },
        )
        .unwrap();
        let out = render_triangles(&m, &r, &cfg).await.unwrap();
        out.validate(&r).unwrap();
        for (name, a, b, tolerance) in [
            (
                "rgb",
                values(out.rgb).await,
                cpu.rgb.into_iter().flatten().collect::<Vec<_>>(),
                2e-4,
            ),
            ("alpha", values(out.alpha).await, cpu.alpha, 2e-4),
            (
                "depth",
                values(out.depth_m.unwrap()).await,
                cpu.depth_m,
                1e-3,
            ),
            (
                "normal",
                values(out.normals.unwrap()).await,
                cpu.normals.into_iter().flatten().collect::<Vec<_>>(),
                2e-4,
            ),
        ] {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    x.is_finite() && (x - y).abs() < tolerance,
                    "case {case} {name}[{i}] GPU{x} CPU{y}"
                );
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn intersection_budget_fails_before_allocation() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let m = TriangleModel::from_seed(&seed(), &d).unwrap();
    let cfg = RenderConfig {
        buffer_budget_bytes: 1,
        ..Default::default()
    };
    assert!(matches!(
        render_triangles(&m, &request(4), &cfg).await,
        Err(ReconError::BufferBudget { .. })
    ));
    // Base projection fits, but the actual nonzero intersection reservation does not.
    let cfg = RenderConfig {
        buffer_budget_bytes: required_buffer_bytes(3, 1, 16, 1, 0).unwrap(),
        ..Default::default()
    };
    assert!(matches!(
        render_triangles(&m, &request(4), &cfg).await,
        Err(ReconError::BufferBudget { .. })
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn equal_depth_order_is_deterministic_and_matches_cpu() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let mut s = seed();
    s.positions = vec![
        [0., 0., 2.],
        [8., 0., 2.],
        [0., 8., 2.],
        [0., 0., 2.],
        [8., 0., 2.],
        [0., 8., 2.],
    ];
    // More than one radix workgroup: equal-depth red IDs must precede blue IDs.
    s.faces = (0..600)
        .map(|i| if i < 300 { [0, 1, 2] } else { [3, 4, 5] })
        .collect();
    s.colors = vec![[0.8, 0.1, 0.1]; 3];
    s.colors.extend([[0.1, 0.1, 0.8]; 3]);
    let m = TriangleModel::from_seed(&s, &d).unwrap();
    let r = request(17);
    let cfg = RenderConfig::default();
    let c: Vec<[f32; 3]> = values(m.rgb_logits.clone())
        .await
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let w = values(m.weight_logits.clone()).await;
    let expected = render_reference(&s, &c, &w, &r, &ReferenceConfig::default()).unwrap();
    let first = values(render_triangles(&m, &r, &cfg).await.unwrap().rgb).await;
    let second = values(render_triangles(&m, &r, &cfg).await.unwrap().rgb).await;
    assert_eq!(first, second);
    for (a, b) in first.iter().zip(expected.rgb.into_iter().flatten()) {
        assert!((a - b).abs() < 2e-4);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn empty_model_uses_logical_counts_and_produces_only_background() {
    let d = Device::from(brush_cube::test_helpers::test_device().await);
    let s = TriangleSeed {
        positions: vec![],
        faces: vec![],
        colors: vec![],
    };
    let m = TriangleModel::from_seed(&s, &d).unwrap();
    assert_eq!(m.num_vertices(), 0);
    assert_eq!(m.num_faces(), 0);
    let r = request(17);
    let out = render_triangles(&m, &r, &RenderConfig::default())
        .await
        .unwrap();
    assert!(values(out.alpha).await.iter().all(|v| *v == 0.));
    assert!(values(out.depth_m.unwrap()).await.iter().all(|v| *v == 0.));
    let rgb = values(out.rgb).await;
    assert!(rgb.chunks_exact(3).all(|v| v == r.background.to_array()));
}
