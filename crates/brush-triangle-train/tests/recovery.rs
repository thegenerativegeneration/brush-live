use brush_recon::{ReconFrame, RenderRequest, RenderResult};
use brush_render::{camera::Camera, kernels::camera_model::CameraModel};
use brush_triangle_render::{
    RenderConfig, TriangleModel, TriangleSeed,
    reference::{ReferenceConfig, render_reference},
    render_triangles,
};
use brush_triangle_train::{
    TrainConfig, TriangleTrainer,
    optimizer::{DeviceAdam, cap_vertex_displacement},
    schedule::{ScheduleConfig, step_context},
};
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, UVec2, Vec2, Vec3};
use image::RgbImage;
use std::sync::Arc;

const SIZE: usize = 24;
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
            Vec2::splat(0.5),
            CameraModel::Pinhole,
        ),
        size: UVec2::splat(SIZE as u32),
        background: Vec3::ZERO,
        depth: true,
        normals: true,
    }
}
fn reference(
    seed: &TriangleSeed,
    r: &RenderRequest,
) -> brush_triangle_render::reference::ReferenceImage {
    let colors: Vec<_> = seed
        .colors
        .iter()
        .map(|c| c.map(|v| (v / (1. - v)).ln()))
        .collect();
    render_reference(
        seed,
        &colors,
        &vec![(0.1_f32 / 0.9).ln(); seed.positions.len()],
        r,
        &ReferenceConfig {
            sigma: 0.6,
            weight_floor: 0.8,
            ..Default::default()
        },
    )
    .unwrap()
}
async fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().try_to_vec().unwrap()
}
async fn scalar(t: Tensor<1>) -> f32 {
    values(t).await[0]
}

#[tokio::test(flavor = "current_thread")]
async fn fixed_topology_training_recovers_heldout_rgb_without_retaining_graphs() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let target_seed = seed();
    let frames: Vec<_> = [-0.25, 0., 0.25]
        .into_iter()
        .enumerate()
        .map(|(i, shift)| {
            let r = request(shift);
            let image = reference(&target_seed, &r);
            let bytes = image
                .rgb
                .iter()
                .flatten()
                .map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
                .collect();
            ReconFrame {
                id: i as u64,
                camera: r.camera,
                rgb: Arc::new(RgbImage::from_raw(SIZE as u32, SIZE as u32, bytes).unwrap()),
                lidar: None,
                roi: Arc::new(vec![255; SIZE * SIZE]),
            }
        })
        .collect();
    let mut initial = target_seed.clone();
    for (i, p) in initial.positions.iter_mut().enumerate() {
        p[0] += if i % 2 == 0 { 0.07 } else { -0.06 };
        p[1] -= 0.05;
        p[2] += 0.12;
    }
    for color in &mut initial.colors {
        for c in color {
            *c = *c * 0.75 + 0.08;
        }
    }
    let mut cfg = TrainConfig::default();
    cfg.vertex_lr = 0.005;
    cfg.rgb_lr = 0.03;
    cfg.loss.seed_weight = 0.;
    cfg.loss.depth_weight = 0.;
    cfg.schedule = ScheduleConfig {
        sigma_start: 0.6,
        sigma_end: 0.6,
        weight_floor_start: 0.8,
        weight_floor_end: 0.8,
    };
    let mut trainer = TriangleTrainer::new(
        TriangleModel::from_seed(&initial, &d).unwrap(),
        initial,
        cfg,
    )
    .unwrap();
    let r = request(0.125);
    let target = reference(&target_seed, &r);
    let target = Tensor::<3>::from_data(
        TensorData::new(
            target.rgb.into_iter().flatten().collect::<Vec<_>>(),
            [SIZE, SIZE, 3],
        ),
        &d.clone().inner(),
    );
    let before = scalar(
        (trainer.render(&r).await.unwrap().rgb - target.clone())
            .powi_scalar(2)
            .mean(),
    )
    .await;
    for i in 0..400 {
        trainer
            .step(&frames[i % 3], step_context(i as u64, 400).unwrap())
            .await
            .unwrap();
    }
    let after = scalar(
        (trainer.render(&r).await.unwrap().rgb - target)
            .powi_scalar(2)
            .mean(),
    )
    .await;
    assert!(
        after.is_finite() && after < before * 0.1,
        "held-out RGB MSE {before} -> {after}"
    );
    assert_eq!(trainer.model.num_vertices(), 4);
    assert_eq!(trainer.model.num_faces(), 2);
    assert!(
        !trainer.model.positions.is_autodiff()
            && !trainer.model.rgb_logits.is_autodiff()
            && !trainer.model.weight_logits.is_autodiff()
    );
    assert_eq!(
        trainer.optimizer_scalar_readbacks(),
        3 * 400,
        "finite-state scalars only; parameter and gradient arrays remain on the device"
    );
}

fn geometry_objective(
    out: RenderResult,
    depth: &Tensor<2>,
    normal: &Tensor<3>,
    mask: &Tensor<2>,
) -> (Tensor<1>, Tensor<1>) {
    let depth_loss = ((out.depth_m.unwrap() - depth.clone()).powi_scalar(2) * mask.clone()).sum()
        / mask.clone().sum();
    let cosine = (out.normals.unwrap() * normal.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    let normal_loss = ((cosine.neg() + 1.) * mask.clone()).sum() / mask.clone().sum();
    (depth_loss, normal_loss)
}

#[tokio::test(flavor = "current_thread")]
async fn synthetic_only_depth_normal_objective_recovers_a_tilted_plane() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let target = TriangleSeed {
        positions: vec![[-0.8, -0.6, 2.], [0.8, -0.6, 2.3], [-0.8, 0.8, 2.1]],
        faces: vec![[0, 1, 2]],
        colors: vec![[0.3, 0.4, 0.7]; 3],
    };
    let r = request(0.);
    let truth = reference(&target, &r);
    let depth = Tensor::from_data(TensorData::new(truth.depth_m, [SIZE, SIZE]), &d);
    let normal = Tensor::from_data(
        TensorData::new(
            truth.normals.into_iter().flatten().collect::<Vec<_>>(),
            [SIZE, SIZE, 3],
        ),
        &d,
    );
    let mask = Tensor::<2>::from_data(
        TensorData::new(
            truth
                .alpha
                .iter()
                .map(|a| if *a > 0.2 { 1. } else { 0. })
                .collect::<Vec<_>>(),
            [SIZE, SIZE],
        ),
        &d,
    );
    let mut initial = target.clone();
    for (i, z) in [0.10, -0.07, 0.08].into_iter().enumerate() {
        initial.positions[i][2] += z;
    }
    let mut model = TriangleModel::from_seed(&initial, &d).unwrap();
    let mut adam = DeviceAdam::new(&model.positions);
    let cfg = RenderConfig {
        sigma: 0.6,
        weight_floor: 0.8,
        ..Default::default()
    };
    let (a, b) = geometry_objective(
        render_triangles(&model, &r, &cfg).await.unwrap(),
        &depth,
        &normal,
        &mask,
    );
    let before = [scalar(a).await, scalar(b).await];
    for _ in 0..500 {
        model.positions = model.positions.detach().autodiff().require_grad();
        let (a, b) = geometry_objective(
            render_triangles(&model, &r, &cfg).await.unwrap(),
            &depth,
            &normal,
            &mask,
        );
        let g = (a + b).backward();
        let proposal = adam
            .update(&model.positions, model.positions.grad(&g).unwrap(), 0.005)
            .await
            .unwrap();
        model.positions = cap_vertex_displacement(model.positions.clone(), proposal, 0.01).unwrap();
    }
    let (a, b) = geometry_objective(
        render_triangles(&model, &r, &cfg).await.unwrap(),
        &depth,
        &normal,
        &mask,
    );
    let after = [scalar(a).await, scalar(b).await];
    assert!(
        after[0].is_finite() && after[0] < before[0] * 0.1,
        "depth MSE {before:?} -> {after:?}"
    );
    assert!(
        after[1].is_finite() && after[1] < before[1] * 0.1,
        "normal cosine error {before:?} -> {after:?}"
    );
}
