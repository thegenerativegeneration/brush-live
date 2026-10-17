use brush_loss::{ImageLossConfig, image_loss};
use brush_recon::*;
use brush_recon_bench::{
    gaussian::{GaussianBackend, mask_scene_batch},
    triangle::TriangleBackend,
};
use brush_render::{AlphaMode, camera::Camera, kernels::camera_model::CameraModel};
use brush_triangle_render::TriangleSeed;
use brush_triangle_train::{
    TrainConfig,
    loss::{LossConfig, triangle_loss},
};
use burn::tensor::{Device, Int, Tensor, s};
use glam::{Quat, UVec2, Vec2, Vec3};
use image::{Rgb, RgbImage};
use std::sync::Arc;

fn seed() -> TriangleSeed {
    TriangleSeed {
        positions: vec![
            [-0.5, -0.5, 2.],
            [0.5, -0.5, 2.],
            [0., 0.5, 2.],
            [0.5, 0.5, 2.],
        ],
        faces: vec![[0, 1, 2], [1, 3, 2]],
        colors: vec![[0.8, 0.2, 0.3]; 4],
    }
}
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
        rgb: Arc::new(RgbImage::from_pixel(16, 16, Rgb([255, 0, 0]))),
        lidar: None,
        roi: Arc::new(vec![255; 256]),
    }
}
async fn scalar(t: Tensor<1>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()[0]
}

async fn exercise<B: ReconstructionBackend>(b: &mut B) {
    let f = Arc::new(frame());
    b.initialize(&[f.clone()]).await.unwrap();
    let r = RenderRequest {
        camera: f.camera,
        size: UVec2::splat(16),
        background: Vec3::ZERO,
        depth: false,
        normals: false,
    };
    b.render(&r).await.unwrap().validate(&r).unwrap();
    let stats = b
        .train_step(
            &f,
            StepContext {
                iteration: 0,
                progress: 0.,
            },
        )
        .await
        .unwrap();
    assert_eq!(stats.iteration, 1);
    assert!(stats.total_loss.is_finite());
    let mut unknown = (*f).clone();
    unknown.id = 999;
    assert!(
        b.train_step(
            &unknown,
            StepContext {
                iteration: 1,
                progress: 0.5
            }
        )
        .await
        .is_err()
    );
    b.synchronize().await.unwrap();
    let state = b.export_state().await.unwrap();
    assert_eq!(state.kind, b.kind());
    assert_eq!(state.format_version, 1);
    assert!(!state.payload.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn real_backends_obey_generic_contract_and_reject_heldout_training_ids() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut g = GaussianBackend::new(seed(), d.clone(), 0);
    exercise(&mut g).await;
    assert_eq!(g.stats().gaussians, 4);
    let f = frame();
    let r = RenderRequest {
        camera: f.camera,
        size: UVec2::splat(16),
        background: Vec3::ZERO,
        depth: true,
        normals: false,
    };
    assert!(matches!(
        g.render(&r).await,
        Err(ReconError::UnsupportedOutput("depth"))
    ));
    let mut t = TriangleBackend::new(seed(), d, TrainConfig::default());
    exercise(&mut t).await;
    assert_eq!(t.stats().vertices, 4);
    assert_eq!(t.stats().triangles, 2);
    let out = t.render(&r).await.unwrap();
    assert!(out.depth_m.is_some() && out.normals.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn sparse_roi_gaussian_loss_equals_triangle_rgb_loss_without_alpha_matching() {
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut f = frame();
    f.roi = Arc::new((0..256).map(|i| if i == 0 { 255 } else { 0 }).collect());
    let batch = mask_scene_batch(&f).unwrap();
    assert!(batch.has_alpha);
    assert_eq!(batch.alpha_mode, AlphaMode::Masked);
    let gt = Tensor::<2, Int>::from_data(batch.img_packed, &d.clone().inner());
    let rgba = Tensor::<3>::full([16, 16, 4], 0.5, &d);
    let old = image_loss(
        rgba.clone(),
        gt,
        ImageLossConfig {
            l1_weight: 1.,
            ssim_weight: 0.,
            composite_bg: None,
            mask: true,
            alpha_weight: 0.,
        },
    );
    let out = RenderResult {
        rgb: rgba.clone().slice(s![.., .., 0..3]),
        alpha: rgba.slice(s![.., .., 3..4]).reshape([16, 16]),
        depth_m: None,
        normals: None,
    };
    let p = Tensor::zeros([1, 3], &d);
    let new = triangle_loss(&out, &f, &p, &p, &LossConfig::default()).unwrap();
    let a = scalar(old).await;
    let b = scalar(new.rgb).await;
    assert!(
        (a - b).abs() < 1e-7 && (a - 0.5 / 256.).abs() < 1e-7,
        "Gaussian {a} triangle {b}"
    );
}
