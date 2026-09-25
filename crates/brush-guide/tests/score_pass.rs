mod test_scene;

use brush_guide::scores::pass::{PassConfig, PassView, score_pass};
use brush_render::bwd::render_splats;
use burn::module::Module;
use burn::tensor::{Tensor, s};
use glam::{UVec2, Vec3};
use test_scene::{camera_at, device, splats_from};

const SIZE: UVec2 = UVec2::new(16, 16);

/// Exact `Σ_p Σ_c J Jᵀ` over `[mean, log_scale]` by one backward per pixel/channel.
async fn exact_fisher(
    splats: &brush_render::gaussian_splats::Splats,
    view: &PassView,
) -> Vec<[f32; 36]> {
    let n = splats.num_splats() as usize;
    let mut h = vec![[0f32; 36]; n];
    for y in 0..SIZE.y as usize {
        for x in 0..SIZE.x as usize {
            for c in 0..3 {
                let s = splats.clone().train();
                let out = render_splats(s.clone(), &view.camera, SIZE, Vec3::ZERO).await;
                let px: Tensor<1> = out.img.slice(s![y..y + 1, x..x + 1, c..c + 1]).reshape([1]);
                let mut grads = px.sum().backward();
                let g = s.transforms.val().grad_remove(&mut grads).unwrap();
                let j: Vec<f32> = Tensor::cat(
                    vec![g.clone().slice(s![.., 0..3]), g.slice(s![.., 7..10])],
                    1,
                )
                .into_data_async()
                .await
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
                for i in 0..n {
                    for a in 0..6 {
                        for b in 0..6 {
                            h[i][a * 6 + b] += j[i * 6 + a] * j[i * 6 + b];
                        }
                    }
                }
            }
        }
    }
    h
}

#[tokio::test]
async fn hutchinson_matches_exact_fisher() {
    let device = device().await.autodiff();
    let splats = splats_from(
        &[[0.0, 0.0, 2.0], [0.3, -0.2, 2.5], [-0.3, 0.2, 3.0]],
        -2.0,
        0.6,
        &device,
    );
    let view = PassView {
        camera: camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)),
        img_size: SIZE,
    };

    let exact = exact_fisher(&splats, &view).await;
    let cfg = PassConfig {
        hutchinson_samples: 4000,
        ..Default::default()
    };
    let est = score_pass(&splats, &[view], &cfg).await;

    for (e, h) in exact.iter().zip(&est.fisher) {
        let norm_e = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        let diff = e
            .iter()
            .zip(h)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(norm_e > 0.0);
        assert!(diff / norm_e < 0.1, "relative error {}", diff / norm_e);
    }
}

#[tokio::test]
async fn opposite_views_cancel_direction_sum() {
    let device = device().await.autodiff();
    let splats = splats_from(&[[0.0, 0.0, 0.0]], -2.0, 0.6, &device);
    let views = [
        PassView {
            camera: camera_at(Vec3::new(0.0, 0.0, -2.0), Vec3::ZERO),
            img_size: SIZE,
        },
        PassView {
            camera: camera_at(Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO),
            img_size: SIZE,
        },
    ];
    let out = score_pass(&splats, &views, &PassConfig::default()).await;
    assert_eq!(out.weight[0], 2.0);
    let s = Vec3::from(out.dir_sum[0]);
    assert!(s.length() < 1e-3, "{s}");
    assert!(out.max_px_per_m[0] > 0.0);
}

#[tokio::test]
async fn occluded_gaussian_is_not_observed() {
    let device = device().await.autodiff();
    // Three stacked large opaque splats hide a small one at z=3 from a camera at
    // the origin. One layer is not enough: per-splat alpha is clamped below 1.
    let mut splats = splats_from(
        &[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [0.0, 0.0, 1.2]],
        0.5,
        0.999,
        &device,
    );
    let hidden = splats_from(&[[0.0, 0.0, 3.0]], -3.0, 0.9, &device);
    splats.transforms = splats
        .transforms
        .map(|t| Tensor::cat(vec![t, hidden.transforms.val()], 0));
    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|t| Tensor::cat(vec![t, hidden.sh_coeffs.val()], 0));
    splats.raw_opacities = splats
        .raw_opacities
        .map(|t| Tensor::cat(vec![t, hidden.raw_opacities.val()], 0));

    let view = PassView {
        camera: camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)),
        img_size: SIZE,
    };
    let out = score_pass(&splats, &[view], &PassConfig::default()).await;
    assert_eq!(out.weight[0..3], [1.0, 1.0, 1.0]);
    assert_eq!(out.weight[3], 0.0);
}

#[tokio::test]
async fn empty_views_give_zeros() {
    let device = device().await.autodiff();
    let splats = splats_from(&[[0.0, 0.0, 2.0]], -2.0, 0.6, &device);
    let out = score_pass(&splats, &[], &PassConfig::default()).await;
    assert_eq!(out.weight, vec![0.0]);
    assert_eq!(out.fisher[0], [0.0; 36]);
}
