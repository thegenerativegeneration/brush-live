//! The backward's per-splat importance `Σ_px (∂I/∂g)²` (Speedy-Splat) against
//! host references built from forward renders of the splats alone.

use brush_render::bwd::render_splats_with_pass;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, SplatRenderMode, Splats, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::{Device, Tensor, s};
use glam::{UVec2, Vec3};

/// SH DC 0 gives colour 0.5 per channel.
const C: f32 = 0.5;
/// `ALPHA_CUTOFF_MID` in brush-render's `kernels/helpers.rs`.
const CUT: f32 = 1.0 / 255.0;

async fn device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

/// Pinhole at the origin looking down +z, 60° field of view.
fn cam() -> Camera {
    let fov = 60f64.to_radians();
    Camera::new(
        Vec3::ZERO,
        glam::Quat::IDENTITY,
        fov,
        fov,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

const SIZE: UVec2 = UVec2::new(64, 64);

/// Splats at the given (x, y, z) with one log-scale and opacity each, grey, axis-aligned.
fn splats(points: &[(Vec3, f32, f32)], device: &Device) -> Splats {
    Splats::from_raw(
        points.iter().flat_map(|(p, _, _)| p.to_array()).collect(),
        [1.0, 0.0, 0.0, 0.0].repeat(points.len()),
        points.iter().flat_map(|(_, s, _)| [*s; 3]).collect(),
        vec![0.0; points.len() * 3],
        points.iter().map(|(_, _, o)| inverse_sigmoid(*o)).collect(),
        SplatRenderMode::Default,
        device,
    )
}

/// Per-pixel accumulated alpha of a forward render (channel 3 of `img`).
async fn alpha_map(s: Splats, cam: &Camera) -> Vec<f32> {
    let out = render_splats_with_pass(s, cam, SIZE, Vec3::ZERO, RasterPass::Backward).await;
    let img: Tensor<3> = out.img;
    let a = img.slice(s![.., .., 3..4]).flatten::<1>(0, 2);
    a.into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

/// Importance per splat from one render + backward of `img.sum()`.
async fn importance(s: Splats, cam: &Camera) -> Vec<f32> {
    let out = render_splats_with_pass(s, cam, SIZE, Vec3::ZERO, RasterPass::Backward).await;
    let grads = out.img.sum().backward();
    let g = out
        .importance_holder
        .grad(&grads)
        .expect("importance gradient");
    g.into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

fn in_branch(a: f32) -> bool {
    (CUT..=0.999).contains(&a)
}

#[tokio::test]
async fn single_splat_matches_alpha_map() {
    let device = device().await;
    let cam = cam();
    // Alone on black: ∂C/∂α = c per channel, so importance = o² · 3c² · #pixels in (CUT, 0.999].
    let o = 0.6;
    let s = splats(&[(Vec3::new(0.0, 0.0, 2.0), -2.5, o)], &device);
    let a = alpha_map(s.clone(), &cam).await;
    let n = a.iter().filter(|&&x| in_branch(x)).count() as f32;
    let expect = o * o * 3.0 * C * C * n;
    let got = importance(s, &cam).await[0];
    assert!(n > 10.0, "splat covers too few pixels: {n}");
    assert!((got - expect).abs() <= 2e-3 * expect, "{got} vs {expect}");
}

#[tokio::test]
async fn two_splats_match_front_to_back_derivatives() {
    let device = device().await;
    let cam = cam();
    // A in front of B, same grey: ∂C/∂α_A = c(1 − α_B), ∂C/∂α_B = c(1 − α_A).
    let (a, b) = (
        (Vec3::new(0.05, 0.0, 1.5), -2.6, 0.5),
        (Vec3::new(-0.05, 0.0, 2.5), -2.2, 0.7),
    );
    let am = alpha_map(splats(&[a], &device), &cam).await;
    let bm = alpha_map(splats(&[b], &device), &cam).await;
    let mut ea = 0.0;
    let mut eb = 0.0;
    for (&x, &y) in am.iter().zip(&bm) {
        if in_branch(x) {
            ea += (1.0 - y) * (1.0 - y);
        }
        if in_branch(y) {
            eb += (1.0 - x) * (1.0 - x);
        }
    }
    let (ea, eb) = (a.2 * a.2 * 3.0 * C * C * ea, b.2 * b.2 * 3.0 * C * C * eb);
    let got = importance(splats(&[a, b], &device), &cam).await;
    assert!((got[0] - ea).abs() <= 5e-3 * ea, "front {} vs {ea}", got[0]);
    assert!((got[1] - eb).abs() <= 5e-3 * eb, "back {} vs {eb}", got[1]);
}

#[tokio::test]
async fn splat_behind_opaque_layers_scores_zero() {
    let device = device().await;
    let cam = cam();
    // Two large near-opaque layers drop transmittance below the kernel's 1e-4
    // early-out across the hidden splat's whole footprint (~5 px radius): with
    // log-scale 0.5 the layers' σ is ~90 px, so T ≈ 5e-5 there.
    let front = |z| (Vec3::new(0.0, 0.0, z), 0.5, 0.995);
    let hidden = (Vec3::new(0.0, 0.0, 3.0), -2.5, 0.9);
    let got = importance(splats(&[front(1.0), front(1.2), hidden], &device), &cam).await;
    assert_eq!(got[2], 0.0);
    assert!(got[0] > 0.0);
}
