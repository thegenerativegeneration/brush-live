//! Tests of `sh_background.rs`.

use super::*;
use brush_render::kernels::camera_model::CameraModel::Pinhole;

const EPS: f32 = 1e-5;

async fn device() -> Device {
    brush_cube::test_helpers::test_device().await.into()
}

fn dirs_tensor_on(dirs: &[Vec3], device: &Device) -> Tensor<2> {
    let flat: Vec<f32> = dirs.iter().flat_map(|d| [d.x, d.y, d.z]).collect();
    Tensor::<1>::from_floats(flat.as_slice(), device).reshape([dirs.len(), 3])
}

async fn dirs_tensor(dirs: &[Vec3]) -> Tensor<2> {
    dirs_tensor_on(dirs, &device().await)
}

async fn row(t: &Tensor<2>, i: usize) -> [f32; 9] {
    let data: Vec<f32> = t
        .clone()
        .into_data_async()
        .await
        .expect("tensor data")
        .try_into_vec()
        .expect("tensor data");
    let mut out = [0.0; 9];
    out.copy_from_slice(&data[i * 9..i * 9 + 9]);
    out
}

#[tokio::test]
async fn sh_basis_matches_analytic_axis_values() {
    let dirs = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    let basis = sh_basis(dirs_tensor(&dirs).await);

    for i in 0..6 {
        let b = row(&basis, i).await;
        assert!((b[0] - SH_C0).abs() < EPS, "band0 const at dir {i}: {b:?}");
    }

    // +x: b1_0 = -SH_C1*y = 0, b1_1 = SH_C1*z = 0, b1_2 = -SH_C1*x = -SH_C1
    let b = row(&basis, 0).await;
    assert!((b[1] - 0.0).abs() < EPS);
    assert!((b[2] - 0.0).abs() < EPS);
    assert!((b[3] - (-SH_C1)).abs() < EPS);
    // band2 x^2-y^2 term = SH_C2_XY at +-x
    assert!((b[8] - SH_C2_XY).abs() < EPS);

    // +y: b1_0 = -SH_C1*y = -SH_C1
    let b = row(&basis, 2).await;
    assert!((b[1] - (-SH_C1)).abs() < EPS);
    assert!((b[8] - (-SH_C2_XY)).abs() < EPS);

    // +z: b1_1 = SH_C1*z = SH_C1; band2 z^2 term = SH_C2_Z2 + SH_C2_Z2_OFFSET
    let b = row(&basis, 4).await;
    assert!((b[2] - SH_C1).abs() < EPS);
    assert!((b[6] - (SH_C2_Z2 + SH_C2_Z2_OFFSET)).abs() < EPS);

    // -z: b1_1 = -SH_C1
    let b = row(&basis, 5).await;
    assert!((b[2] - (-SH_C1)).abs() < EPS);
}

fn test_camera(rotation: glam::Quat) -> Camera {
    Camera::new(
        Vec3::ZERO,
        rotation,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

#[test]
fn pixel_dirs_center_is_forward_axis() {
    let size = UVec2::new(101, 101);
    let cam = test_camera(glam::Quat::IDENTITY);
    let dirs = pixel_dirs(&cam, size);
    // Centre pixel (50, 50): index 50*101+50.
    let idx = (50 * 101 + 50) * 3;
    let d = Vec3::new(dirs[idx], dirs[idx + 1], dirs[idx + 2]);
    assert!((d - Vec3::Z).length() < 1e-4, "centre dir {d:?}");
}

#[test]
fn pixel_dirs_corner_is_near_45_degrees() {
    let size = UVec2::new(101, 101);
    let cam = test_camera(glam::Quat::IDENTITY);
    let dirs = pixel_dirs(&cam, size);
    // Left edge, vertical centre: pixel (0, 50).
    let idx = (50 * 101 + 0) * 3;
    let d = Vec3::new(dirs[idx], dirs[idx + 1], dirs[idx + 2]);
    let angle = d.dot(Vec3::Z).acos().to_degrees();
    assert!((angle - 45.0).abs() < 1.0, "corner angle {angle}");
}

#[test]
fn pixel_dirs_flips_under_180_degree_yaw() {
    let size = UVec2::new(101, 101);
    let cam = test_camera(glam::Quat::from_rotation_y(std::f32::consts::PI));
    let dirs = pixel_dirs(&cam, size);
    let idx = (50 * 101 + 50) * 3;
    let d = Vec3::new(dirs[idx], dirs[idx + 1], dirs[idx + 2]);
    assert!((d - Vec3::NEG_Z).length() < 1e-3, "flipped centre dir {d:?}");
}

#[test]
fn pixel_dirs_90_degree_yaw_moves_centre_to_plus_x() {
    // Catches a transposed/inverted rotation that a 180-degree test
    // can't: 180 degrees is its own inverse, so `rotation.inverse() *
    // local` (world-to-camera, the wrong direction) would also pass
    // that test. A 90-degree yaw breaks the symmetry: camera-to-world
    // rotation puts the centre ray at +X; the inverse would give -X.
    let size = UVec2::new(101, 101);
    let cam = test_camera(glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
    let dirs = pixel_dirs(&cam, size);
    let idx = (50 * 101 + 50) * 3;
    let d = Vec3::new(dirs[idx], dirs[idx + 1], dirs[idx + 2]);
    assert!((d - Vec3::X).length() < 1e-3, "90-degree-yaw centre dir {d:?}");
}

#[test]
#[should_panic(expected = "pixel_dirs only supports a pinhole camera_model")]
fn pixel_dirs_rejects_non_pinhole_cameras() {
    let cam = Camera::new(
        Vec3::ZERO,
        glam::Quat::IDENTITY,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
        glam::vec2(0.5, 0.5),
        CameraModel::KannalaBrandt4(Default::default()),
    );
    let _ = pixel_dirs(&cam, UVec2::new(16, 16));
}

#[tokio::test]
async fn sh_background_starts_uniform_grey() {
    let device = device().await.autodiff();
    let bg = ShBackground::new(&device, 0.2);
    let dirs = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Y];
    let basis = sh_basis(dirs_tensor_on(&dirs, &device));
    let image = bg.image(basis);
    let data: Vec<f32> = image
        .into_data_async()
        .await
        .expect("tensor data")
        .try_into_vec()
        .expect("tensor data");
    for v in data {
        assert!((v - 0.5).abs() < EPS, "expected grey, got {v}");
    }
}

async fn read(t: Tensor<2>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("tensor data")
        .try_into_vec()
        .expect("tensor data")
}

#[tokio::test]
async fn world_dirs_match_pixel_dirs_for_a_rotated_off_centre_camera() {
    let device = device().await;
    let rotation = glam::Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.5).normalize(), 0.65);
    let cam = Camera::new(
        Vec3::new(1.0, -2.0, 0.5),
        rotation,
        1.1,
        0.8,
        glam::vec2(0.42, 0.57),
        Pinhole,
    );
    let size = UVec2::new(37, 23);
    let expected = pixel_dirs(&cam, size);
    let got = read(world_dirs(pixel_centres(size, &device), &cam, size)).await;
    assert_eq!(got.len(), expected.len());
    let worst = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "max deviation from pixel_dirs: {worst}");
}

#[tokio::test]
async fn background_match_mask_keeps_matching_blocks_and_drops_isolated_pixels() {
    let device = device().await;
    let (h, w) = (8usize, 8usize);
    let bg = Tensor::<3>::full([h, w, 3], 0.5, &device);
    // GT equals bg in the left half; the right half differs except for a
    // single matching pixel at (2, 6).
    let mut gt = vec![0.5f32; h * w * 3];
    for y in 0..h {
        for x in w / 2..w {
            if (y, x) != (2, 6) {
                for c in 0..3 {
                    gt[(y * w + x) * 3 + c] = 0.9;
                }
            }
        }
    }
    let gt = Tensor::<1>::from_floats(gt.as_slice(), &device).reshape([h, w, 3]);
    let mask = read(background_match_mask(gt, bg)).await;
    let at = |y: usize, x: usize| mask[y * w + x];
    assert_eq!(at(4, 1), 1.0, "interior of the matching block");
    assert_eq!(at(4, 6), 0.0, "non-matching block");
    assert_eq!(at(2, 6), 0.0, "an isolated match is not kept");
    assert_eq!(at(0, 0), 0.0, "corner: zero padding leaves 4 of 9");
    assert_eq!(at(0, 1), 1.0, "edge: zero padding leaves 6 of 9");
}

#[tokio::test]
async fn higher_bands_step_at_the_rest_lr_ratio() {
    let device = device().await.autodiff();
    let mut bg = ShBackground::new(&device, 0.2);
    let dirs = [Vec3::new(0.6, 0.0, 0.8), Vec3::new(0.0, 0.6, -0.8), Vec3::new(-0.8, 0.6, 0.0)];
    let basis = sh_basis(dirs_tensor_on(&dirs, &device));
    let loss = (bg.image(basis) - 0.9).powi_scalar(2).sum();
    let mut grads = loss.backward();
    bg.step(&mut grads, 0.01);
    let c = read(bg.coeffs().inner()).await;
    // Adam's first step moves every coefficient with a gradient by ~lr.
    assert!((c[0].abs() - 0.01).abs() < 1e-4, "DC step {}", c[0]);
    let rest_max = c[3..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!((rest_max - 0.002).abs() < 1e-4, "rest step {rest_max}");
}

#[tokio::test]
async fn image_is_not_clamped_above_one() {
    let device = device().await.autodiff();
    let mut bg = ShBackground::new(&device, 1.0);
    let basis = sh_basis(dirs_tensor_on(&[Vec3::Z], &device));
    for _ in 0..200 {
        let loss = (bg.image(basis.clone()) - 1.5).powi_scalar(2).sum();
        let mut grads = loss.backward();
        bg.step(&mut grads, 0.05);
    }
    let v = read(bg.image(basis).inner()).await;
    assert!(v.iter().all(|x| *x > 1.2), "globe should exceed 1 freely, got {v:?}");
}

/// A globe with every band set, so the upsampling tests see real
/// variation across the image.
fn varied_globe(device: &Device) -> ShBackground {
    let mut bg = ShBackground::new(device, 1.0);
    let coeffs: Vec<f32> = (0..27).map(|i| 0.35 * ((i as f32) * 1.7).sin()).collect();
    bg.coeffs = Tensor::<1>::from_floats(coeffs.as_slice(), device)
        .reshape([9, 3])
        .require_grad();
    bg
}

fn wide_camera() -> Camera {
    Camera::new(
        Vec3::ZERO,
        glam::Quat::from_axis_angle(Vec3::new(0.2, 1.0, 0.1).normalize(), 0.8),
        1.3,
        1.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

fn per_pixel_and_upsampled(
    bg: &ShBackground,
    cam: &Camera,
    size: UVec2,
    device: &Device,
) -> (Tensor<3>, Tensor<3>) {
    let (h, w) = (size.y as usize, size.x as usize);
    let full = bg
        .image(sh_basis(world_dirs(pixel_centres(size, device), cam, size)))
        .reshape([h, w, 3]);
    let grid = block_grid_size(size);
    let low = bg
        .image(sh_basis(world_dirs(pixel_centres(grid, device), cam, grid)))
        .reshape([grid.y as usize, grid.x as usize, 3]);
    (full, upsample_background(low, h, w))
}

#[tokio::test]
async fn upsampled_globe_matches_per_pixel_evaluation() {
    let device = device().await.autodiff();
    let bg = varied_globe(&device);
    // 500 px long side, as in training; 375 doesn't divide by the block.
    let size = UVec2::new(500, 375);
    let (full, up) = per_pixel_and_upsampled(&bg, &wide_camera(), size, &device);
    let diff = read((full.clone() - up).abs().reshape([-1, 1]).inner()).await;
    let worst = diff.iter().fold(0.0f32, |m, v| m.max(*v));
    let range = read(full.reshape([-1, 1]).inner()).await;
    let span = range.iter().fold(f32::MIN, |m, v| m.max(*v))
        - range.iter().fold(f32::MAX, |m, v| m.min(*v));
    assert!(span > 0.2, "test globe should vary, span {span}");
    assert!(worst < 4.0 / 255.0, "worst per-pixel deviation {worst}");
}

#[tokio::test]
async fn upsampled_globe_gradient_matches_per_pixel_gradient() {
    let device = device().await.autodiff();
    let size = UVec2::new(160, 120);
    let cam = wide_camera();
    let target = Tensor::<3>::full([120, 160, 3], 0.8, &device);
    let grad_of = |use_upsampled: bool| {
        let bg = varied_globe(&device);
        let (full, up) = per_pixel_and_upsampled(&bg, &cam, size, &device);
        let img = if use_upsampled { up } else { full };
        let mut grads = (img - target.clone()).powi_scalar(2).sum().backward();
        bg.coeffs().grad_remove(&mut grads).expect("coeff grad")
    };
    let exact = grad_of(false);
    let approx = grad_of(true);
    let err = read((exact.clone() - approx).powi_scalar(2).sum().sqrt().reshape([1, 1])).await[0];
    let norm = read(exact.powi_scalar(2).sum().sqrt().reshape([1, 1])).await[0];
    assert!(err / norm < 0.02, "relative gradient error {}", err / norm);
}
