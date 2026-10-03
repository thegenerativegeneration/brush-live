//! Degree-2 spherical-harmonic environment background.
//!
//! A learned per-direction colour, composited into the training pixels a
//! splat's alpha leaves transparent (sky, at-infinity surfaces) so training
//! doesn't have to grow near-camera floaters to explain them. Behind the
//! `sh_background` config flag; see `crate::train` for the integration.

use crate::adam_scaled::{AdamScaled, AdamState};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::module::{adaptive_avg_pool2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Gradients, Tensor};
use glam::{UVec2, Vec3};

/// Band-0 constant, matching `brush_render::kernels::sh::SH_C0`.
const SH_C0: f32 = 0.282_094_8;
/// Band-1 constant, matching `brush_render::kernels::sh`'s `f0a`.
const SH_C1: f32 = 0.488_602_5;
/// Band-2 `xy`/`x²-y²` constant, matching `brush_render::kernels::sh`'s `f1a`.
const SH_C2_XY: f32 = 0.546_274_24;
/// Band-2 `yz`/`xz` constant, matching `brush_render::kernels::sh`'s `f0b`'s
/// per-`z` factor.
const SH_C2_Z: f32 = -1.092_548_5;
/// Band-2 `z²` constant, matching `brush_render::kernels::sh`'s `p_sh6`.
const SH_C2_Z2: f32 = 0.946_174_7;
/// Band-2 `z²` offset, matching `brush_render::kernels::sh`'s `p_sh6`.
const SH_C2_Z2_OFFSET: f32 = -0.315_391_57;

/// Degree-2 real SH basis of unit directions `dirs` (`[n, 3]`) -> `[n, 9]`.
///
/// Coefficient order mirrors `brush_render::kernels::sh::sh_coeffs_to_color`
/// exactly: band 0 (DC); band 1 as `{-y, z, -x} * SH_C1`; band 2 as `{2xy,
/// yz, 3z²-1, xz, x²-y²}` with Brush's own constants. Read that function
/// first before touching these — a sign or ordering mismatch there produces
/// a globe that silently fits the wrong angular pattern.
pub fn sh_basis(dirs: Tensor<2>) -> Tensor<2> {
    let device = dirs.device();
    let n = dirs.dims()[0];
    let x = dirs.clone().slice([0..n, 0..1]);
    let y = dirs.clone().slice([0..n, 1..2]);
    let z = dirs.slice([0..n, 2..3]);

    let b0 = Tensor::<2>::full([n, 1], SH_C0, &device);
    let b1_0 = y.clone() * -SH_C1;
    let b1_1 = z.clone() * SH_C1;
    let b1_2 = x.clone() * -SH_C1;

    let z2 = z.clone() * z.clone();
    let fc1 = x.clone() * x.clone() - y.clone() * y.clone();
    let fs1 = x.clone() * y.clone() * 2.0;
    let b2_xy = fs1 * SH_C2_XY;
    let b2_yz = z.clone() * y * SH_C2_Z;
    let b2_z2 = z2 * SH_C2_Z2 + SH_C2_Z2_OFFSET;
    let b2_xz = z * x * SH_C2_Z;
    let b2_x2y2 = fc1 * SH_C2_XY;

    Tensor::cat(
        vec![b0, b1_0, b1_1, b1_2, b2_xy, b2_yz, b2_z2, b2_xz, b2_x2y2],
        1,
    )
}

/// World-space unit ray directions of every pixel of `camera` at `size`
/// (`[h*w, 3]`, row-major — pixel `(x, y)` at index `y * w + x`).
///
/// Pinhole convention matches
/// `brush_guide::geometry::tsdf::projection::Projection` exactly (pixel
/// `(i, j)`'s centre at `i + 0.5`, local ray `((px - cx)/fx, (py - cy)/fy,
/// 1)`); the world rotation matches Brush's forward axis used throughout
/// `brush-guide` (`camera.rotation * Vec3::Z`, see e.g. `live.rs`'s
/// `view_cones`).
///
/// Panics if `camera.camera_model` isn't `Pinhole`: a fisheye/KB4 camera's
/// `focal()` comes from that model's own (non-linear) FOV formula, which
/// this pinhole unprojection would silently misinterpret into the wrong
/// direction for every pixel but the centre. `brush-guide` keyframes are
/// always pinhole; a non-pinhole caller needs its own unprojection, not a
/// silently-wrong one.
pub fn pixel_dirs(camera: &Camera, size: UVec2) -> Vec<f32> {
    assert!(
        matches!(camera.camera_model, CameraModel::Pinhole),
        "pixel_dirs only supports a pinhole camera_model, got {:?}",
        camera.camera_model
    );
    let focal = camera.focal(size);
    let centre = camera.center(size);
    let rotation = camera.rotation;
    let mut out = Vec::with_capacity((size.x * size.y * 3) as usize);
    for y in 0..size.y {
        for x in 0..size.x {
            let local = Vec3::new(
                (x as f32 + 0.5 - centre.x) / focal.x,
                (y as f32 + 0.5 - centre.y) / focal.y,
                1.0,
            );
            let world = (rotation * local).normalize();
            out.push(world.x);
            out.push(world.y);
            out.push(world.z);
        }
    }
    out
}

/// Side, in training pixels, of the square block the globe is evaluated on.
/// A degree-2 globe barely changes across 8 pixels (about a degree at the
/// 500 px training size), so evaluating it per block and upsampling costs
/// almost nothing in accuracy and removes most of its per-pixel work.
pub const BG_BLOCK_PX: u32 = 8;

/// The block grid's size for a render of `size`: [`BG_BLOCK_PX`]-pixel
/// blocks, the last row and column partial when `size` doesn't divide.
pub fn block_grid_size(size: UVec2) -> UVec2 {
    UVec2::new(
        size.x.div_ceil(BG_BLOCK_PX).max(1),
        size.y.div_ceil(BG_BLOCK_PX).max(1),
    )
}

/// Upsamples a block-grid background `[hl, wl, 3]` to `[h, w, 3]`. The value
/// is bilinear (half-pixel centres); the gradient flows through a
/// nearest-neighbour upsample instead, i.e. each block receives the sum of
/// its pixels' gradients, because the GPU backend has no bilinear backward.
/// For a field this smooth the two gradients differ negligibly.
pub fn upsample_background(low: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
    let [hl, wl, c] = low.dims();
    let nchw = low.permute([2, 0, 1]).reshape([1, c, hl, wl]);
    let smooth = interpolate(
        nchw.clone().detach(),
        [h, w],
        InterpolateOptions {
            mode: InterpolateMode::Bilinear,
            align_corners: false,
        },
    );
    let blocky = interpolate(nchw, [h, w], InterpolateOptions::new(InterpolateMode::Nearest));
    let out = smooth + blocky.clone() - blocky.detach();
    out.reshape([c, h, w]).permute([1, 2, 0])
}

/// Mean of `x` (`[h, w, c]`) over each cell of a `grid`-sized block grid,
/// `[grid.y, grid.x, c]`. Differentiable.
pub fn block_mean(x: Tensor<3>, grid: UVec2) -> Tensor<3> {
    let [h, w, c] = x.dims();
    let (hl, wl) = (grid.y as usize, grid.x as usize);
    let nchw = x.permute([2, 0, 1]).reshape([1, c, h, w]);
    adaptive_avg_pool2d(nchw, [hl, wl])
        .reshape([c, hl, wl])
        .permute([1, 2, 0])
}

/// Pixel centres `(x + 0.5, y + 0.5)` of every pixel at `size`, `[h*w, 2]`,
/// in [`pixel_dirs`]'s order. Depends only on the size, so the trainer builds
/// it once per render size and [`world_dirs`] turns it into directions on the
/// GPU for each camera.
pub fn pixel_centres(size: UVec2, device: &Device) -> Tensor<2> {
    let n = (size.x * size.y) as usize;
    let mut out = Vec::with_capacity(n * 2);
    for y in 0..size.y {
        for x in 0..size.x {
            out.push(x as f32 + 0.5);
            out.push(y as f32 + 0.5);
        }
    }
    Tensor::<1>::from_floats(out.as_slice(), device).reshape([n, 2])
}

/// The tensor form of [`pixel_dirs`]: world-space unit ray directions,
/// `[h*w, 3]`, from [`pixel_centres`]. Only the camera's intrinsics and
/// rotation are uploaded per call. Same pinhole-only restriction.
pub fn world_dirs(centres: Tensor<2>, camera: &Camera, size: UVec2) -> Tensor<2> {
    assert!(
        matches!(camera.camera_model, CameraModel::Pinhole),
        "world_dirs only supports a pinhole camera_model, got {:?}",
        camera.camera_model
    );
    let device = centres.device();
    let n = centres.dims()[0];
    let focal = camera.focal(size);
    let centre = camera.center(size);
    let offset = Tensor::<1>::from_floats([centre.x, centre.y], &device).reshape([1, 2]);
    let inv_focal =
        Tensor::<1>::from_floats([1.0 / focal.x, 1.0 / focal.y], &device).reshape([1, 2]);
    let local = Tensor::cat(
        vec![(centres - offset) * inv_focal, Tensor::ones([n, 1], &device)],
        1,
    );
    // Row vectors times Rᵀ; Rᵀ in row-major order is R's column-major array.
    let rotation_t = Tensor::<1>::from_floats(
        glam::Mat3::from_quat(camera.rotation).to_cols_array(),
        &device,
    )
    .reshape([3, 3]);
    let world = (local.reshape([n, 3, 1]) * rotation_t.reshape([1, 3, 3]))
        .sum_dim(1)
        .reshape([n, 3]);
    let norm = world.clone().powi_scalar(2).sum_dim(1).sqrt();
    world / norm
}

/// Mean absolute RGB difference below which a GT pixel counts as explained
/// by the background.
const BG_MATCH_EPS: f32 = 0.003;
/// Share of a pixel's 3x3 neighbourhood that must match for the pixel to
/// count, so isolated matches inside textured regions are ignored.
const BG_MATCH_NEIGHBOURHOOD: f32 = 0.6;

/// Pixels the globe already explains, `[h, w]` of 0/1: mean `|gt - bg|`
/// over RGB below [`BG_MATCH_EPS`], kept where more than
/// [`BG_MATCH_NEIGHBOURHOOD`] of the zero-padded 3x3 neighbourhood matches.
/// The mask of Splatfacto-W's background alpha loss.
pub fn background_match_mask(gt: Tensor<3>, bg: Tensor<3>) -> Tensor<2> {
    let [h, w, _] = gt.dims();
    let device = gt.device();
    let close = (gt - bg)
        .abs()
        .mean_dim(2)
        .reshape([h, w])
        .lower_elem(BG_MATCH_EPS)
        .float();
    let row = Tensor::<2>::zeros([1, w], &device);
    let padded = Tensor::cat(vec![row.clone(), close, row], 0);
    let col = Tensor::<2>::zeros([h + 2, 1], &device);
    let padded = Tensor::cat(vec![col.clone(), padded, col], 1);
    let mut sum = Tensor::<2>::zeros([h, w], &device);
    for dy in 0..3 {
        for dx in 0..3 {
            sum = sum + padded.clone().slice([dy..dy + h, dx..dx + w]);
        }
    }
    (sum / 9.0).greater_elem(BG_MATCH_NEIGHBOURHOOD).float()
}

/// The learned globe: degree-2 SH coefficients `[9, 3]`, with their own tiny
/// Adam state (reusing `crate::adam_scaled`, the trainer's own optimizer, so
/// the globe gets the same numerics as every other learned parameter).
pub struct ShBackground {
    coeffs: Tensor<2>,
    adam: AdamScaled,
    state: AdamState<2>,
}

impl ShBackground {
    /// Zero coefficients: `image(..)` is uniform mid-grey everywhere, so
    /// turning the flag on never perturbs a step's loss before the globe
    /// has learned anything. `rest_lr_ratio` scales the learning rate of the
    /// eight non-DC rows relative to the DC row's `lr` passed to [`step`].
    pub fn new(device: &Device, rest_lr_ratio: f32) -> Self {
        let mut rows = [rest_lr_ratio; 9];
        rows[0] = 1.0;
        let scaling = Tensor::<1>::from_floats(rows, &device.clone().inner()).reshape([9, 1]);
        Self {
            coeffs: Tensor::<2>::zeros([9, 3], device).require_grad(),
            adam: AdamScaled::new(1e-8),
            state: AdamState::new(Some(scaling), false),
        }
    }

    /// The globe's colour at each row of `basis` (an `[n, 9]` SH basis from
    /// [`sh_basis`]): `max(basis · coeffs + 0.5, 0)`, `[n, 3]`. The
    /// `+ 0.5` matches `brush_render::sh::rgb_to_sh`'s convention that SH
    /// coefficients of zero decode to mid-grey, not black. No upper clamp:
    /// it would zero the gradient wherever the globe reaches white, which
    /// overexposed sky does all the time.
    pub fn image(&self, basis: Tensor<2>) -> Tensor<2> {
        // Broadcast-and-sum rather than `matmul`: the coefficient gradient is
        // then a sum over all pixels, which reduces in parallel, whereas the
        // matmul backward (`[9, n] x [n, 3]`) has only 27 outputs to spread a
        // reduction over hundreds of thousands of pixels across.
        let [n, k] = basis.dims();
        let weighted = basis.reshape([n, k, 1]) * self.coeffs.clone().reshape([1, k, 3]);
        (weighted.sum_dim(1).reshape([n, 3]) + 0.5).clamp_min(0.0)
    }

    /// The coefficients, on the autodiff graph — composite with this (not
    /// with [`image`]'s output during training) so gradients reach the
    /// globe through the same graph the splats use, and a single
    /// `backward()` covers both.
    pub fn coeffs(&self) -> Tensor<2> {
        self.coeffs.clone()
    }

    /// Extracts the coefficient gradient from `grads` (populated by the same
    /// `backward()` call that produced the splat gradients — see
    /// `crate::train`) and runs one Adam step. A no-op if the coefficients
    /// never entered this step's graph (flag off, or an untracked path).
    pub fn step(&mut self, grads: &mut Gradients, lr: f64) {
        let Some(grad) = self.coeffs.clone().grad_remove(grads) else {
            return;
        };
        let stepped = self
            .adam
            .step(lr, self.coeffs.clone().inner(), &grad, &mut self.state);
        self.coeffs = Tensor::from_inner(stepped).require_grad();
    }

    /// The coefficients as a small hand-written JSON array (`[9][3]` f32),
    /// for the session-dir dump. Async: tensor readback crosses the GPU
    /// boundary, matching every other readback in this crate
    /// (`into_data_async`).
    pub async fn coeffs_json(&self) -> String {
        let data = self
            .coeffs
            .clone()
            .into_data_async()
            .await
            .expect("Failed to read SH background coeffs")
            .try_into_vec::<f32>()
            .expect("Failed to read SH background coeffs");
        let mut out = String::from("[");
        for row in 0..9 {
            if row > 0 {
                out.push(',');
            }
            out.push('[');
            for c in 0..3 {
                if c > 0 {
                    out.push(',');
                }
                out.push_str(&data[row * 3 + c].to_string());
            }
            out.push(']');
        }
        out.push(']');
        out
    }
}

#[cfg(test)]
mod tests {
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
}
