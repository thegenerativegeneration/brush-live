//! Degree-2 spherical-harmonic environment background.
//!
//! A learned per-direction colour, composited into the training pixels a
//! splat's alpha leaves transparent (sky, at-infinity surfaces) so training
//! doesn't have to grow near-camera floaters to explain them. Behind the
//! `sh_background` config flag; see `crate::train` for the integration.

use crate::adam_scaled::{AdamScaled, AdamState};
use brush_render::camera::Camera;
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
/// `view_cones`). A non-pinhole `camera_model` would need its own
/// unprojection; callers render training views as pinhole, so this only
/// supports that.
pub fn pixel_dirs(camera: &Camera, size: UVec2) -> Vec<f32> {
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
    /// has learned anything.
    pub fn new(device: &Device) -> Self {
        Self {
            coeffs: Tensor::<2>::zeros([9, 3], device).require_grad(),
            adam: AdamScaled::new(1e-8),
            state: AdamState::new(None, false),
        }
    }

    /// The globe's colour at each row of `basis` (an `[n, 9]` SH basis from
    /// [`sh_basis`]): `clamp(basis · coeffs + 0.5, 0, 1)`, `[n, 3]`. The
    /// `+ 0.5` matches `brush_render::sh::rgb_to_sh`'s convention that SH
    /// coefficients of zero decode to mid-grey, not black.
    pub fn image(&self, basis: Tensor<2>) -> Tensor<2> {
        (basis.matmul(self.coeffs.clone()) + 0.5).clamp(0.0, 1.0)
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

    #[tokio::test]
    async fn sh_background_starts_uniform_grey() {
        let device = device().await.autodiff();
        let bg = ShBackground::new(&device);
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
}
