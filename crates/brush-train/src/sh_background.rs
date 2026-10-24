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
use burn::tensor::{Device, Gradients, Int, Tensor, s};
use glam::UVec2;
#[cfg(any(test, feature = "test-support"))]
use glam::Vec3;

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
/// Pinhole convention: pixel `(i, j)`'s centre at `i + 0.5`, local ray
/// `((px - cx)/fx, (py - cy)/fy, 1)`; the world rotation matches Brush's
/// forward axis used throughout `brush-guide` (`camera.rotation * Vec3::Z`,
/// see [`crate::cone::ViewCone::of_camera`]).
///
/// Panics if `camera.camera_model` isn't `Pinhole`: a fisheye/KB4 camera's
/// `focal()` comes from that model's own (non-linear) FOV formula, which
/// this pinhole unprojection would silently misinterpret into the wrong
/// direction for every pixel but the centre. `brush-guide` keyframes are
/// always pinhole; a non-pinhole caller needs its own unprojection, not a
/// silently-wrong one.
#[cfg(test)]
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

/// `max(basis · coeffs + 0.5, 0)`, `[n, 3]`, for an `[n, 9]` basis and
/// `[9, 3]` coefficients.
fn globe_colour(coeffs: Tensor<2>, basis: Tensor<2>) -> Tensor<2> {
    // Broadcast-and-sum rather than `matmul`: the coefficient gradient is
    // then a sum over all pixels, which reduces in parallel, whereas the
    // matmul backward (`[9, n] x [n, 3]`) has only 27 outputs to spread a
    // reduction over hundreds of thousands of pixels across.
    let [n, k] = basis.dims();
    let weighted = basis.reshape([n, k, 1]) * coeffs.reshape([1, k, 3]);
    (weighted.sum_dim(1).reshape([n, 3]) + 0.5).clamp_min(0.0)
}

/// The globe of `coeffs` (`[9, 3]`) behind a render of `camera` at `size`: its colour on the [`BG_BLOCK_PX`] block
/// grid, `[gh, gw, 3]`, and that grid upsampled to the render, `[h, w, 3]`. `centres` are [`pixel_centres`] of
/// [`block_grid_size`]`(size)`. Training passes the coefficients on the autodiff graph, evals off it.
pub fn globe_background(
    coeffs: Tensor<2>,
    centres: Tensor<2>,
    camera: &Camera,
    size: UVec2,
) -> (Tensor<3>, Tensor<3>) {
    let grid = block_grid_size(size);
    let low = globe_colour(coeffs, sh_basis(world_dirs(centres, camera, grid)))
        .reshape([grid.y as usize, grid.x as usize, 3]);
    let image = upsample_background(low.clone(), size.y as usize, size.x as usize);
    (low, image)
}

/// The learned globe: degree-2 SH coefficients `[9, 3]`, with their own tiny
/// Adam state (reusing `crate::adam_scaled`, the trainer's own optimizer, so
/// the globe gets the same numerics as every other learned parameter).
#[cfg_attr(test, derive(Clone))]
pub struct ShBackground {
    coeffs: Tensor<2>,
    adam: AdamScaled,
    state: AdamState<2>,
    /// [`pixel_centres`] of the block grid at the last composited size; directions are
    /// rebuilt from it on the GPU each step, so nothing per view is cached.
    grid_centres: Option<(UVec2, Tensor<2>)>,
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
            grid_centres: None,
        }
    }

    /// The globe's colour at each row of `basis` (an `[n, 9]` SH basis from
    /// [`sh_basis`]): `max(basis · coeffs + 0.5, 0)`, `[n, 3]`. The
    /// `+ 0.5` matches `brush_render::sh::rgb_to_sh`'s convention that SH
    /// coefficients of zero decode to mid-grey, not black. No upper clamp:
    /// it would zero the gradient wherever the globe reaches white, which
    /// overexposed sky does all the time.
    #[cfg(test)]
    pub fn image(&self, basis: Tensor<2>) -> Tensor<2> {
        globe_colour(self.coeffs.clone(), basis)
    }

    /// A globe of one colour in every direction (band 0 only), with fresh
    /// Adam state.
    #[cfg(any(test, feature = "test-support"))]
    pub fn constant(device: &Device, colour: Vec3) -> Self {
        let mut bg = Self::new(device, 1.0);
        let mut coeffs = [0.0f32; 27];
        for (c, v) in colour.to_array().into_iter().enumerate() {
            coeffs[c] = (v - 0.5) / SH_C0;
        }
        bg.coeffs = Tensor::<1>::from_floats(coeffs, device).reshape([9, 3]).require_grad();
        bg
    }

    /// The background `[h, w, 3]` a training step composites behind a render
    /// of `camera` at `size`: the globe on the [`BG_BLOCK_PX`] block grid,
    /// upsampled. Off the autodiff graph, on the coefficients' inner device.
    pub fn render_background(&self, camera: &Camera, size: UVec2) -> Tensor<3> {
        let coeffs = self.coeffs.clone().inner();
        let centres = pixel_centres(block_grid_size(size), &coeffs.device());
        globe_background(coeffs, centres, camera, size).1
    }

    /// [`pixel_centres`] of the block grid `grid`, kept while the grid stays the same size.
    fn grid_centres(&mut self, grid: UVec2) -> Tensor<2> {
        match &self.grid_centres {
            Some((s, c)) if *s == grid => c.clone(),
            _ => {
                let c = pixel_centres(grid, &self.coeffs.device());
                self.grid_centres = Some((grid, c.clone()));
                c
            }
        }
    }

    /// The prediction `pred` (`[h, w, 4]`, RGB + alpha) with the globe composited into
    /// RGB behind the render's own alpha; the alpha channel is untouched (the alpha-match
    /// loss reads it). That alpha is the true accumulated alpha: the rasterizer writes it
    /// into channel 3 whatever the render background (`kernels::rasterize`'s `final_a`). Also the
    /// background alpha penalty (Splatfacto-W) when `alpha_weight > 0`: where the
    /// globe already matches the photo `gt_packed`, opacity is a floater. The globe
    /// is evaluated on a coarse block grid and upsampled; the penalty uses the same grid.
    pub fn composite(
        &mut self,
        pred: Tensor<3>,
        gt_packed: Tensor<2, Int>,
        camera: &Camera,
        img_size: UVec2,
        alpha_weight: f32,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
        let grid = block_grid_size(img_size);
        let (grid_h, grid_w) = (grid.y as usize, grid.x as usize);
        let centres = self.grid_centres(grid);
        let (bg_grid, bg_image) = globe_background(self.coeffs(), centres, camera, img_size);
        let rgb = pred.clone().slice(s![.., .., 0..3]);
        let alpha = pred.slice(s![.., .., 3..4]);
        let mut penalty = None;
        if alpha_weight > 0.0 {
            let gt_grid = block_mean(brush_loss::unpack_gt_rgb(gt_packed, None), grid);
            let mask: Tensor<2> =
                Tensor::from_inner(background_match_mask(gt_grid, bg_grid.clone().inner()));
            let alpha_grid = block_mean(alpha.clone(), grid).reshape([grid_h, grid_w]);
            let covered = alpha_grid * mask.clone();
            penalty = Some(covered.sum() / mask.sum().clamp_min(1.0) * alpha_weight);
        }
        let composited = rgb + (1.0f32 - alpha.clone()) * bg_image;
        (Tensor::cat(vec![composited, alpha], 2), penalty)
    }

    /// The coefficients, on the autodiff graph. [`Self::composite`] builds the training
    /// background from these (not from [`image`]'s output) so gradients reach the globe
    /// through the same graph the splats use, and a single `backward()` covers both.
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
            .step(lr, self.coeffs.clone().inner(), &grad, None, &mut self.state, None);
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
#[path = "sh_background_tests.rs"]
pub(crate) mod tests;
