use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    adam_scaled::{AdamScaled, AdamState},
    config::TrainConfig,
    evict::{
        EvictConfig, Eviction, ProtectCone, SplatLife, growth_demand, read_bool, read_f32,
        recent_demand, select_evictions,
    },
    msg::{RefineStats, TrainStepStats},
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    sh_background::{
        ShBackground, background_match_mask, block_grid_size, block_mean, pixel_centres, sh_basis,
        upsample_background, world_dirs,
    },
    splat_init::bounds_from_pos,
    stats::RefineRecord,
};
use brush_dataset::scene::SceneBatch;
use brush_loss::{ImageLossConfig, image_loss};
use brush_render::bwd::render_splats;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
use burn::{
    module::Param,
    tensor::{
        Bool, Device, Distribution, Gradients, IndexingUpdateOp::Assign, Int, Tensor, TensorData,
        activation::sigmoid, s,
    },
};

use hashbrown::HashSet;
use rand::SeedableRng;
use crate::profile::{StepProfile, device_sync, lap};
use tracing::{Instrument, trace_span};

pub const BOUND_PERCENTILE: f32 = 0.8;

const MIN_OPACITY: f32 = 1.0 / 255.0;

/// Fraction of training after which the Mip-Splatting 3D-filter floor stops
/// being recomputed and is held frozen (still applied), so splats settle
/// against a fixed target instead of chasing a moving floor.
const MIN_SCALE_FREEZE_FRAC: f32 = 0.9;

/// The three per-parameter Adam states of a [`Splats`] module, owned directly
/// so the trainer can update LR scaling every step and surgically edit the
/// momentum tensors during refine — all GPU-side, no record round-trips.
struct SplatOptim {
    adam: AdamScaled,
    transforms: AdamState<2>,
    sh_coeffs: AdamState<3>,
    opacities: AdamState<1>,
}

/// Step one parameter: pull its gradient, run Adam on the inner
/// (autodiff-free) tensor, and re-wrap tracking. Parameters without a
/// gradient this step are left untouched.
fn step_param<const D: usize>(
    adam: &AdamScaled,
    lr: f64,
    param: Param<Tensor<D>>,
    state: &mut AdamState<D>,
    grads: &mut Gradients,
    grad_sq_mean: Option<Tensor<D>>,
) -> Param<Tensor<D>> {
    param.map(|t| {
        let Some(grad) = t.grad_remove(grads) else {
            return t;
        };
        let stepped = adam.step(lr, t.inner(), &grad, grad_sq_mean, state);
        Tensor::from_inner(stepped).require_grad()
    })
}

pub struct SplatTrainer {
    config: TrainConfig,
    /// Per-step multiplier of the exponential mean-LR schedule:
    /// `lr(n) = lr_mean * decay^(n-1)`.
    lr_mean_decay: f64,
    /// Per-column LR scales for `transforms` (`means(3) + rotations(4) +
    /// log_scales(3)`) with the mean columns zeroed, and a mask of those
    /// columns: the scheduled mean LR is mixed in on the device each step so
    /// the optimizer never waits on a host upload.
    lr_scaling_fixed: Tensor<2>,
    lr_mean_columns: Tensor<2>,
    refine_record: Option<RefineRecord>,
    optim: Option<SplatOptim>,
    ssim_enabled: bool,
    bounds: BoundingBox,
    step_count: u32,
    max_sh_degree: u32,
    rng: rand::rngs::StdRng,
    /// Per-train-view (world center, focal in px at native res) for the
    /// Mip-Splatting 3D filter. Empty disables it. The floor itself lives on
    /// the splats (recomputed at each refine), not here.
    view_cams: Vec<(glam::Vec3, f32)>,
    /// Keeps the splat count within `max_splats` by evicting; `None` caps growth as Brush does.
    evict: Option<Eviction>,
    /// Per-phase step timing, when on (`set_profiling`).
    profile: Option<StepProfile>,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel>,
    /// The learned SH environment background; `None` when `config.sh_background`
    /// is off, so the flag-off step never builds a basis, a background image,
    /// or any extra graph node (see `step`'s `if let Some(..)` guard).
    sh_background: Option<ShBackground>,
    /// Pixel centres for the SH background at the last render size (see
    /// `sh_background::pixel_centres`); directions are rebuilt from it on the
    /// GPU each step, so nothing per view is cached.
    sh_pixel_centres: Option<(glam::UVec2, Tensor<2>)>,
}

fn inv_sigmoid(x: Tensor<1>) -> Tensor<1> {
    (x.clone() / (1.0f32 - x)).log()
}

/// Per-splat world-space scale floor for the Mip-Splatting 3D filter:
/// `f_i = sqrt(factor) · min_v(||mean_i - cam_v|| / focal_px_v)`. `means` and
/// the result are on the inner (non-autodiff) backend; `f` is a frozen
/// constant. Returns `None` if disabled or there are no cameras.
fn compute_min_scale(
    means: &Tensor<2>,
    view_cams: &[(glam::Vec3, f32)],
    factor: f32,
) -> Option<Tensor<1>> {
    if factor <= 0.0 || view_cams.is_empty() {
        return None;
    }
    let device = means.device();
    let n = means.dims()[0] as i32;

    let mut min_ratio: Option<Tensor<1>> = None;
    for (center, focal) in view_cams {
        let c = Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let diff = means.clone() - c;
        let dist = diff.clone().mul(diff).sum_dim(1).sqrt().reshape([n]);
        let ratio = dist.div_scalar(focal.max(1e-6));
        min_ratio = Some(match min_ratio {
            Some(m) => m.min_pair(ratio),
            None => ratio,
        });
    }
    min_ratio.map(|r| r.mul_scalar(factor.sqrt()))
}

pub async fn get_splat_bounds(splats: Splats, percentile: f32) -> BoundingBox {
    let means: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .try_to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    pub fn new(config: &TrainConfig, device: &Device, bounds: BoundingBox) -> Self {
        Self::new_seeded(config, device, bounds, 42)
    }

    pub fn new_seeded(
        config: &TrainConfig,
        device: &Device,
        bounds: BoundingBox,
        seed: u64,
    ) -> Self {
        // The per-step decay reaching lr_mean_end at the last iteration. With
        // one iteration or fewer there is nothing to decay over (and the
        // exponent 1/iters would be undefined), so hold the LR.
        let decay = if config.total_train_iters > 1 {
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64)
        } else {
            1.0
        };

        let ssim_enabled = config.ssim_weight > 0.0;

        // Optimizer state lives on the inner device.
        let opt_device = device.clone().inner();
        let (rot, scale) = (config.lr_rotation as f32, config.lr_scale as f32);
        let lr_scaling_fixed = Tensor::<1>::from_floats(
            [0.0, 0.0, 0.0, rot, rot, rot, rot, scale, scale, scale],
            &opt_device,
        )
        .reshape([1, 10]);
        let lr_mean_columns = Tensor::<1>::from_floats(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &opt_device,
        )
        .reshape([1, 10]);

        // Growth is gated on the global iter. LOD phases run past
        // total_train_iters but their refines should never grow — clamp
        // here so growth_stop is never effectively past end-of-training,
        // and growth_start never past growth_stop.
        let mut config = config.clone();
        config.growth_stop_iter = config.growth_stop_iter.min(config.total_train_iters);
        config.growth_start_iter = config.growth_start_iter.min(config.growth_stop_iter);

        #[cfg(not(target_family = "wasm"))]
        let lpips = (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device));

        let sh_background = config.sh_background.then(|| {
            ShBackground::new(
                device,
                (config.sh_background_rest_lr / config.sh_background_lr) as f32,
            )
        });

        Self {
            config,
            lr_mean_decay: decay,
            lr_scaling_fixed,
            lr_mean_columns,
            optim: None,
            refine_record: None,
            ssim_enabled,
            bounds,
            step_count: 0,
            max_sh_degree: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            view_cams: Vec::new(),
            evict: None,
            sh_background,
            sh_pixel_centres: None,
            profile: None,
            #[cfg(not(target_family = "wasm"))]
            lpips,
        }
    }

    /// Turns per-phase step timing on or off; on, every step waits for the
    /// GPU at each phase boundary (see `profile`).
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(StepProfile::default);
    }

    /// The phase times since the last call; `None` when profiling is off.
    pub fn take_profile(&mut self) -> Option<StepProfile> {
        self.profile.as_mut().map(std::mem::take)
    }

    /// Percentile bounding box of the splats, refreshed on each refine.
    pub fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    /// Supply per-train-view (world center, focal-px at native res) to enable
    /// the Mip-Splatting 3D filter (gated on `config.min_scale_factor > 0`).
    pub fn set_view_cams(&mut self, view_cams: Vec<(glam::Vec3, f32)>) {
        self.view_cams = view_cams;
    }

    /// Turns on eviction (see [`crate::evict`]). Call before training starts.
    pub fn enable_eviction(&mut self, config: EvictConfig) {
        self.evict = Some(Eviction::new(config));
    }

    /// Per-splat importance for eviction, in splat order; higher is kept
    /// longer, NaN keeps the previous value. Ignored without eviction or
    /// when the length does not match.
    pub fn set_importance(&mut self, importance: &[f32]) {
        let Some(e) = self.evict.as_mut() else {
            return;
        };
        let Some(life) = e.life.as_mut() else {
            return;
        };
        if importance.len() != life.len() {
            log::warn!(
                "importance for {} splats, model has {}; ignored",
                importance.len(),
                life.len()
            );
            return;
        }
        life.set_importance(importance);
        e.fresh = true;
    }

    /// Splats inside `cone` are never evicted.
    pub fn set_protect_cone(&mut self, cone: Option<ProtectCone>) {
        if let Some(e) = self.evict.as_mut() {
            e.protect = cone;
        }
    }

    /// Records seeds dropped because the budget was full; the next refine
    /// evicts to make room.
    pub fn note_seed_shortfall(&mut self, dropped: u32) {
        if let Some(e) = self.evict.as_mut() {
            e.seed_shortfall = e.seed_shortfall.saturating_add(dropped);
        }
    }

    /// Records a keyframe's seed positions (flat xyz, all seeds before the
    /// budget drops any); cells it seeds count as newly observed.
    pub fn note_keyframe_seeds(&mut self, seed_means: &[f32]) {
        if let Some(e) = self.evict.as_mut() {
            e.recent.note_keyframe(seed_means);
        }
    }

    /// The count splitting and growth may fill up to.
    fn growth_cap(&self) -> u32 {
        match &self.evict {
            Some(e) => crate::evict::growth_limit(self.config.max_splats, e.config.headroom),
            None => self.config.max_splats,
        }
    }

    /// Adds this refine's evictions to the `dead` prune mask. Returns the
    /// mask and the number evicted.
    async fn add_evictions(
        &mut self,
        iter: u32,
        splats: &Splats,
        refiner: &RefineRecord,
        dead: Tensor<1, Bool>,
    ) -> (Tensor<1, Bool>, u32) {
        let Some(ev) = self.evict.as_mut() else {
            return (dead, 0);
        };
        let Some(life) = ev.life.as_mut() else {
            return (dead, 0);
        };
        life.tick();
        let window = ev.config.recent_refines;
        let active = ev.recent.active(window);
        if !active {
            ev.recent.backlog = 0;
            if ev.seed_shortfall == 0 {
                return (dead, 0);
            }
        }
        let current = splats.num_splats();
        let mut means = None;
        if active {
            let num_dead = dead
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("dead count readback") as u32;
            let demand = growth_demand(refiner, &self.config, iter, num_dead).await;
            let limit = crate::evict::growth_limit(self.config.max_splats, ev.config.headroom);
            if current.saturating_add(demand) > limit {
                let m = read_f32(splats.means()).await;
                let in_recent = ev.recent.mask(&m, window);
                let oversized = if self.config.split_at_screen_size > 0.0 {
                    read_bool(refiner.above_screen_size(self.config.split_at_screen_size)).await
                } else {
                    vec![]
                };
                let growing =
                    iter >= self.config.growth_start_iter && iter < self.config.growth_stop_iter;
                let above = if growing {
                    read_bool(refiner.above_threshold(self.config.growth_grad_threshold)).await
                } else {
                    vec![]
                };
                let recent = recent_demand(
                    &in_recent,
                    &oversized,
                    &above,
                    self.config.growth_select_fraction,
                );
                ev.recent.backlog = ev.recent.backlog.saturating_add(recent);
                means = Some(m);
            } else {
                ev.recent.backlog = 0;
            }
        }
        if !ev.fresh {
            return (dead, 0);
        }
        let shortfall = std::mem::take(&mut ev.seed_shortfall);
        let backlog = ev.recent.backlog;
        let want = crate::evict::evict_count(
            current,
            backlog,
            shortfall,
            self.config.max_splats,
            ev.config.headroom,
        );
        if want == 0 {
            return (dead, 0);
        }
        let means = match means {
            Some(m) => m,
            None => read_f32(splats.means()).await,
        };
        let (mask, count) = select_evictions(
            life,
            splats.means(),
            &means,
            dead.clone(),
            ev.protect,
            &ev.config,
            want,
        )
        .await;
        if count > 0 {
            ev.fresh = false;
            ev.recent.backlog = 0;
        }
        log::info!(
            "evict: {count} of {want} wanted (recent backlog {backlog}, seed shortfall {shortfall}, {current} splats)"
        );
        (dead.bool_or(mask), count)
    }

    /// Add `new` splats to `splats`, keeping optimizer and refine state aligned.
    /// Both must be on the inner backend and share the SH degree. If `splats`
    /// already carries a Mip-Splatting floor it is kept (not baked): the
    /// existing floor values are reused as-is and extended for the new
    /// splats via [`compute_min_scale`], so the filter stays live between
    /// refines instead of silently switching off until the next one.
    /// Otherwise the (absent) floor is a no-op as before. `new`'s own floor,
    /// if any, is always baked into its raw params before concatenation.
    pub fn append_splats(&mut self, splats: Splats, new: Splats) -> Splats {
        let n = new.num_splats() as usize;
        if n == 0 {
            return splats;
        }
        let opt_device = splats.device().inner();
        let existing_floor = splats.min_scale.clone();
        let splats = if existing_floor.is_some() {
            splats
        } else {
            splats.bake_min_scale()
        };
        let new_means = new.means().inner();
        let new = new.bake_min_scale();
        let (nt, ns, no) = (
            new.transforms.val(),
            new.sh_coeffs.val(),
            new.raw_opacities.val(),
        );

        let mut splats = if let Some(optim) = self.optim.as_mut() {
            map_splats_and_opt(
                splats,
                optim,
                |x| Tensor::cat(vec![x, nt], 0),
                |x| Tensor::cat(vec![x, ns], 0),
                |x| Tensor::cat(vec![x, no], 0),
                |x: Tensor<2>| {
                    let d1 = x.dims()[1];
                    Tensor::cat(vec![x, Tensor::zeros([n, d1], &opt_device)], 0)
                },
                |x: Tensor<3>| {
                    let [_, d1, d2] = x.dims();
                    Tensor::cat(vec![x, Tensor::zeros([n, d1, d2], &opt_device)], 0)
                },
                |x: Tensor<1>| Tensor::cat(vec![x, Tensor::zeros([n], &opt_device)], 0),
            )
        } else {
            let mut s = splats;
            s.transforms = s.transforms.map(|x| Tensor::cat(vec![x, nt], 0));
            s.sh_coeffs = s.sh_coeffs.map(|x| Tensor::cat(vec![x, ns], 0));
            s.raw_opacities = s.raw_opacities.map(|x| Tensor::cat(vec![x, no], 0));
            s
        };

        if let Some(existing_floor) = existing_floor {
            let new_floor =
                compute_min_scale(&new_means, &self.view_cams, self.config.min_scale_factor)
                    .unwrap_or_else(|| Tensor::zeros([n], &opt_device));
            splats = splats.with_min_scale(Tensor::cat(vec![existing_floor, new_floor], 0));
        }

        if let Some(record) = self.refine_record.take() {
            self.refine_record = Some(record.pad(n));
        }
        if let Some(life) = self.evict.as_mut().and_then(|e| e.life.as_mut()) {
            life.pad(n);
        }
        splats
    }

    /// The SH background's coefficients as JSON, or `None` when the flag is
    /// off.
    pub async fn sh_background_json(&self) -> Option<String> {
        match &self.sh_background {
            Some(bg) => Some(bg.coeffs_json().await),
            None => None,
        }
    }

    /// SH basis `[h*w, 9]` of `camera`'s world-space pixel directions at
    /// `size` (the globe's block grid, not the render size).
    fn sh_basis_for(
        &mut self,
        camera: &brush_render::camera::Camera,
        size: glam::UVec2,
        device: &Device,
    ) -> Tensor<2> {
        let centres = match &self.sh_pixel_centres {
            Some((s, c)) if *s == size => c.clone(),
            _ => {
                let c = pixel_centres(size, device);
                self.sh_pixel_centres = Some((size, c.clone()));
                c
            }
        };
        sh_basis(world_dirs(centres, camera, size))
    }

    pub async fn step(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        let mut splats = splats;

        // Track max SH degree from the first splats we see.
        if self.step_count == 0 {
            self.max_sh_degree = splats.sh_degree();
        }
        self.step_count += 1;

        let profiling = self.profile.is_some();
        let mut phases = [0.0f64; 5];
        if profiling {
            device_sync(&splats.device());
        }
        let mut clock = web_time::Instant::now();

        let [img_h, img_w] = batch.img_size();
        let camera = batch.camera;

        let device = splats.device();
        let has_alpha = batch.has_alpha;
        // GT lives on the GPU as packed `[H, W]` u32 (RGBA u8). All mixing
        // (bg compositing, alpha matching, mask) is folded into the loss
        // kernels; no f32 GT image is ever materialised here.
        // GT is pure data — never differentiated. Build it on the inner
        // backend so it doesn't inherit the autodiff device's residual
        // checkpointing flag (the LPIPS `unpack_gt_rgb` path, via
        // `unwrap_wgpu_int`, expects a clean Wgpu tensor).
        let gt_packed: Tensor<2, Int> =
            Tensor::from_data(batch.img_packed, &device.clone().inner());
        let img_size = glam::uvec2(img_w as u32, img_h as u32);
        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(
            base_bg,
            self.config.background_noise_strength,
            &mut self.rng,
        );

        let median_scale = self.bounds.median_size();

        let lr_mean = self.config.lr_mean
            * self.lr_mean_decay.powi(self.step_count as i32 - 1)
            * median_scale as f64;
        let masked_alpha = batch.alpha_mode == AlphaMode::Masked;
        let do_alpha_match = has_alpha && !masked_alpha && self.config.match_alpha_weight > 0.0;

        // The optimizer runs with base LR 1 and these per-column scales.
        let lr_scaling =
            self.lr_scaling_fixed.clone() + self.lr_mean_columns.clone() * lr_mean as f32;

        let (
            mut grads,
            visible,
            opacities,
            num_visible,
            loss_inner,
            refine_weight,
            max_radius,
            coeffs_grad_sq,
        ) = {
            // The splats already carry their 3D-filter floor (set at refine);
            // the render path folds it in. Optimizer/refine work on raw params.
            let render_input = splats.clone();
            // With the SH background on, the render must not bake in any flat
            // colour — the globe supplies it per pixel below instead — so the
            // render background is forced to zero. Off, this is exactly
            // `background`, the existing path, unchanged.
            let render_bg = if self.sh_background.is_some() {
                glam::Vec3::ZERO
            } else {
                background
            };
            let diff_out = render_splats(render_input, &camera, img_size, render_bg)
                .instrument(trace_span!("Forward"))
                .await;
            if profiling {
                device_sync(&device);
                phases[0] = lap(&mut clock);
            }

            let pred_image = diff_out.img;
            let refine_weight_holder = diff_out.refine_weight_holder;
            let visible = diff_out.visible;
            let max_radius = diff_out.max_radius;
            let opacities = diff_out.opacities;

            // RGB loss is `(1 - w) * L1 + (-w) * SSIM` per pixel. Bg
            // compositing always runs in the kernel; for synthesised opaque
            // alpha or zero bg it's a no-op. Mask multiplies the loss-map
            // by `gt.a`; for synthesised opaque alpha that's a no-op too.
            // Alpha matching needs a real alpha source (synthesised
            // a = 1 would pull predicted alpha to fully opaque); we feed
            // `pred` with 4 channels and the kernel's `c == 3` workgroup
            // emits `|pred.a - gt.a|` into the alpha channel.
            let (l1_w, ssim_w) = if self.ssim_enabled {
                (1.0 - self.config.ssim_weight, -self.config.ssim_weight)
            } else {
                (1.0, 0.0)
            };
            // Only composite when there's a real alpha channel and a non-zero
            // bg to mix in; the kernel skips the per-pixel `(1-a)*bg` math
            // entirely when this is None. `ImageLossConfig::composite_bg`
            // only takes one flat `Vec3`, so it can't express the SH
            // background's per-pixel globe — when the globe is on, the gt
            // side gets no compositing at all (not even flat-colour), and
            // only the pred side is asked to explain transparent/sky pixels.
            // That's exact for the globe's real target (photographed sky:
            // `has_alpha` is false there, so this branch is `None` either
            // way). It under-composites transparent GT for a
            // synthetic/matted dataset with real alpha and the globe on —
            // out of scope here; see `docs/superpowers/notes/sh-globe.md`.
            let composite_bg = (has_alpha && background != glam::Vec3::ZERO
                && self.sh_background.is_none())
            .then_some(background);
            let cfg = ImageLossConfig {
                l1_weight: l1_w,
                ssim_weight: ssim_w,
                composite_bg,
                mask: masked_alpha,
                alpha_weight: if do_alpha_match {
                    self.config.match_alpha_weight
                } else {
                    0.0
                },
            };

            // Flag off: `pred_final` is exactly `pred_image`, so the loss
            // call below is byte-for-byte today's call. Flag on: composite
            // the globe into the RGB channels using the render's own alpha
            // as the weight (`brush_render`'s rasterizer writes the true
            // accumulated alpha into channel 3 regardless of the render
            // background — see `kernels::rasterize`'s `final_a`), keeping
            // the alpha channel itself untouched for the alpha-match path.
            let mut globe_alpha_penalty = None;
            let pred_final = if self.sh_background.is_some() {
                // The globe is evaluated on a coarse block grid and upsampled;
                // the alpha penalty works on the same grid.
                let grid = block_grid_size(img_size);
                let (grid_h, grid_w) = (grid.y as usize, grid.x as usize);
                let basis = self.sh_basis_for(&camera, grid, &device);
                let bg_grid = self
                    .sh_background
                    .as_ref()
                    .expect("checked is_some above")
                    .image(basis)
                    .reshape([grid_h, grid_w, 3]);
                let bg_image = upsample_background(bg_grid.clone(), img_h, img_w);
                let rgb = pred_image.clone().slice(s![.., .., 0..3]);
                let alpha = pred_image.clone().slice(s![.., .., 3..4]);
                let weight = self.config.sh_background_alpha_weight;
                if weight > 0.0 {
                    // Splatfacto-W's background alpha loss: where the globe
                    // already matches the photo, opacity there is a floater.
                    let gt_grid =
                        block_mean(brush_loss::unpack_gt_rgb(gt_packed.clone(), None), grid);
                    let mask: Tensor<2> = Tensor::from_inner(background_match_mask(
                        gt_grid,
                        bg_grid.clone().inner(),
                    ));
                    let alpha_grid = block_mean(alpha.clone(), grid).reshape([grid_h, grid_w]);
                    let covered = alpha_grid * mask.clone();
                    globe_alpha_penalty =
                        Some(covered.sum() / mask.sum().clamp_min(1.0) * weight);
                }
                let composited = rgb + (1.0f32 - alpha.clone()) * bg_image;
                Tensor::cat(vec![composited, alpha], 2)
            } else {
                pred_image.clone()
            };

            // The kernel takes the RGBA image as rendered (or, with the
            // globe on, the RGBA image with the globe composited in).
            // `loss` is only reassigned by the LPIPS path below, which is
            // compiled out on wasm — so `mut` is unused there.
            let mut loss = image_loss(pred_final.clone(), gt_packed.clone(), cfg);
            if let Some(penalty) = globe_alpha_penalty {
                loss = loss + penalty;
            }

            // LPIPS still needs an f32 RGB tensor for VGG. Materialising it
            // here costs ~99 MB at 4K, only when LPIPS is enabled.
            #[cfg(not(target_family = "wasm"))]
            if let Some(lpips) = &self.lpips {
                let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
                let gt_rgb_diff: Tensor<3> = Tensor::from_inner(gt_rgb);
                loss = loss
                    + lpips.lpips(
                        pred_final.clone().slice(s![.., .., 0..3]).unsqueeze_dim(0),
                        gt_rgb_diff.unsqueeze_dim(0),
                    ) * self.config.lpips_loss_weight;
            }

            // Strip the autodiff graph off the loss so consumers can read the
            // scalar later without keeping the backward pass alive.
            let loss_inner = loss.clone().inner();
            if profiling {
                device_sync(&device);
                phases[1] = lap(&mut clock);
            }
            let mut grads = splats.bwd_validate(loss).await;

            // The globe's coefficients are a leaf in this same graph (built
            // from `sh_bg.image(..)` above, composited with the splats'
            // render by ordinary tensor ops) — this is the same `backward()`
            // the splat gradients just came out of, not a second one.
            if let Some(sh_bg) = self.sh_background.as_mut() {
                sh_bg.step(&mut grads, self.config.sh_background_lr);
            }

            let refine_weight = refine_weight_holder
                .grad_remove(&mut grads)
                .expect("XY gradients need to be calculated.")
                .without_autodiff();
            if profiling {
                device_sync(&device);
                phases[2] = lap(&mut clock);
            }
            // Reduced in the backward off the compact rows, so the dense SH
            // gradient never gets squared just to be summed away.
            let coeffs_grad_sq = diff_out
                .coeffs_grad_sq_holder
                .grad_remove(&mut grads)
                .map(Tensor::without_autodiff);

            (
                grads,
                visible,
                opacities,
                diff_out.num_visible,
                loss_inner,
                refine_weight,
                max_radius,
                coeffs_grad_sq,
            )
        };

        // The optimizer strips autodiff before stepping, so optimizer state
        // (scaling, momentum) lives on the inner device.
        let opt_device = device.clone().inner();
        let optimizer =
            self.optim.get_or_insert_with(|| {
                let sh_degree = splats.sh_degree();
                let num_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

                // DC (band 0) uses full LR; bands 1+ are scaled down.
                let mut scales = vec![1.0f32; num_coeffs];
                let rest_scale = 1.0 / self.config.lr_coeffs_sh_scale;
                for s in &mut scales[1..] {
                    *s = rest_scale;
                }
                let sh_lr_scales = Tensor::<1>::from_floats(scales.as_slice(), &opt_device)
                    .reshape([1, num_coeffs as i32, 1]);

                SplatOptim {
                    adam: AdamScaled::new(1e-15),
                    transforms: AdamState::new(None, false),
                    sh_coeffs: AdamState::new(Some(sh_lr_scales), true),
                    opacities: AdamState::new(None, false),
                }
            });

        optimizer.transforms.scaling = Some(lr_scaling);

        splats = trace_span!("Optimizer step").in_scope(|| {
            splats.transforms = trace_span!("Transforms step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    1.0,
                    splats.transforms,
                    &mut optimizer.transforms,
                    &mut grads,
                    None,
                )
            });
            splats.sh_coeffs = trace_span!("SH Coeffs step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    self.config.lr_coeffs_dc,
                    splats.sh_coeffs,
                    &mut optimizer.sh_coeffs,
                    &mut grads,
                    coeffs_grad_sq,
                )
            });
            splats.raw_opacities = trace_span!("Opacity step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    self.config.lr_opac,
                    splats.raw_opacities,
                    &mut optimizer.opacities,
                    &mut grads,
                    None,
                )
            });
            splats
        });
        if profiling {
            device_sync(&device);
            phases[3] = lap(&mut clock);
        }

        trace_span!("Housekeeping").in_scope(|| {
            // Refine state accumulates on the inner (non-autodiff) device.
            // Kept after the optimizer so it doesn't sit between the
            // backward's gradient gathers and the step that consumes them.
            let device = splats.device().inner();
            let record = self
                .refine_record
                .get_or_insert_with(|| RefineRecord::new(splats.num_splats(), &device));
            record.gather_stats(refine_weight, visible.clone(), max_radius);
            if let Some(e) = self.evict.as_mut() {
                e.life
                    .get_or_insert_with(|| SplatLife::new(splats.num_splats() as usize, &device));
            }
        });

        // Noise uses the forward's floored opacities, without building an autodiff graph.
        let samples = Tensor::random(
            [splats.num_splats() as usize, 3],
            Distribution::Normal(0.0, 1.0),
            &splats.device().inner(),
        );

        splats.transforms = splats.transforms.map(|t| {
            let inner = t.inner();
            // Resolve random generation and views before the arithmetic so the
            // gate through means addition can fuse as one rank-2 expression.
            let means = inner.clone().slice(s![.., 0..3]);
            let opacities = opacities.unsqueeze_dim::<2>(1);
            let visible = visible.unsqueeze_dim::<2>(1);
            let inv_opac: Tensor<2> = 1.0 - opacities;
            let noise_weight = inv_opac.powi_scalar(150.0).clamp(0.0, 1.0) * visible;
            let noise_weight_means =
                noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);
            let noise_m = (samples * noise_weight_means).clamp(-median_scale, median_scale);
            let noised_means = means + noise_m;
            let out = inner.slice_assign(s![.., 0..3], noised_means);
            Tensor::from_inner(out).require_grad()
        });

        if let Some(p) = self.profile.as_mut() {
            device_sync(&device);
            phases[4] = lap(&mut clock);
            p.add(phases);
        }

        let stats = TrainStepStats {
            num_visible,
            lr_mean,
            lr_rotation: self.config.lr_rotation,
            lr_scale: self.config.lr_scale,
            lr_coeffs: self.config.lr_coeffs_dc,
            lr_opac: self.config.lr_opac,
            loss: loss_inner,
        };

        (splats, stats)
    }

    pub async fn refine(&mut self, iter: u32, splats: Splats) -> (Splats, RefineStats) {
        let progress = iter as f32 / self.config.total_train_iters.max(1) as f32;
        // Refine manipulates the canonical (un-floored) params, so bake the
        // current 3D-filter floor into them first — split/clone/prune then see
        // the splat's true scales with no double-apply. A freshly recomputed
        // floor is attached at the end (below), once positions/count are known.
        let splats = splats.bake_min_scale();
        let device = splats.device();

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        let max_allowed_bounds = self.bounds.extent.max_element() * 100.0;

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let mut optim = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized");
        let alpha_mask = splats.opacities().lower_elem(MIN_OPACITY);
        let scales = splats.scales();

        // Note: we do NOT cull on a minimum scale. A genuinely flat splat
        // (a thin "pancake" representing a surface) legitimately has a tiny
        // smallest axis, so there's no correct min-scale threshold — the
        // non-finite check below still removes actually-degenerate splats.
        let scale_big = scales
            .clone()
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means() - bound_center).abs();
        let bound_mask = splat_dists
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Prune parameter that's NaN.
        fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
            t.clone().is_finite().bool_not().any_dim(1).squeeze_dim(1)
        }
        let transforms_bad = row_non_finite(&splats.transforms.val());
        let sh_bad = row_non_finite(&splats.sh_coeffs.val().flatten(1, 2));
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let non_finite_mask = transforms_bad.bool_or(sh_bad).bool_or(opac_bad);
        let num_pruned_non_finite = non_finite_mask
            .clone()
            .int()
            .sum()
            .into_scalar_async::<i32>()
            .await
            .expect("Failed to count non-finite splats") as u32;

        let prune_mask = alpha_mask
            .bool_or(scale_big)
            .bool_or(bound_mask)
            .bool_or(non_finite_mask);

        let (prune_mask, num_evicted) = self
            .add_evictions(iter, &splats, &refiner, prune_mask)
            .await;
        let life = self.evict.as_mut().and_then(|e| e.life.as_mut());
        let (mut splats, refiner, pruned_count) =
            prune_points(splats, &mut optim, refiner, life, prune_mask).await;
        let num_dead = pruned_count.saturating_sub(num_evicted);
        let mut split_inds = HashSet::new();
        // After an eviction, the freed room goes to newly observed cells only.
        let recent_mask = match self.evict.as_mut() {
            Some(e) => {
                let mask = if num_evicted > 0 {
                    let means = read_f32(splats.means()).await;
                    Some(e.recent.mask(&means, e.config.recent_refines))
                } else {
                    None
                };
                e.recent.tick();
                mask
            }
            None => None,
        };

        // Always replace dead gaussians, so that the pruned budget is reused.
        // Evicted ones are not replaced: freeing their budget is the point.
        if num_dead > 0 {
            // Replacement weighting: opacity × visibility.
            let vis_f = refiner.vis_mask().float();
            let resampled_weights = splats.opacities() * vis_f.clone();
            let resampled_weights = resampled_weights
                .into_data_async()
                .await
                .expect("Failed to get weights")
                .try_into_vec::<f32>()
                .expect("Failed to read weights");
            let resampled_inds = multinomial_sample(&mut self.rng, &resampled_weights, num_dead);
            split_inds.extend(resampled_inds);
        }

        // Force-split splats that are too big on screen (every refine). Rather
        // than killing them (the old `kill_at_screen_size`), we split them and
        // shrink the children down to `split_at_screen_size` on screen — see
        // `refine_splats`. Capped by the remaining `max_splats` budget.
        let pre_oversized = split_inds.len();
        if self.config.split_at_screen_size > 0.0 {
            let oversized = refiner.above_screen_size(self.config.split_at_screen_size);
            let oversized_inds = oversized.argwhere_async().await;
            if oversized_inds.dims()[0] > 0 {
                let oversized_inds = oversized_inds
                    .squeeze_dim::<1>(1)
                    .into_data_async()
                    .await
                    .expect("Failed to get oversized indices")
                    .try_into_vec::<i32>()
                    .expect("Failed to read oversized indices");
                let mut budget = self
                    .growth_cap()
                    .saturating_sub(splats.num_splats() + split_inds.len() as u32);
                for ind in oversized_inds {
                    if budget == 0 {
                        break;
                    }
                    if recent_mask.as_ref().is_some_and(|m| !m[ind as usize]) {
                        continue;
                    }
                    if split_inds.insert(ind) {
                        budget -= 1;
                    }
                }
            }
        }
        let num_split_oversized = (split_inds.len() - pre_oversized) as u32;

        let pre_high_grad = split_inds.len();
        if iter >= self.config.growth_start_iter && iter < self.config.growth_stop_iter {
            let above_threshold = refiner.above_threshold(self.config.growth_grad_threshold);

            let threshold_count = above_threshold
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("Failed to get threshold") as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(num_dead);

            // Saturating — cur_splats can exceed max_splats if the scene
            // was loaded above cap, and the u32 underflow would request
            // ~4B new splats.
            let cur_splats = splats.num_splats() + split_inds.len() as u32;
            let headroom = self.growth_cap().saturating_sub(cur_splats);
            let grow_count = sample_high_grad.min(headroom);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                let weights = above_threshold.float() * refiner.refine_weight_norm.clone();
                let mut weights = weights
                    .into_data_async()
                    .await
                    .expect("Failed to get weights")
                    .try_into_vec::<f32>()
                    .expect("Failed to read weights");
                if let Some(mask) = &recent_mask {
                    for (w, &recent) in weights.iter_mut().zip(mask) {
                        if !recent {
                            *w = 0.0;
                        }
                    }
                }
                // Sampling is without replacement, so it needs that many candidates.
                let candidates = weights.iter().filter(|w| w.is_finite() && **w > 0.0).count();
                let grow_count = grow_count.min(candidates as u32);
                if grow_count > 0 {
                    let growth_inds = multinomial_sample(&mut self.rng, &weights, grow_count);
                    split_inds.extend(growth_inds);
                }
            }
        }

        let num_split_high_grad = (split_inds.len() - pre_high_grad) as u32;
        let refine_count = split_inds.len();
        // Per-splat max on-screen extent, used by `refine_splats` to cap the
        // split shrink so oversized splats' children land at `split_at_screen_size`.
        let screen_sizes = refiner.max_screen_size.clone();
        splats = self.refine_splats(&device, optim, splats, split_inds, screen_sizes, iter);

        // Update current bounds based on the splats.
        self.bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        device.memory_cleanup();

        // Recompute the per-splat 3D-filter floor against the new positions/
        // count and attach it — the floor is part of the splat from here until
        // the next refine. Past the freeze fraction we stop refreshing and leave
        // it baked in, so the tail settles against fixed params.
        if progress < MIN_SCALE_FREEZE_FRAC {
            // `splats` is already on the inner backend here, so `means()` is too.
            // No-op when there are no view cameras (e.g. unit tests).
            let means = splats.means();
            if let Some(f) =
                compute_min_scale(&means, &self.view_cams, self.config.min_scale_factor)
            {
                splats = splats.with_min_scale(f);
            }
        }

        let splat_count = splats.num_splats();

        (
            splats,
            RefineStats {
                num_added: refine_count as u32,
                num_split_oversized,
                num_split_high_grad,
                num_pruned: pruned_count,
                num_pruned_non_finite,
                num_evicted,
                total_splats: splat_count,
            },
        )
    }

    fn refine_splats(
        &mut self,
        device: &Device,
        mut optim: SplatOptim,
        mut splats: Splats,
        split_inds: HashSet<i32>,
        screen_sizes: Tensor<1>,
        iter: u32,
    ) -> Splats {
        let refine_count = split_inds.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect::<Vec<_>>(), [refine_count]),
                device,
            );

            let cur_transforms = splats.transforms.val().select(0, refine_inds.clone());
            let cur_means = cur_transforms.clone().slice(s![.., 0..3]);
            let cur_rots_raw = cur_transforms.clone().slice(s![.., 3..7]);
            let magnitudes = Tensor::clamp_min(
                Tensor::sum_dim(cur_rots_raw.clone().powi_scalar(2), 1).sqrt(),
                1e-32,
            );
            let cur_rots = cur_rots_raw.clone() / magnitudes;
            let cur_log_scale = cur_transforms.slice(s![.., 7..10]);
            let cur_sh_coeffs = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            let cur_scales = cur_log_scale.clone().exp();

            let cur_opac = sigmoid(cur_raw_opac);
            let inv_opac: Tensor<1> = 1.0 - cur_opac;
            // Post-split child opacity as a power law in transmittance,
            // p = 0.5 would keep the transmittance for cloning splats but as we offset them
            // choose a higher p.
            let new_opac: Tensor<1> = 1.0 - inv_opac.powf_scalar(FRAC_1_SQRT_2);
            let new_raw_opac = inv_sigmoid(new_opac.clamp(MIN_OPACITY, 1.0 - MIN_OPACITY));

            // Smooth covariance-aware split. Per-axis shrink + mass-conserving
            // deterministic offset (one child at +offset, the other at -offset).
            // Children inherit the
            // parent's rotation; the split is the scale shrink + ±offset.
            let cur_scales_sq = cur_scales.clone().powi_scalar(2);
            let max_scale_sq = cur_scales_sq.clone().max_dim(1).clamp_min(1e-30);
            let ratio = cur_scales_sq / max_scale_sq;
            // Max-axis shrink factor `k` (per splat). The standard split uses
            // 1/√2 (mass-conserving). When `split_at_screen_size` is set, splats
            // that are too big on screen shrink harder so their children land at
            // (at most) the cap: `k = min(1/√2, split_at_screen_size / screen)`.
            // Splats already within √2× of the cap are unaffected (min → 1/√2).
            let k_per_axis: Tensor<2> = if self.config.split_at_screen_size > 0.0 {
                let k_max = screen_sizes
                    .select(0, refine_inds.clone())
                    .unsqueeze_dim(1)
                    .clamp_min(1e-6)
                    .recip()
                    .mul_scalar(self.config.split_at_screen_size)
                    .clamp_max(FRAC_1_SQRT_2);
                -(ratio * (-k_max + 1.0)) + 1.0
            } else {
                -(ratio * (1.0_f32 - FRAC_1_SQRT_2)) + 1.0
            };
            let offset_factor = (-k_per_axis.clone().powi_scalar(2) + 1.0)
                .clamp_min(0.0)
                .sqrt();
            let offset_local = offset_factor * cur_scales;
            let samples = quaternion_vec_multiply(cur_rots.clone(), offset_local);
            let new_log_scales = cur_log_scale + k_per_axis.log();
            let child_rots = cur_rots;

            let parent_transforms = Tensor::cat(
                vec![
                    cur_means.clone() - samples.clone(),
                    cur_rots_raw,
                    new_log_scales.clone(),
                ],
                1,
            );
            splats.transforms = splats
                .transforms
                .map(|t| t.select_assign(0, refine_inds.clone(), parent_transforms, Assign));
            splats.raw_opacities = splats
                .raw_opacities
                .map(|m| m.select_assign(0, refine_inds.clone(), new_raw_opac.clone(), Assign));

            // Child sits at parent_mean + samples (parent moves to
            // parent_mean - samples) — anti-correlated, centroid-preserving.
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let new_transforms =
                Tensor::cat(vec![cur_means + samples, child_rots, new_log_scales], 1);

            // Optimizer state lives on the inner (non-autodiff) device.
            let opt_device = device.clone().inner();
            let opt_inds = refine_inds.to_device(&opt_device);

            splats = map_splats_and_opt(
                splats,
                &mut optim,
                |x| Tensor::cat(vec![x, new_transforms], 0),
                |x| Tensor::cat(vec![x, cur_sh_coeffs], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x: Tensor<2>| {
                    let d1 = x.dims()[1];
                    let zeros = Tensor::zeros([refine_count, d1], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
                |x: Tensor<3>| {
                    let [_, d1, d2] = x.dims();
                    let zeros = Tensor::zeros([refine_count, d1, d2], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
                |x: Tensor<1>| {
                    let zeros = Tensor::zeros([refine_count], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
            );
            if let Some(life) = self.evict.as_mut().and_then(|e| e.life.as_mut()) {
                life.split(opt_inds);
            }
        }

        let train_t = (iter as f32 / self.config.total_train_iters.max(1) as f32).clamp(0.0, 1.0);
        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f) - minus_opac;
            inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
        });

        self.optim = Some(optim);
        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats,
    optim: &mut SplatOptim,
    map_transforms: impl FnOnce(Tensor<2>) -> Tensor<2>,
    map_sh_coeffs: impl FnOnce(Tensor<3>) -> Tensor<3>,
    map_opac: impl FnOnce(Tensor<1>) -> Tensor<1>,

    map_opt_transforms: impl Fn(Tensor<2>) -> Tensor<2>,
    map_opt_sh_coeffs: impl Fn(Tensor<3>) -> Tensor<3>,
    map_opt_opac: impl Fn(Tensor<1>) -> Tensor<1>,
) -> Splats {
    splats.transforms = splats.transforms.map(map_transforms);
    optim.transforms.map_momentum(map_opt_transforms);
    splats.sh_coeffs = splats.sh_coeffs.map(map_sh_coeffs);
    optim.sh_coeffs.map_momentum(map_opt_sh_coeffs);
    splats.raw_opacities = splats.raw_opacities.map(map_opac);
    optim.opacities.map_momentum(map_opt_opac);
    splats
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
async fn prune_points(
    mut splats: Splats,
    optim: &mut SplatOptim,
    mut refiner: RefineRecord,
    life: Option<&mut SplatLife>,
    prune: Tensor<1, Bool>,
) -> (Splats, RefineRecord, u32) {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    let prune_count = prune.dims()[0];
    if prune_count == 0 {
        return (splats, refiner, 0);
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return (splats, refiner, 0);
    }

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0] as u32;
    if new_points < start_splats {
        let valid_inds = valid_inds.squeeze_dim(1);
        let inner_valid_inds = valid_inds.clone().without_autodiff();
        splats = map_splats_and_opt(
            splats,
            optim,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
        );
        if let Some(life) = life {
            life.keep(inner_valid_inds.clone());
        }
        refiner = refiner.keep(inner_valid_inds);
    }
    (splats, refiner, start_splats - new_points)
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color<R: rand::Rng + ?Sized>(
    base: glam::Vec3,
    strength: f32,
    rng: &mut R,
) -> glam::Vec3 {
    if strength <= 0.0 {
        return base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    }
    use rand::RngExt as _;
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod append_tests {
    use super::*;
    use brush_dataset::scene::{SceneBatch, view_to_packed_data};
    use brush_render::AlphaMode;
    use brush_render::camera::Camera;
    use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
    use brush_render::kernels::camera_model::CameraModel;
    use burn::module::Module;
    use clap::Parser;

    fn splats(n: usize, z: f32, device: &Device) -> Splats {
        Splats::from_raw(
            (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, z]).collect(),
            [1.0, 0.0, 0.0, 0.0].repeat(n),
            vec![-3.0; n * 3],
            vec![0.5; n * 3],
            vec![inverse_sigmoid(0.5); n],
            SplatRenderMode::Default,
            device,
        )
    }

    fn batch() -> SceneBatch {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            32,
            32,
            image::Rgb([255, 0, 0]),
        ));
        let (img_packed, has_alpha) = view_to_packed_data(img, AlphaMode::Transparent);
        let fov = 60f64.to_radians();
        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Transparent,
            camera: Camera::new(
                glam::Vec3::ZERO,
                glam::Quat::IDENTITY,
                fov,
                fov,
                glam::vec2(0.5, 0.5),
                CameraModel::Pinhole,
            ),
        }
    }

    #[tokio::test]
    async fn append_then_step_and_refine() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let config = TrainConfig::parse_from(["test"]);
        let base = splats(50, 2.0, &device);
        let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
        let mut trainer = SplatTrainer::new(&config, &device, bounds);

        let (s, _) = trainer.step(batch(), base.train()).await;
        let s = s.valid();
        let s = trainer.append_splats(s, splats(20, 2.5, &device).valid());
        assert_eq!(s.num_splats(), 70);

        let (s, _) = trainer.step(batch(), s.train()).await;
        let (s, _) = trainer.refine(1, s.valid()).await;
        assert!(s.num_splats() >= 1);
    }

    #[tokio::test]
    async fn append_before_first_step() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let config = TrainConfig::parse_from(["test"]);
        let base = splats(10, 2.0, &device).valid();
        let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
        let mut trainer = SplatTrainer::new(&config, &device, bounds);
        let s = trainer.append_splats(base, splats(5, 2.0, &device).valid());
        assert_eq!(s.num_splats(), 15);
        let (s, _) = trainer.step(batch(), s.train()).await;
        assert_eq!(s.num_splats(), 15);
    }

    #[tokio::test]
    async fn append_keeps_mip_floor() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let config = TrainConfig::parse_from(["test"]);
        let base = splats(50, 2.0, &device);
        let bounds = get_splat_bounds(base.clone(), BOUND_PERCENTILE).await;
        let mut trainer = SplatTrainer::new(&config, &device, bounds);
        trainer.set_view_cams(vec![(glam::Vec3::ZERO, 30.0)]);

        let (s, _) = trainer.step(batch(), base.train()).await;
        let (s, _) = trainer.refine(1, s.valid()).await;
        let pre_count = s.num_splats() as usize;
        let pre_floor = s
            .min_scale
            .clone()
            .expect("refine with view cams should attach a Mip floor")
            .into_data_async()
            .await
            .expect("floor readback")
            .try_into_vec::<f32>()
            .expect("floor readback");

        let s = trainer.append_splats(s, splats(20, 2.5, &device).valid());
        assert_eq!(s.num_splats() as usize, pre_count + 20);

        let floor = s
            .min_scale
            .clone()
            .expect("Mip floor should survive append_splats");
        assert_eq!(floor.dims()[0], pre_count + 20);
        let floor_vals = floor
            .into_data_async()
            .await
            .expect("floor readback")
            .try_into_vec::<f32>()
            .expect("floor readback");
        assert_eq!(&floor_vals[..pre_count], &pre_floor[..]);

        let (s, _) = trainer.step(batch(), s.train()).await;
        assert_eq!(s.num_splats() as usize, pre_count + 20);
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "train_evict_tests.rs"]
mod evict_tests;

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "profile_tests.rs"]
mod profile_tests;

#[cfg(all(test, not(target_family = "wasm")))]
#[path = "sh_background_integration_tests.rs"]
mod sh_background_integration_tests;
