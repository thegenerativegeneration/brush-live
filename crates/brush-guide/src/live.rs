use crate::config::GuideConfig;
use crate::keyframe::DecodedKeyframe;
use crate::mono::{MonoSeed, ScaleFit, mono_seed_for};
use crate::seed::{SeedInput, Seeds, seed_points, stride_for};
use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::scene::{Scene, SceneBatch, SceneView, view_to_packed_data};
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, TextureMode, render_splats};
use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use brush_train::config::TrainConfig;
use brush_train::evict::{EvictConfig, ProtectCone};
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_train::{RandomSplatsConfig, create_random_splats, to_init_splats};
use burn::module::Module;
use burn::tensor::{Device, s};
use clap::Parser;
use glam::{UVec2, Vec3};
use rand::{RngExt as _, SeedableRng};
use std::collections::HashSet;
use std::ops::Range;

/// `LoadDatasetConfig` is a clap `Args` group; this wrapper parses its CLI defaults.
#[derive(Parser)]
struct LoadArgs {
    #[command(flatten)]
    load: LoadDatasetConfig,
}

/// Recent-window loader is rebuilt every this many new keyframes (and for each
/// of the first ones): every rebuild spawns a fresh set of loader threads.
const RECENT_LOADER_REBUILD_EVERY: usize = 3;

/// With debug timing on, the step profile is logged every this many steps.
const PROFILE_LOG_EVERY: u32 = 100;

/// Splats trained incrementally as keyframes arrive. Splats are kept on the
/// inner (non-autodiff) device between steps.
pub struct LiveModel {
    config: GuideConfig,
    device: Device,
    load_config: LoadDatasetConfig,
    train_config: TrainConfig,
    splats: Option<Splats>,
    trainer: Option<SplatTrainer>,
    views: Vec<SceneView>,
    /// Per view: (camera position, focal length in px) for the Mip 3D filter.
    view_cams: Vec<(Vec3, f32)>,
    ids: HashSet<u64>,
    last_id: Option<u64>,
    recent: Option<SceneLoader>,
    all: Option<SceneLoader>,
    /// Views in the recent loader; those after `views_in_recent` are in no
    /// recent pool yet and get drawn directly in `train_step`.
    recent_len: usize,
    views_in_recent: usize,
    views_in_all: usize,
    iter: u32,
    /// Splats evicted to stay within the budget, in total.
    num_evicted: u64,
    /// Refines and their time since `take_refine_stats`, for debug timing.
    refine_stats: (u32, f64),
    /// Seconds spent getting training batches since the last profile log.
    batch_s: f64,
    /// Whether the trainer times its phases; `batch_s` only accumulates then.
    profiling: bool,
    rng: rand::rngs::StdRng,
}

impl LiveModel {
    /// `device` is the autodiff device.
    pub fn new(config: GuideConfig, device: Device) -> Self {
        let mut train_config = TrainConfig::parse_from(["brush-guide"]);
        train_config.total_train_iters = 1_000_000_000;
        train_config.lr_mean_end = train_config.lr_mean;
        train_config.growth_stop_iter = 1_000_000_000;
        train_config.max_splats = config.max_splats;
        train_config.refine_every = config.refine_every;
        train_config.sh_background = config.sh_background;
        train_config.sh_background_alpha_weight = config.sh_background_alpha_weight;
        train_config.ssim_every = config.ssim_every;
        let mut load_config = LoadArgs::parse_from(["brush-guide"]).load;
        load_config.max_scene_batch_cache_size = config.loader_cache_bytes;
        let rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        Self {
            config,
            device,
            load_config,
            train_config,
            splats: None,
            trainer: None,
            views: Vec::new(),
            view_cams: Vec::new(),
            ids: HashSet::new(),
            last_id: None,
            recent: None,
            all: None,
            recent_len: 0,
            views_in_recent: 0,
            views_in_all: 0,
            iter: 0,
            num_evicted: 0,
            refine_stats: (0, 0.0),
            batch_s: 0.0,
            profiling: false,
            rng,
        }
    }

    pub fn splats(&self) -> Option<&Splats> {
        self.splats.as_ref()
    }

    /// The SH background's coefficients as JSON, or `None` when the flag is
    /// off or training hasn't started. For the session-dir dump on finish.
    pub async fn sh_background_json(&self) -> Option<String> {
        self.trainer.as_ref()?.sh_background_json().await
    }

    pub fn views(&self) -> &[SceneView] {
        &self.views
    }

    /// True if a keyframe with this id was already added.
    pub fn contains(&self, id: u64) -> bool {
        self.ids.contains(&id)
    }

    pub fn last_keyframe_id(&self) -> Option<u64> {
        self.last_id
    }

    /// Splats evicted to stay within the budget since the model started.
    pub fn num_evicted(&self) -> u64 {
        self.num_evicted
    }

    /// The trainer's eviction importance in current splat order; `None`
    /// without eviction or before training starts.
    pub async fn importance(&self) -> Option<Vec<f32>> {
        self.trainer.as_ref()?.importance().await
    }

    /// Refines and their seconds since the last call.
    pub fn take_refine_stats(&mut self) -> (u32, f64) {
        std::mem::take(&mut self.refine_stats)
    }

    /// Number of training steps taken.
    #[allow(clippy::iter_not_returning_iterator)]
    pub fn iter(&self) -> u32 {
        self.iter
    }

    /// Adds a view and seeds splats where the current model is transparent.
    /// Returns false if a keyframe with this id was already added.
    pub async fn add_keyframe(&mut self, mut kf: DecodedKeyframe) -> bool {
        if !self.ids.insert(kf.id) {
            return false;
        }
        kf.depth = kf.depth.map(|d| d.masked(self.config.min_depth_confidence));
        let size = UVec2::new(kf.image.width(), kf.image.height());
        self.view_cams
            .push((kf.camera.position, kf.camera.focal(size).x));

        let splats = match self.splats.take() {
            None => {
                let init = self.initial_splats(&kf, size);
                let bounds = get_splat_bounds(init.clone(), BOUND_PERCENTILE).await;
                let mut trainer = SplatTrainer::new_seeded(
                    &self.train_config,
                    &self.device,
                    bounds,
                    self.config.seed,
                );
                trainer.set_view_cams(self.view_cams.clone());
                trainer.set_profiling(self.config.profile_steps);
                self.profiling = self.config.profile_steps;
                if self.config.evict {
                    trainer.enable_eviction(EvictConfig {
                        headroom: self.config.evict_headroom,
                        min_age: self.config.evict_min_age,
                        max_cell_fraction: self.config.evict_max_cell_fraction,
                        recent_refines: self.config.evict_recent_refines,
                    });
                }
                self.trainer = Some(trainer);
                init
            }
            Some(current) => {
                let new = self.seed_from_mask(current.clone(), &kf, size).await;
                let trainer = self
                    .trainer
                    .as_mut()
                    .expect("trainer exists once splats exist");
                // Set before appending so the new splats' Mip floor sees this view.
                trainer.set_view_cams(self.view_cams.clone());
                match new {
                    Some(new) => trainer.append_splats(current, new),
                    None => current,
                }
            }
        };
        self.splats = Some(splats);
        if let Some(t) = self.trainer.as_mut() {
            t.set_protect_cone(Some(view_cone(&kf.view.camera)));
        }

        self.views.push(kf.view);
        self.last_id = Some(kf.id);
        self.rebuild_loaders();
        true
    }

    fn rebuild_loaders(&mut self) {
        let n = self.views.len();
        if n <= RECENT_LOADER_REBUILD_EVERY
            || n >= self.views_in_recent + RECENT_LOADER_REBUILD_EVERY
        {
            let start = n.saturating_sub(self.config.recent_window);
            self.recent_len = n - start;
            let recent = Scene::new(self.views[start..].to_vec());
            self.recent = Some(SceneLoader::new(
                &recent,
                self.config.seed + n as u64,
                &self.load_config,
            ));
            self.views_in_recent = n;
        }
        if self.all.is_none()
            || self.views.len() >= self.views_in_all + self.config.all_loader_rebuild_every
        {
            let all = Scene::new(self.views.clone());
            self.all = Some(SceneLoader::new(&all, self.config.seed, &self.load_config));
            self.views_in_all = self.views.len();
        }
    }

    /// Keeps a seeded random subset of `seeds` that fits under `max_splats`
    /// next to `current` existing splats.
    /// Also marks the cells `seeds` fall in as newly observed for eviction.
    fn cap_seeds(&mut self, seeds: Seeds, current: u32) -> Seeds {
        if let Some(t) = self.trainer.as_mut() {
            t.note_keyframe_seeds(&seeds.means);
        }
        let room = self.config.max_splats.saturating_sub(current) as usize;
        if seeds.colors.len() <= room {
            return seeds;
        }
        if let Some(t) = self.trainer.as_mut() {
            t.note_seed_shortfall((seeds.colors.len() - room) as u32);
        }
        let mut keep = rand::seq::index::sample(&mut self.rng, seeds.colors.len(), room).into_vec();
        keep.sort_unstable();
        Seeds {
            means: keep
                .iter()
                .flat_map(|&i| seeds.means[i * 3..i * 3 + 3].iter().copied())
                .collect(),
            colors: keep.iter().map(|&i| seeds.colors[i]).collect(),
            mono: seeds.mono,
        }
    }

    fn mono_input<'a>(
        &self,
        kf: &'a DecodedKeyframe,
        size: UVec2,
    ) -> (Option<MonoSeed<'a>>, ScaleFit) {
        mono_seed_for(
            kf.mono.as_ref(),
            kf.depth.as_ref(),
            &kf.camera,
            size,
            &kf.points,
            &self.config,
        )
    }

    /// Seed grid spacing for a keyframe decoded at `size`.
    fn seed_stride(&self, size: UVec2) -> u32 {
        stride_for(
            size.x,
            size.y,
            self.config.max_splats,
            self.config.seed_view_fraction,
            self.config.seed_stride_px,
        )
    }

    fn initial_splats(&mut self, kf: &DecodedKeyframe, size: UVec2) -> Splats {
        let (mono, fit) = self.mono_input(kf, size);
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &vec![0.0; (size.x * size.y) as usize],
            alpha_size: size,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: self.seed_stride(size),
            alpha_threshold: self.config.seed_alpha_threshold,
            mono,
        });
        log::debug!(
            target: crate::timing::TARGET,
            "keyframe {}: {} initial seeds (depth {}, mono {fit}, {} mono seeds)",
            kf.id,
            seeds.colors.len(),
            kf.depth.is_some(),
            seeds.mono
        );
        let seeds = self.cap_seeds(seeds, 0);
        let splats = if seeds.colors.len() >= 3 {
            self.seeds_to_splats(seeds.means, &seeds.colors)
        } else {
            let count = self
                .config
                .init_random_count
                .min(self.config.max_splats as usize);
            let cfg = RandomSplatsConfig::new().with_init_count(count);
            create_random_splats(
                &cfg,
                &[kf.camera],
                None,
                &mut self.rng,
                SplatRenderMode::Default,
                &self.device.clone().inner(),
            )
        };
        splats.with_sh_degree(self.config.sh_degree)
    }

    async fn seed_from_mask(
        &mut self,
        splats: Splats,
        kf: &DecodedKeyframe,
        size: UVec2,
    ) -> Option<Splats> {
        let current = splats.num_splats();
        let small = (size / 4).max(UVec2::ONE);
        let (img, _) = render_splats(
            splats,
            &kf.camera,
            small,
            Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
        let alpha = img
            .slice(s![.., .., 3..4])
            .into_data_async()
            .await
            .expect("alpha readback")
            .try_to_vec::<f32>()
            .expect("f32 alpha");
        let (mono, fit) = self.mono_input(kf, size);
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &alpha,
            alpha_size: small,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: (self.seed_stride(size) / 4).max(1),
            alpha_threshold: self.config.seed_alpha_threshold,
            mono,
        });
        log::debug!(
            target: crate::timing::TARGET,
            "keyframe {}: {} seeds (depth {}, mono {fit}, {} mono seeds) onto {current} splats",
            kf.id,
            seeds.colors.len(),
            kf.depth.is_some(),
            seeds.mono
        );
        let seeds = self.cap_seeds(seeds, current);
        (seeds.colors.len() >= 3).then(|| {
            self.seeds_to_splats(seeds.means, &seeds.colors)
                .with_sh_degree(self.config.sh_degree)
        })
    }

    fn seeds_to_splats(&self, means: Vec<f32>, colors: &[Vec3]) -> Splats {
        let sh: Vec<f32> = colors
            .iter()
            .flat_map(|c| rgb_to_sh(*c).to_array())
            .collect();
        let data = SplatData {
            means,
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(sh),
            raw_opacities: None,
        };
        to_init_splats(data, SplatRenderMode::Default, &self.device.clone().inner())
    }

    /// One optimisation step on a batch drawn from the recent window (with
    /// probability `recent_fraction`) or from all views. No-op without views.
    pub async fn train_step(&mut self) {
        let (Some(splats), Some(trainer)) = (self.splats.take(), self.trainer.as_mut()) else {
            return;
        };
        let use_recent = self.rng.random::<f32>() < self.config.recent_fraction;
        let pending = use_recent
            .then(|| {
                pick_pending(
                    &mut self.rng,
                    self.recent_len,
                    self.views_in_recent..self.views.len(),
                )
            })
            .flatten();
        let t_batch = web_time::Instant::now();
        let batch = if let Some(i) = pending {
            load_batch(&self.views[i]).await
        } else {
            let loader = if use_recent {
                self.recent.as_mut()
            } else {
                self.all.as_mut()
            };
            loader
                .expect("loaders exist once views exist")
                .next_batch()
                .await
        };
        if self.profiling {
            self.batch_s += t_batch.elapsed().as_secs_f64();
        }
        let (stepped, _) = trainer.step(batch, splats.train()).await;
        let mut splats = stepped.valid();
        self.iter += 1;
        if self.iter.is_multiple_of(PROFILE_LOG_EVERY)
            && let Some(p) = trainer.take_profile()
            && p.steps > 0
        {
            let ms = |s: f64| s * 1e3 / f64::from(p.steps);
            log::debug!(
                target: crate::timing::TARGET,
                "step profile over {} steps at {} splats: batch {:.2} ms, forward {:.2}, loss {:.2}, backward {:.2}, optimizer {:.2}, stats+noise {:.2} ms",
                p.steps,
                splats.num_splats(),
                ms(std::mem::take(&mut self.batch_s)),
                ms(p.forward_s),
                ms(p.loss_s),
                ms(p.backward_s),
                ms(p.optimizer_s),
                ms(p.stats_noise_s)
            );
        }
        if self.iter.is_multiple_of(self.config.refine_every) {
            crate::timing::sync_splats(&splats).await;
            let t = web_time::Instant::now();
            let before = splats.num_splats();
            let (refined, stats) = trainer.refine(self.iter, splats).await;
            splats = refined;
            crate::timing::sync_splats(&splats).await;
            let dt = t.elapsed().as_secs_f64();
            self.refine_stats.0 += 1;
            self.refine_stats.1 += dt;
            log::debug!(
                target: crate::timing::TARGET,
                "refine at iter {}: {:.0} ms, splats {before} -> {}, {} evicted",
                self.iter,
                dt * 1e3,
                splats.num_splats(),
                stats.num_evicted
            );
            self.num_evicted += u64::from(stats.num_evicted);
        }
        self.splats = Some(splats);
    }
}

/// For a recent-window draw: one of the `pending` views (added since the
/// recent loader was built) to train on directly, or `None` to draw from the
/// loader's `loader_len` views. Every view gets the same share.
fn pick_pending(
    rng: &mut impl rand::Rng,
    loader_len: usize,
    pending: Range<usize>,
) -> Option<usize> {
    if pending.is_empty() {
        return None;
    }
    let r = rng.random_range(0..loader_len + pending.len());
    (r >= loader_len).then(|| pending.start + r - loader_len)
}

/// A training batch for one view, decoded and packed as `SceneLoader` does.
async fn load_batch(view: &SceneView) -> SceneBatch {
    let raw = view.image.load().await.expect("keyframe image decodes");
    let (img_packed, has_alpha) = view_to_packed_data(raw, view.image.alpha_mode());
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: view.image.alpha_mode(),
        camera: view.camera,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_pending_views_leave_the_rng_untouched() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut fresh = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..10 {
            assert!(pick_pending(&mut rng, 20, 20..20).is_none());
        }
        assert_eq!(rng.random::<u64>(), fresh.random::<u64>());
    }

    #[test]
    fn pending_views_after_the_window_slid_map_to_their_indices() {
        // 35 views pooled at the last rebuild (the loader holds the last 20), 2 new since.
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let picked: std::collections::HashSet<usize> = (0..2000)
            .filter_map(|_| pick_pending(&mut rng, 20, 35..37))
            .collect();
        assert_eq!(picked, [35, 36].into_iter().collect());
    }

    #[test]
    fn pending_views_get_a_loader_views_share() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let draws = 22_000;
        let mut hits = [0usize; 2];
        for _ in 0..draws {
            if let Some(i) = pick_pending(&mut rng, 20, 20..22) {
                hits[i - 20] += 1;
            }
        }
        // Each of the 22 views should get ~1/22 of the draws (1000).
        for h in hits {
            assert!((850..1150).contains(&h), "pending view drawn {h} times");
        }
    }

    #[test]
    fn no_pending_views_always_uses_the_loader() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!((0..1000).all(|_| pick_pending(&mut rng, 20, 20..20).is_none()));
    }
}

/// The camera's viewing cone, as `session::worker` approximates a view.
fn view_cone(c: &brush_render::camera::Camera) -> ProtectCone {
    ProtectCone {
        position: c.position,
        forward: c.rotation * Vec3::Z,
        cos_half_fov: (0.5 * c.fov_x.max(c.fov_y) as f32).cos(),
    }
}
