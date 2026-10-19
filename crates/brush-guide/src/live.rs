use crate::config::GuideConfig;
use crate::keyframe::DecodedKeyframe;
use crate::mono::ScaleHistory;
use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::scene::SceneView;
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::Splats;
use brush_train::config::TrainConfig;
use brush_train::evict::{EvictConfig, ProtectCone};
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::module::Module;
use burn::tensor::Device;
use clap::Parser;
use glam::{UVec2, Vec3};
use rand::SeedableRng;
use std::collections::HashMap;
use std::path::PathBuf;

mod area;
mod area_active;
mod area_budget;
mod area_export;
mod area_freeze;
mod area_park;
mod area_restore;
mod area_sight;
mod draw;
mod loaders;
mod poses;
mod seed;
use area_restore::RestoreReason;
use draw::{Draw, choose_draw, load_batch};
pub use area_export::FinalExport;
pub use poses::{PoseUpdateStats, ViewPose};

#[cfg(test)]
mod scale_history_tests;

/// `LoadDatasetConfig` is a clap `Args` group; this wrapper parses its CLI defaults.
#[derive(Parser)]
struct LoadArgs {
    #[command(flatten)]
    load: LoadDatasetConfig,
}

/// With debug timing on, the step profile is logged every this many steps.
const PROFILE_LOG_EVERY: u32 = 100;

/// Splats trained incrementally as keyframes arrive. Splats are kept on the
/// inner (non-autodiff) device between steps.
/// One training step's mean learning rate and the scene size that scaled it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepLr {
    pub lr_mean: f64,
    /// The bounds' median size in metres, which scales the mean LR and clamps the position noise.
    pub position_scale_m: f32,
}

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
    /// View index by keyframe id.
    index: HashMap<u64, usize>,
    last_id: Option<u64>,
    /// Scales of the last LiDAR / feature mono fits, for keyframes with too
    /// few samples of their own; `reset` keeps it.
    scale_history: ScaleHistory,
    /// Area training's state; `None` with it off (see [`Self::enable_area`]).
    area: Option<area::AreaState>,
    /// Where parked tiles live; kept by `reset`, which turns area training on again with it.
    tiles_dir: Option<PathBuf>,
    recent: Option<SceneLoader>,
    all: Option<SceneLoader>,
    /// Views (indices into `views`) in the recent loader.
    recent_members: Vec<usize>,
    /// `views.len()` when the recent loader was built: training views from this index on are in no recent pool
    /// yet and get drawn directly in `train_step`.
    views_in_recent: usize,
    /// Views in the all-views loader.
    all_members: Vec<usize>,
    /// How often the all-views loader was built.
    all_loader_builds: u32,
    iter: u32,
    /// Splats evicted to stay within the budget, in total.
    num_evicted: u64,
    /// Splats the last refine evicted for the total (not for a tile's own cap); a floor for what the next refine parks
    /// before it evicts.
    last_refine_evicted: u32,
    /// Refines and their time since `take_refine_stats`, for debug timing.
    refine_stats: (u32, f64),
    /// Refine wall time since the model started, kept apart from `take_refine_stats`.
    refine_total_s: f64,
    last_step_lr: StepLr,
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
        train_config.train_radius_m = config.train_radius_m;
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
            index: HashMap::new(),
            last_id: None,
            scale_history: ScaleHistory::default(),
            area: None,
            tiles_dir: None,
            recent: None,
            all: None,
            recent_members: Vec::new(),
            views_in_recent: 0,
            all_members: Vec::new(),
            all_loader_builds: 0,
            iter: 0,
            num_evicted: 0,
            last_refine_evicted: 0,
            refine_stats: (0, 0.0),
            refine_total_s: 0.0,
            last_step_lr: StepLr::default(),
            batch_s: 0.0,
            profiling: false,
            rng,
        }
    }

    /// Starts over from no keyframes with the same config and device, but
    /// keeps the mono scale history: a scale relates the network's depth to
    /// camera depth and does not depend on the segment's world. With a tiles
    /// directory, area training starts again on it (emptied).
    pub fn reset(&mut self) {
        let history = std::mem::take(&mut self.scale_history);
        let tiles_dir = self.tiles_dir.take();
        *self = Self::new(self.config.clone(), self.device.clone());
        self.scale_history = history;
        if let Some(dir) = tiles_dir {
            self.enable_area(dir);
        }
    }

    /// The scale a keyframe with too few samples of its own would use
    /// (`ScaleSource::Recent`), `None` before the first LiDAR or feature fit.
    pub fn scale_median(&self) -> Option<f32> {
        self.scale_history.median()
    }

    pub fn splats(&self) -> Option<&Splats> {
        self.splats.as_ref()
    }

    /// The SH background's coefficients as JSON, or `None` when the flag is
    /// off or training hasn't started. For the session-dir dump on finish.
    pub async fn sh_background_json(&self) -> Option<String> {
        self.trainer.as_ref()?.sh_background_json().await
    }

    /// The trainer's SH environment background, which held-out evals composite behind their renders; `None`
    /// when the flag is off or training hasn't started.
    pub fn sh_background(&self) -> Option<&brush_train::sh_background::ShBackground> {
        self.trainer.as_ref()?.sh_background()
    }

    pub fn views(&self) -> &[SceneView] {
        &self.views
    }

    /// True if a keyframe with this id was already added.
    pub fn contains(&self, id: u64) -> bool {
        self.index.contains_key(&id)
    }

    pub fn last_keyframe_id(&self) -> Option<u64> {
        self.last_id
    }

    /// Splats evicted to stay within the budget since the model started.
    pub fn num_evicted(&self) -> u64 {
        self.num_evicted
    }

    /// Optimizer elements whose step was clamped since the model started, counted up to the last refine.
    pub fn adam_clamped(&self) -> u64 {
        self.trainer.as_ref().map_or(0, SplatTrainer::adam_clamped)
    }

    /// Splats the last training step left alone for lying beyond `train_radius_m`; `None` with the radius off or
    /// before the first step.
    pub async fn masked_by_radius(&self) -> Option<u32> {
        let count = self.trainer.as_ref()?.masked_by_radius()?;
        let data = count.into_data_async().await.ok()?.convert::<i64>();
        data.try_to_vec::<i64>().ok()?.first().map(|&n| n as u32)
    }

    /// Splats refine pruned for a diverged colour since the model started.
    pub fn sh_pruned(&self) -> u64 {
        self.trainer.as_ref().map_or(0, SplatTrainer::sh_pruned)
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

    /// Milliseconds spent in refine since the model started.
    pub fn refine_total_ms(&self) -> f64 {
        self.refine_total_s * 1e3
    }

    /// The last training step's mean LR and position scale; zeros before the first.
    pub fn last_step_lr(&self) -> StepLr {
        self.last_step_lr
    }

    /// Median size of the trainer's bounds, which scales the mean learning rate; 0 before training.
    pub fn bounds_median_size(&self) -> f32 {
        self.trainer.as_ref().map_or(0.0, |t| t.bounds().median_size())
    }

    /// Makes the trainer work towards `config.max_splats`, the one splat budget (it may change between refines).
    fn sync_max_splats(&mut self) {
        self.train_config.max_splats = self.config.max_splats;
        if let Some(t) = self.trainer.as_mut() {
            t.set_max_splats(self.config.max_splats);
        }
    }

    /// Number of training steps taken.
    #[allow(clippy::iter_not_returning_iterator)]
    pub fn iter(&self) -> u32 {
        self.iter
    }

    /// Adds a view and seeds splats where the current model is transparent. With area training, parked tiles within
    /// the radius of its camera come back first, then those its seeds hit nearby (see `seed_from_mask`); with
    /// `area_freeze` inactive(_at_cap) the keyframe ranks the active tiles and the rows outside them are held at its end
    /// (inactive_at_cap: at the cap only). Seeds into a tile at the per-tile cap are dropped.
    /// Returns false if a keyframe with this id was already added.
    pub async fn add_keyframe(&mut self, mut kf: DecodedKeyframe) -> bool {
        if self.index.contains_key(&kf.id) {
            return false;
        }
        self.index.insert(kf.id, self.views.len());
        let slot = self.views.len() as u32;
        kf.depth = kf.depth.map(|d| d.masked(self.config.min_depth_confidence));
        let size = UVec2::new(kf.image.width(), kf.image.height());
        self.view_cams
            .push((kf.camera.position, kf.camera.focal(size).x));
        if let Some(area) = self.area.as_mut() {
            area.registry.note_walker(kf.camera.position);
        }

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
                if self.config.move_anchored_splats {
                    trainer.track_origins(&init, slot);
                }
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
                let pos = kf.camera.position;
                self.hold_left_rows_at_keyframe(&current, pos).await;
                let near = self.area.as_ref().map(|a| a.registry.parked_within_radius(pos)).unwrap_or_default();
                let current = self.restore_tiles(current, &near, RestoreReason::Distance, pos).await;
                let (current, new) = self.seed_from_mask(current, &kf, size).await;
                // Set before appending so the new splats' Mip floor sees this view and the tiles just restored.
                self.refresh_floor_cams();
                let trainer = self
                    .trainer
                    .as_mut()
                    .expect("trainer exists once splats exist");
                match new {
                    Some(new) => trainer.append_splats_from(current, new, slot),
                    None => current,
                }
            }
        };
        self.hold_inactive_if_stale(&splats).await;
        self.splats = Some(splats);
        if let Some(t) = self.trainer.as_mut() {
            t.set_protect_cone(Some(view_cone(&kf.view.camera)));
        }

        self.views.push(kf.view);
        self.last_id = Some(kf.id);
        self.refresh_area_views();
        self.refresh_floor_cams();
        self.refresh_junction_views();
        self.rebuild_loaders();
        self.end_deferral_streaks();
        true
    }

    /// One optimisation step. While a junction refresh lasts, every other step draws one of its views; otherwise
    /// the batch comes from the recent window (with probability `recent_fraction`) or else from all training views
    /// (see [`choose_draw`]). No-op without views.
    pub async fn train_step(&mut self) {
        if self.splats.is_none() || self.trainer.is_none() {
            return;
        }
        if let Some(splats) = self.splats.clone() {
            self.hold_inactive_if_stale(&splats).await;
        }
        let pending = self.pending_views();
        let junction = self.area.as_mut().and_then(|a| a.junction.pick());
        let t_batch = web_time::Instant::now();
        let batch = if let Some(i) = junction {
            self.junction_batch(i).await
        } else {
            let draw = choose_draw(
                &mut self.rng,
                self.config.recent_fraction,
                self.recent_members.len(),
                &pending,
            );
            match draw {
                Draw::Pending(i) => load_batch(&self.views[i]).await,
                Draw::RecentLoader | Draw::AllLoader => {
                    let loader = if draw == Draw::RecentLoader {
                        self.recent.as_mut()
                    } else {
                        self.all.as_mut()
                    };
                    loader
                        .expect("loaders exist once views exist")
                        .next_batch()
                        .await
                }
            }
        };
        if self.profiling {
            self.batch_s += t_batch.elapsed().as_secs_f64();
        }
        let (Some(splats), Some(trainer)) = (self.splats.take(), self.trainer.as_mut()) else {
            return;
        };
        let position_scale = trainer.bounds().median_size();
        let (stepped, stats) = trainer.step(batch, splats.train()).await;
        self.last_step_lr = StepLr {
            lr_mean: stats.lr_mean,
            position_scale_m: position_scale,
        };
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
            splats = self.refine(splats).await;
        }
        self.splats = Some(splats);
    }

    /// Refines the splats (densify, prune, evict). With area training it first notes which tiles the window's views
    /// drew and hands the trainer the tiles' row budget, and parks the tiles an eviction or the held cap would
    /// otherwise make room for; afterwards it parks the tiles the capture has left and holds the rows `area_freeze`
    /// names (outside the active tiles, or of left tiles still resident).
    async fn refine(&mut self, splats: Splats) -> Splats {
        self.sync_max_splats();
        crate::timing::sync_splats(&splats).await;
        self.note_visibility(&splats).await;
        let splats = self.park_for_room(splats).await;
        self.refresh_floor_cams();
        let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
        let t = web_time::Instant::now();
        let before = splats.num_splats();
        let (splats, stats) = trainer.refine(self.iter, splats).await;
        crate::timing::sync_splats(&splats).await;
        let dt = t.elapsed().as_secs_f64();
        self.refine_stats.0 += 1;
        self.refine_stats.1 += dt;
        self.refine_total_s += dt;
        log::debug!(
            target: crate::timing::TARGET,
            "refine at iter {}: {:.0} ms, splats {before} -> {}, {} evicted",
            self.iter,
            dt * 1e3,
            splats.num_splats(),
            stats.num_evicted
        );
        self.num_evicted += u64::from(stats.num_evicted);
        self.last_refine_evicted = stats.evicted_for_total;
        self.note_budget_evictions();
        let splats = self.park_after_refine(splats).await;
        self.hold_rows_after_refine(&splats).await;
        splats
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
