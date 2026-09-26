use crate::config::GuideConfig;
use crate::keyframe::DecodedKeyframe;
use crate::seed::{SeedInput, Seeds, seed_points};
use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::scene::{Scene, SceneView};
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, TextureMode, render_splats};
use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_train::{RandomSplatsConfig, create_random_splats, to_init_splats};
use burn::module::Module;
use burn::tensor::{Device, s};
use clap::Parser;
use glam::{UVec2, Vec3};
use rand::{RngExt as _, SeedableRng};
use std::collections::HashSet;

/// `LoadDatasetConfig` is a clap `Args` group; this wrapper parses its CLI defaults.
#[derive(Parser)]
struct LoadArgs {
    #[command(flatten)]
    load: LoadDatasetConfig,
}

/// Recent-window loader is rebuilt every this many new keyframes (and for each
/// of the first ones): every rebuild spawns a fresh set of loader threads.
const RECENT_LOADER_REBUILD_EVERY: usize = 3;

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
    views_in_recent: usize,
    views_in_all: usize,
    iter: u32,
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
            views_in_recent: 0,
            views_in_all: 0,
            iter: 0,
            rng,
        }
    }

    pub fn splats(&self) -> Option<&Splats> {
        self.splats.as_ref()
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
    fn cap_seeds(&mut self, seeds: Seeds, current: u32) -> Seeds {
        let room = self.config.max_splats.saturating_sub(current) as usize;
        if seeds.colors.len() <= room {
            return seeds;
        }
        let mut keep = rand::seq::index::sample(&mut self.rng, seeds.colors.len(), room).into_vec();
        keep.sort_unstable();
        Seeds {
            means: keep
                .iter()
                .flat_map(|&i| seeds.means[i * 3..i * 3 + 3].iter().copied())
                .collect(),
            colors: keep.iter().map(|&i| seeds.colors[i]).collect(),
        }
    }

    fn initial_splats(&mut self, kf: &DecodedKeyframe, size: UVec2) -> Splats {
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &vec![0.0; (size.x * size.y) as usize],
            alpha_size: size,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: self.config.seed_stride_px,
            alpha_threshold: self.config.seed_alpha_threshold,
        });
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
        let seeds = seed_points(&SeedInput {
            camera: &kf.camera,
            alpha: &alpha,
            alpha_size: small,
            rgb: &kf.image,
            depth: kf.depth.as_ref(),
            points: &kf.points,
            stride: (self.config.seed_stride_px / 4).max(1),
            alpha_threshold: self.config.seed_alpha_threshold,
        });
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
        let loader = if use_recent {
            self.recent.as_mut()
        } else {
            self.all.as_mut()
        }
        .expect("loaders exist once views exist");
        let batch = loader.next_batch().await;
        let (stepped, _) = trainer.step(batch, splats.train()).await;
        let mut splats = stepped.valid();
        self.iter += 1;
        if self.iter.is_multiple_of(self.config.refine_every) {
            splats = trainer.refine(self.iter, splats).await.0;
        }
        self.splats = Some(splats);
    }
}
