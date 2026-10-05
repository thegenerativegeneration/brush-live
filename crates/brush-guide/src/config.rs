use crate::scores::{
    metrics::{CoverageParams, FisherRidge},
    pass::PassConfig,
    voxel::UncertaintyScale,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GuideConfig {
    pub voxel_size: f32,
    pub min_cell_opacity: f32,
    pub max_splats: u32,
    /// Evict the least important splats when growth hits `max_splats`
    /// (see `brush_train::evict`). Off: growth stops at the cap. On by
    /// default, since a growing scene reaches any fixed budget eventually.
    pub evict: bool,
    /// Which per-splat score eviction ranks by (see `EvictionImportance`).
    pub eviction_importance: EvictionImportance,
    /// Fraction of `max_splats` one eviction frees.
    pub evict_headroom: f32,
    /// Refines a splat must survive before it can be evicted.
    pub evict_min_age: u32,
    /// Most of a 1 m cell's splats one eviction may take.
    pub evict_max_cell_fraction: f32,
    /// Refines a cell seeded by a keyframe counts as newly observed for;
    /// eviction only makes room there, so it stops this many refines after
    /// the last keyframe that seeded anything.
    pub evict_recent_refines: u32,
    pub recent_window: usize,
    pub recent_fraction: f32,
    pub refine_every: u32,
    pub all_loader_rebuild_every: usize,
    /// Decoded-frame cache per scene loader, in bytes.
    pub loader_cache_bytes: u64,
    /// Training iterations per second at most (0: uncapped). The worker sleeps
    /// between steps, which saves power on the phone.
    pub max_iters_per_s: f32,
    /// Long side of the keyframes the client sends, in pixels. Sizes the
    /// warm-up so GPU autotuning covers the session's image size.
    pub keyframe_long_side: u32,
    /// Share of wall time for voxel rounds (score set from the splat
    /// parameters).
    pub score_budget: f32,
    /// Minimum start-to-start interval of voxel rounds.
    pub min_score_interval_s: f32,
    /// Share of wall time for Fisher passes (coverage and uncertainty).
    pub fisher_budget: f32,
    /// Minimum start-to-start interval of Fisher passes.
    pub min_fisher_interval_s: f32,
    /// Views per Fisher pass at most: the newest third plus one view per
    /// stratum of the rest, rotating (`schedule::ViewSample`). Fewer views
    /// (larger strata) let the stripe share drift up as the voxel set
    /// grows. The scheduler shortens a pass to what fits between voxel
    /// rounds.
    pub max_fisher_views: usize,
    /// Fewest views a Fisher pass is shortened to so it ends before the next
    /// voxel round is due.
    pub min_fisher_views: usize,
    /// Run a render, backward and depth render on dummy splats at server
    /// start so GPU autotuning happens before the first session.
    pub warmup: bool,
    pub sh_degree: u32,
    pub seed_stride_px: u32,
    pub seed_alpha_threshold: f32,
    pub init_random_count: usize,
    /// Absolute ridge on a voxel's summed position Fisher before inversion.
    pub fisher_lambda: f32,
    /// Ridge relative to that Fisher's mean eigenvalue (`tr(H)/3`).
    pub fisher_lambda_rel: f32,
    /// Pixel noise on [0, 1] RGB that scales a voxel's inverse position
    /// Fisher to a positional σ in metres. Only the raw dump sees the
    /// scale; the uncertainty byte ranks σ within a round and ignores it.
    pub sigma_pix: f32,
    /// Appends each round's per-voxel positional σ and coverage to
    /// `raw_uncertainty.jsonl` in the session directory, for calibration.
    pub dump_raw_uncertainty: bool,
    /// At `finish`, run one Fisher pass over every view (weight 1) and one
    /// voxel round, so the final score set carries coverage and uncertainty
    /// from all views, and write `importance.json` (`{"fisher": [..],
    /// "train": [..]}`, splat order, non-finite as `null`) to the session
    /// directory. For offline gates; the phone leaves it off.
    pub finish_fisher: bool,
    pub pass: PassConfig,
    pub coverage: CoverageParams,
    pub seed: u64,
    /// ARKit confidence below which depth is ignored for seeding (0 low, 1 medium, 2 high).
    pub min_depth_confidence: u8,
    /// Plumbed into `TrainConfig::sh_background` — see its docs. Off by
    /// default; flag-off training is unchanged.
    pub sh_background: bool,
    /// Plumbed into `TrainConfig::sh_background_alpha_weight`.
    pub sh_background_alpha_weight: f32,
    /// Stop training after this many steps; keyframes are still ingested and
    /// `finish` still exports. For fixed-step measurement runs; `None` trains
    /// until the session ends.
    pub max_train_steps: Option<u32>,
    /// Hold every this many new keyframes out of training and seeding and
    /// score the splats on them (PSNR, SSIM in the log, black background), for
    /// replays. 0 off; 2 or more (1 would hold out every keyframe and counts as 0).
    pub holdout_every: u32,
    /// Seconds between held-out scorings; there is one more at finish.
    pub eval_interval_s: f32,
    /// Time training-step phases (GPU synced at each boundary; slower steps)
    /// and log them every 100 steps on `brush_guide::timing` at debug.
    pub profile_steps: bool,
    /// Plumbed into `TrainConfig::ssim_every`: compute the SSIM loss term
    /// every this many training steps (1 every step, 0 never).
    pub ssim_every: u32,
    /// Seed from the phone's mono-depth block where LiDAR and feature points
    /// leave a pixel empty (`mono`). Frames without the block are unaffected.
    pub mono_seeding: bool,
    /// With a LiDAR depth map, mono depth seeds only at this depth or beyond.
    pub mono_min_depth_with_lidar_m: f32,
    /// Accepted per-keyframe mono scale, inclusive; a fit outside is rejected
    /// and the keyframe seeds without mono depth.
    pub mono_scale_range: (f32, f32),
}

impl Default for GuideConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.10,
            min_cell_opacity: 0.1,
            max_splats: 750_000,
            evict: true,
            eviction_importance: EvictionImportance::Train,
            evict_headroom: 0.1,
            evict_min_age: 3,
            evict_max_cell_fraction: 0.3,
            evict_recent_refines: 10,
            recent_window: 20,
            recent_fraction: 0.7,
            refine_every: 100,
            all_loader_rebuild_every: 10,
            loader_cache_bytes: 1 << 30,
            max_iters_per_s: 0.0,
            keyframe_long_side: 960,
            score_budget: 0.25,
            min_score_interval_s: 2.0,
            fisher_budget: 0.2,
            min_fisher_interval_s: 3.0,
            max_fisher_views: 60,
            min_fisher_views: 12,
            warmup: true,
            sh_degree: 1,
            seed_stride_px: 8,
            seed_alpha_threshold: 0.5,
            init_random_count: 5000,
            fisher_lambda: 1e-6,
            fisher_lambda_rel: 1e-3,
            sigma_pix: 0.05,
            dump_raw_uncertainty: false,
            finish_fisher: false,
            pass: PassConfig::default(),
            coverage: CoverageParams::default(),
            seed: 42,
            min_depth_confidence: 2,
            sh_background: false,
            sh_background_alpha_weight: 0.0,
            max_train_steps: None,
            holdout_every: 0,
            eval_interval_s: 30.0,
            profile_steps: false,
            ssim_every: 1,
            mono_seeding: true,
            mono_min_depth_with_lidar_m: 4.5,
            mono_scale_range: (0.3, 3.0),
        }
    }
}

/// Source of the per-splat score eviction ranks by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvictionImportance {
    /// Accumulated by the trainer's backward pass over training steps.
    Train,
    /// From the live Fisher pass (`scores::importance`).
    Fisher,
}

impl GuideConfig {
    pub fn fisher_ridge(&self) -> FisherRidge {
        FisherRidge {
            abs: self.fisher_lambda,
            rel: self.fisher_lambda_rel,
        }
    }

    pub fn uncertainty_scale(&self) -> UncertaintyScale {
        UncertaintyScale {
            ridge: self.fisher_ridge(),
            sigma_pix: self.sigma_pix,
        }
    }
}

#[cfg(test)]
mod ssim_every_tests {
    use super::GuideConfig;

    #[test]
    fn missing_field_defaults_to_every_step() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.ssim_every, 1);
    }

    #[test]
    fn field_parses() {
        let cfg: GuideConfig = serde_json::from_str(r#"{"ssim_every": 4}"#).expect("parses");
        assert_eq!(cfg.ssim_every, 4);
    }
}

#[cfg(test)]
mod eviction_importance_tests {
    use super::{EvictionImportance, GuideConfig};

    #[test]
    fn defaults_to_train_and_parses_fisher() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.eviction_importance, EvictionImportance::Train);
        assert!(!cfg.finish_fisher);
        let cfg: GuideConfig =
            serde_json::from_str(r#"{"eviction_importance": "fisher", "finish_fisher": true}"#)
                .expect("parses");
        assert_eq!(cfg.eviction_importance, EvictionImportance::Fisher);
        assert!(cfg.finish_fisher);
    }
}

#[cfg(test)]
mod mono_tests {
    use super::GuideConfig;

    #[test]
    fn mono_defaults_and_overrides() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert!(cfg.mono_seeding);
        assert_eq!(cfg.mono_min_depth_with_lidar_m, 4.5);
        assert_eq!(cfg.mono_scale_range, (0.3, 3.0));
        let cfg: GuideConfig =
            serde_json::from_str(r#"{"mono_seeding": false, "mono_scale_range": [0.5, 2.0]}"#)
                .expect("parses");
        assert!(!cfg.mono_seeding);
        assert_eq!(cfg.mono_scale_range, (0.5, 2.0));
    }
}
