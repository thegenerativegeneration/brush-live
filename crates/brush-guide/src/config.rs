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
    /// Share of wall time for voxel rounds (score set from the splat
    /// parameters, TSDF fusion, meshing).
    pub score_budget: f32,
    /// Minimum start-to-start interval of voxel rounds.
    pub min_score_interval_s: f32,
    /// Share of wall time for Fisher passes (coverage and uncertainty).
    pub fisher_budget: f32,
    /// Minimum start-to-start interval of Fisher passes.
    pub min_fisher_interval_s: f32,
    /// Views per Fisher pass at most: the newest third plus one view per
    /// stratum of the rest, rotating (`schedule::ViewSample`).
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
    pub pass: PassConfig,
    pub coverage: CoverageParams,
    pub seed: u64,
    /// ARKit confidence below which depth is ignored for seeding (0 low, 1 medium, 2 high).
    pub min_depth_confidence: u8,
}

impl Default for GuideConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.10,
            min_cell_opacity: 0.1,
            max_splats: 750_000,
            evict: true,
            evict_headroom: 0.1,
            evict_min_age: 3,
            evict_max_cell_fraction: 0.3,
            evict_recent_refines: 10,
            recent_window: 20,
            recent_fraction: 0.7,
            refine_every: 100,
            all_loader_rebuild_every: 10,
            loader_cache_bytes: 1 << 30,
            score_budget: 0.25,
            min_score_interval_s: 2.0,
            fisher_budget: 0.1,
            min_fisher_interval_s: 3.0,
            max_fisher_views: 30,
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
            pass: PassConfig::default(),
            coverage: CoverageParams::default(),
            seed: 42,
            min_depth_confidence: 2,
        }
    }
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
