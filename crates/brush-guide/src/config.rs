use crate::scores::{
    metrics::{CoverageParams, FisherRidge},
    pass::PassConfig,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GuideConfig {
    pub voxel_size: f32,
    pub min_cell_opacity: f32,
    pub max_splats: u32,
    pub recent_window: usize,
    pub recent_fraction: f32,
    pub refine_every: u32,
    pub all_loader_rebuild_every: usize,
    /// Decoded-frame cache per scene loader, in bytes.
    pub loader_cache_bytes: u64,
    pub score_budget: f32,
    pub min_score_interval_s: f32,
    /// Views per scoring round: the newest 40 plus a stratified sample of the rest.
    pub max_score_views: usize,
    pub sh_degree: u32,
    pub seed_stride_px: u32,
    pub seed_alpha_threshold: f32,
    pub init_random_count: usize,
    /// Absolute ridge on the Fisher before its log-determinant.
    pub fisher_lambda: f32,
    /// Ridge relative to the Fisher's mean eigenvalue (`tr(H)/6`).
    pub fisher_lambda_rel: f32,
    pub pass: PassConfig,
    pub coverage: CoverageParams,
    pub seed: u64,
}

impl Default for GuideConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.10,
            min_cell_opacity: 0.1,
            max_splats: 1_500_000,
            recent_window: 20,
            recent_fraction: 0.7,
            refine_every: 100,
            all_loader_rebuild_every: 10,
            loader_cache_bytes: 1 << 30,
            score_budget: 0.25,
            min_score_interval_s: 2.0,
            max_score_views: 120,
            sh_degree: 1,
            seed_stride_px: 8,
            seed_alpha_threshold: 0.5,
            init_random_count: 5000,
            fisher_lambda: 1e-6,
            fisher_lambda_rel: 1e-3,
            pass: PassConfig::default(),
            coverage: CoverageParams::default(),
            seed: 42,
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
}
