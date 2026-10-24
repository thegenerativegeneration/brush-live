use serde_with::with_prefix;

use crate::scores::{
    metrics::{CoverageParams, FisherRidge},
    pass::PassConfig,
    voxel::UncertaintyScale,
};

/// Which resident rows area training holds still (no step, decay, floor bake, growth or eviction) until they park.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AreaFreeze {
    /// While the resident rows stay below `area_freeze_at` of `max_splats` every view and every row trains as with
    /// area training off (the active set is still ranked and reported); from there the views that see an active tile
    /// train and the rows outside the active tiles are held, as with `inactive`, until the resident rows drop below
    /// `area_freeze_at` − 0.05.
    #[default]
    InactiveAtCap,
    /// Only the active tiles train: the tiles within 2.5 m (at most half a tile) of the newest keyframe's camera and
    /// up to `area_active_tiles` tiles it sees within `area_view_m`. The training views are the keyframes that see an
    /// active tile, keyframes seed only into active tiles and ground no tile holds yet, and every other resident row
    /// is held.
    Inactive,
    /// The tiles the walker has left are held while resident; the views within `area_radius_m` train.
    Left,
    /// Nothing is held; the views within `area_radius_m` train.
    #[serde(rename = "none")]
    Off,
}

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
    /// Eviction: the `evict_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_eviction")]
    pub eviction: EvictConfig,
    pub recent_window: usize,
    pub recent_fraction: f32,
    pub refine_every: u32,
    pub all_loader_rebuild_every: usize,
    /// Decoded-frame cache per scene loader, in bytes.
    pub loader_cache_bytes: u64,
    /// Training iterations per second at most (0: uncapped). The worker sleeps
    /// between steps, which saves power on the phone.
    pub max_iters_per_s: f32,
    /// Send a status frame after every training step, not only once a second, so a client sees `train_iters`
    /// within a step (lockstep replays send the next keyframe on it). Off on the phone.
    pub status_every_step: bool,
    /// Long side of the keyframes the client sends, in pixels. Sizes the
    /// warm-up so GPU autotuning covers the session's image size.
    pub keyframe_long_side: u32,
    /// Share of wall time for voxel rounds (score set from the splat
    /// parameters).
    pub score_budget: f32,
    /// Minimum start-to-start interval of voxel rounds.
    pub min_score_interval_s: f32,
    /// Run a render, backward and depth render on dummy splats at server
    /// start so GPU autotuning happens before the first session.
    pub warmup: bool,
    pub sh_degree: u32,
    /// Keyframe seeding: the `seed_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_seeding")]
    pub seeding: SeedConfig,
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
    /// Held-out views: the `holdout_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_holdout")]
    pub holdout: HoldoutConfig,
    /// Held-out evals: the `eval_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_eval")]
    pub eval: EvalConfig,
    /// Time training-step phases (GPU synced at each boundary; slower steps)
    /// and log them every 100 steps on `brush_guide::timing` at debug.
    pub profile_steps: bool,
    /// Plumbed into `TrainConfig::ssim_every`: compute the SSIM loss term
    /// every this many training steps (1 every step, 0 never).
    pub ssim_every: u32,
    /// Mono-depth seeding: the `mono_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_mono")]
    pub mono: MonoConfig,
    /// On a pose update, move each splat with the view whose seeds created it (`brush_train`'s origin tags). Off:
    /// only the views' cameras change and training alone has to move the drifted splats.
    pub move_anchored_splats: bool,
    /// Area training: the `area_*` keys in the JSON, flattened.
    #[serde(flatten, with = "prefix_area")]
    pub area: AreaConfig,
    /// Plumbed into `TrainConfig::train_radius_m`: splats farther than this from the training camera get no update
    /// in that step. 0: off.
    pub train_radius_m: f32,
}

with_prefix!(prefix_eviction "evict_");

/// The `evict_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EvictConfig {
    /// Fraction of `max_splats` one eviction frees.
    pub headroom: f32,
    /// Refines a splat must survive before it can be evicted.
    pub min_age: u32,
    /// Most of a 1 m cell's splats one eviction may take.
    pub max_cell_fraction: f32,
    /// Refines a cell seeded by a keyframe counts as newly observed for;
    /// eviction only makes room there, so it stops this many refines after
    /// the last keyframe that seeded anything.
    pub recent_refines: u32,
}

impl Default for EvictConfig {
    fn default() -> Self {
        Self {
            headroom: 0.1,
            min_age: 3,
            max_cell_fraction: 0.3,
            recent_refines: 10,
        }
    }
}

with_prefix!(prefix_seeding "seed_");

/// The `seed_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SeedConfig {
    /// Smallest spacing, in pixels, of the grid a keyframe seeds new splats
    /// on. The spacing itself follows from `seed_view_fraction`.
    pub stride_px: u32,
    /// A fully uncovered view seeds at most about this fraction of
    /// `max_splats`; the seed grid's spacing follows from it.
    pub view_fraction: f32,
    pub alpha_threshold: f32,
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            stride_px: 4,
            view_fraction: 0.03,
            alpha_threshold: 0.5,
        }
    }
}

with_prefix!(prefix_holdout "holdout_");

/// The `holdout_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HoldoutConfig {
    /// Hold every this many new keyframes out of training and seeding and
    /// score the splats on them (PSNR, SSIM in the log; renders over the SH background when on, else black), for
    /// replays. 0 off; 2 or more (1 would hold out every keyframe and counts as 0).
    pub every: u32,
    /// Keyframes received after this many are never held out (0: no limit), e.g. a replay's walk back.
    pub max_received: u32,
}

impl Default for HoldoutConfig {
    fn default() -> Self {
        Self {
            every: 0,
            max_received: 0,
        }
    }
}

with_prefix!(prefix_eval "eval_");

/// The `eval_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EvalConfig {
    /// Seconds between held-out scorings; there is one more at finish.
    pub interval_s: f32,
    /// Held-out evals every this many received keyframes (0: every `eval_interval_s` seconds).
    pub every_keyframes: u32,
    /// Received-keyframe counts at which the first eval at or after carries the extras (LPIPS, seam split,
    /// saved renders); finish always does.
    pub extras_at: Vec<u32>,
    /// Score LPIPS in evals with extras.
    pub lpips: bool,
    /// Width in metres of the seam band at tile borders for the split PSNR (0: no split).
    pub seam_band_m: f32,
    /// Keyframe ids of held-out views whose renders are saved to `eval_renders/` in evals with extras.
    pub save_views: Vec<u64>,
    /// Held-out keyframe ids scored with LPIPS in evals with extras (empty: every held-out view); PSNR and SSIM
    /// always cover every held-out view.
    pub extras_views: Vec<u64>,
    /// Evals with extras at or after this many received keyframes carry the seam split, and so does the finish
    /// eval (0: every eval with extras).
    pub seam_from: u32,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            interval_s: 30.0,
            every_keyframes: 0,
            extras_at: Vec::new(),
            lpips: false,
            seam_band_m: 0.0,
            save_views: Vec::new(),
            extras_views: Vec::new(),
            seam_from: 0,
        }
    }
}

with_prefix!(prefix_mono "mono_");

/// The `mono_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MonoConfig {
    /// Seed from the phone's mono-depth block where LiDAR and feature points
    /// leave a pixel empty (`mono`). Frames without the block are unaffected.
    pub seeding: bool,
    /// With a LiDAR depth map, mono depth seeds only at this depth or beyond.
    pub min_depth_with_lidar_m: f32,
    /// Accepted per-keyframe mono scale, inclusive; a fit outside is rejected
    /// and the keyframe seeds without mono depth.
    pub scale_range: (f32, f32),
}

impl Default for MonoConfig {
    fn default() -> Self {
        Self {
            seeding: true,
            min_depth_with_lidar_m: 4.5,
            scale_range: (0.3, 3.0),
        }
    }
}

with_prefix!(prefix_area "area_");

/// The `area_*` keys of [`GuideConfig`], without the prefix.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AreaConfig {
    /// Edge of the area-training tiles in x and z, metres; also the grid of the eval's seam band.
    pub tile_size_m: f32,
    /// Area training: tiles whose footprint lies within this many metres (horizontal) of the newest keyframe train.
    /// 0 or less: off.
    pub radius_m: f32,
    /// A resident tile is parked only beyond `area_radius_m` plus this margin, so tiles at the edge do not flap.
    pub hysteresis_m: f32,
    /// A resident tile counts as recently seen for this many training steps after a view last drew it.
    pub seen_window_steps: u32,
    /// A restored tile stays resident for at least this many keyframes before it may be parked again.
    pub min_resident_keyframes: u32,
    /// A parked tile is restored when at least this many fresh seeds fall into it.
    pub restore_min_seeds: u32,
    /// Training steps that blend a restored tile in at the junction with the live area.
    pub junction_steps: u32,
    /// Largest disagreement, metres, between a restored tile and the fresh geometry that still restores it as is.
    pub restore_inconsistency_m: f32,
    /// Seeds in a parked tile are restored only when the tile lies within `area_radius_m` plus this many metres. Only
    /// tiles the walker never visited (parked for room) can use the band beyond `area_hysteresis_m`: seeds in a
    /// visited tile farther than `area_radius_m` + `area_hysteresis_m` (a left tile) are dropped, so a visited tile
    /// comes back only within that distance.
    pub far_seed_m: f32,
    /// The resident rows in tiles the walker has left (held or training, whatever `area_freeze` does with them) keep
    /// at most this share of `max_splats`; at each refine of an engaged engine the farthest left tiles park, even in
    /// view, until the rest fit. 1.0 or more: no cap.
    pub held_fraction: f32,
    /// Which resident rows area training holds still until they park (see [`AreaFreeze`]).
    pub freeze: AreaFreeze,
    /// With `area_freeze` inactive(_at_cap): the newest keyframe makes the tiles it sees up to this many metres deep
    /// active.
    pub view_m: f32,
    /// With `area_freeze` inactive(_at_cap): at most this many tiles in view are active, besides the near tiles (within
    /// 2.5 m of the camera, at most half a tile).
    pub active_tiles: u32,
    /// With `area_freeze` inactive(_at_cap): a tile in view needs this many rows to become active.
    pub active_min_rows: u32,
    /// With `area_freeze` inactive_at_cap: the share of `max_splats` at which the rows outside the active tiles are
    /// held (released again below this share − 0.05).
    pub freeze_at: f32,
    /// Per-tile cap as a share of `max_splats`; 0 (default): no per-tile cap, tiles share the total budget. Above 0
    /// (0.4: 10k of 25k), once area training has engaged: seeds into a full tile are dropped, its rows do not
    /// densify, and a tile over it evicts its least important rows down to it at the next refine (a tile still
    /// filling up can overshoot by one refine's growth). Held rows no step ever trained stay eviction victims at the
    /// total cap either way.
    pub tile_max_fraction: f32,
    /// Area training engages once the walker has left a tile (a visited tile lies beyond `area_radius_m` plus
    /// `area_hysteresis_m`) or a tile parked, and stays on until the session resets. Until then the engine trains as
    /// with area training off: every view trains, nothing is held or parked, and no per-tile cap, row budget or
    /// eviction quota applies (the active set is still ranked and reported). false: engaged from the first keyframe.
    pub engage_on_leave: bool,
}

impl Default for AreaConfig {
    fn default() -> Self {
        Self {
            tile_size_m: 10.0,
            radius_m: 50.0,
            hysteresis_m: 5.0,
            seen_window_steps: 1000,
            min_resident_keyframes: 20,
            restore_min_seeds: 8,
            junction_steps: 200,
            restore_inconsistency_m: 1.0,
            far_seed_m: 10.0,
            held_fraction: 0.25,
            freeze: AreaFreeze::InactiveAtCap,
            view_m: 15.0,
            active_tiles: 3,
            active_min_rows: 50,
            freeze_at: 0.9,
            tile_max_fraction: 0.0,
            engage_on_leave: true,
        }
    }
}

impl Default for GuideConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.10,
            min_cell_opacity: 0.1,
            max_splats: 750_000,
            evict: true,
            eviction: EvictConfig::default(),
            recent_window: 20,
            recent_fraction: 0.7,
            refine_every: 100,
            all_loader_rebuild_every: 10,
            loader_cache_bytes: 1 << 30,
            max_iters_per_s: 0.0,
            status_every_step: false,
            keyframe_long_side: 960,
            score_budget: 0.25,
            min_score_interval_s: 2.0,
            warmup: true,
            sh_degree: 1,
            seeding: SeedConfig::default(),
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
            holdout: HoldoutConfig::default(),
            eval: EvalConfig::default(),
            profile_steps: false,
            ssim_every: 1,
            mono: MonoConfig::default(),
            move_anchored_splats: true,
            area: AreaConfig::default(),
            train_radius_m: 0.0,
        }
    }
}

impl GuideConfig {
    /// Rejects values that would stall or poison a session: a non-finite or non-positive `voxel_size`,
    /// `min_cell_opacity` or `score_budget`. Reports the first bad field by its JSON key.
    pub fn validate(&self) -> Result<(), String> {
        for (key, value) in [
            ("voxel_size", self.voxel_size),
            ("min_cell_opacity", self.min_cell_opacity),
            ("score_budget", self.score_budget),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{key} must be finite and above 0, got {value}"));
            }
        }
        Ok(())
    }

    /// Most splats one tile holds: `area_tile_max_fraction` of `max_splats`, rounded; 0 for no per-tile cap.
    pub fn area_tile_cap(&self) -> u32 {
        if self.area.tile_max_fraction > 0.0 {
            (f64::from(self.area.tile_max_fraction) * f64::from(self.max_splats)).round() as u32
        } else {
            0
        }
    }

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
#[path = "config_tests.rs"]
mod tests;
