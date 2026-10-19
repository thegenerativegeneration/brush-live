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
    /// Smallest spacing, in pixels, of the grid a keyframe seeds new splats
    /// on. The spacing itself follows from `seed_view_fraction`.
    pub seed_stride_px: u32,
    /// A fully uncovered view seeds at most about this fraction of
    /// `max_splats`; the seed grid's spacing follows from it.
    pub seed_view_fraction: f32,
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
    /// score the splats on them (PSNR, SSIM in the log; renders over the SH background when on, else black), for
    /// replays. 0 off; 2 or more (1 would hold out every keyframe and counts as 0).
    pub holdout_every: u32,
    /// Keyframes received after this many are never held out (0: no limit), e.g. a replay's walk back.
    pub holdout_max_received: u32,
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
    /// On a pose update, move each splat with the view whose seeds created it (`brush_train`'s origin tags). Off:
    /// only the views' cameras change and training alone has to move the drifted splats.
    pub move_anchored_splats: bool,
    /// Held-out evals every this many received keyframes (0: every `eval_interval_s` seconds).
    pub eval_every_keyframes: u32,
    /// Received-keyframe counts at which the first eval at or after carries the extras (LPIPS, seam split,
    /// saved renders); finish always does.
    pub eval_extras_at: Vec<u32>,
    /// Score LPIPS in evals with extras.
    pub eval_lpips: bool,
    /// Width in metres of the seam band at tile borders for the split PSNR (0: no split).
    pub eval_seam_band_m: f32,
    /// Keyframe ids of held-out views whose renders are saved to `eval_renders/` in evals with extras.
    pub eval_save_views: Vec<u64>,
    /// Held-out keyframe ids scored with LPIPS in evals with extras (empty: every held-out view); PSNR and SSIM
    /// always cover every held-out view.
    pub eval_extras_views: Vec<u64>,
    /// Evals with extras at or after this many received keyframes carry the seam split, and so does the finish
    /// eval (0: every eval with extras).
    pub eval_seam_from: u32,
    /// Edge of the area-training tiles in x and z, metres; also the grid of the eval's seam band.
    pub area_tile_size_m: f32,
    /// Area training: tiles whose footprint lies within this many metres (horizontal) of the newest keyframe train.
    /// 0 or less: off.
    pub area_radius_m: f32,
    /// A resident tile is parked only beyond `area_radius_m` plus this margin, so tiles at the edge do not flap.
    pub area_hysteresis_m: f32,
    /// A resident tile counts as recently seen for this many training steps after a view last drew it.
    pub area_seen_window_steps: u32,
    /// A restored tile stays resident for at least this many keyframes before it may be parked again.
    pub area_min_resident_keyframes: u32,
    /// A parked tile is restored when at least this many fresh seeds fall into it.
    pub area_restore_min_seeds: u32,
    /// Training steps that blend a restored tile in at the junction with the live area.
    pub area_junction_steps: u32,
    /// Largest disagreement, metres, between a restored tile and the fresh geometry that still restores it as is.
    pub area_restore_inconsistency_m: f32,
    /// Seeds in a parked tile are restored only when the tile lies within `area_radius_m` plus this many metres. Only
    /// tiles the walker never visited (parked for room) can use the band beyond `area_hysteresis_m`: seeds in a
    /// visited tile farther than `area_radius_m` + `area_hysteresis_m` (a left tile) are dropped, so a visited tile
    /// comes back only within that distance.
    pub area_far_seed_m: f32,
    /// The resident rows in tiles the walker has left (held or training, whatever `area_freeze` does with them) keep
    /// at most this share of `max_splats`; at each refine of an engaged engine the farthest left tiles park, even in
    /// view, until the rest fit. 1.0 or more: no cap.
    pub area_held_fraction: f32,
    /// Which resident rows area training holds still until they park (see [`AreaFreeze`]).
    pub area_freeze: AreaFreeze,
    /// With `area_freeze` inactive(_at_cap): the newest keyframe makes the tiles it sees up to this many metres deep
    /// active.
    pub area_view_m: f32,
    /// With `area_freeze` inactive(_at_cap): at most this many tiles in view are active, besides the near tiles (within
    /// 2.5 m of the camera, at most half a tile).
    pub area_active_tiles: u32,
    /// With `area_freeze` inactive(_at_cap): a tile in view needs this many rows to become active.
    pub area_active_min_rows: u32,
    /// With `area_freeze` inactive_at_cap: the share of `max_splats` at which the rows outside the active tiles are
    /// held (released again below this share − 0.05).
    pub area_freeze_at: f32,
    /// Per-tile cap as a share of `max_splats`; 0 (default): no per-tile cap, tiles share the total budget. Above 0
    /// (0.4: 10k of 25k), once area training has engaged: seeds into a full tile are dropped, its rows do not
    /// densify, and a tile over it evicts its least important rows down to it at the next refine (a tile still
    /// filling up can overshoot by one refine's growth). Held rows no step ever trained stay eviction victims at the
    /// total cap either way.
    pub area_tile_max_fraction: f32,
    /// Area training engages once the walker has left a tile (a visited tile lies beyond `area_radius_m` plus
    /// `area_hysteresis_m`) or a tile parked, and stays on until the session resets. Until then the engine trains as
    /// with area training off: every view trains, nothing is held or parked, and no per-tile cap, row budget or
    /// eviction quota applies (the active set is still ranked and reported). false: engaged from the first keyframe.
    pub area_engage_on_leave: bool,
    /// Plumbed into `TrainConfig::train_radius_m`: splats farther than this from the training camera get no update
    /// in that step. 0: off.
    pub train_radius_m: f32,
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
            max_iters_per_s: 0.0,
            status_every_step: false,
            keyframe_long_side: 960,
            score_budget: 0.25,
            min_score_interval_s: 2.0,
            warmup: true,
            sh_degree: 1,
            seed_stride_px: 4,
            seed_view_fraction: 0.03,
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
            holdout_max_received: 0,
            eval_interval_s: 30.0,
            profile_steps: false,
            ssim_every: 1,
            mono_seeding: true,
            mono_min_depth_with_lidar_m: 4.5,
            mono_scale_range: (0.3, 3.0),
            move_anchored_splats: true,
            eval_every_keyframes: 0,
            eval_extras_at: Vec::new(),
            eval_lpips: false,
            eval_seam_band_m: 0.0,
            eval_save_views: Vec::new(),
            eval_extras_views: Vec::new(),
            eval_seam_from: 0,
            area_tile_size_m: 10.0,
            area_radius_m: 50.0,
            area_hysteresis_m: 5.0,
            area_seen_window_steps: 1000,
            area_min_resident_keyframes: 20,
            area_restore_min_seeds: 8,
            area_junction_steps: 200,
            area_restore_inconsistency_m: 1.0,
            area_far_seed_m: 10.0,
            area_held_fraction: 0.25,
            area_freeze: AreaFreeze::InactiveAtCap,
            area_view_m: 15.0,
            area_active_tiles: 3,
            area_active_min_rows: 50,
            area_freeze_at: 0.9,
            area_tile_max_fraction: 0.0,
            area_engage_on_leave: true,
            train_radius_m: 0.0,
        }
    }
}

impl GuideConfig {
    /// Most splats one tile holds: `area_tile_max_fraction` of `max_splats`, rounded; 0 for no per-tile cap.
    pub fn area_tile_cap(&self) -> u32 {
        if self.area_tile_max_fraction > 0.0 {
            (f64::from(self.area_tile_max_fraction) * f64::from(self.max_splats)).round() as u32
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
mod seed_stride_tests {
    use super::GuideConfig;

    #[test]
    fn defaults_and_old_configs_parse() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.seed_stride_px, 4);
        assert_eq!(cfg.seed_view_fraction, 0.03);
        let cfg: GuideConfig = serde_json::from_str(r#"{"seed_stride_px": 8}"#).expect("parses");
        assert_eq!(cfg.seed_stride_px, 8);
        assert_eq!(cfg.seed_view_fraction, 0.03);
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
mod finish_fisher_tests {
    use super::GuideConfig;

    #[test]
    fn defaults_to_off_and_parses() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert!(!cfg.finish_fisher);
        let cfg: GuideConfig = serde_json::from_str(r#"{"finish_fisher": true}"#).expect("parses");
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

#[cfg(test)]
mod anchored_splat_tests {
    use super::GuideConfig;

    #[test]
    fn moving_anchored_splats_is_on_by_default_and_can_be_turned_off() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert!(cfg.move_anchored_splats);
        let cfg: GuideConfig =
            serde_json::from_str(r#"{"move_anchored_splats": false}"#).expect("parses");
        assert!(!cfg.move_anchored_splats);
    }
}

#[cfg(test)]
mod eval_extra_tests {
    use super::GuideConfig;

    #[test]
    fn eval_extras_default_off_and_parse() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!((cfg.eval_every_keyframes, cfg.eval_lpips, cfg.eval_seam_band_m), (0, false, 0.0));
        assert!(cfg.eval_extras_at.is_empty());
        assert!(cfg.eval_save_views.is_empty());
        assert!(cfg.eval_extras_views.is_empty());
        assert_eq!((cfg.eval_seam_from, cfg.holdout_max_received), (0, 0));
        assert_eq!(cfg.area_tile_size_m, 10.0);
        let cfg: GuideConfig = serde_json::from_str(
            r#"{"eval_every_keyframes": 50, "eval_extras_at": [117, 413], "eval_lpips": true, "eval_save_views": [19, 59]}"#,
        )
        .expect("parses");
        assert_eq!(cfg.eval_save_views, vec![19, 59]);
        let cfg: GuideConfig = serde_json::from_str(
            r#"{"eval_extras_views": [9, 109], "eval_seam_from": 674, "holdout_max_received": 674}"#,
        )
        .expect("parses");
        assert_eq!((cfg.eval_extras_views, cfg.eval_seam_from, cfg.holdout_max_received), (vec![9, 109], 674, 674));
    }
}

#[cfg(test)]
mod train_radius_tests {
    use super::GuideConfig;

    #[test]
    fn defaults_to_off_and_parses() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.train_radius_m, 0.0);
        let cfg: GuideConfig = serde_json::from_str(r#"{"train_radius_m": 25}"#).expect("parses");
        assert_eq!(cfg.train_radius_m, 25.0);
    }
}

#[cfg(test)]
mod area_tests {
    use super::{AreaFreeze, GuideConfig};

    #[test]
    fn area_defaults_and_overrides() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!((cfg.area_radius_m, cfg.area_hysteresis_m, cfg.area_seen_window_steps), (50.0, 5.0, 1000));
        assert_eq!((cfg.area_min_resident_keyframes, cfg.area_junction_steps, cfg.area_restore_min_seeds), (20, 200, 8));
        assert_eq!((cfg.area_restore_inconsistency_m, cfg.area_far_seed_m), (1.0, 10.0));
        let cfg: GuideConfig = serde_json::from_str(r#"{"area_radius_m": 25}"#).expect("parses");
        assert_eq!(cfg.area_radius_m, 25.0);
    }

    #[test]
    fn held_fraction_defaults_to_a_quarter_and_parses() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.area_held_fraction, 0.25);
        let cfg: GuideConfig = serde_json::from_str(r#"{"area_held_fraction": 1.0}"#).expect("parses");
        assert_eq!(cfg.area_held_fraction, 1.0);
    }

    #[test]
    fn freeze_defaults_to_inactive_at_cap_and_parses() {
        let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.area_freeze, AreaFreeze::InactiveAtCap);
        assert_eq!((cfg.area_view_m, cfg.area_active_tiles, cfg.area_active_min_rows), (15.0, 3, 50));
        assert_eq!(cfg.area_freeze_at, 0.9);
        assert!(cfg.area_engage_on_leave);
        let capped = GuideConfig { area_tile_max_fraction: 0.4, max_splats: 25_000, ..cfg.clone() };
        assert_eq!(capped.area_tile_cap(), 10_000);
        assert_eq!(GuideConfig { area_tile_max_fraction: 0.0, ..cfg }.area_tile_cap(), 0);
        let modes = [
            ("left", AreaFreeze::Left),
            ("none", AreaFreeze::Off),
            ("inactive", AreaFreeze::Inactive),
            ("inactive_at_cap", AreaFreeze::InactiveAtCap),
        ];
        for (json, want) in modes {
            let cfg: GuideConfig =
                serde_json::from_str(&format!(r#"{{"area_freeze": "{json}"}}"#)).expect("parses");
            assert_eq!(cfg.area_freeze, want);
        }
        let cfg: GuideConfig =
            serde_json::from_str(r#"{"area_tile_max_fraction": 0.25, "max_splats": 20000}"#).expect("parses");
        assert_eq!(cfg.area_tile_cap(), 5_000);
        let cfg: GuideConfig = serde_json::from_str(r#"{"area_view_m": 20, "area_active_tiles": 5}"#).expect("parses");
        assert_eq!((cfg.area_view_m, cfg.area_active_tiles), (20.0, 5));
        let cfg: GuideConfig =
            serde_json::from_str(r#"{"area_freeze_at": 0.8, "area_tile_max_fraction": 0}"#).expect("parses");
        assert_eq!((cfg.area_freeze_at, cfg.area_tile_max_fraction), (0.8, 0.0));
    }

    #[test]
    fn the_default_config_has_no_per_tile_cap() {
        let cfg = GuideConfig::default();
        assert_eq!(cfg.area_tile_max_fraction, 0.0);
        assert_eq!(cfg.area_tile_cap(), 0);
        let parsed: GuideConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(parsed.area_tile_cap(), 0);
    }
}
