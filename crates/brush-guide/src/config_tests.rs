use super::{AreaFreeze, GuideConfig};
use serde_json::Value;

/// Every top-level key of the serialized default config, sorted.
const KEYS: &[&str] = &[
    "all_loader_rebuild_every",
    "area_active_min_rows",
    "area_active_tiles",
    "area_engage_on_leave",
    "area_far_seed_m",
    "area_freeze",
    "area_freeze_at",
    "area_held_fraction",
    "area_hysteresis_m",
    "area_junction_steps",
    "area_min_resident_keyframes",
    "area_radius_m",
    "area_restore_inconsistency_m",
    "area_restore_min_seeds",
    "area_seen_window_steps",
    "area_tile_max_fraction",
    "area_tile_size_m",
    "area_view_m",
    "coverage",
    "dump_raw_uncertainty",
    "eval_every_keyframes",
    "eval_extras_at",
    "eval_extras_views",
    "eval_interval_s",
    "eval_lpips",
    "eval_save_views",
    "eval_seam_band_m",
    "eval_seam_from",
    "evict",
    "evict_headroom",
    "evict_max_cell_fraction",
    "evict_min_age",
    "evict_recent_refines",
    "finish_fisher",
    "fisher_lambda",
    "fisher_lambda_rel",
    "holdout_every",
    "holdout_max_received",
    "init_random_count",
    "keyframe_long_side",
    "loader_cache_bytes",
    "max_iters_per_s",
    "max_splats",
    "max_train_steps",
    "min_cell_opacity",
    "min_depth_confidence",
    "min_score_interval_s",
    "mono_min_depth_with_lidar_m",
    "mono_scale_range",
    "mono_seeding",
    "move_anchored_splats",
    "pass",
    "profile_steps",
    "recent_fraction",
    "recent_window",
    "refine_every",
    "score_budget",
    "seed",
    "seed_alpha_threshold",
    "seed_stride_px",
    "seed_view_fraction",
    "sh_background",
    "sh_background_alpha_weight",
    "sh_degree",
    "sigma_pix",
    "ssim_every",
    "status_every_step",
    "train_radius_m",
    "voxel_size",
    "warmup",
];

/// Every key set to a value other than its default. Values are exact in `f32`; `voxel_size` and `area_radius_m`
/// are integer literals on purpose.
const NON_DEFAULT_JSON: &str = r#"{
    "all_loader_rebuild_every": 4,
    "area_active_min_rows": 25,
    "area_active_tiles": 5,
    "area_engage_on_leave": false,
    "area_far_seed_m": 20.0,
    "area_freeze": "none",
    "area_freeze_at": 0.75,
    "area_held_fraction": 0.5,
    "area_hysteresis_m": 2.5,
    "area_junction_steps": 100,
    "area_min_resident_keyframes": 10,
    "area_radius_m": 25,
    "area_restore_inconsistency_m": 0.5,
    "area_restore_min_seeds": 4,
    "area_seen_window_steps": 500,
    "area_tile_max_fraction": 0.5,
    "area_tile_size_m": 20.0,
    "area_view_m": 20.0,
    "coverage": {"n_target": 16.0, "spread_target": 0.5, "min_px_per_m": 150.0, "low_res_penalty": 0.25},
    "dump_raw_uncertainty": true,
    "eval_every_keyframes": 50,
    "eval_extras_at": [117, 413],
    "eval_extras_views": [9, 109],
    "eval_interval_s": 15.5,
    "eval_lpips": true,
    "eval_save_views": [19, 59],
    "eval_seam_band_m": 1.5,
    "eval_seam_from": 674,
    "evict": false,
    "evict_headroom": 0.25,
    "evict_max_cell_fraction": 0.5,
    "evict_min_age": 5,
    "evict_recent_refines": 7,
    "finish_fisher": true,
    "fisher_lambda": 0.5,
    "fisher_lambda_rel": 0.25,
    "holdout_every": 2,
    "holdout_max_received": 10,
    "init_random_count": 100,
    "keyframe_long_side": 1280,
    "loader_cache_bytes": 1048576,
    "max_iters_per_s": 8.0,
    "max_splats": 1000,
    "max_train_steps": 100,
    "min_cell_opacity": 0.25,
    "min_depth_confidence": 1,
    "min_score_interval_s": 4.5,
    "mono_min_depth_with_lidar_m": 2.5,
    "mono_scale_range": [0.5, 2.0],
    "mono_seeding": false,
    "move_anchored_splats": false,
    "pass": {"hutchinson_samples": 2, "observed_rel_eps": 0.5, "render_scale": 0.25},
    "profile_steps": true,
    "recent_fraction": 0.5,
    "recent_window": 30,
    "refine_every": 50,
    "score_budget": 0.5,
    "seed": 7,
    "seed_alpha_threshold": 0.25,
    "seed_stride_px": 8,
    "seed_view_fraction": 0.25,
    "sh_background": true,
    "sh_background_alpha_weight": 0.5,
    "sh_degree": 2,
    "sigma_pix": 0.5,
    "ssim_every": 4,
    "status_every_step": true,
    "train_radius_m": 25,
    "voxel_size": 2,
    "warmup": false
}"#;

/// Numbers compared as `f64`, so an integer literal for a float field equals the float it serializes to.
fn numbers_as_f64(value: Value) -> Value {
    match value {
        Value::Number(n) => serde_json::json!(n.as_f64().expect("finite")),
        Value::Array(items) => Value::Array(items.into_iter().map(numbers_as_f64).collect()),
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, numbers_as_f64(v))).collect()),
        other => other,
    }
}

fn default_json() -> Value {
    serde_json::to_value(GuideConfig::default()).expect("serializes")
}

#[test]
fn default_json_keys_are_pinned() {
    let value = default_json();
    let mut keys: Vec<&str> = value.as_object().expect("object").keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, KEYS);
}

#[test]
fn every_key_round_trips() {
    let given: Value = serde_json::from_str(NON_DEFAULT_JSON).expect("valid json");
    let given_keys: Vec<&str> = given.as_object().expect("object").keys().map(String::as_str).collect();
    assert_eq!(given_keys, KEYS, "the round-trip JSON sets every key");
    let defaults = default_json();
    for key in KEYS {
        assert_ne!(
            numbers_as_f64(given[key].clone()),
            numbers_as_f64(defaults[key].clone()),
            "{key} is set to its default"
        );
    }
    let cfg: GuideConfig = serde_json::from_str(NON_DEFAULT_JSON).expect("parses");
    let back = serde_json::to_value(cfg).expect("serializes");
    assert_eq!(numbers_as_f64(back), numbers_as_f64(given));
}

#[test]
fn empty_json_is_the_default() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(serde_json::to_value(cfg).expect("serializes"), default_json());
}

#[test]
fn unknown_keys_are_ignored() {
    let cfg: GuideConfig =
        serde_json::from_str(r#"{"gpu_autotune_level": 2, "no_such_key": 1}"#).expect("parses");
    assert_eq!(serde_json::to_value(cfg).expect("serializes"), default_json());
}

#[test]
fn defaults_and_old_configs_parse() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(cfg.seed_stride_px, 4);
    assert_eq!(cfg.seed_view_fraction, 0.03);
    let cfg: GuideConfig = serde_json::from_str(r#"{"seed_stride_px": 8}"#).expect("parses");
    assert_eq!(cfg.seed_stride_px, 8);
    assert_eq!(cfg.seed_view_fraction, 0.03);
}

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

#[test]
fn finish_fisher_defaults_to_off_and_parses() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert!(!cfg.finish_fisher);
    let cfg: GuideConfig = serde_json::from_str(r#"{"finish_fisher": true}"#).expect("parses");
    assert!(cfg.finish_fisher);
}

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

#[test]
fn moving_anchored_splats_is_on_by_default_and_can_be_turned_off() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert!(cfg.move_anchored_splats);
    let cfg: GuideConfig =
        serde_json::from_str(r#"{"move_anchored_splats": false}"#).expect("parses");
    assert!(!cfg.move_anchored_splats);
}

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

#[test]
fn train_radius_defaults_to_off_and_parses() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(cfg.train_radius_m, 0.0);
    let cfg: GuideConfig = serde_json::from_str(r#"{"train_radius_m": 25}"#).expect("parses");
    assert_eq!(cfg.train_radius_m, 25.0);
}

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
