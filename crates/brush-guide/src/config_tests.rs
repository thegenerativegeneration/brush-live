use super::{AreaConfig, AreaFreeze, GuideConfig};
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

/// `GuideConfig::default()` serialized, as printed by `serde_json` (`f32` fields as their `f64` widening).
const DEFAULT_JSON: &str = r#"{
  "all_loader_rebuild_every": 10,
  "area_active_min_rows": 50,
  "area_active_tiles": 3,
  "area_engage_on_leave": true,
  "area_far_seed_m": 10.0,
  "area_freeze": "inactive_at_cap",
  "area_freeze_at": 0.8999999761581421,
  "area_held_fraction": 0.25,
  "area_hysteresis_m": 5.0,
  "area_junction_steps": 200,
  "area_min_resident_keyframes": 20,
  "area_radius_m": 50.0,
  "area_restore_inconsistency_m": 1.0,
  "area_restore_min_seeds": 8,
  "area_seen_window_steps": 1000,
  "area_tile_max_fraction": 0.0,
  "area_tile_size_m": 10.0,
  "area_view_m": 15.0,
  "coverage": {
    "low_res_penalty": 0.5,
    "min_px_per_m": 300.0,
    "n_target": 8.0,
    "spread_target": 0.20000000298023224
  },
  "dump_raw_uncertainty": false,
  "eval_every_keyframes": 0,
  "eval_extras_at": [],
  "eval_extras_views": [],
  "eval_interval_s": 30.0,
  "eval_lpips": false,
  "eval_save_views": [],
  "eval_seam_band_m": 0.0,
  "eval_seam_from": 0,
  "evict": true,
  "evict_headroom": 0.10000000149011612,
  "evict_max_cell_fraction": 0.30000001192092896,
  "evict_min_age": 3,
  "evict_recent_refines": 10,
  "finish_fisher": false,
  "fisher_lambda": 9.999999974752427e-7,
  "fisher_lambda_rel": 0.0010000000474974513,
  "holdout_every": 0,
  "holdout_max_received": 0,
  "init_random_count": 5000,
  "keyframe_long_side": 960,
  "loader_cache_bytes": 1073741824,
  "max_iters_per_s": 0.0,
  "max_splats": 750000,
  "max_train_steps": null,
  "min_cell_opacity": 0.10000000149011612,
  "min_depth_confidence": 2,
  "min_score_interval_s": 2.0,
  "mono_min_depth_with_lidar_m": 4.5,
  "mono_scale_range": [
    0.30000001192092896,
    3.0
  ],
  "mono_seeding": true,
  "move_anchored_splats": true,
  "pass": {
    "hutchinson_samples": 1,
    "observed_rel_eps": 9.999999974752427e-7,
    "render_scale": 0.5
  },
  "profile_steps": false,
  "recent_fraction": 0.699999988079071,
  "recent_window": 20,
  "refine_every": 100,
  "score_budget": 0.25,
  "seed": 42,
  "seed_alpha_threshold": 0.5,
  "seed_stride_px": 4,
  "seed_view_fraction": 0.029999999329447746,
  "sh_background": false,
  "sh_background_alpha_weight": 0.0,
  "sh_degree": 1,
  "sigma_pix": 0.05000000074505806,
  "ssim_every": 1,
  "status_every_step": false,
  "train_radius_m": 0.0,
  "voxel_size": 0.10000000149011612,
  "warmup": true
}"#;

/// Numbers narrowed to `f32` (kept as `f64`): an integer literal for a float field equals the float it serializes
/// to, and the default parser's last-bit error on long decimals does not matter for `f32` fields.
fn numbers_as_f32(value: Value) -> Value {
    match value {
        Value::Number(n) => serde_json::json!(f64::from(n.as_f64().expect("finite") as f32)),
        Value::Array(items) => Value::Array(items.into_iter().map(numbers_as_f32).collect()),
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, numbers_as_f32(v))).collect()),
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
    let mut given_keys: Vec<&str> = given.as_object().expect("object").keys().map(String::as_str).collect();
    given_keys.sort_unstable();
    assert_eq!(given_keys, KEYS, "the round-trip JSON sets every key");
    let defaults = default_json();
    for key in KEYS {
        assert_ne!(
            numbers_as_f32(given[key].clone()),
            numbers_as_f32(defaults[key].clone()),
            "{key} is set to its default"
        );
    }
    let cfg: GuideConfig = serde_json::from_str(NON_DEFAULT_JSON).expect("parses");
    let back = serde_json::to_value(cfg).expect("serializes");
    assert_eq!(numbers_as_f32(back), numbers_as_f32(given));
}

#[test]
fn default_json_is_pinned() {
    let pinned: Value = serde_json::from_str(DEFAULT_JSON).expect("valid json");
    assert_eq!(numbers_as_f32(default_json()), numbers_as_f32(pinned));
}

#[test]
fn partial_group_override_keeps_the_rest() {
    let cfg: GuideConfig = serde_json::from_str(r#"{"evict_headroom": 0.5, "holdout_every": 3}"#).expect("parses");
    assert_eq!((cfg.eviction.headroom, cfg.holdout.every), (0.5, 3));
    let got = serde_json::to_value(cfg).expect("serializes");
    let defaults = default_json();
    for key in KEYS.iter().filter(|k| !["evict_headroom", "holdout_every"].contains(k)) {
        assert_eq!(got[key], defaults[key], "{key} drifted");
    }
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
    assert_eq!(cfg.seeding.stride_px, 4);
    assert_eq!(cfg.seeding.view_fraction, 0.03);
    let cfg: GuideConfig = serde_json::from_str(r#"{"seed_stride_px": 8}"#).expect("parses");
    assert_eq!(cfg.seeding.stride_px, 8);
    assert_eq!(cfg.seeding.view_fraction, 0.03);
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
    assert!(cfg.mono.seeding);
    assert_eq!(cfg.mono.min_depth_with_lidar_m, 4.5);
    assert_eq!(cfg.mono.scale_range, (0.3, 3.0));
    let cfg: GuideConfig =
        serde_json::from_str(r#"{"mono_seeding": false, "mono_scale_range": [0.5, 2.0]}"#)
            .expect("parses");
    assert!(!cfg.mono.seeding);
    assert_eq!(cfg.mono.scale_range, (0.5, 2.0));
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
    assert_eq!((cfg.eval.every_keyframes, cfg.eval.lpips, cfg.eval.seam_band_m), (0, false, 0.0));
    assert!(cfg.eval.extras_at.is_empty());
    assert!(cfg.eval.save_views.is_empty());
    assert!(cfg.eval.extras_views.is_empty());
    assert_eq!((cfg.eval.seam_from, cfg.holdout.max_received), (0, 0));
    assert_eq!(cfg.area.tile_size_m, 10.0);
    let cfg: GuideConfig = serde_json::from_str(
        r#"{"eval_every_keyframes": 50, "eval_extras_at": [117, 413], "eval_lpips": true, "eval_save_views": [19, 59]}"#,
    )
    .expect("parses");
    assert_eq!(cfg.eval.save_views, vec![19, 59]);
    let cfg: GuideConfig = serde_json::from_str(
        r#"{"eval_extras_views": [9, 109], "eval_seam_from": 674, "holdout_max_received": 674}"#,
    )
    .expect("parses");
    assert_eq!((cfg.eval.extras_views, cfg.eval.seam_from, cfg.holdout.max_received), (vec![9, 109], 674, 674));
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
    assert_eq!((cfg.area.radius_m, cfg.area.hysteresis_m, cfg.area.seen_window_steps), (50.0, 5.0, 1000));
    assert_eq!((cfg.area.min_resident_keyframes, cfg.area.junction_steps, cfg.area.restore_min_seeds), (20, 200, 8));
    assert_eq!((cfg.area.restore_inconsistency_m, cfg.area.far_seed_m), (1.0, 10.0));
    let cfg: GuideConfig = serde_json::from_str(r#"{"area_radius_m": 25}"#).expect("parses");
    assert_eq!(cfg.area.radius_m, 25.0);
}

#[test]
fn held_fraction_defaults_to_a_quarter_and_parses() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(cfg.area.held_fraction, 0.25);
    let cfg: GuideConfig = serde_json::from_str(r#"{"area_held_fraction": 1.0}"#).expect("parses");
    assert_eq!(cfg.area.held_fraction, 1.0);
}

#[test]
fn freeze_defaults_to_inactive_at_cap_and_parses() {
    let cfg: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(cfg.area.freeze, AreaFreeze::InactiveAtCap);
    assert_eq!((cfg.area.view_m, cfg.area.active_tiles, cfg.area.active_min_rows), (15.0, 3, 50));
    assert_eq!(cfg.area.freeze_at, 0.9);
    assert!(cfg.area.engage_on_leave);
    let capped = { let mut c = cfg.clone(); c.area.tile_max_fraction = 0.4; c.max_splats = 25_000; c };
    assert_eq!(capped.area_tile_cap(), 10_000);
    assert_eq!(GuideConfig { area: AreaConfig { tile_max_fraction: 0.0, ..cfg.area }, ..cfg }.area_tile_cap(), 0);
    let modes = [
        ("left", AreaFreeze::Left),
        ("none", AreaFreeze::Off),
        ("inactive", AreaFreeze::Inactive),
        ("inactive_at_cap", AreaFreeze::InactiveAtCap),
    ];
    for (json, want) in modes {
        let cfg: GuideConfig =
            serde_json::from_str(&format!(r#"{{"area_freeze": "{json}"}}"#)).expect("parses");
        assert_eq!(cfg.area.freeze, want);
    }
    let cfg: GuideConfig =
        serde_json::from_str(r#"{"area_tile_max_fraction": 0.25, "max_splats": 20000}"#).expect("parses");
    assert_eq!(cfg.area_tile_cap(), 5_000);
    let cfg: GuideConfig = serde_json::from_str(r#"{"area_view_m": 20, "area_active_tiles": 5}"#).expect("parses");
    assert_eq!((cfg.area.view_m, cfg.area.active_tiles), (20.0, 5));
    let cfg: GuideConfig =
        serde_json::from_str(r#"{"area_freeze_at": 0.8, "area_tile_max_fraction": 0}"#).expect("parses");
    assert_eq!((cfg.area.freeze_at, cfg.area.tile_max_fraction), (0.8, 0.0));
}

#[test]
fn the_default_config_has_no_per_tile_cap() {
    let cfg = GuideConfig::default();
    assert_eq!(cfg.area.tile_max_fraction, 0.0);
    assert_eq!(cfg.area_tile_cap(), 0);
    let parsed: GuideConfig = serde_json::from_str("{}").expect("parses");
    assert_eq!(parsed.area_tile_cap(), 0);
}
