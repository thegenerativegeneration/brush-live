use brush_recon_bench::{
    cli::{Cli, PreflightArgs},
    preflight::{load_frozen, preflight},
    run::{RunBudget, RunConfig, run_backend},
    triangle::TriangleBackend,
};
use brush_triangle_train::artifact::TriangleState;
use burn::tensor::Device;
use clap::Parser;
use half::f16;
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

fn fixture() -> (tempfile::TempDir, PreflightArgs) {
    let d = tempfile::tempdir().unwrap();
    let capture = d.path().join("Capture With Spaces");
    let segment = capture.join("segment-0");
    fs::create_dir_all(segment.join("images")).unwrap();
    fs::create_dir(segment.join("depth")).unwrap();
    fs::write(capture.join("capture.json"), b"{}").unwrap();
    let mut frames = vec![];
    for id in (174..=213).rev() {
        RgbImage::from_pixel(8, 6, Rgb([180, 90, 40]))
            .save(segment.join(format!("images/{id}.jpg")))
            .unwrap();
        let z: Vec<_> = [2.0_f32; 48]
            .into_iter()
            .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        fs::write(segment.join(format!("depth/{id}.f16")), z).unwrap();
        fs::write(segment.join(format!("depth/{id}.conf")), [2; 48]).unwrap();
        frames.push(json!({"file_path":format!("images/{id}.jpg"),"w":8,"h":6,"fl_x":6.,"fl_y":6.,"cx":4.,"cy":3.,"depth_w":8,"depth_h":6,"lidar_depth_file_path":format!("depth/{id}.f16"),"transform_matrix":[[1.,0.,0.,(id-174) as f32*0.005],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]}));
    }
    fs::write(
        segment.join("transforms.json"),
        serde_json::to_vec(&json!({"camera":"wide","frames":frames})).unwrap(),
    )
    .unwrap();
    fs::write(
        segment.join("splat.ply"),
        b"must not use existing trained splat",
    )
    .unwrap();
    let args = PreflightArgs {
        segment,
        output: d.path().join("Frozen Output"),
        first_id: 174,
        last_id: 213,
        holdout_every: 5,
        holdout_offset: 4,
        width: 8,
        seed: 0,
    };
    (d, args)
}
fn inventory(root: &Path, path: &Path) -> BTreeMap<String, String> {
    let mut all = BTreeMap::new();
    for e in fs::read_dir(path).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            all.extend(inventory(root, &p));
        } else {
            all.insert(
                p.strip_prefix(root).unwrap().display().to_string(),
                format!("{:x}", Sha256::digest(fs::read(p).unwrap())),
            );
        }
    }
    all
}

#[test]
fn budget_modes_are_exclusive_and_raw_selection_cannot_override_frozen_manifest() {
    let base = [
        "bench",
        "run",
        "--manifest",
        "frozen.json",
        "--backend",
        "triangle",
        "--output",
        "out",
    ];
    assert!(Cli::try_parse_from(base).is_err());
    let mut a = base.to_vec();
    a.extend(["--steps", "2"]);
    assert!(Cli::try_parse_from(&a).is_ok());
    a.extend(["--train-seconds", "60"]);
    assert!(Cli::try_parse_from(&a).is_err());
    let mut a = base.to_vec();
    a.extend(["--steps", "2", "--first-id", "0"]);
    assert!(Cli::try_parse_from(a).is_err());
}

#[test]
fn preflight_handles_spaces_and_never_modifies_capture_or_overwrites_output() {
    let (_d, a) = fixture();
    let root = a.segment.parent().unwrap();
    let before = inventory(root, root);
    let summary = preflight(&a, json!({"test":true})).unwrap();
    assert_eq!((summary.train_views, summary.holdout_views), (32, 8));
    assert!(summary.triangles > 0);
    assert_eq!(before, inventory(root, root));
    assert!(preflight(&a, json!({})).is_err());
    let frozen = load_frozen(&summary.manifest).unwrap();
    assert_eq!(frozen.split.train[0].id, 174);
    assert_eq!(frozen.split.holdout[0].id, 178);
}

#[test]
fn nested_capture_output_and_symlink_alias_are_rejected() {
    let (d, mut a) = fixture();
    a.output = a.segment.parent().unwrap().join("do-not-create");
    assert!(preflight(&a, json!({})).is_err());
    assert!(!a.output.exists());
    #[cfg(unix)]
    {
        let alias = d.path().join("alias");
        std::os::unix::fs::symlink(a.segment.parent().unwrap(), &alias).unwrap();
        a.output = alias.join("do-not-create");
        assert!(preflight(&a, json!({})).is_err());
        assert!(!a.output.exists());
    }
}

#[test]
fn frozen_geometry_and_masks_do_not_depend_on_heldout_rgb_or_depth() {
    let (_d, mut a) = fixture();
    let first = preflight(&a, json!({})).unwrap();
    let f = load_frozen(&first.manifest).unwrap();
    let hash = f.seed.fingerprint().unwrap();
    let masks: Vec<_> = f
        .split
        .train
        .iter()
        .chain(&f.split.holdout)
        .map(|f| f.roi.clone())
        .collect();
    RgbImage::from_pixel(8, 6, Rgb([0, 255, 0]))
        .save(a.segment.join("images/178.jpg"))
        .unwrap();
    let bytes: Vec<_> = [7.0_f32; 48]
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    fs::write(a.segment.join("depth/178.f16"), bytes).unwrap();
    a.output = a.output.with_file_name("second frozen");
    let second = preflight(&a, json!({})).unwrap();
    let f = load_frozen(&second.manifest).unwrap();
    assert_eq!(hash, f.seed.fingerprint().unwrap());
    let new: Vec<_> = f
        .split
        .train
        .iter()
        .chain(&f.split.holdout)
        .map(|f| f.roi.clone())
        .collect();
    assert_eq!(masks, new);
    assert!(
        load_frozen(&first.manifest).is_err(),
        "source mutation invalidates previous frozen manifest"
    );
}

#[test]
fn wrong_calibration_and_tampered_assets_fail_without_fallback() {
    let (_d, a) = fixture();
    let p = a.segment.join("transforms.json");
    let original = fs::read(&p).unwrap();
    let mut scene: Value = serde_json::from_slice(&original).unwrap();
    scene["camera_model"] = json!("OPENCV");
    fs::write(&p, serde_json::to_vec(&scene).unwrap()).unwrap();
    assert!(preflight(&a, json!({})).is_err());
    assert!(!a.output.exists());
    fs::write(&p, original).unwrap();
    let result = preflight(&a, json!({})).unwrap();
    let manifest: Value = serde_json::from_slice(&fs::read(&result.manifest).unwrap()).unwrap();
    let roi = manifest["frozen"]["roi"]["174"]["file"].as_str().unwrap();
    fs::write(a.output.join(roi), [0]).unwrap();
    assert!(load_frozen(&result.manifest).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn runner_discards_warmup_and_draws_only_training_ids_in_order() {
    let (_d, a) = fixture();
    let summary = preflight(&a, json!({})).unwrap();
    let f = load_frozen(&summary.manifest).unwrap();
    let d = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let record = run_backend(
        || TriangleBackend::new(f.seed.clone(), d.clone(), f.config.clone()),
        &f.split,
        &RunConfig {
            budget: RunBudget::Steps { steps: 2 },
            warmup_steps: 1,
        },
    )
    .await
    .unwrap();
    assert_eq!(
        record.curve.iter().map(|p| p.frame_id).collect::<Vec<_>>(),
        [174, 175]
    );
    let initial: TriangleState = serde_json::from_slice(&record.initial_state.payload).unwrap();
    let final_state: TriangleState = serde_json::from_slice(&record.final_state.payload).unwrap();
    assert_eq!(initial.iterations, 0);
    assert_eq!(final_state.iterations, 2);
    assert_eq!(record.curve[0].progress, 0.);
    assert_eq!(record.curve[1].progress, 1.);
}

#[test]
fn actual_cli_accepts_bare_manifest_and_paths_with_spaces() {
    let (d, a) = fixture();
    preflight(&a, json!({})).unwrap();
    let output = d.path().join("CLI Run With Spaces");
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_brush-recon-bench"))
        .current_dir(&a.output)
        .args([
            "run",
            "--manifest",
            "manifest.json",
            "--backend",
            "triangle",
            "--steps",
            "2",
            "--warmup-steps",
            "0",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(output.join("run.json")).unwrap()).unwrap();
    assert_eq!(report["run"]["curve"].as_array().unwrap().len(), 2);
    assert_eq!(report["run"]["curve"][0]["frame_id"], 174);
}
