use brush_recon::capture::{CaptureSelection, load_capture};
use glam::{UVec2, Vec2, Vec3};
use half::f16;
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("images")).unwrap();
    fs::create_dir(dir.path().join("depth")).unwrap();
    let mut frames = vec![];
    for id in (174..=213).rev() {
        RgbImage::from_pixel(4, 3, Rgb([id as u8, 60, 90]))
            .save(dir.path().join(format!("images/{id}.jpg")))
            .unwrap();
        let bytes: Vec<_> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        fs::write(dir.path().join(format!("depth/{id}.f16")), bytes).unwrap();
        fs::write(dir.path().join(format!("depth/{id}.conf")), [2, 1, 0, 2]).unwrap();
        frames.push(json!({
            "file_path":format!("images/{id}.jpg"), "w":4,"h":3,
            "fl_x":4.0,"fl_y":3.0,"cx":1.5,"cy":1.25,
            "depth_w":2,"depth_h":2,"lidar_depth_file_path":format!("depth/{id}.f16"),
            "transform_matrix":[[1.,0.,0.,1.],[0.,1.,0.,2.],[0.,0.,1.,3.],[0.,0.,0.,1.]]
        }));
    }
    fs::write(
        dir.path().join("transforms.json"),
        serde_json::to_vec(&json!({"camera":"wide","tracker":"arkit","frames":frames})).unwrap(),
    )
    .unwrap();
    // These must not be discovered as inputs or substituted for the raw data.
    fs::write(dir.path().join("splat.ply"), b"not a model").unwrap();
    fs::write(dir.path().join("images/._174.jpg"), b"appledouble").unwrap();
    dir
}

fn selection() -> CaptureSelection {
    CaptureSelection {
        size: UVec2::new(4, 3),
        ..Default::default()
    }
}

fn inventory(root: &Path, path: &Path, result: &mut BTreeMap<String, String>) {
    for e in fs::read_dir(path).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            inventory(root, &p, result);
        } else {
            result.insert(
                p.strip_prefix(root).unwrap().to_string_lossy().into(),
                format!("{:x}", Sha256::digest(fs::read(p).unwrap())),
            );
        }
    }
}

#[test]
fn split_174_213_is_32_train_8_holdout_and_numerically_ordered() {
    let d = fixture();
    let split = load_capture(d.path(), &selection()).unwrap();
    assert_eq!(split.train.len(), 32);
    assert_eq!(
        split.holdout.iter().map(|f| f.id).collect::<Vec<_>>(),
        [178, 183, 188, 193, 198, 203, 208, 213]
    );
    assert_eq!(split.train[0].id, 174);
    assert_eq!(split.train.last().unwrap().id, 212);
    assert!(split.train.windows(2).all(|p| p[0].id < p[1].id));
    assert_eq!(split.manifest.source_kind, "arkit_raw");
    assert_eq!(split.manifest.source_sha256.len(), 121);
    assert!(!split.manifest.source_sha256.contains_key("splat.ply"));
}

#[test]
fn camera_orientation_intrinsics_and_metric_depth_are_preserved() {
    let d = fixture();
    let mut sel = selection();
    sel.size = UVec2::new(8, 6);
    let split = load_capture(d.path(), &sel).unwrap();
    let f = &split.train[0];
    assert_eq!(f.rgb.dimensions(), (8, 6));
    assert!((f.camera.position - Vec3::new(1., 2., 3.)).length() < 1e-5);
    assert!((f.camera.local_to_world().transform_vector3(Vec3::Z) + Vec3::Z).length() < 1e-5);
    assert!((f.camera.local_to_world().transform_vector3(Vec3::Y) + Vec3::Y).length() < 1e-5);
    assert!((f.camera.focal(sel.size) - Vec2::new(8., 6.)).length() < 1e-5);
    assert!((f.camera.center(sel.size) - Vec2::new(3., 2.5)).length() < 1e-5);
    let depth = f.lidar.as_ref().unwrap();
    assert_eq!(depth.size, UVec2::new(2, 2));
    assert_eq!(depth.values_m, [1., 2., 3., 4.]);
    assert_eq!(depth.confidence.as_ref().unwrap(), &[2, 1, 0, 2]);
}

#[test]
fn heldout_payloads_are_not_loaded_into_training_partition() {
    let d = fixture();
    let a = load_capture(d.path(), &selection()).unwrap();
    let modified: Vec<_> = [7.0_f32; 4]
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    fs::write(d.path().join("depth/178.f16"), modified).unwrap();
    let b = load_capture(d.path(), &selection()).unwrap();
    for (x, y) in a.train.iter().zip(&b.train) {
        assert_eq!(
            x.lidar.as_ref().unwrap().values_m,
            y.lidar.as_ref().unwrap().values_m
        );
        assert_eq!(x.rgb.as_raw(), y.rgb.as_raw());
    }
    assert_eq!(b.holdout[0].lidar.as_ref().unwrap().values_m, [7.; 4]);
}

#[test]
fn malformed_depth_is_contextual_error() {
    let d = fixture();
    fs::write(d.path().join("depth/175.f16"), [0, 0]).unwrap();
    let error = load_capture(d.path(), &selection())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("175.f16"), "{error}");
}

#[test]
fn appledouble_entries_are_ignored_and_loading_does_not_write_to_source() {
    let d = fixture();
    let mut before = BTreeMap::new();
    inventory(d.path(), d.path(), &mut before);
    let _ = load_capture(d.path(), &selection()).unwrap();
    let mut after = BTreeMap::new();
    inventory(d.path(), d.path(), &mut after);
    assert_eq!(before, after);
}

#[test]
fn distortion_duplicate_ids_and_missing_frames_fail_explicitly() {
    let d = fixture();
    let path = d.path().join("transforms.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for kind in 0..3 {
        let mut scene = original.clone();
        match kind {
            0 => {
                scene["camera_model"] = json!("OPENCV");
            }
            1 => {
                let f = scene["frames"][0].clone();
                scene["frames"].as_array_mut().unwrap().push(f);
            }
            _ => {
                scene["frames"].as_array_mut().unwrap().pop();
            }
        }
        fs::write(&path, serde_json::to_vec(&scene).unwrap()).unwrap();
        assert!(load_capture(d.path(), &selection()).is_err());
    }
}

#[test]
fn pixel_center_depth_lookup_respects_confidence_and_bounds() {
    let d = brush_recon::MetricDepth {
        size: UVec2::new(2, 2),
        values_m: vec![1., 2., 3., 4.],
        confidence: Some(vec![2, 1, 0, 2]),
    };
    assert_eq!(d.sample_uv(Vec2::new(0.25, 0.25), 2), Some(1.));
    assert_eq!(d.sample_uv(Vec2::new(0.75, 0.75), 2), Some(4.));
    assert_eq!(d.sample_uv(Vec2::new(0.75, 0.25), 2), None);
    assert_eq!(d.sample_uv(Vec2::new(0.75, 0.25), 1), Some(2.));
    for uv in [
        Vec2::new(-0.1, 0.2),
        Vec2::new(1., 0.2),
        Vec2::new(f32::NAN, 0.2),
    ] {
        assert_eq!(d.sample_uv(uv, 2), None);
    }
}
