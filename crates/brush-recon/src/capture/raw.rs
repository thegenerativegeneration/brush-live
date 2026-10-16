use super::{CaptureManifest, CaptureSelection, CaptureSplit};
use crate::{MetricDepth, ReconError, ReconFrame, frame::pixel_count};
use brush_render::{
    camera::{Camera, focal_to_fov},
    kernels::camera_model::CameraModel,
};
use glam::{Mat3, Mat4, Quat, UVec2, Vec2};
use half::f16;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
    sync::Arc,
};

#[derive(Deserialize)]
struct RawScene {
    frames: Vec<RawFrame>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct RawFrame {
    file_path: String,
    w: u32,
    h: u32,
    fl_x: f64,
    fl_y: f64,
    cx: f32,
    cy: f32,
    transform_matrix: [[f32; 4]; 4],
    #[serde(alias = "depth_file_path")]
    lidar_depth_file_path: Option<String>,
    depth_w: Option<u32>,
    depth_h: Option<u32>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

fn calibration(extra: &BTreeMap<String, Value>) -> Result<(), ReconError> {
    if let Some(model) = extra.get("camera_model")
        && !matches!(
            model.as_str(),
            Some("PINHOLE" | "PERSPECTIVE" | "perspective" | "pinhole")
        )
    {
        return Err(ReconError::UnsupportedCamera);
    }
    for name in ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"] {
        if let Some(v) = extra.get(name)
            && v.as_f64() != Some(0.0)
        {
            return Err(ReconError::UnsupportedCamera);
        }
    }
    Ok(())
}

fn read_hashed(
    root: &Path,
    rel: &Path,
    hashes: &mut BTreeMap<String, String>,
) -> Result<Vec<u8>, ReconError> {
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(ReconError::input(format!(
            "unsafe capture path {}",
            rel.display()
        )));
    }
    let path = root.join(rel);
    let canonical = path.canonicalize().map_err(|e| ReconError::io(&path, e))?;
    if !canonical.starts_with(root) {
        return Err(ReconError::input(format!(
            "capture path escapes source: {}",
            path.display()
        )));
    }
    let bytes = std::fs::read(&path).map_err(|e| ReconError::io(&path, e))?;
    hashes.insert(
        rel.to_string_lossy().into_owned(),
        format!("{:x}", Sha256::digest(&bytes)),
    );
    Ok(bytes)
}

fn camera(frame: &RawFrame) -> Result<Camera, ReconError> {
    calibration(&frame.extra)?;
    if frame.w == 0
        || frame.h == 0
        || !frame.fl_x.is_finite()
        || frame.fl_x <= 0.0
        || !frame.fl_y.is_finite()
        || frame.fl_y <= 0.0
        || !frame.cx.is_finite()
        || !frame.cy.is_finite()
    {
        return Err(ReconError::input(format!(
            "{}: invalid camera intrinsics",
            frame.file_path
        )));
    }
    let mut m = Mat4::from_cols_array_2d(&frame.transform_matrix).transpose();
    if !m.is_finite()
        || m.x_axis.w.abs() > 1e-6
        || m.y_axis.w.abs() > 1e-6
        || m.z_axis.w.abs() > 1e-6
        || (m.w_axis.w - 1.0).abs() > 1e-3
    {
        return Err(ReconError::input(format!(
            "{}: invalid homogeneous pose",
            frame.file_path
        )));
    }
    m *= m.w_axis.w.recip();
    let r = Mat3::from_mat4(m);
    if (r.transpose() * r - Mat3::IDENTITY)
        .to_cols_array()
        .iter()
        .any(|v| v.abs() > 1e-3)
        || (r.determinant() - 1.0).abs() > 1e-3
    {
        return Err(ReconError::input(format!(
            "{}: pose is not rigid",
            frame.file_path
        )));
    }
    // Same OpenGL/ARKit convention as brush-dataset and brush-guide: +z forward, y down.
    m.y_axis *= -1.0;
    m.z_axis *= -1.0;
    let result = Camera::new(
        m.w_axis.truncate(),
        Quat::from_mat3(&Mat3::from_mat4(m)).normalize(),
        focal_to_fov(frame.fl_x, frame.w, &CameraModel::Pinhole),
        focal_to_fov(frame.fl_y, frame.h, &CameraModel::Pinhole),
        Vec2::new(frame.cx / frame.w as f32, frame.cy / frame.h as f32),
        CameraModel::Pinhole,
    );
    crate::frame::validate_camera(&result)?;
    Ok(result)
}

fn load_depth(
    root: &Path,
    f: &RawFrame,
    hashes: &mut BTreeMap<String, String>,
) -> Result<Option<Arc<MetricDepth>>, ReconError> {
    let Some(rel) = &f.lidar_depth_file_path else {
        return Ok(None);
    };
    let size = UVec2::new(f.depth_w.unwrap_or(0), f.depth_h.unwrap_or(0));
    let n = pixel_count(size)?;
    if size.max_element() > 4096 {
        return Err(ReconError::input(format!("{rel}: depth size exceeds 4096")));
    }
    let path = Path::new(rel);
    if path.extension().and_then(|s| s.to_str()) != Some("f16") {
        return Err(ReconError::input(format!(
            "{rel}: raw depth must be float16"
        )));
    }
    let bytes = read_hashed(root, path, hashes)?;
    if bytes.len() != n * 2 {
        return Err(ReconError::input(format!(
            "{rel}: expected {} depth bytes, got {}",
            n * 2,
            bytes.len()
        )));
    }
    let conf_path = path.with_extension("conf");
    let confidence = if root
        .join(&conf_path)
        .try_exists()
        .map_err(|e| ReconError::io(root.join(&conf_path), e))?
    {
        let c = read_hashed(root, &conf_path, hashes)?;
        if c.len() != n || c.iter().any(|v| *v > 2) {
            return Err(ReconError::input(format!(
                "{}: malformed confidence",
                conf_path.display()
            )));
        }
        Some(c)
    } else {
        None
    };
    let values_m = bytes
        .chunks_exact(2)
        .map(|v| {
            let d = f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32();
            if d.is_finite() && d > 0.0 { d } else { 0.0 }
        })
        .collect();
    let depth = MetricDepth {
        size,
        values_m,
        confidence,
    };
    depth.validate()?;
    Ok(Some(Arc::new(depth)))
}

pub fn load_capture(
    segment: &Path,
    selection: &CaptureSelection,
) -> Result<CaptureSplit, ReconError> {
    pixel_count(selection.size)?;
    if selection.holdout_every < 2
        || selection.holdout_offset >= selection.holdout_every
        || selection.last_id < selection.first_id
    {
        return Err(ReconError::input(
            "invalid frame range or holdout partition",
        ));
    }
    let root = segment
        .canonicalize()
        .map_err(|e| ReconError::io(segment, e))?;
    let mut hashes = BTreeMap::new();
    let json = read_hashed(&root, Path::new("transforms.json"), &mut hashes)?;
    let scene: RawScene = serde_json::from_slice(&json).map_err(|source| ReconError::Json {
        path: root.join("transforms.json"),
        source,
    })?;
    calibration(&scene.extra)?;
    let mut selected = vec![];
    let mut seen = BTreeSet::new();
    for frame in scene.frames {
        let id = Path::new(&frame.file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| {
                ReconError::input(format!("non-numeric capture frame {}", frame.file_path))
            })?;
        if !seen.insert(id) {
            return Err(ReconError::input(format!("duplicate frame ID {id}")));
        }
        if (selection.first_id..=selection.last_id).contains(&id) {
            selected.push((id, frame));
        }
    }
    selected.sort_by_key(|(id, _)| *id);
    let expected = selection
        .last_id
        .checked_sub(selection.first_id)
        .and_then(|v| v.checked_add(1));
    if expected != Some(selected.len() as u64) {
        return Err(ReconError::input("selected frame range has missing IDs"));
    }
    let mut train = vec![];
    let mut holdout = vec![];
    for (offset, (id, f)) in selected.into_iter().enumerate() {
        let camera = camera(&f)?;
        let bytes = read_hashed(&root, Path::new(&f.file_path), &mut hashes)?;
        let image = image::load_from_memory(&bytes)
            .map_err(|e| ReconError::input(format!("{}: {e}", f.file_path)))?
            .into_rgb8();
        if image.dimensions() != (f.w, f.h) {
            return Err(ReconError::input(format!(
                "{}: JPEG size differs from metadata",
                f.file_path
            )));
        }
        let rgb = image::imageops::resize(
            &image,
            selection.size.x,
            selection.size.y,
            image::imageops::FilterType::Triangle,
        );
        let lidar = load_depth(&root, &f, &mut hashes)?;
        let frame = Arc::new(ReconFrame {
            id,
            camera,
            rgb: Arc::new(rgb),
            lidar,
            roi: Arc::new(vec![255; pixel_count(selection.size)?]),
        });
        frame.validate()?;
        if offset % selection.holdout_every == selection.holdout_offset {
            holdout.push(frame);
        } else {
            train.push(frame);
        }
    }
    if train.is_empty() || holdout.is_empty() {
        return Err(ReconError::input(
            "capture split needs training and held-out views",
        ));
    }
    let manifest = CaptureManifest {
        format_version:1,source:root,source_kind:"arkit_raw".into(),pose_convention:"camera-to-world OpenGL/ARKit; converted to Brush +z forward/y down; depth camera-z metres".into(),
        selection:selection.clone(),train_ids:train.iter().map(|f|f.id).collect(),holdout_ids:holdout.iter().map(|f|f.id).collect(),source_sha256:hashes,
        selection_note:"Depth-support-selected subset, including eventual held-out depth statistics; optimization-isolated, not selection-blind".into(),
    };
    Ok(CaptureSplit {
        train,
        holdout,
        manifest,
    })
}
