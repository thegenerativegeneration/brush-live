//! Reading a nerfstudio-style capture export: `transforms.json`, depth and
//! confidence files, feature points.

use clap::ValueEnum;
use glam::Vec3;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, ValueEnum)]
pub(crate) enum DepthMode {
    /// Send no depth.
    None,
    /// Send depth without confidence; the server uses every value.
    All,
    /// Send depth with confidence; the server masks low-confidence depth.
    High,
}

#[derive(Deserialize)]
pub(crate) struct Transforms {
    pub(crate) fl_x: Option<f32>,
    pub(crate) fl_y: Option<f32>,
    pub(crate) cx: Option<f32>,
    pub(crate) cy: Option<f32>,
    pub(crate) w: Option<u32>,
    pub(crate) h: Option<u32>,
    pub(crate) ply_file_path: Option<String>,
    pub(crate) frames: Vec<Frame>,
}

#[derive(Deserialize)]
pub(crate) struct Frame {
    pub(crate) file_path: String,
    pub(crate) transform_matrix: [[f32; 4]; 4],
    pub(crate) fl_x: Option<f32>,
    pub(crate) fl_y: Option<f32>,
    pub(crate) cx: Option<f32>,
    pub(crate) cy: Option<f32>,
    pub(crate) w: Option<u32>,
    pub(crate) h: Option<u32>,
    /// Raw f16 lidar depth. Exports before the rename used `depth_file_path`, which
    /// nerfstudio-style loaders read as a depth image, so it is only a fallback.
    pub(crate) lidar_depth_file_path: Option<String>,
    pub(crate) depth_file_path: Option<String>,
    pub(crate) depth_w: Option<u32>,
    pub(crate) depth_h: Option<u32>,
}

impl Frame {
    pub(crate) fn depth_path(&self) -> Option<&str> {
        self.lidar_depth_file_path
            .as_deref()
            .or(self.depth_file_path.as_deref())
    }
}

/// A frame's depth bytes, confidence bytes and depth size.
pub(crate) type Depth = (Vec<u8>, Option<Vec<u8>>, [u32; 2]);

/// Loads a frame's depth (and, under `High`, confidence) from the export directory.
///
/// Depth bytes are the raw little-endian f16 samples on disk, returned unchanged.
/// Confidence is one byte per pixel at the path with the depth file's extension
/// replaced by `.conf`. Missing confidence under `High` is not an error here; the
/// caller decides whether/how to warn and falls back to sending depth alone.
pub(crate) fn load_depth(
    dir: &Path,
    frame: &Frame,
    mode: DepthMode,
) -> anyhow::Result<Option<Depth>> {
    if matches!(mode, DepthMode::None) {
        return Ok(None);
    }
    let (Some(path), Some(w), Some(h)) = (frame.depth_path(), frame.depth_w, frame.depth_h) else {
        return Ok(None);
    };
    let depth_bytes = std::fs::read(dir.join(path))?;
    let confidence = if matches!(mode, DepthMode::High) {
        let conf_path = Path::new(path).with_extension("conf");
        std::fs::read(dir.join(&conf_path)).ok()
    } else {
        None
    };
    Ok(Some((depth_bytes, confidence, [w, h])))
}

/// Reads an ASCII PLY whose first three vertex properties are x y z.
fn load_points(path: &Path) -> Result<Vec<Vec3>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (header, body) = text
        .split_once("end_header")
        .ok_or_else(|| format!("{} has no PLY header", path.display()))?;
    if !header.contains("format ascii") {
        return Err(format!("{} is not an ASCII PLY", path.display()));
    }
    Ok(body
        .lines()
        .filter_map(|l| {
            let v: Vec<f32> = l
                .split_whitespace()
                .take(3)
                .filter_map(|t| t.parse().ok())
                .collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .collect())
}

/// The export's feature points; empty, with a warning, if there are none.
pub(crate) fn feature_points(dataset: &Path, t: &Transforms) -> Vec<Vec3> {
    match &t.ply_file_path {
        None => {
            eprintln!(
                "warning: transforms.json has no ply_file_path; sending keyframes without feature points"
            );
            Vec::new()
        }
        Some(p) => match load_points(&dataset.join(p)) {
            Ok(points) if points.is_empty() => {
                eprintln!("warning: {p} has no points; sending keyframes without feature points");
                points
            }
            Ok(points) => points,
            Err(e) => {
                eprintln!("warning: {e}; sending keyframes without feature points");
                Vec::new()
            }
        },
    }
}
