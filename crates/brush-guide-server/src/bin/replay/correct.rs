//! Live poses and one correction for replays of a phone export: the poses the phone sent during capture
//! (`transforms_live.json`) and, after `--correct-after` keyframes, one `poses` frame with the export's refined poses
//! (`transforms.json`), as the phone's engine gets them after a loop closure.

use crate::dataset::Transforms;
use brush_guide::protocol::{ClientHeader, PoseUpdate, encode_frame};
use glam::Mat4;
use std::path::Path;

/// Column-major camera-to-world of a frame's row-major `transform_matrix`.
pub(crate) fn pose(transform_matrix: &[[f32; 4]; 4]) -> [f32; 16] {
    Mat4::from_cols_array_2d(transform_matrix)
        .transpose()
        .to_cols_array()
}

/// The capture-time pose of each frame of `t`, from `transforms_live.json`; an error when that file is missing or
/// does not list the same images in the same order.
pub(crate) fn live_poses(dir: &Path, t: &Transforms) -> anyhow::Result<Vec<[f32; 16]>> {
    let live: Transforms =
        serde_json::from_slice(&std::fs::read(dir.join("transforms_live.json"))?)?;
    anyhow::ensure!(
        live.frames.len() == t.frames.len()
            && live
                .frames
                .iter()
                .zip(&t.frames)
                .all(|(a, b)| a.file_path == b.file_path),
        "transforms_live.json does not list the frames of transforms.json"
    );
    Ok(live
        .frames
        .iter()
        .map(|f| pose(&f.transform_matrix))
        .collect())
}

/// One `poses` frame for keyframe ids `0..sent`, each with its frame's refined pose (ids cycle over the frames as
/// `--loops` sends them).
pub(crate) fn correction_frame(t: &Transforms, sent: usize) -> Vec<u8> {
    let poses = (0..sent)
        .map(|i| PoseUpdate {
            id: i as u64,
            pose: pose(&t.frames[i % t.frames.len()].transform_matrix),
        })
        .collect();
    encode_frame(&ClientHeader::Poses { poses }, &[])
}
