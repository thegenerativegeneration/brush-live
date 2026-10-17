use crate::ReconFrame;
use glam::UVec2;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaptureSelection {
    pub first_id: u64,
    pub last_id: u64,
    pub holdout_every: usize,
    pub holdout_offset: usize,
    pub size: UVec2,
}

impl Default for CaptureSelection {
    fn default() -> Self {
        Self {
            first_id: 174,
            last_id: 213,
            holdout_every: 5,
            holdout_offset: 4,
            size: UVec2::new(320, 240),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub format_version: u32,
    pub source: PathBuf,
    pub source_kind: String,
    pub pose_convention: String,
    pub selection: CaptureSelection,
    pub train_ids: Vec<u64>,
    pub holdout_ids: Vec<u64>,
    /// Relative source paths to SHA-256, including the complete transforms JSON.
    pub source_sha256: BTreeMap<String, String>,
    pub selection_note: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen: Option<FrozenAssets>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetReference {
    pub file: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenAssets {
    pub seed: AssetReference,
    pub roi: BTreeMap<u64, AssetReference>,
    pub seed_config: serde_json::Value,
    pub loss_config: serde_json::Value,
    pub schedule_config: serde_json::Value,
    pub render_config: serde_json::Value,
    pub triangle_train_config: serde_json::Value,
    pub gaussian_train_config: serde_json::Value,
    pub rng_seed: u64,
    pub git_revision: String,
    pub dirty_patch_sha256: String,
    pub environment: serde_json::Value,
}

pub struct CaptureSplit {
    pub train: Vec<Arc<ReconFrame>>,
    pub holdout: Vec<Arc<ReconFrame>>,
    pub manifest: CaptureManifest,
}
