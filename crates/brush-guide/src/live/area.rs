//! Area training in the live model: only views whose camera lies within `area_radius_m` (horizontally) of the
//! newest keyframe's camera train; tiles the capture has left are parked on disk ([`super::area_park`]) and the eval
//! model loads them back.

use super::LiveModel;
use crate::area::{AreaCounters, AreaParams, AreaReport, TileKey, TileRegistry, area_views};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::tensor::{Tensor, s};
use glam::Vec3;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Area training's state: the tile registry, the views that train, and what parking did.
pub(crate) struct AreaState {
    pub(super) registry: TileRegistry,
    /// Views within the radius of the newest view, ascending; refreshed on every keyframe and pose update.
    views: Vec<usize>,
    pub(super) counters: AreaCounters,
    /// Parked tiles that resident rows lay in at the last visibility readback (restored from Task 10 on).
    pub(super) conflicts: Vec<TileKey>,
    /// File size of every parked tile.
    pub(super) file_bytes: BTreeMap<TileKey, u64>,
}

impl AreaState {
    pub(crate) fn new(params: AreaParams) -> Self {
        Self {
            registry: TileRegistry::new(params),
            views: Vec::new(),
            counters: AreaCounters::default(),
            conflicts: Vec::new(),
            file_bytes: BTreeMap::new(),
        }
    }
}

impl LiveModel {
    /// Turns area training on with `tiles_dir` holding parked tiles: the directory is emptied (a reused session
    /// directory may hold tiles of an earlier capture) and created. Area training stays off when `area_radius_m` is
    /// 0, or when the directory cannot be prepared (logged). Meant for a model without keyframes; with views it
    /// recomputes the area and rebuilds the loaders.
    pub fn enable_area(&mut self, tiles_dir: PathBuf) {
        let prepared = match std::fs::remove_dir_all(&tiles_dir) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => std::fs::create_dir_all(&tiles_dir),
        };
        self.area = match prepared {
            Ok(()) => AreaParams::from_config(&self.config).map(AreaState::new),
            Err(e) => {
                log::warn!("area training off: cannot prepare {}: {e}", tiles_dir.display());
                None
            }
        };
        self.tiles_dir = Some(tiles_dir);
        if !self.views.is_empty() {
            self.refresh_area_views();
            self.rebuild_all_loaders();
        }
    }

    /// The views the loaders train on, ascending: the area's views, or every view with area training off.
    pub(super) fn training_views(&self) -> Vec<usize> {
        match &self.area {
            Some(a) => a.views.clone(),
            None => (0..self.views.len()).collect(),
        }
    }

    /// Training views from index `first` on, ascending.
    pub(super) fn training_views_from(&self, first: usize) -> Vec<usize> {
        match &self.area {
            Some(a) => a.views[a.views.partition_point(|&i| i < first)..].to_vec(),
            None => (first..self.views.len()).collect(),
        }
    }

    /// Recomputes the area's views around the newest view's current camera; no-op with area training off. The
    /// newest view is always among them, even when its position is not finite.
    pub(super) fn refresh_area_views(&mut self) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        let positions: Vec<Vec3> = self.view_cams.iter().map(|(p, _)| *p).collect();
        let Some(&centre) = positions.last() else {
            area.views.clear();
            return;
        };
        area.views = area_views(&positions, centre, area.registry.params().radius);
        // Only a non-finite centre leaves the newest view out; the loaders need at least one view.
        let newest = positions.len() - 1;
        if area.views.last() != Some(&newest) {
            area.views.push(newest);
        }
    }

    /// Views in the area; `None` with area training off.
    pub fn area_view_count(&self) -> Option<u32> {
        self.area.as_ref().map(|a| a.views.len() as u32)
    }

    /// Tile counts by state around the newest view (as of the last refine's readback), the counters and every
    /// tile's state; `None` with area training off.
    pub fn area_report(&self) -> Option<AreaReport> {
        let area = self.area.as_ref()?;
        let centre = self.view_cams.last().map_or(Vec3::ZERO, |(p, _)| *p);
        Some(AreaReport {
            summary: area.registry.summary(centre),
            counters: area.counters,
            tiles: area.registry.tile_states(centre),
        })
    }

    /// The splats a held-out eval renders: the trainer's, and with parked tiles the full model, the resident splats
    /// and every parked tile (read from its file and moved by the pose corrections made since it was parked), each
    /// with its Mip floor baked in. A tile file that cannot be read, or holds another SH degree, is left out with a
    /// warning. Loading the tiles is logged with its time.
    pub async fn eval_splats(&self) -> Option<Splats> {
        let resident = self.splats.clone()?;
        let keys = self.area.as_ref().map(|a| a.registry.parked_keys()).unwrap_or_default();
        if keys.is_empty() {
            return Some(resident);
        }
        let t = web_time::Instant::now();
        let device = self.device.clone().inner();
        let coeffs = resident.sh_coeffs.dims()[1] as u32;
        let mode = if resident.render_mip { SplatRenderMode::Mip } else { SplatRenderMode::Default };
        let mut parts = vec![resident.bake_min_scale()];
        let (mut tiles, mut rows) = (0u32, 0usize);
        for key in keys {
            match self.load_parked(key) {
                Ok(r) if r.sh_coeffs_per_channel != coeffs => log::warn!(
                    "area: eval leaves out parked tile {}: {} SH coefficients, the model has {coeffs}",
                    key.stem(),
                    r.sh_coeffs_per_channel
                ),
                Ok(r) if r.is_empty() => {}
                Ok(r) => {
                    tiles += 1;
                    rows += r.len();
                    parts.push(r.baked().to_splats_raw(&device));
                }
                Err(e) => log::warn!("area: eval leaves out parked tile {}: {e}", key.stem()),
            }
        }
        let splats = concat_splats(&parts, mode);
        log::info!(
            "area: eval loads {tiles} parked tiles ({rows} splats) in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(splats)
    }
}

/// The rows of every part in order, as one set of splats without a floor.
fn concat_splats(parts: &[Splats], mode: SplatRenderMode) -> Splats {
    let transforms = Tensor::cat(parts.iter().map(|p| p.transforms.val()).collect(), 0);
    Splats::from_tensor_data(
        transforms.clone().slice(s![.., 0..3]),
        transforms.clone().slice(s![.., 3..7]),
        transforms.slice(s![.., 7..10]),
        Tensor::cat(parts.iter().map(|p| p.sh_coeffs.val()).collect(), 0),
        Tensor::cat(parts.iter().map(|p| p.raw_opacities.val()).collect(), 0),
        mode,
    )
}

#[cfg(test)]
#[path = "area_tests.rs"]
pub(in crate::live) mod tests;
