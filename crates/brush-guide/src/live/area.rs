//! Area training in the live model: only views whose camera lies within `area_radius_m` (horizontally) of the
//! newest keyframe's camera train.

use super::LiveModel;
use crate::area::{AreaParams, TileRegistry, area_views};
use glam::Vec3;
use std::path::PathBuf;

/// Area training's state: the tile registry and the views that train.
pub(crate) struct AreaState {
    registry: TileRegistry,
    /// Views within the radius of the newest view, ascending; refreshed on every keyframe and pose update.
    views: Vec<usize>,
}

impl AreaState {
    pub(crate) fn new(params: AreaParams) -> Self {
        Self {
            registry: TileRegistry::new(params),
            views: Vec::new(),
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
    /// newest view is always among them.
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
        debug_assert_eq!(area.views.last(), Some(&(positions.len() - 1)), "newest view outside its own area");
    }

    /// Views in the area; `None` with area training off.
    pub fn area_view_count(&self) -> Option<u32> {
        self.area.as_ref().map(|a| a.views.len() as u32)
    }
}

#[cfg(test)]
#[path = "area_tests.rs"]
mod tests;
