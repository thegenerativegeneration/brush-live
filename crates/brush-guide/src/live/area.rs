//! Area training in the live model: only views whose camera lies within `area_radius_m` (horizontally) of the
//! newest keyframe's camera train; tiles the capture has left are held still while resident ([`super::area_freeze`]),
//! parked on disk ([`super::area_park`]), come back when it returns ([`super::area_restore`]), and the eval model loads
//! them back.

use super::LiveModel;
use brush_dataset::scene::SceneBatch;
use crate::area::{AreaCounters, AreaParams, AreaReport, TileKey, TileRegistry, area_views};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::tensor::{Tensor, s};
use glam::Vec3;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Area training's state: the tile registry, the views that train, and what parking did.
pub(crate) struct AreaState {
    pub(super) registry: TileRegistry,
    /// Views within the radius of the newest view, ascending; refreshed on every keyframe and pose update.
    pub(super) views: Vec<usize>,
    pub(super) counters: AreaCounters,
    /// Parked tiles that resident rows lay in at the last visibility readback; the refine's park step restores them.
    pub(super) conflicts: Vec<TileKey>,
    /// File size of every parked tile.
    pub(super) file_bytes: BTreeMap<TileKey, u64>,
    /// The last park was skipped for taking every resident row (logged once per streak).
    pub(super) skipping: bool,
    /// A tile file could not be written: no parking until the next reset.
    pub(super) parking_off: bool,
    /// Training on the views that see tiles just restored.
    pub(super) junction: Junction,
    /// Seed counts of the last keyframe that seeded again after restoring tiles its seeds hit: (before, after).
    pub(super) last_reseed: Option<(u32, u32)>,
    /// Parked tiles in a deferral streak (deferred at the last keyframe or since, and not back), with the streak's
    /// deferrals so far. A streak logs once and notes its shortfall every `DEFER_RENOTE_EVERY` deferrals.
    pub(super) deferring: BTreeMap<TileKey, u32>,
    /// Tiles deferred since the last keyframe ended; a streak whose tile is not among them ends with the keyframe.
    pub(super) deferred_now: BTreeSet<TileKey>,
    /// The left resident tiles whose rows the trainer holds ([`super::area_freeze`]).
    pub(super) frozen_keys: BTreeSet<TileKey>,
    /// Rows the trainer held when the mask was last built.
    pub(super) frozen_rows: u32,
}

/// The junction refresh after a restore: for `steps_left` training steps every other step draws one of `views`, the
/// views that see a restored tile's box, so the restored tiles and the live area train together at their seam.
#[derive(Debug, Default)]
pub(super) struct Junction {
    /// Bounding boxes (min, max) of the restored rows the junction blends in.
    pub(super) boxes: Vec<(Vec3, Vec3)>,
    /// Views that see one of `boxes`, ascending.
    pub(super) views: Vec<usize>,
    pub(super) steps_left: u32,
    /// Whether the last call drew a junction view.
    pub(super) flip: bool,
    /// Decoded batches of the junction views drawn so far, at most [`JUNCTION_CACHE`].
    pub(super) cache: Vec<(usize, SceneBatch)>,
}

/// Junction views kept decoded; the junction usually has fewer.
const JUNCTION_CACHE: usize = 16;

impl Junction {
    /// Counts one training step while the junction lasts and returns the view it draws: every other step one of
    /// `views` in turn, `None` on the steps between, without views, or once the junction is over. The step that ends
    /// the junction clears `boxes`, `views` and the decoded batches.
    pub(super) fn pick(&mut self) -> Option<usize> {
        if self.steps_left == 0 {
            return None;
        }
        self.steps_left -= 1;
        self.flip = !self.flip;
        let pick = (self.flip && !self.views.is_empty())
            .then(|| self.views[(self.steps_left / 2) as usize % self.views.len()]);
        if self.steps_left == 0 {
            self.boxes.clear();
            self.views.clear();
            self.cache.clear();
            self.flip = false;
        }
        pick
    }
}

impl AreaState {
    pub(crate) fn new(params: AreaParams) -> Self {
        Self {
            registry: TileRegistry::new(params),
            views: Vec::new(),
            counters: AreaCounters::default(),
            conflicts: Vec::new(),
            file_bytes: BTreeMap::new(),
            skipping: false,
            parking_off: false,
            junction: Junction::default(),
            last_reseed: None,
            deferring: BTreeMap::new(),
            deferred_now: BTreeSet::new(),
            frozen_keys: BTreeSet::new(),
            frozen_rows: 0,
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
            parking_off: area.parking_off,
            frozen_rows: area.frozen_rows,
        })
    }

    /// The training batch of junction view `i`: decoded once and kept while the junction lasts (up to
    /// [`JUNCTION_CACHE`] views), with the view's current camera.
    pub(super) async fn junction_batch(&mut self, i: usize) -> SceneBatch {
        let camera = self.views[i].camera;
        let cached = self.area.as_ref().and_then(|a| a.junction.cache.iter().find(|(v, _)| *v == i));
        if let Some((_, batch)) = cached {
            return SceneBatch { camera, ..batch.clone() };
        }
        let batch = super::draw::load_batch(&self.views[i]).await;
        if let Some(a) = self.area.as_mut().filter(|a| a.junction.cache.len() < JUNCTION_CACHE) {
            a.junction.cache.push((i, batch.clone()));
        }
        batch
    }

    /// Junction steps still to come after the last restore; 0 with area training off.
    pub fn junction_steps_left(&self) -> u32 {
        self.area.as_ref().map_or(0, |a| a.junction.steps_left)
    }

    /// The seed counts (before, after) of the last keyframe that seeded again once the tiles its seeds hit were
    /// restored; `None` before any such keyframe or with area training off.
    pub fn last_reseed(&self) -> Option<(u32, u32)> {
        self.area.as_ref()?.last_reseed
    }

    /// The splats a held-out eval renders: the trainer's, and with parked tiles the full model, the resident splats
    /// and every parked tile (read from its file and moved by the pose corrections made since it was parked), each
    /// with its Mip floor baked in. A tile file that cannot be read, holds another SH degree or rows out of shape
    /// ([`Self::full_model_rows`]) is left out with a warning. Loading the tiles is logged with its time.
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
            match self.full_model_rows(key, coeffs) {
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
