//! Area training in the live model: only the views that see an active tile ([`super::area_active`]), or that lie
//! within `area_radius_m` (horizontally) of the newest keyframe's camera, train (with `area_freeze` inactive_at_cap
//! every view until the cap, [`super::area_budget`]); rows outside the active tiles, or
//! of tiles the capture has left, are held still while resident ([`super::area_freeze`]),
//! parked on disk ([`super::area_park`]), come back when it returns ([`super::area_restore`]), and the eval model loads
//! them back. All of this waits until area training engages ([`super::area_engage`]: the walker has left a tile);
//! until then the engine trains as with area training off.

use super::LiveModel;
use super::area_active::{ActiveSet, TileHeights};
use super::area_budget::BudgetTiles;
use super::area_freeze::HeldMask;
use super::area_park::Parking;
use super::area_restore::Deferrals;
use super::area_sight::Sight;
use brush_dataset::scene::SceneBatch;
use crate::config::AreaFreeze;
use crate::area::{AreaCounters, AreaParams, AreaReport, TileRegistry, area_views, tile_splats, views_seeing};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::rows::SplatRows;
use burn::tensor::{Tensor, s};
use glam::Vec3;
use std::path::PathBuf;

/// Area training's state: the tile registry, the views that train, and what parking did.
pub(crate) struct AreaState {
    pub(super) registry: TileRegistry,
    /// The training views ([`LiveModel::refresh_area_views`]), ascending; refreshed on every keyframe, pose update
    /// and inactive_at_cap switch.
    pub(super) views: Vec<usize>,
    pub(super) counters: AreaCounters,
    /// Parked tiles, their files and whether parking is on.
    pub(super) parking: Parking,
    /// Training on the views that see tiles just restored.
    pub(super) junction: Junction,
    /// Seed counts of the last keyframe that seeded again after restoring tiles its seeds hit: (before, after).
    pub(super) last_reseed: Option<(u32, u32)>,
    /// The deferral streaks of parked tiles.
    pub(super) defer: Deferrals,
    /// The mask that holds rows still ([`super::area_freeze`]).
    pub(super) mask: HeldMask,
    /// The active tiles ([`super::area_active`]).
    pub(super) active: ActiveSet,
    /// The tiles' height bands.
    pub(super) heights: TileHeights,
    /// With `area_freeze` inactive_at_cap: the rows outside the active tiles are held ([`LiveModel::note_cap`]).
    pub(super) at_cap: bool,
    /// inactive_at_cap switched while a keyframe was being added (its camera noted, its view not yet in): the
    /// keyframe's loader rebuild builds both loaders afresh ([`LiveModel::rebuild_loaders`]).
    pub(super) loaders_stale: bool,
    /// The row budget's tiles and evictions ([`super::area_budget`]).
    pub(super) budget: BudgetTiles,
    /// The views that see the resident model ([`super::area_sight`]).
    pub(super) sight: Sight,
    /// Area training is engaged ([`super::area_engage`]); until then the engine trains as with it off.
    pub(super) engaged: bool,
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
            parking: Parking::default(),
            junction: Junction::default(),
            last_reseed: None,
            defer: Deferrals::default(),
            mask: HeldMask::default(),
            active: ActiveSet::default(),
            heights: TileHeights::default(),
            at_cap: false,
            loaders_stale: false,
            budget: BudgetTiles::default(),
            sight: Sight::default(),
            engaged: false,
        }
    }
}

impl LiveModel {
    /// Turns area training on with `tiles_dir` holding parked tiles: the directory is emptied (a reused session
    /// directory may hold tiles of an earlier capture) and created. Area training stays off when `area_radius_m` is
    /// 0 (the directory is then left alone), or when the directory cannot be prepared (logged). `tiles_dir` is kept
    /// either way for [`Self::reset`]. Meant for a model without keyframes; with views it recomputes the area and
    /// rebuilds the loaders.
    pub fn enable_area(&mut self, tiles_dir: PathBuf) {
        self.area = AreaParams::from_config(&self.config).and_then(|p| {
            let prepared = match std::fs::remove_dir_all(&tiles_dir) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                _ => std::fs::create_dir_all(&tiles_dir),
            };
            match prepared {
                Ok(()) => Some(AreaState { engaged: !self.config.area_engage_on_leave, ..AreaState::new(p) }),
                Err(e) => {
                    log::warn!("area training off: cannot prepare {}: {e}", tiles_dir.display());
                    None
                }
            }
        });
        self.tiles_dir = Some(tiles_dir);
        if !self.views.is_empty() {
            self.refresh_area_views();
            self.refresh_floor_cams();
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

    /// Recomputes the area's views; no-op with area training off. Before area training engages
    /// ([`super::area_engage`]), and with `area_freeze` inactive_at_cap below the cap ([`Self::holds_inactive`]), they
    /// are every view, as with area training off. Otherwise, with an active set and
    /// an active tile that holds rows, they are the views that see an active tile up to `area_view_m` deep
    /// ([`crate::area::views_seeing`]); else the views within the radius of the newest view's current camera.
    /// The newest view is always among them, even when its position is not finite. A changed active set is logged
    /// with the view count.
    pub(super) fn refresh_area_views(&mut self) {
        let view_m = self.config.area_view_m;
        let engaged = self.area_engaged();
        let every = !engaged || (self.config.area_freeze == AreaFreeze::InactiveAtCap && !self.holds_inactive());
        let Some(area) = self.area.as_mut() else {
            return;
        };
        let positions: Vec<Vec3> = self.view_cams.iter().map(|(p, _)| *p).collect();
        let Some(&centre) = positions.last() else {
            area.views.clear();
            return;
        };
        area.views = if every {
            (0..positions.len()).collect()
        } else if area.active.boxes.is_empty() {
            area_views(&positions, centre, area.registry.params().radius)
        } else {
            let cameras: Vec<_> = self.views.iter().map(|v| v.camera).collect();
            views_seeing(&cameras, &area.active.boxes, view_m)
        };
        if std::mem::take(&mut area.active.changed) {
            let stems: Vec<String> = area.active.tiles.iter().map(|k| k.stem()).collect();
            let how = match (every, area.active.boxes.is_empty()) {
                (true, _) if !engaged => "every view until a tile is left",
                (true, _) => "every view below the cap",
                (false, true) => "radius fallback",
                (false, false) => "by sight",
            };
            log::info!("area: active tiles {} ({} training views, {how})", stems.join(", "), area.views.len());
        }
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
            parking_off: area.parking.off,
            frozen_rows: area.mask.rows,
            left_splats: area.registry.left_resident_splats(centre),
            active_tiles: area.active.tiles.iter().map(|k| [k.x, k.z]).collect(),
            active_splats: area.mask.active_splats,
            held_untrained_rows: area.mask.untrained_rows,
            at_cap: area.at_cap,
            tile_splats: tile_splats(&area.registry),
            tile_evictions: area.budget.evictions.iter().map(|(k, &n)| [k.x, k.z, n as i32]).collect(),
            tile_cap: self.config.area_tile_cap(),
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
        let mut parked = Vec::new();
        for key in keys {
            match self.full_model_rows(key, coeffs) {
                Ok(r) if r.is_empty() => {}
                Ok(r) => parked.push(r.baked()),
                Err(e) => log::warn!("area: eval leaves out parked tile {}: {e}", key.stem()),
            }
        }
        let (tiles, parked) = (parked.len(), SplatRows::concat(&parked));
        let rows = parked.len();
        // One part for all parked tiles: a kernel joining one buffer per tile exceeds Metal's 31 buffer bindings.
        let mut parts = vec![resident.bake_min_scale()];
        if rows > 0 {
            parts.push(parked.to_splats_raw(&device));
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
