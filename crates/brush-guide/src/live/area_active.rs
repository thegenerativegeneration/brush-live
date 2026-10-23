//! Training the active tiles (`area_freeze` inactive or inactive_at_cap): at each keyframe and pose update the active
//! set is ranked from the newest camera ([`crate::area::rank_active`]): the tiles within 2.5 m (at most half a tile)
//! of it, then up to `area_active_tiles` tiles it sees within `area_view_m`, largest on its image first, with
//! hysteresis. The views that see an active tile train ([`LiveModel::refresh_area_views`]; with inactive_at_cap only
//! at the cap). With inactive (and
//! inactive_at_cap at the cap, [`super::area_budget`]) keyframes seed only into active tiles and onto new ground, and
//! every other resident row is held in the trainer until it parks ([`super::area_freeze`]).

use super::LiveModel;
use super::area_freeze::Held;
use crate::area::{
    ActiveRule, Heights, SeedSort, TileKey, candidate_boxes, keep_active_seeds, rank_active, tile_heights, widen_all,
};
use crate::config::AreaFreeze;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use glam::Vec3;
use std::collections::{BTreeMap, BTreeSet};

/// The active tiles, their boxes and the set before.
#[derive(Debug, Default)]
pub(super) struct ActiveSet {
    /// With `area_freeze` inactive or inactive_at_cap: the active tiles, the one under the newest camera first.
    pub(super) tiles: Vec<TileKey>,
    /// Boxes (min, max) of the active tiles that hold rows; the training views are the views that see one.
    pub(super) boxes: Vec<(Vec3, Vec3)>,
    /// The active set before the current one, to count changes back to it.
    pub(super) previous: Vec<TileKey>,
    /// The active set changed since the training views were last logged.
    pub(super) changed: bool,
}

/// The height bands per tile, from the last readback and from the seeds since.
#[derive(Debug, Default)]
pub(super) struct TileHeights {
    /// Height band and rows per tile at the last visibility readback.
    pub(super) last: BTreeMap<TileKey, Heights>,
    /// Height band and rows per tile of the rows seeded since the last visibility readback.
    pub(super) fresh: BTreeMap<TileKey, Heights>,
}

impl LiveModel {
    /// Whether area training is on with an active set (`area_freeze` inactive or inactive_at_cap).
    pub(super) fn trains_active_tiles(&self) -> bool {
        self.area.is_some() && matches!(self.config.area.freeze, AreaFreeze::Inactive | AreaFreeze::InactiveAtCap)
    }

    /// Ranks the active set for `camera` ([`rank_active`], the last set as incumbents) over the resident tiles that
    /// are not left, with the height bands of the last readback, of the rows seeded since and of `seed_means` (flat
    /// xyz, the keyframe's seeds before they are sorted), and keeps the boxes of the active tiles that have one for
    /// the training views. A changed set (in any order) is counted, a change back to the set before the last one also
    /// as a reversal, and logged with the next view refresh. No-op unless [`Self::trains_active_tiles`].
    pub(super) fn refresh_active(&mut self, camera: &Camera, seed_means: &[f32]) {
        if !self.trains_active_tiles() {
            return;
        }
        let rule = ActiveRule {
            view_m: self.config.area.view_m,
            max: self.config.area.active_tiles,
            min_rows: self.config.area.active_min_rows,
            tile_size: self.config.area.tile_size_m,
        };
        let area = self.area.as_mut().expect("area training is on");
        let mut heights = area.heights.last.clone();
        widen_all(&mut heights, &area.heights.fresh);
        widen_all(&mut heights, &tile_heights(seed_means, rule.tile_size));
        let candidates = candidate_boxes(&area.registry, &heights, camera.position);
        let active = rank_active(camera, &candidates, rule, &area.active.tiles);
        let set = |keys: &[TileKey]| keys.iter().copied().collect::<BTreeSet<_>>();
        let (new, old) = (set(&active), set(&area.active.tiles));
        if new != old {
            area.counters.active_changes += 1;
            if new == set(&area.active.previous) {
                area.counters.active_reversals += 1;
            }
            area.active.previous = std::mem::take(&mut area.active.tiles);
            area.active.changed = true;
        }
        area.active.boxes = active.iter().filter_map(|k| candidates.get(k)).map(|c| (c.min, c.max)).collect();
        area.active.tiles = active;
    }

    /// Clears `keep` for the seeds (flat xyz) that land in a tile holding rows outside the active set
    /// ([`keep_active_seeds`]); seeds kept on new ground outside it mark the mask stale, so they are held from the
    /// keyframe's end on. No-op (nothing sorted) unless the rows outside the active set are held
    /// ([`Self::holds_inactive`]).
    pub(super) fn keep_active(&mut self, seed_means: &[f32], keep: &mut [bool]) -> SeedSort {
        if !self.holds_inactive() {
            return SeedSort::default();
        }
        let area = self.area.as_mut().expect("area training is on");
        let sort = keep_active_seeds(&area.registry, &area.heights.fresh, &area.active.tiles, seed_means, keep);
        area.mask.stale |= sort.new_ground > 0;
        sort
    }

    /// Notes the height bands of rows just seeded (flat xyz) per tile, until the next readback shows them to the
    /// registry. No-op without area training.
    pub(super) fn note_seeded(&mut self, means: &[f32]) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        let seeded = tile_heights(means, area.registry.params().tile_size);
        widen_all(&mut area.heights.fresh, &seeded);
    }

    /// Records the height bands of every resident row from a readback (flat xyz `means`) and forgets the rows
    /// seeded before it. No-op without area training.
    pub(super) fn note_heights(&mut self, means: &[f32]) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        area.heights.last = tile_heights(means, area.registry.params().tile_size);
        area.heights.fresh.clear();
    }

    /// Holds every row of `splats` outside the active tiles ([`Self::hold_inactive`]) when the active set changed
    /// since the mask was built, or the mask went stale (rows appended outside the set: restores, seeds on new
    /// ground; a pose update; inactive_at_cap reaching or leaving the cap, [`Self::note_cap`]). Called at the end of a
    /// keyframe and before each training step. No-op unless [`Self::trains_active_tiles`].
    pub(super) async fn hold_inactive_if_stale(&mut self, splats: &Splats) {
        if !self.trains_active_tiles() {
            return;
        }
        self.note_cap(splats.num_splats());
        let Some(area) = self.area.as_ref() else {
            return;
        };
        let mut active = area.active.tiles.clone();
        active.sort_unstable();
        if area.mask.stale || area.mask.built_for.as_ref() != Some(&active) {
            self.hold_inactive(splats).await;
        }
    }

    /// Holds every row of `splats` outside the active tiles, or none while inactive_at_cap is below the cap
    /// ([`Self::holds_inactive`]), and records the mask's active set, the resident tiles it holds and the rows in the
    /// active tiles.
    pub(super) async fn hold_inactive(&mut self, splats: &Splats) {
        let Some(area) = self.area.as_ref() else {
            return;
        };
        let active: BTreeSet<TileKey> = area.active.tiles.iter().copied().collect();
        let holds = self.holds_inactive();
        let held = if holds { Held::OutsideOf(active.clone()) } else { Held::In(BTreeSet::new()) };
        self.hold_rows(splats, held).await;
        let n = splats.num_splats();
        let area = self.area.as_mut().expect("area training is on");
        area.mask.keys = match holds {
            true => area.registry.resident_keys().into_iter().filter(|k| !active.contains(k)).collect(),
            false => BTreeSet::new(),
        };
        area.mask.built_for = Some(active.into_iter().collect());
        area.mask.stale = false;
        area.mask.active_splats = n.saturating_sub(area.mask.rows);
    }
}

#[cfg(test)]
#[path = "area_active_tests.rs"]
mod tests;
