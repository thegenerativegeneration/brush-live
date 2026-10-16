//! Training only the active tiles (`area_freeze` inactive): at each keyframe and pose update the active set is ranked
//! from the newest camera ([`crate::area::rank_active`]): the tile under it and the tiles it sees within
//! `area_view_m`, largest on its image first, `area_active_tiles` in all. The views that see an active tile train
//! ([`LiveModel::refresh_area_views`]), keyframes seed only into active tiles and onto new ground, and every other
//! resident row is held in the trainer until it parks ([`super::area_freeze`]).

use super::LiveModel;
use crate::area::{TileKey, candidate_boxes, keep_active_seeds, rank_active, seed_heights};
use crate::config::AreaFreeze;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use std::collections::BTreeSet;

impl LiveModel {
    /// Whether area training is on with `area_freeze` inactive.
    pub(super) fn trains_active_tiles(&self) -> bool {
        self.area.is_some() && self.config.area_freeze == AreaFreeze::Inactive
    }

    /// Ranks the active set for `camera` ([`rank_active`]) over the resident tiles that are not left, the rows
    /// seeded since the last readback and `seed_means` (flat xyz, the keyframe's seeds before they are sorted), and
    /// keeps the boxes of the active tiles that hold rows for the training views. A changed set (in any order) is
    /// logged with the next view refresh. No-op unless [`Self::trains_active_tiles`].
    pub(super) fn refresh_active(&mut self, camera: &Camera, seed_means: &[f32]) {
        if !self.trains_active_tiles() {
            return;
        }
        let (view_m, max) = (self.config.area_view_m, self.config.area_active_tiles);
        let area = self.area.as_mut().expect("area training is on");
        let size = area.registry.params().tile_size;
        let mut fresh = area.fresh.clone();
        for (key, (lo, hi)) in seed_heights(seed_means, size) {
            let r = fresh.entry(key).or_insert((lo, hi));
            *r = (r.0.min(lo), r.1.max(hi));
        }
        let boxes = candidate_boxes(&area.registry, &fresh, camera.position);
        let active = rank_active(camera, &boxes, view_m, max, size);
        let set = |keys: &[TileKey]| keys.iter().copied().collect::<BTreeSet<_>>();
        area.active_changed |= set(&active) != set(&area.active);
        area.active_boxes = active.iter().filter_map(|k| boxes.get(k).copied()).collect();
        area.active = active;
    }

    /// Clears `keep` for the seeds (flat xyz) that land in a tile holding rows outside the active set
    /// ([`keep_active_seeds`]); seeds kept on new ground outside it mark the mask stale, so they are held from the
    /// keyframe's end on. Logs what it dropped. No-op unless [`Self::trains_active_tiles`].
    pub(super) fn keep_active(&mut self, seed_means: &[f32], keep: &mut [bool]) {
        if !self.trains_active_tiles() {
            return;
        }
        let area = self.area.as_mut().expect("area training is on");
        let sort = keep_active_seeds(&area.registry, &area.fresh, &area.active, seed_means, keep);
        if sort.dropped > 0 {
            log::info!("area: dropped {} seeds outside the active tiles", sort.dropped);
        }
        area.mask_stale |= sort.new_ground > 0;
    }

    /// Notes the heights of rows just seeded (flat xyz) per tile, until the next readback shows them to the
    /// registry. No-op without area training.
    pub(super) fn note_seeded(&mut self, means: &[f32]) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        for (key, (lo, hi)) in seed_heights(means, area.registry.params().tile_size) {
            let r = area.fresh.entry(key).or_insert((lo, hi));
            *r = (r.0.min(lo), r.1.max(hi));
        }
    }

    /// At the end of a keyframe: holds every row of `splats` outside the active tiles when the active set changed
    /// since the mask was built, or rows outside it were appended since (restores, seeds on new ground). No-op
    /// unless [`Self::trains_active_tiles`].
    pub(super) async fn hold_inactive_at_keyframe(&mut self, splats: &Splats) {
        let Some(area) = self.area.as_ref().filter(|_| self.trains_active_tiles()) else {
            return;
        };
        let mut active = area.active.clone();
        active.sort_unstable();
        if area.mask_stale || area.mask_active.as_ref() != Some(&active) {
            self.hold_inactive(splats).await;
        }
    }

    /// Holds every row of `splats` outside the active tiles and records the mask's active set, the resident tiles
    /// it holds and the rows in the active tiles.
    pub(super) async fn hold_inactive(&mut self, splats: &Splats) {
        let Some(area) = self.area.as_ref() else {
            return;
        };
        let active: BTreeSet<TileKey> = area.active.iter().copied().collect();
        self.hold_rows(splats, active.clone(), true).await;
        let n = splats.num_splats();
        let area = self.area.as_mut().expect("area training is on");
        area.frozen_keys =
            area.registry.resident_keys().into_iter().filter(|k| !active.contains(k)).collect();
        area.mask_active = Some(active.into_iter().collect());
        area.mask_stale = false;
        area.active_splats = n.saturating_sub(area.frozen_rows);
    }
}

#[cfg(test)]
#[path = "area_active_tests.rs"]
mod tests;
