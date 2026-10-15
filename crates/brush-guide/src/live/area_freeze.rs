//! Holding left tiles still: a resident tile the walker has left ([`crate::area::TileRegistry::left_keys`]: visited,
//! now beyond radius + hysteresis) can stay resident for a while, since views near the walker may still see it. Its
//! rows are held in the trainer ([`brush_train::train::SplatTrainer::set_frozen_rows`]): no step trains them, refine
//! neither decays, re-floors, densifies nor evicts them, so the area stays as the walker left it until it parks or the
//! walker returns. Rows appended later (seeds, restores) train until the mask is next built. Held rows keep at most
//! `area_held_fraction` of the splat budget: past it, the farthest held tiles park after a refine, even in view. With
//! `area_hold_left` off nothing is held (left rows train until they park) and the cap counts the left tiles' rows.

use super::LiveModel;
use crate::area::{TileKey, rows_in};
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Bool, Tensor, TensorData};
use glam::Vec3;
use std::collections::BTreeSet;

impl LiveModel {
    /// After a refine and its parks (the refine dropped the trainer's mask, and parks and restores moved rows): marks
    /// the tiles around the newest view as visited and holds the rows of the left resident tiles. Without area
    /// training nothing happens; with `area_hold_left` off the tiles are marked and nothing is held.
    pub(super) async fn hold_left_rows_after_refine(&mut self, splats: &Splats) {
        let Some(centre) = self.view_cams.last().map(|(p, _)| *p) else {
            return;
        };
        if let Some(keys) = self.left_resident_keys(centre)
            && self.config.area_hold_left
        {
            self.hold_rows(splats, keys).await;
        }
    }

    /// At a keyframe whose camera stands at `centre`, before its restores and seeds: marks the tiles around it as
    /// visited and, when the left resident tiles differ from the resident ones the mask holds, holds the new set (the
    /// walker left a tile, or came back to one). A held tile parked since (its rows left with it) is no change.
    /// Without area training nothing happens; with `area_hold_left` off the tiles are marked and nothing is held.
    pub(super) async fn hold_left_rows_at_keyframe(&mut self, splats: &Splats, centre: Vec3) {
        let Some(keys) = self.left_resident_keys(centre) else {
            return;
        };
        if !self.config.area_hold_left {
            return;
        }
        let changed = self.area.as_ref().is_some_and(|a| {
            let resident = |k: &&TileKey| a.registry.get(**k).is_some_and(|t| t.resident);
            !a.frozen_keys.iter().filter(resident).eq(keys.iter())
        });
        if changed {
            self.hold_rows(splats, keys).await;
        }
    }

    /// Right before a refine that will evict (a seed shortfall noted, or blocked growth backlog while keyframes seed
    /// new cells, [`brush_train::train::SplatTrainer::pending_evictions`]): parks resident left tiles, farthest first,
    /// holding as many splats as the eviction would free ([`crate::area::TileRegistry::left_to_park`]), so the
    /// eviction takes from the area the walker left instead of the live one. The splats the previous refine evicted,
    /// less the room left below `max_splats`, are a floor for that count: a refine's own split and growth demand is
    /// not pending beforehand, and at the cap the next refine usually evicts about as many again, while with room
    /// to spare it evicts nothing. Needs the registry as the visibility readback just left it. No-op without area
    /// training, with parking off, or when nothing would be evicted.
    pub(super) async fn park_left_before_eviction(&mut self, splats: Splats) -> Splats {
        let (Some(area), Some(trainer), Some(&(centre, _))) =
            (self.area.as_ref(), self.trainer.as_ref(), self.view_cams.last())
        else {
            return splats;
        };
        let current = splats.num_splats();
        let room = self.train_config.max_splats.saturating_sub(current);
        let evict = trainer.pending_evictions(current).max(self.last_refine_evicted.saturating_sub(room));
        if area.parking_off || evict == 0 {
            return splats;
        }
        let keys = area.registry.left_to_park(evict, current, centre, self.views.len() as u32);
        if keys.is_empty() {
            return splats;
        }
        log::info!("area: park {} left tiles before evicting {evict} splats", keys.len());
        self.park_tiles(splats, &keys).await
    }

    /// After a refine's parks, before the mask is built: while the resident left tiles' splats (the rows held still,
    /// or with `area_hold_left` off the rows still training there) exceed `area_held_fraction` of `max_splats`, parks
    /// left tiles farthest first, seen or not ([`crate::area::TileRegistry::held_over_cap`]), so the live area keeps
    /// the rest of the budget. A held tile parks as it was held; a parked left tile is not restored until the walker
    /// returns. Counted in `parks_for_cap` and logged once
    /// per refine that parks. No-op without area training, with parking off, or with a fraction of 1.0 or more.
    pub(super) async fn park_held_over_cap(&mut self, splats: Splats) -> Splats {
        let fraction = self.config.area_held_fraction;
        let max_splats = self.train_config.max_splats;
        let (Some(area), Some(&(centre, _))) = (self.area.as_mut(), self.view_cams.last()) else {
            return splats;
        };
        if area.parking_off || !(fraction < 1.0) {
            return splats;
        }
        area.registry.note_walker(centre);
        let cap = (fraction.max(0.0) * max_splats as f32) as u32;
        let before = splats.num_splats();
        let keys = area.registry.held_over_cap(cap, before, centre, self.views.len() as u32);
        if keys.is_empty() {
            return splats;
        }
        let parks = area.counters.parks;
        let splats = self.park_tiles(splats, &keys).await;
        let area = self.area.as_mut().expect("area training is on");
        let tiles = area.counters.parks - parks;
        if tiles > 0 {
            area.counters.parks_for_cap += tiles;
            log::info!(
                "area: park held over cap {tiles} tiles, {} splats",
                before.saturating_sub(splats.num_splats())
            );
        }
        splats
    }

    /// Notes the walker at `centre` and returns the resident tiles it has left; `None` without area training.
    fn left_resident_keys(&mut self, centre: Vec3) -> Option<BTreeSet<TileKey>> {
        let registry = &mut self.area.as_mut()?.registry;
        registry.note_walker(centre);
        let mut keys = registry.left_keys(centre);
        keys.retain(|&k| registry.get(k).is_some_and(|t| t.resident));
        Some(keys)
    }

    /// Holds the rows of `splats` lying in `keys` in the trainer (one means readback; none when `keys` is empty) and
    /// records the tiles and the row count.
    async fn hold_rows(&mut self, splats: &Splats, keys: BTreeSet<TileKey>) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_mut()) else {
            return;
        };
        let mut held = 0;
        let mut mask = None;
        if !keys.is_empty() && splats.num_splats() > 0 {
            let t = web_time::Instant::now();
            let means: Vec<f32> = splats
                .means()
                .into_data_async()
                .await
                .expect("means readback")
                .convert::<f32>()
                .try_into_vec()
                .expect("f32 means");
            let keys: Vec<TileKey> = keys.iter().copied().collect();
            let flags = rows_in(&keys, &means, area.registry.params().tile_size);
            held = flags.iter().filter(|&&f| f).count() as u32;
            let n = flags.len();
            mask = (held > 0).then(|| Tensor::<1, Bool>::from_data(TensorData::new(flags, [n]), &splats.device()));
            log::debug!(
                target: crate::timing::TARGET,
                "hold at iter {}: {held} rows in {} left tiles, {:.1} ms",
                self.iter,
                keys.len(),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        trainer.set_frozen_rows(mask);
        area.frozen_keys = keys;
        area.frozen_rows = held;
    }
}

#[cfg(test)]
#[path = "area_freeze_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "area_cap_tests.rs"]
mod cap_tests;
