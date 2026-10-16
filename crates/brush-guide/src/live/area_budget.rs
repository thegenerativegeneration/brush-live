//! The splat budget of area training in the live model:
//! - Per tile (`area_tile_max_splats`): seeds into a full tile are dropped; at a refine the trainer gets a row budget
//!   ([`crate::area::row_budget`]) under which a full tile's rows do not densify and a tile over the cap evicts its
//!   least important rows down to it.
//! - At the total cap (`max_splats`): before a refine, the tiles the walker has left park first, then the tiles
//!   neither active nor near beyond the radius ([`LiveModel::park_for_room`]); an eviction that still bites takes only
//!   rows of active tiles above their fair share (`max_splats` / active tiles), or held rows no step ever trained.
//! - With `area_freeze` inactive_at_cap the rows outside the active tiles train while the resident rows stay below
//!   `area_freeze_at` of `max_splats` and are held from there ([`LiveModel::note_cap`]).

use super::LiveModel;
use crate::area::{FairShare, TileKey, far_inactive_to_park, keep_under_tile_cap, near_tiles, row_budget, tile_counts};
use crate::config::AreaFreeze;
use brush_render::gaussian_splats::Splats;

/// With `area_freeze` inactive_at_cap, held rows are released once the resident rows drop this share of
/// `max_splats` below `area_freeze_at`, so the hold does not toggle at every refine.
pub(super) const FREEZE_RELEASE: f32 = 0.05;

impl LiveModel {
    /// Whether the rows outside the active tiles are held now: always with `area_freeze` inactive, with
    /// inactive_at_cap while the resident rows are at the cap ([`Self::note_cap`]), never otherwise.
    pub(super) fn holds_inactive(&self) -> bool {
        match (&self.area, self.config.area_freeze) {
            (Some(_), AreaFreeze::Inactive) => true,
            (Some(a), AreaFreeze::InactiveAtCap) => a.at_cap,
            _ => false,
        }
    }

    /// With `area_freeze` inactive_at_cap: at `resident` rows from `area_freeze_at` × `max_splats` on the rows
    /// outside the active tiles are held, below (`area_freeze_at` − [`FREEZE_RELEASE`]) × `max_splats` they train
    /// again. A switch marks the mask stale and is logged. No-op in other modes or without area training.
    pub(super) fn note_cap(&mut self, resident: u32) {
        if self.config.area_freeze != AreaFreeze::InactiveAtCap {
            return;
        }
        let (max, at) = (self.config.max_splats as f32, self.config.area_freeze_at);
        let Some(area) = self.area.as_mut() else {
            return;
        };
        let bound = if area.at_cap { (at - FREEZE_RELEASE) * max } else { at * max };
        let at_cap = resident as f32 >= bound;
        if at_cap != area.at_cap {
            area.at_cap = at_cap;
            area.mask_stale = true;
            let what = if at_cap { "held" } else { "train again" };
            log::info!("area: rows outside the active tiles {what} at {resident} of {max} splats");
        }
    }

    /// Per seed (flat xyz), whether a tile has room for it under `area_tile_max_splats`, counting the registry's rows,
    /// the rows seeded since its last readback and the seeds before it ([`keep_under_tile_cap`]); the seeds turned
    /// away are counted. `None` without area training or a per-tile cap.
    pub(super) fn tile_cap_keep(&mut self, seed_means: &[f32]) -> Option<Vec<bool>> {
        let cap = self.config.area_tile_max_splats;
        let area = self.area.as_mut().filter(|_| cap > 0)?;
        let counts = tile_counts(&area.registry, &area.fresh);
        let mut keep = vec![true; seed_means.len() / 3];
        let cleared = keep_under_tile_cap(&counts, cap, seed_means, area.registry.params().tile_size, &mut keep);
        area.counters.seeds_over_tile_cap += cleared;
        (cleared > 0).then_some(keep)
    }

    /// Hands the trainer the row budget for the next refine from a readback of every resident row (flat xyz `means`):
    /// the per-tile cap, and with an active set (`area_freeze` inactive or inactive_at_cap) the fair share an
    /// eviction may take from each active tile. No-op without area training.
    pub(super) fn set_row_budget(&mut self, means: &[f32]) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_mut()) else {
            return;
        };
        let has_active = matches!(self.config.area_freeze, AreaFreeze::Inactive | AreaFreeze::InactiveAtCap);
        let fair = has_active.then(|| FairShare { active: &area.active, max_splats: self.config.max_splats });
        let tile_size = area.registry.params().tile_size;
        let budget = row_budget(means, tile_size, self.config.area_tile_max_splats, fair);
        let (keys, budget) = budget.map_or((Vec::new(), None), |(k, b)| (k, Some(b)));
        area.budget_keys = keys;
        trainer.set_row_budget(budget);
    }

    /// After a refine: adds the evictions per tile the trainer reports for its row budget to the totals and logs
    /// them. No-op without area training.
    pub(super) fn note_budget_evictions(&mut self) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_ref()) else {
            return;
        };
        let mut parts = Vec::new();
        for (key, &n) in area.budget_keys.iter().zip(trainer.budget_evictions()) {
            if n > 0 {
                *area.tile_evictions.entry(*key).or_default() += n;
                parts.push(format!("{} {n}", key.stem()));
            }
        }
        if !parts.is_empty() {
            log::info!("area: evicted per tile {}", parts.join(", "));
        }
    }

    /// Right before a refine, from the registry as the visibility readback just left it: makes the room the refine
    /// would otherwise evict for, and keeps the resident left tiles' rows within `area_held_fraction` of
    /// `max_splats`. The need is the larger of
    /// - the eviction: what the refine evicts at least ([`brush_train::train::SplatTrainer::pending_evictions`]: a
    ///   seed shortfall noted, or blocked growth while keyframes seed new cells), floored by the splats the previous
    ///   refine evicted beyond the room left below `max_splats` (a refine's own split and growth demand is not
    ///   pending beforehand);
    /// - the left tiles' rows beyond the held cap.
    ///
    /// Left tiles park first, farthest first, seen or not ([`crate::area::TileRegistry::left_to_park`]); then, for
    /// the eviction and with an active set, tiles neither active nor near beyond the radius
    /// ([`far_inactive_to_park`]). Never an active or near tile, never every row. Counted in `parks_for_room`, the
    /// left tiles the held cap needed also in `parks_for_cap`. No-op without area training, with parking off, or when
    /// nothing is needed.
    pub(super) async fn park_for_room(&mut self, splats: Splats) -> Splats {
        let has_active = self.trains_active_tiles();
        let (Some(area), Some(trainer), Some(&(centre, _))) =
            (self.area.as_mut(), self.trainer.as_ref(), self.view_cams.last())
        else {
            return splats;
        };
        if area.parking_off {
            return splats;
        }
        area.registry.note_walker(centre);
        let current = splats.num_splats();
        let max_splats = self.train_config.max_splats;
        let room = max_splats.saturating_sub(current);
        let evict = trainer.pending_evictions(current).max(self.last_refine_evicted.saturating_sub(room));
        let fraction = self.config.area_held_fraction;
        let held_over = if fraction < 1.0 {
            let cap = (fraction.max(0.0) * max_splats as f32) as u32;
            area.registry.left_resident_splats(centre).saturating_sub(cap)
        } else {
            0
        };
        let need = evict.max(held_over);
        if need == 0 {
            return splats;
        }
        let keyframes = self.views.len() as u32;
        let registry = &area.registry;
        let splats_of = |k: &TileKey| registry.get(*k).map_or(0, |t| t.splats);
        let left = registry.left_to_park(need, current, centre, keyframes);
        let freed: u32 = left.iter().map(splats_of).sum();
        // The left tiles the held cap needs, in parking order.
        let mut for_cap = 0u32;
        let mut covered = 0u32;
        for k in &left {
            if covered >= held_over {
                break;
            }
            covered += splats_of(k);
            for_cap += 1;
        }
        let far = if has_active && evict > freed {
            let mut keep = area.active.clone();
            keep.extend(near_tiles(centre, registry.params().tile_size));
            far_inactive_to_park(registry, evict - freed, current - freed, centre, keyframes, &keep)
        } else {
            Vec::new()
        };
        if left.is_empty() && far.is_empty() {
            return splats;
        }
        let far_splats: u32 = far.iter().map(splats_of).sum();
        log::info!(
            "area: park for room {} left tiles ({freed} splats), {} far inactive tiles ({far_splats} splats); \
             eviction {evict}, held over cap {held_over}",
            left.len(),
            far.len()
        );
        let parks = area.counters.parks;
        let keys: Vec<TileKey> = left.into_iter().chain(far).collect();
        let splats = self.park_tiles(splats, &keys).await;
        let area = self.area.as_mut().expect("area training is on");
        let parked = area.counters.parks - parks;
        area.counters.parks_for_room += parked;
        area.counters.parks_for_cap += parked.min(for_cap);
        splats
    }
}

#[cfg(test)]
#[path = "area_budget_tests.rs"]
mod tests;
