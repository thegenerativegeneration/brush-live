//! The splat budget of area training in the live model:
//! - Per tile (`area_tile_max_fraction` of `max_splats`): seeds into a full tile are dropped; at a refine the trainer
//!   gets a row budget
//!   ([`crate::area::row_budget`]) under which a full tile's rows do not densify and a tile over the cap evicts its
//!   least important rows down to it.
//! - At the total cap (`max_splats`): before a refine, the tiles the walker has left park first, then far tiles
//!   neither active nor near nor in view ([`LiveModel::park_for_room`]). While rows outside the active tiles are
//!   held, an eviction that still bites takes only rows of active tiles above their fair share (`max_splats` / active
//!   tiles), or held rows no step ever trained; when that and the parks fall short, it may take any row outside the
//!   active and near tiles, held or not, and with every resident tile active or near any row, as without area
//!   training ([`LiveModel::set_eviction_fallback`]).
//! - With `area_freeze` inactive_at_cap every row and every view trains (as without area training) while the resident
//!   rows stay below `area_freeze_at` of `max_splats`; from there the rows outside the active tiles are held and the
//!   views that see an active tile train ([`LiveModel::note_cap`]).

use super::LiveModel;
use crate::area::{
    FairShare, TileKey, camera_sees_box, far_inactive_to_park, keep_under_tile_cap, near_tiles, row_budget,
    tile_counts,
};
use crate::config::AreaFreeze;
use brush_render::gaussian_splats::Splats;
use brush_train::budget::EvictionRelax;
use glam::Vec3;

/// With `area_freeze` inactive_at_cap, held rows are released once the resident rows drop this share of
/// `max_splats` below `area_freeze_at`, so the hold does not toggle at every refine.
pub(super) const FREEZE_RELEASE: f32 = 0.05;

impl LiveModel {
    /// Whether the rows outside the active tiles are held now: once area training engaged
    /// ([`super::area_engage`]), always with `area_freeze` inactive, with inactive_at_cap while the resident rows are
    /// at the cap ([`Self::note_cap`]), never otherwise.
    pub(super) fn holds_inactive(&self) -> bool {
        match (&self.area, self.config.area_freeze) {
            (Some(a), AreaFreeze::Inactive) => a.engaged,
            (Some(a), AreaFreeze::InactiveAtCap) => a.at_cap,
            _ => false,
        }
    }

    /// With `area_freeze` inactive_at_cap: at `resident` rows from `area_freeze_at` × `max_splats` on the rows
    /// outside the active tiles are held and the views that see an active tile train, below
    /// (`area_freeze_at` − [`FREEZE_RELEASE`]) × `max_splats` every row and every view trains again. A switch marks
    /// the mask stale, recomputes the training views and the Mip floor's cameras, rebuilds both loaders and is logged;
    /// during a keyframe (its view not yet in) the keyframe does that once its view is in. No-op in other modes,
    /// without area training or before it engages ([`super::area_engage`]).
    pub(super) fn note_cap(&mut self, resident: u32) {
        if self.config.area_freeze != AreaFreeze::InactiveAtCap || !self.area_engaged() {
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
            if self.views.len() != self.view_cams.len() {
                area.loaders_stale = true;
                return;
            }
            self.refresh_area_views();
            if !self.views.is_empty() {
                self.refresh_floor_cams();
                self.rebuild_all_loaders();
            }
        }
    }

    /// Per seed (flat xyz), whether a tile has room for it under the per-tile cap, counting the registry's rows,
    /// the rows seeded since its last readback and the seeds before it ([`keep_under_tile_cap`]); the seeds turned
    /// away are counted. `None` without area training, before it engages or without a per-tile cap.
    pub(super) fn tile_cap_keep(&mut self, seed_means: &[f32]) -> Option<Vec<bool>> {
        let cap = self.config.area_tile_cap();
        let area = self.area.as_mut().filter(|a| a.engaged && cap > 0)?;
        let counts = tile_counts(&area.registry, &area.fresh);
        let mut keep = vec![true; seed_means.len() / 3];
        let cleared = keep_under_tile_cap(&counts, cap, seed_means, area.registry.params().tile_size, &mut keep);
        area.counters.seeds_over_tile_cap += cleared;
        (cleared > 0).then_some(keep)
    }

    /// Hands the trainer the row budget for the next refine from a readback of every resident row (flat xyz `means`):
    /// the per-tile cap, and while the rows outside the active tiles are held ([`Self::holds_inactive`]) the fair
    /// share an eviction may take from each active tile (otherwise the eviction is not limited). No budget before area
    /// training engages ([`super::area_engage`]); no-op without area training.
    pub(super) fn set_row_budget(&mut self, means: &[f32]) {
        let holds = self.holds_inactive();
        let cap = self.config.area_tile_cap();
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_mut()) else {
            return;
        };
        if !area.engaged {
            area.budget_keys.clear();
            trainer.set_row_budget(None);
            return;
        }
        let fair = holds.then(|| FairShare { active: &area.active, max_splats: self.config.max_splats });
        let tile_size = area.registry.params().tile_size;
        let budget = row_budget(means, tile_size, cap, fair);
        let (keys, budget) = budget.map_or((Vec::new(), None), |(k, b)| (k, Some(b)));
        area.budget_keys = keys;
        trainer.set_row_budget(budget);
    }

    /// After a refine: adds the evictions per tile the trainer reports for its row budget to the totals and logs
    /// them, and counts a widened eviction in `eviction_fallbacks`. No-op without area training.
    pub(super) fn note_budget_evictions(&mut self) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_ref()) else {
            return;
        };
        if trainer.eviction_fell_back() {
            area.counters.eviction_fallbacks += 1;
        }
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
    /// `max_splats` ([`Self::park_room_tiles`]); then tells the trainer how its eviction widens when the parks and the
    /// active tiles' fair shares fall short ([`Self::set_eviction_fallback`]). No-op without area training, before it
    /// engages ([`super::area_engage`]), or without a trainer or views.
    pub(super) async fn park_for_room(&mut self, splats: Splats) -> Splats {
        if !self.area_engaged() {
            return splats;
        }
        let Some(need) = self.room_need(splats.num_splats()) else {
            return splats;
        };
        let splats = self.park_room_tiles(splats, need).await;
        self.set_eviction_fallback();
        splats
    }

    /// What the next refine needs room for with `current` resident rows; `None` without area training, a trainer or
    /// views. See [`RoomNeed`].
    fn room_need(&self, current: u32) -> Option<RoomNeed> {
        let (Some(area), Some(trainer), Some(&(centre, _))) =
            (self.area.as_ref(), self.trainer.as_ref(), self.view_cams.last())
        else {
            return None;
        };
        let max_splats = self.config.max_splats;
        let room = max_splats.saturating_sub(current);
        let evict = trainer.pending_evictions(current).max(self.last_refine_evicted.saturating_sub(room));
        let fraction = self.config.area_held_fraction;
        let held_over = if fraction < 1.0 {
            let cap = (fraction.max(0.0) * max_splats as f32) as u32;
            area.registry.left_resident_splats(centre).saturating_sub(cap)
        } else {
            0
        };
        Some(RoomNeed { evict, held_over, centre })
    }

    /// Parks for `need` (the larger of its two parts): left tiles first, farthest first, seen or not
    /// ([`crate::area::TileRegistry::left_to_park`]); then, for the eviction and with an active set, far tiles
    /// neither active nor near nor seen by the newest camera ([`far_inactive_to_park`]). Never an active or near tile,
    /// never every row. Counted in `parks_for_room`, the left tiles the held cap needed also in `parks_for_cap`. No-op
    /// with parking off or nothing needed.
    async fn park_room_tiles(&mut self, splats: Splats, need: RoomNeed) -> Splats {
        let has_active = self.trains_active_tiles();
        let camera = self.views.last().map(|v| v.camera);
        let RoomNeed { evict, held_over, centre } = need;
        let total = evict.max(held_over);
        let Some(area) = self.area.as_mut().filter(|a| !a.parking_off && total > 0) else {
            return splats;
        };
        area.registry.note_walker(centre);
        let current = splats.num_splats();
        let keyframes = self.views.len() as u32;
        let registry = &area.registry;
        let splats_of = |k: &TileKey| registry.get(*k).map_or(0, |t| t.splats);
        let left = registry.left_to_park(total, current, centre, keyframes);
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
            let seen = |k: TileKey| {
                let (Some(cam), Some((lo, hi))) = (camera.as_ref(), registry.tile_box(k)) else {
                    return false;
                };
                camera_sees_box(cam, lo, hi, SIGHT_M)
            };
            far_inactive_to_park(registry, evict - freed, current - freed, centre, keyframes, &keep, seen)
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

    /// When the trainer's eviction is limited to the active tiles' fair shares, tells it how to widen when those
    /// quotas give fewer rows than the refine wants ([`brush_train::train::SplatTrainer::set_eviction_fallback`]),
    /// which it only knows during the refine (its own growth demand): it may then take rows of the resident tiles
    /// neither active nor near, held rows included, least important first and outside the newest keyframe's cone;
    /// with every resident tile active or near, any row not held, as without area training. Refines that widen are
    /// counted in `eviction_fallbacks` ([`Self::note_budget_evictions`]).
    fn set_eviction_fallback(&mut self) {
        let near = self.view_cams.last().map(|(p, _)| *p);
        let (Some(area), Some(trainer), Some(centre)) = (self.area.as_ref(), self.trainer.as_mut(), near) else {
            return;
        };
        if !trainer.row_budget().is_some_and(|b| b.limit_eviction) {
            return;
        }
        let mut keep = area.active.clone();
        keep.extend(near_tiles(centre, area.registry.params().tile_size));
        let open: Vec<u32> = area
            .budget_keys
            .iter()
            .enumerate()
            .filter(|&(_, k)| !keep.contains(k) && area.registry.get(*k).is_some_and(|t| t.resident))
            .map(|(g, _)| g as u32)
            .collect();
        trainer.set_eviction_fallback(if open.is_empty() { EvictionRelax::Unlimited } else { EvictionRelax::Groups(open) });
    }
}

/// What a refine needs room for ([`LiveModel::park_for_room`]).
#[derive(Clone, Copy, Debug)]
struct RoomNeed {
    /// What the refine evicts at least ([`brush_train::train::SplatTrainer::pending_evictions`]: a seed shortfall
    /// noted, or blocked growth while keyframes seed new cells), floored by the splats the previous refine evicted
    /// for the total beyond the room left below `max_splats` (a refine's own split and growth demand is not pending
    /// beforehand).
    evict: u32,
    /// The left tiles' rows beyond the held cap (`area_held_fraction` of `max_splats`).
    held_over: u32,
    /// The newest view's camera position.
    centre: Vec3,
}

/// How deep the newest camera's view reaches when room parking spares the far tiles it sees, metres.
const SIGHT_M: f32 = 100.0;

#[cfg(test)]
#[path = "area_budget_tests.rs"]
mod tests;
