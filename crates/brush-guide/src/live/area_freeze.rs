//! Holding rows still: with `area_freeze` inactive every resident row outside the active tiles
//! ([`super::area_active`]), with inactive_at_cap the same once the resident rows reach the cap
//! ([`super::area_budget`]), with left the rows of the resident tiles the walker has left
//! ([`crate::area::TileRegistry::left_keys`]: visited, now beyond radius + hysteresis), which views near the walker may
//! still see. Held rows stay in the trainer ([`brush_train::train::SplatTrainer::set_frozen_rows`]): no step trains
//! them, refine neither decays, re-floors nor densifies them, and they are no eviction victims unless no step ever
//! trained them, so they stay as they are until they park, the walker returns or their tile becomes active. Rows
//! appended later (seeds, restores) train until the mask is next built. The left tiles' rows keep at most
//! `area_held_fraction` of the splat budget: past it, the farthest left tiles park before a refine, even in view
//! ([`LiveModel::park_for_room`]). With `area_freeze` none nothing is held (left rows train until they park) and the
//! cap still counts the left tiles' rows.

use super::LiveModel;
use crate::area::{TileKey, rows_in};
use crate::config::AreaFreeze;
use crate::readback::read_f32;
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Bool, Tensor, TensorData};
use glam::Vec3;
use std::collections::BTreeSet;

/// The mask that holds rows still, and what it was built for.
#[derive(Debug, Default)]
pub(super) struct HeldMask {
    /// The resident tiles whose rows the trainer holds: the left ones, or with `area_freeze` inactive every
    /// resident tile not active.
    pub(super) keys: BTreeSet<TileKey>,
    /// Rows the trainer held when the mask was last built.
    pub(super) rows: u32,
    /// Held rows no step ever trained, when the mask was last built.
    pub(super) untrained_rows: u32,
    /// The active tiles the mask was last built for (sorted); `None` before any mask with `area_freeze` inactive.
    pub(super) built_for: Option<Vec<TileKey>>,
    /// Rows outside the active tiles were appended since the mask was built (restores, seeds on new ground).
    pub(super) stale: bool,
    /// Rows in the active tiles when the mask was last built.
    pub(super) active_splats: u32,
}

impl LiveModel {
    /// After a refine and its parks (the refine dropped the trainer's mask, and parks and restores moved rows): marks
    /// the tiles around the newest view as visited and holds the rows outside the active tiles (`area_freeze`
    /// inactive, and inactive_at_cap at the cap, [`Self::note_cap`]) or of the left resident tiles (left). Without
    /// area training nothing happens; with `area_freeze` none the tiles are marked and nothing is held.
    pub(super) async fn hold_rows_after_refine(&mut self, splats: &Splats) {
        let Some(centre) = self.view_cams.last().map(|(p, _)| *p) else {
            return;
        };
        let Some(keys) = self.left_resident_keys(centre) else {
            return;
        };
        match self.config.area.freeze {
            AreaFreeze::Inactive | AreaFreeze::InactiveAtCap => {
                self.note_cap(splats.num_splats());
                self.hold_inactive(splats).await;
            }
            AreaFreeze::Left => self.hold_rows(splats, Held::In(keys)).await,
            AreaFreeze::Off => {}
        }
    }

    /// At a keyframe whose camera stands at `centre`, before its restores and seeds: marks the tiles around it as
    /// visited and, when the left resident tiles differ from the resident ones the mask holds, holds the new set (the
    /// walker left a tile, or came back to one). A held tile parked since (its rows left with it) is no change.
    /// Without area training nothing happens; unless `area_freeze` is left the tiles are marked and nothing is held
    /// here (inactive holds at the keyframe's end, [`Self::hold_inactive_if_stale`]).
    pub(super) async fn hold_left_rows_at_keyframe(&mut self, splats: &Splats, centre: Vec3) {
        let Some(keys) = self.left_resident_keys(centre) else {
            return;
        };
        if self.config.area.freeze != AreaFreeze::Left {
            return;
        }
        let changed = self.area.as_ref().is_some_and(|a| {
            let resident = |k: &&TileKey| a.registry.get(**k).is_some_and(|t| t.resident);
            !a.mask.keys.iter().filter(resident).eq(keys.iter())
        });
        if changed {
            self.hold_rows(splats, Held::In(keys)).await;
        }
    }

    /// Notes the walker at `centre` and returns the resident tiles it has left; `None` without area training.
    fn left_resident_keys(&mut self, centre: Vec3) -> Option<BTreeSet<TileKey>> {
        let registry = &mut self.area.as_mut()?.registry;
        registry.note_walker(centre);
        let mut keys = registry.left_keys(centre);
        keys.retain(|&k| registry.get(k).is_some_and(|t| t.resident));
        Some(keys)
    }

    /// Holds the rows of `splats` that `held` names in the trainer (one means readback; none for no tiles in
    /// [`Held::In`]), and records the tiles, the held row count and how many of them no step ever trained.
    pub(super) async fn hold_rows(&mut self, splats: &Splats, held: Held) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_mut()) else {
            return;
        };
        let (keys, outside) = match held {
            Held::In(keys) => (keys, false),
            Held::OutsideOf(keys) => (keys, true),
        };
        let mut count = 0;
        let mut mask = None;
        if (outside || !keys.is_empty()) && splats.num_splats() > 0 {
            let t = web_time::Instant::now();
            let means: Vec<f32> = read_f32(splats.means()).await;
            let list: Vec<TileKey> = keys.iter().copied().collect();
            let flags: Vec<bool> =
                rows_in(&list, &means, area.registry.params().tile_size).into_iter().map(|f| f != outside).collect();
            count = flags.iter().filter(|&&f| f).count() as u32;
            let n = flags.len();
            mask = (count > 0).then(|| Tensor::<1, Bool>::from_data(TensorData::new(flags, [n]), &splats.device()));
            log::debug!(
                target: crate::timing::TARGET,
                "hold at iter {}: {count} rows {} {} tiles, {:.1} ms",
                self.iter,
                if outside { "outside" } else { "in" },
                list.len(),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        trainer.set_frozen_rows(mask);
        let untrained = match trainer.held_untrained_rows() {
            Some(c) => c.into_data_async().await.ok().and_then(|d| d.convert::<i64>().try_to_vec::<i64>().ok()),
            None => None,
        };
        area.mask.untrained_rows = untrained.and_then(|v| v.first().copied()).map_or(0, |v| v.max(0) as u32);
        area.mask.keys = keys;
        area.mask.rows = count;
    }
}

/// The rows [`LiveModel::hold_rows`] holds.
pub(super) enum Held {
    /// The rows lying in these tiles; no tiles hold nothing.
    In(BTreeSet<TileKey>),
    /// The rows lying in none of these tiles; no tiles (nothing active) hold every row.
    OutsideOf(BTreeSet<TileKey>),
}

#[cfg(test)]
#[path = "area_freeze_tests.rs"]
pub(in crate::live) mod tests;

#[cfg(test)]
#[path = "area_cap_tests.rs"]
mod cap_tests;
