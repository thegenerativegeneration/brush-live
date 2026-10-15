//! Restoring: a parked tile comes back into the trainer when the capture returns within the radius, when enough of a
//! new keyframe's seeds land in it nearby, or when resident rows drift into it. Its rows move by the pose corrections
//! their seeding views had while it was parked; a tile whose views disagree about where it is gets dropped and
//! retrains from its keyframes. A restore stays within the splat budget and starts a junction refresh. (Rows that
//! drift into a parked tile far from the walker join its file instead: [`super::area_park`].)

use super::LiveModel;
use crate::area::{TileKey, apply_moves, junction_views, move_spread, remove_tile_files};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::rows::SplatRows;
use burn::tensor::s;
use glam::Vec3;
use std::collections::BTreeSet;
use std::path::Path;

/// How far away a junction view may see a restored tile's box, in metres.
const JUNCTION_FAR_M: f32 = 100.0;

/// A restored tile keeps its rows' eviction importance unless it was parked for more than this many seen windows;
/// then the importance is `+inf` (unscored) until the next refine scores it.
const STALE_WINDOWS: u32 = 10;

/// A deferral streak notes its shortfall at its first deferral and every this many after, so a tile larger than the
/// room eviction keeps free still gets room eventually, without evicting the live area at every refine.
pub(super) const DEFER_RENOTE_EVERY: u32 = 10;

/// Why parked tiles come back; the log names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestoreReason {
    /// The tile lies within the radius of the newest keyframe.
    Distance,
    /// Enough of the newest keyframe's seeds land in it within the radius plus `area_far_seed_m`.
    Seeds,
    /// Resident rows lie in it (drifted across a border).
    Rows,
}

/// What [`LiveModel::restore_one`] did with a tile.
enum Restore {
    Restored { splats: u32, spread: f32 },
    Deferred,
    Dropped,
}

impl std::fmt::Display for RestoreReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Distance => "distance",
            Self::Seeds => "seeds",
            Self::Rows => "rows",
        })
    }
}

impl LiveModel {
    /// Brings the parked tiles among `keys` back into the trainer, nearest to `centre` first ([`Self::restore_one`]);
    /// keys that are not parked are skipped. Logs one line per tile and one for the whole call, whose time (GPU
    /// synced before and after) includes reads, drops, parking for room and row appends. When a tile came back, the
    /// junction refresh starts over.
    pub(super) async fn restore_tiles(
        &mut self,
        mut splats: Splats,
        keys: &[TileKey],
        reason: RestoreReason,
        centre: Vec3,
    ) -> Splats {
        let (Some(area), Some(dir)) = (self.area.as_ref(), self.tiles_dir.clone()) else {
            return splats;
        };
        let tile_size = area.registry.params().tile_size;
        let mut keys: Vec<TileKey> = keys
            .iter()
            .copied()
            .filter(|&k| area.registry.get(k).is_some_and(|t| !t.resident))
            .collect();
        if keys.is_empty() {
            return splats;
        }
        keys.sort_by(|a, b| {
            a.footprint_distance(centre, tile_size)
                .total_cmp(&b.footprint_distance(centre, tile_size))
                .then(a.cmp(b))
        });
        keys.dedup();
        sync(&splats).await;
        let t = web_time::Instant::now();
        let (mut tiles, mut restored) = (0u32, 0u32);
        for key in keys {
            let tile_t = web_time::Instant::now();
            let outcome;
            (splats, outcome) = self.restore_one(splats, &dir, key, centre).await;
            if let Restore::Restored { splats: n, spread } = outcome {
                tiles += 1;
                restored += n;
                log::info!(
                    "area: restore tile {} ({n} splats, {reason}) in {:.1} ms, spread {spread:.2} m",
                    key.stem(),
                    tile_t.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        sync(&splats).await;
        let junction_steps = self.config.area_junction_steps;
        let area = self.area.as_mut().expect("area training is on");
        area.counters.tile_bytes = area.file_bytes.values().sum();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        area.counters.restore_ms += ms;
        area.counters.max_restore_ms = area.counters.max_restore_ms.max(ms);
        log::info!("area: restore event {tiles} tiles, {restored} splats in {ms:.1} ms ({reason})");
        if tiles > 0 {
            area.junction.steps_left = junction_steps;
            // A running junction starts over with other views: its decoded batches would fill the cache.
            area.junction.cache.clear();
            self.refresh_junction_views();
        }
        splats
    }

    /// Restores parked tile `key`, or defers or drops it:
    /// 1. Its recorded splat count must fit within `max_splats` next to the resident splats, with room made by
    ///    parking unseen tiles beyond the radius, then left tiles ([`crate::area::TileRegistry::plan_room`]);
    ///    otherwise the restore is deferred before the file is read and before anything parks (the first deferral of
    ///    a streak raises the next refine's eviction).
    /// 2. A file that cannot be read, holds another SH degree, or whose seeding views carry the tile more than
    ///    `area_restore_inconsistency_m` apart is dropped ([`Self::drop_parked`]).
    /// 3. The planned tiles park; if that fell short (parking off, a skipped park), the restore is deferred.
    /// 4. The rows, moved by their views' pose corrections since the park, are appended (zero Adam moments; their
    ///    floor, age and importance kept, the importance unscored after more than `STALE_WINDOWS` seen windows), the
    ///    file goes, and the rows' box joins the junction.
    async fn restore_one(&mut self, splats: Splats, dir: &Path, key: TileKey, centre: Vec3) -> (Splats, Restore) {
        let max_splats = self.config.max_splats;
        let keyframes = self.views.len() as u32;
        let registry = &self.area.as_ref().expect("area training is on").registry;
        let recorded = registry.get(key).map_or(0, |t| t.splats);
        let plan = registry.plan_room(recorded, splats.num_splats(), max_splats, centre, self.iter, keyframes);
        if !plan.restore {
            self.defer(key, plan.shortfall);
            return (splats, Restore::Deferred);
        }
        let (mut tile, moves) = match self.parked_with_moves(key) {
            Ok(read) => read,
            Err(e) => {
                log::warn!("area: drop unreadable tile {}: {e}", key.stem());
                self.drop_parked(dir, key);
                return (splats, Restore::Dropped);
            }
        };
        let coeffs = splats.sh_coeffs.dims()[1] as u32;
        if tile.rows.sh_coeffs_per_channel != coeffs {
            log::warn!(
                "area: drop tile {} of another SH degree: {} coefficients, the model has {coeffs}",
                key.stem(),
                tile.rows.sh_coeffs_per_channel
            );
            self.drop_parked(dir, key);
            return (splats, Restore::Dropped);
        }
        let spread = move_spread(&tile.rows, &moves);
        let limit = self.config.area_restore_inconsistency_m;
        if spread > limit {
            log::info!(
                "area: drop inconsistent tile {} (views disagree by {spread:.2} m, limit {limit:.2} m)",
                key.stem()
            );
            self.drop_parked(dir, key);
            return (splats, Restore::Dropped);
        }
        let splats = self.park_tiles(splats, &plan.park_first).await;
        let n = tile.rows.len() as u32;
        let over = (splats.num_splats() + n).saturating_sub(max_splats);
        if over > 0 {
            self.defer(key, over);
            return (splats, Restore::Deferred);
        }
        apply_moves(&mut tile.rows, &moves);
        let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
        let area = self.area.as_mut().expect("area training is on");
        let parked_at = area.registry.get(key).map_or(self.iter, |t| t.parked_at_step);
        if self.iter.saturating_sub(parked_at) > STALE_WINDOWS * area.registry.params().seen_window {
            tile.rows.importance.fill(f32::INFINITY);
        }
        let splats = trainer.append_rows(splats, &tile.rows);
        area.registry.mark_restored(key, keyframes, self.iter);
        remove_tile_files(dir, key);
        area.file_bytes.remove(&key);
        area.deferring.remove(&key);
        area.deferred_now.remove(&key);
        area.junction.boxes.extend(rows_box(&tile.rows));
        area.counters.restores += 1;
        (splats, Restore::Restored { splats: n, spread })
    }

    /// Defers the restore of `key`, `shortfall` splats over the budget. The first deferral of a streak is logged;
    /// it and every [`DEFER_RENOTE_EVERY`]th after it note the shortfall, so the next refine evicts for it. The ones
    /// between only count, so a tile that stays deferred does not evict the live area at every refine.
    pub(super) fn defer(&mut self, key: TileKey, shortfall: u32) {
        let area = self.area.as_mut().expect("area training is on");
        area.counters.deferrals += 1;
        area.deferred_now.insert(key);
        let streak = area.deferring.entry(key).or_insert(0);
        *streak += 1;
        if *streak == 1 {
            log::info!("area: restore of tile {} deferred, {shortfall} splats over the budget", key.stem());
        }
        if (*streak - 1).is_multiple_of(DEFER_RENOTE_EVERY) {
            let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
            trainer.note_seed_shortfall(shortfall);
        }
    }

    /// Ends the deferral streaks of tiles not deferred again since the last keyframe ended; call at the end of each
    /// keyframe. No-op without area training.
    pub(super) fn end_deferral_streaks(&mut self) {
        if let Some(area) = self.area.as_mut() {
            let now = std::mem::take(&mut area.deferred_now);
            area.deferring.retain(|k, _| now.contains(k));
        }
    }

    /// Forgets parked tile `key`: its files go, the registry drops it (so seeds may land there again) and it counts
    /// as a drop.
    pub(super) fn drop_parked(&mut self, dir: &Path, key: TileKey) {
        remove_tile_files(dir, key);
        let area = self.area.as_mut().expect("area training is on");
        area.registry.drop_tile(key);
        area.file_bytes.remove(&key);
        area.deferring.remove(&key);
        area.deferred_now.remove(&key);
        area.counters.drops += 1;
    }

    /// The junction's views: the area's views and the newest view that see one of the junction's boxes up to
    /// [`JUNCTION_FAR_M`]. No-op without area training; without boxes there are no junction views.
    pub(super) fn refresh_junction_views(&mut self) {
        let Some(area) = self.area.as_mut() else {
            return;
        };
        if area.junction.boxes.is_empty() {
            area.junction.views.clear();
            return;
        }
        let mut candidates: BTreeSet<usize> = area.views.iter().copied().collect();
        candidates.extend(self.views.len().checked_sub(1));
        let cameras: Vec<(usize, Camera)> =
            candidates.into_iter().filter_map(|i| Some((i, self.views.get(i)?.camera))).collect();
        let mut views = BTreeSet::new();
        for &(lo, hi) in &area.junction.boxes {
            views.extend(junction_views(&cameras, lo, hi, JUNCTION_FAR_M));
        }
        area.junction.views = views.into_iter().collect();
    }
}

/// Waits for the GPU work queued on `splats`, so a restore event's time holds the work it queued and none before it.
async fn sync(splats: &Splats) {
    if splats.num_splats() > 0 {
        let _ = splats.means().slice(s![0..1, ..]).into_data_async().await;
    }
}

/// The bounding box (min, max) of the rows' means; `None` without rows.
fn rows_box(rows: &SplatRows) -> Option<(Vec3, Vec3)> {
    (!rows.is_empty()).then(|| {
        (0..rows.len()).fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), i| {
            let p = rows.mean(i);
            (lo.min(p), hi.max(p))
        })
    })
}

#[cfg(test)]
#[path = "area_restore_tests.rs"]
pub(in crate::live) mod tests;

#[cfg(test)]
#[path = "area_room_tests.rs"]
mod room_tests;
