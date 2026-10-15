//! Restoring: a parked tile comes back into the trainer when the capture returns within the radius, when enough of a
//! new keyframe's seeds land in it nearby, or when resident rows drift into it. Its rows move by the pose corrections
//! their seeding views had while it was parked; a tile whose views disagree about where it is gets dropped and
//! retrains from its keyframes. A restore stays within the splat budget and starts a junction refresh.

use super::LiveModel;
use crate::area::{TileKey, apply_moves, junction_views, move_spread, read_tile, remove_tile_files, slot_moves};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::rows::SplatRows;
use glam::Vec3;
use std::collections::BTreeSet;
use std::path::Path;

/// How far away a junction view may see a restored tile's box, in metres.
const JUNCTION_FAR_M: f32 = 100.0;

/// A restored tile keeps its rows' eviction importance unless it was parked for more than this many seen windows;
/// then the importance is `+inf` (unscored) until the next refine scores it.
const STALE_WINDOWS: u32 = 10;

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
    /// Brings the parked tiles among `keys` back into the trainer, nearest to `centre` first; keys that are not
    /// parked are skipped. Per tile: its file is read (a tile that cannot be read, or holds another SH degree, is
    /// dropped); the pose corrections of its seeding views since the park are worked out, and a tile those views
    /// carry more than `area_restore_inconsistency_m` apart is dropped (file removed, registry entry gone, so its
    /// area reseeds). Room within `max_splats` comes from parking unseen tiles beyond the radius, farthest first;
    /// without enough room the restore is deferred (the tile stays parked and the shortfall raises the next refine's
    /// eviction). Otherwise the moved rows are appended (zero Adam moments, their floor, age and importance kept; the
    /// importance is reset to unscored after a long park), the file removed, and the rows' box joins the junction.
    /// Logs one line per tile and one for the whole call, whose time includes reads, drops and parking for room.
    /// When a tile came back, the junction refresh starts over.
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
        let t = web_time::Instant::now();
        let (mut tiles, mut restored) = (0u32, 0u32);
        for key in keys {
            let tile_t = web_time::Instant::now();
            let mut tile = match read_tile(&dir, key) {
                Ok(tile) => tile,
                Err(e) => {
                    log::warn!("area: drop unreadable tile {}: {e}", key.stem());
                    self.drop_parked(&dir, key);
                    continue;
                }
            };
            let coeffs = splats.sh_coeffs.dims()[1] as u32;
            if tile.rows.sh_coeffs_per_channel != coeffs {
                log::warn!(
                    "area: drop tile {}: {} SH coefficients, the model has {coeffs}",
                    key.stem(),
                    tile.rows.sh_coeffs_per_channel
                );
                self.drop_parked(&dir, key);
                continue;
            }
            let current = |slot: u32| self.views.get(slot as usize).map(|v| (v.camera.position, v.camera.rotation));
            let moves = slot_moves(&tile.slot_poses, &current);
            let spread = move_spread(&tile.rows, &moves);
            let limit = self.config.area_restore_inconsistency_m;
            if spread > limit {
                log::info!(
                    "area: drop inconsistent tile {} (views disagree by {spread:.2} m, limit {limit:.2} m)",
                    key.stem()
                );
                self.drop_parked(&dir, key);
                continue;
            }
            let n = tile.rows.len() as u32;
            let max_splats = self.config.max_splats;
            let keyframes = self.views.len() as u32;
            let registry = &self.area.as_ref().expect("area training is on").registry;
            let plan = registry.plan_room(n, splats.num_splats(), max_splats, centre, self.iter, keyframes);
            splats = self.park_tiles(splats, &plan.park_first).await;
            // Parking can fall short of the plan (parking off, a skipped park): the budget decides.
            let over = (splats.num_splats() + n).saturating_sub(max_splats);
            let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
            let area = self.area.as_mut().expect("area training is on");
            if !plan.restore || over > 0 {
                let shortfall = plan.shortfall.max(over);
                trainer.note_seed_shortfall(shortfall);
                area.counters.deferrals += 1;
                log::info!("area: restore of tile {} deferred, {shortfall} splats over the budget", key.stem());
                continue;
            }
            apply_moves(&mut tile.rows, &moves);
            let parked_at = area.registry.get(key).map_or(self.iter, |t| t.parked_at_step);
            if self.iter.saturating_sub(parked_at) > STALE_WINDOWS * area.registry.params().seen_window {
                tile.rows.importance.fill(f32::INFINITY);
            }
            splats = trainer.append_rows(splats, &tile.rows);
            area.registry.mark_restored(key, keyframes, self.iter);
            remove_tile_files(&dir, key);
            area.file_bytes.remove(&key);
            area.junction.boxes.extend(rows_box(&tile.rows));
            area.counters.restores += 1;
            tiles += 1;
            restored += n;
            log::info!(
                "area: restore tile {} ({n} splats, {reason}) in {:.1} ms, spread {spread:.2} m",
                key.stem(),
                tile_t.elapsed().as_secs_f64() * 1e3
            );
        }
        let junction_steps = self.config.area_junction_steps;
        let area = self.area.as_mut().expect("area training is on");
        area.counters.tile_bytes = area.file_bytes.values().sum();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        area.counters.restore_ms += ms;
        area.counters.max_restore_ms = area.counters.max_restore_ms.max(ms);
        log::info!("area: restore event {tiles} tiles, {restored} splats in {ms:.1} ms ({reason})");
        if tiles > 0 {
            area.junction.steps_left = junction_steps;
            self.refresh_junction_views();
        }
        splats
    }

    /// Forgets parked tile `key`: its files go, the registry drops it (so seeds may land there again) and it counts
    /// as a drop.
    fn drop_parked(&mut self, dir: &Path, key: TileKey) {
        remove_tile_files(dir, key);
        let area = self.area.as_mut().expect("area training is on");
        area.registry.drop_tile(key);
        area.file_bytes.remove(&key);
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
mod tests;
