//! Parking: at each refine the live model reads back which splats the window's training views drew, and after the
//! refine the resident tiles the capture has left (beyond the radius plus hysteresis, unseen for the window) leave
//! the trainer for tile files. Parked rows are out of the trainer, so nothing trains, refines, decays or evicts them
//! until they are restored.

use super::LiveModel;
use super::area_restore::RestoreReason;
use crate::area::{
    ParkedTile, SlotPose, TileKey, apply_moves, group_rows, move_spread, read_tile, rows_in, slot_moves, write_tile,
};
use crate::readback::read_f32;
use brush_render::gaussian_splats::Splats;
use brush_train::rows::SplatRows;
use burn::tensor::Transaction;
use glam::Affine3A;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// What parking tracks: the conflicts of the last readback, the tile files and whether parking is on.
#[derive(Debug, Default)]
pub(super) struct Parking {
    /// Parked tiles that resident rows lay in at the last visibility readback; the refine's park step restores them.
    pub(super) conflicts: Vec<TileKey>,
    /// File size of every parked tile.
    pub(super) file_bytes: BTreeMap<TileKey, u64>,
    /// The last park was skipped for taking every resident row (logged once per streak).
    pub(super) skipping: bool,
    /// A tile file could not be written: no parking until the next reset.
    pub(super) off: bool,
}

impl LiveModel {
    /// Records per resident tile whether a training step since the last refine drew one of its splats (the refine
    /// record's `importance_views`, occlusion included) from one readback of the means and that count, and hands the
    /// trainer the row budget from the same means ([`Self::set_row_budget`]). No-op without area training or before a
    /// step since the last refine; call right before `refine`.
    pub(super) async fn note_visibility(&mut self, splats: &Splats) {
        let (Some(area), Some(trainer)) = (self.area.as_mut(), self.trainer.as_ref()) else {
            return;
        };
        let Some(views) = trainer.seen_since_refine() else {
            return;
        };
        let t = web_time::Instant::now();
        let mut data = Transaction::default()
            .register(splats.means())
            .register(views)
            .execute_async()
            .await
            .expect("visibility readback")
            .into_iter()
            .map(|d| d.convert::<f32>().try_into_vec::<f32>().expect("f32 visibility readback"));
        let (means, views) = (data.next().expect("means"), data.next().expect("views"));
        // A record out of step with the splats would park tiles in view: count every row as seen then.
        let seen: Vec<bool> = if views.len() * 3 == means.len() {
            views.iter().map(|&v| v > 0.0).collect()
        } else {
            log::error!(
                "area: refine record holds {} rows for {} splats; every splat counts as seen",
                views.len(),
                means.len() / 3
            );
            vec![true; means.len() / 3]
        };
        let conflicts = area.registry.note_resident(&means, &seen, self.iter);
        log::debug!(
            target: crate::timing::TARGET,
            "visibility at iter {}: {:.1} ms for {} splats",
            self.iter,
            t.elapsed().as_secs_f64() * 1e3,
            seen.len()
        );
        if conflicts != area.parking.conflicts {
            if !conflicts.is_empty() {
                let stems: Vec<String> = conflicts.iter().map(|k| k.stem()).collect();
                log::info!("area: resident splats lie in parked tiles {}", stems.join(", "));
            }
            area.parking.conflicts = conflicts;
        }
        self.note_heights(&means);
        self.set_row_budget(&means);
    }

    /// After a refine: parked tiles that resident rows drifted into come back ([`RestoreReason::Rows`]) when they lie
    /// within radius + hysteresis of the newest view; farther ones take the rows into their file
    /// ([`Self::merge_into_parked`]), so a walker's drifting splats do not pull a far area back. Then the tiles
    /// [`crate::area::TileRegistry::to_park`] names park ([`Self::park_tiles`]). No-op without area training, before
    /// it engages ([`super::area_engage`]: nothing is parked yet) or without views; with parking off every such tile
    /// comes back and nothing parks.
    pub(super) async fn park_after_refine(&mut self, splats: Splats) -> Splats {
        let Some(centre) = self.view_cams.last().map(|(p, _)| *p) else {
            return splats;
        };
        if !self.area_engaged() {
            return splats;
        }
        let Some(area) = self.area.as_mut() else {
            return splats;
        };
        let p = *area.registry.params();
        let near = |k: &TileKey| k.footprint_distance(centre, p.tile_size) <= p.radius + p.hysteresis;
        // With parking off a far tile's file cannot take the rows: the tile comes back instead.
        let (back, far): (Vec<TileKey>, Vec<TileKey>) =
            area.parking.conflicts.iter().partition(|k| near(k) || area.parking.off);
        let splats = self.restore_tiles(splats, &back, RestoreReason::Rows, centre).await;
        let splats = self.merge_into_parked(splats, &far).await;
        let Some(area) = self.area.as_mut().filter(|a| !a.parking.off) else {
            return splats;
        };
        let keys = area.registry.to_park(centre, self.iter, self.views.len() as u32);
        if keys.is_empty() {
            area.parking.skipping = false;
            return splats;
        }
        self.park_tiles(splats, &keys).await
    }

    /// Parks the resident tiles `keys`: their rows leave the trainer and go to one tile file each, with the pose of
    /// every view that seeded them. A park that would take every resident row is skipped (counted). The rows of a
    /// tile whose file cannot be written go back to the trainer (with their Adam moments) and parking stays off until
    /// the next reset. Each park logs one line per tile and one for the whole event, whose time includes the row
    /// readback. After a park the all-views loader drops views that left the area. No-op with parking off.
    pub(super) async fn park_tiles(&mut self, splats: Splats, keys: &[TileKey]) -> Splats {
        let Some(area) = self.area.as_mut().filter(|a| !a.parking.off && !keys.is_empty()) else {
            return splats;
        };
        let t = web_time::Instant::now();
        let tile_size = area.registry.params().tile_size;
        let means: Vec<f32> = read_f32(splats.means()).await;
        let take = rows_in(keys, &means, tile_size);
        let taken = take.iter().filter(|&&t| t).count();
        let skip = taken == take.len() && taken > 0;
        if skip {
            area.counters.parks_skipped += 1;
            if !area.parking.skipping {
                log::info!("area: park skipped, it would take every resident splat");
            }
        }
        area.parking.skipping = skip;
        if skip || taken == 0 {
            // With nothing taken, the refine pruned these tiles empty; the next readback drops them.
            return splats;
        }
        let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
        let (splats, rows) = trainer.take_rows(splats, &take).await;
        let rows = rows.expect("rows taken and rows kept");
        let (parked, tiles) = self.write_tiles(&rows, tile_size);
        let splats = if parked.len() < rows.len() {
            let back: Vec<usize> = (0..rows.len()).filter(|i| !parked.contains(i)).collect();
            let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
            trainer.append_rows(splats, &rows.select(&back))
        } else {
            splats
        };
        let area = self.area.as_mut().expect("area training is on");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        area.counters.note_park(ms, area.parking.file_bytes.values().sum());
        log::info!("area: park event {tiles} tiles, {} splats in {ms:.1} ms", parked.len());
        self.drop_departed_views();
        splats
    }

    /// Moves the resident rows lying in the far parked tiles among `keys` (keys not parked are skipped) into their
    /// files: per tile, the file's rows take the pose corrections since the park (as a restore would), the drifted
    /// rows join them, every slot records its current pose (so the merge resets the tile's pose baseline, spec D1),
    /// and the file is written as the next version; the tile stays parked and gets no junction. A tile whose views
    /// disagree (as on restore), whose file cannot be read or holds another SH degree is dropped
    /// ([`Self::drop_parked`]) and the drifted rows stay resident; a failed write keeps them out too and turns parking
    /// off. Rows kept out go back to the trainer. Logged per tile and as one park event. No-op when every resident row
    /// would go.
    async fn merge_into_parked(&mut self, splats: Splats, keys: &[TileKey]) -> Splats {
        let (Some(area), Some(dir)) = (self.area.as_ref(), self.tiles_dir.clone()) else {
            return splats;
        };
        let keys: Vec<TileKey> =
            keys.iter().copied().filter(|&k| area.registry.get(k).is_some_and(|t| !t.resident)).collect();
        if keys.is_empty() {
            return splats;
        }
        let t = web_time::Instant::now();
        let tile_size = area.registry.params().tile_size;
        let means: Vec<f32> = read_f32(splats.means()).await;
        let take = rows_in(&keys, &means, tile_size);
        let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
        let (splats, rows) = trainer.take_rows(splats, &take).await;
        let Some(rows) = rows else {
            return splats;
        };
        let (mut merged, mut tiles) = (BTreeSet::new(), 0u32);
        for (key, idx) in group_rows(&rows.means, tile_size) {
            let tile_t = web_time::Instant::now();
            let Some((total, version, bytes)) = self.merge_tile(&dir, key, rows.select(&idx)) else {
                continue;
            };
            log::info!(
                "area: merge {} drifted splats into parked tile {} (now {total}, v{version}) in {:.1} ms, file {bytes} B",
                idx.len(),
                key.stem(),
                tile_t.elapsed().as_secs_f64() * 1e3
            );
            merged.extend(idx);
            tiles += 1;
        }
        let splats = if merged.len() < rows.len() {
            let back: Vec<usize> = (0..rows.len()).filter(|i| !merged.contains(i)).collect();
            let trainer = self.trainer.as_mut().expect("trainer exists once splats exist");
            trainer.append_rows(splats, &rows.select(&back))
        } else {
            splats
        };
        let area = self.area.as_mut().expect("area training is on");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        area.counters.note_park(ms, area.parking.file_bytes.values().sum());
        log::info!("area: park event {tiles} tiles, {} splats in {ms:.1} ms", merged.len());
        splats
    }

    /// Writes parked tile `key` again with `new` rows added (see [`Self::merge_into_parked`]); returns its row count,
    /// version and file size, or `None` when the rows stay out.
    fn merge_tile(&mut self, dir: &Path, key: TileKey, new: SplatRows) -> Option<(u32, u32, u64)> {
        if self.area.as_ref().is_none_or(|a| a.parking.off) {
            return None;
        }
        let (tile, moves, _) =
            self.validated_parked(dir, key, new.sh_coeffs_per_channel, ", the rows drifted into it stay resident")?;
        let mut old = tile.rows;
        apply_moves(&mut old, &moves);
        let rows = SplatRows::concat(&[old, new]);
        let slot_poses = self.slot_poses(&rows);
        let area = self.area.as_mut().expect("area training is on");
        let version = area.registry.get(key)?.version + 1;
        let total = rows.len() as u32;
        match write_tile(dir, &ParkedTile { key, version, slot_poses, rows }) {
            Ok(bytes) => {
                area.registry.note_merged(key, total);
                area.parking.file_bytes.insert(key, bytes);
                Some((total, version, bytes))
            }
            Err(e) => {
                log::warn!("area: parking off until reset, tile {} cannot be written: {e}", key.stem());
                area.parking.off = true;
                None
            }
        }
    }

    /// Writes one tile file per tile of the taken `rows` and marks it parked; returns the indices (into `rows`) of
    /// the rows parked and the number of tiles they make. A tile the registry does not know, or whose file cannot be
    /// written, stays out; the first write failure turns parking off until the next reset, so the tiles after it
    /// stay out too.
    fn write_tiles(&mut self, rows: &SplatRows, tile_size: f32) -> (BTreeSet<usize>, u32) {
        let dir = self.tiles_dir.clone().expect("area training has a tiles directory");
        let (mut parked, mut tiles) = (BTreeSet::new(), 0);
        for (key, idx) in group_rows(&rows.means, tile_size) {
            if self.area.as_ref().is_none_or(|a| a.parking.off) {
                break;
            }
            let t = web_time::Instant::now();
            let tile_rows = rows.select(&idx);
            let n = tile_rows.len() as u32;
            let slot_poses = self.slot_poses(&tile_rows);
            let area = self.area.as_mut().expect("area training is on");
            let Some(version) = area.registry.get(key).map(|t| t.version + 1) else {
                log::error!("area: tile {} is not in the registry; its rows stay resident", key.stem());
                continue;
            };
            let tile = ParkedTile { key, version, slot_poses, rows: tile_rows };
            let bytes = match write_tile(&dir, &tile) {
                Ok(bytes) => bytes,
                Err(e) => {
                    log::warn!("area: parking off until reset, tile {} cannot be written: {e}", key.stem());
                    area.parking.off = true;
                    continue;
                }
            };
            area.registry.mark_parked(key, n, self.iter);
            area.parking.file_bytes.insert(key, bytes);
            area.counters.parks += 1;
            parked.extend(idx);
            tiles += 1;
            log::info!(
                "area: park tile {} ({n} splats, v{version}) in {:.1} ms, file {bytes} B; {} resident tiles, {} parked",
                key.stem(),
                t.elapsed().as_secs_f64() * 1e3,
                area.registry.resident_keys().len(),
                area.registry.parked_keys().len()
            );
        }
        (parked, tiles)
    }

    /// The pose of every view slot `rows` name as an origin, as its camera stands now.
    fn slot_poses(&self, rows: &SplatRows) -> Vec<SlotPose> {
        let slots: BTreeSet<u32> = rows.origin.iter().flatten().copied().collect();
        let poses: Vec<SlotPose> = slots
            .iter()
            .filter_map(|&slot| {
                let c = &self.views.get(slot as usize)?.camera;
                Some(SlotPose { slot, position: c.position, rotation: c.rotation })
            })
            .collect();
        debug_assert_eq!(poses.len(), slots.len(), "an origin slot names no view");
        poses
    }

    /// Parked tile `key` with its moves and their spread, or `None` after dropping it ([`Self::drop_parked`]) because
    /// its file cannot be read, holds other than `coeffs` SH coefficients per channel, or its seeding views carry it
    /// more than `area_restore_inconsistency_m` apart. `note` ends the warnings of the first two.
    pub(super) fn validated_parked(
        &mut self,
        dir: &Path,
        key: TileKey,
        coeffs: u32,
        note: &str,
    ) -> Option<(ParkedTile, Vec<(u32, Affine3A)>, f32)> {
        let (tile, moves) = match self.parked_with_moves(key) {
            Ok(read) => read,
            Err(e) => {
                log::warn!("area: drop unreadable tile {}{note}: {e}", key.stem());
                self.drop_parked(dir, key);
                return None;
            }
        };
        if tile.rows.sh_coeffs_per_channel != coeffs {
            log::warn!(
                "area: drop tile {} of another SH degree: {} coefficients, the model has {coeffs}{note}",
                key.stem(),
                tile.rows.sh_coeffs_per_channel
            );
            self.drop_parked(dir, key);
            return None;
        }
        let spread = move_spread(&tile.rows, &moves);
        let limit = self.config.area_restore_inconsistency_m;
        if spread > limit {
            log::info!(
                "area: drop inconsistent tile {} (views disagree by {spread:.2} m, limit {limit:.2} m)",
                key.stem()
            );
            self.drop_parked(dir, key);
            return None;
        }
        Some((tile, moves, spread))
    }

    /// The rows of parked tile `key` from its file, moved by the pose corrections their seeding views had since the
    /// park ([`slot_moves`]).
    pub(super) fn load_parked(&self, key: TileKey) -> std::io::Result<SplatRows> {
        let (tile, moves) = self.parked_with_moves(key)?;
        let mut rows = tile.rows;
        apply_moves(&mut rows, &moves);
        Ok(rows)
    }

    /// Parked tile `key` as its file holds it, and the rigid move per seeding view slot from its recorded pose to
    /// its current one ([`slot_moves`]).
    pub(super) fn parked_with_moves(&self, key: TileKey) -> std::io::Result<(ParkedTile, Vec<(u32, Affine3A)>)> {
        let dir = self
            .tiles_dir
            .as_deref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no tiles directory"))?;
        let tile = read_tile(dir, key)?;
        let current = |slot: u32| self.views.get(slot as usize).map(|v| (v.camera.position, v.camera.rotation));
        let moves = slot_moves(&tile.slot_poses, &current);
        Ok((tile, moves))
    }
}

#[cfg(test)]
#[path = "area_park_tests.rs"]
mod tests;
