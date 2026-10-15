//! Parking: at each refine the live model reads back which splats the window's training views drew, and after the
//! refine the resident tiles the capture has left (beyond the radius plus hysteresis, unseen for the window) leave
//! the trainer for tile files. Parked rows are out of the trainer, so nothing trains, refines, decays or evicts them
//! until they are restored.

use super::LiveModel;
use crate::area::{ParkedTile, SlotPose, TileKey, apply_moves, group_rows, read_tile, rows_in, slot_moves, write_tile};
use brush_render::gaussian_splats::Splats;
use brush_train::rows::SplatRows;
use burn::tensor::Transaction;
use std::collections::BTreeSet;

impl LiveModel {
    /// Records per resident tile whether a training step since the last refine drew one of its splats (the refine
    /// record's `importance_views`, occlusion included) from one readback of the means and that count. No-op without
    /// area training or before a step since the last refine; call right before `refine`.
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
        if conflicts != area.conflicts {
            if !conflicts.is_empty() {
                let stems: Vec<String> = conflicts.iter().map(|k| k.stem()).collect();
                log::info!("area: resident splats lie in parked tiles {}", stems.join(", "));
            }
            area.conflicts = conflicts;
        }
    }

    /// Parks the tiles [`crate::area::TileRegistry::to_park`] names: their rows leave the trainer and go to one tile
    /// file each, with the pose of every view that seeded them. A park that would take every resident row is
    /// skipped (counted). The rows of a tile whose file cannot be written go back to the trainer (with zero Adam
    /// moments) and parking stays off until the next reset. Each park logs one line per tile and one for the whole
    /// event, whose time includes the row readback. After a park the all-views loader drops views that left the area.
    pub(super) async fn park_after_refine(&mut self, splats: Splats) -> Splats {
        let Some(centre) = self.view_cams.last().map(|(p, _)| *p) else {
            return splats;
        };
        let Some(area) = self.area.as_mut().filter(|a| !a.parking_off) else {
            return splats;
        };
        let keys = area.registry.to_park(centre, self.iter, self.views.len() as u32);
        if keys.is_empty() {
            area.skipping = false;
            return splats;
        }
        let t = web_time::Instant::now();
        let tile_size = area.registry.params().tile_size;
        let means: Vec<f32> = splats
            .means()
            .into_data_async()
            .await
            .expect("means readback")
            .convert::<f32>()
            .try_into_vec()
            .expect("f32 means");
        let take = rows_in(&keys, &means, tile_size);
        let taken = take.iter().filter(|&&t| t).count();
        let skip = taken == take.len() && taken > 0;
        if skip {
            area.counters.parks_skipped += 1;
            if !area.skipping {
                log::info!("area: park skipped, it would take every resident splat");
            }
        }
        area.skipping = skip;
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
        area.counters.tile_bytes = area.file_bytes.values().sum();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        area.counters.park_ms += ms;
        area.counters.max_park_ms = area.counters.max_park_ms.max(ms);
        log::info!("area: park event {tiles} tiles, {} splats in {ms:.1} ms", parked.len());
        self.drop_departed_views();
        splats
    }

    /// Writes one tile file per tile of the taken `rows` and marks it parked; returns the indices (into `rows`) of
    /// the rows parked and the number of tiles they make. A tile the registry does not know, or whose file cannot be
    /// written, stays out; the first write failure turns parking off until the next reset, so the tiles after it
    /// stay out too.
    fn write_tiles(&mut self, rows: &SplatRows, tile_size: f32) -> (BTreeSet<usize>, u32) {
        let dir = self.tiles_dir.clone().expect("area training has a tiles directory");
        let (mut parked, mut tiles) = (BTreeSet::new(), 0);
        for (key, idx) in group_rows(&rows.means, tile_size) {
            if self.area.as_ref().is_none_or(|a| a.parking_off) {
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
                    area.parking_off = true;
                    continue;
                }
            };
            area.registry.mark_parked(key, n, self.iter);
            area.file_bytes.insert(key, bytes);
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

    /// The rows of parked tile `key` from its file, moved by the pose corrections their seeding views had since the
    /// park ([`slot_moves`]).
    pub(super) fn load_parked(&self, key: TileKey) -> std::io::Result<SplatRows> {
        let dir = self
            .tiles_dir
            .as_deref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no tiles directory"))?;
        let tile = read_tile(dir, key)?;
        let current = |slot: u32| self.views.get(slot as usize).map(|v| (v.camera.position, v.camera.rotation));
        let moves = slot_moves(&tile.slot_poses, &current);
        let mut rows = tile.rows;
        apply_moves(&mut rows, &moves);
        Ok(rows)
    }
}

#[cfg(test)]
#[path = "area_park_tests.rs"]
mod tests;
