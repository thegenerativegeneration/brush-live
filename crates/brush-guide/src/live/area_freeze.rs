//! Holding left tiles still: a resident tile the walker has left ([`crate::area::TileRegistry::left_keys`]: visited,
//! now beyond radius + hysteresis) can stay resident for a while, since views near the walker may still see it. Its
//! rows are held in the trainer ([`brush_train::train::SplatTrainer::set_frozen_rows`]): no step trains them, refine
//! neither decays, re-floors, densifies nor evicts them, so the area stays as the walker left it until it parks or the
//! walker returns. Rows appended later (seeds, restores) train until the mask is next built.

use super::LiveModel;
use crate::area::{TileKey, rows_in};
use brush_render::gaussian_splats::Splats;
use burn::tensor::{Bool, Tensor, TensorData};
use glam::Vec3;
use std::collections::BTreeSet;

impl LiveModel {
    /// After a refine and its parks (the refine dropped the trainer's mask, and parks and restores moved rows): marks
    /// the tiles around the newest view as visited and holds the rows of the left resident tiles. No-op without area
    /// training.
    pub(super) async fn hold_left_rows_after_refine(&mut self, splats: &Splats) {
        let Some(centre) = self.view_cams.last().map(|(p, _)| *p) else {
            return;
        };
        if let Some(keys) = self.left_resident_keys(centre) {
            self.hold_rows(splats, keys).await;
        }
    }

    /// At a keyframe whose camera stands at `centre`, before its restores and seeds: marks the tiles around it as
    /// visited and, when the left resident tiles differ from those the mask holds, holds the new set (the walker left
    /// a tile, or came back to one). No-op without area training.
    pub(super) async fn hold_left_rows_at_keyframe(&mut self, splats: &Splats, centre: Vec3) {
        let Some(keys) = self.left_resident_keys(centre) else {
            return;
        };
        if self.area.as_ref().is_some_and(|a| a.frozen_keys != keys) {
            self.hold_rows(splats, keys).await;
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
