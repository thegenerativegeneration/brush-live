//! The final export: the whole walk in one `splat.ply`, the resident splats plus every parked tile (moved by the
//! pose corrections made since it parked, floors baked), streamed tile by tile so a long walk never sits in memory
//! at once. Without parked tiles the file is `brush_serde::splat_to_ply` of the trainer's splats, as before area
//! training.

use super::LiveModel;
use crate::area::{PlyStreamWriter, TileKey, group_rows};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_train::rows::SplatRows;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// What the final export wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FinalExport {
    /// Size of the PLY, header included.
    pub ply_bytes: u64,
    /// Tiles with rows written (resident and parked, each once); 0 when the splats were written in one piece (no
    /// parked tile).
    pub tiles: u32,
    pub splats: u64,
    /// Parked tiles left out: unreadable, or of another SH degree than the model.
    pub skipped_tiles: u32,
    pub ms: f64,
}

/// `e` with its kind, naming `path`: `writing splat <path>: …`.
fn context(path: &Path) -> impl Fn(io::Error) -> io::Error + '_ {
    move |e| io::Error::new(e.kind(), format!("writing splat {}: {e}", path.display()))
}

fn other(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(e.to_string())
}

impl LiveModel {
    /// Writes the final export to `ply_path` through `<name>.tmp` beside it, synced to disk and renamed on success,
    /// removed on failure, so a failed export leaves no partial file and an earlier `ply_path` as it was. File errors
    /// keep their kind and name the path (`writing splat <path>: …`). `NotFound` without splats, before any file is
    /// touched.
    pub async fn write_final(&self, ply_path: &Path) -> io::Result<FinalExport> {
        if self.splats.is_none() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no splats yet"));
        }
        let name = ply_path.file_name().ok_or_else(|| other(format!("{} names no file", ply_path.display())))?;
        let tmp = ply_path.with_file_name(format!("{}.tmp", name.to_string_lossy()));
        let written = async {
            let file = std::fs::File::create(&tmp).map_err(context(&tmp))?;
            let (out, e) = self.write_final_to(BufWriter::new(file)).await?;
            let file = out.into_inner().map_err(io::IntoInnerError::into_error).map_err(context(&tmp))?;
            file.sync_all().map_err(context(&tmp))?;
            std::fs::rename(&tmp, ply_path).map_err(context(ply_path))?;
            Ok(e)
        }
        .await;
        if written.is_err() {
            std::fs::remove_file(&tmp).ok();
        }
        written
    }

    /// The final export's bytes, as [`Self::write_final`] writes them.
    pub async fn final_ply(&self) -> io::Result<(Vec<u8>, FinalExport)> {
        self.write_final_to(Vec::new()).await
    }

    /// Writes the final export to `out`. Without parked tiles: `splat_to_ply` of the trainer's splats. Otherwise in
    /// two passes over the parked tiles: the first reads and checks each one ([`Self::full_model_rows`]) and counts
    /// its rows, leaving out (with a warning) every tile that fails, so the header's count is exact; the second
    /// writes tile by tile in key order, the resident rows of a key (one readback, grouped) before its parked rows.
    /// A tile that fails only in the second pass is an error, since the header is already written.
    async fn write_final_to<W: Write>(&self, mut out: W) -> io::Result<(W, FinalExport)> {
        let t = web_time::Instant::now();
        let resident = self.splats.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no splats yet"))?;
        let parked = self.area.as_ref().map(|a| a.registry.parked_keys()).unwrap_or_default();
        if parked.is_empty() {
            let bytes = brush_serde::splat_to_ply(resident.clone(), None).await.map_err(other)?;
            out.write_all(&bytes)?;
            let e = FinalExport {
                ply_bytes: bytes.len() as u64,
                splats: u64::from(resident.num_splats()),
                ms: t.elapsed().as_secs_f64() * 1e3,
                ..FinalExport::default()
            };
            return Ok((out, e));
        }
        let trainer = self.trainer.as_ref().expect("trainer exists once splats exist");
        let rows = trainer.read_model_rows(resident).await;
        rows.check().map_err(other)?;
        let coeffs = rows.sh_coeffs_per_channel;
        let tile_size = self.area.as_ref().expect("parked tiles imply area training").registry.params().tile_size;
        let groups = group_rows(&rows.means, tile_size);
        let mut counts: BTreeMap<TileKey, usize> = BTreeMap::new();
        let mut skipped = 0u32;
        for key in parked {
            match self.full_model_rows(key, coeffs) {
                Ok(r) => {
                    counts.insert(key, r.len());
                }
                Err(e) => {
                    log::warn!("area: final export skips unreadable tile {}: {e}", key.stem());
                    skipped += 1;
                }
            }
        }
        let total = rows.len() + counts.values().sum::<usize>();
        let degree = brush_render::sh::sh_degree_from_coeffs(coeffs);
        let mode = if resident.render_mip { SplatRenderMode::Mip } else { SplatRenderMode::Default };
        let mut writer = PlyStreamWriter::new(out, total as u64, degree, mode)?;
        let keys: BTreeSet<TileKey> = groups.keys().chain(counts.keys()).copied().collect();
        let mut tiles = 0u32;
        for key in keys {
            let before = writer.rows();
            if let Some(idx) = groups.get(&key) {
                writer.write_rows(&rows.select(idx))?;
            }
            if let Some(&n) = counts.get(&key) {
                let r = self.full_model_rows(key, coeffs).map_err(|e| {
                    other(format!("parked tile {} failed after it was counted: {e}", key.stem()))
                })?;
                if r.len() != n {
                    return Err(other(format!("parked tile {} holds {} rows, counted {n}", key.stem(), r.len())));
                }
                writer.write_rows(&r)?;
            }
            tiles += u32::from(writer.rows() > before);
        }
        let ply_bytes = writer.bytes();
        let out = writer.finish()?;
        let e = FinalExport {
            ply_bytes,
            tiles,
            splats: total as u64,
            skipped_tiles: skipped,
            ms: t.elapsed().as_secs_f64() * 1e3,
        };
        log::info!(
            "area: final export {} splats in {tiles} tiles ({skipped} skipped), splat.ply {ply_bytes} B in {:.0} ms",
            e.splats,
            e.ms
        );
        Ok((out, e))
    }

    /// The rows parked tile `key` adds to the full model: read from its file and moved by the pose corrections since
    /// it parked ([`Self::load_parked`]), floor not baked. `InvalidData` when they hold another number of SH
    /// coefficients per channel than the model's `coeffs`, or do not fit their row count.
    pub(super) fn full_model_rows(&self, key: TileKey, coeffs: u32) -> io::Result<SplatRows> {
        let rows = self.load_parked(key)?;
        if rows.sh_coeffs_per_channel != coeffs {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} SH coefficients, the model has {coeffs}", rows.sh_coeffs_per_channel),
            ));
        }
        rows.check().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(rows)
    }
}

#[cfg(test)]
#[path = "area_export_tests.rs"]
mod tests;
