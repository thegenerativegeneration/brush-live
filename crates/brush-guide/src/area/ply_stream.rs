//! A PLY writer that takes splat rows batch by batch, so the final export can stream every tile of a long walk
//! into one file without holding them all. The bytes are those of `brush_serde::splat_to_ply` for the same splats in
//! the same order: binary little-endian, its four header comments, one `vertex` element of f32 properties `x y z
//! scale_0..2 opacity rot_0..3 f_dc_0..2 f_rest_*` (rest coefficients channel-major), the Mip floor folded in and
//! rotations normalised.

use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::sh::sh_coeffs_for_degree;
use brush_train::rows::SplatRows;
use std::borrow::Cow;
use std::io::{self, Write};

/// Writes a PLY of a row count fixed up front ([`Self::new`]) from batches of splat rows ([`Self::write_rows`]).
pub struct PlyStreamWriter<W: Write> {
    out: W,
    total: u64,
    rows: u64,
    bytes: u64,
    /// SH coefficients per channel the header declares.
    coeffs: u32,
    /// One row's bytes, reused.
    row: Vec<u8>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

impl<W: Write> PlyStreamWriter<W> {
    /// Writes the header for `total` rows of SH degree `sh_degree` rendered as `mode`, the header
    /// `splat_to_ply(splats, None)` writes (vertical axis y).
    pub fn new(mut out: W, total: u64, sh_degree: u32, mode: SplatRenderMode) -> io::Result<Self> {
        let coeffs = sh_coeffs_for_degree(sh_degree);
        let render = if mode == SplatRenderMode::Mip { "mip" } else { "default" };
        let mut header = format!(
            "ply\nformat binary_little_endian 1.0\ncomment Exported from Brush\ncomment Vertical axis: y\n\
             comment SH degree: {sh_degree}\ncomment SplatRenderMode: {render}\nelement vertex {total}\n"
        );
        let names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"];
        let dc = ["f_dc_0", "f_dc_1", "f_dc_2"];
        for name in names.iter().chain(&dc) {
            header.push_str(&format!("property float {name}\n"));
        }
        for i in 0..(coeffs - 1) * 3 {
            header.push_str(&format!("property float f_rest_{i}\n"));
        }
        header.push_str("end_header\n");
        out.write_all(header.as_bytes())?;
        Ok(Self { out, total, rows: 0, bytes: header.len() as u64, coeffs, row: Vec::new() })
    }

    /// Appends `rows` (their floor folded in, [`SplatRows::baked`]). `InvalidInput`, with nothing written, for rows
    /// of another SH degree, rows whose columns do not fit their count ([`SplatRows::check`]), or rows beyond the
    /// header's count.
    pub fn write_rows(&mut self, rows: &SplatRows) -> io::Result<()> {
        rows.check().map_err(invalid)?;
        if rows.sh_coeffs_per_channel != self.coeffs {
            return Err(invalid(format!(
                "rows hold {} SH coefficients per channel, the file {}",
                rows.sh_coeffs_per_channel, self.coeffs
            )));
        }
        let n = rows.len() as u64;
        if self.rows + n > self.total {
            return Err(invalid(format!("{} rows exceed the header's {}", self.rows + n, self.total)));
        }
        let rows = if rows.min_scale.is_empty() { Cow::Borrowed(rows) } else { Cow::Owned(rows.baked()) };
        let c = self.coeffs as usize;
        for i in 0..rows.len() {
            self.row.clear();
            let mut put = |v: f32| self.row.extend_from_slice(&v.to_le_bytes());
            rows.means[i * 3..i * 3 + 3].iter().for_each(|&v| put(v));
            rows.log_scales[i * 3..i * 3 + 3].iter().for_each(|&v| put(v));
            put(rows.raw_opacities[i]);
            // Normalised as `splat_to_ply` does it, operation for operation.
            let r = &rows.rotations[i * 4..i * 4 + 4];
            let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt().max(1e-12);
            r.iter().for_each(|&v| put(v / rn));
            // SH rows are `[coeffs, 3]`: DC per channel, then the rest of each channel in turn.
            let sh = &rows.sh_coeffs[i * c * 3..(i + 1) * c * 3];
            (0..3).for_each(|ch| put(sh[ch]));
            for ch in 0..3 {
                (1..c).for_each(|k| put(sh[k * 3 + ch]));
            }
            self.out.write_all(&self.row)?;
            self.bytes += self.row.len() as u64;
        }
        self.rows += n;
        Ok(())
    }

    /// Rows written so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes written so far, header included.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Flushes and returns the output; `InvalidData` when fewer rows than the header's count were written (the file
    /// would not parse).
    pub fn finish(mut self) -> io::Result<W> {
        if self.rows != self.total {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} rows written, the header declares {}", self.rows, self.total),
            ));
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
#[path = "ply_stream_tests.rs"]
pub(crate) mod tests;
