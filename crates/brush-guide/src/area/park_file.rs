//! The tile file: the splat rows of one parked tile plus the poses of the views that seeded them at park time, so a
//! restore can replay the pose corrections made while the tile was parked ([`super::pose_replay`]).
//!
//! Layout, little-endian: `b"BGTL"`, u32 file version, i32 key x, i32 key z, u32 tile version, u32 row count n, u32
//! SH coefficients per channel, u32 flags (bit 0: floor), u32 pose count; per pose u32 slot, 3×f32 position, 4×f32
//! rotation (x y z w); then the columns: means f32×3n, rotations f16×4n (normalised, w x y z), log-scales f16×3n,
//! raw opacities f16×n, SH f16×(n·coeffs·3), floor f16×n (when flagged), origin u32×n (slot + 1, 0 none), age f32×n,
//! importance f32×n. Means stay f32 so splats far from the world origin keep their position exactly.

use super::tiles::TileKey;
use brush_train::rows::SplatRows;
use glam::{Quat, Vec3};
use half::f16;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// The layout this module writes and the only one it reads.
pub const TILE_FILE_VERSION: u32 = 1;

const MAGIC: &[u8; 4] = b"BGTL";
/// Flags bit 0: the file carries a Mip floor column.
const FLAG_FLOOR: u32 = 1;
/// Magic plus eight u32 / i32 header fields.
const HEADER_BYTES: usize = 36;

/// A view slot's camera pose (position, rotation) when its tile was parked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotPose {
    pub slot: u32,
    pub position: Vec3,
    pub rotation: Quat,
}

/// One parked tile as stored on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct ParkedTile {
    pub key: TileKey,
    /// The tile's parking count ([`super::TileInfo::version`]) when it was written.
    pub version: u32,
    /// The pose at park time of every view slot the rows' origins name.
    pub slot_poses: Vec<SlotPose>,
    pub rows: SplatRows,
}

/// `dir/<stem>.bin`, e.g. `dir/-3_12.bin`.
pub fn tile_path(dir: &Path, key: TileKey) -> PathBuf {
    dir.join(format!("{}.bin", key.stem()))
}

fn temp_path(dir: &Path, key: TileKey) -> PathBuf {
    dir.join(format!("{}.bin.tmp", key.stem()))
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
}

fn put_f16s(out: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    out.extend(
        values
            .into_iter()
            .flat_map(|v| f16::from_f32(v).to_le_bytes()),
    );
}

/// `q` (w x y z) scaled to unit length; a zero or non-finite length leaves it as it is.
fn normalised(q: &[f32]) -> [f32; 4] {
    let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    let s = if l > 0.0 && l.is_finite() {
        1.0 / l
    } else {
        1.0
    };
    [q[0] * s, q[1] * s, q[2] * s, q[3] * s]
}

/// `Err` when `rows` cannot be written in this layout: they fail [`SplatRows::check`], hold more than `u32::MAX`
/// rows, or name slot `u32::MAX` (which has no tag) as an origin.
fn writable(rows: &SplatRows) -> Result<(), String> {
    rows.check()?;
    if u32::try_from(rows.len()).is_err() {
        return Err(format!("{} rows do not fit a tile file", rows.len()));
    }
    if rows.origin.contains(&Some(u32::MAX)) {
        return Err("origin slot u32::MAX has no tag".to_owned());
    }
    Ok(())
}

/// The file bytes of `t`. Panics when `t.rows` cannot be written in this layout (see [`write_tile`], which returns
/// an error instead).
pub fn encode_tile(t: &ParkedTile) -> Vec<u8> {
    let r = &t.rows;
    if let Err(e) = writable(r) {
        panic!("encode_tile: {e}");
    }
    let n = r.len() as u32;
    let floor = !r.min_scale.is_empty();
    // Header, poses, then per row: means, rotations, log-scales, opacity, floor, origin, age, importance; and SH.
    let row_bytes = 12 + 8 + 6 + 2 + 2 + 4 + 4 + 4;
    let mut out = Vec::with_capacity(
        HEADER_BYTES + t.slot_poses.len() * 32 + r.len() * row_bytes + r.sh_coeffs.len() * 2,
    );
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, TILE_FILE_VERSION);
    put_i32(&mut out, t.key.x);
    put_i32(&mut out, t.key.z);
    put_u32(&mut out, t.version);
    put_u32(&mut out, n);
    put_u32(&mut out, r.sh_coeffs_per_channel);
    put_u32(&mut out, if floor { FLAG_FLOOR } else { 0 });
    put_u32(
        &mut out,
        u32::try_from(t.slot_poses.len()).expect("pose count fits u32"),
    );
    for p in &t.slot_poses {
        put_u32(&mut out, p.slot);
        put_f32s(&mut out, &p.position.to_array());
        put_f32s(&mut out, &p.rotation.to_array());
    }
    put_f32s(&mut out, &r.means);
    put_f16s(&mut out, r.rotations.chunks_exact(4).flat_map(normalised));
    put_f16s(&mut out, r.log_scales.iter().copied());
    put_f16s(&mut out, r.raw_opacities.iter().copied());
    put_f16s(&mut out, r.sh_coeffs.iter().copied());
    put_f16s(&mut out, r.min_scale.iter().copied());
    for o in &r.origin {
        let tag = o.map_or(0, |s| s + 1);
        put_u32(&mut out, tag);
    }
    put_f32s(&mut out, &r.age);
    put_f32s(&mut out, &r.importance);
    out
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Reads values off the front of a byte slice; running past its end is an `InvalidData` error.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if n > self.bytes.len() {
            return Err(invalid("truncated tile file"));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    /// The next `count` values of `width` bytes each, after checking that they are all there.
    fn chunks(
        &mut self,
        count: usize,
        width: usize,
    ) -> io::Result<std::slice::ChunksExact<'a, u8>> {
        let len = count
            .checked_mul(width)
            .ok_or_else(|| invalid("tile file sizes overflow"))?;
        Ok(self.take(len)?.chunks_exact(width))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u32s(&mut self, count: usize) -> io::Result<Vec<u32>> {
        Ok(self
            .chunks(count, 4)?
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect())
    }

    fn f32s(&mut self, count: usize) -> io::Result<Vec<f32>> {
        Ok(self
            .chunks(count, 4)?
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect())
    }

    fn f16s(&mut self, count: usize) -> io::Result<Vec<f32>> {
        Ok(self
            .chunks(count, 2)?
            .map(|b| f16::from_le_bytes(b.try_into().expect("2 bytes")).to_f32())
            .collect())
    }
}

/// The fixed-size head of a tile file.
struct Header {
    key: TileKey,
    version: u32,
    n: usize,
    coeffs: u32,
    floor: bool,
    poses: usize,
}

fn read_header(r: &mut Reader<'_>) -> io::Result<Header> {
    if r.take(4)? != MAGIC {
        return Err(invalid("not a tile file"));
    }
    let file_version = r.u32()?;
    if file_version != TILE_FILE_VERSION {
        return Err(invalid(format!("tile file version {file_version}")));
    }
    let key = TileKey {
        x: r.i32()?,
        z: r.i32()?,
    };
    let version = r.u32()?;
    let n = r.u32()? as usize;
    let coeffs = r.u32()?;
    let flags = r.u32()?;
    if flags & !FLAG_FLOOR != 0 {
        return Err(invalid(format!("unknown tile file flags {flags:#x}")));
    }
    let poses = r.u32()? as usize;
    Ok(Header {
        key,
        version,
        n,
        coeffs,
        floor: flags & FLAG_FLOOR != 0,
        poses,
    })
}

/// The tile in `bytes`. An `InvalidData` error for a truncated or foreign file, another file version, unknown
/// flags, bytes after the last column, or columns that fail [`SplatRows::check`]; never a panic.
pub fn decode_tile(bytes: &[u8]) -> io::Result<ParkedTile> {
    let mut r = Reader { bytes };
    let h = read_header(&mut r)?;
    let slot_poses = r
        .chunks(h.poses, 32)?
        .map(|b| {
            let f = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
            SlotPose {
                slot: u32::from_le_bytes(b[0..4].try_into().expect("4 bytes")),
                position: Vec3::new(f(4), f(8), f(12)),
                rotation: Quat::from_xyzw(f(16), f(20), f(24), f(28)),
            }
        })
        .collect();
    let sh_width = (h.coeffs as usize)
        .checked_mul(3)
        .ok_or_else(|| invalid("tile file sizes overflow"))?;
    let three =
        h.n.checked_mul(3)
            .ok_or_else(|| invalid("tile file sizes overflow"))?;
    let four =
        h.n.checked_mul(4)
            .ok_or_else(|| invalid("tile file sizes overflow"))?;
    let sh =
        h.n.checked_mul(sh_width)
            .ok_or_else(|| invalid("tile file sizes overflow"))?;
    let rows = SplatRows {
        means: r.f32s(three)?,
        rotations: r.f16s(four)?,
        log_scales: r.f16s(three)?,
        raw_opacities: r.f16s(h.n)?,
        sh_coeffs: r.f16s(sh)?,
        min_scale: if h.floor { r.f16s(h.n)? } else { Vec::new() },
        origin: r.u32s(h.n)?.into_iter().map(|t| t.checked_sub(1)).collect(),
        age: r.f32s(h.n)?,
        importance: r.f32s(h.n)?,
        sh_coeffs_per_channel: h.coeffs,
        trained: Vec::new(),
        moments: None,
    };
    if !r.bytes.is_empty() {
        return Err(invalid(format!(
            "{} bytes after the last tile column",
            r.bytes.len()
        )));
    }
    rows.check().map_err(invalid)?;
    Ok(ParkedTile {
        key: h.key,
        version: h.version,
        slot_poses,
        rows,
    })
}

/// Writes `t` to [`tile_path`] through a temporary file and a rename, so a reader never sees a half-written tile.
/// Returns the file's size in bytes. `InvalidInput` for rows this layout cannot hold (nothing is written); on any
/// other failure the temporary file is removed. The file is not synced: a tile only has to outlive the session
/// that parked it.
pub fn write_tile(dir: &Path, t: &ParkedTile) -> io::Result<u64> {
    writable(&t.rows).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let bytes = encode_tile(t);
    let tmp = temp_path(dir, t.key);
    let written =
        std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, tile_path(dir, t.key)));
    if let Err(e) = written {
        std::fs::remove_file(&tmp).ok();
        return Err(e);
    }
    Ok(bytes.len() as u64)
}

fn check_key(path: &Path, want: TileKey, got: TileKey) -> io::Result<()> {
    if want == got {
        Ok(())
    } else {
        Err(invalid(format!(
            "{} holds tile {}, not {}",
            path.display(),
            got.stem(),
            want.stem()
        )))
    }
}

/// The parked tile `key` from `dir`; `InvalidData` when the file fails [`decode_tile`] or holds another tile.
pub fn read_tile(dir: &Path, key: TileKey) -> io::Result<ParkedTile> {
    let path = tile_path(dir, key);
    let t = decode_tile(&std::fs::read(&path)?)?;
    check_key(&path, key, t.key)?;
    Ok(t)
}

/// The row count of the parked tile `key`, from the file's header alone.
pub fn read_tile_count(dir: &Path, key: TileKey) -> io::Result<u32> {
    let path = tile_path(dir, key);
    let mut head = [0u8; HEADER_BYTES];
    match std::fs::File::open(&path)?.read_exact(&mut head) {
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(invalid("truncated tile file"));
        }
        other => other?,
    }
    let h = read_header(&mut Reader { bytes: &head })?;
    check_key(&path, key, h.key)?;
    Ok(h.n as u32)
}

/// Removes the tile file of `key`, a leftover temporary file, and its coarse levels `<stem>_L*.bin`. Missing files
/// are fine; any other failure is logged.
pub fn remove_tile_files(dir: &Path, key: TileKey) {
    let remove = |p: &Path| match std::fs::remove_file(p) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => {
            log::warn!("removing {}: {e}", p.display())
        }
        _ => {}
    };
    remove(&tile_path(dir, key));
    remove(&temp_path(dir, key));
    let level_prefix = format!("{}_L", key.stem());
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                log::warn!("listing {}: {e}", dir.display());
            }
            return;
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&level_prefix) && name.ends_with(".bin") {
            remove(&entry.path());
        }
    }
}

#[cfg(test)]
#[path = "park_file_tests.rs"]
mod tests;
