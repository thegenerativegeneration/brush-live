//! Score set cells: fixed-size records with an octahedral-encoded normal.

use super::ProtocolError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    /// Opacity-weighted mean position of the voxel's Gaussians; lies inside the voxel.
    pub center: [f32; 3],
    pub coverage: u8,
    pub uncertainty: u8,
    pub age: u8,
    /// Unit surface normal facing the observing cameras; `None` when the voxel is not planar enough.
    pub normal: Option<[f32; 3]>,
    /// `min(255, round(Σ opacity · 32))` over the voxel's visible Gaussians.
    pub density: u8,
}

pub const CELL_BYTES: usize = 19;
pub const CELL_FLAG_NORMAL: u8 = 1;

fn sign_not_zero(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// Octahedral encoding of a unit vector into two bytes.
pub fn oct_encode(n: glam::Vec3) -> [u8; 2] {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs());
    let (mut x, mut y) = (n.x, n.y);
    if n.z < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * sign_not_zero(ox);
        y = (1.0 - ox.abs()) * sign_not_zero(oy);
    }
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
    [q(x), q(y)]
}

pub fn oct_decode(b: [u8; 2]) -> glam::Vec3 {
    let x = b[0] as f32 / 255.0 * 2.0 - 1.0;
    let y = b[1] as f32 / 255.0 * 2.0 - 1.0;
    let mut n = glam::Vec3::new(x, y, 1.0 - x.abs() - y.abs());
    let t = (-n.z).max(0.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize()
}

pub fn encode_cells(cells: &[Cell]) -> Vec<u8> {
    let mut out = Vec::with_capacity(cells.len() * CELL_BYTES);
    for c in cells {
        for v in c.center {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[c.coverage, c.uncertainty, c.age]);
        let (oct, flags) = match c.normal {
            Some(n) => (oct_encode(glam::Vec3::from(n)), CELL_FLAG_NORMAL),
            None => ([0, 0], 0),
        };
        out.extend_from_slice(&oct);
        out.extend_from_slice(&[c.density, flags]);
    }
    out
}

pub fn decode_cells(bytes: &[u8]) -> Result<Vec<Cell>, ProtocolError> {
    if bytes.len() % CELL_BYTES != 0 {
        return Err(ProtocolError::PayloadSize {
            expected: bytes.len().next_multiple_of(CELL_BYTES),
            actual: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(CELL_BYTES)
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            Cell {
                center: [f(0), f(1), f(2)],
                coverage: c[12],
                uncertainty: c[13],
                age: c[14],
                normal: (c[18] & CELL_FLAG_NORMAL != 0)
                    .then(|| oct_decode([c[15], c[16]]).to_array()),
                density: c[17],
            }
        })
        .collect())
}
