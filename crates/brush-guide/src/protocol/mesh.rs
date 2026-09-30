//! `mesh_bricks` payload: per brick its key, quantised vertices,
//! octahedral normals, optional 8-bit sRGB vertex colours and u16 triangle
//! indices.

use super::ProtocolError;
use super::cells::{oct_decode, oct_encode};
use crate::geometry::mesh::BrickMesh;
use crate::geometry::tsdf::{BRICK, BrickKey, VOXEL};

/// One brick of a `mesh_bricks` message: its new mesh, or that it has none
/// any more.
#[derive(Debug, Clone)]
pub enum MeshBrick {
    Mesh(BrickMesh),
    Removed(BrickKey),
}

impl MeshBrick {
    pub fn key(&self) -> BrickKey {
        match self {
            Self::Mesh(m) => m.key,
            Self::Removed(k) => *k,
        }
    }
}

pub const MESH_BRICK_REMOVED: u8 = 1;
/// Brick flag: `u8×3` sRGB per vertex follow the normals.
pub const MESH_BRICK_COLOURS: u8 = 2;

/// Bytes before a brick's vertex data: key, flags, vertex and index counts.
pub(super) const MESH_BRICK_HEADER: usize = 21;

/// Brick side in metres.
const BRICK_SIZE: f32 = BRICK as f32 * VOXEL;

/// Quantised positions span the brick box widened by this on every side:
/// surface-nets vertices of cells straddling the border lie up to half a
/// voxel outside the box.
pub const BRICK_MARGIN: f32 = 0.5 * VOXEL;

const QUANT_SPAN: f32 = BRICK_SIZE + 2.0 * BRICK_MARGIN;

fn quantise(local: f32) -> u16 {
    ((local + BRICK_MARGIN) / QUANT_SPAN * 65535.0)
        .round()
        .clamp(0.0, 65535.0) as u16
}

fn dequantise(q: u16) -> f32 {
    q as f32 / 65535.0 * QUANT_SPAN - BRICK_MARGIN
}

/// Most vertices a brick's mesh may have: u16 indices address 0..=65 535.
/// A brick has at most 21³ = 9 261.
pub const MAX_BRICK_VERTICES: usize = u16::MAX as usize + 1;

/// Encodes bricks as the `mesh_bricks` payload. Panics if a mesh has more
/// than `MAX_BRICK_VERTICES` (65 536) vertices, or colours but not one per
/// vertex.
pub fn encode_mesh_bricks<'a>(bricks: impl IntoIterator<Item = &'a MeshBrick>) -> Vec<u8> {
    let mut out = Vec::new();
    for brick in bricks {
        for v in brick.key().0.to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let MeshBrick::Mesh(mesh) = brick else {
            out.push(MESH_BRICK_REMOVED);
            out.extend_from_slice(&[0; 8]);
            continue;
        };
        assert!(
            mesh.positions.len() <= MAX_BRICK_VERTICES,
            "brick {:?} has {} vertices, more than 65 536 (u16 indices)",
            mesh.key,
            mesh.positions.len()
        );
        let coloured = !mesh.colours.is_empty();
        assert!(
            !coloured || mesh.colours.len() == mesh.positions.len(),
            "brick {:?} has {} colours for {} vertices",
            mesh.key,
            mesh.colours.len(),
            mesh.positions.len()
        );
        out.push(if coloured { MESH_BRICK_COLOURS } else { 0 });
        out.extend_from_slice(&(mesh.positions.len() as u32).to_le_bytes());
        out.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
        let origin = (mesh.key.0 * BRICK).as_vec3() * VOXEL;
        for &p in &mesh.positions {
            let local = glam::Vec3::from(p) - origin;
            for v in local.to_array() {
                out.extend_from_slice(&quantise(v).to_le_bytes());
            }
        }
        for &n in &mesh.normals {
            out.extend_from_slice(&oct_encode(glam::Vec3::from(n)));
        }
        out.extend_from_slice(mesh.colours.as_flattened());
        for &i in &mesh.indices {
            out.extend_from_slice(&(i as u16).to_le_bytes());
        }
    }
    out
}

fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], ProtocolError> {
    if rest.len() < n {
        return Err(ProtocolError::Truncated);
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

/// Decodes a `mesh_bricks` payload of `num_bricks` bricks. Positions come
/// back in world metres, within half a quantisation step (8 µm) of the
/// encoded ones. Rejects a removed brick with non-zero counts, an index
/// count that is not a multiple of 3 and indices past the vertices.
pub fn decode_mesh_bricks(bytes: &[u8], num_bricks: u32) -> Result<Vec<MeshBrick>, ProtocolError> {
    let mut rest = bytes;
    let bricks = (0..num_bricks)
        .map(|_| decode_brick(&mut rest))
        .collect::<Result<Vec<_>, _>>()?;
    if !rest.is_empty() {
        return Err(ProtocolError::PayloadSize {
            expected: bytes.len() - rest.len(),
            actual: bytes.len(),
        });
    }
    Ok(bricks)
}

/// Decodes the brick at the start of `rest` and advances past it.
fn decode_brick(rest: &mut &[u8]) -> Result<MeshBrick, ProtocolError> {
    let h = take(rest, MESH_BRICK_HEADER)?;
    let int = |i: usize| i32::from_le_bytes(h[i * 4..i * 4 + 4].try_into().unwrap());
    let key = BrickKey(glam::IVec3::new(int(0), int(1), int(2)));
    let flags = h[12];
    let num_vertices = u32::from_le_bytes(h[13..17].try_into().unwrap()) as usize;
    let num_indices = u32::from_le_bytes(h[17..21].try_into().unwrap()) as usize;
    if flags & MESH_BRICK_REMOVED != 0 {
        if num_vertices != 0 || num_indices != 0 {
            return Err(ProtocolError::RemovedBrickWithData);
        }
        return Ok(MeshBrick::Removed(key));
    }
    if !num_indices.is_multiple_of(3) {
        return Err(ProtocolError::PartialTriangle(num_indices));
    }
    let coloured = flags & MESH_BRICK_COLOURS != 0;
    decode_mesh(rest, key, num_vertices, num_indices, coloured).map(MeshBrick::Mesh)
}

/// Decodes the vertex and index data of brick `key` from `rest`, with
/// vertex colours if `coloured`.
fn decode_mesh(
    rest: &mut &[u8],
    key: BrickKey,
    num_vertices: usize,
    num_indices: usize,
    coloured: bool,
) -> Result<BrickMesh, ProtocolError> {
    let origin = (key.0 * BRICK).as_vec3() * VOXEL;
    let positions = take(
        rest,
        num_vertices
            .checked_mul(6)
            .ok_or(ProtocolError::SizeOverflow)?,
    )?
    .as_chunks::<6>()
    .0
    .iter()
    .map(|c| {
        let q = |i: usize| dequantise(u16::from_le_bytes([c[i * 2], c[i * 2 + 1]]));
        (origin + glam::Vec3::new(q(0), q(1), q(2))).to_array()
    })
    .collect();
    let normals = take(rest, num_vertices * 2)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| oct_decode(c).to_array())
        .collect();
    let colours = if coloured {
        take(rest, num_vertices * 3)?.as_chunks::<3>().0.to_vec()
    } else {
        Vec::new()
    };
    let indices: Vec<u32> = take(
        rest,
        num_indices
            .checked_mul(2)
            .ok_or(ProtocolError::SizeOverflow)?,
    )?
    .as_chunks::<2>()
    .0
    .iter()
    .map(|&c| u16::from_le_bytes(c) as u32)
    .collect();
    if let Some(&index) = indices.iter().find(|&&i| i as usize >= num_vertices) {
        return Err(ProtocolError::IndexOutOfRange {
            index,
            num_vertices,
        });
    }
    Ok(BrickMesh {
        key,
        positions,
        normals,
        colours,
        indices,
    })
}
