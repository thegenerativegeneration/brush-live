use crate::geometry::mesh::BrickMesh;
use crate::geometry::tsdf::{BRICK, BrickKey, VOXEL};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("frame too short")]
    Truncated,
    #[error("header length {0} exceeds frame")]
    BadHeaderLength(usize),
    #[error("invalid header json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("payload size mismatch: expected {expected}, got {actual}")]
    PayloadSize { expected: usize, actual: usize },
    #[error("declared payload size overflows")]
    SizeOverflow,
    #[error("depth_confidence set without depth_size")]
    ConfidenceWithoutDepth,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KeyframeHeader {
    pub id: u64,
    pub timestamp: f64,
    /// Camera-to-world, ARKit world frame, column-major.
    pub pose: [f32; 16],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub jpeg_len: u32,
    pub depth_size: Option<[u32; 2]>,
    #[serde(default)]
    pub depth_confidence: bool,
    pub num_points: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientHeader {
    Hello {
        session_id: String,
        device_model: String,
        has_lidar: bool,
    },
    Keyframe(KeyframeHeader),
    Finish,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerHeader {
    Ack {
        keyframe_id: u64,
    },
    ScoreSet {
        version: u64,
        based_on_keyframe_id: u64,
        voxel_size: f32,
        num_cells: u32,
        cell_bytes: u32,
    },
    Status {
        num_keyframes: u32,
        num_splats: u32,
        train_iters_per_s: f32,
        last_score_ms: u32,
    },
    Splat {
        ply_len: u64,
    },
    MeshBricks {
        version: u64,
        num_bricks: u32,
        /// Depth rendering, TSDF fusion and meshing time of the round, ms.
        mesh_ms: u32,
    },
    Error {
        message: String,
    },
}

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

pub fn encode_frame<H: Serialize>(header: &H, payload: &[u8]) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("header serialises");
    let mut out = Vec::with_capacity(4 + json.len() + payload.len());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(payload);
    out
}

pub fn decode_frame<H: DeserializeOwned>(frame: &[u8]) -> Result<(H, &[u8]), ProtocolError> {
    let len_bytes: [u8; 4] = frame
        .get(0..4)
        .ok_or(ProtocolError::Truncated)?
        .try_into()
        .unwrap();
    let len = u32::from_le_bytes(len_bytes) as usize;
    let json = frame
        .get(4..4 + len)
        .ok_or(ProtocolError::BadHeaderLength(len))?;
    Ok((serde_json::from_slice(json)?, &frame[4 + len..]))
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

/// Bytes before a brick's vertex data: key, flags, vertex and index counts.
const MESH_BRICK_HEADER: usize = 21;

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

/// Encodes bricks as the `mesh_bricks` payload. Panics if a mesh has 65 536
/// vertices or more (a brick has at most 21³).
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
            mesh.positions.len() <= u16::MAX as usize + 1,
            "brick {:?} has {} vertices, more than u16 indices address",
            mesh.key,
            mesh.positions.len()
        );
        out.push(0);
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
/// encoded ones.
pub fn decode_mesh_bricks(bytes: &[u8], num_bricks: u32) -> Result<Vec<MeshBrick>, ProtocolError> {
    let mut rest = bytes;
    let mut bricks = Vec::new();
    for _ in 0..num_bricks {
        let h = take(&mut rest, MESH_BRICK_HEADER)?;
        let int = |i: usize| i32::from_le_bytes(h[i * 4..i * 4 + 4].try_into().unwrap());
        let key = BrickKey(glam::IVec3::new(int(0), int(1), int(2)));
        let flags = h[12];
        let num_vertices = u32::from_le_bytes(h[13..17].try_into().unwrap()) as usize;
        let num_indices = u32::from_le_bytes(h[17..21].try_into().unwrap()) as usize;
        if flags & MESH_BRICK_REMOVED != 0 {
            bricks.push(MeshBrick::Removed(key));
            continue;
        }
        let origin = (key.0 * BRICK).as_vec3() * VOXEL;
        let positions = take(
            &mut rest,
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
        let normals = take(&mut rest, num_vertices * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| oct_decode(c).to_array())
            .collect();
        let indices = take(
            &mut rest,
            num_indices
                .checked_mul(2)
                .ok_or(ProtocolError::SizeOverflow)?,
        )?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_le_bytes(c) as u32)
        .collect();
        bricks.push(MeshBrick::Mesh(BrickMesh {
            key,
            positions,
            normals,
            indices,
        }));
    }
    if !rest.is_empty() {
        return Err(ProtocolError::PayloadSize {
            expected: bytes.len() - rest.len(),
            actual: bytes.len(),
        });
    }
    Ok(bricks)
}

pub struct KeyframePayload<'a> {
    pub jpeg: &'a [u8],
    pub depth: Option<Vec<f32>>,
    pub confidence: Option<Vec<u8>>,
    pub points: Vec<[f32; 3]>,
}

/// (depth bytes, confidence bytes, total payload bytes) declared by the
/// header, `None` on overflow.
fn payload_sizes(h: &KeyframeHeader) -> Option<(usize, usize, usize)> {
    let depth_len = match h.depth_size {
        None => 0,
        Some([w, d]) => (w as usize).checked_mul(d as usize)?.checked_mul(2)?,
    };
    let confidence_len = if h.depth_confidence {
        match h.depth_size {
            None => 0,
            Some([w, d]) => (w as usize).checked_mul(d as usize)?,
        }
    } else {
        0
    };
    let points_len = (h.num_points as usize).checked_mul(12)?;
    let total = (h.jpeg_len as usize)
        .checked_add(depth_len)?
        .checked_add(confidence_len)?
        .checked_add(points_len)?;
    Some((depth_len, confidence_len, total))
}

pub fn split_keyframe_payload<'a>(
    h: &KeyframeHeader,
    payload: &'a [u8],
) -> Result<KeyframePayload<'a>, ProtocolError> {
    if h.depth_confidence && h.depth_size.is_none() {
        return Err(ProtocolError::ConfidenceWithoutDepth);
    }
    let jpeg_len = h.jpeg_len as usize;
    let (depth_len, confidence_len, expected) =
        payload_sizes(h).ok_or(ProtocolError::SizeOverflow)?;
    if payload.len() != expected {
        return Err(ProtocolError::PayloadSize {
            expected,
            actual: payload.len(),
        });
    }
    let (jpeg, rest) = payload.split_at(jpeg_len);
    let (depth_bytes, rest) = rest.split_at(depth_len);
    let (confidence_bytes, point_bytes) = rest.split_at(confidence_len);
    let depth = h.depth_size.map(|_| {
        depth_bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    });
    let confidence = h.depth_confidence.then(|| confidence_bytes.to_vec());
    let points = point_bytes
        .chunks_exact(12)
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            [f(0), f(1), f(2)]
        })
        .collect();
    Ok(KeyframePayload {
        jpeg,
        depth,
        confidence,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf_header() -> KeyframeHeader {
        KeyframeHeader {
            id: 7,
            timestamp: 12.5,
            pose: [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.5, 1.0, -2.0, 1.,
            ],
            fx: 700.0,
            fy: 700.0,
            cx: 480.0,
            cy: 360.0,
            width: 960,
            height: 720,
            jpeg_len: 3,
            depth_size: Some([2, 1]),
            depth_confidence: true,
            num_points: 1,
        }
    }

    #[test]
    fn frame_roundtrip() {
        let h = ClientHeader::Keyframe(kf_header());
        let frame = encode_frame(&h, b"abc");
        let (back, payload): (ClientHeader, &[u8]) = decode_frame(&frame).unwrap();
        assert_eq!(back, h);
        assert_eq!(payload, b"abc");
    }

    #[test]
    fn header_json_uses_type_tag() {
        let frame = encode_frame(&ServerHeader::Ack { keyframe_id: 3 }, &[]);
        let len = u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&frame[4..4 + len]).unwrap();
        assert_eq!(json, r#"{"type":"ack","keyframe_id":3}"#);
    }

    #[test]
    fn truncated_and_oversized_frames_are_errors() {
        assert!(decode_frame::<ServerHeader>(&[1, 0]).is_err());
        let mut frame = encode_frame(&ServerHeader::Ack { keyframe_id: 1 }, &[]);
        frame[0] = 200;
        assert!(decode_frame::<ServerHeader>(&frame).is_err());
        let bad = encode_frame(&serde_json::json!({"type": "nope"}), &[]);
        assert!(decode_frame::<ServerHeader>(&bad).is_err());
    }

    #[test]
    fn cells_roundtrip_and_size() {
        let cells = vec![
            Cell {
                center: [1.0, -2.0, 3.5],
                coverage: 10,
                uncertainty: 250,
                age: 255,
                normal: None,
                density: 0,
            },
            Cell {
                center: [0.0, 0.0, 0.0],
                coverage: 0,
                uncertainty: 0,
                age: 0,
                normal: None,
                density: 0,
            },
        ];
        let bytes = encode_cells(&cells);
        assert_eq!(bytes.len(), 2 * CELL_BYTES);
        assert_eq!(decode_cells(&bytes).unwrap(), cells);
        assert!(decode_cells(&bytes[..16]).is_err());
    }

    #[test]
    fn keyframe_payload_split() {
        let h = kf_header();
        let mut payload = b"jpg".to_vec();
        for v in [1.5f32, 2.0] {
            payload.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        payload.extend_from_slice(&[2, 0]);
        for v in [0.1f32, 0.2, 0.3] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let p = split_keyframe_payload(&h, &payload).unwrap();
        assert_eq!(p.jpeg, b"jpg");
        assert_eq!(p.depth.unwrap(), vec![1.5, 2.0]);
        assert_eq!(p.confidence.unwrap(), vec![2, 0]);
        assert_eq!(p.points, vec![[0.1, 0.2, 0.3]]);
        assert!(split_keyframe_payload(&h, &payload[..payload.len() - 1]).is_err());
    }

    #[test]
    fn octahedral_round_trip_within_two_degrees() {
        let mut worst = 0.0f32;
        for i in 0..40 {
            for j in 0..80 {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / 40.0;
                let phi = 2.0 * std::f32::consts::PI * j as f32 / 80.0;
                let n = glam::Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let back = oct_decode(oct_encode(n));
                worst = worst.max(n.dot(back).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        for n in [
            glam::Vec3::X,
            glam::Vec3::NEG_X,
            glam::Vec3::Y,
            glam::Vec3::NEG_Y,
            glam::Vec3::Z,
            glam::Vec3::NEG_Z,
        ] {
            worst = worst.max(
                n.dot(oct_decode(oct_encode(n)))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees(),
            );
        }
        assert!(worst < 2.0, "worst error {worst}°");
    }

    #[test]
    fn cells_round_trip_with_and_without_normal() {
        let cells = [
            Cell {
                center: [1.0, -2.0, 3.5],
                coverage: 10,
                uncertainty: 250,
                age: 3,
                normal: Some([0.6, 0.0, -0.8]),
                density: 40,
            },
            Cell {
                center: [0.25, 0.5, -1.0],
                coverage: 200,
                uncertainty: 5,
                age: 30,
                normal: None,
                density: 3,
            },
        ];
        let bytes = encode_cells(&cells);
        assert_eq!(bytes.len(), 2 * CELL_BYTES);
        assert_eq!(bytes[18], CELL_FLAG_NORMAL);
        assert_eq!(bytes[CELL_BYTES + 18], 0);
        let back = decode_cells(&bytes).unwrap();
        assert_eq!(back[1], cells[1]);
        let n = glam::Vec3::from(back[0].normal.unwrap());
        assert!(n.dot(glam::Vec3::new(0.6, 0.0, -0.8)) > 0.999);
        assert_eq!((back[0].center, back[0].density), (cells[0].center, 40));
    }

    #[test]
    fn score_set_header_carries_cell_bytes() {
        let msg = crate::session::ScoreSetMsg {
            version: 1,
            based_on_keyframe_id: 2,
            voxel_size: 0.1,
            cells: vec![],
        };
        let frame = msg.to_frame();
        let (h, _): (serde_json::Value, _) = decode_frame(&frame).unwrap();
        assert_eq!(h["cell_bytes"], 19);
    }

    /// A 1 m right triangle in brick (1, −2, 0), facing +z, and the removal
    /// of brick (−1, 0, 3): the bricks of the golden fixture.
    fn fixture_bricks() -> [MeshBrick; 2] {
        let key = BrickKey(glam::IVec3::new(1, -2, 0));
        let o = glam::Vec3::new(1.0, -2.0, 0.0);
        [
            MeshBrick::Mesh(BrickMesh {
                key,
                positions: [o, o + glam::Vec3::X, o + glam::Vec3::Y]
                    .map(|p| p.to_array())
                    .to_vec(),
                normals: vec![[0.0, 0.0, 1.0]; 3],
                indices: vec![0, 1, 2],
            }),
            MeshBrick::Removed(BrickKey(glam::IVec3::new(-1, 0, 3))),
        ]
    }

    #[test]
    fn mesh_bricks_round_trip_with_removal() {
        let key = BrickKey(glam::IVec3::new(-3, 0, 2));
        let o = glam::Vec3::new(-3.0, 0.0, 2.0);
        let positions = [
            glam::Vec3::new(0.3, 0.7, 0.1),
            glam::Vec3::new(-0.025, 0.5, 1.025),
            glam::Vec3::new(1.0, 0.0, 0.5),
            glam::Vec3::new(0.123_45, 0.987_65, 0.5),
        ]
        .map(|p| (o + p).to_array());
        let normals = [
            glam::Vec3::new(0.6, 0.0, -0.8),
            glam::Vec3::Z,
            glam::Vec3::new(-0.48, 0.6, 0.64),
            glam::Vec3::NEG_Y,
        ];
        let bricks = [
            MeshBrick::Mesh(BrickMesh {
                key,
                positions: positions.to_vec(),
                normals: normals.map(|n| n.to_array()).to_vec(),
                indices: vec![0, 1, 2, 2, 1, 3],
            }),
            MeshBrick::Removed(BrickKey(glam::IVec3::new(7, -8, 9))),
        ];
        let bytes = encode_mesh_bricks(&bricks);
        assert_eq!(bytes.len(), 2 * MESH_BRICK_HEADER + 4 * 8 + 6 * 2);
        let back = decode_mesh_bricks(&bytes, 2).unwrap();
        let MeshBrick::Mesh(mesh) = &back[0] else {
            panic!("{:?}", back[0])
        };
        assert_eq!(mesh.key, key);
        assert_eq!(mesh.indices, vec![0, 1, 2, 2, 1, 3]);
        for (p, q) in positions.iter().zip(&mesh.positions) {
            let d = (glam::Vec3::from(*p) - glam::Vec3::from(*q))
                .abs()
                .max_element();
            assert!(d < 1e-5, "{p:?} vs {q:?}");
        }
        for (n, m) in normals.iter().zip(&mesh.normals) {
            assert!(n.dot(glam::Vec3::from(*m)) > 0.999, "{n} vs {m:?}");
        }
        assert!(matches!(back[1], MeshBrick::Removed(k) if k.0 == glam::IVec3::new(7, -8, 9)));
    }

    #[test]
    fn mesh_bricks_byte_layout() {
        let bytes = encode_mesh_bricks(&fixture_bricks());
        let int = |i: usize| i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let half = |i: usize| u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap());
        assert_eq!((int(0), int(4), int(8)), (1, -2, 0));
        assert_eq!(bytes[12], 0, "flags");
        assert_eq!((word(13), word(17)), (3, 3));
        // Vertex (1, 0, 0) in the brick: x at the far side of the box, y and z at its origin.
        let v1 = 21 + 6;
        let at = |local: f32| ((local + 0.025) / 1.05 * 65535.0).round() as u16;
        assert_eq!(
            (half(v1), half(v1 + 2), half(v1 + 4)),
            (at(1.0), at(0.0), at(0.0))
        );
        let normals = 21 + 18;
        assert_eq!(&bytes[normals..normals + 2], &oct_encode(glam::Vec3::Z));
        let indices = normals + 6;
        assert_eq!(
            (half(indices), half(indices + 2), half(indices + 4)),
            (0, 1, 2)
        );
        let removed = indices + 6;
        assert_eq!(
            (int(removed), int(removed + 4), int(removed + 8)),
            (-1, 0, 3)
        );
        assert_eq!(bytes[removed + 12], MESH_BRICK_REMOVED);
        assert_eq!(&bytes[removed + 13..], &[0; 8]);
    }

    #[test]
    fn mesh_bricks_reject_truncated_and_trailing_bytes() {
        let bytes = encode_mesh_bricks(&fixture_bricks());
        assert!(decode_mesh_bricks(&bytes[..bytes.len() - 1], 2).is_err());
        assert!(decode_mesh_bricks(&bytes, 1).is_err());
        assert!(decode_mesh_bricks(&bytes, 3).is_err());
    }

    #[test]
    fn mesh_bricks_header_json() {
        let frame = encode_frame(
            &ServerHeader::MeshBricks {
                version: 5,
                num_bricks: 2,
                mesh_ms: 40,
            },
            &[],
        );
        let (h, _): (serde_json::Value, _) = decode_frame(&frame).unwrap();
        assert_eq!(
            h,
            serde_json::json!({"type": "mesh_bricks", "version": 5, "num_bricks": 2, "mesh_ms": 40})
        );
    }

    #[test]
    fn decode_cells_rejects_old_15_byte_payload() {
        assert!(decode_cells(&[0u8; 15]).is_err());
    }

    #[test]
    fn header_without_confidence_field_defaults_to_false() {
        let json = serde_json::json!({
            "id": 1, "timestamp": 0.0,
            "pose": [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.],
            "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0,
            "width": 1, "height": 1, "jpeg_len": 0,
            "depth_size": null, "num_points": 0,
        });
        let h: KeyframeHeader = serde_json::from_value(json).unwrap();
        assert!(!h.depth_confidence);
    }

    #[test]
    fn confidence_without_depth_size_is_an_error() {
        let mut h = kf_header();
        h.depth_size = None;
        h.depth_confidence = true;
        assert!(matches!(
            split_keyframe_payload(&h, b"jpg"),
            Err(ProtocolError::ConfidenceWithoutDepth)
        ));
    }

    #[test]
    fn oversized_payload_sizes_are_errors_not_panics() {
        let mut h = kf_header();
        h.depth_size = Some([100_000, 100_000]);
        assert!(split_keyframe_payload(&h, b"jpg").is_err());
        h.depth_size = Some([u32::MAX, u32::MAX]);
        h.jpeg_len = u32::MAX;
        h.num_points = u32::MAX;
        assert!(split_keyframe_payload(&h, b"jpg").is_err());
    }

    /// `WRITE_FIXTURES=/abs/path cargo test -p brush-guide write_golden_fixtures`
    #[test]
    fn write_golden_fixtures() {
        let Ok(dir) = std::env::var("WRITE_FIXTURES") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cells = [
            Cell {
                center: [1.0, -2.0, 3.5],
                coverage: 10,
                uncertainty: 250,
                age: 3,
                normal: Some([0.6, 0.0, -0.8]),
                density: 40,
            },
            Cell {
                center: [0.25, 0.5, -1.0],
                coverage: 200,
                uncertainty: 5,
                age: 30,
                normal: None,
                density: 3,
            },
        ];
        let fixtures: Vec<(&str, Vec<u8>)> = vec![
            (
                "ack.bin",
                encode_frame(&ServerHeader::Ack { keyframe_id: 3 }, &[]),
            ),
            (
                "scoreset.bin",
                encode_frame(
                    &ServerHeader::ScoreSet {
                        version: 2,
                        based_on_keyframe_id: 9,
                        voxel_size: 0.1,
                        num_cells: 2,
                        cell_bytes: CELL_BYTES as u32,
                    },
                    &encode_cells(&cells),
                ),
            ),
            (
                "status.bin",
                encode_frame(
                    &ServerHeader::Status {
                        num_keyframes: 4,
                        num_splats: 1000,
                        train_iters_per_s: 55.5,
                        last_score_ms: 1200,
                    },
                    &[],
                ),
            ),
            (
                "error.bin",
                encode_frame(
                    &ServerHeader::Error {
                        message: "bad".into(),
                    },
                    &[],
                ),
            ),
            (
                "mesh_bricks.bin",
                encode_frame(
                    &ServerHeader::MeshBricks {
                        version: 2,
                        num_bricks: 2,
                        mesh_ms: 35,
                    },
                    &encode_mesh_bricks(&fixture_bricks()),
                ),
            ),
            (
                "keyframe_header.json",
                serde_json::to_vec_pretty(&ClientHeader::Keyframe(kf_header())).unwrap(),
            ),
        ];
        for (name, bytes) in fixtures {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
}
