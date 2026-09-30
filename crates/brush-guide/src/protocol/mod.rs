mod cells;
mod mesh;
#[cfg(test)]
mod mesh_tests;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub use cells::{
    CELL_BYTES, CELL_FLAG_NORMAL, Cell, decode_cells, encode_cells, oct_decode, oct_encode,
};
pub use mesh::{
    BRICK_MARGIN, MESH_BRICK_REMOVED, MeshBrick, decode_mesh_bricks, encode_mesh_bricks,
};

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
