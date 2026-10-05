mod cells;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub use cells::{
    CELL_BYTES, CELL_FLAG_NORMAL, CELL_FLAG_UNINFORMED, Cell, decode_cells, encode_cells,
    oct_decode, oct_encode,
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
    /// `[w, h]` of the mono-depth block after the feature points; `None`
    /// (absent or `null`) without one.
    #[serde(default)]
    pub mono_depth_size: Option<[u32; 2]>,
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
        #[serde(default)]
        train_iters: u64,
        #[serde(default)]
        train_ms: u64,
        #[serde(default)]
        ingest_ms: u64,
        #[serde(default)]
        preview_ms: u64,
        #[serde(default)]
        voxel_ms: u64,
        #[serde(default)]
        fisher_ms: u64,
        #[serde(default)]
        uptime_ms: u64,
        #[serde(default)]
        throttle_ms: u64,
    },
    Splat {
        ply_len: u64,
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
    /// Mono depth in metres, row-major `mono_depth_size`, 0 = invalid.
    pub mono: Option<Vec<f32>>,
}

/// Byte lengths of a keyframe payload's blocks as declared by the header.
struct PayloadSizes {
    depth: usize,
    confidence: usize,
    points: usize,
    mono: usize,
    total: usize,
}

/// Pixels of an optional `[w, h]` grid, `None` on overflow.
fn grid_pixels(size: Option<[u32; 2]>) -> Option<usize> {
    match size {
        None => Some(0),
        Some([w, h]) => (w as usize).checked_mul(h as usize),
    }
}

/// Block sizes declared by the header, `None` on overflow. Order on the
/// wire: JPEG, depth, confidence, points, mono depth.
fn payload_sizes(h: &KeyframeHeader) -> Option<PayloadSizes> {
    let depth_pixels = grid_pixels(h.depth_size)?;
    let depth = depth_pixels.checked_mul(2)?;
    let confidence = if h.depth_confidence { depth_pixels } else { 0 };
    let points = (h.num_points as usize).checked_mul(12)?;
    let mono = grid_pixels(h.mono_depth_size)?.checked_mul(2)?;
    let total = (h.jpeg_len as usize)
        .checked_add(depth)?
        .checked_add(confidence)?
        .checked_add(points)?
        .checked_add(mono)?;
    Some(PayloadSizes {
        depth,
        confidence,
        points,
        mono,
        total,
    })
}

/// The declared block sizes, after checking that the payload has exactly
/// their total length.
fn checked_sizes(h: &KeyframeHeader, payload: &[u8]) -> Result<PayloadSizes, ProtocolError> {
    let sizes = payload_sizes(h).ok_or(ProtocolError::SizeOverflow)?;
    if payload.len() != sizes.total {
        return Err(ProtocolError::PayloadSize {
            expected: sizes.total,
            actual: payload.len(),
        });
    }
    Ok(sizes)
}

fn f16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

pub fn split_keyframe_payload<'a>(
    h: &KeyframeHeader,
    payload: &'a [u8],
) -> Result<KeyframePayload<'a>, ProtocolError> {
    if h.depth_confidence && h.depth_size.is_none() {
        return Err(ProtocolError::ConfidenceWithoutDepth);
    }
    let sizes = checked_sizes(h, payload)?;
    let (jpeg, rest) = payload.split_at(h.jpeg_len as usize);
    let (depth_bytes, rest) = rest.split_at(sizes.depth);
    let (confidence_bytes, rest) = rest.split_at(sizes.confidence);
    let (point_bytes, mono_bytes) = rest.split_at(sizes.points);
    let points = point_bytes
        .chunks_exact(12)
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
            [f(0), f(1), f(2)]
        })
        .collect();
    Ok(KeyframePayload {
        jpeg,
        depth: h.depth_size.map(|_| f16_values(depth_bytes)),
        confidence: h.depth_confidence.then(|| confidence_bytes.to_vec()),
        points,
        mono: h.mono_depth_size.map(|_| f16_values(mono_bytes)),
    })
}

/// The mono-depth block's size and raw float16 bytes (the payload's last
/// block), `None` when the header declares none. Checks the payload length
/// like `split_keyframe_payload`.
pub fn mono_block<'a>(
    h: &KeyframeHeader,
    payload: &'a [u8],
) -> Result<Option<([u32; 2], &'a [u8])>, ProtocolError> {
    let sizes = checked_sizes(h, payload)?;
    Ok(h.mono_depth_size
        .map(|size| (size, &payload[payload.len() - sizes.mono..])))
}
