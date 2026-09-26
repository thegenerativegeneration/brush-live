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
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    pub center: [f32; 3],
    pub coverage: u8,
    pub uncertainty: u8,
    pub age: u8,
}

pub const CELL_BYTES: usize = 15;

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
            }
        })
        .collect())
}

pub struct KeyframePayload<'a> {
    pub jpeg: &'a [u8],
    pub depth: Option<Vec<f32>>,
    pub points: Vec<[f32; 3]>,
}

/// (depth bytes, total payload bytes) declared by the header, `None` on overflow.
fn payload_sizes(h: &KeyframeHeader) -> Option<(usize, usize)> {
    let depth_len = match h.depth_size {
        None => 0,
        Some([w, d]) => (w as usize).checked_mul(d as usize)?.checked_mul(2)?,
    };
    let points_len = (h.num_points as usize).checked_mul(12)?;
    let total = (h.jpeg_len as usize)
        .checked_add(depth_len)?
        .checked_add(points_len)?;
    Some((depth_len, total))
}

pub fn split_keyframe_payload<'a>(
    h: &KeyframeHeader,
    payload: &'a [u8],
) -> Result<KeyframePayload<'a>, ProtocolError> {
    let jpeg_len = h.jpeg_len as usize;
    let (depth_len, expected) = payload_sizes(h).ok_or(ProtocolError::SizeOverflow)?;
    if payload.len() != expected {
        return Err(ProtocolError::PayloadSize {
            expected,
            actual: payload.len(),
        });
    }
    let (jpeg, rest) = payload.split_at(jpeg_len);
    let (depth_bytes, point_bytes) = rest.split_at(depth_len);
    let depth = h.depth_size.map(|_| {
        depth_bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    });
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
            },
            Cell {
                center: [0.0, 0.0, 0.0],
                coverage: 0,
                uncertainty: 0,
                age: 0,
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
        for v in [0.1f32, 0.2, 0.3] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let p = split_keyframe_payload(&h, &payload).unwrap();
        assert_eq!(p.jpeg, b"jpg");
        assert_eq!(p.depth.unwrap(), vec![1.5, 2.0]);
        assert_eq!(p.points, vec![[0.1, 0.2, 0.3]]);
        assert!(split_keyframe_payload(&h, &payload[..payload.len() - 1]).is_err());
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
        let cells = [Cell {
            center: [1.0, -2.0, 3.5],
            coverage: 10,
            uncertainty: 250,
            age: 3,
        }];
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
                        num_cells: 1,
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
                "keyframe_header.json",
                serde_json::to_vec_pretty(&ClientHeader::Keyframe(kf_header())).unwrap(),
            ),
        ];
        for (name, bytes) in fixtures {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
}
