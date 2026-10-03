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
fn uninformed_is_flags_bit_1_next_to_the_normal_bit() {
    let cell = Cell {
        center: [0.5; 3],
        coverage: 0,
        uncertainty: 255,
        age: 3,
        normal: Some([0.0, 0.0, 1.0]),
        density: 40,
        uninformed: true,
    };
    let bytes = encode_cells(&[cell]);
    assert_eq!(bytes[18], CELL_FLAG_NORMAL | CELL_FLAG_UNINFORMED);
    let back = decode_cells(&bytes).unwrap()[0];
    assert!(back.uninformed && back.normal.is_some());
    let plain = encode_cells(&[Cell {
        uninformed: false,
        normal: None,
        ..cell
    }]);
    assert_eq!(
        plain[18], 0,
        "an informed cell without a normal keeps flags 0"
    );
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
fn cells_roundtrip_with_and_without_a_normal() {
    let cells = [
        Cell {
            center: [1.0, -2.0, 3.5],
            coverage: 10,
            uncertainty: 250,
            age: 3,
            normal: Some([0.6, 0.0, -0.8]),
            density: 40,
            uninformed: false,
        },
        Cell {
            center: [0.25, 0.5, -1.0],
            coverage: 200,
            uncertainty: 5,
            age: 30,
            normal: None,
            density: 3,
            uninformed: false,
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

    assert!(decode_cells(&bytes[..16]).is_err(), "truncated payload");
    assert!(decode_cells(&[0u8; 15]).is_err(), "old 15-byte payload");
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
            uninformed: false,
        },
        Cell {
            center: [0.25, 0.5, -1.0],
            coverage: 200,
            uncertainty: 5,
            age: 30,
            normal: None,
            density: 3,
            uninformed: false,
        },
        Cell {
            center: [-0.5, 1.5, 2.0],
            coverage: 0,
            uncertainty: 255,
            age: 1,
            normal: Some([0.0, 0.0, 1.0]),
            density: 12,
            uninformed: true,
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
                    num_cells: cells.len() as u32,
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
            "keyframe_header.json",
            serde_json::to_vec_pretty(&ClientHeader::Keyframe(kf_header())).unwrap(),
        ),
    ];
    for (name, bytes) in fixtures {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
}
