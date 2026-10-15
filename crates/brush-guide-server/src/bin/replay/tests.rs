use std::collections::HashSet;
use std::path::PathBuf;

use super::correct::{correction_frame, live_capture, pose};
use super::dataset::{DepthMode, Frame, Transforms, load_depth, load_mono};
use brush_guide::protocol::{ClientHeader, KeyframeHeader, decode_frame, encode_frame};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "brush-guide-replay-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn frame_with_depth(depth_file_path: Option<&str>) -> Frame {
    Frame {
        file_path: "images/0.jpg".into(),
        transform_matrix: [[0.0; 4]; 4],
        fl_x: None,
        fl_y: None,
        cx: None,
        cy: None,
        w: None,
        h: None,
        lidar_depth_file_path: depth_file_path.map(String::from),
        depth_file_path: None,
        depth_w: depth_file_path.map(|_| 2),
        depth_h: depth_file_path.map(|_| 1),
    }
}

/// Depth-mode table: none sends nothing, all sends depth without confidence, high sends both when confidence is
/// present and falls back to depth-only without it.
#[test]
fn depth_mode_controls_what_is_sent() {
    let dir = TempDir::new("none");
    let f = frame_with_depth(Some("depth/0.f16"));
    assert!(load_depth(&dir.0, &f, DepthMode::None).unwrap().is_none());

    let dir = TempDir::new("all");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    std::fs::write(dir.0.join("depth/0.conf"), [9u8, 9]).unwrap();
    let f = frame_with_depth(Some("depth/0.f16"));
    let (depth, confidence, size) = load_depth(&dir.0, &f, DepthMode::All)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(confidence, None, "all mode ignores confidence");
    assert_eq!(size, [2, 1]);

    let dir = TempDir::new("high-with-conf");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    std::fs::write(dir.0.join("depth/0.conf"), [2u8, 1]).unwrap();
    let f = frame_with_depth(Some("depth/0.f16"));
    let (depth, confidence, size) = load_depth(&dir.0, &f, DepthMode::High)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(confidence, Some(vec![2, 1]));
    assert_eq!(size, [2, 1]);

    let dir = TempDir::new("high-no-conf");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    let f = frame_with_depth(Some("depth/0.f16"));
    let (depth, confidence, size) = load_depth(&dir.0, &f, DepthMode::High)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(confidence, None, "high falls back to depth-only");
    assert_eq!(size, [2, 1]);
}

/// Depth-key table: the legacy key is read when the new key is absent, the new key wins when both are present, and
/// `Frame` JSON parses either key name.
#[test]
fn depth_key_prefers_the_new_name_over_the_legacy_one() {
    let dir = TempDir::new("legacy-key");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    let mut f = frame_with_depth(Some("depth/0.f16"));
    f.depth_file_path = f.lidar_depth_file_path.take();
    let (depth, _, size) = load_depth(&dir.0, &f, DepthMode::All)
        .unwrap()
        .expect("depth present via depth_file_path");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(size, [2, 1]);

    let dir = TempDir::new("both-keys");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    std::fs::write(dir.0.join("depth/old.f16"), [7u8, 7, 7, 7]).unwrap();
    let mut f = frame_with_depth(Some("depth/0.f16"));
    f.depth_file_path = Some("depth/old.f16".into());
    let (depth, _, _) = load_depth(&dir.0, &f, DepthMode::All)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);

    let base = r#""file_path":"images/0.jpg","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"depth_w":2,"depth_h":1"#;
    let new: Frame = serde_json::from_str(&format!(
        r#"{{{base},"lidar_depth_file_path":"depth/0.f16"}}"#
    ))
    .unwrap();
    let old: Frame =
        serde_json::from_str(&format!(r#"{{{base},"depth_file_path":"depth/0.f16"}}"#)).unwrap();
    assert_eq!(new.depth_path(), Some("depth/0.f16"));
    assert_eq!(old.depth_path(), Some("depth/0.f16"));
}

fn wire_header(mono: Option<[u32; 2]>) -> KeyframeHeader {
    KeyframeHeader {
        id: 0,
        timestamp: 0.0,
        pose: glam::Mat4::IDENTITY.to_cols_array(),
        fx: 1.0,
        fy: 1.0,
        cx: 1.0,
        cy: 1.0,
        width: 2,
        height: 2,
        jpeg_len: 3,
        depth_size: None,
        depth_confidence: false,
        num_points: 1,
        mono_depth_size: mono,
    }
}

fn write_wire(dir: &std::path::Path, header: KeyframeHeader, tail: &[u8]) {
    std::fs::create_dir_all(dir.join("wire")).unwrap();
    let mut payload = b"jpg".to_vec();
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(tail);
    std::fs::write(
        dir.join("wire/0.bin"),
        encode_frame(&ClientHeader::Keyframe(header), &payload),
    )
    .unwrap();
}

#[test]
fn load_mono_copies_the_block_from_the_wire_file() {
    let dir = TempDir::new("mono");
    let block = vec![1u8, 2, 3, 4, 5, 6];
    write_wire(&dir.0, wire_header(Some([3, 1])), &block);
    let f = frame_with_depth(None);
    let (size, bytes) = load_mono(&dir.0, &f).unwrap().expect("block present");
    assert_eq!(size, [3, 1]);
    assert_eq!(bytes, block);
}

#[test]
fn load_mono_is_none_without_a_wire_file_or_a_block() {
    let dir = TempDir::new("no-mono");
    let f = frame_with_depth(None);
    assert!(load_mono(&dir.0, &f).unwrap().is_none(), "no wire folder");
    write_wire(&dir.0, wire_header(None), &[]);
    assert!(
        load_mono(&dir.0, &f).unwrap().is_none(),
        "frame without a block"
    );
    write_wire(&dir.0, wire_header(Some([3, 1])), &[1, 2]);
    assert!(
        load_mono(&dir.0, &f).is_err(),
        "declared block missing bytes"
    );
}

#[test]
fn load_mono_rejects_out_of_range_sides() {
    let dir = TempDir::new("mono-range");
    let f = frame_with_depth(None);
    write_wire(&dir.0, wire_header(Some([0, 1])), &[]);
    assert!(load_mono(&dir.0, &f).is_err(), "zero side");
    write_wire(&dir.0, wire_header(Some([1025, 1])), &vec![0u8; 2050]);
    assert!(load_mono(&dir.0, &f).is_err(), "side above 1024");
}

#[test]
fn keyframe_payload_puts_the_mono_block_last_and_leaves_frames_without_it_unchanged() {
    let points = [glam::Vec3::new(0.1, 0.2, 0.3)];
    let plain = crate::prepare::keyframe_payload(b"jpg".to_vec(), None, &points, None);
    let mut expected = b"jpg".to_vec();
    for v in [0.1f32, 0.2, 0.3] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(plain, expected);
    let with = crate::prepare::keyframe_payload(b"jpg".to_vec(), None, &points, Some(&[9, 8]));
    assert_eq!(&with[..plain.len()], &plain[..]);
    assert_eq!(&with[plain.len()..], &[9, 8]);
}

fn transforms(frames: Vec<Frame>) -> Transforms {
    Transforms {
        fl_x: Some(500.0),
        fl_y: None,
        cx: None,
        cy: None,
        w: Some(640),
        h: Some(480),
        ply_file_path: None,
        frames,
    }
}

fn frame_at(file: &str, x: f32) -> Frame {
    Frame {
        file_path: file.into(),
        transform_matrix: [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        ..frame_with_depth(None)
    }
}

/// The correction names every keyframe sent so far, ids cycling over the frames, with the refined poses.
#[test]
fn correction_frame_carries_refined_poses_of_sent_keyframes() {
    let t = transforms(vec![
        frame_at("images/0.jpg", 0.5),
        frame_at("images/1.jpg", 1.5),
    ]);
    let frame = correction_frame(&t, 3);
    let (header, payload) = decode_frame::<ClientHeader>(&frame).unwrap();
    assert!(payload.is_empty());
    let ClientHeader::Poses { poses } = header else {
        panic!("{header:?}")
    };
    assert_eq!(
        poses.iter().map(|p| p.id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(poses[2].pose, pose(&t.frames[0].transform_matrix));
    assert_eq!(poses[1].pose[12], 1.5);
}

/// Live poses come from transforms_live.json by position; a file listing other images is an error.
#[test]
fn live_poses_must_list_the_same_frames() {
    let dir = TempDir::new("live");
    let t = transforms(vec![frame_at("images/0.jpg", 0.5)]);
    let live = r#"{"fl_x":500,"frames":[{"file_path":"images/0.jpg","transform_matrix":[[1,0,0,0.2],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}"#;
    std::fs::write(dir.0.join("transforms_live.json"), live).unwrap();
    assert_eq!(live_capture(&dir.0, &t).unwrap().poses[0][12], 0.2);
    let other = transforms(vec![frame_at("images/9.jpg", 0.5)]);
    assert!(live_capture(&dir.0, &other).is_err());
}

/// A drifted frame of a refined export: its live pose sits 0.3 m right of the refined one, and points.ply (on the
/// live poses) holds the surface point 2 m ahead of the live camera. The replay sends the point unchanged, and the
/// keyframe's camera sees it at the image centre 2 m deep, where that frame's depth seeds the same surface: feature
/// points, depth and pose agree. The refined camera would see it 75 px off.
#[test]
fn live_capture_keeps_feature_points_on_the_live_poses() {
    let dir = TempDir::new("live-points");
    let t = transforms(vec![frame_at("images/0.jpg", 0.0)]);
    let live = r#"{"fl_x":500,"ply_file_path":"points.ply","frames":[{"file_path":"images/0.jpg","transform_matrix":[[1,0,0,0.3],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}"#;
    std::fs::write(dir.0.join("transforms_live.json"), live).unwrap();
    let ply = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.3 0 -2\n";
    std::fs::write(dir.0.join("points.ply"), ply).unwrap();
    let surface = glam::Vec3::new(0.3, 0.0, -2.0);

    let capture = live_capture(&dir.0, &t).unwrap();
    assert_eq!(capture.points, vec![surface]);
    let header = |pose: [f32; 16]| KeyframeHeader {
        pose,
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        width: 640,
        height: 480,
        ..wire_header(None)
    };
    let live_header = header(capture.poses[0]);
    assert_eq!(
        crate::prepare::visible_points(&live_header, &capture.points, 10).unwrap(),
        vec![surface]
    );
    let size = glam::UVec2::new(live_header.width, live_header.height);
    let centre = glam::Vec2::new(live_header.cx, live_header.cy);
    let seen = |h: &KeyframeHeader| {
        let camera = brush_guide::keyframe::arkit_to_camera(h).unwrap();
        brush_guide::seed::project(&camera, size, surface).unwrap()
    };
    let (px, depth) = seen(&live_header);
    assert!(
        (px - centre).length() < 1e-3 && (depth - 2.0).abs() < 1e-5,
        "{px} at {depth} m"
    );
    let (px, _) = seen(&header(pose(&t.frames[0].transform_matrix)));
    assert!(
        (px - centre).length() > 70.0,
        "the refined camera sees it at {px}"
    );
}

#[test]
fn send_order_walks_back_without_the_last_frame_and_skips_forward_held_frames() {
    use crate::order::send_order;
    let none = HashSet::new();
    assert_eq!(
        send_order(3, 1, false, None, &none),
        vec![(0, 0), (1, 1), (2, 2)]
    );
    assert_eq!(
        send_order(5, 1, true, None, &none),
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 3),
            (6, 2),
            (7, 1),
            (8, 0)
        ]
    );
    // Forward frames 1 and 3 are held out with holdout_every 2; the walk back leaves them out.
    let back: Vec<_> = send_order(5, 1, true, Some(2), &none)
        .into_iter()
        .skip(5)
        .collect();
    assert_eq!(back, vec![(6, 2), (8, 0)]);
    // holdout_every 1 is off on the server, so nothing is skipped.
    assert_eq!(
        send_order(5, 1, true, Some(1), &none),
        send_order(5, 1, true, None, &none)
    );
    // After two loops the return ids start above every forward id.
    let two = send_order(3, 2, true, None, &none);
    assert_eq!(&two[6..], &[(6, 1), (7, 0)]);
}

#[test]
fn dropped_frames_are_never_sent_and_held_frames_follow_the_send_count() {
    use crate::order::send_order;
    let dropped: HashSet<usize> = [2, 3].into();
    let order = send_order(8, 1, true, Some(2), &dropped);
    // Forward ids stay frame indices; frames 1, 5 and 7 go out as the 2nd, 4th and 6th keyframe (held out with
    // holdout_every 2), so the walk back leaves them out along with the dropped ones.
    assert_eq!(
        order,
        vec![
            (0, 0),
            (1, 1),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 6),
            (10, 4),
            (14, 0)
        ]
    );
}

#[test]
fn drop_ranges_parse() {
    use crate::order::parse_frame_range;
    assert_eq!(parse_frame_range("485-520"), Ok(485..=520));
    assert_eq!(parse_frame_range("7"), Ok(7..=7));
    assert!(parse_frame_range("9-3").is_err());
    assert!(parse_frame_range("a-3").is_err());
}

#[test]
fn keyframes_go_out_on_a_fixed_schedule() {
    use crate::order::send_time;
    use std::time::Duration;
    assert_eq!(send_time(0, 2.0), Duration::ZERO);
    assert_eq!(send_time(3, 2.0), Duration::from_millis(1500));
    assert_eq!(send_time(10, 4.0), Duration::from_millis(2500));
}
