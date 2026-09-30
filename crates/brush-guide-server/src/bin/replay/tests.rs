use brush_guide::geometry::mesh::BrickMesh;
use brush_guide::geometry::tsdf::BrickKey;
use brush_guide::protocol::MeshBrick;
use std::path::PathBuf;

use super::dataset::{DepthMode, Frame, load_depth};
use super::receive::handle_mesh_bricks;

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

#[test]
fn mesh_dump_writes_ply_and_round_line() {
    let dir = TempDir::new("mesh-dump");
    let key = BrickKey(glam::IVec3::new(1, -2, 0));
    let bricks = [
        MeshBrick::Mesh(BrickMesh {
            key,
            positions: vec![[1.0, -2.0, 0.0], [2.0, -2.0, 0.0], [1.0, -1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
        }),
        MeshBrick::Removed(BrickKey(glam::IVec3::new(-1, 0, 3))),
    ];
    let rec = rerun::RecordingStream::disabled();
    let line = handle_mesh_bricks(&rec, Some(&dir.0), 7, 12, 100, &bricks).unwrap();
    assert!(
        line.contains("2 bricks (1 removed, 1 triangles), 100 bytes"),
        "{line}"
    );
    let ply = std::fs::read_to_string(dir.0.join("v00007_brick_1_-2_0.ply")).unwrap();
    assert!(
        ply.contains("element vertex 3\n") && ply.ends_with("3 0 1 2\n"),
        "{ply}"
    );
    let rounds = std::fs::read_to_string(dir.0.join("rounds.jsonl")).unwrap();
    let round: serde_json::Value = serde_json::from_str(rounds.trim()).unwrap();
    assert_eq!(round["version"], 7);
    assert_eq!(round["bricks"][1]["removed"], true);
}

#[test]
fn none_mode_sends_no_depth() {
    let dir = TempDir::new("none");
    let f = frame_with_depth(Some("depth/0.f16"));
    assert!(load_depth(&dir.0, &f, DepthMode::None).unwrap().is_none());
}

#[test]
fn all_mode_sends_depth_without_confidence() {
    let dir = TempDir::new("all");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    std::fs::write(dir.0.join("depth/0.conf"), [9u8, 9]).unwrap();
    let f = frame_with_depth(Some("depth/0.f16"));
    let (depth, confidence, size) = load_depth(&dir.0, &f, DepthMode::All)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(confidence, None);
    assert_eq!(size, [2, 1]);
}

#[test]
fn high_mode_with_confidence_sends_both() {
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
}

#[test]
fn high_mode_without_confidence_falls_back_to_depth_only() {
    let dir = TempDir::new("high-no-conf");
    std::fs::create_dir_all(dir.0.join("depth")).unwrap();
    std::fs::write(dir.0.join("depth/0.f16"), [0u8, 1, 2, 3]).unwrap();
    let f = frame_with_depth(Some("depth/0.f16"));
    let (depth, confidence, size) = load_depth(&dir.0, &f, DepthMode::High)
        .unwrap()
        .expect("depth present");
    assert_eq!(depth, vec![0, 1, 2, 3]);
    assert_eq!(confidence, None);
    assert_eq!(size, [2, 1]);
}

#[test]
fn legacy_depth_key_is_read_when_the_new_key_is_absent() {
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
}

#[test]
fn new_depth_key_wins_over_the_legacy_key() {
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
}

#[test]
fn frame_json_accepts_either_depth_key() {
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
