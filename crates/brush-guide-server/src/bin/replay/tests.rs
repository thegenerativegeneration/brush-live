use std::path::PathBuf;

use super::dataset::{DepthMode, Frame, load_depth};

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
