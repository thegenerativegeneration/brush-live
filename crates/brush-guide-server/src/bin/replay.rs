use brush_guide::protocol::{
    CELL_BYTES, Cell, ClientHeader, KeyframeHeader, ServerHeader, decode_cells, decode_frame,
    encode_frame,
};
use brush_guide::seed::project;
use clap::{Parser, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use glam::{Mat4, UVec2, Vec3};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

#[derive(Parser)]
struct Args {
    dataset: PathBuf,
    #[arg(long, default_value = "ws://127.0.0.1:8765")]
    server: String,
    #[arg(long, default_value_t = 2.0)]
    rate: f32,
    #[arg(long, default_value_t = 960)]
    long_side: u32,
    #[arg(long, value_enum, default_value_t = Mode::Both)]
    mode: Mode,
    #[arg(long, default_value_t = 300)]
    points_per_frame: usize,
    /// Write the rerun recording to this .rrd file instead of spawning a viewer
    /// (useful headless, where spawning a viewer process fails).
    #[arg(long)]
    save: Option<PathBuf>,
    /// Whether to send the export's depth (and confidence) alongside each keyframe.
    #[arg(long, value_enum, default_value_t = DepthMode::High)]
    depth: DepthMode,
    /// Append one JSON line per received `score_set` to this file.
    #[arg(long)]
    dump_scores: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum DepthMode {
    /// Send no depth.
    None,
    /// Send depth without confidence; the server uses every value.
    All,
    /// Send depth with confidence; the server masks low-confidence depth.
    High,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Red/yellow by coverage.
    Coverage,
    /// Red/yellow by Fisher uncertainty.
    Uncertainty,
    /// Red/yellow if either metric flags the cell.
    Both,
    /// Weak cells only, coloured by which metric flags them.
    Agreement,
}

const GREY: [u8; 3] = [150, 150, 150];
const RED: [u8; 3] = [230, 40, 40];
const YELLOW: [u8; 3] = [240, 200, 40];
const BLUE: [u8; 3] = [40, 110, 240];
const PURPLE: [u8; 3] = [180, 60, 220];

const AGREEMENT_LEGEND: &str = "Agreement mode, weak cells only:\n\n\
    * red: weak by coverage and by uncertainty\n\
    * blue: weak by coverage only\n\
    * purple: weak by uncertainty only\n\
    * grey: first seen less than 3 s ago\n\n\
    Uncertainty is ranked within each round, so its share of weak cells stays roughly \
    constant; compare where the colours sit, not how many there are.";

#[derive(Deserialize)]
struct Transforms {
    fl_x: Option<f32>,
    fl_y: Option<f32>,
    cx: Option<f32>,
    cy: Option<f32>,
    w: Option<u32>,
    h: Option<u32>,
    ply_file_path: Option<String>,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Frame {
    file_path: String,
    transform_matrix: [[f32; 4]; 4],
    fl_x: Option<f32>,
    fl_y: Option<f32>,
    cx: Option<f32>,
    cy: Option<f32>,
    w: Option<u32>,
    h: Option<u32>,
    /// Raw f16 lidar depth. Exports before the rename used `depth_file_path`, which
    /// nerfstudio-style loaders read as a depth image, so it is only a fallback.
    lidar_depth_file_path: Option<String>,
    depth_file_path: Option<String>,
    depth_w: Option<u32>,
    depth_h: Option<u32>,
}

impl Frame {
    fn depth_path(&self) -> Option<&str> {
        self.lidar_depth_file_path
            .as_deref()
            .or(self.depth_file_path.as_deref())
    }
}

/// Loads a frame's depth (and, under `High`, confidence) from the export directory.
///
/// Depth bytes are the raw little-endian f16 samples on disk, returned unchanged.
/// Confidence is one byte per pixel at the path with the depth file's extension
/// replaced by `.conf`. Missing confidence under `High` is not an error here; the
/// caller decides whether/how to warn and falls back to sending depth alone.
fn load_depth(
    dir: &Path,
    frame: &Frame,
    mode: DepthMode,
) -> anyhow::Result<Option<(Vec<u8>, Option<Vec<u8>>, [u32; 2])>> {
    if matches!(mode, DepthMode::None) {
        return Ok(None);
    }
    let (Some(path), Some(w), Some(h)) = (frame.depth_path(), frame.depth_w, frame.depth_h) else {
        return Ok(None);
    };
    let depth_bytes = std::fs::read(dir.join(path))?;
    let confidence = if matches!(mode, DepthMode::High) {
        let conf_path = Path::new(path).with_extension("conf");
        std::fs::read(dir.join(&conf_path)).ok()
    } else {
        None
    };
    Ok(Some((depth_bytes, confidence, [w, h])))
}

/// Reads an ASCII PLY whose first three vertex properties are x y z.
fn load_points(path: &Path) -> Result<Vec<Vec3>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (header, body) = text
        .split_once("end_header")
        .ok_or_else(|| format!("{} has no PLY header", path.display()))?;
    if !header.contains("format ascii") {
        return Err(format!("{} is not an ASCII PLY", path.display()));
    }
    Ok(body
        .lines()
        .filter_map(|l| {
            let v: Vec<f32> = l
                .split_whitespace()
                .take(3)
                .filter_map(|t| t.parse().ok())
                .collect();
            (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
        })
        .collect())
}

fn weak_coverage(c: &Cell) -> bool {
    c.coverage < 80
}

fn weak_uncertainty(c: &Cell) -> bool {
    c.uncertainty > 200
}

fn cell_color(c: &Cell, mode: Mode) -> Option<[u8; 3]> {
    if c.age < 3 {
        return Some(GREY);
    }
    let (wc, wu) = (weak_coverage(c), weak_uncertainty(c));
    if matches!(mode, Mode::Agreement) {
        return match (wc, wu) {
            (true, true) => Some(RED),
            (true, false) => Some(BLUE),
            (false, true) => Some(PURPLE),
            (false, false) => None,
        };
    }
    let border_cov = c.coverage < 160;
    let border_unc = c.uncertainty > 140;
    let (weak, border) = match mode {
        Mode::Coverage => (wc, border_cov),
        Mode::Uncertainty => (wu, border_unc),
        Mode::Both | Mode::Agreement => (wc || wu, border_cov || border_unc),
    };
    if weak {
        Some(RED)
    } else if border {
        Some(YELLOW)
    } else {
        None
    }
}

/// Per-round counts of settled cells (not pending) flagged weak by each metric.
fn log_agreement_counts(rec: &rerun::RecordingStream, cells: &[Cell]) {
    let settled = cells.iter().filter(|c| c.age >= 3);
    let (mut both, mut cov_only, mut unc_only) = (0u32, 0u32, 0u32);
    for c in settled {
        match (weak_coverage(c), weak_uncertainty(c)) {
            (true, true) => both += 1,
            (true, false) => cov_only += 1,
            (false, true) => unc_only += 1,
            (false, false) => {}
        }
    }
    let _ = rec.log("agreement/both", &rerun::Scalars::new(vec![both as f64]));
    let _ = rec.log(
        "agreement/coverage_only",
        &rerun::Scalars::new(vec![cov_only as f64]),
    );
    let _ = rec.log(
        "agreement/uncertainty_only",
        &rerun::Scalars::new(vec![unc_only as f64]),
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t: Transforms =
        serde_json::from_slice(&std::fs::read(args.dataset.join("transforms.json"))?)?;
    let points = match &t.ply_file_path {
        None => {
            eprintln!(
                "warning: transforms.json has no ply_file_path; sending keyframes without feature points"
            );
            Vec::new()
        }
        Some(p) => match load_points(&args.dataset.join(p)) {
            Ok(points) if points.is_empty() => {
                eprintln!("warning: {p} has no points; sending keyframes without feature points");
                points
            }
            Ok(points) => points,
            Err(e) => {
                eprintln!("warning: {e}; sending keyframes without feature points");
                Vec::new()
            }
        },
    };
    let rec = match &args.save {
        Some(path) => rerun::RecordingStreamBuilder::new("capture-guidance-replay").save(path)?,
        None => rerun::RecordingStreamBuilder::new("capture-guidance-replay").spawn()?,
    };

    let (ws, _) = tokio_tungstenite::connect_async(&args.server).await?;
    let (mut sink, mut source) = ws.split();
    let session_id = format!("replay-{}", std::process::id());
    sink.send(Message::binary(encode_frame(
        &ClientHeader::Hello {
            session_id,
            device_model: "replay".into(),
            has_lidar: false,
        },
        &[],
    )))
    .await?;

    let rec_rx = rec.clone();
    let mode = args.mode;
    if matches!(mode, Mode::Agreement) {
        rec.log_static(
            "legend",
            &rerun::TextDocument::from_markdown(AGREEMENT_LEGEND),
        )?;
    }
    let mut dump_file = match &args.dump_scores {
        Some(p) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)?,
        ),
        None => None,
    };
    let start = std::time::Instant::now();
    let receiver = tokio::spawn(async move {
        while let Some(Ok(msg)) = source.next().await {
            let Message::Binary(bytes) = msg else {
                continue;
            };
            let Ok((header, payload)) = decode_frame::<ServerHeader>(&bytes) else {
                continue;
            };
            match header {
                ServerHeader::ScoreSet {
                    version,
                    voxel_size,
                    cell_bytes,
                    ..
                } => {
                    if cell_bytes != CELL_BYTES as u32 {
                        eprintln!(
                            "server and replay cell formats differ: server cell_bytes={cell_bytes}, replay expects {CELL_BYTES}"
                        );
                        continue;
                    }
                    let cells = decode_cells(payload).unwrap_or_default();
                    if let Some(file) = dump_file.as_mut() {
                        let cells_json: Vec<[f64; 6]> = cells
                            .iter()
                            .map(|c| {
                                [
                                    c.center[0] as f64,
                                    c.center[1] as f64,
                                    c.center[2] as f64,
                                    c.coverage as f64,
                                    c.uncertainty as f64,
                                    c.age as f64,
                                ]
                            })
                            .collect();
                        let line = serde_json::json!({
                            "version": version,
                            "received_s": start.elapsed().as_secs_f64(),
                            "voxel_size": voxel_size,
                            "cells": cells_json,
                        });
                        let _ = writeln!(file, "{line}");
                    }
                    let (pos, col): (Vec<[f32; 3]>, Vec<[u8; 3]>) = cells
                        .iter()
                        .filter_map(|c| cell_color(c, mode).map(|col| (c.center, col)))
                        .unzip();
                    log_agreement_counts(&rec_rx, &cells);
                    let _ = rec_rx.log(
                        "scores",
                        &rerun::Points3D::new(pos)
                            .with_colors(col)
                            .with_radii([voxel_size * 0.4]),
                    );
                }
                ServerHeader::Status {
                    num_splats,
                    train_iters_per_s,
                    last_score_ms,
                    ..
                } => {
                    let _ = rec_rx.log(
                        "status/splats",
                        &rerun::Scalars::new(vec![num_splats as f64]),
                    );
                    let _ = rec_rx.log(
                        "status/iters_per_s",
                        &rerun::Scalars::new(vec![train_iters_per_s as f64]),
                    );
                    let _ = rec_rx.log(
                        "status/score_ms",
                        &rerun::Scalars::new(vec![last_score_ms as f64]),
                    );
                }
                ServerHeader::Error { message } => eprintln!("server error: {message}"),
                _ => {}
            }
        }
    });

    let mut path_points = Vec::new();
    let mut warned_missing_conf = false;
    for (i, f) in t.frames.iter().enumerate() {
        let img = image::open(args.dataset.join(&f.file_path))?.into_rgb8();
        let (w0, h0) = (
            f.w.or(t.w).unwrap_or(img.width()),
            f.h.or(t.h).unwrap_or(img.height()),
        );
        let scale = args.long_side as f32 / w0.max(h0) as f32;
        let (w, h) = (
            ((w0 as f32) * scale).round() as u32,
            ((h0 as f32) * scale).round() as u32,
        );
        let img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )?;

        let c2w = Mat4::from_cols_array_2d(&f.transform_matrix).transpose();
        let fl_x = f.fl_x.or(t.fl_x).ok_or_else(|| {
            anyhow::anyhow!(
                "frame {}: no fl_x in the frame or transforms.json",
                f.file_path
            )
        })?;
        let (fx, fy) = (fl_x * scale, f.fl_y.or(t.fl_y).unwrap_or(fl_x) * scale);
        let (cx, cy) = (
            f.cx.or(t.cx).unwrap_or(w0 as f32 / 2.0) * scale,
            f.cy.or(t.cy).unwrap_or(h0 as f32 / 2.0) * scale,
        );
        let depth = load_depth(&args.dataset, f, args.depth)?;
        if let Some((_, confidence, _)) = &depth {
            if args.depth == DepthMode::High && confidence.is_none() && !warned_missing_conf {
                eprintln!(
                    "warning: {}: no .conf file; sending depth without confidence (like --depth all)",
                    f.file_path
                );
                warned_missing_conf = true;
            }
        }
        let header = KeyframeHeader {
            id: i as u64,
            timestamp: i as f64 / args.rate as f64,
            pose: c2w.to_cols_array(),
            fx,
            fy,
            cx,
            cy,
            width: w,
            height: h,
            jpeg_len: jpeg.len() as u32,
            depth_size: depth.as_ref().map(|(_, _, size)| *size),
            depth_confidence: depth.as_ref().is_some_and(|(_, c, _)| c.is_some()),
            num_points: 0,
        };

        // Emulate ARKit feature points: dataset points visible in this frame.
        let camera = brush_guide::keyframe::arkit_to_camera(&header)?;
        let visible: Vec<Vec3> = points
            .iter()
            .filter(|p| project(&camera, UVec2::new(w, h), **p).is_some())
            .take(args.points_per_frame)
            .copied()
            .collect();
        let header = KeyframeHeader {
            num_points: visible.len() as u32,
            ..header
        };
        let mut payload = jpeg;
        if let Some((depth_bytes, confidence, _)) = &depth {
            payload.extend_from_slice(depth_bytes);
            if let Some(confidence) = confidence {
                payload.extend_from_slice(confidence);
            }
        }
        for p in &visible {
            for v in p.to_array() {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        sink.send(Message::binary(encode_frame(
            &ClientHeader::Keyframe(header),
            &payload,
        )))
        .await?;

        path_points.push(c2w.w_axis.truncate().to_array());
        rec.log(
            "camera_path",
            &rerun::LineStrips3D::new([path_points.clone()]),
        )?;
        tokio::time::sleep(Duration::from_secs_f32(1.0 / args.rate)).await;
    }

    println!("All frames sent. Leave running to watch scores settle; Ctrl-C to quit.");
    receiver.await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            serde_json::from_str(&format!(r#"{{{base},"depth_file_path":"depth/0.f16"}}"#))
                .unwrap();
        assert_eq!(new.depth_path(), Some("depth/0.f16"));
        assert_eq!(old.depth_path(), Some("depth/0.f16"));
    }
}
