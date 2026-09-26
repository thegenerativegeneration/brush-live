use brush_guide::protocol::{
    ClientHeader, KeyframeHeader, ServerHeader, decode_cells, decode_frame, encode_frame,
};
use brush_guide::seed::project;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use glam::{Mat4, UVec2, Vec3};
use serde::Deserialize;
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
    #[arg(long, default_value = "both")]
    mode: String,
    #[arg(long, default_value_t = 300)]
    points_per_frame: usize,
    /// Write the rerun recording to this .rrd file instead of spawning a viewer
    /// (useful headless, where spawning a viewer process fails).
    #[arg(long)]
    save: Option<PathBuf>,
}

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

fn cell_color(c: &brush_guide::protocol::Cell, mode: &str) -> Option<[u8; 3]> {
    if c.age < 3 {
        return Some([150, 150, 150]);
    }
    let weak_cov = c.coverage < 80;
    let border_cov = c.coverage < 160;
    let weak_unc = c.uncertainty > 200;
    let border_unc = c.uncertainty > 140;
    let (weak, border) = match mode {
        "coverage" => (weak_cov, border_cov),
        "uncertainty" => (weak_unc, border_unc),
        _ => (weak_cov || weak_unc, border_cov || border_unc),
    };
    if weak {
        Some([230, 40, 40])
    } else if border {
        Some([240, 200, 40])
    } else {
        None
    }
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
    let mode = args.mode.clone();
    let receiver = tokio::spawn(async move {
        while let Some(Ok(msg)) = source.next().await {
            let Message::Binary(bytes) = msg else {
                continue;
            };
            let Ok((header, payload)) = decode_frame::<ServerHeader>(&bytes) else {
                continue;
            };
            match header {
                ServerHeader::ScoreSet { voxel_size, .. } => {
                    let cells = decode_cells(payload).unwrap_or_default();
                    let (pos, col): (Vec<[f32; 3]>, Vec<[u8; 3]>) = cells
                        .iter()
                        .filter_map(|c| cell_color(c, &mode).map(|col| (c.center, col)))
                        .unzip();
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
            depth_size: None,
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
