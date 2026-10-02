//! Replays a capture export to a guidance server as keyframes and shows
//! what the server sends back in rerun.

mod dataset;
mod dump;
mod receive;
#[cfg(test)]
mod tests;
mod viz;

use brush_guide::protocol::{ClientHeader, KeyframeHeader, encode_frame};
use brush_guide::seed::project;
use clap::Parser;
use dataset::{Depth, DepthMode, Frame, Transforms, feature_points, load_depth};
use dump::ScoreDump;
use futures_util::{Sink, SinkExt, StreamExt};
use glam::{Mat4, UVec2, Vec3};
use receive::Receiver;
use std::path::PathBuf;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use viz::{AGREEMENT_LEGEND, Mode};

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
    /// Record nothing to rerun (no viewer, no .rrd); for scripted runs that
    /// read only the dumps and stderr.
    #[arg(long, conflicts_with = "save")]
    no_viz: bool,
    /// Send the export this many times in a row, each pass with new keyframe
    /// ids, so the server holds `loops` views per frame (a longer capture).
    #[arg(long, default_value_t = 1)]
    loops: usize,
    /// Whether to send the export's depth (and confidence) alongside each keyframe.
    #[arg(long, value_enum, default_value_t = DepthMode::High)]
    depth: DepthMode,
    /// Append one JSON line per received `score_set` to this file. Each cell is
    /// `[x, y, z, coverage, uncertainty, age, nx, ny, nz, density,
    /// uninformed]`, with `nx, ny, nz = 0, 0, 0` when the cell has no normal.
    #[arg(long)]
    dump_scores: Option<PathBuf>,
    /// With `--dump-scores`, skip score sets arriving less than this many
    /// seconds after the last dumped one.
    #[arg(long, default_value_t = 0.0)]
    dump_every: f32,
    /// Write every received `mesh_bricks` round to this directory: each
    /// brick's mesh as `v<version>_brick_<x>_<y>_<z>.ply` (ASCII, world
    /// metres, sRGB vertex colours when sent) and one JSON line per round in `rounds.jsonl` with its bricks
    /// (`removed` for bricks that lost their mesh), frame bytes and
    /// `mesh_ms`.
    #[arg(long)]
    dump_mesh: Option<PathBuf>,
    /// Send `finish` this many seconds after connecting (once all frames are
    /// sent), wait for the server's `splat` reply and exit. The server writes
    /// the model to `<root>/<session_id>/splat.ply`.
    #[arg(long)]
    finish_after: Option<f32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t: Transforms =
        serde_json::from_slice(&std::fs::read(args.dataset.join("transforms.json"))?)?;
    if t.frames.is_empty() {
        println!("transforms.json has no frames (empty segment); skipping");
        return Ok(());
    }
    let points = feature_points(&args.dataset, &t);
    let rec = match &args.save {
        _ if args.no_viz => rerun::RecordingStream::disabled(),
        Some(path) => rerun::RecordingStreamBuilder::new("capture-guidance-replay").save(path)?,
        None => rerun::RecordingStreamBuilder::new("capture-guidance-replay").spawn()?,
    };

    let (ws, _) = tokio_tungstenite::connect_async(&args.server).await?;
    let started = tokio::time::Instant::now();
    let (mut sink, source) = ws.split();
    let session_id = format!("replay-{}", std::process::id());
    sink.send(Message::binary(encode_frame(
        &ClientHeader::Hello {
            session_id: session_id.clone(),
            device_model: "replay".into(),
            has_lidar: false,
        },
        &[],
    )))
    .await?;

    if matches!(args.mode, Mode::Agreement) {
        rec.log_static(
            "legend",
            &rerun::TextDocument::from_markdown(AGREEMENT_LEGEND),
        )?;
    }
    let receiver = Receiver {
        rec: rec.clone(),
        mode: args.mode,
        score_dump: args
            .dump_scores
            .as_deref()
            .map(ScoreDump::open)
            .transpose()?,
        mesh_dump: args.dump_mesh.clone(),
        stop_on_splat: args.finish_after.is_some(),
        dump_every_s: args.dump_every,
        started: started.into_std(),
        last_dump: None,
    };
    let receiver = tokio::spawn(receiver.run(source));

    send_keyframes(&args, &t, &points, &mut sink, &rec).await?;
    if let Some(after) = args.finish_after {
        tokio::time::sleep_until(started + Duration::from_secs_f32(after)).await;
        sink.send(Message::binary(encode_frame(&ClientHeader::Finish, &[])))
            .await?;
        println!("finish sent; session {session_id}");
    } else {
        println!("All frames sent. Leave running to watch scores settle; Ctrl-C to quit.");
    }
    receiver.await?;
    rec.flush_blocking()?;
    Ok(())
}

/// Sends every frame of the export `args.loops` times at `args.rate` and logs
/// the camera path.
async fn send_keyframes<S>(
    args: &Args,
    t: &Transforms,
    points: &[Vec3],
    sink: &mut S,
    rec: &rerun::RecordingStream,
) -> anyhow::Result<()>
where
    S: Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let mut path_points = Vec::new();
    let mut warned_missing_conf = false;
    let frames = t.frames.iter().cycle().take(t.frames.len() * args.loops);
    for (i, f) in frames.enumerate() {
        let (header, jpeg) = keyframe_header(args, t, f, i)?;
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
            depth_size: depth.as_ref().map(|(_, _, size)| *size),
            depth_confidence: depth.as_ref().is_some_and(|(_, c, _)| c.is_some()),
            ..header
        };

        // Emulate ARKit feature points: dataset points visible in this frame.
        let camera = brush_guide::keyframe::arkit_to_camera(&header)?;
        let size = UVec2::new(header.width, header.height);
        let visible: Vec<Vec3> = points
            .iter()
            .filter(|p| project(&camera, size, **p).is_some())
            .take(args.points_per_frame)
            .copied()
            .collect();
        let header = KeyframeHeader {
            num_points: visible.len() as u32,
            ..header
        };
        let payload = keyframe_payload(jpeg, depth.as_ref(), &visible);
        sink.send(Message::binary(encode_frame(
            &ClientHeader::Keyframe(header),
            &payload,
        )))
        .await?;

        let c2w = Mat4::from_cols_array_2d(&f.transform_matrix).transpose();
        path_points.push(c2w.w_axis.truncate().to_array());
        rec.log(
            "camera_path",
            &rerun::LineStrips3D::new([path_points.clone()]),
        )?;
        tokio::time::sleep(Duration::from_secs_f32(1.0 / args.rate)).await;
    }
    Ok(())
}

/// Frame `i`'s image as JPEG, scaled to `args.long_side`, and its header
/// without depth or points.
fn keyframe_header(
    args: &Args,
    t: &Transforms,
    f: &Frame,
    i: usize,
) -> anyhow::Result<(KeyframeHeader, Vec<u8>)> {
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
        depth_confidence: false,
        num_points: 0,
    };
    Ok((header, jpeg))
}

/// JPEG, then depth and confidence if any, then the points as f32 xyz.
fn keyframe_payload(jpeg: Vec<u8>, depth: Option<&Depth>, points: &[Vec3]) -> Vec<u8> {
    let mut payload = jpeg;
    if let Some((depth_bytes, confidence, _)) = depth {
        payload.extend_from_slice(depth_bytes);
        if let Some(confidence) = confidence {
            payload.extend_from_slice(confidence);
        }
    }
    for p in points {
        for v in p.to_array() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    payload
}
