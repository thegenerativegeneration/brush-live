//! Replays a capture export to a guidance server as keyframes and shows
//! what the server sends back in rerun.

mod correct;
mod dataset;
mod dump;
mod order;
mod pace;
mod prepare;
mod receive;
#[cfg(test)]
mod tests;
mod viz;

use brush_guide::protocol::{ClientHeader, encode_frame};
use clap::Parser;
use dataset::{DepthMode, MonoMode, Transforms, feature_points};
use dump::ScoreDump;
use futures_util::{Sink, SinkExt, StreamExt};
use glam::Mat4;
use pace::{Lockstep, Pace, Paced};
use prepare::{PrepOptions, Source, spawn_prepare};
use receive::Receiver;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use viz::{AGREEMENT_LEGEND, Mode};

/// Keyframes prepared ahead of the one waiting to go out.
const PREPARE_AHEAD: usize = 4;
/// A lockstep replay gives up when the server's step count stands still this long.
const LOCKSTEP_STALL: Duration = Duration::from_secs(300);

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
    /// Whether to copy the phone's mono-depth block from `wire/<id>.bin` into
    /// each keyframe that has one. The server's feature-point scale fallback
    /// then sees this tool's feature points (from `points.ply`), not the
    /// phone's per-frame points, so that fallback is not reproduced exactly.
    #[arg(long, value_enum, default_value_t = MonoMode::On)]
    mono: MonoMode,
    /// Append one JSON line per received `score_set` to this file. Each cell is
    /// `[x, y, z, coverage, uncertainty, age, nx, ny, nz, density,
    /// uninformed]`, with `nx, ny, nz = 0, 0, 0` when the cell has no normal.
    #[arg(long)]
    dump_scores: Option<PathBuf>,
    /// With `--dump-scores`, skip score sets arriving less than this many
    /// seconds after the last dumped one.
    #[arg(long, default_value_t = 0.0)]
    dump_every: f32,
    /// Send `finish` this many seconds after connecting (once all frames are
    /// sent), wait for the server's `splat` reply and exit. The server writes
    /// the model to `<root>/<session_id>/splat.ply`.
    #[arg(long)]
    finish_after: Option<f32>,
    /// Like `--finish-after`, but this many seconds after the last keyframe was sent.
    #[arg(long, conflicts_with = "finish_after")]
    tail: Option<f32>,
    /// Send each keyframe with the pose the phone sent during capture (`transforms_live.json`) instead of the
    /// refined one in `transforms.json`.
    #[arg(long)]
    live_poses: bool,
    /// After this many keyframes, send one `poses` frame with the refined poses (`transforms.json`) of every
    /// keyframe sent so far, as the phone's engine gets them after a loop closure.
    #[arg(long, requires = "live_poses")]
    correct_after: Option<usize>,
    /// After the last frame, walk back: send the frames again in reverse order (without the last one) under new
    /// ids, like a capture that returns to its start.
    #[arg(long = "return", conflicts_with = "live_poses")]
    return_pass: bool,
    /// With `--return`, leave out frames `j` with `(j + 1) % N == 0`: the forward pass's held-out frames when N is
    /// the server's `holdout_every`, so their images are never trained on.
    #[arg(long, requires = "return_pass")]
    return_skip_every: Option<usize>,
    /// Never send these frames of transforms.json, forward or back: inclusive `A-B` ranges or single indices,
    /// comma-separated or repeated. The other frames keep their ids.
    #[arg(long, value_delimiter = ',', value_parser = order::parse_frame_range)]
    drop_frames: Vec<std::ops::RangeInclusive<usize>>,
    /// Lockstep: send the k-th keyframe (0-based, in send order) only once the server has trained round(k · K)
    /// steps, K = this value, instead of on the --rate schedule. The steps per keyframe then no longer depend on
    /// how fast the machine trains. The server must run with `status_every_step` (status after every step).
    #[arg(long, value_parser = positive_f32)]
    steps_per_keyframe: Option<f32>,
    /// With --steps-per-keyframe: send `finish` once the server has trained this many steps beyond the last
    /// keyframe's target plus K (n · K + N for n keyframes), wait for the server's `splat` reply and exit.
    #[arg(long, requires = "steps_per_keyframe", conflicts_with_all = ["tail", "finish_after"])]
    tail_steps: Option<u64>,
}

fn positive_f32(s: &str) -> Result<f32, String> {
    match s.parse::<f32>() {
        Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(format!("{s}: expected a number above 0")),
    }
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
    // Live-pose replays send what the phone sent: live poses and the export's feature points, which stay on the live
    // poses (see `correct`).
    let (live_poses, points) = if args.live_poses {
        let live = correct::live_capture(&args.dataset, &t)?;
        (Some(live.poses), live.points)
    } else {
        (None, feature_points(&args.dataset, &t))
    };
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
    let (steps_tx, steps_rx) = tokio::sync::watch::channel(0u64);
    let receiver = Receiver {
        rec: rec.clone(),
        mode: args.mode,
        score_dump: args
            .dump_scores
            .as_deref()
            .map(|p| ScoreDump::open(p, started.into_std()))
            .transpose()?,
        stop_on_splat: args.finish_after.is_some()
            || args.tail.is_some()
            || args.tail_steps.is_some(),
        dump_every_s: args.dump_every,
        started: started.into_std(),
        last_dump: None,
        skipped: None,
        steps: args.steps_per_keyframe.map(|_| steps_tx),
        last_status_print: None,
    };
    let receiver = tokio::spawn(receiver.run(source));

    let pace = match args.steps_per_keyframe {
        Some(k) => Pace::Lockstep(Lockstep::new(steps_rx, k, LOCKSTEP_STALL)),
        None => Pace::Wall {
            started,
            rate: args.rate,
        },
    };
    let source = Arc::new(Source {
        t,
        live_poses,
        points,
    });
    let (sent, mut pace) = send_keyframes(&args, &source, pace, &mut sink, &rec).await?;
    let finish_at = match (args.tail, args.finish_after) {
        (Some(tail), _) => Some(tokio::time::Instant::now() + Duration::from_secs_f32(tail)),
        (None, Some(after)) => Some(started + Duration::from_secs_f32(after)),
        (None, None) => None,
    };
    let finish = match (&mut pace, args.tail_steps) {
        (Pace::Lockstep(lock), Some(tail)) => {
            let target = pace::step_target(sent, lock.per_keyframe()) + tail;
            let at = lock.wait_for(target).await?;
            println!("tail: finish at {at} steps (target {target})");
            true
        }
        _ => match finish_at {
            Some(at) => {
                tokio::time::sleep_until(at).await;
                true
            }
            None => false,
        },
    };
    if finish {
        sink.send(Message::binary(encode_frame(&ClientHeader::Finish, &[])))
            .await?;
        println!(
            "[{:8.2} s] finish sent; session {session_id}",
            started.elapsed().as_secs_f64()
        );
    } else {
        println!("All frames sent. Leave running to watch scores settle; Ctrl-C to quit.");
    }
    receiver.await?;
    rec.flush_blocking()?;
    Ok(())
}

/// Sends the frames in [`order::send_order`] (the export `args.loops` times, then the walk back with `--return`,
/// without `--drop-frames`), each when `pace` lets it go, and logs the camera path. Keyframes are prepared on
/// another thread up to [`PREPARE_AHEAD`] ahead. Returns how many went out; a lockstep replay prints how many steps
/// the server trained per keyframe.
async fn send_keyframes<S>(
    args: &Args,
    source: &Arc<Source>,
    pace: Pace,
    sink: &mut S,
    rec: &rerun::RecordingStream,
) -> anyhow::Result<(usize, Pace)>
where
    S: Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let t = &source.t;
    let dropped: std::collections::HashSet<usize> =
        args.drop_frames.iter().cloned().flatten().collect();
    if !dropped.is_empty() {
        let outside = dropped.iter().filter(|&&j| j >= t.frames.len()).count();
        println!(
            "dropping {} frames{}",
            dropped.len() - outside,
            if outside > 0 {
                format!(" ({outside} given indices lie past the last frame)")
            } else {
                String::new()
            }
        );
    }
    let order = order::send_order(
        t.frames.len(),
        args.loops,
        args.return_pass,
        args.return_skip_every,
        &dropped,
    );
    let opts = PrepOptions {
        dataset: args.dataset.clone(),
        long_side: args.long_side,
        rate: args.rate,
        depth: args.depth,
        mono: args.mono,
        points_per_frame: args.points_per_frame,
    };
    let mut paced = Paced::new(
        spawn_prepare(opts, source.clone(), order, PREPARE_AHEAD),
        pace,
    );
    let mut path_points = Vec::new();
    let mut mono_frames = 0usize;
    while let Some(kf) = paced.next().await {
        let kf = kf??;
        mono_frames += usize::from(kf.carried_mono);
        sink.send(Message::binary(kf.bytes)).await?;
        if args.correct_after == Some(paced.sent()) {
            sink.send(Message::binary(correct::correction_frame(t, paced.sent())))
                .await?;
            println!("poses sent for {} keyframes", paced.sent());
        }
        let c2w = Mat4::from_cols_array_2d(&t.frames[kf.frame].transform_matrix).transpose();
        path_points.push(c2w.w_axis.truncate().to_array());
        rec.log(
            "camera_path",
            &rerun::LineStrips3D::new([path_points.clone()]),
        )?;
    }
    if args.mono == MonoMode::On {
        println!("{mono_frames} keyframes carried the phone's mono depth");
    }
    let sent = paced.sent();
    let pace = paced.into_pace();
    if let Pace::Lockstep(lock) = &pace {
        let s = lock.summary();
        println!(
            "lockstep: {} keyframes at {} steps each: {:.2} steps per keyframe (min {}, max {}), {} late",
            s.keyframes,
            lock.per_keyframe(),
            s.mean,
            s.min,
            s.max,
            s.late
        );
        if s.late * 10 > s.keyframes {
            eprintln!(
                "warning: over a tenth of the keyframes went out late; does the server run with status_every_step?"
            );
        }
    }
    Ok((sent, pace))
}
