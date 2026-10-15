//! Building each keyframe frame from the export: image, depth, feature points and mono depth. Runs on its own
//! thread ahead of the sender, so preparing keyframe k overlaps the wait before it goes out.

use crate::dataset::{Depth, DepthMode, Frame, MonoMode, Transforms, load_depth, load_mono};
use brush_guide::protocol::{ClientHeader, KeyframeHeader, encode_frame};
use brush_guide::seed::project;
use glam::{Mat4, UVec2, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// The replay arguments keyframe preparation reads.
#[derive(Clone)]
pub(crate) struct PrepOptions {
    pub(crate) dataset: PathBuf,
    pub(crate) long_side: u32,
    /// Keyframes per second; stamps keyframe k with `k / rate` s.
    pub(crate) rate: f32,
    pub(crate) depth: DepthMode,
    pub(crate) mono: MonoMode,
    pub(crate) points_per_frame: usize,
}

/// What the export holds for preparing keyframes, shared with the preparing thread.
pub(crate) struct Source {
    pub(crate) t: Transforms,
    /// Capture-time poses per frame, sent instead of the refined ones.
    pub(crate) live_poses: Option<Vec<[f32; 16]>>,
    pub(crate) points: Vec<Vec3>,
}

/// One encoded keyframe frame, ready to send.
pub(crate) struct Prepared {
    /// Index of the frame in transforms.json.
    pub(crate) frame: usize,
    pub(crate) bytes: Vec<u8>,
    pub(crate) carried_mono: bool,
}

/// Prepares the keyframes of `order` ((keyframe id, frame index) in send order) on a new thread into a channel
/// that holds at most `ahead` prepared ones; the first error ends the channel after it.
pub(crate) fn spawn_prepare(
    opts: PrepOptions,
    source: Arc<Source>,
    order: Vec<(u64, usize)>,
    ahead: usize,
) -> mpsc::Receiver<anyhow::Result<Prepared>> {
    let (tx, rx) = mpsc::channel(ahead.max(1));
    std::thread::spawn(move || {
        let mut warned_missing_conf = false;
        for (k, (id, frame)) in order.into_iter().enumerate() {
            let prepared = prepare(&opts, &source, k, id, frame, &mut warned_missing_conf);
            let failed = prepared.is_err();
            if tx.blocking_send(prepared).is_err() || failed {
                break;
            }
        }
    });
    rx
}

/// The `k`-th keyframe sent (keyframe `id`, frame `frame`) as a keyframe frame. Warns once when depth goes out
/// without confidence under `--depth high`.
fn prepare(
    opts: &PrepOptions,
    source: &Source,
    k: usize,
    id: u64,
    frame: usize,
    warned_missing_conf: &mut bool,
) -> anyhow::Result<Prepared> {
    let f = &source.t.frames[frame];
    let (header, jpeg) = keyframe_header(opts, &source.t, f, id, k)?;
    let header = match &source.live_poses {
        Some(poses) => KeyframeHeader {
            pose: poses[frame],
            ..header
        },
        None => header,
    };
    let depth = load_depth(&opts.dataset, f, opts.depth)?;
    if let Some((_, confidence, _)) = &depth
        && opts.depth == DepthMode::High
        && confidence.is_none()
        && !*warned_missing_conf
    {
        eprintln!(
            "warning: {}: no .conf file; sending depth without confidence (like --depth all)",
            f.file_path
        );
        *warned_missing_conf = true;
    }
    let header = KeyframeHeader {
        depth_size: depth.as_ref().map(|(_, _, size)| *size),
        depth_confidence: depth.as_ref().is_some_and(|(_, c, _)| c.is_some()),
        ..header
    };
    let visible = visible_points(&header, &source.points, opts.points_per_frame)?;
    let mono = match opts.mono {
        MonoMode::Off => None,
        MonoMode::On => load_mono(&opts.dataset, f).unwrap_or_else(|e| {
            eprintln!("warning: {}: mono depth skipped: {e}", f.file_path);
            None
        }),
    };
    let header = KeyframeHeader {
        num_points: visible.len() as u32,
        mono_depth_size: mono.as_ref().map(|(size, _)| *size),
        ..header
    };
    let payload = keyframe_payload(
        jpeg,
        depth.as_ref(),
        &visible,
        mono.as_ref().map(|(_, block)| block.as_slice()),
    );
    Ok(Prepared {
        frame,
        bytes: encode_frame(&ClientHeader::Keyframe(header), &payload),
        carried_mono: mono.is_some(),
    })
}

/// Emulates ARKit feature points: up to `max` of `points` that the keyframe's camera (its header pose) sees.
pub(crate) fn visible_points(
    header: &KeyframeHeader,
    points: &[Vec3],
    max: usize,
) -> anyhow::Result<Vec<Vec3>> {
    let camera = brush_guide::keyframe::arkit_to_camera(header)?;
    let size = UVec2::new(header.width, header.height);
    Ok(points
        .iter()
        .filter(|p| project(&camera, size, **p).is_some())
        .take(max)
        .copied()
        .collect())
}

/// Frame `f`'s image as JPEG, scaled to `opts.long_side`, and its header (keyframe `id`, the `send_index`-th
/// keyframe sent) without depth or points.
fn keyframe_header(
    opts: &PrepOptions,
    t: &Transforms,
    f: &Frame,
    id: u64,
    send_index: usize,
) -> anyhow::Result<(KeyframeHeader, Vec<u8>)> {
    let img = image::open(opts.dataset.join(&f.file_path))?.into_rgb8();
    let (w0, h0) = (
        f.w.or(t.w).unwrap_or(img.width()),
        f.h.or(t.h).unwrap_or(img.height()),
    );
    let scale = opts.long_side as f32 / w0.max(h0) as f32;
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
        id,
        timestamp: send_index as f64 / opts.rate as f64,
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
        mono_depth_size: None,
    };
    Ok((header, jpeg))
}

/// JPEG, then depth and confidence if any, then the points as f32 xyz, then
/// the mono-depth block if any.
pub(crate) fn keyframe_payload(
    jpeg: Vec<u8>,
    depth: Option<&Depth>,
    points: &[Vec3],
    mono: Option<&[u8]>,
) -> Vec<u8> {
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
    if let Some(mono) = mono {
        payload.extend_from_slice(mono);
    }
    payload
}
