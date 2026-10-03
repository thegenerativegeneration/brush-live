//! Spike (throwaway): replays a capture's `wire/*.bin` keyframe frames into a
//! `GuideSession` at capture rate and appends one JSON line of engine metrics
//! per sample interval. Runs the same code on the Mac (CLI) and the iPhone
//! (C entry point called from a harness app).

use anyhow::{Context, bail};
use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, KeyframeHeader, decode_frame};
use brush_guide::session::GuideSession;
// cubecl's AutoGraphicsApi picks Metal only on macOS and asks for Vulkan on iOS.
use burn_wgpu::graphics::Metal;
use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice};
use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use serde::Serialize;
use std::ffi::{CStr, c_char};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct BenchParams {
    pub max_splats: u32,
    /// Keyframe image width; frames wider than this are downscaled, intrinsics scaled with them.
    pub width: u32,
    pub duration_s: f32,
    /// Keyframes per second; the capture loops with fresh ids until `duration_s`.
    pub rate: f32,
    pub sample_s: f32,
    pub loader_cache_mb: u32,
    /// Training iterations per second at most; 0 is uncapped.
    pub max_iters_per_s: f32,
}

#[derive(Serialize)]
struct Sample {
    t_s: f32,
    keyframes_sent: u64,
    num_keyframes: u32,
    num_splats: u32,
    train_iters_per_s: f32,
    /// Cumulative; the difference between samples over their interval is the true average rate.
    train_iters: u64,
    last_score_ms: u32,
    score_version: u64,
    /// Slowest `push_keyframe` since the previous sample.
    max_push_ms: u32,
    phys_footprint_mb: f64,
    peak_phys_footprint_mb: f64,
    #[serde(flatten)]
    host: HostSample,
}

/// What the host app reports per sample; -1 when unknown.
#[repr(C)]
#[derive(Clone, Copy, Serialize)]
pub struct HostSample {
    /// `ProcessInfo.ThermalState` raw value.
    pub thermal: i32,
    /// `os_proc_available_memory()`: headroom below the jetsam limit.
    pub available_mb: f64,
    /// `UIDevice.batteryLevel`, 0–1 in 1 % steps.
    pub battery_level: f32,
    /// `UIDevice.BatteryState` raw value (1 unplugged, 2 charging, 3 full).
    pub battery_state: i32,
}

impl HostSample {
    pub const UNKNOWN: Self = Self {
        thermal: -1,
        available_mb: -1.0,
        battery_level: -1.0,
        battery_state: -1,
    };
}

struct Frame {
    header: KeyframeHeader,
    payload: Vec<u8>,
}

/// Loads the keyframe frames in numeric file order.
fn load_frames(wire_dir: &Path, width: u32) -> anyhow::Result<Vec<Frame>> {
    let mut files: Vec<(u64, PathBuf)> = std::fs::read_dir(wire_dir)
        .with_context(|| format!("reading {}", wire_dir.display()))?
        .filter_map(|e| {
            let path = e.ok()?.path();
            let n = path.file_stem()?.to_str()?.parse().ok()?;
            Some((n, path))
        })
        .collect();
    files.sort_unstable_by_key(|(n, _)| *n);
    let mut frames = Vec::with_capacity(files.len());
    for (_, path) in files {
        let bytes = std::fs::read(&path)?;
        let (header, payload) = decode_frame::<ClientHeader>(&bytes)?;
        let ClientHeader::Keyframe(header) = header else {
            bail!("{} is not a keyframe", path.display());
        };
        frames.push(downscale(header, payload, width)?);
    }
    Ok(frames)
}

fn downscale(mut h: KeyframeHeader, payload: &[u8], width: u32) -> anyhow::Result<Frame> {
    if h.width <= width {
        return Ok(Frame {
            header: h,
            payload: payload.to_vec(),
        });
    }
    let (jpeg, rest) = payload.split_at(h.jpeg_len as usize);
    let s = width as f32 / h.width as f32;
    let height = (h.height as f32 * s).round() as u32;
    let img = image::load_from_memory(jpeg)?.to_rgb8();
    let img = image::imageops::resize(&img, width, height, FilterType::Triangle);
    let mut out = Vec::new();
    // Quality 85 matches the phone's FrameEncoder.
    JpegEncoder::new_with_quality(&mut out, 85).write_image(
        img.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;
    h.fx *= s;
    h.fy *= s;
    h.cx *= s;
    h.cy *= s;
    h.width = width;
    h.height = height;
    h.jpeg_len = out.len() as u32;
    out.extend_from_slice(rest);
    Ok(Frame {
        header: h,
        payload: out,
    })
}

/// Current and lifetime-peak physical footprint in MB (what jetsam counts on iOS).
fn footprint_mb() -> (f64, f64) {
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&raw mut info).cast::<libc::rusage_info_t>(),
        )
    };
    if ok != 0 {
        return (-1.0, -1.0);
    }
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    (
        mb(info.ri_phys_footprint),
        mb(info.ri_lifetime_max_phys_footprint),
    )
}

pub async fn run(
    params: BenchParams,
    wire_dir: &Path,
    out_dir: &Path,
    host: impl Fn() -> HostSample,
) -> anyhow::Result<()> {
    let frames = load_frames(wire_dir, params.width)?;
    if frames.is_empty() {
        bail!("no frames in {}", wire_dir.display());
    }
    std::fs::create_dir_all(out_dir)?;
    let cap = if params.max_iters_per_s > 0.0 {
        format!("{}its", params.max_iters_per_s)
    } else {
        "uncapped".to_owned()
    };
    let stem = format!(
        "bench-{}k-{}px-{cap}",
        params.max_splats / 1000,
        params.width
    );
    let mut log = std::fs::File::create(out_dir.join(format!("{stem}.jsonl")))?;

    // Once per process: the host app runs several benches in a row.
    static DEVICE_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !DEVICE_READY.swap(true, std::sync::atomic::Ordering::SeqCst) {
        burn_wgpu::init_setup_async::<Metal>(
            &WgpuDevice::default(),
            RuntimeOptions {
                tasks_max: 64,
                memory_config: MemoryConfiguration::ExclusivePages,
            },
        )
        .await;
    }
    let device = burn::tensor::Device::from(WgpuDevice::default()).autodiff();
    let config = GuideConfig {
        max_splats: params.max_splats,
        loader_cache_bytes: u64::from(params.loader_cache_mb) << 20,
        max_iters_per_s: params.max_iters_per_s,
        ..GuideConfig::default()
    };
    let session = GuideSession::start(config, device, out_dir.join("session"));
    let status = session.status();
    let scores = session.scores();

    let start = Instant::now();
    let duration = Duration::from_secs_f32(params.duration_s);
    let interval = Duration::from_secs_f32(1.0 / params.rate);
    let sample_every = Duration::from_secs_f32(params.sample_s);
    let mut next_frame = start;
    let mut next_sample = start + sample_every;
    let mut sent: u64 = 0;
    let mut max_push = Duration::ZERO;

    while start.elapsed() < duration {
        let now = Instant::now();
        if now >= next_frame {
            let f = &frames[(sent % frames.len() as u64) as usize];
            let mut header = f.header.clone();
            header.id = sent;
            header.timestamp = sent as f64 / f64::from(params.rate);
            let t = Instant::now();
            session.push_keyframe(header, f.payload.clone()).await?;
            max_push = max_push.max(t.elapsed());
            sent += 1;
            next_frame += interval;
        }
        if now >= next_sample {
            let st = status.borrow().clone();
            let (phys, peak) = footprint_mb();
            let sample = Sample {
                t_s: start.elapsed().as_secs_f32(),
                keyframes_sent: sent,
                num_keyframes: st.num_keyframes,
                num_splats: st.num_splats,
                train_iters_per_s: st.train_iters_per_s,
                train_iters: st.train_iters,
                last_score_ms: st.last_score_ms,
                score_version: scores.borrow().as_ref().map_or(0, |s| s.version),
                max_push_ms: max_push.as_millis() as u32,
                phys_footprint_mb: phys,
                peak_phys_footprint_mb: peak,
                host: host(),
            };
            let line = serde_json::to_string(&sample)?;
            eprintln!("{line}");
            writeln!(log, "{line}")?;
            log.flush()?;
            max_push = Duration::ZERO;
            next_sample += sample_every;
        }
        let wake = next_frame.min(next_sample);
        tokio::time::sleep(wake.saturating_duration_since(Instant::now())).await;
    }

    // The end state, for comparing guidance quality across runs: splat PLY,
    // and the score set as a wire frame.
    std::fs::write(
        out_dir.join(format!("{stem}.ply")),
        session.export_splat().await?,
    )?;
    if let Some(s) = scores.borrow().as_ref() {
        std::fs::write(out_dir.join(format!("{stem}-scores.bin")), s.to_frame())?;
    }
    Ok(())
}

/// C parameters for [`guide_bench_run`].
#[repr(C)]
pub struct GuideBenchParams {
    pub max_splats: u32,
    pub width: u32,
    pub duration_s: f32,
    pub rate: f32,
    pub sample_s: f32,
    pub loader_cache_mb: u32,
    pub max_iters_per_s: f32,
}

/// Runs one bench to completion on the calling thread. Returns 0 on success, 1 on error
/// (the error is printed to stderr).
///
/// # Safety
/// `wire_dir` and `out_dir` must be valid NUL-terminated UTF-8 paths and `params` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn guide_bench_run(
    wire_dir: *const c_char,
    out_dir: *const c_char,
    params: *const GuideBenchParams,
    host: Option<extern "C" fn() -> HostSample>,
) -> i32 {
    let (wire_dir, out_dir, p) = unsafe {
        (
            CStr::from_ptr(wire_dir).to_string_lossy().into_owned(),
            CStr::from_ptr(out_dir).to_string_lossy().into_owned(),
            &*params,
        )
    };
    let params = BenchParams {
        max_splats: p.max_splats,
        width: p.width,
        duration_s: p.duration_s,
        rate: p.rate,
        sample_s: p.sample_s,
        loader_cache_mb: p.loader_cache_mb,
        max_iters_per_s: p.max_iters_per_s,
    };
    // A crash report has no panic message, so keep it next to the logs.
    let panic_path = Path::new(&out_dir).join("panic.txt");
    std::panic::set_hook(Box::new(move |info| {
        let text = format!(
            "{info}\n{}",
            std::backtrace::Backtrace::force_capture()
        );
        eprintln!("{text}");
        let _ = std::fs::write(&panic_path, text);
    }));
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("bench: runtime: {e}");
            return 1;
        }
    };
    let host = move || host.map_or(HostSample::UNKNOWN, |f| f());
    match runtime.block_on(run(
        params,
        Path::new(&wire_dir),
        Path::new(&out_dir),
        host,
    )) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("bench: {e:#}");
            1
        }
    }
}
