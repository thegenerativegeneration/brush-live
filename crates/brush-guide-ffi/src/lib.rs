//! C interface to one live guidance session for the iOS app. Keyframe and poses
//! wire frames go in (`bge_push`); the frames the server would send come out
//! through the `out` callback (docs/protocol.md).

use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ServerHeader, encode_frame};
use brush_guide::session::{GuideSession, Pushed, forward_frames};
use brush_guide::warmup::Warmup;
use burn::cubecl::config::autotune::AutotuneLevel;
use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use brush_guide::gpu::{Metal, init_device};
use burn_wgpu::WgpuDevice;
use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tokio::sync::watch;

mod prewarm;
mod preview;
pub use preview::*;

pub type BgeOut = extern "C" fn(ctx: *mut c_void, frame: *const u8, len: usize);

struct Out {
    f: BgeOut,
    ctx: *mut c_void,
}

// The callback and its context are the host's; the host promises they may be
// called from any thread, several at a time (see the header).
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    fn send(&self, frame: &[u8]) {
        (self.f)(self.ctx, frame.as_ptr(), frame.len());
    }

    fn error(&self, message: impl Into<String>) {
        self.send(&encode_frame(&ServerHeader::Error { message: message.into() }, &[]));
    }
}

pub struct BgeEngine {
    runtime: tokio::runtime::Runtime,
    session: GuideSession,
    out: Arc<Out>,
    forwarder: tokio::task::JoinHandle<()>,
    warmup: Option<WarmupControl>,
    session_paused: AtomicBool,
    /// This engine holds the process-wide pre-warm pause.
    holds_prewarm: AtomicBool,
}

/// A running warm-up and the flag that pauses it between splat counts.
struct WarmupControl {
    warmup: Warmup,
    pause: watch::Sender<bool>,
}

impl WarmupControl {
    /// Pauses the warm-up; true once it is done or gone (it panicked, which
    /// also releases the session), so the session may be commanded. False
    /// once it is parked.
    async fn pause(&self) -> bool {
        let _ = self.pause.send(true);
        prewarm::parked_or_done(self.warmup.parked(), self.warmup.ready()).await
    }
}

/// Kernels per Metal command buffer. iOS aborts a foreground app's command
/// buffers that hold up display rendering (kIOGPUCommandBufferCallbackError
/// ImpactingInteractivity); short buffers let RealityKit's frames interleave.
const GPU_TASKS_PER_SUBMIT: usize = 8;

/// Sets up wgpu with the Metal graphics API once per process, and returns
/// the autodiff device. `AutoGraphicsApi` would ask for Vulkan on iOS, so a
/// failed setup is retried, never skipped. `None` if there is no GPU adapter.
fn device(runtime: &tokio::runtime::Runtime) -> Option<burn::tensor::Device> {
    static READY: Mutex<bool> = Mutex::new(false);
    let mut ready = READY.lock().unwrap_or_else(PoisonError::into_inner);
    if *ready {
        return Some(burn::tensor::Device::from(WgpuDevice::default()).autodiff());
    }
    let init = init_device::<Metal>(GPU_TASKS_PER_SUBMIT);
    let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.block_on(init))).ok()?;
    *ready = true;
    Some(device)
}

/// Appends every panic's message and backtrace to `path` (on iOS, stderr is
/// lost), then runs the previous hook. Installed once per process.
fn record_panics(path: PathBuf) {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        let thread = std::thread::current();
        let text = format!(
            "--- panic on thread {:?} at {:?}\n{info}\n{}\n",
            thread.name().unwrap_or("?"),
            std::time::SystemTime::now(),
            std::backtrace::Backtrace::force_capture()
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = f.write_all(text.as_bytes());
        }
        previous(info);
    }));
}

/// Applies the optional `"gpu_autotune_level"` (`minimal`, `balanced`,
/// `extensive`, `full`) and `"gpu_autotune_samples"` (benchmark samples per
/// candidate, at least 1) keys of the engine config to cubecl's process-wide
/// config. It is read once, at the first GPU use, so this runs before
/// `device`. An invalid value or an already-read config is logged and
/// otherwise ignored.
fn set_autotune(json: &str) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        return;
    };
    let mut config = CubeClRuntimeConfig::default();
    let mut changed = false;
    if let Some(name) = value.get("gpu_autotune_level").and_then(|v| v.as_str()) {
        match serde_json::from_value::<AutotuneLevel>(serde_json::Value::String(name.to_owned())) {
            Ok(level) => {
                config.autotune.level = level;
                changed = true;
            }
            Err(_) => log::warn!("gpu_autotune_level {name:?} is not minimal, balanced, extensive or full; ignored"),
        }
    }
    if let Some(raw) = value.get("gpu_autotune_samples") {
        match raw.as_u64().filter(|n| *n >= 1) {
            Some(n) => {
                let n = usize::try_from(n).unwrap_or(usize::MAX);
                let bench = &mut config.autotune.bench;
                bench.max_samples = n;
                bench.min_samples = bench.min_samples.min(n);
                bench.short_circuit_samples = bench.short_circuit_samples.min(n);
                changed = true;
            }
            None => log::warn!("gpu_autotune_samples {raw} is not an integer of at least 1; ignored"),
        }
    }
    if changed && !CubeClRuntimeConfig::try_set(config) {
        log::warn!("gpu_autotune settings ignored: cubecl config was already read");
    }
}

/// Parses the `GuideConfig` overrides in `json` (empty for the defaults) and validates the result.
fn parse_config(json: &str) -> Result<GuideConfig, String> {
    let config = if json.trim().is_empty() {
        GuideConfig::default()
    } else {
        serde_json::from_str(json).map_err(|e| e.to_string())?
    };
    config.validate()?;
    Ok(config)
}

/// Why `init` failed; `Display` is the text `bge_new` sends as an `Error` frame.
enum InitError {
    Config(String),
    Runtime(String),
    NoGpu,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "engine config: {e}"),
            Self::Runtime(e) => write!(f, "engine runtime: {e}"),
            Self::NoGpu => f.write_str("engine: no Metal GPU adapter"),
        }
    }
}

/// Parses the config, applies the autotune keys, then builds the runtime and the process-wide device, in that order
/// (cubecl reads its config at the first GPU use).
fn init(json: &str) -> Result<(GuideConfig, tokio::runtime::Runtime, burn::tensor::Device), InitError> {
    let config = parse_config(json).map_err(InitError::Config)?;
    set_autotune(json);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| InitError::Runtime(e.to_string()))?;
    let device = device(&runtime).ok_or(InitError::NoGpu)?;
    Ok((config, runtime, device))
}

unsafe fn str_arg<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Starts a session. `config_json` holds `GuideConfig` overrides (may be
/// empty). On failure, passes one `Error` frame to `out` and returns null.
///
/// # Safety
/// `config_json` and `session_dir` are NUL-terminated UTF-8; `out` and `ctx`
/// stay valid until `bge_free` and may be called from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_new(
    config_json: *const c_char,
    session_dir: *const c_char,
    out: BgeOut,
    ctx: *mut c_void,
) -> *mut BgeEngine {
    let out = Arc::new(Out { f: out, ctx });
    let json = unsafe { str_arg(config_json) }.unwrap_or("");
    let Some(dir) = (unsafe { str_arg(session_dir) }) else {
        out.error("engine: session directory missing");
        return std::ptr::null_mut();
    };
    record_panics(PathBuf::from(dir).join("engine-panic.txt"));
    let (config, runtime, device) = match init(json) {
        Ok(parts) => parts,
        Err(e) => {
            out.error(e.to_string());
            return std::ptr::null_mut();
        }
    };
    let warmup = config.warmup.then(|| {
        let (pause, pause_rx) = watch::channel(false);
        let warmup = Warmup::spawn_pausable(config.clone(), device.clone(), pause_rx);
        WarmupControl { warmup, pause }
    });
    let ready = warmup.as_ref().map(|w| w.warmup.ready());
    let session = GuideSession::start_when(config, device, PathBuf::from(dir), ready);
    let forwarder = runtime.spawn({
        let session = session.clone();
        let out = out.clone();
        async move {
            forward_frames(&session, |frame| {
                out.send(&frame);
                std::future::ready(true)
            })
            .await;
        }
    });
    Box::into_raw(Box::new(BgeEngine {
        runtime,
        session,
        out,
        forwarder,
        warmup,
        session_paused: AtomicBool::new(false),
        holds_prewarm: AtomicBool::new(false),
    }))
}

/// Adds one keyframe frame (emits `Ack`) or applies one poses frame (emits nothing); blocks until done. A rejected
/// frame emits `Error` "keyframe <id>: …" or "poses: …". Waits while paused. Returns 0 on success, 1 otherwise.
///
/// # Safety
/// `e` comes from `bge_new`; `frame` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_push(e: *mut BgeEngine, frame: *const u8, len: usize) -> i32 {
    let e = unsafe { &*e };
    let bytes = unsafe { std::slice::from_raw_parts(frame, len) };
    match e.runtime.block_on(e.session.push_frame(bytes)) {
        Ok(Pushed::Keyframe(id)) => {
            e.out.send(&encode_frame(&ServerHeader::Ack { keyframe_id: id }, &[]));
            0
        }
        Ok(Pushed::Poses(_)) => 0,
        Err(message) => {
            e.out.error(message);
            1
        }
    }
}

/// Writes the splat as PLY to `ply_path` and emits `Splat`; training pauses
/// until the next keyframe. Returns 0 on success, 1 otherwise (`Error` emitted).
///
/// # Safety
/// `e` comes from `bge_new`; `ply_path` is NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_finish(e: *mut BgeEngine, ply_path: *const c_char) -> i32 {
    let e = unsafe { &*e };
    let Some(path) = (unsafe { str_arg(ply_path) }) else {
        e.out.error("finish: no path");
        return 1;
    };
    match e.runtime.block_on(e.session.finish(std::path::Path::new(path))) {
        Ok(ply_len) => {
            e.out.send(&encode_frame(&ServerHeader::Splat { ply_len }, &[]));
            0
        }
        Err(err) => {
            e.out.error(format!("finish: {err}"));
            1
        }
    }
}

/// Clears the session for a new segment (new world frame). Preview
/// snapshots published afterwards carry `generation`.
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_reset(e: *mut BgeEngine, generation: u64) {
    let e = unsafe { &*e };
    e.runtime.block_on(e.session.reset(generation));
}

/// Stops all GPU work, including any `bge_warm_up` running on another
/// thread; returns after the step, round or warm-up size in progress.
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_pause(e: *mut BgeEngine) {
    let e = unsafe { &*e };
    if !e.holds_prewarm.swap(true, Ordering::SeqCst) {
        prewarm::hold();
    }
    let own_warmup_parked = e
        .warmup
        .as_ref()
        .is_some_and(|warmup| !e.runtime.block_on(warmup.pause()));
    // The session takes no commands during its own warm-up; `bge_resume`
    // lets the warm-up go on.
    if !own_warmup_parked {
        e.runtime.block_on(e.session.set_paused(true));
        e.session_paused.store(true, Ordering::SeqCst);
    }
    e.runtime.block_on(prewarm::settle());
}

/// Resumes after `bge_pause`; held keyframes are added in order.
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_resume(e: *mut BgeEngine) {
    let e = unsafe { &*e };
    if let Some(warmup) = &e.warmup {
        let _ = warmup.pause.send(false);
    }
    if e.session_paused.swap(false, Ordering::SeqCst) {
        e.runtime.block_on(e.session.set_paused(false));
    }
    if e.holds_prewarm.swap(false, Ordering::SeqCst) {
        prewarm::release();
    }
}

/// Runs the session warm-up to completion. `config_json` is read as in
/// `bge_new`. Returns 0 on success, 1 on a config error, no GPU adapter or a
/// panic in the warm-up. It compiles the kernels of every warm-up size (and
/// tunes them where autotune is enabled) on the process-wide device, so later
/// sessions reuse them. While any engine is paused, the warm-up waits between
/// splat counts.
///
/// # Safety
/// `config_json` is NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_warm_up(config_json: *const c_char) -> i32 {
    let json = unsafe { str_arg(config_json) }.unwrap_or("");
    let (config, runtime, device) = match init(json) {
        Ok(parts) => parts,
        Err(InitError::Config(e)) => {
            log::warn!("warm-up config: {e}");
            return 1;
        }
        Err(_) => return 1,
    };
    let warmup = Warmup::spawn_pausable(config, device, prewarm::flag());
    let registration = prewarm::register(&warmup);
    let mut ready = warmup.ready();
    let done = runtime.block_on(ready.wait_for(|r| *r)).is_ok();
    drop(registration);
    drop(warmup);
    runtime.shutdown_background();
    i32::from(!done)
}

/// Writes the warm-up's progress: sizes done and sizes in total of the warm-up running now (0, 0 before any).
///
/// # Safety
/// `done` and `total` are valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_warm_up_progress(done: *mut u32, total: *mut u32) {
    let (d, t) = brush_guide::warmup::progress();
    unsafe {
        *done = d;
        *total = t;
    }
}

/// Stops the session and frees the engine; `out` is not called afterwards.
///
/// # Safety
/// `e` comes from `bge_new` and is not used again; no other `bge_*` call is
/// running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_free(e: *mut BgeEngine) {
    if e.is_null() {
        return;
    }
    let e = unsafe { Box::from_raw(e) };
    if e.holds_prewarm.swap(false, Ordering::SeqCst) {
        prewarm::release();
    }
    let BgeEngine {
        runtime,
        session,
        forwarder,
        ..
    } = *e;
    // An aborted task is finished only once its future is dropped; after
    // this, `out` is never called again.
    forwarder.abort();
    let _ = runtime.block_on(forwarder);
    drop(session);
    runtime.shutdown_background();
}
