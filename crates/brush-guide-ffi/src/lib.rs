//! C interface to one live guidance session for the iOS app. Keyframe wire
//! frames go in (`bge_push`); the frames the server would send come out
//! through the `out` callback (docs/protocol.md).

use brush_guide::config::GuideConfig;
use brush_guide::protocol::{ClientHeader, ServerHeader, decode_frame, encode_frame};
use brush_guide::session::{GuideSession, forward_frames};
use brush_guide::warmup::Warmup;
use burn_wgpu::graphics::Metal;
use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice};
use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tokio::sync::watch;

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
}

/// A running warm-up and the flag that pauses it between splat counts.
struct WarmupControl {
    warmup: Warmup,
    pause: watch::Sender<bool>,
}

impl WarmupControl {
    /// Pauses the warm-up; true once it is parked or done, so the session
    /// may be commanded (it takes no commands before the warm-up is done).
    async fn pause(&self) -> bool {
        let _ = self.pause.send(true);
        let mut parked = self.warmup.parked();
        let mut ready = self.warmup.ready();
        tokio::select! {
            _ = parked.wait_for(|p| *p) => false,
            _ = ready.wait_for(|r| *r) => true,
        }
    }
}

/// Sets up wgpu with the Metal graphics API once per process, and returns
/// the autodiff device. `AutoGraphicsApi` would ask for Vulkan on iOS, so a
/// failed setup is retried, never skipped. `None` if there is no GPU adapter.
fn device(runtime: &tokio::runtime::Runtime) -> Option<burn::tensor::Device> {
    static READY: Mutex<bool> = Mutex::new(false);
    let mut ready = READY.lock().unwrap_or_else(PoisonError::into_inner);
    if !*ready {
        let init = async {
            burn_wgpu::init_setup_async::<Metal>(
                &WgpuDevice::default(),
                RuntimeOptions {
                    tasks_max: 64,
                    memory_config: MemoryConfiguration::ExclusivePages,
                },
            )
            .await;
        };
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.block_on(init))).ok()?;
        *ready = true;
    }
    Some(burn::tensor::Device::from(WgpuDevice::default()).autodiff())
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
    let config: GuideConfig = match if json.trim().is_empty() { Ok(GuideConfig::default()) } else { serde_json::from_str(json) } {
        Ok(c) => c,
        Err(e) => {
            out.error(format!("engine config: {e}"));
            return std::ptr::null_mut();
        }
    };
    let Some(dir) = (unsafe { str_arg(session_dir) }) else {
        out.error("engine: session directory missing");
        return std::ptr::null_mut();
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => {
            out.error(format!("engine runtime: {e}"));
            return std::ptr::null_mut();
        }
    };
    let Some(device) = device(&runtime) else {
        out.error("engine: no Metal GPU adapter");
        return std::ptr::null_mut();
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
    }))
}

/// Adds one keyframe frame; blocks until it is added (emits `Ack`) or
/// rejected (emits `Error` "keyframe <id>: …"). Waits while paused.
/// Returns 0 on success, 1 otherwise.
///
/// # Safety
/// `e` comes from `bge_new`; `frame` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_push(e: *mut BgeEngine, frame: *const u8, len: usize) -> i32 {
    let e = unsafe { &*e };
    let bytes = unsafe { std::slice::from_raw_parts(frame, len) };
    let (header, payload) = match decode_frame::<ClientHeader>(bytes) {
        Ok((ClientHeader::Keyframe(h), p)) => (h, p),
        Ok(_) => {
            e.out.error("expected a keyframe frame");
            return 1;
        }
        Err(err) => {
            e.out.error(err.to_string());
            return 1;
        }
    };
    let id = header.id;
    match e.runtime.block_on(e.session.push_keyframe(header, payload.to_vec())) {
        Ok(()) => {
            e.out.send(&encode_frame(&ServerHeader::Ack { keyframe_id: id }, &[]));
            0
        }
        Err(err) => {
            e.out.error(format!("keyframe {id}: {err}"));
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

/// Clears the session for a new segment (new world frame).
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_reset(e: *mut BgeEngine) {
    let e = unsafe { &*e };
    e.runtime.block_on(e.session.reset());
}

/// Stops all GPU work; returns after the step, round or warm-up size in
/// progress.
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_pause(e: *mut BgeEngine) {
    let e = unsafe { &*e };
    if let Some(warmup) = &e.warmup
        && !e.runtime.block_on(warmup.pause())
    {
        // The session takes no commands during the warm-up; `bge_resume`
        // lets the warm-up go on.
        return;
    }
    e.runtime.block_on(e.session.set_paused(true));
    e.session_paused.store(true, Ordering::SeqCst);
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
