//! The warm-up call has its own binary: it sets the process-wide cubecl config and device.

use brush_guide_ffi::*;
use std::ffi::CString;

fn warm_up(config: &str) -> i32 {
    let config = CString::new(config).unwrap();
    unsafe { bge_warm_up(config.as_ptr()) }
}

#[test]
fn warm_up_returns_zero_for_a_small_budget_and_one_for_malformed_json() {
    assert_eq!(warm_up("{not json"), 1);
    assert_eq!(
        warm_up(r#"{"max_splats": 20000, "keyframe_long_side": 500, "gpu_autotune_level": "minimal", "gpu_autotune_samples": 2}"#),
        0
    );
}

extern "C" fn ignore(_ctx: *mut std::ffi::c_void, _frame: *const u8, _len: usize) {}

#[test]
fn a_paused_engine_holds_a_warm_up_on_another_thread_until_resumed() {
    const SMALL: &str = r#"{"max_splats": 4096, "keyframe_long_side": 500, "gpu_autotune_level": "minimal", "gpu_autotune_samples": 2}"#;
    let t = std::time::Instant::now();
    assert_eq!(warm_up(SMALL), 0);
    let unpaused = t.elapsed();

    let dir = std::env::temp_dir().join(format!("bge-warm-up-pause-{}", std::process::id()));
    let config = CString::new(r#"{"max_splats": 4096, "warmup": false}"#).unwrap();
    let session_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), session_dir.as_ptr(), ignore, std::ptr::null_mut()) };
    assert!(!e.is_null());
    unsafe { bge_pause(e) };
    let warm = std::thread::spawn(|| warm_up(SMALL));
    std::thread::sleep((unpaused * 3).max(std::time::Duration::from_secs(1)));
    assert!(!warm.is_finished(), "the warm-up waits while an engine is paused");
    unsafe { bge_resume(e) };
    assert_eq!(warm.join().unwrap(), 0);
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
