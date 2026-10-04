//! Preview snapshots through the C functions.

use brush_guide_ffi::*;
use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::time::{Duration, Instant};

extern "C" fn ignore(_ctx: *mut c_void, _frame: *const u8, _len: usize) {}

fn wire(id: u64) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../../../datasets/segment-1/wire/{id}.bin"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn latest(e: *mut BgeEngine, after: u64) -> Option<BgePreview> {
    let mut p = BgePreview::default();
    (unsafe { bge_preview_latest(e, after, &mut p) } == 1).then_some(p)
}

fn wait_newer(e: *mut BgeEngine, after: u64, secs: u64) -> BgePreview {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(p) = latest(e, after) {
            return p;
        }
        assert!(Instant::now() < deadline, "no snapshot newer than {after}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn snapshots_follow_training_pause_off_and_reset() {
    let dir = std::env::temp_dir().join(format!("bge-preview-{}", std::process::id()));
    let config = CString::new(r#"{"max_splats": 20000, "warmup": false, "keyframe_long_side": 500}"#).unwrap();
    let session_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), session_dir.as_ptr(), ignore, std::ptr::null_mut()) };
    assert!(!e.is_null());

    // On before any keyframe: nothing to snapshot.
    unsafe { bge_set_preview(e, 0) };
    std::thread::sleep(Duration::from_millis(200));
    assert!(latest(e, 0).is_none());

    let f = wire(0);
    assert_eq!(unsafe { bge_push(e, f.as_ptr(), f.len()) }, 0);
    let first = wait_newer(e, 0, 30);
    assert!(first.count > 0);
    assert_eq!(first.generation, 0, "generation before any reset");
    let floats = unsafe { std::slice::from_raw_parts(first.data, first.count as usize * 14) };
    for s in floats.chunks_exact(14) {
        let q = (s[3] * s[3] + s[4] * s[4] + s[5] * s[5] + s[6] * s[6]).sqrt();
        assert!((q - 1.0).abs() < 1e-3, "unit rotation, got {q}");
        assert!(s[7..10].iter().all(|v| v.is_finite() && *v > 0.0), "positive scales");
        assert!((0.0..=1.0).contains(&s[10]), "opacity {}", s[10]);
    }

    // An older snapshot stays readable after a newer one exists.
    let second = wait_newer(e, first.version, 10);
    assert!(second.version > first.version);
    assert!(floats[0].is_finite());
    unsafe { bge_preview_release(first.handle) };
    unsafe { bge_preview_release(second.handle) };

    // Paused: no GPU work, so no new snapshot.
    unsafe { bge_pause(e) };
    std::thread::sleep(Duration::from_millis(500));
    let v = latest(e, second.version).map_or(second.version, |p| {
        unsafe { bge_preview_release(p.handle) };
        p.version
    });
    std::thread::sleep(Duration::from_millis(500));
    assert!(latest(e, v).is_none(), "snapshot while paused");
    unsafe { bge_resume(e) };

    // Snapshots flow again after resume.
    let flowing = wait_newer(e, v, 10);
    let v = flowing.version;
    unsafe { bge_preview_release(flowing.handle) };

    // Off: no new snapshot.
    unsafe { bge_set_preview(e, -1) };
    std::thread::sleep(Duration::from_millis(500));
    let v = latest(e, v).map_or(v, |p| {
        unsafe { bge_preview_release(p.handle) };
        p.version
    });
    std::thread::sleep(Duration::from_millis(500));
    assert!(latest(e, v).is_none(), "snapshot while off");

    // Reset publishes an empty, newer snapshot of the new generation, which
    // later snapshots carry too.
    unsafe { bge_reset(e, 7) };
    let empty = wait_newer(e, v, 5);
    assert_eq!((empty.count, empty.generation), (0, 7));
    let v = empty.version;
    unsafe { bge_preview_release(empty.handle) };
    unsafe { bge_set_preview(e, 0) };
    let f = wire(1);
    assert_eq!(unsafe { bge_push(e, f.as_ptr(), f.len()) }, 0);
    let trained = wait_newer(e, v, 30);
    assert!(trained.count > 0);
    assert_eq!(trained.generation, 7);
    unsafe { bge_preview_release(trained.handle) };

    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
