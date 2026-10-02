//! Drives the C functions with recorded keyframe frames and decodes what comes back.

use brush_guide::protocol::{ServerHeader, decode_frame};
use brush_guide_ffi::*;
use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static FRAMES: Mutex<Vec<ServerHeader>> = Mutex::new(Vec::new());

extern "C" fn collect(_ctx: *mut c_void, frame: *const u8, len: usize) {
    let bytes = unsafe { std::slice::from_raw_parts(frame, len) };
    let (h, _) = decode_frame::<ServerHeader>(bytes).expect("server frame");
    FRAMES.lock().unwrap().push(h);
}

fn wire(id: u64) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../../../datasets/segment-1/wire/{id}.bin"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn wait_for(what: &str, secs: u64, pred: impl Fn(&[ServerHeader]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while !pred(&FRAMES.lock().unwrap()) {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn keyframes_in_acks_status_scores_pause_and_splat_out() {
    let dir = std::env::temp_dir().join(format!("bge-ffi-{}", std::process::id()));
    let config = CString::new(r#"{"max_splats": 20000, "warmup": false, "keyframe_long_side": 960}"#).unwrap();
    let session_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), session_dir.as_ptr(), collect, std::ptr::null_mut()) };
    assert!(!e.is_null());

    for id in 0..3 {
        let f = wire(id);
        assert_eq!(unsafe { bge_push(e, f.as_ptr(), f.len()) }, 0);
    }
    wait_for("three acks", 5, |fs| fs.iter().filter(|h| matches!(h, ServerHeader::Ack { .. })).count() == 3);
    wait_for("a status", 2, |fs| fs.iter().any(|h| matches!(h, ServerHeader::Status { .. })));
    wait_for("a score set", 60, |fs| fs.iter().any(|h| matches!(h, ServerHeader::ScoreSet { .. })));

    unsafe { bge_pause(e) };
    let acks = |fs: &[ServerHeader]| fs.iter().filter(|h| matches!(h, ServerHeader::Ack { .. })).count();
    let pusher = std::thread::spawn({
        let e = e as usize;
        move || {
            let f = wire(3);
            unsafe { bge_push(e as *mut BgeEngine, f.as_ptr(), f.len()) }
        }
    });
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(acks(&FRAMES.lock().unwrap()), 3, "no ack while paused");
    unsafe { bge_resume(e) };
    assert_eq!(pusher.join().unwrap(), 0);
    wait_for("the fourth ack", 10, |fs| acks(fs) == 4);

    let ply = dir.join("splat.ply");
    let path = CString::new(ply.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { bge_finish(e, path.as_ptr()) }, 0);
    wait_for("the splat frame", 5, |fs| fs.iter().any(|h| matches!(h, ServerHeader::Splat { ply_len } if *ply_len > 0)));
    assert!(std::fs::read(&ply).unwrap().starts_with(b"ply"));

    let bad = [0u8; 3];
    assert_eq!(unsafe { bge_push(e, bad.as_ptr(), bad.len()) }, 1);
    wait_for("an error frame", 2, |fs| fs.iter().any(|h| matches!(h, ServerHeader::Error { .. })));

    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
