//! `bge_new` rejects a config that fails `GuideConfig::validate`: one `Error` frame, null engine.

use brush_guide::protocol::{ServerHeader, decode_frame};
use brush_guide_ffi::*;
use std::ffi::{CString, c_void};
use std::sync::Mutex;

type Frames = Mutex<Vec<ServerHeader>>;

extern "C" fn collect_into(ctx: *mut c_void, frame: *const u8, len: usize) {
    let frames = unsafe { &*(ctx as *const Frames) };
    let bytes = unsafe { std::slice::from_raw_parts(frame, len) };
    let (h, _) = decode_frame::<ServerHeader>(bytes).expect("server frame");
    frames.lock().unwrap().push(h);
}

#[test]
fn a_zero_min_cell_opacity_is_reported_and_no_engine_starts() {
    let frames: Frames = Mutex::new(Vec::new());
    let config = CString::new(r#"{"min_cell_opacity": 0}"#).unwrap();
    let dir = CString::new(std::env::temp_dir().to_str().unwrap()).unwrap();
    let ctx = &frames as *const Frames as *mut c_void;
    let e = unsafe { bge_new(config.as_ptr(), dir.as_ptr(), collect_into, ctx) };
    assert!(e.is_null());
    let frames = frames.lock().unwrap();
    let [ServerHeader::Error { message }] = &frames[..] else {
        panic!("expected one Error frame, got {frames:?}");
    };
    assert_eq!(message, "engine config: min_cell_opacity must be finite and above 0, got 0");
}
