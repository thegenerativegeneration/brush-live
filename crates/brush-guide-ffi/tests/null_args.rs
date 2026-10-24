//! Null pointers from the host: calls without an engine do nothing and report failure where they have a result.

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
fn every_call_accepts_a_null_engine() {
    let e = std::ptr::null_mut::<BgeEngine>();
    let frame = [0u8; 4];
    let mut preview = BgePreview::default();
    unsafe {
        assert_eq!(bge_push(e, frame.as_ptr(), frame.len()), 1);
        assert_eq!(bge_finish(e, c"x.ply".as_ptr()), 1);
        bge_reset(e, 1);
        bge_pause(e);
        bge_resume(e);
        bge_set_preview(e, 0);
        assert_eq!(bge_preview_latest(e, 0, &mut preview), 0);
        bge_warm_up_progress(std::ptr::null_mut(), std::ptr::null_mut());
    }
}

#[test]
fn an_empty_push_with_a_null_pointer_is_rejected_not_undefined() {
    let frames = Frames::default();
    let ctx = std::ptr::from_ref::<Frames>(&frames).cast_mut().cast::<c_void>();
    let dir = std::env::temp_dir().join(format!("bge-null-args-{}", std::process::id()));
    let config = CString::new(r#"{"max_splats": 20000, "warmup": false}"#).unwrap();
    let session_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), session_dir.as_ptr(), collect_into, ctx) };
    assert!(!e.is_null());
    assert_eq!(unsafe { bge_push(e, std::ptr::null(), 0) }, 1);
    let errors: Vec<_> = frames
        .lock()
        .unwrap()
        .iter()
        .filter(|h| matches!(h, ServerHeader::Error { .. }))
        .cloned()
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
