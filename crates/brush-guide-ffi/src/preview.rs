//! Preview snapshots for the app's splat overlay.

use crate::BgeEngine;
use brush_guide::session::PreviewSnapshot;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

/// One snapshot as handed to the host; `data` stays valid until
/// `bge_preview_release(handle)`.
#[repr(C)]
pub struct BgePreview {
    pub version: u64,
    pub count: u32,
    pub readback_ms: f32,
    pub data: *const f32,
    pub handle: *const c_void,
}

const _: () = assert!(
    std::mem::size_of::<BgePreview>() == 32
        && std::mem::offset_of!(BgePreview, data) == 16
        && std::mem::offset_of!(BgePreview, handle) == 24
);

impl Default for BgePreview {
    fn default() -> Self {
        Self {
            version: 0,
            count: 0,
            readback_ms: 0.0,
            data: std::ptr::null(),
            handle: std::ptr::null(),
        }
    }
}

/// `interval_ms` 0: a snapshot after every training step; > 0: at most every
/// `interval_ms`; < 0: off.
///
/// # Safety
/// `e` comes from `bge_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_set_preview(e: *mut BgeEngine, interval_ms: i32) {
    let e = unsafe { &*e };
    let interval = u64::try_from(interval_ms).ok().map(Duration::from_millis);
    e.runtime.block_on(e.session.set_preview(interval));
}

/// Fills `out` with the newest snapshot if its version is above
/// `after_version` and returns 1; returns 0 otherwise.
///
/// # Safety
/// `e` comes from `bge_new`; `out` points to a writable `BgePreview`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_preview_latest(e: *mut BgeEngine, after_version: u64, out: *mut BgePreview) -> i32 {
    let e = unsafe { &*e };
    let Some(snap) = e.session.preview().borrow().clone() else { return 0 };
    if snap.version <= after_version {
        return 0;
    }
    let preview = BgePreview {
        version: snap.version,
        count: snap.count,
        readback_ms: snap.readback_ms,
        data: snap.data.as_ptr(),
        handle: Arc::into_raw(snap).cast(),
    };
    unsafe { out.write(preview) };
    1
}

/// Releases a snapshot from `bge_preview_latest`; null is ignored. Valid
/// after `bge_free`.
///
/// # Safety
/// `handle` comes from `bge_preview_latest` and is released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bge_preview_release(handle: *const c_void) {
    if !handle.is_null() {
        drop(unsafe { Arc::from_raw(handle.cast::<PreviewSnapshot>()) });
    }
}
