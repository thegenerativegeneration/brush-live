//! The process-wide panic hook writes to the newest engine's session folder.

use brush_guide_ffi::*;
use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};

extern "C" fn ignore(_ctx: *mut c_void, _frame: *const u8, _len: usize) {}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("panic-log-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn start(dir: &Path) -> *mut BgeEngine {
    let config = CString::new("{}").unwrap();
    let dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), dir.as_ptr(), ignore, std::ptr::null_mut()) };
    assert!(!e.is_null());
    e
}

#[test]
fn a_panic_goes_to_the_newest_engines_folder() {
    let (a, b) = (fresh_dir("a"), fresh_dir("b"));
    unsafe { bge_free(start(&a)) };
    let second = start(&b);
    let _ = std::panic::catch_unwind(|| panic!("panic-log probe"));
    let in_b = std::fs::read_to_string(b.join("engine-panic.txt")).unwrap_or_default();
    let in_a = std::fs::read_to_string(a.join("engine-panic.txt")).unwrap_or_default();
    unsafe { bge_free(second) };
    assert!(in_b.contains("panic-log probe"), "B: {in_b}");
    assert!(!in_a.contains("panic-log probe"), "A: {in_a}");
}
