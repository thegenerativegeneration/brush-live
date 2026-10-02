//! The cubecl config is process-wide and set once, so this test has its own binary.

use burn::cubecl::config::autotune::AutotuneLevel;
use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use brush_guide_ffi::*;
use std::ffi::{CString, c_void};

extern "C" fn ignore(_ctx: *mut c_void, _frame: *const u8, _len: usize) {}

fn start(config: &str, dir: &std::path::Path) -> *mut BgeEngine {
    let config = CString::new(config).unwrap();
    let dir = CString::new(dir.to_str().unwrap()).unwrap();
    unsafe { bge_new(config.as_ptr(), dir.as_ptr(), ignore, std::ptr::null_mut()) }
}

#[test]
fn autotune_level_reaches_cubecl_and_an_unknown_level_is_ignored() {
    let dir = std::env::temp_dir().join(format!("bge-autotune-{}", std::process::id()));
    let e = start(r#"{"warmup": false, "gpu_autotune_level": "minimal", "gpu_autotune_samples": 2}"#, &dir);
    assert!(!e.is_null());
    assert!(matches!(CubeClRuntimeConfig::get().autotune.level, AutotuneLevel::Minimal));
    let bench = &CubeClRuntimeConfig::get().autotune.bench;
    assert_eq!((bench.min_samples, bench.max_samples, bench.short_circuit_samples), (2, 2, 2));
    unsafe { bge_free(e) };

    let e = start(r#"{"warmup": false, "gpu_autotune_level": "bogus"}"#, &dir);
    assert!(!e.is_null());
    assert!(matches!(CubeClRuntimeConfig::get().autotune.level, AutotuneLevel::Minimal));
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
