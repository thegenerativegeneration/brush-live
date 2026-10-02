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
