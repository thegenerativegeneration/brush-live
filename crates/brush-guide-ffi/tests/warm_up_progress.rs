//! Own binary: the progress counters are process-wide, so other warm-ups in the same process would disturb them.

use brush_guide_ffi::*;
use std::ffi::CString;

fn warm_up(config: &str) -> i32 {
    let config = CString::new(config).unwrap();
    unsafe { bge_warm_up(config.as_ptr()) }
}

#[test]
fn progress_reaches_the_number_of_sizes() {
    assert_eq!(warm_up(r#"{"max_splats": 1000, "keyframe_long_side": 320}"#), 0);
    let (mut done, mut total) = (0u32, 0u32);
    unsafe { bge_warm_up_progress(&mut done, &mut total) };
    assert_eq!((done, total), (4, 4)); // 128, 256, 512, 1000
}
