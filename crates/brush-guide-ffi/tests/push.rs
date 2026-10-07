//! `bge_push` with poses frames after a recorded keyframe frame: applied silently, unknown ids ignored, invalid
//! poses rejected with an `Error`.

use brush_guide::protocol::{ClientHeader, PoseUpdate, ServerHeader, decode_frame, encode_frame};
use brush_guide_ffi::*;
use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::sync::Mutex;

type Frames = Mutex<Vec<ServerHeader>>;

extern "C" fn collect_into(ctx: *mut c_void, frame: *const u8, len: usize) {
    let frames = unsafe { &*(ctx as *const Frames) };
    let bytes = unsafe { std::slice::from_raw_parts(frame, len) };
    let (h, _) = decode_frame::<ServerHeader>(bytes).expect("server frame");
    frames.lock().unwrap().push(h);
}

fn wire(id: u64) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../../../datasets/segment-1/wire/{id}.bin"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The frames `bge_push` emits itself (status and score sets come from the forwarder).
fn replies(frames: &Frames) -> Vec<ServerHeader> {
    frames
        .lock()
        .unwrap()
        .iter()
        .filter(|h| {
            !matches!(
                h,
                ServerHeader::Status { .. } | ServerHeader::ScoreSet { .. }
            )
        })
        .cloned()
        .collect()
}

/// An engine that holds keyframe 0 of the recording; returns it with keyframe 0's pose.
fn engine_with_keyframe(name: &str, frames: &Frames) -> (*mut BgeEngine, PathBuf, [f32; 16]) {
    let ctx = std::ptr::from_ref::<Frames>(frames)
        .cast_mut()
        .cast::<c_void>();
    let dir = std::env::temp_dir().join(format!("bge-push-{name}-{}", std::process::id()));
    let config =
        CString::new(r#"{"max_splats": 20000, "warmup": false, "keyframe_long_side": 960}"#)
            .unwrap();
    let session_dir = CString::new(dir.to_str().unwrap()).unwrap();
    let e = unsafe { bge_new(config.as_ptr(), session_dir.as_ptr(), collect_into, ctx) };
    assert!(!e.is_null());
    let f = wire(0);
    let Ok((ClientHeader::Keyframe(h), _)) = decode_frame::<ClientHeader>(&f) else {
        panic!("wire/0.bin is a keyframe frame")
    };
    assert_eq!(unsafe { bge_push(e, f.as_ptr(), f.len()) }, 0);
    assert!(matches!(
        replies(frames)[..],
        [ServerHeader::Ack { keyframe_id: 0 }]
    ));
    (e, dir, h.pose)
}

fn push_poses(e: *mut BgeEngine, poses: Vec<PoseUpdate>) -> i32 {
    let f = encode_frame(&ClientHeader::Poses { poses }, &[]);
    unsafe { bge_push(e, f.as_ptr(), f.len()) }
}

#[test]
fn poses_for_a_known_keyframe_return_0_and_emit_nothing() {
    let frames = Frames::default();
    let (e, dir, mut pose) = engine_with_keyframe("known", &frames);
    pose[12] += 0.05;
    assert_eq!(push_poses(e, vec![PoseUpdate { id: 0, pose }]), 0);
    assert_eq!(
        replies(&frames).len(),
        1,
        "only keyframe 0's ack: {:?}",
        replies(&frames)
    );
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn poses_for_an_unknown_keyframe_are_ignored() {
    let frames = Frames::default();
    let (e, dir, pose) = engine_with_keyframe("unknown", &frames);
    assert_eq!(push_poses(e, vec![PoseUpdate { id: 99, pose }]), 0);
    assert_eq!(
        replies(&frames).len(),
        1,
        "no error for an unknown id: {:?}",
        replies(&frames)
    );
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_non_rigid_pose_returns_1_with_a_poses_error() {
    let frames = Frames::default();
    let (e, dir, mut pose) = engine_with_keyframe("invalid", &frames);
    // Not rigid (a scaled axis). A NaN cannot be sent: JSON writes it as null, a decode error.
    pose[0] *= 2.0;
    assert_eq!(push_poses(e, vec![PoseUpdate { id: 0, pose }]), 1);
    let replies = replies(&frames);
    let [_, ServerHeader::Error { message }] = &replies[..] else {
        panic!("expected one error after the ack: {replies:?}")
    };
    assert!(message.starts_with("poses: keyframe 0 pose: "), "{message}");
    unsafe { bge_free(e) };
    std::fs::remove_dir_all(dir).ok();
}
