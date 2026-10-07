//! Plan D's Mac check of engine pose updates: a wall captured by six keyframes, two of which report a pose off by a
//! rigid drift (ORB-SLAM3 drift: 5° of yaw and 0.3 m) until a correction arrives. Measured at the true poses of those
//! two views, and right after the correction on the splats themselves.

use crate::test_scene;
use brush_guide::config::GuideConfig;
use brush_guide::keyframe::decode_keyframe;
use brush_guide::live::{LiveModel, ViewPose};
use brush_render::gaussian_splats::Splats;
use glam::{Affine3A, Mat4, Quat, Vec3};

const STEPS_PER_KEYFRAME: usize = 50;
const STEPS_AFTER: usize = 300;
/// The drifted keyframes' views; their ids are their view slots.
const DRIFTED: [u32; 2] = [3, 4];
/// Splat positions after the move must match the corrected transform this closely (metres).
const MOVE_TOL_M: f32 = 1e-4;

/// The drifted keyframes' error, where the phone thought it was relative to where it was: 5° of yaw about a point on
/// the wall in front of the drifted cameras, then a 0.3 m slide. The yaw tilts the drifted seeds' surface and swings
/// the cameras; with a rotation in it, the move's order and its rotation both matter. (About a pivot near the cameras
/// instead, the yaw would carry the seeds back by about 0.22 m and cancel most of the slide.)
fn drift() -> Mat4 {
    let pivot = Vec3::new(1.2, 0.0, 0.0);
    Mat4::from_translation(Vec3::new(0.3, 0.0, 0.1))
        * Mat4::from_translation(pivot)
        * Mat4::from_rotation_y(5f32.to_radians())
        * Mat4::from_translation(-pivot)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    NoUpdate,
    PoseOnly,
    PoseAndAnchoredSplats,
}

/// Keyframes 0–2 see the left half (true poses), 3–4 the right half (drifted), 5 both halves from the middle
/// (true pose: the loop has closed).
fn shots() -> Vec<(Mat4, bool)> {
    let at = |x: f32, look_x: f32| {
        Mat4::look_at_rh(Vec3::new(x, 0.0, 2.5), Vec3::new(look_x, 0.0, 0.0), Vec3::Y).inverse()
    };
    vec![
        (at(-1.6, -1.6), false),
        (at(-1.0, -1.0), false),
        (at(-0.4, -0.6), false),
        (at(0.8, 1.0), true),
        (at(1.5, 1.6), true),
        (at(0.0, 0.0), false),
    ]
}

async fn floats(t: burn::tensor::Tensor<2>) -> Vec<f32> {
    t.into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

/// Means and rotations (`[w, x, y, z]`, as stored) per splat.
async fn poses_of(splats: &Splats) -> Vec<(Vec3, Quat)> {
    let means = floats(splats.means()).await;
    let rots = floats(splats.rotations()).await;
    means
        .chunks_exact(3)
        .zip(rots.chunks_exact(4))
        .map(|(m, q)| (Vec3::from_slice(m), Quat::from_xyzw(q[1], q[2], q[3], q[0])))
        .collect()
}

/// Every splat seeded by a drifted view moved by `correction` (mean `Δ·μ`, rotation `q_Δ ⊗ q`); every other splat
/// stayed exactly where it was. Splats of both drifted views and of the others exist.
fn assert_anchored_move(
    before: &[(Vec3, Quat)],
    after: &[(Vec3, Quat)],
    origins: &[Option<u32>],
    correction: Affine3A,
) {
    assert_eq!(before.len(), after.len());
    assert_eq!(origins.len(), after.len());
    let (_, q_delta, _) = correction.to_scale_rotation_translation();
    let mut moved = [0usize; 2];
    let mut kept = 0usize;
    for (i, (((m0, q0), (m1, q1)), origin)) in before.iter().zip(after).zip(origins).enumerate() {
        match origin.and_then(|o| DRIFTED.iter().position(|d| *d == o)) {
            Some(k) => {
                let want = correction.transform_point3(*m0);
                assert!(
                    (*m1 - want).length() < MOVE_TOL_M,
                    "splat {i} of view {origin:?}: mean {m1}, want {want} (was {m0})"
                );
                let want_q = (q_delta * *q0).normalize();
                assert!(
                    q1.normalize().dot(want_q).abs() > 1.0 - 1e-5,
                    "splat {i} of view {origin:?}: rotation {q1}, want {want_q} (was {q0})"
                );
                moved[k] += 1;
            }
            None => {
                assert!(
                    (*m1 - *m0).length() < 1e-6 && (*q1 - *q0).length() < 1e-6,
                    "splat {i} of view {origin:?} moved: {m0} -> {m1}, {q0} -> {q1}"
                );
                kept += 1;
            }
        }
    }
    assert!(
        moved.iter().all(|n| *n > 0) && kept > 0,
        "moved per drifted view {moved:?}, kept {kept}"
    );
}

/// Mean RGB error at the drifted views' true poses after training through the capture and `STEPS_AFTER` more steps.
async fn run(arm: Arm) -> f32 {
    let device = test_scene::device().await;
    let wall = test_scene::wall(48, 2.5, &device);
    let dir =
        std::env::temp_dir().join(format!("brush-guide-drift-{arm:?}-{}", std::process::id()));
    let config = GuideConfig {
        max_splats: 20_000,
        move_anchored_splats: arm != Arm::PoseOnly,
        ..GuideConfig::default()
    };
    let mut live = LiveModel::new(config, device.autodiff());
    let shots = shots();
    for (id, (pose, drifted)) in shots.iter().enumerate() {
        let sent = if *drifted { drift() * *pose } else { *pose };
        let (h, p) = test_scene::wall_keyframe(id as u64, &wall, *pose, sent).await;
        assert!(
            live.add_keyframe(decode_keyframe(&h, &p, &dir).await.unwrap())
                .await
        );
        for _ in 0..STEPS_PER_KEYFRAME {
            live.train_step().await;
        }
    }
    if arm != Arm::NoUpdate {
        let before = poses_of(live.splats().unwrap()).await;
        let corrected: Vec<ViewPose> = shots
            .iter()
            .enumerate()
            .filter(|(_, (_, drifted))| *drifted)
            .map(|(id, (pose, _))| ViewPose::from_arkit(id as u64, &pose.to_cols_array()).unwrap())
            .collect();
        assert_eq!(live.update_poses(&corrected).moved, 2);
        let after = poses_of(live.splats().unwrap()).await;
        match live.splat_origins().await {
            // Each drifted camera moves from drift · pose back to pose: by drift⁻¹ in the world.
            Some(origins) => assert_anchored_move(
                &before,
                &after,
                &origins,
                Affine3A::from_mat4(drift().inverse()),
            ),
            None => {
                assert_eq!(arm, Arm::PoseOnly, "anchored arm tracks origins");
                assert_eq!(before, after, "pose only: no splat moves");
            }
        }
    }
    for _ in 0..STEPS_AFTER {
        live.train_step().await;
    }
    let mut error = 0.0;
    for (pose, _) in shots.iter().filter(|(_, drifted)| *drifted) {
        let camera = test_scene::wall_camera(*pose);
        let reference = test_scene::render_rgb(&wall, &camera).await;
        error += test_scene::rgb_error(live.splats().unwrap(), &camera, &reference).await;
    }
    std::fs::remove_dir_all(dir).ok();
    error / 2.0
}

#[tokio::test]
async fn corrected_poses_bring_drifted_views_back() {
    let none = run(Arm::NoUpdate).await;
    let pose_only = run(Arm::PoseOnly).await;
    let anchored = run(Arm::PoseAndAnchoredSplats).await;
    eprintln!(
        "drift check, mean RGB error at the drifted views' true poses: no update {none:.4}, pose only \
         {pose_only:.4}, pose + anchored splats {anchored:.4}"
    );
    assert!(
        anchored < 0.8 * none,
        "a correction with anchored splats must explain the drifted views better: {anchored} vs {none}"
    );
}
