//! Plan D's Mac check of engine pose updates: a wall captured by six keyframes, two of which report a pose 0.3 m
//! off (ORB-SLAM3 drift) until a correction arrives. Measured at the true poses of those two views.

use crate::test_scene;
use brush_guide::config::GuideConfig;
use brush_guide::keyframe::decode_keyframe;
use brush_guide::live::{LiveModel, ViewPose};
use glam::{Mat4, Vec3};

const STEPS_PER_KEYFRAME: usize = 50;
const STEPS_AFTER: usize = 300;
/// The drifted keyframes' error: where the phone thought it was, relative to where it was.
const DRIFT: Vec3 = Vec3::new(0.3, 0.0, 0.1);

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
        let sent = if *drifted {
            Mat4::from_translation(DRIFT) * *pose
        } else {
            *pose
        };
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
        let corrected: Vec<ViewPose> = shots
            .iter()
            .enumerate()
            .filter(|(_, (_, drifted))| *drifted)
            .map(|(id, (pose, _))| ViewPose::from_arkit(id as u64, &pose.to_cols_array()).unwrap())
            .collect();
        assert_eq!(live.update_poses(&corrected).moved, 2);
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
