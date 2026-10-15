use super::*;
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::tensor::{Device, Tensor};
use clap::Parser;

fn one_row(origin: Option<u32>, mean: Vec3, rot: Quat) -> SplatRows {
    SplatRows {
        means: mean.to_array().to_vec(),
        rotations: vec![rot.w, rot.x, rot.y, rot.z],
        log_scales: vec![-3.0; 3],
        sh_coeffs: vec![0.0; 3],
        raw_opacities: vec![0.0],
        min_scale: Vec::new(),
        origin: vec![origin],
        age: vec![0.0],
        importance: vec![f32::INFINITY],
        sh_coeffs_per_channel: 1,
    }
}

fn quat(rows: &SplatRows) -> Quat {
    Quat::from_xyzw(
        rows.rotations[1],
        rows.rotations[2],
        rows.rotations[3],
        rows.rotations[0],
    )
}

fn step(a: CameraPose, b: CameraPose) -> Affine3A {
    Affine3A::from_rotation_translation(b.1, b.0)
        * Affine3A::from_rotation_translation(a.1, a.0).inverse()
}

#[test]
fn a_splat_follows_its_view_rigidly() {
    let old = (Vec3::new(1.0, 0.0, 2.0), Quat::from_rotation_y(0.2));
    let new = (Vec3::new(1.1, 0.05, 2.0), Quat::from_rotation_y(0.25));
    let mean = Vec3::new(0.5, 0.2, 4.0);
    let mut rows = one_row(Some(3), mean, Quat::IDENTITY);
    let recorded = [SlotPose {
        slot: 3,
        position: old.0,
        rotation: old.1,
    }];
    apply_moves(
        &mut rows,
        &slot_moves(&recorded, &|s| (s == 3).then_some(new)),
    );
    // The mean keeps its coordinates in the view's camera frame.
    let local_before = Affine3A::from_rotation_translation(old.1, old.0)
        .inverse()
        .transform_point3(mean);
    let local_after = Affine3A::from_rotation_translation(new.1, new.0)
        .inverse()
        .transform_point3(rows.mean(0));
    assert!((local_before - local_after).length() < 1e-5);
    assert!(quat(&rows).dot(new.1 * old.1.inverse()).abs() > 1.0 - 1e-5);
}

/// Review Focus 4.
#[test]
fn two_corrections_while_parked_compose_to_the_latest_pose() {
    let p0 = (Vec3::new(1.0, 0.0, 2.0), Quat::from_rotation_y(0.1));
    let p1 = (Vec3::new(1.2, 0.0, 2.0), Quat::from_rotation_y(0.15));
    let p2 = (
        Vec3::new(1.25, 0.05, 1.9),
        Quat::from_rotation_x(0.2) * Quat::from_rotation_y(0.3),
    );
    let start = one_row(
        Some(3),
        Vec3::new(0.5, 0.2, 4.0),
        Quat::from_rotation_z(0.4),
    );
    let recorded = [SlotPose {
        slot: 3,
        position: p0.0,
        rotation: p0.1,
    }];
    let mut restored = start.clone();
    apply_moves(&mut restored, &slot_moves(&recorded, &|_| Some(p2)));
    let mut resident = start;
    apply_moves(&mut resident, &[(3, step(p0, p1))]);
    apply_moves(&mut resident, &[(3, step(p1, p2))]);
    assert!((restored.mean(0) - resident.mean(0)).length() < 1e-5);
    assert!(quat(&restored).dot(quat(&resident)).abs() > 1.0 - 1e-5);
}

#[test]
fn rows_without_origin_or_a_current_pose_stay() {
    let recorded = [SlotPose {
        slot: 1,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    }];
    let moves = slot_moves(&recorded, &|_| None);
    assert!(moves.is_empty());
    let mut rows = one_row(None, Vec3::ONE, Quat::IDENTITY);
    apply_moves(&mut rows, &[(1, Affine3A::from_translation(Vec3::X))]);
    assert_eq!(rows.mean(0), Vec3::ONE);
}

/// A view still at its park-time pose yields no move, so its rows come back bit for bit (no float noise from an
/// almost-identity transform), as `LiveModel::update_poses` skips a view already at the given pose.
#[test]
fn a_view_still_at_its_recorded_pose_moves_nothing() {
    let p = (
        Vec3::new(412.7, 1.3, -655.1),
        Quat::from_rotation_y(2.1) * Quat::from_rotation_x(0.3),
    );
    let recorded = [SlotPose {
        slot: 0,
        position: p.0,
        rotation: p.1,
    }];
    assert!(slot_moves(&recorded, &|_| Some(p)).is_empty());
}

fn group(origin: Option<u32>, n: usize) -> SplatRows {
    SplatRows::concat(&vec![
        one_row(
            origin,
            Vec3::new(1.0, 0.0, 3.0),
            Quat::IDENTITY
        );
        n
    ])
}

/// User decision D1 and Review Focus 5: one view carrying the tile far is consistent; two views disagreeing by
/// more than the limit are not; a stray view with a few rows does not decide.
#[test]
fn one_view_moving_far_is_consistent_two_views_disagreeing_are_not() {
    let shift = |x: f32| Affine3A::from_translation(Vec3::new(x, 0.0, 0.0));
    let one = group(Some(0), 40);
    assert_eq!(move_spread(&one, &[(0, shift(5.0))]), 0.0);
    let two = SplatRows::concat(&[group(Some(0), 50), group(Some(1), 50)]);
    assert!((move_spread(&two, &[(0, shift(0.6)), (1, shift(-0.6))]) - 1.2).abs() < 1e-5);
    let stray = SplatRows::concat(&[group(Some(0), 99), group(Some(1), 1)]);
    assert_eq!(
        move_spread(&stray, &[(0, shift(0.2)), (1, shift(3.0))]),
        0.0,
        "1 % of the rows: ignored"
    );
    let unanchored = SplatRows::concat(&[group(None, 50), group(Some(0), 50)]);
    assert!(
        (move_spread(&unanchored, &[(0, shift(2.0))]) - 2.0).abs() < 1e-5,
        "unanchored rows stay behind"
    );
    // A rotation of one view about its camera is rigid for that view's rows: consistent.
    let turn =
        Affine3A::from_rotation_translation(Quat::from_rotation_y(0.5), Vec3::new(0.3, 0.0, 0.0));
    assert_eq!(move_spread(&one, &[(0, turn)]), 0.0);
    assert_eq!(
        move_spread(&SplatRows::default(), &[(0, turn)]),
        0.0,
        "no rows"
    );
}

/// A slot without a move joins the unmoved rows rather than forming a group of its own.
#[test]
fn views_without_a_move_count_with_the_unanchored_rows() {
    let shift = Affine3A::from_translation(Vec3::new(0.0, 0.0, 1.5));
    let rows = SplatRows::concat(&[group(None, 30), group(Some(4), 30), group(Some(0), 40)]);
    assert!((move_spread(&rows, &[(0, shift)]) - 1.5).abs() < 1e-5);
}

async fn floats(t: Tensor<2>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_to_vec::<f32>()
        .expect("floats")
}

fn camera(p: CameraPose) -> Camera {
    Camera::new(
        p.0,
        p.1,
        1.0,
        1.0,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

/// The replay ends where the resident path would: the trainer's GPU move (`SplatTrainer::move_by_origin`), fed two
/// corrections built as `LiveModel::update_poses` builds them, against one replay from the park-time pose.
#[tokio::test]
async fn the_replay_matches_the_resident_move_on_the_gpu() {
    let p0 = (Vec3::new(30.0, 1.0, -42.0), Quat::from_rotation_y(0.1));
    let p1 = (Vec3::new(30.2, 1.0, -42.1), Quat::from_rotation_y(0.18));
    let p2 = (
        Vec3::new(30.1, 1.1, -41.9),
        Quat::from_rotation_x(0.1) * Quat::from_rotation_y(0.3),
    );
    let start = SplatRows::concat(
        &(0..6)
            .map(|i| {
                let tilt = Quat::from_rotation_z(0.3 * i as f32) * Quat::from_rotation_x(0.2);
                let mut r = one_row(Some(3), p0.0 + Vec3::new(i as f32 * 0.4, 0.1, 2.0), tilt);
                // Stored rotations are unnormalised; both paths keep their length.
                r.rotations
                    .iter_mut()
                    .for_each(|v| *v *= 1.0 + i as f32 * 0.5);
                r
            })
            .collect::<Vec<_>>(),
    );

    let device = Device::from(brush_cube::test_helpers::test_device().await);
    let splats = start.to_splats_raw(&device);
    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&TrainConfig::parse_from(["test"]), &device, bounds);
    trainer.track_origins(&splats, 3);
    let resident_move =
        |a: CameraPose, b: CameraPose| camera(b).local_to_world() * camera(a).world_to_local();
    let splats = trainer.move_by_origin(splats, &[(3, resident_move(p0, p1))]);
    let splats = trainer.move_by_origin(splats, &[(3, resident_move(p1, p2))]);
    let (means, rotations) = (
        floats(splats.means()).await,
        floats(splats.rotations()).await,
    );

    let mut restored = start;
    let recorded = [SlotPose {
        slot: 3,
        position: p0.0,
        rotation: p0.1,
    }];
    apply_moves(&mut restored, &slot_moves(&recorded, &|_| Some(p2)));
    for (a, b) in restored.means.iter().zip(&means) {
        assert!(
            (a - b).abs() < 1e-4,
            "means {:?} vs {means:?}",
            restored.means
        );
    }
    for (a, b) in restored.rotations.chunks(4).zip(rotations.chunks(4)) {
        let (qa, qb) = (
            Quat::from_xyzw(a[1], a[2], a[3], a[0]),
            Quat::from_xyzw(b[1], b[2], b[3], b[0]),
        );
        assert!(
            (qa.length() - qb.length()).abs() < 1e-4,
            "lengths {qa} vs {qb}"
        );
        assert!(
            qa.normalize().dot(qb.normalize()).abs() > 1.0 - 1e-5,
            "{qa} vs {qb}"
        );
    }
}
