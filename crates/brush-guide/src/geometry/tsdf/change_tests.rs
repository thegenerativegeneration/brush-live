//! Tests of stale-brick detection: `changed`, `mark_meshed`, `take_changed`.

use glam::vec3;

use super::fixtures::{depth_image, integrate, look_at, plane_setup, plane_z};
use super::*;

#[test]
fn take_changed_follows_mean_change() {
    let (mut tsdf, cam) = plane_setup();
    let brick = BrickKey(IVec3::new(0, 0, 2));
    let changed = tsdf.take_changed();
    assert!(changed.contains(&brick), "first fusion reports {changed:?}");
    assert!(
        tsdf.take_changed().is_empty(),
        "nothing left after extraction"
    );

    // Identical depth on a converged volume changes nothing.
    let same = depth_image(&cam, plane_z(2.5));
    for _ in 0..5 {
        tsdf.integrate(&same, &cam);
    }
    assert!(tsdf.take_changed().is_empty());

    // Weight 6 now: one view of a plane 1 cm further moves the
    // near-surface voxels by 1/7 cm.
    let shifted = depth_image(&cam, plane_z(2.51));
    tsdf.integrate(&shifted, &cam);
    assert!(
        tsdf.take_changed().is_empty(),
        "sub-threshold change reported"
    );

    // Converging on the shifted plane moves them by nearly 1 cm.
    for _ in 0..30 {
        tsdf.integrate(&shifted, &cam);
    }
    assert!(
        tsdf.take_changed().contains(&brick),
        "1 cm shift of a converged plane not reported"
    );

    // A 20 cm jump moves every band voxel by centimetres.
    for _ in 0..3 {
        tsdf.integrate(&depth_image(&cam, plane_z(2.7)), &cam);
    }
    assert!(tsdf.take_changed().contains(&brick));
}

/// Wall at z = 2.9 m across bricks A = (0, 0, 2) and its +x neighbour B.
fn wall_pair() -> (Tsdf, BrickKey, BrickKey) {
    let (a, b) = (BrickKey(IVec3::new(0, 0, 2)), BrickKey(IVec3::new(1, 0, 2)));
    let tsdf = Tsdf::from_sdf(&[a, b], |p| Some(2.9 - p.z));
    (tsdf, a, b)
}

#[test]
fn changed_stays_pending_until_marked() {
    let (mut tsdf, a, b) = wall_pair();
    assert_eq!(tsdf.changed(), vec![a, b]);
    tsdf.mark_meshed(&[a]);
    assert_eq!(tsdf.changed(), vec![b], "unmarked brick dropped");
    assert_eq!(tsdf.changed(), vec![b], "changed() consumed state");
    tsdf.mark_meshed(&[b]);
    assert!(tsdf.changed().is_empty());
}

/// A 15 cm cube (27 voxels) appearing half a metre in front of the wall:
/// its mean change over the wall's near-surface band is ~2 mm, but 27
/// voxels change sign.
#[test]
fn small_object_next_to_a_wall_is_reported() {
    let (mut tsdf, a, b) = wall_pair();
    tsdf.take_changed();
    let inside = |p: Vec3| {
        (p - vec3(0.375, 0.375, 2.375))
            .abs()
            .cmple(Vec3::splat(0.075))
            .all()
    };
    tsdf.fill_sdf(&[a], |p| Some(if inside(p) { -0.01 } else { 2.9 - p.z }));
    assert_eq!(tsdf.changed(), vec![a], "{b:?} unaffected");
}

#[test]
fn neighbour_allocation_and_seam_layer_changes_are_reported() {
    let (mut tsdf, a, b) = wall_pair();
    tsdf.take_changed();

    // An empty brick next to A and diagonally next to B.
    let above = BrickKey(IVec3::new(0, 1, 2));
    tsdf.fill_sdf(&[above], |_| None);
    assert_eq!(tsdf.changed(), vec![a, b], "allocated neighbour");
    tsdf.take_changed();

    // B's voxel at the seam, just in front of the wall, drops below
    // UNOBSERVED_WEIGHT: A (padded with it) and B are stale.
    let seam = IVec3::new(BRICK, 5, 2 * BRICK + 17);
    let (t, _) = tsdf.voxel(seam).unwrap();
    tsdf.set_voxel(seam, t, 0.5 * UNOBSERVED_WEIGHT);
    assert!(tsdf.changed().contains(&a), "{:?}", tsdf.changed());
    tsdf.take_changed();

    // A seam voxel more than half the truncation band from the surface
    // turning unobserved, or observed again, is no change.
    let far = IVec3::new(BRICK, 5, 2 * BRICK + 10);
    let (t, w) = tsdf.voxel(far).unwrap();
    assert!(t.abs() >= 0.5, "{t}");
    tsdf.set_voxel(far, t, 0.5 * UNOBSERVED_WEIGHT);
    assert!(tsdf.changed().is_empty(), "{:?}", tsdf.changed());
    tsdf.set_voxel(far, t, w);
    assert!(tsdf.changed().is_empty(), "{:?}", tsdf.changed());

    // B's seam layer moves 2 cm; B's own mean over its band stays small.
    for y in 0..BRICK {
        for z in 2 * BRICK + 14..2 * BRICK + 20 {
            let g = IVec3::new(BRICK, y, z);
            let (t, w) = tsdf.voxel(g).unwrap();
            tsdf.set_voxel(g, t - 0.02 / TRUNC, w);
        }
    }
    let changed = tsdf.changed();
    assert!(changed.contains(&a), "{changed:?}");
    assert!(!changed.contains(&b), "{changed:?}");

    // Voxels of B away from the seam do not concern A.
    tsdf.take_changed();
    let inner = IVec3::new(BRICK + 5, 5, 2 * BRICK + 17);
    let (t, _) = tsdf.voxel(inner).unwrap();
    tsdf.set_voxel(inner, t, 0.0);
    assert!(!tsdf.changed().contains(&a));
}

/// A static plane seen by eight views, re-integrated four views per
/// round with decay, as the session does after the last keyframe: once
/// meshed, no brick turns stale again.
#[test]
fn static_plane_drains_after_the_last_keyframe() {
    let target = vec3(0.0, 0.0, 2.0);
    let cams: Vec<Camera> = (0..8)
        .map(|i| {
            let a = i as f32 * 0.15 - 0.5;
            look_at(vec3(a, 0.1 * a, 0.0), target)
        })
        .collect();
    let mut tsdf = Tsdf::new();
    integrate(&mut tsdf, &cams, &plane_z(2.0));
    let mut counts = Vec::new();
    let mut cursor = 0;
    for _ in 0..60 {
        let stale = tsdf.changed();
        counts.push(stale.len());
        tsdf.mark_meshed(&stale);
        tsdf.decay(0.95);
        for _ in 0..4 {
            let cam = &cams[cursor % cams.len()];
            cursor += 1;
            tsdf.integrate(&depth_image(cam, plane_z(2.0)), cam);
        }
    }
    assert!(
        counts[5..].iter().all(|&n| n == 0),
        "stale bricks per round: {counts:?}"
    );
}
