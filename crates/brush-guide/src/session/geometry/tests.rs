use super::*;
use crate::geometry::tsdf::fixtures::{colour_image, depth_image, look_at, plane_z};

fn mesh(x: i32) -> MeshBrick {
    MeshBrick::Mesh(crate::geometry::mesh::BrickMesh {
        key: BrickKey(glam::IVec3::new(x, 0, 0)),
        positions: vec![[x as f32, 0.0, 0.0]; 3],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        colours: Vec::new(),
        indices: vec![0, 1, 2],
    })
}

fn keys(msg: &MeshBricksMsg) -> Vec<(i32, bool)> {
    msg.bricks
        .iter()
        .map(|b| (b.key().0.x, matches!(**b, MeshBrick::Removed(_))))
        .collect()
}

/// A reader that last sent round 0 and reads the log after rounds 1 and 2
/// were recorded gets both rounds in one message of version 2: versions
/// can run ahead of the score set the reader just saw.
#[test]
fn a_late_reader_gets_coalesced_rounds_with_the_newer_version() {
    let mut log = MeshLog::default();
    log.record(1, 10, vec![mesh(1)]);
    log.record(2, 20, vec![mesh(2)]);
    let msg = log.since(0).unwrap();
    assert_eq!(msg.version, 2);
    assert_eq!(keys(&msg), vec![(1, false), (2, false)]);
}

#[test]
fn mesh_log_catches_up_with_the_latest_state() {
    let mut log = MeshLog::default();
    assert!(log.since(0).is_none());
    log.record(1, 10, vec![mesh(1), mesh(2)]);
    log.record(
        2,
        20,
        vec![MeshBrick::Removed(BrickKey(glam::IVec3::X)), mesh(3)],
    );
    let all = log.since(0).unwrap();
    assert_eq!((all.version, all.mesh_ms), (2, 20));
    assert_eq!(keys(&all), vec![(2, false), (1, true), (3, false)]);
    assert_eq!(keys(&log.since(1).unwrap()), vec![(1, true), (3, false)]);
    assert!(log.since(2).is_none());
    log.record(3, 5, Vec::new());
    let empty = log.since(2).unwrap();
    assert_eq!((empty.version, empty.bricks.len()), (3, 0));
}

/// 20 new views at once are fused over three rounds, 8 new per round,
/// none lost, while older views keep being refreshed.
#[test]
fn new_views_are_capped_per_round_and_carried_over() {
    let mut g = Geometry::new();
    assert_eq!(g.views_for_round(20), (0..8).collect::<Vec<_>>());
    assert_eq!(
        g.views_for_round(20),
        [0, 1, 2, 3].into_iter().chain(8..16).collect::<Vec<_>>()
    );
    assert_eq!(
        g.views_for_round(20),
        [4, 5, 6, 7].into_iter().chain(16..20).collect::<Vec<_>>()
    );
    assert_eq!(g.views_for_round(20), vec![8, 9, 10, 11], "only refreshes");
}

#[test]
fn rounds_integrate_new_views_and_rotate_through_older_ones() {
    let mut g = Geometry::new();
    assert_eq!(g.views_for_round(3), vec![0, 1, 2]);
    assert_eq!(g.views_for_round(5), vec![0, 1, 2, 3, 4]);
    assert_eq!(g.views_for_round(5), vec![3, 4, 0, 1]);
    assert_eq!(g.views_for_round(6), vec![2, 3, 4, 0, 5]);
    assert_eq!(g.views_for_round(6), vec![1, 2, 3, 4]);
}

/// The cap the queueing tests below are written for.
const CAP: usize = 24;

#[test]
fn default_cap_is_96_bricks_per_round() {
    assert_eq!(Geometry::new().max_bricks, 96);
}

/// A row of 30 bricks along x holding the plane z = 0.5, meshed at most
/// `CAP` per round.
fn plane_row() -> Geometry {
    let keys: Vec<BrickKey> = (0..30)
        .map(|x| BrickKey(glam::IVec3::new(x, 0, 0)))
        .collect();
    Geometry {
        tsdf: Tsdf::from_sdf(&keys, |p| Some(p.z - 0.5)),
        max_bricks: CAP,
        ..Geometry::new()
    }
}

#[test]
fn meshing_is_capped_nearest_first_and_the_rest_stays_pending() {
    let mut g = plane_row();
    let (bricks, pending) = g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    assert_eq!(pending, 30);
    let xs: Vec<i32> = bricks.iter().map(|b| b.key().0.x).collect();
    assert_eq!(xs, (6..30).rev().collect::<Vec<_>>());
    assert!(bricks.iter().all(|b| matches!(b, MeshBrick::Mesh(_))));
    let rest: Vec<i32> = g.tsdf.changed().iter().map(|k| k.0.x).collect();
    assert_eq!(rest, (0..6).collect::<Vec<_>>());

    let (bricks, pending) = g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    assert_eq!(pending, 6);
    let xs: Vec<i32> = bricks.iter().map(|b| b.key().0.x).collect();
    assert_eq!(xs, (0..6).rev().collect::<Vec<_>>());
    let (bricks, pending) = g.mesh_changed(Vec3::ZERO);
    assert_eq!((bricks.len(), pending), (0, 0));
}

/// Near bricks that keep changing must not starve far ones: with 30
/// stale bricks and the cap of 24, every brick is meshed within two
/// rounds even though the 24 nearest change again in between.
#[test]
fn meshing_queue_is_fair() {
    let mut g = plane_row();
    let eye = Vec3::new(29.5, 0.5, 0.5);
    let mut meshed = HashSet::new();
    for round in 0..2 {
        let (bricks, _) = g.mesh_changed(eye);
        meshed.extend(bricks.iter().map(|b| b.key().0.x));
        // The nearest 24 bricks move by 2 cm (alternating).
        let near: Vec<BrickKey> = (6..30)
            .map(|x| BrickKey(glam::IVec3::new(x, 0, 0)))
            .collect();
        let z = if round % 2 == 0 { 0.52 } else { 0.5 };
        g.tsdf.fill_sdf(&near, |p| Some(p.z - z));
    }
    let missing: Vec<i32> = (0..30).filter(|x| !meshed.contains(x)).collect();
    assert!(missing.is_empty(), "never meshed: {missing:?}");
}

/// Among bricks meshed before, the one stale longest goes first: with
/// every brick meshed once and all 30 stale again, the 6 far ones wait one
/// round behind the 24 near ones, then go ahead of them although the near
/// ones changed again in between.
#[test]
fn meshing_queue_is_fair_among_bricks_meshed_before() {
    let mut g = plane_row();
    let eye = Vec3::new(29.5, 0.5, 0.5);
    g.mesh_changed(eye);
    g.mesh_changed(eye);
    assert!(g.tsdf.changed().is_empty(), "all bricks meshed");
    let row = |xs: std::ops::Range<i32>| -> Vec<BrickKey> {
        xs.map(|x| BrickKey(glam::IVec3::new(x, 0, 0))).collect()
    };
    g.tsdf.fill_sdf(&row(0..30), |p| Some(p.z - 0.52));
    let mut far_meshed = HashSet::new();
    for round in 0..2 {
        let (bricks, pending) = g.mesh_changed(eye);
        assert_eq!(pending, 30, "round {round}");
        assert_eq!(bricks.len(), CAP, "round {round}");
        far_meshed.extend(bricks.iter().map(|b| b.key().0.x).filter(|&x| x < 6));
        // The near bricks move by 2 cm again (alternating).
        let z = if round % 2 == 0 { 0.5 } else { 0.52 };
        g.tsdf.fill_sdf(&row(6..30), |p| Some(p.z - z));
    }
    let missing: Vec<i32> = (0..6).filter(|x| !far_meshed.contains(x)).collect();
    assert!(missing.is_empty(), "far bricks never meshed: {missing:?}");
}

#[test]
fn a_brick_losing_its_mesh_is_sent_as_removed() {
    let mut g = plane_row();
    g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    let key = BrickKey(glam::IVec3::new(29, 0, 0));
    // Free space seen through bricks 28 and 29 clears their surface: they
    // are observed 10 cm in front of any surface, so 29 has no zero
    // crossing left, not even in the layer it is padded with from 28.
    let cleared = [BrickKey(glam::IVec3::new(28, 0, 0)), key];
    g.tsdf.fill_sdf(&cleared, |_| Some(0.1));
    let (bricks, _) = g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    assert!(
        bricks
            .iter()
            .any(|b| matches!(b, MeshBrick::Removed(k) if *k == key)),
        "{:?}",
        bricks.iter().map(MeshBrick::key).collect::<Vec<_>>()
    );
    assert!(!g.meshed.contains(&key));
}

#[test]
fn a_brick_that_never_had_a_mesh_is_not_sent() {
    let key = BrickKey(glam::IVec3::ZERO);
    // Near the surface everywhere but no zero crossing: stale, no mesh.
    let mut g = Geometry {
        tsdf: Tsdf::from_sdf(&[key], |_| Some(0.05)),
        ..Geometry::new()
    };
    let (bricks, pending) = g.mesh_changed(Vec3::ZERO);
    assert_eq!((bricks.len(), pending), (0, 1));
    assert!(g.tsdf.changed().is_empty(), "marked meshed");
}

/// Fuses plane z = 2.5, seen head-on from the origin and painted `rgb`
/// (linear), `n` times.
fn paint_plane(g: &mut Geometry, rgb: [f32; 3], n: usize) {
    let cam = look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 2.5));
    let depth = depth_image(&cam, plane_z(2.5));
    let colour = colour_image(&cam, plane_z(2.5), |_| rgb);
    for _ in 0..n {
        g.tsdf.integrate(&depth, Some(&colour), &cam);
    }
}

#[test]
fn a_repainted_plane_is_remeshed_with_its_new_colour() {
    let mut g = Geometry::new();
    paint_plane(&mut g, [0.2; 3], 30);
    let (bricks, pending) = g.mesh_changed(Vec3::ZERO);
    assert!(!bricks.is_empty() && pending <= MAX_BRICKS_PER_ROUND);
    paint_plane(&mut g, [1.0, 0.0, 0.0], 150);
    assert!(g.tsdf.changed().is_empty(), "geometry unchanged");

    let (bricks, pending) = g.mesh_changed(Vec3::ZERO);
    assert_eq!(bricks.len(), pending);
    assert!(!bricks.is_empty(), "repainted bricks re-meshed");
    for b in &bricks {
        let MeshBrick::Mesh(m) = b else {
            panic!("{:?} removed", b.key())
        };
        assert_eq!(m.colours.len(), m.positions.len());
        for c in &m.colours {
            assert!(c[0] >= 250 && c[1] <= 5 && c[2] <= 5, "{c:?}");
        }
    }
    assert_eq!(g.mesh_changed(Vec3::ZERO).1, 0, "nothing stale after");
}

/// Geometry changes go first: with 26 bricks moved (25 and the neighbour
/// padded with one of them) and 4 repainted, the repainted ones wait
/// behind the cap of 24 and follow in the next round.
#[test]
fn colour_only_changes_wait_behind_geometry_changes() {
    let row = |xs: std::ops::Range<i32>| -> Vec<BrickKey> {
        xs.map(|x| BrickKey(glam::IVec3::new(x, 0, 0))).collect()
    };
    let mut g = plane_row();
    g.tsdf.fill_colour(&row(0..30), [0.2; 3]);
    let eye = Vec3::new(29.5, 0.5, 0.5);
    g.mesh_changed(eye);
    g.mesh_changed(eye);
    assert_eq!(g.mesh_changed(eye).1, 0, "all meshed");

    g.tsdf.fill_colour(&row(0..4), [1.0, 0.0, 0.0]);
    g.tsdf.fill_sdf(&row(5..30), |p| Some(p.z - 0.52));
    let (bricks, pending) = g.mesh_changed(eye);
    assert_eq!((bricks.len(), pending), (CAP, 30));
    let xs: Vec<i32> = bricks.iter().map(|b| b.key().0.x).collect();
    assert!(xs.iter().all(|&x| x >= 4), "geometry first: {xs:?}");

    let (bricks, pending) = g.mesh_changed(eye);
    assert_eq!(pending, 6);
    let xs: Vec<i32> = bricks.iter().map(|b| b.key().0.x).collect();
    assert_eq!(xs.len(), 6, "{xs:?}");
    assert!(
        xs[..2].iter().all(|&x| x >= 4),
        "geometry leftovers first: {xs:?}"
    );
    let mut repainted = xs[2..].to_vec();
    repainted.sort_unstable();
    assert_eq!(repainted, vec![0, 1, 2, 3]);
}
