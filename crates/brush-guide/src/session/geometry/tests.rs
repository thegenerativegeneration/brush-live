use super::*;

fn mesh(x: i32) -> MeshBrick {
    MeshBrick::Mesh(crate::geometry::mesh::BrickMesh {
        key: BrickKey(glam::IVec3::new(x, 0, 0)),
        positions: vec![[x as f32, 0.0, 0.0]; 3],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        indices: vec![0, 1, 2],
    })
}

fn keys(msg: &MeshBricksMsg) -> Vec<(i32, bool)> {
    msg.bricks
        .iter()
        .map(|b| (b.key().0.x, matches!(**b, MeshBrick::Removed(_))))
        .collect()
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

#[test]
fn rounds_integrate_new_views_and_rotate_through_older_ones() {
    let mut g = Geometry::new();
    assert_eq!(g.views_for_round(3), vec![0, 1, 2]);
    assert_eq!(g.views_for_round(5), vec![0, 1, 2, 3, 4]);
    assert_eq!(g.views_for_round(5), vec![3, 4, 0, 1]);
    assert_eq!(g.views_for_round(6), vec![2, 3, 4, 0, 5]);
    assert_eq!(g.views_for_round(6), vec![1, 2, 3, 4]);
}

/// A row of 30 bricks along x holding the plane z = 0.5.
fn plane_row() -> Geometry {
    let keys: Vec<BrickKey> = (0..30)
        .map(|x| BrickKey(glam::IVec3::new(x, 0, 0)))
        .collect();
    Geometry {
        tsdf: Tsdf::from_sdf(&keys, |p| Some(p.z - 0.5)),
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

#[test]
fn a_brick_losing_its_mesh_is_sent_as_removed() {
    let mut g = plane_row();
    g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    g.mesh_changed(Vec3::new(29.5, 0.5, 0.5));
    let key = BrickKey(glam::IVec3::new(29, 0, 0));
    // Brick 29 loses its observations (decayed below MIN_WEIGHT).
    g.tsdf.fill_sdf(&[key], |_| None);
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
