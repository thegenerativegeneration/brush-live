use super::mesh::MESH_BRICK_HEADER;
use super::tests::fixture_bricks;
use super::*;
use crate::geometry::mesh::BrickMesh;
use crate::geometry::tsdf::BrickKey;

#[test]
fn mesh_bricks_round_trip_with_removal() {
    let key = BrickKey(glam::IVec3::new(-3, 0, 2));
    let o = glam::Vec3::new(-3.0, 0.0, 2.0);
    let positions = [
        glam::Vec3::new(0.3, 0.7, 0.1),
        glam::Vec3::new(-0.025, 0.5, 1.025),
        glam::Vec3::new(1.0, 0.0, 0.5),
        glam::Vec3::new(0.123_45, 0.987_65, 0.5),
    ]
    .map(|p| (o + p).to_array());
    let normals = [
        glam::Vec3::new(0.6, 0.0, -0.8),
        glam::Vec3::Z,
        glam::Vec3::new(-0.48, 0.6, 0.64),
        glam::Vec3::NEG_Y,
    ];
    let bricks = [
        MeshBrick::Mesh(BrickMesh {
            key,
            positions: positions.to_vec(),
            normals: normals.map(|n| n.to_array()).to_vec(),
            indices: vec![0, 1, 2, 2, 1, 3],
        }),
        MeshBrick::Removed(BrickKey(glam::IVec3::new(7, -8, 9))),
    ];
    let bytes = encode_mesh_bricks(&bricks);
    assert_eq!(bytes.len(), 2 * MESH_BRICK_HEADER + 4 * 8 + 6 * 2);
    let back = decode_mesh_bricks(&bytes, 2).unwrap();
    let MeshBrick::Mesh(mesh) = &back[0] else {
        panic!("{:?}", back[0])
    };
    assert_eq!(mesh.key, key);
    assert_eq!(mesh.indices, vec![0, 1, 2, 2, 1, 3]);
    for (p, q) in positions.iter().zip(&mesh.positions) {
        let d = (glam::Vec3::from(*p) - glam::Vec3::from(*q))
            .abs()
            .max_element();
        assert!(d < 1e-5, "{p:?} vs {q:?}");
    }
    for (n, m) in normals.iter().zip(&mesh.normals) {
        assert!(n.dot(glam::Vec3::from(*m)) > 0.999, "{n} vs {m:?}");
    }
    assert!(matches!(back[1], MeshBrick::Removed(k) if k.0 == glam::IVec3::new(7, -8, 9)));
}

#[test]
fn mesh_bricks_byte_layout() {
    let bytes = encode_mesh_bricks(&fixture_bricks());
    let int = |i: usize| i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let half = |i: usize| u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap());
    assert_eq!((int(0), int(4), int(8)), (1, -2, 0));
    assert_eq!(bytes[12], 0, "flags");
    assert_eq!((word(13), word(17)), (3, 3));
    // Vertex (1, 0, 0) in the brick: x at the far side of the box, y and z at its origin.
    let v1 = 21 + 6;
    let at = |local: f32| ((local + 0.025) / 1.05 * 65535.0).round() as u16;
    assert_eq!(
        (half(v1), half(v1 + 2), half(v1 + 4)),
        (at(1.0), at(0.0), at(0.0))
    );
    let normals = 21 + 18;
    assert_eq!(&bytes[normals..normals + 2], &oct_encode(glam::Vec3::Z));
    let indices = normals + 6;
    assert_eq!(
        (half(indices), half(indices + 2), half(indices + 4)),
        (0, 1, 2)
    );
    let removed = indices + 6;
    assert_eq!(
        (int(removed), int(removed + 4), int(removed + 8)),
        (-1, 0, 3)
    );
    assert_eq!(bytes[removed + 12], MESH_BRICK_REMOVED);
    assert_eq!(&bytes[removed + 13..], &[0; 8]);
}

#[test]
fn mesh_bricks_reject_truncated_and_trailing_bytes() {
    let bytes = encode_mesh_bricks(&fixture_bricks());
    assert!(decode_mesh_bricks(&bytes[..bytes.len() - 1], 2).is_err());
    assert!(decode_mesh_bricks(&bytes, 1).is_err());
    assert!(decode_mesh_bricks(&bytes, 3).is_err());
}

#[test]
fn mesh_bricks_header_json() {
    let frame = encode_frame(
        &ServerHeader::MeshBricks {
            version: 5,
            num_bricks: 2,
            mesh_ms: 40,
        },
        &[],
    );
    let (h, _): (serde_json::Value, _) = decode_frame(&frame).unwrap();
    assert_eq!(
        h,
        serde_json::json!({"type": "mesh_bricks", "version": 5, "num_bricks": 2, "mesh_ms": 40})
    );
}

fn triangle_with_indices(indices: Vec<u32>) -> [MeshBrick; 1] {
    let MeshBrick::Mesh(mut mesh) = fixture_bricks()[0].clone() else {
        unreachable!()
    };
    mesh.indices = indices;
    [MeshBrick::Mesh(mesh)]
}

#[test]
fn mesh_bricks_reject_a_removed_brick_with_counts() {
    let mut bytes = encode_mesh_bricks(&fixture_bricks()[1..]);
    assert!(decode_mesh_bricks(&bytes, 1).is_ok());
    bytes[13] = 3;
    assert!(matches!(
        decode_mesh_bricks(&bytes, 1),
        Err(ProtocolError::RemovedBrickWithData)
    ));
}

#[test]
fn mesh_bricks_reject_an_index_count_not_a_multiple_of_three() {
    let bytes = encode_mesh_bricks(&triangle_with_indices(vec![0, 1, 2, 0]));
    assert!(matches!(
        decode_mesh_bricks(&bytes, 1),
        Err(ProtocolError::PartialTriangle(4))
    ));
}

#[test]
fn mesh_bricks_reject_indices_past_the_vertices() {
    let bytes = encode_mesh_bricks(&triangle_with_indices(vec![0, 1, 3]));
    assert!(matches!(
        decode_mesh_bricks(&bytes, 1),
        Err(ProtocolError::IndexOutOfRange {
            index: 3,
            num_vertices: 3
        })
    ));
}

fn mesh_with_vertices(n: usize) -> [MeshBrick; 1] {
    [MeshBrick::Mesh(BrickMesh {
        key: BrickKey(glam::IVec3::ZERO),
        positions: vec![[0.5; 3]; n],
        normals: vec![[0.0, 0.0, 1.0]; n],
        indices: vec![0, 1, (n - 1) as u32],
    })]
}

#[test]
fn mesh_bricks_encode_up_to_65536_vertices() {
    let bytes = encode_mesh_bricks(&mesh_with_vertices(65_536));
    let back = decode_mesh_bricks(&bytes, 1).unwrap();
    let MeshBrick::Mesh(mesh) = &back[0] else {
        panic!("{:?}", back[0].key())
    };
    assert_eq!(mesh.positions.len(), 65_536);
    assert_eq!(mesh.indices, vec![0, 1, 65_535]);
}

#[test]
#[should_panic(expected = "more than 65 536")]
fn mesh_bricks_reject_more_than_65536_vertices() {
    encode_mesh_bricks(&mesh_with_vertices(65_537));
}
