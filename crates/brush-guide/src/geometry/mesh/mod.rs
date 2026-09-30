//! Triangle meshes of TSDF bricks by Naive Surface Nets
//! (`fast-surface-nets`, MIT/Apache-2.0).
//!
//! Each brick is meshed as a padded chunk, the crate's scheme for seamless
//! chunks: `Tsdf::brick_samples` adds a one-voxel border from the
//! neighbours, and `surface_nets` over the whole padded grid emits the faces
//! of exactly the grid edges whose far end lies in the brick, so adjacent
//! bricks share border vertices and neither duplicate nor miss faces.
//!
//! Like voxblox, only fully observed cells are meshed: a quad is dropped
//! when any of its four cells has an unobserved corner sample, which would
//! otherwise read as a fake surface between observed-inside and unobserved.
//!
//! Vertex colours are the volume's fused colour at the vertex
//! (`Tsdf::rgb`), encoded as 8-bit sRGB.

#[cfg(test)]
mod tests;

use fast_surface_nets::ndshape::ConstShape3u32;
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use glam::Vec3;

use super::colour::linear_to_srgb8;
use super::tsdf::{BRICK, BrickKey, PADDED, Tsdf, VOXEL};

const SIDE: u32 = PADDED as u32;

/// Shape of `BrickSamples`: x fastest, `x + PADDED·(y + PADDED·z)`.
type PaddedShape = ConstShape3u32<SIDE, SIDE, SIDE>;

/// Index offsets of a cell's eight corner samples from its minimal corner.
const CELL_CORNERS: [u32; 8] = [
    0,
    1,
    SIDE,
    SIDE + 1,
    SIDE * SIDE,
    SIDE * SIDE + 1,
    SIDE * SIDE + SIDE,
    SIDE * SIDE + SIDE + 1,
];

/// Mesh of one brick in world coordinates (metres). Normals are unit
/// gradients of the distance field, pointing out of the surface; triangles
/// wind counter-clockwise seen from the side the normals point to.
/// `colours` are 8-bit sRGB per vertex, or empty when the volume has no
/// colour at some vertex.
#[derive(Clone, Debug)]
pub struct BrickMesh {
    pub key: BrickKey,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colours: Vec<[u8; 3]>,
    pub indices: Vec<u32>,
}

/// Surface of brick `key` at distance 0; `None` when the brick does not
/// exist or has no observed surface.
pub fn mesh_brick(tsdf: &Tsdf, key: BrickKey) -> Option<BrickMesh> {
    let samples = tsdf.brick_samples(key)?;
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(
        &samples.sdf,
        &PaddedShape {},
        [0; 3],
        [SIDE - 1; 3],
        &mut buffer,
    );

    let observed: Vec<bool> = buffer
        .surface_strides
        .iter()
        .map(|&stride| {
            CELL_CORNERS
                .iter()
                .all(|&c| samples.weight[(stride + c) as usize] > 0.0)
        })
        .collect();

    // Sample `s` of the padded grid is the centre of global voxel
    // `key·BRICK + s − 1`, at `(key·BRICK + s − 0.5)·VOXEL`.
    let origin = (key.0 * BRICK).as_vec3() - 0.5;
    let mut mesh = BrickMesh {
        key,
        positions: Vec::new(),
        normals: Vec::new(),
        colours: Vec::new(),
        indices: Vec::new(),
    };
    let mut remap = vec![u32::MAX; buffer.positions.len()];
    // `surface_nets` emits every quad as six indices (two triangles).
    for quad in buffer.indices.as_chunks::<6>().0 {
        if !quad.iter().all(|&v| observed[v as usize]) {
            continue;
        }
        for &v in quad {
            let v = v as usize;
            if remap[v] == u32::MAX {
                remap[v] = mesh.positions.len() as u32;
                let local = Vec3::from(buffer.positions[v]);
                mesh.positions.push(((origin + local) * VOXEL).to_array());
                let normal = Vec3::from(buffer.normals[v]).normalize_or_zero();
                mesh.normals.push(normal.to_array());
            }
            mesh.indices.push(remap[v]);
        }
    }
    if mesh.indices.is_empty() {
        return None;
    }
    fill_zero_normals(&mut mesh);
    mesh.colours = vertex_colours(tsdf, &mesh.positions).unwrap_or_default();
    Some(mesh)
}

/// 8-bit sRGB of the volume's colour at every position, `None` if some
/// position has no colour.
fn vertex_colours(tsdf: &Tsdf, positions: &[[f32; 3]]) -> Option<Vec<[u8; 3]>> {
    positions
        .iter()
        .map(|&p| tsdf.rgb(Vec3::from(p)).map(linear_to_srgb8))
        .collect()
}

/// Gives vertices whose distance gradient vanished the normalised sum of
/// the area-weighted normals of their triangles.
fn fill_zero_normals(mesh: &mut BrickMesh) {
    let zero: Vec<bool> = mesh.normals.iter().map(|n| *n == [0.0; 3]).collect();
    if !zero.contains(&true) {
        return;
    }
    let mut sums = vec![Vec3::ZERO; mesh.positions.len()];
    for tri in mesh.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from(mesh.positions[i as usize]));
        let face = (b - a).cross(c - a);
        for &i in tri {
            sums[i as usize] += face;
        }
    }
    for (i, sum) in sums.into_iter().enumerate() {
        if zero[i] {
            mesh.normals[i] = sum.normalize_or_zero().to_array();
        }
    }
}
