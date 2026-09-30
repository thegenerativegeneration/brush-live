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

use fast_surface_nets::ndshape::ConstShape3u32;
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use glam::Vec3;

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
#[derive(Clone, Debug)]
pub struct BrickMesh {
    pub key: BrickKey,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
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
    fill_zero_normals(&mut mesh);
    (!mesh.indices.is_empty()).then_some(mesh)
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use glam::{IVec3, vec3};

    use super::super::tsdf::TRUNC;
    use super::super::tsdf::tests::{
        ROD_CENTRES, depth_image, look_at, plane_z, rod_tsdf, thin_sheet_tsdf,
    };
    use super::*;

    fn triangles(mesh: &BrickMesh) -> impl Iterator<Item = [Vec3; 3]> + '_ {
        mesh.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| t.map(|i| Vec3::from(mesh.positions[i as usize])))
    }

    fn face_normal([a, b, c]: [Vec3; 3]) -> Vec3 {
        (b - a).cross(c - a)
    }

    fn area(meshes: &[BrickMesh]) -> f32 {
        meshes
            .iter()
            .flat_map(triangles)
            .map(|t| 0.5 * face_normal(t).length())
            .sum()
    }

    fn mesh_all(tsdf: &Tsdf, keys: &[BrickKey]) -> Vec<BrickMesh> {
        keys.iter().filter_map(|&k| mesh_brick(tsdf, k)).collect()
    }

    /// Welds vertices of all meshes closer than 10 µm (bricks compute shared
    /// border vertices from different local offsets) and returns the
    /// triangles as welded vertex ids.
    fn welded_triangles(meshes: &[BrickMesh]) -> Vec<[usize; 3]> {
        const CELL: f32 = 1e-5;
        let mut ids: HashMap<IVec3, usize> = HashMap::new();
        let mut next = 0;
        let mut weld = |p: Vec3| -> usize {
            let q = (p / CELL).round().as_ivec3();
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(&id) = ids.get(&(q + IVec3::new(dx, dy, dz))) {
                            return id;
                        }
                    }
                }
            }
            ids.insert(q, next);
            next += 1;
            next - 1
        };
        meshes
            .iter()
            .flat_map(|m| {
                m.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.map(|i| Vec3::from(m.positions[i as usize])))
            })
            .map(|t| t.map(&mut weld))
            .collect()
    }

    fn keys_around_origin() -> Vec<BrickKey> {
        let mut keys = Vec::new();
        for z in -1..=0 {
            for y in -1..=0 {
                for x in -1..=0 {
                    keys.push(BrickKey(IVec3::new(x, y, z)));
                }
            }
        }
        keys
    }

    const SPHERE_CENTRE: Vec3 = Vec3::new(0.013, -0.021, 0.007);
    const SPHERE_RADIUS: f32 = 0.4;

    fn sphere_meshes() -> Vec<BrickMesh> {
        let keys = keys_around_origin();
        let tsdf = Tsdf::from_sdf(&keys, |p| {
            Some((p - SPHERE_CENTRE).length() - SPHERE_RADIUS)
        });
        mesh_all(&tsdf, &keys)
    }

    #[test]
    fn sphere_over_eight_bricks_has_its_area() {
        let meshes = sphere_meshes();
        assert_eq!(meshes.len(), 8, "every brick holds part of the sphere");
        let expected = 4.0 * std::f32::consts::PI * SPHERE_RADIUS * SPHERE_RADIUS;
        let area = area(&meshes);
        assert!(
            (area / expected - 1.0).abs() < 0.05,
            "area {area} vs {expected}"
        );
    }

    /// Seam vertices sit in cells straddling the brick border, so a brick's
    /// vertices reach up to half a voxel beyond its box on every side; the
    /// wire format quantises positions over that widened box.
    #[test]
    fn vertices_lie_within_half_a_voxel_of_their_brick() {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for m in sphere_meshes() {
            let origin = (m.key.0 * BRICK).as_vec3() * VOXEL;
            for &p in &m.positions {
                let local = Vec3::from(p) - origin;
                lo = lo.min(local.min_element());
                hi = hi.max(local.max_element());
            }
        }
        let size = BRICK as f32 * VOXEL;
        assert!(
            lo >= -0.5 * VOXEL - 1e-5 && hi <= size + 0.5 * VOXEL + 1e-5,
            "local range [{lo}, {hi}]"
        );
        assert!(lo < 0.0 && hi > size, "local range [{lo}, {hi}]");
    }

    #[test]
    fn sphere_is_closed_across_brick_seams() {
        assert_closed(&sphere_meshes());
    }

    /// Asserts that every edge of `meshes` is shared by exactly two
    /// consistently oriented triangles.
    fn assert_closed(meshes: &[BrickMesh]) {
        let triangles = welded_triangles(meshes);
        assert!(!triangles.is_empty());
        let mut directed: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for [a, b, c] in &triangles {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *directed.entry((*u, *v)).or_default() += 1;
            }
        }
        for (&(u, v), &n) in &directed {
            assert_eq!(n, 1, "edge {u}->{v} used by {n} triangles");
            let back = directed.get(&(v, u)).copied().unwrap_or(0);
            assert_eq!(back, 1, "edge {u}-{v} shared by {} triangles", n + back);
        }
    }

    /// Brick A is meshed alone, then its neighbour B is filled: A is
    /// re-meshed, and the meshes kept per brick close at the seam.
    #[test]
    fn neighbour_filled_later_remeshes_the_seam() {
        let centre = vec3(1.013, 0.52, 0.47);
        let sphere = |p: Vec3| Some((p - centre).length() - 0.3);
        let (a, b) = (BrickKey(IVec3::ZERO), BrickKey(IVec3::X));
        let mut tsdf = Tsdf::from_sdf(&[a], sphere);
        let mut meshes: BTreeMap<BrickKey, Option<BrickMesh>> = BTreeMap::new();
        let remesh = |tsdf: &mut Tsdf, meshes: &mut BTreeMap<_, _>| {
            for key in tsdf.take_changed() {
                meshes.insert(key, mesh_brick(tsdf, key));
            }
        };
        remesh(&mut tsdf, &mut meshes);
        assert!(meshes.contains_key(&a));

        tsdf.fill_sdf(&[b], sphere);
        remesh(&mut tsdf, &mut meshes);
        let kept: Vec<BrickMesh> = meshes.into_values().flatten().collect();
        assert_eq!(kept.len(), 2);
        assert_closed(&kept);
    }

    #[test]
    fn sphere_normals_point_outwards_and_match_winding() {
        let meshes = sphere_meshes();
        assert!(!meshes.is_empty());
        for mesh in meshes {
            for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
                let (p, n) = (Vec3::from(*p), Vec3::from(*n));
                let radial = p - SPHERE_CENTRE;
                assert!(
                    (radial.length() - SPHERE_RADIUS).abs() < 0.01,
                    "vertex {p} off the sphere"
                );
                assert!((n.length() - 1.0).abs() < 1e-4, "normal {n} not unit");
                assert!(n.dot(radial.normalize()) > 0.95, "normal {n} at {p}");
            }
            for (t, idx) in triangles(&mesh).zip(mesh.indices.as_chunks::<3>().0) {
                let vertex_normal = Vec3::from(mesh.normals[idx[0] as usize]);
                assert!(face_normal(t).dot(vertex_normal) > 0.0, "winding of {t:?}");
            }
        }
    }

    #[test]
    fn tilted_plane_gives_flat_triangles() {
        let normal = vec3(0.2, -0.3, 1.0).normalize();
        let point = vec3(0.5, 0.5, 2.5);
        let key = BrickKey(IVec3::new(0, 0, 2));
        let tsdf = Tsdf::from_sdf(&[key], |p| Some(normal.dot(p - point)));
        let mesh = mesh_brick(&tsdf, key).expect("plane crosses the brick");

        let max_angle = 2f32.to_radians();
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let d = normal.dot(Vec3::from(*p) - point);
            assert!(d.abs() < 1e-4, "vertex {p:?} {d} m off the plane");
            assert!(
                Vec3::from(*n).angle_between(normal) < max_angle,
                "normal {n:?}"
            );
        }
        for t in triangles(&mesh) {
            let angle = face_normal(t).angle_between(normal);
            assert!(angle < max_angle, "face {t:?} at {}°", angle.to_degrees());
        }
        // One brick without neighbours: the border cells touch unobserved
        // padding, so the mesh covers the plane inside [0.025, 0.975]² of
        // the brick's footprint in x and y, about 0.9 m² tilted.
        let area = area(std::slice::from_ref(&mesh));
        assert!(area > 0.7, "area {area}");
    }

    #[test]
    fn zero_gradient_falls_back_to_the_face_normal() {
        let mut mesh = BrickMesh {
            key: BrickKey(IVec3::ZERO),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0; 3]],
            indices: vec![0, 1, 2],
        };
        fill_zero_normals(&mut mesh);
        assert_eq!(
            mesh.normals,
            vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn free_space_and_missing_bricks_have_no_mesh() {
        let key = BrickKey(IVec3::new(0, 0, 0));
        let free = Tsdf::from_sdf(&[key], |_| Some(1.0));
        assert!(mesh_brick(&free, key).is_none());
        let solid = Tsdf::from_sdf(&[key], |_| Some(-1.0));
        assert!(mesh_brick(&solid, key).is_none());
        let unobserved = Tsdf::from_sdf(&[key], |_| None);
        assert!(mesh_brick(&unobserved, key).is_none());
        assert!(mesh_brick(&free, BrickKey(IVec3::new(1, 0, 0))).is_none());
    }

    /// Behind an observed surface the volume turns unobserved (distance +1):
    /// no faces between the observed-inside band and the unobserved region.
    #[test]
    fn unobserved_samples_make_no_faces() {
        let key = BrickKey(IVec3::new(0, 0, 2));
        let tsdf = Tsdf::from_sdf(&[key], |p| {
            let d = 2.5 - p.z;
            (d > -TRUNC && p.x > 0.5).then_some(d)
        });
        let mesh = mesh_brick(&tsdf, key).expect("plane observed");
        for p in &mesh.positions {
            assert!((p[2] - 2.5).abs() < 1e-4, "vertex {p:?} off the plane");
            assert!(p[0] > 0.5, "vertex {p:?} in the unobserved half");
        }

        let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
        let mut fused = Tsdf::new();
        fused.integrate(&depth_image(&cam, plane_z(2.5)), &cam);
        let mesh = mesh_brick(&fused, key).expect("fused plane");
        for p in &mesh.positions {
            assert!(
                (p[2] - 2.5).abs() < 0.01,
                "vertex {p:?} off the fused plane"
            );
        }
    }

    #[test]
    fn thin_sheet_is_meshed() {
        let mut tsdf = thin_sheet_tsdf();
        let keys = tsdf.take_changed();
        let meshes = mesh_all(&tsdf, &keys);
        // Near the rim one camera sees past the sheet, and its negative band
        // behind the sheet is not cleared by the other view; those fused
        // voxels hold a real zero crossing off the sheet. Only vertices
        // more than 10 cm inside the 0.6 m half-width rim are checked.
        let inner: Vec<&[f32; 3]> = meshes
            .iter()
            .flat_map(|m| &m.positions)
            .filter(|p| p[0].abs() < 0.5 && p[1].abs() < 0.5)
            .collect();
        assert!(inner.len() > 100, "{} vertices inside the rim", inner.len());
        for p in inner {
            assert!((p[2] - 2.0).abs() <= VOXEL, "vertex {p:?} off the sheet");
        }
    }

    #[test]
    fn rod_is_meshed() {
        for center in ROD_CENTRES {
            let mut tsdf = rod_tsdf(center, 0.05);
            let keys = tsdf.take_changed();
            let meshes = mesh_all(&tsdf, &keys);
            for y in [-0.3, 0.0, 0.3] {
                let near_rod = meshes.iter().flat_map(|m| &m.positions).any(|p| {
                    let p = Vec3::from(*p) - center;
                    vec3(p.x, 0.0, p.z).length() < 0.025 + VOXEL && (p.y - y).abs() < 0.1
                });
                assert!(near_rod, "rod at {center}: no mesh near height {y}");
            }
        }
    }
}
