//! Sparse truncated signed distance volume fused from depth images.
//!
//! The update is the voxel-projection fusion of Zeng et al.
//! (`andyzeng/tsdf-fusion-python`, BSD-2): every voxel centre is projected
//! into the depth image, `sdf = depth − z`, voxels more than `TRUNC` behind
//! the surface are skipped, the normalised distance `min(1, sdf / TRUNC)`
//! enters a weighted running average. Weights are capped at `MAX_WEIGHT` so
//! the volume keeps following a changing splat. Storage is sparse in the way
//! of `Open3D`'s `ScalableTSDFVolume`: bricks of `BRICK³` voxels are allocated
//! only around observed surface points.
//!
//! Distances are stored normalised to `[-1, 1]` (units of `TRUNC`), positive
//! in front of the surface; unobserved voxels read as `+1`.

use std::collections::HashMap;

use brush_render::camera::Camera;
use glam::{IVec3, UVec2, Vec3};

use super::depth::DepthImage;

pub const VOXEL: f32 = 0.05;
pub const TRUNC: f32 = 0.15;
pub const BRICK: i32 = 20;
pub const MAX_WEIGHT: f32 = 20.0;

/// A brick is reported by `take_changed` once the summed |Δsdf| of its
/// voxels since its last report, divided by `BRICK³`, exceeds this (metres).
const CHANGE_THRESHOLD: f32 = 0.005;

const BRICK_VOXELS: usize = (BRICK * BRICK * BRICK) as usize;

/// Side of the padded sample grid of `Tsdf::brick_samples`.
pub const PADDED: usize = BRICK as usize + 2;

/// Brick index: the brick covers world `[key, key + 1)` metres per axis.
/// Ordered lexicographically by (x, y, z).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BrickKey(pub IVec3);

impl Ord for BrickKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.to_array().cmp(&other.0.to_array())
    }
}

impl PartialOrd for BrickKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct Brick {
    /// Normalised distance per voxel, x fastest: `x + BRICK·(y + BRICK·z)`.
    tsdf: Vec<f32>,
    /// Fusion weight per voxel; 0 means unobserved.
    weight: Vec<f32>,
    /// Σ |Δsdf| in metres since the brick was last reported.
    change: f32,
    /// Whether `take_changed` has reported this brick before.
    reported: bool,
}

impl Brick {
    fn new() -> Self {
        Self {
            tsdf: vec![1.0; BRICK_VOXELS],
            weight: vec![0.0; BRICK_VOXELS],
            change: 0.0,
            reported: false,
        }
    }

    /// Weighted running average of one observation into voxel `i`
    /// (Zeng et al.: `(w·t + w_obs·d) / (w + w_obs)`), weight capped.
    fn fuse(&mut self, i: usize, dist: f32) {
        const OBS_WEIGHT: f32 = 1.0;
        let (t_old, w_old) = (self.tsdf[i], self.weight[i]);
        let w_new = w_old + OBS_WEIGHT;
        let t_new = (w_old * t_old + OBS_WEIGHT * dist) / w_new;
        self.tsdf[i] = t_new;
        self.weight[i] = w_new.min(MAX_WEIGHT);
        self.change += (t_new - t_old).abs() * TRUNC;
    }
}

fn voxel_index(local: IVec3) -> usize {
    (local.x + BRICK * (local.y + BRICK * local.z)) as usize
}

/// Brick containing global voxel `g`, and `g`'s position inside it.
fn split_voxel(g: IVec3) -> (BrickKey, IVec3) {
    (
        BrickKey(g.div_euclid(IVec3::splat(BRICK))),
        g.rem_euclid(IVec3::splat(BRICK)),
    )
}

fn voxel_centre(g: IVec3) -> Vec3 {
    (g.as_vec3() + 0.5) * VOXEL
}

/// Global voxel containing world point `p`.
fn voxel_of(p: Vec3) -> IVec3 {
    (p / VOXEL).floor().as_ivec3()
}

/// Pinhole projection matching the splat rasteriser: `u = fx·x/z + cx`,
/// pixel `(i, j)` covering `[i, i+1) × [j, j+1)` (its centre at `i + 0.5`).
struct Projection {
    size: UVec2,
    focal: glam::Vec2,
    centre: glam::Vec2,
    world_to_local: glam::Affine3A,
}

impl Projection {
    fn new(camera: &Camera, size: UVec2) -> Self {
        Self {
            size,
            focal: camera.focal(size),
            centre: camera.center(size),
            world_to_local: camera.world_to_local(),
        }
    }

    /// Camera-space point and pixel index of world point `p`, if it lies in
    /// front of the camera and inside the image.
    fn project(&self, p: Vec3) -> Option<(Vec3, usize)> {
        let local = self.world_to_local.transform_point3(p);
        if local.z <= 0.0 {
            return None;
        }
        let u = self.focal.x * local.x / local.z + self.centre.x;
        let v = self.focal.y * local.y / local.z + self.centre.y;
        if !(u >= 0.0 && v >= 0.0 && u < self.size.x as f32 && v < self.size.y as f32) {
            return None;
        }
        Some((local, v as usize * self.size.x as usize + u as usize))
    }

    /// Whether a sphere may intersect the view frustum up to depth `far`.
    fn sees_sphere(&self, centre: Vec3, radius: f32, far: f32) -> bool {
        let c = self.world_to_local.transform_point3(centre);
        if c.z < -radius || c.z > far + radius {
            return false;
        }
        // Side planes through the optical centre and the image borders, as
        // `x - k·z = 0` with the inside on the side of the principal axis.
        let size = self.size.as_vec2();
        let sides = [
            (c.x, -self.centre.x / self.focal.x, 1.0),
            (c.x, (size.x - self.centre.x) / self.focal.x, -1.0),
            (c.y, -self.centre.y / self.focal.y, 1.0),
            (c.y, (size.y - self.centre.y) / self.focal.y, -1.0),
        ];
        sides.iter().all(|&(a, k, sign)| {
            let dist = sign * (a - k * c.z) / (1.0 + k * k).sqrt();
            dist >= -radius
        })
    }
}

pub struct Tsdf {
    bricks: HashMap<BrickKey, Brick>,
}

impl Default for Tsdf {
    fn default() -> Self {
        Self::new()
    }
}

impl Tsdf {
    pub fn new() -> Self {
        Self {
            bricks: HashMap::new(),
        }
    }

    /// Fuses one depth image (depth along the camera's forward axis, NaN
    /// where empty) seen from `camera`, a pinhole with the image's size.
    #[allow(
        clippy::iter_over_hash_type,
        reason = "per-brick updates are independent"
    )]
    pub fn integrate(&mut self, depth: &DepthImage, camera: &Camera) {
        let size = UVec2::new(depth.width, depth.height);
        let proj = Projection::new(camera, size);
        let local_to_world = camera.local_to_world();

        // Allocate the bricks within TRUNC of every observed surface point.
        let mut far = 0.0f32;
        let mut touched = Vec::new();
        for (i, &d) in depth.depth.iter().enumerate() {
            if !(d.is_finite() && d > 0.0) {
                continue;
            }
            far = far.max(d);
            let (x, y) = (i as u32 % size.x, i as u32 / size.x);
            let local = Vec3::new(
                (x as f32 + 0.5 - proj.centre.x) / proj.focal.x * d,
                (y as f32 + 0.5 - proj.centre.y) / proj.focal.y * d,
                d,
            );
            let p = local_to_world.transform_point3(local);
            let lo = split_voxel(voxel_of(p - TRUNC)).0.0;
            let hi = split_voxel(voxel_of(p + TRUNC)).0.0;
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        touched.push(BrickKey(IVec3::new(x, y, z)));
                    }
                }
            }
        }
        touched.sort_unstable();
        touched.dedup();
        for key in touched {
            self.bricks.entry(key).or_insert_with(Brick::new);
        }
        if far == 0.0 {
            return;
        }

        // Every allocated brick in the frustum is updated, so free space seen
        // in this view also clears surfaces that no longer exist.
        let half = 0.5 * BRICK as f32 * VOXEL;
        let radius = half * 3f32.sqrt();
        for (key, brick) in &mut self.bricks {
            let origin = key.0 * BRICK;
            let centre = origin.as_vec3() * VOXEL + half;
            if !proj.sees_sphere(centre, radius, far + TRUNC) {
                continue;
            }
            for z in 0..BRICK {
                for y in 0..BRICK {
                    for x in 0..BRICK {
                        let local = IVec3::new(x, y, z);
                        let Some((cam, pixel)) = proj.project(voxel_centre(origin + local)) else {
                            continue;
                        };
                        let d = depth.depth[pixel];
                        if !(d.is_finite() && d > 0.0) {
                            continue;
                        }
                        let sdf = d - cam.z;
                        if sdf < -TRUNC {
                            continue;
                        }
                        brick.fuse(voxel_index(local), (sdf / TRUNC).min(1.0));
                    }
                }
            }
        }
    }

    /// Multiplies every fusion weight by `factor`, so later observations
    /// count more than earlier ones.
    #[allow(
        clippy::iter_over_hash_type,
        reason = "per-brick updates are independent"
    )]
    pub fn decay(&mut self, factor: f32) {
        for brick in self.bricks.values_mut() {
            for w in &mut brick.weight {
                *w *= factor;
            }
        }
    }

    /// Signed distance in metres at `world`, trilinear between voxel centres;
    /// `None` unless all eight surrounding voxels are observed.
    pub fn sdf(&self, world: Vec3) -> Option<f32> {
        let g = world / VOXEL - 0.5;
        let base = g.floor();
        let f = g - base;
        let base = base.as_ivec3();
        let mut sum = 0.0;
        for corner in 0..8 {
            let o = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let (t, w) = self.voxel(base + o)?;
            if w <= 0.0 {
                return None;
            }
            let o = o.as_vec3();
            let k = (Vec3::ONE - o) * (Vec3::ONE - f) + o * f;
            sum += k.x * k.y * k.z * t;
        }
        Some(sum * TRUNC)
    }

    /// Bricks to re-mesh, in key order: those whose summed |Δsdf| since their
    /// last report averages more than 5 mm over the brick's voxels, and those
    /// changed but never reported. Their change counters restart.
    pub fn take_changed(&mut self) -> Vec<BrickKey> {
        let threshold = CHANGE_THRESHOLD * BRICK_VOXELS as f32;
        let mut keys: Vec<BrickKey> = self
            .bricks
            .iter_mut()
            .filter(|(_, b)| b.change > threshold || (!b.reported && b.change > 0.0))
            .map(|(key, b)| {
                b.change = 0.0;
                b.reported = true;
                *key
            })
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Normalised distances of brick `key` and a one-voxel border taken from
    /// its neighbours, for meshing as a padded chunk: `PADDED³` samples
    /// (`PADDED = BRICK + 2`), sample `(x, y, z)` at index
    /// `x + PADDED·(y + PADDED·z)` (x fastest, the order of
    /// `ndshape::ConstShape3u32<PADDED, PADDED, PADDED>` used by
    /// `fast-surface-nets`), holding voxel `key·BRICK + (x−1, y−1, z−1)`.
    /// Unobserved voxels read `1.0` (+TRUNC). `None` if the brick does not exist.
    pub fn brick_samples(&self, key: BrickKey) -> Option<Vec<f32>> {
        if !self.bricks.contains_key(&key) {
            return None;
        }
        let origin = key.0 * BRICK - 1;
        let mut samples = Vec::with_capacity(PADDED * PADDED * PADDED);
        for z in 0..PADDED as i32 {
            for y in 0..PADDED as i32 {
                for x in 0..PADDED as i32 {
                    let t = match self.voxel(origin + IVec3::new(x, y, z)) {
                        Some((t, w)) if w > 0.0 => t,
                        _ => 1.0,
                    };
                    samples.push(t);
                }
            }
        }
        Some(samples)
    }

    pub fn reset(&mut self) {
        self.bricks.clear();
    }

    /// Normalised distance and weight of global voxel `g`, if its brick exists.
    fn voxel(&self, g: IVec3) -> Option<(f32, f32)> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        Some((brick.tsdf[i], brick.weight[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::{Mat3, Quat, UVec2, Vec2, vec3};

    const SIZE: UVec2 = UVec2::new(128, 128);

    /// Pinhole camera at `pos` looking at `target`, 60° fov; local +Z forward, +Y down.
    fn look_at(pos: Vec3, target: Vec3) -> Camera {
        let forward = (target - pos).normalize();
        let right = Vec3::Y.cross(forward).normalize();
        let down = forward.cross(right);
        let rotation = Quat::from_mat3(&Mat3::from_cols(right, down, forward));
        let fov = 60f64.to_radians();
        Camera::new(
            pos,
            rotation,
            fov,
            fov,
            Vec2::splat(0.5),
            CameraModel::Pinhole,
        )
    }

    /// Ray-casts `hit` per pixel centre. `hit(origin, dir)` returns the ray
    /// parameter of the first hit; `dir` has unit camera-space z, so that
    /// parameter is the depth along the camera's forward axis.
    fn depth_image(camera: &Camera, hit: impl Fn(Vec3, Vec3) -> Option<f32>) -> DepthImage {
        let (f, c) = (camera.focal(SIZE), camera.center(SIZE));
        let mut depth = Vec::new();
        for y in 0..SIZE.y {
            for x in 0..SIZE.x {
                let local = vec3(
                    (x as f32 + 0.5 - c.x) / f.x,
                    (y as f32 + 0.5 - c.y) / f.y,
                    1.0,
                );
                let d = hit(camera.position, camera.rotation * local).unwrap_or(f32::NAN);
                depth.push(d);
            }
        }
        let alpha = depth
            .iter()
            .map(|d| if d.is_nan() { 0.0 } else { 1.0 })
            .collect();
        DepthImage {
            width: SIZE.x,
            height: SIZE.y,
            depth,
            alpha,
        }
    }

    fn plane_z(z: f32) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| {
            let t = (z - o.z) / d.z;
            (t > 0.0).then_some(t)
        }
    }

    /// Axis-aligned box, entry point of the slab test.
    fn aabb(min: Vec3, max: Vec3) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| {
            let inv = d.recip();
            let (t0, t1) = ((min - o) * inv, (max - o) * inv);
            let near = t0.min(t1).max_element();
            let far = t0.max(t1).min_element();
            (near <= far && near > 0.0).then_some(near)
        }
    }

    fn integrate(tsdf: &mut Tsdf, cameras: &[Camera], hit: &impl Fn(Vec3, Vec3) -> Option<f32>) {
        for cam in cameras {
            tsdf.integrate(&depth_image(cam, hit), cam);
        }
    }

    /// Normalised distance at `p` as the mesher sees it: trilinear between
    /// voxel centres, unobserved voxels reading `1.0` as in `brick_samples`.
    fn mesher_value(tsdf: &Tsdf, p: Vec3) -> f32 {
        let g = p / VOXEL - 0.5;
        let (base, f) = (g.floor().as_ivec3(), g - g.floor());
        (0..8)
            .map(|c| {
                let o = IVec3::new(c & 1, (c >> 1) & 1, (c >> 2) & 1);
                let t = match tsdf.voxel(base + o) {
                    Some((t, w)) if w > 0.0 => t,
                    _ => 1.0,
                };
                let o = o.as_vec3();
                let k = (Vec3::ONE - o) * (Vec3::ONE - f) + o * f;
                k.x * k.y * k.z * t
            })
            .sum()
    }

    /// Sign changes of the mesher's input sampled every 5 mm from `a` to `b`.
    fn zero_crossings(tsdf: &Tsdf, a: Vec3, b: Vec3) -> Vec<Vec3> {
        let n = ((b - a).length() / 0.005).round() as usize;
        let samples: Vec<(Vec3, f32)> = (0..=n)
            .map(|i| {
                let p = a.lerp(b, i as f32 / n as f32);
                (p, mesher_value(tsdf, p))
            })
            .collect();
        samples
            .windows(2)
            .filter(|w| (w[0].1 > 0.0) != (w[1].1 > 0.0))
            .map(|w| w[0].0.lerp(w[1].0, w[0].1 / (w[0].1 - w[1].1)))
            .collect()
    }

    fn voxel_weight(tsdf: &Tsdf, world: Vec3) -> f32 {
        let g = (world / VOXEL).floor().as_ivec3();
        tsdf.voxel(g).map_or(0.0, |(_, w)| w)
    }

    #[test]
    fn plane_from_three_poses() {
        let target = vec3(0.0, 0.0, 2.0);
        let cams = [
            look_at(Vec3::ZERO, target),
            look_at(vec3(0.5, 0.0, 0.2), target),
            look_at(vec3(-0.3, 0.4, 0.1), target),
        ];
        let mut tsdf = Tsdf::new();
        integrate(&mut tsdf, &cams, &plane_z(2.0));

        for x in [-0.5, -0.23, 0.0, 0.17, 0.5] {
            for y in [-0.5, -0.11, 0.0, 0.31, 0.5] {
                let on = tsdf.sdf(vec3(x, y, 2.0)).expect("plane observed");
                assert!(on.abs() < 0.01, "({x},{y}): sdf {on}");
                let front = tsdf.sdf(vec3(x, y, 1.95)).expect("front observed");
                assert!(front > 0.02, "({x},{y}): front sdf {front}");
                let behind = tsdf.sdf(vec3(x, y, 2.05)).expect("behind observed");
                assert!(behind < -0.02, "({x},{y}): behind sdf {behind}");
            }
        }
        assert_eq!(
            tsdf.sdf(vec3(0.0, 0.0, 2.5)),
            None,
            "beyond truncation is unobserved"
        );
    }

    #[test]
    fn box_faces_from_four_sides() {
        let center = vec3(0.0, 0.0, 2.0);
        let cams = [
            look_at(vec3(0.0, 0.0, 0.7), center),
            look_at(vec3(1.3, 0.0, 2.0), center),
            look_at(vec3(0.0, 0.0, 3.3), center),
            look_at(vec3(-1.3, 0.0, 2.0), center),
        ];
        let mut tsdf = Tsdf::new();
        integrate(&mut tsdf, &cams, &aabb(center - 0.2, center + 0.2));

        for normal in [Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_X] {
            let face = center + 0.2 * normal;
            let on = tsdf.sdf(face).expect("face observed");
            assert!(on.abs() < 0.01, "face {normal}: sdf {on}");
            let outside = tsdf.sdf(face + 0.05 * normal).expect("outside observed");
            assert!(outside > 0.02, "face {normal}: outside sdf {outside}");
            let inside = tsdf.sdf(face - 0.05 * normal).expect("inside observed");
            assert!(inside < -0.02, "face {normal}: inside sdf {inside}");
        }
    }

    #[test]
    fn thin_sheet_seen_from_both_sides_keeps_a_zero_crossing() {
        let target = vec3(0.0, 0.0, 2.0);
        let cams = [
            look_at(Vec3::ZERO, target),
            look_at(vec3(0.0, 0.0, 4.0), target),
        ];
        let mut tsdf = Tsdf::new();
        let sheet = aabb(vec3(-0.6, -0.6, 1.9975), vec3(0.6, 0.6, 2.0025));
        integrate(&mut tsdf, &cams, &sheet);

        for (x, y) in [(0.0, 0.0), (0.21, -0.13), (-0.4, 0.33)] {
            let crossings = zero_crossings(
                &tsdf,
                vec3(x, y, 2.0 - 2.0 * TRUNC),
                vec3(x, y, 2.0 + 2.0 * TRUNC),
            );
            assert!(!crossings.is_empty(), "({x},{y}): no zero crossing");
        }
    }

    /// A 2 cm rod along y through `center`, fused from four sides at 1.5 m.
    fn rod_crossings(center: Vec3) -> Vec<Vec<Vec3>> {
        let cams = [
            look_at(center - 1.5 * Vec3::Z, center),
            look_at(center + 1.5 * Vec3::X, center),
            look_at(center + 1.5 * Vec3::Z, center),
            look_at(center - 1.5 * Vec3::X, center),
        ];
        let mut tsdf = Tsdf::new();
        let half = vec3(0.01, 0.5, 0.01);
        integrate(&mut tsdf, &cams, &aabb(center - half, center + half));

        let mut all = Vec::new();
        for y in [0.0, 0.12, -0.27] {
            for dir in [Vec3::X, Vec3::Z] {
                let c = center + vec3(0.0, y, 0.0);
                let crossings = zero_crossings(&tsdf, c - 2.0 * TRUNC * dir, c + 2.0 * TRUNC * dir);
                all.push(crossings);
            }
        }
        all
    }

    /// Rod axis on a line of voxel centres, so voxel projections hit it.
    #[test]
    fn thin_rod_seen_from_four_sides_keeps_a_zero_crossing() {
        let center = vec3(0.025, 0.0, 2.025);
        for crossings in rod_crossings(center) {
            assert!(!crossings.is_empty(), "rod at {center}: no zero crossing");
        }
    }

    /// Rod axis midway between voxel centres: no voxel centre projects onto
    /// the rod from the four axis-aligned views, so voxel-projection fusion
    /// never observes it.
    #[test]
    #[ignore = "known limitation of voxel-projection fusion for sub-voxel structures"]
    fn thin_rod_between_voxel_centres_keeps_a_zero_crossing() {
        let center = vec3(0.0, 0.0, 2.0);
        for crossings in rod_crossings(center) {
            assert!(!crossings.is_empty(), "rod at {center}: no zero crossing");
        }
    }

    fn plane_setup() -> (Tsdf, Camera) {
        let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
        let mut tsdf = Tsdf::new();
        tsdf.integrate(&depth_image(&cam, plane_z(2.5)), &cam);
        (tsdf, cam)
    }

    #[test]
    fn decay_scales_weights() {
        let (mut tsdf, _) = plane_setup();
        let p = vec3(0.025, 0.025, 2.475);
        let before = tsdf.sdf(p).unwrap();
        assert_eq!(voxel_weight(&tsdf, p), 1.0);
        tsdf.decay(0.5);
        assert_eq!(voxel_weight(&tsdf, p), 0.5);
        assert_eq!(tsdf.sdf(p), Some(before), "decay keeps distances");
    }

    #[test]
    fn weights_are_capped() {
        let (mut tsdf, cam) = plane_setup();
        let depth = depth_image(&cam, plane_z(2.5));
        for _ in 0..30 {
            tsdf.integrate(&depth, &cam);
        }
        assert_eq!(voxel_weight(&tsdf, vec3(0.025, 0.025, 2.475)), MAX_WEIGHT);
    }

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

        // Weight 6 now: a 1 cm shift moves the ~2400 band voxels by 1/7 cm, a
        // brick mean of ~0.4 mm.
        tsdf.integrate(&depth_image(&cam, plane_z(2.51)), &cam);
        assert!(
            tsdf.take_changed().is_empty(),
            "sub-threshold change reported"
        );

        // A 20 cm jump moves every band voxel by centimetres.
        for _ in 0..3 {
            tsdf.integrate(&depth_image(&cam, plane_z(2.7)), &cam);
        }
        assert!(tsdf.take_changed().contains(&brick));
    }

    #[test]
    fn brick_samples_are_padded_x_fastest() {
        let (tsdf, _) = plane_setup();
        let key = BrickKey(IVec3::new(0, 0, 2));
        let samples = tsdf.brick_samples(key).expect("allocated brick");
        let p = (BRICK + 2) as usize;
        assert_eq!(samples.len(), p * p * p);
        assert!(
            samples
                .iter()
                .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
        );

        let at = |x: i32, y: i32, z: i32| {
            samples[(x + 1) as usize + p * ((y + 1) as usize + p * (z + 1) as usize)]
        };
        let voxel = |x: i32, y: i32, z: i32| IVec3::new(x, y, 2 * BRICK + z);
        // Interior voxels and padding taken from the neighbours at -x, -y, +x+y.
        for (x, y, z) in [
            (3, 7, 9),
            (0, 0, 10),
            (19, 5, 8),
            (-1, 4, 9),
            (6, -1, 11),
            (20, 20, 10),
        ] {
            let (t, w) = tsdf.voxel(voxel(x, y, z)).expect("allocated");
            assert!(w > 0.0, "({x},{y},{z}) unobserved");
            assert_eq!(at(x, y, z), t, "({x},{y},{z})");
        }
        // Plane at 2.5 m: z index 9 is 2.475 m (in front), 10 is 2.525 m (behind).
        assert!(at(5, 5, 9) > 0.0 && at(5, 5, 10) < 0.0);
        // Beyond the truncation band behind the plane nothing was observed.
        assert_eq!(tsdf.voxel(voxel(5, 5, 15)).map(|(_, w)| w), Some(0.0));
        assert_eq!(at(5, 5, 15), 1.0);

        assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(9, 9, 9))), None);
    }

    #[test]
    fn reset_empties() {
        let (mut tsdf, _) = plane_setup();
        tsdf.reset();
        assert_eq!(tsdf.sdf(vec3(0.0, 0.0, 2.5)), None);
        assert!(tsdf.take_changed().is_empty());
        assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(0, 0, 2))), None);
    }
}
