//! Sparse truncated signed distance volume fused from depth images.
//!
//! The update is the voxel-projection fusion of Zeng et al.
//! (`andyzeng/tsdf-fusion-python`, BSD-2): every voxel centre is projected
//! into the depth image, `sdf = depth − z`, voxels more than `TRUNC` behind
//! the surface are skipped, the normalised distance `min(1, sdf / TRUNC)`
//! enters a weighted running average. Behind the surface the observation
//! weight falls off linearly (voxblox `TsdfIntegratorBase::updateTsdfVoxel`,
//! BSD-3), which keeps thin structures seen from both sides from averaging
//! their zero crossing away. Weights are capped at `MAX_WEIGHT` so
//! the volume keeps following a changing splat. Storage is sparse in the way
//! of `Open3D`'s `ScalableTSDFVolume`: bricks of `BRICK³` voxels are allocated
//! only around observed surface points.
//!
//! Distances are stored normalised to `[-1, 1]` (units of `TRUNC`), positive
//! in front of the surface; unobserved voxels read as `+1`.

use std::collections::HashMap;

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use glam::{IVec3, UVec2, Vec3};
use half::f16;

use super::depth::DepthImage;

pub const VOXEL: f32 = 0.05;
pub const TRUNC: f32 = 0.15;
pub const BRICK: i32 = 20;
pub const MAX_WEIGHT: f32 = 20.0;
/// A voxel becomes observed once its fusion weight reaches this.
pub const MIN_WEIGHT: f32 = 0.1;
/// An observed voxel stays observed until its weight falls below this.
pub const UNOBSERVED_WEIGHT: f32 = 0.05;
/// `decay` never lowers a weight below this.
pub const DECAY_FLOOR: f32 = 2.0;
/// Depth beyond this (metres) is not integrated, like voxblox's
/// `max_ray_length_m`.
pub const MAX_DEPTH: f32 = 8.0;

/// Mean |Δsdf| (metres) over the near-surface voxels of a brick, or of a
/// neighbour's layer facing it, since the brick was meshed, from which
/// `changed` reports it.
const CHANGE_THRESHOLD: f32 = 0.005;

/// Number of a brick's voxels whose sign flipped since it was meshed, from
/// which `changed` reports it: catches small objects appearing next to a
/// large surface, whose change is too small for the mean.
const SIGN_CHANGES: usize = 20;

const BRICK_VOXELS: usize = (BRICK * BRICK * BRICK) as usize;

/// Side of the padded sample grid of `Tsdf::brick_samples`.
pub const PADDED: usize = BRICK as usize + 2;

const PADDED_SAMPLES: usize = PADDED * PADDED * PADDED;

/// Index of the brick itself among the 27 bricks around it,
/// `(o.x + 1) + 3·(o.y + 1) + 9·(o.z + 1)` for offset `o`.
const SELF_REGION: usize = 13;

fn region_index(o: IVec3) -> usize {
    ((o.x + 1) + 3 * (o.y + 1) + 9 * (o.z + 1)) as usize
}

fn region_offset(r: usize) -> IVec3 {
    let r = r as i32;
    IVec3::new(r % 3, (r / 3) % 3, r / 9) - 1
}

/// Padded sample coordinates covered by the neighbour at offset `o` along
/// one axis, and the neighbour-local voxel coordinate of the first.
fn padded_span(o: i32) -> (std::ops::Range<usize>, i32) {
    match o {
        -1 => (0..1, BRICK - 1),
        0 => (1..PADDED - 1, 0),
        _ => (PADDED - 1..PADDED, 0),
    }
}

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
    /// Fusion weight per voxel.
    weight: Vec<f32>,
    /// Observed status per voxel, with hysteresis (`observed_after`).
    observed: Vec<bool>,
    /// State the brick's mesh was made from, set by `Tsdf::mark_meshed`.
    meshed: Option<MeshedState>,
}

/// What the mesher saw of a brick: its padded samples, normalised distance
/// or NaN where unobserved, and which of its 26 neighbours existed (bit
/// `region_index(offset)`).
struct MeshedState {
    sdf: Vec<f16>,
    neighbours: u32,
}

impl Brick {
    fn new() -> Self {
        Self {
            tsdf: vec![1.0; BRICK_VOXELS],
            weight: vec![0.0; BRICK_VOXELS],
            observed: vec![false; BRICK_VOXELS],
            meshed: None,
        }
    }

    /// Weighted running average of one observation into voxel `i`
    /// (Zeng et al.: `(w·t + w_obs·d) / (w + w_obs)`), weight capped.
    fn fuse(&mut self, i: usize, dist: f32, w_obs: f32) {
        if w_obs <= 0.0 {
            return;
        }
        let (t_old, w_old) = (self.tsdf[i], self.weight[i]);
        let w_new = w_old + w_obs;
        let t_new = (w_old * t_old + w_obs * dist) / w_new;
        self.tsdf[i] = t_new;
        self.set_weight(i, w_new.min(MAX_WEIGHT));
    }

    fn set_weight(&mut self, i: usize, w: f32) {
        self.weight[i] = w;
        self.observed[i] = observed_after(self.observed[i], w);
    }
}

/// Region (`region_index`) of the brick or neighbour padded sample `i`
/// comes from.
fn padded_region(i: usize) -> usize {
    let axis = |c: usize| match c {
        0 => -1,
        c if c == PADDED - 1 => 1,
        _ => 0,
    };
    let (x, y, z) = (i % PADDED, (i / PADDED) % PADDED, i / (PADDED * PADDED));
    region_index(IVec3::new(axis(x), axis(y), axis(z)))
}

/// Observation weight for signed distance `sdf` (metres): 1 down to one
/// voxel behind the surface, then linearly to 0 at `-TRUNC` (voxblox weight
/// drop-off with `dropoff_epsilon` = voxel size).
fn observation_weight(sdf: f32) -> f32 {
    if sdf < -VOXEL {
        ((TRUNC + sdf) / (TRUNC - VOXEL)).max(0.0)
    } else {
        1.0
    }
}

/// Observed status of a voxel whose weight became `w`: it turns observed
/// at `MIN_WEIGHT` and unobserved below `UNOBSERVED_WEIGHT`, so weights
/// hovering around one threshold do not toggle it.
fn observed_after(observed: bool, w: f32) -> bool {
    w >= if observed {
        UNOBSERVED_WEIGHT
    } else {
        MIN_WEIGHT
    }
}

/// A voxel whose observed status changed counts as a change only within
/// this normalised distance of the surface.
const STATUS_CHANGE_BAND: f32 = 0.5;

fn usable_depth(d: f32) -> bool {
    d.is_finite() && d > 0.0 && d <= MAX_DEPTH
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

/// Padded samples of one brick, see `Tsdf::brick_samples`.
#[derive(Clone, Debug, PartialEq)]
pub struct BrickSamples {
    /// Normalised distance (units of `TRUNC`, positive in front of the
    /// surface); unobserved samples read `1.0`.
    pub sdf: Vec<f32>,
    /// Fusion weight; 0 where unobserved.
    pub weight: Vec<f32>,
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
    /// Depth beyond `MAX_DEPTH` is ignored.
    #[allow(
        clippy::iter_over_hash_type,
        reason = "per-brick updates are independent"
    )]
    pub fn integrate(&mut self, depth: &DepthImage, camera: &Camera) {
        debug_assert!(
            matches!(camera.camera_model, CameraModel::Pinhole),
            "integrate projects with a pinhole camera"
        );
        let size = UVec2::new(depth.width, depth.height);
        let proj = Projection::new(camera, size);
        let local_to_world = camera.local_to_world();

        // Allocate the bricks within TRUNC of every observed surface point.
        let mut far = 0.0f32;
        let mut touched = Vec::new();
        for (i, &d) in depth.depth.iter().enumerate() {
            if !usable_depth(d) {
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
                        if !usable_depth(d) {
                            continue;
                        }
                        let sdf = d - cam.z;
                        if sdf < -TRUNC {
                            continue;
                        }
                        brick.fuse(
                            voxel_index(local),
                            (sdf / TRUNC).min(1.0),
                            observation_weight(sdf),
                        );
                    }
                }
            }
        }
    }

    /// Multiplies every fusion weight above `DECAY_FLOOR` by `factor`, not
    /// going below the floor, so later observations count more than earlier
    /// ones while a surface seen only a few times is kept.
    #[allow(
        clippy::iter_over_hash_type,
        reason = "per-brick updates are independent"
    )]
    pub fn decay(&mut self, factor: f32) {
        for brick in self.bricks.values_mut() {
            for i in 0..BRICK_VOXELS {
                let w = brick.weight[i];
                if w > DECAY_FLOOR {
                    brick.set_weight(i, (w * factor).max(DECAY_FLOOR));
                }
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
            let t = self.observed_voxel(base + o)?;
            let o = o.as_vec3();
            let k = (Vec3::ONE - o) * (Vec3::ONE - f) + o * f;
            sum += k.x * k.y * k.z * t;
        }
        Some(sum * TRUNC)
    }

    /// Bricks whose mesh is stale, in key order: never meshed but holding
    /// near-surface voxels (observed, |sdf| < `TRUNC`), or, since
    /// `mark_meshed`,
    /// - their near-surface voxels (now or then) moved by a mean |Δsdf| of
    ///   5 mm or more, unobserved voxels counting as distance `TRUNC`,
    /// - at least `SIGN_CHANGES` of their voxels changed sign,
    /// - a neighbour was allocated,
    /// - the layer of a neighbour they are padded with moved by a mean
    ///   |Δsdf| of 5 mm or more, or
    /// - a voxel of theirs or of that layer became observed or unobserved.
    ///
    /// A voxel that became observed or unobserved counts (in all of the
    /// above) only if its observed distance is within
    /// `STATUS_CHANGE_BAND` of the surface.
    pub fn changed(&self) -> Vec<BrickKey> {
        let mut keys: Vec<BrickKey> = self
            .bricks
            .keys()
            .copied()
            .filter(|&key| self.is_stale(key))
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Records the state bricks `keys` were meshed from; `changed` measures
    /// their later changes against it.
    pub fn mark_meshed(&mut self, keys: &[BrickKey]) {
        for &key in keys {
            let Some(samples) = self.brick_samples(key) else {
                continue;
            };
            let sdf = samples
                .sdf
                .iter()
                .zip(&samples.weight)
                .map(|(&t, &w)| f16::from_f32(if w > 0.0 { t } else { f32::NAN }))
                .collect();
            let neighbours = self.neighbour_mask(key);
            if let Some(brick) = self.bricks.get_mut(&key) {
                brick.meshed = Some(MeshedState { sdf, neighbours });
            }
        }
    }

    /// `changed`, then `mark_meshed` of the result.
    pub fn take_changed(&mut self) -> Vec<BrickKey> {
        let keys = self.changed();
        self.mark_meshed(&keys);
        keys
    }

    fn neighbour_mask(&self, key: BrickKey) -> u32 {
        (0..27)
            .filter(|&r| r != SELF_REGION)
            .filter(|&r| {
                self.bricks
                    .contains_key(&BrickKey(key.0 + region_offset(r)))
            })
            .fold(0, |mask, r| mask | 1 << r)
    }

    fn is_stale(&self, key: BrickKey) -> bool {
        let Some(samples) = self.brick_samples(key) else {
            return false;
        };
        let Some(meshed) = self.bricks.get(&key).and_then(|b| b.meshed.as_ref()) else {
            return (0..PADDED_SAMPLES)
                .filter(|&i| padded_region(i) == SELF_REGION)
                .any(|i| samples.weight[i] > 0.0 && samples.sdf[i].abs() < 1.0);
        };
        if self.neighbour_mask(key) & !meshed.neighbours != 0 {
            return true;
        }

        let mut sum = [0.0f32; 27];
        let mut count = [0usize; 27];
        let mut flipped = [false; 27];
        let mut sign_changes = 0;
        for i in 0..PADDED_SAMPLES {
            let before = meshed.sdf[i].to_f32();
            let now = if samples.weight[i] > 0.0 {
                f16::from_f32(samples.sdf[i]).to_f32()
            } else {
                f32::NAN
            };
            let r = padded_region(i);
            if before.is_nan() != now.is_nan() {
                let seen = if before.is_nan() { now } else { before };
                if seen.abs() >= STATUS_CHANGE_BAND {
                    continue;
                }
                flipped[r] = true;
            }
            let (before, now) = (
                if before.is_nan() { 1.0 } else { before },
                if now.is_nan() { 1.0 } else { now },
            );
            if before.abs() < 1.0 || now.abs() < 1.0 {
                sum[r] += (now - before).abs();
                count[r] += 1;
            }
            if r == SELF_REGION && (before < 0.0) != (now < 0.0) {
                sign_changes += 1;
            }
        }
        let moved = |r: usize| count[r] > 0 && sum[r] / count[r] as f32 * TRUNC >= CHANGE_THRESHOLD;
        sign_changes >= SIGN_CHANGES || (0..27).any(|r| flipped[r] || moved(r))
    }

    /// Samples of brick `key` and a one-voxel border taken from its
    /// neighbours, for meshing as a padded chunk: `PADDED³` samples
    /// (`PADDED = BRICK + 2`), sample `(x, y, z)` at index
    /// `x + PADDED·(y + PADDED·z)` (x fastest, the order of
    /// `ndshape::ConstShape3u32<PADDED, PADDED, PADDED>` used by
    /// `fast-surface-nets`), holding voxel `key·BRICK + (x−1, y−1, z−1)`.
    /// Unobserved voxels read as weight 0, distance 1.
    /// `None` if the brick does not exist.
    pub fn brick_samples(&self, key: BrickKey) -> Option<BrickSamples> {
        if !self.bricks.contains_key(&key) {
            return None;
        }
        let mut samples = BrickSamples {
            sdf: vec![1.0; PADDED_SAMPLES],
            weight: vec![0.0; PADDED_SAMPLES],
        };
        for r in 0..27 {
            let o = region_offset(r);
            let Some(brick) = self.bricks.get(&BrickKey(key.0 + o)) else {
                continue;
            };
            let ((xs, x0), (ys, y0), (zs, z0)) =
                (padded_span(o.x), padded_span(o.y), padded_span(o.z));
            for (lz, z) in (z0..).zip(zs) {
                for (ly, y) in (y0..).zip(ys.clone()) {
                    for (lx, x) in (x0..).zip(xs.clone()) {
                        let v = voxel_index(IVec3::new(lx, ly, lz));
                        if brick.observed[v] {
                            let i = x + PADDED * (y + PADDED * z);
                            samples.sdf[i] = brick.tsdf[v];
                            samples.weight[i] = brick.weight[v];
                        }
                    }
                }
            }
        }
        Some(samples)
    }

    pub fn reset(&mut self) {
        self.bricks.clear();
    }

    /// Volume over bricks `keys` holding `sdf(voxel centre)` (metres,
    /// `None` for unobserved) with weight 1.
    #[cfg(test)]
    pub(crate) fn from_sdf(keys: &[BrickKey], sdf: impl Fn(Vec3) -> Option<f32>) -> Self {
        let mut tsdf = Self::new();
        tsdf.fill_sdf(keys, sdf);
        tsdf
    }

    /// Allocates bricks `keys` if needed and overwrites their voxels as in
    /// `from_sdf`.
    #[cfg(test)]
    pub(crate) fn fill_sdf(&mut self, keys: &[BrickKey], sdf: impl Fn(Vec3) -> Option<f32>) {
        for &key in keys {
            let brick = self.bricks.entry(key).or_insert_with(Brick::new);
            for z in 0..BRICK {
                for y in 0..BRICK {
                    for x in 0..BRICK {
                        let local = IVec3::new(x, y, z);
                        let i = voxel_index(local);
                        (brick.tsdf[i], brick.weight[i], brick.observed[i]) =
                            match sdf(voxel_centre(key.0 * BRICK + local)) {
                                Some(d) => ((d / TRUNC).clamp(-1.0, 1.0), 1.0, true),
                                None => (1.0, 0.0, false),
                            };
                    }
                }
            }
        }
    }

    /// Sets normalised distance and weight of global voxel `g` in an
    /// existing brick.
    #[cfg(test)]
    fn set_voxel(&mut self, g: IVec3, t: f32, w: f32) {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get_mut(&key).expect("allocated brick");
        let i = voxel_index(local);
        brick.tsdf[i] = t;
        brick.set_weight(i, w);
    }

    /// Normalised distance of global voxel `g`, if it is observed.
    fn observed_voxel(&self, g: IVec3) -> Option<f32> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        brick.observed[i].then_some(brick.tsdf[i])
    }

    /// Normalised distance and weight of global voxel `g`, if its brick exists.
    #[cfg(test)]
    fn voxel(&self, g: IVec3) -> Option<(f32, f32)> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        Some((brick.tsdf[i], brick.weight[i]))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::{Mat3, Quat, UVec2, Vec2, vec3};

    const SIZE: UVec2 = UVec2::new(128, 128);

    /// Pinhole camera at `pos` looking at `target`, 60° fov; local +Z forward, +Y down.
    pub(crate) fn look_at(pos: Vec3, target: Vec3) -> Camera {
        let fov = 60f64.to_radians();
        look_at_with(pos, target, fov, fov, Vec2::splat(0.5))
    }

    fn look_at_with(pos: Vec3, target: Vec3, fov_x: f64, fov_y: f64, center_uv: Vec2) -> Camera {
        let forward = (target - pos).normalize();
        let right = Vec3::Y.cross(forward).normalize();
        let down = forward.cross(right);
        let rotation = Quat::from_mat3(&Mat3::from_cols(right, down, forward));
        Camera::new(pos, rotation, fov_x, fov_y, center_uv, CameraModel::Pinhole)
    }

    pub(crate) fn depth_image(
        camera: &Camera,
        hit: impl Fn(Vec3, Vec3) -> Option<f32>,
    ) -> DepthImage {
        depth_image_sized(camera, SIZE, hit)
    }

    /// Ray-casts `hit` per pixel centre. `hit(origin, dir)` returns the ray
    /// parameter of the first hit; `dir` has unit camera-space z, so that
    /// parameter is the depth along the camera's forward axis.
    fn depth_image_sized(
        camera: &Camera,
        size: UVec2,
        hit: impl Fn(Vec3, Vec3) -> Option<f32>,
    ) -> DepthImage {
        let (f, c) = (camera.focal(size), camera.center(size));
        let mut depth = Vec::new();
        for y in 0..size.y {
            for x in 0..size.x {
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
            width: size.x,
            height: size.y,
            depth,
            alpha,
        }
    }

    pub(crate) fn plane_z(z: f32) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| {
            let t = (z - o.z) / d.z;
            (t > 0.0).then_some(t)
        }
    }

    /// Axis-aligned box, entry point of the slab test.
    pub(crate) fn aabb(min: Vec3, max: Vec3) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| {
            let inv = d.recip();
            let (t0, t1) = ((min - o) * inv, (max - o) * inv);
            let near = t0.min(t1).max_element();
            let far = t0.max(t1).min_element();
            (near <= far && near > 0.0).then_some(near)
        }
    }

    /// Inside of an axis-aligned room: where a ray from inside leaves it.
    fn room(min: Vec3, max: Vec3) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| {
            let inv = d.recip();
            let (t0, t1) = ((min - o) * inv, (max - o) * inv);
            Some(t0.max(t1).min_element()).filter(|t| *t > 0.0)
        }
    }

    /// Nearest hit of two scenes.
    fn union(
        a: impl Fn(Vec3, Vec3) -> Option<f32>,
        b: impl Fn(Vec3, Vec3) -> Option<f32>,
    ) -> impl Fn(Vec3, Vec3) -> Option<f32> {
        move |o, d| match (a(o, d), b(o, d)) {
            (Some(s), Some(t)) => Some(s.min(t)),
            (s, t) => s.or(t),
        }
    }

    pub(crate) fn integrate(
        tsdf: &mut Tsdf,
        cameras: &[Camera],
        hit: &impl Fn(Vec3, Vec3) -> Option<f32>,
    ) {
        for cam in cameras {
            tsdf.integrate(&depth_image(cam, hit), cam);
        }
    }

    /// Sign changes of the sdf sampled every 5 mm from `a` to `b`, counting
    /// only samples whose eight surrounding voxels are all observed (the
    /// cells a mesher keeps when it drops cells touching weight 0).
    fn zero_crossings(tsdf: &Tsdf, a: Vec3, b: Vec3) -> Vec<Vec3> {
        let n = ((b - a).length() / 0.005).round() as usize;
        let samples: Vec<(Vec3, Option<f32>)> = (0..=n)
            .map(|i| {
                let p = a.lerp(b, i as f32 / n as f32);
                (p, tsdf.sdf(p))
            })
            .collect();
        samples
            .windows(2)
            .filter_map(|w| match (w[0], w[1]) {
                ((p0, Some(s0)), (p1, Some(s1))) if (s0 > 0.0) != (s1 > 0.0) => {
                    Some(p0.lerp(p1, s0 / (s0 - s1)))
                }
                _ => None,
            })
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

    /// A 5 mm sheet at z = 2 m, 1.2 m square, fused from both sides.
    pub(crate) fn thin_sheet_tsdf() -> Tsdf {
        let target = vec3(0.0, 0.0, 2.0);
        let cams = [
            look_at(Vec3::ZERO, target),
            look_at(vec3(0.0, 0.0, 4.0), target),
        ];
        let mut tsdf = Tsdf::new();
        let sheet = aabb(vec3(-0.6, -0.6, 1.9975), vec3(0.6, 0.6, 2.0025));
        integrate(&mut tsdf, &cams, &sheet);
        tsdf
    }

    #[test]
    fn thin_sheet_seen_from_both_sides_keeps_a_zero_crossing() {
        let tsdf = thin_sheet_tsdf();

        for (x, y) in [(0.0, 0.0), (0.21, -0.13), (-0.4, 0.33)] {
            let crossings = zero_crossings(
                &tsdf,
                vec3(x, y, 2.0 - 2.0 * TRUNC),
                vec3(x, y, 2.0 + 2.0 * TRUNC),
            );
            assert!(!crossings.is_empty(), "({x},{y}): no zero crossing");
            for c in &crossings {
                assert!(
                    (c.z - 2.0).abs() <= VOXEL,
                    "({x},{y}): crossing at z = {}",
                    c.z
                );
            }
        }
    }

    /// A square rod of side `width` along y through `center`, 1 m long,
    /// inside a 5 m room, fused from four sides at 1.5 m. The room walls give
    /// every pixel a depth, so free space around the rod is observed as it
    /// would be in a real scene.
    pub(crate) fn rod_tsdf(center: Vec3, width: f32) -> Tsdf {
        let cams = [
            look_at(center - 1.5 * Vec3::Z, center),
            look_at(center + 1.5 * Vec3::X, center),
            look_at(center + 1.5 * Vec3::Z, center),
            look_at(center - 1.5 * Vec3::X, center),
        ];
        let mut tsdf = Tsdf::new();
        let half = vec3(0.5 * width, 0.5, 0.5 * width);
        let scene = union(
            aabb(center - half, center + half),
            room(center - 2.5, center + 2.5),
        );
        integrate(&mut tsdf, &cams, &scene);
        tsdf
    }

    /// Zero crossings of the rod of `rod_tsdf` along x and z at three heights.
    fn rod_crossings(center: Vec3, width: f32) -> Vec<(Vec3, Vec<Vec3>)> {
        let tsdf = rod_tsdf(center, width);
        let mut all = Vec::new();
        for y in [0.0, 0.12, -0.27] {
            for dir in [Vec3::X, Vec3::Z] {
                let c = center + vec3(0.0, y, 0.0);
                let crossings = zero_crossings(&tsdf, c - 2.0 * TRUNC * dir, c + 2.0 * TRUNC * dir);
                all.push((dir, crossings));
            }
        }
        all
    }

    /// Rod axes on a line of voxel centres and midway between them.
    pub(crate) const ROD_CENTRES: [Vec3; 2] =
        [Vec3::new(0.025, 0.0, 2.025), Vec3::new(0.0, 0.0, 2.0)];

    #[test]
    fn rod_seen_from_four_sides_keeps_a_zero_crossing() {
        for center in ROD_CENTRES {
            for (dir, crossings) in rod_crossings(center, 0.05) {
                assert!(
                    !crossings.is_empty(),
                    "rod at {center} along {dir}: no zero crossing"
                );
                for c in crossings {
                    let off_axis = ((c - center) * dir).length();
                    assert!(
                        (off_axis - 0.025).abs() <= VOXEL,
                        "rod at {center} along {dir}: crossing {off_axis} m off the axis"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "below the 5 cm voxel resolution"]
    fn thin_rod_seen_from_four_sides_keeps_a_zero_crossing() {
        for center in ROD_CENTRES {
            for (dir, crossings) in rod_crossings(center, 0.02) {
                assert!(
                    !crossings.is_empty(),
                    "rod at {center} along {dir}: no zero crossing"
                );
            }
        }
    }

    fn plane_setup() -> (Tsdf, Camera) {
        let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 2.5));
        let mut tsdf = Tsdf::new();
        tsdf.integrate(&depth_image(&cam, plane_z(2.5)), &cam);
        (tsdf, cam)
    }

    #[test]
    fn decay_scales_weights_down_to_the_floor() {
        let (mut tsdf, cam) = plane_setup();
        let p = vec3(0.025, 0.025, 2.475);
        let depth = depth_image(&cam, plane_z(2.5));
        for _ in 0..9 {
            tsdf.integrate(&depth, &cam);
        }
        let before = tsdf.sdf(p).unwrap();
        assert_eq!(voxel_weight(&tsdf, p), 10.0);
        tsdf.decay(0.5);
        assert_eq!(voxel_weight(&tsdf, p), 5.0);
        assert_eq!(tsdf.sdf(p), Some(before), "decay keeps distances");
        tsdf.decay(0.5);
        assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR + 0.5);
        tsdf.decay(0.5);
        assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR, "clamped to the floor");
        tsdf.decay(0.5);
        assert_eq!(voxel_weight(&tsdf, p), DECAY_FLOOR);

        let (mut single, _) = plane_setup();
        single.decay(0.5);
        assert_eq!(voxel_weight(&single, p), 1.0, "below the floor: untouched");
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

    fn sample_index(x: i32, y: i32, z: i32) -> usize {
        (x + 1) as usize + PADDED * ((y + 1) as usize + PADDED * (z + 1) as usize)
    }

    #[test]
    fn brick_samples_are_padded_x_fastest() {
        let (tsdf, _) = plane_setup();
        let key = BrickKey(IVec3::new(0, 0, 2));
        let samples = tsdf.brick_samples(key).expect("allocated brick");
        let n = PADDED * PADDED * PADDED;
        assert_eq!((samples.sdf.len(), samples.weight.len()), (n, n));
        assert!(
            samples
                .sdf
                .iter()
                .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
        );

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
            let i = sample_index(x, y, z);
            assert_eq!((samples.sdf[i], samples.weight[i]), (t, w), "({x},{y},{z})");
        }
        // Plane at 2.5 m: z index 9 is 2.475 m (in front), 10 is 2.525 m (behind).
        assert!(samples.sdf[sample_index(5, 5, 9)] > 0.0);
        assert!(samples.sdf[sample_index(5, 5, 10)] < 0.0);

        assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(9, 9, 9))), None);
    }

    /// Seen from one side only, everything more than TRUNC behind the plane
    /// stays unobserved: weight 0 and distance +1.
    #[test]
    fn one_sided_plane_leaves_samples_beyond_truncation_unobserved() {
        let (tsdf, _) = plane_setup();
        let samples = tsdf.brick_samples(BrickKey(IVec3::new(0, 0, 2))).unwrap();
        for x in -1..=BRICK {
            for y in -1..=BRICK {
                // z index 13 is 2.675 m, 17.5 cm behind the plane at 2.5 m.
                for z in 13..=BRICK {
                    let i = sample_index(x, y, z);
                    assert_eq!(samples.weight[i], 0.0, "({x},{y},{z})");
                    assert_eq!(samples.sdf[i], 1.0, "({x},{y},{z})");
                }
                // Up to one voxel behind the plane observations count fully.
                assert_eq!(samples.weight[sample_index(x, y, 10)], 1.0, "({x},{y},10)");
            }
        }
    }

    /// A voxel becomes observed at `MIN_WEIGHT` and stays observed down to
    /// `UNOBSERVED_WEIGHT`.
    #[test]
    fn observed_status_has_hysteresis() {
        let (mut tsdf, _) = plane_setup();
        let key = BrickKey(IVec3::new(0, 0, 2));
        let p = vec3(0.025, 0.025, 2.475);
        let g = voxel_of(p);
        let i = sample_index(0, 0, 9);
        let (t, _) = tsdf.voxel(g).unwrap();
        tsdf.take_changed();

        tsdf.set_voxel(g, t, 0.07);
        assert!(
            tsdf.sdf(p).is_some(),
            "observed voxel stays observed at 0.07"
        );
        assert_eq!(tsdf.brick_samples(key).unwrap().weight[i], 0.07);

        tsdf.set_voxel(g, t, 0.04);
        assert_eq!(tsdf.sdf(p), None, "below UNOBSERVED_WEIGHT");
        let samples = tsdf.brick_samples(key).unwrap();
        assert_eq!((samples.sdf[i], samples.weight[i]), (1.0, 0.0));
        assert!(
            tsdf.take_changed().contains(&key),
            "near-surface voxel lost is a change"
        );

        tsdf.set_voxel(g, t, 0.07);
        assert_eq!(
            tsdf.sdf(p),
            None,
            "unobserved voxel stays unobserved at 0.07"
        );
        tsdf.set_voxel(g, t, MIN_WEIGHT);
        assert!(tsdf.sdf(p).is_some(), "observed again at MIN_WEIGHT");
    }

    /// A 160×120 image with the principal point off centre and different
    /// horizontal and vertical fov: a box in front of a wall, seen from one
    /// pose, lands where it is.
    #[test]
    fn non_square_image_with_offset_principal_point() {
        let size = UVec2::new(160, 120);
        let cam = look_at_with(
            Vec3::ZERO,
            Vec3::Z,
            70f64.to_radians(),
            50f64.to_radians(),
            Vec2::new(0.4, 0.6),
        );
        let (lo, hi) = (vec3(0.2, -0.4, 1.5), vec3(0.6, -0.1, 1.9));
        let scene = union(aabb(lo, hi), plane_z(2.5));
        let mut tsdf = Tsdf::new();
        tsdf.integrate(&depth_image_sized(&cam, size, &scene), &cam);

        for (x, y) in [(0.25, -0.35), (0.55, -0.15), (0.4, -0.25)] {
            let on = tsdf.sdf(vec3(x, y, 1.5)).expect("box face observed");
            assert!(on.abs() < 0.01, "box face ({x},{y}): sdf {on}");
        }
        // Beside the box the rays reach the wall: free space at the face depth.
        for (x, y) in [(0.1, -0.25), (0.7, -0.25), (0.4, -0.5), (0.4, 0.0)] {
            let beside = tsdf.sdf(vec3(x, y, 1.5)).expect("free space observed");
            assert!(beside > 0.1, "beside the box ({x},{y}): sdf {beside}");
        }
        for (x, y) in [(-0.3, 0.3), (1.0, 0.5), (-0.6, -0.9), (0.1, -0.25)] {
            let wall = tsdf.sdf(vec3(x, y, 2.5)).expect("wall observed");
            assert!(wall.abs() < 0.01, "wall ({x},{y}): sdf {wall}");
        }
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

    #[test]
    fn reset_empties() {
        let (mut tsdf, _) = plane_setup();
        tsdf.reset();
        assert_eq!(tsdf.sdf(vec3(0.0, 0.0, 2.5)), None);
        assert!(tsdf.take_changed().is_empty());
        assert_eq!(tsdf.brick_samples(BrickKey(IVec3::new(0, 0, 2))), None);
    }

    #[test]
    fn depth_beyond_max_depth_is_ignored() {
        let cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 1.0));
        let mut tsdf = Tsdf::new();
        tsdf.integrate(&depth_image(&cam, plane_z(MAX_DEPTH + 0.5)), &cam);
        assert!(tsdf.bricks.is_empty(), "no bricks beyond MAX_DEPTH");

        let near = MAX_DEPTH - 0.5;
        tsdf.integrate(&depth_image(&cam, plane_z(near)), &cam);
        let sdf = tsdf.sdf(vec3(0.0, 0.0, near)).expect("observed");
        assert!(sdf.abs() < 0.01, "{sdf}");
    }

    #[test]
    #[should_panic(expected = "pinhole")]
    #[cfg(debug_assertions)]
    fn integrate_rejects_non_pinhole_cameras() {
        let mut cam = look_at(Vec3::ZERO, vec3(0.0, 0.0, 1.0));
        let depth = depth_image(&cam, plane_z(2.0));
        cam.camera_model = CameraModel::KannalaBrandt4(Default::default());
        Tsdf::new().integrate(&depth, &cam);
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
}
