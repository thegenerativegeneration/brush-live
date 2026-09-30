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
//! Colour, when given, is fused into the same voxels (`colour`).
//!
//! Distances are stored normalised to `[-1, 1]` (units of `TRUNC`), positive
//! in front of the surface; unobserved voxels read as `+1`.

mod changes;
mod colour;
mod projection;
mod samples;

#[cfg(test)]
mod change_tests;
#[cfg(test)]
mod colour_tests;
#[cfg(test)]
pub(super) mod fixtures;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use glam::{Affine3A, IVec3, UVec2, Vec3};

use super::depth::{ColourImage, DepthImage};
use changes::MeshedState;
use colour::VoxelColours;
use projection::Projection;
pub use samples::{BrickSamples, PADDED};

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

const BRICK_VOXELS: usize = (BRICK * BRICK * BRICK) as usize;

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
    /// Linear RGB per voxel, once the brick had a coloured observation.
    colour: Option<VoxelColours>,
    /// State the brick's mesh was made from, set by `Tsdf::mark_meshed`.
    meshed: Option<MeshedState>,
}

impl Brick {
    fn new() -> Self {
        Self {
            tsdf: vec![1.0; BRICK_VOXELS],
            weight: vec![0.0; BRICK_VOXELS],
            observed: vec![false; BRICK_VOXELS],
            colour: None,
            meshed: None,
        }
    }

    /// Weighted running average of one observation into voxel `i`
    /// (Zeng et al.: `(w·t + w_obs·d) / (w + w_obs)`), weight capped; its
    /// colour `rgb`, if finite, is averaged with the same weights.
    fn fuse(&mut self, i: usize, dist: f32, rgb: Option<[f32; 3]>, w_obs: f32) {
        if w_obs <= 0.0 {
            return;
        }
        let (t_old, w_old) = (self.tsdf[i], self.weight[i]);
        let w_new = w_old + w_obs;
        let t_new = (w_old * t_old + w_obs * dist) / w_new;
        self.tsdf[i] = t_new;
        self.set_weight(i, w_new.min(MAX_WEIGHT));
        if let Some(rgb) = rgb.filter(|c| c.iter().all(|v| v.is_finite())) {
            self.colour
                .get_or_insert_with(VoxelColours::new)
                .fuse(i, rgb, w_old, w_obs);
        }
    }

    fn set_weight(&mut self, i: usize, w: f32) {
        self.weight[i] = w;
        self.observed[i] = observed_after(self.observed[i], w);
    }

    /// Fuses every voxel of this brick (first global voxel `origin`) that
    /// projects into `depth` and lies at most `TRUNC` behind the surface,
    /// with the colour of its pixel if `colour` is given.
    fn fuse_view(
        &mut self,
        origin: IVec3,
        depth: &DepthImage,
        colour: Option<&ColourImage>,
        proj: &Projection,
    ) {
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
                    self.fuse(
                        voxel_index(local),
                        (sdf / TRUNC).min(1.0),
                        colour.map(|c| c.rgb[pixel]),
                        observation_weight(sdf),
                    );
                }
            }
        }
    }
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
    /// where empty) seen from `camera`, a pinhole with the image's size,
    /// and the colour image of the same view if given. Depth beyond
    /// `MAX_DEPTH` is ignored. Panics if the images differ in size.
    pub fn integrate(&mut self, depth: &DepthImage, colour: Option<&ColourImage>, camera: &Camera) {
        debug_assert!(
            matches!(camera.camera_model, CameraModel::Pinhole),
            "integrate projects with a pinhole camera"
        );
        if let Some(c) = colour {
            assert!(
                (c.width, c.height) == (depth.width, depth.height),
                "colour size {}×{} differs from depth size {}×{}",
                c.width,
                c.height,
                depth.width,
                depth.height
            );
        }
        let proj = Projection::new(camera, UVec2::new(depth.width, depth.height));
        let far = self.allocate_around_surface(depth, &proj, &camera.local_to_world());
        if far == 0.0 {
            return;
        }
        self.fuse_in_frustum(depth, colour, &proj, far);
    }

    /// Allocates the bricks within `TRUNC` of every observed surface point;
    /// returns the largest usable depth, 0 if there is none.
    fn allocate_around_surface(
        &mut self,
        depth: &DepthImage,
        proj: &Projection,
        local_to_world: &Affine3A,
    ) -> f32 {
        let mut far = 0.0f32;
        let mut touched = Vec::new();
        for (i, &d) in depth.depth.iter().enumerate() {
            if !usable_depth(d) {
                continue;
            }
            far = far.max(d);
            let p = local_to_world.transform_point3(proj.unproject(i, d));
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
        far
    }

    /// Updates every allocated brick in the frustum up to depth `far`, so
    /// free space seen in this view also clears surfaces that no longer
    /// exist.
    #[allow(
        clippy::iter_over_hash_type,
        reason = "per-brick updates are independent"
    )]
    fn fuse_in_frustum(
        &mut self,
        depth: &DepthImage,
        colour: Option<&ColourImage>,
        proj: &Projection,
        far: f32,
    ) {
        let half = 0.5 * BRICK as f32 * VOXEL;
        let radius = half * 3f32.sqrt();
        for (key, brick) in &mut self.bricks {
            let origin = key.0 * BRICK;
            let centre = origin.as_vec3() * VOXEL + half;
            if proj.sees_sphere(centre, radius, far + TRUNC) {
                brick.fuse_view(origin, depth, colour, proj);
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

    pub fn reset(&mut self) {
        self.bricks.clear();
    }

    /// Normalised distance of global voxel `g`, if it is observed.
    fn observed_voxel(&self, g: IVec3) -> Option<f32> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        brick.observed[i].then_some(brick.tsdf[i])
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

    /// Normalised distance and weight of global voxel `g`, if its brick exists.
    #[cfg(test)]
    fn voxel(&self, g: IVec3) -> Option<(f32, f32)> {
        let (key, local) = split_voxel(g);
        let brick = self.bricks.get(&key)?;
        let i = voxel_index(local);
        Some((brick.tsdf[i], brick.weight[i]))
    }
}
