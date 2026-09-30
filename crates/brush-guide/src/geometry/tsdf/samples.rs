//! A brick's voxels padded with a one-voxel border from its neighbours, the
//! chunk the mesher works on.

use glam::IVec3;

use super::{BRICK, BrickKey, Tsdf, voxel_index};

/// Side of the padded sample grid of `Tsdf::brick_samples`.
pub const PADDED: usize = BRICK as usize + 2;

pub(super) const PADDED_SAMPLES: usize = PADDED * PADDED * PADDED;

/// Index of the brick itself among the 27 bricks around it,
/// `(o.x + 1) + 3·(o.y + 1) + 9·(o.z + 1)` for offset `o`.
pub(super) const SELF_REGION: usize = 13;

pub(super) fn region_index(o: IVec3) -> usize {
    ((o.x + 1) + 3 * (o.y + 1) + 9 * (o.z + 1)) as usize
}

pub(super) fn region_offset(r: usize) -> IVec3 {
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

/// Region (`region_index`) of the brick or neighbour padded sample `i`
/// comes from.
pub(super) fn padded_region(i: usize) -> usize {
    let axis = |c: usize| match c {
        0 => -1,
        c if c == PADDED - 1 => 1,
        _ => 0,
    };
    let (x, y, z) = (i % PADDED, (i / PADDED) % PADDED, i / (PADDED * PADDED));
    region_index(IVec3::new(axis(x), axis(y), axis(z)))
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

impl Tsdf {
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
}
