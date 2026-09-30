//! Which bricks' meshes are stale: each brick keeps the padded samples it
//! was last meshed from, and later samples are compared against them.

use half::f16;

use super::samples::{BrickSamples, PADDED_SAMPLES, SELF_REGION, padded_region, region_offset};
use super::{BrickKey, TRUNC, Tsdf};

/// Mean |Δsdf| (metres) over the near-surface voxels of a brick, or of a
/// neighbour's layer facing it, since the brick was meshed, from which
/// `changed` reports it.
const CHANGE_THRESHOLD: f32 = 0.005;

/// Number of a brick's voxels whose sign flipped since it was meshed, from
/// which `changed` reports it: catches small objects appearing next to a
/// large surface, whose change is too small for the mean.
const SIGN_CHANGES: usize = 20;

/// A voxel whose observed status changed counts as a change only within
/// this normalised distance of the surface.
const STATUS_CHANGE_BAND: f32 = 0.5;

/// What the mesher saw of a brick: its padded samples, normalised distance
/// or NaN where unobserved, and which of its 26 neighbours existed (bit
/// `region_index(offset)`).
pub(super) struct MeshedState {
    sdf: Vec<f16>,
    neighbours: u32,
}

/// Per-region change of padded samples between the meshed state and now.
struct RegionChanges {
    sum: [f32; 27],
    count: [usize; 27],
    /// A voxel near the surface became observed or unobserved.
    flipped: [bool; 27],
    /// Voxels of the brick itself whose sign changed.
    sign_changes: usize,
}

impl RegionChanges {
    fn new() -> Self {
        Self {
            sum: [0.0; 27],
            count: [0; 27],
            flipped: [false; 27],
            sign_changes: 0,
        }
    }

    /// Adds padded sample `i`, normalised distance `before` and `now`, NaN
    /// where unobserved.
    fn add(&mut self, i: usize, before: f32, now: f32) {
        let r = padded_region(i);
        if before.is_nan() != now.is_nan() {
            let seen = if before.is_nan() { now } else { before };
            if seen.abs() >= STATUS_CHANGE_BAND {
                return;
            }
            self.flipped[r] = true;
        }
        let (before, now) = (
            if before.is_nan() { 1.0 } else { before },
            if now.is_nan() { 1.0 } else { now },
        );
        if before.abs() < 1.0 || now.abs() < 1.0 {
            self.sum[r] += (now - before).abs();
            self.count[r] += 1;
        }
        if r == SELF_REGION && (before < 0.0) != (now < 0.0) {
            self.sign_changes += 1;
        }
    }

    fn moved(&self, r: usize) -> bool {
        self.count[r] > 0 && self.sum[r] / self.count[r] as f32 * TRUNC >= CHANGE_THRESHOLD
    }

    fn stale(&self) -> bool {
        self.sign_changes >= SIGN_CHANGES || (0..27).any(|r| self.flipped[r] || self.moved(r))
    }
}

/// Normalised distance of padded sample `i` as `MeshedState` stores it.
fn stored(samples: &BrickSamples, i: usize) -> f32 {
    if samples.weight[i] > 0.0 {
        f16::from_f32(samples.sdf[i]).to_f32()
    } else {
        f32::NAN
    }
}

impl Tsdf {
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
        let mut changes = RegionChanges::new();
        for i in 0..PADDED_SAMPLES {
            changes.add(i, meshed.sdf[i].to_f32(), stored(&samples, i));
        }
        changes.stale()
    }
}
