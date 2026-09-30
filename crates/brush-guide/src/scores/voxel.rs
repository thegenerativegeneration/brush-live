use crate::protocol::Cell;
use crate::scores::metrics::{FisherRidge, position_sigma};
use glam::{IVec3, Mat3, Vec3};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct GaussianScore {
    pub pos: Vec3,
    pub opacity: f32,
    pub coverage: f32,
    /// Position block of the Gaussian's Fisher, row-major 3×3.
    pub fisher_pos: [f32; 9],
    /// Unit direction of the Gaussian's shortest scale axis in world space (sign arbitrary).
    pub axis: Vec3,
    /// `1 − s_min / s_mid` of the sorted scales: 0 for a round Gaussian, whose
    /// `axis` carries no orientation, towards 1 for a flat disc. Scales the
    /// Gaussian's vote on the voxel normal.
    pub flatness: f32,
}

/// A keyframe camera approximated as a cone for "does this camera see the voxel".
pub struct ViewCone {
    pub position: Vec3,
    pub forward: Vec3,
    pub cos_half_fov: f32,
}

impl ViewCone {
    pub fn sees(&self, p: Vec3) -> bool {
        let d = p - self.position;
        let len = d.length();
        len > 1e-4 && d.dot(self.forward) >= self.cos_half_fov * len
    }
}

/// Minimum share of the axis tensor's trace on its dominant eigenvector for a
/// voxel to count as planar; a two-plane crease scores 0.5.
const MIN_PLANARITY: f32 = 0.7;
/// Minimum Σ opacity in a voxel for a normal.
const MIN_NORMAL_WEIGHT: f32 = 0.3;
/// Minimum Σ opacity · flatness of Gaussians with a usable axis for a normal;
/// a voxel of only round Gaussians has no orientation.
const MIN_FLAT_WEIGHT: f32 = 0.1;

pub struct VoxelAggregator {
    voxel_size: f32,
    min_opacity: f32,
    scale: UncertaintyScale,
    first_seen: HashMap<IVec3, f64>,
    /// EMA (α = 0.3 on the new round) of the sent uncertainty byte per voxel.
    unc_ema: HashMap<IVec3, f32>,
    raw: Vec<RawVoxel>,
}

/// How a voxel's summed position Fisher becomes its uncertainty byte.
#[derive(Clone, Copy, Debug)]
pub struct UncertaintyScale {
    pub ridge: FisherRidge,
    /// Pixel noise on [0, 1] RGB that turns information into metres.
    pub sigma_pix: f32,
    /// Positional σ (metres) sent as byte 0.
    pub sigma_good: f32,
    /// Positional σ (metres) sent as byte 255.
    pub sigma_bad: f32,
}

/// A voxel's scores before quantisation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawVoxel {
    pub key: IVec3,
    /// Opacity-weighted mean coverage.
    pub coverage: f32,
    /// Positional standard deviation in metres; infinite without information.
    pub sigma: f32,
}

struct Acc {
    w: f32,
    cov: f32,
    /// Σ opacity · position Fisher over Gaussians with a finite block.
    info: [f64; 9],
    pos: Vec3,
    /// Σ opacity · flatness · a aᵀ over Gaussians with a usable axis.
    tensor: Mat3,
    /// Σ opacity · flatness over the same Gaussians.
    axis_w: f32,
}

impl Acc {
    // glam's Mat3::default() is the identity, so the accumulator is built explicitly.
    fn new() -> Self {
        Self {
            w: 0.0,
            cov: 0.0,
            info: [0.0; 9],
            pos: Vec3::ZERO,
            tensor: Mat3::ZERO,
            axis_w: 0.0,
        }
    }
}

/// Dominant eigenvector of a symmetric PSD matrix and its share of the trace.
fn dominant_axis(t: Mat3) -> Option<(Vec3, f32)> {
    let trace = t.x_axis.x + t.y_axis.y + t.z_axis.z;
    if !trace.is_finite() || trace <= 0.0 {
        return None;
    }
    let mut best: Option<(Vec3, f32)> = None;
    for start in [t.x_axis, t.y_axis, t.z_axis] {
        if start.length_squared() == 0.0 {
            continue;
        }
        let mut v = start.normalize();
        for _ in 0..32 {
            let nv = t * v;
            let l = nv.length();
            if l == 0.0 {
                break;
            }
            v = nv / l;
        }
        let lambda = v.dot(t * v);
        if best.is_none_or(|(_, b)| lambda > b) {
            best = Some((v, lambda));
        }
    }
    best.map(|(v, lambda)| (v, lambda / trace))
}

/// `clamp((σ − good) / (bad − good), 0, 1) · 255`, rounded; 255 for a
/// non-finite σ, 0 for an empty range.
fn uncertainty_byte(sigma: f32, good: f32, bad: f32) -> u8 {
    if !sigma.is_finite() {
        255
    } else if bad > good {
        (((sigma - good) / (bad - good)).clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        0
    }
}

/// Flips `n` towards the cameras that see `p` (majority vote); with no seeing
/// camera or a tie, towards the nearest camera.
fn orient(n: Vec3, p: Vec3, cameras: &[ViewCone]) -> Vec3 {
    let vote = |c: &ViewCone| if n.dot(c.position - p) >= 0.0 { 1 } else { -1 };
    let mut votes: i32 = cameras.iter().filter(|c| c.sees(p)).map(vote).sum();
    if votes == 0
        && let Some(c) = cameras.iter().min_by(|a, b| {
            a.position
                .distance_squared(p)
                .total_cmp(&b.position.distance_squared(p))
        })
    {
        votes = vote(c);
    }
    if votes < 0 { -n } else { n }
}

impl VoxelAggregator {
    pub fn new(voxel_size: f32, min_opacity: f32, scale: UncertaintyScale) -> Self {
        Self {
            voxel_size,
            min_opacity,
            scale,
            first_seen: HashMap::new(),
            unc_ema: HashMap::new(),
            raw: Vec::new(),
        }
    }

    /// Per-voxel raw scores of the latest `aggregate` call.
    pub fn raw_round(&self) -> &[RawVoxel] {
        &self.raw
    }

    pub fn reset(&mut self) {
        self.first_seen.clear();
        self.unc_ema.clear();
    }

    pub fn aggregate(
        &mut self,
        gaussians: &[GaussianScore],
        cameras: &[ViewCone],
        now_s: f64,
    ) -> Vec<Cell> {
        let mut acc: HashMap<IVec3, Acc> = HashMap::new();
        for g in gaussians {
            let visible = g.opacity.is_finite() && g.opacity >= self.min_opacity;
            if !visible || !g.pos.is_finite() {
                continue;
            }
            let coverage = if g.coverage.is_finite() {
                g.coverage
            } else {
                0.0
            };
            let key = (g.pos / self.voxel_size).floor().as_ivec3();
            let a = acc.entry(key).or_insert_with(Acc::new);
            a.w += g.opacity;
            a.cov += g.opacity * coverage;
            if g.fisher_pos.iter().all(|v| v.is_finite()) {
                for (s, h) in a.info.iter_mut().zip(g.fisher_pos) {
                    *s += f64::from(g.opacity) * f64::from(h);
                }
            }
            a.pos += g.opacity * g.pos;
            let axis_w = if g.flatness.is_finite() {
                g.opacity * g.flatness.clamp(0.0, 1.0)
            } else {
                0.0
            };
            if axis_w > 0.0 && g.axis.is_finite() && g.axis.length_squared() > 0.5 {
                let ax = g.axis.normalize();
                a.tensor += Mat3::from_cols(ax * ax.x, ax * ax.y, ax * ax.z) * axis_w;
                a.axis_w += axis_w;
            }
        }

        self.raw.clear();
        acc.into_iter()
            .map(|(key, a)| {
                let first = *self.first_seen.entry(key).or_insert(now_s);
                let sc = self.scale;
                let sigma = position_sigma(&a.info, sc.ridge, sc.sigma_pix);
                self.raw.push(RawVoxel {
                    key,
                    coverage: a.cov / a.w,
                    sigma,
                });
                let u8_unc = uncertainty_byte(sigma, sc.sigma_good, sc.sigma_bad);
                let prev = *self.unc_ema.get(&key).unwrap_or(&(u8_unc as f32));
                let smoothed = 0.3 * u8_unc as f32 + 0.7 * prev;
                self.unc_ema.insert(key, smoothed);
                let center = a.pos / a.w;
                let normal = (a.w >= MIN_NORMAL_WEIGHT && a.axis_w >= MIN_FLAT_WEIGHT)
                    .then(|| dominant_axis(a.tensor))
                    .flatten()
                    .filter(|(_, planarity)| *planarity >= MIN_PLANARITY)
                    .map(|(n, _)| orient(n, center, cameras).to_array());
                let density = (a.w * 32.0).round().min(255.0) as u8;
                Cell {
                    center: center.to_array(),
                    coverage: ((a.cov / a.w).clamp(0.0, 1.0) * 255.0).round() as u8,
                    uncertainty: smoothed.round() as u8,
                    age: (now_s - first).clamp(0.0, 255.0) as u8,
                    normal,
                    density,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
