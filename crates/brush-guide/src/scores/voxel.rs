use crate::protocol::Cell;
use crate::scores::metrics::{FisherRidge, position_sigma};
use glam::{IVec3, Mat3, Vec3};
use std::collections::HashMap;

/// What the voxel round needs of one Gaussian; read from the splat
/// parameters, no render.
#[derive(Clone, Copy, Debug)]
pub struct SplatGeom {
    pub pos: Vec3,
    pub opacity: f32,
    /// Unit direction of the Gaussian's shortest scale axis in world space (sign arbitrary).
    pub axis: Vec3,
    /// `1 − s_min / s_mid` of the sorted scales: 0 for a round Gaussian, whose
    /// `axis` carries no orientation, towards 1 for a flat disc. Scales the
    /// Gaussian's vote on the voxel normal.
    pub flatness: f32,
}

/// One Gaussian with the Fisher pass's scores.
#[derive(Clone, Copy, Debug)]
pub struct GaussianScore {
    pub pos: Vec3,
    pub opacity: f32,
    pub coverage: f32,
    /// Position block of the Gaussian's Fisher, row-major 3×3.
    pub fisher_pos: [f32; 9],
    pub axis: Vec3,
    pub flatness: f32,
}

impl GaussianScore {
    pub fn geom(&self) -> SplatGeom {
        SplatGeom {
            pos: self.pos,
            opacity: self.opacity,
            axis: self.axis,
            flatness: self.flatness,
        }
    }
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

/// Coverage byte of a voxel no Fisher pass has scored yet ("unseen").
pub const UNINFORMED_COVERAGE: u8 = 0;
/// Uncertainty byte of a voxel no Fisher pass has scored yet (one with
/// information about it); such cells also carry `Cell::uninformed`.
pub const UNINFORMED_UNCERTAINTY: u8 = 255;

/// Builds score-set cells in two parts: [`Self::update_fisher`] takes a
/// Fisher pass and keeps each voxel's coverage and smoothed uncertainty
/// byte; [`Self::cells`] builds the cells from the splat parameters alone
/// and attaches the latest Fisher bytes per voxel.
pub struct VoxelAggregator {
    voxel_size: f32,
    min_opacity: f32,
    scale: UncertaintyScale,
    first_seen: HashMap<IVec3, f64>,
    /// Per voxel, as of the latest Fisher pass that scored it.
    fisher: HashMap<IVec3, FisherBytes>,
    record_raw: bool,
    raw: Vec<RawVoxel>,
}

#[derive(Clone, Copy, Debug)]
struct FisherBytes {
    coverage: u8,
    /// EMA (α = 0.3 on the new pass) of the uncertainty byte.
    unc_ema: f32,
}

/// How a voxel's summed position Fisher becomes its positional σ.
#[derive(Clone, Copy, Debug)]
pub struct UncertaintyScale {
    pub ridge: FisherRidge,
    /// Pixel noise on [0, 1] RGB that turns information into metres.
    pub sigma_pix: f32,
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

/// A voxel's Fisher-pass sums.
struct FisherAcc {
    w: f32,
    cov: f32,
    /// Σ opacity · position Fisher over Gaussians with a finite block.
    info: [f64; 9],
}

/// A voxel's geometry sums.
struct GeomAcc {
    w: f32,
    pos: Vec3,
    /// Σ opacity · flatness · a aᵀ over Gaussians with a usable axis.
    tensor: Mat3,
    /// Σ opacity · flatness over the same Gaussians.
    axis_w: f32,
}

impl GeomAcc {
    // glam's Mat3::default() is the identity, so the accumulator is built explicitly.
    fn new() -> Self {
        Self {
            w: 0.0,
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

/// The round's 5th and 95th percentile of `ln σ` over finite σ.
fn log_sigma_range(sigmas: &[f32]) -> (f32, f32) {
    let mut logs: Vec<f32> = sigmas
        .iter()
        .filter(|s| s.is_finite() && **s > 0.0)
        .map(|s| s.ln())
        .collect();
    logs.sort_by(f32::total_cmp);
    let pct = |p: f32| {
        logs.get(((logs.len() as f32 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    (pct(0.05), pct(0.95))
}

/// `clamp((ln σ − lo) / (hi − lo), 0, 1) · 255`, rounded; 255 for a
/// non-finite σ. For an empty range, 255 above it and 0 at or below.
fn uncertainty_byte(sigma: f32, (lo, hi): (f32, f32)) -> u8 {
    if !sigma.is_finite() {
        255
    } else if hi > lo {
        (((sigma.ln() - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8
    } else if sigma.ln() > hi {
        255
    } else {
        0
    }
}

/// Positional σ of a voxel's summed position Fisher; infinite when it holds
/// no information, so the ridge alone never yields a finite σ.
fn voxel_sigma(info: &[f64; 9], scale: &UncertaintyScale) -> f32 {
    let trace = info[0] + info[4] + info[8];
    if trace > 0.0 {
        position_sigma(info, scale.ridge, scale.sigma_pix)
    } else {
        f32::INFINITY
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
            fisher: HashMap::new(),
            record_raw: false,
            raw: Vec::new(),
        }
    }

    /// Whether `update_fisher` keeps each voxel's raw scores for `raw_round`.
    pub fn record_raw(&mut self, on: bool) {
        self.record_raw = on;
        if !on {
            self.raw = Vec::new();
        }
    }

    /// Per-voxel raw scores of the latest `update_fisher` call; empty unless
    /// recording is on.
    pub fn raw_round(&self) -> &[RawVoxel] {
        &self.raw
    }

    pub fn reset(&mut self) {
        self.first_seen.clear();
        self.fisher.clear();
    }

    /// The voxel a Gaussian counts towards, if it is opaque enough: the
    /// occupancy rule both parts share.
    fn key(&self, pos: Vec3, opacity: f32) -> Option<IVec3> {
        let visible = opacity.is_finite() && opacity >= self.min_opacity;
        (visible && pos.is_finite()).then(|| (pos / self.voxel_size).floor().as_ivec3())
    }

    /// Takes one Fisher pass: per voxel, the opacity-weighted mean coverage
    /// and the positional σ of the summed position Fisher, mapped to a byte
    /// between this pass's 5th and 95th percentile of `ln σ` and smoothed
    /// per voxel across passes. Voxels this pass does not hold, or holds
    /// without usable information (no view of the pass observed them, or
    /// only broken Fisher blocks, so σ is not finite), keep their previous
    /// bytes unchanged. Returns the number of voxels scored.
    pub fn update_fisher(&mut self, gaussians: &[GaussianScore]) -> usize {
        let mut acc: HashMap<IVec3, FisherAcc> = HashMap::new();
        for g in gaussians {
            let Some(key) = self.key(g.pos, g.opacity) else {
                continue;
            };
            let coverage = if g.coverage.is_finite() {
                g.coverage
            } else {
                0.0
            };
            let a = acc.entry(key).or_insert(FisherAcc {
                w: 0.0,
                cov: 0.0,
                info: [0.0; 9],
            });
            a.w += g.opacity;
            a.cov += g.opacity * coverage;
            if g.fisher_pos.iter().all(|v| v.is_finite()) {
                for (s, h) in a.info.iter_mut().zip(g.fisher_pos) {
                    *s += f64::from(g.opacity) * f64::from(h);
                }
            }
        }

        let sc = self.scale;
        let voxels: Vec<(IVec3, FisherAcc, f32)> = acc
            .into_iter()
            .map(|(key, a)| {
                let sigma = voxel_sigma(&a.info, &sc);
                (key, a, sigma)
            })
            .collect();
        let sigmas: Vec<f32> = voxels.iter().map(|v| v.2).collect();
        let range = log_sigma_range(&sigmas);
        self.raw.clear();
        let mut scored = 0;
        for (key, a, sigma) in voxels {
            if self.record_raw {
                self.raw.push(RawVoxel {
                    key,
                    coverage: a.cov / a.w,
                    sigma,
                });
            }
            // No usable information, typically because no view of this pass
            // observed the voxel. That says nothing about it, so it keeps
            // what earlier passes gave it, or stays uninformed.
            if !sigma.is_finite() {
                continue;
            }
            scored += 1;
            let u8_unc = f32::from(uncertainty_byte(sigma, range));
            let prev = self.fisher.get(&key).map_or(u8_unc, |f| f.unc_ema);
            self.fisher.insert(
                key,
                FisherBytes {
                    coverage: ((a.cov / a.w).clamp(0.0, 1.0) * 255.0).round() as u8,
                    unc_ema: 0.3 * u8_unc + 0.7 * prev,
                },
            );
        }
        scored
    }

    /// The score set's cells from the splat parameters: centre, density and
    /// normal from this call's Gaussians, age from each voxel's first
    /// appearance here, coverage and uncertainty from the latest Fisher pass
    /// that scored the voxel ([`UNINFORMED_COVERAGE`] and
    /// [`UNINFORMED_UNCERTAINTY`] if none has).
    pub fn cells(
        &mut self,
        gaussians: &[SplatGeom],
        cameras: &[ViewCone],
        now_s: f64,
    ) -> Vec<Cell> {
        let mut acc: HashMap<IVec3, GeomAcc> = HashMap::new();
        for g in gaussians {
            let Some(key) = self.key(g.pos, g.opacity) else {
                continue;
            };
            let a = acc.entry(key).or_insert_with(GeomAcc::new);
            a.w += g.opacity;
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
        acc.into_iter()
            .map(|(key, a)| {
                let first = *self.first_seen.entry(key).or_insert(now_s);
                let fisher = self.fisher.get(&key);
                let (coverage, uncertainty) = fisher
                    .map_or((UNINFORMED_COVERAGE, UNINFORMED_UNCERTAINTY), |f| {
                        (f.coverage, f.unc_ema.round() as u8)
                    });
                let center = a.pos / a.w;
                let normal = (a.w >= MIN_NORMAL_WEIGHT && a.axis_w >= MIN_FLAT_WEIGHT)
                    .then(|| dominant_axis(a.tensor))
                    .flatten()
                    .filter(|(_, planarity)| *planarity >= MIN_PLANARITY)
                    .map(|(n, _)| orient(n, center, cameras).to_array());
                Cell {
                    center: center.to_array(),
                    coverage,
                    uncertainty,
                    age: (now_s - first).clamp(0.0, 255.0) as u8,
                    normal,
                    density: (a.w * 32.0).round().min(255.0) as u8,
                    uninformed: fisher.is_none(),
                }
            })
            .collect()
    }

    /// A Fisher pass and the cells of the same Gaussians, as one call.
    pub fn aggregate(
        &mut self,
        gaussians: &[GaussianScore],
        cameras: &[ViewCone],
        now_s: f64,
    ) -> Vec<Cell> {
        self.update_fisher(gaussians);
        let geoms: Vec<SplatGeom> = gaussians.iter().map(GaussianScore::geom).collect();
        self.cells(&geoms, cameras, now_s)
    }
}

#[cfg(test)]
mod tests;
