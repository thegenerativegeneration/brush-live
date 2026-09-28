use crate::protocol::Cell;
use glam::{IVec3, Mat3, Vec3};
use std::collections::HashMap;

pub struct GaussianScore {
    pub pos: Vec3,
    pub opacity: f32,
    pub coverage: f32,
    pub uncertainty: f32,
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
    /// Stands in for a non-finite uncertainty: an unscorable Gaussian is
    /// maximally uncertain, not absent.
    uncertainty_cap: f32,
    first_seen: HashMap<IVec3, f64>,
    /// EMA (α = 0.3 on the new round) of the sent uncertainty byte per voxel,
    /// smoothing the per-round percentile normalisation across rounds.
    unc_ema: HashMap<IVec3, f32>,
}

struct Acc {
    w: f32,
    cov: f32,
    unc: f32,
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
            unc: 0.0,
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
    pub fn new(voxel_size: f32, min_opacity: f32, uncertainty_cap: f32) -> Self {
        Self {
            voxel_size,
            min_opacity,
            uncertainty_cap,
            first_seen: HashMap::new(),
            unc_ema: HashMap::new(),
        }
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
            let uncertainty = if g.uncertainty.is_finite() {
                g.uncertainty
            } else {
                self.uncertainty_cap
            };
            let key = (g.pos / self.voxel_size).floor().as_ivec3();
            let a = acc.entry(key).or_insert_with(Acc::new);
            a.w += g.opacity;
            a.cov += g.opacity * coverage;
            a.unc += g.opacity * uncertainty;
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

        let mut unc: Vec<f32> = acc.values().map(|a| a.unc / a.w).collect();
        unc.sort_by(f32::total_cmp);
        let pct = |p: f32| {
            unc.get(((unc.len() as f32 - 1.0) * p).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let (lo, hi) = (pct(0.05), pct(0.95));

        acc.into_iter()
            .map(|(key, a)| {
                let first = *self.first_seen.entry(key).or_insert(now_s);
                let u = a.unc / a.w;
                let u8_unc = if hi > lo {
                    (((u - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8
                } else {
                    0
                };
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
mod tests {
    use super::*;

    const CAP: f32 = 83.0;

    fn g(pos: [f32; 3], opacity: f32, coverage: f32, uncertainty: f32) -> GaussianScore {
        GaussianScore {
            pos: Vec3::from(pos),
            opacity,
            coverage,
            uncertainty,
            axis: Vec3::Z,
            flatness: 1.0,
        }
    }

    fn ga(pos: [f32; 3], opacity: f32, axis: [f32; 3]) -> GaussianScore {
        GaussianScore {
            pos: Vec3::from(pos),
            opacity,
            coverage: 0.5,
            uncertainty: 1.0,
            axis: Vec3::from(axis),
            flatness: 1.0,
        }
    }

    fn ga_u(pos: [f32; 3], uncertainty: f32) -> GaussianScore {
        GaussianScore {
            pos: Vec3::from(pos),
            opacity: 0.8,
            coverage: 0.5,
            uncertainty,
            axis: Vec3::Z,
            flatness: 1.0,
        }
    }

    fn cam(pos: [f32; 3], fwd: [f32; 3]) -> ViewCone {
        ViewCone {
            position: Vec3::from(pos),
            forward: Vec3::from(fwd).normalize(),
            cos_half_fov: 0.5,
        }
    }

    #[test]
    fn empty_input_gives_no_cells() {
        let mut agg = VoxelAggregator::new(0.1, 0.1, CAP);
        assert!(agg.aggregate(&[], &[], 0.0).is_empty());
    }

    #[test]
    fn groups_by_voxel_and_weights_by_opacity() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let cells = agg.aggregate(
            &[
                g([0.2, 0.2, 0.2], 0.9, 1.0, 0.0),
                g([0.8, 0.1, 0.5], 0.3, 0.0, 0.0),
                g([5.5, 0.5, 0.5], 0.05, 1.0, 0.0),
            ],
            &[],
            0.0,
        );
        assert_eq!(cells.len(), 1, "low-opacity voxel dropped");
        // (0.9·(0.2, 0.2, 0.2) + 0.3·(0.8, 0.1, 0.5)) / 1.2
        let expected = [0.35, 0.175, 0.275];
        for (got, want) in cells[0].center.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        assert_eq!(cells[0].coverage, (0.75f32 * 255.0).round() as u8);
    }

    #[test]
    fn uncertainty_is_percentile_normalised() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let gs: Vec<_> = (0..100)
            .map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, i as f32))
            .collect();
        let cells = agg.aggregate(&gs, &[], 0.0);
        let by_x = |x: f32| cells.iter().find(|c| c.center[0] == x).unwrap().uncertainty;
        assert_eq!(by_x(0.5), 0);
        assert_eq!(by_x(99.5), 255);
        assert!((100..160).contains(&by_x(50.5)));
    }

    #[test]
    fn identical_uncertainty_gives_zero() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(
            &[
                g([0.5; 3], 1.0, 0.0, 3.0),
                g([1.5, 0.5, 0.5], 1.0, 0.0, 3.0),
            ],
            &[],
            0.0,
        );
        assert!(cells.iter().all(|c| c.uncertainty == 0));
    }

    #[test]
    fn uncertainty_is_smoothed_across_rounds_per_voxel() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        // Two voxels so the per-round percentile normalisation has a range.
        let low = |u: f32| vec![ga_u([0.5, 0.5, 0.5], u), ga_u([5.5, 0.5, 0.5], 1.0)];
        let first = agg.aggregate(&low(50.0), &[], 0.0);
        let a0 = first.iter().find(|c| c.center[0] < 1.0).unwrap().uncertainty;
        assert_eq!(a0, 255); // highest in its round
        let second = agg.aggregate(&low(0.5), &[], 1.0);
        let a1 = second.iter().find(|c| c.center[0] < 1.0).unwrap().uncertainty;
        // Round value is 0 (now the lowest); sent value = round(0.3·0 + 0.7·255) = 179.
        assert_eq!(a1, 179);
    }

    #[test]
    fn reset_clears_uncertainty_history() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let set = |u: f32| vec![ga_u([0.5, 0.5, 0.5], u), ga_u([5.5, 0.5, 0.5], 1.0)];
        agg.aggregate(&set(50.0), &[], 0.0);
        agg.reset();
        let c = agg.aggregate(&set(0.5), &[], 1.0);
        assert_eq!(c.iter().find(|c| c.center[0] < 1.0).unwrap().uncertainty, 0);
    }

    #[test]
    fn age_counts_from_first_appearance_and_saturates() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], &[], 10.0);
        let cells = agg.aggregate(
            &[
                g([0.5; 3], 1.0, 0.0, 0.0),
                g([2.5, 0.5, 0.5], 1.0, 0.0, 0.0),
            ],
            &[],
            13.4,
        );
        let age = |x: f32| cells.iter().find(|c| c.center[0] == x).unwrap().age;
        assert_eq!(age(0.5), 3);
        assert_eq!(age(2.5), 0);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], &[], 1000.0);
        assert_eq!(cells[0].age, 255);
    }

    #[test]
    fn infinite_uncertainty_keeps_coverage() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 1.0, f32::INFINITY)], &[], 0.0);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].coverage, 255);
    }

    #[test]
    fn nan_scores_count_as_uncovered_and_maximally_uncertain() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(
            &[
                g([0.5; 3], 1.0, f32::NAN, f32::NAN),
                g([0.6; 3], 1.0, 1.0, 1.0),
                g([1.5, 0.5, 0.5], 1.0, 1.0, 1.0),
            ],
            &[],
            0.0,
        );
        assert_eq!(cells.len(), 2);
        let first = cells.iter().find(|c| c.center[0] < 1.0).unwrap();
        assert!((first.center[0] - 0.55).abs() < 1e-6);
        assert_eq!(first.coverage, 128);
        assert_eq!(
            first.uncertainty, 255,
            "(CAP + 1) / 2 is the higher cell mean"
        );
    }

    #[test]
    fn non_finite_position_or_opacity_is_skipped() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(
            &[
                g([f32::NAN, 0.5, 0.5], 1.0, 1.0, 1.0),
                g([0.5; 3], f32::NAN, 0.0, 0.0),
                g([0.6; 3], 1.0, 1.0, 1.0),
            ],
            &[],
            0.0,
        );
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].coverage, 255);
    }

    #[test]
    fn flat_patch_gets_normal_facing_the_cameras() {
        // Patch at z = 0.5 inside voxel [0,1)^3; Gaussian axes point ±z (unsigned).
        let gs: Vec<_> = (0..20)
            .map(|i| {
                let x = 0.1 + 0.04 * i as f32;
                ga(
                    [x, 0.9 - 0.04 * i as f32, 0.5],
                    0.8,
                    if i % 2 == 0 {
                        [0.0, 0.0, 1.0]
                    } else {
                        [0.0, 0.0, -1.0]
                    },
                )
            })
            .collect();
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let above = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
        assert!(Vec3::from(above[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
        let below = agg.aggregate(&gs, &[cam([0.5, 0.5, -3.0], [0.0, 0.0, 1.0])], 0.0);
        assert!(Vec3::from(below[0].normal.unwrap()).dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn crease_voxel_has_no_normal() {
        let gs: Vec<_> = (0..10)
            .map(|i| {
                ga(
                    [0.5, 0.5, 0.5],
                    0.8,
                    if i < 5 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, 1.0]
                    },
                )
            })
            .collect();
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        assert_eq!(agg.aggregate(&gs, &[], 0.0)[0].normal, None);
    }

    #[test]
    fn light_voxel_has_no_normal() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        assert_eq!(
            agg.aggregate(&[ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])], &[], 0.0)[0].normal,
            None
        );
    }

    #[test]
    fn center_is_opacity_weighted_mean() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let c = agg.aggregate(
            &[
                ga([0.2, 0.5, 0.5], 0.9, [0.0, 0.0, 1.0]),
                ga([0.8, 0.5, 0.5], 0.1, [0.0, 0.0, 1.0]),
            ],
            &[],
            0.0,
        );
        assert!((c[0].center[0] - 0.26).abs() < 1e-5);
    }

    #[test]
    fn density_sums_opacity_and_saturates() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let three: Vec<_> = (0..3)
            .map(|_| ga([0.5, 0.5, 0.5], 0.5, [0.0, 0.0, 1.0]))
            .collect();
        assert_eq!(agg.aggregate(&three, &[], 0.0)[0].density, 48);
        let many: Vec<_> = (0..10)
            .map(|_| ga([0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 1.0]))
            .collect();
        assert_eq!(agg.aggregate(&many, &[], 0.0)[0].density, 255);
    }

    #[test]
    fn unseen_voxel_orients_toward_nearest_camera() {
        let gs: Vec<_> = (0..5)
            .map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0]))
            .collect();
        // Camera looks away (+x) from the voxel, so it does not "see" it; nearest-camera fallback applies.
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let c = agg.aggregate(&gs, &[cam([0.5, 0.5, -4.0], [1.0, 0.0, 0.0])], 0.0);
        assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn round_gaussians_do_not_vote_on_the_normal() {
        // 20 round Gaussians whose arbitrary shortest axes point along x, plus
        // 3 flat ones along z: the normal comes from the flat ones.
        let mut gs: Vec<_> = (0..20)
            .map(|_| GaussianScore {
                flatness: 0.02,
                ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
            })
            .collect();
        gs.extend((0..3).map(|_| ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])));
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
        assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn only_round_gaussians_give_no_normal() {
        let gs: Vec<_> = (0..20)
            .map(|_| GaussianScore {
                flatness: 0.0,
                ..ga([0.5, 0.5, 0.5], 0.8, [0.0, 0.0, 1.0])
            })
            .collect();
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let c = agg.aggregate(&gs, &[], 0.0);
        assert_eq!(c[0].normal, None);
        assert_eq!(c[0].density, 255, "density stays opacity-weighted");
    }

    #[test]
    fn weight_rule_uses_opacity_mass_and_a_smaller_flat_mass() {
        let round = |n: usize| {
            (0..n).map(|_| GaussianScore {
                flatness: 0.0,
                ..ga([0.5, 0.5, 0.5], 0.8, [1.0, 0.0, 0.0])
            })
        };
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        // Opacity mass 1.0, flat mass 0.2: enough for a normal.
        let mut gs: Vec<_> = round(1).collect();
        gs.push(ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0]));
        let c = agg.aggregate(&gs, &[cam([0.5, 0.5, 3.0], [0.0, 0.0, -1.0])], 0.0);
        assert!(Vec3::from(c[0].normal.unwrap()).dot(Vec3::Z) > 0.99);
        // Opacity mass 1.8, flat mass 0.05: too little orientation.
        let mut gs: Vec<_> = round(2).collect();
        gs.push(GaussianScore {
            flatness: 0.25,
            ..ga([0.5, 0.5, 0.5], 0.2, [0.0, 0.0, 1.0])
        });
        assert_eq!(agg.aggregate(&gs, &[], 0.0)[0].normal, None);
    }
}
