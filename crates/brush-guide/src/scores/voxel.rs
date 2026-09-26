use crate::protocol::Cell;
use glam::{IVec3, Vec3};
use std::collections::HashMap;

pub struct GaussianScore {
    pub pos: Vec3,
    pub opacity: f32,
    pub coverage: f32,
    pub uncertainty: f32,
}

pub struct VoxelAggregator {
    voxel_size: f32,
    min_opacity: f32,
    /// Stands in for a non-finite uncertainty: an unscorable Gaussian is
    /// maximally uncertain, not absent.
    uncertainty_cap: f32,
    first_seen: HashMap<IVec3, f64>,
}

#[derive(Default)]
struct Acc {
    w: f32,
    cov: f32,
    unc: f32,
}

impl VoxelAggregator {
    pub fn new(voxel_size: f32, min_opacity: f32, uncertainty_cap: f32) -> Self {
        Self {
            voxel_size,
            min_opacity,
            uncertainty_cap,
            first_seen: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.first_seen.clear();
    }

    pub fn aggregate(&mut self, gaussians: &[GaussianScore], now_s: f64) -> Vec<Cell> {
        let mut acc: HashMap<IVec3, Acc> = HashMap::new();
        for g in gaussians {
            let visible = g.opacity.is_finite() && g.opacity >= self.min_opacity;
            if !visible || !g.pos.is_finite() {
                continue;
            }
            let coverage = if g.coverage.is_finite() { g.coverage } else { 0.0 };
            let uncertainty = if g.uncertainty.is_finite() {
                g.uncertainty
            } else {
                self.uncertainty_cap
            };
            let key = (g.pos / self.voxel_size).floor().as_ivec3();
            let a = acc.entry(key).or_default();
            a.w += g.opacity;
            a.cov += g.opacity * coverage;
            a.unc += g.opacity * uncertainty;
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
                let center = (key.as_vec3() + 0.5) * self.voxel_size;
                Cell {
                    center: center.to_array(),
                    coverage: ((a.cov / a.w).clamp(0.0, 1.0) * 255.0).round() as u8,
                    uncertainty: u8_unc,
                    age: (now_s - first).clamp(0.0, 255.0) as u8,
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
        GaussianScore { pos: Vec3::from(pos), opacity, coverage, uncertainty }
    }

    #[test]
    fn empty_input_gives_no_cells() {
        let mut agg = VoxelAggregator::new(0.1, 0.1, CAP);
        assert!(agg.aggregate(&[], 0.0).is_empty());
    }

    #[test]
    fn groups_by_voxel_and_weights_by_opacity() {
        let mut agg = VoxelAggregator::new(1.0, 0.1, CAP);
        let cells = agg.aggregate(
            &[g([0.2, 0.2, 0.2], 0.9, 1.0, 0.0), g([0.8, 0.1, 0.5], 0.3, 0.0, 0.0), g([5.5, 0.5, 0.5], 0.05, 1.0, 0.0)],
            0.0,
        );
        assert_eq!(cells.len(), 1, "low-opacity voxel dropped");
        assert_eq!(cells[0].center, [0.5, 0.5, 0.5]);
        assert_eq!(cells[0].coverage, (0.75f32 * 255.0).round() as u8);
    }

    #[test]
    fn uncertainty_is_percentile_normalised() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let gs: Vec<_> = (0..100).map(|i| g([i as f32 + 0.5, 0.5, 0.5], 1.0, 0.0, i as f32)).collect();
        let cells = agg.aggregate(&gs, 0.0);
        let by_x = |x: f32| cells.iter().find(|c| c.center[0] == x).unwrap().uncertainty;
        assert_eq!(by_x(0.5), 0);
        assert_eq!(by_x(99.5), 255);
        assert!((100..160).contains(&by_x(50.5)));
    }

    #[test]
    fn identical_uncertainty_gives_zero() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 3.0), g([1.5, 0.5, 0.5], 1.0, 0.0, 3.0)], 0.0);
        assert!(cells.iter().all(|c| c.uncertainty == 0));
    }

    #[test]
    fn age_counts_from_first_appearance_and_saturates() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], 10.0);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0), g([2.5, 0.5, 0.5], 1.0, 0.0, 0.0)], 13.4);
        let age = |x: f32| cells.iter().find(|c| c.center[0] == x).unwrap().age;
        assert_eq!(age(0.5), 3);
        assert_eq!(age(2.5), 0);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 0.0, 0.0)], 1000.0);
        assert_eq!(cells[0].age, 255);
    }

    #[test]
    fn infinite_uncertainty_keeps_coverage() {
        let mut agg = VoxelAggregator::new(1.0, 0.0, CAP);
        let cells = agg.aggregate(&[g([0.5; 3], 1.0, 1.0, f32::INFINITY)], 0.0);
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
            0.0,
        );
        assert_eq!(cells.len(), 2);
        let first = cells.iter().find(|c| c.center[0] == 0.5).unwrap();
        assert_eq!(first.coverage, 128);
        assert_eq!(first.uncertainty, 255, "(CAP + 1) / 2 is the higher cell mean");
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
            0.0,
        );
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].coverage, 255);
    }
}
