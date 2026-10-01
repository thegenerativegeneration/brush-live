//! The single-pass aggregation the score round used before the voxel round
//! and the Fisher pass were split (brush 295c49a5), kept as the reference
//! the split must reproduce.

use super::*;
use rand::{RngExt as _, SeedableRng};

struct LegacyAcc {
    w: f32,
    cov: f32,
    info: [f64; 9],
    pos: Vec3,
    tensor: Mat3,
    axis_w: f32,
}

struct Legacy {
    voxel_size: f32,
    min_opacity: f32,
    scale: UncertaintyScale,
    first_seen: HashMap<IVec3, f64>,
    unc_ema: HashMap<IVec3, f32>,
}

impl Legacy {
    fn new(voxel_size: f32, min_opacity: f32, scale: UncertaintyScale) -> Self {
        Self {
            voxel_size,
            min_opacity,
            scale,
            first_seen: HashMap::new(),
            unc_ema: HashMap::new(),
        }
    }

    fn aggregate(&mut self, gaussians: &[GaussianScore], cameras: &[ViewCone], now_s: f64) -> Vec<Cell> {
        let mut acc: HashMap<IVec3, LegacyAcc> = HashMap::new();
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
            let a = acc.entry(key).or_insert(LegacyAcc {
                w: 0.0,
                cov: 0.0,
                info: [0.0; 9],
                pos: Vec3::ZERO,
                tensor: Mat3::ZERO,
                axis_w: 0.0,
            });
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

        let sc = self.scale;
        let voxels: Vec<(IVec3, LegacyAcc, f32)> = acc
            .into_iter()
            .map(|(key, a)| {
                let sigma = voxel_sigma(&a.info, &sc);
                (key, a, sigma)
            })
            .collect();
        let sigmas: Vec<f32> = voxels.iter().map(|v| v.2).collect();
        let range = log_sigma_range(&sigmas);
        voxels
            .into_iter()
            .map(|(key, a, sigma)| {
                let first = *self.first_seen.entry(key).or_insert(now_s);
                let u8_unc = uncertainty_byte(sigma, range);
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

/// A seeded scene: flat patches with jittered axes, round blobs, faint and
/// broken Gaussians, a spread of information so the percentiles matter.
fn fixture(seed: u64, n: usize) -> Vec<GaussianScore> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let pos = Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-0.5..0.5),
                if i % 3 == 0 { rng.random_range(-1.0..1.0) } else { 0.3 + rng.random_range(-0.01..0.01) },
            );
            let opacity = match i % 17 {
                0 => 0.02,
                1 => f32::NAN,
                _ => rng.random_range(0.05..1.0),
            };
            let s = 10f32.powf(rng.random_range(-3.0..-1.0));
            let mut fisher_pos = info(s);
            fisher_pos[1] = rng.random_range(-0.1..0.1) / (s * s);
            fisher_pos[3] = fisher_pos[1];
            if i % 23 == 0 {
                fisher_pos = [0.0; 9];
            }
            let axis = if i % 3 == 0 {
                Vec3::new(rng.random(), rng.random(), rng.random())
            } else {
                Vec3::new(rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1), 1.0)
            };
            GaussianScore {
                pos: if i % 41 == 0 { Vec3::NAN } else { pos },
                opacity,
                coverage: if i % 29 == 0 { f32::NAN } else { rng.random() },
                fisher_pos,
                axis,
                flatness: rng.random(),
            }
        })
        .collect()
}

fn cameras() -> Vec<ViewCone> {
    vec![
        cam([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
        cam([2.0, 0.0, 2.0], [-1.0, 0.0, -1.0]),
        cam([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]),
    ]
}

fn sorted(mut cells: Vec<Cell>) -> Vec<Cell> {
    cells.sort_by(|a, b| a.center.partial_cmp(&b.center).expect("finite centres"));
    cells
}

const PROD_SCALE: UncertaintyScale = UncertaintyScale {
    ridge: FisherRidge { abs: 1e-6, rel: 1e-3 },
    sigma_pix: 0.05,
};

#[test]
fn split_path_reproduces_the_legacy_score_round() {
    let cams = cameras();
    let mut legacy = Legacy::new(0.1, 0.1, PROD_SCALE);
    let mut split = VoxelAggregator::new(0.1, 0.1, PROD_SCALE);
    // Several rounds, so the per-voxel EMA and first-seen ages are compared too.
    for (round, seed) in [11u64, 12, 13, 14].into_iter().enumerate() {
        let gs = fixture(seed, 4000);
        let now = round as f64 * 2.5;
        let want = sorted(legacy.aggregate(&gs, &cams, now));
        // The worker's order: the Fisher pass, then the voxel round from the
        // splat parameters alone.
        split.update_fisher(&gs);
        let geoms: Vec<SplatGeom> = gs.iter().map(GaussianScore::geom).collect();
        let got = sorted(split.cells(&geoms, &cams, now));
        assert!(want.len() > 300, "fixture spans many voxels: {}", want.len());
        assert_eq!(got, want, "round {round}");
    }
}

#[test]
fn cells_between_fisher_passes_keep_the_last_bytes_and_mark_new_voxels_uninformed() {
    let cams = cameras();
    let mut agg = VoxelAggregator::new(0.1, 0.1, PROD_SCALE);
    let gs = fixture(5, 3000);
    let with_fisher = sorted(agg.aggregate(&gs, &cams, 0.0));
    // The splats move on: shift everything by one voxel along x, so most
    // voxels are the same keys and the last column is new.
    let moved: Vec<SplatGeom> = gs
        .iter()
        .map(|g| SplatGeom {
            pos: g.pos + Vec3::new(0.1, 0.0, 0.0),
            ..g.geom()
        })
        .collect();
    let later = agg.cells(&moved, &cams, 3.0);
    let by_key = |cells: &[Cell]| -> HashMap<IVec3, Cell> {
        cells
            .iter()
            .map(|c| ((Vec3::from(c.center) / 0.1).floor().as_ivec3(), *c))
            .collect()
    };
    let before = by_key(&with_fisher);
    let mut new_keys = 0;
    for (key, c) in by_key(&later) {
        match before.get(&key) {
            Some(b) => {
                assert_eq!((c.coverage, c.uncertainty), (b.coverage, b.uncertainty), "{key}");
            }
            None => {
                new_keys += 1;
                assert_eq!(
                    (c.coverage, c.uncertainty),
                    (UNINFORMED_COVERAGE, UNINFORMED_UNCERTAINTY),
                    "{key}"
                );
            }
        }
    }
    assert!(new_keys > 0);
}

#[test]
fn voxels_before_any_fisher_pass_are_uninformed() {
    let mut agg = VoxelAggregator::new(1.0, 0.1, SCALE);
    let cells = agg.cells(&[g([0.5; 3], 1.0, 1.0, 0.1).geom()], &[], 0.0);
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].coverage, 0);
    assert_eq!(cells[0].uncertainty, 255);
}
