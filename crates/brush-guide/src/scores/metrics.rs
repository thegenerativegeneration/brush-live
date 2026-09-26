use super::pass::PassOutput;
use glam::Vec3;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CoverageParams {
    /// Views needed for full credit.
    pub n_target: f32,
    /// Angular spread (1 − mean resultant length) that counts as full. A uniform
    /// hemisphere gives 0.5; a ±45° arc about 0.1.
    pub spread_target: f32,
    /// Best observed pixels per metre below which the Gaussian counts as blurry.
    pub min_px_per_m: f32,
    pub low_res_penalty: f32,
}

impl Default for CoverageParams {
    fn default() -> Self {
        Self {
            n_target: 8.0,
            spread_target: 0.3,
            min_px_per_m: 300.0,
            low_res_penalty: 0.5,
        }
    }
}

pub fn coverage_score(dir_sum: Vec3, weight: f32, max_px_per_m: f32, p: &CoverageParams) -> f32 {
    if weight <= 0.0 {
        return 0.0;
    }
    let spread = 1.0 - (dir_sum.length() / weight).min(1.0);
    let mut c = (weight / p.n_target).min(1.0) * (spread / p.spread_target).min(1.0);
    if max_px_per_m < p.min_px_per_m {
        c *= p.low_res_penalty;
    }
    c
}

/// Ridge added to the Fisher before its log-determinant: an absolute floor plus
/// a fraction of the mean eigenvalue, so rank-deficient `H` (fewer than six
/// independent views) stays positive definite at any gradient scale.
#[derive(Clone, Copy, Debug)]
pub struct FisherRidge {
    pub abs: f32,
    pub rel: f32,
}

/// Uncertainty of `H = 0`, the largest value `uncertainty_score` returns.
pub fn uncertainty_cap(ridge: FisherRidge) -> f32 {
    (-6.0 * f64::from(ridge.abs.max(f32::MIN_POSITIVE)).ln()) as f32
}

fn log_det_6x6_f64(m: &[f64; 36]) -> Option<f64> {
    let mut l = [0.0f64; 36];
    for j in 0..6 {
        let diag = m[j * 6 + j] - (0..j).map(|k| l[j * 6 + k] * l[j * 6 + k]).sum::<f64>();
        if !(diag > 0.0 && diag.is_finite()) {
            return None;
        }
        l[j * 6 + j] = diag.sqrt();
        for i in (j + 1)..6 {
            let sum: f64 = (0..j).map(|k| l[i * 6 + k] * l[j * 6 + k]).sum();
            l[i * 6 + j] = (m[i * 6 + j] - sum) / l[j * 6 + j];
        }
    }
    Some(2.0 * (0..6).map(|i| l[i * 6 + i].ln()).sum::<f64>())
}

/// `−log det(H + λI)` with `λ = rel · max(tr(H)/6, 1e-12) + abs`, in f64.
/// Always finite: if the factorisation still fails the cap is returned.
pub fn uncertainty_score(h: &[f32; 36], ridge: FisherRidge) -> f32 {
    let mut m: [f64; 36] = std::array::from_fn(|i| f64::from(h[i]));
    let mean_eig = (0..6).map(|i| m[i * 6 + i]).sum::<f64>() / 6.0;
    let lambda = f64::from(ridge.rel) * mean_eig.max(1e-12) + f64::from(ridge.abs);
    for i in 0..6 {
        m[i * 6 + i] += lambda;
    }
    match log_det_6x6_f64(&m) {
        Some(ld) if ld.is_finite() => (-ld) as f32,
        _ => uncertainty_cap(ridge),
    }
}

pub fn gaussian_metrics(
    out: &PassOutput,
    p: &CoverageParams,
    ridge: FisherRidge,
) -> (Vec<f32>, Vec<f32>) {
    let coverage = (0..out.weight.len())
        .map(|i| {
            coverage_score(
                Vec3::from(out.dir_sum[i]),
                out.weight[i],
                out.max_px_per_m[i],
                p,
            )
        })
        .collect();
    let uncertainty = out
        .fisher
        .iter()
        .map(|h| uncertainty_score(h, ridge))
        .collect();
    (coverage, uncertainty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> CoverageParams {
        CoverageParams::default()
    }

    fn sum_dirs(dirs: &[Vec3]) -> (Vec3, f32) {
        (dirs.iter().map(|d| d.normalize()).sum(), dirs.len() as f32)
    }

    #[test]
    fn unseen_is_zero() {
        assert_eq!(coverage_score(Vec3::ZERO, 0.0, 0.0, &p()), 0.0);
    }

    #[test]
    fn many_views_one_direction_is_low() {
        let (s, w) = sum_dirs(&[Vec3::Z; 20]);
        assert!(coverage_score(s, w, 1000.0, &p()) < 0.05);
    }

    #[test]
    fn wide_arc_with_enough_views_is_full() {
        let dirs: Vec<Vec3> = (0..12)
            .map(|i| {
                let a = (i as f32 / 11.0 - 0.5) * std::f32::consts::PI; // -90°..90°
                Vec3::new(a.sin(), 0.0, a.cos())
            })
            .collect();
        let (s, w) = sum_dirs(&dirs);
        assert!((coverage_score(s, w, 1000.0, &p()) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn few_views_scale_down_and_low_res_penalised() {
        let (s, w) = sum_dirs(&[Vec3::X, Vec3::NEG_X]);
        let full_res = coverage_score(s, w, 1000.0, &p());
        assert!((full_res - 2.0 / 8.0).abs() < 1e-5);
        assert!((coverage_score(s, w, 100.0, &p()) - full_res * 0.5).abs() < 1e-5);
    }

    const RIDGE: FisherRidge = FisherRidge {
        abs: 1e-6,
        rel: 1e-3,
    };

    #[test]
    fn uncertainty_orders_by_information() {
        let mut weak = [0f32; 36];
        let mut strong = [0f32; 36];
        for i in 0..6 {
            weak[i * 6 + i] = 1e-3;
            strong[i * 6 + i] = 10.0;
        }
        let u_none = uncertainty_score(&[0.0; 36], RIDGE);
        let u_weak = uncertainty_score(&weak, RIDGE);
        let u_strong = uncertainty_score(&strong, RIDGE);
        assert!(u_none > u_weak && u_weak > u_strong);
        assert!(u_none.is_finite());
    }

    /// Fisher from `k` random per-view gradients at the magnitude a real pass produces.
    fn fisher_from_views(k: usize, seed: u64) -> [f32; 36] {
        use rand::{RngExt as _, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut h = [0f32; 36];
        for _ in 0..k {
            let g: [f32; 6] = std::array::from_fn(|_| (rng.random::<f32>() * 2.0 - 1.0) * 100.0);
            for i in 0..6 {
                for j in 0..6 {
                    h[i * 6 + j] += g[i] * g[j];
                }
            }
        }
        h
    }

    #[test]
    fn rank_deficient_fisher_is_finite_and_ordered() {
        let u = |h: &[f32; 36]| uncertainty_score(h, RIDGE);
        let (r1, r3, full) = (
            u(&fisher_from_views(1, 1)),
            u(&fisher_from_views(3, 2)),
            u(&fisher_from_views(12, 3)),
        );
        assert!(
            r1.is_finite() && r3.is_finite() && full.is_finite(),
            "{r1} {r3} {full}"
        );
        assert!(r1 > r3 && r3 > full, "{r1} {r3} {full}");
    }

    #[test]
    fn zero_and_non_finite_fisher_give_the_cap() {
        let cap = uncertainty_cap(RIDGE);
        assert!((uncertainty_score(&[0.0; 36], RIDGE) - cap).abs() < 1e-3);
        let mut nan = [0f32; 36];
        nan[0] = f32::NAN;
        assert_eq!(uncertainty_score(&nan, RIDGE), cap);
        assert_eq!(uncertainty_score(&[f32::INFINITY; 36], RIDGE), cap);
    }
}
