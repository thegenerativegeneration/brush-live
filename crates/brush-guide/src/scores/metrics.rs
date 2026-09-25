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

pub fn uncertainty_score(h: &[f32; 36], lambda: f32) -> f32 {
    let mut m = *h;
    for i in 0..6 {
        m[i * 6 + i] += lambda;
    }
    -brush_train::lod::log_det_6x6(&m)
}

pub fn gaussian_metrics(out: &PassOutput, p: &CoverageParams, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let coverage = (0..out.weight.len())
        .map(|i| coverage_score(Vec3::from(out.dir_sum[i]), out.weight[i], out.max_px_per_m[i], p))
        .collect();
    let uncertainty = out.fisher.iter().map(|h| uncertainty_score(h, lambda)).collect();
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

    #[test]
    fn uncertainty_orders_by_information() {
        let mut weak = [0f32; 36];
        let mut strong = [0f32; 36];
        for i in 0..6 {
            weak[i * 6 + i] = 1e-3;
            strong[i * 6 + i] = 10.0;
        }
        let u_none = uncertainty_score(&[0.0; 36], 1e-6);
        let u_weak = uncertainty_score(&weak, 1e-6);
        let u_strong = uncertainty_score(&strong, 1e-6);
        assert!(u_none > u_weak && u_weak > u_strong);
        assert!(u_none.is_finite());
    }
}
