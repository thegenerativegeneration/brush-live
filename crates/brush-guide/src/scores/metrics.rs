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
    /// Best observed pixels per metre (at a REFERENCE_LONG_SIDE image) below which the Gaussian counts as blurry.
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

/// Image long side at which `CoverageParams::min_px_per_m` is defined.
pub const REFERENCE_LONG_SIDE: f32 = 960.0;

/// A view's focal length in pixels rescaled to a `REFERENCE_LONG_SIDE` image,
/// so pixels per metre mean the same working distance at any keyframe size.
pub fn reference_focal(focal: f32, img_size: glam::UVec2) -> f32 {
    focal * REFERENCE_LONG_SIDE / img_size.max_element().max(1) as f32
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

/// Ridge added to a voxel's summed position Fisher before inversion: an
/// absolute floor plus a fraction of the mean eigenvalue, so a rank-deficient
/// block (too few independent views) stays invertible at any gradient scale.
#[derive(Clone, Copy, Debug)]
pub struct FisherRidge {
    pub abs: f32,
    pub rel: f32,
}

/// Rows and columns 0..3 of the 6×6 Fisher (means first, then log-scales):
/// the block of the Gaussian's position, row-major.
pub fn position_block(h: &[f32; 36]) -> [f32; 9] {
    std::array::from_fn(|i| h[(i / 3) * 6 + i % 3])
}

/// Positional standard deviation in metres for a position Fisher `h`
/// (row-major 3×3, information per unit pixel noise):
/// `σ_pix · sqrt(tr((h + λI)⁻¹) / 3)` with `λ = rel · max(tr(h)/3, 1e-12) + abs`.
/// Infinite when `h + λI` is not finite and positive definite.
pub fn position_sigma(h: &[f64; 9], ridge: FisherRidge, sigma_pix: f32) -> f32 {
    let sym = |i: usize, j: usize| 0.5 * (h[i * 3 + j] + h[j * 3 + i]);
    let trace = sym(0, 0) + sym(1, 1) + sym(2, 2);
    let lambda = f64::from(ridge.rel) * (trace / 3.0).max(1e-12) + f64::from(ridge.abs);
    let m = |i: usize, j: usize| sym(i, j) + if i == j { lambda } else { 0.0 };
    let minor = |i: usize, j: usize| m(i, i) * m(j, j) - m(i, j) * m(i, j);
    let det = m(0, 0) * minor(1, 2) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    let positive_definite = m(0, 0) > 0.0 && minor(0, 1) > 0.0 && det > 0.0;
    if !(positive_definite && det.is_finite()) {
        return f32::INFINITY;
    }
    // tr(M⁻¹) is the sum of the principal 2×2 minors over det(M).
    let trace_inv = (minor(1, 2) + minor(0, 2) + minor(0, 1)) / det;
    (f64::from(sigma_pix) * (trace_inv / 3.0).sqrt()) as f32
}

/// Per-Gaussian coverage and position Fisher block.
pub fn gaussian_metrics(out: &PassOutput, p: &CoverageParams) -> (Vec<f32>, Vec<[f32; 9]>) {
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
    let fisher_pos = out.fisher.iter().map(position_block).collect();
    (coverage, fisher_pos)
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

    /// Coverage formula table (spec: the phone's `CoverageScore` port must match): unseen is zero, many views from
    /// one direction stay low, a wide arc with enough views saturates, and few views scale down with a low-res
    /// penalty.
    #[test]
    fn coverage_score_matches_the_spec_formula() {
        assert_eq!(coverage_score(Vec3::ZERO, 0.0, 0.0, &p()), 0.0);

        let (s, w) = sum_dirs(&[Vec3::Z; 20]);
        assert!(coverage_score(s, w, 1000.0, &p()) < 0.05);

        let dirs: Vec<Vec3> = (0..12)
            .map(|i| {
                let a = (i as f32 / 11.0 - 0.5) * std::f32::consts::PI; // -90°..90°
                Vec3::new(a.sin(), 0.0, a.cos())
            })
            .collect();
        let (s, w) = sum_dirs(&dirs);
        assert!((coverage_score(s, w, 1000.0, &p()) - 1.0).abs() < 1e-5);

        let (s, w) = sum_dirs(&[Vec3::X, Vec3::NEG_X]);
        let full_res = coverage_score(s, w, 1000.0, &p());
        assert!((full_res - 2.0 / 8.0).abs() < 1e-5);
        assert!((coverage_score(s, w, 100.0, &p()) - full_res * 0.5).abs() < 1e-5);
    }

    const RIDGE: FisherRidge = FisherRidge {
        abs: 1e-6,
        rel: 1e-3,
    };

    fn diag(v: f64) -> [f64; 9] {
        [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]
    }

    #[test]
    fn position_block_takes_the_means_rows_and_columns() {
        let h: [f32; 36] = std::array::from_fn(|i| i as f32);
        assert_eq!(
            position_block(&h),
            [0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0]
        );
    }

    /// σ = sqrt(tr Σ / 3): an isotropic information gives its inverse square root exactly, and the trace of the
    /// inverse averages the per-axis variances regardless of rotation.
    #[test]
    fn position_sigma_is_the_square_root_of_the_mean_inverse_variance() {
        let exact = FisherRidge { abs: 0.0, rel: 0.0 };
        let s = position_sigma(&diag(400.0), exact, 0.05);
        assert!((s - 0.05 / 20.0).abs() < 1e-9, "{s}");

        let h = [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 16.0];
        let want = ((1.0 + 0.25 + 0.0625) / 3.0f64).sqrt() as f32;
        assert!((position_sigma(&h, exact, 1.0) - want).abs() < 1e-6);
        // A rotated copy of the same information has the same trace.
        let r = [0.6, -0.8, 0.0, 0.8, 0.6, 0.0, 0.0, 0.0, 1.0];
        let rh: [f64; 9] = std::array::from_fn(|k| {
            let (i, j) = (k / 3, k % 3);
            (0..3)
                .map(|a| r[i * 3 + a] * h[a * 3 + a] * r[j * 3 + a])
                .sum()
        });
        assert!((position_sigma(&rh, exact, 1.0) - want).abs() < 1e-6);
    }

    /// Degenerate information is bounded by the ridge (zero information) or infinite (non-finite or indefinite).
    #[test]
    fn degenerate_information_is_bounded_or_infinite() {
        let s = position_sigma(&[0.0; 9], RIDGE, 0.05);
        assert!((s - 0.05 / 1e-6f32.sqrt()).abs() < 1e-2, "{s}");

        let mut nan = diag(1.0);
        nan[4] = f64::NAN;
        assert_eq!(position_sigma(&nan, RIDGE, 0.05), f32::INFINITY);
        assert_eq!(
            position_sigma(&diag(f64::INFINITY), RIDGE, 0.05),
            f32::INFINITY
        );
        assert_eq!(position_sigma(&diag(-5.0), RIDGE, 0.05), f32::INFINITY);
    }

    #[test]
    fn reference_focal_is_independent_of_image_size() {
        let at_960 = reference_focal(1334.0, glam::UVec2::new(960, 720));
        let at_500 = reference_focal(1334.0 * 500.0 / 960.0, glam::UVec2::new(500, 375));
        assert!((at_960 - 1334.0).abs() < 1e-3);
        assert!((at_500 - at_960).abs() < 1e-2, "{at_500} vs {at_960}");
        // Portrait images use their long side too.
        let portrait = reference_focal(1334.0 * 500.0 / 960.0, glam::UVec2::new(375, 500));
        assert!((portrait - at_960).abs() < 1e-2);
    }
}
