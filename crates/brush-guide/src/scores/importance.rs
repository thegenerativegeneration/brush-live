//! Per-Gaussian Fisher importance `I = tr(D Rᵀ F_μμ R D + F_ss) / n` from the score pass's 6×6 Fisher. The offline
//! Fisher pass at finish writes it to `importance.json` next to the trainer's eviction importance, for comparison;
//! the terms and the reasons for them are in the capture-guidance docs, "How it works" → "Splat importance".

use super::pass::PassOutput;
use glam::{Mat3, Quat};

/// Importance of one Gaussian, NaN if no view observed it; `rot` is Brush's `[w, x, y, z]`, `scale` the
/// three linear scales.
pub fn gaussian_importance(fisher: &[f32; 36], weight: f32, rot: &[f32], scale: &[f32]) -> f32 {
    if weight <= 0.0 {
        return f32::NAN;
    }
    let q = Quat::from_xyzw(rot[1], rot[2], rot[3], rot[0]);
    let r = if q.is_finite() && q.length_squared() > 0.0 {
        Mat3::from_quat(q.normalize())
    } else {
        Mat3::IDENTITY
    };
    let f_pos = Mat3::from_cols_array(&std::array::from_fn(|i| fisher[(i % 3) * 6 + i / 3]));
    let pos: f32 = (0..3)
        .map(|k| {
            let axis = r.col(k);
            scale[k] * scale[k] * axis.dot(f_pos * axis)
        })
        .sum();
    let log_scale = fisher[21] + fisher[28] + fisher[35];
    let value = (pos + log_scale) / weight;
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Importance of every Gaussian of a pass; `rots` and `scales` are the
/// splats' flat rotation (4 per splat) and scale (3 per splat) data.
pub fn importances(out: &PassOutput, rots: &[f32], scales: &[f32]) -> Vec<f32> {
    out.fisher
        .iter()
        .zip(&out.weight)
        .enumerate()
        .map(|(i, (f, &w))| {
            gaussian_importance(f, w, &rots[i * 4..i * 4 + 4], &scales[i * 3..i * 3 + 3])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

    /// `Σ_views J Jᵀ` for `views` identical views with Jacobian `j`.
    fn fisher(j: [f32; 6], views: f32) -> [f32; 36] {
        std::array::from_fn(|i| views * j[i / 6] * j[i % 6])
    }

    fn trace(f: &[f32; 36]) -> f32 {
        (0..6).map(|k| f[k * 6 + k]).sum()
    }

    /// Ruling 37: per-observation normalisation. Summed over views, a weak but often-observed Gaussian can look
    /// more important than a strongly observed one; per observation it does not, and the same per-view sensitivity
    /// scores the same regardless of how many views contributed.
    #[test]
    fn importance_is_normalised_per_observation() {
        let scale = [0.05; 3];
        let once = fisher([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 1.0);
        let weak = fisher([0.0, 0.0, 0.0, 0.2, 0.2, 0.2], 100.0);
        assert!(trace(&weak) > trace(&once));
        let i_once = gaussian_importance(&once, 1.0, &IDENTITY, &scale);
        let i_weak = gaussian_importance(&weak, 100.0, &IDENTITY, &scale);
        assert!(i_once > i_weak, "{i_once} vs {i_weak}");
        assert!((i_once - 3.0).abs() < 1e-5);
        assert!((i_weak - 0.12).abs() < 1e-5);

        let j = [0.5, -0.2, 0.1, 0.3, 0.0, 0.4];
        let scale2 = [0.1, 0.05, 0.02];
        let a = gaussian_importance(&fisher(j, 2.0), 2.0, &IDENTITY, &scale2);
        let b = gaussian_importance(&fisher(j, 200.0), 200.0, &IDENTITY, &scale2);
        assert!((a - b).abs() < 1e-5 * a.max(1.0));
    }

    #[test]
    fn position_is_measured_along_the_gaussians_own_axes() {
        // Sensitivity to moving along world x, for a needle of length 1 and
        // width 0.01: along its long axis a shift by its extent is large.
        let f = fisher([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        let scale = [1.0, 0.01, 0.01];
        let along = gaussian_importance(&f, 1.0, &IDENTITY, &scale);
        // Rotated 90° about z: the needle lies along y, x is a thin axis.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let across = gaussian_importance(&f, 1.0, &[h, 0.0, 0.0, h], &scale);
        assert!((along - 1.0).abs() < 1e-5);
        assert!((across - 1e-4).abs() < 1e-6);
    }

    #[test]
    fn unobserved_is_nan_and_degenerate_is_zero() {
        let f = fisher([1.0; 6], 1.0);
        assert!(gaussian_importance(&f, 0.0, &IDENTITY, &[0.1; 3]).is_nan());
        let nan = [f32::NAN; 36];
        assert_eq!(gaussian_importance(&nan, 1.0, &IDENTITY, &[0.1; 3]), 0.0);
        // A zero quaternion falls back to the world axes.
        let z = gaussian_importance(&f, 1.0, &[0.0; 4], &[1.0; 3]);
        assert!((z - 6.0).abs() < 1e-5);
    }
}
