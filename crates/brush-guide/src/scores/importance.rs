//! Per-Gaussian importance for eviction, from the score pass's 6×6 Fisher.
//!
//! `I = tr(D Rᵀ F_μμ R D + F_ss) / n`
//!
//! - `F_μμ`, `F_ss`: the position and log-scale blocks of the Gaussian's
//!   Fisher summed over the round's views (each view weighted by its
//!   sampling weight).
//! - `R`, `D = diag(s)`: the Gaussian's rotation and scales. `D Rᵀ F_μμ R D`
//!   is the position Fisher in the Gaussian's own axes, in units of its own
//!   extent, so a shift by one standard deviation weighs like a log-scale
//!   change of 1. Without it the position block (1/m²) would dwarf the
//!   unitless log-scale block and favour tiny Gaussians.
//! - `n`: the round's weighted count of views in which the Gaussian has a
//!   non-zero Fisher trace (`PassOutput::weight`).
//!
//! `I` is the mean, over the views that observe the Gaussian, of how much the
//! render changes when the Gaussian moves or resizes by its own extent: a
//! per-observation PUP-style sensitivity. Summed Fisher (PUP's log det of
//! `F`) grows with the number of views, so a well-trained old room would
//! outrank a new room that only a few keyframes have seen, and eviction
//! would take the new room's Gaussians first. Dividing by `n` removes that
//! count: a Gaussian seen once that matters in that view outranks one seen in
//! a hundred views that barely matters in any. The trace replaces PUP's log
//! det because one Rademacher probe per view gives a rank-1 contribution, so a
//! Gaussian seen in fewer than six views has det 0.
//!
//! Gaussians no sampled view observes this round get NaN, which the trainer
//! reads as "keep the previous score" (0 if never scored). A round samples
//! the newest views plus a stratified sample of all, so the score does not
//! decay while a room is out of sight, and a Gaussian missed by one round's
//! sample keeps its last value. The trainer's age and view-cone protection
//! keep new Gaussians out of eviction until they have been scored.

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

    #[test]
    fn one_strong_view_beats_many_weak_views() {
        let scale = [0.05; 3];
        let once = fisher([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 1.0);
        let weak = fisher([0.0, 0.0, 0.0, 0.2, 0.2, 0.2], 100.0);
        // Summed over views the weak, well-observed Gaussian looks more
        // important; per observation it is not.
        assert!(trace(&weak) > trace(&once));
        let i_once = gaussian_importance(&once, 1.0, &IDENTITY, &scale);
        let i_weak = gaussian_importance(&weak, 100.0, &IDENTITY, &scale);
        assert!(i_once > i_weak, "{i_once} vs {i_weak}");
        assert!((i_once - 3.0).abs() < 1e-5);
        assert!((i_weak - 0.12).abs() < 1e-5);
    }

    #[test]
    fn same_per_view_sensitivity_scores_the_same_at_any_view_count() {
        let j = [0.5, -0.2, 0.1, 0.3, 0.0, 0.4];
        let scale = [0.1, 0.05, 0.02];
        let a = gaussian_importance(&fisher(j, 2.0), 2.0, &IDENTITY, &scale);
        let b = gaussian_importance(&fisher(j, 200.0), 200.0, &IDENTITY, &scale);
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
