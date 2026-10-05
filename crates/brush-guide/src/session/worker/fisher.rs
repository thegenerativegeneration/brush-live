//! The offline Fisher pass: at finish, render and backward over every view
//! at one model state, giving each voxel its coverage and uncertainty (the
//! uncertainty byte ranks voxels within the pass, 5th–95th percentile of
//! `ln σ`) and writing `importance.json` with the pass's per-splat Fisher
//! importance next to the trainer's, for comparing the two. The live
//! eviction importance comes from training alone (`brush_train::evict`).

use super::{Worker, append_ingredient_round, append_raw_round};
use crate::scores::importance::importances;
use crate::scores::metrics::gaussian_metrics;
use crate::scores::pass::{PassOutput, PassView, score_pass};
use crate::session::splat_read::SplatRead;
use web_time::Instant;

/// One pass's outputs, for the caller's log and dumps.
struct PassRun {
    out: PassOutput,
    read: SplatRead,
}

impl Worker {
    /// At finish: one Fisher pass over every view (weight 1), and
    /// `importance.json` with the pass's and the trainer's importance in
    /// splat order. The caller runs the voxel round that publishes it.
    pub(super) async fn finish_pass(&mut self, start: f64) {
        let views: Vec<PassView> = self
            .live
            .views()
            .iter()
            .zip(&self.sizes)
            .map(|(v, &img_size)| PassView {
                camera: v.camera,
                img_size,
                weight: 1.0,
            })
            .collect();
        let num_views = views.len();
        let train = self.live.importance().await;
        let run = self.run_pass(views).await;
        let fisher = importances(&run.out, &run.read.rots, &run.read.scales);
        let path = self.session_dir.join("importance.json");
        let json = serde_json::json!({
            "fisher": finite_or_null(&fisher),
            "train": train.as_deref().map(finite_or_null),
        });
        if let Err(e) = std::fs::write(&path, json.to_string()) {
            log::warn!("importance dump to {}: {e}", path.display());
        }
        log::info!(
            "finish fisher pass: {num_views} views, {:.0} ms",
            (self.clock.elapsed().as_secs_f64() - start) * 1e3
        );
    }

    /// Render and backward over `views`: per-voxel coverage and uncertainty
    /// into the voxel state, and the raw dumps.
    async fn run_pass(&mut self, views: Vec<PassView>) -> PassRun {
        let config = &self.config;
        let splats = self.live.splats().expect("views imply splats").clone();
        let t = Instant::now();
        let out = score_pass(&splats, &views, &config.pass).await;
        let t_pass = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (coverage, fisher_pos) = gaussian_metrics(&out, &config.coverage);
        let t_metrics = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let read = SplatRead::new(&splats).await;
        let t_read = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let gaussians = read.scores(&coverage, &fisher_pos);
        let t_prep = t.elapsed().as_secs_f64();
        let t = Instant::now();
        self.voxels.update_fisher(&gaussians);
        let t_agg = t.elapsed().as_secs_f64();
        if self.config.dump_raw_uncertainty {
            let path = self.session_dir.join("raw_uncertainty.jsonl");
            // The score set that first carries this pass.
            if let Err(e) = append_raw_round(&path, self.version + 1, self.voxels.raw_round()) {
                log::warn!("raw uncertainty dump to {}: {e}", path.display());
            }
            let rows = coverage_ingredients(
                &read.means,
                &read.opac,
                &out.weight,
                &out.dir_sum,
                &out.max_px_per_m,
                &coverage,
                self.config.voxel_size,
                self.config.min_cell_opacity,
            );
            let path = self.session_dir.join("raw_coverage.jsonl");
            if let Err(e) = append_ingredient_round(&path, self.version + 1, &rows) {
                log::warn!("raw coverage dump to {}: {e}", path.display());
            }
        }
        log::debug!(
            target: crate::timing::TARGET,
            "fisher: {} views, pass {:.0} ms, metrics {:.0} ms, splat readback {:.0} ms, \
             gaussian prep {:.0} ms, voxel aggregate {:.0} ms",
            views.len(),
            t_pass * 1e3,
            t_metrics * 1e3,
            t_read * 1e3,
            t_prep * 1e3,
            t_agg * 1e3
        );
        PassRun { out, read }
    }
}

/// Values as JSON numbers, non-finite ones as `null`.
fn finite_or_null(values: &[f32]) -> Vec<Option<f32>> {
    values.iter().map(|&v| v.is_finite().then_some(v)).collect()
}

/// Per-voxel coverage ingredients of one Fisher pass, for the calibration
/// dump: `(key, n, spread, max px/m, mean coverage)` over Gaussians at or
/// above the opacity floor with finite values. `n` and `coverage` are
/// opacity-weighted means; `spread` treats the voxel as one surface
/// (1 − |Σ op·dir| / Σ op·w); `px/m` is the voxel's best.
#[allow(clippy::too_many_arguments)]
fn coverage_ingredients(
    means: &[f32],
    opac: &[f32],
    weight: &[f32],
    dir_sum: &[[f32; 3]],
    ppm: &[f32],
    coverage: &[f32],
    voxel_size: f32,
    min_opacity: f32,
) -> Vec<(glam::IVec3, f32, f32, f32, f32)> {
    use glam::Vec3;
    struct Acc {
        op: f32,
        n: f32,
        dir: Vec3,
        ppm: f32,
        cov: f32,
    }
    let mut acc: std::collections::HashMap<glam::IVec3, Acc> = std::collections::HashMap::new();
    for i in 0..opac.len() {
        let op = opac[i];
        let pos = Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]);
        if !(op.is_finite() && op >= min_opacity && pos.is_finite()) {
            continue;
        }
        let (w, d, p, c) = (weight[i], Vec3::from(dir_sum[i]), ppm[i], coverage[i]);
        if !(w.is_finite() && d.is_finite() && p.is_finite() && c.is_finite()) {
            continue;
        }
        let key = (pos / voxel_size).floor().as_ivec3();
        let a = acc.entry(key).or_insert(Acc {
            op: 0.0,
            n: 0.0,
            dir: Vec3::ZERO,
            ppm: 0.0,
            cov: 0.0,
        });
        a.op += op;
        a.n += op * w;
        a.dir += op * d;
        a.ppm = a.ppm.max(p);
        a.cov += op * c;
    }
    acc.into_iter()
        .map(|(k, a)| {
            let spread = if a.n > 0.0 {
                1.0 - (a.dir.length() / a.n).min(1.0)
            } else {
                0.0
            };
            (k, a.n / a.op, spread, a.ppm, a.cov / a.op)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    /// Two Gaussians in one 1 m voxel (weights 2 and 4 views, opposite view
    /// directions), one below the opacity floor, one in another voxel.
    #[test]
    fn ingredients_aggregate_per_voxel() {
        let means = vec![0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 5.5, 0.5, 0.5];
        let opac = vec![1.0, 1.0, 0.01, 1.0];
        let weight = vec![2.0, 4.0, 9.0, 1.0];
        let dir_sum = vec![
            [2.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let ppm = vec![100.0, 300.0, 9999.0, 50.0];
        let cov = vec![0.5, 0.25, 1.0, 0.75];
        let rows = coverage_ingredients(&means, &opac, &weight, &dir_sum, &ppm, &cov, 1.0, 0.1);
        let v = rows.iter().find(|r| r.0 == IVec3::ZERO).unwrap();
        assert!((v.1 - 3.0).abs() < 1e-6, "n = (1·2 + 1·4)/2");
        // Σ op·dir = (2,0,0) + (−4,0,0) = (−2,0,0); Σ op·w = 6 → spread = 1 − 2/6.
        assert!((v.2 - (1.0 - 2.0 / 6.0)).abs() < 1e-6);
        assert_eq!(v.3, 300.0, "max ppm of the voxel's opaque Gaussians");
        assert!((v.4 - 0.375).abs() < 1e-6, "opacity-weighted mean coverage");
        let other = rows.iter().find(|r| r.0 == IVec3::new(5, 0, 0)).unwrap();
        assert!((other.1 - 1.0).abs() < 1e-6);
    }
}
