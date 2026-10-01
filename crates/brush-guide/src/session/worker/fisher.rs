//! The Fisher pass: render and backward over a sample of the views, giving
//! each voxel its coverage and uncertainty and each splat its eviction
//! importance.
//!
//! A pass is self-contained: its views are weighted by Horvitz–Thompson
//! (`schedule::ViewSample`), so its sums estimate sums over every view at
//! one model state, and the uncertainty byte ranks voxels within the pass
//! (5th–95th percentile of `ln σ`). Passes are combined only through the
//! per-voxel EMA of that byte. Summing or averaging raw Fisher across passes
//! would mix model states: with no new views, a voxel's σ rises ×1.58 over
//! about a dozen rounds as training lowers the information per unit opacity
//! (Task 5 spike), so older passes would make a region look more certain
//! than a fresh pass of the same views. Ranks within a pass are free of that
//! drift.

use super::{Worker, append_raw_round};
use crate::schedule::{FisherCost, ViewSample};
use crate::scores::importance::importances;
use crate::scores::metrics::gaussian_metrics;
use crate::scores::pass::{PassView, score_pass};
use crate::session::splat_read::SplatRead;
use web_time::Instant;

impl Worker {
    /// One Fisher pass over at most `max_views` views; `start` is when it
    /// started. Its bytes reach the phone with the next voxel round.
    pub(super) async fn fisher_pass(&mut self, start: f64, max_views: usize) {
        let config = &self.config;
        let splats = self.live.splats().expect("views imply splats").clone();
        let num_views = self.live.views().len();
        let sample = ViewSample::new(max_views);
        let pass = self.fisher_passes;
        self.fisher_passes += 1;
        let views: Vec<PassView> = sample
            .select(num_views, pass, config.seed)
            .into_iter()
            .map(|i| PassView {
                camera: self.live.views()[i].camera,
                img_size: self.sizes[i],
                // Sums over the sample estimate sums over every view, so
                // `CoverageParams::n_target` and σ refer to the whole capture.
                weight: sample.weight(i, num_views),
            })
            .collect();

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
        if config.evict {
            let importance = importances(&out, &read.rots, &read.scales);
            self.live.set_importance(&importance);
        }
        let t_importance = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let gaussians = read.scores(&coverage, &fisher_pos);
        let t_prep = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let scored = self.voxels.update_fisher(&gaussians);
        let t_agg = t.elapsed().as_secs_f64();
        if self.config.dump_raw_uncertainty {
            let path = self.session_dir.join("raw_uncertainty.jsonl");
            // The score set that first carries this pass.
            if let Err(e) = append_raw_round(&path, self.version + 1, self.voxels.raw_round()) {
                log::warn!("raw uncertainty dump to {}: {e}", path.display());
            }
        }

        let duration = self.clock.elapsed().as_secs_f64() - start;
        self.scheduler.fisher.record(start, duration);
        self.fisher_cost = Some((
            FisherCost {
                per_view_s: t_pass / views.len().max(1) as f64,
                fixed_s: (duration - t_pass).max(0.0),
            },
            splats.num_splats(),
        ));
        log::info!(
            "fisher pass {pass} at {start:.2} s: {} of {num_views} views (up to {max_views}), {:.0} ms, \
             {scored} voxels, {} splats",
            views.len(),
            duration * 1e3,
            splats.num_splats()
        );
        log::debug!(
            target: crate::timing::TARGET,
            "fisher: {} views, pass {:.0} ms, metrics {:.0} ms, splat readback {:.0} ms, importance {:.0} ms, \
             gaussian prep {:.0} ms, voxel aggregate {:.0} ms",
            views.len(),
            t_pass * 1e3,
            t_metrics * 1e3,
            t_read * 1e3,
            t_importance * 1e3,
            t_prep * 1e3,
            t_agg * 1e3
        );
    }
}
