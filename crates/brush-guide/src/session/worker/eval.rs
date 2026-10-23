//! The held-out eval: scores the splats on the held-out views and logs one record.

use super::holdout::EvalExtras;
use super::{eval_log, jsonl, Worker};
use web_time::Instant;

impl Worker {
    /// Scores the splats on the held-out views and logs the result; `extras` adds LPIPS, the seam split and
    /// saved renders as configured ([`EvalExtras::for_eval`]), `finish` marks the finish eval. Whether any view
    /// was scored (not without splats or renderable views).
    pub(super) async fn eval_holdout(&mut self, extras: bool, finish: bool) -> bool {
        // Nothing to score: skip the full model, which loads every parked tile.
        if !self.holdout.has_views() {
            return false;
        }
        let now = self.clock.elapsed().as_secs_f64();
        let eval_extras = EvalExtras::for_eval(
            &self.config,
            extras,
            self.holdout.received(),
            finish,
            &self.session_dir,
            self.live.iter(),
        );
        let t_load = Instant::now();
        let Some(splats) = self.live.eval_splats().await else {
            return false;
        };
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
        // Loading the parked tiles does not count against the training step rate.
        self.rate_window.0 += load_ms / 1e3;
        let background = self.live.sh_background();
        let composited = if background.is_some() { eval_log::EvalBackground::Sh } else { eval_log::EvalBackground::Black };
        let Some(r) = self.holdout.eval(&splats, now, &eval_extras, background).await else {
            return false;
        };
        // Nor does the eval itself.
        self.rate_window.0 += r.ms / 1e3;
        log::info!(
            "eval at {now:.2} s: {} held-out views, psnr {:.2} dB, ssim {:.4}; iter {}, {} splats ({:.0} ms, {} background)",
            r.views,
            r.psnr,
            r.ssim,
            self.live.iter(),
            splats.num_splats(),
            r.ms,
            if composited == eval_log::EvalBackground::Sh { "SH" } else { "black" }
        );
        let record = eval_log::EvalRecord {
            unix_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64()),
            t_s: now,
            iter: self.live.iter(),
            keyframes: self.live.views().len() as u32,
            received_keyframes: self.holdout.received(),
            last_keyframe_id: self.live.last_keyframe_id(),
            splats: splats.num_splats(),
            resident_splats: self.live.splats().map_or(0, |s| s.num_splats()),
            loader_cache_bytes: self.live.loader_cache_bytes(),
            train_ms: (self.totals_s.train * 1e3) as u64,
            refine_ms: self.live.refine_total_ms(),
            eval_ms: r.ms,
            lpips_ms: r.lpips_ms,
            seam_ms: r.seam_ms,
            bounds_median_size: self.live.bounds_median_size(),
            lr_mean: self.live.last_step_lr().lr_mean,
            position_scale_m: self.live.last_step_lr().position_scale_m,
            adam_clamped: self.live.adam_clamped(),
            masked_by_radius: self.live.masked_by_radius().await,
            area_views: self.live.area_view_count(),
            sight_views: self.live.sight_view_count(),
            sight_voters: self.live.sight_voter_count(),
            parked_load_ms: self.live.area_view_count().map(|_| load_ms),
            area: self.live.area_report(),
            extras,
            background: composited,
            views: r.per_view.clone(),
        };
        if let Err(e) = jsonl::append_jsonl(&self.session_dir.join("eval.jsonl"), &record) {
            log::warn!("could not append eval.jsonl: {e}");
        }
        true
    }
}
