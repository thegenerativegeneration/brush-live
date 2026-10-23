//! Finishing a session: export, finish eval, optional Fisher pass.

use super::Worker;
use std::path::Path;
use web_time::Instant;

impl Worker {
    /// Writes the whole model to `path` ([`crate::live::LiveModel::write_final`]), then runs the finish eval (and with
    /// `finish_fisher` the Fisher pass and a last round) and pauses training; returns the file size.
    pub(super) async fn finish(&mut self, path: &Path) -> Result<u64, String> {
        if self.live.splats().is_none() {
            return Err("no splats yet".to_owned());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let result = self.live.write_final(path).await.map(|e| e.ply_bytes).map_err(|e| e.to_string());
        if result.is_ok() {
            self.eval_holdout(true, true).await;
            if self.config.finish_fisher {
                let t = Instant::now();
                self.finish_pass(self.clock.elapsed().as_secs_f64()).await;
                self.totals_s.fisher += t.elapsed().as_secs_f64();
                let t = Instant::now();
                self.voxel_round(self.clock.elapsed().as_secs_f64()).await;
                self.totals_s.voxel += t.elapsed().as_secs_f64();
            }
            self.finished = true;
            self.channels
                .status
                .send_modify(|s| s.train_iters_per_s = 0.0);
            // The exported PLY does not include the SH background's globe; this JSON does.
            if let Some(json) = self.live.sh_background_json().await {
                let path = self.session_dir.join("sh_background.json");
                let _ = tokio::fs::write(&path, json).await;
            }
        }
        result
    }
}
