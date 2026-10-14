//! One JSON line per held-out eval in `session_dir/eval.jsonl`, for the replay measurement report.

use serde::Serialize;
use std::path::Path;

/// One held-out view's scores.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(super) struct ViewScore {
    pub id: u64,
    pub psnr: f32,
    pub ssim: f32,
    pub lpips: Option<f32>,
    /// PSNR over pixels whose surface lies within the seam band of a tile border.
    pub psnr_band: Option<f32>,
    /// PSNR over the other pixels with a surface.
    pub psnr_interior: Option<f32>,
}

/// One eval: when, on what model, and every view's scores.
#[derive(Debug, Serialize)]
pub(super) struct EvalRecord {
    /// Wall clock (seconds since the Unix epoch), to align with the footprint sampler.
    pub unix_s: f64,
    /// Seconds since the worker started.
    pub t_s: f64,
    pub iter: u32,
    /// Training views.
    pub keyframes: u32,
    /// Keyframes received, trained or held out: the replay's send count, which aligns evals across runs.
    pub received_keyframes: u32,
    pub last_keyframe_id: Option<u64>,
    /// Splats the eval rendered (the full model).
    pub splats: u32,
    /// Splats in the trainer.
    pub resident_splats: u32,
    pub loader_cache_bytes: u64,
    /// Cumulative training wall time.
    pub train_ms: u64,
    /// Cumulative refine wall time (part of `train_ms`).
    pub refine_ms: f64,
    pub eval_ms: f64,
    /// The trainer's bounds' median size, which scales the mean learning rate.
    pub bounds_median_size: f32,
    pub adam_clamped: u64,
    /// The eval carried the extras (LPIPS, seam split, saved renders).
    pub extras: bool,
    pub views: Vec<ViewScore>,
}

/// Appends `record` as one line.
pub(super) fn append(path: &Path, record: &EvalRecord) -> std::io::Result<()> {
    use std::io::Write;
    let line = serde_json::to_string(record).map_err(std::io::Error::other)?;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(iter: u32) -> EvalRecord {
        EvalRecord {
            unix_s: 1.5e9,
            t_s: 12.5,
            iter,
            keyframes: 40,
            received_keyframes: 44,
            last_keyframe_id: Some(43),
            splats: 1000,
            resident_splats: 800,
            loader_cache_bytes: 4096,
            train_ms: 9000,
            refine_ms: 400.0,
            eval_ms: 31.0,
            bounds_median_size: 0.8,
            adam_clamped: 0,
            extras: false,
            views: vec![ViewScore { id: 9, psnr: 21.5, ssim: 0.7, lpips: None, psnr_band: Some(20.0), psnr_interior: None }],
        }
    }

    #[test]
    fn records_append_as_json_lines() {
        let dir = std::env::temp_dir().join(format!("eval-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("eval.jsonl");
        append(&path, &record(100)).unwrap();
        append(&path, &record(200)).unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["iter"], 200);
        assert_eq!(lines[0]["views"][0]["id"], 9);
        assert!(lines[0]["views"][0]["lpips"].is_null());
        assert_eq!(lines[0]["last_keyframe_id"], 43);
        std::fs::remove_dir_all(dir).ok();
    }
}
