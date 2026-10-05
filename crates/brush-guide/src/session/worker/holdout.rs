//! Held-out keyframes for measuring image quality in replays: every
//! `holdout_every`-th new keyframe is kept out of training and seeding, and
//! the splats are scored on those views every `eval_interval_s` and at finish.

use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::eval::eval_stats;
use std::collections::HashSet;
use std::path::Path;
use web_time::Instant;

pub(super) struct Holdout {
    every: u32,
    interval_s: f64,
    /// New keyframes seen, held out or not.
    received: u32,
    ids: HashSet<u64>,
    /// Images are loaded at eval time, so they do not sit in memory.
    views: Vec<SceneView>,
    last_eval_s: Option<f64>,
}

/// Mean quality over the held-out views.
pub(super) struct EvalResult {
    pub views: usize,
    pub psnr: f32,
    pub ssim: f32,
    pub ms: f64,
}

impl Holdout {
    /// `every` 0 holds nothing out; 1 would hold out everything and counts as 0.
    pub(super) fn new(every: u32, interval_s: f32) -> Self {
        let every = if every == 1 {
            log::warn!("holdout_every 1 would hold out every keyframe; holdout off");
            0
        } else {
            every
        };
        Self {
            every,
            interval_s: f64::from(interval_s),
            received: 0,
            ids: HashSet::new(),
            views: Vec::new(),
            last_eval_s: None,
        }
    }

    /// True if keyframe `id` was held out before (its resends are skipped).
    pub(super) fn contains(&self, id: u64) -> bool {
        self.ids.contains(&id)
    }

    /// Whether the next new keyframe is held out (every `every`-th one; with
    /// `every` >= 2 the first is never held out). Changes no state.
    pub(super) fn next_is_held(&self) -> bool {
        self.every != 0 && (self.received + 1).is_multiple_of(self.every)
    }

    /// A new keyframe went to training.
    pub(super) fn note_trained(&mut self) {
        self.received += 1;
    }

    /// Holds keyframe `id` out and records the held-out ids in
    /// `session_dir/holdout.json`.
    pub(super) fn add_held(
        &mut self,
        id: u64,
        camera: Camera,
        view: SceneView,
        session_dir: &Path,
    ) {
        self.received += 1;
        self.ids.insert(id);
        self.views.push(SceneView { camera, ..view });
        let mut ids: Vec<u64> = self.ids.iter().copied().collect();
        ids.sort_unstable();
        let json = serde_json::to_string(&ids).expect("ids serialize");
        if let Err(e) = std::fs::write(session_dir.join("holdout.json"), json) {
            log::warn!("could not write holdout.json: {e}");
        }
    }

    pub(super) fn due(&self, now_s: f64) -> bool {
        !self.views.is_empty()
            && self
                .last_eval_s
                .is_none_or(|last| now_s - last >= self.interval_s)
    }

    /// Scores `splats` on every held-out view that renders and loads; `None`
    /// if no view could be scored.
    pub(super) async fn eval(&mut self, splats: &Splats, now_s: f64) -> Option<EvalResult> {
        if self.views.is_empty() {
            return None;
        }
        self.last_eval_s = Some(now_s);
        let start = Instant::now();
        let device = splats.device();
        let (mut psnr, mut ssim, mut n) = (0.0f32, 0.0f32, 0usize);
        for view in &self.views {
            match score_view(splats, view, &device).await {
                Ok((p, s)) => {
                    psnr += p;
                    ssim += s;
                    n += 1;
                }
                Err(e) => log::warn!("held-out view skipped: {e}"),
            }
        }
        (n > 0).then(|| EvalResult {
            views: n,
            psnr: psnr / n as f32,
            ssim: ssim / n as f32,
            ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

async fn score_view(
    splats: &Splats,
    view: &SceneView,
    device: &burn::tensor::Device,
) -> Result<(f32, f32), String> {
    let image = view.image.load().await.map_err(|e| e.to_string())?;
    let sample = eval_stats(
        splats.clone(),
        &view.camera,
        image,
        view.image.alpha_mode(),
        device,
    )
    .await
    .map_err(|e| e.to_string())?;
    let psnr = sample
        .psnr
        .into_scalar_async::<f32>()
        .await
        .map_err(|e| e.to_string())?;
    let ssim = sample
        .ssim
        .into_scalar_async::<f32>()
        .await
        .map_err(|e| e.to_string())?;
    Ok((psnr, ssim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_dataset::load_image::LoadImage;
    use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
    use brush_vfs::BrushVfs;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// A view over a 2x2 JPEG written to `dir`.
    fn view(dir: &Path, id: u64) -> SceneView {
        let rel = PathBuf::from(format!("{id}.jpg"));
        image::RgbImage::new(2, 2).save(dir.join(&rel)).unwrap();
        let vfs = Arc::new(BrushVfs::from_directory_files(dir, vec![rel.clone()]));
        SceneView {
            image: LoadImage::new(vfs, rel, None, 2, None, false),
            camera: Camera::default(),
        }
    }

    #[test]
    fn every_nth_new_keyframe_is_held_out_with_non_sequential_ids() {
        let dir = std::env::temp_dir().join(format!("holdout-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut h = Holdout::new(3, 30.0);
        let mut held = Vec::new();
        for id in [10u64, 11, 40, 41, 77, 90, 91, 92, 99] {
            if h.next_is_held() {
                h.add_held(id, Camera::default(), view(&dir, id), &dir);
                held.push(id);
            } else {
                h.note_trained();
            }
        }
        assert_eq!(held, vec![40, 90, 99]);
        assert!(h.contains(90) && !h.contains(91));
        let json = std::fs::read_to_string(dir.join("holdout.json")).unwrap();
        assert_eq!(json, "[40,90,99]");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn next_is_held_changes_no_state() {
        let h = Holdout::new(2, 30.0);
        assert!(!h.next_is_held() && !h.next_is_held());
        let mut h = h;
        h.note_trained();
        assert!(h.next_is_held() && h.next_is_held());
    }

    #[test]
    fn zero_holds_nothing_out() {
        let mut h = Holdout::new(0, 30.0);
        for _ in 0..20 {
            assert!(!h.next_is_held());
            h.note_trained();
        }
    }

    #[test]
    fn one_is_treated_as_off() {
        let h = Holdout::new(1, 30.0);
        assert!(!h.next_is_held());
    }

    #[test]
    fn due_only_with_views_and_after_the_interval() {
        let dir = std::env::temp_dir().join(format!("holdout-due-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut h = Holdout::new(2, 30.0);
        assert!(!h.due(100.0), "no views yet");
        h.views.push(view(&dir, 0));
        assert!(h.due(0.0));
        h.last_eval_s = Some(10.0);
        assert!(!h.due(39.0));
        assert!(h.due(40.0));
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn eval_scores_views_that_load_and_skips_the_missing_one() {
        let dir = std::env::temp_dir().join(format!("holdout-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            1.0,
            1.0,
            glam::vec2(0.5, 0.5),
            brush_render::kernels::camera_model::CameraModel::Pinhole,
        );
        let mut h = Holdout::new(2, 30.0);
        h.add_held(1, camera.clone(), view(&dir, 1), &dir);
        h.add_held(2, camera, view(&dir, 2), &dir);
        std::fs::remove_file(dir.join("2.jpg")).unwrap();

        let device = brush_cube::test_helpers::test_device().await.into();
        let n = 20;
        let splats = Splats::from_raw(
            (0..n).flat_map(|i| [i as f32 * 0.01, 0.0, 2.0]).collect(),
            [1.0, 0.0, 0.0, 0.0].repeat(n),
            vec![-3.0; n * 3],
            vec![0.5; n * 3],
            vec![inverse_sigmoid(0.5); n],
            SplatRenderMode::Default,
            &device,
        );
        let result = h.eval(&splats, 0.0).await.expect("one view scores");
        assert_eq!(result.views, 1);
        assert!(result.psnr.is_finite(), "psnr {}", result.psnr);
        std::fs::remove_dir_all(dir).ok();
    }
}
