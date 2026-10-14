//! Held-out keyframes for measuring image quality in replays: every
//! `holdout_every`-th new keyframe is kept out of training and seeding, and
//! the splats are scored on those views on the `EvalSchedule` and at finish.

use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use super::eval_log::ViewScore;
use super::holdout_seam::{border_band, split_psnr, surface_points};
use brush_train::eval::{EvalSample, eval_stats};
use burn::tensor::{Tensor, TensorData};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use web_time::Instant;

pub(super) struct Holdout {
    every: u32,
    /// New keyframes seen, held out or not.
    received: u32,
    ids: HashSet<u64>,
    /// Images are loaded at eval time, so they do not sit in memory.
    views: Vec<(u64, SceneView)>,
    /// Loaded on the first eval with LPIPS.
    lpips: Option<lpips::LpipsModel>,
    /// Ids whose ground truth image was already saved.
    gt_saved: HashSet<u64>,
}

/// What one eval computes besides PSNR and SSIM.
pub(super) struct EvalExtras {
    pub lpips: bool,
    /// Width of the seam band in metres; 0 skips the split.
    pub seam_band_m: f32,
    pub tile_size_m: f32,
    /// Where renders of `save_ids` go; `None` saves nothing.
    pub save_dir: Option<PathBuf>,
    pub save_ids: Vec<u64>,
    pub iter: u32,
}

impl EvalExtras {
    /// PSNR and SSIM only.
    #[cfg(test)]
    pub(super) fn none() -> Self {
        Self {
            lpips: false,
            seam_band_m: 0.0,
            tile_size_m: 10.0,
            save_dir: None,
            save_ids: Vec::new(),
            iter: 0,
        }
    }
}

/// Mean quality over the held-out views.
pub(super) struct EvalResult {
    pub views: usize,
    pub psnr: f32,
    pub ssim: f32,
    pub ms: f64,
    /// Scores of each view that was scored, in holdout order.
    pub per_view: Vec<ViewScore>,
}

impl Holdout {
    /// `every` 0 holds nothing out; 1 would hold out everything and counts as 0. The eval interval belongs to
    /// the `EvalSchedule`.
    pub(super) fn new(every: u32, _interval_s: f32) -> Self {
        let every = if every == 1 {
            log::warn!("holdout_every 1 would hold out every keyframe; holdout off");
            0
        } else {
            every
        };
        Self {
            every,
            received: 0,
            ids: HashSet::new(),
            views: Vec::new(),
            lpips: None,
            gt_saved: HashSet::new(),
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
        self.views.push((id, SceneView { camera, ..view }));
        let mut ids: Vec<u64> = self.ids.iter().copied().collect();
        ids.sort_unstable();
        let json = serde_json::to_string(&ids).expect("ids serialize");
        if let Err(e) = std::fs::write(session_dir.join("holdout.json"), json) {
            log::warn!("could not write holdout.json: {e}");
        }
    }

    /// Moves held-out views to corrected poses; returns how many moved.
    pub(super) fn update_poses(&mut self, poses: &[crate::live::ViewPose]) -> u32 {
        let mut moved = 0;
        for p in poses {
            if let Some((_, view)) = self.views.iter_mut().find(|(id, _)| *id == p.id)
                && !p.matches(&view.camera)
            {
                view.camera = p.applied_to(&view.camera);
                moved += 1;
            }
        }
        moved
    }

    /// Keyframes received so far, trained or held out.
    pub(super) fn received(&self) -> u32 {
        self.received
    }

    /// Whether any view is held out.
    pub(super) fn has_views(&self) -> bool {
        !self.views.is_empty()
    }

    /// Scores `splats` on every held-out view that renders and loads; `None`
    /// if no view could be scored.
    pub(super) async fn eval(
        &mut self,
        splats: &Splats,
        _now_s: f64,
        extras: &EvalExtras,
    ) -> Option<EvalResult> {
        if self.views.is_empty() {
            return None;
        }
        if extras.lpips && self.lpips.is_none() {
            self.lpips = Some(lpips::load_vgg_lpips(&splats.device()));
        }
        let start = Instant::now();
        let device = splats.device();
        let (mut psnr, mut ssim, mut n) = (0.0f32, 0.0f32, 0usize);
        let mut per_view = Vec::new();
        for (id, view) in &self.views {
            match score_view(*id, splats, view, &device, extras, self.lpips.as_ref()).await {
                Ok((score, gt_to_save)) => {
                    psnr += score.psnr;
                    ssim += score.ssim;
                    n += 1;
                    per_view.push(score);
                    if let Some((path, img)) = gt_to_save.filter(|_| self.gt_saved.insert(*id)) {
                        if let Err(e) = img.save(&path) {
                            log::warn!("could not save {path:?}: {e}");
                        }
                    }
                }
                Err(e) => log::warn!("held-out view skipped: {e}"),
            }
        }
        (n > 0).then(|| EvalResult {
            views: n,
            psnr: psnr / n as f32,
            ssim: ssim / n as f32,
            ms: start.elapsed().as_secs_f64() * 1e3,
            per_view,
        })
    }
}

/// Scores one view; also returns the ground truth image and its path when `extras` saves this view.
async fn score_view(
    id: u64,
    splats: &Splats,
    view: &SceneView,
    device: &burn::tensor::Device,
    extras: &EvalExtras,
    lpips: Option<&lpips::LpipsModel>,
) -> Result<(ViewScore, Option<(PathBuf, image::RgbImage)>), String> {
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
    let scalar = |t: Tensor<1>| async move { t.into_scalar_async::<f32>().await.map_err(|e| e.to_string()) };
    let mut score = ViewScore {
        id,
        psnr: scalar(sample.psnr.clone()).await?,
        ssim: scalar(sample.ssim.clone()).await?,
        lpips: None,
        psnr_band: None,
        psnr_interior: None,
    };
    if let Some(model) = lpips.filter(|_| extras.lpips) {
        let gt = gt_tensor(&sample, device);
        score.lpips = Some(scalar(model.lpips(sample.rendered.clone().unsqueeze_dim(0), gt)).await?);
    }
    if extras.seam_band_m > 0.0 {
        let [h, w, _] = sample.rendered.dims();
        let points = surface_points(splats, &view.camera, glam::uvec2(w as u32, h as u32)).await;
        let band = border_band(&points, extras.tile_size_m, extras.seam_band_m);
        let render = sample
            .rendered
            .clone()
            .into_data_async()
            .await
            .map_err(|e| e.to_string())?
            .try_to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        let gt = sample.gt_img.to_rgb32f().into_raw();
        (score.psnr_band, score.psnr_interior) = split_psnr(&render, &gt, &band);
    }
    let mut gt_to_save = None;
    if let Some(dir) = extras.save_dir.as_ref().filter(|_| extras.save_ids.contains(&id)) {
        let path = dir.join(format!("{id}_{}.png", extras.iter));
        if let Err(e) = sample.save_to_disk(&path).await {
            log::warn!("could not save {path:?}: {e}");
        }
        gt_to_save = Some((dir.join(format!("gt_{id}.png")), sample.gt_img.to_rgb8()));
    }
    Ok((score, gt_to_save))
}

/// The ground truth as a `[1, h, w, 3]` float tensor.
fn gt_tensor(sample: &EvalSample, device: &burn::tensor::Device) -> Tensor<4> {
    let rgb = sample.gt_img.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Tensor::from_data(TensorData::new(rgb.into_raw(), [1, h, w, 3]), device)
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
    fn corrected_poses_move_held_out_views() {
        let dir = std::env::temp_dir().join(format!("holdout-poses-{}", std::process::id()));
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
        h.add_held(4, camera, view(&dir, 4), &dir);
        let p = crate::live::ViewPose {
            id: 4,
            position: glam::Vec3::new(0.3, 0.0, 0.0),
            rotation: glam::Quat::IDENTITY,
        };
        assert_eq!(
            h.update_poses(&[p, crate::live::ViewPose { id: 5, ..p }]),
            1
        );
        assert_eq!(h.views[0].1.camera.position, glam::Vec3::new(0.3, 0.0, 0.0));
        assert_eq!(h.update_poses(&[p]), 0);
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
        let result = h.eval(&splats, 0.0, &EvalExtras::none()).await.expect("one view scores");
        assert_eq!(result.views, 1);
        assert!(result.psnr.is_finite(), "psnr {}", result.psnr);
        assert_eq!(result.per_view.len(), 1);
        assert_eq!(result.per_view[0].id, 1);
        assert!((result.per_view[0].psnr - result.psnr).abs() < 1e-6);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn extras_add_lpips_and_save_renders() {
        let dir = std::env::temp_dir().join(format!("holdout-extras-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rel = PathBuf::from("5.jpg");
        image::RgbImage::from_pixel(64, 64, image::Rgb([90, 120, 30])).save(dir.join(&rel)).unwrap();
        let vfs = Arc::new(BrushVfs::from_directory_files(&dir, vec![rel.clone()]));
        let view = SceneView { image: LoadImage::new(vfs, rel, None, 64, None, false), camera: Camera::default() };
        let camera = Camera::new(glam::Vec3::ZERO, glam::Quat::IDENTITY, 1.0, 1.0, glam::vec2(0.5, 0.5),
            brush_render::kernels::camera_model::CameraModel::Pinhole);
        let mut h = Holdout::new(2, 30.0);
        h.add_held(5, camera, view, &dir);
        let device = brush_cube::test_helpers::test_device().await.into();
        let splats = Splats::from_raw(vec![0.0, 0.0, 2.0], vec![1.0, 0.0, 0.0, 0.0], vec![-1.0; 3], vec![0.5; 3],
            vec![inverse_sigmoid(0.5)], SplatRenderMode::Default, &device);
        let extras = EvalExtras { lpips: true, seam_band_m: 0.5, tile_size_m: 10.0,
            save_dir: Some(dir.join("eval_renders")), save_ids: vec![5], iter: 7 };
        let r = h.eval(&splats, 0.0, &extras).await.expect("scores");
        assert!(r.per_view[0].lpips.is_some_and(f32::is_finite));
        assert!(dir.join("eval_renders/5_7.png").exists());
        assert!(dir.join("eval_renders/gt_5.png").exists());
        std::fs::remove_dir_all(dir).ok();
    }
}
