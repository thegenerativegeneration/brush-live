//! Held-out keyframes for measuring image quality in replays: every
//! `holdout_every`-th new keyframe is kept out of training and seeding, and
//! the splats are scored on those views on the `EvalSchedule` and at finish.

use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use super::eval_log::ViewScore;
use super::holdout_seam::{border_band, read_means, split_psnr, surface_points};
use crate::config::GuideConfig;
use brush_train::eval::{EvalSample, composite_over, eval_stats_over};
use brush_train::sh_background::ShBackground;
use burn::tensor::{Tensor, TensorData};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use web_time::Instant;

pub(super) struct Holdout {
    every: u32,
    /// Received count after which nothing is held out; 0 no limit.
    max_received: u32,
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
    /// Views scored with LPIPS when `lpips` is set; empty: every view.
    pub lpips_ids: Vec<u64>,
    /// Width of the seam band in metres; 0 skips the split.
    pub seam_band_m: f32,
    pub tile_size_m: f32,
    /// Where renders of `save_ids` go; `None` saves nothing.
    pub save_dir: Option<PathBuf>,
    pub save_ids: Vec<u64>,
    pub iter: u32,
}

impl EvalExtras {
    /// What an eval at `received` keyframes computes under `config`. With `extras`: LPIPS on
    /// `eval_extras_views` (every view when empty), the seam split from `eval_seam_from` received keyframes on
    /// and at `finish`, and the saved renders.
    pub(super) fn for_eval(
        config: &GuideConfig,
        extras: bool,
        received: u32,
        finish: bool,
        session_dir: &Path,
        iter: u32,
    ) -> Self {
        let seam = extras && (finish || received >= config.eval.seam_from);
        Self {
            lpips: extras && config.eval.lpips,
            lpips_ids: config.eval.extras_views.clone(),
            seam_band_m: if seam { config.eval.seam_band_m } else { 0.0 },
            tile_size_m: config.area.tile_size_m,
            save_dir: (extras && !config.eval.save_views.is_empty()).then(|| session_dir.join("eval_renders")),
            save_ids: config.eval.save_views.clone(),
            iter,
        }
    }

    fn lpips_for(&self, id: u64) -> bool {
        self.lpips && (self.lpips_ids.is_empty() || self.lpips_ids.contains(&id))
    }

    /// PSNR and SSIM only.
    #[cfg(test)]
    pub(super) fn none() -> Self {
        Self {
            lpips: false,
            lpips_ids: Vec::new(),
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
    /// Wall time of the whole eval, including `lpips_ms` and `seam_ms`.
    pub ms: f64,
    /// Time spent on LPIPS (model load included) and on the seam split.
    pub lpips_ms: f64,
    pub seam_ms: f64,
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
            max_received: 0,
            received: 0,
            ids: HashSet::new(),
            views: Vec::new(),
            lpips: None,
            gt_saved: HashSet::new(),
        }
    }

    /// Holdout under `config`: every `holdout_every`-th keyframe up to `holdout_max_received` received ones.
    pub(super) fn from_config(config: &GuideConfig) -> Self {
        Self {
            max_received: config.holdout.max_received,
            ..Self::new(config.holdout.every, config.eval.interval_s)
        }
    }

    /// True if keyframe `id` was held out before (its resends are skipped).
    pub(super) fn contains(&self, id: u64) -> bool {
        self.ids.contains(&id)
    }

    /// Whether the next new keyframe is held out (every `every`-th one up to `max_received`; with
    /// `every` >= 2 the first is never held out). Changes no state.
    pub(super) fn next_is_held(&self) -> bool {
        self.every != 0
            && (self.max_received == 0 || self.received < self.max_received)
            && (self.received + 1).is_multiple_of(self.every)
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

    /// Scores `splats` on every held-out view that renders and loads, composited over `background` when given
    /// (black otherwise); `None` if no view could be scored.
    pub(super) async fn eval(
        &mut self,
        splats: &Splats,
        _now_s: f64,
        extras: &EvalExtras,
        background: Option<&ShBackground>,
    ) -> Option<EvalResult> {
        if self.views.is_empty() {
            return None;
        }
        let start = Instant::now();
        let mut cost = ViewCost::default();
        if extras.lpips && self.lpips.is_none() {
            self.lpips = Some(lpips::load_vgg_lpips(&splats.device()));
            cost.lpips_ms += start.elapsed().as_secs_f64() * 1e3;
        }
        let device = splats.device();
        let means = if extras.seam_band_m > 0.0 {
            let t = Instant::now();
            let means = read_means(splats).await.inspect_err(|e| log::warn!("seam split skipped: {e}")).ok();
            cost.seam_ms += t.elapsed().as_secs_f64() * 1e3;
            means
        } else {
            None
        };
        let (mut psnr, mut ssim, mut n) = (0.0f32, 0.0f32, 0usize);
        let mut per_view = Vec::new();
        for (id, view) in &self.views {
            let ctx = ViewContext { splats, device: &device, extras, lpips: self.lpips.as_ref(), means: means.as_deref(), background };
            let scored = score_view(*id, view, &ctx, &mut cost).await;
            match scored {
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
            lpips_ms: cost.lpips_ms,
            seam_ms: cost.seam_ms,
            per_view,
        })
    }
}

/// Milliseconds an eval spent on its extras.
#[derive(Default)]
struct ViewCost {
    lpips_ms: f64,
    seam_ms: f64,
}

/// What every view of one eval shares.
struct ViewContext<'a> {
    splats: &'a Splats,
    device: &'a burn::tensor::Device,
    extras: &'a EvalExtras,
    lpips: Option<&'a lpips::LpipsModel>,
    /// The splats' means on the host; enables the seam split.
    means: Option<&'a [f32]>,
    /// Composited behind every render; black without.
    background: Option<&'a ShBackground>,
}

/// Scores one view; also returns the ground truth image and its path when `extras` saves this view. Time spent on
/// extras is added to `cost`.
async fn score_view(
    id: u64,
    view: &SceneView,
    ctx: &ViewContext<'_>,
    cost: &mut ViewCost,
) -> Result<(ViewScore, Option<(PathBuf, image::RgbImage)>), String> {
    let ViewContext { splats, device, extras, lpips, means, background } = *ctx;
    let image = view.image.load().await.map_err(|e| e.to_string())?;
    let sample = eval_stats_over(
        splats.clone(),
        &view.camera,
        image,
        view.image.alpha_mode(),
        background,
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
    if let Some(model) = lpips.filter(|_| extras.lpips_for(id)) {
        let t = Instant::now();
        let gt = gt_tensor(&sample, device);
        score.lpips = Some(scalar(model.lpips(sample.rendered.clone().unsqueeze_dim(0), gt)).await?);
        cost.lpips_ms += t.elapsed().as_secs_f64() * 1e3;
    }
    if let Some(means) = means {
        let t = Instant::now();
        let [h, w, _] = sample.rendered.dims();
        let points = surface_points(splats, means, &view.camera, glam::uvec2(w as u32, h as u32)).await;
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
        cost.seam_ms += t.elapsed().as_secs_f64() * 1e3;
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

/// Compiles (and, where autotune is on, tunes) the eval's kernels at keyframe `size` before a session starts: one
/// LPIPS forward on noise, one seam render of `splats` and one composite over a fresh SH background, as configured.
/// Nothing without a holdout.
pub(crate) async fn warm_up_eval(config: &GuideConfig, splats: &Splats, camera: &Camera, size: glam::UVec2) {
    if config.holdout.every < 2 {
        return;
    }
    let device = splats.device();
    if config.sh_background {
        let globe = ShBackground::new(&device.clone().autodiff(), 1.0);
        let render = Tensor::<3>::zeros([size.y as usize, size.x as usize, 4], &device);
        if let Err(e) = composite_over(render, Some(&globe), camera).into_data_async().await {
            log::warn!("warm-up: SH background composite failed: {e}");
        }
    }
    if config.eval.lpips {
        let model = lpips::load_vgg_lpips(&device);
        let noise = || {
            Tensor::<4>::random(
                [1, size.y as usize, size.x as usize, 3],
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &device,
            )
        };
        if let Err(e) = model.lpips(noise(), noise()).into_scalar_async::<f32>().await {
            log::warn!("warm-up: LPIPS failed: {e}");
        }
    }
    if config.eval.seam_band_m > 0.0 {
        match read_means(splats).await {
            Ok(means) => drop(surface_points(splats, &means, camera, size).await),
            Err(e) => log::warn!("warm-up: seam render skipped: {e}"),
        }
    }
}

/// The ground truth as a `[1, h, w, 3]` float tensor.
fn gt_tensor(sample: &EvalSample, device: &burn::tensor::Device) -> Tensor<4> {
    let rgb = sample.gt_img.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Tensor::from_data(TensorData::new(rgb.into_raw(), [1, h, w, 3]), device)
}

#[cfg(test)]
#[path = "holdout_tests.rs"]
mod tests;
