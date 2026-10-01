use brush_render::bwd::render_splats;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use burn::module::Module;
use burn::tensor::{Distribution, Tensor, s};
use glam::{UVec2, Vec3};

pub struct PassView {
    pub camera: Camera,
    pub img_size: UVec2,
    /// Multiplies the view's Fisher, observation count and direction, e.g.
    /// a sampling weight so a round's sums estimate sums over all views.
    pub weight: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PassConfig {
    /// Random-sign probes per view. One is enough when summing over many views.
    pub hutchinson_samples: u32,
    /// A Gaussian counts as observed in a view if its per-view Fisher trace exceeds this.
    pub observed_eps: f32,
    /// Multiplies each view's resolution for the pass.
    pub render_scale: f32,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            hutchinson_samples: 1,
            observed_eps: 1e-12,
            render_scale: 0.5,
        }
    }
}

pub struct PassOutput {
    pub fisher: Vec<[f32; 36]>,
    pub dir_sum: Vec<[f32; 3]>,
    pub weight: Vec<f32>,
    pub max_px_per_m: Vec<f32>,
}

async fn read_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("score readback")
        .try_to_vec::<f32>()
        .expect("f32 scores")
}

/// One forward/backward per view and probe. Back-propagating `Σ r·render` with
/// Rademacher `r` gives gradients `g` with `E[g gᵀ] = Σ_p J_p J_pᵀ`, the Fisher
/// of the render, independent of any ground-truth image.
pub async fn score_pass(splats: &Splats, views: &[PassView], cfg: &PassConfig) -> PassOutput {
    let n = splats.num_splats() as usize;
    // A detached snapshot: each probe lifts it to a fresh autodiff leaf, so the
    // pass works whatever graph the caller's params are part of.
    let base = splats.valid();
    let inner = base.device();
    let samples = cfg.hutchinson_samples.max(1);

    let mut fisher: Tensor<3> = Tensor::zeros([n, 6, 6], &inner);
    let mut dir_sum: Tensor<2> = Tensor::zeros([n, 3], &inner);
    let mut weight: Tensor<1> = Tensor::zeros([n], &inner);
    let mut max_ppm: Tensor<1> = Tensor::zeros([n], &inner);
    let means = base.means();
    let t_start = web_time::Instant::now();

    for view in views {
        let size = (view.img_size.as_vec2() * cfg.render_scale)
            .as_uvec2()
            .max(UVec2::ONE);
        let mut trace: Tensor<1> = Tensor::zeros([n], &inner);

        for _ in 0..samples {
            let s = base.clone().train();
            let out = render_splats(s.clone(), &view.camera, size, Vec3::ZERO).await;
            let rgb = out.img.slice(s![.., .., 0..3]);
            let signs: Tensor<3> =
                Tensor::random(rgb.dims(), Distribution::Bernoulli(0.5), &rgb.device()) * 2.0 - 1.0;
            let mut grads = (rgb * signs).sum().backward();
            let g = s
                .transforms
                .val()
                .grad_remove(&mut grads)
                .expect("transform grads");
            let j = Tensor::cat(
                vec![g.clone().slice(s![.., 0..3]), g.slice(s![.., 7..10])],
                1,
            );
            let outer = j.clone().unsqueeze_dim::<3>(2) * j.clone().unsqueeze_dim::<3>(1);
            fisher = fisher + outer * (view.weight / samples as f32);
            trace = trace + j.powi_scalar(2).sum_dim(1).squeeze_dim(1) / samples as f32;
        }

        let observed = trace.greater_elem(cfg.observed_eps).float();
        let cam = view.camera.position;
        let cam_t = Tensor::<1>::from_floats([cam.x, cam.y, cam.z], &inner).reshape([1, 3]);
        let to_g = means.clone() - cam_t;
        let dist = to_g
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-6); // [n,1]
        let dir = to_g / dist.clone();
        dir_sum = dir_sum + dir * (observed.clone() * view.weight).unsqueeze_dim::<2>(1);
        weight = weight + observed.clone() * view.weight;
        let focal = view.camera.focal(view.img_size).x;
        let ppm = dist.squeeze_dim::<1>(1).recip() * focal * observed;
        max_ppm = max_ppm.max_pair(ppm);
        crate::timing::sync(&max_ppm).await;
    }
    let t_views = t_start.elapsed().as_secs_f64();

    let t_read = web_time::Instant::now();
    let fisher_v = read_vec(fisher).await;
    let dir_v = read_vec(dir_sum).await;
    let out = PassOutput {
        fisher: fisher_v.as_chunks::<36>().0.to_vec(),
        dir_sum: dir_v.as_chunks::<3>().0.to_vec(),
        weight: read_vec(weight).await,
        max_px_per_m: read_vec(max_ppm).await,
    };
    log::debug!(
        target: crate::timing::TARGET,
        "score_pass: {} views, {n} splats, render+bwd {:.0} ms ({:.1} ms/view), readback {:.0} ms",
        views.len(),
        t_views * 1e3,
        t_views * 1e3 / views.len().max(1) as f64,
        t_read.elapsed().as_secs_f64() * 1e3
    );
    out
}
