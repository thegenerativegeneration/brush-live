//! Benchmark for `score_pass`: prints total and per-view milliseconds for a
//! synthetic splat cloud at two render scales.
//!
//! Usage: `cargo run --release -p brush-guide --features brush-cube/metal
//! --example score_bench -- <n_splats> <n_views>` (defaults: 500000 300).

use brush_guide::scores::pass::{PassConfig, PassView, score_pass};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid};
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::Device;
use glam::{Mat4, UVec2, Vec3, vec2};
use rand::{RngExt, SeedableRng};

fn camera_at(pos: Vec3, look_at: Vec3) -> Camera {
    let mut c2w = Mat4::look_at_rh(pos, look_at, Vec3::Y).inverse();
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    let fov = 60f64.to_radians();
    Camera::new(
        translation,
        rotation,
        fov,
        fov * 0.75,
        vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (n_splats, n_views) = (
        args.first().copied().unwrap_or(500_000),
        args.get(1).copied().unwrap_or(300),
    );
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let means: Vec<f32> = (0..n_splats * 3)
        .map(|_| rng.random_range(-2.0..2.0))
        .collect();
    let splats = Splats::from_raw(
        means,
        [1.0, 0.0, 0.0, 0.0].repeat(n_splats),
        vec![-4.0; n_splats * 3],
        vec![0.5; n_splats * 3],
        vec![inverse_sigmoid(0.5); n_splats],
        SplatRenderMode::Default,
        &device,
    );
    let views: Vec<PassView> = (0..n_views)
        .map(|i| {
            let a = i as f32 / n_views as f32 * std::f32::consts::TAU;
            PassView {
                camera: camera_at(Vec3::new(4.0 * a.cos(), 0.5, 4.0 * a.sin()), Vec3::ZERO),
                img_size: UVec2::new(960, 720),
            }
        })
        .collect();
    for scale in [1.0, 0.5] {
        let cfg = PassConfig {
            render_scale: scale,
            ..Default::default()
        };
        score_pass(&splats, &views[..2], &cfg).await; // warm-up
        let t = web_time::Instant::now();
        score_pass(&splats, &views, &cfg).await;
        let ms = t.elapsed().as_millis();
        println!(
            "splats={n_splats} views={n_views} scale={scale}: {ms} ms total, {:.1} ms/view",
            ms as f32 / n_views as f32
        );
    }
}
