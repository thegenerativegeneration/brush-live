//! Training-step cost on a recorded capture: ms per step for two splat counts
//! and three image sizes, measured in alternating blocks so that background
//! load on the machine spreads over all configurations, then a per-phase
//! profile (which syncs the GPU at phase boundaries). The forward probe
//! assumes 4:3 images; splats use SH degree 1; measured blocks run no refine.
//!
//! Usage: `cargo run --release -p brush-guide --features brush-cube/metal
//! --example step_bench [-- <dataset dir> [<ssim_every>]]` from `server/brush`
//! (default `../../datasets/segment-1`; `ssim_every` default 1, 0 never).

use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::load_dataset;
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::config::TrainConfig;
use brush_train::to_init_splats;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_vfs::BrushVfs;
use burn::module::Module;
use burn::tensor::{Device, s};
use clap::Parser;
use std::path::Path;
use std::sync::Arc;

#[derive(Parser)]
struct LoadArgs {
    #[command(flatten)]
    load: LoadDatasetConfig,
}

const SIZES: [u32; 3] = [500, 250, 125];
const TARGETS: [usize; 2] = [50_000, 10_000];
const BLOCK: usize = 100;
const ROUNDS: usize = 3;

async fn sync(splats: &Splats) {
    let _ = splats.device().sync();
}

async fn steps(trainer: &mut SplatTrainer, loader: &mut SceneLoader, splats: Splats, n: usize) -> Splats {
    let mut splats = splats;
    for _ in 0..n {
        let batch = loader.next_batch().await;
        let (stepped, _) = trainer.step(batch, splats.train()).await;
        splats = stepped.valid();
    }
    splats
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../../datasets/segment-1".to_owned());
    let ssim_every: u32 = std::env::args()
        .nth(2)
        .map_or(1, |s| s.parse().expect("ssim_every is a number"));
    println!("ssim_every {ssim_every}");
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let vfs = Arc::new(BrushVfs::from_path(Path::new(&dir)).await.expect("dataset dir"));

    let mut load = LoadArgs::parse_from(["step_bench"]).load;
    let mut loaders = Vec::new();
    let mut points = None;
    for size in SIZES {
        load.max_resolution = size;
        let result = load_dataset(vfs.clone(), &load).await.expect("dataset loads");
        points = points.or(result.init_splat);
        loaders.push(SceneLoader::new(&result.dataset.train, 0, &load));
    }
    let points = points.expect("dataset has points.ply").data;
    let total = points.num_splats();

    let mut train_config = TrainConfig::parse_from(["step_bench"]);
    train_config.total_train_iters = 1_000_000_000;
    train_config.ssim_every = ssim_every;
    let mut sets = Vec::new();
    for target in TARGETS {
        let stride = (total / target).max(1);
        let keep: Vec<usize> = (0..total).step_by(stride).take(target).collect();
        let mut data = points.clone();
        data.means = keep.iter().flat_map(|&i| points.means[i * 3..i * 3 + 3].to_vec()).collect();
        data.sh_coeffs = points
            .sh_coeffs
            .as_ref()
            .map(|c| {
                let per = c.len() / total;
                keep.iter().flat_map(|&i| c[i * per..(i + 1) * per].to_vec()).collect()
            });
        data.rotations = None;
        data.log_scales = None;
        data.raw_opacities = None;
        let splats = to_init_splats(data, SplatRenderMode::Default, &device.clone().inner()).with_sh_degree(1);
        let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        let mut trainer = SplatTrainer::new_seeded(&train_config, &device, bounds, 0);
        // Settle sizes and opacities at full size before measuring.
        let splats = steps(&mut trainer, &mut loaders[0], splats, 300).await;
        sync(&splats).await;
        println!("set: {} splats (of {total} points)", splats.num_splats());
        sets.push((trainer, splats));
    }

    let mut ms = vec![vec![0.0f64; SIZES.len()]; TARGETS.len()];
    for _ in 0..ROUNDS {
        for (t, (trainer, splats)) in sets.iter_mut().enumerate() {
            for (i, loader) in loaders.iter_mut().enumerate() {
                let start = web_time::Instant::now();
                let stepped = steps(trainer, loader, splats.clone(), BLOCK).await;
                sync(&stepped).await;
                ms[t][i] += start.elapsed().as_secs_f64() * 1e3 / (BLOCK * ROUNDS) as f64;
                *splats = stepped;
            }
        }
    }
    println!("\nms per step (no profiling), {ROUNDS} interleaved blocks of {BLOCK}:");
    println!("{:>8} {}", "splats", SIZES.map(|s| format!("{s:>7} px")).join(""));
    for (t, (_, splats)) in sets.iter().enumerate() {
        println!(
            "{:>8} {}",
            splats.num_splats(),
            ms[t].iter().map(|v| format!("{v:>10.2}")).collect::<String>()
        );
    }

    // What a step's fixed part is made of: a bare one-element readback (the
    // round trip the renderer makes mid-forward) and a forward render alone.
    let (_, splats) = &sets[0];
    let probe = splats.valid();
    let n = 200;
    let start = web_time::Instant::now();
    for _ in 0..n {
        sync(&probe).await;
    }
    println!("\nbare 1-element readback: {:.3} ms", start.elapsed().as_secs_f64() * 1e3 / f64::from(n));
    for (i, loader) in loaders.iter_mut().enumerate() {
        let camera = loader.next_batch().await.camera;
        let size = glam::uvec2(SIZES[i], SIZES[i] * 3 / 4);
        let start = web_time::Instant::now();
        for _ in 0..n {
            let (img, _) = brush_render::gaussian_splats::render_splats(
                probe.clone(),
                &camera,
                size,
                glam::Vec3::ZERO,
                None,
                brush_render::gaussian_splats::TextureMode::Float,
            )
            .await;
            let _ = img.slice(s![0..1, 0..1, ..]).into_data_async().await;
        }
        println!(
            "forward render alone at {} px, {} splats: {:.2} ms",
            SIZES[i],
            probe.num_splats(),
            start.elapsed().as_secs_f64() * 1e3 / f64::from(n)
        );
    }

    println!("\nphase profile at {} px (GPU synced at each boundary), ms per step:", SIZES[0]);
    for (trainer, splats) in &mut sets {
        trainer.set_profiling(true);
        let stepped = steps(trainer, &mut loaders[0], splats.clone(), 200).await;
        let p = trainer.take_profile().expect("profiling on");
        trainer.set_profiling(false);
        let n = f64::from(p.steps);
        println!(
            "{:>8} splats: forward {:.2}, loss {:.2}, backward {:.2}, optimizer {:.2}, stats+noise {:.2}",
            stepped.num_splats(),
            p.forward_s * 1e3 / n,
            p.loss_s * 1e3 / n,
            p.backward_s * 1e3 / n,
            p.optimizer_s * 1e3 / n,
            p.stats_noise_s * 1e3 / n
        );
        *splats = stepped;
    }
}
