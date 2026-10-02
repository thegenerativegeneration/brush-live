use brush_guide_bench::{BenchParams, HostSample, run};
use clap::Parser;
use std::path::PathBuf;

/// Spike: replays a capture's wire frames into the live engine and logs metrics.
#[derive(Parser)]
struct Args {
    /// A capture's `wire/` directory.
    wire_dir: PathBuf,
    #[arg(long, default_value = "bench-out")]
    out: PathBuf,
    #[arg(long, default_value_t = 100_000)]
    max_splats: u32,
    #[arg(long, default_value_t = 960)]
    width: u32,
    #[arg(long, default_value_t = 600.0)]
    duration_s: f32,
    #[arg(long, default_value_t = 2.0)]
    rate: f32,
    #[arg(long, default_value_t = 10.0)]
    sample_s: f32,
    #[arg(long, default_value_t = 1024)]
    loader_cache_mb: u32,
    /// Training iterations per second at most; 0 is uncapped.
    #[arg(long, default_value_t = 0.0)]
    max_iters_per_s: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let params = BenchParams {
        max_splats: a.max_splats,
        width: a.width,
        duration_s: a.duration_s,
        rate: a.rate,
        sample_s: a.sample_s,
        loader_cache_mb: a.loader_cache_mb,
        max_iters_per_s: a.max_iters_per_s,
    };
    run(params, &a.wire_dir, &a.out, || HostSample::UNKNOWN).await
}
