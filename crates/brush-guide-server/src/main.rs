use brush_guide::config::GuideConfig;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Where keyframe images are stored, one folder per session.
    #[arg(long, default_value = "guide-sessions")]
    root: PathBuf,
    /// Optional JSON file with GuideConfig overrides.
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let config: GuideConfig = match &args.config {
        Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
        None => GuideConfig::default(),
    };
    let device = brush_process::burn_init_setup().await.autodiff();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await?;
    log::info!("listening on {}", listener.local_addr()?);
    brush_guide_server::server::serve(listener, config, device, args.root).await
}
