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
    /// Disable Bonjour advertisement.
    #[arg(long)]
    no_mdns: bool,
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
    let mdns_guard = if args.no_mdns {
        None
    } else {
        match brush_guide_server::mdns::advertise(args.port) {
            Ok(d) => {
                log::info!(
                    "advertising {} on port {}",
                    brush_guide_server::mdns::SERVICE_TYPE,
                    args.port
                );
                Some(d)
            }
            Err(e) => {
                log::warn!("Bonjour advertisement failed: {e}");
                None
            }
        }
    };
    // Run until the server stops on its own, or until we're asked to shut down; either
    // way, drop `mdns_guard` before returning so the Bonjour registration is torn down.
    let result = tokio::select! {
        result = brush_guide_server::server::serve(listener, config, device, args.root) => result,
        () = shutdown_signal() => {
            log::info!("shutting down");
            Ok(())
        }
    };
    drop(mdns_guard);
    result
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
