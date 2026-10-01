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
    /// Overrides GuideConfig's max_splats cap (default 750000).
    #[arg(long)]
    max_splats: Option<u32>,
    /// Evict the least important splats when growth hits the cap, instead of
    /// stopping growth (GuideConfig's default: on).
    #[arg(long, overrides_with = "no_evict")]
    evict: bool,
    /// Stop growth at the cap instead of evicting.
    #[arg(long, overrides_with = "evict")]
    no_evict: bool,
    /// Fraction of the cap one eviction frees (with --evict).
    #[arg(long)]
    evict_headroom: Option<f32>,
}

impl Args {
    /// Applies the splat budget flags on top of the defaults or config file.
    fn apply_budget(&self, config: &mut GuideConfig) {
        if let Some(max_splats) = self.max_splats {
            config.max_splats = max_splats;
        }
        if self.evict {
            config.evict = true;
        }
        if self.no_evict {
            config.evict = false;
        }
        if let Some(h) = self.evict_headroom {
            config.evict_headroom = h;
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let mut config: GuideConfig = match &args.config {
        Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
        None => GuideConfig::default(),
    };
    args.apply_budget(&mut config);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(flags: &[&str]) -> GuideConfig {
        let args = Args::try_parse_from(std::iter::once("server").chain(flags.iter().copied())).unwrap();
        let mut config = GuideConfig::default();
        args.apply_budget(&mut config);
        config
    }

    #[test]
    fn eviction_is_on_by_default_and_can_be_switched_off() {
        let c = budget(&[]);
        assert_eq!((c.max_splats, c.evict), (750_000, true));
        assert!(!budget(&["--no-evict"]).evict);
        assert!(budget(&["--no-evict", "--evict"]).evict, "the last flag wins");
        assert!(!budget(&["--evict", "--no-evict"]).evict, "the last flag wins");
        assert_eq!(budget(&["--max-splats", "200000"]).max_splats, 200_000);
    }
}
