use brush_recon_bench::cli::Cli;
use clap::Parser;

#[tokio::main]
async fn main() {
    if let Err(error) = brush_recon_bench::cli::execute(Cli::parse()).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
