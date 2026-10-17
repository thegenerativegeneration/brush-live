use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Offline native-Metal Gaussian/triangle reconstruction experiment")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Subcommand)]
pub enum Command {
    Preflight(PreflightArgs),
    Run(RunArgs),
}
#[derive(Args, Clone, Debug)]
pub struct PreflightArgs {
    #[arg(long)]
    pub segment: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 174)]
    pub first_id: u64,
    #[arg(long, default_value_t = 213)]
    pub last_id: u64,
    #[arg(long, default_value_t = 5)]
    pub holdout_every: usize,
    #[arg(long, default_value_t = 4)]
    pub holdout_offset: usize,
    #[arg(long,default_value_t=320,value_parser=clap::value_parser!(u32).range(4..=4096))]
    pub width: u32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum BackendChoice {
    Gaussian,
    Triangle,
}
#[derive(Args, Clone, Debug)]
#[command(group(clap::ArgGroup::new("budget").required(true).multiple(false).args(["steps","train_seconds"])))]
pub struct RunArgs {
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long, value_enum)]
    pub backend: BackendChoice,
    #[arg(long,value_parser=clap::value_parser!(u64).range(2..))]
    pub steps: Option<u64>,
    #[arg(long)]
    pub train_seconds: Option<f64>,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 2)]
    pub warmup_steps: u64,
}

pub async fn execute(cli: Cli) -> Result<(), brush_recon::ReconError> {
    use crate::{
        gaussian::GaussianBackend,
        io,
        preflight::{load_frozen, preflight, verify_sources},
        run::{RunBudget, RunConfig, run_backend},
        triangle::TriangleBackend,
    };
    match cli.command {
        Command::Preflight(args) => {
            let (_, environment) = crate::runtime::initialize().await?;
            let result = preflight(&args, environment)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&result)
                    .map_err(|e| brush_recon::ReconError::input(e.to_string()))?
            );
        }
        Command::Run(args) => {
            let budget = match (args.steps, args.train_seconds) {
                (Some(steps), None) => RunBudget::Steps { steps },
                (None, Some(seconds)) => RunBudget::Time { seconds },
                _ => {
                    return Err(brush_recon::ReconError::input(
                        "choose exactly one training budget",
                    ));
                }
            };
            budget.validate()?;
            let config = RunConfig {
                budget,
                warmup_steps: args.warmup_steps,
            };
            let loading = std::time::Instant::now();
            let frozen = load_frozen(&args.manifest)?;
            let source_load_seconds = loading.elapsed().as_secs_f64();
            let output = io::validate_output(&frozen.split.manifest.source, &args.output)?;
            let (device, environment) = crate::runtime::initialize().await?;
            io::create_output(&output)?;
            let record = match args.backend {
                BackendChoice::Triangle => {
                    run_backend(
                        || {
                            TriangleBackend::new(
                                frozen.seed.clone(),
                                device.clone(),
                                frozen.config.clone(),
                            )
                        },
                        &frozen.split,
                        &config,
                    )
                    .await?
                }
                BackendChoice::Gaussian => {
                    let seed = frozen
                        .split
                        .manifest
                        .frozen
                        .as_ref()
                        .ok_or_else(|| {
                            brush_recon::ReconError::input("missing frozen seed metadata")
                        })?
                        .rng_seed;
                    run_backend(
                        || GaussianBackend::new(frozen.seed.clone(), device.clone(), seed),
                        &frozen.split,
                        &config,
                    )
                    .await?
                }
            };
            verify_sources(&frozen.split.manifest)?;
            let extension = match record.backend {
                brush_recon::BackendKind::Triangle => "state.json",
                brush_recon::BackendKind::Gaussian => "ply",
            };
            io::write_new(
                &output.join(format!("initial.{extension}")),
                &record.initial_state.payload,
            )?;
            io::write_new(
                &output.join(format!("final.{extension}")),
                &record.final_state.payload,
            )?;
            io::json(
                &output.join("capture-manifest.json"),
                &frozen.split.manifest,
            )?;
            let (revision, dirty) = io::provenance()?;
            io::json(
                &output.join("run.json"),
                &serde_json::json!({"run":record,"manifest":args.manifest,"source_load_seconds":source_load_seconds,"environment":environment,"git_revision":revision,"dirty_patch_sha256":dirty,"renderer_memory_note":"triangle bytes are a conservative reservation bound; Gaussian renderer bytes are not instrumented"}),
            )?;
            println!("{}", output.join("run.json").display());
        }
    }
    Ok(())
}
