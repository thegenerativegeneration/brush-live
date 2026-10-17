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
    let cli_started = std::time::Instant::now();
    let started_unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| brush_recon::ReconError::input(format!("system clock: {e}")))?
        .as_secs();
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
            let manifest_path = args
                .manifest
                .canonicalize()
                .map_err(|e| brush_recon::ReconError::io(&args.manifest, e))?;
            let frozen = load_frozen(&manifest_path)?;
            let source_load_seconds = loading.elapsed().as_secs_f64();
            let output = io::validate_output(&frozen.split.manifest.source, &args.output)?;
            let (device, environment) = crate::runtime::initialize().await?;
            io::create_output(&output)?;
            let attempt = match args.backend {
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
                        &frozen.seed,
                    )
                    .await
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
                        &frozen.seed,
                    )
                    .await
                }
            };
            let record = match attempt {
                Ok(record) => record,
                Err(error) => {
                    let source_verification_error = verify_sources(&frozen.split.manifest)
                        .err()
                        .map(|e| e.to_string());
                    io::json(
                        &output.join("failure.json"),
                        &serde_json::json!({"error":error.to_string(),"manifest":manifest_path,"source_verification_error":source_verification_error,"environment":environment,"started_unix_s":started_unix_s}),
                    )?;
                    return Err(error);
                }
            };
            verify_sources(&frozen.split.manifest)?;
            crate::report::save_artifacts(&record, &output)?;
            crate::report::copy_frozen_assets(&manifest_path, &frozen.split.manifest, &output)?;
            let (revision, dirty) = io::provenance()?;
            io::json(
                &output.join("run.json"),
                &serde_json::json!({"run":record,"manifest":manifest_path,"source_load_seconds":source_load_seconds,"environment":environment,"git_revision":revision,"dirty_patch_sha256":dirty,"started_unix_s":started_unix_s,"cli_seconds_before_final_report":cli_started.elapsed().as_secs_f64(),"renderer_memory_note":"triangle bytes are a conservative reservation bound; Gaussian renderer bytes are not instrumented"}),
            )?;
            println!("{}", output.join("run.json").display());
        }
    }
    Ok(())
}
