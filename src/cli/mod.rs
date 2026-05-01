//! CLI entry point and dispatcher.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.
//! `src/main.rs` is now a thin shim that calls `cli::run()`.

use clap::Parser;

pub(crate) mod analyze_tla;
pub(crate) mod args;
pub(crate) mod list_checkpoints;
pub(crate) mod probe;
pub(crate) mod run_adaptive_branching;
pub(crate) mod run_counter_grid;
pub(crate) mod run_flurm_lifecycle;
pub(crate) mod run_high_branching;
pub(crate) mod run_model;
pub(crate) mod run_tla;
pub(crate) mod shared;

pub(crate) use args::Command;

#[derive(Parser, Debug)]
#[command(name = "tlaplusplus")]
#[command(about = "Prototype scalable runtime for TLA+ model checking", long_about = None)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// Main CLI entry point. Called from `src/main.rs`.
pub fn run() -> anyhow::Result<()> {
    // T11: when built with --features failpoints, initialize the FailScenario
    // so the standard `FAILPOINTS=name=action[;...]` env var configures
    // failpoints for this process. Without this call, externally-set FAILPOINTS
    // is ignored. The teardown is handled implicitly on process exit.
    //
    // Risk #4 of the refactor plan: this binding must outlive the dispatcher,
    // so it lives in `run()` (not `main()`) and the `_` prefix suppresses
    // unused-binding warnings in non-failpoints builds.
    #[cfg(feature = "failpoints")]
    let _failpoint_scenario = fail::FailScenario::setup();

    let cli = Cli::parse();

    match cli.command {
        Command::RunCounterGrid {
            max_x,
            max_y,
            max_sum,
            runtime,
            storage,
            s3,
            cluster,
        } => run_counter_grid::handle(max_x, max_y, max_sum, runtime, storage, s3, cluster),
        Command::RunFlurmLifecycle {
            max_jobs,
            max_time_limit,
            runtime,
            storage,
            s3,
        } => run_flurm_lifecycle::handle(max_jobs, max_time_limit, runtime, storage, s3),
        Command::RunHighBranching {
            max_depth,
            branching_factor,
            runtime,
            storage,
            s3,
        } => run_high_branching::handle(max_depth, branching_factor, runtime, storage, s3),
        Command::RunAdaptiveBranching {
            max_depth,
            min_branching,
            max_branching,
            memory_threshold_pct,
            adjustment_interval_secs,
            runtime,
            storage,
            s3,
        } => run_adaptive_branching::handle(
            max_depth,
            min_branching,
            max_branching,
            memory_threshold_pct,
            adjustment_interval_secs,
            runtime,
            storage,
            s3,
        ),
        Command::AnalyzeTla { module, config } => analyze_tla::handle(module, config),
        Command::RunTla {
            module,
            config,
            fetch_module,
            fetch_config,
            init,
            next,
            allow_deadlock,
            simulate,
            simulate_depth,
            simulate_traces,
            simulate_seed,
            swarm,
            por,
            coverage,
            dump,
            dump_format,
            difftrace,
            runtime,
            storage,
            s3,
            cluster,
        } => run_tla::handle(
            module,
            config,
            fetch_module,
            fetch_config,
            init,
            next,
            allow_deadlock,
            simulate,
            simulate_depth,
            simulate_traces,
            simulate_seed,
            swarm,
            por,
            coverage,
            dump,
            dump_format,
            difftrace,
            runtime,
            storage,
            s3,
            cluster,
        ),
        Command::ListCheckpoints {
            work_dir,
            s3_bucket,
            s3_prefix,
            s3_region,
            validate,
        } => list_checkpoints::handle(work_dir, s3_bucket, s3_prefix, s3_region, validate),
    }
}
