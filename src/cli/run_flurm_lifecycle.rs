//! `run-flurm-lifecycle` subcommand handler.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use crate::models::flurm_job_lifecycle::FlurmJobLifecycleModel;

use super::run_model::run_model_with_s3;
use super::shared::{build_engine_config, print_stats, run_system_checks};

pub(crate) fn handle(
    max_jobs: usize,
    max_time_limit: u16,
    runtime: crate::cli::args::RuntimeArgs,
    storage: crate::cli::args::StorageArgs,
    s3: crate::cli::args::S3Args,
) -> anyhow::Result<()> {
    run_system_checks(runtime.skip_system_checks);
    let model = FlurmJobLifecycleModel::new(max_jobs, max_time_limit);
    let config = build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
    let outcome = run_model_with_s3(model, config, &s3)?;
    print_stats("flurm-job-lifecycle", &outcome.stats);
    if let Some(violation) = outcome.violation {
        println!("violation=true");
        println!("violation_message={}", violation.message);
        println!("violation_state={:?}", violation.state);
    } else {
        println!("violation=false");
    }
    Ok(())
}
