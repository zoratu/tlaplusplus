//! `run-high-branching` subcommand handler.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use crate::models::high_branching::HighBranchingModel;

use super::run_model::run_model_with_s3;
use super::shared::{build_engine_config, print_stats, run_system_checks};

pub(crate) fn handle(
    max_depth: u32,
    branching_factor: u32,
    runtime: crate::cli::args::RuntimeArgs,
    storage: crate::cli::args::StorageArgs,
    s3: crate::cli::args::S3Args,
) -> anyhow::Result<()> {
    run_system_checks(runtime.skip_system_checks);
    let model = HighBranchingModel::new(max_depth, branching_factor);
    let config = build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
    let outcome = run_model_with_s3(model, config, &s3)?;
    print_stats("high-branching", &outcome.stats);
    if let Some(violation) = outcome.violation {
        println!("violation=true");
        println!("violation_message={}", violation.message);
        println!("violation_state={:?}", violation.state);
    } else {
        println!("violation=false");
    }
    Ok(())
}
