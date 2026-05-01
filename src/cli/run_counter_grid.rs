//! `run-counter-grid` subcommand handler.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use crate::models::counter_grid::CounterGridModel;

use super::run_model::run_model_with_s3;
use super::shared::{build_engine_config, maybe_setup_cluster, print_stats, run_system_checks};

pub(crate) fn handle(
    max_x: u32,
    max_y: u32,
    max_sum: u32,
    runtime: crate::cli::args::RuntimeArgs,
    storage: crate::cli::args::StorageArgs,
    s3: crate::cli::args::S3Args,
    cluster: crate::cli::args::ClusterArgs,
) -> anyhow::Result<()> {
    run_system_checks(runtime.skip_system_checks);
    let model = CounterGridModel::new(max_x, max_y, max_sum);
    let mut config = build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
    // Wire distributed-cluster mode if --cluster-listen was passed.
    // T6: enables cross-node work stealing across the cluster.
    maybe_setup_cluster(&cluster, &mut config)?;
    let outcome = run_model_with_s3(model, config, &s3)?;
    print_stats("counter-grid", &outcome.stats);
    if let Some(violation) = outcome.violation {
        println!("violation=true");
        println!("violation_message={}", violation.message);
        println!("violation_state={:?}", violation.state);
    } else {
        println!("violation=false");
    }
    Ok(())
}
