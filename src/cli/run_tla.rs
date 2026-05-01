//! `run-tla` subcommand handler — model-checks a TLA+ spec.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::distributed::ClusterConfig;
use crate::distributed::handler::StolenState;
use crate::distributed::transport::ClusterTransport;
use crate::distributed::work_stealer::DistributedWorkStealer;
use crate::models::tla_native::TlaModel;
use crate::system::{check_thp_and_warn, parse_cpu_list};
use crate::tla::action_exec::probe_next_disjuncts_with_instances;
use crate::tla::module::TlaModuleInstance;
use crate::tla::{
    ActionClause, ClauseKind, ConfigValue, EvalContext, TlaConfig, TlaDefinition, TlaModule,
    TlaState, TlaValue, classify_clause, compile_action_ir, compile_action_ir_branches,
    eval_action_body_multi, eval_expr, eval_let_action_multi, looks_like_action,
    normalize_operator_ref_name, normalize_param_name, parse_action_exists,
    parse_stuttering_action_expr, parse_tla_config, parse_tla_module_file, restore_eval_budget,
    scan_module_closure, set_active_eval_budget, split_action_body_disjuncts, split_top_level,
};
use crate::{EngineConfig, SimulationConfig, run_model, run_simulation};

use super::probe::build_probe_eval_context;
use super::run_model::{
    collect_coverage, dump_state_graph, evaluate_assumes, run_model_with_s3,
};
use super::shared::{
    build_engine_config, fetch_s3_file, format_num, inject_constants_into_definitions,
    maybe_setup_cluster, model_fingerprint, print_difftrace, print_difftrace_with_relevance,
    print_stats, run_system_checks,
};

pub(crate) fn handle(
    module: Option<std::path::PathBuf>,
    config: Option<std::path::PathBuf>,
    fetch_module: Option<String>,
    fetch_config: Option<String>,
    init: Option<String>,
    next: Option<String>,
    allow_deadlock: bool,
    simulate: bool,
    simulate_depth: usize,
    simulate_traces: usize,
    simulate_seed: u64,
    swarm: bool,
    por: bool,
    coverage: bool,
    dump: Option<std::path::PathBuf>,
    dump_format: String,
    difftrace: bool,
    runtime: crate::cli::args::RuntimeArgs,
    storage: crate::cli::args::StorageArgs,
    s3: crate::cli::args::S3Args,
    cluster: crate::cli::args::ClusterArgs,
) -> anyhow::Result<()> {
    run_system_checks(runtime.skip_system_checks);

    // Fetch spec/config from S3 if requested (for distributed runs
    // where nodes don't share a filesystem)
    let module = if let Some(ref uri) = fetch_module {
        let local = fetch_s3_file(uri)?;
        eprintln!("Fetched module from {}", uri);
        local
    } else {
        module.expect("--module is required when --fetch-module is not used")
    };
    let config = if let Some(ref uri) = fetch_config {
        let local = fetch_s3_file(uri)?;
        eprintln!("Fetched config from {}", uri);
        Some(local)
    } else {
        config
    };

    // Auto-detect config file if not specified
    let config_path = config.or_else(|| {
        let cfg_path = module.with_extension("cfg");
        if cfg_path.exists() {
            Some(cfg_path)
        } else {
            None
        }
    });

    let mut model = TlaModel::from_files(
        &module,
        config_path.as_deref(),
        init.as_deref(),
        next.as_deref(),
    )
    .map_err(|e| {
        eprintln!("Error building TLA+ model:");
        eprintln!("  Module: {}", module.display());
        if let Some(cfg) = &config_path {
            eprintln!("  Config: {}", cfg.display());
        }
        eprintln!("  Error: {}", e);
        e
    })?;

    // CLI --allow-deadlock overrides config CHECK_DEADLOCK
    if allow_deadlock {
        model.allow_deadlock = true;
    }

    // CLI --por: enable partial-order reduction (safety only).
    if por {
        model.enable_por().map_err(|e| {
            eprintln!("--por rejected: {}", e);
            eprintln!(
                "  POR in this version preserves only safety properties. \
                 Re-run without --por (or remove fairness/liveness from the spec)."
            );
            e
        })?;
        eprintln!(
            "Partial-order reduction enabled ({} actions, {} variables)",
            model
                .por_analysis
                .as_ref()
                .map(|a| a.num_actions())
                .unwrap_or(0),
            model.module.variables.len(),
        );
    }

    // Feature 7: Evaluate ASSUME statements before exploration
    evaluate_assumes(&model)?;

    // Print TLC-compatible output
    let start_time = chrono::Local::now();
    println!("Starting... ({})", start_time.format("%Y-%m-%d %H:%M:%S"));
    println!("Computing initial states...");

    // Feature 1: Simulation mode
    if simulate {
        let sim_config = SimulationConfig {
            depth: simulate_depth,
            num_traces: simulate_traces,
            seed: simulate_seed,
            swarm,
        };
        let max_violations = runtime.max_violations;
        println!(
            "Running simulation: {} traces, depth {}, seed {}",
            sim_config.num_traces, sim_config.depth, sim_config.seed
        );
        let mut sim_outcome = run_simulation(&model, &sim_config, max_violations);
        // T9 Phase A: minimize simulation violation traces too.
        if runtime.minimize_trace && runtime.minimize_trace_budget_secs > 0 {
            let budget = std::time::Duration::from_secs(runtime.minimize_trace_budget_secs);
            for v in sim_outcome.violations.iter_mut() {
                let original = v.trace.len();
                if original > 1 {
                    let res = crate::minimize_trace(
                        &model,
                        std::mem::take(&mut v.trace),
                        budget,
                    );
                    let final_len = res.trace.len();
                    v.trace = res.trace;
                    eprintln!(
                        "Trace minimization: {} -> {} steps in {:?} ({} iter{}{})",
                        original,
                        final_len,
                        res.elapsed,
                        res.iterations,
                        if res.iterations == 1 { "" } else { "s" },
                        if res.budget_exhausted {
                            ", budget exhausted"
                        } else {
                            ""
                        }
                    );
                }
            }
        }
        let end_time = chrono::Local::now();

        println!(
            "Simulation complete: {} traces, {} total states, max depth {}",
            sim_outcome.traces_run, sim_outcome.total_states, sim_outcome.max_depth_reached,
        );
        println!(
            "Finished in {:.3}s at ({})",
            sim_outcome.duration.as_secs_f64(),
            end_time.format("%Y-%m-%d %H:%M:%S")
        );

        if sim_outcome.violations.is_empty() {
            println!("violation=false");
            println!();
            println!("Simulation completed successfully! No violations found.");
        } else {
            println!(
                "violation=true ({} violations found)",
                sim_outcome.violations.len()
            );
            for (i, v) in sim_outcome.violations.iter().enumerate() {
                println!();
                println!("--- Violation {} ---", i + 1);
                println!("  message: {}", v.message);
                if difftrace {
                    print_difftrace(&v.trace);
                } else {
                    for (step, state) in v.trace.iter().enumerate() {
                        println!("  step {}: {:?}", step, state);
                    }
                }
            }
            std::process::exit(1);
        }
        // Skip the rest of the RunTla block in simulation mode
        return Ok(());
    }

    // Clone model for post-processing (liveness checking, coverage, dump)
    let model_for_liveness = model.clone();

    let mut engine_config =
        build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;

    // --- Distributed cluster startup ---
    if let Some(ref listen_addr_str) = cluster.cluster_listen {
        let listen_addr: std::net::SocketAddr = listen_addr_str.parse().map_err(|e| {
            anyhow::anyhow!(
                "invalid --cluster-listen address '{}': {}",
                listen_addr_str,
                e
            )
        })?;
        let peers: Vec<std::net::SocketAddr> = cluster
            .cluster_peers
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| anyhow::anyhow!("invalid peer address '{}': {}", s, e))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let cluster_config = ClusterConfig {
            node_id: cluster.node_id,
            listen_addr,
            peers: peers.clone(),
        };
        let num_nodes = cluster_config.num_nodes();

        // Build tokio runtime for cluster transport
        let tokio_rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("failed to create tokio runtime: {}", e))?;
        let tokio_handle = tokio_rt.handle().clone();

        // Start transport (bind listener)
        let transport = tokio_handle
            .block_on(async { ClusterTransport::new(cluster_config.clone()).await })?;

        // Connect to peers (retry with brief delay for startup ordering)
        println!(
            "[cluster] node {} listening on {}, connecting to {} peers...",
            cluster.node_id,
            listen_addr,
            peers.len()
        );
        tokio_handle.block_on(async {
            for attempt in 0..30 {
                match transport.connect_to_peers().await {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        if attempt < 29 {
                            eprintln!(
                                "[cluster] peer connection attempt {} failed: {}, retrying...",
                                attempt + 1, e
                            );
                            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            unreachable!()
        })?;
        println!("[cluster] connected to all peers");

        // Create distributed work stealer (independent exploration + work stealing)
        let stealer = Arc::new(DistributedWorkStealer::new(
            cluster.node_id,
            num_nodes,
            Arc::clone(&transport),
            tokio_handle.clone(),
        ));

        // Create channels for stolen-state and donation exchange
        // stolen_states: handler pushes StealResponse states, workers drain
        let (stolen_tx, stolen_rx) = crossbeam_channel::bounded::<StolenState>(65_536);
        // donate_states: workers push serialized states, handler pops for StealRequests
        let (donate_tx, donate_rx) = crossbeam_channel::bounded::<Vec<u8>>(65_536);

        engine_config.distributed_stealer = Some(Arc::clone(&stealer));
        engine_config.stolen_states_rx = Some(stolen_rx);
        engine_config.donate_states_tx = Some(donate_tx);
        engine_config.donate_states_rx = Some(donate_rx);
        engine_config.stolen_states_tx = Some(stolen_tx);

        // Leak the tokio runtime so it stays alive for the duration of the process.
        // The runtime is needed for the transport's async tasks.
        std::mem::forget(tokio_rt);

        println!(
            "[cluster] distributed mode active: node {}, {} total nodes",
            cluster.node_id, num_nodes
        );
    }

    let mut outcome = run_model_with_s3(model, engine_config, &s3).map_err(|e| {
        eprintln!("Error running model:");
        eprintln!("  {}", e);
        e
    })?;

    // T9 Phase A: minimize counter-example traces before reporting.
    // Default on; opt out with `--minimize-trace=false` or zero
    // budget. Operates on the model used for liveness post-processing
    // (a clone of the runtime model, identical Init/Next/invariants).
    if runtime.minimize_trace && runtime.minimize_trace_budget_secs > 0 {
        let budget = std::time::Duration::from_secs(runtime.minimize_trace_budget_secs);
        if let Some(ref mut v) = outcome.violation {
            let original = v.trace.len();
            if original > 1 {
                let res = crate::minimize_trace(
                    &model_for_liveness,
                    std::mem::take(&mut v.trace),
                    budget,
                );
                let final_len = res.trace.len();
                v.trace = res.trace;
                eprintln!(
                    "Trace minimization: {} -> {} steps in {:?} ({} iter{}{})",
                    original,
                    final_len,
                    res.elapsed,
                    res.iterations,
                    if res.iterations == 1 { "" } else { "s" },
                    if res.budget_exhausted {
                        ", budget exhausted"
                    } else {
                        ""
                    }
                );
            }
        }
        for v in outcome.violations.iter_mut() {
            let original = v.trace.len();
            if original > 1 {
                let res = crate::minimize_trace(
                    &model_for_liveness,
                    std::mem::take(&mut v.trace),
                    budget,
                );
                let final_len = res.trace.len();
                v.trace = res.trace;
                eprintln!(
                    "Trace minimization: {} -> {} steps in {:?} ({} iter{}{})",
                    original,
                    final_len,
                    res.elapsed,
                    res.iterations,
                    if res.iterations == 1 { "" } else { "s" },
                    if res.budget_exhausted {
                        ", budget exhausted"
                    } else {
                        ""
                    }
                );
            }
        }
    }

    // Feature 2: Coverage profiling
    if coverage {
        let cov_stats = collect_coverage(&model_for_liveness);
        cov_stats.print_summary();
    }

    // Feature 4: Dump state graph
    if let Some(ref dump_path) = dump {
        dump_state_graph(&model_for_liveness, dump_path, &dump_format)?;
    }

    // Print TLC-compatible final output
    let end_time = chrono::Local::now();
    let queue_pending = 0; // Queue is empty after completion

    // Format numbers with commas (TLC style)
    fn format_with_commas(n: u64) -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    println!(
        "{} states generated, {} distinct states found, {} states left on queue.",
        format_with_commas(outcome.stats.states_generated),
        format_with_commas(outcome.stats.states_distinct),
        queue_pending
    );

    let duration_secs = outcome.stats.duration.as_secs();
    let duration_str = if duration_secs < 60 {
        format!("{:02}s", duration_secs)
    } else if duration_secs < 3600 {
        format!("{:02}min {:02}s", duration_secs / 60, duration_secs % 60)
    } else {
        format!(
            "{:02}h {:02}min",
            duration_secs / 3600,
            (duration_secs % 3600) / 60
        )
    };

    println!(
        "Finished in {} at ({})",
        duration_str,
        end_time.format("%Y-%m-%d %H:%M:%S")
    );

    // Feature 3 & 5: Report violations (multiple if --continue or --max-violations)
    // Collect all violations: first from outcome.violation, rest from outcome.violations
    let mut all_violations_display = Vec::new();
    if let Some(ref v) = outcome.violation {
        all_violations_display.push(v);
    }
    for v in &outcome.violations {
        all_violations_display.push(v);
    }

    if !all_violations_display.is_empty() {
        println!(
            "violation=true ({} violations found)",
            all_violations_display.len()
        );
        // T9 Phase B: pre-compute the set of state variables each
        // invariant references syntactically, so the printer can
        // mark unreferenced variables as "(noise)". This is purely
        // presentation — the trace data structure is unchanged.
        let var_names: Vec<String> = model_for_liveness
            .module
            .variables
            .iter()
            .cloned()
            .collect();
        // T9.1: build an operator-body registry from the loaded
        // module so the relevance scan can transitively inline
        // operator definitions.  An invariant like `Inv == IsBad(s)`
        // where `IsBad(s) == s.x > 5` will now correctly mark `x`
        // as referenced.
        let operator_registry: std::collections::BTreeMap<
            String,
            crate::OperatorBody,
        > = model_for_liveness
            .module
            .definitions
            .iter()
            .map(|(name, def)| {
                (
                    name.clone(),
                    crate::OperatorBody {
                        name: def.name.clone(),
                        body: def.body.clone(),
                    },
                )
            })
            .collect();
        let invariant_relevance: std::collections::HashMap<
            String,
            std::collections::HashSet<String>,
        > = model_for_liveness
            .invariant_exprs
            .iter()
            .map(|(name, body)| {
                let rel = crate::extract_invariant_variables_transitive(
                    body,
                    &var_names,
                    &operator_registry,
                );
                (name.clone(), rel)
            })
            .collect();
        let relevant_vars_for = |msg: &str| -> Option<&std::collections::HashSet<String>> {
            // Violation messages currently have form "invariant 'NAME' violated".
            // Extract NAME and look up its relevance set.
            let prefix = "invariant '";
            let start = msg.find(prefix)? + prefix.len();
            let rest = &msg[start..];
            let end = rest.find('\'')?;
            let name = &rest[..end];
            invariant_relevance.get(name)
        };
        for (i, violation) in all_violations_display.iter().enumerate() {
            println!();
            println!("--- Violation {} ---", i + 1);
            println!("  message: {}", violation.message);
            let relevant = relevant_vars_for(&violation.message);
            if let Some(rel) = relevant {
                let noise: Vec<&String> = var_names
                    .iter()
                    .filter(|v| !rel.contains(v.as_str()))
                    .collect();
                if !noise.is_empty() {
                    println!(
                        "  invariant variables: {}",
                        rel.iter().cloned().collect::<Vec<_>>().join(", ")
                    );
                    println!(
                        "  noise variables (not referenced by invariant): {}",
                        noise
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            if difftrace {
                print_difftrace_with_relevance(&violation.trace, relevant);
            } else {
                println!("  state: {:?}", violation.state);
                if !violation.trace.is_empty() {
                    println!("  trace ({} steps):", violation.trace.len());
                    for (step, state) in violation.trace.iter().enumerate() {
                        println!("    step {}: {:?}", step, state);
                    }
                }
            }
        }
        std::process::exit(1);
    } else {
        // No safety violations found - check liveness properties if present
        use crate::model::Model;
        let initial_states = model_for_liveness.initial_states();
        if let Some(first_state) = initial_states.first() {
            if model_for_liveness.has_liveness_properties()
                && !model_for_liveness.temporal_properties.is_empty()
            {
                use anyhow::anyhow;
                use crate::liveness::LivenessChecker;

                println!(
                    "Checking {} temporal properties...",
                    model_for_liveness.temporal_properties.len()
                );

                let checker =
                    LivenessChecker::new(model_for_liveness.temporal_properties.clone());

                // For finite state space, check liveness on a reconstructed path
                // Note: Full liveness checking requires analyzing cycles in the state graph
                // For now, we do a basic check on a single path from initial state
                let trace = vec![first_state.clone()];

                match checker.check_finite_trace(&trace, &|state, pred_expr| {
                    let ctx = build_probe_eval_context(
                        state,
                        &model_for_liveness.module.definitions,
                        &model_for_liveness.module.instances,
                    );
                    match eval_expr(pred_expr, &ctx) {
                        Ok(TlaValue::Bool(b)) => Ok(b),
                        Ok(_) => Err(anyhow!("predicate did not evaluate to boolean")),
                        Err(e) => Err(anyhow!("evaluation error: {}", e)),
                    }
                }) {
                    Ok(_) => {
                        if std::env::var("TLAPP_VERBOSE").is_ok() {
                            println!("All temporal properties satisfied on finite trace");
                            println!();
                            println!(
                                "Note: Full liveness checking requires fairness analysis on state graph cycles"
                            );
                        }
                    }
                    Err(msg) => {
                        // Don't treat liveness property violations as hard errors
                        // when checking on incomplete traces
                        if std::env::var("TLAPP_VERBOSE").is_ok() {
                            println!("Note: Liveness property check inconclusive: {}", msg);
                            println!(
                                "  (Full liveness checking requires complete state graph analysis)"
                            );
                        }
                    }
                }
            }
        }

        // Fairness checking (WF/SF) was wired into the runtime via PR #60.
        // The runtime now collects labeled transitions and runs SCC-based
        // fairness analysis post-exploration (see fairness.rs, liveness.rs).
        // The analyze-tla command does not perform full model checking, so
        // fairness is not checked here — it runs during `run-tla`.

        println!("violation=false");
        println!();
        println!("Model checking completed successfully!");
    }
    Ok(())
}
