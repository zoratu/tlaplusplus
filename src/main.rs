use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use tlaplusplus::models::counter_grid::CounterGridModel;
use tlaplusplus::models::flurm_job_lifecycle::FlurmJobLifecycleModel;
use tlaplusplus::models::high_branching::HighBranchingModel;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::system::parse_cpu_list;
use tlaplusplus::tla::{
    ActionClause, ClauseKind, ConfigValue, EvalContext, TlaState, TlaValue, classify_clause,
    compile_action_ir, eval_expr, looks_like_action, parse_tla_config, parse_tla_module_file,
    probe_next_disjuncts, scan_module_closure, split_top_level,
};
use tlaplusplus::{EngineConfig, run_model};

#[derive(Parser, Debug)]
#[command(name = "tlaplusplus")]
#[command(about = "Prototype scalable runtime for TLA+ model checking", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone, Debug)]
struct RuntimeArgs {
    #[arg(
        long,
        default_value_t = 0,
        help = "Worker threads (0 = auto, cgroup-aware)"
    )]
    workers: usize,
    #[arg(
        long,
        help = "CPU IDs/ranges, e.g. 2-127 or 2-63,96-127. Intersected with cgroup cpuset."
    )]
    core_ids: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    enforce_cgroups: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    numa_pinning: bool,
    #[arg(
        long,
        help = "Hard memory ceiling (bytes) used to tune runtime caches/queues"
    )]
    memory_max_bytes: Option<u64>,
    #[arg(
        long,
        default_value_t = 256,
        help = "Estimated bytes per in-memory state"
    )]
    estimated_state_bytes: usize,
    #[arg(long, default_value = "./.tlapp")]
    work_dir: std::path::PathBuf,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    clean_work_dir: bool,
    #[arg(
        long,
        default_value_t = false,
        action = clap::ArgAction::Set,
        help = "Resume from existing checkpoint/work dir"
    )]
    resume: bool,
    #[arg(long, default_value_t = 30)]
    checkpoint_interval_secs: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    checkpoint_on_exit: bool,
    #[arg(long, default_value_t = 1)]
    poll_sleep_ms: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stop_on_violation: bool,
}

#[derive(Args, Clone, Debug)]
struct StorageArgs {
    #[arg(long, default_value_t = 64)]
    fp_shards: usize,
    #[arg(long, default_value_t = 100_000_000)]
    fp_expected_items: usize,
    #[arg(long, default_value_t = 0.01)]
    fp_fpr: f64,
    #[arg(long, default_value_t = 1_000_000)]
    fp_hot_entries: usize,
    #[arg(long, default_value_t = 1_073_741_824)]
    fp_cache_bytes: u64,
    #[arg(long, default_value_t = 10_000)]
    fp_flush_every_ms: u64,
    #[arg(long, default_value_t = 512)]
    fp_batch_size: usize,
    #[arg(long, default_value_t = 5_000_000)]
    queue_inmem_limit: usize,
    #[arg(long, default_value_t = 50_000)]
    queue_spill_batch: usize,
    #[arg(long, default_value_t = 128)]
    queue_spill_channel_bound: usize,
}

#[derive(Subcommand, Debug)]
enum Command {
    RunCounterGrid {
        #[arg(long, default_value_t = 5000)]
        max_x: u32,
        #[arg(long, default_value_t = 5000)]
        max_y: u32,
        #[arg(long, default_value_t = 10000)]
        max_sum: u32,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
    },
    RunFlurmLifecycle {
        #[arg(long, default_value_t = 3)]
        max_jobs: usize,
        #[arg(long, default_value_t = 3)]
        max_time_limit: u16,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
    },
    RunHighBranching {
        #[arg(long, default_value_t = 8)]
        max_depth: u32,
        #[arg(long, default_value_t = 16)]
        branching_factor: u32,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
    },
    AnalyzeTla {
        #[arg(long)]
        module: std::path::PathBuf,
        #[arg(long)]
        config: Option<std::path::PathBuf>,
    },
    RunTla {
        #[arg(long)]
        module: std::path::PathBuf,
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        #[arg(long)]
        init: Option<String>,
        #[arg(long)]
        next: Option<String>,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
    },
}

fn build_engine_config(
    runtime: &RuntimeArgs,
    storage: &StorageArgs,
) -> anyhow::Result<EngineConfig> {
    let core_ids = match &runtime.core_ids {
        Some(spec) => Some(parse_cpu_list(spec)?),
        None => None,
    };

    let fp_flush_every_ms = if storage.fp_flush_every_ms == 0 {
        None
    } else {
        Some(storage.fp_flush_every_ms)
    };

    Ok(EngineConfig {
        workers: runtime.workers,
        core_ids,
        enforce_cgroups: runtime.enforce_cgroups,
        numa_pinning: runtime.numa_pinning,
        memory_max_bytes: runtime.memory_max_bytes,
        estimated_state_bytes: runtime.estimated_state_bytes,
        work_dir: runtime.work_dir.clone(),
        clean_work_dir: runtime.clean_work_dir,
        resume_from_checkpoint: runtime.resume,
        checkpoint_interval_secs: runtime.checkpoint_interval_secs,
        checkpoint_on_exit: runtime.checkpoint_on_exit,
        poll_sleep_ms: runtime.poll_sleep_ms,
        stop_on_violation: runtime.stop_on_violation,
        fp_shards: storage.fp_shards,
        fp_expected_items: storage.fp_expected_items,
        fp_false_positive_rate: storage.fp_fpr,
        fp_hot_entries_per_shard: storage.fp_hot_entries,
        fp_cache_capacity_bytes: storage.fp_cache_bytes,
        fp_flush_every_ms,
        fp_batch_size: storage.fp_batch_size,
        queue_inmem_limit: storage.queue_inmem_limit,
        queue_spill_batch: storage.queue_spill_batch,
        queue_spill_channel_bound: storage.queue_spill_channel_bound,
    })
}

fn print_stats(model_name: &str, stats: &tlaplusplus::RunStats) {
    let duration_sec = stats.duration.as_secs_f64().max(0.000_001);
    println!("model={}", model_name);
    println!("duration_sec={:.3}", duration_sec);
    println!("states_generated={}", stats.states_generated);
    println!("states_processed={}", stats.states_processed);
    println!("states_distinct={}", stats.states_distinct);
    println!("duplicates={}", stats.duplicates);
    println!(
        "throughput_states_per_sec={:.2}",
        (stats.states_processed as f64) / duration_sec
    );
    println!("checkpoints={}", stats.checkpoints);
    println!("configured_workers={}", stats.configured_workers);
    println!("actual_workers={}", stats.actual_workers);
    println!("allowed_cpu_count={}", stats.allowed_cpu_count);
    println!(
        "cgroup_cpuset_cores={}",
        stats
            .cgroup_cpuset_cores
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "cgroup_quota_cores={}",
        stats
            .cgroup_quota_cores
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("numa_nodes_used={}", stats.numa_nodes_used);
    println!(
        "effective_memory_max_bytes={}",
        stats
            .effective_memory_max_bytes
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("resumed_from_checkpoint={}", stats.resumed_from_checkpoint);
    println!("fingerprints.inmem=true",);
    println!(
        "fingerprints.batch_calls={}",
        stats.fingerprints.batch_calls
    );
    println!(
        "fingerprints.batch_items={}",
        stats.fingerprints.batch_items
    );
    println!("queue.spilled_items={}", stats.queue.spilled_items);
    println!("queue.spill_batches={}", stats.queue.spill_batches);
    println!("queue.loaded_segments={}", stats.queue.loaded_segments);
    println!("queue.loaded_items={}", stats.queue.loaded_items);
    println!("queue.max_inmem_len={}", stats.queue.max_inmem_len);
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::RunCounterGrid {
            max_x,
            max_y,
            max_sum,
            runtime,
            storage,
        } => {
            let model = CounterGridModel::new(max_x, max_y, max_sum);
            let config = build_engine_config(&runtime, &storage)?;
            let outcome = run_model(model, config)?;
            print_stats("counter-grid", &outcome.stats);
            if let Some(violation) = outcome.violation {
                println!("violation=true");
                println!("violation_message={}", violation.message);
                println!("violation_state={:?}", violation.state);
            } else {
                println!("violation=false");
            }
        }
        Command::RunFlurmLifecycle {
            max_jobs,
            max_time_limit,
            runtime,
            storage,
        } => {
            let model = FlurmJobLifecycleModel::new(max_jobs, max_time_limit);
            let config = build_engine_config(&runtime, &storage)?;
            let outcome = run_model(model, config)?;
            print_stats("flurm-job-lifecycle", &outcome.stats);
            if let Some(violation) = outcome.violation {
                println!("violation=true");
                println!("violation_message={}", violation.message);
                println!("violation_state={:?}", violation.state);
            } else {
                println!("violation=false");
            }
        }
        Command::RunHighBranching {
            max_depth,
            branching_factor,
            runtime,
            storage,
        } => {
            let model = HighBranchingModel::new(max_depth, branching_factor);
            let config = build_engine_config(&runtime, &storage)?;
            let outcome = run_model(model, config)?;
            print_stats("high-branching", &outcome.stats);
            if let Some(violation) = outcome.violation {
                println!("violation=true");
                println!("violation_message={}", violation.message);
                println!("violation_state={:?}", violation.state);
            } else {
                println!("violation=false");
            }
        }
        Command::AnalyzeTla { module, config } => {
            let parsed_module = parse_tla_module_file(&module)?;
            let scan = scan_module_closure(&module)?;
            let parsed_cfg = match config.as_ref() {
                Some(cfg_path) => {
                    let raw = std::fs::read_to_string(cfg_path)?;
                    Some(parse_tla_config(&raw)?)
                }
                None => None,
            };
            println!("entry_module={}", module.display());
            println!(
                "parsed_module_name={} parsed_constants={} parsed_variables={} parsed_definitions={}",
                parsed_module.name,
                parsed_module.constants.len(),
                parsed_module.variables.len(),
                parsed_module.definitions.len()
            );
            println!(
                "parsed_has_init={} parsed_has_next={} parsed_has_spec={}",
                parsed_module.definitions.contains_key("Init"),
                parsed_module.definitions.contains_key("Next"),
                parsed_module.definitions.contains_key("Spec")
            );
            if let Some(next_def) = parsed_module.definitions.get("Next") {
                let disjuncts = split_top_level(&next_def.body, "\\/");
                println!("next_top_level_disjuncts={}", disjuncts.len());

                let mut max_conjuncts = 0usize;
                let mut total_primed_assignments = 0usize;
                let mut total_unchanged_clauses = 0usize;
                for branch in &disjuncts {
                    let conjuncts = split_top_level(branch, "/\\");
                    max_conjuncts = max_conjuncts.max(conjuncts.len());
                    for clause in conjuncts {
                        match classify_clause(&clause) {
                            ClauseKind::PrimedAssignment { .. } => {
                                total_primed_assignments += 1;
                            }
                            ClauseKind::Unchanged { .. } => {
                                total_unchanged_clauses += 1;
                            }
                            _ => {}
                        }
                    }
                }
                println!("next_max_top_level_conjuncts={}", max_conjuncts);
                println!(
                    "next_detected_primed_assignments={}",
                    total_primed_assignments
                );
                println!(
                    "next_detected_unchanged_clauses={}",
                    total_unchanged_clauses
                );
            }

            let mut action_defs = 0usize;
            let mut action_conjunct_max = 0usize;
            let mut action_primed_assignments = 0usize;
            let mut action_unchanged_clauses = 0usize;
            let mut action_guard_clauses = 0usize;
            for def in parsed_module.definitions.values() {
                if !looks_like_action(def) {
                    continue;
                }
                action_defs += 1;
                let ir = compile_action_ir(def);
                action_conjunct_max = action_conjunct_max.max(ir.clauses.len());
                for clause in ir.clauses {
                    match clause {
                        ActionClause::PrimedAssignment { .. } => action_primed_assignments += 1,
                        ActionClause::Unchanged { .. } => action_unchanged_clauses += 1,
                        ActionClause::Guard { .. } => action_guard_clauses += 1,
                    }
                }
            }
            println!("action_like_definitions={}", action_defs);
            println!("action_conjunct_max={}", action_conjunct_max);
            println!(
                "action_detected_primed_assignments={}",
                action_primed_assignments
            );
            println!(
                "action_detected_unchanged_clauses={}",
                action_unchanged_clauses
            );
            println!("action_detected_guard_clauses={}", action_guard_clauses);

            let mut probe_state: TlaState = BTreeMap::new();
            if let Some(cfg) = parsed_cfg.as_ref() {
                for (k, v) in &cfg.constants {
                    if let Some(tv) = config_value_to_tla(v) {
                        probe_state.insert(k.clone(), tv);
                    }
                }
            }
            let mut probe_init_seeded = 0usize;
            let mut probe_init_unresolved = 0usize;
            let mut probe_init_unresolved_vars: Vec<String> = Vec::new();
            if let Some(init_def) = parsed_module.definitions.get("Init") {
                let mut pending = Vec::new();
                for clause in split_top_level(&init_def.body, "/\\") {
                    if let ClauseKind::UnprimedEquality { var, expr } = classify_clause(&clause) {
                        pending.push((var, expr));
                    }
                }

                for _ in 0..pending.len().saturating_add(1) {
                    if pending.is_empty() {
                        break;
                    }
                    let mut progress = false;
                    let mut next_pending = Vec::new();
                    for (var, expr) in pending {
                        let ctx =
                            EvalContext::with_definitions(&probe_state, &parsed_module.definitions);
                        match eval_expr(&expr, &ctx) {
                            Ok(value) => {
                                probe_state.insert(var, value);
                                probe_init_seeded += 1;
                                progress = true;
                            }
                            Err(_) => next_pending.push((var, expr)),
                        }
                    }

                    if !progress {
                        probe_init_unresolved = next_pending.len();
                        probe_init_unresolved_vars =
                            next_pending.iter().map(|(var, _)| var.clone()).collect();
                        break;
                    }
                    pending = next_pending;
                }
            }
            for var in &parsed_module.variables {
                probe_state.entry(var.clone()).or_insert(TlaValue::Int(0));
            }
            println!("probe_init_seeded={probe_init_seeded}");
            println!("probe_init_unresolved={probe_init_unresolved}");
            if !probe_init_unresolved_vars.is_empty() {
                println!(
                    "probe_init_unresolved_vars={}",
                    probe_init_unresolved_vars.join(",")
                );
            }
            println!("modules_scanned={}", scan.modules.len());
            println!("operators_total={}", scan.operator_names.len());
            println!("features_total={}", scan.combined_features.len());
            println!("native_frontend.module_parse=true");
            println!("native_frontend.cfg_parse={}", config.is_some());
            println!("native_frontend.value_domain=true");
            let mut action_eval_ready = false;
            if let Some(next_def) = parsed_module.definitions.get("Next") {
                let next_probe =
                    probe_next_disjuncts(&next_def.body, &parsed_module.definitions, &probe_state);
                println!("next_branch_probe_total={}", next_probe.total_disjuncts);
                println!(
                    "next_branch_probe_supported={}",
                    next_probe.supported_disjuncts
                );
                println!(
                    "next_branch_probe_generated_successors={}",
                    next_probe.generated_successors
                );
                println!(
                    "next_branch_probe_unsupported={}",
                    next_probe
                        .total_disjuncts
                        .saturating_sub(next_probe.supported_disjuncts)
                );
                action_eval_ready = next_probe.total_disjuncts > 0
                    && next_probe.supported_disjuncts == next_probe.total_disjuncts;
                if !next_probe.failures.is_empty() {
                    println!("--- next_branch_probe_errors ---");
                    for (idx, (err, count)) in next_probe.failures.iter().enumerate() {
                        if idx >= 10 {
                            break;
                        }
                        println!("error_count={count} error={err}");
                    }
                }
            }
            let mut expr_total = 0usize;
            let mut expr_ok = 0usize;
            let mut expr_errors: BTreeMap<String, u64> = BTreeMap::new();
            let mut expr_error_examples: BTreeMap<String, String> = BTreeMap::new();
            for def in parsed_module.definitions.values() {
                if !looks_like_action(def) {
                    continue;
                }
                let ir = compile_action_ir(def);
                let mut ctx =
                    EvalContext::with_definitions(&probe_state, &parsed_module.definitions);
                for param in &ir.params {
                    let sample = sample_param_value(param, &probe_state);
                    ctx.locals.insert(param.clone(), sample);
                }
                for clause in ir.clauses {
                    match clause {
                        ActionClause::Guard { expr }
                        | ActionClause::PrimedAssignment { expr, .. } => {
                            expr_total += 1;
                            match eval_expr(&expr, &ctx) {
                                Ok(_) => expr_ok += 1,
                                Err(err) => {
                                    let key = err.to_string();
                                    *expr_errors.entry(key).or_insert(0) += 1;
                                    expr_error_examples.entry(err.to_string()).or_insert_with(
                                        || {
                                            format!(
                                                "def={} expr={}",
                                                def.name,
                                                expr.replace('\n', " ")
                                            )
                                        },
                                    );
                                }
                            }
                        }
                        ActionClause::Unchanged { .. } => {}
                    }
                }
            }
            let expr_eval_ready = expr_total > 0 && expr_ok == expr_total;
            println!("native_frontend.expr_eval={expr_eval_ready}");
            println!("native_frontend.action_eval={action_eval_ready}");
            println!(
                "native_frontend.unsupported_feature_count={}",
                scan.combined_features.len()
            );
            println!("expr_probe_total={expr_total}");
            println!("expr_probe_ok={expr_ok}");
            println!("expr_probe_failed={}", expr_total.saturating_sub(expr_ok));
            if !expr_errors.is_empty() {
                println!("--- expr_probe_errors ---");
                for (idx, (err, count)) in expr_errors.iter().enumerate() {
                    if idx >= 20 {
                        break;
                    }
                    println!("error_count={count} error={err}");
                    if let Some(example) = expr_error_examples.get(err) {
                        println!("error_example={example}");
                    }
                }
            }
            println!("--- modules ---");
            for m in &scan.modules {
                println!(
                    "module={} path={} operators={} features={}",
                    if m.module_name.is_empty() {
                        "unknown"
                    } else {
                        m.module_name.as_str()
                    },
                    m.path.display(),
                    m.operators.len(),
                    m.features.len()
                );
            }
            println!("--- feature_counts ---");
            for (feature, count) in &scan.combined_features {
                println!("{feature}={count}");
            }
            println!("--- operators ---");
            for op in &scan.operator_names {
                println!("{op}");
            }

            if let Some(cfg) = parsed_cfg.as_ref() {
                println!("--- config ---");
                println!(
                    "specification={}",
                    cfg.specification.as_deref().unwrap_or("none")
                );
                println!("init={}", cfg.init.as_deref().unwrap_or("none"));
                println!("next={}", cfg.next.as_deref().unwrap_or("none"));
                println!("symmetry={}", cfg.symmetry.as_deref().unwrap_or("none"));
                println!(
                    "check_deadlock={}",
                    cfg.check_deadlock
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "none".to_string())
                );
                println!("constants_count={}", cfg.constants.len());
                println!("invariants_count={}", cfg.invariants.len());
                println!("properties_count={}", cfg.properties.len());
                println!("constraints_count={}", cfg.constraints.len());
                println!("action_constraints_count={}", cfg.action_constraints.len());
                for (k, v) in &cfg.constants {
                    println!("constant.{k}={v:?}");
                }
            }
        }
        Command::RunTla {
            module,
            config,
            init,
            next,
            runtime,
            storage,
        } => {
            let model =
                TlaModel::from_files(&module, config.as_deref(), init.as_deref(), next.as_deref())
                    .map_err(|e| {
                        eprintln!("Error building TLA+ model:");
                        eprintln!("  Module: {}", module.display());
                        if let Some(cfg) = &config {
                            eprintln!("  Config: {}", cfg.display());
                        }
                        eprintln!("  Error: {}", e);
                        e
                    })?;

            println!("Running TLA+ model: {}", module.display());
            println!("  Init: {}", model.init_name);
            println!("  Next: {}", model.next_name);
            println!("  Variables: {:?}", model.module.variables);
            println!("  Invariants: {}", model.invariant_exprs.len());
            println!();

            // Clone model for post-processing (liveness checking)
            let model_for_liveness = model.clone();

            let config = build_engine_config(&runtime, &storage)?;
            let outcome = run_model(model, config).map_err(|e| {
                eprintln!("Error running model:");
                eprintln!("  {}", e);
                e
            })?;

            println!();
            print_stats("tla-native", &outcome.stats);

            if let Some(violation) = outcome.violation {
                println!("violation=true");
                println!("violation_message={}", violation.message);
                println!("violation_state={:?}", violation.state);
                std::process::exit(1);
            } else {
                // No safety violations found - check liveness properties if present
                use tlaplusplus::model::Model;
                let initial_states = model_for_liveness.initial_states();
                if let Some(first_state) = initial_states.first() {
                    if model_for_liveness.has_liveness_properties()
                        && !model_for_liveness.temporal_properties.is_empty()
                    {
                        use anyhow::anyhow;
                        use tlaplusplus::liveness::LivenessChecker;

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
                            let ctx = EvalContext::with_definitions(
                                state,
                                &model_for_liveness.module.definitions,
                            );
                            match eval_expr(pred_expr, &ctx) {
                                Ok(TlaValue::Bool(b)) => Ok(b),
                                Ok(_) => Err(anyhow!("predicate did not evaluate to boolean")),
                                Err(e) => Err(anyhow!("evaluation error: {}", e)),
                            }
                        }) {
                            Ok(_) => {
                                println!("All temporal properties satisfied on finite trace");
                                println!();
                                println!(
                                    "Note: Full liveness checking requires fairness analysis on state graph cycles"
                                );
                            }
                            Err(msg) => {
                                println!("Temporal property violation: {}", msg);
                                std::process::exit(1);
                            }
                        }
                    }
                }

                // TODO: Fairness checking
                // Fairness constraints (WF/SF) require analyzing strongly connected components
                // in the labeled transition graph. This needs runtime integration to:
                // 1. Collect labeled transitions during exploration (using model.next_states_labeled)
                // 2. Build the transition graph
                // 3. Find SCCs using Tarjan's algorithm
                // 4. Check fairness constraints on each SCC using BuchiChecker
                //
                // The infrastructure is in place (see src/fairness.rs, src/liveness.rs),
                // but integration requires modifying the generic runtime to optionally collect
                // labeled transitions when fairness constraints are present.

                println!("violation=false");
                println!();
                println!("Model checking completed successfully!");
            }
        }
    }

    Ok(())
}

fn config_value_to_tla(value: &ConfigValue) -> Option<TlaValue> {
    match value {
        ConfigValue::Int(v) => Some(TlaValue::Int(*v)),
        ConfigValue::Bool(v) => Some(TlaValue::Bool(*v)),
        ConfigValue::String(v) => Some(TlaValue::String(v.clone())),
        ConfigValue::ModelValue(v) => Some(TlaValue::ModelValue(v.clone())),
        ConfigValue::OperatorRef(_) => None,
        ConfigValue::Tuple(values) => Some(TlaValue::Seq(
            values.iter().filter_map(config_value_to_tla).collect(),
        )),
        ConfigValue::Set(values) => Some(TlaValue::Set(
            values.iter().filter_map(config_value_to_tla).collect(),
        )),
    }
}

fn sample_param_value(param: &str, probe_state: &TlaState) -> TlaValue {
    let lower = param.to_ascii_lowercase();

    let from_named_set = |name: &str| -> Option<TlaValue> {
        match probe_state.get(name) {
            Some(TlaValue::Set(values)) => values.iter().next().cloned(),
            _ => None,
        }
    };

    let set_hint = match lower.as_str() {
        "bot" | "buyer" | "holder" | "b" => Some("Bots"),
        "seller" | "writer" | "s" => Some("Sellers"),
        "mm" => Some("MarketMakers"),
        "asset" | "a" => Some("Assets"),
        "node" | "n" | "n1" | "n2" | "other" => Some("Nodes"),
        "client" | "p" | "p1" | "p2" => Some("Participants"),
        _ => None,
    };
    if let Some(set_name) = set_hint
        && let Some(v) = from_named_set(set_name)
    {
        return v;
    }

    if lower.contains("qty")
        || lower.contains("price")
        || lower.contains("strike")
        || lower.contains("premium")
        || lower.contains("expiry")
        || lower.contains("time")
        || lower.contains("drift")
        || lower == "delta"
    {
        return TlaValue::Int(1);
    }

    if lower.contains("key") || lower.contains("response") || lower.contains("operation") {
        return TlaValue::String("sample".to_string());
    }

    if lower == "args" {
        return TlaValue::Set(Default::default());
    }

    if lower.ends_with("id") {
        return TlaValue::String(format!("{param}_0"));
    }

    TlaValue::ModelValue(param.to_string())
}
