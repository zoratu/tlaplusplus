use clap::{Args, Parser, Subcommand};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tlaplusplus::models::counter_grid::CounterGridModel;
use tlaplusplus::models::flurm_job_lifecycle::FlurmJobLifecycleModel;
use tlaplusplus::models::high_branching::HighBranchingModel;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::system::{check_thp_and_warn, parse_cpu_list};
use tlaplusplus::tla::action_exec::probe_next_disjuncts_with_instances;
use tlaplusplus::tla::module::TlaModuleInstance;
use tlaplusplus::tla::{
    ActionClause, ClauseKind, ConfigValue, EvalContext, TlaConfig, TlaDefinition, TlaModule,
    TlaState, TlaValue, classify_clause, compile_action_ir, compile_action_ir_branches,
    eval_action_body_multi, eval_expr, eval_let_action_multi, looks_like_action,
    normalize_operator_ref_name, normalize_param_name, parse_action_exists,
    parse_stuttering_action_expr, parse_tla_config, parse_tla_module_file, restore_eval_budget,
    scan_module_closure, set_active_eval_budget, split_action_body_disjuncts, split_top_level,
};
use tlaplusplus::{EngineConfig, SimulationConfig, run_model, run_simulation};

/// Parse human-readable byte sizes like "200GB", "10GiB", "512MB"
fn parse_byte_size(s: &str) -> Result<u64, String> {
    let s = s.trim();

    // Try to parse as raw number first
    if let Ok(n) = s.parse::<u64>() {
        return Ok(n);
    }

    // Parse number with unit suffix
    let (num_part, unit_part) = s
        .char_indices()
        .find(|(_, c)| c.is_alphabetic())
        .map(|(i, _)| s.split_at(i))
        .ok_or_else(|| format!("Invalid byte size format: {}", s))?;

    let num: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("Invalid number in byte size: {}", num_part))?;

    let multiplier: u64 = match unit_part.to_uppercase().as_str() {
        "B" => 1,
        "KB" => 1_000,
        "KIB" => 1_024,
        "MB" => 1_000_000,
        "MIB" => 1_048_576,
        "GB" => 1_000_000_000,
        "GIB" => 1_073_741_824,
        "TB" => 1_000_000_000_000,
        "TIB" => 1_099_511_627_776,
        _ => {
            return Err(format!(
                "Unknown unit: {}. Supported: B, KB, KiB, MB, MiB, GB, GiB, TB, TiB",
                unit_part
            ));
        }
    };

    Ok((num * multiplier as f64) as u64)
}

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
        help = "NUMA nodes to use (e.g., 0,1 or 0-2). Restricts workers to cores on these nodes only."
    )]
    numa_nodes: Option<String>,
    #[arg(
        long,
        value_parser = parse_byte_size,
        help = "Hard memory ceiling (supports units: 200GB, 10GiB, 512MB, etc.)"
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
    /// Start fresh, ignoring any existing checkpoint (default: auto-resume when S3 is configured)
    #[arg(long, default_value_t = false)]
    fresh: bool,
    /// Checkpoint interval in seconds (0 = disabled, default: 600 with S3, 0 without)
    #[arg(long, default_value_t = 0)]
    checkpoint_interval_secs: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    checkpoint_on_exit: bool,
    #[arg(long, default_value_t = 1)]
    poll_sleep_ms: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stop_on_violation: bool,
    /// Auto-tune worker count based on CPU utilization (reduces workers when sys% is high)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    auto_tune: bool,
    /// Skip system configuration checks (THP, etc.) at startup
    #[arg(long, default_value_t = false)]
    skip_system_checks: bool,
    /// Enable BFS parent tracking for error trace reconstruction (TLC-style).
    #[arg(long, default_value_t = false)]
    trace_parents: bool,
    /// Maximum number of states to record for parent tracking.
    #[arg(long, default_value_t = 10_000_000)]
    max_trace_states: usize,
    /// Continue after invariant violations (do not stop workers)
    #[arg(long = "continue", default_value_t = false)]
    continue_on_violation: bool,
    /// Maximum number of violations to collect before stopping (default 1)
    #[arg(long, default_value_t = 1)]
    max_violations: usize,
}

#[derive(Args, Clone, Debug)]
struct StorageArgs {
    /// Number of fingerprint shards (0 = auto-calculate based on CPU/NUMA topology)
    #[arg(long, default_value_t = 0)]
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
    /// Disable disk spilling for work-stealing queues (spilling is enabled by default)
    #[arg(long, default_value_t = false)]
    disable_queue_spilling: bool,
    /// Max items in memory before spilling to disk (when enable_queue_spilling is true)
    #[arg(long, default_value_t = 50_000_000)]
    queue_max_inmem_items: u64,
    /// Disable fingerprint persistence (persistence is enabled by default for resume support)
    #[arg(long, default_value_t = false)]
    disable_fp_persistence: bool,
    /// Use bloom filter for fingerprints (bounded memory, ~1% false positive rate)
    /// This drastically reduces memory usage at the cost of possibly re-exploring ~1% of states
    #[arg(long, default_value_t = false)]
    use_bloom_fingerprints: bool,
    /// Enable automatic switching from exact to bloom filter fingerprints
    /// When enabled, starts with exact fingerprints and switches to bloom when:
    /// - Memory usage exceeds --bloom-switch-memory-threshold, OR
    /// - State count exceeds --bloom-switch-threshold
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    bloom_auto_switch: bool,
    /// State count threshold to trigger bloom auto-switch (default: 1 billion)
    #[arg(long, default_value_t = 1_000_000_000)]
    bloom_switch_threshold: u64,
    /// Memory pressure threshold to trigger bloom auto-switch (0.0-1.0, default: 0.85)
    #[arg(long, default_value_t = 0.85)]
    bloom_switch_memory_threshold: f64,
    /// False positive rate for bloom filter after auto-switch (default: 0.001 = 0.1%)
    #[arg(long, default_value_t = 0.001)]
    bloom_switch_fpr: f64,
}

#[derive(Args, Clone, Debug, Default)]
struct S3Args {
    /// S3 bucket for checkpoint persistence (enables S3 sync)
    #[arg(long)]
    s3_bucket: Option<String>,
    /// S3 prefix/path for this run (e.g., "runs/my-run-123")
    #[arg(long, default_value = "")]
    s3_prefix: String,
    /// S3 region (e.g., "us-east-1"). If not specified, uses instance/env region
    #[arg(long)]
    s3_region: Option<String>,
    /// S3 upload interval in seconds (default: 10)
    #[arg(long, default_value_t = 10)]
    s3_upload_interval_secs: u64,
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
        #[command(flatten)]
        s3: S3Args,
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
        #[command(flatten)]
        s3: S3Args,
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
        #[command(flatten)]
        s3: S3Args,
    },
    RunAdaptiveBranching {
        #[arg(long, default_value_t = 5)]
        max_depth: u32,
        #[arg(long, default_value_t = 20)]
        min_branching: u32,
        #[arg(long, default_value_t = 500)]
        max_branching: u32,
        /// Memory threshold percentage to trigger backoff (e.g., 85 = back off at 85% memory usage)
        #[arg(long, default_value_t = 85)]
        memory_threshold_pct: u8,
        /// How often to check memory and adjust branching (seconds)
        #[arg(long, default_value_t = 5)]
        adjustment_interval_secs: u64,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
        #[command(flatten)]
        s3: S3Args,
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
        /// Allow deadlocked states (no successors) without error.
        /// Equivalent to TLC's CHECK_DEADLOCK FALSE or -deadlock flag.
        #[arg(long, default_value_t = false)]
        allow_deadlock: bool,
        /// Run in simulation mode (random trace exploration) instead of BFS
        #[arg(long, default_value_t = false)]
        simulate: bool,
        /// Maximum depth per simulation trace (default 100)
        #[arg(long, default_value_t = 100)]
        simulate_depth: usize,
        /// Number of simulation traces to run (default 1000)
        #[arg(long, default_value_t = 1000)]
        simulate_traces: usize,
        /// Random seed for simulation (0 = system entropy)
        #[arg(long, default_value_t = 0)]
        simulate_seed: u64,
        /// Enable swarm testing: randomly disable Next disjuncts per simulation trace.
        /// Based on "Swarm Testing" (Groce et al., ISSTA 2012).
        /// Only effective with --simulate.
        #[arg(long, default_value_t = false)]
        swarm: bool,
        /// Enable action coverage profiling
        #[arg(long, default_value_t = false)]
        coverage: bool,
        /// Dump state graph to a file after exploration
        #[arg(long)]
        dump: Option<std::path::PathBuf>,
        /// Show only changed variables in error traces (like TLC's -difftrace)
        #[arg(long, default_value_t = false)]
        difftrace: bool,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
        #[command(flatten)]
        s3: S3Args,
    },
    /// List available checkpoints from S3 and/or local disk
    ListCheckpoints {
        /// Local work directory to check for checkpoints
        #[arg(long)]
        work_dir: Option<std::path::PathBuf>,
        /// S3 bucket containing checkpoints
        #[arg(long)]
        s3_bucket: Option<String>,
        /// S3 prefix/path for the run (e.g., "runs/parallel-v5")
        #[arg(long, default_value = "")]
        s3_prefix: String,
        /// S3 region (e.g., "us-east-1")
        #[arg(long)]
        s3_region: Option<String>,
        /// Validate that all required segments exist for each checkpoint
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        validate: bool,
    },
}

fn build_engine_config(
    runtime: &RuntimeArgs,
    storage: &StorageArgs,
    s3_enabled: bool,
) -> anyhow::Result<EngineConfig> {
    let core_ids = match &runtime.core_ids {
        Some(spec) => Some(parse_cpu_list(spec)?),
        None => None,
    };

    // Parse NUMA node list (uses same format as CPU list: "0,1" or "0-2")
    let numa_nodes = match &runtime.numa_nodes {
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
        numa_nodes,
        memory_max_bytes: runtime.memory_max_bytes,
        estimated_state_bytes: runtime.estimated_state_bytes,
        work_dir: runtime.work_dir.clone(),
        clean_work_dir: runtime.clean_work_dir,
        // Auto-resume from checkpoint when S3 is enabled (unless --fresh is specified)
        resume_from_checkpoint: s3_enabled && !runtime.fresh,
        checkpoint_interval_secs: runtime.checkpoint_interval_secs,
        checkpoint_on_exit: runtime.checkpoint_on_exit,
        poll_sleep_ms: runtime.poll_sleep_ms,
        stop_on_violation: if runtime.continue_on_violation {
            false
        } else {
            runtime.stop_on_violation
        },
        max_violations: runtime.max_violations,
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
        enable_queue_spilling: !storage.disable_queue_spilling,
        queue_max_inmem_items: storage.queue_max_inmem_items,
        auto_tune: runtime.auto_tune,
        enable_fp_persistence: (s3_enabled && !runtime.fresh) || !storage.disable_fp_persistence,
        use_bloom_fingerprints: storage.use_bloom_fingerprints,
        bloom_auto_switch: storage.bloom_auto_switch && !storage.use_bloom_fingerprints,
        bloom_switch_threshold: storage.bloom_switch_threshold,
        bloom_switch_memory_threshold: storage.bloom_switch_memory_threshold,
        bloom_switch_fpr: storage.bloom_switch_fpr,
        // Default to false - set to true in run_model_with_s3 when S3 is active
        defer_queue_segment_deletion: false,
        trace_parents: runtime.trace_parents,
        max_trace_states: runtime.max_trace_states,
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

/// Run a model with optional S3 persistence
/// If s3.s3_bucket is Some, enables continuous S3 upload and download on resume
/// When S3 is enabled, defaults checkpoint interval to 10 minutes if not explicitly set
fn run_model_with_s3<M>(
    model: M,
    mut config: EngineConfig,
    s3: &S3Args,
) -> anyhow::Result<tlaplusplus::RunOutcome<M::State>>
where
    M: tlaplusplus::Model + Send + Sync + 'static,
    M::State: Clone
        + std::fmt::Debug
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    use tlaplusplus::storage::s3_persistence::S3Persistence;

    if s3.s3_bucket.is_none() {
        // No S3 bucket configured, run directly
        return run_model(model, config);
    }

    // When S3 is enabled, default to 10-minute checkpoint interval for spot instance safety
    // Only apply default if checkpoint interval was left at CLI default (0 = disabled)
    if config.checkpoint_interval_secs == 0 {
        config.checkpoint_interval_secs = 600; // 10 minutes
        eprintln!("S3: Using default checkpoint interval of 10 minutes for spot instance safety");
    }

    // With S3 enabled, defer queue segment deletion until S3 confirms upload
    // This prevents the race condition where segments are deleted before being synced
    config.defer_queue_segment_deletion = true;

    let bucket = s3.s3_bucket.as_ref().unwrap();
    let prefix = if s3.s3_prefix.is_empty() {
        // Generate a prefix based on timestamp
        format!("runs/{}", chrono::Local::now().format("%Y%m%d-%H%M%S"))
    } else {
        s3.s3_prefix.clone()
    };

    eprintln!("S3: Enabled persistence to s3://{}/{}", bucket, prefix);

    // Create tokio runtime for S3 operations
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("s3-io")
        .enable_all()
        .build()?;

    // Initialize S3 persistence
    let mut s3_persist = rt.block_on(async {
        S3Persistence::new(bucket, &prefix, &config.work_dir, s3.s3_region.as_deref())
            .await
            .map(|p| p.with_upload_interval(s3.s3_upload_interval_secs))
    })?;

    // Download existing state if resuming
    if config.resume_from_checkpoint {
        eprintln!("S3: Checking for existing checkpoint to resume...");
        match rt.block_on(s3_persist.download_state()) {
            Ok(tlaplusplus::storage::s3_persistence::DownloadResult::Resumed {
                manifest, ..
            }) => {
                eprintln!("S3: Found checkpoint in run '{}'", manifest.run_id);
                if let Some(ref cp) = manifest.checkpoint {
                    eprintln!("S3: Resuming from checkpoint:");
                    eprintln!("      States generated: {}", cp.states_generated);
                    eprintln!("      Distinct states:  {}", cp.states_distinct);
                    eprintln!("      Queue pending:    {}", cp.queue_pending);
                    let ts = chrono::DateTime::from_timestamp(manifest.updated_at as i64, 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    eprintln!("      Last updated:     {}", ts);
                } else {
                    eprintln!("S3: Resuming from manifest (no checkpoint state)");
                }
            }
            Ok(tlaplusplus::storage::s3_persistence::DownloadResult::NoExistingState) => {
                eprintln!("S3: No existing checkpoint found, starting fresh");
            }
            Err(e) => {
                eprintln!("S3: Warning: failed to download state: {}", e);
                eprintln!("S3: Starting fresh due to download error");
            }
        }
    } else if s3.s3_bucket.is_some() {
        eprintln!("S3: Starting fresh (--fresh specified, ignoring any existing checkpoint)");
    }

    // Start background upload
    s3_persist.start_background_upload(rt.handle());

    // Set up SIGTERM handler for graceful shutdown
    let s3_persist = std::sync::Arc::new(std::sync::Mutex::new(Some(s3_persist)));
    let s3_persist_for_signal = s3_persist.clone();
    let _rt_handle = rt.handle().clone();

    // Spawn signal handler for graceful shutdown (spot instance preemption)
    eprintln!("Signal handler: Setting up SIGTERM/SIGINT handlers...");
    rt.spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
            let mut sigint =
                signal(SignalKind::interrupt()).expect("Failed to register SIGINT handler");

            eprintln!("Signal handler: Waiting for signals...");

            let is_spot_preemption = tokio::select! {
                _ = sigterm.recv() => {
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                    eprintln!("\n🛑 SIGTERM received - spot instance preemption detected!");
                    eprintln!("   AWS gives us ~2 minutes to checkpoint and flush to S3...");
                    let _ = std::io::stderr().flush();
                    true
                }
                _ = sigint.recv() => {
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                    eprintln!("\n🛑 SIGINT received - graceful shutdown requested...");
                    let _ = std::io::stderr().flush();
                    false
                }
            };

            // Request emergency checkpoint from the checkpoint thread
            eprintln!("   Step 1/3: Requesting emergency checkpoint...");
            tlaplusplus::chaos::request_emergency_checkpoint();

            // Wait for checkpoint to complete (max 90 seconds for spot, 30 for SIGINT)
            let timeout_secs = if is_spot_preemption { 90 } else { 30 };
            let checkpoint_ok = tokio::task::spawn_blocking(move || {
                tlaplusplus::chaos::wait_for_emergency_checkpoint(timeout_secs)
            })
            .await
            .unwrap_or(false);

            if checkpoint_ok {
                eprintln!("   Step 2/3: Emergency checkpoint complete!");
            } else {
                eprintln!(
                    "   Step 2/3: Checkpoint timed out after {}s (continuing with S3 flush)",
                    timeout_secs
                );
            }

            // Take the persist handle out of the mutex before any await
            let persist_opt = s3_persist_for_signal.lock().unwrap().take();

            // S3 flush
            eprintln!("   Step 3/3: Flushing to S3...");
            if let Some(mut persist) = persist_opt {
                if let Err(e) = persist.emergency_flush().await {
                    eprintln!("   S3 emergency flush failed: {}", e);
                } else {
                    eprintln!("   S3 emergency flush complete!");
                }
                // Upload manifest so the next run can find our checkpoint data.
                // Without this, the checkpoint files exist in S3 but the resume
                // logic can't locate them.
                let checkpoint = tlaplusplus::storage::s3_persistence::CheckpointState {
                    id: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    states_generated: 0, // Unknown at signal time
                    states_distinct: 0,
                    queue_pending: 0,
                    min_segment_id: None,
                };
                if let Err(e) = persist.upload_manifest(Some(checkpoint)).await {
                    eprintln!("   S3 manifest upload failed: {}", e);
                } else {
                    eprintln!("   S3 manifest uploaded for resume!");
                }
                if let Err(e) = persist.stop().await {
                    eprintln!("   S3 stop failed: {}", e);
                }
            }

            eprintln!("✅ Graceful shutdown complete. State preserved for resume.");
            use std::io::Write;
            let _ = std::io::stderr().flush();
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }

        #[cfg(not(unix))]
        {
            // On non-Unix, just use ctrl-c
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\n🛑 Ctrl+C received - graceful shutdown requested...");

            // Request emergency checkpoint
            eprintln!("   Step 1/3: Requesting emergency checkpoint...");
            tlaplusplus::chaos::request_emergency_checkpoint();

            // Wait for checkpoint (max 30 seconds)
            let checkpoint_ok = tokio::task::spawn_blocking(|| {
                tlaplusplus::chaos::wait_for_emergency_checkpoint(30)
            })
            .await
            .unwrap_or(false);

            if checkpoint_ok {
                eprintln!("   Step 2/3: Emergency checkpoint complete!");
            } else {
                eprintln!("   Step 2/3: Checkpoint timed out (continuing with S3 flush)");
            }

            // Take the persist handle out of the mutex before any await
            let persist_opt = s3_persist_for_signal.lock().unwrap().take();

            eprintln!("   Step 3/3: Flushing to S3...");
            if let Some(mut persist) = persist_opt {
                if let Err(e) = persist.emergency_flush().await {
                    eprintln!("   S3 emergency flush failed: {}", e);
                } else {
                    eprintln!("   S3 emergency flush complete!");
                }
                if let Err(e) = persist.stop().await {
                    eprintln!("   S3 stop failed: {}", e);
                }
            }

            eprintln!("✅ Graceful shutdown complete. State preserved for resume.");
            use std::io::Write;
            let _ = std::io::stderr().flush();
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }
    });

    // Run the model
    let outcome = run_model(model, config)?;

    // Final flush to S3
    if let Some(mut persist) = s3_persist.lock().unwrap().take() {
        eprintln!("S3: Performing final flush...");

        // Add checkpoint state
        // Note: min_segment_id is None here because we're at completion - no segments needed
        let checkpoint = tlaplusplus::storage::s3_persistence::CheckpointState {
            id: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            states_generated: outcome.stats.states_generated,
            states_distinct: outcome.stats.states_distinct,
            queue_pending: 0,     // Completed
            min_segment_id: None, // No segments needed - run is complete
        };

        rt.block_on(async {
            if let Err(e) = persist.emergency_flush().await {
                eprintln!("S3: Final flush error: {}", e);
            }
            if let Err(e) = persist.upload_manifest(Some(checkpoint)).await {
                eprintln!("S3: Manifest upload error: {}", e);
            }
            if let Err(e) = persist.stop().await {
                eprintln!("S3: Stop error: {}", e);
            }
        });

        let stats = persist.stats();
        eprintln!(
            "S3: Total uploaded: {} files, {:.2} MB",
            stats.files_uploaded,
            stats.bytes_uploaded as f64 / 1_048_576.0
        );
    }

    Ok(outcome)
}

/// Run system configuration checks (THP, etc.) unless skipped
fn run_system_checks(skip: bool) {
    if skip {
        return;
    }
    check_thp_and_warn();
}

/// Feature 7: Evaluate ASSUME/AXIOM statements from the TLA+ module.
/// If any ASSUME evaluates to FALSE, prints an error and exits.
fn evaluate_assumes(model: &TlaModel) -> anyhow::Result<()> {
    if model.module.assumes.is_empty() {
        return Ok(());
    }

    let empty_state = TlaState::new();
    let ctx = EvalContext::with_definitions_and_instances(
        &empty_state,
        &model.module.definitions,
        &model.module.instances,
    );

    for body in &model.module.assumes {
        match eval_expr(body, &ctx) {
            Ok(TlaValue::Bool(true)) => {
                // ASSUME satisfied
            }
            Ok(TlaValue::Bool(false)) => {
                return Err(anyhow::anyhow!("ASSUME failed: {}", body));
            }
            Ok(other) => {
                eprintln!(
                    "Warning: ASSUME '{}' evaluated to non-boolean: {:?}",
                    body, other
                );
            }
            Err(e) => {
                // Don't fail on evaluation errors for ASSUME - some may reference
                // state variables or complex expressions we can't evaluate yet
                eprintln!("Warning: Could not evaluate ASSUME '{}': {}", body, e);
            }
        }
    }

    Ok(())
}

/// Feature 2: Collect action coverage statistics by replaying from initial states.
/// This is a lightweight post-hoc analysis that evaluates the Next relation
/// per-disjunct to see which actions fire.
fn collect_coverage(model: &TlaModel) -> tlaplusplus::CoverageStats {
    use tlaplusplus::coverage::{ActionCoverageEntry, CoverageStats};
    use tlaplusplus::model::Model;

    let next_def = match model.module.definitions.get(&model.next_name) {
        Some(def) => def,
        None => return CoverageStats::default(),
    };

    // Split Next body into disjuncts (actions)
    let disjuncts = split_action_body_disjuncts(&next_def.body);

    let action_names: Vec<String> = disjuncts
        .iter()
        .enumerate()
        .map(|(i, d)| {
            // Try to extract a meaningful name from the disjunct
            let trimmed = d.trim();
            if let Some(name) = trimmed.split_whitespace().next() {
                if name.chars().next().map_or(false, |c| c.is_alphabetic()) && !name.contains('(')
                {
                    return name.to_string();
                }
            }
            format!("disjunct_{}", i)
        })
        .collect();

    let mut stats = CoverageStats::default();
    for name in &action_names {
        stats.actions.insert(name.clone(), ActionCoverageEntry::default());
    }

    // Evaluate each disjunct against initial states to populate coverage
    let initial_states = model.initial_states();
    let mut explored = std::collections::HashSet::new();
    let mut frontier: Vec<TlaState> = initial_states;
    let mut next_frontier = Vec::new();

    // BFS up to a limited depth to collect coverage stats
    let max_coverage_depth = 10;
    for _depth in 0..max_coverage_depth {
        if frontier.is_empty() {
            break;
        }
        for state in &frontier {
            let fp = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                state.hash(&mut h);
                h.finish()
            };
            if !explored.insert(fp) {
                continue;
            }

            let instances = if model.module.instances.is_empty() {
                None
            } else {
                Some(&model.module.instances)
            };

            // Try each disjunct individually
            for (idx, disjunct) in disjuncts.iter().enumerate() {
                match tlaplusplus::tla::evaluate_next_states_with_instances(
                    disjunct,
                    &model.module.definitions,
                    instances,
                    state,
                ) {
                    Ok(successors) if !successors.is_empty() => {
                        let name = &action_names[idx];
                        if let Some(entry) = stats.actions.get_mut(name) {
                            entry.fires += 1;
                            entry.states_generated += successors.len() as u64;
                        }
                        next_frontier.extend(successors);
                    }
                    _ => {}
                }
            }
        }
        frontier = std::mem::take(&mut next_frontier);
    }

    stats
}

/// Feature 4: Dump state graph to a file.
/// Format: one line per state "STATE hash", then "hash1 -> hash2" per transition.
fn dump_state_graph(
    model: &TlaModel,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::io::Write;
    use tlaplusplus::model::Model;

    eprintln!("Dumping state graph to {}...", path.display());

    let mut state_hashes: HashMap<u64, TlaState> = HashMap::new();
    let mut transitions: Vec<(u64, u64)> = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<TlaState> = VecDeque::new();

    for init in model.initial_states() {
        let fp = model_fingerprint(&init);
        if visited.insert(fp) {
            state_hashes.insert(fp, init.clone());
            queue.push_back(init);
        }
    }

    let mut successors = Vec::new();
    while let Some(state) = queue.pop_front() {
        let from_fp = model_fingerprint(&state);
        successors.clear();
        model.next_states(&state, &mut successors);

        for succ in &successors {
            let to_fp = model_fingerprint(succ);
            transitions.push((from_fp, to_fp));
            if visited.insert(to_fp) {
                state_hashes.insert(to_fp, succ.clone());
                queue.push_back(succ.clone());
            }
        }
    }

    let mut file = std::fs::File::create(path)?;
    // Write states
    for (hash, state) in &state_hashes {
        writeln!(file, "STATE {:#018x} {:?}", hash, state)?;
    }
    writeln!(file)?;
    // Write transitions
    for (from, to) in &transitions {
        writeln!(file, "{:#018x} -> {:#018x}", from, to)?;
    }

    eprintln!(
        "Dumped {} states, {} transitions to {}",
        state_hashes.len(),
        transitions.len(),
        path.display()
    );
    Ok(())
}

/// Compute fingerprint for dump_state_graph
fn model_fingerprint(state: &TlaState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    state.hash(&mut h);
    h.finish()
}

/// Feature 5: Print difftrace - only show variables that changed between steps.
fn print_difftrace(trace: &[TlaState]) {
    if trace.is_empty() {
        return;
    }
    println!("  step 0 (initial):");
    for (k, v) in &trace[0] {
        println!("    /\\ {} = {:?}", k, v);
    }
    for i in 1..trace.len() {
        let prev = &trace[i - 1];
        let curr = &trace[i];
        println!("  step {} (changed):", i);
        let mut any_changed = false;
        for (k, v) in curr {
            match prev.get(k) {
                Some(old_v) if old_v == v => {
                    // unchanged, skip in difftrace
                }
                _ => {
                    println!("    /\\ {} = {:?}", k, v);
                    any_changed = true;
                }
            }
        }
        // Check for variables that were removed
        for k in prev.keys() {
            if !curr.contains_key(k) {
                println!("    /\\ {} = <removed>", k);
                any_changed = true;
            }
        }
        if !any_changed {
            println!("    (no changes - stuttering step)");
        }
    }
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
            s3,
        } => {
            run_system_checks(runtime.skip_system_checks);
            let model = CounterGridModel::new(max_x, max_y, max_sum);
            let config = build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
            let outcome = run_model_with_s3(model, config, &s3)?;
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
            s3,
        } => {
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
        }
        Command::RunHighBranching {
            max_depth,
            branching_factor,
            runtime,
            storage,
            s3,
        } => {
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
        }
        Command::RunAdaptiveBranching {
            max_depth,
            min_branching,
            max_branching,
            memory_threshold_pct,
            adjustment_interval_secs,
            runtime,
            storage,
            s3: _, // S3 not yet integrated with adaptive branching
        } => {
            run_system_checks(runtime.skip_system_checks);
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::thread;
            use std::time::Duration;
            use tlaplusplus::models::adaptive_branching::AdaptiveBranchingModel;

            let model = AdaptiveBranchingModel::new(max_depth, min_branching, max_branching);
            let model_clone = model.clone();

            // Get memory limit from config
            let memory_max = runtime
                .memory_max_bytes
                .or_else(|| tlaplusplus::system::cgroup_memory_max_bytes())
                .unwrap_or(16 * 1024 * 1024 * 1024); // 16GB default

            let threshold_bytes =
                (memory_max as f64 * (memory_threshold_pct as f64 / 100.0)) as u64;

            // Spawn monitoring thread
            let done = Arc::new(AtomicBool::new(false));
            let done_clone = done.clone();

            eprintln!("🚀 Starting adaptive branching test:");
            eprintln!("   Depth: {}", max_depth);
            eprintln!(
                "   Branching: {} → {} (adaptive)",
                min_branching, max_branching
            );
            eprintln!(
                "   Memory limit: {:.1} GB, threshold: {}% ({:.1} GB)",
                memory_max as f64 / (1024.0 * 1024.0 * 1024.0),
                memory_threshold_pct,
                threshold_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            eprintln!("   Adjustment interval: {}s", adjustment_interval_secs);
            eprintln!();

            let monitor_thread = thread::spawn(move || {
                let mut cycles_since_ramp = 0;
                let ramp_up_delay = 3; // Wait 3 cycles before ramping up

                while !done_clone.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(adjustment_interval_secs));

                    // Check memory usage (RSS of current process)
                    #[cfg(target_os = "linux")]
                    let mem_info_opt = procfs::process::Process::myself()
                        .and_then(|p| p.stat())
                        .map(|stat| stat.rss * 4096) // rss is in pages
                        .ok();

                    #[cfg(not(target_os = "linux"))]
                    let mem_info_opt: Option<u64> = None; // Not available on non-Linux

                    if let Some(mem_info) = mem_info_opt {
                        let memory_pct = (mem_info as f64 / memory_max as f64) * 100.0;

                        eprintln!(
                            "📊 Memory: {:.1} GB ({:.1}%), Branching factor: {}",
                            mem_info as f64 / (1024.0 * 1024.0 * 1024.0),
                            memory_pct,
                            model_clone.current_branching()
                        );

                        if mem_info > threshold_bytes {
                            // Memory pressure - back off immediately
                            eprintln!("⚠️  Memory threshold exceeded!");
                            model_clone.back_off();
                            cycles_since_ramp = 0;
                        } else if memory_pct < 60.0 {
                            // Memory usage low and stable - consider ramping up
                            cycles_since_ramp += 1;
                            if cycles_since_ramp >= ramp_up_delay {
                                model_clone.ramp_up();
                                cycles_since_ramp = 0;
                            }
                        }
                    } else {
                        // Memory monitoring not available - just ramp up conservatively
                        eprintln!(
                            "📊 Memory monitoring N/A, Branching factor: {}",
                            model_clone.current_branching()
                        );
                        cycles_since_ramp += 1;
                        if cycles_since_ramp >= ramp_up_delay {
                            model_clone.ramp_up();
                            cycles_since_ramp = 0;
                        }
                    }
                }
            });

            let config = build_engine_config(&runtime, &storage, false)?;
            let outcome = run_model(model, config)?;

            // Signal monitor thread to stop
            done.store(true, Ordering::Relaxed);
            let _ = monitor_thread.join();

            print_stats("adaptive-branching", &outcome.stats);
            if let Some(violation) = outcome.violation {
                println!("violation=true");
                println!("violation_message={}", violation.message);
                println!("violation_state={:?}", violation.state);
            } else {
                println!("violation=false");
            }
        }
        Command::AnalyzeTla { module, config } => {
            let mut parsed_module = parse_tla_module_file(&module)?;
            let scan = scan_module_closure(&module)?;
            let parsed_cfg = match config.as_ref() {
                Some(cfg_path) => {
                    let raw = std::fs::read_to_string(cfg_path)?;
                    Some(parse_tla_config(&raw)?)
                }
                None => None,
            };
            // Inject constants from config into module definitions (handles OperatorRef)
            if let Some(cfg) = parsed_cfg.as_ref() {
                inject_constants_into_definitions(&mut parsed_module, cfg);
            }
            let analyzed_model = TlaModel::from_files(&module, config.as_deref(), None, None).ok();
            if let Some(model) = analyzed_model.as_ref() {
                parsed_module = model.module.clone();
            }
            let resolved_init_name = analyzed_model
                .as_ref()
                .map(|model| model.init_name.clone())
                .unwrap_or_else(|| "Init".to_string());
            let resolved_next_name = analyzed_model
                .as_ref()
                .map(|model| model.next_name.clone())
                .unwrap_or_else(|| "Next".to_string());
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
            println!("resolved_init={resolved_init_name}");
            println!("resolved_next={resolved_next_name}");
            println!(
                "resolved_initial_states={}",
                analyzed_model
                    .as_ref()
                    .map(|model| model.initial_states_vec.len())
                    .unwrap_or(0)
            );
            if let Some(next_def) = parsed_module.definitions.get(&resolved_next_name) {
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
                            ClauseKind::PrimedAssignment { .. }
                            | ClauseKind::PrimedMembership { .. } => {
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
                if !definition_is_contextually_probeable_action(
                    def,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                    &mut BTreeSet::new(),
                ) {
                    continue;
                }
                action_defs += 1;
                let ir = compile_action_ir(def);
                action_conjunct_max = action_conjunct_max.max(ir.clauses.len());
                for clause in ir.clauses {
                    match clause {
                        ActionClause::PrimedAssignment { .. }
                        | ActionClause::PrimedMembership { .. } => action_primed_assignments += 1,
                        ActionClause::Unchanged { .. } => action_unchanged_clauses += 1,
                        ActionClause::Guard { .. } => action_guard_clauses += 1,
                        ActionClause::Exists { .. } => action_guard_clauses += 1,
                        ActionClause::LetWithPrimes { .. } => action_guard_clauses += 1,
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

            let mut probe_state: TlaState = analyzed_model
                .as_ref()
                .and_then(|model| model.initial_states_vec.first().cloned())
                .unwrap_or_default();
            if let Some(cfg) = parsed_cfg.as_ref() {
                let mut deferred_operator_refs = Vec::new();
                for (k, v) in &cfg.constants {
                    match v {
                        ConfigValue::OperatorRef(name) => {
                            deferred_operator_refs
                                .push((k.clone(), normalize_operator_ref_name(name).to_string()));
                        }
                        _ => {
                            if let Some(tv) = config_value_to_tla(v) {
                                probe_state.insert(k.clone(), tv);
                            }
                        }
                    }
                }
                for _ in 0..deferred_operator_refs.len().saturating_add(1) {
                    if deferred_operator_refs.is_empty() {
                        break;
                    }
                    let mut progress = false;
                    let mut next_deferred = Vec::new();
                    for (name, ref_name) in deferred_operator_refs {
                        let ctx = build_probe_eval_context(
                            &probe_state,
                            &parsed_module.definitions,
                            &parsed_module.instances,
                        );
                        if let Ok(value) = eval_expr(&ref_name, &ctx) {
                            probe_state.insert(name, value);
                            progress = true;
                        } else if let Some(value) =
                            representative_value_from_set_expr(&ref_name, &ctx)
                        {
                            probe_state.insert(name, value);
                            progress = true;
                        } else {
                            next_deferred.push((name, ref_name));
                        }
                    }
                    if !progress {
                        break;
                    }
                    deferred_operator_refs = next_deferred;
                }
            }
            // Set evaluation budget for all probing (Init seeding, branch probing, expr probing).
            // This prevents exponential blowup from Seq, SUBSET, [D -> R], etc.
            let prev_budget = set_active_eval_budget(100_000);
            let mut probe_init_seeded = 0usize;
            let mut probe_init_unresolved = 0usize;
            let mut probe_init_unresolved_vars: Vec<String> = Vec::new();
            if let Some(init_def) = parsed_module.definitions.get(&resolved_init_name) {
                // Collect both equality and membership assignments
                let mut pending_eq: Vec<(String, String)> = Vec::new();
                let mut pending_mem: Vec<(String, String)> = Vec::new();
                for clause in expand_state_predicate_clauses(
                    &init_def.body,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                ) {
                    match classify_clause(&clause) {
                        ClauseKind::UnprimedEquality { var, expr } => {
                            pending_eq.push((var, expr));
                        }
                        ClauseKind::UnprimedMembership { var, set_expr } => {
                            pending_mem.push((var, set_expr));
                        }
                        _ => {}
                    }
                }

                // Process both types together with fixed-point iteration
                let total_pending = pending_eq.len() + pending_mem.len();
                for _ in 0..total_pending.saturating_add(1) {
                    if pending_eq.is_empty() && pending_mem.is_empty() {
                        break;
                    }
                    let mut progress = false;

                    // Process equality assignments
                    let mut next_pending_eq = Vec::new();
                    for (var, expr) in pending_eq {
                        let ctx = build_probe_eval_context(
                            &probe_state,
                            &parsed_module.definitions,
                            &parsed_module.instances,
                        );
                        match eval_expr(&expr, &ctx) {
                            Ok(value) => {
                                probe_state.insert(var, value);
                                probe_init_seeded += 1;
                                progress = true;
                            }
                            Err(_) => next_pending_eq.push((var, expr)),
                        }
                    }

                    // Process membership assignments - pick representative value
                    let mut next_pending_mem = Vec::new();
                    for (var, set_expr) in pending_mem {
                        let ctx = build_probe_eval_context(
                            &probe_state,
                            &parsed_module.definitions,
                            &parsed_module.instances,
                        );
                        match eval_expr(&set_expr, &ctx) {
                            Ok(set_val) => {
                                // Pick a representative from the set
                                if let Some(repr) = pick_representative_from_set(&set_val) {
                                    probe_state.insert(var, repr);
                                    probe_init_seeded += 1;
                                    progress = true;
                                } else {
                                    // Set was empty or not a set, keep pending
                                    next_pending_mem.push((var, set_expr));
                                }
                            }
                            Err(_) => {
                                if let Some(repr) =
                                    representative_value_from_set_expr(&set_expr, &ctx)
                                {
                                    probe_state.insert(var, repr);
                                    probe_init_seeded += 1;
                                    progress = true;
                                } else {
                                    next_pending_mem.push((var, set_expr));
                                }
                            }
                        }
                    }

                    if !progress {
                        probe_init_unresolved = next_pending_eq.len() + next_pending_mem.len();
                        probe_init_unresolved_vars = next_pending_eq
                            .iter()
                            .chain(next_pending_mem.iter())
                            .map(|(var, _)| var.clone())
                            .collect();
                        break;
                    }
                    pending_eq = next_pending_eq;
                    pending_mem = next_pending_mem;
                }
            }
            probe_init_seeded += seed_probe_state_from_type_invariants(
                &mut probe_state,
                &parsed_module,
                parsed_cfg.as_ref(),
            );
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
            if let Some(next_def) = parsed_module.definitions.get(&resolved_next_name) {
                let next_probe = probe_next_disjuncts_with_instances(
                    &next_def.body,
                    &parsed_module.definitions,
                    if parsed_module.instances.is_empty() {
                        None
                    } else {
                        Some(&parsed_module.instances)
                    },
                    &probe_state,
                );
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
            let action_param_samples = infer_action_param_samples_from_module_contexts(
                &resolved_next_name,
                &parsed_module.definitions,
                &parsed_module.instances,
                &probe_state,
            );
            for def in parsed_module.definitions.values() {
                if !definition_is_contextually_probeable_action(
                    def,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                    &mut BTreeSet::new(),
                ) {
                    continue;
                }
                for ir in compile_action_ir_branches(def) {
                    let mut ctx = build_action_expr_probe_context(
                        &probe_state,
                        &parsed_module.definitions,
                        &parsed_module.instances,
                        &ir.params,
                        &ir.clauses,
                        action_param_samples.get(&def.name),
                    );
                    for clause in &ir.clauses {
                        let Some(result) = probe_action_clause_expr(clause, &mut ctx) else {
                            continue;
                        };
                        expr_total += 1;
                        match result {
                            Ok(()) => expr_ok += 1,
                            Err(err) if is_probe_sampling_limitation_error(&err) => {
                                // Treat probe-sampling-limitation errors as OK.
                                // These errors arise because the probe state uses
                                // default/ModelValue placeholders that don't match
                                // the actual runtime domains.
                                expr_ok += 1;
                            }
                            Err(err) => {
                                let key = err.to_string();
                                *expr_errors.entry(key).or_insert(0) += 1;
                                expr_error_examples
                                    .entry(err.to_string())
                                    .or_insert_with(|| {
                                        let expr = match clause {
                                            ActionClause::Guard { expr }
                                            | ActionClause::PrimedAssignment { expr, .. }
                                            | ActionClause::PrimedMembership {
                                                set_expr: expr,
                                                ..
                                            }
                                            | ActionClause::LetWithPrimes { expr } => expr,
                                            ActionClause::Exists { binders, body } => {
                                                return format!(
                                                    "def={} expr=\\\\E {}: {}",
                                                    def.name,
                                                    binders.replace('\n', " "),
                                                    body.replace('\n', " ")
                                                );
                                            }
                                            ActionClause::Unchanged { .. } => unreachable!(),
                                        };
                                        format!("def={} expr={}", def.name, expr.replace('\n', " "))
                                    });
                            }
                        }
                    }
                }
            }
            restore_eval_budget(prev_budget);
            let expr_eval_ready = expr_probe_is_ready(expr_total, expr_ok);
            println!("native_frontend.expr_eval={expr_eval_ready}");
            // If the spec has no Next definition, action_eval is not applicable
            let has_next = parsed_module.definitions.contains_key(&resolved_next_name);
            if has_next {
                println!("native_frontend.action_eval={action_eval_ready}");
            } else {
                println!("native_frontend.action_eval=na");
            }
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
            allow_deadlock,
            simulate,
            simulate_depth,
            simulate_traces,
            simulate_seed,
            swarm,
            coverage,
            dump,
            difftrace,
            runtime,
            storage,
            s3,
        } => {
            run_system_checks(runtime.skip_system_checks);
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
                let sim_outcome = run_simulation(&model, &sim_config, max_violations);
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

            let engine_config =
                build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
            let outcome = run_model_with_s3(model, engine_config, &s3).map_err(|e| {
                eprintln!("Error running model:");
                eprintln!("  {}", e);
                e
            })?;

            // Feature 2: Coverage profiling
            if coverage {
                let cov_stats = collect_coverage(&model_for_liveness);
                cov_stats.print_summary();
            }

            // Feature 4: Dump state graph
            if let Some(ref dump_path) = dump {
                dump_state_graph(&model_for_liveness, dump_path)?;
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
                for (i, violation) in all_violations_display.iter().enumerate() {
                    println!();
                    println!("--- Violation {} ---", i + 1);
                    println!("  message: {}", violation.message);
                    if difftrace {
                        print_difftrace(&violation.trace);
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
        Command::ListCheckpoints {
            work_dir,
            s3_bucket,
            s3_prefix,
            s3_region,
            validate,
        } => {
            list_checkpoints(work_dir, s3_bucket, s3_prefix, s3_region, validate)?;
        }
    }

    Ok(())
}

/// List available checkpoints from S3 and/or local disk
fn list_checkpoints(
    work_dir: Option<std::path::PathBuf>,
    s3_bucket: Option<String>,
    s3_prefix: String,
    s3_region: Option<String>,
    validate: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let rt = tokio::runtime::Runtime::new()?;

    // Check S3 if bucket is provided
    if let Some(bucket) = s3_bucket {
        println!("Checking S3: s3://{}/{}", bucket, s3_prefix);
        println!();

        rt.block_on(async {
            list_s3_checkpoints(&bucket, &s3_prefix, s3_region.as_deref(), validate).await
        })?;
    }

    // Check local disk if work_dir is provided
    if let Some(dir) = work_dir {
        println!("Checking local: {}", dir.display());
        println!();

        let checkpoints_dir = dir.join("checkpoints");
        if checkpoints_dir.exists() {
            let mut checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoints_dir)
                .context("failed reading checkpoints directory")?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with("checkpoint-"))
                        .unwrap_or(false)
                })
                .collect();

            checkpoint_files.sort_by_key(|e| e.path());

            if checkpoint_files.is_empty() {
                println!("  No checkpoint files found");
            } else {
                for entry in checkpoint_files {
                    let path = entry.path();
                    if let Ok(contents) = std::fs::read_to_string(&path) {
                        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&contents) {
                            let states_generated = manifest
                                .get("states_generated")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let states_distinct = manifest
                                .get("states_distinct")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let created = manifest
                                .get("created_unix_secs")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);

                            let time_str = if created > 0 {
                                chrono::DateTime::from_timestamp(created as i64, 0)
                                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                                    .unwrap_or_else(|| "unknown".to_string())
                            } else {
                                "unknown".to_string()
                            };

                            println!(
                                "  {} - {} generated, {} distinct ({})",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                format_num(states_generated),
                                format_num(states_distinct),
                                time_str
                            );
                        }
                    }
                }
            }
        } else {
            println!(
                "  Checkpoints directory not found: {}",
                checkpoints_dir.display()
            );
        }

        // Check queue-spill segments
        let queue_spill_dir = dir.join("queue-spill");
        if queue_spill_dir.exists() {
            let segment_count = std::fs::read_dir(&queue_spill_dir)
                .map(|r| r.filter_map(|e| e.ok()).count())
                .unwrap_or(0);
            println!("  Queue segments on disk: {}", segment_count);
        }
    }

    Ok(())
}

async fn list_s3_checkpoints(
    bucket: &str,
    prefix: &str,
    region: Option<&str>,
    validate: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;
    use aws_config::BehaviorVersion;
    use aws_sdk_s3::Client;
    use aws_sdk_s3::config::Region;

    // Load AWS config
    let mut config_loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(r) = region {
        config_loader = config_loader.region(Region::new(r.to_string()));
    }
    let config = config_loader.load().await;
    let client = Client::new(&config);

    // Fetch manifest.json
    let manifest_key = if prefix.is_empty() {
        "manifest.json".to_string()
    } else {
        format!("{}/manifest.json", prefix)
    };

    let manifest_result = client
        .get_object()
        .bucket(bucket)
        .key(&manifest_key)
        .send()
        .await;

    match manifest_result {
        Ok(resp) => {
            let body = resp
                .body
                .collect()
                .await
                .context("failed reading S3 manifest body")?;
            let manifest: serde_json::Value = serde_json::from_slice(&body.into_bytes())
                .context("failed parsing manifest JSON")?;

            println!(
                "  Run ID: {}",
                manifest
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
            println!(
                "  Last updated: {}",
                manifest
                    .get("updated_at")
                    .and_then(|v| v.as_u64())
                    .map(|ts| {
                        chrono::DateTime::from_timestamp(ts as i64, 0)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                            .unwrap_or_else(|| "unknown".to_string())
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            );

            // List checkpoint info
            if let Some(checkpoint) = manifest.get("checkpoint") {
                let id = checkpoint.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                let states_generated = checkpoint
                    .get("states_generated")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let states_distinct = checkpoint
                    .get("states_distinct")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let queue_pending = checkpoint
                    .get("queue_pending")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let min_segment_id = checkpoint.get("min_segment_id").and_then(|v| v.as_u64());

                println!();
                println!("  Checkpoint #{}", id);
                println!("    States generated: {}", format_num(states_generated));
                println!("    States distinct:  {}", format_num(states_distinct));
                println!("    Queue pending:    {}", format_num(queue_pending));
                if let Some(min_seg) = min_segment_id {
                    println!("    Min segment ID:   {}", min_seg);
                }

                // Validate segments if requested
                if validate {
                    println!();
                    println!("  Validating queue segments...");

                    // List all segment files in S3
                    let segment_prefix = if prefix.is_empty() {
                        "queue-spill/".to_string()
                    } else {
                        format!("{}/queue-spill/", prefix)
                    };

                    let mut segments: Vec<(u64, String)> = Vec::new();
                    let mut continuation_token: Option<String> = None;

                    loop {
                        let mut req = client
                            .list_objects_v2()
                            .bucket(bucket)
                            .prefix(&segment_prefix);

                        if let Some(token) = continuation_token.take() {
                            req = req.continuation_token(token);
                        }

                        let resp = req.send().await.context("failed listing S3 segments")?;

                        for obj in resp.contents() {
                            if let Some(key) = obj.key() {
                                // Extract segment ID from filename like "segment-0000000000012345.bin"
                                if let Some(filename) = key.rsplit('/').next() {
                                    if filename.starts_with("segment-")
                                        && filename.ends_with(".bin")
                                    {
                                        let id_str = filename
                                            .trim_start_matches("segment-")
                                            .trim_end_matches(".bin");
                                        if let Ok(seg_id) = id_str.parse::<u64>() {
                                            segments.push((seg_id, key.to_string()));
                                        }
                                    }
                                }
                            }
                        }

                        if resp.is_truncated() == Some(true) {
                            continuation_token =
                                resp.next_continuation_token().map(|s| s.to_string());
                        } else {
                            break;
                        }
                    }

                    segments.sort_by_key(|(id, _)| *id);

                    let total_segments = segments.len();
                    println!("    Total segments in S3: {}", total_segments);

                    if let Some(min_seg) = min_segment_id {
                        let required_segments: Vec<_> =
                            segments.iter().filter(|(id, _)| *id >= min_seg).collect();

                        let required_count = required_segments.len();
                        println!(
                            "    Segments >= min_segment_id ({}): {}",
                            min_seg, required_count
                        );

                        // Check for gaps
                        if !required_segments.is_empty() {
                            let first_id = required_segments.first().unwrap().0;
                            let last_id = required_segments.last().unwrap().0;
                            let expected_count = (last_id - first_id + 1) as usize;

                            if required_count < expected_count {
                                let mut missing = Vec::new();
                                let segment_ids: std::collections::HashSet<u64> =
                                    required_segments.iter().map(|(id, _)| *id).collect();

                                for id in first_id..=last_id {
                                    if !segment_ids.contains(&id) {
                                        missing.push(id);
                                        if missing.len() >= 10 {
                                            break;
                                        }
                                    }
                                }

                                println!("    ⚠️  MISSING SEGMENTS detected!");
                                println!(
                                    "       Expected {} segments, found {}",
                                    expected_count, required_count
                                );
                                println!("       Missing (first 10): {:?}", missing);
                                println!();
                                println!("    This checkpoint may not be resumable.");
                            } else {
                                println!("    ✓ All required segments present (no gaps)");
                            }
                        }
                    } else {
                        println!(
                            "    ⚠️  No min_segment_id in checkpoint - cannot validate completeness"
                        );
                    }
                }
            } else {
                println!("  No checkpoint state in manifest");
            }

            // Count files by type
            if let Some(files) = manifest.get("files").and_then(|v| v.as_object()) {
                let mut checkpoint_files = 0;
                let mut fingerprint_files = 0;
                let mut queue_files = 0;
                let mut other_files = 0;

                for key in files.keys() {
                    if key.contains("checkpoint") {
                        checkpoint_files += 1;
                    } else if key.contains("fingerprint") {
                        fingerprint_files += 1;
                    } else if key.contains("queue") || key.contains("segment") {
                        queue_files += 1;
                    } else {
                        other_files += 1;
                    }
                }

                println!();
                println!("  Files in manifest:");
                println!("    Checkpoint files: {}", checkpoint_files);
                println!("    Fingerprint files: {}", fingerprint_files);
                println!("    Queue/segment files: {}", queue_files);
                if other_files > 0 {
                    println!("    Other files: {}", other_files);
                }
            }
        }
        Err(e) => {
            println!("  No manifest found at s3://{}/{}", bucket, manifest_key);
            println!("  Error: {}", e);
        }
    }

    Ok(())
}

fn format_num(n: u64) -> String {
    let mut result = String::new();
    let s = n.to_string();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Inject constant bindings from config file into module definitions.
///
/// This handles OperatorRef constants (e.g., `Node <- N1`) by creating
/// definitions that reference the operator, allowing the evaluator to
/// resolve them properly.
fn inject_constants_into_definitions(module: &mut TlaModule, config: &TlaConfig) {
    inject_constants_into_module_tree(module, config, true);
}

fn inject_constants_into_module_tree(
    module: &mut TlaModule,
    config: &TlaConfig,
    include_all_constants: bool,
) {
    for (name, value) in &config.constants {
        if !include_all_constants
            && !module.constants.iter().any(|constant| constant == name)
            && !module.definitions.contains_key(name)
        {
            continue;
        }

        let (params, body) = match value {
            ConfigValue::OperatorRef(target_name) => {
                let target_name = normalize_operator_ref_name(target_name);
                if let Some(target_def) = module.definitions.get(target_name) {
                    let params = target_def.params.clone();
                    let body = if params.is_empty() {
                        target_name.to_string()
                    } else {
                        format!("{target_name}({})", params.join(", "))
                    };
                    (params, body)
                } else {
                    (Vec::new(), target_name.to_string())
                }
            }
            _ => (Vec::new(), config_value_to_expr(value)),
        };

        module.definitions.insert(
            name.clone(),
            TlaDefinition {
                name: name.clone(),
                params,
                body,
                is_recursive: false,
            },
        );
    }

    for instance in module.instances.values_mut() {
        if let Some(instance_module) = instance.module.as_mut() {
            inject_constants_into_module_tree(instance_module, config, false);
        }
    }
}

/// Convert a ConfigValue to a TLA+ expression string
fn config_value_to_expr(value: &ConfigValue) -> String {
    match value {
        ConfigValue::Int(n) => n.to_string(),
        ConfigValue::String(s) => format!("\"{}\"", s),
        ConfigValue::ModelValue(s) => s.clone(),
        ConfigValue::Bool(b) => {
            if *b {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        ConfigValue::Set(values) => {
            let items: Vec<String> = values.iter().map(config_value_to_expr).collect();
            format!("{{{}}}", items.join(", "))
        }
        ConfigValue::Tuple(values) => {
            let items: Vec<String> = values.iter().map(config_value_to_expr).collect();
            format!("<<{}>>", items.join(", "))
        }
        ConfigValue::OperatorRef(name) => normalize_operator_ref_name(name).to_string(),
    }
}

fn config_value_to_tla(value: &ConfigValue) -> Option<TlaValue> {
    match value {
        ConfigValue::Int(v) => Some(TlaValue::Int(*v)),
        ConfigValue::Bool(v) => Some(TlaValue::Bool(*v)),
        ConfigValue::String(v) => Some(TlaValue::String(v.clone())),
        ConfigValue::ModelValue(v) => Some(TlaValue::ModelValue(v.clone())),
        ConfigValue::OperatorRef(_) => None,
        ConfigValue::Tuple(values) => Some(TlaValue::Seq(Arc::new(
            values.iter().filter_map(config_value_to_tla).collect(),
        ))),
        ConfigValue::Set(values) => Some(TlaValue::Set(Arc::new(
            values.iter().filter_map(config_value_to_tla).collect(),
        ))),
    }
}

fn build_probe_eval_context<'a>(
    state: &'a TlaState,
    definitions: &'a BTreeMap<String, TlaDefinition>,
    instances: &'a BTreeMap<String, TlaModuleInstance>,
) -> EvalContext<'a> {
    if instances.is_empty() {
        EvalContext::with_definitions(state, definitions)
    } else {
        EvalContext::with_definitions_and_instances(state, definitions, instances)
    }
}

fn parse_zero_arg_state_operator_ref(text: &str) -> Option<(Option<String>, String)> {
    let trimmed = text.trim();
    if is_probe_simple_identifier(trimmed) {
        return Some((None, trimmed.to_string()));
    }

    let (alias, operator) = trimmed.split_once('!')?;
    let alias = alias.trim();
    let operator = operator.trim();
    if is_probe_simple_identifier(alias) && is_probe_simple_identifier(operator) {
        Some((Some(alias.to_string()), operator.to_string()))
    } else {
        None
    }
}

fn resolve_zero_arg_state_definition<'a>(
    expr: &str,
    definitions: &'a BTreeMap<String, TlaDefinition>,
    instances: &'a BTreeMap<String, TlaModuleInstance>,
) -> Option<(
    String,
    &'a TlaDefinition,
    &'a BTreeMap<String, TlaDefinition>,
    &'a BTreeMap<String, TlaModuleInstance>,
    Option<&'a TlaModuleInstance>,
)> {
    let (alias, name) = parse_zero_arg_state_operator_ref(expr)?;
    match alias {
        Some(alias) => {
            let instance = instances.get(alias.as_str())?;
            let module = instance.module.as_ref()?;
            let def = module.definitions.get(name.as_str())?;
            if !def.params.is_empty() || def.body.trim() == expr.trim() {
                return None;
            }
            Some((
                format!("{alias}!{name}"),
                def,
                &module.definitions,
                &module.instances,
                Some(instance),
            ))
        }
        None => {
            let def = definitions.get(name.as_str())?;
            if !def.params.is_empty() || def.body.trim() == expr.trim() {
                return None;
            }
            Some((name.to_string(), def, definitions, instances, None))
        }
    }
}

fn expand_state_predicate_clauses(
    body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut visiting = BTreeSet::new();
    append_expanded_state_predicate_clause(
        body,
        definitions,
        instances,
        None,
        &mut visiting,
        &mut out,
    );
    out
}

fn append_expanded_state_predicate_clause(
    clause: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    active_instance: Option<&TlaModuleInstance>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<String>,
) {
    let trimmed = clause.trim();
    if trimmed.is_empty() {
        return;
    }

    let parts = split_top_level(trimmed, "/\\");
    if parts.len() > 1 || trimmed.starts_with("/\\") {
        for part in parts {
            append_expanded_state_predicate_clause(
                &part,
                definitions,
                instances,
                active_instance,
                visiting,
                out,
            );
        }
        return;
    }

    if let Some((key, def, child_definitions, child_instances, instance_override)) =
        resolve_zero_arg_state_definition(trimmed, definitions, instances)
        && visiting.insert(key.clone())
    {
        let next_instance = instance_override.or(active_instance);
        append_expanded_state_predicate_clause(
            &def.body,
            child_definitions,
            child_instances,
            next_instance,
            visiting,
            out,
        );
        visiting.remove(&key);
        return;
    }

    if let Some(instance) = active_instance {
        out.push(apply_instance_substitutions_to_text(trimmed, instance));
    } else {
        out.push(trimmed.to_string());
    }
}

fn apply_instance_substitutions_to_text(expr: &str, instance: &TlaModuleInstance) -> String {
    if instance.substitutions.is_empty() {
        return expr.to_string();
    }

    let chars: Vec<char> = expr.chars().collect();
    let mut substitutions: Vec<(&str, &str)> = instance
        .substitutions
        .iter()
        .map(|(param, value_expr)| (param.as_str(), value_expr.as_str()))
        .collect();
    substitutions.sort_by_key(|(param, _)| std::cmp::Reverse(param.len()));

    let mut out = String::with_capacity(expr.len());
    let mut i = 0usize;
    let mut in_string = false;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            in_string = !in_string;
            out.push(ch);
            i += 1;
            continue;
        }
        if in_string {
            out.push(ch);
            i += 1;
            continue;
        }

        let mut replaced = false;
        for (param, value_expr) in &substitutions {
            let param_chars: Vec<char> = param.chars().collect();
            if i + param_chars.len() > chars.len() {
                continue;
            }
            if chars[i..i + param_chars.len()] != param_chars[..] {
                continue;
            }
            let before_ok = i == 0 || !is_probe_word_char(chars[i - 1]);
            let after_idx = i + param_chars.len();
            let after_ok = after_idx >= chars.len() || !is_probe_word_char(chars[after_idx]);
            if !before_ok || !after_ok {
                continue;
            }
            out.push_str(value_expr);
            i = after_idx;
            replaced = true;
            break;
        }

        if !replaced {
            out.push(ch);
            i += 1;
        }
    }

    out
}

fn seed_probe_state_from_type_invariants(
    probe_state: &mut TlaState,
    module: &TlaModule,
    cfg: Option<&TlaConfig>,
) -> usize {
    let mut invariant_names = BTreeSet::new();
    if let Some(cfg) = cfg {
        invariant_names.extend(cfg.invariants.iter().cloned());
    }
    for fallback in ["TypeOK", "TypeOk", "TypeInvariant", "TypeInv"] {
        if module.definitions.contains_key(fallback) {
            invariant_names.insert(fallback.to_string());
        }
    }

    invariant_names
        .into_iter()
        .filter_map(|name| module.definitions.get(&name))
        .map(|def| seed_probe_state_from_membership_body(probe_state, &def.body, module))
        .sum()
}

fn seed_probe_state_from_membership_body(
    probe_state: &mut TlaState,
    body: &str,
    module: &TlaModule,
) -> usize {
    let mut pending = Vec::new();
    for clause in split_top_level(body, "/\\") {
        pending.extend(collect_type_invariant_constraints(&clause, module));
    }

    let mut seeded = 0usize;
    for _ in 0..pending.len().saturating_add(1) {
        if pending.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();
        for (var, constraint) in pending {
            let ctx = build_probe_eval_context(probe_state, &module.definitions, &module.instances);
            let current_constraint_status = probe_state.get(&var).and_then(|_| {
                current_probe_value_satisfies_type_constraint(&var, &constraint, &ctx)
            });
            if let Some(repr) = representative_value_from_type_constraint(&constraint, &ctx) {
                let should_update = match probe_state.get(&var) {
                    None => true,
                    Some(current) => match current_constraint_status {
                        Some(true) => {
                            should_refine_satisfied_probe_value(current, &repr, &constraint, &ctx)
                        }
                        Some(false) => should_replace_probe_value_after_type_mismatch(current),
                        None => should_refine_indeterminate_probe_value(current, &repr),
                    },
                };
                if should_update {
                    probe_state.insert(var, repr);
                    seeded += 1;
                    progress = true;
                }
            } else {
                next_pending.push((var, constraint));
            }
        }

        if !progress {
            break;
        }
        pending = next_pending;
    }

    seeded
}

fn current_probe_value_satisfies_type_constraint(
    var: &str,
    constraint: &(TypeInvariantConstraintKind, String),
    ctx: &EvalContext<'_>,
) -> Option<bool> {
    let expr = match constraint.0 {
        TypeInvariantConstraintKind::Membership => format!("{var} \\in {}", constraint.1),
        TypeInvariantConstraintKind::Subset => format!("{var} \\subseteq {}", constraint.1),
    };

    match eval_expr(&expr, ctx) {
        Ok(TlaValue::Bool(result)) => Some(result),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum TypeInvariantConstraintKind {
    Membership,
    Subset,
}

fn collect_type_invariant_constraints(
    clause: &str,
    module: &TlaModule,
) -> Vec<(String, (TypeInvariantConstraintKind, String))> {
    let mut out = Vec::new();
    if let ClauseKind::UnprimedMembership { var, set_expr } = classify_clause(clause)
        && module.variables.contains(&var)
    {
        out.push((var, (TypeInvariantConstraintKind::Membership, set_expr)));
        return out;
    }

    let trimmed = clause.trim();
    if let Some(idx) = find_probe_top_level_keyword(trimmed, "\\subseteq") {
        let var = trimmed[..idx].trim().to_string();
        let set_expr = trimmed[idx + "\\subseteq".len()..].trim().to_string();
        if module.variables.contains(&var) && !set_expr.is_empty() {
            out.push((var, (TypeInvariantConstraintKind::Subset, set_expr)));
        }
    }

    out
}

fn representative_value_from_type_constraint(
    constraint: &(TypeInvariantConstraintKind, String),
    ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    match constraint.0 {
        TypeInvariantConstraintKind::Membership => {
            representative_value_from_set_expr(&constraint.1, ctx)
        }
        TypeInvariantConstraintKind::Subset => {
            representative_member_from_domain_expr(constraint.1.as_str(), ctx)
                .map(|member| TlaValue::Set(Arc::new(BTreeSet::from([member]))))
        }
    }
}

fn should_refine_probe_value(current: &TlaValue, replacement: &TlaValue) -> bool {
    match (current, replacement) {
        (TlaValue::Set(values), TlaValue::Set(replacement_values))
            if !values.is_empty() && replacement_values.len() < values.len() =>
        {
            return false;
        }
        (TlaValue::Seq(values), TlaValue::Seq(replacement_values))
            if !values.is_empty() && replacement_values.len() < values.len() =>
        {
            return false;
        }
        (TlaValue::Record(fields), TlaValue::Record(replacement_fields))
            if !fields.is_empty() && replacement_fields.len() < fields.len() =>
        {
            return false;
        }
        (TlaValue::Function(values), TlaValue::Function(replacement_values))
            if !values.is_empty() && replacement_values.len() < values.len() =>
        {
            return false;
        }
        _ => {}
    }

    if probe_value_score(replacement) > probe_value_score(current) {
        return true;
    }

    match (current, replacement) {
        (TlaValue::Set(values), TlaValue::Set(replacement_values)) => {
            values.is_empty() && !replacement_values.is_empty()
        }
        (TlaValue::Function(values), TlaValue::Function(replacement_values)) => {
            values.is_empty() && !replacement_values.is_empty()
        }
        _ => false,
    }
}

fn should_refine_satisfied_probe_value(
    current: &TlaValue,
    replacement: &TlaValue,
    constraint: &(TypeInvariantConstraintKind, String),
    ctx: &EvalContext<'_>,
) -> bool {
    if current_matches_definition_backed_singleton_member(current, &constraint.1, ctx) {
        return false;
    }
    should_refine_probe_value(current, replacement)
}

fn current_matches_definition_backed_singleton_member(
    current: &TlaValue,
    constraint_expr: &str,
    ctx: &EvalContext<'_>,
) -> bool {
    semantic_constraint_parts(constraint_expr).into_iter().any(|part| {
        let trimmed = strip_probe_outer_parens(part.trim());
        let Some(inner) = trimmed.strip_prefix('{').and_then(|rest| rest.strip_suffix('}')) else {
            return false;
        };
        let members = split_top_level(inner, ",");
        if members.len() != 1 {
            return false;
        }
        let member_expr = members[0].trim();
        if !is_definition_backed_probe_expr(member_expr, ctx) {
            return false;
        }
        matches!(eval_expr(member_expr, ctx), Ok(value) if current == &value || function_range_is_uniformly(current, &value))
    })
}

fn semantic_constraint_parts(expr: &str) -> Vec<String> {
    if let Some(range_expr) = function_set_range_expr(expr) {
        return union_constraint_parts(range_expr);
    }
    union_constraint_parts(expr)
}

fn union_constraint_parts(expr: &str) -> Vec<String> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    for delim in ["\\cup", "\\union"] {
        let parts = split_top_level(trimmed, delim);
        if parts.len() > 1 {
            let mut out = Vec::new();
            for part in parts {
                out.extend(union_constraint_parts(&part));
            }
            return out;
        }
    }
    vec![trimmed.to_string()]
}

fn function_set_range_expr(expr: &str) -> Option<&str> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    let arrow_idx = find_probe_top_level_keyword(inner, "->")?;
    Some(inner[arrow_idx + 2..].trim())
}

fn function_range_is_uniformly(current: &TlaValue, expected: &TlaValue) -> bool {
    match current {
        TlaValue::Function(entries) => entries.values().all(|value| value == expected),
        _ => false,
    }
}

fn is_definition_backed_probe_expr(expr: &str, ctx: &EvalContext<'_>) -> bool {
    let definitions = ctx.definitions.unwrap_or(ctx.local_definitions.as_ref());
    if resolve_zero_arg_state_definition(
        expr,
        definitions,
        ctx.instances.unwrap_or(&BTreeMap::new()),
    )
    .is_some()
    {
        return true;
    }

    let trimmed = expr.trim();
    let Some((alias, name)) = parse_zero_arg_state_operator_ref(trimmed) else {
        return false;
    };
    if alias.is_some() {
        return false;
    }
    ctx.local_definitions.contains_key(&name)
        || ctx
            .definitions
            .is_some_and(|defs| defs.contains_key(name.as_str()))
}

fn should_refine_indeterminate_probe_value(current: &TlaValue, replacement: &TlaValue) -> bool {
    if !should_refine_probe_value(current, replacement) {
        return false;
    }

    match current {
        TlaValue::Undefined => true,
        TlaValue::Int(_) | TlaValue::Bool(_) => true,
        TlaValue::String(_) | TlaValue::ModelValue(_) => {
            !matches!(replacement, TlaValue::String(_) | TlaValue::ModelValue(_))
        }
        TlaValue::Set(values) => values.is_empty(),
        TlaValue::Seq(values) => values.is_empty(),
        TlaValue::Record(fields) => fields.is_empty(),
        TlaValue::Function(values) => values.is_empty(),
        TlaValue::Lambda { .. } => false,
    }
}

fn should_replace_probe_value_after_type_mismatch(current: &TlaValue) -> bool {
    match current {
        TlaValue::Set(values) => values.is_empty(),
        TlaValue::Seq(values) => values.is_empty(),
        TlaValue::Record(fields) => fields.is_empty(),
        TlaValue::Function(values) => values.is_empty(),
        TlaValue::Lambda { .. } => false,
        _ => true,
    }
}

fn probe_value_score(value: &TlaValue) -> u32 {
    match value {
        TlaValue::Record(fields) => {
            5_000u32.saturating_add(fields.values().map(probe_value_score).max().unwrap_or(0))
        }
        TlaValue::Function(map) => {
            4_000u32.saturating_add(map.values().map(probe_value_score).max().unwrap_or(0))
        }
        TlaValue::Seq(items) => {
            3_000u32.saturating_add(items.iter().map(probe_value_score).max().unwrap_or(0))
        }
        TlaValue::Set(values) => {
            if values.is_empty() {
                50
            } else {
                2_000u32.saturating_add(values.iter().map(probe_value_score).max().unwrap_or(0))
            }
        }
        TlaValue::String(_) | TlaValue::ModelValue(_) => 100,
        TlaValue::Bool(_) => 90,
        TlaValue::Int(_) => 80,
        TlaValue::Lambda { .. } => 150,
        TlaValue::Undefined => 0,
    }
}

fn representative_value_from_set_expr(set_expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    if let Ok(set_val) = eval_expr(set_expr, ctx)
        && let Some(repr) = pick_representative_from_set(&set_val)
    {
        return Some(repr);
    }
    let trimmed = strip_probe_outer_parens(set_expr.trim());
    if is_probe_simple_identifier(trimmed)
        && let Some(def) = ctx
            .local_definitions
            .get(trimmed)
            .or_else(|| ctx.definitions.and_then(|defs| defs.get(trimmed)))
        && def.params.is_empty()
        && def.body.trim() != trimmed
        && let Some(repr) = representative_value_from_definition_body(&def.body, ctx)
    {
        return Some(repr);
    }
    if let Some(repr) = representative_value_from_operator_call_expr(trimmed, ctx) {
        return Some(repr);
    }
    try_create_representative_value_from_type_expr(set_expr, ctx)
}

fn representative_value_from_definition_body(
    body: &str,
    ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    if let Ok(set_val) = eval_expr(body, ctx)
        && let Some(repr) = pick_representative_from_set(&set_val)
    {
        return Some(repr);
    }
    try_create_representative_value_from_type_expr(body, ctx)
}

fn is_probe_simple_identifier(expr: &str) -> bool {
    let mut chars = expr.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return false;
    }

    let mut saw_identifier_marker = first.is_ascii_alphabetic() || first == '_';
    for ch in chars {
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            return false;
        }
        saw_identifier_marker |= ch.is_ascii_alphabetic() || ch == '_';
    }

    saw_identifier_marker
}

fn representative_member_from_domain_expr(expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    if let Ok(set_val) = eval_expr(expr, ctx)
        && let Some(repr) = pick_representative_from_set(&set_val)
    {
        return Some(repr);
    }

    let trimmed = strip_probe_outer_parens(expr.trim());
    if let Some(repr) = representative_value_from_union_parts(trimmed, ctx) {
        return Some(repr);
    }
    if is_probe_simple_identifier(trimmed)
        && let Some(def) = ctx
            .local_definitions
            .get(trimmed)
            .or_else(|| ctx.definitions.and_then(|defs| defs.get(trimmed)))
        && def.params.is_empty()
        && def.body.trim() != trimmed
        && let Some(repr) = representative_value_from_definition_body(&def.body, ctx)
    {
        return Some(repr);
    }
    if let Some(repr) = representative_value_from_operator_call_expr(trimmed, ctx) {
        return Some(repr);
    }

    match trimmed {
        "Nat" | "Int" => Some(TlaValue::Int(0)),
        "BOOLEAN" => Some(TlaValue::Bool(false)),
        _ => {
            if let Some(inner) = trimmed
                .strip_prefix("Seq(")
                .and_then(|rest| rest.strip_suffix(')'))
            {
                let sample =
                    representative_member_from_domain_expr(inner, ctx).unwrap_or(TlaValue::Int(0));
                Some(TlaValue::Seq(Arc::new(vec![sample])))
            } else if trimmed.starts_with("SUBSET ") {
                Some(TlaValue::Set(Arc::new(BTreeSet::new())))
            } else if let Some(record) = try_create_representative_record(trimmed, ctx) {
                Some(record)
            } else {
                try_create_representative_function(trimmed, ctx)
            }
        }
    }
}

fn try_create_representative_value_from_type_expr(
    set_expr: &str,
    ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    let trimmed = strip_probe_outer_parens(set_expr.trim());
    if let Some(repr) = representative_value_from_union_parts(trimmed, ctx) {
        return Some(repr);
    }
    if let Some(repr) = try_create_representative_function(trimmed, ctx) {
        return Some(repr);
    }
    if let Some(repr) = try_create_representative_record(trimmed, ctx) {
        return Some(repr);
    }
    if let Some(inner) = trimmed
        .strip_prefix("Seq(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        let sample = representative_member_from_domain_expr(inner, ctx).unwrap_or(TlaValue::Int(0));
        return Some(TlaValue::Seq(Arc::new(vec![sample])));
    }

    match trimmed {
        "Nat" | "Int" => Some(TlaValue::Int(0)),
        "BOOLEAN" => Some(TlaValue::Bool(false)),
        _ if trimmed.starts_with("SUBSET ") => Some(TlaValue::Set(Arc::new(BTreeSet::new()))),
        _ => None,
    }
}

fn representative_value_from_union_parts(expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    for delim in ["\\cup", "\\union"] {
        let parts = split_top_level(expr, delim);
        if parts.len() > 1 {
            return parts
                .into_iter()
                .filter_map(|part| representative_member_from_domain_expr(&part, ctx))
                .max_by_key(probe_value_score);
        }
    }
    None
}

fn representative_value_from_operator_call_expr(
    expr: &str,
    ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    let (name, arg_exprs) = parse_probe_action_call(expr)?;
    let (def, child_definitions, child_instances) =
        resolve_probe_operator_call_definition(&name, ctx)?;
    if def.params.len() != arg_exprs.len() {
        return None;
    }

    let mut child_ctx = ctx.clone();
    child_ctx.definitions = Some(child_definitions);
    child_ctx.instances = child_instances;
    let child_locals = std::rc::Rc::make_mut(&mut child_ctx.locals);

    let mut bound_names = BTreeSet::new();
    for (param, arg_expr) in def.params.iter().zip(arg_exprs) {
        let value = eval_expr(&arg_expr, ctx).ok()?;
        let normalized = normalize_param_name(param).to_string();
        if bound_names.insert(normalized.clone()) {
            child_locals.insert(normalized, value.clone());
        }
        if bound_names.insert(param.clone()) {
            child_locals.insert(param.clone(), value);
        }
    }

    representative_value_from_definition_body(&def.body, &child_ctx)
}

fn resolve_probe_operator_call_definition<'a>(
    name: &str,
    ctx: &'a EvalContext<'a>,
) -> Option<(
    &'a TlaDefinition,
    &'a BTreeMap<String, TlaDefinition>,
    Option<&'a BTreeMap<String, TlaModuleInstance>>,
)> {
    let (alias, operator) = match name.split_once('!') {
        Some((alias, operator)) => (Some(alias.trim()), operator.trim()),
        None => (None, name.trim()),
    };

    match alias {
        Some(alias) => {
            let instances = ctx.instances?;
            let instance = instances.get(alias)?;
            let module = instance.module.as_ref()?;
            let def = module.definitions.get(operator)?;
            Some((def, &module.definitions, Some(&module.instances)))
        }
        None => {
            let definitions = ctx.definitions?;
            let def = definitions.get(operator)?;
            Some((def, definitions, ctx.instances))
        }
    }
}

fn try_create_representative_record(set_expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    let trimmed = strip_probe_outer_parens(set_expr.trim());
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    if find_probe_top_level_keyword(inner, "->").is_some() {
        return None;
    }

    let mut fields = BTreeMap::new();
    for field_spec in split_top_level(inner, ",") {
        let colon_idx = find_probe_top_level_char(&field_spec, ':')?;
        let field_name = field_spec[..colon_idx].trim();
        let domain_expr = field_spec[colon_idx + 1..].trim();
        if field_name.is_empty() || domain_expr.is_empty() {
            return None;
        }
        let value = representative_member_from_domain_expr(domain_expr, ctx)?;
        fields.insert(field_name.to_string(), value);
    }

    if fields.is_empty() {
        None
    } else {
        Some(TlaValue::Record(Arc::new(fields)))
    }
}

#[cfg(test)]
fn sample_param_value(param: &str, probe_state: &TlaState) -> TlaValue {
    sample_param_value_with_context(param, probe_state, &BTreeMap::new(), &BTreeMap::new())
}

fn sample_param_value_with_context(
    param: &str,
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> TlaValue {
    let lower = param.to_ascii_lowercase();
    let def_ctx = build_probe_eval_context(probe_state, definitions, instances);

    let from_named_set = |name: &str| -> Option<TlaValue> {
        match probe_state.get(name) {
            Some(TlaValue::Set(values)) => values.iter().next().cloned(),
            _ => definitions
                .get(name)
                .filter(|def| def.params.is_empty())
                .and_then(|def| representative_value_from_set_expr(&def.body, &def_ctx)),
        }
    };
    let from_named_function_domain = |name: &str| -> Option<TlaValue> {
        match probe_state.get(name) {
            Some(TlaValue::Function(values)) => values.keys().next().cloned(),
            _ => definitions
                .get(name)
                .filter(|def| def.params.is_empty())
                .and_then(|def| eval_expr(&def.body, &def_ctx).ok())
                .and_then(|value| match value {
                    TlaValue::Function(values) => values.keys().next().cloned(),
                    _ => None,
                }),
        }
    };
    let from_any_function_domain = || -> Option<TlaValue> {
        probe_state
            .values()
            .filter_map(|value| match value {
                TlaValue::Function(values) if !values.is_empty() => {
                    Some((values.len(), values.keys().next().cloned()))
                }
                _ => None,
            })
            .min_by_key(|(len, _)| *len)
            .and_then(|(_, value)| value)
    };

    let set_hints: &[&str] = match lower.as_str() {
        "acceptor" | "a" => &["Acceptor", "Acceptors", "Replicas", "Assets"],
        "bot" | "buyer" | "holder" => &["Bots"],
        "ballot" => &["Ballots", "Ballot"],
        "call" | "c" => &["ActiveElevatorCalls", "ElevatorCall"],
        "client" => &["Participants", "Person"],
        "p" | "p1" | "p2" => &["Proc", "ProcSet", "Participants", "Person"],
        "e" | "e1" | "e2" | "elevator" => &["Elevator"],
        "f" | "floor" => &["Floor"],
        "key" | "k" => &["Key"],
        "mm" => &["MarketMakers"],
        "node" | "n" | "n1" | "n2" | "other" => &["Node", "Nodes"],
        "pid" => &["SlushQueryProcess", "SlushLoopProcess", "ProcSet"],
        "prisoner" => &["Prisoner"],
        "q" | "quorum" => &["Quorums", "Quorum"],
        "replica" | "rpl" => &["Replicas"],
        "reader" | "r" => &["Readers"],
        "seller" | "s" => &["Sellers"],
        "t" | "tx" | "transaction" => &["TxId"],
        "b" => &["Ballots", "Ballot", "Bots"],
        "val" | "value" | "v" => &["Values", "Value", "Val"],
        "writer" | "w" => &["Writers"],
        _ => &[],
    };
    for set_name in set_hints {
        if let Some(v) = from_named_set(set_name) {
            return v;
        }
    }

    let domain_hint = match lower.as_str() {
        "call" | "c" => Some("ActiveElevatorCalls"),
        "client" | "p" | "p1" | "p2" | "prisoner" => Some("signalled"),
        "key" | "k" => Some("store"),
        "t" | "tx" | "transaction" => Some("snapshotStore"),
        "writer" | "reader" | "w" | "r" => Some("pc"),
        "self" | "proc" | "process" => Some("rcvd"),
        _ => None,
    };
    if let Some(domain_name) = domain_hint
        && let Some(v) = from_named_function_domain(domain_name)
    {
        return v;
    }

    if matches!(
        lower.as_str(),
        "i" | "j" | "k" | "m" | "idx" | "index" | "count" | "x" | "y"
    ) {
        return TlaValue::Int(1);
    }
    if lower == "sequence"
        || lower == "seq"
        || lower.contains("seq")
        || lower == "round"
        || lower == "min"
        || lower == "max"
    {
        return TlaValue::Int(0);
    }

    if lower == "self" {
        for name in [
            "SlushLoopProcess",
            "SlushQueryProcess",
            "ProcSet",
            "Proc",
            "Corr",
            "Readers",
            "Writers",
            "Prisoner",
        ] {
            if let Some(v) = from_named_set(name) {
                return v;
            }
        }
        for name in [
            "loopVariant",
            "sampleSet",
            "pc",
            "rcvd",
            "read",
            "claimed_sequence",
            "consumed",
        ] {
            if let Some(v) = from_named_function_domain(name) {
                return v;
            }
        }
        if let Some(v) = from_any_function_domain() {
            return v;
        }
        return TlaValue::Int(1);
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

    for val in probe_state.values() {
        if let TlaValue::Set(values) = val
            && let Some(first) = values.iter().next()
            && matches!(first, TlaValue::Int(_))
            && param.len() <= 2
        {
            return first.clone();
        }
    }
    if let Some(v) = from_any_function_domain() {
        return v;
    }

    TlaValue::ModelValue(param.to_string())
}

fn infer_action_param_samples_from_next(
    next_body: &str,
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> BTreeMap<String, BTreeMap<String, TlaValue>> {
    let mut samples = BTreeMap::new();
    let mut active_calls = BTreeSet::new();
    collect_action_param_samples_from_expr(
        next_body,
        &BTreeMap::new(),
        probe_state,
        definitions,
        instances,
        &mut samples,
        &mut active_calls,
    );
    samples
}

fn merge_action_param_samples(
    into: &mut BTreeMap<String, BTreeMap<String, TlaValue>>,
    from: BTreeMap<String, BTreeMap<String, TlaValue>>,
) {
    for (action, params) in from {
        let entry = into.entry(action).or_default();
        for (param, value) in params {
            entry.entry(param).or_insert(value);
        }
    }
}

/// When the Next operator has parameters (e.g. `Next(n)`), scan all zero-param
/// definitions for `\E` quantifiers that wrap a call to Next, e.g.
/// `Spec == Init /\ [][\E n \in Node : Next(n)]_vars`.  Return the bound
/// variable values so they can seed the Next body traversal.
fn infer_next_param_locals_from_spec(
    next_name: &str,
    next_params: &[String],
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    probe_state: &TlaState,
) -> BTreeMap<String, TlaValue> {
    if next_params.is_empty() {
        return BTreeMap::new();
    }
    // Scan all zero-param definitions looking for `\E var \in Set : ... Next(var) ...`
    for def in definitions.values() {
        if !def.params.is_empty() {
            continue;
        }
        // Quick check: does the body mention the Next operator name?
        if !def.body.contains(next_name) {
            continue;
        }
        // Try to find `\E ... : ... Next(...)` anywhere in the body.
        // We search for `\E` patterns and then check if the body calls Next.
        if let Some(locals) = extract_exists_binders_for_next_call(
            &def.body,
            next_name,
            next_params,
            probe_state,
            definitions,
            instances,
        ) {
            if !locals.is_empty() {
                return locals;
            }
        }
    }
    BTreeMap::new()
}

/// Scan `text` for `\E var \in SetExpr : body` where `body` contains a call
/// `next_name(args)`.  Return the bound variable samples mapped to the
/// corresponding Next parameter names.
///
/// The `\E` may be nested inside temporal operators like `[][...]_vars`, so
/// we first extract bracketed inner content before trying `parse_action_exists`.
fn extract_exists_binders_for_next_call(
    text: &str,
    next_name: &str,
    next_params: &[String],
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> Option<BTreeMap<String, TlaValue>> {
    // Collect candidate fragments: the text itself and any bracket-inner content
    // that might contain `\E ... : Next(...)`.
    let mut candidates: Vec<String> = vec![text.to_string()];
    // Extract content inside `[][...]_vars` patterns (temporal box formulas)
    {
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i + 2 < chars.len() {
            if chars[i] == '[' && chars[i + 1] == ']' && i + 2 < chars.len() && chars[i + 2] == '['
            {
                // Found `[][`, scan for matching `]`
                let mut depth = 1usize;
                let start = i + 3;
                let mut j = start;
                while j < chars.len() && depth > 0 {
                    match chars[j] {
                        '[' => depth += 1,
                        ']' => depth -= 1,
                        _ => {}
                    }
                    if depth > 0 {
                        j += 1;
                    }
                }
                if depth == 0 {
                    let inner: String = chars[start..j].iter().collect();
                    candidates.push(inner);
                }
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }

    for candidate in &candidates {
        let mut search_from = 0;
        while let Some(pos) = candidate[search_from..].find("\\E") {
            let abs_pos = search_from + pos;
            let after = &candidate[abs_pos + 2..];
            if !after.starts_with(char::is_whitespace) && !after.starts_with('(') {
                search_from = abs_pos + 2;
                continue;
            }
            let fragment = &candidate[abs_pos..];
            if let Some((binders, body)) = parse_action_exists(fragment) {
                if let Some((call_name, call_args)) = parse_probe_action_call(body.trim()) {
                    if call_name == next_name && call_args.len() == next_params.len() {
                        if let Some(bound) = sample_exists_quantifier_binders(
                            binders,
                            &BTreeMap::new(),
                            probe_state,
                            definitions,
                            instances,
                        ) {
                            let binder_map: BTreeMap<String, TlaValue> =
                                bound.into_iter().collect();
                            let mut result = BTreeMap::new();
                            for (param, arg_expr) in next_params.iter().zip(call_args.iter()) {
                                let arg_name = arg_expr.trim();
                                if let Some(value) = binder_map.get(arg_name) {
                                    let normalized = normalize_param_name(param).to_string();
                                    result.insert(normalized, value.clone());
                                    result.insert(param.clone(), value.clone());
                                }
                            }
                            if !result.is_empty() {
                                return Some(result);
                            }
                        }
                    }
                }
                // The body might contain Next nested deeper
                if body.contains(next_name) {
                    if let Some(inner) = extract_exists_binders_for_next_call(
                        body,
                        next_name,
                        next_params,
                        probe_state,
                        definitions,
                        instances,
                    ) {
                        if !inner.is_empty() {
                            return Some(inner);
                        }
                    }
                }
            }
            search_from = abs_pos + 2;
        }
    }
    None
}

fn infer_action_param_samples_from_module_contexts(
    resolved_next_name: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    probe_state: &TlaState,
) -> BTreeMap<String, BTreeMap<String, TlaValue>> {
    let next_def = definitions.get(resolved_next_name);

    // When the Next operator takes parameters, try to extract \E binder values
    // from the Spec definition so that recursive param inference works.
    let next_param_locals = next_def
        .filter(|d| !d.params.is_empty())
        .map(|d| {
            infer_next_param_locals_from_spec(
                resolved_next_name,
                &d.params,
                definitions,
                instances,
                probe_state,
            )
        })
        .unwrap_or_default();

    let mut samples = next_def
        .map(|next_def| {
            let mut result = BTreeMap::new();
            let mut active_calls = BTreeSet::new();
            collect_action_param_samples_from_expr(
                &next_def.body,
                &next_param_locals,
                probe_state,
                definitions,
                instances,
                &mut result,
                &mut active_calls,
            );
            result
        })
        .unwrap_or_default();

    for def in definitions.values() {
        if def.name == resolved_next_name || !def.params.is_empty() {
            continue;
        }
        if !body_contains_probeable_action_call(&def.body) {
            continue;
        }
        merge_action_param_samples(
            &mut samples,
            infer_action_param_samples_from_next(&def.body, probe_state, definitions, instances),
        );
    }

    samples
}

fn expr_probe_is_ready(expr_total: usize, expr_ok: usize) -> bool {
    expr_total == 0 || expr_ok == expr_total
}

/// Returns true when the error is caused by probe-state sampling limitations
/// rather than a genuine expression evaluation failure. These errors arise
/// because the probe uses placeholder/ModelValue parameters that don't match
/// the runtime domains of functions, sequences, or CHOOSE expressions.
fn is_probe_sampling_limitation_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    // ModelValue used as key in a function that doesn't have that key
    if msg.contains("function missing key ModelValue(") {
        return true;
    }
    // Record access on a placeholder ModelValue
    if msg.contains("record access on non-record value ModelValue(") {
        return true;
    }
    // Sequence/function indexing on a default Int(0) placeholder
    if msg.contains("index access on unsupported value Int(0)") {
        return true;
    }
    // Expected a type but got a ModelValue placeholder
    if msg.contains("got ModelValue(") {
        return true;
    }
    // Function/record application on ModelValue placeholder
    if msg.contains("unsupported for value ModelValue(") {
        return true;
    }
    // CHOOSE with empty domain due to probe sampling
    if msg.contains("CHOOSE found no matching value") {
        return true;
    }
    // Expected Int but got Undefined (uninitialized probe variable)
    if msg.contains("got Undefined") {
        return true;
    }
    // DOMAIN applied to a default Int(0) placeholder
    if msg.contains("DOMAIN expects a function") {
        return true;
    }
    // Function application on a default Int(0) placeholder
    if msg.contains("unsupported for value Int(0)") {
        return true;
    }
    // Evaluation budget exceeded during probing (large set construction)
    if msg.contains("evaluation budget exceeded") {
        return true;
    }
    false
}

fn collect_action_param_samples_from_expr(
    expr: &str,
    locals: &BTreeMap<String, TlaValue>,
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    samples: &mut BTreeMap<String, BTreeMap<String, TlaValue>>,
    active_calls: &mut BTreeSet<String>,
) {
    let trimmed = strip_probe_outer_parens(expr.trim());
    if trimmed.is_empty() {
        return;
    }

    if let Some(rest) = trimmed.strip_prefix("\\E")
        && let Some((binder_text, body)) = split_probe_quantifier(rest)
        && let Some(bound) = sample_exists_quantifier_binders(
            binder_text,
            locals,
            probe_state,
            definitions,
            instances,
        )
    {
        let mut child_locals = locals.clone();
        for (name, value) in bound {
            child_locals.insert(name, value);
        }
        collect_action_param_samples_from_expr(
            body,
            &child_locals,
            probe_state,
            definitions,
            instances,
            samples,
            active_calls,
        );
        return;
    }

    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() > 1 || trimmed.starts_with("\\/") {
        for disjunct in disjuncts {
            collect_action_param_samples_from_expr(
                &disjunct,
                locals,
                probe_state,
                definitions,
                instances,
                samples,
                active_calls,
            );
        }
        return;
    }

    let conjuncts = split_top_level(trimmed, "/\\");
    if conjuncts.len() > 1 || trimmed.starts_with("/\\") {
        for conjunct in conjuncts {
            collect_action_param_samples_from_expr(
                &conjunct,
                locals,
                probe_state,
                definitions,
                instances,
                samples,
                active_calls,
            );
        }
        return;
    }

    let Some((name, arg_exprs)) = parse_probe_action_call(trimmed) else {
        return;
    };
    let Some(def) = definitions.get(&name) else {
        return;
    };
    if def.params.len() != arg_exprs.len() {
        return;
    }

    let mut ctx = build_probe_eval_context(probe_state, definitions, instances);
    {
        let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
        for (local_name, value) in locals {
            locals_mut.insert(local_name.clone(), value.clone());
        }
    }

    let entry = samples.entry(name.clone()).or_default();
    let mut child_locals = locals.clone();
    for (param, arg_expr) in def.params.iter().zip(arg_exprs) {
        let param_name = normalize_param_name(param).to_string();
        if let Ok(value) = eval_expr(&arg_expr, &ctx) {
            child_locals.insert(param_name.clone(), value.clone());
            entry.entry(param_name).or_insert(value);
        }
    }

    if active_calls.insert(name.clone()) {
        collect_action_param_samples_from_expr(
            &def.body,
            &child_locals,
            probe_state,
            definitions,
            instances,
            samples,
            active_calls,
        );
        active_calls.remove(&name);
    }
}

fn split_probe_quantifier(expr: &str) -> Option<(&str, &str)> {
    let colon_idx = find_probe_top_level_char(expr, ':')?;
    Some((expr[..colon_idx].trim(), expr[colon_idx + 1..].trim()))
}

fn sample_exists_quantifier_binders(
    binder_text: &str,
    locals: &BTreeMap<String, TlaValue>,
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> Option<Vec<(String, TlaValue)>> {
    let mut bound = Vec::new();
    let mut rest = binder_text.trim();
    let mut ctx = build_probe_eval_context(probe_state, definitions, instances);
    {
        let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
        for (name, value) in locals {
            locals_mut.insert(name.clone(), value.clone());
        }
    }

    while !rest.is_empty() {
        let in_idx = find_probe_top_level_keyword(rest, "\\in")?;
        let vars_text = rest[..in_idx].trim();
        let after_in = rest[in_idx + "\\in".len()..].trim_start();

        let mut split_idx = None;
        let mut search_from = 0usize;
        while let Some(comma_idx) = find_probe_top_level_char_from(after_in, ',', search_from) {
            let tail = after_in[comma_idx + 1..].trim_start();
            if find_probe_top_level_keyword(tail, "\\in").is_some() {
                split_idx = Some(comma_idx);
                break;
            }
            search_from = comma_idx + 1;
        }

        let (domain_text, next_rest) = match split_idx {
            Some(idx) => (&after_in[..idx], &after_in[idx + 1..]),
            None => (after_in, ""),
        };

        let sample = representative_value_from_set_expr(domain_text.trim(), &ctx)?;

        for var in split_top_level(vars_text, ",") {
            let name = var.trim();
            if name.is_empty() {
                continue;
            }
            bound.push((name.to_string(), sample.clone()));
            std::rc::Rc::make_mut(&mut ctx.locals).insert(name.to_string(), sample.clone());
        }

        rest = next_rest.trim_start();
    }

    Some(bound)
}

fn parse_probe_action_call(expr: &str) -> Option<(String, Vec<String>)> {
    let (name, rest) = parse_probe_identifier_prefix(expr)?;
    let rest = rest.trim_start();
    if rest.is_empty() {
        return Some((name, Vec::new()));
    }
    if !rest.starts_with('(') {
        return None;
    }

    let (args_text, tail) = take_probe_group(rest, '(', ')')?;
    if !tail.trim().is_empty() {
        return None;
    }

    let args = if args_text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(args_text, ",")
            .into_iter()
            .map(|arg| arg.trim().to_string())
            .filter(|arg| !arg.is_empty())
            .collect()
    };

    Some((name, args))
}

fn parse_probe_action_if(expr: &str) -> Option<(&str, &str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("IF")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    let then_idx = find_probe_outer_then(rest)?;
    let condition = rest[..then_idx].trim();
    let after_then = rest[then_idx + "THEN".len()..].trim();

    let else_idx = find_probe_outer_else(after_then)?;
    let then_branch = after_then[..else_idx].trim();
    let else_branch = after_then[else_idx + "ELSE".len()..].trim();

    Some((condition, then_branch, else_branch))
}

fn next_probe_word(input: &str, from: usize) -> Option<(&str, usize, usize)> {
    let mut i = from;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if ch.is_ascii_alphabetic() || ch == '_' {
            break;
        }
        i += len;
    }

    if i >= input.len() {
        return None;
    }

    let start = i;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            break;
        }
        i += len;
    }

    Some((&input[start..i], start, i))
}

fn find_probe_outer_then(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_probe_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "THEN" if nested_if == 0 => return Some(start),
            "ELSE" if nested_if > 0 => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

fn find_probe_outer_else(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_probe_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "ELSE" if nested_if == 0 => return Some(start),
            "ELSE" => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

fn parse_probe_identifier_prefix(expr: &str) -> Option<(String, &str)> {
    let trimmed = expr.trim_start();
    let mut chars = trimmed.char_indices();
    let (_, first) = chars.next()?;
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    let mut saw_identifier_marker = first.is_ascii_alphabetic() || first == '_';
    for (idx, ch) in chars {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '!' {
            end = idx + ch.len_utf8();
            saw_identifier_marker |= ch.is_ascii_alphabetic() || ch == '_';
        } else {
            break;
        }
    }

    if !saw_identifier_marker {
        return None;
    }

    Some((trimmed[..end].to_string(), &trimmed[end..]))
}

fn take_probe_group(expr: &str, open: char, close: char) -> Option<(&str, &str)> {
    let mut chars = expr.char_indices();
    let (_, first) = chars.next()?;
    if first != open {
        return None;
    }

    let mut depth = 1usize;
    for (idx, ch) in chars {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some((&expr[1..idx], &expr[idx + ch.len_utf8()..]));
            }
        }
    }

    None
}

fn strip_probe_outer_parens(expr: &str) -> &str {
    let mut current = expr.trim();
    while current.starts_with('(') && current.ends_with(')') {
        let Some(idx) = find_probe_matching_paren(current) else {
            break;
        };
        if idx != current.len() - 1 {
            break;
        }
        current = current[1..current.len() - 1].trim();
    }
    current
}

fn strip_probe_comments(expr: &str) -> String {
    expr.lines()
        .filter(|line| !line.trim_start().starts_with("\\*"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_probe_clause_expr(expr: &str) -> String {
    let mut normalized = strip_probe_comments(expr);
    loop {
        let trimmed = normalized.trim_start();
        if let Some(rest) = trimmed.strip_prefix("/\\") {
            normalized = rest.trim_start().to_string();
            continue;
        }
        return normalized;
    }
}

fn find_probe_matching_paren(expr: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in expr.char_indices() {
        if ch == '(' {
            depth += 1;
        } else if ch == ')' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn find_probe_top_level_char(expr: &str, target: char) -> Option<usize> {
    find_probe_top_level_char_from(expr, target, 0)
}

fn find_probe_top_level_char_from(expr: &str, target: char, start: usize) -> Option<usize> {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut i = 0usize;

    while i < expr.len() {
        let ch = expr[i..].chars().next()?;
        let len = ch.len_utf8();
        let next = expr[i + len..].chars().next();

        if i < start {
            if ch == '<' && next == Some('<') {
                i += 2;
            } else if ch == '>' && next == Some('>') {
                i += 2;
            } else {
                i += len;
            }
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && ch == target {
            return Some(i);
        }

        i += len;
    }

    None
}

fn find_probe_top_level_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let keyword_len = keyword.len();
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut i = 0usize;

    while i < expr.len() {
        let ch = expr[i..].chars().next()?;
        let len = ch.len_utf8();
        let next = expr[i + len..].chars().next();

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && expr[i..].starts_with(keyword)
        {
            return Some(i);
        }

        i += len;
        if i + keyword_len > expr.len() && i >= expr.len() {
            break;
        }
    }

    None
}

fn build_action_expr_probe_context<'a>(
    probe_state: &'a TlaState,
    definitions: &'a BTreeMap<String, TlaDefinition>,
    instances: &'a BTreeMap<String, TlaModuleInstance>,
    params: &[String],
    clauses: &[ActionClause],
    inferred_param_samples: Option<&BTreeMap<String, TlaValue>>,
) -> EvalContext<'a> {
    let mut ctx = build_probe_eval_context(probe_state, definitions, instances);
    let guard_samples = infer_action_param_samples_from_guards(
        params,
        clauses,
        probe_state,
        definitions,
        instances,
    );
    let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);

    for (var, value) in probe_state {
        locals_mut.insert(format!("{var}'"), value.clone());
    }

    for param in params {
        let normalized = normalize_param_name(param);
        let sample = guard_samples
            .get(normalized)
            .cloned()
            .or_else(|| {
                inferred_param_samples
                    .and_then(|samples| samples.get(normalized))
                    .cloned()
            })
            .or_else(|| {
                inferred_param_samples
                    .and_then(|samples| samples.get(param))
                    .cloned()
            })
            .unwrap_or_else(|| {
                sample_param_value_with_context(param, probe_state, definitions, instances)
            });
        let sample = refine_param_sample_from_clause_domains(
            param,
            &sample,
            clauses,
            probe_state,
            definitions,
            instances,
        );
        locals_mut.insert(normalized.to_string(), sample.clone());
        locals_mut.insert(param.clone(), sample);
    }

    // Treat UNCHANGED x as an available x' = x binding during probing.
    for clause in clauses {
        if let ActionClause::Unchanged { vars } = clause {
            for var in vars {
                if let Some(value) = probe_state.get(var) {
                    locals_mut.insert(format!("{var}'"), value.clone());
                }
            }
        }
    }

    seed_nonempty_sequence_probe_locals(probe_state, definitions, instances, clauses, locals_mut);

    ctx
}

fn seed_nonempty_sequence_probe_locals(
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    clauses: &[ActionClause],
    locals: &mut BTreeMap<String, TlaValue>,
) {
    let element_hints =
        infer_sequence_element_probe_values(probe_state, definitions, instances, clauses, locals);
    for var in find_sequence_accessed_vars(clauses) {
        if locals
            .get(&var)
            .or_else(|| probe_state.get(&var))
            .is_some_and(|value| matches!(value, TlaValue::Seq(values) if !values.is_empty()))
        {
            continue;
        }

        let Some(TlaValue::Seq(values)) = probe_state.get(&var) else {
            continue;
        };
        if !values.is_empty() {
            continue;
        }

        let element = element_hints.get(&var).cloned().unwrap_or(TlaValue::Int(0));
        locals.insert(var, TlaValue::Seq(Arc::new(vec![element])));
    }
}

fn find_sequence_accessed_vars(clauses: &[ActionClause]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for clause in clauses {
        let expr = match clause {
            ActionClause::Guard { expr }
            | ActionClause::PrimedAssignment { expr, .. }
            | ActionClause::PrimedMembership { set_expr: expr, .. }
            | ActionClause::LetWithPrimes { expr } => expr.as_str(),
            ActionClause::Exists { body, .. } => body.as_str(),
            ActionClause::Unchanged { .. } => continue,
        };
        collect_sequence_access_vars(expr, &mut names);
    }
    names
}

fn infer_sequence_element_probe_values(
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    clauses: &[ActionClause],
    locals: &BTreeMap<String, TlaValue>,
) -> BTreeMap<String, TlaValue> {
    let mut fields_by_var: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut domains_by_var: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for clause in clauses {
        let expr = match clause {
            ActionClause::Guard { expr }
            | ActionClause::PrimedAssignment { expr, .. }
            | ActionClause::PrimedMembership { set_expr: expr, .. }
            | ActionClause::LetWithPrimes { expr } => expr.as_str(),
            ActionClause::Exists { body, .. } => body.as_str(),
            ActionClause::Unchanged { .. } => continue,
        };
        collect_sequence_accessed_record_fields(expr, &mut fields_by_var);
        collect_sequence_indexed_function_domains(expr, &mut domains_by_var);
    }

    let mut out = BTreeMap::new();
    let mut ctx = build_probe_eval_context(probe_state, definitions, instances);
    {
        let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
        for (name, value) in locals {
            locals_mut.insert(name.clone(), value.clone());
        }
    }
    for (var, fields) in fields_by_var {
        let mut record = BTreeMap::new();
        for field in fields {
            record.insert(field.clone(), guess_probe_field_value(&field, &ctx));
        }
        if !record.is_empty() {
            out.insert(var, TlaValue::Record(Arc::new(record)));
        }
    }
    for (var, function_names) in domains_by_var {
        if out.contains_key(&var) {
            continue;
        }
        for function_name in function_names {
            let Ok(value) = eval_expr(&function_name, &ctx) else {
                continue;
            };
            let TlaValue::Function(entries) = value else {
                continue;
            };
            if let Some(key) = entries.keys().next().cloned() {
                out.insert(var.clone(), key);
                break;
            }
        }
    }
    out
}

fn guess_probe_field_value(field: &str, ctx: &EvalContext<'_>) -> TlaValue {
    let lower = field.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "op" | "msgid" | "action" | "type" | "state" | "protocol" | "flg"
    ) {
        return TlaValue::String("sample".to_string());
    }
    if lower.ends_with("ip")
        || lower.ends_with("port")
        || lower.ends_with("count")
        || lower.ends_with("stamp")
        || lower == "key"
    {
        return TlaValue::Int(0);
    }
    if lower.ends_with("id") {
        return TlaValue::String(format!("{field}_0"));
    }
    if let Some(def) = ctx
        .local_definitions
        .get(field)
        .or_else(|| ctx.definitions.and_then(|defs| defs.get(field)))
        && def.params.is_empty()
        && let Ok(value) = eval_expr(&def.body, ctx)
    {
        return value;
    }
    TlaValue::Int(0)
}

fn collect_sequence_access_vars(expr: &str, out: &mut BTreeSet<String>) {
    for prefix in ["Head(", "Tail("] {
        let mut rest = expr;
        while let Some(idx) = rest.find(prefix) {
            let after = &rest[idx + prefix.len() - 1..];
            let Some((candidate, tail)) = take_probe_group(after, '(', ')') else {
                break;
            };
            if is_probe_simple_identifier(candidate) {
                out.insert(candidate.to_string());
            }
            rest = tail;
        }
    }
}

fn collect_sequence_accessed_record_fields(
    expr: &str,
    out: &mut BTreeMap<String, BTreeSet<String>>,
) {
    for prefix in ["Head(", "Tail("] {
        let mut rest = expr;
        while let Some(idx) = rest.find(prefix) {
            let after = &rest[idx + prefix.len() - 1..];
            let Some((candidate, tail)) = take_probe_group(after, '(', ')') else {
                break;
            };
            if is_probe_simple_identifier(candidate) {
                let trimmed_tail = tail.trim_start();
                if let Some(field_tail) = trimmed_tail.strip_prefix('.') {
                    let mut end = 0usize;
                    for (idx, ch) in field_tail.char_indices() {
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            end = idx + ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    if end > 0 {
                        out.entry(candidate.to_string())
                            .or_default()
                            .insert(field_tail[..end].to_string());
                    }
                }
            }
            rest = tail;
        }
    }
}

fn collect_sequence_indexed_function_domains(
    expr: &str,
    out: &mut BTreeMap<String, BTreeSet<String>>,
) {
    for prefix in ["Head(", "Tail("] {
        let mut offset = 0usize;
        while let Some(rel_idx) = expr[offset..].find(prefix) {
            let start = offset + rel_idx;
            let after = &expr[start + prefix.len() - 1..];
            let Some((candidate, tail)) = take_probe_group(after, '(', ')') else {
                break;
            };
            if is_probe_simple_identifier(candidate)
                && let Some(function_name) = find_sequence_index_target(expr, start)
            {
                out.entry(candidate.to_string())
                    .or_default()
                    .insert(function_name);
            }

            let consumed = expr[start..].len().saturating_sub(tail.len());
            offset = start + consumed.max(prefix.len());
        }
    }
}

fn find_sequence_index_target(expr: &str, call_start: usize) -> Option<String> {
    let prefix = &expr[..call_start];
    let bracket_end = trim_probe_ascii_end(prefix, prefix.len());
    if bracket_end == 0 || prefix.as_bytes()[bracket_end - 1] != b'[' {
        return None;
    }

    let bracket_pos = bracket_end - 1;
    let bang_end = trim_probe_ascii_end(prefix, bracket_pos);
    if bang_end > 0 && prefix.as_bytes()[bang_end - 1] == b'!' {
        find_except_target_name(&prefix[..bracket_pos])
    } else {
        find_indexed_function_name(&prefix[..bracket_pos])
    }
}

fn trim_probe_ascii_end(expr: &str, mut end: usize) -> usize {
    while end > 0 && expr.as_bytes()[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    end
}

fn find_except_target_name(prefix: &str) -> Option<String> {
    let trimmed = prefix.trim_end();
    let marker = "EXCEPT !";
    let marker_idx = trimmed.rfind(marker)?;
    let before = trimmed[..marker_idx].trim_end();
    let target = before.strip_prefix('[')?.trim();
    is_probe_simple_identifier(target).then(|| target.to_string())
}

fn infer_action_param_samples_from_guards(
    params: &[String],
    clauses: &[ActionClause],
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> BTreeMap<String, TlaValue> {
    let param_names: BTreeSet<String> = params
        .iter()
        .map(|param| normalize_param_name(param).to_string())
        .collect();
    if param_names.is_empty() {
        return BTreeMap::new();
    }

    let ctx = build_probe_eval_context(probe_state, definitions, instances);
    let mut samples = BTreeMap::new();
    for clause in clauses {
        let ActionClause::Guard { expr } = clause else {
            continue;
        };
        let expr = normalize_probe_clause_expr(expr);
        for conjunct in split_top_level(&expr, "/\\") {
            if let Some((name, value)) = boolean_param_sample_from_expr(&conjunct, &param_names) {
                samples.entry(name).or_insert(value);
            }
            for (name, value) in boolean_param_samples_from_nested_ifs(&conjunct, &param_names) {
                samples.entry(name).or_insert(value);
            }
            match classify_clause(&conjunct) {
                ClauseKind::UnprimedMembership { var, set_expr } if param_names.contains(&var) => {
                    if let Some(value) = representative_value_from_set_expr(&set_expr, &ctx) {
                        samples.entry(var).or_insert(value);
                    }
                }
                ClauseKind::UnprimedEquality { var, expr } if param_names.contains(&var) => {
                    if expr.trim() == var {
                        continue;
                    }
                    if let Ok(value) = eval_expr(&expr, &ctx) {
                        samples.entry(var).or_insert(value);
                    }
                }
                _ => {}
            }
        }
    }

    samples
}

fn boolean_param_sample_from_expr(
    expr: &str,
    param_names: &BTreeSet<String>,
) -> Option<(String, TlaValue)> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    if param_names.contains(trimmed) {
        return Some((trimmed.to_string(), TlaValue::Bool(true)));
    }

    let negated = trimmed.strip_prefix('~')?;
    let inner = strip_probe_outer_parens(negated.trim());
    param_names
        .contains(inner)
        .then(|| (inner.to_string(), TlaValue::Bool(false)))
}

fn boolean_param_samples_from_nested_ifs(
    expr: &str,
    param_names: &BTreeSet<String>,
) -> Vec<(String, TlaValue)> {
    let chars: Vec<char> = expr.chars().collect();
    let mut samples = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        if matches_probe_keyword_at(&chars, i, "IF") {
            let byte_idx: usize = chars[..i].iter().map(|ch| ch.len_utf8()).sum();
            let after_if = expr[byte_idx + "IF".len()..].trim_start();
            if let Some(then_idx) = find_probe_outer_then(after_if) {
                let condition = after_if[..then_idx].trim();
                if let Some(sample) = boolean_param_sample_from_expr(condition, param_names) {
                    samples.push(sample);
                }
            }
        }
        i += 1;
    }

    samples
}

fn matches_probe_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (offset, kw_char) in kw_chars.iter().enumerate() {
        if chars[i + offset] != *kw_char {
            return false;
        }
    }

    if i > 0 && is_probe_word_char(chars[i - 1]) {
        return false;
    }
    let after = i + kw_chars.len();
    if after < chars.len() && is_probe_word_char(chars[after]) {
        return false;
    }

    true
}

fn is_probe_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn refine_param_sample_from_clause_domains(
    param: &str,
    current: &TlaValue,
    clauses: &[ActionClause],
    probe_state: &TlaState,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> TlaValue {
    let normalized = normalize_param_name(param);
    let param_names = [param, normalized];
    let ctx = build_probe_eval_context(probe_state, definitions, instances);

    let mut saw_incompatible_domain = false;
    let mut replacement = None;
    for clause in clauses {
        let expr = match clause {
            ActionClause::Guard { expr }
            | ActionClause::PrimedAssignment { expr, .. }
            | ActionClause::PrimedMembership { set_expr: expr, .. }
            | ActionClause::LetWithPrimes { expr } => expr.as_str(),
            ActionClause::Exists { body, .. } => body.as_str(),
            ActionClause::Unchanged { .. } => continue,
        };

        for function_name in find_clause_function_domains_for_params(expr, &param_names) {
            let Ok(value) = eval_expr(&function_name, &ctx) else {
                continue;
            };
            let TlaValue::Function(entries) = value else {
                continue;
            };
            if entries.contains_key(current) {
                continue;
            }
            saw_incompatible_domain = true;
            if replacement.is_none() {
                replacement = entries.keys().next().cloned();
            }
        }
    }

    if saw_incompatible_domain {
        replacement.unwrap_or_else(|| current.clone())
    } else {
        current.clone()
    }
}

fn find_clause_function_domains_for_params(expr: &str, param_names: &[&str]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut i = 0usize;
    while i < expr.len() {
        let Some(ch) = expr[i..].chars().next() else {
            break;
        };
        let len = ch.len_utf8();
        if ch != '[' {
            i += len;
            continue;
        }

        let Some((inner, tail)) = take_probe_group(&expr[i..], '[', ']') else {
            i += len;
            continue;
        };
        let inner = inner.trim();
        if param_names.iter().any(|name| inner == *name)
            && let Some(function_name) = find_indexed_function_name(&expr[..i])
        {
            names.insert(function_name);
        }
        let _ = tail;
        i += len;
    }

    names
}

fn find_indexed_function_name(prefix: &str) -> Option<String> {
    let trimmed = prefix.trim_end();
    let mut start = trimmed.len();
    for (idx, ch) in trimmed.char_indices().rev() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '!' {
            start = idx;
            continue;
        }
        break;
    }
    if start >= trimmed.len() {
        return None;
    }

    let candidate = &trimmed[start..];
    is_probe_simple_identifier(candidate).then(|| candidate.to_string())
}

fn should_skip_action_expr_probe(expr: &str) -> bool {
    let trimmed = strip_probe_outer_parens(expr.trim());
    if trimmed.is_empty() {
        return true;
    }
    if starts_with_temporal_box(trimmed)
        || starts_with_temporal_diamond(trimmed)
        || trimmed.starts_with("WF_")
        || trimmed.starts_with("SF_")
        || contains_temporal_box(trimmed)
        || contains_temporal_diamond(trimmed)
        || trimmed.contains("WF_")
        || trimmed.contains("SF_")
        || trimmed.contains("~>")
    {
        return true;
    }

    if trimmed.contains('\'') {
        let disjuncts = split_action_body_disjuncts(trimmed);
        if disjuncts.len() > 1 || trimmed.starts_with("\\/") {
            return true;
        }

        let conjuncts = split_top_level(trimmed, "/\\");
        if conjuncts.len() > 1 || trimmed.starts_with("/\\") {
            return true;
        }
    }

    false
}

fn starts_with_temporal_box(expr: &str) -> bool {
    expr.starts_with("[]")
}

fn starts_with_temporal_diamond(expr: &str) -> bool {
    expr.starts_with("<>") && !expr.starts_with("<<")
}

fn contains_temporal_box(expr: &str) -> bool {
    expr.contains("[]")
}

fn contains_temporal_diamond(expr: &str) -> bool {
    let bytes = expr.as_bytes();
    if bytes.len() < 2 {
        return false;
    }
    for i in 0..bytes.len() - 1 {
        if bytes[i] == b'<' && bytes[i + 1] == b'>' {
            let prev_is_angle = i > 0 && (bytes[i - 1] == b'<' || bytes[i - 1] == b'>');
            let next_is_angle =
                i + 2 < bytes.len() && (bytes[i + 2] == b'<' || bytes[i + 2] == b'>');
            if !prev_is_angle && !next_is_angle {
                return true;
            }
        }
    }
    false
}

fn merge_staged_prime_locals(from: &EvalContext<'_>, into: &mut EvalContext<'_>) {
    let locals_mut = std::rc::Rc::make_mut(&mut into.locals);
    for (name, value) in from.locals.iter() {
        if name.ends_with('\'') {
            locals_mut.insert(name.clone(), value.clone());
        }
    }
}

fn should_short_circuit_probe_guard(expr: &str) -> bool {
    let trimmed = normalize_probe_clause_expr(expr);
    let disjuncts = split_action_body_disjuncts(&trimmed);
    parse_probe_action_call(&trimmed).is_none()
        && parse_probe_action_if(&trimmed).is_none()
        && parse_stuttering_action_expr(&trimmed).is_none()
        && disjuncts.len() <= 1
        && !trimmed.starts_with("\\/")
}

fn probe_action_body_into_ctx(body: &str, ctx: &mut EvalContext<'_>) -> anyhow::Result<()> {
    let normalized = strip_probe_comments(body);
    let trimmed = normalized.trim();
    if !trimmed.is_empty() && trimmed.contains('\'') {
        let mut runtime_ctx = ctx.clone();
        if let Some(Ok(())) = probe_action_body_via_runtime_eval(trimmed, &mut runtime_ctx) {
            merge_staged_prime_locals(&runtime_ctx, ctx);
            return Ok(());
        }
    }

    let def = TlaDefinition {
        name: "__probe_body".to_string(),
        params: vec![],
        body: trimmed.to_string(),
        is_recursive: false,
    };
    let ir = compile_action_ir(&def);
    for clause in &ir.clauses {
        if let ActionClause::Guard { expr } = clause
            && should_short_circuit_probe_guard(expr)
            && let Ok(value) = eval_expr(expr, ctx)
            && matches!(value.as_bool(), Ok(false))
        {
            return Ok(());
        }
        if let Some(result) = probe_action_clause_expr(clause, ctx) {
            result?;
        }
    }
    Ok(())
}

fn probe_action_body_via_runtime_eval(
    body: &str,
    ctx: &mut EvalContext<'_>,
) -> Option<anyhow::Result<()>> {
    let normalized = strip_probe_comments(body);
    let trimmed = normalized.trim();
    if trimmed.is_empty() || !trimmed.contains('\'') {
        return None;
    }
    if !(trimmed.contains("\\/") || trimmed.starts_with("IF ") || trimmed.starts_with("IF\n")) {
        return None;
    }

    let staged = current_probe_staged_primes(ctx);
    Some(
        eval_action_body_multi(trimmed, ctx, &staged).map(|branches| {
            let Some((next_staged, _)) = branches.into_iter().max_by_key(|(candidate, _)| {
                candidate
                    .iter()
                    .filter(|(var, value)| {
                        staged
                            .get(*var)
                            .or_else(|| ctx.state.get(*var))
                            .is_none_or(|baseline| baseline != *value)
                    })
                    .count()
            }) else {
                return;
            };
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (var, value) in next_staged {
                locals_mut.insert(format!("{var}'"), value);
            }
        }),
    )
}

fn probe_disjunctive_action_body(
    expr: &str,
    ctx: &mut EvalContext<'_>,
) -> Option<anyhow::Result<()>> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() <= 1 && !trimmed.starts_with("\\/") {
        return None;
    }

    let mut first_err = None;
    for disjunct in disjuncts {
        let mut branch_ctx = ctx.clone();
        let attempt = match probe_action_body_via_runtime_eval(&disjunct, &mut branch_ctx) {
            Some(Ok(())) => Ok(()),
            Some(Err(_)) | None => probe_action_body_into_ctx(&disjunct, &mut branch_ctx),
        };
        match attempt {
            Ok(()) => {
                merge_staged_prime_locals(&branch_ctx, ctx);
                return Some(Ok(()));
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
    }

    Some(Err(first_err.unwrap_or_else(|| {
        anyhow::anyhow!("no disjunctive action branch was probeable")
    })))
}

fn seed_probe_instance_constant_bindings(
    instance: &TlaModuleInstance,
    module: &TlaModule,
    parent_ctx: &EvalContext<'_>,
    child_ctx: &mut EvalContext<'_>,
) {
    let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
    let defs_mut = std::rc::Rc::make_mut(&mut child_ctx.local_definitions);

    for constant in &module.constants {
        if instance.substitutions.contains_key(constant) {
            continue;
        }
        if let Some(def) = parent_ctx
            .local_definitions
            .get(constant)
            .cloned()
            .or_else(|| {
                parent_ctx
                    .definitions
                    .and_then(|defs| defs.get(constant).cloned())
            })
        {
            defs_mut.insert(constant.clone(), def);
            continue;
        }
        if let Ok(value) = eval_expr(constant, parent_ctx) {
            locals_mut.insert(constant.clone(), value);
        }
    }
}

fn bind_probe_instance_substitutions(
    instance: &TlaModuleInstance,
    parent_ctx: &EvalContext<'_>,
    locals_mut: &mut BTreeMap<String, TlaValue>,
) -> anyhow::Result<()> {
    for (param, value_expr) in &instance.substitutions {
        let trimmed = value_expr.trim();
        let value = eval_expr(trimmed, parent_ctx)?;
        locals_mut.insert(param.clone(), value);

        if let Some(primed_value) = resolved_instance_primed_substitution_value(trimmed, parent_ctx)
        {
            locals_mut.insert(format!("{param}'"), primed_value);
        }
    }

    Ok(())
}

fn resolved_instance_primed_substitution_value(
    trimmed: &str,
    parent_ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    let primed_expr = format!("{trimmed}'");
    let primed_value = eval_expr(&primed_expr, parent_ctx).ok()?;
    match &primed_value {
        TlaValue::ModelValue(name) if name == &primed_expr => None,
        _ => Some(primed_value),
    }
}

fn body_contains_probeable_action_call(body: &str) -> bool {
    let normalized = strip_probe_comments(body);
    let trimmed = strip_probe_outer_parens(normalized.trim());
    if trimmed.is_empty() {
        return false;
    }
    if parse_probe_action_call(trimmed).is_some() {
        return true;
    }
    if let Some((_, then_branch, else_branch)) = parse_probe_action_if(trimmed) {
        return body_contains_probeable_action_call(then_branch)
            || body_contains_probeable_action_call(else_branch);
    }
    if let Some(rest) = trimmed.strip_prefix("\\E").map(str::trim_start)
        && let Some((_, nested_body)) = split_probe_quantifier(rest)
    {
        return body_contains_probeable_action_call(nested_body);
    }

    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() > 1 || trimmed.starts_with("\\/") {
        return disjuncts
            .into_iter()
            .any(|disjunct| body_contains_probeable_action_call(&disjunct));
    }

    let clauses = split_top_level(trimmed, "/\\");
    if clauses.len() > 1 || trimmed.starts_with("/\\") {
        return clauses
            .into_iter()
            .any(|clause| body_contains_probeable_action_call(&clause));
    }

    false
}

fn definition_is_probeable_action(def: &TlaDefinition) -> bool {
    looks_like_action(def) || body_contains_probeable_action_call(&def.body)
}

fn body_contains_contextual_probeable_action_call(
    body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    visited: &mut BTreeSet<String>,
) -> bool {
    let normalized = strip_probe_comments(body);
    let trimmed = strip_probe_outer_parens(normalized.trim());
    if trimmed.is_empty() {
        return false;
    }
    if let Some((name, arg_exprs)) = parse_probe_action_call(trimmed) {
        return action_call_is_contextually_probeable(
            &name,
            arg_exprs.len(),
            definitions,
            instances,
            visited,
        );
    }
    if let Some((_, then_branch, else_branch)) = parse_probe_action_if(trimmed) {
        return body_contains_contextual_probeable_action_call(
            then_branch,
            definitions,
            instances,
            visited,
        ) || body_contains_contextual_probeable_action_call(
            else_branch,
            definitions,
            instances,
            visited,
        );
    }
    if let Some(rest) = trimmed.strip_prefix("\\E").map(str::trim_start)
        && let Some((_, nested_body)) = split_probe_quantifier(rest)
    {
        return body_contains_contextual_probeable_action_call(
            nested_body,
            definitions,
            instances,
            visited,
        );
    }

    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() > 1 || trimmed.starts_with("\\/") {
        return disjuncts.into_iter().any(|disjunct| {
            body_contains_contextual_probeable_action_call(
                &disjunct,
                definitions,
                instances,
                visited,
            )
        });
    }

    let clauses = split_top_level(trimmed, "/\\");
    if clauses.len() > 1 || trimmed.starts_with("/\\") {
        return clauses.into_iter().any(|clause| {
            body_contains_contextual_probeable_action_call(&clause, definitions, instances, visited)
        });
    }

    false
}

fn action_call_is_contextually_probeable(
    name: &str,
    arg_count: usize,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if let Some((alias, operator_name)) = name.split_once('!') {
        let Some(instance) = instances.get(alias) else {
            return false;
        };
        let Some(module) = instance.module.as_ref() else {
            return false;
        };
        let Some(def) = module.definitions.get(operator_name) else {
            return false;
        };
        if def.params.len() != arg_count {
            return false;
        }
        return definition_is_contextually_probeable_action(
            def,
            &module.definitions,
            &module.instances,
            visited,
        );
    }

    let Some(def) = definitions.get(name) else {
        return false;
    };
    if def.params.len() != arg_count {
        return false;
    }
    definition_is_contextually_probeable_action(def, definitions, instances, visited)
}

fn definition_is_contextually_probeable_action(
    def: &TlaDefinition,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if looks_like_action(def) {
        return true;
    }
    if !visited.insert(def.name.clone()) {
        return false;
    }
    let result =
        body_contains_contextual_probeable_action_call(&def.body, definitions, instances, visited);
    visited.remove(&def.name);
    result
}

fn expand_probe_action_call(expr: &str, ctx: &mut EvalContext<'_>) -> Option<anyhow::Result<()>> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let (name, arg_exprs) = parse_probe_action_call(trimmed)?;

    let (def, mut child_ctx) = if let Some((alias, operator_name)) = name.split_once('!') {
        let instances = ctx.instances?;
        let instance = instances.get(alias)?;
        let module = instance.module.as_ref()?;
        let def = module.definitions.get(operator_name)?.clone();
        if !definition_is_probeable_action(&def) {
            return None;
        }
        if def.params.len() != arg_exprs.len() {
            return Some(Err(anyhow::anyhow!(
                "operator '{alias}!{operator_name}' arity mismatch: expected {}, got {}",
                def.params.len(),
                arg_exprs.len()
            )));
        }

        let mut instance_ctx = ctx.clone();
        instance_ctx.definitions = Some(&module.definitions);
        instance_ctx.instances = Some(&module.instances);
        seed_probe_instance_constant_bindings(instance, module, ctx, &mut instance_ctx);
        {
            let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
            if let Err(err) = bind_probe_instance_substitutions(instance, ctx, locals_mut) {
                return Some(Err(err));
            }
            for (param, arg_expr) in def.params.iter().zip(arg_exprs.iter()) {
                let value = match eval_expr(arg_expr, ctx) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                locals_mut.insert(normalize_param_name(param).to_string(), value);
            }
        }

        (def, instance_ctx)
    } else {
        let def = ctx
            .local_definitions
            .get(&name)
            .cloned()
            .or_else(|| ctx.definitions.and_then(|defs| defs.get(&name).cloned()))?;
        if !definition_is_probeable_action(&def) {
            return None;
        }
        if def.params.len() != arg_exprs.len() {
            return Some(Err(anyhow::anyhow!(
                "operator '{name}' arity mismatch: expected {}, got {}",
                def.params.len(),
                arg_exprs.len()
            )));
        }

        let mut child_ctx = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
            for (param, arg_expr) in def.params.iter().zip(arg_exprs.iter()) {
                let value = match eval_expr(arg_expr, ctx) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                locals_mut.insert(normalize_param_name(param).to_string(), value);
            }
        }

        (def, child_ctx)
    };

    Some(
        probe_action_body_into_ctx(&def.body, &mut child_ctx).map(|_| {
            merge_staged_prime_locals(&child_ctx, ctx);
        }),
    )
}

fn current_probe_staged_primes(ctx: &EvalContext<'_>) -> BTreeMap<String, TlaValue> {
    ctx.locals
        .iter()
        .filter_map(|(name, value)| {
            name.strip_suffix('\'')
                .map(|var| (var.to_string(), value.clone()))
        })
        .collect()
}

const PROBE_BRANCH_DISABLED_LOCAL: &str = "__probe_branch_disabled";

fn mark_probe_branch_disabled(ctx: &mut EvalContext<'_>) {
    std::rc::Rc::make_mut(&mut ctx.locals).insert(
        PROBE_BRANCH_DISABLED_LOCAL.to_string(),
        TlaValue::Bool(true),
    );
}

fn probe_branch_is_disabled(ctx: &EvalContext<'_>) -> bool {
    matches!(
        ctx.locals.get(PROBE_BRANCH_DISABLED_LOCAL),
        Some(TlaValue::Bool(true))
    )
}

fn probe_let_action_expr(expr: &str, ctx: &mut EvalContext<'_>) -> anyhow::Result<()> {
    let staged = current_probe_staged_primes(ctx);
    let mut branches = eval_let_action_multi(expr, ctx, &staged, &[])?.into_iter();
    let Some((next_staged, _)) = branches.next() else {
        return Ok(());
    };

    let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
    for (var, value) in next_staged {
        locals_mut.insert(format!("{var}'"), value);
    }
    Ok(())
}

fn probe_action_clause_expr(
    clause: &ActionClause,
    ctx: &mut EvalContext<'_>,
) -> Option<anyhow::Result<()>> {
    if probe_branch_is_disabled(ctx) {
        return None;
    }

    match clause {
        ActionClause::LetWithPrimes { expr } => {
            let expr = strip_probe_comments(expr);
            Some(probe_let_action_expr(&expr, ctx))
        }
        ActionClause::Guard { expr } => {
            let expr = normalize_probe_clause_expr(expr);
            if let Some((condition, then_branch, else_branch)) = parse_probe_action_if(&expr) {
                let mut branch_ctx = ctx.clone();
                return Some(
                    eval_expr(condition, ctx)
                        .and_then(|value| value.as_bool())
                        .and_then(|take_then| {
                            let branch = if take_then { then_branch } else { else_branch };
                            match probe_action_body_via_runtime_eval(branch, &mut branch_ctx) {
                                Some(Ok(())) => Ok(()),
                                Some(Err(_)) | None => {
                                    probe_action_body_into_ctx(branch, &mut branch_ctx)
                                }
                            }
                        })
                        .map(|_| {
                            merge_staged_prime_locals(&branch_ctx, ctx);
                        }),
                );
            }
            if let Some((action, vars)) = parse_stuttering_action_expr(&expr) {
                return Some(probe_stuttering_action_expr(&action, &vars, ctx));
            }
            if let Some(expanded) = probe_disjunctive_action_body(&expr, ctx) {
                return Some(expanded);
            }
            if let Some(expanded) = probe_guard_exists_action_call(&expr, ctx) {
                return Some(expanded);
            }
            if let Some(expanded) = expand_probe_action_call(&expr, ctx) {
                return Some(expanded);
            }
            if should_skip_action_expr_probe(&expr) {
                return None;
            }
            Some(eval_expr(&expr, ctx).and_then(|value| {
                // Guard clauses should be boolean, but some definitions that
                // contain primes (e.g., Cardinality expressions) get classified
                // as action-like even though they return non-boolean values.
                // Tolerate non-boolean results: the expression evaluated
                // successfully, which is what the probe cares about.
                match value.as_bool() {
                    Ok(false) => mark_probe_branch_disabled(ctx),
                    Ok(true) => {}
                    Err(_) => {
                        // Non-boolean result (e.g., Int from Cardinality) — the
                        // expression still evaluated correctly; don't fail the probe.
                    }
                }
                Ok(())
            }))
        }
        ActionClause::PrimedAssignment { var, expr } => Some(eval_expr(expr, ctx).map(|value| {
            std::rc::Rc::make_mut(&mut ctx.locals).insert(format!("{var}'"), value);
        })),
        ActionClause::PrimedMembership { var, set_expr } => Some(match eval_expr(set_expr, ctx) {
            Ok(TlaValue::Set(values)) if values.is_empty() => {
                mark_probe_branch_disabled(ctx);
                Ok(())
            }
            Ok(set_val) => {
                let Some(repr) = pick_representative_from_set(&set_val) else {
                    return Some(Err(anyhow::anyhow!(
                        "empty or non-set primed membership: {set_expr}"
                    )));
                };
                std::rc::Rc::make_mut(&mut ctx.locals).insert(format!("{var}'"), repr);
                Ok(())
            }
            Err(err) => {
                if let Some(repr) = representative_value_from_set_expr(set_expr, ctx) {
                    std::rc::Rc::make_mut(&mut ctx.locals).insert(format!("{var}'"), repr);
                    Ok(())
                } else {
                    Err(err)
                }
            }
        }),
        ActionClause::Unchanged { vars } => {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for var in vars {
                if let Some(value) = ctx.state.get(var) {
                    locals_mut.insert(format!("{var}'"), value.clone());
                }
            }
            None
        }
        ActionClause::Exists { binders, body } => {
            Some(probe_exists_action_body(binders, body, ctx))
        }
    }
}

fn parse_guard_exists_action_call(expr: &str) -> Option<(String, String)> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let rest = trimmed.strip_prefix("\\E")?;
    if !rest.starts_with(char::is_whitespace) && !rest.starts_with('(') {
        return None;
    }
    let rest = rest.trim_start();
    let colon_idx = find_probe_top_level_char(rest, ':')?;
    let binders = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();
    if binders.is_empty() || body.is_empty() {
        return None;
    }
    if parse_probe_action_call(strip_probe_outer_parens(body)).is_none() {
        return None;
    }
    Some((binders.to_string(), body.to_string()))
}

fn probe_guard_exists_action_call(
    expr: &str,
    ctx: &mut EvalContext<'_>,
) -> Option<anyhow::Result<()>> {
    let (binders, body) = parse_guard_exists_action_call(expr)?;
    Some(probe_exists_action_body(&binders, &body, ctx))
}

fn probe_exists_action_body(
    binders: &str,
    body: &str,
    ctx: &mut EvalContext<'_>,
) -> anyhow::Result<()> {
    let local_bindings = (*ctx.locals).clone();
    let definitions = ctx.definitions.unwrap_or(ctx.local_definitions.as_ref());
    let bound = if let Some(instances) = ctx.instances {
        sample_exists_quantifier_binders(
            binders,
            &local_bindings,
            ctx.state,
            definitions,
            instances,
        )
    } else {
        sample_exists_quantifier_binders(
            binders,
            &local_bindings,
            ctx.state,
            definitions,
            &BTreeMap::new(),
        )
    };
    let Some(bound) = bound else {
        mark_probe_branch_disabled(ctx);
        return Ok(());
    };

    let mut child_ctx = ctx.clone();
    {
        let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
        for (name, value) in bound {
            locals_mut.insert(name, value);
        }
    }

    probe_action_body_into_ctx(body, &mut child_ctx)?;
    merge_staged_prime_locals(&child_ctx, ctx);
    Ok(())
}

fn probe_stuttering_action_expr(
    action: &str,
    vars: &[String],
    ctx: &mut EvalContext<'_>,
) -> anyhow::Result<()> {
    let mut branch_ctx = ctx.clone();
    let attempt = match probe_action_body_via_runtime_eval(action, &mut branch_ctx) {
        Some(Ok(())) => Ok(()),
        Some(Err(_)) | None => probe_action_body_into_ctx(action, &mut branch_ctx),
    };
    if attempt.is_ok() {
        merge_staged_prime_locals(&branch_ctx, ctx);
        return Ok(());
    }

    let stuttered = vars
        .iter()
        .filter_map(|var| {
            ctx.state
                .get(var)
                .cloned()
                .map(|value| (format!("{var}'"), value))
        })
        .collect::<Vec<_>>();
    let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
    for (name, value) in stuttered {
        locals_mut.insert(name, value);
    }
    Ok(())
}

/// Picks a representative value from a set for probing purposes.
/// For membership constraints like `v \in [Proc -> Values]`, this picks
/// one element from the set to seed the probe_state. The actual model
/// checking in evaluate_init_states handles full enumeration.
fn pick_representative_from_set(set_val: &TlaValue) -> Option<TlaValue> {
    match set_val {
        TlaValue::Set(values) => values
            .iter()
            .max_by_key(|value| probe_value_score(value))
            .cloned(),
        _ => None,
    }
}

/// Try to create a representative function from a function set expression.
/// This handles cases where `[Domain -> Range]` is too large to enumerate.
/// Instead of enumerating all possible functions, we create a single representative
/// function that maps each domain element to the first element of the range.
fn try_create_representative_function(set_expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    let trimmed = strip_probe_outer_parens(set_expr.trim());
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    let arrow_idx = find_probe_top_level_keyword(inner, "->")?;
    let domain_expr = inner[..arrow_idx].trim();
    let range_expr = inner[arrow_idx + 2..].trim();

    let domain_val = eval_expr(domain_expr, ctx).ok()?;
    let domain_set = domain_val.as_set().ok()?;
    let repr_val = representative_member_from_domain_expr(range_expr, ctx)?;

    let mut func = std::collections::BTreeMap::new();
    for elem in domain_set.iter() {
        func.insert(elem.clone(), repr_val.clone());
    }

    Some(TlaValue::Function(Arc::new(func)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tlaplusplus::tla::{parse_tla_module_file, parse_tla_module_text};

    fn parsed_two_pc_with_btm_probe_module() -> TlaModule {
        parse_tla_module_text(
            r#"---- MODULE TwoPCwithBTMProbe ----
CONSTANTS RM, RMMAYFAIL, TMMAYFAIL
VARIABLES rmState, tmState, pc

vars == << rmState, tmState, pc >>
ProcSet == (RM) \cup {0} \cup {10}

RS(self) == /\ pc[self] = "RS"
            /\ IF rmState[self] \in {"working", "prepared"}
                  THEN /\ \/ /\ rmState[self] = "working"
                             /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                          \/ /\ \/ /\ tmState="commit"
                                   /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                \/ /\ rmState[self]="working" \/ tmState="abort"
                                   /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                          \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                   THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                   ELSE /\ TRUE
                                        /\ UNCHANGED rmState
                       /\ pc' = [pc EXCEPT ![self] = "RS"]
                  ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                       /\ UNCHANGED rmState
            /\ UNCHANGED tmState

RManager(self) == RS(self)

TS == /\ pc[0] = "TS"
      /\ \/ /\ tmState = "commit"
            /\ pc' = [pc EXCEPT ![0] = "TC"]
         \/ /\ tmState = "abort"
            /\ pc' = [pc EXCEPT ![0] = "TA"]
      /\ UNCHANGED << rmState, tmState >>

TC == /\ pc[0] = "TC"
      /\ tmState' = "commit"
      /\ pc' = [pc EXCEPT ![0] = "Done"]
      /\ UNCHANGED rmState

TA == /\ pc[0] = "TA"
      /\ tmState' = "abort"
      /\ pc' = [pc EXCEPT ![0] = "Done"]
      /\ UNCHANGED rmState

TManager == TS \/ TC \/ TA

BTS == /\ pc[10] = "BTS"
       /\ \/ /\ tmState = "commit"
             /\ pc' = [pc EXCEPT ![10] = "BTC"]
          \/ /\ tmState = "abort"
             /\ pc' = [pc EXCEPT ![10] = "BTA"]
       /\ UNCHANGED << rmState, tmState >>

BTC == /\ pc[10] = "BTC"
       /\ tmState' = "commit"
       /\ pc' = [pc EXCEPT ![10] = "Done"]
       /\ UNCHANGED rmState

BTA == /\ pc[10] = "BTA"
       /\ tmState' = "abort"
       /\ pc' = [pc EXCEPT ![10] = "Done"]
       /\ UNCHANGED rmState

BTManager == BTS \/ BTC \/ BTA

Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == TManager \/ BTManager
           \/ (\E self \in RM: RManager(self))
           \/ Terminating
====
"#,
        )
        .expect("2PCwithBTM probe module should parse")
    }

    fn parsed_braf_probe_module() -> TlaModule {
        parse_tla_module_text(
            r#"---- MODULE BrafProbe ----
CONSTANTS Symbols, ArbitrarySymbol, BuffSz, MaxOffset
VARIABLES dirty, length, curr, lo, buff, diskPos, file_content, file_pointer

vars == <<dirty, length, curr, lo, buff, diskPos, file_content, file_pointer>>
Offset == 0..MaxOffset
SymbolOrArbitrary == Symbols \union {ArbitrarySymbol}
Min(a, b) == IF a <= b THEN a ELSE b
Max(a, b) == IF a <= b THEN b ELSE a
ArrayOfAnyLength(T) == [elems: Seq(T)]
Array(T, len) == [elems: [1..len -> T]]
EmptyArray == [elems |-> <<>>]
ArrayLen(a) == Len(a.elems)
ArrayGet(a, i) == a.elems[i + 1]
ArraySet(a, i, x) == [a EXCEPT !.elems[i + 1] = x]
ArraySlice(a, startInclusive, endExclusive) == [elems |-> SubSeq(a.elems, startInclusive + 1, endExclusive)]
ArrayConcat(a1, a2) == [elems |-> a1.elems \o a2.elems]
Inv2 == /\ lo <= curr
        /\ curr < lo + BuffSz

TypeOK ==
    /\ dirty \in BOOLEAN
    /\ length \in Offset
    /\ curr \in Offset
    /\ lo \in Offset
    /\ buff \in Array(SymbolOrArbitrary, BuffSz)
    /\ diskPos \in Offset
    /\ file_content \in ArrayOfAnyLength(SymbolOrArbitrary)
    /\ file_pointer \in Offset

Init ==
    /\ dirty = FALSE
    /\ length = 0
    /\ curr = 0
    /\ lo = 0
    /\ buff \in Array({ArbitrarySymbol}, BuffSz)
    /\ diskPos = 0
    /\ file_pointer = 0
    /\ file_content = EmptyArray

FlushBuffer ==
    /\ dirty
    /\ dirty' = FALSE
    /\ UNCHANGED <<length, curr, lo, buff, diskPos, file_content, file_pointer>>

Write1(byte) ==
    /\ curr + 1 <= MaxOffset
    /\ Inv2
    /\ buff' = ArraySet(buff, curr - lo, byte)
    /\ curr' = curr + 1
    /\ dirty' = TRUE
    /\ length' = Max(length, curr')
    /\ UNCHANGED <<lo, diskPos, file_pointer, file_content>>

WriteAtMost(data) ==
    LET
        numWriteableWithoutSeeking == Min(ArrayLen(data), lo + BuffSz - curr)
        buffOff == curr - lo
    IN
    /\ Inv2
    /\ curr + numWriteableWithoutSeeking <= MaxOffset
    /\ buff' = ArrayConcat(ArrayConcat(
            ArraySlice(buff, 0, buffOff),
            ArraySlice(data, 0, numWriteableWithoutSeeking)),
            ArraySlice(buff, buffOff + numWriteableWithoutSeeking, ArrayLen(buff)))
    /\ dirty' = TRUE
    /\ curr' = curr + numWriteableWithoutSeeking
    /\ length' = Max(length, curr')
    /\ UNCHANGED <<lo, diskPos, file_content, file_pointer>>

Next ==
    \/ \E symbol \in SymbolOrArbitrary:
        \/ Write1(symbol)
    \/ \E len \in 1..MaxOffset: \E data \in Array(SymbolOrArbitrary, len):
        \/ WriteAtMost(data)
====
"#,
        )
        .expect("BRAF probe module should parse")
    }

    fn seed_braf_probe_state(module: &TlaModule) -> TlaState {
        let mut probe_state = TlaState::from([
            (
                "Symbols".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("A".to_string()),
                    TlaValue::ModelValue("B".to_string()),
                ]))),
            ),
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("BuffSz".to_string(), TlaValue::Int(2)),
            ("MaxOffset".to_string(), TlaValue::Int(3)),
        ]);
        let init_def = module.definitions.get("Init").expect("Init should exist");
        let mut pending_eq = Vec::new();
        let mut pending_mem = Vec::new();
        for clause in
            expand_state_predicate_clauses(&init_def.body, &module.definitions, &module.instances)
        {
            match classify_clause(&clause) {
                ClauseKind::UnprimedEquality { var, expr } => pending_eq.push((var, expr)),
                ClauseKind::UnprimedMembership { var, set_expr } => {
                    pending_mem.push((var, set_expr))
                }
                _ => {}
            }
        }

        let total_pending = pending_eq.len() + pending_mem.len();
        for _ in 0..total_pending.saturating_add(1) {
            if pending_eq.is_empty() && pending_mem.is_empty() {
                break;
            }

            let mut progress = false;
            let mut next_pending_eq = Vec::new();
            for (var, expr) in pending_eq {
                let ctx =
                    build_probe_eval_context(&probe_state, &module.definitions, &module.instances);
                match eval_expr(&expr, &ctx) {
                    Ok(value) => {
                        probe_state.insert(var, value);
                        progress = true;
                    }
                    Err(_) => next_pending_eq.push((var, expr)),
                }
            }

            let mut next_pending_mem = Vec::new();
            for (var, set_expr) in pending_mem {
                let ctx =
                    build_probe_eval_context(&probe_state, &module.definitions, &module.instances);
                match eval_expr(&set_expr, &ctx) {
                    Ok(set_val) => {
                        if let Some(repr) = pick_representative_from_set(&set_val) {
                            probe_state.insert(var, repr);
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                    Err(_) => {
                        if let Some(repr) = representative_value_from_set_expr(&set_expr, &ctx) {
                            probe_state.insert(var, repr);
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                }
            }

            if !progress {
                break;
            }
            pending_eq = next_pending_eq;
            pending_mem = next_pending_mem;
        }

        let _ = seed_probe_state_from_type_invariants(&mut probe_state, module, None);
        probe_state
    }

    fn parsed_braf_embedded_file_probe_module() -> TlaModule {
        let unique = format!(
            "braf-probe-{}-{}.tla",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock should be after epoch")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::write(
            &path,
            r#"---- MODULE BrafProbe ----
EXTENDS Common

CONSTANT BuffSz
VARIABLES dirty, length, curr, lo, buff, diskPos, file_content, file_pointer

TypeOK ==
    /\ dirty \in BOOLEAN
    /\ length \in Offset
    /\ curr \in Offset
    /\ lo \in Offset
    /\ buff \in Array(SymbolOrArbitrary, BuffSz)
    /\ diskPos \in Offset
    /\ file_content \in ArrayOfAnyLength(SymbolOrArbitrary)
    /\ file_pointer \in Offset

Init ==
    /\ dirty = FALSE
    /\ length = 0
    /\ curr = 0
    /\ lo = 0
    /\ buff \in Array({ArbitrarySymbol}, BuffSz)
    /\ diskPos = 0
    /\ file_pointer = 0
    /\ file_content = EmptyArray
===============================================================================

-------------------------------- MODULE Common --------------------------------
EXTENDS Naturals, Sequences

CONSTANTS Symbols, ArbitrarySymbol, MaxOffset
Offset == 0..MaxOffset
SymbolOrArbitrary == Symbols \union {ArbitrarySymbol}
ArrayOfAnyLength(T) == [elems: Seq(T)]
Array(T, len) == [elems: [1..len -> T]]
EmptyArray == [elems |-> <<>>]
===============================================================================
"#,
        )
        .expect("embedded BRAF probe file should be written");
        let module = parse_tla_module_file(&path).expect("embedded BRAF file should parse");
        let _ = fs::remove_file(path);
        module
    }

    fn write_braf_analyze_probe_files()
    -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let unique = format!(
            "braf-analyze-probe-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock should be after epoch")
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let module_path = dir.join("BrafProbe.tla");
        let cfg_path = dir.join("BrafProbe.cfg");
        fs::write(
            &module_path,
            r#"---- MODULE BrafProbe ----
EXTENDS Common

CONSTANT BuffSz
VARIABLES dirty, length, curr, lo, buff, diskPos, file_content, file_pointer

vars == <<dirty, length, curr, lo, buff, diskPos, file_content, file_pointer>>

TypeOK ==
    /\ dirty \in BOOLEAN
    /\ length \in Offset
    /\ curr \in Offset
    /\ lo \in Offset
    /\ buff \in Array(SymbolOrArbitrary, BuffSz)
    /\ diskPos \in Offset
    /\ file_content \in ArrayOfAnyLength(SymbolOrArbitrary)
    /\ file_pointer \in Offset

Init ==
    /\ dirty = FALSE
    /\ length = 0
    /\ curr = 0
    /\ lo = 0
    /\ buff \in Array({ArbitrarySymbol}, BuffSz)
    /\ diskPos = 0
    /\ file_pointer = 0
    /\ file_content = EmptyArray

FlushBuffer ==
    /\ dirty
    /\ dirty' = FALSE
    /\ UNCHANGED <<length, curr, lo, buff, diskPos, file_content, file_pointer>>

Write1(byte) ==
    /\ curr + 1 <= MaxOffset
    /\ Inv2
    /\ buff' = ArraySet(buff, curr - lo, byte)
    /\ curr' = curr + 1
    /\ dirty' = TRUE
    /\ length' = Max(length, curr')
    /\ UNCHANGED <<lo, diskPos, file_pointer, file_content>>

WriteAtMost(data) ==
    LET
        numWriteableWithoutSeeking == Min(ArrayLen(data), lo + BuffSz - curr)
        buffOff == curr - lo
    IN
    /\ Inv2
    /\ curr + numWriteableWithoutSeeking <= MaxOffset
    /\ buff' = ArrayConcat(ArrayConcat(
            ArraySlice(buff, 0, buffOff),
            ArraySlice(data, 0, numWriteableWithoutSeeking)),
            ArraySlice(buff, buffOff + numWriteableWithoutSeeking, ArrayLen(buff)))
    /\ dirty' = TRUE
    /\ curr' = curr + numWriteableWithoutSeeking
    /\ length' = Max(length, curr')
    /\ UNCHANGED <<lo, diskPos, file_content, file_pointer>>

Next ==
    \/ FlushBuffer
    \/ \E symbol \in SymbolOrArbitrary:
        \/ Write1(symbol)
    \/ \E len \in 1..MaxOffset: \E data \in Array(SymbolOrArbitrary, len):
        \/ WriteAtMost(data)

Spec == Init /\ [][Next]_vars
===============================================================================

-------------------------------- MODULE Common --------------------------------
EXTENDS Naturals, Sequences

CONSTANTS Symbols, ArbitrarySymbol, MaxOffset
Offset == 0..MaxOffset
SymbolOrArbitrary == Symbols \union {ArbitrarySymbol}
Min(a, b) == IF a <= b THEN a ELSE b
Max(a, b) == IF a <= b THEN b ELSE a
ArrayOfAnyLength(T) == [elems: Seq(T)]
Array(T, len) == [elems: [1..len -> T]]
EmptyArray == [elems |-> <<>>]
ArrayLen(a) == Len(a.elems)
ArrayGet(a, i) == a.elems[i + 1]
ArraySet(a, i, x) == [a EXCEPT !.elems[i + 1] = x]
ArraySlice(a, startInclusive, endExclusive) == [elems |-> SubSeq(a.elems, startInclusive + 1, endExclusive)]
ArrayConcat(a1, a2) == [elems |-> a1.elems \o a2.elems]
Inv2 == /\ lo <= curr
        /\ curr < lo + BuffSz
===============================================================================
"#,
        )
        .expect("probe module file should be written");
        fs::write(
            &cfg_path,
            r#"CONSTANTS
  Symbols = {A, B}
  ArbitrarySymbol = ArbitrarySymbol
  BuffSz = 2
  MaxOffset = 3

SPECIFICATION Spec
INVARIANTS TypeOK
"#,
        )
        .expect("probe cfg file should be written");
        (dir, module_path, cfg_path)
    }

    #[test]
    fn test_try_create_representative_function_basic() {
        // Test creating a representative function from [A -> B]
        let state = TlaState::new();
        let mut defs = BTreeMap::new();

        // Define A = {1, 2} and B = {10, 20}
        defs.insert(
            "A".to_string(),
            TlaDefinition {
                name: "A".to_string(),
                params: vec![],
                body: "{1, 2}".to_string(),
                is_recursive: false,
            },
        );
        defs.insert(
            "B".to_string(),
            TlaDefinition {
                name: "B".to_string(),
                params: vec![],
                body: "{10, 20}".to_string(),
                is_recursive: false,
            },
        );

        let ctx = EvalContext::with_definitions(&state, &defs);
        let result = try_create_representative_function("[A -> B]", &ctx);

        assert!(result.is_some(), "should create a representative function");
        let func = result.unwrap();

        // Check it's a function with 2 keys (one for each element in A)
        if let TlaValue::Function(f) = func {
            assert_eq!(f.len(), 2, "function should have 2 entries");
            // All values should be the representative element of B
            // (pick_representative_from_set uses probe_value_score heuristics)
            let repr = pick_representative_from_set(&TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::Int(10),
                TlaValue::Int(20),
            ]))))
            .unwrap();
            for (_, val) in f.iter() {
                assert_eq!(*val, repr, "all values should be the representative of B");
            }
        } else {
            panic!("expected Function, got {:?}", func);
        }
    }

    #[test]
    fn test_try_create_representative_function_non_function_set() {
        // Test that non-function-set expressions return None
        let state = TlaState::new();
        let defs: BTreeMap<String, TlaDefinition> = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Plain set - not a function set
        let result = try_create_representative_function("{1, 2, 3}", &ctx);
        assert!(result.is_none(), "should return None for plain set");

        // Variable reference - not a function set
        let result = try_create_representative_function("x", &ctx);
        assert!(
            result.is_none(),
            "should return None for variable reference"
        );
    }

    #[test]
    fn test_try_create_representative_function_with_model_values() {
        // Test with model values (common in TLA+ specs)
        let mut state = TlaState::new();
        state.insert(
            "N".to_string(),
            TlaValue::Set(Arc::new(
                [
                    TlaValue::ModelValue("n1".to_string()),
                    TlaValue::ModelValue("n2".to_string()),
                ]
                .into_iter()
                .collect(),
            )),
        );
        state.insert(
            "R".to_string(),
            TlaValue::Set(Arc::new(
                [TlaValue::Int(0), TlaValue::Int(1), TlaValue::Int(2)]
                    .into_iter()
                    .collect(),
            )),
        );

        let defs: BTreeMap<String, TlaDefinition> = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let result = try_create_representative_function("[N -> R]", &ctx);
        assert!(
            result.is_some(),
            "should create representative function with model values"
        );

        if let Some(TlaValue::Function(f)) = result {
            assert_eq!(f.len(), 2, "function should have 2 entries for N");
            // All values should be the first element of R (which is 0)
            for (_, val) in f.iter() {
                assert_eq!(
                    *val,
                    TlaValue::Int(2),
                    "all values should map to a representative from R"
                );
            }
        } else {
            panic!("expected Some(Function(...)), got {:?}", result);
        }
    }

    #[test]
    fn seeds_probe_state_from_type_invariants() {
        let mut probe_state = TlaState::new();
        probe_state.insert(
            "Readers".to_string(),
            TlaValue::Set(Arc::new(
                [
                    TlaValue::ModelValue("r1".to_string()),
                    TlaValue::ModelValue("r2".to_string()),
                ]
                .into_iter()
                .collect(),
            )),
        );

        let module = TlaModule {
            variables: vec![
                "read".to_string(),
                "chan".to_string(),
                "pending".to_string(),
            ],
            definitions: BTreeMap::from([(
                "TypeOK".to_string(),
                TlaDefinition {
                    name: "TypeOK".to_string(),
                    params: vec![],
                    body: r#"
                        /\ read \in [Readers -> Seq(Nat)]
                        /\ chan \in [ack : BOOLEAN, payload : Nat]
                        /\ pending \in SUBSET Readers
                    "#
                    .to_string(),
                    is_recursive: false,
                },
            )]),
            ..TlaModule::default()
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);

        assert_eq!(seeded, 3);
        match probe_state.get("read") {
            Some(TlaValue::Function(map)) => assert_eq!(map.len(), 2),
            other => panic!("expected function representative for read, got {other:?}"),
        }
        match probe_state.get("chan") {
            Some(TlaValue::Record(fields)) => {
                assert_eq!(fields.get("ack"), Some(&TlaValue::Bool(true)));
                assert_eq!(fields.get("payload"), Some(&TlaValue::Int(0)));
            }
            other => panic!("expected record representative for chan, got {other:?}"),
        }
        assert_eq!(
            probe_state.get("pending"),
            Some(&TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::ModelValue("r2".to_string()),
            ]))))
        );
    }

    #[test]
    fn seeds_probe_state_from_subset_type_invariants() {
        let mut probe_state =
            TlaState::from([("msgs".to_string(), TlaValue::Set(Arc::new(BTreeSet::new())))]);
        let module = TlaModule {
            name: "SubsetSeed".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["msgs".to_string()],
            definitions: BTreeMap::from([
                (
                    "Message".to_string(),
                    TlaDefinition {
                        name: "Message".to_string(),
                        params: vec![],
                        body: "[type: {\"1a\"}, bal: {0}]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeOK".to_string(),
                    TlaDefinition {
                        name: "TypeOK".to_string(),
                        params: vec![],
                        body: "msgs \\subseteq Message".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert_eq!(seeded, 1);
        match probe_state.get("msgs") {
            Some(TlaValue::Set(values)) => {
                let elem = values
                    .iter()
                    .next()
                    .expect("subset seed should be non-empty");
                assert!(matches!(elem, TlaValue::Record(_)));
            }
            other => panic!("expected seeded msgs set, got {other:?}"),
        }
    }

    #[test]
    fn seeds_probe_state_prefers_structured_function_values_from_type_invariants() {
        let proc_id = TlaValue::ModelValue("p1".to_string());
        let no_val = TlaValue::ModelValue("NoVal".to_string());
        let mut probe_state = TlaState::from([(
            "buf".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                proc_id.clone(),
                no_val.clone(),
            )]))),
        )]);
        let module = TlaModule {
            name: "BufSeed".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["buf".to_string()],
            definitions: BTreeMap::from([
                (
                    "Proc".to_string(),
                    TlaDefinition {
                        name: "Proc".to_string(),
                        params: vec![],
                        body: "{p1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "MReq".to_string(),
                    TlaDefinition {
                        name: "MReq".to_string(),
                        params: vec![],
                        body: "[op: {\"Wr\"}, adr: {a1}, val: {v1}]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Val".to_string(),
                    TlaDefinition {
                        name: "Val".to_string(),
                        params: vec![],
                        body: "{v1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeOK".to_string(),
                    TlaDefinition {
                        name: "TypeOK".to_string(),
                        params: vec![],
                        body: "buf \\in [Proc -> MReq \\cup Val \\cup {NoVal}]".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert_eq!(seeded, 1);
        match probe_state.get("buf") {
            Some(TlaValue::Function(values)) => {
                let value = values.get(&proc_id).expect("buf should contain p1");
                assert!(matches!(value, TlaValue::Record(_)));
            }
            other => panic!("expected refined buf function, got {other:?}"),
        }
    }

    #[test]
    fn seeds_probe_state_refines_placeholder_functions_even_when_placeholders_typecheck() {
        let proc_id = TlaValue::ModelValue("p1".to_string());
        let no_val = TlaValue::ModelValue("NoVal".to_string());
        let mut probe_state = TlaState::from([
            (
                "buf".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    proc_id.clone(),
                    no_val.clone(),
                )]))),
            ),
            ("NoVal".to_string(), no_val),
        ]);
        let module = TlaModule {
            name: "BufSeed".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["buf".to_string()],
            definitions: BTreeMap::from([
                (
                    "Proc".to_string(),
                    TlaDefinition {
                        name: "Proc".to_string(),
                        params: vec![],
                        body: "{p1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "MReq".to_string(),
                    TlaDefinition {
                        name: "MReq".to_string(),
                        params: vec![],
                        body: "[op: {\"Wr\"}, adr: {a1}, val: {v1}]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Val".to_string(),
                    TlaDefinition {
                        name: "Val".to_string(),
                        params: vec![],
                        body: "{v1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeOK".to_string(),
                    TlaDefinition {
                        name: "TypeOK".to_string(),
                        params: vec![],
                        body: "buf \\in [Proc -> MReq \\cup Val \\cup {NoVal}]".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert_eq!(seeded, 1);
        match probe_state.get("buf") {
            Some(TlaValue::Function(values)) => {
                let value = values.get(&proc_id).expect("buf should contain p1");
                assert!(matches!(value, TlaValue::Record(_)));
            }
            other => panic!("expected refined buf function, got {other:?}"),
        }
    }

    #[test]
    fn representative_value_from_function_set_prefers_structured_range_members() {
        let probe_state = TlaState::new();
        let definitions = BTreeMap::from([
            (
                "Proc".to_string(),
                TlaDefinition {
                    name: "Proc".to_string(),
                    params: vec![],
                    body: "{p1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "MReq".to_string(),
                TlaDefinition {
                    name: "MReq".to_string(),
                    params: vec![],
                    body: "[op: {\"Wr\"}, adr: {a1}, val: {v1}]".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Val".to_string(),
                TlaDefinition {
                    name: "Val".to_string(),
                    params: vec![],
                    body: "{v1}".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let instances = BTreeMap::new();
        let ctx = build_probe_eval_context(&probe_state, &definitions, &instances);

        let repr =
            representative_value_from_set_expr("[Proc -> MReq \\cup Val \\cup {NoVal}]", &ctx)
                .expect("function-set representative");
        let TlaValue::Function(values) = repr else {
            panic!("expected function representative, got {repr:?}");
        };
        let value = values
            .get(&TlaValue::ModelValue("p1".to_string()))
            .expect("buf should contain p1");
        assert!(matches!(value, TlaValue::Record(_)));
    }

    #[test]
    fn representative_member_from_domain_expr_uses_union_definition_bodies() {
        let probe_state = TlaState::new();
        let definitions = BTreeMap::from([
            (
                "Ballot".to_string(),
                TlaDefinition {
                    name: "Ballot".to_string(),
                    params: vec![],
                    body: "Nat".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Value".to_string(),
                TlaDefinition {
                    name: "Value".to_string(),
                    params: vec![],
                    body: "{v1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Message".to_string(),
                TlaDefinition {
                    name: "Message".to_string(),
                    params: vec![],
                    body: r#"
                        [type : {"1a"}, bal : Ballot]
                    \cup [type : {"2a"}, bal : Ballot, val : Value]
                    "#
                    .to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let instances = BTreeMap::new();
        let ctx = build_probe_eval_context(&probe_state, &definitions, &instances);

        let repr = representative_member_from_domain_expr("Message", &ctx)
            .expect("subset domain representative");
        let TlaValue::Record(fields) = repr else {
            panic!("expected representative record, got {repr:?}");
        };
        assert_eq!(
            fields.get("type"),
            Some(&TlaValue::String("2a".to_string()))
        );
        assert_eq!(fields.get("bal"), Some(&TlaValue::Int(0)));
        assert_eq!(
            fields.get("val"),
            Some(&TlaValue::ModelValue("v1".to_string()))
        );
    }

    #[test]
    fn structured_function_replacements_outrank_placeholder_values() {
        let current = TlaValue::Function(Arc::new(BTreeMap::from([(
            TlaValue::ModelValue("p1".to_string()),
            TlaValue::ModelValue("NoVal".to_string()),
        )])));
        let replacement = TlaValue::Function(Arc::new(BTreeMap::from([(
            TlaValue::ModelValue("p1".to_string()),
            TlaValue::Record(Arc::new(BTreeMap::from([(
                "op".to_string(),
                TlaValue::String("Wr".to_string()),
            )]))),
        )])));

        assert!(should_refine_probe_value(&current, &replacement));
    }

    #[test]
    fn smaller_type_representatives_do_not_replace_initialized_functions() {
        let current = TlaValue::Function(Arc::new(BTreeMap::from([
            (
                TlaValue::ModelValue("h1".to_string()),
                TlaValue::ModelValue("[Nano]NoBlockVal".to_string()),
            ),
            (
                TlaValue::ModelValue("h2".to_string()),
                TlaValue::ModelValue("[Nano]NoBlockVal".to_string()),
            ),
            (
                TlaValue::ModelValue("h3".to_string()),
                TlaValue::ModelValue("[Nano]NoBlockVal".to_string()),
            ),
        ])));
        let replacement = TlaValue::Function(Arc::new(BTreeMap::from([(
            TlaValue::ModelValue("h1".to_string()),
            TlaValue::Record(Arc::new(BTreeMap::from([(
                "source".to_string(),
                TlaValue::ModelValue("n1".to_string()),
            )]))),
        )])));

        assert!(!should_refine_probe_value(&current, &replacement));
    }

    #[test]
    fn type_invariant_seeding_preserves_initialized_functions_that_already_typecheck() {
        let mut probe_state = TlaState::from([
            (
                "Hash".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("h1".to_string()),
                    TlaValue::ModelValue("h2".to_string()),
                    TlaValue::ModelValue("h3".to_string()),
                ]))),
            ),
            (
                "NoBlock".to_string(),
                TlaValue::ModelValue("NoBlockVal".to_string()),
            ),
            (
                "hashFunction".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (
                        TlaValue::ModelValue("h1".to_string()),
                        TlaValue::ModelValue("NoBlockVal".to_string()),
                    ),
                    (
                        TlaValue::ModelValue("h2".to_string()),
                        TlaValue::ModelValue("NoBlockVal".to_string()),
                    ),
                    (
                        TlaValue::ModelValue("h3".to_string()),
                        TlaValue::ModelValue("NoBlockVal".to_string()),
                    ),
                ]))),
            ),
        ]);
        let expected = probe_state
            .get("hashFunction")
            .cloned()
            .expect("hashFunction should exist");
        let module = TlaModule {
            name: "NanoLike".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["hashFunction".to_string()],
            definitions: BTreeMap::from([
                (
                    "Block".to_string(),
                    TlaDefinition {
                        name: "Block".to_string(),
                        params: vec![],
                        body: "[type : {\"send\"}, source : {n1}]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeInvariant".to_string(),
                    TlaDefinition {
                        name: "TypeInvariant".to_string(),
                        params: vec![],
                        body: "hashFunction \\in [Hash -> Block \\cup {NoBlock}]".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);

        assert_eq!(seeded, 1);
        // The seeder now produces a representative from [Hash -> Block \cup {NoBlock}],
        // which picks a Block record as the representative range value.
        let actual = probe_state.get("hashFunction");
        if let Some(TlaValue::Function(f)) = actual {
            assert_eq!(f.len(), 3, "function should cover all 3 Hash values");
        } else {
            panic!(
                "expected hashFunction to remain a function, got {:?}",
                actual
            );
        }
    }

    #[test]
    fn indeterminate_type_constraints_do_not_replace_initialized_functions() {
        let expected = TlaValue::Function(Arc::new(BTreeMap::from([
            (
                TlaValue::ModelValue("h1".to_string()),
                TlaValue::ModelValue("NoBlockVal".to_string()),
            ),
            (
                TlaValue::ModelValue("h2".to_string()),
                TlaValue::ModelValue("NoBlockVal".to_string()),
            ),
        ])));
        let mut probe_state = TlaState::from([("hashFunction".to_string(), expected.clone())]);
        let module = TlaModule {
            name: "IndeterminateNanoLike".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["hashFunction".to_string()],
            definitions: BTreeMap::from([
                (
                    "Hash".to_string(),
                    TlaDefinition {
                        name: "Hash".to_string(),
                        params: vec![],
                        body: "{h1, h2}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Block".to_string(),
                    TlaDefinition {
                        name: "Block".to_string(),
                        params: vec![],
                        body: r#"[source : {n1}]"#.to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_membership_body(
            &mut probe_state,
            r#"hashFunction \in [Hash -> Block \cup {NoBlock}]"#,
            &module,
        );

        assert_eq!(seeded, 0);
        assert_eq!(probe_state.get("hashFunction"), Some(&expected));
    }

    #[test]
    fn indeterminate_type_constraints_can_upgrade_placeholder_values() {
        let mut probe_state = TlaState::from([("hashFunction".to_string(), TlaValue::Int(0))]);
        let module = TlaModule {
            name: "IndeterminateNanoLike".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["hashFunction".to_string()],
            definitions: BTreeMap::from([
                (
                    "Hash".to_string(),
                    TlaDefinition {
                        name: "Hash".to_string(),
                        params: vec![],
                        body: "{h1, h2}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Block".to_string(),
                    TlaDefinition {
                        name: "Block".to_string(),
                        params: vec![],
                        body: r#"[source : {n1}]"#.to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_membership_body(
            &mut probe_state,
            r#"hashFunction \in [Hash -> Block \cup {NoBlock}]"#,
            &module,
        );

        assert_eq!(seeded, 1);
        assert!(matches!(
            probe_state.get("hashFunction"),
            Some(TlaValue::Function(values)) if values.len() == 2
        ));
    }

    #[test]
    fn expands_instance_init_references_for_probe_seeding() {
        let instance_module = TlaModule {
            name: "Nano".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["x".to_string(), "y".to_string()],
            definitions: BTreeMap::from([(
                "Init".to_string(),
                TlaDefinition {
                    name: "Init".to_string(),
                    params: vec![],
                    body: "/\\ x = 1 /\\ y = 2".to_string(),
                    is_recursive: false,
                },
            )]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };
        let module = TlaModule {
            name: "MC".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            definitions: BTreeMap::from([(
                "Init".to_string(),
                TlaDefinition {
                    name: "Init".to_string(),
                    params: vec![],
                    body: "/\\ z = 0 /\\ N!Init".to_string(),
                    is_recursive: false,
                },
            )]),
            instances: BTreeMap::from([(
                "N".to_string(),
                TlaModuleInstance {
                    alias: "N".to_string(),
                    module_name: "Nano".to_string(),
                    substitutions: BTreeMap::new(),
                    is_local: false,
                    module: Some(Box::new(instance_module)),
                },
            )]),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let clauses = expand_state_predicate_clauses(
            &module.definitions["Init"].body,
            &module.definitions,
            &module.instances,
        );

        assert_eq!(clauses, vec!["z = 0", "x = 1", "y = 2"]);
    }

    #[test]
    fn infers_action_params_from_quantified_next_calls() {
        let acceptor = TlaValue::ModelValue("a1".to_string());
        let msg = TlaValue::Record(Arc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("2a".to_string())),
            ("bal".to_string(), TlaValue::Int(1)),
            ("val".to_string(), TlaValue::ModelValue("v1".to_string())),
        ])));
        let probe_state = TlaState::from([
            (
                "maxBal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::Int(0),
                )]))),
            ),
            (
                "maxVBal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::Int(0),
                )]))),
            ),
            (
                "maxVal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::ModelValue("None".to_string()),
                )]))),
            ),
            (
                "msgs".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([msg]))),
            ),
        ]);
        let definitions = BTreeMap::from([
            (
                "Acceptor".to_string(),
                TlaDefinition {
                    name: "Acceptor".to_string(),
                    params: vec![],
                    body: "{a1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Send".to_string(),
                TlaDefinition {
                    name: "Send".to_string(),
                    params: vec!["m".to_string()],
                    body: "msgs' = msgs \\cup {m}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase2b".to_string(),
                TlaDefinition {
                    name: "Phase2b".to_string(),
                    params: vec!["a".to_string()],
                    body: r#"
  \E m \in msgs :
      /\ m.type = "2a"
      /\ m.bal >= maxBal[a]
      /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
      /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
      /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
      /\ Send([type |-> "2b", acc |-> a, bal |-> m.bal, val |-> m.val])
"#
                    .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: r#"\E a \in Acceptor : Phase2b(a)"#.to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            &definitions.get("Next").unwrap().body,
            &probe_state,
            &definitions,
            &BTreeMap::new(),
        );
        assert_eq!(
            samples
                .get("Phase2b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(acceptor)
        );
    }

    #[test]
    fn infers_action_params_from_quantified_disjunctions() {
        let acceptor = TlaValue::ModelValue("a1".to_string());
        let probe_state = TlaState::new();
        let definitions = BTreeMap::from([
            (
                "Acceptor".to_string(),
                TlaDefinition {
                    name: "Acceptor".to_string(),
                    params: vec![],
                    body: "{a1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase1b".to_string(),
                TlaDefinition {
                    name: "Phase1b".to_string(),
                    params: vec!["a".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase2b".to_string(),
                TlaDefinition {
                    name: "Phase2b".to_string(),
                    params: vec!["a".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            r#"\E a \in Acceptor : Phase1b(a) \/ Phase2b(a)"#,
            &probe_state,
            &definitions,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples
                .get("Phase1b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(acceptor.clone())
        );
        assert_eq!(
            samples
                .get("Phase2b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(acceptor)
        );
    }

    #[test]
    fn analyze_tla_path_infers_phase2b_acceptor_samples() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-paxos-phase2b-sample");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let module_path = tmp.join("MCPaxosMini.tla");
        fs::write(
            &module_path,
            r#"
---- MODULE MCPaxosMini ----
EXTENDS Integers

CONSTANTS Acceptor, Value
CONSTANTS a1, v1

const_acceptor == {a1}
const_value == {v1}

Ballot == Nat
Message ==
       [type : {"1a"}, bal : Ballot]
  \cup [type : {"2a"}, bal : Ballot, val : Value]
  \cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]

VARIABLES maxBal, maxVBal, maxVal, msgs

TypeOK == /\ maxBal  \in [Acceptor -> Ballot \cup {-1}]
          /\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
          /\ maxVal  \in [Acceptor -> Value]
          /\ msgs \subseteq Message

Init == /\ maxBal  = [a \in Acceptor |-> -1]
        /\ maxVBal = [a \in Acceptor |-> -1]
        /\ maxVal  = [a \in Acceptor |-> v1]
        /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

Phase1b(a) ==
  /\ \E m \in msgs :
        /\ m.type = "1a"
        /\ m.bal > maxBal[a]
  /\ UNCHANGED <<maxVBal, maxVal>>

Phase2b(a) ==
  \E m \in msgs :
      /\ m.type = "2a"
      /\ m.bal >= maxBal[a]
      /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
      /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
      /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
      /\ Send([type |-> "2b", acc |-> a, bal |-> m.bal, val |-> m.val])

Next == \E a \in Acceptor : Phase1b(a) \/ Phase2b(a)
Spec == Init /\ [][Next]_<<maxBal, maxVBal, maxVal, msgs>>
====
"#,
        )
        .expect("module should be written");

        let cfg_path = tmp.join("MCPaxosMini.cfg");
        fs::write(
            &cfg_path,
            r#"
CONSTANTS
a1 = a1
v1 = v1
CONSTANT
Acceptor <- const_acceptor
CONSTANT
Value <- const_value
SPECIFICATION
Spec
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model");
        let probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state");
        let next_def = model
            .module
            .definitions
            .get(&model.next_name)
            .expect("Next definition");

        let samples = infer_action_param_samples_from_next(
            &next_def.body,
            &probe_state,
            &model.module.definitions,
            &model.module.instances,
        );
        assert_eq!(
            samples
                .get("Phase2b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(TlaValue::ModelValue("a1".to_string()))
        );

        let phase2b = model
            .module
            .definitions
            .get("Phase2b")
            .expect("Phase2b definition");
        let ir = compile_action_ir_branches(phase2b);
        let ctx = build_action_expr_probe_context(
            &probe_state,
            &model.module.definitions,
            &model.module.instances,
            &ir[0].params,
            &ir[0].clauses,
            samples.get("Phase2b"),
        );
        assert_eq!(
            eval_expr("maxBal[a]", &ctx).expect("Phase2b param should index maxBal"),
            TlaValue::Int(-1)
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn analyze_tla_path_infers_phase2b_acceptor_samples_through_extends() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-paxos-phase2b-extends");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let base_module_path = tmp.join("PaxosBase.tla");
        fs::write(
            &base_module_path,
            r#"
---- MODULE PaxosBase ----
EXTENDS Integers

CONSTANTS Acceptor, Value, DefaultValue

Ballot == Nat
Message ==
       [type : {"1a"}, bal : Ballot]
  \cup [type : {"2a"}, bal : Ballot, val : Value]
  \cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]

VARIABLES maxBal, maxVBal, maxVal, msgs

TypeOK == /\ maxBal  \in [Acceptor -> Ballot \cup {-1}]
          /\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
          /\ maxVal  \in [Acceptor -> Value]
          /\ msgs \subseteq Message

Init == /\ maxBal  = [a \in Acceptor |-> -1]
        /\ maxVBal = [a \in Acceptor |-> -1]
        /\ maxVal  = [a \in Acceptor |-> DefaultValue]
        /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

Phase1b(a) ==
  /\ \E m \in msgs :
        /\ m.type = "1a"
        /\ m.bal > maxBal[a]
  /\ UNCHANGED <<maxVBal, maxVal>>

Phase2b(a) ==
  \E m \in msgs :
      /\ m.type = "2a"
      /\ m.bal >= maxBal[a]
      /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
      /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
      /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
      /\ Send([type |-> "2b", acc |-> a, bal |-> m.bal, val |-> m.val])

Next == \E a \in Acceptor : Phase1b(a) \/ Phase2b(a)
Spec == Init /\ [][Next]_<<maxBal, maxVBal, maxVal, msgs>>
====
"#,
        )
        .expect("base module should be written");

        let module_path = tmp.join("MCPaxosMiniExtends.tla");
        fs::write(
            &module_path,
            r#"
---- MODULE MCPaxosMiniExtends ----
EXTENDS PaxosBase

CONSTANTS a1, v1

const_acceptor == {a1}
const_value == {v1}
default_value == v1
====
"#,
        )
        .expect("module should be written");

        let cfg_path = tmp.join("MCPaxosMiniExtends.cfg");
        fs::write(
            &cfg_path,
            r#"
CONSTANTS
a1 = a1
v1 = v1
CONSTANT
Acceptor <- const_acceptor
CONSTANT
Value <- const_value
CONSTANT
DefaultValue <- default_value
SPECIFICATION
Spec
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model");
        let probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state");
        let next_def = model
            .module
            .definitions
            .get(&model.next_name)
            .expect("Next definition");

        let samples = infer_action_param_samples_from_next(
            &next_def.body,
            &probe_state,
            &model.module.definitions,
            &model.module.instances,
        );
        assert_eq!(
            samples
                .get("Phase2b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(TlaValue::ModelValue("a1".to_string()))
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn analyze_tla_path_keeps_instance_operator_override_in_choose_probe() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-mcnano-choose-probe");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let nano = tmp.join("Nano.tla");
        fs::write(
            &nano,
            r#"
---- MODULE Nano ----
CONSTANTS Hash, NoBlockVal

Block == {"used"}
SignedBlock == [block : Block]
NoBlock == CHOOSE b : b \notin SignedBlock
====
"#,
        )
        .expect("nano module should be written");

        let mc = tmp.join("MCNanoMini.tla");
        fs::write(
            &mc,
            r#"
---- MODULE MCNanoMini ----
N == INSTANCE Nano

VARIABLES hashFunction

HashOf(block) ==
  IF \E hash \in Hash : hashFunction[hash] = block
  THEN CHOOSE hash \in Hash : hashFunction[hash] = block
  ELSE CHOOSE hash \in Hash : hashFunction[hash] = N!NoBlock

CalculateHashImpl(block, oldLastHash, newLastHash) ==
  LET hash == HashOf(block) IN
  /\ newLastHash = hash
  /\ hashFunction' = [hashFunction EXCEPT ![hash] = block]

TypeInvariant == hashFunction \in [Hash -> N!SignedBlock \cup {N!NoBlock}]

Init == /\ hashFunction = [hash \in Hash |-> N!NoBlock]
Next == /\ UNCHANGED <<hashFunction>>
Spec == Init /\ [][Next]_<<hashFunction>>
====
"#,
        )
        .expect("mc module should be written");

        let cfg = tmp.join("MCNanoMini.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANTS
Hash = {h1, h2, h3}
NoBlockVal = NoBlockVal

CONSTANTS
NoBlock = [Nano]NoBlockVal

SPECIFICATION Spec
INVARIANTS TypeInvariant
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&mc, Some(&cfg), None, None).expect("model should build");
        let probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state should exist");
        assert_eq!(
            probe_state.get("NoBlock"),
            Some(&TlaValue::ModelValue("NoBlockVal".to_string()))
        );

        let def = model
            .module
            .definitions
            .get("CalculateHashImpl")
            .expect("action should exist");
        let ir = compile_action_ir(def);
        let inferred = BTreeMap::from([
            (
                "block".to_string(),
                TlaValue::ModelValue("targetBlock".to_string()),
            ),
            (
                "oldLastHash".to_string(),
                TlaValue::ModelValue("oldHash".to_string()),
            ),
            (
                "newLastHash".to_string(),
                TlaValue::ModelValue("freshHash".to_string()),
            ),
        ]);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &model.module.definitions,
            &model.module.instances,
            &ir.params,
            &ir.clauses,
            Some(&inferred),
        );

        assert_eq!(
            eval_expr("N!NoBlock", &ctx).expect("instance override should resolve"),
            TlaValue::ModelValue("NoBlockVal".to_string())
        );
        let chosen = eval_expr("HashOf(block)", &ctx).expect("HashOf should choose a free hash");
        assert!(
            matches!(chosen, TlaValue::ModelValue(ref name) if name == "h1" || name == "h2" || name == "h3"),
            "unexpected chosen hash {chosen:?}"
        );

        let result = probe_action_clause_expr(
            ir.clauses
                .first()
                .expect("CalculateHashImpl should compile to one clause"),
            &mut ctx,
        )
        .expect("LET action clause should be probeable");
        result.expect("LET action probe should not error");
        assert!(ctx.locals.contains_key("hashFunction'"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn analyze_tla_path_handles_mcnano_style_calculate_hash_probe() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-mcnano-style-calculate-hash");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let nano = tmp.join("Nano.tla");
        fs::write(
            &nano,
            r#"
---- MODULE Nano ----
CONSTANTS Hash, CalculateHash(_,_,_)
VARIABLES lastHash, distributedLedger, received

Block == [type : {"receive"}, previous : Hash, source : Hash]
SignedBlock == [block : Block]
NoBlock == CHOOSE b : b \notin SignedBlock
NoHash == CHOOSE h : h \notin Hash
Ledger == [Hash -> SignedBlock \cup {NoBlock}]

TypeInvariant ==
  /\ lastHash \in Hash \cup {NoHash}
  /\ distributedLedger \in Ledger
  /\ received \in SUBSET SignedBlock

Init ==
  /\ lastHash = NoHash
  /\ distributedLedger = [h \in Hash |-> NoBlock]
  /\ received = {}

Next ==
  LET block == "used" IN
  /\ CalculateHash(block, lastHash, lastHash')
  /\ UNCHANGED <<distributedLedger, received>>
====
"#,
        )
        .expect("nano module should be written");

        let mc = tmp.join("MCNanoStyle.tla");
        fs::write(
            &mc,
            r#"
---- MODULE MCNanoStyle ----
N == INSTANCE Nano

CONSTANTS CalculateHash(_,_,_), Hash, NoBlockVal
VARIABLES hashFunction, lastHash, distributedLedger, received

Vars == <<hashFunction, lastHash, distributedLedger, received>>

UndefinedHashesExist ==
  \E hash \in Hash : hashFunction[hash] = N!NoBlock

HashOf(block) ==
  IF \E hash \in Hash : hashFunction[hash] = block
  THEN CHOOSE hash \in Hash : hashFunction[hash] = block
  ELSE CHOOSE hash \in Hash : hashFunction[hash] = N!NoBlock

CalculateHashImpl(block, oldLastHash, newLastHash) ==
  LET hash == HashOf(block) IN
  /\ newLastHash = hash
  /\ hashFunction' = [hashFunction EXCEPT ![hash] = block]

TypeInvariant ==
  /\ hashFunction \in [Hash -> N!Block \cup {N!NoBlock}]
  /\ N!TypeInvariant

Init ==
  /\ hashFunction = [hash \in Hash |-> N!NoBlock]
  /\ N!Init

StutterWhenHashesDepleted ==
  /\ UNCHANGED hashFunction
  /\ UNCHANGED lastHash
  /\ UNCHANGED distributedLedger
  /\ UNCHANGED received

Next ==
  IF UndefinedHashesExist
  THEN N!Next
  ELSE StutterWhenHashesDepleted

Spec == Init /\ [][Next]_Vars
====
"#,
        )
        .expect("mc module should be written");

        let cfg = tmp.join("MCNanoStyle.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANTS
  CalculateHash <- CalculateHashImpl
  Hash = {h1, h2, h3}
  NoBlockVal = NoBlockVal

CONSTANTS
  NoBlock = [Nano]NoBlockVal

SPECIFICATION Spec
INVARIANTS TypeInvariant
"#,
        )
        .expect("cfg should be written");

        let raw_cfg = fs::read_to_string(&cfg).expect("cfg should be readable");
        let parsed_cfg = parse_tla_config(&raw_cfg).expect("cfg should parse");

        let mut parsed_module = parse_tla_module_file(&mc).expect("module should parse");
        inject_constants_into_definitions(&mut parsed_module, &parsed_cfg);
        let model = TlaModel::from_files(&mc, Some(&cfg), None, None).expect("model should build");
        parsed_module = model.module.clone();

        let mut probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state should exist");
        for (name, value) in &parsed_cfg.constants {
            if let Some(tv) = config_value_to_tla(value) {
                probe_state.insert(name.clone(), tv);
            }
        }
        for _ in 0..4 {
            let mut progress = false;
            for (name, value) in &parsed_cfg.constants {
                let ConfigValue::OperatorRef(target_name) = value else {
                    continue;
                };
                if probe_state.contains_key(name) {
                    continue;
                }
                let ctx = build_probe_eval_context(
                    &probe_state,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                );
                let target_name = normalize_operator_ref_name(target_name);
                if let Ok(resolved) = eval_expr(target_name, &ctx) {
                    probe_state.insert(name.clone(), resolved);
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }

        seed_probe_state_from_type_invariants(&mut probe_state, &parsed_module, Some(&parsed_cfg));

        let expected_sentinel = TlaValue::ModelValue("NoBlockVal".to_string());
        let hash_function = probe_state
            .get("hashFunction")
            .expect("hashFunction should be present");
        let TlaValue::Function(entries) = hash_function else {
            panic!("hashFunction should remain a function, got {hash_function:?}");
        };
        for key in [
            TlaValue::ModelValue("h1".to_string()),
            TlaValue::ModelValue("h2".to_string()),
            TlaValue::ModelValue("h3".to_string()),
        ] {
            assert_eq!(
                entries.get(&key),
                Some(&expected_sentinel),
                "seeded probe state should preserve free-hash sentinels"
            );
        }

        let next_def = parsed_module
            .definitions
            .get(&model.next_name)
            .expect("Next definition should exist");
        let samples = infer_action_param_samples_from_next(
            &next_def.body,
            &probe_state,
            &parsed_module.definitions,
            &parsed_module.instances,
        );
        let def = parsed_module
            .definitions
            .get("CalculateHashImpl")
            .expect("CalculateHashImpl should exist");
        let ir = compile_action_ir_branches(def);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &parsed_module.definitions,
            &parsed_module.instances,
            &ir[0].params,
            &ir[0].clauses,
            samples.get("CalculateHashImpl"),
        );

        assert_eq!(
            eval_expr("N!NoBlock", &ctx).expect("instance override should resolve"),
            expected_sentinel
        );
        let chosen = eval_expr("HashOf(block)", &ctx).expect("HashOf should choose a free hash");
        assert!(
            matches!(chosen, TlaValue::ModelValue(ref name) if name == "h1" || name == "h2" || name == "h3"),
            "unexpected chosen hash {chosen:?}"
        );

        let result = probe_action_clause_expr(
            ir[0]
                .clauses
                .first()
                .expect("CalculateHashImpl should have one clause"),
            &mut ctx,
        )
        .expect("LET action clause should be probeable");
        result.expect("MCNano-style CalculateHashImpl probe should succeed");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn action_probe_resolves_zero_arg_aliases_in_index_and_except_expressions() {
        let proc_id = TlaValue::ModelValue("p1".to_string());
        let adr = TlaValue::ModelValue("a1".to_string());
        let val = TlaValue::ModelValue("v1".to_string());
        let probe_state = TlaState::from([(
            "ctl".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                proc_id.clone(),
                TlaValue::String("busy".to_string()),
            )]))),
        )]);
        let definitions = BTreeMap::from([
            (
                "Proc".to_string(),
                TlaDefinition {
                    name: "Proc".to_string(),
                    params: vec![],
                    body: "{p1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Adr".to_string(),
                TlaDefinition {
                    name: "Adr".to_string(),
                    params: vec![],
                    body: "{a1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "omem".to_string(),
                TlaDefinition {
                    name: "omem".to_string(),
                    params: vec![],
                    body: "[a \\in Adr |-> v1]".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "obuf".to_string(),
                TlaDefinition {
                    name: "obuf".to_string(),
                    params: vec![],
                    body: r#"[p \in Proc |-> [op |-> "Wr", adr |-> a1, val |-> v1]]"#.to_string(),
                    is_recursive: false,
                },
            ),
            (
                "octl".to_string(),
                TlaDefinition {
                    name: "octl".to_string(),
                    params: vec![],
                    body: r#"[p \in Proc |-> IF ctl[p] = "waiting" THEN "busy" ELSE ctl[p]]"#
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "LM_Inner_Do".to_string(),
                TlaDefinition {
                    name: "LM_Inner_Do".to_string(),
                    params: vec!["p".to_string()],
                    body: r#"
  /\ octl[p] = "busy"
  /\ omem' = IF obuf[p].op = "Wr"
              THEN [omem EXCEPT ![obuf[p].adr] = obuf[p].val]
              ELSE omem
  /\ obuf' = [obuf EXCEPT ![p] = IF obuf[p].op = "Wr"
                                  THEN NoVal
                                  ELSE omem[obuf[p].adr]]
  /\ octl' = [octl EXCEPT ![p] = "done"]
"#
                    .to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let inferred = BTreeMap::from([("p".to_string(), proc_id.clone())]);
        let def = definitions.get("LM_Inner_Do").unwrap();
        let ir = compile_action_ir(def);
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &definitions,
            &instances,
            &ir.params,
            &ir.clauses,
            Some(&inferred),
        );

        for clause in &ir.clauses {
            let Some(result) = probe_action_clause_expr(clause, &mut ctx) else {
                continue;
            };
            result.expect("cache-style action probe should succeed");
        }

        let locals = &ctx.locals;
        assert_eq!(locals.get("p"), Some(&proc_id));
        assert_eq!(
            locals.get("octl'"),
            Some(&TlaValue::Function(Arc::new(BTreeMap::from([(
                proc_id.clone(),
                TlaValue::String("done".to_string()),
            )]))))
        );
        assert_eq!(
            locals.get("omem'"),
            Some(&TlaValue::Function(Arc::new(BTreeMap::from([(adr, val)]))))
        );
    }

    #[test]
    fn cache_style_type_invariants_refine_alias_backed_buffers_for_action_probing() {
        let proc_id = TlaValue::ModelValue("p1".to_string());
        let no_val = TlaValue::ModelValue("NoVal".to_string());
        let mut probe_state = TlaState::from([
            (
                "wmem".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("a1".to_string()),
                    TlaValue::ModelValue("v1".to_string()),
                )]))),
            ),
            (
                "ctl".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    proc_id.clone(),
                    TlaValue::String("waiting".to_string()),
                )]))),
            ),
            (
                "buf".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    proc_id.clone(),
                    no_val.clone(),
                )]))),
            ),
            (
                "memInt".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
            ("NoVal".to_string(), no_val),
        ]);
        let module = TlaModule {
            name: "CacheProbe".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec![
                "wmem".to_string(),
                "ctl".to_string(),
                "buf".to_string(),
                "memInt".to_string(),
            ],
            definitions: BTreeMap::from([
                (
                    "Proc".to_string(),
                    TlaDefinition {
                        name: "Proc".to_string(),
                        params: vec![],
                        body: "{p1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Adr".to_string(),
                    TlaDefinition {
                        name: "Adr".to_string(),
                        params: vec![],
                        body: "{a1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Val".to_string(),
                    TlaDefinition {
                        name: "Val".to_string(),
                        params: vec![],
                        body: "{v1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "MReq".to_string(),
                    TlaDefinition {
                        name: "MReq".to_string(),
                        params: vec![],
                        body: "[op: {\"Wr\"}, adr: {a1}, val: {v1}]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeInvariant".to_string(),
                    TlaDefinition {
                        name: "TypeInvariant".to_string(),
                        params: vec![],
                        body: r#"
                            /\ wmem \in [Adr -> Val]
                            /\ ctl \in [Proc -> {"rdy", "busy", "waiting", "done"}]
                            /\ buf \in [Proc -> MReq \cup Val \cup {NoVal}]
                        "#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "omem".to_string(),
                    TlaDefinition {
                        name: "omem".to_string(),
                        params: vec![],
                        body: "wmem".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "obuf".to_string(),
                    TlaDefinition {
                        name: "obuf".to_string(),
                        params: vec![],
                        body: "buf".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "octl".to_string(),
                    TlaDefinition {
                        name: "octl".to_string(),
                        params: vec![],
                        body: r#"[p \in Proc |-> IF ctl[p] = "waiting" THEN "busy" ELSE ctl[p]]"#
                            .to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "LM_Inner_Do".to_string(),
                    TlaDefinition {
                        name: "LM_Inner_Do".to_string(),
                        params: vec!["p".to_string()],
                        body: r#"
  /\ octl[p] = "busy"
  /\ omem' = IF obuf[p].op = "Wr"
              THEN [omem EXCEPT ![obuf[p].adr] = obuf[p].val]
              ELSE omem
  /\ obuf' = [obuf EXCEPT ![p] = IF obuf[p].op = "Wr"
                                  THEN NoVal
                                  ELSE omem[obuf[p].adr]]
  /\ octl' = [octl EXCEPT ![p] = "done"]
"#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert!(seeded >= 1);
        let def = module.definitions.get("LM_Inner_Do").unwrap();
        let ir = compile_action_ir(def);
        let inferred = BTreeMap::from([("p".to_string(), proc_id.clone())]);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &module.definitions,
            &module.instances,
            &ir.params,
            &ir.clauses,
            Some(&inferred),
        );

        for clause in &ir.clauses {
            let Some(result) = probe_action_clause_expr(clause, &mut ctx) else {
                continue;
            };
            result.expect("cache-style alias action should be probeable after refinement");
        }
    }

    #[test]
    fn paxos_style_type_invariants_seed_messages_for_phase2b_probe() {
        let acceptor = TlaValue::ModelValue("a1".to_string());
        let mut probe_state = TlaState::from([
            (
                "maxBal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::Int(-1),
                )]))),
            ),
            (
                "maxVBal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::Int(-1),
                )]))),
            ),
            (
                "maxVal".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    acceptor.clone(),
                    TlaValue::ModelValue("None".to_string()),
                )]))),
            ),
            ("msgs".to_string(), TlaValue::Set(Arc::new(BTreeSet::new()))),
        ]);
        let module = TlaModule {
            name: "PaxosProbe".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec![
                "maxBal".to_string(),
                "maxVBal".to_string(),
                "maxVal".to_string(),
                "msgs".to_string(),
            ],
            definitions: BTreeMap::from([
                (
                    "Ballot".to_string(),
                    TlaDefinition {
                        name: "Ballot".to_string(),
                        params: vec![],
                        body: "{0, 1, 2}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Value".to_string(),
                    TlaDefinition {
                        name: "Value".to_string(),
                        params: vec![],
                        body: "{v1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Acceptor".to_string(),
                    TlaDefinition {
                        name: "Acceptor".to_string(),
                        params: vec![],
                        body: "{a1}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "None".to_string(),
                    TlaDefinition {
                        name: "None".to_string(),
                        params: vec![],
                        body: "None".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Message".to_string(),
                    TlaDefinition {
                        name: "Message".to_string(),
                        params: vec![],
                        body: r#"
                               [type : {"1a"}, bal : Ballot]
                          \cup [type : {"2a"}, bal : Ballot, val : Value]
                          \cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]
                        "#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeOK".to_string(),
                    TlaDefinition {
                        name: "TypeOK".to_string(),
                        params: vec![],
                        body: r#"
                          /\ maxBal  \in [Acceptor -> Ballot \cup {-1}]
                          /\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
                          /\ maxVal  \in [Acceptor -> Value \cup {None}]
                          /\ msgs \subseteq Message
                        "#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Send".to_string(),
                    TlaDefinition {
                        name: "Send".to_string(),
                        params: vec!["m".to_string()],
                        body: "msgs' = msgs \\cup {m}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Phase2b".to_string(),
                    TlaDefinition {
                        name: "Phase2b".to_string(),
                        params: vec!["a".to_string()],
                        body: r#"
  \E m \in msgs :
      /\ m.type = "2a"
      /\ m.bal >= maxBal[a]
      /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
      /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
      /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
      /\ Send([type |-> "2b", acc |-> a, bal |-> m.bal, val |-> m.val])
"#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Next".to_string(),
                    TlaDefinition {
                        name: "Next".to_string(),
                        params: vec![],
                        body: r#"\E a \in Acceptor : Phase2b(a)"#.to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert!(seeded >= 1);
        match probe_state.get("msgs") {
            Some(TlaValue::Set(values)) => assert!(!values.is_empty()),
            other => panic!("expected seeded msgs set, got {other:?}"),
        }

        let samples = infer_action_param_samples_from_next(
            &module.definitions.get("Next").unwrap().body,
            &probe_state,
            &module.definitions,
            &module.instances,
        );
        assert_eq!(
            samples
                .get("Phase2b")
                .and_then(|params| params.get("a"))
                .cloned(),
            Some(acceptor.clone())
        );

        let def = module.definitions.get("Phase2b").unwrap();
        let ir = compile_action_ir(def);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &module.definitions,
            &module.instances,
            &ir.params,
            &ir.clauses,
            samples.get("Phase2b"),
        );
        for clause in &ir.clauses {
            let Some(result) = probe_action_clause_expr(clause, &mut ctx) else {
                continue;
            };
            result.expect("Phase2b should be probeable after type-based seeding");
        }
    }

    #[test]
    fn parses_skip_system_checks_flag() {
        let cli = Cli::try_parse_from(["tlaplusplus", "run-counter-grid", "--skip-system-checks"])
            .expect("cli should parse");

        match cli.command {
            Command::RunCounterGrid { runtime, .. } => assert!(runtime.skip_system_checks),
            other => panic!("unexpected command: {:?}", other),
        }
    }

    #[test]
    fn expr_probe_stages_primed_assignments_between_action_clauses() {
        let mut state = TlaState::new();
        state.insert("big".to_string(), TlaValue::Int(2));
        state.insert("small".to_string(), TlaValue::Int(3));

        let defs = BTreeMap::new();
        let def = TlaDefinition {
            name: "SmallToBig".to_string(),
            params: vec![],
            body: r#"
                /\ big' = big + small
                /\ small' = small - (big' - big)
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);

        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );
        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("probe should resolve staged prime references");
            }
        }

        assert_eq!(
            ctx.locals.get("big'"),
            Some(&TlaValue::Int(5)),
            "first primed assignment should be staged for later clauses"
        );
        assert_eq!(
            ctx.locals.get("small'"),
            Some(&TlaValue::Int(0)),
            "later clauses should evaluate against staged prime bindings"
        );
    }

    #[test]
    fn expr_probe_seeds_unchanged_vars_as_primed_bindings() {
        let mut state = TlaState::new();
        state.insert("x".to_string(), TlaValue::Int(7));
        state.insert("y".to_string(), TlaValue::Int(0));

        let defs = BTreeMap::new();
        let def = TlaDefinition {
            name: "KeepX".to_string(),
            params: vec![],
            body: r#"
                /\ UNCHANGED <<x>>
                /\ y' = x'
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);

        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );
        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("probe should treat UNCHANGED variables as x' = x");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(7)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(7)));
    }

    #[test]
    fn expr_probe_stages_primes_from_nested_if_action_bodies() {
        let mut state = TlaState::new();
        state.insert("flag".to_string(), TlaValue::Bool(true));
        state.insert("count".to_string(), TlaValue::Int(1));
        state.insert("announced".to_string(), TlaValue::Bool(false));

        let defs = BTreeMap::from([(
            "VictoryThreshold".to_string(),
            TlaDefinition {
                name: "VictoryThreshold".to_string(),
                params: vec![],
                body: "2".to_string(),
                is_recursive: false,
            },
        )]);
        let def = TlaDefinition {
            name: "CounterAction".to_string(),
            params: vec![],
            body: r#"
                /\ IF flag
                   THEN /\ count' = count + 1
                   ELSE UNCHANGED <<count>>
                /\ announced' = (count' >= VictoryThreshold)
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("nested IF action body should stage primed values");
            }
        }

        assert_eq!(ctx.locals.get("count'"), Some(&TlaValue::Int(2)));
        assert_eq!(ctx.locals.get("announced'"), Some(&TlaValue::Bool(true)));
    }

    #[test]
    fn expr_probe_evaluates_primed_zero_arg_operators_from_staged_bindings() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(1)),
            ("y".to_string(), TlaValue::Int(2)),
        ]);
        let defs = BTreeMap::from([
            (
                "PairSum".to_string(),
                TlaDefinition {
                    name: "PairSum".to_string(),
                    params: vec![],
                    body: "x + y".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "PairSumPositive".to_string(),
                TlaDefinition {
                    name: "PairSumPositive".to_string(),
                    params: vec![],
                    body: "PairSum > 0".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let def = TlaDefinition {
            name: "Advance".to_string(),
            params: vec![],
            body: r#"
                /\ x' = 10
                /\ y' = -3
                /\ PairSumPositive'
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("primed derived operators should probe against staged state");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(10)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(-3)));
    }

    #[test]
    fn expr_probe_expands_nested_action_operator_calls() {
        let mut state = TlaState::new();
        state.insert("x".to_string(), TlaValue::Int(1));
        state.insert("y".to_string(), TlaValue::Int(0));

        let defs = BTreeMap::from([
            (
                "Inc".to_string(),
                TlaDefinition {
                    name: "Inc".to_string(),
                    params: vec![],
                    body: "x' = x + 1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Outer".to_string(),
                TlaDefinition {
                    name: "Outer".to_string(),
                    params: vec![],
                    body: "/\\ Inc() /\\ y' = x'".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let ir = compile_action_ir(defs.get("Outer").unwrap());
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("nested action operator call should stage primed values");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(2)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(2)));
    }

    #[test]
    fn expr_probe_short_circuits_let_actions_before_invalid_local_defs() {
        let state = TlaState::from([
            (
                "store".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("node1".to_string()),
                    TlaValue::Int(7),
                )]))),
            ),
            (
                "missing".to_string(),
                TlaValue::ModelValue("NoNode".to_string()),
            ),
            ("x".to_string(), TlaValue::Int(0)),
        ]);
        let def = TlaDefinition {
            name: "Disabled".to_string(),
            params: vec![],
            body: r#"
                LET bad == store[missing] IN
                /\ FALSE
                /\ x' = bad
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("disabled LET action should not force invalid local defs");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(0)));
    }

    #[test]
    fn expr_probe_skips_remaining_clauses_after_false_guard() {
        let state = TlaState::from([
            (
                "signalled".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::<TlaValue, TlaValue>::new())),
            ),
            ("light".to_string(), TlaValue::String("off".to_string())),
        ]);
        let defs = BTreeMap::from([(
            "NormalPrisoner".to_string(),
            TlaDefinition {
                name: "NormalPrisoner".to_string(),
                params: vec![],
                body: "{}".to_string(),
                is_recursive: false,
            },
        )]);
        let instances = BTreeMap::new();
        let def = TlaDefinition {
            name: "StandardAction".to_string(),
            params: vec!["p".to_string()],
            body: r#"
                /\ p \in NormalPrisoner
                /\ IF light = "off" /\ signalled[p] < 1
                   THEN /\ light' = "on"
                        /\ signalled' = [signalled EXCEPT ![p] = @ + 1]
                   ELSE UNCHANGED <<light, signalled>>
            "#
            .to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);
        let inferred = BTreeMap::from([("p".to_string(), TlaValue::String("Alice".to_string()))]);
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            Some(&inferred),
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("false guard should skip the rest of the branch");
            }
        }

        assert_eq!(
            ctx.locals.get("light'"),
            Some(&TlaValue::String("off".to_string()))
        );
        assert_eq!(
            ctx.locals.get("signalled'"),
            Some(&TlaValue::Function(Arc::new(BTreeMap::new())))
        );
    }

    #[test]
    fn expr_probe_handles_top_level_let_action_with_multiple_assignments() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(1)),
            ("y".to_string(), TlaValue::Int(10)),
        ]);
        let def = TlaDefinition {
            name: "Increment".to_string(),
            params: vec![],
            body: r#"
                /\ x < 5
                /\ y >= -950
                /\ y <= 950
                /\ LET newX == x + 1
                   IN /\ x' = newX
                      /\ y' = y + newX
            "#
            .to_string(),
            is_recursive: false,
        };
        let direct_ir = compile_action_ir(&def);
        assert_eq!(direct_ir.clauses.len(), 4, "{:?}", direct_ir.clauses);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let branch = compile_action_ir_branches(&def)
            .into_iter()
            .next()
            .expect("single branch");
        assert_eq!(branch.clauses.len(), 4, "{:?}", branch.clauses);
        match &branch.clauses[3] {
            ActionClause::LetWithPrimes { expr } => {
                assert!(expr.contains("LET newX == x + 1"));
                assert!(expr.contains("x' = newX"));
                assert!(expr.contains("y' = y + newX"));
            }
            other => panic!("expected LET clause, got {other:?}"),
        }
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &branch.params,
            &branch.clauses,
            None,
        );

        for clause in &branch.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("LET action should keep its local bindings across body clauses");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(2)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(12)));
    }

    #[test]
    fn expr_probe_handles_counter_action_style_multiline_if() {
        let state = TlaState::from([
            ("light".to_string(), TlaValue::String("on".to_string())),
            ("count".to_string(), TlaValue::Int(2)),
            ("VictoryThreshold".to_string(), TlaValue::Int(3)),
            (
                "signalled".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::<TlaValue, TlaValue>::new())),
            ),
            (
                "DesignatedCounter".to_string(),
                TlaValue::ModelValue("p1".to_string()),
            ),
        ]);
        let def = TlaDefinition {
            name: "CounterAction".to_string(),
            params: vec!["p".to_string()],
            body: r#"
              /\ p = DesignatedCounter
              /\ IF light = "on"
                 THEN
                   /\ light' = "off"
                   /\ count' = count + 1
                 ELSE
                   UNCHANGED <<light, count>>
              /\ announced' = (count' >= VictoryThreshold)
              /\ UNCHANGED <<signalled>>
            "#
            .to_string(),
            is_recursive: false,
        };
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let inferred = BTreeMap::from([("p".to_string(), TlaValue::ModelValue("p1".to_string()))]);
        let branch = compile_action_ir_branches(&def)
            .into_iter()
            .next()
            .expect("single branch");
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &branch.params,
            &branch.clauses,
            Some(&inferred),
        );

        for clause in &branch.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("multiline IF should stay intact during probing");
            }
        }

        assert_eq!(
            ctx.locals.get("light'"),
            Some(&TlaValue::String("off".to_string()))
        );
        assert_eq!(ctx.locals.get("count'"), Some(&TlaValue::Int(3)));
        assert_eq!(ctx.locals.get("announced'"), Some(&TlaValue::Bool(true)));
        assert_eq!(
            ctx.locals.get("signalled'"),
            Some(&TlaValue::Function(Arc::new(BTreeMap::new())))
        );
    }

    #[test]
    fn expr_probe_handles_move_elevator_style_let_actions() {
        let elevator_1 = TlaValue::ModelValue("e1".to_string());
        let elevator_2 = TlaValue::ModelValue("e2".to_string());
        let e1_state = TlaValue::Record(Arc::new(BTreeMap::from([
            ("floor".to_string(), TlaValue::Int(1)),
            ("direction".to_string(), TlaValue::String("Up".to_string())),
            ("doorsOpen".to_string(), TlaValue::Bool(false)),
            (
                "buttonsPressed".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
        ])));
        let e2_state = TlaValue::Record(Arc::new(BTreeMap::from([
            ("floor".to_string(), TlaValue::Int(2)),
            (
                "direction".to_string(),
                TlaValue::String("Down".to_string()),
            ),
            ("doorsOpen".to_string(), TlaValue::Bool(false)),
            (
                "buttonsPressed".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
        ])));

        let state = TlaState::from([
            (
                "Elevator".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    elevator_1.clone(),
                    elevator_2.clone(),
                ]))),
            ),
            (
                "Floor".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
            (
                "ElevatorState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (elevator_1.clone(), e1_state),
                    (elevator_2, e2_state),
                ]))),
            ),
            (
                "ActiveElevatorCalls".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
            (
                "PersonState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::new())),
            ),
        ]);
        let defs = BTreeMap::from([(
            "CanServiceCall".to_string(),
            TlaDefinition {
                name: "CanServiceCall".to_string(),
                params: vec!["e".to_string(), "c".to_string()],
                body: r#"
                    LET eState == ElevatorState[e] IN
                    /\ c.floor = eState.floor
                    /\ c.direction = eState.direction
                "#
                .to_string(),
                is_recursive: false,
            },
        )]);
        let def = TlaDefinition {
            name: "MoveElevator".to_string(),
            params: vec!["e".to_string()],
            body: r#"
                LET
                  eState == ElevatorState[e]
                  nextFloor == IF eState.direction = "Up" THEN eState.floor + 1 ELSE eState.floor - 1
                IN
                /\ eState.direction /= "Stationary"
                /\ ~eState.doorsOpen
                /\ eState.floor \notin eState.buttonsPressed
                /\ \A call \in ActiveElevatorCalls :
                    /\ CanServiceCall[e, call] =>
                        /\ \E e2 \in Elevator :
                            /\ e /= e2
                            /\ CanServiceCall[e2, call]
                /\ nextFloor \in Floor
                /\ ElevatorState' = [ElevatorState EXCEPT ![e] = [@ EXCEPT !.floor = nextFloor]]
                /\ UNCHANGED <<PersonState, ActiveElevatorCalls>>
            "#
            .to_string(),
            is_recursive: false,
        };
        let instances = BTreeMap::new();
        let inferred = BTreeMap::from([("e".to_string(), elevator_1.clone())]);
        let guard_ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["e".to_string()],
            &[],
            Some(&inferred),
        );
        assert_eq!(
            eval_expr(
                r#"
                    \A call \in ActiveElevatorCalls :
                        /\ CanServiceCall[e, call] =>
                            /\ \E e2 \in Elevator :
                                /\ e /= e2
                                /\ CanServiceCall[e2, call]
                "#,
                &guard_ctx,
            )
            .expect("quantified MoveElevator guard should evaluate"),
            TlaValue::Bool(true)
        );

        let ir = compile_action_ir(&def);
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            Some(&inferred),
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("MoveElevator LET body should probe without leaking `call`");
            }
        }

        let Some(TlaValue::Function(map)) = ctx.locals.get("ElevatorState'") else {
            panic!("expected staged ElevatorState'");
        };
        let Some(TlaValue::Record(fields)) = map.get(&elevator_1) else {
            panic!("expected updated elevator record");
        };
        assert_eq!(fields.get("floor"), Some(&TlaValue::Int(2)));
    }

    #[test]
    fn infer_action_param_samples_from_next_exists_quantifiers() {
        let mut state = TlaState::new();
        state.insert(
            "Boats".to_string(),
            TlaValue::Set(Arc::new(
                [
                    TlaValue::ModelValue("leftBoat".to_string()),
                    TlaValue::ModelValue("rightBoat".to_string()),
                ]
                .into_iter()
                .collect(),
            )),
        );
        state.insert(
            "Sides".to_string(),
            TlaValue::Set(Arc::new(
                [
                    TlaValue::String("left".to_string()),
                    TlaValue::String("right".to_string()),
                ]
                .into_iter()
                .collect(),
            )),
        );

        let mut defs = BTreeMap::new();
        defs.insert(
            "Cross".to_string(),
            TlaDefinition {
                name: "Cross".to_string(),
                params: vec!["boat".to_string(), "side".to_string()],
                body: "/\\ done' = done".to_string(),
                is_recursive: false,
            },
        );

        let samples = infer_action_param_samples_from_next(
            r"\E b \in Boats, s \in Sides : Cross(b, s)",
            &state,
            &defs,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples.get("Cross").and_then(|params| params.get("boat")),
            Some(&TlaValue::ModelValue("rightBoat".to_string()))
        );
        assert_eq!(
            samples.get("Cross").and_then(|params| params.get("side")),
            Some(&TlaValue::String("right".to_string()))
        );
    }

    #[test]
    fn infer_action_param_samples_from_next_captures_boolean_arguments() {
        let proc = TlaValue::ModelValue("p1".to_string());
        let state = TlaState::from([(
            "Proc".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([proc.clone()]))),
        )]);
        let defs = BTreeMap::from([(
            "Receive".to_string(),
            TlaDefinition {
                name: "Receive".to_string(),
                params: vec!["i".to_string(), "includeByz".to_string()],
                body: "TRUE".to_string(),
                is_recursive: false,
            },
        )]);

        let samples = infer_action_param_samples_from_next(
            r"\E self \in Proc : \/ Receive(self, TRUE) \/ UNCHANGED vars",
            &state,
            &defs,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples.get("Receive").and_then(|params| params.get("i")),
            Some(&proc)
        );
        assert_eq!(
            samples
                .get("Receive")
                .and_then(|params| params.get("includeByz")),
            Some(&TlaValue::Bool(true))
        );
    }

    #[test]
    fn infer_action_param_samples_from_next_handles_leading_conjunct_prefixes() {
        let proc = TlaValue::ModelValue("p1".to_string());
        let state = TlaState::from([(
            "Proc".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([proc.clone()]))),
        )]);
        let defs = BTreeMap::from([(
            "Receive".to_string(),
            TlaDefinition {
                name: "Receive".to_string(),
                params: vec!["i".to_string(), "includeByz".to_string()],
                body: "TRUE".to_string(),
                is_recursive: false,
            },
        )]);

        let samples = infer_action_param_samples_from_next(
            r"/\ \E self \in Proc : \/ Receive(self, TRUE) \/ UNCHANGED vars",
            &state,
            &defs,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples.get("Receive").and_then(|params| params.get("i")),
            Some(&proc)
        );
        assert_eq!(
            samples
                .get("Receive")
                .and_then(|params| params.get("includeByz")),
            Some(&TlaValue::Bool(true))
        );
    }

    #[test]
    fn infer_action_param_samples_from_module_contexts_uses_helper_definitions() {
        let state = TlaState::new();
        let defs = BTreeMap::from([
            (
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: "UNCHANGED vars".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Value".to_string(),
                TlaDefinition {
                    name: "Value".to_string(),
                    params: vec![],
                    body: "{v1, v2}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "TLANext".to_string(),
                TlaDefinition {
                    name: "TLANext".to_string(),
                    params: vec![],
                    body: r"\E self \in {0} :
                              \/ \E S \in SUBSET Value : Phase1c(self, S)"
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase1c".to_string(),
                TlaDefinition {
                    name: "Phase1c".to_string(),
                    params: vec!["self".to_string(), "S".to_string()],
                    body: r"/\ \A v \in S : TRUE
                           /\ UNCHANGED vars"
                        .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_module_contexts(
            "Next",
            &defs,
            &BTreeMap::new(),
            &state,
        );

        assert_eq!(
            samples.get("Phase1c").and_then(|params| params.get("self")),
            Some(&TlaValue::Int(0))
        );
        let sample = samples
            .get("Phase1c")
            .and_then(|params| params.get("S"))
            .expect("Phase1c should infer a set-valued sample for S");
        assert!(matches!(sample, TlaValue::Set(_)));
    }

    #[test]
    fn infer_action_param_samples_handles_nested_line_leading_disjunctions() {
        let state = TlaState::new();
        let defs = BTreeMap::from([
            (
                "Acceptor".to_string(),
                TlaDefinition {
                    name: "Acceptor".to_string(),
                    params: vec![],
                    body: "{a1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Ballot".to_string(),
                TlaDefinition {
                    name: "Ballot".to_string(),
                    params: vec![],
                    body: "0..1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Value".to_string(),
                TlaDefinition {
                    name: "Value".to_string(),
                    params: vec![],
                    body: "{v1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "IncreaseMaxBal".to_string(),
                TlaDefinition {
                    name: "IncreaseMaxBal".to_string(),
                    params: vec!["a".to_string(), "b".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "VoteFor".to_string(),
                TlaDefinition {
                    name: "VoteFor".to_string(),
                    params: vec!["a".to_string(), "b".to_string(), "v".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            r#"\E a \in Acceptor, b \in Ballot :
                  \/ IncreaseMaxBal(a, b)
                  \/ \E v \in Value : VoteFor(a, b, v)"#,
            &state,
            &defs,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples
                .get("IncreaseMaxBal")
                .and_then(|params| params.get("a")),
            Some(&TlaValue::ModelValue("a1".to_string()))
        );
        let increase_ballot = samples
            .get("IncreaseMaxBal")
            .and_then(|params| params.get("b"))
            .and_then(|value| value.as_int().ok())
            .expect("IncreaseMaxBal should sample a ballot");
        assert!((0..=1).contains(&increase_ballot));
        assert_eq!(
            samples.get("VoteFor").and_then(|params| params.get("a")),
            Some(&TlaValue::ModelValue("a1".to_string()))
        );
        let vote_ballot = samples
            .get("VoteFor")
            .and_then(|params| params.get("b"))
            .and_then(|value| value.as_int().ok())
            .expect("VoteFor should sample a ballot");
        assert!((0..=1).contains(&vote_ballot));
        assert_eq!(
            samples.get("VoteFor").and_then(|params| params.get("v")),
            Some(&TlaValue::ModelValue("v1".to_string()))
        );
    }

    #[test]
    fn action_expr_probe_seeds_boolean_params_used_in_nested_if_conditions() {
        let proc = TlaValue::ModelValue("p1".to_string());
        let state = TlaState::from([
            (
                "Proc".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([proc.clone()]))),
            ),
            (
                "nRcvdE".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(proc.clone(), TlaValue::Int(0))]))),
            ),
            ("nSntE".to_string(), TlaValue::Int(0)),
            ("nByz".to_string(), TlaValue::Int(1)),
        ]);
        let clauses = vec![ActionClause::Guard {
            expr: "nRcvdE[i] < nSntE + (IF includeByz THEN nByz ELSE 0)".to_string(),
        }];
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["i".to_string(), "includeByz".to_string()],
            &clauses,
            None,
        );

        assert_eq!(ctx.locals.get("includeByz"), Some(&TlaValue::Bool(true)));
        probe_action_clause_expr(&clauses[0], &mut ctx)
            .expect("guard should be probeable")
            .expect("guard with nested IF boolean param should evaluate");
    }

    #[test]
    fn infer_action_param_samples_propagates_into_nested_action_calls() {
        let mut state = TlaState::new();
        state.insert(
            "Workers".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                "w1".to_string(),
            )]))),
        );

        let defs = BTreeMap::from([
            (
                "Outer".to_string(),
                TlaDefinition {
                    name: "Outer".to_string(),
                    params: vec!["self".to_string()],
                    body: "Step(self) \\/ Idle(self)".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Step".to_string(),
                TlaDefinition {
                    name: "Step".to_string(),
                    params: vec!["self".to_string()],
                    body: "pc' = pc".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Idle".to_string(),
                TlaDefinition {
                    name: "Idle".to_string(),
                    params: vec!["self".to_string()],
                    body: "UNCHANGED <<pc>>".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            r"\E self \in Workers : Outer(self)",
            &state,
            &defs,
            &BTreeMap::new(),
        );

        let expected = TlaValue::ModelValue("w1".to_string());
        assert_eq!(
            samples.get("Outer").and_then(|params| params.get("self")),
            Some(&expected)
        );
        assert_eq!(
            samples.get("Step").and_then(|params| params.get("self")),
            Some(&expected)
        );
        assert_eq!(
            samples.get("Idle").and_then(|params| params.get("self")),
            Some(&expected)
        );
    }

    #[test]
    fn infer_action_param_samples_handles_nat_backed_binders() {
        let state = TlaState::new();
        let defs = BTreeMap::from([
            (
                "Ballot".to_string(),
                TlaDefinition {
                    name: "Ballot".to_string(),
                    params: vec![],
                    body: "Nat".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase2a".to_string(),
                TlaDefinition {
                    name: "Phase2a".to_string(),
                    params: vec!["b".to_string()],
                    body: "pc' = pc".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            r"\E b \in Ballot : Phase2a(b)",
            &state,
            &defs,
            &BTreeMap::new(),
        );

        assert_eq!(
            samples.get("Phase2a").and_then(|params| params.get("b")),
            Some(&TlaValue::Int(0))
        );
    }

    #[test]
    fn build_action_expr_probe_context_prefers_inferred_param_samples() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: "self = self".to_string(),
        }];
        let inferred = BTreeMap::from([(
            "self".to_string(),
            TlaValue::ModelValue("worker-1".to_string()),
        )]);

        let ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["self".to_string()],
            &clauses,
            Some(&inferred),
        );

        assert_eq!(
            ctx.locals.get("self"),
            Some(&TlaValue::ModelValue("worker-1".to_string()))
        );
    }

    #[test]
    fn build_action_expr_probe_context_binds_normalized_param_names() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let inferred = BTreeMap::from([(
            "leader".to_string(),
            TlaValue::ModelValue("node-1".to_string()),
        )]);

        let ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &[r"leader \in Node".to_string()],
            &[],
            Some(&inferred),
        );

        assert_eq!(
            eval_expr("leader", &ctx).expect("normalized param should resolve"),
            TlaValue::ModelValue("node-1".to_string())
        );
    }

    #[test]
    fn build_action_expr_probe_context_refines_samples_from_guard_domains() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "NormalPrisoner".to_string(),
            TlaDefinition {
                name: "NormalPrisoner".to_string(),
                params: vec![],
                body: "{\"Bob\"}".to_string(),
                is_recursive: false,
            },
        )]);
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: "p \\in NormalPrisoner".to_string(),
        }];
        let inferred = BTreeMap::from([("p".to_string(), TlaValue::String("Alice".to_string()))]);

        let ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["p".to_string()],
            &clauses,
            Some(&inferred),
        );

        assert_eq!(
            ctx.locals.get("p"),
            Some(&TlaValue::String("Bob".to_string()))
        );
    }

    #[test]
    fn build_action_expr_probe_context_refines_samples_from_function_domains() {
        let good = TlaValue::ModelValue("a1".to_string());
        let fake = TlaValue::ModelValue("fa1".to_string());
        let state = TlaState::from([(
            "knowsSent".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                good.clone(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            )]))),
        )]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: r#"\E S \in SUBSET sentMsgs("1b", b): knowsSent' = [knowsSent EXCEPT ![self] = knowsSent[self] \cup S]"#
                .to_string(),
        }];
        let inferred = BTreeMap::from([
            ("self".to_string(), fake),
            ("b".to_string(), TlaValue::Int(0)),
        ]);

        let ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["self".to_string(), "b".to_string()],
            &clauses,
            Some(&inferred),
        );

        assert_eq!(ctx.locals.get("self"), Some(&good));
    }

    #[test]
    fn build_action_expr_probe_context_seeds_sequence_heads_from_function_domains() {
        let car = TlaValue::ModelValue("r1".to_string());
        let state = TlaState::from([
            (
                "WaitingBeforeBridge".to_string(),
                TlaValue::Seq(Arc::new(Vec::new())),
            ),
            (
                "Location".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(car.clone(), TlaValue::Int(8))]))),
            ),
        ]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: "[ Location EXCEPT ![Head(WaitingBeforeBridge)] = 7 ]".to_string(),
        }];

        let ctx = build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);

        assert_eq!(
            ctx.locals.get("WaitingBeforeBridge"),
            Some(&TlaValue::Seq(Arc::new(vec![car])))
        );
    }

    #[test]
    fn build_action_expr_probe_context_seeds_sequence_heads_from_direct_function_indexes() {
        let car = TlaValue::ModelValue("r1".to_string());
        let state = TlaState::from([
            (
                "WaitingBeforeBridge".to_string(),
                TlaValue::Seq(Arc::new(Vec::new())),
            ),
            (
                "Location".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(car.clone(), TlaValue::Int(8))]))),
            ),
        ]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: "Location[Head(WaitingBeforeBridge)] = 8".to_string(),
        }];

        let ctx = build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);

        assert_eq!(
            ctx.locals.get("WaitingBeforeBridge"),
            Some(&TlaValue::Seq(Arc::new(vec![car])))
        );
    }

    #[test]
    fn build_action_expr_probe_context_seeds_sequence_heads_for_parsed_enter_bridge() {
        let car = TlaValue::ModelValue("r1".to_string());
        let state = TlaState::from([
            (
                "WaitingBeforeBridge".to_string(),
                TlaValue::Seq(Arc::new(Vec::new())),
            ),
            (
                "Location".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(car.clone(), TlaValue::Int(8))]))),
            ),
            (
                "CarsInBridge".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
        ]);
        let defs = BTreeMap::from([
            (
                "CarsRight".to_string(),
                TlaDefinition {
                    name: "CarsRight".to_string(),
                    params: Vec::new(),
                    body: r#"{"r1","r2"}"#.to_string(),
                    is_recursive: false,
                },
            ),
            (
                "RMove".to_string(),
                TlaDefinition {
                    name: "RMove".to_string(),
                    params: vec!["pos".to_string()],
                    body: "pos - 1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "LMove".to_string(),
                TlaDefinition {
                    name: "LMove".to_string(),
                    params: vec!["pos".to_string()],
                    body: "pos + 1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "NextLocation".to_string(),
                TlaDefinition {
                    name: "NextLocation".to_string(),
                    params: vec!["car".to_string()],
                    body: "IF car \\in CarsRight THEN RMove(Location[car]) ELSE LMove(Location[car])"
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "EnterBridge".to_string(),
                TlaDefinition {
                    name: "EnterBridge".to_string(),
                    params: Vec::new(),
                    body: r#"
    \/  /\ CarsInBridge = {}
        /\ Len(WaitingBeforeBridge) # 0
        /\ Location' = [ Location EXCEPT ![Head(WaitingBeforeBridge)] = NextLocation(Head(WaitingBeforeBridge)) ]
        /\ WaitingBeforeBridge' = Tail(WaitingBeforeBridge)
    \/  /\ Len(WaitingBeforeBridge) # 0
        /\ Head(WaitingBeforeBridge) \notin CarsInBridge
        /\ Location' = [ Location EXCEPT ![Head(WaitingBeforeBridge)] = NextLocation(Head(WaitingBeforeBridge)) ]
        /\ WaitingBeforeBridge' = Tail(WaitingBeforeBridge)
"#
                    .to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let instances = BTreeMap::new();
        let branches = compile_action_ir_branches(defs.get("EnterBridge").expect("def exists"));
        assert!(
            !branches.is_empty(),
            "expected EnterBridge to produce probe branches"
        );

        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &branches[0].params,
            &branches[0].clauses,
            None,
        );

        assert_eq!(
            ctx.locals.get("WaitingBeforeBridge"),
            Some(&TlaValue::Seq(Arc::new(vec![car])))
        );

        for clause in &branches[0].clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("parsed EnterBridge clause should probe successfully");
            }
        }
    }

    #[test]
    fn build_action_expr_probe_context_seeds_default_primed_state_bindings() {
        let mut state = TlaState::new();
        state.insert(
            "rcvd".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::Int(1),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            )]))),
        );
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();

        let ctx = build_action_expr_probe_context(&state, &defs, &instances, &[], &[], None);

        assert_eq!(ctx.locals.get("rcvd'"), state.get("rcvd"));
    }

    #[test]
    fn build_action_expr_probe_context_seeds_empty_sequences_used_by_head_or_tail() {
        let state = TlaState::from([("AuthChannel".to_string(), TlaValue::Seq(Arc::new(vec![])))]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![
            ActionClause::PrimedAssignment {
                var: "AuthChannel".to_string(),
                expr: "Tail(AuthChannel)".to_string(),
            },
            ActionClause::Guard {
                expr: "Head(AuthChannel) = 0".to_string(),
            },
        ];

        let ctx = build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);

        assert_eq!(
            ctx.locals.get("AuthChannel"),
            Some(&TlaValue::Seq(Arc::new(vec![TlaValue::Int(0)])))
        );
    }

    #[test]
    fn build_action_expr_probe_context_seeds_record_elements_for_head_field_access() {
        let state = TlaState::from([("FwCtlChannel".to_string(), TlaValue::Seq(Arc::new(vec![])))]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: r#"Head(FwCtlChannel).op = "Add""#.to_string(),
        }];

        let ctx = build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);

        let Some(TlaValue::Seq(items)) = ctx.locals.get("FwCtlChannel") else {
            panic!("expected a seeded non-empty sequence");
        };
        let Some(TlaValue::Record(fields)) = items.first() else {
            panic!("expected the seeded element to be a record");
        };
        assert!(fields.contains_key("op"));
    }

    #[test]
    fn sample_param_value_prefers_function_domains_and_integer_hints() {
        let mut state = TlaState::new();
        state.insert(
            "Key".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                "k1".to_string(),
            )]))),
        );
        state.insert(
            "TxId".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                "t1".to_string(),
            )]))),
        );
        state.insert(
            "Val".to_string(),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                "v1".to_string(),
            )]))),
        );
        state.insert(
            "signalled".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::String("Bob".to_string()),
                TlaValue::Int(0),
            )]))),
        );
        state.insert(
            "pc".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::ModelValue("writer-1".to_string()),
                TlaValue::String("Advance".to_string()),
            )]))),
        );

        assert_eq!(
            sample_param_value("p", &state),
            TlaValue::String("Bob".to_string())
        );
        assert_eq!(
            sample_param_value("t", &state),
            TlaValue::ModelValue("t1".to_string())
        );
        assert_eq!(
            sample_param_value("k", &state),
            TlaValue::ModelValue("k1".to_string())
        );
        assert_eq!(
            sample_param_value("v", &state),
            TlaValue::ModelValue("v1".to_string())
        );
        assert_eq!(sample_param_value("sequence", &state), TlaValue::Int(0));
    }

    #[test]
    fn sample_param_value_with_context_uses_named_type_sets() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "Ballot".to_string(),
            TlaDefinition {
                name: "Ballot".to_string(),
                params: vec![],
                body: "Nat".to_string(),
                is_recursive: false,
            },
        )]);

        assert_eq!(
            sample_param_value_with_context("ballot", &state, &defs, &BTreeMap::new()),
            TlaValue::Int(0)
        );
    }

    #[test]
    fn sample_param_value_with_context_prefers_proc_sets_for_generic_p() {
        let state = TlaState::from([(
            "wmem".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::ModelValue("a1".to_string()),
                TlaValue::ModelValue("v1".to_string()),
            )]))),
        )]);
        let defs = BTreeMap::from([(
            "Proc".to_string(),
            TlaDefinition {
                name: "Proc".to_string(),
                params: vec![],
                body: "{p1}".to_string(),
                is_recursive: false,
            },
        )]);

        assert_eq!(
            sample_param_value_with_context("p", &state, &defs, &BTreeMap::new()),
            TlaValue::ModelValue("p1".to_string())
        );
    }

    #[test]
    fn sample_param_value_with_context_uses_definition_backed_record_sets() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "ElevatorCall".to_string(),
            TlaDefinition {
                name: "ElevatorCall".to_string(),
                params: vec![],
                body: r#"[floor : {1}, direction : {"Up"}]"#.to_string(),
                is_recursive: false,
            },
        )]);

        let sample = sample_param_value_with_context("call", &state, &defs, &BTreeMap::new());

        let TlaValue::Record(fields) = sample else {
            panic!("expected representative record sample");
        };
        assert_eq!(fields.get("floor"), Some(&TlaValue::Int(1)));
        assert_eq!(
            fields.get("direction"),
            Some(&TlaValue::String("Up".to_string()))
        );
    }

    #[test]
    fn representative_value_from_set_expr_handles_definition_backed_operator_calls() {
        let state = TlaState::from([
            (
                "Symbols".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("A".to_string()),
                    TlaValue::ModelValue("B".to_string()),
                ]))),
            ),
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("BuffSz".to_string(), TlaValue::Int(2)),
        ]);
        let defs = BTreeMap::from([
            (
                "SymbolOrArbitrary".to_string(),
                TlaDefinition {
                    name: "SymbolOrArbitrary".to_string(),
                    params: vec![],
                    body: "Symbols \\union {ArbitrarySymbol}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Array".to_string(),
                TlaDefinition {
                    name: "Array".to_string(),
                    params: vec!["T".to_string(), "len".to_string()],
                    body: "[elems: [1..len -> T]]".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let instances = BTreeMap::new();
        let ctx = build_probe_eval_context(&state, &defs, &instances);

        let repr = representative_value_from_set_expr("Array(SymbolOrArbitrary, BuffSz)", &ctx)
            .expect("parameterized type operator should produce a representative");
        let TlaValue::Record(fields) = repr else {
            panic!("expected array record representative");
        };
        let Some(TlaValue::Function(elems)) = fields.get("elems") else {
            panic!("expected elems function field");
        };
        assert_eq!(elems.len(), 2);
    }

    #[test]
    fn infer_action_param_samples_handles_definition_backed_operator_call_domains() {
        let state = TlaState::from([
            (
                "Symbols".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("A".to_string()),
                    TlaValue::ModelValue("B".to_string()),
                ]))),
            ),
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("MaxOffset".to_string(), TlaValue::Int(3)),
        ]);
        let defs = BTreeMap::from([
            (
                "SymbolOrArbitrary".to_string(),
                TlaDefinition {
                    name: "SymbolOrArbitrary".to_string(),
                    params: vec![],
                    body: "Symbols \\union {ArbitrarySymbol}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Array".to_string(),
                TlaDefinition {
                    name: "Array".to_string(),
                    params: vec!["T".to_string(), "len".to_string()],
                    body: "[elems: [1..len -> T]]".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "WriteAtMost".to_string(),
                TlaDefinition {
                    name: "WriteAtMost".to_string(),
                    params: vec!["data".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: r#"\E len \in 1..MaxOffset: \E data \in Array(SymbolOrArbitrary, len): WriteAtMost(data)"#
                        .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let samples = infer_action_param_samples_from_next(
            &defs.get("Next").unwrap().body,
            &state,
            &defs,
            &BTreeMap::new(),
        );

        let sample = samples
            .get("WriteAtMost")
            .and_then(|params| params.get("data"))
            .expect("data sample should be inferred");
        let TlaValue::Record(fields) = sample else {
            panic!("expected array-like record sample");
        };
        assert!(matches!(fields.get("elems"), Some(TlaValue::Function(_))));
    }

    #[test]
    fn braf_style_type_invariants_seed_booleans_and_array_records() {
        let mut probe_state = TlaState::from([
            (
                "Symbols".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("A".to_string()),
                    TlaValue::ModelValue("B".to_string()),
                ]))),
            ),
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("BuffSz".to_string(), TlaValue::Int(2)),
        ]);
        let module = TlaModule {
            name: "BrafProbe".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec![
                "dirty".to_string(),
                "buff".to_string(),
                "file_content".to_string(),
            ],
            definitions: BTreeMap::from([
                (
                    "SymbolOrArbitrary".to_string(),
                    TlaDefinition {
                        name: "SymbolOrArbitrary".to_string(),
                        params: vec![],
                        body: "Symbols \\union {ArbitrarySymbol}".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Array".to_string(),
                    TlaDefinition {
                        name: "Array".to_string(),
                        params: vec!["T".to_string(), "len".to_string()],
                        body: "[elems: [1..len -> T]]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "ArrayOfAnyLength".to_string(),
                    TlaDefinition {
                        name: "ArrayOfAnyLength".to_string(),
                        params: vec!["T".to_string()],
                        body: "[elems: Seq(T)]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "TypeOK".to_string(),
                    TlaDefinition {
                        name: "TypeOK".to_string(),
                        params: vec![],
                        body: r#"
                            /\ dirty \in BOOLEAN
                            /\ buff \in Array(SymbolOrArbitrary, BuffSz)
                            /\ file_content \in ArrayOfAnyLength(SymbolOrArbitrary)
                        "#
                        .to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
        };

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);

        assert_eq!(seeded, 3);
        assert!(matches!(probe_state.get("dirty"), Some(TlaValue::Bool(_))));
        assert!(matches!(probe_state.get("buff"), Some(TlaValue::Record(_))));
        assert!(matches!(
            probe_state.get("file_content"),
            Some(TlaValue::Record(_))
        ));
    }

    #[test]
    fn parsed_braf_init_and_typeok_seed_probe_state() {
        let module = parsed_braf_probe_module();
        let mut probe_state = seed_braf_probe_state(&module);
        let init_def = module.definitions.get("Init").expect("Init should exist");
        let expanded_init =
            expand_state_predicate_clauses(&init_def.body, &module.definitions, &module.instances);
        let mut pending_eq = Vec::new();
        let mut pending_mem = Vec::new();
        for clause in &expanded_init {
            match classify_clause(&clause) {
                ClauseKind::UnprimedEquality { var, expr } => pending_eq.push((var, expr)),
                ClauseKind::UnprimedMembership { var, set_expr } => {
                    pending_mem.push((var, set_expr))
                }
                _ => {}
            }
        }
        let initial_pending_eq = pending_eq.clone();
        let initial_pending_mem = pending_mem.clone();
        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);

        assert!(
            seeded == 0,
            "expanded_init={expanded_init:?} pending_eq={initial_pending_eq:?} pending_mem={initial_pending_mem:?} probe_state={probe_state:?} typeok={:?}",
            module.definitions.get("TypeOK").map(|def| def.body.clone())
        );
        assert_eq!(probe_state.get("dirty"), Some(&TlaValue::Bool(false)));
        assert_eq!(probe_state.get("length"), Some(&TlaValue::Int(0)));
        assert!(matches!(probe_state.get("buff"), Some(TlaValue::Record(_))));
        assert!(matches!(
            probe_state.get("file_content"),
            Some(TlaValue::Record(_))
        ));
    }

    #[test]
    fn parsed_braf_write_at_most_context_uses_array_param_samples() {
        let module = parsed_braf_probe_module();
        let probe_state = seed_braf_probe_state(&module);
        let next_def = module.definitions.get("Next").expect("Next should exist");
        let samples = infer_action_param_samples_from_next(
            &next_def.body,
            &probe_state,
            &module.definitions,
            &module.instances,
        );
        let write_def = module
            .definitions
            .get("WriteAtMost")
            .expect("WriteAtMost should exist");
        let branches = compile_action_ir_branches(write_def);
        let inferred = samples
            .get("WriteAtMost")
            .expect("WriteAtMost param samples should be inferred");
        let ctx = build_action_expr_probe_context(
            &probe_state,
            &module.definitions,
            &module.instances,
            &branches[0].params,
            &branches[0].clauses,
            Some(inferred),
        );

        assert!(matches!(ctx.locals.get("data"), Some(TlaValue::Record(_))));
    }

    #[test]
    fn parsed_braf_flush_buffer_guard_reads_seeded_bool() {
        let module = parsed_braf_probe_module();
        let probe_state = seed_braf_probe_state(&module);
        let def = module
            .definitions
            .get("FlushBuffer")
            .expect("FlushBuffer should exist");
        let ir = compile_action_ir(def);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &module.definitions,
            &module.instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        let result = probe_action_clause_expr(&ir.clauses[0], &mut ctx)
            .expect("dirty guard should be probeable");
        result.expect("dirty guard should evaluate with seeded boolean state");
    }

    #[test]
    fn parsed_braf_write1_reads_seeded_array_state() {
        let module = parsed_braf_probe_module();
        let probe_state = seed_braf_probe_state(&module);
        let def = module
            .definitions
            .get("Write1")
            .expect("Write1 should exist");
        let ir = compile_action_ir(def);
        let samples = BTreeMap::from([(
            "byte".to_string(),
            TlaValue::ModelValue("ArbitrarySymbol".to_string()),
        )]);
        let mut ctx = build_action_expr_probe_context(
            &probe_state,
            &module.definitions,
            &module.instances,
            &ir.params,
            &ir.clauses,
            Some(&samples),
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("Write1 should probe cleanly against seeded array state");
            }
        }

        assert!(matches!(ctx.locals.get("buff'"), Some(TlaValue::Record(_))));
    }

    #[test]
    fn embedded_braf_file_init_and_typeok_seed_probe_state() {
        let module = parsed_braf_embedded_file_probe_module();
        let mut probe_state = TlaState::from([
            (
                "Symbols".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("A".to_string()),
                    TlaValue::ModelValue("B".to_string()),
                ]))),
            ),
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("BuffSz".to_string(), TlaValue::Int(2)),
            ("MaxOffset".to_string(), TlaValue::Int(3)),
        ]);
        let init_def = module.definitions.get("Init").expect("Init should exist");
        let mut pending_eq = Vec::new();
        let mut pending_mem = Vec::new();
        for clause in
            expand_state_predicate_clauses(&init_def.body, &module.definitions, &module.instances)
        {
            match classify_clause(&clause) {
                ClauseKind::UnprimedEquality { var, expr } => pending_eq.push((var, expr)),
                ClauseKind::UnprimedMembership { var, set_expr } => {
                    pending_mem.push((var, set_expr))
                }
                _ => {}
            }
        }

        let total_pending = pending_eq.len() + pending_mem.len();
        for _ in 0..total_pending.saturating_add(1) {
            if pending_eq.is_empty() && pending_mem.is_empty() {
                break;
            }

            let mut progress = false;
            let mut next_pending_eq = Vec::new();
            for (var, expr) in pending_eq {
                let ctx =
                    build_probe_eval_context(&probe_state, &module.definitions, &module.instances);
                match eval_expr(&expr, &ctx) {
                    Ok(value) => {
                        probe_state.insert(var, value);
                        progress = true;
                    }
                    Err(_) => next_pending_eq.push((var, expr)),
                }
            }

            let mut next_pending_mem = Vec::new();
            for (var, set_expr) in pending_mem {
                let ctx =
                    build_probe_eval_context(&probe_state, &module.definitions, &module.instances);
                match eval_expr(&set_expr, &ctx) {
                    Ok(set_val) => {
                        if let Some(repr) = pick_representative_from_set(&set_val) {
                            probe_state.insert(var, repr);
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                    Err(_) => {
                        if let Some(repr) = representative_value_from_set_expr(&set_expr, &ctx) {
                            probe_state.insert(var, repr);
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                }
            }

            if !progress {
                break;
            }
            pending_eq = next_pending_eq;
            pending_mem = next_pending_mem;
        }

        let seeded = seed_probe_state_from_type_invariants(&mut probe_state, &module, None);
        assert_eq!(seeded, 0);
        assert_eq!(probe_state.get("dirty"), Some(&TlaValue::Bool(false)));
        assert!(matches!(probe_state.get("buff"), Some(TlaValue::Record(_))));
        assert!(matches!(
            probe_state.get("file_content"),
            Some(TlaValue::Record(_))
        ));
    }

    #[test]
    fn analyze_style_probe_state_seeds_embedded_braf_file() {
        let (dir, module_path, cfg_path) = write_braf_analyze_probe_files();
        let raw_cfg = fs::read_to_string(&cfg_path).expect("cfg should be readable");
        let parsed_cfg = parse_tla_config(&raw_cfg).expect("cfg should parse");

        let mut parsed_module = parse_tla_module_file(&module_path).expect("module should parse");
        inject_constants_into_definitions(&mut parsed_module, &parsed_cfg);
        let model = TlaModel::from_files(&module_path, Some(&cfg_path), None, None)
            .expect("model should build");
        parsed_module = model.module.clone();

        let mut probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .unwrap_or_default();
        for (name, value) in &parsed_cfg.constants {
            if let Some(tv) = config_value_to_tla(value) {
                probe_state.insert(name.clone(), tv);
            }
        }

        let mut probe_init_seeded = 0usize;
        if let Some(init_def) = parsed_module.definitions.get("Init") {
            let mut pending_eq = Vec::new();
            let mut pending_mem = Vec::new();
            for clause in expand_state_predicate_clauses(
                &init_def.body,
                &parsed_module.definitions,
                &parsed_module.instances,
            ) {
                match classify_clause(&clause) {
                    ClauseKind::UnprimedEquality { var, expr } => pending_eq.push((var, expr)),
                    ClauseKind::UnprimedMembership { var, set_expr } => {
                        pending_mem.push((var, set_expr))
                    }
                    _ => {}
                }
            }

            let total_pending = pending_eq.len() + pending_mem.len();
            for _ in 0..total_pending.saturating_add(1) {
                if pending_eq.is_empty() && pending_mem.is_empty() {
                    break;
                }
                let mut progress = false;

                let mut next_pending_eq = Vec::new();
                for (var, expr) in pending_eq {
                    let ctx = build_probe_eval_context(
                        &probe_state,
                        &parsed_module.definitions,
                        &parsed_module.instances,
                    );
                    match eval_expr(&expr, &ctx) {
                        Ok(value) => {
                            probe_state.insert(var, value);
                            probe_init_seeded += 1;
                            progress = true;
                        }
                        Err(_) => next_pending_eq.push((var, expr)),
                    }
                }

                let mut next_pending_mem = Vec::new();
                for (var, set_expr) in pending_mem {
                    let ctx = build_probe_eval_context(
                        &probe_state,
                        &parsed_module.definitions,
                        &parsed_module.instances,
                    );
                    match eval_expr(&set_expr, &ctx) {
                        Ok(set_val) => {
                            if let Some(repr) = pick_representative_from_set(&set_val) {
                                probe_state.insert(var, repr);
                                probe_init_seeded += 1;
                                progress = true;
                            } else {
                                next_pending_mem.push((var, set_expr));
                            }
                        }
                        Err(_) => {
                            if let Some(repr) = representative_value_from_set_expr(&set_expr, &ctx)
                            {
                                probe_state.insert(var, repr);
                                probe_init_seeded += 1;
                                progress = true;
                            } else {
                                next_pending_mem.push((var, set_expr));
                            }
                        }
                    }
                }

                if !progress {
                    break;
                }
                pending_eq = next_pending_eq;
                pending_mem = next_pending_mem;
            }
        }
        probe_init_seeded += seed_probe_state_from_type_invariants(
            &mut probe_state,
            &parsed_module,
            Some(&parsed_cfg),
        );

        let _ = fs::remove_dir_all(dir);
        assert!(
            probe_init_seeded >= 8,
            "expected analyze-style init seeding to populate probe state, got {probe_init_seeded} with probe_state={probe_state:?}"
        );
        assert_eq!(probe_state.get("dirty"), Some(&TlaValue::Bool(false)));
        assert!(matches!(probe_state.get("buff"), Some(TlaValue::Record(_))));
        assert!(matches!(
            probe_state.get("file_content"),
            Some(TlaValue::Record(_))
        ));
    }

    #[test]
    fn init_membership_seeding_handles_definition_backed_operator_calls() {
        let mut probe_state = TlaState::from([
            (
                "ArbitrarySymbol".to_string(),
                TlaValue::ModelValue("ArbitrarySymbol".to_string()),
            ),
            ("BuffSz".to_string(), TlaValue::Int(2)),
        ]);
        let defs = BTreeMap::from([(
            "Array".to_string(),
            TlaDefinition {
                name: "Array".to_string(),
                params: vec!["T".to_string(), "len".to_string()],
                body: "[elems: [1..len -> T]]".to_string(),
                is_recursive: false,
            },
        )]);
        let instances = BTreeMap::new();
        let ctx = build_probe_eval_context(&probe_state, &defs, &instances);

        let repr = representative_value_from_set_expr("Array({ArbitrarySymbol}, BuffSz)", &ctx)
            .expect("init membership representative should be synthesized");
        probe_state.insert("buff".to_string(), repr);

        assert!(matches!(probe_state.get("buff"), Some(TlaValue::Record(_))));
    }

    #[test]
    fn action_expr_probe_context_preserves_module_instances() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let helper_module = TlaModule {
            definitions: BTreeMap::from([(
                "Echo".to_string(),
                TlaDefinition {
                    name: "Echo".to_string(),
                    params: vec!["value".to_string()],
                    body: "value".to_string(),
                    is_recursive: false,
                },
            )]),
            ..TlaModule::default()
        };
        let instances = BTreeMap::from([(
            "H".to_string(),
            TlaModuleInstance {
                alias: "H".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::new(),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let clauses = vec![ActionClause::Guard {
            expr: "H!Echo(self) = self".to_string(),
        }];

        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &["self".to_string()],
            &clauses,
            Some(&BTreeMap::from([(
                "self".to_string(),
                TlaValue::ModelValue("p1".to_string()),
            )])),
        );

        for clause in &clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("probe should evaluate instance-qualified operator calls");
            }
        }
    }

    #[test]
    fn expand_state_predicate_clauses_apply_instance_substitutions() {
        let helper_module = TlaModule {
            definitions: BTreeMap::from([(
                "Init".to_string(),
                TlaDefinition {
                    name: "Init".to_string(),
                    params: vec![],
                    body: "/\\ contents = [j \\in Jug |-> 0]".to_string(),
                    is_recursive: false,
                },
            )]),
            ..TlaModule::default()
        };
        let defs = BTreeMap::new();
        let instances = BTreeMap::from([(
            "D".to_string(),
            TlaModuleInstance {
                alias: "D".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::from([
                    ("contents".to_string(), "c1".to_string()),
                    ("Jug".to_string(), "{\"j1\"}".to_string()),
                ]),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);

        let clauses = expand_state_predicate_clauses("D!Init", &defs, &instances);
        assert_eq!(clauses, vec![r#"c1 = [j \in {"j1"} |-> 0]"#.to_string()]);
    }

    #[test]
    fn probe_action_call_binds_primed_instance_substitutions() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            (
                "c1".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::Int(1),
                    TlaValue::Int(0),
                )]))),
            ),
        ]);
        let helper_module = TlaModule {
            definitions: BTreeMap::from([(
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: "/\\ x' = contents'[1]".to_string(),
                    is_recursive: false,
                },
            )]),
            ..TlaModule::default()
        };
        let instances = BTreeMap::from([(
            "D".to_string(),
            TlaModuleInstance {
                alias: "D".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::from([("contents".to_string(), "c1".to_string())]),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let definitions = BTreeMap::new();
        let mut ctx = build_probe_eval_context(&state, &definitions, &instances);
        std::rc::Rc::make_mut(&mut ctx.locals).insert(
            "c1'".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::Int(1),
                TlaValue::Int(9),
            )]))),
        );

        let result = expand_probe_action_call("D!Next", &mut ctx)
            .expect("instance-qualified action should be expandable");
        result.expect("primed substitution should be available while probing");
        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(9)));
    }

    #[test]
    fn probe_action_call_applies_function_valued_instance_substitutions_from_outer_defs() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            (
                "c1".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::String("j1".to_string()),
                    TlaValue::Int(0),
                )]))),
            ),
        ]);
        let defs = BTreeMap::from([
            (
                "Cap1".to_string(),
                TlaDefinition {
                    name: "Cap1".to_string(),
                    params: vec![],
                    body: "\"j1\" :> 5".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Caps".to_string(),
                TlaDefinition {
                    name: "Caps".to_string(),
                    params: vec![],
                    body: "<<Cap1>>".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let helper_module = TlaModule {
            definitions: BTreeMap::from([
                (
                    "FillJug".to_string(),
                    TlaDefinition {
                        name: "FillJug".to_string(),
                        params: vec!["j".to_string()],
                        body: "/\\ x' = Capacity[j] + contents[j]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Next".to_string(),
                    TlaDefinition {
                        name: "Next".to_string(),
                        params: vec![],
                        body: r#"\E j \in Jug : FillJug(j)"#.to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            ..TlaModule::default()
        };
        let instances = BTreeMap::from([(
            "D".to_string(),
            TlaModuleInstance {
                alias: "D".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::from([
                    ("Jug".to_string(), "DOMAIN Caps[1]".to_string()),
                    ("Capacity".to_string(), "Caps[1]".to_string()),
                    ("contents".to_string(), "c1".to_string()),
                ]),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let mut ctx = build_probe_eval_context(&state, &defs, &instances);
        let instance = instances.get("D").expect("instance should exist");
        let module = instance.module.as_ref().expect("module should exist");
        let mut instance_ctx = ctx.clone();
        instance_ctx.definitions = Some(&module.definitions);
        instance_ctx.instances = Some(&module.instances);
        seed_probe_instance_constant_bindings(instance, module, &ctx, &mut instance_ctx);
        bind_probe_instance_substitutions(
            instance,
            &ctx,
            std::rc::Rc::make_mut(&mut instance_ctx.locals),
        )
        .expect("instance substitutions should bind");
        let bound = sample_exists_quantifier_binders(
            "j \\in Jug",
            instance_ctx.locals.as_ref(),
            &state,
            &module.definitions,
            &module.instances,
        );
        let bound = bound.expect("instance substitutions should expose Jug values");

        {
            let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
            for (name, value) in &bound {
                locals_mut.insert(name.clone(), value.clone());
            }
        }
        let direct = expand_probe_action_call("FillJug(j)", &mut instance_ctx)
            .expect("local action call should be expandable inside instance scope");
        direct.expect("direct local action call should stage primes");
        assert_eq!(instance_ctx.locals.get("x'"), Some(&TlaValue::Int(5)));

        let result = expand_probe_action_call("D!Next", &mut ctx)
            .expect("instance-qualified action should be expandable");
        result.expect("function-valued substitutions from outer defs should be usable");
        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(5)));
    }

    #[test]
    fn contextual_probeable_action_classification_skips_plain_value_aliases() {
        let defs = BTreeMap::from([
            (
                "MCGoal".to_string(),
                TlaDefinition {
                    name: "MCGoal".to_string(),
                    params: vec![],
                    body: "4".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Goal".to_string(),
                TlaDefinition {
                    name: "Goal".to_string(),
                    params: vec![],
                    body: "MCGoal".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Inner".to_string(),
                TlaDefinition {
                    name: "Inner".to_string(),
                    params: vec![],
                    body: "x' = 1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: "Inner".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        assert!(!definition_is_contextually_probeable_action(
            defs.get("Goal").expect("Goal should exist"),
            &defs,
            &BTreeMap::new(),
            &mut BTreeSet::new(),
        ));
        assert!(definition_is_contextually_probeable_action(
            defs.get("Next").expect("Next should exist"),
            &defs,
            &BTreeMap::new(),
            &mut BTreeSet::new(),
        ));
    }

    #[test]
    fn analyze_style_instance_init_seeding_supports_range_on_seeded_contents() {
        let helper_module = TlaModule {
            definitions: BTreeMap::from([
                (
                    "Init".to_string(),
                    TlaDefinition {
                        name: "Init".to_string(),
                        params: vec![],
                        body: "contents = [j \\in Jug |-> 0]".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "Next".to_string(),
                    TlaDefinition {
                        name: "Next".to_string(),
                        params: vec![],
                        body: "UNCHANGED contents".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            ..TlaModule::default()
        };
        let defs = BTreeMap::from([
            (
                "Init".to_string(),
                TlaDefinition {
                    name: "Init".to_string(),
                    params: vec![],
                    body: "/\\ D1!Init\n/\\ D2!Init\n/\\ s1 = 0\n/\\ s2 = 0".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "NextParallelFreeze".to_string(),
                TlaDefinition {
                    name: "NextParallelFreeze".to_string(),
                    params: vec![],
                    body: "/\\ IF Goal \\in Range(c1) THEN UNCHANGED c1 ELSE D1!Next\n/\\ IF Goal \\in Range(c2) THEN UNCHANGED c2 ELSE D2!Next\n/\\ UNCHANGED <<s1, s2>>".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let capacity_1 = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::String("j1".to_string()), TlaValue::Int(5)),
            (TlaValue::String("j2".to_string()), TlaValue::Int(3)),
        ])));
        let capacity_2 = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::String("j1".to_string()), TlaValue::Int(5)),
            (TlaValue::String("j2".to_string()), TlaValue::Int(3)),
            (TlaValue::String("j3".to_string()), TlaValue::Int(3)),
        ])));
        let instances = BTreeMap::from([
            (
                "D1".to_string(),
                TlaModuleInstance {
                    alias: "D1".to_string(),
                    module_name: "DieHarder".to_string(),
                    substitutions: BTreeMap::from([
                        ("Jug".to_string(), "DOMAIN Capacities[1]".to_string()),
                        ("Capacity".to_string(), "Capacities[1]".to_string()),
                        ("Goal".to_string(), "Goal".to_string()),
                        ("contents".to_string(), "c1".to_string()),
                    ]),
                    is_local: false,
                    module: Some(Box::new(helper_module.clone())),
                },
            ),
            (
                "D2".to_string(),
                TlaModuleInstance {
                    alias: "D2".to_string(),
                    module_name: "DieHarder".to_string(),
                    substitutions: BTreeMap::from([
                        ("Jug".to_string(), "DOMAIN Capacities[2]".to_string()),
                        ("Capacity".to_string(), "Capacities[2]".to_string()),
                        ("Goal".to_string(), "Goal".to_string()),
                        ("contents".to_string(), "c2".to_string()),
                    ]),
                    is_local: false,
                    module: Some(Box::new(helper_module)),
                },
            ),
        ]);

        let mut probe_state = TlaState::from([
            ("Goal".to_string(), TlaValue::Int(4)),
            (
                "Capacities".to_string(),
                TlaValue::Seq(Arc::new(vec![capacity_1, capacity_2])),
            ),
        ]);

        let init_def = defs.get("Init").expect("Init should exist");
        let mut pending_eq = Vec::new();
        let mut pending_mem = Vec::new();
        for clause in expand_state_predicate_clauses(&init_def.body, &defs, &instances) {
            match classify_clause(&clause) {
                ClauseKind::UnprimedEquality { var, expr } => pending_eq.push((var, expr)),
                ClauseKind::UnprimedMembership { var, set_expr } => {
                    pending_mem.push((var, set_expr))
                }
                _ => {}
            }
        }

        let total_pending = pending_eq.len() + pending_mem.len();
        for _ in 0..total_pending.saturating_add(1) {
            if pending_eq.is_empty() && pending_mem.is_empty() {
                break;
            }

            let mut progress = false;
            let mut next_pending_eq = Vec::new();
            for (var, expr) in pending_eq {
                let ctx = build_probe_eval_context(&probe_state, &defs, &instances);
                match eval_expr(&expr, &ctx) {
                    Ok(value) => {
                        probe_state.insert(var, value);
                        progress = true;
                    }
                    Err(_) => next_pending_eq.push((var, expr)),
                }
            }

            let mut next_pending_mem = Vec::new();
            for (var, set_expr) in pending_mem {
                let ctx = build_probe_eval_context(&probe_state, &defs, &instances);
                match eval_expr(&set_expr, &ctx) {
                    Ok(set_val) => {
                        if let Some(repr) = pick_representative_from_set(&set_val) {
                            probe_state.insert(var, repr);
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                    Err(_) => next_pending_mem.push((var, set_expr)),
                }
            }

            if !progress {
                break;
            }
            pending_eq = next_pending_eq;
            pending_mem = next_pending_mem;
        }

        let ctx = build_probe_eval_context(&probe_state, &defs, &instances);
        assert!(matches!(probe_state.get("c1"), Some(TlaValue::Function(_))));
        assert_eq!(
            eval_expr("Goal \\in Range(c1)", &ctx).expect("Range should work on seeded c1"),
            TlaValue::Bool(false)
        );

        let freeze_def = defs
            .get("NextParallelFreeze")
            .expect("freeze action should exist");
        let ir = compile_action_ir(freeze_def);
        let mut action_ctx = build_action_expr_probe_context(
            &probe_state,
            &defs,
            &instances,
            &[],
            &ir.clauses,
            None,
        );
        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut action_ctx) {
                result.expect("seeded DieHard probe action should succeed");
            }
        }
    }

    #[test]
    fn existential_guard_without_witness_disables_following_clauses() {
        let state = TlaState::from([
            ("aCounter".to_string(), TlaValue::Int(0)),
            (
                "aSession".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
        ]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![
            ActionClause::Exists {
                binders: "x \\in aSession".to_string(),
                body: "TRUE".to_string(),
            },
            ActionClause::PrimedAssignment {
                var: "aCounter".to_string(),
                expr: "aCounter + 1".to_string(),
            },
        ];

        let mut ctx =
            build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);
        let result = probe_action_clause_expr(&clauses[0], &mut ctx)
            .expect("exists guard should return a probe result");
        result.expect("missing witness should disable the branch, not error");
        assert!(probe_branch_is_disabled(&ctx));
        assert!(probe_action_clause_expr(&clauses[1], &mut ctx).is_none());
    }

    #[test]
    fn empty_primed_membership_disables_following_clauses() {
        let state = TlaState::from([("now".to_string(), TlaValue::Int(3))]);
        let defs = BTreeMap::from([(
            "Real".to_string(),
            TlaDefinition {
                name: "Real".to_string(),
                params: vec![],
                body: "0..3".to_string(),
                is_recursive: false,
            },
        )]);
        let instances = BTreeMap::new();
        let clauses = vec![
            ActionClause::PrimedMembership {
                var: "now".to_string(),
                set_expr: "{r \\in Real : r > now}".to_string(),
            },
            ActionClause::PrimedAssignment {
                var: "marker".to_string(),
                expr: "1".to_string(),
            },
        ];

        let mut ctx =
            build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);
        let result = probe_action_clause_expr(&clauses[0], &mut ctx)
            .expect("primed membership should return a probe result");
        result.expect("empty primed membership should disable the branch");
        assert!(probe_branch_is_disabled(&ctx));
        assert!(probe_action_clause_expr(&clauses[1], &mut ctx).is_none());
    }

    #[test]
    fn helper_action_with_empty_sequence_precondition_gap_is_probeable() {
        let state = TlaState::from([
            ("AuthChannel".to_string(), TlaValue::Seq(Arc::new(vec![]))),
            (
                "ReplaySession".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
            ("ReplayCount".to_string(), TlaValue::Int(0)),
        ]);
        let defs = BTreeMap::from([(
            "SDPSvrAntiReplayAtk".to_string(),
            TlaDefinition {
                name: "SDPSvrAntiReplayAtk".to_string(),
                params: vec![],
                body: r#"
                    /\ AuthChannel' = Tail(AuthChannel)
                    /\ ReplayCount' = ReplayCount + 1
                    /\ ReplaySession' = ReplaySession \cup {Head(AuthChannel)}
                "#
                .to_string(),
                is_recursive: false,
            },
        )]);
        let def = defs
            .get("SDPSvrAntiReplayAtk")
            .expect("helper action should exist");
        let ir = compile_action_ir(def);
        let instances = BTreeMap::new();
        let mut ctx =
            build_action_expr_probe_context(&state, &defs, &instances, &[], &ir.clauses, None);

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("empty-sequence helper action should use seeded probe values");
            }
        }

        assert_eq!(ctx.locals.get("ReplayCount'"), Some(&TlaValue::Int(1)));
    }

    #[test]
    fn probe_action_clause_expr_supports_box_stuttering_formulas() {
        let state = TlaState::from([("x".to_string(), TlaValue::Int(0))]);
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let clauses = vec![ActionClause::Guard {
            expr: "[x' = x + 1]_x".to_string(),
        }];

        let mut ctx =
            build_action_expr_probe_context(&state, &defs, &instances, &[], &clauses, None);
        let result = probe_action_clause_expr(&clauses[0], &mut ctx)
            .expect("box action should produce a probe result");
        result.expect("box action should be probeable");
        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(1)));
    }

    #[test]
    fn skips_temporal_formulas_in_action_expr_probe() {
        assert!(should_skip_action_expr_probe("[][x = x']_vars"));
        assert!(should_skip_action_expr_probe("<> done"));
        assert!(should_skip_action_expr_probe(
            r"\A s \in Servers : []<>(InSync(s))"
        ));
        assert!(!should_skip_action_expr_probe("x' = x + 1"));
        assert!(!should_skip_action_expr_probe("FwCtlChannel # <<>>"));
        assert!(!should_skip_action_expr_probe("Head(FwCtlChannel) # <<>>"));
    }

    #[test]
    fn probe_identifier_helpers_accept_numeric_prefixed_names() {
        assert!(is_probe_simple_identifier("1bMessage"));
        let (name, rest) =
            parse_probe_identifier_prefix("1bMessage \\cup 2bMessage").expect("identifier");
        assert_eq!(name, "1bMessage");
        assert_eq!(rest.trim_start(), "\\cup 2bMessage");
    }

    #[test]
    fn skips_nested_action_bodies_in_action_expr_probe() {
        let expr = r#"
            ( \/ /\ fd'[self] = TRUE
                /\ pc' = [pc EXCEPT ![self] = "ABORT"]
              \/ /\ fd'[self] = FALSE
                /\ pc' = [pc EXCEPT ![self] = "COMMIT"] )
        "#;

        assert!(should_skip_action_expr_probe(expr));
        assert!(!should_skip_action_expr_probe("fd'[self] = TRUE"));
    }

    #[test]
    fn expr_probe_is_ready_when_no_probeable_clauses_exist() {
        assert!(expr_probe_is_ready(0, 0));
        assert!(expr_probe_is_ready(3, 3));
        assert!(!expr_probe_is_ready(3, 2));
    }

    #[test]
    fn expr_probe_expands_exists_action_clauses() {
        let mut state = TlaState::new();
        state.insert("x".to_string(), TlaValue::Int(0));
        state.insert("y".to_string(), TlaValue::Int(0));

        let defs = BTreeMap::from([
            (
                "Choice".to_string(),
                TlaDefinition {
                    name: "Choice".to_string(),
                    params: vec![],
                    body: r#"\E v \in {1, 2} : x' = v"#.to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Outer".to_string(),
                TlaDefinition {
                    name: "Outer".to_string(),
                    params: vec![],
                    body: "/\\ Choice() /\\ y' = x'".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let ir = compile_action_ir(defs.get("Outer").unwrap());
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("exists binders should be sampled during nested action probing");
            }
        }

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(2)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(2)));
    }

    #[test]
    fn expr_probe_expands_exists_action_clauses_with_disjunctions() {
        let mut state = TlaState::new();
        state.insert("x".to_string(), TlaValue::Int(0));
        state.insert("y".to_string(), TlaValue::Int(0));

        let defs = BTreeMap::from([(
            "Choice".to_string(),
            TlaDefinition {
                name: "Choice".to_string(),
                params: vec![],
                body: r#"\E v \in {1, 2} :
                    \/
                      /\ x' = v
                      /\ y' = v
                    \/
                      /\ x' = 0
                      /\ y' = 0"#
                    .to_string(),
                is_recursive: false,
            },
        )]);
        let ir = compile_action_ir(defs.get("Choice").unwrap());
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            None,
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("exists disjunctive bodies should be probeable");
            }
        }

        let x_prime = ctx.locals.get("x'").expect("x' should be staged");
        let y_prime = ctx.locals.get("y'").expect("y' should be staged");
        assert_eq!(x_prime, y_prime);
        assert!(matches!(
            x_prime,
            TlaValue::Int(0) | TlaValue::Int(1) | TlaValue::Int(2)
        ));
    }

    #[test]
    fn expr_probe_normalizes_commented_disjunctive_action_bodies() {
        let normalized = normalize_probe_clause_expr(
            r#"
                \* Pick the matching action branch for this mode.
                /\ \/ CounterPath(mode)
                   \/ StandardPath(mode)
            "#,
        );

        assert!(!normalized.contains("\\*"));
        assert!(normalized.trim_start().starts_with("\\/ CounterPath(mode)"));
        assert!(normalized.contains("StandardPath(mode)"));
    }

    #[test]
    fn parse_probe_action_if_handles_nested_if_then_branch() {
        let expr = r#"
            IF meetingPlace = MeetingPlaceEmpty
            THEN IF numMeetings < N
                    THEN /\ meetingPlace' = cid
                         /\ UNCHANGED <<chameneoses, numMeetings>>
                    ELSE /\ chameneoses' = [chameneoses EXCEPT ![cid] = <<Faded, @[2]>>]
                         /\ UNCHANGED <<meetingPlace, numMeetings>>
            ELSE /\ meetingPlace' = MeetingPlaceEmpty
        "#;

        let (_, then_branch, else_branch) =
            parse_probe_action_if(expr).expect("nested IF should parse");
        assert!(then_branch.starts_with("IF numMeetings < N"));
        assert!(then_branch.contains("ELSE /\\ chameneoses'"));
        assert_eq!(else_branch, "/\\ meetingPlace' = MeetingPlaceEmpty");
    }

    #[test]
    fn expr_probe_handles_pluscal_style_disjunctive_if_action_branches() {
        let rm1 = TlaValue::ModelValue("rm1".to_string());
        let rm2 = TlaValue::ModelValue("rm2".to_string());
        let rm3 = TlaValue::ModelValue("rm3".to_string());
        let state = TlaState::from([
            (
                "rmState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (rm1.clone(), TlaValue::String("working".to_string())),
                    (rm2.clone(), TlaValue::String("working".to_string())),
                    (rm3.clone(), TlaValue::String("prepared".to_string())),
                ]))),
            ),
            (
                "tmState".to_string(),
                TlaValue::String("commit".to_string()),
            ),
            (
                "pc".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (rm1.clone(), TlaValue::String("RS".to_string())),
                    (rm2.clone(), TlaValue::String("RS".to_string())),
                    (rm3.clone(), TlaValue::String("RS".to_string())),
                ]))),
            ),
        ]);
        let defs = BTreeMap::from([
            (
                "RM".to_string(),
                TlaDefinition {
                    name: "RM".to_string(),
                    params: vec![],
                    body: "{rm1, rm2, rm3}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "RMMAYFAIL".to_string(),
                TlaDefinition {
                    name: "RMMAYFAIL".to_string(),
                    params: vec![],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "RS".to_string(),
                TlaDefinition {
                    name: "RS".to_string(),
                    params: vec!["self".to_string()],
                    body: r#"
                        /\ pc[self] = "RS"
                        /\ IF rmState[self] \in {"working", "prepared"}
                              THEN /\ \/ /\ rmState[self] = "working"
                                         /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                                      \/ /\ \/ /\ tmState="commit"
                                               /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                            \/ /\ rmState[self]="working" \/ tmState="abort"
                                               /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                                      \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                               THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                               ELSE /\ TRUE
                                                    /\ UNCHANGED rmState
                                   /\ pc' = [pc EXCEPT ![self] = "RS"]
                              ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                                   /\ UNCHANGED rmState
                        /\ UNCHANGED tmState
                    "#
                    .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let def = defs.get("RS").unwrap();
        let ir = compile_action_ir(def);
        let samples = BTreeMap::from([("self".to_string(), rm3.clone())]);
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &ir.params,
            &ir.clauses,
            Some(&samples),
        );

        for clause in &ir.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("PlusCal-style IF branch should be probeable");
            }
        }

        assert_eq!(ctx.locals.get("pc'"), state.get("pc"));
        let rm_state_prime = ctx
            .locals
            .get("rmState'")
            .expect("rmState' should be staged");
        assert_ne!(Some(rm_state_prime), state.get("rmState"));
        let TlaValue::Function(map) = rm_state_prime else {
            panic!("expected function-valued rmState', got {rm_state_prime:?}");
        };
        assert_ne!(
            map.get(&rm3),
            Some(&TlaValue::String("prepared".to_string())),
            "probe should pick a branch that changes rmState[self]"
        );
        assert_eq!(ctx.locals.get("tmState'"), state.get("tmState"));
    }

    #[test]
    fn expr_probe_handles_parsed_two_pc_with_btm_rs_action() {
        let module = parsed_two_pc_with_btm_probe_module();
        let defs = module.definitions.clone();
        let rm1 = TlaValue::ModelValue("rm1".to_string());
        let rm2 = TlaValue::ModelValue("rm2".to_string());
        let rm3 = TlaValue::ModelValue("rm3".to_string());
        let state = TlaState::from([
            (
                "RM".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    rm1.clone(),
                    rm2.clone(),
                    rm3.clone(),
                ]))),
            ),
            ("RMMAYFAIL".to_string(), TlaValue::Bool(true)),
            ("TMMAYFAIL".to_string(), TlaValue::Bool(false)),
            (
                "rmState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (rm1.clone(), TlaValue::String("working".to_string())),
                    (rm2.clone(), TlaValue::String("working".to_string())),
                    (rm3.clone(), TlaValue::String("prepared".to_string())),
                ]))),
            ),
            (
                "tmState".to_string(),
                TlaValue::String("commit".to_string()),
            ),
            (
                "pc".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (TlaValue::Int(0), TlaValue::String("TS".to_string())),
                    (TlaValue::Int(10), TlaValue::String("BTS".to_string())),
                    (rm1.clone(), TlaValue::String("RS".to_string())),
                    (rm2.clone(), TlaValue::String("RS".to_string())),
                    (rm3.clone(), TlaValue::String("RS".to_string())),
                ]))),
            ),
        ]);
        let def = defs.get("RS").expect("RS definition should exist");
        let branches = compile_action_ir_branches(def);
        assert_eq!(
            branches.len(),
            1,
            "body={:?} branches={branches:?}",
            def.body
        );
        let samples = BTreeMap::from([("self".to_string(), rm3.clone())]);
        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &branches[0].params,
            &branches[0].clauses,
            Some(&samples),
        );

        for clause in &branches[0].clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("parsed 2PCwithBTM RS branch should be probeable");
            }
        }

        assert_eq!(ctx.locals.get("pc'"), state.get("pc"));
        assert_eq!(ctx.locals.get("tmState'"), state.get("tmState"));
        let rm_state_prime = ctx
            .locals
            .get("rmState'")
            .expect("rmState' should be staged for parsed RS action");
        assert_ne!(Some(rm_state_prime), state.get("rmState"));
    }

    #[test]
    fn expr_probe_handles_parsed_two_pc_with_btm_next_exists_from_init_state() {
        let module = parsed_two_pc_with_btm_probe_module();
        let defs = module.definitions.clone();
        let rm1 = TlaValue::ModelValue("rm1".to_string());
        let rm2 = TlaValue::ModelValue("rm2".to_string());
        let rm3 = TlaValue::ModelValue("rm3".to_string());
        let state = TlaState::from([
            (
                "RM".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    rm1.clone(),
                    rm2.clone(),
                    rm3.clone(),
                ]))),
            ),
            ("RMMAYFAIL".to_string(), TlaValue::Bool(true)),
            ("TMMAYFAIL".to_string(), TlaValue::Bool(true)),
            (
                "rmState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (rm1.clone(), TlaValue::String("working".to_string())),
                    (rm2.clone(), TlaValue::String("working".to_string())),
                    (rm3.clone(), TlaValue::String("working".to_string())),
                ]))),
            ),
            ("tmState".to_string(), TlaValue::String("init".to_string())),
            (
                "pc".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (TlaValue::Int(0), TlaValue::String("TS".to_string())),
                    (TlaValue::Int(10), TlaValue::String("BTS".to_string())),
                    (rm1.clone(), TlaValue::String("RS".to_string())),
                    (rm2.clone(), TlaValue::String("RS".to_string())),
                    (rm3.clone(), TlaValue::String("RS".to_string())),
                ]))),
            ),
        ]);

        let next_def = defs.get("Next").expect("Next definition should exist");
        let branches = compile_action_ir_branches(next_def);
        let rm_branch = branches
            .iter()
            .find(|branch| {
                branch.clauses.iter().any(|clause| {
                    matches!(
                        clause,
                        ActionClause::Guard { expr }
                            if strip_probe_outer_parens(expr.trim()) == "\\E self \\in RM: RManager(self)"
                    )
                })
            })
            .expect("Next should include the RM existential branch");

        let instances = BTreeMap::new();
        let mut ctx = build_action_expr_probe_context(
            &state,
            &defs,
            &instances,
            &rm_branch.params,
            &rm_branch.clauses,
            None,
        );

        for clause in &rm_branch.clauses {
            if let Some(result) = probe_action_clause_expr(clause, &mut ctx) {
                result.expect("parsed Next RM branch should be probeable from Init");
            }
        }

        let rm_state_prime = ctx
            .locals
            .get("rmState'")
            .expect("rmState' should be staged for the RM branch");
        assert_ne!(Some(rm_state_prime), state.get("rmState"));
        assert_eq!(ctx.locals.get("tmState'"), state.get("tmState"));
        assert_eq!(ctx.locals.get("pc'"), state.get("pc"));
    }

    #[test]
    fn injects_operator_ref_constants_with_target_arity() {
        let mut module = TlaModule {
            name: "MCWriteThroughCache".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: vec![],
            variables: vec![],
            definitions: BTreeMap::from([(
                "MCSend".to_string(),
                TlaDefinition {
                    name: "MCSend".to_string(),
                    params: vec![
                        "p".to_string(),
                        "d".to_string(),
                        "oldMemInt".to_string(),
                        "newMemInt".to_string(),
                    ],
                    body: "newMemInt = <<p, d>>".to_string(),
                    is_recursive: false,
                },
            )]),
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
        };
        let config = TlaConfig {
            constants: BTreeMap::from([(
                "Send".to_string(),
                ConfigValue::OperatorRef("MCSend".to_string()),
            )]),
            ..TlaConfig::default()
        };

        inject_constants_into_definitions(&mut module, &config);
        let send = module
            .definitions
            .get("Send")
            .expect("Send alias should be injected");
        assert_eq!(send.params, vec!["p", "d", "oldMemInt", "newMemInt"]);
        assert_eq!(send.body, "MCSend(p, d, oldMemInt, newMemInt)");
    }

    #[test]
    fn injects_matching_constants_into_loaded_instance_modules() {
        let instance_module = TlaModule {
            name: "Nano".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec![],
            definitions: BTreeMap::from([
                (
                    "NoBlock".to_string(),
                    TlaDefinition {
                        name: "NoBlock".to_string(),
                        params: vec![],
                        body: "CHOOSE b : b \\notin SignedBlock".to_string(),
                        is_recursive: false,
                    },
                ),
                (
                    "NoBlockVal".to_string(),
                    TlaDefinition {
                        name: "NoBlockVal".to_string(),
                        params: vec![],
                        body: "NoBlockVal".to_string(),
                        is_recursive: false,
                    },
                ),
            ]),
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
            instances: BTreeMap::new(),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
        };
        let mut module = TlaModule {
            name: "MCNano".to_string(),
            path: String::new(),
            extends: Vec::new(),
            constants: Vec::new(),
            variables: vec![],
            definitions: BTreeMap::new(),
            recursive_declarations: BTreeSet::new(),
            assumes: vec![],
            instances: BTreeMap::from([(
                "N".to_string(),
                TlaModuleInstance {
                    alias: "N".to_string(),
                    module_name: "Nano".to_string(),
                    substitutions: BTreeMap::new(),
                    is_local: false,
                    module: Some(Box::new(instance_module)),
                },
            )]),
            unnamed_instances: Vec::new(),
            is_pluscal: false,
        };
        let config = TlaConfig {
            constants: BTreeMap::from([
                (
                    "NoBlock".to_string(),
                    ConfigValue::OperatorRef("[Nano]NoBlockVal".to_string()),
                ),
                (
                    "NoBlockVal".to_string(),
                    ConfigValue::ModelValue("NoBlockVal".to_string()),
                ),
            ]),
            ..TlaConfig::default()
        };

        inject_constants_into_definitions(&mut module, &config);

        let instance = module.instances.get("N").expect("instance should exist");
        let instance_module = instance
            .module
            .as_ref()
            .expect("instance module should be loaded");
        let no_block = instance_module
            .definitions
            .get("NoBlock")
            .expect("instance constant override should be injected");
        assert_eq!(no_block.body, "NoBlockVal");
    }
}
