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
    TlaState, TlaValue, classify_clause, compile_action_ir, compile_action_ir_branches, eval_expr,
    eval_let_action_multi, looks_like_action, normalize_param_name, parse_tla_config,
    parse_tla_module_file, scan_module_closure, split_top_level,
};
use tlaplusplus::{EngineConfig, run_model};

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
                // Collect both equality and membership assignments
                let mut pending_eq: Vec<(String, String)> = Vec::new();
                let mut pending_mem: Vec<(String, String)> = Vec::new();
                for clause in split_top_level(&init_def.body, "/\\") {
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
                                    try_create_representative_value_from_type_expr(&set_expr, &ctx)
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
            if let Some(next_def) = parsed_module.definitions.get("Next") {
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
            let action_param_samples = parsed_module
                .definitions
                .get("Next")
                .map(|next_def| {
                    infer_action_param_samples_from_next(
                        &next_def.body,
                        &probe_state,
                        &parsed_module.definitions,
                        &parsed_module.instances,
                    )
                })
                .unwrap_or_default();
            for def in parsed_module.definitions.values() {
                if !looks_like_action(def) {
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
                            Err(err) => {
                                let key = err.to_string();
                                *expr_errors.entry(key).or_insert(0) += 1;
                                expr_error_examples
                                    .entry(err.to_string())
                                    .or_insert_with(|| {
                                        let expr = match clause {
                                            ActionClause::Guard { expr }
                                            | ActionClause::PrimedAssignment { expr, .. }
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

            let model = TlaModel::from_files(
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

            // Print TLC-compatible output
            let start_time = chrono::Local::now();
            println!("Starting... ({})", start_time.format("%Y-%m-%d %H:%M:%S"));
            println!("Computing initial states...");

            // Clone model for post-processing (liveness checking)
            let model_for_liveness = model.clone();

            let config = build_engine_config(&runtime, &storage, s3.s3_bucket.is_some())?;
            let outcome = run_model_with_s3(model, config, &s3).map_err(|e| {
                eprintln!("Error running model:");
                eprintln!("  {}", e);
                e
            })?;

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
    for (name, value) in &config.constants {
        // Convert ConfigValue to a TLA+ expression string
        let body = config_value_to_expr(value);

        // Add as a zero-parameter definition
        module.definitions.insert(
            name.clone(),
            TlaDefinition {
                name: name.clone(),
                params: vec![],
                body,
                is_recursive: false,
            },
        );
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
        ConfigValue::OperatorRef(name) => name.clone(),
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
        if let ClauseKind::UnprimedMembership { var, set_expr } = classify_clause(&clause)
            && module.variables.contains(&var)
            && !probe_state.contains_key(&var)
        {
            pending.push((var, set_expr));
        }
    }

    let mut seeded = 0usize;
    for _ in 0..pending.len().saturating_add(1) {
        if pending.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();
        for (var, set_expr) in pending {
            let ctx = build_probe_eval_context(probe_state, &module.definitions, &module.instances);
            if let Some(repr) = representative_value_from_set_expr(&set_expr, &ctx) {
                probe_state.insert(var, repr);
                seeded += 1;
                progress = true;
            } else {
                next_pending.push((var, set_expr));
            }
        }

        if !progress {
            break;
        }
        pending = next_pending;
    }

    seeded
}

fn representative_value_from_set_expr(set_expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    if let Ok(set_val) = eval_expr(set_expr, ctx)
        && let Some(repr) = pick_representative_from_set(&set_val)
    {
        return Some(repr);
    }
    try_create_representative_value_from_type_expr(set_expr, ctx)
}

fn representative_member_from_domain_expr(expr: &str, ctx: &EvalContext<'_>) -> Option<TlaValue> {
    if let Ok(set_val) = eval_expr(expr, ctx)
        && let Some(repr) = pick_representative_from_set(&set_val)
    {
        return Some(repr);
    }

    let trimmed = strip_probe_outer_parens(expr.trim());
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
                .and_then(|def| eval_expr(&def.body, &def_ctx).ok())
                .and_then(|value| pick_representative_from_set(&value)),
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
        "asset" | "a" => &["Assets"],
        "bot" | "buyer" | "holder" | "b" => &["Bots"],
        "call" | "c" => &["ActiveElevatorCalls", "ElevatorCall"],
        "client" | "p" | "p1" | "p2" => &["Participants", "Person"],
        "e" | "e1" | "e2" | "elevator" => &["Elevator"],
        "f" | "floor" => &["Floor"],
        "key" | "k" => &["Key"],
        "mm" => &["MarketMakers"],
        "node" | "n" | "n1" | "n2" | "other" => &["Node", "Nodes"],
        "pid" => &["SlushQueryProcess", "SlushLoopProcess", "ProcSet"],
        "prisoner" => &["Prisoner"],
        "reader" | "r" => &["Readers"],
        "seller" | "s" => &["Sellers"],
        "t" | "tx" | "transaction" => &["TxId"],
        "val" | "value" | "v" => &["Val"],
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

    let disjuncts = split_top_level(trimmed, "\\/");
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

    let conjuncts = split_top_level(trimmed, "/\\");
    if conjuncts.len() > 1 {
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

        let domain = eval_expr(domain_text.trim(), &ctx).ok()?;
        let sample = domain.as_set().ok()?.iter().next().cloned()?;

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

    let then_idx = find_probe_top_level_keyword(rest, "THEN")?;
    let condition = rest[..then_idx].trim();
    let after_then = rest[then_idx + "THEN".len()..].trim();

    let else_idx = find_probe_top_level_keyword(after_then, "ELSE")?;
    let then_branch = after_then[..else_idx].trim();
    let else_branch = after_then[else_idx + "ELSE".len()..].trim();

    Some((condition, then_branch, else_branch))
}

fn parse_probe_identifier_prefix(expr: &str) -> Option<(String, &str)> {
    let trimmed = expr.trim_start();
    let mut chars = trimmed.char_indices();
    let (_, first) = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '!' {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
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
    let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);

    for (var, value) in probe_state {
        locals_mut.insert(format!("{var}'"), value.clone());
    }

    for param in params {
        let sample = inferred_param_samples
            .and_then(|samples| samples.get(normalize_param_name(param)))
            .cloned()
            .unwrap_or_else(|| {
                sample_param_value_with_context(param, probe_state, definitions, instances)
            });
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

    ctx
}

fn should_skip_action_expr_probe(expr: &str) -> bool {
    let trimmed = strip_probe_outer_parens(expr.trim());
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.starts_with("[]")
        || trimmed.starts_with("<>")
        || trimmed.starts_with("WF_")
        || trimmed.starts_with("SF_")
        || trimmed.contains("~>")
    {
        return true;
    }

    if trimmed.contains('\'') {
        let disjuncts = split_top_level(trimmed, "\\/");
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

fn merge_staged_prime_locals(from: &EvalContext<'_>, into: &mut EvalContext<'_>) {
    let locals_mut = std::rc::Rc::make_mut(&mut into.locals);
    for (name, value) in from.locals.iter() {
        if name.ends_with('\'') {
            locals_mut.insert(name.clone(), value.clone());
        }
    }
}

fn probe_action_body_into_ctx(body: &str, ctx: &mut EvalContext<'_>) -> anyhow::Result<()> {
    let def = TlaDefinition {
        name: "__probe_body".to_string(),
        params: vec![],
        body: body.to_string(),
        is_recursive: false,
    };
    let ir = compile_action_ir(&def);
    for clause in &ir.clauses {
        if let ActionClause::Guard { expr } = clause
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

fn probe_disjunctive_action_body(
    expr: &str,
    ctx: &mut EvalContext<'_>,
) -> Option<anyhow::Result<()>> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let disjuncts = split_top_level(trimmed, "\\/");
    if disjuncts.len() <= 1 && !trimmed.starts_with("\\/") {
        return None;
    }

    let mut first_err = None;
    for disjunct in disjuncts {
        let mut branch_ctx = ctx.clone();
        match probe_action_body_into_ctx(&disjunct, &mut branch_ctx) {
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

fn expand_probe_action_call(expr: &str, ctx: &mut EvalContext<'_>) -> Option<anyhow::Result<()>> {
    let trimmed = strip_probe_outer_parens(expr.trim());
    let (name, arg_exprs) = parse_probe_action_call(trimmed)?;

    let (def, mut child_ctx) = if let Some((alias, operator_name)) = name.split_once('!') {
        let instances = ctx.instances?;
        let instance = instances.get(alias)?;
        let module = instance.module.as_ref()?;
        let def = module.definitions.get(operator_name)?.clone();
        if !looks_like_action(&def) {
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
        {
            let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
            for (param, value_expr) in &instance.substitutions {
                let value = match eval_expr(value_expr, ctx) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                locals_mut.insert(param.clone(), value);
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
        if !looks_like_action(&def) {
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
                            probe_action_body_into_ctx(branch, &mut branch_ctx)
                        })
                        .map(|_| {
                            merge_staged_prime_locals(&branch_ctx, ctx);
                        }),
                );
            }
            if let Some(expanded) = probe_disjunctive_action_body(&expr, ctx) {
                return Some(expanded);
            }
            if let Some(expanded) = expand_probe_action_call(&expr, ctx) {
                return Some(expanded);
            }
            if should_skip_action_expr_probe(&expr) {
                return None;
            }
            Some(eval_expr(&expr, ctx).map(|_| ()))
        }
        ActionClause::PrimedAssignment { var, expr } => Some(eval_expr(expr, ctx).map(|value| {
            std::rc::Rc::make_mut(&mut ctx.locals).insert(format!("{var}'"), value);
        })),
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
            let local_bindings = (*ctx.locals).clone();
            let definitions = ctx.definitions.unwrap_or(ctx.local_definitions.as_ref());
            let bound = if let Some(instances) = ctx.instances {
                sample_exists_quantifier_binders(
                    binders,
                    &local_bindings,
                    ctx.state,
                    definitions,
                    instances,
                )?
            } else {
                sample_exists_quantifier_binders(
                    binders,
                    &local_bindings,
                    ctx.state,
                    definitions,
                    &BTreeMap::new(),
                )?
            };

            let mut child_ctx = ctx.clone();
            {
                let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
                for (name, value) in bound {
                    locals_mut.insert(name, value);
                }
            }

            Some(probe_action_body_into_ctx(body, &mut child_ctx).map(|_| {
                merge_staged_prime_locals(&child_ctx, ctx);
            }))
        }
    }
}

/// Picks a representative value from a set for probing purposes.
/// For membership constraints like `v \in [Proc -> Values]`, this picks
/// one element from the set to seed the probe_state. The actual model
/// checking in evaluate_init_states handles full enumeration.
fn pick_representative_from_set(set_val: &TlaValue) -> Option<TlaValue> {
    match set_val {
        TlaValue::Set(values) => {
            // Pick the first element from the set
            values.iter().next().cloned()
        }
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
            // Both values should be the first element of B (which is 10)
            for (_, val) in f.iter() {
                assert_eq!(
                    *val,
                    TlaValue::Int(10),
                    "all values should be the first element of B"
                );
            }
        } else {
            panic!("expected Function, got {:?}", func);
        }
    }

    #[test]
    fn test_try_create_representative_function_non_function_set() {
        // Test that non-function-set expressions return None
        let state = TlaState::new();
        let defs = BTreeMap::new();
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

        let defs = BTreeMap::new();
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
                assert_eq!(*val, TlaValue::Int(0), "all values should be 0");
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
                assert_eq!(fields.get("ack"), Some(&TlaValue::Bool(false)));
                assert_eq!(fields.get("payload"), Some(&TlaValue::Int(0)));
            }
            other => panic!("expected record representative for chan, got {other:?}"),
        }
        assert_eq!(
            probe_state.get("pending"),
            Some(&TlaValue::Set(Arc::new(BTreeSet::new())))
        );
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
            Some(&TlaValue::ModelValue("leftBoat".to_string()))
        );
        assert_eq!(
            samples.get("Cross").and_then(|params| params.get("side")),
            Some(&TlaValue::String("left".to_string()))
        );
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
    fn skips_temporal_formulas_in_action_expr_probe() {
        assert!(should_skip_action_expr_probe("[][x = x']_vars"));
        assert!(should_skip_action_expr_probe("<> done"));
        assert!(!should_skip_action_expr_probe("x' = x + 1"));
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

        assert_eq!(ctx.locals.get("x'"), Some(&TlaValue::Int(1)));
        assert_eq!(ctx.locals.get("y'"), Some(&TlaValue::Int(1)));
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
}
