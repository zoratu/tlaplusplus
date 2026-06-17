//! CLI argument types and clap derive structs.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use clap::{Args, Subcommand};

pub(crate) fn parse_byte_size(s: &str) -> Result<u64, String> {
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

#[derive(Args, Clone, Debug)]
pub(crate) struct RuntimeArgs {
    #[arg(
        long,
        default_value_t = 0,
        help = "Worker threads (0 = auto, cgroup-aware)"
    )]
    pub(crate) workers: usize,
    #[arg(
        long,
        help = "CPU IDs/ranges, e.g. 2-127 or 2-63,96-127. Intersected with cgroup cpuset."
    )]
    pub(crate) core_ids: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) enforce_cgroups: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) numa_pinning: bool,
    #[arg(
        long,
        help = "NUMA nodes to use (e.g., 0,1 or 0-2). Restricts workers to cores on these nodes only."
    )]
    pub(crate) numa_nodes: Option<String>,
    #[arg(
        long,
        value_parser = parse_byte_size,
        help = "Hard memory ceiling (supports units: 200GB, 10GiB, 512MB, etc.)"
    )]
    pub(crate) memory_max_bytes: Option<u64>,
    #[arg(
        long,
        default_value_t = 256,
        help = "Estimated bytes per in-memory state"
    )]
    pub(crate) estimated_state_bytes: usize,
    #[arg(long, default_value = "./.tlapp")]
    pub(crate) work_dir: std::path::PathBuf,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) clean_work_dir: bool,
    /// Start fresh, ignoring any existing checkpoint (default: auto-resume when S3 is configured)
    #[arg(long, default_value_t = false)]
    pub(crate) fresh: bool,
    /// Checkpoint interval in seconds (0 = disabled, default: 600 with S3, 0 without)
    #[arg(long, default_value_t = 0)]
    pub(crate) checkpoint_interval_secs: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) checkpoint_on_exit: bool,
    #[arg(long, default_value_t = 1)]
    pub(crate) poll_sleep_ms: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) stop_on_violation: bool,
    /// Auto-tune worker count based on CPU utilization (reduces workers when sys% is high)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) auto_tune: bool,
    /// Skip system configuration checks (THP, etc.) at startup
    #[arg(long, default_value_t = false)]
    pub(crate) skip_system_checks: bool,
    /// Enable BFS parent tracking for error trace reconstruction (TLC-style).
    #[arg(long, default_value_t = false)]
    pub(crate) trace_parents: bool,
    /// Maximum number of states to record for parent tracking.
    #[arg(long, default_value_t = 10_000_000)]
    pub(crate) max_trace_states: usize,
    /// Continue after invariant violations (do not stop workers)
    #[arg(long = "continue", default_value_t = false)]
    pub(crate) continue_on_violation: bool,
    /// Maximum number of violations to collect before stopping (default 1)
    #[arg(long, default_value_t = 1)]
    pub(crate) max_violations: usize,
    /// Minimize counter-example traces before reporting (T9).
    /// Default on. Runs a bounded delta-debug pass that searches for
    /// shorter alternative paths reaching the same violating state.
    /// Disable with `--minimize-trace=false`.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) minimize_trace: bool,
    /// Wall-time budget for trace minimization, in seconds.
    /// On expiry the best trace found so far is reported (always still a
    /// valid counter-example; never longer than the original).
    #[arg(long, default_value_t = 30)]
    pub(crate) minimize_trace_budget_secs: u64,
    /// T10.2 — opt-in streaming-SCC liveness oracle (nested DFS).
    ///
    /// When set, after the parallel BFS exploration finishes, the runtime runs
    /// a nested-DFS pass over the same fingerprint adjacency map and
    /// cross-validates against the existing Tarjan-based fairness check.
    /// Reports a diagnostic line if the two diverge. This is the staging
    /// flag for the v1.1.0 in-exploration streaming variant; it does not
    /// (yet) affect memory usage during BFS — see
    /// `docs/T10.2-streaming-scc-design.md` for the full design.
    #[arg(long, default_value_t = false)]
    pub(crate) liveness_streaming: bool,
    /// T10.2 stage 3 — opt-in single-worker DFS exploration with in-line
    /// page-aligned color-map coloring.
    ///
    /// When set AND the model has fairness constraints AND the run is
    /// single-node (no `--cluster-listen`), the runtime replaces the
    /// parallel BFS worker fleet with a single DFS worker
    /// (`runtime/dfs_worker.rs`) that walks the state graph in depth-
    /// first order, coloring each fingerprint via the production
    /// `PageAlignedColorMap` (2 bits per fingerprint, NUMA-shard-placed,
    /// lock-free CAS). On every accepting-state pop a nested-DFS red
    /// probe runs in-band against the same color map.
    ///
    /// The DFS worker still populates the labeled-transitions adjacency
    /// map, so the existing post-processing fairness pipeline produces
    /// the canonical Liveness verdict — gate-7 parity tests assert DFS
    /// and Tarjan paths agree on every fairness fixture.
    ///
    /// Default off and additionally gated on `has_fairness_constraints`:
    /// safety-only specs always use the BFS path regardless of this
    /// flag. Distributed runs (`--cluster-listen`) also fall back to BFS
    /// (the DFS path is single-node by design).
    ///
    /// See `docs/T10.2-phase2-refined.md` §10 for the staged plan; stage
    /// 4 lifts the labeled-transitions population so the fairness verdict
    /// comes purely from in-band cycle detection (the actual memory win).
    #[arg(long, default_value_t = false)]
    pub(crate) liveness_streaming_exploration: bool,

    /// T10.2 stage 5 — DFS pool worker count (active only with
    /// `--liveness-streaming-exploration`). `0` (default) auto-sizes the
    /// pool to the BFS fleet's worker count. `1` keeps the stage-4
    /// single-worker DFS. Anything > 1 enables the multi-worker DFS pool
    /// with cross-partition routing.
    ///
    /// Clamped to the BFS fleet ceiling so it can't oversubscribe the
    /// cgroup CPU quota. Ignored unless
    /// `--liveness-streaming-exploration` is also set and the model has
    /// fairness constraints.
    #[arg(long, default_value_t = 0)]
    pub(crate) dfs_workers: usize,
}

#[derive(Args, Clone, Debug)]
pub(crate) struct StorageArgs {
    /// Number of fingerprint shards (0 = auto-calculate based on CPU/NUMA topology)
    #[arg(long, default_value_t = 0)]
    pub(crate) fp_shards: usize,
    #[arg(long, default_value_t = 100_000_000)]
    pub(crate) fp_expected_items: usize,
    #[arg(long, default_value_t = 0.01)]
    pub(crate) fp_fpr: f64,
    #[arg(long, default_value_t = 1_000_000)]
    pub(crate) fp_hot_entries: usize,
    #[arg(long, default_value_t = 1_073_741_824)]
    pub(crate) fp_cache_bytes: u64,
    #[arg(long, default_value_t = 10_000)]
    pub(crate) fp_flush_every_ms: u64,
    #[arg(long, default_value_t = 512)]
    pub(crate) fp_batch_size: usize,
    #[arg(long, default_value_t = 5_000_000)]
    pub(crate) queue_inmem_limit: usize,
    #[arg(long, default_value_t = 50_000)]
    pub(crate) queue_spill_batch: usize,
    #[arg(long, default_value_t = 128)]
    pub(crate) queue_spill_channel_bound: usize,
    /// Disable disk spilling for work-stealing queues (spilling is enabled by default)
    #[arg(long, default_value_t = false)]
    pub(crate) disable_queue_spilling: bool,
    /// Max items in memory before spilling to disk (when enable_queue_spilling is true)
    #[arg(long, default_value_t = 50_000_000)]
    pub(crate) queue_max_inmem_items: u64,
    /// Max BYTES of in-memory pending state before spilling to disk
    /// (supports units: 80GB, 10GiB, 512MB; 0 = byte-based trigger off).
    /// When set, the queue spills if EITHER the item-count cap above OR this
    /// byte budget is crossed — whichever fires first. This bounds memory on
    /// specs with large states, where a pending count well under
    /// --queue-max-inmem-items can still occupy tens of GB. The byte
    /// footprint is estimated from a sampled serialized-size EWMA seeded by
    /// --estimated-state-bytes.
    #[arg(long, default_value_t = 0, value_parser = parse_byte_size)]
    pub(crate) queue_max_inmem_bytes: u64,
    /// Spill to disk when actual process RSS exceeds this percentage of total
    /// RAM (default 75; 0 = off). Unlike --queue-max-inmem-bytes (a
    /// serialized-size estimate that under-counts real heap), this uses
    /// ground-truth process memory: when RSS crosses the ceiling, workers
    /// spill successors to disk, the loader caps the hot queue low so the bulk
    /// of pending state stays on disk, and freed heap is returned to the OS —
    /// bounding the footprint instead of OOM-killing on specs whose in-memory
    /// queue would outgrow RAM. Specs whose peak RSS stays under the ceiling
    /// never spill, so the default is protective without penalizing runs that
    /// fit in memory. Set 0 to disable, or lower (e.g. 60) for more headroom.
    #[arg(long, default_value_t = 75)]
    pub(crate) queue_memory_ceiling_pct: u8,
    /// Enable in-memory zstd compression for overflow segments (T8).
    /// When on, batches that would otherwise spill to disk are first
    /// compressed and held in a bounded in-memory ring (default 256MB).
    /// This typically defers disk I/O for 1-2GB-equivalent of state.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) queue_compression: bool,
    /// Hard cap on resident compressed-ring bytes (default 256MB).
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    pub(crate) queue_compression_max_bytes: usize,
    /// zstd compression level (1-22; 1 fastest, 22 best ratio).
    #[arg(long, default_value_t = 1)]
    pub(crate) queue_compression_level: i32,
    /// Disable fingerprint persistence (persistence is enabled by default for resume support)
    #[arg(long, default_value_t = false)]
    pub(crate) disable_fp_persistence: bool,
    /// Use bloom filter for fingerprints (bounded memory, ~1% false positive rate)
    /// This drastically reduces memory usage at the cost of possibly re-exploring ~1% of states
    #[arg(long, default_value_t = false)]
    pub(crate) use_bloom_fingerprints: bool,
    /// Enable automatic switching from exact to bloom filter fingerprints
    /// When enabled, starts with exact fingerprints and switches to bloom when:
    /// - Memory usage exceeds --bloom-switch-memory-threshold, OR
    /// - State count exceeds --bloom-switch-threshold
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) bloom_auto_switch: bool,
    /// State count threshold to trigger bloom auto-switch (default: 1 billion)
    #[arg(long, default_value_t = 1_000_000_000)]
    pub(crate) bloom_switch_threshold: u64,
    /// Memory pressure threshold to trigger bloom auto-switch (0.0-1.0, default: 0.85)
    #[arg(long, default_value_t = 0.85)]
    pub(crate) bloom_switch_memory_threshold: f64,
    /// False positive rate for bloom filter after auto-switch (default: 0.001 = 0.1%)
    #[arg(long, default_value_t = 0.001)]
    pub(crate) bloom_switch_fpr: f64,
}

#[derive(Args, Clone, Debug, Default)]
pub(crate) struct S3Args {
    /// S3 bucket for checkpoint persistence (enables S3 sync)
    #[arg(long)]
    pub(crate) s3_bucket: Option<String>,
    /// S3 prefix/path for this run (e.g., "runs/my-run-123")
    #[arg(long, default_value = "")]
    pub(crate) s3_prefix: String,
    /// S3 region (e.g., "us-east-1"). If not specified, uses instance/env region
    #[arg(long)]
    pub(crate) s3_region: Option<String>,
    /// S3 upload interval in seconds (default: 10)
    #[arg(long, default_value_t = 10)]
    pub(crate) s3_upload_interval_secs: u64,
}

#[derive(Args, Clone, Debug)]
pub(crate) struct ClusterArgs {
    /// Listen address for T6 independent-exploration cluster mode
    /// (e.g., 0.0.0.0:7878). When set, enables BFS-based multi-node
    /// model checking with cross-node work stealing. Mutually exclusive
    /// with `--dfs-cluster-listen` — pick one cluster mode per run.
    #[arg(long)]
    pub(crate) cluster_listen: Option<String>,
    /// Comma-separated peer addresses (e.g., 10.0.0.2:7878,10.0.0.3:7878).
    /// Used by both `--cluster-listen` (T6 BFS) and `--dfs-cluster-listen`
    /// (T10.2 DFS pool).
    #[arg(long, value_delimiter = ',')]
    pub(crate) cluster_peers: Vec<String>,
    /// This node's ID in the cluster (must be unique per node). Shared
    /// by both cluster modes.
    #[arg(long, default_value_t = 0)]
    pub(crate) node_id: u32,
    /// Listen address for T10.2 phase 2 stage 5 Layer B DFS-cluster pool
    /// (e.g., 0.0.0.0:7900). When set, enables multi-node fingerprint-
    /// partitioned DFS exploration with cross-partition routing over the
    /// network transport. Mutually exclusive with `--cluster-listen`.
    /// Use the same `--cluster-peers` and `--node-id` as the BFS mode.
    #[arg(long)]
    pub(crate) dfs_cluster_listen: Option<String>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
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
        #[command(flatten)]
        cluster: ClusterArgs,
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
        /// Path to the TLA+ module file. Required unless --fetch-module is used.
        #[arg(long, required_unless_present = "fetch_module")]
        module: Option<std::path::PathBuf>,
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        /// Fetch spec module from S3 URI (e.g., s3://bucket/path/Spec.tla).
        /// Downloaded to a temp directory before parsing. For distributed
        /// model checking where nodes don't share a filesystem.
        #[arg(long)]
        fetch_module: Option<String>,
        /// Fetch config from S3 URI (e.g., s3://bucket/path/Spec.cfg).
        #[arg(long)]
        fetch_config: Option<String>,
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
        /// Enable partial-order reduction (POR) via stubborn sets.
        ///
        /// For specs with independent actions (typical of distributed
        /// protocols), POR explores only one representative interleaving
        /// per equivalence class — often cutting state space 2x to 100x.
        ///
        /// LIMITATION: This implementation preserves only safety properties.
        /// It is rejected automatically when fairness constraints (WF/SF) or
        /// liveness/temporal properties are present in the spec.
        #[arg(long, default_value_t = false)]
        por: bool,
        /// Enable action coverage profiling
        #[arg(long, default_value_t = false)]
        coverage: bool,
        /// Dump state graph to a file after exploration
        #[arg(long)]
        dump: Option<std::path::PathBuf>,
        /// Format for --dump output: "dot" (GraphViz, default) or "raw" (legacy)
        #[arg(long, default_value = "dot")]
        dump_format: String,
        /// Show only changed variables in error traces (like TLC's -difftrace)
        #[arg(long, default_value_t = false)]
        difftrace: bool,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        storage: StorageArgs,
        #[command(flatten)]
        s3: S3Args,
        #[command(flatten)]
        cluster: ClusterArgs,
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
