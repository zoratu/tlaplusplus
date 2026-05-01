//! Shared CLI helpers used by multiple subcommand handlers.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use std::sync::Arc;

use crate::distributed::ClusterConfig;
use crate::distributed::handler::StolenState;
use crate::distributed::transport::ClusterTransport;
use crate::distributed::work_stealer::DistributedWorkStealer;
use crate::system::{check_thp_and_warn, parse_cpu_list};
use crate::tla::{
    ConfigValue, TlaConfig, TlaDefinition, TlaModule, TlaState, TlaValue,
    normalize_operator_ref_name,
};
use crate::EngineConfig;

use super::args::{ClusterArgs, RuntimeArgs, StorageArgs};

pub(crate) fn build_engine_config(
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
        queue_compression: storage.queue_compression,
        queue_compression_max_bytes: storage.queue_compression_max_bytes,
        queue_compression_level: storage.queue_compression_level,
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
        distributed_stealer: None, // Set later when --cluster-listen is specified
        stolen_states_rx: None,
        donate_states_tx: None,
        donate_states_rx: None,
        stolen_states_tx: None,
        // T10.2 — opt-in streaming-SCC liveness oracle. When enabled, the
        // post-exploration phase runs nested-DFS over the same fingerprint
        // adjacency and cross-validates against Tarjan-based fairness.
        // Defaults to off (set via --liveness-streaming).
        liveness_streaming: runtime.liveness_streaming,
    })
}

pub(crate) fn print_stats(model_name: &str, stats: &crate::RunStats) {
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
    // T11.4 — non-zero indicates the spill pipeline dropped N states on
    // the floor (permanent disk failure, queue_spill_fail=return failpoint,
    // etc.). Result is unsound under that condition.
    println!(
        "queue.spill_lost_permanently={}",
        stats.queue.spill_lost_permanently
    );
    if stats.queue.spill_lost_permanently > 0 {
        eprintln!(
            "WARNING: {} states were silently dropped by the spill pipeline — model-check result is UNSOUND",
            stats.queue.spill_lost_permanently
        );
    }
}

pub(crate) fn run_system_checks(skip: bool) {
    if skip {
        return;
    }
    check_thp_and_warn();
}

/// Feature 7: Evaluate ASSUME/AXIOM statements from the TLA+ module.
/// If any ASSUME evaluates to FALSE, prints an error and exits.

pub(crate) fn model_fingerprint(state: &TlaState) -> u64 {
    use ahash::AHasher;
    use std::hash::{Hash, Hasher};
    let mut h = AHasher::default();
    state.hash(&mut h);
    h.finish()
}

/// Feature 5: Print difftrace - only show variables that changed between steps.
///
/// `relevant_vars`, when `Some`, is the subset of state variables that the
/// invariant references (T9 Phase B). Variables outside this set are
/// printed with a "(noise)" tag so the user can quickly tell which
/// changes actually drive the violation.

pub(crate) fn print_difftrace(trace: &[TlaState]) {
    print_difftrace_with_relevance(trace, None);
}

pub(crate) fn print_difftrace_with_relevance(
    trace: &[TlaState],
    relevant_vars: Option<&std::collections::HashSet<String>>,
) {
    if trace.is_empty() {
        return;
    }
    let is_relevant = |k: &str| -> bool { relevant_vars.is_none_or(|set| set.contains(k)) };
    let tag = |k: &str| -> &'static str {
        if relevant_vars.is_some() && !is_relevant(k) {
            " (noise)"
        } else {
            ""
        }
    };
    println!("  step 0 (initial):");
    for (k, v) in &trace[0] {
        println!("    /\\ {} = {:?}{}", k, v, tag(k.as_ref()));
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
                    println!("    /\\ {} = {:?}{}", k, v, tag(k.as_ref()));
                    any_changed = true;
                }
            }
        }
        // Check for variables that were removed
        for k in prev.keys() {
            if !curr.contains_key(k) {
                println!("    /\\ {} = <removed>{}", k, tag(k.as_ref()));
                any_changed = true;
            }
        }
        if !any_changed {
            println!("    (no changes - stuttering step)");
        }
    }
}

/// Download a file from an S3 URI (s3://bucket/path/to/file) to a local temp directory.
/// Returns the local path to the downloaded file.

pub(crate) fn fetch_s3_file(uri: &str) -> anyhow::Result<std::path::PathBuf> {
    let stripped = uri
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow::anyhow!("S3 URI must start with s3://, got: {}", uri))?;
    let (bucket, key) = stripped
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("S3 URI must have bucket/key format: {}", uri))?;

    let filename = std::path::Path::new(key)
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("S3 key has no filename: {}", key))?;

    let tmp_dir = std::env::temp_dir().join("tlaplusplus-fetch");
    std::fs::create_dir_all(&tmp_dir)?;
    let local_path = tmp_dir.join(filename);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    rt.block_on(async {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .load()
            .await;
        let client = aws_sdk_s3::Client::new(&config);

        let resp = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch s3://{}/{}: {:?}", bucket, key, e))?;

        let bytes = resp
            .body
            .collect()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read S3 response body: {}", e))?;

        std::fs::write(&local_path, bytes.into_bytes())?;
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(local_path)
}

/// Wire up an `EngineConfig` for distributed cluster mode: bind the listener,
/// connect to peers, create the work stealer, and install the steal/donate
/// channels.
///
/// Mutates `engine_config` in place. Leaks the tokio runtime so it stays
/// alive for the lifetime of the process (the transport's async tasks need it).
///
/// No-op (returns Ok) when `cluster.cluster_listen` is None.

pub(crate) fn maybe_setup_cluster(
    cluster: &ClusterArgs,
    engine_config: &mut EngineConfig,
) -> anyhow::Result<()> {
    let Some(ref listen_addr_str) = cluster.cluster_listen else {
        return Ok(());
    };
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
    let transport =
        tokio_handle.block_on(async { ClusterTransport::new(cluster_config.clone()).await })?;

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
                            attempt + 1,
                            e
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

    let stealer = Arc::new(DistributedWorkStealer::new(
        cluster.node_id,
        num_nodes,
        Arc::clone(&transport),
        tokio_handle.clone(),
    ));

    let (stolen_tx, stolen_rx) = crossbeam_channel::bounded::<StolenState>(65_536);
    let (donate_tx, donate_rx) = crossbeam_channel::bounded::<Vec<u8>>(65_536);

    engine_config.distributed_stealer = Some(Arc::clone(&stealer));
    engine_config.stolen_states_rx = Some(stolen_rx);
    engine_config.donate_states_tx = Some(donate_tx);
    engine_config.donate_states_rx = Some(donate_rx);
    engine_config.stolen_states_tx = Some(stolen_tx);

    // Leak the tokio runtime so it stays alive for the duration of the process.
    std::mem::forget(tokio_rt);

    println!(
        "[cluster] distributed mode active: node {}, {} total nodes",
        cluster.node_id, num_nodes
    );
    Ok(())
}


pub(crate) fn format_num(n: u64) -> String {
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

pub(crate) fn inject_constants_into_definitions(module: &mut TlaModule, config: &TlaConfig) {
    inject_constants_into_module_tree(module, config, true);
}

pub(crate) fn inject_constants_into_module_tree(
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
pub(crate) fn config_value_to_expr(value: &ConfigValue) -> String {
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

pub(crate) fn config_value_to_tla(value: &ConfigValue) -> Option<TlaValue> {
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

