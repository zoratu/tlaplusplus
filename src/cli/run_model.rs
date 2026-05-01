//! Model-running helpers (S3 wrapper, ASSUME evaluation, coverage, state-graph dump).
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use crate::models::tla_native::TlaModel;
use crate::run_model;
use crate::tla::{
    EvalContext, TlaState, TlaValue, eval_expr, split_action_body_disjuncts,
};
use crate::EngineConfig;

use super::args::S3Args;
use super::shared::model_fingerprint;

/// Run a model with optional S3 persistence
/// If s3.s3_bucket is Some, enables continuous S3 upload and download on resume
/// When S3 is enabled, defaults checkpoint interval to 10 minutes if not explicitly set
pub(crate) fn run_model_with_s3<M>(
    model: M,
    mut config: EngineConfig,
    s3: &S3Args,
) -> anyhow::Result<crate::RunOutcome<M::State>>
where
    M: crate::Model + Send + Sync + 'static,
    M::State: Clone
        + std::fmt::Debug
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    use crate::storage::s3_persistence::S3Persistence;

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
            Ok(crate::storage::s3_persistence::DownloadResult::Resumed {
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
            Ok(crate::storage::s3_persistence::DownloadResult::NoExistingState) => {
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

            // All `let _ = stderr/stdout.flush()` calls in this signal handler
            // are best-effort: we're racing a 2-minute spot-preemption budget
            // and the worst case (closed/broken stdio) means dropping a banner
            // line, never losing checkpoint data.
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
            crate::chaos::request_emergency_checkpoint();

            // Wait for checkpoint to complete (max 90 seconds for spot, 30 for SIGINT)
            let timeout_secs = if is_spot_preemption { 90 } else { 30 };
            let checkpoint_ok = tokio::task::spawn_blocking(move || {
                crate::chaos::wait_for_emergency_checkpoint(timeout_secs)
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
                let checkpoint = crate::storage::s3_persistence::CheckpointState {
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
            crate::chaos::request_emergency_checkpoint();

            // Wait for checkpoint (max 30 seconds)
            let checkpoint_ok = tokio::task::spawn_blocking(|| {
                crate::chaos::wait_for_emergency_checkpoint(30)
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
        let checkpoint = crate::storage::s3_persistence::CheckpointState {
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

pub(crate) fn evaluate_assumes(model: &TlaModel) -> anyhow::Result<()> {
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
                // TLCGet is TLC-specific runtime introspection — downgrade to warning
                if body.contains("TLCGet") {
                    eprintln!(
                        "Warning: ASSUME '{}' failed (TLCGet is TLC-specific, ignoring)",
                        body
                    );
                } else {
                    return Err(anyhow::anyhow!("ASSUME failed: {}", body));
                }
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

pub(crate) fn collect_coverage(model: &TlaModel) -> crate::CoverageStats {
    use crate::coverage::{ActionCoverageEntry, CoverageStats};
    use crate::model::Model;

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
                if name.chars().next().map_or(false, |c| c.is_alphabetic()) && !name.contains('(') {
                    return name.to_string();
                }
            }
            format!("disjunct_{}", i)
        })
        .collect();

    let mut stats = CoverageStats::default();
    for name in &action_names {
        stats
            .actions
            .insert(name.clone(), ActionCoverageEntry::default());
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
                use ahash::AHasher;
                use std::hash::{Hash, Hasher};
                let mut h = AHasher::default();
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

            // Try each disjunct individually, timing each evaluation
            for (idx, disjunct) in disjuncts.iter().enumerate() {
                let eval_start = std::time::Instant::now();
                let result = crate::tla::evaluate_next_states_with_instances(
                    disjunct,
                    &model.module.definitions,
                    instances,
                    state,
                );
                let elapsed_nanos = eval_start.elapsed().as_nanos() as u64;
                match result {
                    Ok(successors) if !successors.is_empty() => {
                        let name = &action_names[idx];
                        if let Some(entry) = stats.actions.get_mut(name) {
                            entry.fires += 1;
                            entry.states_generated += successors.len() as u64;
                            entry.elapsed_nanos += elapsed_nanos;
                        }
                        next_frontier.extend(successors);
                    }
                    _ => {
                        // Still track time for non-firing evaluations
                        let name = &action_names[idx];
                        if let Some(entry) = stats.actions.get_mut(name) {
                            entry.elapsed_nanos += elapsed_nanos;
                        }
                    }
                }
            }
        }
        frontier = std::mem::take(&mut next_frontier);
    }

    stats
}

/// Feature 4: Dump state graph to a file.
///
/// Supports two formats controlled by `format`:
/// - `"dot"` (default): GraphViz DOT format with styled initial and violating states
/// - `"raw"`: Legacy format with `STATE hash {state}` lines and `hash -> hash` edges

pub(crate) fn dump_state_graph(model: &TlaModel, path: &std::path::Path, format: &str) -> anyhow::Result<()> {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::io::Write;
    use crate::model::Model;

    eprintln!(
        "Dumping state graph to {} (format={})...",
        path.display(),
        format
    );

    let mut state_hashes: HashMap<u64, TlaState> = HashMap::new();
    let mut transitions: Vec<(u64, u64)> = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<TlaState> = VecDeque::new();
    let mut initial_fps: HashSet<u64> = HashSet::new();

    for init in model.initial_states() {
        let fp = model_fingerprint(&init);
        initial_fps.insert(fp);
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

    // Check which states violate invariants
    let mut violating_fps: HashSet<u64> = HashSet::new();
    for (fp, state) in &state_hashes {
        if model.check_invariants(state).is_err() {
            violating_fps.insert(*fp);
        }
    }

    let mut file = std::fs::File::create(path)?;

    match format {
        "raw" => {
            // Legacy format
            for (hash, state) in &state_hashes {
                writeln!(file, "STATE {:#018x} {:?}", hash, state)?;
            }
            writeln!(file)?;
            for (from, to) in &transitions {
                writeln!(file, "{:#018x} -> {:#018x}", from, to)?;
            }
        }
        _ => {
            // DOT format (default)
            writeln!(file, "digraph StateGraph {{")?;
            writeln!(file, "  node [shape=box, fontsize=10];")?;
            writeln!(file)?;

            // Write node definitions with labels
            for (hash, state) in &state_hashes {
                // Build label: var=value lines
                let label: String = state
                    .iter()
                    .map(|(var, val)| format!("{}={:?}", var, val))
                    .collect::<Vec<_>>()
                    .join("\\n");

                // Determine style based on state role
                let is_initial = initial_fps.contains(hash);
                let is_violating = violating_fps.contains(hash);

                let style = match (is_initial, is_violating) {
                    (true, true) => " style=filled, fillcolor=red",
                    (true, false) => " style=filled, fillcolor=lightblue",
                    (false, true) => " style=filled, fillcolor=red",
                    (false, false) => "",
                };

                writeln!(file, "  \"{:#018x}\" [label=\"{}\"{style}];", hash, label,)?;
            }

            writeln!(file)?;

            // Write edges
            for (from, to) in &transitions {
                writeln!(file, "  \"{:#018x}\" -> \"{:#018x}\";", from, to)?;
            }

            writeln!(file, "}}")?;
        }
    }

    eprintln!(
        "Dumped {} states, {} transitions to {}",
        state_hashes.len(),
        transitions.len(),
        path.display()
    );
    Ok(())
}
