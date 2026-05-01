//! Checkpoint manifest schema and persistence helpers.
//!
//! On-disk schema (`CheckpointManifest`), atomic-rename writer with
//! failpoints for chaos testing, validated rolling-window writer, and
//! the dead `CheckpointContext` / `checkpoint_once` helpers preserved
//! here for history. Follow-up: delete the dead helpers in a separate
//! commit after the runtime refactor lands.

use crate::storage::fingerprint_store::{
    FingerprintStats as OldFingerprintStats, FingerprintStore,
};
use crate::storage::queue::{DiskBackedQueue, QueueStats};
use crate::system::WorkerPlan;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::pause::PauseController;
use super::{AtomicRunStats, EngineConfig};

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct CheckpointManifest {
    pub(super) version: u32,
    pub(super) model: String,
    pub(super) created_unix_secs: u64,
    pub(super) duration_millis: u64,
    pub(super) states_generated: u64,
    pub(super) states_processed: u64,
    pub(super) states_distinct: u64,
    pub(super) duplicates: u64,
    pub(super) enqueued: u64,
    pub(super) checkpoints: u64,
    pub(super) configured_workers: usize,
    pub(super) actual_workers: usize,
    pub(super) allowed_cpu_count: usize,
    pub(super) cgroup_cpuset_cores: Option<usize>,
    pub(super) cgroup_quota_cores: Option<usize>,
    pub(super) numa_nodes_used: usize,
    pub(super) effective_memory_max_bytes: Option<u64>,
    pub(super) resumed_from_checkpoint: bool,
    pub(super) queue: QueueStats,
    pub(super) fingerprints: OldFingerprintStats,
}

pub(super) fn write_checkpoint_manifest(
    path: &Path,
    manifest: &CheckpointManifest,
) -> Result<()> {
    // Chaos: fail point for testing checkpoint write failures
    crate::fail_point!("checkpoint_write_fail");

    // Chaos: apply I/O latency if configured
    crate::chaos::apply_io_latency();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed creating checkpoint dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(manifest).context("failed serializing checkpoint")?;

    // Chaos: fail point for disk write
    crate::fail_point!("checkpoint_disk_write_fail");

    std::fs::write(&tmp, bytes)
        .with_context(|| format!("failed writing checkpoint temp file {}", tmp.display()))?;

    // Chaos: fail point for atomic rename
    crate::fail_point!("checkpoint_rename_fail");

    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed atomically moving checkpoint to {}", path.display()))?;
    Ok(())
}

/// Load checkpoint manifest from disk (for resume)
pub(super) fn load_checkpoint_manifest(path: &Path) -> Result<Option<CheckpointManifest>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed reading checkpoint {}", path.display()))?;
    let manifest: CheckpointManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed parsing checkpoint {}", path.display()))?;
    Ok(Some(manifest))
}

/// Maximum number of checkpoint files to retain (rolling window)
pub(super) const MAX_CHECKPOINT_FILES: usize = 5;

/// Write a checkpoint with validation and rolling retention.
///
/// This function:
/// 1. Writes checkpoint to a timestamped file (e.g., checkpoint-1709612345.json)
/// 2. Reads it back and validates the JSON parses correctly and key fields match
/// 3. Only if valid: updates latest.json to point to the new checkpoint
/// 4. Prunes old checkpoints keeping only the last MAX_CHECKPOINT_FILES
///
/// The `latest_path` should be the path to latest.json (e.g., work_dir/checkpoints/latest.json)
pub(super) fn write_validated_rolling_checkpoint(
    latest_path: &Path,
    manifest: &CheckpointManifest,
) -> Result<()> {
    // Get the checkpoints directory
    let checkpoint_dir = latest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("checkpoint path has no parent directory"))?;

    std::fs::create_dir_all(checkpoint_dir).with_context(|| {
        format!(
            "failed creating checkpoint dir {}",
            checkpoint_dir.display()
        )
    })?;

    // Create timestamped checkpoint filename
    let timestamp = manifest.created_unix_secs;
    let timestamped_name = format!("checkpoint-{}.json", timestamp);
    let timestamped_path = checkpoint_dir.join(&timestamped_name);

    // Step 1: Write checkpoint to timestamped file
    write_checkpoint_manifest(&timestamped_path, manifest).with_context(|| {
        format!(
            "failed writing checkpoint to {}",
            timestamped_path.display()
        )
    })?;

    // Step 2: Read back and validate
    let readback_bytes = std::fs::read(&timestamped_path).with_context(|| {
        format!(
            "failed reading back checkpoint from {}",
            timestamped_path.display()
        )
    })?;

    let readback_manifest: CheckpointManifest = serde_json::from_slice(&readback_bytes)
        .with_context(|| {
            format!(
                "failed parsing checkpoint JSON from {}",
                timestamped_path.display()
            )
        })?;

    // Validate key fields match what we wrote
    if readback_manifest.version != manifest.version
        || readback_manifest.model != manifest.model
        || readback_manifest.created_unix_secs != manifest.created_unix_secs
        || readback_manifest.states_generated != manifest.states_generated
        || readback_manifest.states_distinct != manifest.states_distinct
    {
        // Validation failed — best-effort cleanup of the corrupt checkpoint.
        // We're already returning Err to the caller; if the unlink fails (e.g.
        // race with another process), the prune step on the next successful
        // checkpoint will eventually evict it.
        if let Err(e) = std::fs::remove_file(&timestamped_path) {
            eprintln!(
                "warning: failed to remove invalid checkpoint {}: {}",
                timestamped_path.display(),
                e
            );
        }
        return Err(anyhow::anyhow!(
            "checkpoint validation failed: read-back data does not match written data"
        ));
    }

    // Step 3: Update latest.json to be a copy of the validated checkpoint
    // We copy rather than symlink for S3 compatibility
    std::fs::copy(&timestamped_path, latest_path)
        .with_context(|| format!("failed copying checkpoint to {}", latest_path.display()))?;

    // Step 4: Prune old checkpoints (keep last MAX_CHECKPOINT_FILES)
    prune_old_checkpoints(checkpoint_dir, MAX_CHECKPOINT_FILES)?;

    Ok(())
}

/// Prune old checkpoint files, keeping only the most recent `keep_count` files.
/// Only removes files matching the pattern "checkpoint-*.json".
pub(super) fn prune_old_checkpoints(checkpoint_dir: &Path, keep_count: usize) -> Result<()> {
    let mut checkpoint_files: Vec<_> = std::fs::read_dir(checkpoint_dir)
        .with_context(|| format!("failed reading checkpoint dir {}", checkpoint_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            // SAFE: lossy is fine here — checkpoint files are written by us with
            // ASCII-only names ("checkpoint-{timestamp}.json"). A non-UTF-8 sibling
            // file in the dir simply gets U+FFFD-converted and won't match the
            // ASCII prefix below, so it is correctly excluded from pruning.
            let name_str = name.to_string_lossy();
            // Match checkpoint-{timestamp}.json but NOT latest.json
            name_str.starts_with("checkpoint-") && name_str.ends_with(".json")
        })
        .collect();

    // Sort by filename (which contains timestamp, so oldest first)
    checkpoint_files.sort_by_key(|e| e.file_name());

    // Remove oldest files if we have more than keep_count
    if checkpoint_files.len() > keep_count {
        let to_remove = checkpoint_files.len() - keep_count;
        for entry in checkpoint_files.into_iter().take(to_remove) {
            let path = entry.path();
            if let Err(e) = std::fs::remove_file(&path) {
                // Log but don't fail - pruning is best-effort
                eprintln!(
                    "Checkpoint: warning: failed to prune old checkpoint {}: {}",
                    path.display(),
                    e
                );
            } else {
                eprintln!("Checkpoint: pruned old checkpoint {}", path.display());
            }
        }
    }

    Ok(())
}

/// Pre-T11.5 checkpoint context. Superseded by the inline checkpoint
/// thread body in `run_model`. Kept here to preserve history; should be
/// deleted in a follow-up commit after the runtime refactor lands.
#[allow(dead_code)]
pub(super) struct CheckpointContext<'a, T> {
    pub(super) checkpoint_path: &'a Path,
    pub(super) model_name: &'a str,
    pub(super) started_at: Instant,
    pub(super) run_stats: &'a AtomicRunStats,
    pub(super) queue: &'a DiskBackedQueue<T>,
    pub(super) fp_store: &'a FingerprintStore,
    pub(super) pause: &'a PauseController,
    pub(super) active_workers: &'a AtomicUsize,
    pub(super) live_workers: &'a AtomicUsize,
    pub(super) stop: &'a AtomicBool,
    pub(super) worker_plan: &'a WorkerPlan,
    pub(super) config: &'a EngineConfig,
    pub(super) effective_memory_max: Option<u64>,
    pub(super) resumed_from_checkpoint: bool,
}

#[allow(dead_code)]
pub(super) fn checkpoint_once<T>(ctx: &CheckpointContext<T>) -> Result<()>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    ctx.pause.request_pause();
    let quiescence_achieved =
        ctx.pause
            .wait_for_quiescence(ctx.stop, ctx.active_workers, ctx.live_workers);

    if !quiescence_achieved {
        // Quiescence timed out - skip this checkpoint and resume workers
        ctx.pause.resume();
        return Ok(());
    }

    let checkpoint_result = (|| -> Result<()> {
        // Chaos: fail point for queue flush
        crate::fail_point!("checkpoint_queue_flush_fail");
        ctx.queue.checkpoint_flush()?;

        // Chaos: fail point for fingerprint flush
        crate::fail_point!("checkpoint_fp_flush_fail");
        let _ = ctx.fp_store.flush()?;

        let (states_generated, states_processed, states_distinct, duplicates, enqueued, _) =
            ctx.run_stats.snapshot();
        let checkpoints = ctx.run_stats.checkpoints.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let manifest = CheckpointManifest {
            version: 1,
            model: ctx.model_name.to_string(),
            created_unix_secs: now,
            duration_millis: ctx.started_at.elapsed().as_millis() as u64,
            states_generated,
            states_processed,
            states_distinct,
            duplicates,
            enqueued,
            checkpoints,
            configured_workers: ctx.config.workers,
            actual_workers: ctx.worker_plan.worker_count,
            allowed_cpu_count: ctx.worker_plan.allowed_cpus.len(),
            cgroup_cpuset_cores: ctx.worker_plan.cgroup_cpuset_cores,
            cgroup_quota_cores: ctx.worker_plan.cgroup_quota_cores,
            numa_nodes_used: ctx.worker_plan.numa_nodes_used,
            effective_memory_max_bytes: ctx.effective_memory_max,
            resumed_from_checkpoint: ctx.resumed_from_checkpoint,
            queue: ctx.queue.stats(),
            fingerprints: {
                let stats = ctx.fp_store.stats();
                // Convert new stats to old format for checkpoint (drop collisions field)
                OldFingerprintStats {
                    checks: stats.checks,
                    hits: stats.hits,
                    inserts: stats.inserts,
                    batch_calls: stats.batch_calls,
                    batch_items: stats.batch_items,
                }
            },
        };
        write_validated_rolling_checkpoint(ctx.checkpoint_path, &manifest)?;
        ctx.run_stats.checkpoints.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })();

    ctx.pause.resume();
    checkpoint_result
}
