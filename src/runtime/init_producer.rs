//! T5.4 streaming Init enumeration producer thread.
//!
//! Spawns a dedicated thread that pulls from
//! `model.initial_states_streaming()` and feeds the standard
//! dedup → fp-store-insert → queue.push_global pipeline. Workers can
//! start consuming the global queue as soon as the first batch lands,
//! instead of waiting for Init enumeration to complete.
//!
//! ### Critical ordering invariants (preserved verbatim from the inline body)
//!
//! - `init_producing` is set to `true` **before** the producer thread
//!   spawns (in the caller). The caller passes the already-set Arc here.
//! - A `Drop` guard inside the producer closure clears `init_producing`
//!   on **any** exit (success, error, panic) so workers cannot hang on
//!   a dead producer.
//! - Workers check `init_producing.load(Acquire)` after `has_pending_work()`
//!   returns false, in this order: queue first, producer second. The
//!   ordering is enforced by the worker loop, not this module.

use crate::model::Model;
use crate::storage::spillable_work_stealing::SpillableWorkStealingQueues;
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use anyhow::Result;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::AtomicRunStats;

/// Spawn the streaming Init-enumeration producer thread.
///
/// Returns `Some(handle)` when the producer is active; returns `None`
/// when resuming from a checkpoint (the queue already holds the seeded
/// initial states, so no streaming is needed).
///
/// On `Some(_)`, the caller must have already created `init_producing`
/// with `AtomicBool::new(true)` and clone it into this call. The Drop
/// guard inside the spawned thread clears the flag on every exit path.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_init_producer<M>(
    resumed_from_checkpoint: bool,
    model: Arc<M>,
    queue: Arc<SpillableWorkStealingQueues<M::State>>,
    fp_store: Arc<UnifiedFingerprintStore>,
    run_stats: Arc<AtomicRunStats>,
    state_map: Option<Arc<DashMap<u64, M::State>>>,
    trace_state_count: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    error_tx: Sender<String>,
    init_producing: Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<Result<u64>>>
where
    M: Model + 'static,
{
    if resumed_from_checkpoint {
        return None;
    }

    let producer_model = model;
    let producer_queue = queue;
    let producer_fp_store = fp_store;
    let producer_run_stats = run_stats;
    let producer_state_map = state_map;
    let producer_trace_count = trace_state_count;
    let producer_stop = stop;
    let producer_error_tx = error_tx;
    let producer_done_flag = init_producing;

    Some(
        std::thread::Builder::new()
            .name("tlapp-init-producer".into())
            .spawn(move || -> Result<u64> {
                // Drop guard ensures `init_producing` is cleared even if
                // the producer panics or returns Err early — workers will
                // not hang waiting for an Init stream that has died.
                struct Guard(Arc<AtomicBool>);
                impl Drop for Guard {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _guard = Guard(producer_done_flag);

                // Batch initial states for fingerprint-store insertion to
                // amortize CAS overhead. The batch size mirrors the worker
                // batch path; small enough that producers feed workers
                // promptly, large enough to keep CAS amortized.
                const INIT_BATCH_SIZE: usize = 256;
                let mut local_dedup: HashSet<u64> =
                    HashSet::with_capacity(INIT_BATCH_SIZE * 4);
                let mut batch_fps: Vec<u64> = Vec::with_capacity(INIT_BATCH_SIZE);
                let mut batch_states: Vec<M::State> = Vec::with_capacity(INIT_BATCH_SIZE);
                let mut seen_flags: Vec<bool> = Vec::with_capacity(INIT_BATCH_SIZE);
                let mut distinct_initial: u64 = 0;

                let mut flush = |batch_states: &mut Vec<M::State>,
                             batch_fps: &mut Vec<u64>,
                             seen_flags: &mut Vec<bool>,
                             distinct_initial: &mut u64|
                 -> Result<()> {
                    if batch_fps.is_empty() {
                        return Ok(());
                    }
                    producer_fp_store.contains_or_insert_batch(batch_fps, seen_flags)?;
                    for (idx, state) in batch_states.drain(..).enumerate() {
                        if seen_flags[idx] {
                            producer_run_stats.duplicates.fetch_add(1, Ordering::Relaxed);
                        } else {
                            if let Some(ref sm) = producer_state_map {
                                let fp = batch_fps[idx];
                                sm.insert(fp, state.clone());
                                producer_trace_count.fetch_add(1, Ordering::Relaxed);
                            }
                            producer_run_stats
                                .states_distinct
                                .fetch_add(1, Ordering::Relaxed);
                            producer_queue.push_global(state);
                            producer_run_stats.enqueued.fetch_add(1, Ordering::Relaxed);
                            *distinct_initial += 1;
                        }
                    }
                    batch_fps.clear();
                    seen_flags.clear();
                    Ok(())
                };

                for state in producer_model.initial_states_streaming() {
                    if producer_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    producer_run_stats
                        .states_generated
                        .fetch_add(1, Ordering::Relaxed);
                    let state = producer_model.canonicalize(state);
                    let fp = producer_model.fingerprint(&state);
                    if !local_dedup.insert(fp) {
                        producer_run_stats.duplicates.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    batch_fps.push(fp);
                    batch_states.push(state);
                    if batch_fps.len() >= INIT_BATCH_SIZE {
                        flush(
                            &mut batch_states,
                            &mut batch_fps,
                            &mut seen_flags,
                            &mut distinct_initial,
                        )?;
                    }
                }
                // Final flush
                flush(
                    &mut batch_states,
                    &mut batch_fps,
                    &mut seen_flags,
                    &mut distinct_initial,
                )?;

                // Print TLC-compatible message
                let now = chrono::Local::now();
                let timestamp = now.format("%Y-%m-%d %H:%M:%S");
                let plural = if distinct_initial == 1 { "state" } else { "states" };
                eprintln!(
                    "Finished computing initial states: {} distinct {} generated at {}.",
                    distinct_initial, plural, timestamp
                );
                let _ = producer_error_tx; // keep alive even if no errors
                Ok(distinct_initial)
            })
            .expect("Failed to spawn Init producer thread"),
    )
}
