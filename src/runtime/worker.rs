//! Worker thread body for the parallel BFS exploration loop.
//!
//! This module owns the per-worker spawn closure that was previously
//! inlined in `run_model`. It is **pure code motion** from the inline
//! body — every concurrency-sensitive ordering documented in
//! `docs/runtime-refactor-plan.md` ("Concurrency-coupling analysis")
//! is preserved verbatim:
//!
//! - **T5.4 init-producer ordering** — the `init_producing.load(Acquire)`
//!   check sits in the empty-pop branch *after* `has_pending_work()` and
//!   *before* the cluster-idle handshake. Reversing these introduces a
//!   miss-state race: a state pushed by the producer between two checks
//!   would be skipped by a terminating worker.
//! - **T6 cluster idle-flag handshake** — the three transitions of
//!   `stealer.set_locally_idle(...)` (true on empty-pop, false on
//!   successful pop, true on last-worker exit) all live inside this
//!   function unchanged. The 10ms sleep on the empty-pop branch is the
//!   T6 termination-consensus knob.
//! - **T11.5 violation finish ordering** — on safety violation that
//!   triggers `should_stop`, the order is fixed: violation_tx.try_send
//!   → violation_count.fetch_add → stop.store(Release) → queue.finish()
//!   → worker_idle → break. `queue.finish()` is the T11.5 fix; without
//!   it, sibling workers spin on orphan items in the violator's local
//!   deque.
//!
//! The `WorkerLocalState<M>` struct bundles the 27 distinct `Arc<...>`
//! and `Option<Arc<...>>` clones that the inline closure captured by
//! move. Per-worker mutable scratch buffers and the
//! `flush_local_stats` / `process_batch` inner closures stay inside
//! `run_worker` — they close over `&mut` locals (12+ each) that would
//! otherwise need to be threaded through method signatures, and the
//! current closure shape is the proven-correct one.

use crate::distributed::handler::StolenState;
use crate::distributed::work_stealer::DistributedWorkStealer;
use crate::model::{LabeledTransition, Model};
use crate::storage::numa::set_preferred_node;
use crate::storage::queue::serialize_compressed;
use crate::storage::spillable_work_stealing::{
    SpillableWorkStealingQueues, SpillableWorkerState,
};
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use crate::system::pin_current_thread_to_cpu;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::AtomicRunStats;
use super::pause::{PauseController, pause_worker_after_empty_pop_during_checkpoint};
use super::{PropertyType, Violation, reconstruct_trace_limited};
use crate::autotune::WorkerThrottle;

/// Bundled handles to all shared state captured by a worker thread.
///
/// Replaces the 27 `Arc::clone(...)` / `.clone()` lines that previously
/// sat at the top of the worker spawn block in `run_model`. Construction
/// is mechanical (one clone per field); the consumer is a single
/// `move`-into-thread call site that hands the whole struct to
/// `run_worker`.
///
/// Field order mirrors the original `let worker_X = ...` declarations
/// for ease of cross-referencing during review.
pub(super) struct WorkerLocalState<M: Model> {
    pub(super) model: Arc<M>,
    pub(super) fp_store: Arc<UnifiedFingerprintStore>,
    pub(super) queue: Arc<SpillableWorkStealingQueues<M::State>>,
    pub(super) stats: Arc<AtomicRunStats>,
    pub(super) stop: Arc<AtomicBool>,
    /// Mirrors the original `_worker_active` clone — it was prefixed
    /// with `_` because the worker body never reads it. Kept here so
    /// the per-worker `Arc::clone` count is identical to the inline
    /// version (which matters for any external observer counting Arc
    /// strong refs).
    #[allow(dead_code)]
    pub(super) active_workers: Arc<AtomicUsize>,
    pub(super) live_workers: Arc<AtomicUsize>,
    pub(super) pause: Arc<PauseController>,
    pub(super) throttle: Arc<WorkerThrottle>,
    pub(super) violation_tx: Sender<Violation<M::State>>,
    pub(super) violation_count: Arc<AtomicUsize>,
    pub(super) max_violations: usize,
    pub(super) error_tx: Sender<String>,
    pub(super) stop_on_violation: bool,
    pub(super) fp_batch_size: usize,
    pub(super) cpu: Option<usize>,
    pub(super) numa_node: usize,
    pub(super) labeled_transitions:
        Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
    pub(super) parent_map: Option<Arc<DashMap<u64, u64>>>,
    pub(super) state_map: Option<Arc<DashMap<u64, M::State>>>,
    pub(super) trace_count: Arc<AtomicU64>,
    pub(super) max_trace_states: u64,
    pub(super) distributed_stealer: Option<Arc<DistributedWorkStealer>>,
    pub(super) stolen_rx: Option<Receiver<StolenState>>,
    pub(super) donate_tx: Option<Sender<Vec<u8>>>,
    pub(super) init_producing: Arc<AtomicBool>,
}

/// Run a single worker thread to completion.
///
/// `worker_state` is moved by value (matches the original semantics —
/// each worker owns its `SpillableWorkerState`). `ctx` carries the
/// shared-state handles. The function returns `()` because all errors
/// are reported through `ctx.error_tx` and all violations through
/// `ctx.violation_tx`; a panic propagates as a thread panic and is
/// recovered by the worker-join loop in `run_model`.
pub(super) fn run_worker<M: Model>(
    worker_id: usize,
    mut worker_state: SpillableWorkerState<M::State>,
    ctx: WorkerLocalState<M>,
) {
    // Destructure once; from here on, the body reads the same local
    // bindings the inline closure used (worker_X) so the diff is purely
    // the field-access wrapping.
    let WorkerLocalState {
        model: worker_model,
        fp_store: worker_fp_store,
        queue: worker_queue,
        stats: worker_stats,
        stop: worker_stop,
        active_workers: _worker_active,
        live_workers: worker_live,
        pause: worker_pause,
        throttle: worker_throttle,
        violation_tx: worker_violation_tx,
        violation_count: worker_violation_count,
        max_violations: worker_max_violations,
        error_tx: worker_error_tx,
        stop_on_violation: worker_stop_on_violation,
        fp_batch_size: worker_fp_batch_size,
        cpu: worker_cpu,
        numa_node: worker_numa_node,
        labeled_transitions: worker_labeled_transitions,
        parent_map: worker_parent_map,
        state_map: worker_state_map,
        trace_count: worker_trace_count,
        max_trace_states: worker_max_trace_states,
        distributed_stealer: worker_distributed_stealer,
        stolen_rx: worker_stolen_rx,
        donate_tx: worker_donate_tx,
        init_producing: worker_init_producing,
    } = ctx;

    // Pin thread to CPU for cache locality
    if let Some(cpu) = worker_cpu
        && let Err(err) = pin_current_thread_to_cpu(cpu)
    {
        // Unbounded channel; send only fails if receiver dropped, which
        // can't happen here — the receiver is read after all workers join.
        let _ = worker_error_tx.send(format!("cpu pinning failed on core {cpu}: {err}"));
        worker_stop.store(true, Ordering::Release);
    }

    // Set NUMA memory policy — all allocations on this thread will prefer
    // the local node. Best-effort: returns Err on non-Linux or kernels
    // without NUMA support; in that case the worker still runs, just
    // without NUMA-locality (the article's "graceful degradation" path).
    let _ = set_preferred_node(worker_numa_node);

    // Register this worker's thread handle for lock-free checkpoint unparking
    worker_pause.register_worker_thread(std::thread::current());

    let mut successors: Vec<M::State> = Vec::with_capacity(64);
    let mut pending_batch: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
    let mut unique_states: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);
    let mut unique_fps: Vec<u64> = Vec::with_capacity(worker_fp_batch_size);
    let mut batch_seen: Vec<bool> = Vec::with_capacity(worker_fp_batch_size);
    let mut local_fp_dedup: HashSet<u64> = HashSet::with_capacity(worker_fp_batch_size * 2);
    // Pre-allocate per-iteration vecs outside the loop to avoid repeated allocation
    let mut states_with_home_numa: Vec<(M::State, usize)> =
        Vec::with_capacity(worker_fp_batch_size);
    let mut fps_to_check: Vec<u64> = Vec::with_capacity(worker_fp_batch_size);
    let mut states_to_check: Vec<M::State> = Vec::with_capacity(worker_fp_batch_size);

    // Per-worker persistent fingerprint cache to reduce global store CAS contention
    // This catches duplicates locally before hitting shared atomics
    // Size: 64K entries = ~512KB per worker, catches most duplicates locally
    const LOCAL_FP_CACHE_SIZE: usize = 65536;
    let mut local_fp_cache: HashSet<u64> = HashSet::with_capacity(LOCAL_FP_CACHE_SIZE);
    let mut local_fp_cache_hits = 0u64;

    // Per-worker stats counters to reduce atomic contention
    // Flushed periodically instead of every state
    let mut local_states_processed = 0u64;
    let mut local_states_generated = 0u64;
    let mut local_duplicates = 0u64;
    let mut local_states_distinct = 0u64;
    let mut local_enqueued = 0u64;
    // Adaptive flush: every 512 states OR every ~1 second (whichever comes first)
    // This balances atomic contention reduction vs stats freshness
    const STATS_FLUSH_INTERVAL: u64 = 512;
    let mut last_stats_flush = Instant::now();

    let flush_local_stats = |processed: &mut u64,
                             generated: &mut u64,
                             duplicates: &mut u64,
                             distinct: &mut u64,
                             enqueued: &mut u64,
                             stats: &AtomicRunStats| {
        if *processed > 0 {
            stats
                .states_processed
                .fetch_add(*processed, Ordering::Relaxed);
            *processed = 0;
        }
        if *generated > 0 {
            stats
                .states_generated
                .fetch_add(*generated, Ordering::Relaxed);
            *generated = 0;
        }
        if *duplicates > 0 {
            stats.duplicates.fetch_add(*duplicates, Ordering::Relaxed);
            *duplicates = 0;
        }
        if *distinct > 0 {
            stats
                .states_distinct
                .fetch_add(*distinct, Ordering::Relaxed);
            *distinct = 0;
        }
        if *enqueued > 0 {
            stats.enqueued.fetch_add(*enqueued, Ordering::Relaxed);
            *enqueued = 0;
        }
    };

    loop {
        // Chaos: check if this worker should crash
        if crate::chaos::should_crash_worker(worker_state.id()) {
            panic!("chaos: simulated worker {} crash", worker_state.id());
        }

        // Chaos: failpoint for worker panic
        #[cfg(feature = "failpoints")]
        if fail_point_is_set!("worker_panic") {
            panic!("chaos: failpoint worker_panic triggered");
        }

        worker_pause.worker_pause_point(&worker_stop, worker_id);
        if worker_stop.load(Ordering::Acquire) {
            break;
        }

        // Auto-tune throttle: workers over the limit yield briefly
        worker_throttle.worker_throttle_point(worker_id);

        // Work-stealing: try local queue first, then steal from others
        // This has zero contention on the common path
        let state = match worker_queue.pop_for_worker(&mut worker_state) {
            Some(state) => state,
            None => {
                // During checkpoint, pop_for_worker returns None because pause_requested
                // is set. Workers MUST pause for quiescence. We call worker_pause_point
                // directly here instead of relying on  to reach the one at
                // the top of the loop. This eliminates a race window where workers
                // could spin in a tight continue-loop without ever pausing:
                //   continue -> pause_point(requested=false yet) -> pop(None) -> continue ...
                // By pausing directly, workers enter quiescence immediately.
                if pause_worker_after_empty_pop_during_checkpoint(
                    worker_queue.as_ref(),
                    &worker_pause,
                    &worker_stop,
                    worker_id,
                ) {
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                // After checkpoint, the loader thread may still be loading items from disk.
                // Workers should not terminate while there's pending disk work - the loader
                // will push items to the global queue that workers can steal.
                // Distributed mode: drain stolen states from remote nodes
                // before deciding to terminate.
                if let Some(ref stolen_rx) = worker_stolen_rx {
                    let mut got_work = false;
                    for _ in 0..worker_fp_batch_size {
                        match stolen_rx.try_recv() {
                            Ok(entry) => {
                                match crate::storage::queue::deserialize_compressed::<M::State>(
                                    &entry.compressed_state,
                                ) {
                                    Ok(state) => {
                                        let fp = worker_model.fingerprint(&state);
                                        let home_numa = worker_fp_store.home_numa(fp);
                                        worker_queue.push_batch_to_numa(
                                            &mut worker_state,
                                            std::iter::once((state, home_numa)),
                                        );
                                        got_work = true;
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[cluster] failed to deserialize stolen state: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    if got_work {
                        continue;
                    }
                }
                if worker_queue.has_pending_work() {
                    // Give the loader thread time to load more items
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                // T5.4: Init producer may still be streaming new initial
                // states into the queue. Workers must NOT terminate while
                // the producer is alive — even an empty queue is normal
                // here, the producer just hasn't pushed the next batch.
                if worker_init_producing.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                }
                // Before terminating, do a final pause check to close
                // the race window between our has_pending_work() check
                // and the checkpoint thread requesting pause.  If a
                // checkpoint was requested in that window, this will
                // block until the checkpoint completes and then we can
                // recheck for new work that may have been loaded.
                worker_pause.worker_pause_point(&worker_stop, worker_id);
                if worker_stop.load(Ordering::Acquire) {
                    break;
                }
                // Recheck work availability — checkpoint may have
                // loaded items from disk while we were paused.
                // Use has_pending_work() rather than pop_for_worker()
                // to avoid consuming a state that the outer match
                // would not see.
                if worker_queue.has_pending_work() {
                    continue;
                }
                // Distributed mode: don't terminate until all nodes
                // agree they're idle. Mark THIS node idle so peers
                // can advance their termination consensus, then wait
                // briefly for either (a) stolen work to arrive or
                // (b) global termination to be declared.
                //
                // T6 fix: previously `set_locally_idle(true)` was only
                // called from the worker-exit path, which is itself
                // gated on global termination — a deadlock. Setting
                // it here lets the cluster actually reach consensus.
                if let Some(ref stealer) = worker_distributed_stealer {
                    if !stealer.is_globally_terminated() {
                        stealer.set_locally_idle(true);
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                }
                // No work available and no checkpoint pending — safe to terminate
                break;
            }
        };

        // Mark worker as active (cache-line padded, no contention with other workers)
        worker_queue.worker_start(worker_state.id());
        local_states_processed += 1;

        // Distributed mode: a worker that just popped a state is no
        // longer idle. Clear the flag so peers can't prematurely
        // declare global termination while we still have work to do.
        // Also reset the local-work timestamp so the steal trigger
        // doesn't fire while we're actively processing.
        if let Some(ref stealer) = worker_distributed_stealer {
            stealer.set_locally_idle(false);
        }

        // Periodically flush local stats to reduce atomic contention
        // Flush either by count OR by time (every ~1 second) for accurate reporting
        let should_flush = local_states_processed % STATS_FLUSH_INTERVAL == 0
            || last_stats_flush.elapsed() >= Duration::from_secs(1);
        if should_flush {
            flush_local_stats(
                &mut local_states_processed,
                &mut local_states_generated,
                &mut local_duplicates,
                &mut local_states_distinct,
                &mut local_enqueued,
                &worker_stats,
            );
            last_stats_flush = Instant::now();
            // Distributed mode: tell the steal trigger that we're
            // actively processing work, so it doesn't fire a remote
            // steal while we're busy.
            if let Some(ref stealer) = worker_distributed_stealer {
                stealer.note_local_work();
            }
        }

        if let Err(message) = worker_model.check_invariants(&state) {
            // Reconstruct trace to violation
            let trace = if let Some(ref pm) = worker_parent_map {
                // Use BFS parent tracking: walk parent chain back to initial state
                // invariant: state_map is always Some when parent_map is Some (both gated by config.trace_parents)
                let sm = worker_state_map
                    .as_ref()
                    .expect("state_map missing but parent_map present");
                let mut chain = vec![state.clone()];
                let mut fp = worker_model.fingerprint(&state);
                while let Some(parent_fp_entry) = pm.get(&fp) {
                    let parent_fp = *parent_fp_entry;
                    if let Some(parent_state) = sm.get(&parent_fp) {
                        chain.push(parent_state.clone());
                        fp = parent_fp;
                    } else {
                        break;
                    }
                }
                chain.reverse();
                chain
            } else {
                // Fall back to re-exploration from init states
                reconstruct_trace_limited(
                    worker_model.as_ref(),
                    &state,
                    100, // Max depth to search
                )
                .unwrap_or_else(|| vec![state.clone()])
            };

            // Channel is bounded(max_violations + 1); when it overflows
            // we've already captured enough violations to satisfy the
            // user's --max-violations cap, so dropping the surplus here
            // matches the documented "stop after N" semantics. The
            // worker_violation_count atomic still tracks the true count.
            let _ = worker_violation_tx.try_send(Violation {
                message,
                state,
                property_type: PropertyType::Safety,
                trace,
            });
            let prev_count = worker_violation_count.fetch_add(1, Ordering::AcqRel);
            let should_stop =
                worker_stop_on_violation && (prev_count + 1) >= worker_max_violations;
            if should_stop {
                worker_stop.store(true, Ordering::Release);
                // T11.5: tell the queue to stop too, so other workers
                // currently spinning in `pop_slow_path` exit promptly.
                // Without this, orphan items left in *this* worker's
                // local deque (siblings of the violating state, not
                // yet popped) keep `should_terminate` returning false
                // for everyone else, causing a hang.
                worker_queue.finish();
            }
            worker_queue.worker_idle(worker_state.id());
            if should_stop {
                break;
            }
            continue;
        }

        // Distributed mode: skip state if remote bloom says already explored.
        // This is probabilistic dedup — avoids redundant next_states() calls
        // for states that another node has likely already expanded.
        if let Some(ref stealer) = worker_distributed_stealer {
            let fp = worker_model.fingerprint(&state);
            if stealer.should_skip_state(fp) {
                worker_queue.worker_idle(worker_state.id());
                continue;
            }
        }

        // Apply backpressure: if queue is too full, skip successor generation
        // This allows workers to process the backlog without deadlock
        // Testing: Increased to 1B to allow full state space exploration
        const MAX_PENDING_STATES: u64 = 1_000_000_000;
        if worker_queue.should_apply_backpressure(MAX_PENDING_STATES) {
            // Don't generate successors, just mark worker as idle and continue
            // This allows us to process the backlog
            worker_queue.worker_idle(worker_state.id());
            continue;
        }

        successors.clear();

        // Clear any stale reached-assertion record before generating successors,
        // so the drain below attributes an assertion to THIS state's action
        // (a prior iteration's trace reconstruction can also run next_states).
        crate::model::take_pending_assertion_violation();

        // Use labeled transitions if fairness constraints exist
        if let Some(ref transitions_map) = worker_labeled_transitions {
            if let Some(labeled_transitions_vec) = worker_model.next_states_labeled(&state)
            {
                // Collect labeled transitions and extract successor states
                let state_fp = worker_model.fingerprint(&state);

                // Store all transitions from this state
                let mut transitions_from_state =
                    Vec::with_capacity(labeled_transitions_vec.len());
                for labeled_trans in labeled_transitions_vec {
                    successors.push(labeled_trans.to.clone());
                    transitions_from_state.push(labeled_trans);
                }

                // Store transitions in the map (keyed by source state fingerprint)
                if !transitions_from_state.is_empty() {
                    transitions_map
                        .entry(state_fp)
                        .or_insert_with(Vec::new)
                        .extend(transitions_from_state);
                }
            } else {
                // Fallback to unlabeled if model doesn't provide labeled transitions
                worker_model.next_states(&state, &mut successors);
            }
        } else {
            // No fairness constraints - use regular next_states
            worker_model.next_states(&state, &mut successors);
        }

        // A reached Assert(FALSE) during the next-state evaluation above is a
        // safety violation (TLC halts on it). `next_states` returns `()` and
        // cannot surface it, so the evaluator recorded it on the reached-
        // assertion side channel; drain and report it here exactly like an
        // invariant violation (same trace reconstruction + channel + stop).
        if let Some(assert_message) = crate::model::take_pending_assertion_violation() {
            let trace = if let Some(ref pm) = worker_parent_map {
                let sm = worker_state_map
                    .as_ref()
                    .expect("state_map missing but parent_map present");
                let mut chain = vec![state.clone()];
                let mut fp = worker_model.fingerprint(&state);
                while let Some(parent_fp_entry) = pm.get(&fp) {
                    let parent_fp = *parent_fp_entry;
                    if let Some(parent_state) = sm.get(&parent_fp) {
                        chain.push(parent_state.clone());
                        fp = parent_fp;
                    } else {
                        break;
                    }
                }
                chain.reverse();
                chain
            } else {
                reconstruct_trace_limited(worker_model.as_ref(), &state, 100)
                    .unwrap_or_else(|| vec![state.clone()])
            };
            let _ = worker_violation_tx.try_send(Violation {
                message: assert_message,
                state: state.clone(),
                property_type: PropertyType::Safety,
                trace,
            });
            let prev_count = worker_violation_count.fetch_add(1, Ordering::AcqRel);
            let should_stop =
                worker_stop_on_violation && (prev_count + 1) >= worker_max_violations;
            if should_stop {
                worker_stop.store(true, Ordering::Release);
                worker_queue.finish();
            }
            worker_queue.worker_idle(worker_state.id());
            if should_stop {
                break;
            }
            continue;
        }

        #[allow(unused_variables)]
        let successors_before_filter = successors.len();
        local_states_generated += successors.len() as u64;

        // Filter successors by state constraints (prune states that don't satisfy constraints)
        successors.retain(|next_state| {
            worker_model.check_state_constraints(next_state).is_ok()
                && worker_model
                    .check_action_constraints(&state, next_state)
                    .is_ok()
        });

        // Debug: track constraint filtering (only in debug builds)
        #[cfg(debug_assertions)]
        {
            static DEBUG_STATES_SAMPLED: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let sample_count = DEBUG_STATES_SAMPLED.fetch_add(1, Ordering::Relaxed);
            if sample_count < 10 {
                eprintln!(
                    "DEBUG [state {}]: {} successors generated, {} after constraint filter",
                    sample_count,
                    successors_before_filter,
                    successors.len()
                );
            }
        }

        // Compute parent fingerprint once for parent tracking
        let current_state_fp = if worker_parent_map.is_some() {
            worker_model.fingerprint(&state)
        } else {
            0
        };

        let mut process_batch = |pending_batch: &mut Vec<M::State>,
                                 local_duplicates: &mut u64,
                                 local_states_distinct: &mut u64,
                                 local_enqueued: &mut u64,
                                 fp_cache: &mut HashSet<u64>,
                                 fp_cache_hits: &mut u64|
         -> anyhow::Result<()> {
            if pending_batch.is_empty() {
                return Ok(());
            }
            unique_states.clear();
            unique_fps.clear();
            local_fp_dedup.clear();
            states_with_home_numa.clear();
            fps_to_check.clear();
            states_to_check.clear();

            let mut duplicates_in_batch = 0u64;
            for candidate in pending_batch.drain(..) {
                // Canonicalize under symmetry reduction before fingerprinting.
                // Two states that differ only by a permutation of symmetric
                // elements will produce the same canonical form and therefore
                // the same fingerprint, letting the FP store dedup them.
                let candidate = worker_model.canonicalize(candidate);
                let fp = worker_model.fingerprint(&candidate);
                // Dedup within this batch first
                if !local_fp_dedup.insert(fp) {
                    duplicates_in_batch += 1;
                    continue;
                }
                // Check persistent local cache - catches duplicates without CAS
                if fp_cache.contains(&fp) {
                    *fp_cache_hits += 1;
                    duplicates_in_batch += 1;
                    continue;
                }
                // Need to check global store
                fps_to_check.push(fp);
                states_to_check.push(candidate);
            }

            if !fps_to_check.is_empty() {
                // --- Unified path: all fingerprints checked locally ---
                // In distributed mode, each node has its own FP store.
                // Bloom filter dedup is checked earlier (before next_states).
                // Use worker-affinity batch to reduce CAS contention.
                worker_fp_store.contains_or_insert_batch_with_affinity(
                    &fps_to_check,
                    &mut batch_seen,
                    worker_id,
                )?;

                for (idx, next_state) in states_to_check.drain(..).enumerate() {
                    let fp = fps_to_check[idx];
                    if batch_seen[idx] {
                        *local_duplicates += 1;
                        if fp_cache.len() < LOCAL_FP_CACHE_SIZE {
                            fp_cache.insert(fp);
                        }
                    } else {
                        *local_states_distinct += 1;
                        if fp_cache.len() < LOCAL_FP_CACHE_SIZE {
                            fp_cache.insert(fp);
                        }
                        // Record in distributed bloom filter for cross-node dedup
                        if let Some(ref stealer) = worker_distributed_stealer {
                            stealer.record_explored(fp);
                        }
                        // Record parent tracking for trace reconstruction
                        if let Some(ref pm) = worker_parent_map {
                            if worker_trace_count.load(Ordering::Relaxed)
                                < worker_max_trace_states
                            {
                                pm.insert(fp, current_state_fp);
                                if let Some(ref sm) = worker_state_map {
                                    sm.insert(fp, next_state.clone());
                                }
                                worker_trace_count.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        let home_numa = worker_fp_store.home_numa(fp);
                        states_with_home_numa.push((next_state, home_numa));
                    }
                }

                // Distributed mode: donate some states for remote steal requests.
                // Push every 8th new state to the donation channel (low overhead).
                // try_send dropping on full is intentional — if the donate
                // channel is full there's already plenty of work pending for
                // remote stealers; dropping a sample doesn't lose state
                // (the producer keeps it in its own queue too) and never
                // blocks the producer's hot path.
                if let Some(ref donate_tx) = worker_donate_tx {
                    for (idx, (state, _)) in states_with_home_numa.iter().enumerate() {
                        if idx % 8 == 0 {
                            if let Ok(compressed) = serialize_compressed(state) {
                                let _ = donate_tx.try_send(compressed);
                            }
                        }
                    }
                }

                // Batch push with NUMA-aware routing
                let pushed = worker_queue.push_batch_to_numa(
                    &mut worker_state,
                    states_with_home_numa.drain(..),
                );
                *local_enqueued += pushed as u64;
            }

            *local_duplicates += duplicates_in_batch;

            // Clear cache if it gets too large (prevents unbounded memory)
            if fp_cache.len() >= LOCAL_FP_CACHE_SIZE {
                fp_cache.clear();
            }

            Ok(())
        };

        for next in successors.drain(..) {
            if worker_stop.load(Ordering::Acquire) {
                break;
            }
            pending_batch.push(next);
            if pending_batch.len() >= worker_fp_batch_size
                && let Err(err) = process_batch(
                    &mut pending_batch,
                    &mut local_duplicates,
                    &mut local_states_distinct,
                    &mut local_enqueued,
                    &mut local_fp_cache,
                    &mut local_fp_cache_hits,
                )
            {
                // Unbounded channel; send is infallible until error_rx drops
                // (after worker join), so the discarded Err is unreachable.
                let _ = worker_error_tx.send(err.to_string());
                worker_stop.store(true, Ordering::Release);
                break;
            }
        }
        if !worker_stop.load(Ordering::Acquire)
            && let Err(err) = process_batch(
                &mut pending_batch,
                &mut local_duplicates,
                &mut local_states_distinct,
                &mut local_enqueued,
                &mut local_fp_cache,
                &mut local_fp_cache_hits,
            )
        {
            // Unbounded channel; send is infallible until error_rx drops
            // (after worker join), so the discarded Err is unreachable.
            let _ = worker_error_tx.send(err.to_string());
            worker_stop.store(true, Ordering::Release);
        }

        // Mark worker as idle (done with this state)
        worker_queue.worker_idle(worker_state.id());
    }

    // Flush any remaining local stats before exiting
    flush_local_stats(
        &mut local_states_processed,
        &mut local_states_generated,
        &mut local_duplicates,
        &mut local_states_distinct,
        &mut local_enqueued,
        &worker_stats,
    );

    // Flush queue counters
    worker_queue.flush_worker_counters(&mut worker_state);

    // Distributed mode: signal locally idle when the last worker exits.
    let remaining = worker_live.fetch_sub(1, Ordering::AcqRel);
    if remaining == 1 {
        if let Some(ref stealer) = worker_distributed_stealer {
            stealer.set_locally_idle(true);
        }
    }
}
