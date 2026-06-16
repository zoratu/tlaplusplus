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
use serde::Serialize;
use serde::de::DeserializeOwned;
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
                // not hang waiting for an Init stream that has died. It also
                // signals the queue that Init enumeration is complete so the
                // work-stealing termination check can release its `has_started`
                // startup guard even when ZERO states were enqueued (e.g. resume
                // where every initial state is already in the fp-store). Without
                // this, workers spin forever in `pop_slow_path`. See
                // bug_checkpoint_resume_hang.
                struct Guard<St>(Arc<AtomicBool>, Arc<SpillableWorkStealingQueues<St>>)
                where
                    St: Serialize + DeserializeOwned + Send + Sync + Clone + 'static;
                impl<St> Drop for Guard<St>
                where
                    St: Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
                {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                        self.1.mark_init_complete();
                    }
                }
                let _guard = Guard::<M::State>(
                    producer_done_flag,
                    Arc::clone(&producer_queue),
                );

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

#[cfg(test)]
mod tests {
    //! Behavioural coverage for the T5.4 streaming Init producer.
    //!
    //! Each test instantiates a small Model (a few or zero or duplicated
    //! initial states) wired up to the same Spillable queue, fingerprint
    //! store, and atomic counters that the runtime uses, then asserts the
    //! producer thread observes the documented ordering invariants.
    use super::*;
    use crate::model::Model;
    use crate::storage::spillable_work_stealing::{
        SpillableConfig, SpillableWorkStealingQueues,
    };
    use crate::storage::unified_fingerprint_store::{
        UnifiedFingerprintConfig, UnifiedFingerprintStore,
    };
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
    struct TinyState {
        v: u32,
    }

    /// Model whose initial states are an explicit sequence (with possible
    /// duplicates), so we can assert producer-side dedup and counter math.
    struct ExplicitInitModel {
        inits: Vec<TinyState>,
    }

    impl Model for ExplicitInitModel {
        type State = TinyState;
        fn name(&self) -> &'static str {
            "explicit-init"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            self.inits.clone()
        }
        fn next_states(&self, _state: &Self::State, _out: &mut Vec<Self::State>) {}
        fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
            Ok(())
        }
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "tlapp-runtime-init-producer-{prefix}-{nanos}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn build_queue() -> Arc<SpillableWorkStealingQueues<TinyState>> {
        let cfg = SpillableConfig {
            max_inmem_items: 10_000,
            max_inmem_bytes: 0,
            est_bytes_per_item_seed: 256,
            spill_dir: temp_dir("queue"),
            spill_batch: 200,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: false,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };
        let (q, _w) =
            SpillableWorkStealingQueues::<TinyState>::new(1, vec![0], cfg).expect("queue");
        q
    }

    fn build_fp_store() -> Arc<UnifiedFingerprintStore> {
        let cfg = UnifiedFingerprintConfig {
            use_bloom: false,
            use_auto_switch: false,
            shard_count: 4,
            expected_items: 1_000,
            false_positive_rate: 0.01,
            shard_size_mb: 4,
            num_numa_nodes: 1,
            auto_switch_config: None,
            backing_dir: None,
        };
        let assigned: Vec<Option<usize>> = vec![None];
        Arc::new(
            UnifiedFingerprintStore::new(cfg, &assigned).expect("fp store init"),
        )
    }

    /// Drive the producer to completion and return the join result + the
    /// final value of `init_producing` (which the producer's Drop guard
    /// must have cleared).
    fn run_producer_to_completion(
        model: ExplicitInitModel,
    ) -> (Result<u64>, bool, Arc<AtomicRunStats>, Arc<SpillableWorkStealingQueues<TinyState>>)
    {
        let model = Arc::new(model);
        let queue = build_queue();
        let fp_store = build_fp_store();
        let run_stats = Arc::new(AtomicRunStats::default());
        let trace_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let init_producing = Arc::new(AtomicBool::new(true));
        let (err_tx, _err_rx) = crossbeam_channel::unbounded::<String>();
        let handle = spawn_init_producer(
            false,
            Arc::clone(&model),
            Arc::clone(&queue),
            Arc::clone(&fp_store),
            Arc::clone(&run_stats),
            None,
            Arc::clone(&trace_count),
            Arc::clone(&stop),
            err_tx,
            Arc::clone(&init_producing),
        )
        .expect("producer should be spawned when not resumed");
        let join_result = handle.join().expect("producer thread did not panic");
        let final_flag = init_producing.load(Ordering::Acquire);
        (join_result, final_flag, run_stats, queue)
    }

    #[test]
    fn returns_none_when_resumed_from_checkpoint() {
        // T5.4 invariant: when we resumed from a checkpoint, the queue is
        // already seeded with initial states from the manifest, so the
        // producer must NOT spawn (returning None signals "no thread").
        let model = Arc::new(ExplicitInitModel {
            inits: vec![TinyState { v: 0 }, TinyState { v: 1 }],
        });
        let queue = build_queue();
        let fp_store = build_fp_store();
        let run_stats = Arc::new(AtomicRunStats::default());
        let trace_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        // Caller passes a flag — when we early-return, we must NOT touch it.
        let init_producing = Arc::new(AtomicBool::new(true));
        let (err_tx, _err_rx) = crossbeam_channel::unbounded::<String>();
        let handle = spawn_init_producer(
            true, // resumed_from_checkpoint
            model,
            queue,
            fp_store,
            run_stats,
            None,
            trace_count,
            stop,
            err_tx,
            Arc::clone(&init_producing),
        );
        assert!(handle.is_none(), "resumed run must not spawn the producer");
        assert!(
            init_producing.load(Ordering::Acquire),
            "resumed early-return must not touch the producer flag"
        );
    }

    #[test]
    fn drop_guard_clears_init_producing_flag_on_success() {
        // T5.4 invariant: the Drop guard inside the producer closure clears
        // `init_producing` on every exit path. After the thread joins, the
        // flag MUST be false so workers won't hang waiting for more states.
        let (result, final_flag, _stats, _queue) = run_producer_to_completion(
            ExplicitInitModel {
                inits: vec![TinyState { v: 0 }, TinyState { v: 1 }, TinyState { v: 2 }],
            },
        );
        assert!(result.is_ok(), "producer ran to completion: {result:?}");
        assert!(
            !final_flag,
            "Drop guard must clear init_producing on successful exit"
        );
    }

    #[test]
    fn drop_guard_clears_init_producing_flag_on_empty_init() {
        // Even with zero initial states the producer must still clear the
        // flag — workers would otherwise spin forever waiting for the
        // never-coming first state.
        let (result, final_flag, stats, queue) = run_producer_to_completion(
            ExplicitInitModel { inits: Vec::new() },
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0, "no inits → no distinct returned");
        assert!(
            !final_flag,
            "Drop guard must clear init_producing even on empty Init"
        );
        let (gen_count, _, dist, dup, enq, _) = stats.snapshot();
        assert_eq!(gen_count, 0);
        assert_eq!(dist, 0);
        assert_eq!(dup, 0);
        assert_eq!(enq, 0);
        assert!(!queue.has_pending_work());
    }

    #[test]
    fn distinct_count_excludes_duplicates() {
        // Two duplicates among five entries → 3 distinct. The producer
        // dedups via a HashSet in addition to the fp-store CAS path.
        let inits = vec![
            TinyState { v: 1 },
            TinyState { v: 2 },
            TinyState { v: 1 }, // dup of #0
            TinyState { v: 3 },
            TinyState { v: 2 }, // dup of #1
        ];
        let (result, final_flag, stats, queue) =
            run_producer_to_completion(ExplicitInitModel { inits });
        assert!(!final_flag);
        let distinct_returned = result.expect("producer ok");
        assert_eq!(distinct_returned, 3, "3 unique states");
        let (gen_count, _, dist, dup, enq, _) = stats.snapshot();
        assert_eq!(gen_count, 5, "states_generated counts every yielded state");
        assert_eq!(dist, 3, "states_distinct counts only unique");
        assert_eq!(dup, 2, "duplicates counts the two repeats");
        assert_eq!(enq, 3, "enqueued = distinct (each unique pushed once)");
        // Queue should report exactly 3 items pending after the producer
        // finishes; this kills mutations that drop the push_global call.
        assert_eq!(queue.pending_count(), 3, "queue holds 3 distinct states");
    }

    #[test]
    fn observed_states_match_input_set() {
        // Sanity: every distinct input fingerprint must appear in the
        // queue, and no extras. Pop everything and compare as a set.
        use std::collections::HashSet;
        let inits = vec![
            TinyState { v: 10 },
            TinyState { v: 20 },
            TinyState { v: 30 },
        ];
        let model = Arc::new(ExplicitInitModel {
            inits: inits.clone(),
        });
        let queue = build_queue();
        let fp_store = build_fp_store();
        let run_stats = Arc::new(AtomicRunStats::default());
        let trace_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let init_producing = Arc::new(AtomicBool::new(true));
        let (err_tx, _err_rx) = crossbeam_channel::unbounded::<String>();

        // Build a real worker_state to drain the queue post-producer.
        let cfg = SpillableConfig {
            max_inmem_items: 10_000,
            max_inmem_bytes: 0,
            est_bytes_per_item_seed: 256,
            spill_dir: temp_dir("drain"),
            spill_batch: 200,
            load_existing: false,
            worker_spill_buffer_size: 50,
            worker_channel_bound: 4,
            defer_segment_deletion: false,
            compression_enabled: false,
            compression_max_bytes: 64 * 1024 * 1024,
            compression_level: 1,
        };
        let (drain_queue, mut workers) =
            SpillableWorkStealingQueues::<TinyState>::new(1, vec![0], cfg)
                .expect("drain queue");
        let handle = spawn_init_producer(
            false,
            Arc::clone(&model),
            Arc::clone(&drain_queue),
            Arc::clone(&fp_store),
            Arc::clone(&run_stats),
            None,
            Arc::clone(&trace_count),
            Arc::clone(&stop),
            err_tx,
            Arc::clone(&init_producing),
        )
        .expect("spawn");
        let _ = handle.join().unwrap().expect("producer ok");
        let mut seen: HashSet<u32> = HashSet::new();
        while let Some(state) = drain_queue.pop_for_worker(&mut workers[0]) {
            seen.insert(state.v);
        }
        let expected: HashSet<u32> = inits.iter().map(|s| s.v).collect();
        assert_eq!(seen, expected, "queue must contain exactly the inits");
    }

    #[test]
    fn stop_flag_aborts_producer_promptly() {
        // T5.4 invariant: the producer must check `stop.load(Relaxed)` on
        // every yielded state so a runtime-wide stop signal halts Init
        // enumeration. To exercise this without a long-running iterator
        // we pre-set stop=true; the producer must yield zero states and
        // still clear init_producing via its Drop guard.
        let model = Arc::new(ExplicitInitModel {
            inits: (0..50).map(|v| TinyState { v }).collect(),
        });
        let queue = build_queue();
        let fp_store = build_fp_store();
        let run_stats = Arc::new(AtomicRunStats::default());
        let trace_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(true)); // pre-set
        let init_producing = Arc::new(AtomicBool::new(true));
        let (err_tx, _err_rx) = crossbeam_channel::unbounded::<String>();
        let handle = spawn_init_producer(
            false,
            model,
            queue,
            fp_store,
            run_stats,
            None,
            trace_count,
            stop,
            err_tx,
            Arc::clone(&init_producing),
        )
        .expect("spawn");
        let result = handle.join().expect("no panic").expect("producer ok");
        assert_eq!(result, 0, "stop=true must abort before any distinct emitted");
        assert!(
            !init_producing.load(Ordering::Acquire),
            "Drop guard must still clear flag on stop-aborted exit"
        );
    }
}
