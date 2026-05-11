//! T10.2 stage 5 — multi-worker DFS pool with cross-partition routing.
//!
//! # What this module adds beyond `dfs_worker`
//!
//! Stage 4 (`runtime/dfs_worker.rs`) shipped a single-worker DFS that drops
//! the labeled-transitions DashMap and runs the fairness verdict in-band
//! on a local triple list. It works correctly for the gate-7 fixtures and
//! delivers a 41 % peak-RSS reduction on the high-fanout grid memory
//! benchmark, but exploration is single-threaded — on a 64-core spot it
//! pegs one CPU and leaves 63 idle.
//!
//! Stage 5 (this module) parallelises exploration across `N` worker
//! threads while keeping the in-band fairness verdict and the dropped
//! `labeled_transitions` DashMap from stage 4. The design:
//!
//! 1. **Static partitioning.** Each fingerprint has exactly one home
//!    worker, computed by `partition_for_fp(fp, N)`. Workers own their
//!    partition: only the home worker ever pushes a state's frame onto
//!    a DFS stack, so there is no per-state lock.
//! 2. **Per-worker channels.** Each worker has a crossbeam mpsc receiver.
//!    When worker A computes a successor whose home is worker B, A sends
//!    `(state, fp, action_label)` over B's channel. Local successors stay
//!    local and avoid the channel hop.
//! 3. **In-flight Mattern counter.** A single `AtomicI64` is incremented
//!    *before* every cross-partition send and decremented *after* every
//!    cross-partition receive. Termination requires this == 0 AND every
//!    worker reports its inbox + stack empty (idle). Crossbeam's
//!    happens-before guarantees the counter and channel are consistent.
//! 4. **Per-worker LocalAdjacency.** Each worker records its own
//!    `(from_fp, to_fp, action)` triples and its own `state_by_fp` cache.
//!    On termination the pool merges all per-worker structs into one
//!    triple list and one state map, then runs the same in-band fairness
//!    check the single-worker path uses (`dfs_worker::run_inband_fairness_check`).
//!    Verdict equivalence is by construction — same predicate, same input
//!    shape after merge.
//! 5. **Color map shared across workers.** The `PageAlignedColorMap` is
//!    already lock-free CAS-based, so concurrent CAS from multiple
//!    workers is safe. A worker only touches Cyan/Blue for its own
//!    partition, so contention is naturally low.
//!
//! # Why per-worker triples + post-exploration merge
//!
//! Alternative #1 was a single shared `Mutex<Vec<(u64,u64,String)>>` —
//! every edge would contend on a single lock, defeating the parallelism.
//! Alternative #2 was a sharded `DashMap<u64, Vec<(u64,String)>>` keyed
//! by `from_fp` — also has fast-path contention on shard locks. The
//! merge-after design pays a single O(total-edges) memmove at the end,
//! which is dwarfed by the exploration time on any non-trivial spec.
//!
//! # What the pool deliberately does NOT do
//!
//! - **No global DFS-stack semantics.** Each worker's blue stack is local
//!   to its partition. The CVWY-style "Cyan back-edge witness" property
//!   only holds *within* a partition. The fairness verdict comes from
//!   the post-exploration in-band Tarjan, not the per-worker red probe.
//!   This is acceptable because the verdict path is the predicate that
//!   is gate-tested for parity (gate-6/7/8), not the in-loop red probe.
//! - **No work-stealing within DFS.** Workers don't steal from each other
//!   — the static partition guarantees load balance only if the state
//!   graph's `partition_for_fp` distribution is roughly uniform, which
//!   it is for any reasonable fingerprint hash. If a real spec exhibits
//!   pathological skew, the right fix is a re-hash, not stealing (which
//!   would break the "exactly one DFS frame per fp" property).
//! - **No checkpoint coordination.** Same as stage 4: the DFS pool runs
//!   to completion or to a violation; checkpointing is a BFS-path
//!   facility.

use crate::model::Model;
use crate::storage::page_aligned_color_map::{Color as MapColor, PageAlignedColorMap};
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::dfs_cluster_bridge::ClusterBridge;
use super::dfs_worker::{
    FpStoreExt, LocalAdjacency, compute_labeled_successors, emit_safety_violation,
    run_inband_fairness_check,
};
use super::stats::AtomicRunStats;
use super::Violation;

/// Static partition assignment. Same bit-mixing used by
/// `PageAlignedFingerprintStore::home_numa` so DFS partitioning aligns
/// with the fingerprint store's NUMA placement when both are present.
#[inline]
pub(super) fn partition_for_fp(fp: u64, num_workers: usize) -> usize {
    debug_assert!(num_workers > 0, "num_workers must be positive");
    if num_workers == 1 {
        return 0;
    }
    let mixed = (fp >> 32) ^ (fp >> 16) ^ fp;
    (mixed as usize) % num_workers
}

/// Cross-partition explore message. Sent from the partition that
/// generated the successor to the partition that owns the successor.
///
/// Layer B note: this is the in-process counterpart of the
/// `Message::PartitionEdge` wire variant. The cluster bridge
/// (`dfs_cluster_bridge`) decodes incoming `PartitionEdge` payloads
/// back into this exact struct before pushing onto a local worker's
/// inbox — keeping the local-vs-remote message handling unified.
pub(super) struct ExploreMsg<S> {
    pub(super) state: S,
    pub(super) fp: u64,
    /// Predecessor fp + action label, recorded by the *receiving* worker
    /// in its LocalAdjacency so the post-exploration merge sees a
    /// complete edge list. We could record it on the sender side too,
    /// but recording on the receiver gives a cleaner "every triple is
    /// recorded by the worker that owns the destination" invariant —
    /// useful for debugging and for any future per-partition reductions.
    pub(super) from_fp: u64,
    pub(super) action: String,
    /// Set true if the source predecessor caller has already filtered out
    /// constraint-failing edges (it never has — but we keep the field so
    /// the wire format matches the in-process logic).
    pub(super) passes_constraints: bool,
}

/// Per-worker pool context. Built once by [`run_dfs_pool`] and handed
/// off by clone to each worker thread.
struct PoolWorker<M: Model> {
    id: usize,
    num_workers: usize,
    model: Arc<M>,
    fp_store: Arc<UnifiedFingerprintStore>,
    color_map: Arc<PageAlignedColorMap>,
    stats: Arc<AtomicRunStats>,
    stop: Arc<AtomicBool>,
    violation_tx: Sender<Violation<M::State>>,
    violation_count: Arc<AtomicUsize>,
    max_violations: usize,
    error_tx: Sender<String>,
    stop_on_violation: bool,
    parent_map: Option<Arc<DashMap<u64, u64>>>,
    state_map: Option<Arc<DashMap<u64, M::State>>>,
    trace_count: Arc<AtomicU64>,
    max_trace_states: u64,
    /// Inbound channel — this worker's own receiver.
    inbox_rx: Receiver<ExploreMsg<M::State>>,
    /// Outbound channels — `outbox_tx[i]` is the sender for worker i.
    /// `outbox_tx[self.id]` is the *self-send* path; we never use it
    /// (local successors go straight to the local stack), but it's kept
    /// in the array for index symmetry and is freed when the workers
    /// drop their context handles.
    outbox_tx: Vec<Sender<ExploreMsg<M::State>>>,
    /// Mattern in-flight counter, shared across all workers.
    inflight: Arc<AtomicI64>,
    /// Per-worker idle flags, indexed by worker id.
    idle_flags: Arc<Vec<AtomicBool>>,
    labeled_supported: bool,
    /// Optional cluster bridge — `Some` in multi-node mode, `None` in
    /// single-node Layer A mode. When present, successors whose home
    /// node != self.node_id are dispatched via the bridge instead of
    /// the in-process `outbox_tx`.
    cluster: Option<Arc<ClusterBridge<M::State>>>,
    /// This node's id in the cluster (or 0 in single-node mode).
    self_node: u32,
    /// Total number of cluster nodes (or 1 in single-node mode).
    num_nodes: u32,
}

/// Per-worker output collected on join. The pool merges these to feed
/// the single in-band fairness check.
struct WorkerOutput<S> {
    triples: Vec<(u64, u64, String)>,
    state_by_fp: HashMap<u64, S>,
    /// Number of states the worker generated (raw, before dedup) — for
    /// stats-flush parity with the BFS path.
    states_seen: u64,
    states_distinct: u64,
    states_duplicates: u64,
}

/// Per-worker DFS frame. Same shape as `dfs_worker::BlueFrame` but
/// duplicated locally to avoid cross-module visibility plumbing on a
/// 5-field struct.
struct PoolFrame<S> {
    fp: u64,
    succs: Vec<(S, String, bool)>,
    succ_idx: usize,
    color_owned: bool,
}

/// Public bundle that runs the multi-worker pool. Mirrors the shape of
/// `dfs_worker::DfsWorkerCtx` plus the worker count and (Layer B) the
/// optional cluster bridge.
pub(super) struct DfsPoolCtx<M: Model> {
    pub(super) num_workers: usize,
    pub(super) model: Arc<M>,
    pub(super) fp_store: Arc<UnifiedFingerprintStore>,
    pub(super) color_map: Arc<PageAlignedColorMap>,
    pub(super) stats: Arc<AtomicRunStats>,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) violation_tx: Sender<Violation<M::State>>,
    pub(super) violation_count: Arc<AtomicUsize>,
    pub(super) max_violations: usize,
    pub(super) error_tx: Sender<String>,
    pub(super) stop_on_violation: bool,
    pub(super) parent_map: Option<Arc<DashMap<u64, u64>>>,
    pub(super) state_map: Option<Arc<DashMap<u64, M::State>>>,
    pub(super) trace_count: Arc<AtomicU64>,
    pub(super) max_trace_states: u64,
    pub(super) dfs_inband_verdict_done: Arc<AtomicBool>,
    /// Layer B cluster setup. When `Some`, the pool's workers route
    /// remote-partition successors via the bridge's transport instead
    /// of the in-process channel. When `None` (the v1.2.5 default),
    /// behavior is identical to single-node Layer A.
    pub(super) cluster: Option<DfsPoolClusterCtx<M::State>>,
}

/// Layer B cluster wiring passed in by the runtime when the DFS pool
/// is part of a multi-node cluster run. The pool itself constructs the
/// `ClusterBridge` once `inbox_tx` handles are available.
pub(super) struct DfsPoolClusterCtx<S>
where
    S: Clone + Send + Sync + 'static + serde::Serialize + serde::de::DeserializeOwned,
{
    pub(super) transport: Arc<dyn crate::distributed::transport::Transport>,
    pub(super) node_id: u32,
    pub(super) num_nodes: u32,
    pub(super) tokio_handle: tokio::runtime::Handle,
    /// Phantom for the type parameter.
    pub(super) _phantom: std::marker::PhantomData<fn() -> S>,
}

/// Run the multi-worker DFS pool to completion. This is the Stage 5
/// entrypoint that replaces the single `run_dfs_worker` call in the
/// runtime when `num_dfs_workers > 1`.
pub(super) fn run_dfs_pool<M: Model>(ctx: DfsPoolCtx<M>) {
    assert!(ctx.num_workers >= 1, "DFS pool needs at least one worker");

    let DfsPoolCtx {
        num_workers,
        model,
        fp_store,
        color_map,
        stats,
        stop,
        violation_tx,
        violation_count,
        max_violations,
        error_tx,
        stop_on_violation,
        parent_map,
        state_map,
        trace_count,
        max_trace_states,
        dfs_inband_verdict_done,
        cluster,
    } = ctx;

    let (self_node, num_nodes) = match cluster.as_ref() {
        Some(cc) => (cc.node_id, cc.num_nodes),
        None => (0u32, 1u32),
    };

    let started_at = Instant::now();

    // ---- Initial-state seeding -----------------------------------------
    let initial_states: Vec<M::State> = model.initial_states();
    if initial_states.is_empty() {
        eprintln!(
            "[dfs-pool n={}] zero initial states — exploration trivially complete",
            num_workers
        );
        let _ = error_tx;
        dfs_inband_verdict_done.store(true, Ordering::Release);
        return;
    }

    let labeled_supported = model.next_states_labeled(&initial_states[0]).is_some();

    // ---- Per-worker channels -------------------------------------------
    //
    // Each worker has its own unbounded receiver. Bounded would be more
    // memory-frugal but introduces backpressure deadlock potential when
    // every worker is blocked sending to a full peer; unbounded is the
    // safe choice for stage-5 single-node multi-worker. Cluster mode
    // (Layer B) will use the network channel instead.
    let mut inbox_rxs: Vec<Receiver<ExploreMsg<M::State>>> = Vec::with_capacity(num_workers);
    let mut outbox_txs: Vec<Sender<ExploreMsg<M::State>>> = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let (tx, rx) = crossbeam_channel::unbounded();
        outbox_txs.push(tx);
        inbox_rxs.push(rx);
    }

    let inflight = Arc::new(AtomicI64::new(0));
    let mut idle_vec: Vec<AtomicBool> = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        idle_vec.push(AtomicBool::new(false));
    }
    let idle_flags = Arc::new(idle_vec);

    // ---- Distribute initial states to their home partitions ------------
    //
    // Each initial state is assigned to its home worker. The dedup +
    // color CAS happens inside the worker, mirroring the single-worker
    // path's seed loop. We bypass the inflight counter for seeds — they
    // are the conceptual "external input" that the pool starts from.
    //
    // Layer B note: in cluster mode every node enumerates the same
    // initial states (model.initial_states() is deterministic), but
    // each node only seeds the ones whose `(node_id, worker_id)` home
    // matches the local node — peer nodes will seed the remainder.
    // This guarantees no duplicate seeding without requiring a network
    // round-trip.
    let mut seeds_per_worker: Vec<Vec<M::State>> = (0..num_workers)
        .map(|_| Vec::new())
        .collect();
    for s in initial_states {
        let canon = model.canonicalize(s);
        let fp = model.fingerprint(&canon);
        let owner_local = if num_nodes > 1 {
            let (owner_node, owner_worker) = super::dfs_cluster_bridge::partition_for_fp_cluster(
                fp,
                num_nodes,
                num_workers,
            );
            if owner_node != self_node {
                continue; // peer node will seed this initial state
            }
            owner_worker
        } else {
            partition_for_fp(fp, num_workers)
        };
        seeds_per_worker[owner_local].push(canon);
    }

    // ---- Build cluster bridge (Layer B only) ---------------------------
    let cluster_bridge: Option<Arc<ClusterBridge<M::State>>> = match cluster {
        Some(cc) => {
            // The bridge needs sender handles to push received messages
            // onto each local worker's inbox. We use clones of the
            // existing per-worker `outbox_txs` (which are already the
            // canonical per-worker inboxes — `outbox_tx[i]` is the
            // sender side of `inbox_rx[i]`).
            let bridge_inboxes: Vec<Sender<ExploreMsg<M::State>>> = outbox_txs.clone();
            let bridge = ClusterBridge::<M::State>::new(
                cc.transport,
                cc.node_id,
                cc.num_nodes,
                num_workers,
                bridge_inboxes,
                Arc::clone(&stop),
                cc.tokio_handle,
                Arc::clone(&inflight),
            );
            // Spawn the recv + broadcast tasks on the tokio runtime.
            Arc::clone(&bridge).start();
            Some(bridge)
        }
        None => None,
    };

    // ---- Spawn workers -------------------------------------------------
    let mut handles: Vec<std::thread::JoinHandle<WorkerOutput<M::State>>> =
        Vec::with_capacity(num_workers);
    let mut seeds_iter = seeds_per_worker.into_iter().enumerate();

    for inbox_rx in inbox_rxs.into_iter() {
        let (id, seeds) = seeds_iter.next().expect("seeds_per_worker length mismatch");
        let worker = PoolWorker::<M> {
            id,
            num_workers,
            model: Arc::clone(&model),
            fp_store: Arc::clone(&fp_store),
            color_map: Arc::clone(&color_map),
            stats: Arc::clone(&stats),
            stop: Arc::clone(&stop),
            violation_tx: violation_tx.clone(),
            violation_count: Arc::clone(&violation_count),
            max_violations,
            error_tx: error_tx.clone(),
            stop_on_violation,
            parent_map: parent_map.clone(),
            state_map: state_map.clone(),
            trace_count: Arc::clone(&trace_count),
            max_trace_states,
            inbox_rx,
            outbox_tx: outbox_txs.clone(),
            inflight: Arc::clone(&inflight),
            idle_flags: Arc::clone(&idle_flags),
            labeled_supported,
            cluster: cluster_bridge.clone(),
            self_node,
            num_nodes,
        };
        handles.push(std::thread::spawn(move || run_one_pool_worker(worker, seeds)));
    }

    // Drop the runtime-side outbox handles so the only remaining senders
    // are the per-worker clones inside `PoolWorker::outbox_tx` AND the
    // cluster bridge's clones. When all workers exit and drop their
    // clones — and the bridge releases its set on shutdown — the
    // channels close cleanly.
    drop(outbox_txs);

    // ---- Join + merge --------------------------------------------------
    let mut all_triples: Vec<(u64, u64, String)> = Vec::new();
    let mut merged_state_by_fp: HashMap<u64, M::State> = HashMap::new();
    let mut total_seen: u64 = 0;
    let mut total_distinct: u64 = 0;
    let mut total_dup: u64 = 0;

    for h in handles {
        match h.join() {
            Ok(out) => {
                total_seen = total_seen.saturating_add(out.states_seen);
                total_distinct = total_distinct.saturating_add(out.states_distinct);
                total_dup = total_dup.saturating_add(out.states_duplicates);
                if all_triples.is_empty() {
                    all_triples = out.triples;
                } else {
                    all_triples.reserve(out.triples.len());
                    all_triples.extend(out.triples);
                }
                for (fp, s) in out.state_by_fp {
                    merged_state_by_fp.entry(fp).or_insert(s);
                }
            }
            Err(panic) => {
                let msg = match panic.downcast_ref::<&'static str>() {
                    Some(s) => format!("dfs pool worker panicked: {}", s),
                    None => match panic.downcast_ref::<String>() {
                        Some(s) => format!("dfs pool worker panicked: {}", s),
                        None => "dfs pool worker panicked (non-string payload)".to_string(),
                    },
                };
                let _ = error_tx.send(msg);
                stop.store(true, Ordering::Release);
            }
        }
    }

    // Stats — single flush at end, mirroring single-worker DFS.
    stats.states_generated.fetch_add(total_seen, Ordering::Relaxed);
    stats.states_processed.fetch_add(total_distinct, Ordering::Relaxed);
    stats.states_distinct.fetch_add(total_distinct, Ordering::Relaxed);
    stats.duplicates.fetch_add(total_dup, Ordering::Relaxed);
    stats.enqueued.fetch_add(total_distinct, Ordering::Relaxed);

    let exploration_elapsed = started_at.elapsed();

    // ---- Single in-band fairness check on merged adjacency ------------
    let inband_started_at = Instant::now();
    let safety_already = violation_count.load(Ordering::Acquire) > 0;
    let constraints = model.fairness_constraints();
    let edge_count = all_triples.len();
    if !safety_already && !constraints.is_empty() && !all_triples.is_empty() {
        run_inband_fairness_check::<M>(
            &model,
            all_triples,
            &constraints,
            &merged_state_by_fp,
            &violation_tx,
            &violation_count,
            stop_on_violation,
            max_violations,
            &stop,
        );
    } else if safety_already {
        eprintln!(
            "[dfs-pool n={}] in-band fairness check skipped: safety violation already reported",
            num_workers
        );
    } else if constraints.is_empty() {
        eprintln!(
            "[dfs-pool n={}] in-band fairness check skipped: no fairness constraints",
            num_workers
        );
    } else {
        eprintln!(
            "[dfs-pool n={}] in-band fairness check skipped: no transitions recorded",
            num_workers
        );
    }

    eprintln!(
        "[dfs-pool n={}] explored {} distinct states ({} edges) in {:.2?}, in-band fairness in {:.2?}; \
         labeled-transitions DashMap: NOT MATERIALIZED",
        num_workers, total_distinct, edge_count, exploration_elapsed, inband_started_at.elapsed(),
    );

    dfs_inband_verdict_done.store(true, Ordering::Release);
}

/// Per-worker main loop. Returns the worker's accumulated triples +
/// state map for post-exploration merge.
fn run_one_pool_worker<M: Model>(
    w: PoolWorker<M>,
    seeds: Vec<M::State>,
) -> WorkerOutput<M::State> {
    let PoolWorker {
        id,
        num_workers,
        model,
        fp_store,
        color_map,
        stats: _stats,
        stop,
        violation_tx,
        violation_count,
        max_violations,
        error_tx,
        stop_on_violation,
        parent_map,
        state_map,
        trace_count,
        max_trace_states,
        inbox_rx,
        outbox_tx,
        inflight,
        idle_flags,
        labeled_supported,
        cluster,
        self_node,
        num_nodes,
    } = w;

    let mut local_fp_cache: HashSet<u64> = HashSet::with_capacity(8192);
    let mut state_by_fp: HashMap<u64, M::State> = HashMap::with_capacity(8192);
    let mut adj = LocalAdjacency::default();

    let mut total_seen: u64 = 0;
    let mut total_distinct: u64 = 0;
    let mut total_duplicates: u64 = 0;

    // Stack of frames currently being explored by this worker. Mirrors
    // the single-worker `BlueFrame` stack but lives in this worker's
    // partition only.
    let mut blue_stack: Vec<PoolFrame<M::State>> = Vec::with_capacity(64);

    // ---- Process seeds (initial states) -------------------------------
    //
    // Each seed is dedup'd through the global fp_store and color-CAS'd
    // before being pushed to the stack. Same shape as
    // `run_dfs_worker::'init_loop:`.
    for seed in seeds {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let fp = model.fingerprint(&seed);
        debug_assert_eq!(
            partition_for_fp(fp, num_workers),
            id,
            "seed routed to wrong worker"
        );
        let was_present = match fp_store.contains_or_insert_unchecked(fp) {
            Ok(p) => p,
            Err(e) => {
                let _ = error_tx.send(format!("dfs-pool init dedup failed: {e}"));
                stop.store(true, Ordering::Release);
                break;
            }
        };
        if was_present {
            total_duplicates += 1;
            continue;
        }
        total_distinct += 1;
        state_by_fp.entry(fp).or_insert_with(|| seed.clone());

        if color_map.cas(fp, MapColor::White, MapColor::Cyan).is_err() {
            // Another worker (extremely rare across init-state CAS races
            // since seeds are partitioned by home, but possible if two
            // distinct seeds canonicalise to the same fp). Skip.
            continue;
        }

        // Inline check_invariants on the seed before computing successors
        // to match the single-worker `run_blue_dfs` ordering.
        if let Err(message) = model.check_invariants(&seed) {
            emit_safety_violation(
                &model,
                seed.clone(),
                message,
                &parent_map,
                &state_map,
                &violation_tx,
                &violation_count,
                max_violations,
                stop_on_violation,
                &stop,
            );
            if stop.load(Ordering::Acquire) {
                let _ = color_map.cas(fp, MapColor::Cyan, MapColor::Blue);
                break;
            }
        }

        let succs = compute_labeled_successors(&model, &seed, labeled_supported);
        blue_stack.push(PoolFrame {
            fp,
            succs,
            succ_idx: 0,
            color_owned: true,
        });
        total_seen += 1;
    }

    // ---- Main loop ----------------------------------------------------
    //
    // We interleave: 1) drain inbox to populate the stack with foreign
    // explore requests; 2) advance the top stack frame; 3) post-order
    // pop when frame exhausted; 4) when stack + inbox both empty, mark
    // idle and check the global termination predicate. Wake on inbox
    // recv via a short blocking call to avoid busy-spinning.

    let mut self_was_idle = false;

    loop {
        if stop.load(Ordering::Acquire) {
            // Drain the stack back to Blue so the color map is consistent.
            while let Some(frame) = blue_stack.pop() {
                if frame.color_owned {
                    let _ = color_map.cas(frame.fp, MapColor::Cyan, MapColor::Blue);
                }
            }
            break;
        }

        // 1) Drain inbox into the stack.
        let drained = drain_inbox::<M>(
            &inbox_rx,
            &model,
            &fp_store,
            &color_map,
            &mut adj,
            &mut state_by_fp,
            &mut local_fp_cache,
            &parent_map,
            &state_map,
            &trace_count,
            max_trace_states,
            &violation_tx,
            &violation_count,
            max_violations,
            stop_on_violation,
            &stop,
            &inflight,
            &mut blue_stack,
            &mut total_seen,
            &mut total_distinct,
            &mut total_duplicates,
            labeled_supported,
        );

        if drained > 0 && self_was_idle {
            idle_flags[id].store(false, Ordering::Release);
            self_was_idle = false;
        }

        // 2) Advance top frame.
        if !blue_stack.is_empty() {
            if self_was_idle {
                idle_flags[id].store(false, Ordering::Release);
                self_was_idle = false;
                if let Some(b) = cluster.as_ref() {
                    b.set_self_idle(false);
                }
            }
            advance_one_step::<M>(
                &model,
                &fp_store,
                &color_map,
                id,
                num_workers,
                self_node,
                num_nodes,
                cluster.as_ref(),
                &outbox_tx,
                &inflight,
                &idle_flags,
                &mut adj,
                &mut state_by_fp,
                &mut local_fp_cache,
                &parent_map,
                &state_map,
                &trace_count,
                max_trace_states,
                &violation_tx,
                &violation_count,
                max_violations,
                stop_on_violation,
                &stop,
                &mut blue_stack,
                &mut total_seen,
                &mut total_distinct,
                &mut total_duplicates,
                labeled_supported,
            );
            continue;
        }

        // 3) Stack empty. Try to pull more from inbox; if nothing,
        // signal idle and re-check global termination.
        if !self_was_idle {
            idle_flags[id].store(true, Ordering::Release);
            self_was_idle = true;
            // In cluster mode the bridge needs to know when the local
            // pool's idle predicate flips so its termination consensus
            // tokens are accurate. We propagate idle when ALL local
            // workers are idle to avoid thrashing the bridge counter.
            if let Some(b) = cluster.as_ref() {
                if all_idle(&idle_flags) && inbox_rx.is_empty() {
                    b.set_self_idle(true);
                }
            }
        }

        // Termination check: all idle + inflight == 0 + my inbox empty.
        // We re-check inflight and inbox after marking idle to close the
        // race where a peer increments inflight then sends to us between
        // our two checks.
        let local_quiescent = inflight.load(Ordering::Acquire) == 0
            && inbox_rx.is_empty()
            && all_idle(&idle_flags);

        if local_quiescent {
            // Double-confirm: re-read inflight after seeing all_idle.
            // Mattern-style 2-round: if we still see 0 and no peer is
            // working we are done.
            let still_quiescent = inflight.load(Ordering::Acquire) == 0
                && inbox_rx.is_empty()
                && all_idle(&idle_flags);
            if still_quiescent {
                match cluster.as_ref() {
                    None => break,
                    Some(b) => {
                        b.set_self_idle(true);
                        if b.is_globally_terminated() {
                            if std::env::var("TLAPP_VERBOSE").is_ok() {
                                eprintln!(
                                    "[dfs-pool n={} w={}] cluster terminated; exiting",
                                    self_node, id
                                );
                            }
                            break;
                        }
                        // Don't break yet — the bridge needs at least
                        // one round of cluster-quiescent broadcasts to
                        // promote local-quiescent to globally-terminated.
                        // Fall through to the recv_timeout below to
                        // park briefly, giving the bridge time to flip
                        // the global flag.
                    }
                }
            }
        }

        // Park briefly waiting for a message. Use a short timeout so
        // termination detection isn't blocked on an indefinite recv.
        match inbox_rx.recv_timeout(std::time::Duration::from_millis(2)) {
            Ok(msg) => {
                idle_flags[id].store(false, Ordering::Release);
                self_was_idle = false;
                if let Some(b) = cluster.as_ref() {
                    b.set_self_idle(false);
                }
                handle_explore_msg::<M>(
                    msg,
                    &model,
                    &fp_store,
                    &color_map,
                    &mut adj,
                    &mut state_by_fp,
                    &mut local_fp_cache,
                    &parent_map,
                    &state_map,
                    &trace_count,
                    max_trace_states,
                    &violation_tx,
                    &violation_count,
                    max_violations,
                    stop_on_violation,
                    &stop,
                    &inflight,
                    &mut blue_stack,
                    &mut total_seen,
                    &mut total_distinct,
                    &mut total_duplicates,
                    labeled_supported,
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Loop and re-check termination.
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // All senders dropped — pool is shutting down.
                break;
            }
        }
    }

    WorkerOutput {
        triples: adj.into_triples(),
        state_by_fp,
        states_seen: total_seen,
        states_distinct: total_distinct,
        states_duplicates: total_duplicates,
    }
}

/// Drain everything currently sitting in the inbox into the worker's
/// local DFS stack. Returns how many messages were drained.
#[allow(clippy::too_many_arguments)]
fn drain_inbox<M: Model>(
    inbox_rx: &Receiver<ExploreMsg<M::State>>,
    model: &Arc<M>,
    fp_store: &Arc<UnifiedFingerprintStore>,
    color_map: &Arc<PageAlignedColorMap>,
    adj: &mut LocalAdjacency,
    state_by_fp: &mut HashMap<u64, M::State>,
    local_fp_cache: &mut HashSet<u64>,
    parent_map: &Option<Arc<DashMap<u64, u64>>>,
    state_map: &Option<Arc<DashMap<u64, M::State>>>,
    trace_count: &Arc<AtomicU64>,
    max_trace_states: u64,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<AtomicUsize>,
    max_violations: usize,
    stop_on_violation: bool,
    stop: &Arc<AtomicBool>,
    inflight: &Arc<AtomicI64>,
    blue_stack: &mut Vec<PoolFrame<M::State>>,
    total_seen: &mut u64,
    total_distinct: &mut u64,
    total_duplicates: &mut u64,
    labeled_supported: bool,
) -> usize {
    let mut drained = 0;
    loop {
        match inbox_rx.try_recv() {
            Ok(msg) => {
                drained += 1;
                handle_explore_msg::<M>(
                    msg,
                    model,
                    fp_store,
                    color_map,
                    adj,
                    state_by_fp,
                    local_fp_cache,
                    parent_map,
                    state_map,
                    trace_count,
                    max_trace_states,
                    violation_tx,
                    violation_count,
                    max_violations,
                    stop_on_violation,
                    stop,
                    inflight,
                    blue_stack,
                    total_seen,
                    total_distinct,
                    total_duplicates,
                    labeled_supported,
                );
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
        }
    }
    drained
}

/// Process one inbound explore message. Records the predecessor edge,
/// dedups against the global fp_store, color-CAS to Cyan, and pushes a
/// frame onto this worker's stack if the state is fresh.
#[allow(clippy::too_many_arguments)]
fn handle_explore_msg<M: Model>(
    msg: ExploreMsg<M::State>,
    model: &Arc<M>,
    fp_store: &Arc<UnifiedFingerprintStore>,
    color_map: &Arc<PageAlignedColorMap>,
    adj: &mut LocalAdjacency,
    state_by_fp: &mut HashMap<u64, M::State>,
    local_fp_cache: &mut HashSet<u64>,
    parent_map: &Option<Arc<DashMap<u64, u64>>>,
    state_map: &Option<Arc<DashMap<u64, M::State>>>,
    trace_count: &Arc<AtomicU64>,
    max_trace_states: u64,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<AtomicUsize>,
    max_violations: usize,
    stop_on_violation: bool,
    stop: &Arc<AtomicBool>,
    inflight: &Arc<AtomicI64>,
    blue_stack: &mut Vec<PoolFrame<M::State>>,
    total_seen: &mut u64,
    total_distinct: &mut u64,
    total_duplicates: &mut u64,
    labeled_supported: bool,
) {
    let ExploreMsg {
        state,
        fp,
        from_fp,
        action,
        passes_constraints,
    } = msg;

    // CRITICAL: decrement inflight as soon as we own the message. This
    // releases the Mattern token even if we early-return below.
    inflight.fetch_sub(1, Ordering::AcqRel);

    *total_seen += 1;

    // Record the predecessor edge in our LocalAdjacency. The triple list
    // is the input to the post-exploration in-band fairness check; every
    // edge must appear exactly once, and recording it here in the owner
    // worker satisfies that.
    adj.record(from_fp, fp, action);

    if !passes_constraints {
        // Constraint-failing edges are recorded but we don't descend.
        return;
    }

    if local_fp_cache.contains(&fp) {
        *total_duplicates += 1;
        return;
    }

    let was_present = match fp_store.contains_or_insert_unchecked(fp) {
        Ok(p) => p,
        Err(e) => {
            let _ = violation_tx.try_send(Violation {
                message: format!("dfs-pool fp_store insert failed: {}", e),
                state: state.clone(),
                property_type: super::PropertyType::Safety,
                trace: vec![state],
            });
            stop.store(true, Ordering::Release);
            return;
        }
    };
    if was_present {
        *total_duplicates += 1;
        if local_fp_cache.len() < 65_536 {
            local_fp_cache.insert(fp);
        }
        return;
    }
    if local_fp_cache.len() < 65_536 {
        local_fp_cache.insert(fp);
    }
    *total_distinct += 1;
    if state_by_fp.len() < (max_trace_states as usize).max(8192) {
        state_by_fp
            .entry(fp)
            .or_insert_with(|| state.clone());
    }

    if let Some(pm) = parent_map.as_ref() {
        if trace_count.load(Ordering::Relaxed) < max_trace_states {
            pm.insert(fp, from_fp);
            if let Some(sm) = state_map.as_ref() {
                sm.insert(fp, state.clone());
            }
            trace_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    let color_owned = color_map.cas(fp, MapColor::White, MapColor::Cyan).is_ok();

    if let Err(message) = model.check_invariants(&state) {
        emit_safety_violation(
            model,
            state.clone(),
            message,
            parent_map,
            state_map,
            violation_tx,
            violation_count,
            max_violations,
            stop_on_violation,
            stop,
        );
        if stop.load(Ordering::Acquire) {
            if color_owned {
                let _ = color_map.cas(fp, MapColor::Cyan, MapColor::Blue);
            }
            return;
        }
    }

    let succs = compute_labeled_successors(model, &state, labeled_supported);
    blue_stack.push(PoolFrame {
        fp,
        succs,
        succ_idx: 0,
        color_owned,
    });
}

/// Advance the top frame by visiting its next successor (or post-order
/// pop if exhausted). Mirrors `dfs_worker::run_blue_dfs`'s body but
/// routes successors via `partition_for_fp` (single-node Layer A) or
/// the cluster bridge (multi-node Layer B).
#[allow(clippy::too_many_arguments)]
fn advance_one_step<M: Model>(
    model: &Arc<M>,
    fp_store: &Arc<UnifiedFingerprintStore>,
    color_map: &Arc<PageAlignedColorMap>,
    self_id: usize,
    num_workers: usize,
    self_node: u32,
    num_nodes: u32,
    cluster: Option<&Arc<ClusterBridge<M::State>>>,
    outbox_tx: &[Sender<ExploreMsg<M::State>>],
    inflight: &Arc<AtomicI64>,
    _idle_flags: &Arc<Vec<AtomicBool>>,
    adj: &mut LocalAdjacency,
    state_by_fp: &mut HashMap<u64, M::State>,
    local_fp_cache: &mut HashSet<u64>,
    parent_map: &Option<Arc<DashMap<u64, u64>>>,
    state_map: &Option<Arc<DashMap<u64, M::State>>>,
    trace_count: &Arc<AtomicU64>,
    max_trace_states: u64,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<AtomicUsize>,
    max_violations: usize,
    stop_on_violation: bool,
    stop: &Arc<AtomicBool>,
    blue_stack: &mut Vec<PoolFrame<M::State>>,
    total_seen: &mut u64,
    total_distinct: &mut u64,
    total_duplicates: &mut u64,
    labeled_supported: bool,
) {
    enum Outcome<S> {
        Descend(u64, Vec<(S, String, bool)>, bool),
        Pop,
        ShouldStop,
    }

    // Capture this here, before the `last_mut()` borrow, so the
    // bridge send below doesn't need a fresh `&blue_stack` while a
    // mutable borrow is already live.
    let stack_depth = blue_stack.len() as u32;
    let outcome = {
        let top = match blue_stack.last_mut() {
            Some(t) => t,
            None => return,
        };
        let top_fp = top.fp;
        let mut chosen: Outcome<M::State> = Outcome::Pop;
        while top.succ_idx < top.succs.len() {
            let (next_state, action_name, passes_constraints) =
                top.succs[top.succ_idx].clone();
            top.succ_idx += 1;

            let next_state = model.canonicalize(next_state);
            let next_fp = model.fingerprint(&next_state);
            *total_seen += 1;

            // Cross-partition routing: if the successor's home is not us,
            // ship it to the owner. The owner records the edge in its
            // own LocalAdjacency, so we deliberately do NOT record here.
            //
            // In single-node mode (`num_nodes == 1`) the owner lookup
            // returns `(0, worker_id)` and routing is purely in-process.
            // In Layer B cluster mode, an owner on another node is
            // dispatched through the bridge instead of `outbox_tx`.
            let (owner_node, owner_worker) = if num_nodes > 1 {
                super::dfs_cluster_bridge::partition_for_fp_cluster(
                    next_fp, num_nodes, num_workers,
                )
            } else {
                (self_node, partition_for_fp(next_fp, num_workers))
            };

            if owner_node != self_node {
                // Cross-NODE successor — bridge dispatch.
                let bridge = cluster.expect("num_nodes>1 implies cluster bridge is set");
                if let Err(e) = bridge.send_remote(
                    owner_node,
                    owner_worker,
                    next_state.clone(),
                    next_fp,
                    top_fp,
                    action_name.clone(),
                    passes_constraints,
                    stack_depth,
                ) {
                    eprintln!(
                        "[dfs-pool n={} w={}] cross-node send failed: {}; \
                         falling back to local routing",
                        self_node, self_id, e
                    );
                    // Best-effort fallback: record the edge locally so
                    // the verdict isn't silently dropped. The peer is
                    // unreachable; the cluster will eventually time out.
                    adj.record(top_fp, next_fp, action_name);
                }
                continue;
            }

            if owner_worker != self_id {
                inflight.fetch_add(1, Ordering::AcqRel);
                let msg = ExploreMsg {
                    state: next_state,
                    fp: next_fp,
                    from_fp: top_fp,
                    action: action_name,
                    passes_constraints,
                };
                if let Err(send_err) = outbox_tx[owner_worker].send(msg) {
                    // Channel closed unexpectedly — the receiver
                    // worker already exited. This shouldn't happen
                    // unless we're shutting down. Roll back inflight
                    // and stop.
                    inflight.fetch_sub(1, Ordering::AcqRel);
                    let _ = send_err; // ignore the unsent ExploreMsg
                    stop.store(true, Ordering::Release);
                    chosen = Outcome::ShouldStop;
                    break;
                }
                continue;
            }

            // Local successor — record + dedup + maybe descend, just
            // like the single-worker path.
            adj.record(top_fp, next_fp, action_name);

            if !passes_constraints {
                continue;
            }

            if local_fp_cache.contains(&next_fp) {
                *total_duplicates += 1;
                continue;
            }

            let was_present = match fp_store.contains_or_insert_unchecked(next_fp) {
                Ok(p) => p,
                Err(e) => {
                    let _ = violation_tx.try_send(Violation {
                        message: format!("dfs-pool fp_store insert failed: {}", e),
                        state: next_state.clone(),
                        property_type: super::PropertyType::Safety,
                        trace: vec![next_state],
                    });
                    stop.store(true, Ordering::Release);
                    chosen = Outcome::ShouldStop;
                    break;
                }
            };
            if was_present {
                *total_duplicates += 1;
                if local_fp_cache.len() < 65_536 {
                    local_fp_cache.insert(next_fp);
                }
                continue;
            }
            if local_fp_cache.len() < 65_536 {
                local_fp_cache.insert(next_fp);
            }
            *total_distinct += 1;
            if state_by_fp.len() < (max_trace_states as usize).max(8192) {
                state_by_fp
                    .entry(next_fp)
                    .or_insert_with(|| next_state.clone());
            }

            if let Some(pm) = parent_map.as_ref() {
                if trace_count.load(Ordering::Relaxed) < max_trace_states {
                    pm.insert(next_fp, top_fp);
                    if let Some(sm) = state_map.as_ref() {
                        sm.insert(next_fp, next_state.clone());
                    }
                    trace_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            let color_owned = color_map
                .cas(next_fp, MapColor::White, MapColor::Cyan)
                .is_ok();

            if let Err(message) = model.check_invariants(&next_state) {
                emit_safety_violation(
                    model,
                    next_state.clone(),
                    message,
                    parent_map,
                    state_map,
                    violation_tx,
                    violation_count,
                    max_violations,
                    stop_on_violation,
                    stop,
                );
                if stop.load(Ordering::Acquire) {
                    if color_owned {
                        let _ = color_map.cas(next_fp, MapColor::Cyan, MapColor::Blue);
                    }
                    chosen = Outcome::ShouldStop;
                    break;
                }
            }

            let succs = compute_labeled_successors(model, &next_state, labeled_supported);
            chosen = Outcome::Descend(next_fp, succs, color_owned);
            break;
        }
        chosen
    };

    match outcome {
        Outcome::Descend(next_fp, succs, color_owned) => {
            blue_stack.push(PoolFrame {
                fp: next_fp,
                succs,
                succ_idx: 0,
                color_owned,
            });
        }
        Outcome::Pop => {
            let popped = blue_stack
                .pop()
                .expect("blue_stack non-empty inside advance_one_step");
            if popped.color_owned {
                let cas_done = color_map.cas(popped.fp, MapColor::Cyan, MapColor::Blue);
                debug_assert!(
                    cas_done.is_ok(),
                    "post-order Cyan->Blue CAS must succeed when color_owned=true; got {:?}",
                    cas_done
                );
            }
        }
        Outcome::ShouldStop => { /* stop flag set; main loop will drain */ }
    }
}

/// Returns true if every worker is currently flagged idle.
fn all_idle(flags: &Arc<Vec<AtomicBool>>) -> bool {
    for f in flags.iter() {
        if !f.load(Ordering::Acquire) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fairness::FairnessConstraint;
    use crate::model::{ActionLabel, LabeledTransition};
    use crossbeam_channel::bounded;
    use serde::{Deserialize, Serialize};

    // Mirror `dfs_worker::tests::fresh_failpoint_scenario`: under the
    // failpoints feature, set up a fresh isolated FailScenario so any
    // failpoints another lib test set don't leak into the pool tests.
    // (The failpoints registry is process-global; without this, the
    // `fp_store_shard_full=return` from a sibling test fires inside our
    // fp_store inserts here and turns the dedup into a soft error.)
    #[cfg(feature = "failpoints")]
    fn fresh_failpoint_scenario() -> fail::FailScenario<'static> {
        fail::FailScenario::setup()
    }
    #[cfg(not(feature = "failpoints"))]
    fn fresh_failpoint_scenario() {}

    #[test]
    fn partition_for_fp_distributes_uniformly() {
        let n = 8usize;
        let mut counts = [0usize; 8];
        for fp in 0u64..10_000 {
            counts[partition_for_fp(fp, n)] += 1;
        }
        let avg = 10_000 / n;
        for c in counts.iter() {
            // Each partition should receive within +/- 25% of the mean.
            assert!(
                (*c as i64 - avg as i64).unsigned_abs() <= (avg / 4) as u64,
                "partition skew too large: {:?}",
                counts
            );
        }
    }

    #[test]
    fn partition_for_fp_single_worker_is_identity() {
        for fp in 0u64..1024 {
            assert_eq!(partition_for_fp(fp, 1), 0);
        }
    }

    // Toy fairness model used by the multi-worker correctness tests
    // below. Exercises both the local-successor path (when num_workers=1)
    // and the cross-partition routing path (when num_workers>1).
    #[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
    struct GridState {
        x: u8,
        y: u8,
    }

    struct Grid {
        dim: u8,
    }

    impl Model for Grid {
        type State = GridState;
        fn name(&self) -> &'static str {
            "dfs-pool-grid"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![GridState { x: 0, y: 0 }]
        }
        fn next_states(&self, s: &Self::State, out: &mut Vec<Self::State>) {
            if s.x + 1 < self.dim {
                out.push(GridState { x: s.x + 1, y: s.y });
            }
            if s.y + 1 < self.dim {
                out.push(GridState { x: s.x, y: s.y + 1 });
            }
            // Self-loop so every state has a non-trivial SCC.
            out.push(s.clone());
        }
        fn next_states_labeled(
            &self,
            s: &Self::State,
        ) -> Option<Vec<LabeledTransition<Self::State>>> {
            let mut out = vec![];
            if s.x + 1 < self.dim {
                out.push(LabeledTransition {
                    from: s.clone(),
                    to: GridState { x: s.x + 1, y: s.y },
                    action: ActionLabel { name: "Right".into(), disjunct_index: None },
                });
            }
            if s.y + 1 < self.dim {
                out.push(LabeledTransition {
                    from: s.clone(),
                    to: GridState { x: s.x, y: s.y + 1 },
                    action: ActionLabel { name: "Up".into(), disjunct_index: None },
                });
            }
            out.push(LabeledTransition {
                from: s.clone(),
                to: s.clone(),
                action: ActionLabel { name: "Right".into(), disjunct_index: None },
            });
            Some(out)
        }
        fn check_invariants(&self, _s: &Self::State) -> Result<(), String> {
            Ok(())
        }
        fn has_fairness_constraints(&self) -> bool {
            true
        }
        fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
            vec![FairnessConstraint::Weak {
                vars: vec!["x".into(), "y".into()],
                action: "Right".into(),
            }]
        }
        fn next_action_name(&self) -> Option<&str> {
            Some("Right")
        }
    }

    fn make_fp_store() -> Arc<UnifiedFingerprintStore> {
        use crate::storage::unified_fingerprint_store::UnifiedFingerprintConfig;
        let cfg = UnifiedFingerprintConfig {
            use_bloom: false,
            use_auto_switch: false,
            shard_count: 4,
            expected_items: 4096,
            false_positive_rate: 0.01,
            shard_size_mb: 2,
            num_numa_nodes: 1,
            auto_switch_config: None,
            backing_dir: None,
        };
        Arc::new(
            UnifiedFingerprintStore::new(cfg, &[None, None, None, None])
                .expect("fp store allocates"),
        )
    }

    fn make_color_map() -> Arc<PageAlignedColorMap> {
        Arc::new(
            PageAlignedColorMap::new(8192, 1, &[0]).expect("color map allocates"),
        )
    }

    fn run_pool(num_workers: usize, dim: u8) -> u64 {
        let model = Arc::new(Grid { dim });
        let fp_store = make_fp_store();
        let color_map = make_color_map();
        let stats = Arc::new(AtomicRunStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (vtx, _vrx) = bounded::<Violation<GridState>>(8);
        let (etx, _erx) = crossbeam_channel::unbounded();
        let verdict_done = Arc::new(AtomicBool::new(false));
        let ctx = DfsPoolCtx {
            num_workers,
            model,
            fp_store,
            color_map,
            stats: Arc::clone(&stats),
            stop,
            violation_tx: vtx,
            violation_count: Arc::new(AtomicUsize::new(0)),
            max_violations: 1,
            error_tx: etx,
            stop_on_violation: false,
            parent_map: None,
            state_map: None,
            trace_count: Arc::new(AtomicU64::new(0)),
            max_trace_states: 0,
            dfs_inband_verdict_done: verdict_done,
            cluster: None,
        };
        run_dfs_pool(ctx);
        let (_g, _p, distinct, _d, _e, _c) = stats.snapshot();
        distinct
    }

    // `serial_test::serial` ensures these tests don't run concurrently
    // with other lib tests that mutate the process-global failpoint
    // registry. The `fp_store_shard_full=return` failpoint set by
    // sibling chaos tests would otherwise short-circuit our fp_store
    // inserts and cause spurious distinct-count drift.
    #[test]
    #[serial_test::serial]
    fn pool_explores_grid_with_one_worker() {
        let _scenario = fresh_failpoint_scenario();
        let distinct = run_pool(1, 5);
        assert_eq!(distinct, 25, "5x5 grid has 25 reachable states");
    }

    #[test]
    #[serial_test::serial]
    fn pool_explores_grid_with_four_workers() {
        let _scenario = fresh_failpoint_scenario();
        let distinct = run_pool(4, 5);
        assert_eq!(distinct, 25, "5x5 grid has 25 reachable states with N=4");
    }

    #[test]
    #[serial_test::serial]
    fn pool_state_count_invariant_across_worker_counts() {
        let _scenario = fresh_failpoint_scenario();
        let baseline = run_pool(1, 6);
        for n in [2usize, 3, 4, 8].iter() {
            let counted = run_pool(*n, 6);
            assert_eq!(
                counted, baseline,
                "state count must be invariant under DFS worker count (N={}, baseline={}, got={})",
                n, baseline, counted
            );
        }
    }
}
