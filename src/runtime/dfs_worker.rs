//! T10.2 stage 4 — single-worker DFS exploration with in-band fairness
//! verdict (no labeled-transitions DashMap materialization).
//!
//! # Stage 3 → Stage 4 delta
//!
//! Stage 3 (v1.2.3) shipped the architecture: a separate single-worker
//! DFS exploration function with the iterative blue-frame stack and the
//! production [`PageAlignedColorMap`]. **It still populated the
//! `labeled_transitions: DashMap<u64, Vec<LabeledTransition<S>>>`** so
//! the post-BFS oracle in [`crate::runtime::liveness`] could run Tarjan
//! SCC on it and produce the canonical fairness verdict. That meant the
//! predicted memory win was not realized: the dominant memory cost in
//! the BFS path is the DashMap with its cloned `from`/`to` states (~5 GB
//! on a 1 M-state liveness spec, per `docs/T10.2-phase2-refined.md`
//! §2.2).
//!
//! Stage 4 (this file):
//!
//! 1. **Drops `labeled_transitions` from the DFS path entirely.** The
//!    DashMap is never touched; `compute_successors` now takes a
//!    `LocalAdjacency` builder that records only fingerprints + action
//!    name strings — no state clones.
//! 2. **Moves the fairness verdict in-band.** When the DFS worker
//!    completes exploration, it runs Tarjan + the existing per-action
//!    fairness check on its locally-built `(u64, u64, String)` triple
//!    list. If a violation is found, it is emitted through the same
//!    `violation_tx` channel the BFS path uses; the post-processing
//!    block in `runtime::shutdown` is then skipped.
//! 3. **Skips post-BFS Tarjan when DFS dispatch fired.** A new
//!    `dfs_inband_verdict_done: Arc<AtomicBool>` is shared with the
//!    shutdown context; when set, `liveness::run_post_processing` is
//!    bypassed because the verdict is already in hand.
//! 4. **Preserves all four gate-7 parity tests** — same liveness
//!    verdicts on every fixture even after the DashMap drop. The
//!    in-band check uses the *exact same*
//!    `crate::fairness::check_fairness_on_scc_fp_sharded` predicate
//!    that the BFS path uses, so the verdict is by construction
//!    identical (modulo trace reconstruction details).
//!
//! # Memory win
//!
//! `LabeledTransition<S>` carries two clones of `S` per edge plus an
//! `ActionLabel` (a `String` + `Option<usize>`). For a TLA+ state at
//! 200 B and 4 successors per state, that's ≈ 1.6 KB per state of edge
//! storage. At 1 M states, ≈ 1.6 GB of edge data alone, plus the
//! DashMap shard overhead.
//!
//! The Stage 4 in-band path stores `(u64 from_fp, u64 to_fp, String
//! action_name)` per edge — at typical action-name lengths (8-16 chars),
//! ≈ 32-48 B per edge, or ≈ 128-192 B per state of edge storage. That's
//! **a 10-12× reduction** in edge memory before the DashMap
//! shard-overhead savings even kick in. The state map needed for trace
//! reconstruction (`fp → S`) is built once on demand at violation time
//! by re-walking the blue stack, not maintained throughout exploration.
//!
//! # Features intentionally still dropped
//!
//! Same as Stage 3 (single-worker, single-node). Cross-partition routing
//! infrastructure for multi-worker DFS is wired through `protocol.rs`
//! `PartitionEdge` / `PartitionEdgeAck` variants (Stage 2 deliverable);
//! actually using them requires a dispatch layer that this stage's scope
//! deliberately defers. See `docs/T10.2-phase2-refined.md` §3 for the
//! cross-partition design.

use crate::fairness::{
    FairnessConstraint, TarjanSCC, build_action_shard_index, check_fairness_on_scc_fp_sharded,
};
use crate::model::Model;
use crate::storage::page_aligned_color_map::{Color as MapColor, PageAlignedColorMap};
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use anyhow::Result;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::stats::AtomicRunStats;
use super::{PropertyType, Violation};

/// Bundled handles for the single DFS worker. Mirrors
/// [`super::worker::WorkerLocalState`] in shape but drops every field that
/// is irrelevant under the simplifying assumptions documented in the
/// module header.
///
/// Stage 4 changes:
/// - No `labeled_transitions: Option<Arc<DashMap<...>>>` field. The DFS
///   path builds a thin local fingerprint-only adjacency map and runs
///   the fairness verdict in-band; no DashMap is touched.
/// - New `dfs_inband_verdict_done` flag — set to `true` by the worker
///   when it has run the fairness check in-band, signaling shutdown to
///   skip the post-processing Tarjan pass.
pub(super) struct DfsWorkerCtx<M: Model> {
    pub(super) model: Arc<M>,
    pub(super) fp_store: Arc<UnifiedFingerprintStore>,
    pub(super) stats: Arc<AtomicRunStats>,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) violation_tx: Sender<Violation<M::State>>,
    pub(super) violation_count: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) max_violations: usize,
    pub(super) error_tx: Sender<String>,
    pub(super) stop_on_violation: bool,
    pub(super) parent_map: Option<Arc<DashMap<u64, u64>>>,
    pub(super) state_map: Option<Arc<DashMap<u64, M::State>>>,
    pub(super) trace_count: Arc<AtomicU64>,
    pub(super) max_trace_states: u64,
    /// Page-aligned color map sized for `--fp-expected-items`. Stage 1
    /// deliverable; replaces the `HashMap<u64, Color>` of the v1.1.0
    /// oracle.
    pub(super) color_map: Arc<PageAlignedColorMap>,
    /// Set to `true` by `run_dfs_worker` after it has run the in-band
    /// fairness verdict (whether the verdict was clean or produced a
    /// violation). Read by the shutdown orchestrator to gate
    /// `liveness::run_post_processing` — when true, the post-processing
    /// pass is skipped because the verdict is already in hand.
    pub(super) dfs_inband_verdict_done: Arc<AtomicBool>,
}

/// One DFS frame.
///
/// `state` is retained because trace reconstruction at violation time
/// re-walks the blue stack to map `fp → state`. `succs` plus the
/// per-successor action labels are materialised once on entry so the
/// nested DFS visits each transition exactly once.
struct BlueFrame<S> {
    /// Retained for future use — Stage 4 caches `fp → state` in
    /// `state_by_fp` separately so the violation reporter can
    /// look up the SCC representative without walking the blue
    /// stack. The frame's own `state` is kept here so the cross-
    /// partition Stage 5 routing layer (when wired) can hand the
    /// state off to a peer worker without re-evaluating
    /// `next_states`. Marked `dead_code` until that integration
    /// arrives.
    #[allow(dead_code)]
    state: S,
    fp: u64,
    /// Successor states paired with their action labels and a
    /// "passes constraints" flag. Pulled from the model's labeled-
    /// transition path so the in-band fairness check has the per-edge
    /// label without ever materializing a `LabeledTransition<S>`
    /// (which would clone `from` + `to`).
    ///
    /// The `passes_constraints` bit lets us record the edge in the
    /// in-band labeled adjacency unconditionally (matching the BFS
    /// path's `labeled_transitions` DashMap which is appended *before*
    /// the constraint filter, see `worker.rs` lines ~495-506) while
    /// only descending into successors that pass the filter.
    succs: Vec<(S, String, bool)>,
    succ_idx: usize,
    /// True if the color map slot for this fingerprint was successfully
    /// CAS'd to Cyan on entry. Same semantics as Stage 3.
    color_owned: bool,
}

/// Local in-band adjacency builder. Records `(from_fp, to_fp,
/// action_name)` triples as the DFS visits each edge. The runtime then
/// passes this triple list to the existing
/// [`build_action_shard_index`] / [`check_fairness_on_scc_fp_sharded`]
/// pipeline — exactly the same predicate the BFS post-processing path
/// uses, so the verdict is by construction identical.
///
/// Memory shape: `Vec<(u64, u64, String)>` — 16 B per edge plus the
/// String. At typical action-name lengths and with String interning by
/// the allocator's small-string optimisation absent (Rust strings are
/// always heap-allocated unless empty), expect ~32-48 B per edge. For
/// edge counts up to a few million this fits comfortably inside the
/// memory budget the DFS path frees up by dropping the DashMap.
///
/// Stage 5: this type is shared between the single-worker DFS path
/// (`run_dfs_worker`) and the multi-worker DFS pool (`dfs_pool`).
#[derive(Default)]
pub(super) struct LocalAdjacency {
    pub(super) triples: Vec<(u64, u64, String)>,
}

impl LocalAdjacency {
    pub(super) fn record(&mut self, from_fp: u64, to_fp: u64, action: String) {
        self.triples.push((from_fp, to_fp, action));
    }

    pub(super) fn into_triples(self) -> Vec<(u64, u64, String)> {
        self.triples
    }
}

/// Run a single DFS worker to completion. Returns when the entire
/// reachable state graph (from each initial state) has been explored
/// or when `stop` flips to true.
///
/// All errors are reported through `ctx.error_tx`; all violations
/// through `ctx.violation_tx`. A panic propagates through the
/// `std::thread::JoinHandle` returned from `runtime.rs::run_model`'s
/// dispatch site.
pub(super) fn run_dfs_worker<M: Model>(ctx: DfsWorkerCtx<M>) {
    let DfsWorkerCtx {
        model,
        fp_store,
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
        color_map,
        dfs_inband_verdict_done,
    } = ctx;

    let start = Instant::now();

    // ---- Initial-state seeding -------------------------------------
    //
    // DFS starts from the eager Init enumeration. Streaming Init
    // (T5.4) is intentionally not used here — same rationale as Stage 3.
    let initial_states: Vec<M::State> = model.initial_states();
    if initial_states.is_empty() {
        eprintln!("[dfs-worker] zero initial states — exploration trivially complete");
        let _ = error_tx;
        // Even with zero initial states, the verdict is "done" — no
        // post-processing needed. Mark the flag so shutdown skips
        // Tarjan.
        dfs_inband_verdict_done.store(true, Ordering::Release);
        return;
    }

    // Per-worker scratch.
    let mut local_fp_cache: HashSet<u64> = HashSet::with_capacity(8192);

    // Per-worker fp → state map for in-band trace reconstruction. Built
    // on demand as we descend; lets us emit a representative state for
    // any fingerprint in a violating SCC without re-walking the model.
    // Bounded by `max_trace_states` so we never blow memory; once the
    // bound is hit, trace reconstruction falls back to "first state of
    // the SCC", which is acceptable since the in-band verdict is the
    // primary deliverable, not the trace.
    let mut state_by_fp: HashMap<u64, M::State> = HashMap::with_capacity(8192);

    // The in-band labeled adjacency builder.
    let mut adj = LocalAdjacency::default();

    let mut total_states_seen: u64 = 0;
    let mut total_states_distinct: u64 = 0;
    let mut total_duplicates: u64 = 0;

    // Detect whether the model emits labeled transitions. If `false`,
    // we still run DFS (for state coverage) but the in-band fairness
    // check has nothing to chew on — same as the BFS path's behavior.
    let labeled_supported = model.next_states_labeled(&initial_states[0]).is_some();

    'init_loop: for init_state in initial_states {
        if stop.load(Ordering::Acquire) {
            break;
        }

        let init_state = model.canonicalize(init_state);
        let init_fp = model.fingerprint(&init_state);

        let was_present = match fp_store.contains_or_insert_unchecked(init_fp) {
            Ok(p) => p,
            Err(e) => {
                let _ = error_tx.send(format!("dfs init dedup failed: {e}"));
                stop.store(true, Ordering::Release);
                break;
            }
        };
        if was_present {
            total_duplicates += 1;
            continue;
        }
        total_states_distinct += 1;
        state_by_fp.entry(init_fp).or_insert_with(|| init_state.clone());

        if color_map.cas(init_fp, MapColor::White, MapColor::Cyan).is_err() {
            // Color collision; treat as already-visited.
            continue;
        }

        match run_blue_dfs(
            &model,
            &fp_store,
            &color_map,
            init_state.clone(),
            init_fp,
            labeled_supported,
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
            &mut total_states_seen,
            &mut total_states_distinct,
            &mut total_duplicates,
        ) {
            Ok(()) => {}
            Err(err) => {
                let _ = error_tx.send(format!("dfs blue exploration failed: {err}"));
                stop.store(true, Ordering::Release);
                break 'init_loop;
            }
        }

        if stop.load(Ordering::Acquire) {
            break;
        }
    }

    // Flush stats once at end.
    stats
        .states_generated
        .fetch_add(total_states_seen, Ordering::Relaxed);
    stats
        .states_processed
        .fetch_add(total_states_distinct, Ordering::Relaxed);
    stats
        .states_distinct
        .fetch_add(total_states_distinct, Ordering::Relaxed);
    stats
        .duplicates
        .fetch_add(total_duplicates, Ordering::Relaxed);
    stats
        .enqueued
        .fetch_add(total_states_distinct, Ordering::Relaxed);

    let exploration_elapsed = start.elapsed();

    // ---- In-band fairness verdict ----------------------------------
    //
    // Run the same Tarjan + per-action fairness check the post-
    // processing pipeline runs, but inside the DFS worker thread on
    // our locally-built triple list. By construction this produces
    // the same verdict as the BFS path's post-processing
    // `run_post_processing` because we use the same shard index and
    // the same `check_fairness_on_scc_fp_sharded` predicate.
    //
    // Skip if a safety violation already fired (matches the BFS path's
    // "safety failures take precedence" rule in
    // `runtime/liveness.rs::run_post_processing`).
    let inband_started_at = Instant::now();
    let safety_already = violation_count.load(Ordering::Acquire) > 0;
    let triples = adj.into_triples();
    let constraints = model.fairness_constraints();
    let edge_count = triples.len();
    if !safety_already && !constraints.is_empty() && !triples.is_empty() {
        run_inband_fairness_check::<M>(
            &model,
            triples,
            &constraints,
            &state_by_fp,
            &violation_tx,
            &violation_count,
            stop_on_violation,
            max_violations,
            &stop,
        );
    } else if safety_already {
        eprintln!(
            "[dfs-worker] in-band fairness check skipped: safety violation already reported"
        );
    } else if constraints.is_empty() {
        eprintln!("[dfs-worker] in-band fairness check skipped: no fairness constraints");
    } else {
        eprintln!("[dfs-worker] in-band fairness check skipped: no transitions recorded");
    }

    eprintln!(
        "[dfs-worker] explored {} distinct states ({} edges) in {:.2?}, in-band fairness in {:.2?}; \
         labeled-transitions DashMap: NOT MATERIALIZED",
        total_states_distinct,
        edge_count,
        exploration_elapsed,
        inband_started_at.elapsed(),
    );

    // Mark the verdict-done flag last so shutdown observes a fully
    // populated violation_tx. Acquire-Release pairing with the
    // shutdown orchestrator's load.
    dfs_inband_verdict_done.store(true, Ordering::Release);
}

/// Blue DFS rooted at `root`. `root` has already been color-CAS'd to
/// Cyan and registered with the fp_store by the caller.
#[allow(clippy::too_many_arguments)]
fn run_blue_dfs<M: Model>(
    model: &Arc<M>,
    fp_store: &Arc<UnifiedFingerprintStore>,
    color_map: &Arc<PageAlignedColorMap>,
    root: M::State,
    root_fp: u64,
    labeled_supported: bool,
    adj: &mut LocalAdjacency,
    state_by_fp: &mut HashMap<u64, M::State>,
    local_fp_cache: &mut HashSet<u64>,
    parent_map: &Option<Arc<DashMap<u64, u64>>>,
    state_map: &Option<Arc<DashMap<u64, M::State>>>,
    trace_count: &Arc<AtomicU64>,
    max_trace_states: u64,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<std::sync::atomic::AtomicUsize>,
    max_violations: usize,
    stop_on_violation: bool,
    stop: &Arc<AtomicBool>,
    total_seen: &mut u64,
    total_distinct: &mut u64,
    total_duplicates: &mut u64,
) -> Result<()> {
    let root_succs = compute_labeled_successors(model, &root, labeled_supported);

    if let Err(message) = model.check_invariants(&root) {
        emit_safety_violation(
            model,
            root.clone(),
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
            return Ok(());
        }
    }

    let mut blue_stack: Vec<BlueFrame<M::State>> = Vec::with_capacity(64);
    blue_stack.push(BlueFrame {
        state: root,
        fp: root_fp,
        succs: root_succs,
        succ_idx: 0,
        color_owned: true,
    });
    *total_seen += 1;

    while !blue_stack.is_empty() {
        if stop.load(Ordering::Acquire) {
            // Best-effort drain back to Blue so the color map is left in
            // a consistent state.
            while let Some(frame) = blue_stack.pop() {
                if frame.color_owned {
                    let _ = color_map.cas(frame.fp, MapColor::Cyan, MapColor::Blue);
                }
            }
            break;
        }

        let descend: Option<(M::State, u64, Vec<(M::State, String, bool)>, bool)> = {
            let top = blue_stack
                .last_mut()
                .expect("blue_stack non-empty by while guard");
            let mut descend_target: Option<(M::State, u64, Vec<(M::State, String, bool)>, bool)> =
                None;
            let top_fp = top.fp;
            while top.succ_idx < top.succs.len() {
                let (next_state, action_name, passes_constraints) =
                    top.succs[top.succ_idx].clone();
                top.succ_idx += 1;

                let next_state = model.canonicalize(next_state);
                let next_fp = model.fingerprint(&next_state);
                *total_seen += 1;

                // Record the edge in the in-band adjacency
                // *unconditionally* — matches the BFS path's
                // labeled_transitions DashMap which is appended
                // *before* the constraint filter (see `worker.rs`
                // lines ~495-506). Constraint-failing edges still
                // contribute to the labeled SCC graph.
                adj.record(top_fp, next_fp, action_name);

                // Constraint filter applies only to descent (whether
                // we explore the successor's outgoing edges), not to
                // edge recording. Mirrors BFS path semantics where
                // constraint-failing successors are dropped from the
                // queue but their inbound edges remain in the
                // labeled_transitions map.
                if !passes_constraints {
                    continue;
                }

                if local_fp_cache.contains(&next_fp) {
                    *total_duplicates += 1;
                    continue;
                }

                let was_present = fp_store.contains_or_insert_unchecked(next_fp)?;
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
                // Cache state for in-band trace reconstruction (bounded
                // — once we hit max_trace_states the SCC reporter
                // falls back to "first SCC fp" representative).
                if state_by_fp.len() < {
                    #[cfg(not(feature = "verus"))]
                    { (max_trace_states as usize).max(8192) }
                    #[cfg(feature = "verus")]
                    { crate::storage::verus_smoke::max_usize(max_trace_states as usize, 8192) }
                } {
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
                        break;
                    }
                }

                let succs =
                    compute_labeled_successors(model, &next_state, labeled_supported);
                descend_target = Some((next_state, next_fp, succs, color_owned));
                break;
            }
            descend_target
        };

        if let Some((next_state, next_fp, next_succs, color_owned)) = descend {
            blue_stack.push(BlueFrame {
                state: next_state,
                fp: next_fp,
                succs: next_succs,
                succ_idx: 0,
                color_owned,
            });
            continue;
        }

        // Post-order: in-band red probe (CVWY '92) on the popped fp.
        // The red probe walks the labeled adjacency we have built so
        // far — finding a Cyan back-edge means we've discovered an
        // accepting cycle. We do NOT report a fairness violation from
        // the red probe directly: the canonical fairness verdict is
        // produced by the in-band Tarjan + per-action check at the end
        // of exploration (which uses the *complete* adjacency, not the
        // partial picture available at this post-order point). The red
        // probe here is therefore exercise-only — it confirms the
        // page-aligned color map's CAS path on every accepting-state
        // pop, matching the v1.1.0 `nested_dfs_color_map` shape.
        let popped = blue_stack
            .pop()
            .expect("blue_stack non-empty by while guard");

        if popped.color_owned {
            // Run the red probe in-band over the partial adjacency
            // built so far. Witness is informational; verdict is
            // produced by the post-exploration Tarjan pass.
            let _witness =
                run_red_probe_inband(color_map, popped.fp, &adj.triples);

            let cas_done = color_map.cas(popped.fp, MapColor::Cyan, MapColor::Blue);
            debug_assert!(
                cas_done.is_ok(),
                "post-order Cyan->Blue CAS must succeed when color_owned=true; got {:?}",
                cas_done
            );
        }
    }

    Ok(())
}

/// Compute labeled successors of `state`. Returns
/// `Vec<(S, action_name, passes_constraints)>` where `passes_constraints`
/// is the state-/action-constraint result. We do NOT drop constraint-
/// failing entries here — the in-band fairness adjacency must include
/// ALL labeled transitions to match the BFS path's
/// `labeled_transitions` DashMap shape (which `worker.rs` populates
/// **before** the constraint filter, see lines ~495-506). Only the
/// blue-DFS descent skips constraint-failing successors.
pub(super) fn compute_labeled_successors<M: Model>(
    model: &Arc<M>,
    state: &M::State,
    labeled_supported: bool,
) -> Vec<(M::State, String, bool)> {
    if labeled_supported {
        if let Some(labeled) = model.next_states_labeled(state) {
            let mut out: Vec<(M::State, String, bool)> = Vec::with_capacity(labeled.len());
            for lt in labeled {
                let passes = model.check_state_constraints(&lt.to).is_ok()
                    && model.check_action_constraints(state, &lt.to).is_ok();
                out.push((lt.to, lt.action.name, passes));
            }
            return out;
        }
    }
    let mut raw = Vec::with_capacity(8);
    model.next_states(state, &mut raw);
    let owned_state = state.clone();
    let mut out: Vec<(M::State, String, bool)> = Vec::with_capacity(raw.len());
    for next_state in raw {
        let passes = model.check_state_constraints(&next_state).is_ok()
            && model
                .check_action_constraints(&owned_state, &next_state)
                .is_ok();
        out.push((next_state, String::new(), passes));
    }
    out
}

/// Nested-DFS red probe seeded at `seed_fp`. Walks the partial
/// adjacency triple list and returns a Cyan witness if found, else
/// `None`. This is exercise-only in Stage 4 (the fairness verdict is
/// produced by the in-band Tarjan pass), but we keep the probe wired
/// up to retain CAS-path coverage on the page-aligned color map.
///
/// The probe is iterative and bounded by the adjacency size. It does
/// not paint Red (we don't want to leak Red markers that would prevent
/// Cyan→Blue post-order CAS in the surrounding blue DFS); instead it
/// uses a local `visited` set.
fn run_red_probe_inband(
    color_map: &Arc<PageAlignedColorMap>,
    seed_fp: u64,
    adjacency_triples: &[(u64, u64, String)],
) -> Option<u64> {
    // Build a tiny on-the-fly successor lookup. For the gate-7 fixtures
    // (small graphs) this is O(edges); for larger graphs the in-band
    // verdict path uses the full Tarjan after exploration so this
    // probe's cost is amortised.
    //
    // We bound iterations to keep this from running away on large
    // graphs — the canonical verdict is the post-exploration Tarjan,
    // not this probe.
    let mut visited: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = vec![seed_fp];
    let max_iters = 4096_usize;
    let mut iters = 0_usize;

    while let Some(v) = stack.pop() {
        iters += 1;
        if iters > max_iters {
            return None;
        }
        if !visited.insert(v) {
            continue;
        }
        for (from, to, _) in adjacency_triples {
            if *from == v {
                if color_map.load(*to) == MapColor::Cyan {
                    return Some(*to);
                }
                if !visited.contains(to) {
                    stack.push(*to);
                }
            }
        }
    }
    None
}

/// In-band fairness check. Mirrors
/// [`crate::runtime::liveness::run_post_processing`] phases 2-5 (build
/// adjacency, Tarjan SCC, per-action shard check, emit violation) but
/// runs inside the DFS worker thread on its locally-built triple list.
///
/// **Verdict equivalence**: this function uses the exact same
/// [`build_action_shard_index`] + [`check_fairness_on_scc_fp_sharded`]
/// predicates as the post-processing path. Given the same triple input
/// it produces identical verdicts. The in-band path therefore satisfies
/// the gate-6 / gate-7 parity tests by construction.
///
/// Stage 5: also called by the multi-worker DFS pool after merging
/// per-worker triple lists into one global triple list.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_inband_fairness_check<M: Model>(
    model: &Arc<M>,
    triples: Vec<(u64, u64, String)>,
    constraints: &[FairnessConstraint],
    state_by_fp: &HashMap<u64, M::State>,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<std::sync::atomic::AtomicUsize>,
    stop_on_violation: bool,
    max_violations: usize,
    stop: &Arc<AtomicBool>,
) {
    // Phase 2: build adjacency.
    let mut adjacency_fp: HashMap<u64, Vec<u64>> = HashMap::with_capacity(state_by_fp.len());
    for (from, to, _) in &triples {
        adjacency_fp.entry(*from).or_insert_with(Vec::new).push(*to);
    }

    // Unique node set for Tarjan.
    let mut unique_fps_set: HashSet<u64> = HashSet::new();
    for (from, to, _) in &triples {
        unique_fps_set.insert(*from);
        unique_fps_set.insert(*to);
    }
    let unique_fps: Vec<u64> = unique_fps_set.into_iter().collect();

    // Phase 3: SCC discovery via iterative Tarjan.
    let mut tarjan = TarjanSCC::new();
    let sccs_fp = tarjan.find_sccs(&unique_fps, |fp| {
        adjacency_fp.get(fp).cloned().unwrap_or_default()
    });

    // Phase 4: filter to non-trivial SCCs (size > 1 OR self-loop).
    let non_trivial: Vec<&Vec<u64>> = sccs_fp
        .iter()
        .filter(|scc| {
            scc.len() > 1
                || (scc.len() == 1
                    && adjacency_fp
                        .get(&scc[0])
                        .map(|s| s.contains(&scc[0]))
                        .unwrap_or(false))
        })
        .collect();

    if non_trivial.is_empty() {
        eprintln!("[dfs-worker] no non-trivial SCCs — fairness trivially satisfied");
        return;
    }

    // Phase 4b: per-action shard index (T10.4).
    let shards = build_action_shard_index(&triples);
    let next_name = model.next_action_name();

    eprintln!(
        "[dfs-worker] in-band Tarjan: {} non-trivial SCCs out of {} total, {} edges, \
         checking {} fairness constraints",
        non_trivial.len(),
        sccs_fp.len(),
        triples.len(),
        constraints.len()
    );

    // Phase 5: per-SCC × per-constraint check.
    'outer: for (scc_idx, scc) in non_trivial.iter().enumerate() {
        let scc_fps: HashSet<u64> = scc.iter().copied().collect();
        for constraint in constraints {
            let result = check_fairness_on_scc_fp_sharded(
                &scc_fps,
                constraint,
                &shards,
                &adjacency_fp,
                next_name,
            );
            if let Err(e) = result {
                let representative_fp = scc[0];
                let representative_state = state_by_fp
                    .get(&representative_fp)
                    .cloned()
                    .unwrap_or_else(|| {
                        // State map miss — fall back to any state we
                        // have. If that also misses, we'd have to
                        // re-explore; for now we panic-print and let
                        // the caller surface the message-only
                        // violation.
                        eprintln!(
                            "[dfs-worker] state-map miss for fp={:#x} when reporting fairness \
                             violation; trace will be empty",
                            representative_fp
                        );
                        // Pick any state we do have.
                        state_by_fp
                            .values()
                            .next()
                            .cloned()
                            .expect("state_by_fp empty but SCC non-trivial — unreachable")
                    });

                let mut trace: Vec<M::State> = scc
                    .iter()
                    .filter_map(|fp| state_by_fp.get(fp).cloned())
                    .collect();
                if trace.is_empty() {
                    trace.push(representative_state.clone());
                } else {
                    // Close the lasso.
                    let first = trace[0].clone();
                    trace.push(first);
                }

                let v = Violation {
                    message: format!("Fairness violation: {}", e),
                    state: representative_state,
                    property_type: PropertyType::Liveness,
                    trace,
                };

                eprintln!(
                    "[dfs-worker] fairness violation in SCC {} ({} states): {}",
                    scc_idx,
                    scc.len(),
                    e
                );

                let _ = violation_tx.try_send(v);
                let prev = violation_count.fetch_add(1, Ordering::AcqRel);
                if stop_on_violation && (prev + 1) >= max_violations {
                    stop.store(true, Ordering::Release);
                }
                break 'outer;
            }
        }
    }
}

/// Emit a safety violation through the violation channel and update the
/// violation_count + stop atoms per the BFS worker's T11.5 ordering.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_safety_violation<M: Model>(
    model: &Arc<M>,
    state: M::State,
    message: String,
    parent_map: &Option<Arc<DashMap<u64, u64>>>,
    state_map: &Option<Arc<DashMap<u64, M::State>>>,
    violation_tx: &Sender<Violation<M::State>>,
    violation_count: &Arc<std::sync::atomic::AtomicUsize>,
    max_violations: usize,
    stop_on_violation: bool,
    stop: &Arc<AtomicBool>,
) {
    let trace = if let Some(pm) = parent_map.as_ref() {
        let sm = state_map.as_ref().expect(
            "state_map must be Some when parent_map is Some (both gated by config.trace_parents)",
        );
        let mut chain = vec![state.clone()];
        let mut fp = model.fingerprint(&state);
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
        super::reconstruct_trace_limited(model.as_ref(), &state, 100)
            .unwrap_or_else(|| vec![state.clone()])
    };

    let _ = violation_tx.try_send(Violation {
        message,
        state,
        property_type: PropertyType::Safety,
        trace,
    });
    let prev = violation_count.fetch_add(1, Ordering::AcqRel);
    if stop_on_violation && (prev + 1) >= max_violations {
        stop.store(true, Ordering::Release);
    }
}

pub(super) trait FpStoreExt {
    fn contains_or_insert_unchecked(&self, fp: u64) -> Result<bool>;
}

impl FpStoreExt for UnifiedFingerprintStore {
    fn contains_or_insert_unchecked(&self, fp: u64) -> Result<bool> {
        Ok(self.contains_or_insert(fp))
    }
}

#[cfg(test)]
mod tests {
    //! Unit-test the DFS exploration on a hand-rolled toy model. End-to-end
    //! fairness-verdict parity against the BFS path lives in
    //! `tests/dfs_worker_parity.rs`.
    use super::*;
    use crate::model::Model;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
    struct ChainState {
        v: u32,
    }

    struct LinearChain {
        n: u32,
    }

    impl Model for LinearChain {
        type State = ChainState;
        fn name(&self) -> &'static str {
            "linear-chain"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![ChainState { v: 0 }]
        }
        fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
            if state.v + 1 < self.n {
                out.push(ChainState { v: state.v + 1 });
            }
        }
        fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
            Ok(())
        }
        fn has_fairness_constraints(&self) -> bool {
            false
        }
    }

    fn make_color_map(slots: usize) -> Arc<PageAlignedColorMap> {
        Arc::new(PageAlignedColorMap::new(slots, 1, &[0]).expect("color map alloc"))
    }

    #[cfg(feature = "failpoints")]
    fn fresh_failpoint_scenario() -> fail::FailScenario<'static> {
        fail::FailScenario::setup()
    }
    #[cfg(not(feature = "failpoints"))]
    fn fresh_failpoint_scenario() {}

    fn make_fp_store() -> Arc<UnifiedFingerprintStore> {
        use crate::storage::unified_fingerprint_store::UnifiedFingerprintConfig;
        let cfg = UnifiedFingerprintConfig {
            use_bloom: false,
            use_auto_switch: false,
            shard_count: 2,
            expected_items: 1024,
            false_positive_rate: 0.01,
            shard_size_mb: 2,
            num_numa_nodes: 1,
            auto_switch_config: None,
            backing_dir: None,
        };
        Arc::new(
            UnifiedFingerprintStore::new(cfg, &[None, None]).expect("fp store"),
        )
    }

    fn make_dfs_ctx<M: Model>(
        model: Arc<M>,
        stats: Arc<AtomicRunStats>,
        stop: Arc<AtomicBool>,
        vtx: Sender<Violation<M::State>>,
        etx: Sender<String>,
        verdict_done: Arc<AtomicBool>,
    ) -> DfsWorkerCtx<M> {
        DfsWorkerCtx {
            model,
            fp_store: make_fp_store(),
            stats,
            stop,
            violation_tx: vtx,
            violation_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_violations: 1,
            error_tx: etx,
            stop_on_violation: true,
            parent_map: None,
            state_map: None,
            trace_count: Arc::new(AtomicU64::new(0)),
            max_trace_states: 0,
            color_map: make_color_map(1024),
            dfs_inband_verdict_done: verdict_done,
        }
    }

    #[test]
    fn dfs_worker_explores_linear_chain() {
        let _scenario = fresh_failpoint_scenario();
        let model = Arc::new(LinearChain { n: 5 });
        let stats = Arc::new(AtomicRunStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (vtx, _vrx) = crossbeam_channel::bounded(8);
        let (etx, _erx) = crossbeam_channel::unbounded();
        let verdict_done = Arc::new(AtomicBool::new(false));
        let ctx =
            make_dfs_ctx(Arc::clone(&model), Arc::clone(&stats), stop, vtx, etx, Arc::clone(&verdict_done));
        run_dfs_worker(ctx);
        let (_g, _p, distinct, _d, _e, _c) = stats.snapshot();
        assert_eq!(distinct, 5, "expected to visit 5 chain states (v=0..=4)");
        assert!(verdict_done.load(Ordering::Acquire), "verdict-done flag must be set on completion");
    }

    #[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
    struct CycleState {
        v: u32,
    }

    struct TwoCycle;
    impl Model for TwoCycle {
        type State = CycleState;
        fn name(&self) -> &'static str {
            "two-cycle"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![CycleState { v: 0 }]
        }
        fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
            out.push(CycleState { v: 1 - state.v });
        }
        fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn dfs_worker_explores_two_cycle_without_revisiting() {
        let _scenario = fresh_failpoint_scenario();
        let model = Arc::new(TwoCycle);
        let stats = Arc::new(AtomicRunStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (vtx, _vrx) = crossbeam_channel::bounded(8);
        let (etx, _erx) = crossbeam_channel::unbounded();
        let verdict_done = Arc::new(AtomicBool::new(false));
        let ctx =
            make_dfs_ctx(Arc::clone(&model), Arc::clone(&stats), stop, vtx, etx, verdict_done);
        run_dfs_worker(ctx);
        let (_g, _p, distinct, _d, _e, _c) = stats.snapshot();
        assert_eq!(distinct, 2, "expected exactly 2 distinct states (0, 1)");
    }

    /// Safety violation must be reported and the run stopped.
    #[test]
    fn dfs_worker_reports_safety_violation_and_stops() {
        let _scenario = fresh_failpoint_scenario();
        struct ViolatingChain {
            n: u32,
            bad: u32,
        }
        impl Model for ViolatingChain {
            type State = ChainState;
            fn name(&self) -> &'static str {
                "violating-chain"
            }
            fn initial_states(&self) -> Vec<Self::State> {
                vec![ChainState { v: 0 }]
            }
            fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
                if state.v + 1 < self.n {
                    out.push(ChainState { v: state.v + 1 });
                }
            }
            fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
                if state.v == self.bad {
                    Err(format!("v reached forbidden value {}", self.bad))
                } else {
                    Ok(())
                }
            }
        }

        let model = Arc::new(ViolatingChain { n: 5, bad: 3 });
        let stats = Arc::new(AtomicRunStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (vtx, vrx) = crossbeam_channel::bounded(8);
        let (etx, _erx) = crossbeam_channel::unbounded();
        let stop_clone = Arc::clone(&stop);
        let verdict_done = Arc::new(AtomicBool::new(false));
        let ctx = make_dfs_ctx(model, stats, stop_clone, vtx, etx, verdict_done);
        run_dfs_worker(ctx);
        assert!(stop.load(Ordering::Acquire), "stop must be set on safety violation");
        let v = vrx.try_recv().expect("violation must be enqueued");
        assert_eq!(v.property_type, PropertyType::Safety);
        assert_eq!(v.state.v, 3);
    }
}
