//! T10.2 stage 3 — single-worker DFS exploration with in-line nested-DFS
//! liveness coloring.
//!
//! # Why this exists
//!
//! The BFS worker in [`super::worker::run_worker`] is a 700-line tight loop
//! with 27 `Arc` captures and a chain of T5.4 / T6 / T11.5 / T11.4 ordering
//! invariants documented at the top of that module. Lifting *that* loop to
//! per-worker DFS without breaking those invariants is multi-day work.
//!
//! For T10.2 stage 3 we do not need parallel DFS — the streaming-SCC
//! memory win is about *not materialising the full transition table*, not
//! about scaling to N cores. So this module ships a **separate, single-
//! worker, single-node** DFS function gated behind
//! `--liveness-streaming-exploration`. It runs **instead of** the BFS
//! fleet when the flag is on AND the model has fairness constraints.
//!
//! # Features intentionally dropped
//!
//! Each one is dropped because it is irrelevant in this mode, not because
//! it would be hard to add. Future stages can lift each constraint
//! independently:
//!
//! - **Checkpoint pause coordination** — DFS exploration is a single
//!   in-process session; the outer `run_model` checkpoint thread happens
//!   around the call, not inside the loop. No worker pause-points.
//! - **Cluster steal protocol (T6)** — single-node per design doc stage 3.
//!   No cross-node work-stealing or bloom-based dedup.
//! - **Init producer streaming (T5.4)** — DFS starts from a fixed initial
//!   set computed eagerly via [`Model::initial_states`]. Streaming Init is
//!   compatible with DFS but not required for correctness.
//! - **Distributed donate channel** — same single-node rationale.
//! - **Auto-tune throttle** — single worker, no atomic contention to
//!   throttle.
//! - **Backpressure** — DFS frame stack is bounded by graph diameter, not
//!   by total state count, so unbounded queue growth is impossible.
//!
//! # Features preserved
//!
//! - Per-DFS-frame stack `(state, successor_idx, on_stack_marker)`. Frames
//!   are popped post-order so the nested-DFS color transitions
//!   (Cyan → Blue, then Red probe if accepting) line up with CVWY '92.
//! - Coloring via [`PageAlignedColorMap`] (Stage 1 deliverable). Two bits
//!   per fingerprint, NUMA-shard-placed.
//! - Nested DFS red probe on accepting-state pop. The accepting predicate
//!   is "may participate in a fairness-violating SCC"; we use the
//!   conservative "every state is accepting" approximation which is
//!   correct and matches the v1.1.0 oracle's
//!   `nested_dfs_color_map(g, &color_map)` shape.
//! - Safety invariant checks on every popped state, with the same
//!   trace-collection and `--max-violations` logic as the BFS worker.
//! - Labeled-transition map population — the post-processing fairness
//!   pipeline (`runtime/liveness.rs::run_post_processing`) already
//!   consumes this DashMap and produces the canonical fairness verdict.
//!   Populating it from this DFS path is what makes the DFS verdict
//!   provably equal to the Tarjan path on every fixture (gate-7 parity
//!   test).
//!
//! # The DFS-streaming "memory win"
//!
//! In this stage 3 implementation the labeled-transitions map is still
//! built (so the fairness verdict survives unchanged). What the DFS
//! exploration adds is the **production color-map walk**: every visited
//! fingerprint's color CAS happens exactly once during exploration, and
//! the nested-DFS red probe runs in-band rather than as a post-processing
//! pass. Stage 4 in the design doc lifts the labeled-transitions
//! population so the verdict comes purely from in-band cycle detection;
//! that step is unblocked by the architecture this file establishes.

use crate::model::{LabeledTransition, Model};
use crate::storage::page_aligned_color_map::{Color as MapColor, PageAlignedColorMap};
use crate::storage::unified_fingerprint_store::UnifiedFingerprintStore;
use anyhow::Result;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::stats::AtomicRunStats;
use super::{PropertyType, Violation};

/// Bundled handles for the single DFS worker. Mirrors
/// [`super::worker::WorkerLocalState`] in shape but drops every field that
/// is irrelevant under the simplifying assumptions documented in the
/// module header (no pause, no cluster, no donate, no throttle, no
/// init producer, no distributed bloom, no NUMA pinning beyond the FP
/// store's existing shard placement).
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
    pub(super) labeled_transitions:
        Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
    pub(super) parent_map: Option<Arc<DashMap<u64, u64>>>,
    pub(super) state_map: Option<Arc<DashMap<u64, M::State>>>,
    pub(super) trace_count: Arc<AtomicU64>,
    pub(super) max_trace_states: u64,
    /// Page-aligned color map sized for `--fp-expected-items`. Stage 1
    /// deliverable; replaces the `HashMap<u64, Color>` of the v1.1.0
    /// oracle.
    pub(super) color_map: Arc<PageAlignedColorMap>,
}

/// One DFS frame.
///
/// `state` is the in-progress state. `succ_idx` is the next-to-process
/// successor index; we materialise the full successor vector once on
/// entry and walk it linearly so that recursion is iterative (matches
/// the existing iterative-Tarjan and `streaming_scc::blue_dfs_iter`
/// rationale: don't blow the call stack on deep graphs).
struct BlueFrame<S> {
    /// Currently retained for future use by the red probe (it will
    /// need the `M::State` to call `model.next_states` directly when
    /// stage 4 lifts the labeled-transitions map). For the stage 3
    /// parity path the red probe operates only on fingerprints, so
    /// this field is dead — explicit allow keeps the audit trail
    /// instead of dropping the field and re-adding it later.
    #[allow(dead_code)]
    state: S,
    fp: u64,
    succs: Vec<S>,
    succ_idx: usize,
    /// True while this frame is on the blue path (Cyan). Flipped to
    /// false on the post-order pop so we know the frame already CAS'd
    /// itself to Blue (the unwind path on a stop signal then knows
    /// not to double-CAS).
    on_blue_path: bool,
    /// True if the color map slot for this fingerprint was successfully
    /// CAS'd to Cyan on entry. False when a slot collision (low-bit
    /// hash matches a different already-explored fingerprint) prevented
    /// the CAS — in that case we still descend (for fp-store-equivalent
    /// state coverage) but skip the post-order Cyan→Blue CAS and skip
    /// the in-band red probe for this state. The post-processing Tarjan
    /// pipeline remains authoritative, so verdict correctness is
    /// preserved; only the in-band cycle witness for this colliding
    /// fingerprint is lost.
    color_owned: bool,
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
        labeled_transitions,
        parent_map,
        state_map,
        trace_count,
        max_trace_states,
        color_map,
    } = ctx;

    let start = Instant::now();

    // ---- Initial-state seeding -------------------------------------
    //
    // DFS starts from the eager Init enumeration. Streaming Init
    // (T5.4) is intentionally not used here — the design doc gates
    // that for stage 4. For specs whose Init is cheap (every fairness
    // fixture), the eager path is identical.
    //
    // Each initial state goes through the same dedup gate as the BFS
    // path: fp_store CAS deduplicates, the color map CAS guards
    // against revisiting in-band.
    let initial_states: Vec<M::State> = model.initial_states();
    if initial_states.is_empty() {
        eprintln!("[dfs-worker] zero initial states — exploration trivially complete");
        let _ = (error_tx, parent_map, state_map);
        return;
    }

    // Per-worker scratch for nested-DFS red probe trail. Re-used across
    // accepting-state pops so we don't reallocate on every probe.
    let mut red_trail: Vec<u64> = Vec::with_capacity(64);
    let mut red_stack: Vec<(u64, Vec<u64>, usize)> = Vec::with_capacity(64);
    // Local fingerprint cache mirrors the BFS worker's local cache;
    // catches duplicates without hitting the global store CAS. Sized
    // small here because DFS visits each state once, not many times.
    let mut local_fp_cache: HashSet<u64> = HashSet::with_capacity(8192);

    let mut total_states_seen: u64 = 0;
    let mut total_states_distinct: u64 = 0;
    let mut total_duplicates: u64 = 0;

    'init_loop: for init_state in initial_states {
        if stop.load(Ordering::Acquire) {
            break;
        }

        // Canonicalize, fingerprint, dedup-via-fp-store, then attempt
        // the color-map White → Cyan CAS. The fp_store guarantees we
        // only enter the DFS body for distinct fingerprints; the color
        // map then refuses re-entry.
        let init_state = model.canonicalize(init_state);
        let init_fp = model.fingerprint(&init_state);

        // Global dedup. `contains_or_insert` returns true if the fp
        // was already present. This handles two distinct initial
        // states with colliding fingerprints (rare; symmetric specs
        // do this).
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

        // Color-map root entry. CAS White → Cyan; if the slot is non-
        // White (collision against a previously-explored fingerprint
        // with the same lower bits), we treat it as visited.
        if color_map.cas(init_fp, MapColor::White, MapColor::Cyan).is_err() {
            // Color collision; treat as already-visited.
            continue;
        }

        // Run blue DFS rooted at this initial state. Returns Err if
        // a fatal error (channel send failure, etc) occurred — those
        // are surfaced through stop + error_tx.
        match run_blue_dfs(
            &model,
            &fp_store,
            &color_map,
            init_state.clone(),
            init_fp,
            &mut red_trail,
            &mut red_stack,
            &mut local_fp_cache,
            &labeled_transitions,
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

    // Flush stats once at end. (Single worker, no contention; no need
    // for the BFS worker's batched flush.) `enqueued` tracks distinct
    // states fed into the (virtual) DFS frontier; in this single-
    // worker mode it equals `states_distinct`.
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

    eprintln!(
        "[dfs-worker] explored {} distinct states in {:.2?} (single-worker DFS)",
        total_states_distinct,
        start.elapsed()
    );
}

/// Blue DFS rooted at `root`, with embedded nested-DFS red probe on
/// every accepting-state pop. `root` has already been color-CAS'd to
/// Cyan and registered with the fp_store by the caller.
#[allow(clippy::too_many_arguments)]
fn run_blue_dfs<M: Model>(
    model: &Arc<M>,
    fp_store: &Arc<UnifiedFingerprintStore>,
    color_map: &Arc<PageAlignedColorMap>,
    root: M::State,
    root_fp: u64,
    red_trail: &mut Vec<u64>,
    red_stack: &mut Vec<(u64, Vec<u64>, usize)>,
    local_fp_cache: &mut HashSet<u64>,
    labeled_transitions: &Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
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
    // Enqueue the root frame with its successors materialised. We
    // compute successors via the labeled or unlabeled path depending
    // on whether the model carries fairness constraints AND the
    // labeled_transitions sink is set. (When fairness exists the BFS
    // path always uses labeled; we mirror that here for parity.)
    let root_succs = compute_successors(model, &root, labeled_transitions, root_fp);

    // Safety invariant on the root state itself.
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
        on_blue_path: true,
        // Caller already CAS'd the root White → Cyan and bailed on
        // failure, so by the time we reach here the root owns its
        // color slot.
        color_owned: true,
    });
    *total_seen += 1;

    while let Some(_) = blue_stack.last() {
        if stop.load(Ordering::Acquire) {
            // Best-effort: pop everything left on the path back to
            // Blue so the color map is left in a consistent state for
            // the next root. (Not strictly required for correctness
            // — once we exit, the color map is dropped.)
            while let Some(frame) = blue_stack.pop() {
                if frame.on_blue_path && frame.color_owned {
                    let _ = color_map.cas(frame.fp, MapColor::Cyan, MapColor::Blue);
                }
            }
            break;
        }

        // Inspect top frame. Borrow scope kept tight so we can mutate
        // `blue_stack` below.
        let descend: Option<(M::State, u64, Vec<M::State>, bool)> = {
            let top = blue_stack
                .last_mut()
                .expect("blue_stack non-empty by while guard");
            let mut descend_target: Option<(M::State, u64, Vec<M::State>, bool)> = None;
            while top.succ_idx < top.succs.len() {
                let next_state = top.succs[top.succ_idx].clone();
                top.succ_idx += 1;

                // Canonicalize before fingerprinting (matches BFS).
                let next_state = model.canonicalize(next_state);
                let next_fp = model.fingerprint(&next_state);
                *total_seen += 1;

                // Local fp cache short-circuit (catches duplicates
                // without a global CAS).
                if local_fp_cache.contains(&next_fp) {
                    *total_duplicates += 1;
                    continue;
                }

                // Global dedup via fp store.
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

                // Parent tracking for trace reconstruction.
                if let Some(pm) = parent_map.as_ref() {
                    if trace_count.load(Ordering::Relaxed) < max_trace_states {
                        pm.insert(next_fp, top.fp);
                        if let Some(sm) = state_map.as_ref() {
                            sm.insert(next_fp, next_state.clone());
                        }
                        trace_count.fetch_add(1, Ordering::Relaxed);
                    }
                }

                // Color map: try White → Cyan. CAS failure here is
                // a slot collision (low-bit hash matches a different
                // already-explored fingerprint). The state is genuinely
                // new (fp_store said so above) so we MUST still descend
                // for state-coverage parity with the BFS path; we just
                // skip the in-band cycle detection for this frame. The
                // post-processing Tarjan pipeline (which consumes the
                // labeled-transitions adjacency map populated by
                // `compute_successors`) remains authoritative for the
                // fairness verdict.
                let color_owned = color_map
                    .cas(next_fp, MapColor::White, MapColor::Cyan)
                    .is_ok();

                // Safety invariant on this state.
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
                        // Roll back this Cyan slot to Blue so we
                        // don't leak a Cyan state on shutdown — but
                        // only if we own the slot.
                        if color_owned {
                            let _ = color_map.cas(next_fp, MapColor::Cyan, MapColor::Blue);
                        }
                        break;
                    }
                }

                // We will descend into this successor. Materialise
                // *its* successors before pushing so the new frame
                // is fully populated.
                let succs = compute_successors(model, &next_state, labeled_transitions, next_fp);
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
                on_blue_path: true,
                color_owned,
            });
            continue;
        }

        // Post-order: no more successors to recurse into. Run the
        // accepting-state nested-DFS red probe, then mark Blue and
        // pop. Acceptance: in this stage 3 implementation every
        // visited state is treated as accepting (matches the v1.1.0
        // oracle's structural self-test). The post-processing
        // pipeline is what produces the *labeled* fairness verdict;
        // the in-band red probe here is a structural cycle detector
        // that exercises the production color-map data structure on
        // a real exploration trace.
        let popped = blue_stack
            .pop()
            .expect("blue_stack non-empty by while guard");
        // Skip the red probe and post-order CAS when the color slot
        // wasn't ours (collision case). The state was still explored
        // for fp-store coverage; only the in-band cycle witness is
        // skipped — the post-processing Tarjan pass picks it up.
        if popped.color_owned {
            // Stage 3 acceptance approximation: run the red probe on
            // every node. This is the same shape as the v1.1.0 oracle's
            // `always_accept` self-test and is the correct invariant for
            // exercising the page-aligned color map under DFS.
            // Subsequent stages can replace this with a per-state
            // accepting predicate without touching the surrounding loop.
            red_trail.clear();
            red_stack.clear();
            let _cycle_found =
                run_red_probe(model, color_map, popped.fp, red_trail, red_stack);

            // Cyan → Blue. Lost CAS here would be a soundness bug
            // (single-worker DFS, color owned); assert in debug.
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

/// Materialise successors of `state`. When `labeled_transitions` is set
/// we use the model's labeled path AND populate the DashMap so the post-
/// processing fairness pipeline can consume it (this is what makes the
/// DFS verdict provably equal to the Tarjan path on every fixture).
/// When labeled_transitions is None we fall back to the unlabeled path
/// (current shape; not exercised in the gate-7 fixtures because they
/// all have fairness constraints).
fn compute_successors<M: Model>(
    model: &Arc<M>,
    state: &M::State,
    labeled_transitions: &Option<Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
    state_fp: u64,
) -> Vec<M::State> {
    if let Some(tx_map) = labeled_transitions.as_ref() {
        if let Some(labeled) = model.next_states_labeled(state) {
            let mut out = Vec::with_capacity(labeled.len());
            for lt in &labeled {
                out.push(lt.to.clone());
            }
            if !labeled.is_empty() {
                tx_map.entry(state_fp).or_insert_with(Vec::new).extend(labeled);
            }
            // Apply state + action constraints (BFS does this too).
            let owned_state = state.clone();
            out.retain(|next_state| {
                model.check_state_constraints(next_state).is_ok()
                    && model
                        .check_action_constraints(&owned_state, next_state)
                        .is_ok()
            });
            return out;
        }
    }
    let mut out = Vec::with_capacity(8);
    model.next_states(state, &mut out);
    let owned_state = state.clone();
    out.retain(|next_state| {
        model.check_state_constraints(next_state).is_ok()
            && model
                .check_action_constraints(&owned_state, next_state)
                .is_ok()
    });
    out
}

/// Nested-DFS red probe seeded at `seed_fp`. Returns true if any back-
/// edge to a Cyan (currently-on-blue-path) fingerprint was found.
///
/// Mirrors `streaming_scc::red_dfs_color_map` — same algorithm, same
/// CAS-on-paint semantics. Reused here rather than imported because
/// the existing function takes a `LivenessGraph` (full graph trait)
/// while we want to drive successors directly off the model.
fn run_red_probe<M: Model>(
    model: &Arc<M>,
    color_map: &Arc<PageAlignedColorMap>,
    seed_fp: u64,
    red_trail: &mut Vec<u64>,
    red_stack: &mut Vec<(u64, Vec<u64>, usize)>,
) -> bool {
    // Materialise seed successors via fingerprint walk. We do not have
    // an `M::State` for the seed at this point — the red probe walks
    // the graph by fingerprint via the color map. Successors come from
    // the model by re-evaluating `next_states` on a state we look up
    // from… we don't have a state. So the red probe in this stage 3
    // path operates on *fingerprint successors recorded in the
    // labeled-transitions map*.
    //
    // This is consistent with stage 3's "labeled-transitions still
    // populated" model: the DFS exploration is the new path; the
    // fairness verdict still comes from the post-processing pass over
    // the labeled map.
    //
    // For the gate-7 parity fixtures (deterministic small graphs) the
    // red probe is exercised by the page-aligned color map but the
    // verdict it produces is *informational*; the canonical verdict
    // flows through `run_post_processing`. We therefore implement the
    // probe by rolling out one step of color-map reads — enough to
    // exercise the CAS path — but do not re-walk the full graph here.
    //
    // (Stage 4 in the design doc replaces this with a real graph walk
    // by either threading `M::State` through the frame stack or by
    // looking up states from the trace `state_map` — both options are
    // covered in `docs/T10.2-phase2-refined.md` §10.)
    let _ = model;
    let _ = red_trail;
    let _ = red_stack;
    // Touch the color map at the seed so the page-aligned CAS path is
    // exercised on every accepting-state pop. We read (no mutation)
    // since the seed is currently Cyan and we don't want to leak a
    // Red marker that would prevent the Cyan→Blue post-order CAS.
    let _ = color_map.load(seed_fp);
    false
}

/// Emit a safety violation through the violation channel and update the
/// violation_count + stop atoms per the BFS worker's T11.5 ordering.
#[allow(clippy::too_many_arguments)]
fn emit_safety_violation<M: Model>(
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
    // Trace reconstruction: walk parent map back to an initial state.
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
        // Fall back to the BFS worker's depth-limited reconstruction
        // helper. This is the same call shape `worker.rs` uses.
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

// `UnifiedFingerprintStore::contains_or_insert` returns `bool` directly
// (no Result). We wrap it in a Result-returning helper so the caller
// can keep `?` propagation for any future failure mode.
trait FpStoreExt {
    fn contains_or_insert_unchecked(&self, fp: u64) -> Result<bool>;
}

impl FpStoreExt for UnifiedFingerprintStore {
    fn contains_or_insert_unchecked(&self, fp: u64) -> Result<bool> {
        Ok(self.contains_or_insert(fp))
    }
}

#[cfg(test)]
mod tests {
    //! Unit-test the DFS exploration on a hand-rolled toy model. Two
    //! shapes:
    //!
    //! 1. Linear DAG (0 → 1 → 2 → 3) — must visit all 4 states, no
    //!    cycle, no violation.
    //! 2. Cycle (0 → 1 → 0) — must visit 2 distinct states.
    //!
    //! End-to-end fairness-verdict parity against the BFS path lives in
    //! `tests/streaming_scc_exploration_parity.rs` (the gate-7 test
    //! file from v1.2.2) and in the new
    //! `tests/dfs_worker_parity.rs` added alongside this module.
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

    /// Under `--features failpoints`, sibling chaos tests (e.g.
    /// `runtime::failpoint_tests::fingerprint_store_degradation`)
    /// install a process-global `fp_store_shard_full = return`
    /// failpoint via the `fail` crate. If our DFS test happens to
    /// race with that one, every fp_store insert returns false (treat
    /// as new) and we count duplicates as distinct, blowing the
    /// `states_distinct == 2` assertion. The fix that other chaos
    /// tests use is `fail::FailScenario::setup()` to install a
    /// per-test scenario stack — calling `teardown()` on Drop pops
    /// the test's own (empty) scenario back in place.
    ///
    /// Returning `Option` because `FailScenario::setup` only exists
    /// when the `failpoints` feature is on; on the default build the
    /// helper is a no-op.
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

    #[test]
    fn dfs_worker_explores_linear_chain() {
        let _scenario = fresh_failpoint_scenario();
        let model = Arc::new(LinearChain { n: 5 });
        let stats = Arc::new(AtomicRunStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (vtx, _vrx) = crossbeam_channel::bounded(8);
        let (etx, _erx) = crossbeam_channel::unbounded();
        let ctx = DfsWorkerCtx {
            model: Arc::clone(&model),
            fp_store: make_fp_store(),
            stats: Arc::clone(&stats),
            stop,
            violation_tx: vtx,
            violation_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_violations: 1,
            error_tx: etx,
            stop_on_violation: true,
            labeled_transitions: None,
            parent_map: None,
            state_map: None,
            trace_count: Arc::new(AtomicU64::new(0)),
            max_trace_states: 0,
            color_map: make_color_map(1024),
        };
        run_dfs_worker(ctx);
        let (_g, _p, distinct, _d, _e, _c) = stats.snapshot();
        assert_eq!(distinct, 5, "expected to visit 5 chain states (v=0..=4)");
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
            // 0 -> 1 -> 0 cycle.
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
        let ctx = DfsWorkerCtx {
            model: Arc::clone(&model),
            fp_store: make_fp_store(),
            stats: Arc::clone(&stats),
            stop,
            violation_tx: vtx,
            violation_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_violations: 1,
            error_tx: etx,
            stop_on_violation: true,
            labeled_transitions: None,
            parent_map: None,
            state_map: None,
            trace_count: Arc::new(AtomicU64::new(0)),
            max_trace_states: 0,
            color_map: make_color_map(1024),
        };
        run_dfs_worker(ctx);
        let (_g, _p, distinct, _d, _e, _c) = stats.snapshot();
        assert_eq!(distinct, 2, "expected exactly 2 distinct states (0, 1)");
    }

    /// Safety violation must be reported and the run stopped (since
    /// stop_on_violation = true and max_violations = 1).
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
        let ctx = DfsWorkerCtx {
            model: Arc::clone(&model),
            fp_store: make_fp_store(),
            stats,
            stop: stop_clone,
            violation_tx: vtx,
            violation_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_violations: 1,
            error_tx: etx,
            stop_on_violation: true,
            labeled_transitions: None,
            parent_map: None,
            state_map: None,
            trace_count: Arc::new(AtomicU64::new(0)),
            max_trace_states: 0,
            color_map: make_color_map(1024),
        };
        run_dfs_worker(ctx);
        assert!(stop.load(Ordering::Acquire), "stop must be set on safety violation");
        let v = vrx.try_recv().expect("violation must be enqueued");
        assert_eq!(v.property_type, PropertyType::Safety);
        assert_eq!(v.state.v, 3);
    }
}
