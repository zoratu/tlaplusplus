//! Post-BFS fairness / liveness pipeline (T10 + T10.1-T10.4 + T10.2 oracle).
//!
//! Runs **after** all workers join. Reads the `labeled_transitions`
//! `DashMap` (populated by workers via `next_states_labeled` during
//! exploration) and:
//!
//! 1. T10.1: parallel-flatten dashmap → triples + state_by_fp via raw
//!    shard locks (rayon `par_iter` over `transitions_map.shards()`).
//!    Falls back to a serial walk for graphs under the threshold.
//! 2. Build adjacency_fp `HashMap<u64, Vec<u64>>`.
//! 3. T10.3: trivial-SCC pre-filter (heuristic-gated).
//! 4. Iterative Tarjan SCC over the candidate set.
//! 5. T10.4: per-action shard index built from `tx_triples`; each
//!    fairness check iterates only its own action's edges.
//! 6. For each non-trivial SCC × each constraint, run
//!    `check_fairness_on_scc_fp_sharded`. First failure produces a
//!    `Violation { property_type: Liveness, .. }`.
//! 7. T10.2 (opt-in): nested-DFS streaming-SCC oracle cross-validates
//!    Tarjan's verdict and emits a divergence diagnostic on disagreement.
//!
//! Returns the discovered fairness violation (or `None` if all SCCs
//! satisfy every constraint).

use crate::fairness::{
    TarjanSCC, build_action_shard_index, check_fairness_on_scc_fp_sharded, trivial_scc_prefilter,
};
use crate::model::{LabeledTransition, Model};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{EngineConfig, PropertyType, Violation};

/// Run T10 liveness post-processing on the labeled transition graph
/// produced by the BFS exploration phase.
///
/// `existing_violation` is `true` when the safety phase already produced
/// a violation; in that case fairness checking is skipped to preserve
/// the original ordering ("safety failures take precedence").
pub(super) fn run_post_processing<M>(
    existing_violation: bool,
    model: &Arc<M>,
    labeled_transitions: Option<&Arc<DashMap<u64, Vec<LabeledTransition<M::State>>>>>,
    config: &EngineConfig,
) -> Option<Violation<M::State>>
where
    M: Model,
{
    if existing_violation {
        return None;
    }
    let transitions_map = labeled_transitions?;

    let liveness_started_at = std::time::Instant::now();
    eprintln!(
        "Checking fairness constraints on {} states with transitions...",
        transitions_map.len()
    );

    // Phase 1 (T10.1) — flatten dashmap → triples + state map.
    //
    // Strategy: serial walk over the dashmap (its iterator is not
    // Send so a true parallel iter would have to materialize
    // first, which on benchmarks costs more than it saves on
    // 4-core boxes). Within the walk we partition into chunks
    // and run **fingerprint hashing in parallel** over each
    // chunk before doing the (cheap) sequential map insertion.
    //
    // For very large graphs the chunked-parallel hashing
    // amortizes worker spin-up. For small graphs the serial
    // fast-path is taken and overhead is zero.
    //
    // The 1024-entry threshold on `transitions_map.len()` was
    // tuned on `LivenessBench` N=8 (32 768 states / 143 361
    // edges, 4 cores aarch64): below ~1K dashmap entries the
    // chunk dispatch cost exceeds the parallel-hash win;
    // above it the parallel hash starts to pay.
    let phase1_start = std::time::Instant::now();
    use rayon::prelude::*;

    // Estimate: total transitions ≈ entries * 4-5 (typical
    // labelled-Next per state). Use a threshold on entries so
    // we don't dispatch rayon for tiny graphs.
    //
    // Tuning: on a 4-core aarch64 (LivenessBench), the parallel
    // path needs to amortize a ~100 ms `dashmap → owned-Vec`
    // materialization step (rayon can't iterate dashmap in
    // place because the iter holds shard locks and isn't Send).
    // Below ~80K entries the materialization cost dominates;
    // above it the parallel-hash win starts to pay. Tune on
    // your model — for >>4-core boxes the threshold could be
    // dropped further.
    let parallel_flatten = transitions_map.len() >= 80_000;

    let (tx_triples, state_by_fp, total_tx) = if parallel_flatten {
        // T10.1 parallel flatten via dashmap raw-api shards.
        //
        // Each dashmap shard is a `RwLock<HashMap<K, SharedValue<V>>>`.
        // We acquire each shard's read lock on a separate rayon
        // thread, walk its entries, and produce per-shard
        // (triples, state_map). Final merge is sequential — but
        // the state map is by-fingerprint, so each shard's
        // unique fingerprints are nearly disjoint (collisions
        // only happen on inter-shard duplicate states).
        //
        // We never materialize Vec<LabeledTransition> chunks —
        // entries are read directly from the shard's locked
        // hashmap, the only clone happening is the State clone
        // for the state_by_fp map. This avoids the ~330ms
        // dashmap → owned-Vec materialization step that the
        // previous fallback path incurred on N=10.
        let model_ref = model;
        let shards = transitions_map.shards();
        use dashmap::SharedValue;
        let per_shard: Vec<(Vec<(u64, u64, String)>, HashMap<u64, M::State>, usize)> =
            shards
                .par_iter()
                .map(|shard_lock| {
                    let guard = shard_lock.read();
                    let mut local_triples: Vec<(u64, u64, String)> =
                        Vec::with_capacity(guard.len() * 4);
                    let mut local_states: HashMap<u64, M::State> =
                        HashMap::with_capacity(guard.len());
                    let mut local_count = 0usize;
                    // Iterate raw buckets of the underlying
                    // hashbrown::raw::RawTable. Each bucket holds
                    // &(K, SharedValue<V>); we deref via
                    // bucket.as_ref(). This is the public
                    // raw-api dashmap path used by their own
                    // OwningIter.
                    unsafe {
                        let raw_iter = guard.iter();
                        for bucket in raw_iter {
                            let (_k, shared_val): &(
                                u64,
                                SharedValue<Vec<LabeledTransition<M::State>>>,
                            ) = bucket.as_ref();
                            let txs = shared_val.get();
                            for trans in txs {
                                local_count += 1;
                                let from_fp = model_ref.fingerprint(&trans.from);
                                let to_fp = model_ref.fingerprint(&trans.to);
                                local_states
                                    .entry(from_fp)
                                    .or_insert_with(|| trans.from.clone());
                                local_states
                                    .entry(to_fp)
                                    .or_insert_with(|| trans.to.clone());
                                local_triples.push((
                                    from_fp,
                                    to_fp,
                                    trans.action.name.clone(),
                                ));
                            }
                        }
                    }
                    (local_triples, local_states, local_count)
                })
                .collect();

        // Sequential merge: append triples, fold state maps.
        let total_tx_cap: usize = per_shard.iter().map(|(t, _, _)| t.len()).sum();
        let unique_cap: usize =
            per_shard.iter().map(|(_, s, _)| s.len()).max().unwrap_or(0);
        let mut tx_triples_local: Vec<(u64, u64, String)> =
            Vec::with_capacity(total_tx_cap);
        let mut state_by_fp_local: HashMap<u64, M::State> =
            HashMap::with_capacity(unique_cap.max(64));
        let mut total_tx_local = 0usize;
        for (mut triples, states, count) in per_shard {
            total_tx_local += count;
            tx_triples_local.append(&mut triples);
            // Merge: first-write-wins. Insert from each shard's
            // state map into the global; duplicates collapse.
            for (k, v) in states {
                state_by_fp_local.entry(k).or_insert(v);
            }
        }

        eprintln!(
            "  Total transitions collected: {} (flatten: {:.2?}, parallel via {} shards)",
            total_tx_local,
            phase1_start.elapsed(),
            shards.len()
        );
        (tx_triples_local, state_by_fp_local, total_tx_local)
    } else {
        // Serial fast path for small graphs (<1024 entries).
        let mut state_by_fp_local: HashMap<u64, M::State> = HashMap::new();
        let mut tx_triples_local: Vec<(u64, u64, String)> = Vec::new();
        let mut total_tx_local = 0usize;
        for entry in transitions_map.iter() {
            for trans in entry.value() {
                total_tx_local += 1;
                let from_fp = model.fingerprint(&trans.from);
                let to_fp = model.fingerprint(&trans.to);
                state_by_fp_local
                    .entry(from_fp)
                    .or_insert_with(|| trans.from.clone());
                state_by_fp_local
                    .entry(to_fp)
                    .or_insert_with(|| trans.to.clone());
                tx_triples_local.push((from_fp, to_fp, trans.action.name.clone()));
            }
        }
        eprintln!(
            "  Total transitions collected: {} (flatten: {:.2?}, serial)",
            total_tx_local,
            phase1_start.elapsed()
        );
        (tx_triples_local, state_by_fp_local, total_tx_local)
    };

    let mut violation: Option<Violation<M::State>> = None;

    if !tx_triples.is_empty() {
        eprintln!("  Unique states in graph: {}", state_by_fp.len());

        // Phase 2: build adjacency in fingerprint space.
        let phase2_start = std::time::Instant::now();
        let mut adjacency_fp: HashMap<u64, Vec<u64>> =
            HashMap::with_capacity(state_by_fp.len());
        for &(from, to, _) in &tx_triples {
            adjacency_fp.entry(from).or_insert_with(Vec::new).push(to);
        }
        let unique_fps: Vec<u64> = state_by_fp.keys().copied().collect();
        eprintln!(
            "  Adjacency built in {:.2?} ({} edges, {} nodes)",
            phase2_start.elapsed(),
            total_tx,
            unique_fps.len()
        );

        // Phase 2b (T10.3): trivial-SCC pre-filter. Iteratively
        // peel sources/sinks (no in/out edges, no self-loop) so
        // that Tarjan's input is the cycle skeleton only. For
        // very-DAG-shaped specs (most nodes are sources or
        // sinks) trim leaves nothing for Tarjan; for one-giant-
        // SCC specs trim leaves everything (and trim is a
        // no-op).
        //
        // Heuristic: skip trim by default. On benchmarks, the
        // O(N + E) reverse-adjacency build that trim needs is
        // comparable to the entire Tarjan pass on the same
        // graph — even on pure DAGs where Tarjan would just
        // walk-then-pop every node. Trim only pays when the
        // graph is large AND extremely sparse AND mostly DAG-
        // shaped. We probe for sinks to detect the DAG case;
        // even then, we only trim if avg out-degree is below
        // 2 (suggests tree-like). The function is exposed as
        // `fairness::trivial_scc_prefilter` for callers that
        // know their workload benefits from it.
        //
        // Empirically on LivenessBench (one giant SCC, avg-
        // outdeg 4.4) trim is correctly skipped. On DagBench
        // N=12 (avg-outdeg 3.7), Tarjan-on-DAG (46 ms) beats
        // trim (95 ms). The break-even point is well above
        // typical liveness specs, so the safer default is to
        // keep trim off and only enable it if the heuristic
        // is confident it will pay.
        let trim_start = std::time::Instant::now();
        let avg_outdeg = if unique_fps.is_empty() {
            0.0
        } else {
            total_tx as f64 / unique_fps.len() as f64
        };
        let try_trim = if avg_outdeg < 2.0 && unique_fps.len() >= 4096 {
            // Sparse + nontrivial size: probe for sinks to
            // confirm DAG-shape before paying trim cost.
            let mut found_sink = false;
            for &v in unique_fps.iter().take(1024) {
                if adjacency_fp.get(&v).map(|s| s.is_empty()).unwrap_or(true) {
                    found_sink = true;
                    break;
                }
            }
            found_sink
        } else {
            false
        };

        let (candidate_fps, candidate_set, trimmed_any) = if try_trim {
            let (candidates, trimmed_any) =
                trivial_scc_prefilter(&unique_fps, &adjacency_fp);
            if trimmed_any {
                let cf: Vec<u64> = candidates.iter().copied().collect();
                (cf, candidates, true)
            } else {
                (unique_fps.clone(), HashSet::new(), false)
            }
        } else {
            (unique_fps.clone(), HashSet::new(), false)
        };
        eprintln!(
            "  Trivial-SCC pre-filter: {} → {} candidate nodes ({:.2?}; avg-outdeg {:.2}, trim {})",
            unique_fps.len(),
            candidate_fps.len(),
            trim_start.elapsed(),
            avg_outdeg,
            if try_trim { "ran" } else { "skipped" }
        );

        // Phase 3: SCC discovery via iterative Tarjan over the
        // candidate nodes. When trim ran, the successor function
        // filters out trimmed nodes (those can't be in any
        // non-trivial SCC). When trim didn't run, the successor
        // function returns the full successor list directly.
        let phase3_start = std::time::Instant::now();
        let mut tarjan = TarjanSCC::new();
        let sccs_fp = if trimmed_any {
            tarjan.find_sccs(&candidate_fps, |fp| {
                adjacency_fp
                    .get(fp)
                    .map(|succs| {
                        succs
                            .iter()
                            .copied()
                            .filter(|s| candidate_set.contains(s))
                            .collect::<Vec<u64>>()
                    })
                    .unwrap_or_default()
            })
        } else {
            tarjan.find_sccs(&candidate_fps, |fp| {
                adjacency_fp.get(fp).cloned().unwrap_or_default()
            })
        };
        eprintln!(
            "  Found {} strongly connected components in {:.2?}",
            sccs_fp.len(),
            phase3_start.elapsed()
        );

        // Phase 4: filter to non-trivial SCCs (size > 1 OR self-loop).
        let phase4_start = std::time::Instant::now();
        let non_trivial_sccs: Vec<&Vec<u64>> = sccs_fp
            .iter()
            .filter(|scc| {
                scc.len() > 1
                    || (scc.len() == 1 && {
                        let fp = scc[0];
                        adjacency_fp
                            .get(&fp)
                            .map(|succs| succs.contains(&fp))
                            .unwrap_or(false)
                    })
            })
            .collect();
        eprintln!(
            "  Non-trivial SCCs: {} (filter: {:.2?})",
            non_trivial_sccs.len(),
            phase4_start.elapsed()
        );

        if !non_trivial_sccs.is_empty() {
            let constraints = model.fairness_constraints();

            if constraints.is_empty() {
                eprintln!("  No fairness constraints to check");
            } else {
                // Phase 4b (T10.4): build per-action shard index
                // so each fairness check only iterates its own
                // action's edges. The wrapper-Next path goes
                // through the adjacency map (any in-SCC edge).
                let shard_start = std::time::Instant::now();
                let shards = build_action_shard_index(&tx_triples);
                eprintln!(
                    "  Per-action shard index: {} actions ({:.2?})",
                    shards.len(),
                    shard_start.elapsed()
                );

                eprintln!(
                    "  Checking {} fairness constraints against {} SCCs",
                    constraints.len(),
                    non_trivial_sccs.len()
                );

                let phase5_start = std::time::Instant::now();
                let next_name = model.next_action_name();
                let mut total_constraint_checks = 0usize;

                'outer: for (scc_idx, scc) in non_trivial_sccs.iter().enumerate() {
                    // Build the SCC fingerprint set once (reused for
                    // every constraint).
                    let scc_fps: HashSet<u64> = scc.iter().copied().collect();

                    for constraint in &constraints {
                        total_constraint_checks += 1;
                        let result = check_fairness_on_scc_fp_sharded(
                            &scc_fps,
                            constraint,
                            &shards,
                            &adjacency_fp,
                            next_name,
                        );
                        if let Err(e) = result {
                            let scc_states: Vec<M::State> = scc
                                .iter()
                                .map(|fp| {
                                    state_by_fp
                                        .get(fp)
                                        .expect("SCC fp must be in state map")
                                        .clone()
                                })
                                .collect();

                            eprintln!(
                                "  Fairness violation in SCC {} ({} states): {}",
                                scc_idx,
                                scc_states.len(),
                                e
                            );

                            let representative = scc_states
                                .first()
                                .cloned()
                                .expect("non-trivial SCC has at least one state");

                            let mut trace = scc_states.clone();
                            if let Some(first) = trace.first().cloned() {
                                trace.push(first);
                            }

                            violation = Some(Violation {
                                message: format!("Fairness violation: {}", e),
                                state: representative,
                                property_type: PropertyType::Liveness,
                                trace,
                            });

                            break 'outer;
                        }
                    }
                }

                eprintln!(
                    "  Fairness check: {} constraint-on-SCC checks in {:.2?}",
                    total_constraint_checks,
                    phase5_start.elapsed()
                );

                if violation.is_none() {
                    eprintln!("  All fairness constraints satisfied");
                }

                // T10.2 — streaming-SCC oracle (opt-in via
                // `--liveness-streaming`). Run nested-DFS over
                // the same fingerprint adjacency map and
                // cross-validate against the Tarjan-based result
                // already computed above. This proves the
                // streaming algorithm is correct on real specs
                // before we lift the DFS into the worker hot
                // loop. See `docs/T10.2-streaming-scc-design.md`.
                if config.liveness_streaming {
                    let stream_started_at = std::time::Instant::now();
                    // Acceptance predicate for the oracle: a
                    // state is "accepting" if it participates
                    // in a non-trivial SCC AND the wrapper-Next
                    // fairness constraint (if present) is the
                    // *only* constraint OR no in-SCC edge bears
                    // the constraint's action label. For the
                    // initial validation pass we use a strict
                    // approximation: every node is "accepting"
                    // iff it lies in a known fairness-violating
                    // SCC (computed from the Tarjan path).
                    //
                    // This means: streaming nested-DFS will
                    // report a cycle iff the Tarjan fairness
                    // check already reported one. Any mismatch
                    // is an algorithmic bug and triggers a
                    // diagnostic (not a runtime abort — we keep
                    // the Tarjan-based result authoritative
                    // for now).
                    let constraints = model.fairness_constraints();
                    let next_name = model.next_action_name();
                    // Compute the set of in-SCC fingerprints
                    // for SCCs that violate at least one
                    // constraint. Acceptance := membership in
                    // this set.
                    let mut accepting_set: HashSet<u64> = HashSet::new();
                    // Need to rebuild shards (cheap) since the
                    // outer scope dropped them once `violation`
                    // was set.
                    let oracle_shards =
                        crate::fairness::build_action_shard_index(&tx_triples);
                    for scc in non_trivial_sccs.iter() {
                        let scc_fps: HashSet<u64> = scc.iter().copied().collect();
                        let mut violated = false;
                        for constraint in &constraints {
                            if crate::fairness::check_fairness_on_scc_fp_sharded(
                                &scc_fps,
                                constraint,
                                &oracle_shards,
                                &adjacency_fp,
                                next_name,
                            )
                            .is_err()
                            {
                                violated = true;
                                break;
                            }
                        }
                        if violated {
                            for fp in scc_fps {
                                accepting_set.insert(fp);
                            }
                        }
                    }

                    // Initial states for the oracle: in absence
                    // of a per-fingerprint init list at this
                    // post-processing point, seed with every
                    // node in `state_by_fp`. The DFS dedup via
                    // colors makes this O(V+E) regardless.
                    let init_fps: Vec<u64> =
                        state_by_fp.keys().copied().collect();
                    let accepting = move |fp: u64| accepting_set.contains(&fp);
                    let oracle_graph = crate::streaming_scc::FingerprintAdjacencyGraph {
                        initial_fps: init_fps,
                        adjacency: &adjacency_fp,
                        accepting: &accepting,
                    };
                    let oracle_result =
                        crate::streaming_scc::nested_dfs(&oracle_graph);
                    let oracle_found_cycle = matches!(
                        oracle_result,
                        crate::streaming_scc::NestedDfsResult::AcceptingCycle { .. }
                    );
                    let tarjan_found_cycle = violation.is_some();

                    eprintln!(
                        "  [T10.2 oracle] streaming nested-DFS in {:.2?} — \
                         oracle_cycle={} tarjan_cycle={}",
                        stream_started_at.elapsed(),
                        oracle_found_cycle,
                        tarjan_found_cycle
                    );
                    if oracle_found_cycle != tarjan_found_cycle {
                        eprintln!(
                            "  [T10.2 oracle] DIVERGENCE — streaming and \
                             Tarjan disagree. This indicates a bug in the \
                             nested-DFS implementation; please file an issue. \
                             The Tarjan result is authoritative."
                        );
                    }

                    // Independent self-test: with `accepting = true`
                    // for all states, nested-DFS must find a cycle
                    // iff a non-trivial SCC exists. This is a
                    // structural sanity check that doesn't depend
                    // on Tarjan's verdict — a divergence here
                    // would mean the DFS itself is broken (e.g.
                    // missed back-edge detection).
                    let always_accept = |_: u64| true;
                    let always_graph = crate::streaming_scc::FingerprintAdjacencyGraph {
                        initial_fps: state_by_fp.keys().copied().collect(),
                        adjacency: &adjacency_fp,
                        accepting: &always_accept,
                    };
                    let always_result =
                        crate::streaming_scc::nested_dfs(&always_graph);
                    let always_found =
                        matches!(
                            always_result,
                            crate::streaming_scc::NestedDfsResult::AcceptingCycle { .. }
                        );
                    let any_nontrivial_scc = !non_trivial_sccs.is_empty();
                    if always_found != any_nontrivial_scc {
                        eprintln!(
                            "  [T10.2 oracle] STRUCTURAL DIVERGENCE — \
                             nested-DFS with always-accept reports cycle={} \
                             but Tarjan finds {} non-trivial SCC(s). The \
                             nested-DFS algorithm itself is broken.",
                            always_found,
                            non_trivial_sccs.len()
                        );
                    } else {
                        eprintln!(
                            "  [T10.2 oracle] structural self-test OK \
                             (always-accept cycle={}, non-trivial-SCCs={})",
                            always_found,
                            non_trivial_sccs.len()
                        );
                    }
                }
            }
        } else {
            eprintln!("  No cycles detected - fairness constraints trivially satisfied");
        }
        eprintln!(
            "  Liveness post-processing total: {:.2?}",
            liveness_started_at.elapsed()
        );
    }

    violation
}
