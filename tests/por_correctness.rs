// T7 — partial-order reduction (POR) correctness gate.
//
// WHY THIS FILE EXISTS
// --------------------
// POR is a soundness-sensitive optimisation: a correct stubborn-set
// implementation MUST find every invariant violation that full enumeration
// finds, and the reduced reachable set MUST be a (typically strict) subset
// of the full reachable set.
//
// This file fixes both gates as integration tests:
//
//   1. **Reachable subset**: POR-reduced reachable state set ⊆ full
//      reachable state set (strict equality is fine — no reduction is also
//      correct).
//   2. **Violation parity**: For specs with an invariant violation, both
//      modes report the violation (and reach a violating state).
//
// We use small synthetic specs designed to exercise common POR patterns:
//   - Independent processes incrementing local counters (massive reduction).
//   - Two processes contending on a shared variable (no reduction expected).
//   - One process disabling/enabling another (dependency closure).

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use tlaplusplus::Model;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::tla::TlaState;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

/// Deterministic BFS reachable-set enumeration via the `Model` trait.
/// Dedupes by canonical-JSON repr (stable across builds).
fn enumerate_reachable(model: &TlaModel) -> BTreeSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<TlaState> = Vec::new();

    for s in model.initial_states() {
        let repr = serde_json::to_string(&s).expect("serialise");
        if seen.insert(repr) {
            frontier.push(s);
        }
    }

    let mut buf: Vec<TlaState> = Vec::new();
    while let Some(state) = frontier.pop() {
        buf.clear();
        model.next_states(&state, &mut buf);
        for next in buf.drain(..) {
            let repr = serde_json::to_string(&next).expect("serialise");
            if seen.insert(repr) {
                frontier.push(next);
            }
        }
    }

    seen.into_iter().collect()
}

fn build_model(module_rel: &str, config_rel: &str) -> TlaModel {
    let module = workspace_path(module_rel);
    let config = workspace_path(config_rel);
    TlaModel::from_files(&module, Some(&config), None, None)
        .unwrap_or_else(|e| panic!("failed to build model {}: {e:#}", module_rel))
}

/// Run the spec twice — once with full enumeration, once with POR — and
/// assert the POR-reduced state set is a subset of the full set.
fn assert_por_subset(module_rel: &str, config_rel: &str) -> (usize, usize) {
    let model_full = build_model(module_rel, config_rel);
    let full = enumerate_reachable(&model_full);

    let mut model_por = build_model(module_rel, config_rel);
    model_por
        .enable_por()
        .expect("POR should be enabled (no fairness/liveness in test specs)");
    let por = enumerate_reachable(&model_por);

    // Stubborn-set safety theorem: every state reached under POR is also
    // reached under full enumeration.
    let extra: Vec<&String> = por.difference(&full).collect();
    assert!(
        extra.is_empty(),
        "POR reached {} state(s) not in full reachable set:\n{}",
        extra.len(),
        extra
            .iter()
            .take(5)
            .map(|s| format!("    {}", s))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // POR set must contain every initial state at minimum.
    assert!(
        !por.is_empty(),
        "POR reachable set is empty (initial states should always survive)"
    );

    (full.len(), por.len())
}

#[test]
fn por_two_independent_counters_yields_reduction() {
    // Two independent processes each increment a local counter 0..N.  Full
    // exploration sees ~N^2 states; POR sees only N+1 (one process at a
    // time per equivalence class) — actually 2N+1 along the diagonal.
    let (full, por) = assert_por_subset(
        "corpus/internals/PorTwoCounters.tla",
        "corpus/internals/PorTwoCounters.cfg",
    );
    eprintln!("PorTwoCounters: full={} por={}", full, por);
    // Full = (N+1)^2 = 16 for N=3; POR <= full and < full (real reduction).
    assert_eq!(full, 16, "full state count mismatch");
    assert!(
        por < full,
        "expected POR to reduce state count (got por={}, full={})",
        por,
        full
    );
}

#[test]
fn por_shared_counter_no_reduction() {
    // Two actions both write the same variable.  They are dependent — POR
    // must fire both at every state.  POR reachable set == full reachable
    // set.
    let (full, por) = assert_por_subset(
        "corpus/internals/PorSharedCounter.tla",
        "corpus/internals/PorSharedCounter.cfg",
    );
    eprintln!("PorSharedCounter: full={} por={}", full, por);
    // Both actions touch x — so dependency makes them all stubborn.
    // POR doesn't help here, but must not reach extra states.
    assert_eq!(
        por, full,
        "POR with all-dependent actions should equal full set"
    );
}

#[test]
fn por_dependency_chain() {
    // Three actions: A, B, C.  A and B are independent, but B and C share
    // a variable.  Stubborn set seeded from B must include C.
    let (full, por) = assert_por_subset(
        "corpus/internals/PorDependencyChain.tla",
        "corpus/internals/PorDependencyChain.cfg",
    );
    eprintln!("PorDependencyChain: full={} por={}", full, por);
    // Subset property is the main correctness gate.  Reduction depends on
    // the seed selection — at minimum POR must not exceed full.
    assert!(por <= full);
}

#[test]
fn por_finds_invariant_violation() {
    // A spec with a known violation: two processes increment a counter,
    // invariant claims counter < 3.  Both full and POR must reach a state
    // violating the invariant.
    let model_full = build_model(
        "corpus/internals/PorViolation.tla",
        "corpus/internals/PorViolation.cfg",
    );
    let mut model_por = build_model(
        "corpus/internals/PorViolation.tla",
        "corpus/internals/PorViolation.cfg",
    );
    model_por.enable_por().expect("POR should enable");

    let full = enumerate_reachable(&model_full);
    let por = enumerate_reachable(&model_por);

    let violating_in = |set: &BTreeSet<String>| -> bool {
        for repr in set {
            // The violation is that `count >= 3` is reachable.  We re-parse
            // to check, since BTreeSet stores canonical JSON.
            let parsed: serde_json::Value = serde_json::from_str(repr).expect("parse");
            if let Some(count) = parsed
                .get("count")
                .and_then(|v| v.get("Int"))
                .and_then(|v| v.as_i64())
            {
                if count >= 3 {
                    return true;
                }
            }
        }
        false
    };

    assert!(
        violating_in(&full),
        "full enumeration must reach a state where count >= 3"
    );
    assert!(
        violating_in(&por),
        "POR must also reach a state where count >= 3 (preserves safety violations)"
    );
}

#[test]
fn por_rejected_with_liveness() {
    // POR opt-out: spec with a liveness property → enable_por must error.
    let mut model = build_model(
        "corpus/internals/PorLiveness.tla",
        "corpus/internals/PorLiveness.cfg",
    );
    let result = model.enable_por();
    assert!(
        result.is_err(),
        "POR must be rejected when liveness/temporal properties are present"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("liveness") || err.contains("temporal") || err.contains("fairness"),
        "error message should mention the limitation, got: {}",
        err
    );
}
