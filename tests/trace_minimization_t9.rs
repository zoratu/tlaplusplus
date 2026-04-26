// T9 — trace minimization on invariant violation.
//
// Two integration gates:
//
//   1. Phase A correctness on a TLA+ spec: starting from a needlessly
//      long trace (constructed by interleaving an unrelated action with
//      the violation-driving action), `minimize_trace` returns a trace
//      that is (a) no longer than the input, (b) starts at an initial
//      state, (c) is a valid sequence of `next_states` transitions, and
//      (d) ends at a state that violates the invariant.
//
//   2. Phase B (variable highlighting) on a real TLA+ invariant:
//      `extract_invariant_variables` correctly identifies that an
//      invariant `count < 5` references `count` but not the other
//      state variables.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use tlaplusplus::Model;
use tlaplusplus::extract_invariant_variables;
use tlaplusplus::minimize_trace;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::tla::TlaState;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

fn build_model(module_rel: &str, config_rel: &str) -> TlaModel {
    let module = workspace_path(module_rel);
    let config = workspace_path(config_rel);
    TlaModel::from_files(&module, Some(&config), None, None)
        .unwrap_or_else(|e| panic!("failed to build model {}: {e:#}", module_rel))
}

/// Find the initial state by name. The TraceMinimizationDiamond spec
/// has exactly one initial state.
fn unique_initial_state(model: &TlaModel) -> TlaState {
    let inits = model.initial_states();
    assert_eq!(
        inits.len(),
        1,
        "test fixture should have exactly one init state, got {}",
        inits.len()
    );
    inits[0].clone()
}

/// Greedy step: try each successor by name match. If `prefer` is Some
/// and that variable in the successor differs from the current state,
/// prefer that one; otherwise pick the first successor.
fn step_first(model: &TlaModel, state: &TlaState) -> TlaState {
    let mut buf = Vec::new();
    model.next_states(state, &mut buf);
    assert!(!buf.is_empty(), "no successors at {:?}", state);
    buf.into_iter().next().unwrap()
}

/// Greedy step: pick a successor that differs from the current state in
/// the named variable. Used to construct a "long" trace by interleaving
/// noise actions before driving the invariant-violating variable.
fn step_changing_var(model: &TlaModel, state: &TlaState, var: &str) -> Option<TlaState> {
    let mut buf = Vec::new();
    model.next_states(state, &mut buf);
    let cur = state.get(var).cloned();
    buf.into_iter().find(|s| s.get(var).cloned() != cur)
}

#[test]
fn phase_a_minimization_shortens_an_inflated_trace_on_real_tla_spec() {
    // The TraceMinimizationDiamond spec has three actions:
    //   - Tick: count' = count + 1 (drives the violation count >= 5)
    //   - Bump: noise' = noise + 1 (orthogonal noise)
    //   - SwapPhase: phase' = swap (orthogonal noise)
    //
    // We hand-construct a long trace that bumps noise twice and swaps
    // phase before driving count to 5. Minimization should discover the
    // direct path of just Tick five times.
    let model = build_model(
        "corpus/internals/TraceMinimizationDiamond.tla",
        "corpus/internals/TraceMinimizationDiamond.cfg",
    );

    let init = unique_initial_state(&model);

    // Step plan: count++ x5 (reaches violation at count=5), then keep
    // doing irrelevant noise steps that don't undo the violation. The
    // BFS-equivalent shortest violating prefix is just five ticks (6
    // states); minimization should find that earlier-violating state
    // already in the trace and truncate.
    let mut trace = Vec::new();
    trace.push(init.clone());

    // Five count ticks — final state of these is already a violation.
    for _ in 0..5 {
        let prev = trace.last().unwrap().clone();
        let next =
            step_changing_var(&model, &prev, "count").expect("count step should be available");
        trace.push(next);
    }
    let mid_violation_index = trace.len() - 1;
    assert!(model.check_invariants(&trace[mid_violation_index]).is_err());
    // Then do noise/phase bumps on top, post-violation. Each one is a
    // valid Next transition, but adds nothing to the violation story.
    for _ in 0..2 {
        let prev = trace.last().unwrap().clone();
        let next =
            step_changing_var(&model, &prev, "noise").expect("noise step should be available");
        trace.push(next);
    }
    {
        let prev = trace.last().unwrap().clone();
        let next =
            step_changing_var(&model, &prev, "phase").expect("phase step should be available");
        trace.push(next);
    }
    // Final state still violates the invariant (Inv = count < 5; count
    // is now 5, plus extra noise bumps that don't touch count).
    assert!(model.check_invariants(trace.last().unwrap()).is_err());

    // Sanity: the final state violates the invariant.
    let last = trace.last().unwrap();
    let inv_check = model.check_invariants(last);
    assert!(
        inv_check.is_err(),
        "constructed trace should end in violation, got {:?} for last={:?}",
        inv_check,
        last
    );

    let original_len = trace.len();
    eprintln!("constructed trace length = {}", original_len);

    // Run minimization with a generous budget — the spec is tiny.
    let result = minimize_trace(&model, trace, Duration::from_secs(10));

    eprintln!(
        "minimized: {} -> {} steps in {:?}, iterations={}, budget_exhausted={}",
        result.original_len,
        result.trace.len(),
        result.elapsed,
        result.iterations,
        result.budget_exhausted
    );

    // === Correctness gate 1: shorter or equal, and never longer ===
    assert!(
        result.trace.len() <= original_len,
        "minimization must never lengthen a trace: {} -> {}",
        original_len,
        result.trace.len()
    );

    // === Correctness gate 2: this fixture must actually shorten ===
    // The direct path is 6 states; the inflated path is 9 states.
    assert!(
        result.trace.len() < original_len,
        "expected strict shortening on diamond fixture: {} -> {}",
        original_len,
        result.trace.len()
    );

    // === Correctness gate 3: starts at an initial state ===
    let inits = model.initial_states();
    assert!(
        inits.contains(result.trace.first().unwrap()),
        "minimized trace must start at an initial state"
    );

    // === Correctness gate 4: every transition is valid Next ===
    let mut buf = Vec::new();
    for window in result.trace.windows(2) {
        buf.clear();
        model.next_states(&window[0], &mut buf);
        assert!(
            buf.contains(&window[1]),
            "minimized trace contains invalid transition\n  from: {:?}\n  to:   {:?}",
            window[0],
            window[1]
        );
    }

    // === Correctness gate 5: final state still violates the invariant ===
    let inv_check = model.check_invariants(result.trace.last().unwrap());
    assert!(
        inv_check.is_err(),
        "minimized trace must end in violation, got {:?}",
        inv_check
    );
}

#[test]
fn phase_a_returns_input_unchanged_when_already_optimal() {
    // BFS-shortest trace through the same spec. Five ticks, 6 states,
    // with no noise. Minimization should not be able to shorten.
    let model = build_model(
        "corpus/internals/TraceMinimizationDiamond.tla",
        "corpus/internals/TraceMinimizationDiamond.cfg",
    );

    let init = unique_initial_state(&model);
    let mut trace = vec![init];
    for _ in 0..5 {
        let prev = trace.last().unwrap().clone();
        let next =
            step_changing_var(&model, &prev, "count").expect("count step should be available");
        trace.push(next);
    }
    let original_len = trace.len();
    assert_eq!(original_len, 6);

    let result = minimize_trace(&model, trace.clone(), Duration::from_secs(5));

    // No shortening possible — minimum is 6.
    assert_eq!(
        result.trace.len(),
        6,
        "already-optimal trace must not be shortened: got {}",
        result.trace.len()
    );
    // Final state still violates.
    assert!(
        model
            .check_invariants(result.trace.last().unwrap())
            .is_err()
    );
}

#[test]
fn phase_b_extract_relevant_variables_from_inv_text() {
    // Pull the invariant body straight from the model and check that
    // Phase B identifies `count` as relevant and `noise`/`phase` as not.
    let model = build_model(
        "corpus/internals/TraceMinimizationDiamond.tla",
        "corpus/internals/TraceMinimizationDiamond.cfg",
    );

    let var_names: Vec<String> = model.module.variables.iter().cloned().collect();
    assert_eq!(
        var_names.iter().cloned().collect::<HashSet<_>>(),
        ["count", "noise", "phase"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );

    let inv_body = model
        .invariant_exprs
        .iter()
        .find(|(name, _)| name == "Inv")
        .map(|(_, body)| body.clone())
        .expect("Inv invariant should be resolved");

    let relevant = extract_invariant_variables(&inv_body, &var_names);
    assert!(
        relevant.contains("count"),
        "Inv = `{}` should reference `count`",
        inv_body
    );
    assert!(
        !relevant.contains("noise"),
        "Inv should not reference `noise`"
    );
    assert!(
        !relevant.contains("phase"),
        "Inv should not reference `phase`"
    );
}

/// PorViolation is a known-violating spec used elsewhere in the suite.
/// Run minimization end-to-end on a real BFS trace to confirm it
/// preserves the invariant-violation property even when no shortening
/// is possible.
#[test]
fn phase_a_on_real_known_violation_spec_preserves_violation() {
    let model = build_model(
        "corpus/internals/PorViolation.tla",
        "corpus/internals/PorViolation.cfg",
    );

    // Walk a deterministic path that drives count to 3 (the violation).
    let init = model.initial_states().into_iter().next().unwrap();
    let mut trace = vec![init];
    for _ in 0..3 {
        let prev = trace.last().unwrap().clone();
        // Tick deterministically increments count; pick the successor
        // where count grew.
        let next = step_changing_var(&model, &prev, "count")
            .expect("Tick should fire from any count<4 state");
        trace.push(next);
    }
    // Sanity: count is now 3, which violates Inv == count < 3.
    assert!(model.check_invariants(trace.last().unwrap()).is_err());

    let result = minimize_trace(&model, trace, Duration::from_secs(5));

    // Should remain valid violations.
    assert!(
        model
            .check_invariants(result.trace.last().unwrap())
            .is_err()
    );
    let inits = model.initial_states();
    assert!(inits.contains(result.trace.first().unwrap()));

    // Validate every transition.
    let mut buf = Vec::new();
    for window in result.trace.windows(2) {
        buf.clear();
        model.next_states(&window[0], &mut buf);
        assert!(
            buf.contains(&window[1]),
            "invalid transition in minimized trace"
        );
    }
}

/// Smoke test: a zero budget should not cause panics or invalid traces.
#[test]
fn phase_a_zero_budget_returns_safe_trace() {
    let model = build_model(
        "corpus/internals/TraceMinimizationDiamond.tla",
        "corpus/internals/TraceMinimizationDiamond.cfg",
    );

    let init = unique_initial_state(&model);
    let mut trace = vec![init.clone()];
    for _ in 0..3 {
        let prev = trace.last().unwrap().clone();
        if let Some(next) = step_changing_var(&model, &prev, "count") {
            trace.push(next);
        } else {
            trace.push(step_first(&model, &prev));
        }
    }
    while model.check_invariants(trace.last().unwrap()).is_ok() {
        let prev = trace.last().unwrap().clone();
        let next = step_changing_var(&model, &prev, "count");
        if let Some(n) = next {
            trace.push(n);
        } else {
            break;
        }
    }
    assert!(model.check_invariants(trace.last().unwrap()).is_err());

    let original_len = trace.len();
    let result = minimize_trace(&model, trace, Duration::from_nanos(0));

    // With zero budget we expect either no change or some lucky
    // shortening; either way the result is a valid violating trace and
    // never longer than the input.
    assert!(result.trace.len() <= original_len);
    assert!(
        model
            .check_invariants(result.trace.last().unwrap())
            .is_err()
    );
}
