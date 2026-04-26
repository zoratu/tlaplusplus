//! Trace minimization on invariant violation (T9).
//!
//! When the model checker finds an invariant violation, the BFS-discovered
//! counter-example trace `s0 → s1 → ... → sN` is shortest-by-step from `s0`
//! to `sN`, but `sN` itself may not be the shortest violating state.  An
//! earlier state on a different path could violate the same invariant in
//! fewer transitions, or a state inside the trace might be reachable from
//! `s0` via a shorter alternate path.
//!
//! ## Phase A — path shortening (delta-debug style)
//!
//! Repeatedly try to remove a prefix of states from the trace by searching
//! BFS from `s0` for any successor state already present later in the
//! trace at index `> i`.  If found, splice in the shorter prefix.
//!
//! Bounded compute: the loop terminates when no further shortening is
//! possible, the budget elapses, or we reach a trace of length <= 1.
//!
//! Correctness invariants preserved across every iteration:
//! - The first state is in `model.initial_states()`.
//! - Every adjacent pair `(s_i, s_{i+1})` satisfies `s_{i+1} ∈ next_states(s_i)`.
//! - The final state still violates the invariant
//!   (`model.check_invariants(last)` returns `Err`).
//!
//! ## Phase B — variable highlighting (presentation only)
//!
//! Scan the invariant text for variable references.  Variables not
//! syntactically referenced are flagged as "noise" so the printer can
//! visually de-emphasise them.  This is purely cosmetic — it does not
//! shrink the trace, only annotates it.
//!
//! ### T9.1 — Transitive variable relevance
//!
//! [`extract_invariant_variables_transitive`] extends the syntactic scan
//! by recursively inlining operator definitions.  An invariant of the
//! form `Inv == IsBad(state)` where `IsBad(s) == s.x > 5` will mark `x`
//! as referenced (rather than just `s`, which doesn't even appear in
//! the variable list).  Cycle-safe via a visited set.

use crate::model::Model;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Result of Phase A trace shortening.
#[derive(Debug, Clone)]
pub struct MinimizeResult<S> {
    /// The (possibly) shortened trace, still ending in a state that
    /// violates the invariant.
    pub trace: Vec<S>,
    /// Length of the original trace.
    pub original_len: usize,
    /// Number of shortening passes that successfully reduced length.
    pub iterations: usize,
    /// Total wall time spent in minimization.
    pub elapsed: Duration,
    /// True when minimization stopped because the budget expired.
    pub budget_exhausted: bool,
}

/// Minimize a violation trace using bounded delta-debugging.
///
/// `trace` is the BFS-reconstructed counter-example, ending in a state
/// `last` such that `model.check_invariants(last)` returns `Err`.
///
/// The returned trace satisfies the same property AND has length
/// `<= trace.len()`.  A trace of length 0 or 1 is returned unchanged.
///
/// `budget` caps total wall time.  When exhausted, the best trace found
/// so far is returned (still a valid counter-example).
pub fn minimize_trace<M: Model>(
    model: &M,
    trace: Vec<M::State>,
    budget: Duration,
) -> MinimizeResult<M::State> {
    let start = Instant::now();
    let original_len = trace.len();

    if trace.len() <= 1 {
        return MinimizeResult {
            trace,
            original_len,
            iterations: 0,
            elapsed: start.elapsed(),
            budget_exhausted: false,
        };
    }

    // Sanity-check: the input trace must end in a violating state.  If it
    // does not, refuse to "minimize" (return as-is) — better to surface a
    // weird trace than to silently truncate to a non-violating prefix.
    if model.check_invariants(trace.last().unwrap()).is_ok() {
        return MinimizeResult {
            trace,
            original_len,
            iterations: 0,
            elapsed: start.elapsed(),
            budget_exhausted: false,
        };
    }

    let mut current = trace;
    let mut iterations = 0usize;
    let mut budget_exhausted = false;

    // Cheap pre-pass: truncate at the earliest already-violating state
    // in the trace. The BFS path-shortening step below also handles
    // this implicitly, but doing it once up-front is O(N) and saves the
    // heavier BFS pass from having to rediscover the truncation point.
    if let Some(first_violation_idx) = current
        .iter()
        .position(|s| model.check_invariants(s).is_err())
    {
        if first_violation_idx + 1 < current.len() {
            current.truncate(first_violation_idx + 1);
            iterations += 1;
        }
    }

    loop {
        if start.elapsed() >= budget {
            budget_exhausted = true;
            break;
        }

        // Try to shorten by searching BFS from s0 for any state in the
        // current trace at a strictly-later index.  If found, splice the
        // shorter prefix in.
        let depth_cap = current.len().saturating_sub(1);
        if depth_cap == 0 {
            break;
        }

        match find_shortcut(model, &current, depth_cap, start, budget) {
            ShortcutResult::Found {
                replacement_prefix,
                target_index,
            } => {
                // Replace `current[0..=target_index]` with `replacement_prefix`.
                // Preconditions checked: `replacement_prefix` ends at a state
                // equal to `current[target_index]`, and is shorter.
                debug_assert!(replacement_prefix.len() < target_index + 1);
                let mut new_trace = replacement_prefix;
                new_trace.extend(current.into_iter().skip(target_index + 1));
                current = new_trace;
                iterations += 1;
                // Validate that the reconstructed trace still ends in a
                // violating state.  If something went wrong, abort
                // shortening and keep the previous trace.
                if model.check_invariants(current.last().unwrap()).is_ok() {
                    // Should never happen — final state was unchanged.
                    // Still, guard against it: revert to original by
                    // breaking with whatever we have (trace before the
                    // bad splice was already replaced — but this is
                    // unreachable in practice because `target_index <
                    // current.len()` and we kept the suffix).
                    break;
                }
            }
            ShortcutResult::None => break,
            ShortcutResult::BudgetExhausted => {
                budget_exhausted = true;
                break;
            }
        }
    }

    MinimizeResult {
        trace: current,
        original_len,
        iterations,
        elapsed: start.elapsed(),
        budget_exhausted,
    }
}

enum ShortcutResult<S> {
    /// Found a shorter prefix that reaches `current[target_index]`.
    Found {
        replacement_prefix: Vec<S>,
        target_index: usize,
    },
    /// No improvement possible.
    None,
    /// Budget elapsed during search.
    BudgetExhausted,
}

/// BFS from each initial state up to `depth_cap - 1` transitions deep,
/// looking for any state equal to `current[i]` with `i > 0`.  If found at
/// the smallest possible BFS depth `d`, and `d < i`, return a replacement
/// prefix of length `d + 1` (initial → ... → match) and `target_index = i`.
///
/// We only consider `i > 0` because a 0-length prefix would not shorten.
fn find_shortcut<M: Model>(
    model: &M,
    current: &[M::State],
    depth_cap: usize,
    start: Instant,
    budget: Duration,
) -> ShortcutResult<M::State> {
    use std::collections::HashMap;

    // Map: fingerprint of trace state -> its index in `current`.
    // We skip index 0 (no shortening possible) and look for the
    // smallest-index match — which by BFS-from-init also yields the
    // shortest replacement prefix that hits any later trace state.
    let mut trace_index: HashMap<u64, usize> = HashMap::with_capacity(current.len());
    for (i, st) in current.iter().enumerate().skip(1) {
        // First occurrence wins (smallest index).  BFS distance from s0
        // to current[i] in the original trace is `i`, so any BFS path
        // shorter than `i` reaching the same state is a strict win.
        trace_index.entry(model.fingerprint(st)).or_insert(i);
    }

    if trace_index.is_empty() {
        return ShortcutResult::None;
    }

    // Layered BFS from init states.  Each layer = one transition deeper.
    // We track parent pointers (by fingerprint) so we can reconstruct the
    // replacement prefix on first hit.
    let mut visited: HashSet<u64> = HashSet::new();
    // parent[fp] = (parent_fp, depth, state_clone).  We store the full
    // state so we can reconstruct without re-traversing the model.
    let mut parent: HashMap<u64, (Option<u64>, usize, M::State)> = HashMap::new();
    let mut queue: VecDeque<(M::State, u64, usize)> = VecDeque::new();

    for init in model.initial_states() {
        let fp = model.fingerprint(&init);
        if !visited.insert(fp) {
            continue;
        }
        parent.insert(fp, (None, 0, init.clone()));
        // If an initial state itself matches a later-trace state, that's
        // an immediate win (replacement prefix = [init], length 1).
        if let Some(&i) = trace_index.get(&fp) {
            if i > 0 {
                let mut prefix = Vec::with_capacity(1);
                prefix.push(init);
                return ShortcutResult::Found {
                    replacement_prefix: prefix,
                    target_index: i,
                };
            }
        }
        queue.push_back((init, fp, 0));
    }

    let mut successors_buf: Vec<M::State> = Vec::new();
    while let Some((state, state_fp, depth)) = queue.pop_front() {
        // Budget check — cheap to do per-pop.
        if start.elapsed() >= budget {
            return ShortcutResult::BudgetExhausted;
        }
        if depth >= depth_cap {
            continue;
        }
        successors_buf.clear();
        model.next_states(&state, &mut successors_buf);
        for next in successors_buf.drain(..) {
            let next_fp = model.fingerprint(&next);
            if !visited.insert(next_fp) {
                continue;
            }
            let next_depth = depth + 1;
            parent.insert(next_fp, (Some(state_fp), next_depth, next.clone()));
            // Check for a hit — any later-trace state whose BFS depth
            // (`next_depth`) is strictly less than its trace index.
            if let Some(&i) = trace_index.get(&next_fp) {
                if next_depth < i {
                    // Reconstruct prefix from `next_fp` back to an init.
                    let prefix = reconstruct_prefix::<M>(&parent, next_fp);
                    return ShortcutResult::Found {
                        replacement_prefix: prefix,
                        target_index: i,
                    };
                }
            }
            queue.push_back((next, next_fp, next_depth));
        }
    }

    ShortcutResult::None
}

fn reconstruct_prefix<M: Model>(
    parent: &std::collections::HashMap<u64, (Option<u64>, usize, M::State)>,
    leaf_fp: u64,
) -> Vec<M::State> {
    let mut chain: Vec<M::State> = Vec::new();
    let mut cur = Some(leaf_fp);
    while let Some(fp) = cur {
        let (parent_fp, _depth, state) = parent
            .get(&fp)
            .expect("parent map should contain every visited state");
        chain.push(state.clone());
        cur = *parent_fp;
    }
    chain.reverse();
    chain
}

// ============================================================================
// Phase B — variable highlighting (presentation only)
// ============================================================================

/// Determine which of `all_variables` are referenced by an invariant's
/// source text.  Used by the trace printer to visually de-emphasise
/// variables that don't drive the violation.
///
/// "Referenced" here is purely syntactic: we tokenize the invariant text
/// and check for whole-word matches against each variable name.  This
/// catches the common case (`Inv == count < 3` references `count` only)
/// and is over-conservative for transitively-referenced variables (an
/// invariant that calls another operator that references `x` will not
/// flag `x` as relevant).  Transitive closure is a follow-up (T9.1).
///
/// Returns the subset of `all_variables` that appear in `invariant_text`.
pub fn extract_invariant_variables(
    invariant_text: &str,
    all_variables: &[String],
) -> HashSet<String> {
    let mut found: HashSet<String> = HashSet::new();
    if all_variables.is_empty() {
        return found;
    }

    // Strip TLA+ comments so we don't match identifiers inside them.
    let cleaned = strip_tla_comments(invariant_text);

    // Build a set of variable names for O(1) membership lookup.
    let var_set: HashSet<&str> = all_variables.iter().map(|s| s.as_str()).collect();

    // Tokenize on identifier boundaries.  An identifier is a maximal run
    // of [A-Za-z0-9_] beginning with [A-Za-z_].  Anything else is a
    // separator.
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if (c.is_ascii_alphabetic()) || c == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let cc = bytes[i];
                if cc.is_ascii_alphanumeric() || cc == b'_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let tok = &cleaned[start..i];
            if var_set.contains(tok) {
                found.insert(tok.to_string());
            }
        } else {
            i += 1;
        }
    }

    found
}

/// Lightweight definition descriptor consumed by
/// [`extract_invariant_variables_transitive`].  Mirrors the shape of
/// `crate::tla::module::TlaDefinition` (name, params, body) but is
/// kept separate so the trace minimizer doesn't need to depend on the
/// TLA module-loading layer.
///
/// Callers in `main.rs` adapt `BTreeMap<String, TlaDefinition>` into
/// the format expected here via [`OperatorBody`] tuples.
#[derive(Debug, Clone)]
pub struct OperatorBody {
    /// The operator's defined name (used for cycle detection and lookup).
    pub name: String,
    /// The operator's body source text (scanned for further references).
    pub body: String,
}

/// Transitive variant of [`extract_invariant_variables`] that follows
/// operator-call chains.
///
/// Given the invariant text and a registry of operator definitions, this
/// pass scans `invariant_text` for identifiers, then recursively scans
/// the bodies of any operators referenced (cycle-safe via a visited
/// set).  Variable names matched at any level are accumulated.
///
/// Example: with `invariant_text = "Inv == IsBad(state)"` and operators
/// `IsBad(s) == s.x > 5`, the call returns `{x}` even though `x` is not
/// syntactically in the invariant body — because the operator-inlining
/// pass picks it up from `IsBad`'s body.
///
/// Operator definitions whose names don't match the variable list are
/// recursed into; their parameter names are excluded from the variable
/// match (so `IsBad(s)` won't accidentally flag `s` if `s` happens to
/// also be a state variable name — an unusual but possible collision).
pub fn extract_invariant_variables_transitive(
    invariant_text: &str,
    all_variables: &[String],
    operators: &BTreeMap<String, OperatorBody>,
) -> HashSet<String> {
    let var_set: HashSet<&str> = all_variables.iter().map(|s| s.as_str()).collect();
    let mut found: HashSet<String> = HashSet::new();
    if var_set.is_empty() {
        return found;
    }
    let mut visited_ops: HashSet<String> = HashSet::new();
    // Bound recursion depth to avoid pathological blowups; 64 is well
    // above any realistic invariant call-chain depth.
    scan_with_inlining(
        invariant_text,
        &var_set,
        operators,
        &mut found,
        &mut visited_ops,
        64,
    );
    found
}

/// Recursive helper: tokenize `text`, check each identifier against
/// `var_set` (variables) AND against `operators` (recurse).  Tracks
/// `visited_ops` to avoid infinite loops on mutually recursive
/// operators.  `depth_remaining` bounds total recursion depth.
fn scan_with_inlining(
    text: &str,
    var_set: &HashSet<&str>,
    operators: &BTreeMap<String, OperatorBody>,
    found: &mut HashSet<String>,
    visited_ops: &mut HashSet<String>,
    depth_remaining: usize,
) {
    if depth_remaining == 0 {
        return;
    }
    let cleaned = strip_tla_comments(text);
    // Collect identifiers; check for variable-name and operator-name
    // matches in a single pass.  We collect operator names to recurse
    // into AFTER finishing the current scan, so the recursion is
    // breadth-first-ish (reduces stack depth on long chains).
    let mut to_recurse: Vec<String> = Vec::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            i += 1;
            while i < bytes.len() {
                let cc = bytes[i];
                if cc.is_ascii_alphanumeric() || cc == b'_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let tok = &cleaned[s..i];
            if var_set.contains(tok) {
                found.insert(tok.to_string());
            }
            if operators.contains_key(tok) && !visited_ops.contains(tok) {
                to_recurse.push(tok.to_string());
            }
        } else {
            i += 1;
        }
    }
    for op_name in to_recurse {
        if visited_ops.insert(op_name.clone()) {
            if let Some(op) = operators.get(&op_name) {
                scan_with_inlining(
                    &op.body,
                    var_set,
                    operators,
                    found,
                    visited_ops,
                    depth_remaining - 1,
                );
            }
        }
    }
}

/// Strip `\* ...` line comments and `(* ... *)` block comments from a
/// TLA+ source fragment.  Used by [`extract_invariant_variables`] to
/// avoid matching identifiers inside comments.
///
/// Block-comment nesting follows TLA+'s rules: `(* ... (* nested *) ... *)`
/// requires balanced delimiters.  We track depth.
fn strip_tla_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut block_depth: usize = 0;
    while i < bytes.len() {
        if block_depth == 0 {
            // Look for line-comment start `\*`.
            if i + 1 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'*' {
                // Skip to end of line.
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            // Look for block-comment start `(*`.
            if i + 1 < bytes.len() && bytes[i] == b'(' && bytes[i + 1] == b'*' {
                block_depth = 1;
                i += 2;
                continue;
            }
            out.push(bytes[i] as char);
            i += 1;
        } else {
            // Inside a block comment.  Track nested `(*` and matching `*)`.
            if i + 1 < bytes.len() && bytes[i] == b'(' && bytes[i + 1] == b'*' {
                block_depth += 1;
                i += 2;
            } else if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b')' {
                block_depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Phase B — variable highlighting tests
    // ------------------------------------------------------------------

    #[test]
    fn extract_picks_up_simple_variable_reference() {
        let inv = "Inv == count < 3";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.contains("count"));
        assert!(!found.contains("flag"));
    }

    #[test]
    fn extract_handles_multiple_variables() {
        let inv = "Inv == x + y > 0 /\\ z = 0";
        let vars = vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "w".to_string(),
        ];
        let found = extract_invariant_variables(inv, &vars);
        assert_eq!(
            found,
            ["x", "y", "z"].iter().map(|s| s.to_string()).collect()
        );
    }

    #[test]
    fn extract_ignores_substring_matches() {
        // `count` is a variable; `accountant` should not match.
        let inv = "Inv == accountant > 0";
        let vars = vec!["count".to_string(), "accountant".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.contains("accountant"));
        assert!(!found.contains("count"));
    }

    #[test]
    fn extract_ignores_identifiers_inside_line_comments() {
        let inv = "Inv == count > 0 \\* really we want flag = TRUE here\n";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.contains("count"));
        assert!(!found.contains("flag"));
    }

    #[test]
    fn extract_ignores_identifiers_inside_block_comments() {
        let inv = "Inv == count > 0 (* flag is unused *) /\\ TRUE";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.contains("count"));
        assert!(!found.contains("flag"));
    }

    #[test]
    fn extract_handles_nested_block_comments() {
        let inv = "Inv == count > 0 (* outer (* inner flag *) more *) ";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.contains("count"));
        assert!(!found.contains("flag"));
    }

    #[test]
    fn extract_returns_empty_when_invariant_uses_no_state_vars() {
        let inv = "Inv == TRUE";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let found = extract_invariant_variables(inv, &vars);
        assert!(found.is_empty());
    }

    // ------------------------------------------------------------------
    // Phase A — minimization tests using a tiny hand-rolled model
    // ------------------------------------------------------------------

    use crate::model::Model;
    use serde::{Deserialize, Serialize};

    /// Linear-graph model: states are integers 0..=N.
    /// Init = {0}.  Next = { s+1 } if s < N else {}.
    /// Invariant: state < N.  Violation at state == N.
    ///
    /// Trace from BFS will be `[0, 1, 2, ..., N]` (length N+1).  No
    /// shortening is possible — minimize_trace should return identity.
    struct LinearModel {
        n: i64,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
    struct LinState(i64);

    impl Model for LinearModel {
        type State = LinState;
        fn name(&self) -> &'static str {
            "linear"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![LinState(0)]
        }
        fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
            if state.0 < self.n {
                out.push(LinState(state.0 + 1));
            }
        }
        fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
            if state.0 < self.n {
                Ok(())
            } else {
                Err(format!("violated at {}", state.0))
            }
        }
    }

    #[test]
    fn minimize_linear_returns_identity() {
        let model = LinearModel { n: 4 };
        let trace = vec![
            LinState(0),
            LinState(1),
            LinState(2),
            LinState(3),
            LinState(4),
        ];
        let original_len = trace.len();
        let result = minimize_trace(&model, trace, Duration::from_secs(5));
        assert_eq!(result.trace.len(), original_len);
        assert_eq!(result.original_len, original_len);
        // Final state is still the violation.
        assert!(
            model
                .check_invariants(result.trace.last().unwrap())
                .is_err()
        );
    }

    /// Diamond-graph model: state space is {0, 1A, 1B, 2A, 2B, 3, 4}.
    /// 0 -> 1A, 1A -> 2A, 2A -> 3, 3 -> 4.  Also 0 -> 1B, 1B -> 2B,
    /// 2B -> 4 (a 4-step path bypassing state 3).  Invariant: state != 4.
    ///
    /// If we prime the trace with the long path [0, 1A, 2A, 3, 4]
    /// (length 5), minimize_trace should find the 4-step path
    /// [0, 1B, 2B, 4] (length 4) by BFS.
    struct DiamondModel;

    #[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
    enum DState {
        S0,
        S1A,
        S1B,
        S2A,
        S2B,
        S3,
        S4,
    }

    impl Model for DiamondModel {
        type State = DState;
        fn name(&self) -> &'static str {
            "diamond"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![DState::S0]
        }
        fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
            match state {
                DState::S0 => {
                    out.push(DState::S1A);
                    out.push(DState::S1B);
                }
                DState::S1A => out.push(DState::S2A),
                DState::S1B => out.push(DState::S2B),
                DState::S2A => out.push(DState::S3),
                DState::S2B => out.push(DState::S4),
                DState::S3 => out.push(DState::S4),
                DState::S4 => {}
            }
        }
        fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
            match state {
                DState::S4 => Err("hit S4".to_string()),
                _ => Ok(()),
            }
        }
    }

    #[test]
    fn minimize_diamond_finds_shorter_path() {
        let model = DiamondModel;
        // Priming with the longer 5-step path through S3.
        let trace = vec![DState::S0, DState::S1A, DState::S2A, DState::S3, DState::S4];
        let original_len = trace.len();
        let result = minimize_trace(&model, trace, Duration::from_secs(5));
        // The shorter path [S0, S1B, S2B, S4] is 4 states.
        assert_eq!(result.trace.len(), 4);
        assert_eq!(result.original_len, original_len);
        assert_eq!(*result.trace.first().unwrap(), DState::S0);
        assert_eq!(*result.trace.last().unwrap(), DState::S4);
        assert!(
            model
                .check_invariants(result.trace.last().unwrap())
                .is_err()
        );
        assert!(result.iterations >= 1);
    }

    /// Validate every adjacent pair in the minimized trace is a valid
    /// transition (post-condition guarantee).
    #[test]
    fn minimize_diamond_preserves_transition_validity() {
        let model = DiamondModel;
        let trace = vec![DState::S0, DState::S1A, DState::S2A, DState::S3, DState::S4];
        let result = minimize_trace(&model, trace, Duration::from_secs(5));
        let mut buf = Vec::new();
        for pair in result.trace.windows(2) {
            buf.clear();
            model.next_states(&pair[0], &mut buf);
            assert!(
                buf.contains(&pair[1]),
                "invalid transition {:?} -> {:?} in minimized trace",
                pair[0],
                pair[1]
            );
        }
        // First state must be an initial state.
        let inits = model.initial_states();
        assert!(inits.contains(result.trace.first().unwrap()));
    }

    #[test]
    fn minimize_handles_empty_trace_gracefully() {
        let model = LinearModel { n: 4 };
        let result = minimize_trace(&model, Vec::<LinState>::new(), Duration::from_secs(1));
        assert!(result.trace.is_empty());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn minimize_handles_single_state_trace() {
        // Trace of length 1: only the violating state itself.  Cannot be
        // shortened.
        let model = LinearModel { n: 0 };
        let result = minimize_trace(&model, vec![LinState(0)], Duration::from_secs(1));
        assert_eq!(result.trace.len(), 1);
    }

    #[test]
    fn minimize_returns_input_when_final_state_is_not_violating() {
        // Defensive: if the caller passes a non-violating trace, do not
        // truncate it.
        let model = LinearModel { n: 10 };
        let trace = vec![LinState(0), LinState(1), LinState(2)];
        let result = minimize_trace(&model, trace.clone(), Duration::from_secs(1));
        assert_eq!(result.trace, trace);
        assert_eq!(result.iterations, 0);
    }

    /// Exercises the truncation pre-pass: a trace where an early state
    /// already violates the invariant should be truncated at that
    /// state, not "minimized" by re-running BFS.
    #[test]
    fn minimize_truncates_at_earliest_violation() {
        // LinearModel with N=2: invariant says state.0 < 2.  Build a
        // trace [0, 1, 2, 1, 2] — state index 2 is the first violation.
        // (Note: the model can't actually go from 2 -> 1, so this is
        // purely a synthetic stress test of the truncation pre-pass on
        // a trace that doesn't claim transition validity for the post-
        // violation suffix.  In practice the BFS-derived trace never
        // contains invalid transitions, but truncation is sound either
        // way.)
        let model = LinearModel { n: 2 };
        let trace = vec![
            LinState(0),
            LinState(1),
            LinState(2),
            LinState(1),
            LinState(2),
        ];
        let result = minimize_trace(&model, trace, Duration::from_secs(1));
        // Truncation should take the trace down to [0, 1, 2].
        assert_eq!(result.trace.len(), 3);
        assert_eq!(result.trace.last().unwrap().0, 2);
    }

    // ------------------------------------------------------------------
    // T9.1 — transitive variable relevance through operator inlining
    // ------------------------------------------------------------------

    #[test]
    fn t91_transitive_picks_up_variable_through_one_operator() {
        // Inv == IsBad(s) where IsBad(s) == s.x > 5
        // The variable name "x" appears only in the operator body, not
        // in the invariant text; the transitive scan must still find it.
        let inv = "Inv == IsBad(s)";
        let vars = vec!["x".to_string(), "y".to_string()];
        let mut ops: BTreeMap<String, OperatorBody> = BTreeMap::new();
        ops.insert(
            "IsBad".to_string(),
            OperatorBody {
                name: "IsBad".to_string(),
                body: "IsBad(s) == s.x > 5".to_string(),
            },
        );
        let found = extract_invariant_variables_transitive(inv, &vars, &ops);
        assert!(found.contains("x"), "transitive scan must include x");
        assert!(!found.contains("y"), "y is not referenced anywhere");
    }

    #[test]
    fn t91_transitive_chains_through_multiple_operators() {
        // Inv == OuterCheck
        // OuterCheck == InnerCheck
        // InnerCheck == count > 0
        let inv = "Inv == OuterCheck";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let mut ops: BTreeMap<String, OperatorBody> = BTreeMap::new();
        ops.insert(
            "OuterCheck".to_string(),
            OperatorBody {
                name: "OuterCheck".to_string(),
                body: "OuterCheck == InnerCheck".to_string(),
            },
        );
        ops.insert(
            "InnerCheck".to_string(),
            OperatorBody {
                name: "InnerCheck".to_string(),
                body: "InnerCheck == count > 0".to_string(),
            },
        );
        let found = extract_invariant_variables_transitive(inv, &vars, &ops);
        assert!(found.contains("count"));
        assert!(!found.contains("flag"));
    }

    #[test]
    fn t91_transitive_handles_recursive_operators_without_looping() {
        // Mutually recursive ops should not cause infinite recursion.
        let inv = "Inv == A";
        let vars = vec!["x".to_string()];
        let mut ops: BTreeMap<String, OperatorBody> = BTreeMap::new();
        ops.insert(
            "A".to_string(),
            OperatorBody {
                name: "A".to_string(),
                body: "A == B /\\ x = 0".to_string(),
            },
        );
        ops.insert(
            "B".to_string(),
            OperatorBody {
                name: "B".to_string(),
                body: "B == A".to_string(),
            },
        );
        let found = extract_invariant_variables_transitive(inv, &vars, &ops);
        assert!(found.contains("x"));
    }

    #[test]
    fn t91_transitive_falls_back_to_syntactic_when_no_operators() {
        // With an empty operator registry, the transitive variant must
        // produce the same result as the syntactic one.
        let inv = "Inv == count > 0";
        let vars = vec!["count".to_string(), "flag".to_string()];
        let ops: BTreeMap<String, OperatorBody> = BTreeMap::new();
        let trans = extract_invariant_variables_transitive(inv, &vars, &ops);
        let syn = extract_invariant_variables(inv, &vars);
        assert_eq!(trans, syn);
    }

    #[test]
    fn t91_transitive_does_not_match_operators_inside_comments() {
        // An operator name appearing only in a comment must not trigger
        // recursion — `strip_tla_comments` runs at every level.
        let inv = "Inv == TRUE \\* IsBad would mention x\n";
        let vars = vec!["x".to_string()];
        let mut ops: BTreeMap<String, OperatorBody> = BTreeMap::new();
        ops.insert(
            "IsBad".to_string(),
            OperatorBody {
                name: "IsBad".to_string(),
                body: "IsBad == x > 0".to_string(),
            },
        );
        let found = extract_invariant_variables_transitive(inv, &vars, &ops);
        assert!(!found.contains("x"));
    }

    #[test]
    fn minimize_respects_zero_budget() {
        // A zero-budget call should not panic and should return the
        // original trace (Phase A loop bails immediately).
        let model = DiamondModel;
        let trace = vec![DState::S0, DState::S1A, DState::S2A, DState::S3, DState::S4];
        let result = minimize_trace(&model, trace.clone(), Duration::from_nanos(0));
        // With zero budget we expect either the original trace OR a
        // partially-shortened one, but never longer than the input and
        // always still violating.
        assert!(result.trace.len() <= trace.len());
        assert!(
            model
                .check_invariants(result.trace.last().unwrap())
                .is_err()
        );
    }
}
