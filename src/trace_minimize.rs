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
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
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

        // T9.2: choose seeds = {initial states} ∪ {median trace state}.
        // The trace endpoint (current.last()) is itself a violation, so
        // forward BFS from there is unhelpful — but the median is a
        // useful interior seed for long traces, where BFS-from-init
        // may exhaust the budget before reaching the back half.
        let seeds = build_seeds(&current);
        match find_shortcut(model, &current, &seeds, depth_cap, start, budget) {
            ShortcutResult::Found {
                replacement_segment,
                source_index,
                target_index,
            } => {
                // Replace `current[source_index..=target_index]` with
                // `replacement_segment`.  Precondition (debug-checked):
                // segment[0] == current[source_index], segment.last() ==
                // current[target_index], and segment is strictly shorter.
                debug_assert!(target_index > source_index);
                debug_assert!(replacement_segment.len() < target_index - source_index + 1);
                debug_assert!(!replacement_segment.is_empty());
                let mut new_trace: Vec<M::State> = Vec::with_capacity(
                    current.len() - (target_index - source_index + 1) + replacement_segment.len(),
                );
                new_trace.extend(current.iter().take(source_index).cloned());
                new_trace.extend(replacement_segment.into_iter());
                new_trace.extend(current.iter().skip(target_index + 1).cloned());
                current = new_trace;
                iterations += 1;
                // Validate that the reconstructed trace still ends in a
                // violating state.  If something went wrong, abort
                // shortening and keep what we have.
                if model.check_invariants(current.last().unwrap()).is_ok() {
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
    /// Found a shorter segment that connects `current[source_index]` to
    /// `current[target_index]`.  When `source_index == 0`, the segment
    /// starts at an initial state and replaces the trace prefix; when
    /// `source_index > 0`, it replaces the interior segment between two
    /// trace states (T9.2 internal-shortcut case).
    Found {
        replacement_segment: Vec<S>,
        source_index: usize,
        target_index: usize,
    },
    /// No improvement possible.
    None,
    /// Budget elapsed during search.
    BudgetExhausted,
}

/// A BFS seed: either an initial state of the model (`Init`) or an
/// interior state of the trace at index `trace_index` (`Trace`).
///
/// T9.2 widens `find_shortcut` to multi-source BFS — initial states
/// plus the median trace state — so very long traces can be shortened
/// from the inside-out, not just from index 0.  Forward BFS from the
/// trace's *final* state is intentionally omitted: it's already a
/// violation, so any path forward leaves the violating-prefix invariant.
///
/// Init seeds are produced inline by [`find_shortcut`] (which calls
/// `model.initial_states()`); only Trace seeds need to be returned by
/// [`build_seeds`], which is why this enum is not exhaustively
/// constructed at the call site.
#[derive(Clone)]
enum Seed<S> {
    #[allow(dead_code)]
    Init(S),
    Trace {
        state: S,
        trace_index: usize,
    },
}

/// Build the seed list for [`find_shortcut`].  Always includes all
/// initial states of the model (Phase A's original behaviour).  For
/// traces of length >= 8 (heuristic threshold to avoid duplication on
/// already-short traces), also seeds with the trace's median state.
///
/// The median seed lets the BFS find shortcuts in the back half of the
/// trace without first having to reach it from `s0`.  On a 1000-state
/// trace, BFS-from-init at depth 500 might exhaust the budget before
/// finding a useful candidate; BFS from `current[500]` at depth 250
/// reaches the violation in many fewer expansions.
fn build_seeds<S: Clone>(current: &[S]) -> Vec<Seed<S>> {
    let mut seeds: Vec<Seed<S>> = Vec::new();
    // Initial-state seeds are deferred to `find_shortcut`, which calls
    // `model.initial_states()` directly (so we don't have to clone the
    // model state into this helper).  We tag this here only via the
    // trace seeds; the BFS always starts from inits unconditionally.
    if current.len() >= 8 {
        let mid = current.len() / 2;
        // Skip seeding at index 0 (handled by Init seeds) and at the
        // last index (already-violating; forward BFS is unhelpful).
        if mid > 0 && mid + 1 < current.len() {
            seeds.push(Seed::Trace {
                state: current[mid].clone(),
                trace_index: mid,
            });
        }
    }
    seeds
}

/// Multi-source BFS from initial states (always) plus any extra
/// `interior_seeds` (Trace variants from [`build_seeds`]).
///
/// For each seed, we look for any later-trace state reachable in fewer
/// transitions than the trace itself takes.  Specifically:
///
/// - From an `Init` seed, depth `d` to `current[i]` is a win iff `d < i`.
/// - From a `Trace { trace_index: src }` seed, depth `d` to `current[j]`
///   is a win iff `j > src + 1` and `d < j - src`.
///
/// On hit we reconstruct the replacement segment from parent pointers.
/// We commit to the *first* hit found in BFS order (which is the lowest
/// depth across all sources, breaking ties by source order).
fn find_shortcut<M: Model>(
    model: &M,
    current: &[M::State],
    interior_seeds: &[Seed<M::State>],
    depth_cap: usize,
    start: Instant,
    budget: Duration,
) -> ShortcutResult<M::State> {
    // Map: fingerprint of trace state -> its index in `current`.
    // First occurrence wins (smallest index) — that's the index we
    // compare against to decide if BFS found a strict shortcut.
    let mut trace_index: HashMap<u64, usize> = HashMap::with_capacity(current.len());
    for (i, st) in current.iter().enumerate() {
        trace_index.entry(model.fingerprint(st)).or_insert(i);
    }

    // SourceTag identifies which BFS root a parent-chain leads back to.
    // Init(fp) means the chain ends at an initial state with that fp;
    // Trace(idx) means the chain ends at the trace's state at `idx`.
    #[derive(Clone, Copy)]
    enum SourceTag {
        Init,
        Trace(usize),
    }

    // parent[fp] = (parent_fp, depth, state_clone, source_tag)
    let mut visited: HashSet<u64> = HashSet::new();
    let mut parent: HashMap<u64, (Option<u64>, usize, M::State, SourceTag)> = HashMap::new();
    let mut queue: VecDeque<(M::State, u64, usize, SourceTag)> = VecDeque::new();

    // Helper: try to plant a seed at `state`.  Returns Some(result) if
    // the seed itself is a shortcut win (an init that already equals a
    // later trace state); otherwise enqueues for BFS expansion.
    let plant_seed = |state: M::State,
                      tag: SourceTag,
                      visited: &mut HashSet<u64>,
                      parent: &mut HashMap<u64, (Option<u64>, usize, M::State, SourceTag)>,
                      queue: &mut VecDeque<(M::State, u64, usize, SourceTag)>|
     -> Option<ShortcutResult<M::State>> {
        let fp = model.fingerprint(&state);
        if !visited.insert(fp) {
            return None;
        }
        parent.insert(fp, (None, 0, state.clone(), tag));
        // Self-hit check: does this seed itself collide with a later trace state?
        if let Some(&i) = trace_index.get(&fp) {
            match tag {
                SourceTag::Init if i > 0 => {
                    // Replacement prefix [init] of length 1 vs trace
                    // distance i.  Win iff i > 0 (segment len 1 < i+1
                    // iff i >= 1).
                    return Some(ShortcutResult::Found {
                        replacement_segment: vec![state],
                        source_index: 0,
                        target_index: i,
                    });
                }
                SourceTag::Trace(src) if i > src + 1 => {
                    // Segment of len 1 vs trace distance i - src.
                    // Win iff i > src + 1.
                    return Some(ShortcutResult::Found {
                        replacement_segment: vec![state],
                        source_index: src,
                        target_index: i,
                    });
                }
                _ => {}
            }
        }
        queue.push_back((state, fp, 0, tag));
        None
    };

    // Plant init seeds first (Phase A baseline).
    for init in model.initial_states() {
        if let Some(hit) = plant_seed(init, SourceTag::Init, &mut visited, &mut parent, &mut queue)
        {
            return hit;
        }
    }

    // Plant T9.2 interior seeds.
    for seed in interior_seeds {
        if let Seed::Trace {
            state,
            trace_index: idx,
        } = seed
        {
            if let Some(hit) = plant_seed(
                state.clone(),
                SourceTag::Trace(*idx),
                &mut visited,
                &mut parent,
                &mut queue,
            ) {
                return hit;
            }
        }
    }

    if trace_index.is_empty() {
        return ShortcutResult::None;
    }

    let mut successors_buf: Vec<M::State> = Vec::new();
    while let Some((state, state_fp, depth, tag)) = queue.pop_front() {
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
            parent.insert(next_fp, (Some(state_fp), next_depth, next.clone(), tag));
            // Check for a hit — depends on this node's source tag.
            if let Some(&i) = trace_index.get(&next_fp) {
                let win = match tag {
                    SourceTag::Init => i > 0 && next_depth < i,
                    SourceTag::Trace(src) => i > src + 1 && next_depth < i - src,
                };
                if win {
                    let segment = reconstruct_segment::<M>(&parent, next_fp);
                    let (source_index, target_index) = match tag {
                        SourceTag::Init => (0, i),
                        SourceTag::Trace(src) => (src, i),
                    };
                    return ShortcutResult::Found {
                        replacement_segment: segment,
                        source_index,
                        target_index,
                    };
                }
            }
            queue.push_back((next, next_fp, next_depth, tag));
        }
    }

    ShortcutResult::None
}

fn reconstruct_segment<M: Model>(
    parent: &HashMap<u64, (Option<u64>, usize, M::State, impl Copy)>,
    leaf_fp: u64,
) -> Vec<M::State> {
    let mut chain: Vec<M::State> = Vec::new();
    let mut cur = Some(leaf_fp);
    while let Some(fp) = cur {
        let (parent_fp, _depth, state, _tag) = parent
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

    // ------------------------------------------------------------------
    // T9.2 — smarter BFS seed (median-state seeding for long traces)
    // ------------------------------------------------------------------

    /// Long-trace model: states are integers 0..=N.  Init = {0}.
    /// Transitions: s -> s+1 always, plus s -> s+10 every 5 steps
    /// (a "fast-forward" shortcut).  This gives the trace minimizer a
    /// chance to find an interior shortcut that's only discoverable
    /// starting from a non-initial state.
    struct ForwardSkipModel {
        n: i64,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
    struct FsState(i64);

    impl Model for ForwardSkipModel {
        type State = FsState;
        fn name(&self) -> &'static str {
            "forward-skip"
        }
        fn initial_states(&self) -> Vec<Self::State> {
            vec![FsState(0)]
        }
        fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
            if state.0 < self.n {
                out.push(FsState(state.0 + 1));
            }
            // Skip-by-10 shortcut at every multiple of 5
            if state.0 % 5 == 0 && state.0 + 10 <= self.n {
                out.push(FsState(state.0 + 10));
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
    fn t92_build_seeds_includes_median_for_long_traces() {
        // Direct unit test of build_seeds: should emit one Trace seed
        // for traces of length >= 8, none for shorter.
        let short: Vec<i32> = (0..7).collect();
        let long: Vec<i32> = (0..16).collect();
        assert!(
            build_seeds(&short).is_empty(),
            "short traces should not get interior seeds"
        );
        let seeds = build_seeds(&long);
        assert_eq!(seeds.len(), 1);
        match &seeds[0] {
            Seed::Trace { trace_index, .. } => {
                assert_eq!(*trace_index, 8, "median of 16-state trace is index 8");
            }
            _ => panic!("expected Trace seed"),
        }
    }

    #[test]
    fn t92_long_trace_minimizes_via_skip_shortcut() {
        // 30-step trace 0,1,2,...,30 (length 31).  State 30 violates.
        // The skip transitions allow paths like 0->10->20->30 (length 4),
        // 0->5->15->25->30 (length 5), etc.  Optimal is 4.
        // This is a regression test: with median seeding, the BFS finds
        // the shortcut even when the trace is much longer.
        let model = ForwardSkipModel { n: 30 };
        let trace: Vec<FsState> = (0..=30).map(FsState).collect();
        let result = minimize_trace(&model, trace, Duration::from_secs(5));
        // Optimal path uses the +10 shortcut 3 times: 0->10->20->30.
        // That's 4 states.
        assert_eq!(
            result.trace.len(),
            4,
            "expected length-4 path 0->10->20->30, got {:?}",
            result.trace.iter().map(|s| s.0).collect::<Vec<_>>()
        );
        assert_eq!(result.trace.first().unwrap().0, 0);
        assert_eq!(result.trace.last().unwrap().0, 30);
        // Validate transitions.
        let mut buf = Vec::new();
        for pair in result.trace.windows(2) {
            buf.clear();
            model.next_states(&pair[0], &mut buf);
            assert!(buf.contains(&pair[1]), "invalid transition in minimized");
        }
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
