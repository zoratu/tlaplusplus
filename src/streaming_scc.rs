//! T10.2 — Streaming SCC discovery for liveness on 100M+-state graphs.
//!
//! # Why this exists
//!
//! The current liveness post-processing (see `runtime.rs` Phase 1-5 around the
//! `labeled_transitions` block) is a **two-pass** scheme: BFS-with-fingerprint-
//! dedup builds the full transition map, then a separate Tarjan pass finds
//! SCCs and a fairness check sweeps them. This is fast for ~100K-state graphs
//! (T10 cut it to 115ms on 32K states) but it must materialize the entire
//! transition set in memory, which is infeasible at 100M+ states (each
//! `LabeledTransition<S>` carries two `S` clones plus an action label; on a
//! TLA+ spec with 100M states and ~5 successors each, that's roughly
//! 100GB-1TB just for the transition map).
//!
//! The textbook fix is **on-the-fly liveness checking** (Courcoubetis, Vardi,
//! Wolper, Yannakakis 1992): interleave the property check with the state
//! exploration so we never store the full transition graph. The classic
//! algorithm is **nested DFS**:
//!
//! - The **outer (blue) DFS** explores the state graph in depth-first order.
//!   When it finishes processing an *accepting* state (one that completes a
//!   "bad" prefix in the Buchi-product sense), it launches an inner DFS.
//! - The **inner (red) DFS** searches for any path back to a state currently
//!   on the blue stack. Such a path proves an accepting cycle, which is a
//!   liveness violation.
//!
//! Memory cost: O(reachable states × 2 bits) for the color tags, plus the DFS
//! stacks. No transition map. Time cost: O(states + edges), same as Tarjan,
//! but with a small constant factor advantage (single-pass).
//!
//! # What this module provides (T10.2 v1.1.0)
//!
//! Phase 1 of T10.2: a **standalone, abstract** nested-DFS implementation
//! that operates on any graph exposing `successors()` and `is_accepting()`.
//! It is correctness-tested on small synthetic graphs with known accepting
//! cycles. It can be plugged into the runtime in two ways:
//!
//! 1. **Post-exploration mode (validation harness)**: run nested-DFS over
//!    the same fingerprint adjacency map that Tarjan currently consumes.
//!    This proves correctness against the existing path and is the entry
//!    point for the `--liveness-streaming` flag in v1.1.0.
//! 2. **In-exploration mode (true streaming, future work)**: replace the
//!    BFS worker loop with per-worker DFS, tagging colors as we go. See
//!    `docs/T10.2-streaming-scc-design.md` for the full design and the
//!    open questions around distributed work-stealing across DFS branches.
//!
//! Today this module ships (1) only. The design doc describes (2) and the
//! corpus revalidation plan.
//!
//! # Correctness sketch
//!
//! Standard CVWY '92 result: nested DFS finds an accepting cycle iff one
//! exists. The two invariants it preserves:
//!
//! - **Blue invariant**: when blue DFS pops state `v`, every state reachable
//!   from `v` has been blue-visited (i.e., the blue tree below `v` is
//!   complete). So if `v` is accepting and we don't find a back-edge to the
//!   blue stack via the red search starting at `v`, no other future
//!   accepting cycle can include `v`.
//! - **Red invariant**: red DFS started at accepting state `v` searches the
//!   subgraph reachable from `v`, looking for any blue-stack member. If
//!   found, the path `v → ... → w → ... → v` (via blue stack) is an
//!   accepting cycle.
//!
//! See proofs in:
//! - Courcoubetis, Vardi, Wolper, Yannakakis. "Memory-efficient algorithms
//!   for the verification of temporal properties." (1992).
//! - Holzmann, Peled, Yannakakis. "On nested depth first search." (1996).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Color tag attached to each state during nested DFS.
///
/// `White`: not yet visited by blue DFS.
/// `Cyan`:  on the current blue DFS stack (frontier — used to detect
///          accepting cycles via red back-edge).
/// `Blue`:  blue DFS finished processing this state (fully expanded).
/// `Red`:   visited by an inner red DFS (cycle search).
///
/// Memory: the color is a 2-bit field; for 100M states we need ~25MB of
/// tags, vs ~1TB for the materialized transition map. That's the entire
/// point of this redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Cyan,
    Blue,
    Red,
}

/// Result of the nested-DFS search.
#[derive(Debug, Clone)]
pub enum NestedDfsResult<S> {
    /// No accepting cycle was found in the explored region.
    NoAcceptingCycle,
    /// An accepting cycle was found. `lasso_prefix` is the path from an
    /// initial state to the accepting state on the cycle; `cycle` is the
    /// states forming the cycle (last entry == first entry of cycle to
    /// signal the loop).
    AcceptingCycle {
        lasso_prefix: Vec<S>,
        cycle: Vec<S>,
    },
}

/// Trait that a graph must expose for `nested_dfs` to consume.
///
/// `S` is the state type (typically `u64` in the runtime — fingerprints —
/// or a richer state in tests/synthetic models). The graph is queried
/// purely on demand: nested-DFS never materializes the full successor
/// table.
pub trait LivenessGraph {
    type State: Clone + Debug + Eq + Hash;

    /// All initial states from which the search begins.
    fn initial_states(&self) -> Vec<Self::State>;

    /// Successors of `state`. Called once per visited state in blue DFS,
    /// and once per visited state in each red DFS.
    fn successors(&self, state: &Self::State) -> Vec<Self::State>;

    /// Acceptance predicate. A state is "accepting" iff a fairness or
    /// Buchi-style obligation has been violated along some prefix ending
    /// here. For weak fairness on a wrapper-Next action, all states are
    /// accepting (any cycle is a violation candidate); for a named
    /// action, only states that have not seen the action recently are
    /// accepting. The integration with the TLA+ fairness mapping is
    /// described in `docs/T10.2-streaming-scc-design.md`.
    fn is_accepting(&self, state: &Self::State) -> bool;
}

/// Run a single-threaded nested DFS to find an accepting cycle.
///
/// This is the reference implementation, used as the correctness oracle
/// for the parallel/in-runtime variants. The algorithm is:
///
/// ```text
/// for s in initial_states:
///     if color[s] == White: blue_dfs(s)
///
/// blue_dfs(v):
///     color[v] = Cyan; push v on blue_stack
///     for w in successors(v):
///         if color[w] == White: blue_dfs(w)
///     if is_accepting(v):
///         red_dfs(v)
///     color[v] = Blue; pop v from blue_stack
///
/// red_dfs(seed):
///     for w in successors(seed):
///         if color[w] == Cyan: report cycle (w on blue stack)
///         if color[w] != Red: color[w] = Red; red_dfs(w)
/// ```
///
/// The implementation here is **iterative** (explicit stack) so it does
/// not blow the call stack on deep state graphs; same rationale as the
/// iterative Tarjan in `fairness.rs`.
pub fn nested_dfs<G: LivenessGraph>(graph: &G) -> NestedDfsResult<G::State> {
    let mut colors: HashMap<G::State, Color> = HashMap::new();
    // The blue DFS path from an initial state; used to reconstruct the
    // lasso prefix and the cycle when red DFS finds a back-edge.
    let mut blue_path: Vec<G::State> = Vec::new();

    for init in graph.initial_states() {
        if colors.get(&init).copied().unwrap_or(Color::White) != Color::White {
            continue;
        }
        if let Some((prefix, cycle)) = blue_dfs_iter(graph, init, &mut colors, &mut blue_path) {
            return NestedDfsResult::AcceptingCycle {
                lasso_prefix: prefix,
                cycle,
            };
        }
        debug_assert!(blue_path.is_empty(), "blue path must drain after each root");
    }

    NestedDfsResult::NoAcceptingCycle
}

/// Iterative blue DFS rooted at `root`. On finding an accepting cycle,
/// returns `Some((lasso_prefix, cycle))`. Otherwise returns `None` after
/// fully expanding the reachable subgraph.
fn blue_dfs_iter<G: LivenessGraph>(
    graph: &G,
    root: G::State,
    colors: &mut HashMap<G::State, Color>,
    blue_path: &mut Vec<G::State>,
) -> Option<(Vec<G::State>, Vec<G::State>)> {
    // Frame: (state, successors, next-index-to-process). We "enter" a
    // state by setting Cyan and pushing onto the blue path; we "exit" by
    // running the accepting check (and red DFS if accepting) and setting
    // Blue.
    enter(colors, blue_path, root.clone());
    let mut stack: Vec<(G::State, Vec<G::State>, usize)> =
        vec![(root.clone(), graph.successors(&root), 0)];

    while let Some((v, succs, mut i)) = stack.pop() {
        let mut recursed = false;
        while i < succs.len() {
            let w = succs[i].clone();
            i += 1;
            match colors.get(&w).copied().unwrap_or(Color::White) {
                Color::White => {
                    enter(colors, blue_path, w.clone());
                    let w_succs = graph.successors(&w);
                    stack.push((v.clone(), succs, i));
                    stack.push((w, w_succs, 0));
                    recursed = true;
                    break;
                }
                _ => continue,
            }
        }
        if recursed {
            continue;
        }

        // All successors processed. Run the post-order acceptance check
        // and (if accepting) red DFS.
        if graph.is_accepting(&v) {
            // Red DFS seed at v. The cycle witness, if found, is the blue
            // path from the witness state back to v plus a final hop
            // closing the cycle.
            if let Some((witness, red_cycle_tail)) = red_dfs(graph, &v, colors, blue_path) {
                let prefix_idx = blue_path
                    .iter()
                    .position(|s| s == &witness)
                    .expect("witness must be on blue path (Cyan)");
                let lasso_prefix: Vec<G::State> = blue_path[..prefix_idx].to_vec();
                // Cycle: from witness, follow blue path down to v,
                // then the red-trail back to witness.
                let mut cycle: Vec<G::State> = blue_path[prefix_idx..].to_vec();
                cycle.extend(red_cycle_tail.into_iter());
                cycle.push(witness.clone());
                return Some((lasso_prefix, cycle));
            }
        }

        // Pop v from blue path; mark Blue.
        let popped = blue_path.pop();
        debug_assert_eq!(popped.as_ref(), Some(&v));
        colors.insert(v, Color::Blue);
    }

    None
}

fn enter<S: Clone + Eq + Hash>(
    colors: &mut HashMap<S, Color>,
    blue_path: &mut Vec<S>,
    s: S,
) {
    colors.insert(s.clone(), Color::Cyan);
    blue_path.push(s);
}

/// Red DFS searching for any state currently on the blue stack
/// (`Color::Cyan`). Returns the witness state plus the trail from `seed`
/// to the witness when found.
fn red_dfs<G: LivenessGraph>(
    graph: &G,
    seed: &G::State,
    colors: &mut HashMap<G::State, Color>,
    blue_path: &[G::State],
) -> Option<(G::State, Vec<G::State>)> {
    // Iterative DFS. The trail is the path from seed → current. When we
    // see a Cyan successor we report (cyan, trail+cyan).
    let blue_set: HashSet<&G::State> = blue_path.iter().collect();

    // Frame: (state, successors, idx). We track `trail` separately so we
    // can return it when we hit a Cyan node.
    let seed_succs = graph.successors(seed);
    let mut stack: Vec<(G::State, Vec<G::State>, usize)> =
        vec![(seed.clone(), seed_succs, 0)];
    let mut trail: Vec<G::State> = vec![seed.clone()];

    while let Some((v, succs, mut i)) = stack.pop() {
        // Re-establish trail: pop until trail's tail == v.
        while trail.last() != Some(&v) {
            trail.pop();
        }
        let mut recursed = false;
        while i < succs.len() {
            let w = succs[i].clone();
            i += 1;
            // Detect cyan back-edge — accepting cycle witness.
            if blue_set.contains(&w) {
                let mut tail = trail.clone();
                // The cycle goes seed → ... → v → w; trail[0] == seed
                // and seed must equal blue_path's tail. The lasso back to
                // an earlier blue-stack node `w` proves the cycle.
                tail.push(w.clone());
                return Some((w, tail[1..].to_vec()));
            }
            // Check color; recurse on non-Red.
            let cw = colors.get(&w).copied().unwrap_or(Color::White);
            if cw == Color::Red {
                continue;
            }
            colors.insert(w.clone(), Color::Red);
            let w_succs = graph.successors(&w);
            stack.push((v.clone(), succs, i));
            trail.push(w.clone());
            stack.push((w, w_succs, 0));
            recursed = true;
            break;
        }
        if recursed {
            continue;
        }
        // Done with v; pop trail.
        if trail.last() == Some(&v) {
            trail.pop();
        }
    }
    None
}

// -------- Adapter: run streaming nested-DFS over an existing fingerprint
// adjacency map. Used by the `--liveness-streaming` runtime path as the
// validation harness. --------

/// Wrap an in-memory fingerprint adjacency map plus an "is-accepting"
/// predicate as a `LivenessGraph`. This is the staging ground for the
/// production `--liveness-streaming` mode: we keep the existing exploration
/// path that builds the adjacency map, then run nested-DFS instead of
/// Tarjan-then-fairness-check. The next step (described in the design doc)
/// is to lift the DFS into the worker loop and drop the adjacency map
/// entirely.
pub struct FingerprintAdjacencyGraph<'a> {
    pub initial_fps: Vec<u64>,
    pub adjacency: &'a HashMap<u64, Vec<u64>>,
    pub accepting: &'a dyn Fn(u64) -> bool,
}

impl<'a> LivenessGraph for FingerprintAdjacencyGraph<'a> {
    type State = u64;

    fn initial_states(&self) -> Vec<u64> {
        self.initial_fps.clone()
    }

    fn successors(&self, state: &u64) -> Vec<u64> {
        self.adjacency.get(state).cloned().unwrap_or_default()
    }

    fn is_accepting(&self, state: &u64) -> bool {
        (self.accepting)(*state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy graph: single accepting cycle 0 → 1 → 2 → 0, all states
    /// non-accepting EXCEPT state 2. Nested DFS must find a cycle.
    struct SmallCycleGraph;
    impl LivenessGraph for SmallCycleGraph {
        type State = i32;
        fn initial_states(&self) -> Vec<i32> {
            vec![0]
        }
        fn successors(&self, state: &i32) -> Vec<i32> {
            match state {
                0 => vec![1],
                1 => vec![2],
                2 => vec![0],
                _ => vec![],
            }
        }
        fn is_accepting(&self, state: &i32) -> bool {
            *state == 2
        }
    }

    #[test]
    fn nested_dfs_finds_accepting_cycle() {
        let r = nested_dfs(&SmallCycleGraph);
        match r {
            NestedDfsResult::AcceptingCycle { cycle, .. } => {
                // Cycle should include state 2 (the accepting one).
                assert!(cycle.contains(&2), "cycle must include accepting state");
            }
            NestedDfsResult::NoAcceptingCycle => {
                panic!("expected accepting cycle, got NoAcceptingCycle");
            }
        }
    }

    /// Cycle exists but no accepting state on it: should NOT report.
    struct CycleNoAccept;
    impl LivenessGraph for CycleNoAccept {
        type State = i32;
        fn initial_states(&self) -> Vec<i32> {
            vec![0]
        }
        fn successors(&self, state: &i32) -> Vec<i32> {
            match state {
                0 => vec![1],
                1 => vec![0],
                _ => vec![],
            }
        }
        fn is_accepting(&self, _state: &i32) -> bool {
            false
        }
    }

    #[test]
    fn nested_dfs_no_accepting_state_returns_clean() {
        let r = nested_dfs(&CycleNoAccept);
        assert!(matches!(r, NestedDfsResult::NoAcceptingCycle));
    }

    /// Pure DAG: should return NoAcceptingCycle even with accepting
    /// states.
    struct AcceptingDag;
    impl LivenessGraph for AcceptingDag {
        type State = i32;
        fn initial_states(&self) -> Vec<i32> {
            vec![0]
        }
        fn successors(&self, state: &i32) -> Vec<i32> {
            match state {
                0 => vec![1, 2],
                1 => vec![3],
                2 => vec![3],
                3 => vec![],
                _ => vec![],
            }
        }
        fn is_accepting(&self, _state: &i32) -> bool {
            true
        }
    }

    #[test]
    fn nested_dfs_dag_no_cycle_even_if_all_accepting() {
        let r = nested_dfs(&AcceptingDag);
        assert!(matches!(r, NestedDfsResult::NoAcceptingCycle));
    }

    /// Two disjoint components, only one has an accepting cycle.
    struct TwoComponents;
    impl LivenessGraph for TwoComponents {
        type State = i32;
        fn initial_states(&self) -> Vec<i32> {
            vec![0, 10]
        }
        fn successors(&self, state: &i32) -> Vec<i32> {
            match state {
                0 => vec![1],
                1 => vec![0], // cycle, no accept
                10 => vec![11],
                11 => vec![12],
                12 => vec![10], // accepting cycle
                _ => vec![],
            }
        }
        fn is_accepting(&self, state: &i32) -> bool {
            *state == 12
        }
    }

    #[test]
    fn nested_dfs_finds_cycle_in_second_component() {
        let r = nested_dfs(&TwoComponents);
        match r {
            NestedDfsResult::AcceptingCycle { cycle, .. } => {
                assert!(cycle.contains(&12), "cycle must include accepting state 12");
            }
            _ => panic!("expected accepting cycle in second component"),
        }
    }

    /// Self-loop on an accepting state: trivial accepting cycle.
    struct SelfLoopAccept;
    impl LivenessGraph for SelfLoopAccept {
        type State = i32;
        fn initial_states(&self) -> Vec<i32> {
            vec![0]
        }
        fn successors(&self, state: &i32) -> Vec<i32> {
            match state {
                0 => vec![1],
                1 => vec![1], // self-loop
                _ => vec![],
            }
        }
        fn is_accepting(&self, state: &i32) -> bool {
            *state == 1
        }
    }

    #[test]
    fn nested_dfs_self_loop_on_accept_reports() {
        let r = nested_dfs(&SelfLoopAccept);
        match r {
            NestedDfsResult::AcceptingCycle { cycle, .. } => {
                assert!(cycle.contains(&1));
            }
            _ => panic!("self-loop on accepting state must be flagged"),
        }
    }

    /// Deep chain with no cycle: stress test stack safety of the
    /// iterative implementation. (Same kind of regression we fixed in
    /// Tarjan via T10.)
    #[test]
    fn nested_dfs_deep_chain_no_stack_overflow() {
        struct DeepChain {
            n: i32,
        }
        impl LivenessGraph for DeepChain {
            type State = i32;
            fn initial_states(&self) -> Vec<i32> {
                vec![0]
            }
            fn successors(&self, state: &i32) -> Vec<i32> {
                if *state + 1 < self.n {
                    vec![state + 1]
                } else {
                    vec![]
                }
            }
            fn is_accepting(&self, _state: &i32) -> bool {
                true // every state accepting; no cycle so still clean
            }
        }
        let g = DeepChain { n: 200_000 };
        let r = nested_dfs(&g);
        assert!(matches!(r, NestedDfsResult::NoAcceptingCycle));
    }

    /// Adapter test: feed the same SmallCycleGraph through the
    /// FingerprintAdjacencyGraph wrapper, simulating the runtime
    /// validation harness path.
    #[test]
    fn nested_dfs_via_fingerprint_adjacency_adapter() {
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(0, vec![1]);
        adj.insert(1, vec![2]);
        adj.insert(2, vec![0]);
        let accept = |s: u64| s == 2;
        let g = FingerprintAdjacencyGraph {
            initial_fps: vec![0],
            adjacency: &adj,
            accepting: &accept,
        };
        let r = nested_dfs(&g);
        match r {
            NestedDfsResult::AcceptingCycle { cycle, .. } => {
                assert!(cycle.contains(&2));
            }
            _ => panic!("expected accepting cycle via adapter"),
        }
    }

    /// Cross-validation: build a graph with an obvious accepting cycle
    /// AND a non-cycle accepting branch; nested-DFS must report exactly
    /// one cycle (not the dead-end branch).
    #[test]
    fn nested_dfs_only_reports_real_cycle_not_dead_end_accept() {
        // 0 -> 1 (accepting, dead end)
        // 0 -> 2 -> 3 -> 2 (cycle, 3 is accepting)
        struct G;
        impl LivenessGraph for G {
            type State = i32;
            fn initial_states(&self) -> Vec<i32> {
                vec![0]
            }
            fn successors(&self, state: &i32) -> Vec<i32> {
                match state {
                    0 => vec![1, 2],
                    1 => vec![],
                    2 => vec![3],
                    3 => vec![2],
                    _ => vec![],
                }
            }
            fn is_accepting(&self, state: &i32) -> bool {
                *state == 1 || *state == 3
            }
        }
        let r = nested_dfs(&G);
        match r {
            NestedDfsResult::AcceptingCycle { cycle, .. } => {
                assert!(cycle.contains(&3) && (cycle.contains(&2)));
                assert!(!cycle.contains(&1), "dead-end accept must not appear");
            }
            _ => panic!("expected to find the 2-3 cycle"),
        }
    }
}
