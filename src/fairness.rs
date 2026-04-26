use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// T10.3 — trivial-SCC pre-filter via iterative leaf trim.
///
/// A node `v` cannot belong to any non-trivial SCC if either
///
///   * `v` has no outgoing edge that stays inside the candidate set (a sink), or
///   * `v` has no incoming edge from inside the candidate set (a source),
///
/// **and** `v` does not have a self-loop. Iteratively peeling such nodes
/// shrinks the input to Tarjan to "states that genuinely participate in a
/// cycle." For state graphs that are mostly DAG-shaped (typical for
/// safety-only specs), trim removes nearly every node and Tarjan's input
/// shrinks dramatically. For one-giant-SCC graphs (the LivenessBench
/// shape) trim removes nothing — costs O(N + E) extra.
///
/// Returns the candidate node set and a boolean indicating whether any
/// trimming actually happened (so the caller can still hand the original
/// adjacency back to Tarjan when trim was a no-op, avoiding the O(N + E)
/// "filter to candidates" wrapper cost).
///
/// Self-loops are preserved: any node with `v -> v` is always a candidate
/// because it forms a non-trivial SCC by itself.
pub fn trivial_scc_prefilter(
    nodes: &[u64],
    adjacency: &HashMap<u64, Vec<u64>>,
) -> (HashSet<u64>, bool) {
    use std::collections::VecDeque;

    // Node fingerprints with self-loops are always candidates — they're
    // a non-trivial SCC by themselves. Compute up front so we never trim
    // them.
    let mut self_loops: HashSet<u64> = HashSet::new();
    for (&v, succs) in adjacency.iter() {
        if succs.iter().any(|&s| s == v) {
            self_loops.insert(v);
        }
    }

    // candidate set starts as all nodes; degrees count edges inside the
    // candidate set in both directions.
    let mut candidates: HashSet<u64> = nodes.iter().copied().collect();

    // out_deg[v] = |{w : v -> w, w in candidates}|
    // in_deg[v]  = |{w : w -> v, w in candidates}|
    let mut out_deg: HashMap<u64, usize> = HashMap::with_capacity(nodes.len());
    let mut in_deg: HashMap<u64, usize> = HashMap::with_capacity(nodes.len());
    let mut reverse: HashMap<u64, Vec<u64>> = HashMap::with_capacity(nodes.len());

    for &v in nodes {
        out_deg.entry(v).or_insert(0);
        in_deg.entry(v).or_insert(0);
    }
    for (&v, succs) in adjacency.iter() {
        for &w in succs {
            // count v -> w (skip self-loops; they don't help cycle detection
            // for the trim algorithm — we keep them via the self_loops set).
            if v == w {
                continue;
            }
            *out_deg.entry(v).or_insert(0) += 1;
            *in_deg.entry(w).or_insert(0) += 1;
            reverse.entry(w).or_insert_with(Vec::new).push(v);
        }
    }

    // Initial worklist: every node with zero in-degree or zero out-degree
    // (and no self-loop).
    let mut work: VecDeque<u64> = VecDeque::new();
    for &v in nodes {
        if self_loops.contains(&v) {
            continue;
        }
        let id = in_deg.get(&v).copied().unwrap_or(0);
        let od = out_deg.get(&v).copied().unwrap_or(0);
        if id == 0 || od == 0 {
            work.push_back(v);
        }
    }

    let mut trimmed_any = false;
    while let Some(v) = work.pop_front() {
        if !candidates.contains(&v) {
            continue;
        }
        if self_loops.contains(&v) {
            continue;
        }
        candidates.remove(&v);
        trimmed_any = true;

        // For each successor w of v: w lost an in-edge.
        if let Some(succs) = adjacency.get(&v) {
            for &w in succs {
                if w == v || !candidates.contains(&w) {
                    continue;
                }
                if let Some(d) = in_deg.get_mut(&w) {
                    *d = d.saturating_sub(1);
                    if *d == 0 && !self_loops.contains(&w) {
                        work.push_back(w);
                    }
                }
            }
        }
        // For each predecessor u of v: u lost an out-edge.
        if let Some(preds) = reverse.get(&v) {
            for &u in preds {
                if u == v || !candidates.contains(&u) {
                    continue;
                }
                if let Some(d) = out_deg.get_mut(&u) {
                    *d = d.saturating_sub(1);
                    if *d == 0 && !self_loops.contains(&u) {
                        work.push_back(u);
                    }
                }
            }
        }
    }

    (candidates, trimmed_any)
}

/// T10.4 — per-action transition shard.
///
/// Builds a `HashMap<action_name, Vec<(from_fp, to_fp)>>` so that each
/// fairness check can iterate only the transitions labelled with its
/// action, instead of every transition in the graph. For a spec with many
/// sub-action fairness constraints (e.g. `\A t \in Threads :
/// WF_vars(StealItem(t))`) this turns the per-constraint work from
/// `O(transitions)` into `O(transitions_for_this_action)`.
///
/// Wrapper-Next constraints (where the constraint targets the model's
/// wrapper next-action name) still need to scan everything, but that case
/// can be dispatched to a single "any in-SCC edge counts" check which
/// uses the original flattened triples or the adjacency map. The runtime
/// handles the wrapper-Next case directly without touching the shard
/// index.
///
/// Implementation note: action names are stored as owned `String` keys
/// because the source tx_triples vec owns the names; once the shard is
/// built, downstream callers borrow the keys.
pub type ActionShardIndex = HashMap<String, Vec<(u64, u64)>>;

pub fn build_action_shard_index(triples: &[(u64, u64, String)]) -> ActionShardIndex {
    // Avoid cloning the action name on every transition. We use the
    // raw_entry API: probe with &str, only clone the String key on the
    // first time we see this action.
    use std::collections::hash_map::Entry;
    let mut shards: ActionShardIndex = HashMap::with_capacity(16);
    for (from, to, name) in triples {
        // Cheap probe-then-insert pattern. The hash is computed once;
        // the key clone happens only when the action is new.
        match shards.get_mut(name) {
            Some(v) => v.push((*from, *to)),
            None => match shards.entry(name.clone()) {
                Entry::Occupied(mut o) => o.get_mut().push((*from, *to)),
                Entry::Vacant(v) => {
                    v.insert(vec![(*from, *to)]);
                }
            },
        }
    }
    shards
}

/// Wrapper-Next variant of [`check_fairness_on_scc_fp`] that uses the
/// adjacency map to short-circuit "any in-SCC edge exists". This is the
/// fast path for `WF_vars(Next)` style constraints.
pub fn scc_has_any_internal_edge(
    scc_fps: &HashSet<u64>,
    adjacency: &HashMap<u64, Vec<u64>>,
) -> bool {
    // For each node in the SCC, check if it has any successor inside the
    // SCC. As soon as we find one, we're done.
    for fp in scc_fps {
        if let Some(succs) = adjacency.get(fp) {
            for &w in succs {
                if scc_fps.contains(&w) {
                    return true;
                }
            }
        }
    }
    false
}

/// T10.4 — fairness check on SCC using a per-action shard index.
///
/// Iterates only the transitions for the constraint's action, instead of
/// the full transition list. Falls back to the wrapper-Next check (any
/// in-SCC edge) when the constraint targets the wrapper.
pub fn check_fairness_on_scc_fp_sharded(
    scc_fps: &HashSet<u64>,
    constraint: &FairnessConstraint,
    shards: &ActionShardIndex,
    adjacency: &HashMap<u64, Vec<u64>>,
    next_action_name: Option<&str>,
) -> Result<()> {
    let action_name = constraint.action_name();
    let constraint_is_wrapper_next = next_action_name.map(|n| n == action_name).unwrap_or(false);

    let action_occurs = if constraint_is_wrapper_next {
        // Wrapper Next: any in-SCC edge counts, regardless of label.
        scc_has_any_internal_edge(scc_fps, adjacency)
    } else if let Some(edges) = shards.get(action_name) {
        // Named action: iterate only this action's edges.
        edges
            .iter()
            .any(|(from, to)| scc_fps.contains(from) && scc_fps.contains(to))
    } else {
        // Action name not in any transition label at all → cannot occur.
        false
    };

    if !action_occurs && !scc_fps.is_empty() {
        return Err(anyhow!(
            "Fairness constraint may be violated: action '{}' does not occur in SCC",
            action_name
        ));
    }

    Ok(())
}

/// Action label for tracking which actions are taken in transitions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActionLabel {
    pub name: String,
    pub disjunct_index: Option<usize>,
}

impl ActionLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            disjunct_index: None,
        }
    }

    pub fn with_disjunct(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            disjunct_index: Some(index),
        }
    }
}

/// Labeled transition in the state graph
#[derive(Debug, Clone)]
pub struct LabeledTransition<S> {
    pub from: S,
    pub to: S,
    pub action: ActionLabel,
}

/// Fairness constraint specification
#[derive(Debug, Clone)]
pub enum FairnessConstraint {
    /// Weak fairness: if action is enabled infinitely often, it must occur infinitely often
    Weak { vars: Vec<String>, action: String },
    /// Strong fairness: if action is continuously enabled, it must occur infinitely often
    Strong { vars: Vec<String>, action: String },
}

impl FairnessConstraint {
    pub fn action_name(&self) -> &str {
        match self {
            FairnessConstraint::Weak { action, .. } => action,
            FairnessConstraint::Strong { action, .. } => action,
        }
    }

    pub fn is_weak(&self) -> bool {
        matches!(self, FairnessConstraint::Weak { .. })
    }

    pub fn is_strong(&self) -> bool {
        matches!(self, FairnessConstraint::Strong { .. })
    }
}

/// Tarjan's algorithm for finding strongly connected components
///
/// This is used to detect cycles in the state graph, which is necessary
/// for checking liveness properties and fairness constraints.
///
/// Implementation note: the SCC search is **iterative** (explicit DFS
/// stack) rather than recursive, so it does not blow the call stack on
/// state graphs with long deep paths (e.g., a million-state chain). Older
/// recursive Tarjans crashed with stack overflow at ~100K depth.
pub struct TarjanSCC<S> {
    index: usize,
    stack: Vec<S>,
    indices: HashMap<S, usize>,
    lowlinks: HashMap<S, usize>,
    on_stack: HashSet<S>,
    sccs: Vec<Vec<S>>,
}

impl<S: Clone + Eq + Hash + Debug> TarjanSCC<S> {
    pub fn new() -> Self {
        Self {
            index: 0,
            stack: Vec::new(),
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            on_stack: HashSet::new(),
            sccs: Vec::new(),
        }
    }

    /// Find all strongly connected components in the graph.
    ///
    /// Uses an iterative DFS to avoid stack overflow on deep graphs.
    pub fn find_sccs<F>(&mut self, nodes: &[S], get_successors: F) -> Vec<Vec<S>>
    where
        F: Fn(&S) -> Vec<S>,
    {
        self.sccs.clear();

        for node in nodes {
            if !self.indices.contains_key(node) {
                self.strongconnect_iterative(node.clone(), &get_successors);
            }
        }

        self.sccs.clone()
    }

    /// Iterative version of Tarjan's `strongconnect`.
    ///
    /// Each work-stack frame is `(node, successors, next_succ_idx)`. We
    /// "enter" a node by inserting its index/lowlink and pushing it on the
    /// SCC stack, then iterate its successors. When a recursive call would
    /// happen (unvisited successor) we push a new frame and resume on the
    /// next iteration. When we return from a child frame we update the
    /// parent's lowlink. When all successors are exhausted and the node is
    /// a root, we pop the SCC.
    fn strongconnect_iterative<F>(&mut self, root: S, get_successors: &F)
    where
        F: Fn(&S) -> Vec<S>,
    {
        // (node, successors, next_index_to_process)
        let mut work: Vec<(S, Vec<S>, usize)> = Vec::new();

        // "Enter" the root.
        self.enter_node(root.clone());
        let succs = get_successors(&root);
        work.push((root, succs, 0));

        while let Some((v, succs, mut i)) = work.pop() {
            let mut recursed = false;
            while i < succs.len() {
                let w = succs[i].clone();
                i += 1;
                if !self.indices.contains_key(&w) {
                    // Recurse on w. Re-push current frame at the new index
                    // (i has been incremented past w), then push the child
                    // frame which is the next thing to execute. We clone v
                    // because it's still needed in the parent frame; the
                    // clone is the price of going iterative.
                    self.enter_node(w.clone());
                    let w_succs = get_successors(&w);
                    work.push((v.clone(), succs, i));
                    work.push((w, w_succs, 0));
                    recursed = true;
                    break;
                } else if self.on_stack.contains(&w) {
                    // Cross/back edge into the current SCC.
                    let v_lowlink = *self.lowlinks.get(&v).unwrap();
                    let w_index = *self.indices.get(&w).unwrap();
                    self.lowlinks.insert(v.clone(), v_lowlink.min(w_index));
                }
            }

            if recursed {
                continue;
            }

            // All successors processed. If v is a root, pop its SCC.
            if self.lowlinks.get(&v) == self.indices.get(&v) {
                let mut scc = Vec::new();
                loop {
                    if let Some(w) = self.stack.pop() {
                        self.on_stack.remove(&w);
                        let is_root = w == v;
                        scc.push(w);
                        if is_root {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                self.sccs.push(scc);
            }

            // Returning to caller frame: propagate v's lowlink up to the parent.
            let v_lowlink = *self.lowlinks.get(&v).unwrap();
            if let Some((parent, _, _)) = work.last() {
                let parent_lowlink = *self.lowlinks.get(parent).unwrap();
                let new_low = parent_lowlink.min(v_lowlink);
                let parent_clone = parent.clone();
                self.lowlinks.insert(parent_clone, new_low);
            }
        }
    }

    fn enter_node(&mut self, v: S) {
        self.indices.insert(v.clone(), self.index);
        self.lowlinks.insert(v.clone(), self.index);
        self.index += 1;
        self.stack.push(v.clone());
        self.on_stack.insert(v);
    }
}

impl<S: Clone + Eq + Hash + Debug> Default for TarjanSCC<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a fairness constraint is satisfied on a strongly connected component
///
/// For weak fairness (WF): if action A is enabled infinitely often in the SCC,
/// then A must occur infinitely often (i.e., at least once in the SCC).
///
/// For strong fairness (SF): if action A is continuously enabled in the SCC,
/// then A must occur infinitely often (i.e., at least once in the SCC).
pub fn check_fairness_on_scc<S>(
    scc: &[S],
    constraint: &FairnessConstraint,
    transitions: &[LabeledTransition<S>],
) -> Result<()>
where
    S: Clone + Eq + Hash + Debug,
{
    check_fairness_on_scc_with_next(scc, constraint, transitions, None)
}

/// Fingerprint-keyed equivalent of [`check_fairness_on_scc_with_next`].
///
/// Callers that already have a u64-fingerprint view of the state graph
/// (e.g., the runtime's liveness post-processing) should use this. It
/// avoids the O(scc_size × transitions) cost of `scc.contains(&t.from)`
/// by accepting an `&HashSet<u64>` of in-SCC fingerprints (O(1)
/// membership) and an iterator over `(from_fp, to_fp, action_name)`
/// triples.
///
/// `scc_fps` MUST contain the fingerprints of every state in this SCC.
/// `transition_iter` is iterated once; each element is `(from_fp, to_fp,
/// action_name)`. The function counts an action as "occurring in the
/// SCC" if any in-SCC transition is labelled with the action name (or,
/// for the wrapper-Next case, any in-SCC transition at all).
pub fn check_fairness_on_scc_fp<'a, I>(
    scc_fps: &HashSet<u64>,
    constraint: &FairnessConstraint,
    transition_iter: I,
    next_action_name: Option<&str>,
) -> Result<()>
where
    I: IntoIterator<Item = (u64, u64, &'a str)>,
{
    let action_name = constraint.action_name();
    let constraint_is_wrapper_next = next_action_name.map(|n| n == action_name).unwrap_or(false);

    let mut action_occurs = false;
    for (from_fp, to_fp, label) in transition_iter {
        if scc_fps.contains(&from_fp) && scc_fps.contains(&to_fp) {
            if constraint_is_wrapper_next || label == action_name {
                action_occurs = true;
                break;
            }
        }
    }

    if !action_occurs && !scc_fps.is_empty() {
        return Err(anyhow!(
            "Fairness constraint may be violated: action '{}' does not occur in SCC",
            action_name
        ));
    }

    Ok(())
}

/// Check fairness on an SCC, with knowledge of the wrapper Next-action name.
///
/// `next_action_name` is the spec's top-level Next definition name (typically
/// `"Next"`). When the fairness constraint references the wrapper Next action
/// (e.g., `WF_vars(Next)`) rather than a specific named subaction, *every*
/// transition in the state graph counts as a Next step, so the constraint is
/// satisfied as long as any edge exists in the SCC.
///
/// Without this distinction, a single-state self-loop SCC (e.g., a `Terminated
/// /\ UNCHANGED vars` stutter that the user added explicitly to model
/// termination) would be incorrectly flagged as a Next-fairness violation,
/// because none of the labeled transitions carry the literal name `"Next"` —
/// the action labeller extracts the head identifier of each disjunct
/// (`Terminated`, `WorkerTakeItem`, `LoaderLoad`, ...).
pub fn check_fairness_on_scc_with_next<S>(
    scc: &[S],
    constraint: &FairnessConstraint,
    transitions: &[LabeledTransition<S>],
    next_action_name: Option<&str>,
) -> Result<()>
where
    S: Clone + Eq + Hash + Debug,
{
    let action_name = constraint.action_name();

    // If the fairness constraint targets the wrapper Next action itself, then
    // every transition in the SCC is, by definition, a Next step. The presence
    // of any edge in the SCC means Next has occurred.
    let constraint_is_wrapper_next = next_action_name.map(|n| n == action_name).unwrap_or(false);

    // Check if the action occurs in this SCC. Either the transition is labelled
    // with the action name directly, or the constraint is on the wrapper Next
    // (in which case any in-SCC transition counts).
    let action_occurs = transitions.iter().any(|t| {
        scc.contains(&t.from)
            && scc.contains(&t.to)
            && (t.action.name == action_name || constraint_is_wrapper_next)
    });

    // For both weak and strong fairness, if the action is enabled in the SCC,
    // it must occur at least once.
    //
    // Note: Proper fairness checking requires tracking enablement, which needs
    // evaluation of action guards. For now, we do a conservative check.

    if !action_occurs {
        // Check if action could be enabled in this SCC
        // If the SCC has any states, we conservatively assume the action could be enabled
        if !scc.is_empty() {
            return Err(anyhow!(
                "Fairness constraint may be violated: action '{}' does not occur in SCC",
                action_name
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tarjan_simple_cycle() {
        let mut tarjan = TarjanSCC::new();

        // Graph: 1 -> 2 -> 3 -> 1 (cycle)
        let nodes = vec![1, 2, 3];
        let get_successors = |n: &i32| -> Vec<i32> {
            match n {
                1 => vec![2],
                2 => vec![3],
                3 => vec![1],
                _ => vec![],
            }
        };

        let sccs = tarjan.find_sccs(&nodes, get_successors);

        // Should have one SCC containing all three nodes
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_tarjan_multiple_sccs() {
        let mut tarjan = TarjanSCC::new();

        // Graph: 1 -> 2 -> 3 -> 2 (2-3 form a cycle), 4 -> 5 (separate component)
        let nodes = vec![1, 2, 3, 4, 5];
        let get_successors = |n: &i32| -> Vec<i32> {
            match n {
                1 => vec![2],
                2 => vec![3],
                3 => vec![2],
                4 => vec![5],
                5 => vec![],
                _ => vec![],
            }
        };

        let sccs = tarjan.find_sccs(&nodes, get_successors);

        // Should have multiple SCCs
        assert!(sccs.len() >= 2);

        // Find the SCC containing 2 and 3
        let cycle_scc = sccs.iter().find(|scc| scc.contains(&2) && scc.contains(&3));
        assert!(cycle_scc.is_some());
        assert_eq!(cycle_scc.unwrap().len(), 2);
    }

    #[test]
    fn test_tarjan_iterative_handles_deep_chain_without_stack_overflow() {
        // T10 regression: the recursive Tarjan blew the call stack on deep
        // single-path chains around ~100K nodes. The iterative version must
        // handle 200K-node chains comfortably (default thread stack is 8 MB,
        // a recursive frame is ~200 bytes, ceiling ≈ 40K depth).
        const N: i32 = 200_000;
        let mut tarjan = TarjanSCC::new();
        let nodes: Vec<i32> = (0..N).collect();
        let get_successors = |n: &i32| -> Vec<i32> {
            // 0 -> 1 -> 2 -> ... -> N-1 (no cycles).
            if *n + 1 < N { vec![*n + 1] } else { vec![] }
        };
        let sccs = tarjan.find_sccs(&nodes, get_successors);
        // Each node is its own trivial SCC.
        assert_eq!(sccs.len(), N as usize);
    }

    #[test]
    fn test_tarjan_iterative_finds_giant_cycle() {
        // Same shape as the T10 benchmark spec: one large SCC where every
        // node is reachable from every other. Confirms the iterative
        // implementation still correctly detects the cycle (i.e., that the
        // lowlink propagation across the work-stack frames is correct).
        const N: i32 = 1_000;
        let mut tarjan = TarjanSCC::new();
        let nodes: Vec<i32> = (0..N).collect();
        let get_successors = |n: &i32| -> Vec<i32> {
            // Linear chain 0 -> 1 -> ... -> N-1, plus a back edge N-1 -> 0
            // making the entire graph one giant SCC.
            if *n + 1 < N { vec![*n + 1] } else { vec![0] }
        };
        let sccs = tarjan.find_sccs(&nodes, get_successors);
        assert_eq!(sccs.len(), 1, "expected exactly one SCC");
        assert_eq!(sccs[0].len(), N as usize);
    }

    #[test]
    fn test_check_fairness_on_scc_fp_action_present_passes() {
        // Action "Step" occurs on an in-SCC transition → pass.
        let scc_fps: HashSet<u64> = [1u64, 2, 3].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Step".to_string(),
        };
        let txs: Vec<(u64, u64, &str)> = vec![(1, 2, "Step"), (2, 3, "Step"), (3, 1, "Step")];
        assert!(check_fairness_on_scc_fp(&scc_fps, &constraint, txs.into_iter(), None).is_ok());
    }

    #[test]
    fn test_check_fairness_on_scc_fp_named_action_missing_fails() {
        // No transition labelled "Step" inside the SCC → fail.
        let scc_fps: HashSet<u64> = [1u64, 2].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Step".to_string(),
        };
        let txs: Vec<(u64, u64, &str)> = vec![(1, 2, "Other"), (2, 1, "Other")];
        assert!(check_fairness_on_scc_fp(&scc_fps, &constraint, txs.into_iter(), None).is_err());
    }

    #[test]
    fn test_check_fairness_on_scc_fp_wrapper_next_passes_on_any_edge() {
        // Constraint targets the wrapper Next action → any in-SCC edge counts.
        let scc_fps: HashSet<u64> = [1u64].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Next".to_string(),
        };
        // Self-loop edge labelled with a subaction name (not literally "Next").
        let txs: Vec<(u64, u64, &str)> = vec![(1, 1, "Terminated")];
        assert!(
            check_fairness_on_scc_fp(&scc_fps, &constraint, txs.into_iter(), Some("Next")).is_ok()
        );
    }

    #[test]
    fn test_check_fairness_on_scc_fp_ignores_out_of_scc_transitions() {
        // The action exists in the transition list but only on edges
        // touching nodes outside the SCC → must NOT count as occurring in
        // this SCC.
        let scc_fps: HashSet<u64> = [10u64, 11].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Step".to_string(),
        };
        // Transitions with from/to outside the SCC.
        let txs: Vec<(u64, u64, &str)> = vec![
            (1, 2, "Step"),    // outside SCC
            (10, 11, "Other"), // inside SCC, wrong action
            (11, 10, "Other"),
        ];
        assert!(check_fairness_on_scc_fp(&scc_fps, &constraint, txs.into_iter(), None).is_err());
    }

    // ---- T10.3 trivial-SCC pre-filter tests ----

    #[test]
    fn test_trim_dag_removes_everything() {
        // Linear DAG 1 -> 2 -> 3 -> 4: no cycles, every node should be
        // trimmed. Only 4 has out-degree 0; only 1 has in-degree 0. We
        // peel them, then 2 and 3 follow.
        let nodes = vec![1u64, 2, 3, 4];
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![3]);
        adj.insert(3, vec![4]);
        adj.insert(4, vec![]);
        let (cands, trimmed) = trivial_scc_prefilter(&nodes, &adj);
        assert!(trimmed);
        assert!(cands.is_empty(), "expected DAG nodes all trimmed");
    }

    #[test]
    fn test_trim_giant_cycle_keeps_everything() {
        // 1 -> 2 -> 3 -> 1: every node is in the cycle. Trim must keep
        // them all, since none has zero in/out within the cycle.
        let nodes = vec![1u64, 2, 3];
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![3]);
        adj.insert(3, vec![1]);
        let (cands, trimmed) = trivial_scc_prefilter(&nodes, &adj);
        assert!(!trimmed, "expected no trimming on giant cycle");
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn test_trim_preserves_self_loop() {
        // 1 -> 2 -> 2 (self-loop). 1 has out-edge but no in-edge → trim.
        // 2 has self-loop → keep (non-trivial SCC by itself).
        let nodes = vec![1u64, 2];
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![2]);
        let (cands, trimmed) = trivial_scc_prefilter(&nodes, &adj);
        assert!(trimmed);
        assert!(cands.contains(&2), "self-loop must survive trim");
        assert!(!cands.contains(&1));
    }

    #[test]
    fn test_trim_mixed_graph() {
        // 1 -> 2 -> 3 -> 2 (cycle 2-3) and 4 -> 5 (DAG tail).
        // Trim should drop {1, 4, 5} and keep {2, 3}.
        let nodes = vec![1u64, 2, 3, 4, 5];
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![3]);
        adj.insert(3, vec![2]);
        adj.insert(4, vec![5]);
        adj.insert(5, vec![]);
        let (cands, trimmed) = trivial_scc_prefilter(&nodes, &adj);
        assert!(trimmed);
        assert_eq!(cands.len(), 2);
        assert!(cands.contains(&2));
        assert!(cands.contains(&3));
    }

    // ---- T10.4 per-action transition shard tests ----

    #[test]
    fn test_action_shard_index_groups_by_name() {
        let triples = vec![
            (1u64, 2, "Step".to_string()),
            (2, 3, "Step".to_string()),
            (3, 1, "Reset".to_string()),
            (5, 6, "Other".to_string()),
        ];
        let shards = build_action_shard_index(&triples);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards.get("Step").unwrap().len(), 2);
        assert_eq!(shards.get("Reset").unwrap().len(), 1);
        assert_eq!(shards.get("Other").unwrap().len(), 1);
    }

    #[test]
    fn test_check_fairness_on_scc_fp_sharded_named_action() {
        // SCC = {1, 2, 3}, action "Step" occurs only outside SCC → fail.
        let scc_fps: HashSet<u64> = [1u64, 2, 3].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Step".to_string(),
        };
        let triples = vec![
            (10, 11, "Step".to_string()), // outside SCC
            (1, 2, "Other".to_string()),
            (2, 3, "Other".to_string()),
            (3, 1, "Other".to_string()),
        ];
        let shards = build_action_shard_index(&triples);
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![3]);
        adj.insert(3, vec![1]);
        adj.insert(10, vec![11]);
        let r = check_fairness_on_scc_fp_sharded(&scc_fps, &constraint, &shards, &adj, None);
        assert!(r.is_err());
    }

    #[test]
    fn test_check_fairness_on_scc_fp_sharded_passes_when_action_in_scc() {
        let scc_fps: HashSet<u64> = [1u64, 2, 3].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Step".to_string(),
        };
        let triples = vec![
            (1, 2, "Step".to_string()),
            (2, 3, "Step".to_string()),
            (3, 1, "Reset".to_string()),
        ];
        let shards = build_action_shard_index(&triples);
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![2]);
        adj.insert(2, vec![3]);
        adj.insert(3, vec![1]);
        let r = check_fairness_on_scc_fp_sharded(&scc_fps, &constraint, &shards, &adj, None);
        assert!(r.is_ok());
    }

    #[test]
    fn test_check_fairness_on_scc_fp_sharded_wrapper_next_uses_adjacency() {
        // Constraint targets "Next" wrapper; SCC has a self-loop edge
        // labelled with a subaction name → wrapper-Next must see the
        // edge and pass.
        let scc_fps: HashSet<u64> = [1u64].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Next".to_string(),
        };
        let triples = vec![(1u64, 1, "Terminated".to_string())];
        let shards = build_action_shard_index(&triples);
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        adj.insert(1, vec![1]);
        let r =
            check_fairness_on_scc_fp_sharded(&scc_fps, &constraint, &shards, &adj, Some("Next"));
        assert!(r.is_ok());
    }

    #[test]
    fn test_check_fairness_on_scc_fp_sharded_wrapper_next_fails_on_isolated_node() {
        // A truly isolated single-node SCC (no edges at all) must still
        // flag wrapper-Next as missing — there's no transition for Next
        // to occur on.
        let scc_fps: HashSet<u64> = [42u64].into_iter().collect();
        let constraint = FairnessConstraint::Weak {
            vars: vec!["v".to_string()],
            action: "Next".to_string(),
        };
        let shards: ActionShardIndex = HashMap::new();
        let adj: HashMap<u64, Vec<u64>> = HashMap::new();
        let r =
            check_fairness_on_scc_fp_sharded(&scc_fps, &constraint, &shards, &adj, Some("Next"));
        assert!(r.is_err());
    }

    #[test]
    fn test_action_label() {
        let label1 = ActionLabel::new("Next");
        assert_eq!(label1.name, "Next");
        assert_eq!(label1.disjunct_index, None);

        let label2 = ActionLabel::with_disjunct("Next", 0);
        assert_eq!(label2.name, "Next");
        assert_eq!(label2.disjunct_index, Some(0));
    }

    #[test]
    fn test_fairness_constraint() {
        let wf = FairnessConstraint::Weak {
            vars: vec!["x".to_string()],
            action: "Increment".to_string(),
        };

        assert!(wf.is_weak());
        assert!(!wf.is_strong());
        assert_eq!(wf.action_name(), "Increment");

        let sf = FairnessConstraint::Strong {
            vars: vec!["y".to_string()],
            action: "Decrement".to_string(),
        };

        assert!(!sf.is_weak());
        assert!(sf.is_strong());
        assert_eq!(sf.action_name(), "Decrement");
    }
}
