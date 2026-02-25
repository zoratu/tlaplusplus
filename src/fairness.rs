use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

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

    /// Find all strongly connected components in the graph
    pub fn find_sccs<F>(&mut self, nodes: &[S], get_successors: F) -> Vec<Vec<S>>
    where
        F: Fn(&S) -> Vec<S>,
    {
        self.sccs.clear();

        for node in nodes {
            if !self.indices.contains_key(node) {
                self.strongconnect(node.clone(), &get_successors);
            }
        }

        self.sccs.clone()
    }

    fn strongconnect<F>(&mut self, v: S, get_successors: &F)
    where
        F: Fn(&S) -> Vec<S>,
    {
        // Set depth index for v
        self.indices.insert(v.clone(), self.index);
        self.lowlinks.insert(v.clone(), self.index);
        self.index += 1;
        self.stack.push(v.clone());
        self.on_stack.insert(v.clone());

        // Consider successors of v
        for w in get_successors(&v) {
            if !self.indices.contains_key(&w) {
                // Successor w has not yet been visited; recurse on it
                self.strongconnect(w.clone(), get_successors);
                let v_lowlink = *self.lowlinks.get(&v).unwrap();
                let w_lowlink = *self.lowlinks.get(&w).unwrap();
                self.lowlinks.insert(v.clone(), v_lowlink.min(w_lowlink));
            } else if self.on_stack.contains(&w) {
                // Successor w is in stack and hence in the current SCC
                let v_lowlink = *self.lowlinks.get(&v).unwrap();
                let w_index = *self.indices.get(&w).unwrap();
                self.lowlinks.insert(v.clone(), v_lowlink.min(w_index));
            }
        }

        // If v is a root node, pop the stack and generate an SCC
        if self.lowlinks.get(&v) == self.indices.get(&v) {
            let mut scc = Vec::new();
            loop {
                if let Some(w) = self.stack.pop() {
                    self.on_stack.remove(&w);
                    scc.push(w.clone());
                    if w == v {
                        break;
                    }
                } else {
                    break;
                }
            }
            self.sccs.push(scc);
        }
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
    let action_name = constraint.action_name();

    // Check if the action occurs in this SCC
    let action_occurs = transitions
        .iter()
        .any(|t| scc.contains(&t.from) && scc.contains(&t.to) && t.action.name == action_name);

    // For both weak and strong fairness, if the action is enabled in the SCC,
    // it must occur at least once
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
