use crate::fairness::{FairnessConstraint, LabeledTransition, TarjanSCC, check_fairness_on_scc};
use crate::tla::{EvalContext, TemporalFormula, TlaValue, eval_expr};
use anyhow::{Result, anyhow};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Liveness property checker for temporal formulas
///
/// This checker evaluates temporal properties over execution paths/traces.
/// It supports:
/// - [] P (always P)
/// - <> P (eventually P)
/// - []<> P (infinitely often P)
/// - <>[] P (eventually always P)
/// - P ~> Q (leads-to)
pub struct LivenessChecker {
    /// Temporal properties to check
    properties: Vec<(String, TemporalFormula)>,
}

impl LivenessChecker {
    pub fn new(properties: Vec<(String, TemporalFormula)>) -> Self {
        Self { properties }
    }

    /// Check if all liveness properties hold for a finite trace
    ///
    /// Note: This can only definitively prove violations for some properties.
    /// Properties like <>P may require examining infinite paths or cycles.
    pub fn check_finite_trace<S, F>(&self, trace: &[S], eval_predicate: &F) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        for (name, formula) in &self.properties {
            if let Err(msg) = self.check_formula_on_trace(formula, trace, eval_predicate) {
                return Err(format!("Property '{}': {}", name, msg));
            }
        }
        Ok(())
    }

    /// Check a single temporal formula on a trace
    fn check_formula_on_trace<S, F>(
        &self,
        formula: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        match formula {
            TemporalFormula::StatePredicate(expr) => {
                // Evaluate the predicate on all states in the trace
                // For now, we expect trace to have exactly one state when checking predicates
                if trace.is_empty() {
                    return Err("Cannot check predicate on empty trace".to_string());
                }

                for state in trace {
                    match eval_predicate(state, expr) {
                        Ok(true) => continue,
                        Ok(false) => {
                            return Err(format!("Predicate '{}' evaluated to false", expr));
                        }
                        Err(e) => return Err(format!("Error evaluating '{}': {}", expr, e)),
                    }
                }

                Ok(())
            }

            TemporalFormula::Always(inner) => self.check_always(inner, trace, eval_predicate),

            TemporalFormula::Eventually(inner) => {
                self.check_eventually(inner, trace, eval_predicate)
            }

            TemporalFormula::InfinitelyOften(inner) => {
                self.check_infinitely_often(inner, trace, eval_predicate)
            }

            TemporalFormula::EventuallyAlways(inner) => {
                self.check_eventually_always(inner, trace, eval_predicate)
            }

            TemporalFormula::LeadsTo(p, q) => self.check_leads_to(p, q, trace, eval_predicate),

            TemporalFormula::And(left, right) => {
                self.check_formula_on_trace(left, trace, eval_predicate)?;
                self.check_formula_on_trace(right, trace, eval_predicate)
            }

            TemporalFormula::Or(left, right) => {
                let left_result = self.check_formula_on_trace(left, trace, eval_predicate);
                let right_result = self.check_formula_on_trace(right, trace, eval_predicate);

                match (left_result, right_result) {
                    (Ok(()), _) | (_, Ok(())) => Ok(()),
                    (Err(e1), Err(e2)) => Err(format!("Both disjuncts failed: {} and {}", e1, e2)),
                }
            }

            TemporalFormula::Not(inner) => {
                // Negation of temporal formula
                match self.check_formula_on_trace(inner, trace, eval_predicate) {
                    Ok(()) => Err("Negated formula holds".to_string()),
                    Err(_) => Ok(()),
                }
            }

            TemporalFormula::WeakFairness { .. } | TemporalFormula::StrongFairness { .. } => {
                // Fairness requires action/transition information, not just states
                // TODO: Implement when we have transition-level trace
                Ok(())
            }

            TemporalFormula::TemporalForAll { formula, .. } => {
                // \AA x: P - Universal quantification
                // For finite trace checking, we check that P holds for the formula
                // Note: Full semantic would require iterating over all values in domain,
                // but that's not feasible without runtime state information
                // For now, we just check the inner formula
                self.check_formula_on_trace(formula, trace, eval_predicate)
            }

            TemporalFormula::TemporalExists { formula, .. } => {
                // \EE x: P - Existential quantification
                // For finite trace checking, we check if P holds for the formula
                // Note: Full semantic would require finding at least one value in domain
                // For now, we just check the inner formula
                self.check_formula_on_trace(formula, trace, eval_predicate)
            }
        }
    }

    /// Check [] P (always P) - P must hold in every state
    fn check_always<S, F>(
        &self,
        inner: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        for (i, state) in trace.iter().enumerate() {
            if let Err(msg) = self.check_formula_on_trace(inner, &[state.clone()], eval_predicate) {
                return Err(format!("[] violated at position {}: {}", i, msg));
            }
        }
        Ok(())
    }

    /// Check <> P (eventually P) - P must hold in at least one state
    fn check_eventually<S, F>(
        &self,
        inner: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        // For finite traces, we can only prove violation if trace is complete
        // (i.e., ends in deadlock or cycle)
        for state in trace {
            if self
                .check_formula_on_trace(inner, &[state.clone()], eval_predicate)
                .is_ok()
            {
                return Ok(());
            }
        }

        // Property didn't hold in any state
        // This is only a definitive violation if the trace is complete
        Err("Property did not hold in any state of finite trace".to_string())
    }

    /// Check []<> P (infinitely often P) - P must hold infinitely many times
    fn check_infinitely_often<S, F>(
        &self,
        inner: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        // For finite traces, this requires cycle detection
        // We need to know if there's a cycle where P holds at least once

        // Simple check: count how many times P holds
        let mut count = 0;
        for state in trace {
            if self
                .check_formula_on_trace(inner, &[state.clone()], eval_predicate)
                .is_ok()
            {
                count += 1;
            }
        }

        // TODO: Proper cycle detection to verify P holds in cycle
        if count == 0 {
            return Err("Property never held in trace".to_string());
        }

        // For now, we can't definitively check without cycle info
        Ok(())
    }

    /// Check <>[] P (eventually always P) - P eventually becomes true and stays true
    fn check_eventually_always<S, F>(
        &self,
        inner: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        // Find a position where P becomes true and stays true
        for i in 0..trace.len() {
            let suffix = &trace[i..];
            let mut holds_everywhere = true;

            for state in suffix {
                if self
                    .check_formula_on_trace(inner, &[state.clone()], eval_predicate)
                    .is_err()
                {
                    holds_everywhere = false;
                    break;
                }
            }

            if holds_everywhere {
                return Ok(());
            }
        }

        Err("Property never stabilized to always true".to_string())
    }

    /// Check P ~> Q (leads-to) - whenever P holds, Q must eventually hold
    fn check_leads_to<S, F>(
        &self,
        p: &TemporalFormula,
        q: &TemporalFormula,
        trace: &[S],
        eval_predicate: &F,
    ) -> Result<(), String>
    where
        S: Clone + Debug + Eq + Hash,
        F: Fn(&S, &str) -> Result<bool>,
    {
        // Find all positions where P holds
        for i in 0..trace.len() {
            let state = &trace[i];

            // Check if P holds at this position
            if self
                .check_formula_on_trace(p, &[state.clone()], eval_predicate)
                .is_ok()
            {
                // P holds, so Q must eventually hold in the suffix
                let suffix = &trace[i..];
                let mut q_holds = false;

                for future_state in suffix {
                    if self
                        .check_formula_on_trace(q, &[future_state.clone()], eval_predicate)
                        .is_ok()
                    {
                        q_holds = true;
                        break;
                    }
                }

                if !q_holds {
                    return Err(format!(
                        "P held at position {} but Q never held afterwards",
                        i
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Cycle detector for finding strongly connected components in state graphs
///
/// Used for liveness checking to detect infinite loops and verify
/// properties like []<>P (infinitely often).
pub struct CycleDetector<S> {
    /// Visited states with their DFS number
    visited: HashMap<S, usize>,
    /// States on the current DFS stack
    on_stack: HashSet<S>,
    /// DFS counter
    counter: usize,
}

impl<S: Clone + Eq + Hash> CycleDetector<S> {
    pub fn new() -> Self {
        Self {
            visited: HashMap::new(),
            on_stack: HashSet::new(),
            counter: 0,
        }
    }

    /// Check if there's a cycle reachable from the given state
    ///
    /// Uses Nested DFS algorithm for cycle detection.
    /// Returns Some(cycle) if a cycle is found, None otherwise.
    pub fn find_cycle<F>(&mut self, start: &S, get_successors: F) -> Option<Vec<S>>
    where
        F: Fn(&S) -> Vec<S>,
    {
        let mut path = Vec::new();
        self.dfs(start, &get_successors, &mut path)
    }

    fn dfs<F>(&mut self, state: &S, get_successors: &F, path: &mut Vec<S>) -> Option<Vec<S>>
    where
        F: Fn(&S) -> Vec<S>,
    {
        if self.on_stack.contains(state) {
            // Found a back edge - extract cycle
            if let Some(pos) = path.iter().position(|s| s == state) {
                return Some(path[pos..].to_vec());
            }
        }

        if self.visited.contains_key(state) {
            return None;
        }

        self.visited.insert(state.clone(), self.counter);
        self.counter += 1;
        self.on_stack.insert(state.clone());
        path.push(state.clone());

        for successor in get_successors(state) {
            if let Some(cycle) = self.dfs(&successor, get_successors, path) {
                return Some(cycle);
            }
        }

        path.pop();
        self.on_stack.remove(state);
        None
    }
}

impl<S: Clone + Eq + Hash> Default for CycleDetector<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Büchi automaton checker for fairness constraints
///
/// This checker verifies fairness properties on the state graph by:
/// 1. Building a transition graph with action labels
/// 2. Finding strongly connected components (SCCs)
/// 3. Checking fairness constraints on each SCC
///
/// A fairness violation occurs when an SCC exists where:
/// - Weak fairness (WF): action is enabled infinitely often but never occurs
/// - Strong fairness (SF): action is continuously enabled but never occurs
pub struct BuchiChecker<S> {
    /// Fairness constraints to check
    constraints: Vec<FairnessConstraint>,
    /// State transition graph
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Eq + Hash + Debug> BuchiChecker<S> {
    pub fn new(constraints: Vec<FairnessConstraint>) -> Self {
        Self {
            constraints,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check fairness constraints on a labeled transition system
    ///
    /// This analyzes the transition graph for fairness violations by:
    /// 1. Finding all SCCs (cycles) in the graph
    /// 2. For each SCC, checking that all fairness constraints are satisfied
    ///
    /// Returns Ok(()) if all constraints are satisfied, Err otherwise.
    pub fn check_fairness(&self, states: &[S], transitions: &[LabeledTransition<S>]) -> Result<()> {
        if self.constraints.is_empty() {
            return Ok(()); // No fairness constraints to check
        }

        // Build adjacency information for SCC detection
        let mut adjacency: HashMap<&S, Vec<&S>> = HashMap::new();
        for state in states {
            adjacency.insert(state, Vec::new());
        }
        for trans in transitions {
            adjacency
                .entry(&trans.from)
                .or_insert_with(Vec::new)
                .push(&trans.to);
        }

        // Find all strongly connected components
        let get_successors =
            |state: &&S| -> Vec<&S> { adjacency.get(state).cloned().unwrap_or_default() };

        let mut tarjan = TarjanSCC::new();
        let sccs = tarjan.find_sccs(&states.iter().collect::<Vec<_>>(), get_successors);

        // Check fairness on each non-trivial SCC (cycles)
        for scc in sccs {
            if scc.len() > 1 || self_loop_exists(&scc, transitions) {
                // This is a cycle - check fairness constraints
                let scc_states: Vec<S> = scc.iter().map(|&s| s.clone()).collect();

                for constraint in &self.constraints {
                    check_fairness_on_scc(&scc_states, constraint, transitions)?;
                }
            }
        }

        Ok(())
    }

    /// Check fairness with a predicate for action enablement
    ///
    /// This is a more precise check that uses a user-provided function
    /// to determine if an action is enabled in a state.
    pub fn check_fairness_with_enablement<F>(
        &self,
        states: &[S],
        transitions: &[LabeledTransition<S>],
        is_enabled: F,
    ) -> Result<()>
    where
        F: Fn(&S, &str) -> bool,
    {
        if self.constraints.is_empty() {
            return Ok(());
        }

        // Build adjacency information
        let mut adjacency: HashMap<&S, Vec<&S>> = HashMap::new();
        for state in states {
            adjacency.insert(state, Vec::new());
        }
        for trans in transitions {
            adjacency
                .entry(&trans.from)
                .or_insert_with(Vec::new)
                .push(&trans.to);
        }

        // Find SCCs
        let get_successors =
            |state: &&S| -> Vec<&S> { adjacency.get(state).cloned().unwrap_or_default() };

        let mut tarjan = TarjanSCC::new();
        let sccs = tarjan.find_sccs(&states.iter().collect::<Vec<_>>(), get_successors);

        // Check each non-trivial SCC
        for scc in sccs {
            if scc.len() > 1 || self_loop_exists(&scc, transitions) {
                let scc_states: Vec<S> = scc.iter().map(|&s| s.clone()).collect();

                for constraint in &self.constraints {
                    self.check_constraint_on_scc(
                        &scc_states,
                        constraint,
                        transitions,
                        &is_enabled,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Check a single fairness constraint on an SCC with enablement checking
    fn check_constraint_on_scc<F>(
        &self,
        scc: &[S],
        constraint: &FairnessConstraint,
        transitions: &[LabeledTransition<S>],
        is_enabled: &F,
    ) -> Result<()>
    where
        F: Fn(&S, &str) -> bool,
    {
        let action_name = constraint.action_name();

        // Check if action occurs in this SCC
        let action_occurs = transitions
            .iter()
            .any(|t| scc.contains(&t.from) && scc.contains(&t.to) && t.action.name == action_name);

        // Check if action is enabled in any state of the SCC
        let action_enabled = scc.iter().any(|state| is_enabled(state, action_name));

        // Fairness violation: enabled but doesn't occur
        if action_enabled && !action_occurs {
            return Err(anyhow!(
                "{} fairness violated: action '{}' is enabled in SCC but never occurs",
                if constraint.is_weak() {
                    "Weak"
                } else {
                    "Strong"
                },
                action_name
            ));
        }

        Ok(())
    }
}

/// Check if a state has a self-loop in the transition system
fn self_loop_exists<S: Clone + Eq + Hash>(
    scc: &[&S],
    transitions: &[LabeledTransition<S>],
) -> bool {
    if scc.len() != 1 {
        return false;
    }

    let state = scc[0];
    transitions
        .iter()
        .any(|t| &t.from == state && &t.to == state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_eval(state: &i32, expr: &str) -> Result<bool> {
        match expr {
            "x > 0" => Ok(*state > 0),
            "x = 5" => Ok(*state == 5),
            "x > 5" => Ok(*state > 5),
            "x < 10" => Ok(*state < 10),
            _ => Ok(false),
        }
    }

    #[test]
    fn test_always_property() {
        let checker = LivenessChecker::new(vec![]);
        let trace = vec![1, 2, 3, 4, 5];

        let formula = TemporalFormula::Always(Box::new(TemporalFormula::StatePredicate(
            "x > 0".to_string(),
        )));

        assert!(
            checker
                .check_formula_on_trace(&formula, &trace, &simple_eval)
                .is_ok()
        );

        let failing_trace = vec![1, 2, 0, 4];
        assert!(
            checker
                .check_formula_on_trace(&formula, &failing_trace, &simple_eval)
                .is_err()
        );
    }

    #[test]
    fn test_eventually_property() {
        let checker = LivenessChecker::new(vec![]);
        let trace = vec![1, 2, 3, 5, 6];

        let formula = TemporalFormula::Eventually(Box::new(TemporalFormula::StatePredicate(
            "x = 5".to_string(),
        )));

        assert!(
            checker
                .check_formula_on_trace(&formula, &trace, &simple_eval)
                .is_ok()
        );

        let failing_trace = vec![1, 2, 3, 4];
        assert!(
            checker
                .check_formula_on_trace(&formula, &failing_trace, &simple_eval)
                .is_err()
        );
    }

    #[test]
    fn test_leads_to() {
        let checker = LivenessChecker::new(vec![]);
        let trace = vec![1, 5, 6, 7];

        let formula = TemporalFormula::LeadsTo(
            Box::new(TemporalFormula::StatePredicate("x = 5".to_string())),
            Box::new(TemporalFormula::StatePredicate("x > 5".to_string())),
        );

        assert!(
            checker
                .check_formula_on_trace(&formula, &trace, &simple_eval)
                .is_ok()
        );
    }

    #[test]
    fn test_cycle_detector() {
        let mut detector = CycleDetector::new();

        // Graph: 0 -> 1 -> 2 -> 1 (cycle)
        let get_successors = |state: &i32| -> Vec<i32> {
            match state {
                0 => vec![1],
                1 => vec![2],
                2 => vec![1],
                _ => vec![],
            }
        };

        let cycle = detector.find_cycle(&0, get_successors);
        assert!(cycle.is_some());
        assert!(cycle.unwrap().contains(&1));
    }

    #[test]
    fn test_buichi_checker_fairness_satisfied() {
        use crate::fairness::{ActionLabel, FairnessConstraint, LabeledTransition};

        // Create a simple cycle: 0 -> 1 -> 2 -> 0
        // with action "Next" occurring on all transitions
        let states = vec![0, 1, 2];
        let transitions = vec![
            LabeledTransition {
                from: 0,
                to: 1,
                action: ActionLabel::new("Next"),
            },
            LabeledTransition {
                from: 1,
                to: 2,
                action: ActionLabel::new("Next"),
            },
            LabeledTransition {
                from: 2,
                to: 0,
                action: ActionLabel::new("Next"),
            },
        ];

        let constraint = FairnessConstraint::Weak {
            vars: vec!["x".to_string()],
            action: "Next".to_string(),
        };

        let checker = BuchiChecker::new(vec![constraint]);

        // Should pass: action occurs in the cycle
        assert!(checker.check_fairness(&states, &transitions).is_ok());
    }

    #[test]
    fn test_buichi_checker_fairness_violated() {
        use crate::fairness::{ActionLabel, FairnessConstraint, LabeledTransition};

        // Create a cycle: 0 -> 1 -> 0
        // with only "ActionA" occurring, but fairness requires "ActionB"
        let states = vec![0, 1];
        let transitions = vec![
            LabeledTransition {
                from: 0,
                to: 1,
                action: ActionLabel::new("ActionA"),
            },
            LabeledTransition {
                from: 1,
                to: 0,
                action: ActionLabel::new("ActionA"),
            },
        ];

        let constraint = FairnessConstraint::Weak {
            vars: vec!["x".to_string()],
            action: "ActionB".to_string(),
        };

        let checker = BuchiChecker::new(vec![constraint]);

        // Should fail: ActionB doesn't occur but might be enabled
        // (conservatively assumes it's enabled)
        assert!(checker.check_fairness(&states, &transitions).is_err());
    }

    #[test]
    fn test_buichi_checker_with_enablement() {
        use crate::fairness::{ActionLabel, FairnessConstraint, LabeledTransition};

        // Cycle: 0 -> 1 -> 0
        let states = vec![0, 1];
        let transitions = vec![
            LabeledTransition {
                from: 0,
                to: 1,
                action: ActionLabel::new("Inc"),
            },
            LabeledTransition {
                from: 1,
                to: 0,
                action: ActionLabel::new("Inc"),
            },
        ];

        let wf = FairnessConstraint::Weak {
            vars: vec!["x".to_string()],
            action: "Dec".to_string(),
        };

        let checker = BuchiChecker::new(vec![wf]);

        // Dec is never enabled, so fairness is satisfied
        let is_enabled = |_state: &i32, action: &str| action != "Dec";

        assert!(
            checker
                .check_fairness_with_enablement(&states, &transitions, is_enabled)
                .is_ok()
        );

        // Dec is enabled but never occurs - fairness violated
        let always_enabled = |_state: &i32, _action: &str| true;

        assert!(
            checker
                .check_fairness_with_enablement(&states, &transitions, always_enabled)
                .is_err()
        );
    }
}
