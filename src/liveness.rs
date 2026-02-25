use crate::tla::{EvalContext, TemporalFormula, TlaValue, eval_expr};
use anyhow::{Result, anyhow};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet, VecDeque};
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
}
