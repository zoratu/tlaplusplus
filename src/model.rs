use crate::fairness::FairnessConstraint;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::hash::Hash;

/// Action label for tracking which actions are taken in transitions
/// This is optional and only used by models that support fairness checking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActionLabel {
    pub name: String,
    pub disjunct_index: Option<usize>,
}

/// Labeled transition - a state transition with an action label
#[derive(Debug, Clone)]
pub struct LabeledTransition<S> {
    pub from: S,
    pub to: S,
    pub action: ActionLabel,
}

pub trait Model: Send + Sync + 'static {
    type State: Clone + Debug + Eq + Hash + Send + Sync + Serialize + DeserializeOwned + 'static;

    fn name(&self) -> &'static str;

    fn initial_states(&self) -> Vec<Self::State>;

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>);

    /// Generate labeled next states for fairness checking
    ///
    /// This is an optional method that models can implement to support fairness checking.
    /// If the model has fairness constraints, this method should return a vector of
    /// labeled transitions (state + action label). Otherwise, it returns None.
    ///
    /// Default implementation returns None, meaning the model doesn't support fairness.
    fn next_states_labeled(
        &self,
        _state: &Self::State,
    ) -> Option<Vec<LabeledTransition<Self::State>>> {
        None
    }

    /// Check if this model has fairness constraints that require labeled transitions
    ///
    /// Default implementation returns false. Models with fairness should override this.
    fn has_fairness_constraints(&self) -> bool {
        false
    }

    /// Return the fairness constraints for this model
    ///
    /// Default implementation returns an empty list. Models with fairness
    /// constraints (e.g. TlaModel with WF/SF specifications) should override this.
    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        Vec::new()
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String>;

    /// Check state constraints - returns Ok(()) if state should be explored, Err if pruned
    /// State constraints are used to prune the state space by excluding states that don't
    /// satisfy certain conditions. They're checked before adding a state to the exploration queue.
    fn check_state_constraints(&self, _state: &Self::State) -> Result<(), String> {
        // Default: no constraints, all states are valid
        Ok(())
    }

    /// Check action constraints - returns Ok(()) if transition should be allowed
    /// Action constraints are predicates on state transitions (current_state, next_state).
    /// They're checked before adding successor states to the queue.
    fn check_action_constraints(
        &self,
        _current: &Self::State,
        _next: &Self::State,
    ) -> Result<(), String> {
        // Default: no constraints, all transitions are valid
        Ok(())
    }

    /// Canonicalize a state under symmetry reduction.
    ///
    /// When a SYMMETRY set is configured, two states that differ only by a
    /// permutation of the symmetric elements are considered equivalent.
    /// This method maps every state to the canonical (lexicographically
    /// smallest) representative of its equivalence class.
    ///
    /// The runtime calls this on each successor state **before**
    /// fingerprinting and enqueueing, so that symmetric states are
    /// recognised as duplicates even when they differ structurally.
    ///
    /// Default implementation returns the state unchanged (no symmetry).
    fn canonicalize(&self, state: Self::State) -> Self::State {
        state
    }

    /// Compute fingerprint of a state
    ///
    /// This method allows models to customize how states are fingerprinted,
    /// which is critical for:
    /// - View functions: fingerprint only the "view" of the state
    /// - Symmetry reduction: fingerprint the canonical representative
    ///
    /// Default implementation uses standard hashing of the full state.
    fn fingerprint(&self, state: &Self::State) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        hasher.finish()
    }

    /// Return the number of Next action disjuncts for swarm testing.
    ///
    /// Swarm testing (Groce et al., ISSTA 2012) randomly disables subsets of
    /// Next disjuncts per simulation trace, creating configuration diversity
    /// that finds more bugs than the "use everything" approach.
    ///
    /// Returns 0 if the model does not support disjunct-level control.
    fn num_next_disjuncts(&self) -> usize {
        0
    }

    /// Generate next states using only the specified subset of disjuncts.
    ///
    /// `enabled_mask` contains the indices of disjuncts to evaluate (from
    /// `0..num_next_disjuncts()`). Only models that return nonzero from
    /// `num_next_disjuncts()` need to implement this.
    ///
    /// Default implementation ignores the mask and calls `next_states()`.
    fn next_states_swarm(
        &self,
        state: &Self::State,
        enabled_mask: &[usize],
        out: &mut Vec<Self::State>,
    ) {
        let _ = enabled_mask;
        self.next_states(state, out);
    }
}
