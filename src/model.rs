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
}
