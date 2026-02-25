use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::hash::Hash;

pub trait Model: Send + Sync + 'static {
    type State: Clone + Debug + Eq + Hash + Send + Sync + Serialize + DeserializeOwned + 'static;

    fn name(&self) -> &'static str;

    fn initial_states(&self) -> Vec<Self::State>;

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>);

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
