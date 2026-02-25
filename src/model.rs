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
}
