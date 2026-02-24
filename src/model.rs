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
}
