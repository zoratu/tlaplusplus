use crate::model::Model;
use serde::{Deserialize, Serialize};

/// High-branching synthetic model for stress testing parallel runtime
/// Each state generates many successors to keep workers busy
#[derive(Clone, Debug)]
pub struct HighBranchingModel {
    max_depth: u32,
    branching_factor: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct HBState {
    path: Vec<u32>,
    value: u64,
}

impl HighBranchingModel {
    pub fn new(max_depth: u32, branching_factor: u32) -> Self {
        Self {
            max_depth,
            branching_factor,
        }
    }
}

impl Model for HighBranchingModel {
    type State = HBState;

    fn name(&self) -> &'static str {
        "high-branching"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        vec![HBState {
            path: vec![],
            value: 0,
        }]
    }

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        if state.path.len() >= self.max_depth as usize {
            return;
        }

        // Generate many successors to keep workers busy
        for i in 0..self.branching_factor {
            let mut new_path = state.path.clone();
            new_path.push(i);

            // Add some computational work to simulate real model checking
            let mut value = state.value;
            for _ in 0..100 {
                value = value.wrapping_mul(31).wrapping_add(i as u64);
            }

            out.push(HBState {
                path: new_path,
                value,
            });
        }
    }

    fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_many_successors() {
        let model = HighBranchingModel::new(10, 16);
        let init = model.initial_states();
        assert_eq!(init.len(), 1);

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert_eq!(next.len(), 16);
    }
}
