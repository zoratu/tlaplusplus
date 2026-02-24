use crate::model::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct CounterGridModel {
    pub max_x: u32,
    pub max_y: u32,
    pub max_sum: u32,
}

impl CounterGridModel {
    pub fn new(max_x: u32, max_y: u32, max_sum: u32) -> Self {
        Self {
            max_x,
            max_y,
            max_sum,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterGridState {
    pub x: u32,
    pub y: u32,
}

impl Model for CounterGridModel {
    type State = CounterGridState;

    fn name(&self) -> &'static str {
        "counter-grid"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        vec![CounterGridState { x: 0, y: 0 }]
    }

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        if state.x < self.max_x {
            out.push(CounterGridState {
                x: state.x + 1,
                y: state.y,
            });
        }
        if state.y < self.max_y {
            out.push(CounterGridState {
                x: state.x,
                y: state.y + 1,
            });
        }
        if state.x > 0 {
            out.push(CounterGridState {
                x: state.x - 1,
                y: state.y,
            });
        }
        if state.y > 0 {
            out.push(CounterGridState {
                x: state.x,
                y: state.y - 1,
            });
        }
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        let sum = state.x + state.y;
        if sum > self.max_sum {
            return Err(format!(
                "sum invariant violated: x + y = {} exceeds {}",
                sum, self.max_sum
            ));
        }
        Ok(())
    }
}
