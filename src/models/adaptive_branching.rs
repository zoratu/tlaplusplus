use crate::model::Model;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Adaptive branching model that progressively ramps up load
/// Monitors memory usage and backs off if needed
#[derive(Clone, Debug)]
pub struct AdaptiveBranchingModel {
    max_depth: u32,
    current_branching_factor: Arc<AtomicU32>,
    min_branching: u32,
    max_branching: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ABState {
    path: Vec<u32>,
    value: u64,
    branching_at_level: u32,
}

impl AdaptiveBranchingModel {
    pub fn new(max_depth: u32, min_branching: u32, max_branching: u32) -> Self {
        Self {
            max_depth,
            current_branching_factor: Arc::new(AtomicU32::new(min_branching)),
            min_branching,
            max_branching,
        }
    }

    /// Increase branching factor (called when system is handling load well)
    pub fn ramp_up(&self) {
        let current = self.current_branching_factor.load(Ordering::Relaxed);
        if current < self.max_branching {
            // Increase by 10% or at least 5
            let increase = ((current as f64 * 0.1) as u32).max(5);
            let new_val = (current + increase).min(self.max_branching);
            self.current_branching_factor
                .store(new_val, Ordering::Relaxed);
            eprintln!("🔼 Ramping up: branching factor {} → {}", current, new_val);
        }
    }

    /// Decrease branching factor (called when memory pressure detected)
    pub fn back_off(&self) {
        let current = self.current_branching_factor.load(Ordering::Relaxed);
        if current > self.min_branching {
            // Decrease by 20% or at least 10 (more aggressive backoff)
            let decrease = ((current as f64 * 0.2) as u32).max(10);
            let new_val = current.saturating_sub(decrease).max(self.min_branching);
            self.current_branching_factor
                .store(new_val, Ordering::Relaxed);
            eprintln!(
                "🔽 Backing off: branching factor {} → {} (memory pressure)",
                current, new_val
            );
        }
    }

    pub fn current_branching(&self) -> u32 {
        self.current_branching_factor.load(Ordering::Relaxed)
    }
}

impl Model for AdaptiveBranchingModel {
    type State = ABState;

    fn name(&self) -> &'static str {
        "adaptive-branching"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        let initial_branching = self.current_branching_factor.load(Ordering::Relaxed);
        vec![ABState {
            path: vec![],
            value: 0,
            branching_at_level: initial_branching,
        }]
    }

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        if state.path.len() >= self.max_depth as usize {
            return;
        }

        // Use current branching factor (may have changed since state was created)
        let branching = self.current_branching_factor.load(Ordering::Relaxed);

        for i in 0..branching {
            let mut new_path = state.path.clone();
            new_path.push(i);

            // Add computational work
            let mut value = state.value;
            for _ in 0..100 {
                value = value.wrapping_mul(31).wrapping_add(i as u64);
            }

            out.push(ABState {
                path: new_path,
                value,
                branching_at_level: branching,
            });
        }
    }

    fn check_invariants(&self, _state: &Self::State) -> Result<(), String> {
        Ok(())
    }
}
