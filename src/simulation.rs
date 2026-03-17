//! Simulation mode: random trace exploration (like TLC's -simulate).
//!
//! Instead of exhaustive BFS, randomly picks one initial state and one successor
//! at each step, checking invariants along the way. Much faster than BFS for
//! finding bugs in large state spaces.

use crate::model::Model;
use crate::runtime::{PropertyType, Violation};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Configuration for simulation mode.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Maximum depth per trace (default 100).
    pub depth: usize,
    /// Number of traces to run (default 1000).
    pub num_traces: usize,
    /// Random seed (0 = use system entropy via time).
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            depth: 100,
            num_traces: 1000,
            seed: 0,
        }
    }
}

/// Result of a simulation run.
#[derive(Debug)]
pub struct SimulationOutcome<S> {
    pub traces_run: usize,
    pub max_depth_reached: usize,
    pub total_states: u64,
    pub duration: Duration,
    pub violations: Vec<Violation<S>>,
}

/// Simple xoshiro256** PRNG (no external dependency needed).
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed xoshiro
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Run simulation mode: random trace exploration.
///
/// For each trace, picks a random initial state and then repeatedly picks
/// a random successor, checking invariants at each step.
pub fn run_simulation<M>(
    model: &M,
    config: &SimulationConfig,
    max_violations: usize,
) -> SimulationOutcome<M::State>
where
    M: Model,
    M::State: Clone + Debug,
{
    let started_at = Instant::now();
    let initial_states = model.initial_states();
    if initial_states.is_empty() {
        return SimulationOutcome {
            traces_run: 0,
            max_depth_reached: 0,
            total_states: 0,
            duration: started_at.elapsed(),
            violations: vec![],
        };
    }

    let seed = if config.seed == 0 {
        // Use time-based seed
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    } else {
        config.seed
    };
    let mut rng = Rng::new(seed);

    let mut violations = Vec::new();
    let mut max_depth_reached = 0usize;
    let mut total_states = 0u64;
    let mut successors = Vec::with_capacity(64);
    let mut traces_completed = 0usize;

    for trace_idx in 0..config.num_traces {
        if violations.len() >= max_violations {
            break;
        }
        traces_completed = trace_idx + 1;

        // Pick a random initial state
        let init_idx = rng.range(initial_states.len());
        let mut current = initial_states[init_idx].clone();
        let mut trace = vec![current.clone()];

        // Check invariant on initial state
        if let Err(message) = model.check_invariants(&current) {
            violations.push(Violation {
                message,
                state: current,
                property_type: PropertyType::Safety,
                trace,
            });
            total_states += 1;
            if violations.len() >= max_violations {
                break;
            }
            continue;
        }

        for step in 0..config.depth {
            successors.clear();
            model.next_states(&current, &mut successors);
            total_states += 1;

            if successors.is_empty() {
                // Deadlocked state - end this trace
                if step + 1 > max_depth_reached {
                    max_depth_reached = step + 1;
                }
                break;
            }

            // Pick a random successor
            let next_idx = rng.range(successors.len());
            current = successors.swap_remove(next_idx);
            trace.push(current.clone());

            // Check invariants
            if let Err(message) = model.check_invariants(&current) {
                violations.push(Violation {
                    message,
                    state: current.clone(),
                    property_type: PropertyType::Safety,
                    trace: trace.clone(),
                });
                if violations.len() >= max_violations {
                    break;
                }
                // Continue the trace even after violation in simulation mode
            }

            if step + 1 > max_depth_reached {
                max_depth_reached = step + 1;
            }
        }

        if traces_completed % 100 == 0 {
            eprintln!(
                "Simulation: {} traces completed, {} violations found",
                traces_completed,
                violations.len()
            );
        }
    }

    SimulationOutcome {
        traces_run: traces_completed,
        max_depth_reached,
        total_states,
        duration: started_at.elapsed(),
        violations,
    }
}
