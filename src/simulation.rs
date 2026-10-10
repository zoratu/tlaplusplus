//! Simulation mode: random trace exploration (like TLC's -simulate).
//!
//! Instead of exhaustive BFS, randomly picks one initial state and one successor
//! at each step, checking invariants along the way. Much faster than BFS for
//! finding bugs in large state spaces.
//!
//! ## Swarm Mode
//!
//! When `swarm` is enabled, each trace randomly selects a subset (50-90%) of
//! Next action disjuncts to enable. This implements "Swarm Testing" (Groce et
//! al., ISSTA 2012): by varying which actions are available per trace, the
//! simulation reaches diverse parts of the state space that the "all actions
//! enabled" approach rarely visits. This is especially effective for finding
//! bugs that only manifest when certain actions are disabled.

use crate::model::Model;
use crate::runtime::{PropertyType, Violation};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Strategy for weighting successor selection during the random walk.
///
/// Motivated by "When random isn't random enough" (ersc.io): random *selection*
/// is best expressed as an explicit weighted choice, and tuning those weights is
/// a powerful lever for steering state-space exploration.
///
/// - `Uniform` reproduces classic TLC `-simulate`: every enabled successor is
///   equiprobable.
/// - `LeastVisited` biases the walk toward successors the simulation has seen
///   least often, pushing traces into under-explored regions of the state space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SimWeighting {
    #[default]
    Uniform,
    LeastVisited,
}

/// Configuration for simulation mode.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Maximum depth per trace (default 100).
    pub depth: usize,
    /// Number of traces to run (default 1000).
    pub num_traces: usize,
    /// Random seed (0 = use system entropy via time).
    pub seed: u64,
    /// Enable swarm testing: randomly disable Next disjuncts per trace.
    pub swarm: bool,
    /// How to weight successor selection during the random walk.
    pub weighting: SimWeighting,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            depth: 100,
            num_traces: 1000,
            seed: 0,
            swarm: false,
            weighting: SimWeighting::Uniform,
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
    /// Per-trace swarm configs: which disjunct indices were enabled.
    /// Only populated when swarm mode is active.
    pub swarm_configs: Vec<Vec<usize>>,
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

    /// Uniform index in `[0, n)` with **no modulo bias**.
    ///
    /// The naive `next_u64() % n` over-weights the first `2^64 mod n` indices —
    /// the "classic blunder" from "When random isn't random enough" (ersc.io).
    /// We reject draws that fall in the biased tail `[0, 2^64 mod n)` so every
    /// index is equiprobable. For all accepted draws (`r >= 2^64 mod n`) the
    /// result is exactly `r % n`, so the RNG stream is unchanged except on the
    /// vanishing `~n/2^64` fraction that would otherwise have been biased.
    fn range(&mut self, n: usize) -> usize {
        #[cfg(not(feature = "verus"))]
        {
            self.range_u64(n as u64) as usize
        }
        #[cfg(feature = "verus")]
        {
            let r = self.next_u64();
            crate::storage::verus_smoke::compute_u64_index_mod(r, n)
        }
    }

    /// Unbiased uniform draw in `[0, n)` over `u64` (see [`Rng::range`]).
    fn range_u64(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0, "range_u64 requires n > 0");
        // reject == 2^64 mod n: `-n mod n` computes it without 128-bit math
        // (2^64 ≡ -n (mod 2^64), so wrapping_neg gives 2^64 - n).
        let reject = n.wrapping_neg() % n;
        loop {
            let r = self.next_u64();
            if r >= reject {
                return r % n;
            }
        }
    }

    /// Pick an index in `[0, weights.len())` with probability proportional to
    /// its integer weight — the `random_choice` primitive advocated in "When
    /// random isn't random enough" (ersc.io). Integer weights (rather than
    /// floats summing to 1.0) keep the distribution exact and easy to reason
    /// about. Zero-weight entries are never chosen; if every weight is zero we
    /// fall back to a uniform pick so the caller always makes progress.
    fn weighted(&mut self, weights: &[u32]) -> usize {
        debug_assert!(!weights.is_empty(), "weighted requires a non-empty slice");
        let total: u64 = weights.iter().map(|&w| u64::from(w)).sum();
        if total == 0 {
            return self.range(weights.len());
        }
        let mut target = self.range_u64(total);
        for (i, &w) in weights.iter().enumerate() {
            let w = u64::from(w);
            if target < w {
                return i;
            }
            target -= w;
        }
        // Unreachable while `target < total`; return the last positive-weight
        // index as a defensive fallback.
        weights.len() - 1
    }

    /// Select a random subset containing between `min_frac` and `max_frac` of
    /// `n` items. Returns a sorted Vec of selected indices.
    fn random_subset(&mut self, n: usize, min_frac: f64, max_frac: f64) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        // Pick a target count between min and max
        let raw_min_count = (n as f64 * min_frac).ceil() as usize;
        let raw_max_count = (n as f64 * max_frac).floor() as usize;
        #[cfg(not(feature = "verus"))]
        let min_count = raw_min_count.max(1);
        #[cfg(feature = "verus")]
        let min_count = crate::storage::verus_smoke::max_usize(raw_min_count, 1);
        #[cfg(not(feature = "verus"))]
        let max_count = raw_max_count.min(n);
        #[cfg(feature = "verus")]
        let max_count = crate::storage::verus_smoke::min_usize(raw_max_count, n);
        let target = if min_count >= max_count {
            min_count
        } else {
            min_count + self.range(max_count - min_count + 1)
        };

        // Fisher-Yates partial shuffle to select `target` items
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..target.min(n) {
            let j = i + self.range(n - i);
            indices.swap(i, j);
        }
        indices.truncate(target);
        indices.sort_unstable();
        indices
    }
}

/// Run simulation mode: random trace exploration.
///
/// For each trace, picks a random initial state and then repeatedly picks
/// a random successor, checking invariants at each step.
///
/// When `config.swarm` is true and the model supports disjunct-level control
/// (`num_next_disjuncts() > 1`), each trace randomly enables 50-90% of the
/// Next action disjuncts, creating diverse exploration configurations.
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
            swarm_configs: vec![],
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

    // Check if swarm mode is possible (need >1 disjunct)
    let num_disjuncts = model.num_next_disjuncts();
    let swarm_active = config.swarm && num_disjuncts > 1;

    if swarm_active {
        eprintln!(
            "Swarm mode: {} Next disjuncts detected, will randomly select 50-90% per trace",
            num_disjuncts
        );
    }

    let mut violations = Vec::new();
    let mut max_depth_reached = 0usize;
    let mut total_states = 0u64;
    let mut successors = Vec::with_capacity(64);
    let mut traces_completed = 0usize;
    let mut swarm_configs = Vec::new();
    // Successor-selection weighting state (LeastVisited only; stays empty and
    // cost-free for the default Uniform walk). `visit_counts` is keyed by the
    // model's deterministic fingerprint and spans all traces, so the bias steers
    // the whole simulation toward globally under-explored states.
    let weighting = config.weighting;
    let mut visit_counts: HashMap<u64, u32> = HashMap::new();
    let mut weights: Vec<u32> = Vec::with_capacity(64);

    for trace_idx in 0..config.num_traces {
        if violations.len() >= max_violations {
            break;
        }
        traces_completed = trace_idx + 1;

        // In swarm mode, select a random subset of disjuncts for this trace
        let enabled_disjuncts = if swarm_active {
            let subset = rng.random_subset(num_disjuncts, 0.5, 0.9);
            swarm_configs.push(subset.clone());
            Some(subset)
        } else {
            None
        };

        // Pick a random initial state
        let init_idx = rng.range(initial_states.len());
        let mut current = initial_states[init_idx].clone();
        if weighting == SimWeighting::LeastVisited {
            *visit_counts.entry(model.fingerprint(&current)).or_insert(0) += 1;
        }
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

            // Use swarm-filtered or full next-state generation
            if let Some(ref enabled) = enabled_disjuncts {
                model.next_states_swarm(&current, enabled, &mut successors);
            } else {
                model.next_states(&current, &mut successors);
            }
            total_states += 1;

            // A reached Assert(FALSE) during the next-state generation above is a
            // safety violation (TLC halts on it). The failing action produced no
            // successors, so `successors` is empty here; drain the reached-
            // assertion side channel and record it, exactly as the exhaustive BFS
            // worker does. See crate::model.
            if let Some(message) = crate::model::take_pending_assertion_violation() {
                violations.push(Violation {
                    message,
                    state: current.clone(),
                    property_type: PropertyType::Safety,
                    trace: trace.clone(),
                });
                if violations.len() >= max_violations {
                    break;
                }
            }

            if successors.is_empty() {
                // Deadlocked state - end this trace
                if step + 1 > max_depth_reached {
                    max_depth_reached = step + 1;
                }
                break;
            }

            // Pick a successor. Uniform reproduces TLC's `-simulate`;
            // LeastVisited uses a weighted choice to steer the walk toward
            // under-explored states (see `SimWeighting`).
            let next_idx = match weighting {
                SimWeighting::Uniform => rng.range(successors.len()),
                SimWeighting::LeastVisited => {
                    // Integer weight per successor: the least-visited gets the
                    // most weight, but every successor keeps a non-zero chance so
                    // the walk never starves a branch. weight = 1 + (max - visits).
                    weights.clear();
                    let mut max_visits = 0u32;
                    for s in successors.iter() {
                        let v = *visit_counts.get(&model.fingerprint(s)).unwrap_or(&0);
                        max_visits = max_visits.max(v);
                    }
                    for s in successors.iter() {
                        let v = *visit_counts.get(&model.fingerprint(s)).unwrap_or(&0);
                        weights.push(1 + (max_visits - v));
                    }
                    rng.weighted(&weights)
                }
            };
            current = successors.swap_remove(next_idx);
            if weighting == SimWeighting::LeastVisited {
                *visit_counts.entry(model.fingerprint(&current)).or_insert(0) += 1;
            }
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

    // In swarm mode, report which disjunct subsets found violations
    if swarm_active && !violations.is_empty() {
        eprintln!(
            "Swarm: {} violations found across {} traces ({} unique disjunct configs)",
            violations.len(),
            traces_completed,
            swarm_configs.len(),
        );
    }

    SimulationOutcome {
        traces_run: traces_completed,
        max_depth_reached,
        total_states,
        duration: started_at.elapsed(),
        violations,
        swarm_configs,
    }
}

#[cfg(test)]
mod rng_tests {
    //! Tests for the random-selection primitives. These guard the ideas from
    //! "When random isn't random enough" (ersc.io): unbiased bounded selection
    //! and exact integer-weighted choice.
    use super::*;

    #[test]
    fn range_is_approximately_uniform_for_non_power_of_two() {
        // n = 7 is the classic modulo-bias trap. Over many draws every bucket
        // should land within ~5% of the expected share; naive `% 7` would skew
        // the low buckets. (The deterministic seed makes this reproducible.)
        let n = 7usize;
        let draws = 700_000u64;
        let mut rng = Rng::new(0xC0FFEE);
        let mut counts = vec![0u64; n];
        for _ in 0..draws {
            counts[rng.range(n)] += 1;
        }
        let expected = draws as f64 / n as f64;
        for (i, &c) in counts.iter().enumerate() {
            let dev = (c as f64 - expected).abs() / expected;
            assert!(
                dev < 0.05,
                "bucket {i} deviated {:.3} from uniform (count={c}, expected={expected:.0})",
                dev
            );
        }
    }

    #[test]
    fn range_always_in_bounds_including_n_one() {
        let mut rng = Rng::new(42);
        for n in 1..=16usize {
            for _ in 0..1000 {
                assert!(rng.range(n) < n);
            }
        }
    }

    #[test]
    fn weighted_only_picks_positive_weight_indices() {
        // Zero-weight entries must never be selected.
        let mut rng = Rng::new(7);
        let weights = [0u32, 5, 0, 3, 0];
        for _ in 0..10_000 {
            let i = rng.weighted(&weights);
            assert!(weights[i] > 0, "picked zero-weight index {i}");
        }
    }

    #[test]
    fn weighted_respects_proportions() {
        // weights (1, 3) => index 1 should appear ~3x as often as index 0.
        let mut rng = Rng::new(0xABCD);
        let weights = [1u32, 3];
        let draws = 400_000u32;
        let mut hits = [0u32; 2];
        for _ in 0..draws {
            hits[rng.weighted(&weights)] += 1;
        }
        let ratio = hits[1] as f64 / hits[0] as f64;
        assert!(
            (ratio - 3.0).abs() < 0.1,
            "expected ~3:1, got {ratio:.3} (hits={hits:?})"
        );
    }

    #[test]
    fn weighted_all_zero_falls_back_to_uniform_in_range() {
        let mut rng = Rng::new(1);
        let weights = [0u32, 0, 0];
        for _ in 0..1000 {
            assert!(rng.weighted(&weights) < weights.len());
        }
    }
}
