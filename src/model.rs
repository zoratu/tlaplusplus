use crate::fairness::FairnessConstraint;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::hash::Hash;

/// Fixed-seed hasher for state fingerprints.
///
/// State fingerprints MUST be identical across processes for the same
/// logical state: multi-node cluster partitioning
/// (`partition_for_fp_cluster`) decides each state's owner node from its
/// fingerprint, and checkpoint/resume reloads persisted fingerprints into
/// a fresh process. `ahash::AHasher::default()` seeds its keys from the OS
/// RNG once per process (ahash's `runtime-rng` feature), so the same state
/// hashes to different values in different processes — which silently
/// breaks both. Always build state-fingerprint hashers via this helper.
///
/// The seed constants are arbitrary (hex digits of pi); any fixed values
/// work, but they must not change across builds that share checkpoints.
#[inline]
pub(crate) fn fingerprint_hasher() -> ahash::AHasher {
    use std::hash::BuildHasher;
    ahash::RandomState::with_seeds(
        0x243F_6A88_85A3_08D3,
        0x1319_8A2E_0370_7344,
        0xA409_3822_299F_31D0,
        0x082E_FA98_EC4E_6C89,
    )
    .build_hasher()
}

// ---------------------------------------------------------------------------
// Reached-assertion side channel
//
// A reached `Assert(FALSE)` in a next-state action is a safety violation (TLC
// halts on it with the assertion message). But `Model::next_states` returns
// `()` — it has no error/violation channel — and an eval error there is
// otherwise silently treated as a disabled branch, so such a violation would
// be masked as under-exploration. To surface it without a trait-signature
// change, the evaluator records the failed assertion in this per-thread slot,
// and the worker drains it immediately after each `next_states` call and
// reports it exactly like an invariant violation.
//
// Two guards keep this sound (no false violations):
//   1. Only an assertion whose argument evaluated to `Bool(false)` is recorded
//      — a benign guard-eval failure (e.g. a record access on a `ModelValue`
//      when a sibling guard is false) errors on a *different* expression and is
//      not recorded. Conjunction short-circuit means a reached assertion had
//      all preceding guards pass.
//   2. It is recorded only while a *committed* next-state generation is in
//      progress (see `CommittedNextStateGuard`), so speculative evaluations
//      (`ENABLED`, invariant/state-constraint checks, action probing) that hit
//      an assertion do not record a spurious pending violation.
use std::cell::{Cell, RefCell};

thread_local! {
    static COMMITTED_NEXT_STATE_DEPTH: Cell<u32> = const { Cell::new(0) };
    static PENDING_ASSERTION_VIOLATION: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// RAII marker: while alive, evaluation is part of committed next-state
/// generation, so a reached `Assert(FALSE)` should be recorded as a pending
/// violation. Nesting is tracked with a depth counter so a re-entrant
/// `next_states` (e.g. POR) restores the enclosing state on drop.
pub struct CommittedNextStateGuard(());

/// Enter committed next-state generation. Hold the returned guard for the
/// duration of the `next_states`/`next_states_labeled` evaluation.
#[must_use]
pub fn enter_committed_next_state() -> CommittedNextStateGuard {
    COMMITTED_NEXT_STATE_DEPTH.with(|d| d.set(d.get() + 1));
    CommittedNextStateGuard(())
}

impl Drop for CommittedNextStateGuard {
    fn drop(&mut self) {
        COMMITTED_NEXT_STATE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// True when evaluation is inside a committed next-state generation.
pub fn in_committed_next_state() -> bool {
    COMMITTED_NEXT_STATE_DEPTH.with(|d| d.get() > 0)
}

/// Record a reached `Assert(FALSE)` encountered during committed next-state
/// generation. Keeps the first message recorded until it is drained (the first
/// assertion reached is the one TLC would halt on).
pub fn record_pending_assertion_violation(message: String) {
    PENDING_ASSERTION_VIOLATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(message);
        }
    });
}

/// Take (and clear) any pending reached-assertion violation for this thread.
pub fn take_pending_assertion_violation() -> Option<String> {
    PENDING_ASSERTION_VIOLATION.with(|slot| slot.borrow_mut().take())
}

/// Peek whether a reached-assertion violation is pending (without clearing).
/// Used by `Model::next_states` to return cleanly (rather than panic) when a
/// reached `Assert(FALSE)` produced the eval error, so the worker can report it.
pub fn has_pending_assertion_violation() -> bool {
    PENDING_ASSERTION_VIOLATION.with(|slot| slot.borrow().is_some())
}

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

    /// Streaming variant of [`Model::initial_states`].
    ///
    /// Returns an iterator that yields initial states as they are produced.
    /// This lets the runtime begin invariant checking and successor exploration
    /// **before** Init enumeration completes — useful for specs whose Init
    /// predicate is expensive to enumerate (e.g. Einstein-class puzzles where
    /// Init enumeration alone can take tens of minutes while the reachable
    /// state graph is small).
    ///
    /// The default implementation defers to [`Model::initial_states`] and
    /// returns its `Vec`'s `into_iter()`. Models with cheap Init enumeration
    /// (counter-grid, hand-rolled Rust models) need not override this — the
    /// runtime treats the eager and streaming paths identically when the
    /// iterator yields all states immediately.
    ///
    /// Models that can produce initial states lazily (e.g. by spawning a
    /// background thread that enumerates Init and pushes through a channel)
    /// should override this method to return an iterator that yields states
    /// as they become available.
    fn initial_states_streaming(&self) -> Box<dyn Iterator<Item = Self::State> + Send + '_> {
        Box::new(self.initial_states().into_iter())
    }

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

    /// Whether the runtime should run the post-BFS fairness / liveness
    /// (SCC) check for this model.
    ///
    /// Fairness constraints (`WF_`/`SF_`) declared inside a `SPECIFICATION`
    /// are *assumptions*, not properties: on their own they can never be
    /// "violated". TLC only evaluates them relative to a declared temporal
    /// PROPERTY. So the fairness SCC pass — and the (more expensive)
    /// labeled-transition collection that feeds it — must run *only* when
    /// the model actually has a liveness property to verify. Otherwise a
    /// safety-only spec that happens to carry `WF_vars(...)` in its `Spec`
    /// would get a spurious liveness "violation" (and pay for building the
    /// labeled-transition graph) where TLC reports SAFE.
    ///
    /// Default: gate purely on the presence of fairness constraints, so
    /// non-TLA models that override `has_fairness_constraints` keep their
    /// prior behaviour. `TlaModel` additionally requires a liveness
    /// property to be present.
    fn should_check_fairness(&self) -> bool {
        self.has_fairness_constraints()
    }

    /// Return the fairness constraints for this model
    ///
    /// Default implementation returns an empty list. Models with fairness
    /// constraints (e.g. TlaModel with WF/SF specifications) should override this.
    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        Vec::new()
    }

    /// Name of the wrapper next-state action (e.g., `"Next"` for a TLA+ spec
    /// that defines `Next == ...`).
    ///
    /// The fairness checker uses this to recognise when a constraint like
    /// `WF_vars(Next)` is targeting the entire next-state relation rather than
    /// a specific named subaction. In that case any transition in an SCC
    /// counts as a Next step.
    ///
    /// Returns `None` if the model has no concept of a wrapper next action
    /// (the default).
    fn next_action_name(&self) -> Option<&str> {
        None
    }

    /// Decide whether a fairness-unfair SCC is an *actual* liveness
    /// counterexample.
    ///
    /// The fairness SCC pass finds cycles in which a fair action is
    /// continuously/infinitely enabled yet never taken. Such a cycle is a
    /// *non-fair* behaviour — but a non-fair behaviour is NOT a counterexample
    /// on its own. TLC only reports a violation when a declared temporal
    /// PROPERTY (e.g. `<>P`, `[]<>P`, `P ~> Q`) actually fails on a behaviour
    /// that stays in the cycle. An SCC that is fairness-unfair but still
    /// satisfies every checked property (e.g. EWD840's environment-only cycle,
    /// where `terminated` never holds so `terminated ~> terminationDetected`
    /// is vacuously true) must NOT be flagged — doing so is a false liveness
    /// violation that TLC does not report.
    ///
    /// Returns:
    ///   * `Some(true)`  — a declared temporal property is definitively
    ///     violated by this SCC → report the liveness violation.
    ///   * `Some(false)` — every declared temporal property that we can
    ///     evaluate is satisfied by this SCC → suppress the violation.
    ///   * `None`        — the model cannot make a property-level judgement
    ///     (no evaluable temporal property, or an unsupported property shape).
    ///     The caller then falls back to the fairness-only verdict (preserving
    ///     prior behaviour for models that don't implement this).
    ///
    /// `scc_states` are the actual states forming the SCC.
    fn scc_violates_liveness_property(&self, _scc_states: &[Self::State]) -> Option<bool> {
        None
    }

    /// Whether the runtime should collect the labeled-transition graph and run
    /// the post-BFS graph-level liveness pass for GRAPH-STRUCTURED temporal
    /// properties that are NOT expressible per-SCC — chiefly
    /// `[](P => <>[]Q)` ("whenever P, eventually-always Q").
    ///
    /// Distinct from `should_check_fairness`: a spec can have such a property
    /// with NO fairness constraints (the RealTime `ErrorTemporal` case). When
    /// this returns true the runtime collects labeled transitions (via
    /// `next_states_labeled`) and calls `graph_liveness_violation` after
    /// exploration, even though `should_check_fairness()` is false.
    ///
    /// Default: false (no such properties).
    fn needs_graph_liveness_check(&self) -> bool {
        false
    }

    /// Graph-level liveness check for `[](P => <>[]Q)`-shaped properties.
    ///
    /// Called once after exploration with the FULL fingerprint adjacency map,
    /// the fp→state map, and the list of non-trivial SCCs (cycles). Unlike
    /// `scc_violates_liveness_property` (which judges a single SCC in
    /// isolation), this receives the whole graph so it can check reachability
    /// from a P-state (which may lie on the stem) to a bad cycle (a reachable
    /// non-trivial cycle where `<>[]Q` fails, i.e. containing a `¬Q` state).
    ///
    /// `fair_scc` reports, per non-trivial SCC (by index into `non_trivial_sccs`),
    /// whether the cycle is FAIR — i.e. whether a behaviour may legitimately
    /// loop in it forever. Absent fairness constraints every non-trivial SCC
    /// is fair (stuttering is always allowed). With `WF_`/`SF_` constraints an
    /// unfair cycle is excluded by TLC and must NOT count as a bad cycle.
    ///
    /// Returns `Some(representative_bad_cycle_state)` describing a genuine
    /// violation, or `None` if no property is violated / the model can't judge.
    ///
    /// Default: `None`.
    fn graph_liveness_violation(
        &self,
        _adjacency_fp: &std::collections::HashMap<u64, Vec<u64>>,
        _state_by_fp: &std::collections::HashMap<u64, Self::State>,
        _non_trivial_sccs: &[Vec<u64>],
        _fair_scc: &[bool],
    ) -> Option<Self::State> {
        None
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
        use std::hash::Hasher;

        let mut hasher = fingerprint_hasher();
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

#[cfg(test)]
mod fingerprint_determinism_tests {
    use super::fingerprint_hasher;
    use std::hash::Hasher;

    /// Cross-process determinism guard for the state-fingerprint hasher.
    ///
    /// The expected constant was captured in a SEPARATE process (a
    /// standalone binary built against the same ahash version with the
    /// same seeds). If `fingerprint_hasher`'s seeds ever change — or if
    /// someone reintroduces `AHasher::default()` (per-process random
    /// seed) — this constant stops matching and the test fails.
    ///
    /// Why this matters: multi-node cluster partitioning
    /// (`partition_for_fp_cluster`) derives each state's owner node from
    /// its fingerprint, and checkpoint/resume reloads persisted
    /// fingerprints into a fresh process. Both silently break (no
    /// scaling / re-explore-everything) if the same state hashes
    /// differently across processes.
    #[test]
    fn fingerprint_hasher_seed_is_stable_across_processes() {
        let mut h = fingerprint_hasher();
        h.write(b"tlaplusplus-fingerprint-determinism-probe");
        assert_eq!(
            h.finish(),
            0x2C20_A680_A4EA_7496,
            "fingerprint_hasher seed changed — this breaks cluster \
             partitioning and checkpoint/resume across processes"
        );
    }

    /// The default `Model::fingerprint` of a fixed state must also be a
    /// stable constant (it routes through `fingerprint_hasher`).
    #[test]
    fn default_model_fingerprint_is_stable() {
        use super::Model;
        use crate::fairness::FairnessConstraint;
        use serde::{Deserialize, Serialize};

        #[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
        struct S {
            a: u32,
            b: String,
        }
        struct M;
        impl Model for M {
            type State = S;
            fn name(&self) -> &'static str {
                "fp-test"
            }
            fn initial_states(&self) -> Vec<S> {
                vec![]
            }
            fn next_states(&self, _s: &S, _out: &mut Vec<S>) {}
            fn check_invariants(&self, _s: &S) -> Result<(), String> {
                Ok(())
            }
            fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
                vec![]
            }
        }
        let fp = M.fingerprint(&S {
            a: 42,
            b: "hello".into(),
        });
        assert_eq!(
            fp, 2_207_073_645_509_949_576,
            "default Model::fingerprint of a fixed state must be a stable \
             cross-process constant"
        );
    }
}
