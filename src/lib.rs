// T13.4 Phase 2 — register the Verus tool-attribute namespace under the
// `verus` cargo feature so `#[verifier::external]` etc. resolve cleanly
// under regular rustc. Cargo-verus uses these markers to know which
// modules to skip when verifying.
#![cfg_attr(feature = "verus", register_tool(verifier))]

#[cfg_attr(feature = "verus", verifier::external)]
pub mod autotune;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod canon;
#[cfg_attr(feature = "verus", verifier::external)]
#[macro_use]
pub mod chaos;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod cli;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod coverage;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod distributed;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod fairness;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod liveness;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod model;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod models;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod runtime;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod simulation;
pub mod storage;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod streaming_scc;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod symmetry;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod system;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod tla;
#[cfg_attr(feature = "verus", verifier::external)]
pub mod trace_minimize;

pub use coverage::{CoverageStats, CoverageTracker};
pub use fairness::{
    ActionLabel, FairnessConstraint, LabeledTransition, TarjanSCC, check_fairness_on_scc,
    check_fairness_on_scc_with_next,
};
pub use liveness::{BuchiChecker, CycleDetector, LivenessChecker};
pub use model::Model;
pub use runtime::{EngineConfig, PropertyType, RunOutcome, RunStats, Violation, run_model};
pub use streaming_scc::{
    Color, FingerprintAdjacencyGraph, LivenessGraph, NestedDfsResult, nested_dfs,
};
pub use simulation::{SimulationConfig, SimulationOutcome, run_simulation};
pub use trace_minimize::{
    MinimizeResult, OperatorBody, extract_invariant_variables,
    extract_invariant_variables_transitive, minimize_trace,
};
