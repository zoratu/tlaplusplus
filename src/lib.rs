// T13.4 Phase A.2 — under `--features verus` we use `assume_specification`
// to bridge `std::vec::Vec::<T, A>::len`, whose actual signature carries
// the unstable `allocator_api` allocator-parameter generic. Default
// builds (no `verus` feature) never see this and stay on stable rustc.
#![cfg_attr(feature = "verus", feature(allocator_api))]

// T13.4 Phase 2 — `#[cfg_attr(feature = "verus", verifier::external)]`
// markers tell cargo-verus to skip modules that aren't being verified.
// Under regular rustc with `--features verus`, these attributes are
// not recognized (the `verifier` tool namespace isn't a known crate),
// so `cargo build --features verus` only works under cargo-verus's
// wrapper. Mac/CI default builds (no feature) are unaffected; the
// supported verification flow is `cargo verus check --features verus`
// on a Verus-built spot.

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
