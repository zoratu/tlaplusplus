//! Metamorphic gate — symmetry reduction on/off.
//!
//! WHY THIS FILE EXISTS
//! --------------------
//! Symmetry reduction is a soundness-sensitive optimisation, and it lives on
//! the fingerprint path (`TlaModel::fingerprint` → `canonicalize_tla_state`),
//! *not* on `next_states`. So the raw-`Model`-trait BFS used by the POR
//! correctness gate (`tests/por_correctness.rs`) cannot exercise it — the
//! reduction only appears in fingerprint-based dedup, i.e. in the real engine's
//! `RunStats::states_distinct`. This gate therefore drives `run_model` and
//! compares two runs of the *same* module that differ only in whether the `.cfg`
//! declares `SYMMETRY`.
//!
//! Two oracle-free metamorphic relations (no TLC needed):
//!
//!   1. **Verdict parity** — toggling symmetry must not change the invariant
//!      verdict. A correct quotient reaches a violating equivalence class iff
//!      the full space does. (Here the invariant holds, so both are clean; the
//!      `metamorphic_symmetry_violation_parity` case covers the violating side.)
//!   2. **Reduced-or-equal distinct count** — the symmetry-reduced distinct
//!      count is `<=` the full count, and for a genuinely symmetric spec it is
//!      *strictly* less (a `==` here would mean symmetry silently did nothing —
//!      exactly the #95 class of bug where canonicalisation was a no-op).
//!
//! The exact counts are also pinned as a regression anchor: with
//! `Procs = {p1,p2,p3}` and each clock cycling mod 3, the full reachable set is
//! `3^3 = 27`; quotienting by `Permutations(Procs)` leaves the multisets of size
//! 3 over `{0,1,2}` = `C(5,3) = 10`.

use std::fs;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, RunOutcome, run_model};

/// A spec whose state is fully symmetric under permutation of `Procs`.
const SYM_MODULE: &str = r#"---- MODULE SymmetryMetamorphic ----
EXTENDS Naturals

CONSTANT Procs

VARIABLE clock

vars == <<clock>>

Init == clock = [p \in Procs |-> 0]

Tick(p) == clock' = [clock EXCEPT ![p] = (clock[p] + 1) % 3]

Next == \E p \in Procs : Tick(p)

Spec == Init /\ [][Next]_vars

\* Always true: no violation, so both runs complete and their verdicts match.
Inv == \A p \in Procs : clock[p] \in 0..2

\* Deliberately violable variant: some clock reaching 2 trips this. The
\* violating states are NOT all in one symmetry class, so a correct quotient
\* still reaches one.
NoTwos == \A p \in Procs : clock[p] # 2

Symmetry == Permutations(Procs)
================================================================================
"#;

const CFG_SYM: &str = r#"SPECIFICATION Spec
CONSTANTS Procs = {p1, p2, p3}
INVARIANT Inv
SYMMETRY Symmetry
"#;

const CFG_FULL: &str = r#"SPECIFICATION Spec
CONSTANTS Procs = {p1, p2, p3}
INVARIANT Inv
"#;

const CFG_SYM_VIOL: &str = r#"SPECIFICATION Spec
CONSTANTS Procs = {p1, p2, p3}
INVARIANT NoTwos
SYMMETRY Symmetry
"#;

const CFG_FULL_VIOL: &str = r#"SPECIFICATION Spec
CONSTANTS Procs = {p1, p2, p3}
INVARIANT NoTwos
"#;

fn run(cfg_src: &str) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let module_path = dir.path().join("SymmetryMetamorphic.tla");
    let cfg_path = dir.path().join("SymmetryMetamorphic.cfg");
    fs::write(&module_path, SYM_MODULE).expect("write module");
    fs::write(&cfg_path, cfg_src).expect("write cfg");

    let mut model =
        TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model loads");
    model.allow_deadlock = true;

    let mut engine_cfg = EngineConfig::default();
    engine_cfg.workers = 2;
    engine_cfg.enforce_cgroups = false;
    engine_cfg.numa_pinning = false;
    engine_cfg.fp_expected_items = 4096;
    engine_cfg.checkpoint_on_exit = false;
    engine_cfg.enable_fp_persistence = false;
    engine_cfg.work_dir = dir.path().join("work");

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

#[test]
fn metamorphic_symmetry_no_violation_parity_and_reduction() {
    let full = run(CFG_FULL);
    let sym = run(CFG_SYM);

    // Relation 1: verdict parity — neither run violates (Inv always holds).
    assert!(
        full.violation.is_none(),
        "full run unexpectedly violated: {:?}",
        full.violation
    );
    assert!(
        sym.violation.is_none(),
        "symmetry run unexpectedly violated: {:?}",
        sym.violation
    );

    // Relation 2: reduced-or-equal, and strictly-less for this symmetric spec.
    assert!(
        sym.stats.states_distinct <= full.stats.states_distinct,
        "symmetry-reduced distinct ({}) must be <= full distinct ({})",
        sym.stats.states_distinct,
        full.stats.states_distinct
    );
    assert!(
        sym.stats.states_distinct < full.stats.states_distinct,
        "symmetry did not reduce the state count ({} == {}) — canonicalisation \
         is a no-op (the #95 regression class)",
        sym.stats.states_distinct,
        full.stats.states_distinct
    );

    // Regression anchors: full = 3^3 = 27; quotient = C(5,3) = 10.
    assert_eq!(
        full.stats.states_distinct, 27,
        "full reachable count changed"
    );
    assert_eq!(
        sym.stats.states_distinct, 10,
        "symmetry-quotient count changed"
    );
}

#[test]
fn metamorphic_symmetry_violation_parity() {
    let full = run(CFG_FULL_VIOL);
    let sym = run(CFG_SYM_VIOL);

    // Verdict parity on the violating side: both must catch it. A quotient that
    // dropped the violating class would report `violation=false` here — a
    // false-safe under-report, the most dangerous symmetry bug.
    assert!(
        full.violation.is_some(),
        "full run missed the NoTwos violation"
    );
    assert!(
        sym.violation.is_some(),
        "symmetry run missed the NoTwos violation — quotient dropped a \
         violating equivalence class (false-safe)"
    );
}
