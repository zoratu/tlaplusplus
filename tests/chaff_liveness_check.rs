//! Chaff gate for `eventually_always_unchecked` (chaffs.toml).
//!
//! The liveness layer ships SILENT false-negatives when it goes wrong: a bare
//! `<>[]P` (eventually-always) PROPERTY that is left unchecked is reported as
//! `violation=false` rather than erroring. `diff_tlc` compares a boolean verdict,
//! but no corpus spec yet exercises a reachable bare-`<>[]P` Progress violation,
//! and the CLI's finite-trace fallback can print `violation=false` for temporal
//! properties — so a diff_tlc gate is not the reliable oracle here.
//!
//! This gate asserts directly on the proven path: `run_model` must detect a
//! `PropertyType::Liveness` violation for a bare `<>[]P` on a spec where `P`
//! provably fails to become always-true (the 0<->1 fair cycle from PR #198's
//! `eventually_always_fair_cycle_violates`). Without the chaff the violation is
//! detected and this test passes; with `TLAPP_CHAFF=eventually_always_unchecked`
//! active, `graph_liveness_shape`'s EventuallyAlways arm returns `None`, the
//! property is left unchecked, no violation is reported, and this test FAILS —
//! i.e. the gate CATCHES the chaff.
//!
//! `scripts/chaff_matrix.sh` runs this crate with that env var set for the
//! `eventually_always_unchecked` chaff (gate = "liveness_check").

use std::fs;

use serial_test::serial;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

fn run_spec(
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let module_path = dir.path().join(format!("{module_name}.tla"));
    let cfg_path = dir.path().join(format!("{module_name}.cfg"));
    fs::write(&module_path, module_src).expect("write module");
    fs::write(&cfg_path, cfg_src).expect("write cfg");

    let mut model =
        TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model loads");
    model.allow_deadlock = true;

    let mut engine_cfg = EngineConfig::default();
    engine_cfg.workers = 2;
    engine_cfg.enforce_cgroups = false;
    engine_cfg.numa_pinning = false;
    engine_cfg.fp_expected_items = 8192;
    engine_cfg.checkpoint_on_exit = false;
    engine_cfg.enable_fp_persistence = false;
    engine_cfg.work_dir = dir.path().join("work");

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

#[test]
#[serial]
fn bare_eventually_always_progress_violation_is_detected() {
    // x toggles 0<->1 forever, so `<>[](x = 1)` is VIOLATED (x = 0 recurs
    // infinitely). The bare-`<>[]P` shape is exactly what the chaff refuses to
    // recognize. Modeled on PR #198's eventually_always_fair_cycle_violates.
    let module_src = r#"---- MODULE ChaffEAFairCycle ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Next == \/ (x = 0 /\ x' = 1)
        \/ (x = 1 /\ x' = 0)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

P == x = 1
Progress == <>[]P
====
"#;
    let cfg_src = "SPECIFICATION Spec\nPROPERTY Progress\n";
    let outcome = run_spec("ChaffEAFairCycle", module_src, cfg_src);

    assert_eq!(
        outcome.stats.states_distinct, 2,
        "x toggles in {{0, 1}} — exploration itself must be unaffected"
    );
    // The `[CHAFF_CATCH_OK]` sentinel marks the INTENDED assertion. The chaff
    // matrix greps the cargo log for it so that only a failure HERE (the missing
    // liveness violation) is credited as a catch — an unrelated panic, a compile
    // error, or a `run_model` crash does not carry the sentinel and is reported as
    // an unknown-mode HOLE instead of a false "caught".
    let v = outcome.violation.as_ref().expect(
        "[CHAFF_CATCH_OK] expected a liveness violation: bare <>[](x = 1) fails on the \
         0<->1 cycle. If this fails under TLAPP_CHAFF=eventually_always_unchecked, the \
         chaff is CAUGHT; if it fails WITHOUT the chaff, the bare-<>[]P liveness gate has \
         regressed.",
    );
    assert_eq!(
        v.property_type,
        PropertyType::Liveness,
        "the <>[]P violation must be reported as a liveness property"
    );
}
