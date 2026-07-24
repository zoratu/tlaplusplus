// Item 3 regression: a reached `Assert(FALSE)` in a next-state action is a
// safety violation (TLC halts on it). `Model::next_states` returns `()` and
// cannot surface an error, so a failing action eval was previously swallowed as
// a disabled branch — masking the violation as under-exploration. The evaluator
// now records a definitively-failed assertion (argument evaluated to `Bool(false)`)
// on a per-thread side channel while a *committed* next-state generation is in
// progress, and the worker drains it and reports it like an invariant violation.
//
// Two cases are exercised end-to-end via `run_model` (with `allow_deadlock =
// true`, the mode in which the swallow bug used to manifest):
//   1. A reached Assert(FALSE) IS reported as a Safety violation.
//   2. An Assert in a branch whose guard is never satisfiable is NOT reached
//      (conjunction short-circuit) and must NOT produce a false violation, and
//      the reachable-state count is unaffected.

use serial_test::serial;
use std::fs;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

fn run_spec(name: &str, module_src: &str, cfg_src: &str) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let module_path = dir.path().join(format!("{name}.tla"));
    let cfg_path = dir.path().join(format!("{name}.cfg"));
    fs::write(&module_path, module_src).expect("write module");
    fs::write(&cfg_path, cfg_src).expect("write cfg");
    let mut model =
        TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model loads");
    model.allow_deadlock = true;

    let mut engine_cfg = EngineConfig::default();
    engine_cfg.workers = 1;
    engine_cfg.enforce_cgroups = false;
    engine_cfg.numa_pinning = false;
    engine_cfg.fp_expected_items = 1024;
    engine_cfg.checkpoint_on_exit = false;
    engine_cfg.enable_fp_persistence = false;
    engine_cfg.work_dir = dir.path().join("work");

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

#[test]
#[serial]
fn reached_assert_false_is_reported_as_safety_violation() {
    // From x=2, Next takes x'=3 and Assert(x' < 3) fails. Pre-fix this was
    // swallowed and the run reported no violation.
    let module_src = r#"---- MODULE AssertReachedInline ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == /\ x < 3
        /\ x' = x + 1
        /\ Assert(x' < 3, "x' reached 3")
====
"#;
    let cfg_src = "INIT Init\nNEXT Next\n";
    let outcome = run_spec("AssertReachedInline", module_src, cfg_src);

    let v = outcome
        .violation
        .expect("a reached Assert(FALSE) must be reported as a violation");
    assert_eq!(
        v.property_type,
        PropertyType::Safety,
        "a reached-assertion violation must be Safety, got {:?}: {}",
        v.property_type,
        v.message
    );
    assert!(
        v.message.contains("assertion failed"),
        "unexpected violation message: {}",
        v.message
    );
}

#[test]
#[serial]
fn assert_in_never_enabled_branch_is_not_a_violation() {
    // The Assert sits behind an unsatisfiable guard (x > 100); conjunction
    // short-circuit means it is never reached, so there must be no violation and
    // the reachable state count (x = 0,1,2,3) is unaffected.
    let module_src = r#"---- MODULE AssertDisabledInline ----
EXTENDS Integers, TLC
VARIABLE x
Init == x = 0
Next == \/ (x < 3 /\ x' = x + 1)
        \/ (x = 3 /\ x' = x)
        \/ (x > 100 /\ x' = x /\ Assert(FALSE, "must never fire"))
====
"#;
    let cfg_src = "INIT Init\nNEXT Next\n";
    let outcome = run_spec("AssertDisabledInline", module_src, cfg_src);

    assert!(
        outcome.violation.is_none(),
        "an Assert in a never-enabled branch must not be a violation, got: {:?}",
        outcome.violation
    );
    assert_eq!(
        outcome.stats.states_distinct, 4,
        "expected 4 distinct states (x = 0..3), got {}",
        outcome.stats.states_distinct
    );
}
