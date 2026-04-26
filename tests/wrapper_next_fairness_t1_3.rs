// T1.3 regression: WF_vars(Next) on a spec with an explicit
// `Terminated /\ UNCHANGED vars` stutter disjunct must NOT report
// a fairness violation. Pre-fix, the fairness checker compared the
// constraint's action name ("Next") against the per-disjunct labels
// ("Step", "Done", ...) and reported a false-positive Buchi-style
// violation on the single-state self-loop SCC at the terminated
// state.
//
// The fix adds `Model::next_action_name()` and threads it through to
// `check_fairness_on_scc_with_next`: when the constraint targets the
// wrapper Next action (i.e., `WF_vars(Next)`), every transition in
// the SCC counts as a Next step, so the constraint is satisfied as
// long as any edge exists.
//
// This file exercises the bug end-to-end via `run_model`, which is
// the path that the diff harness (and the WorkQueue.tla spec)
// trips on.

use serial_test::serial;
use std::fs;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

fn write_spec(
    dir: &TempDir,
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let module_path = dir.path().join(format!("{}.tla", module_name));
    let cfg_path = dir.path().join(format!("{}.cfg", module_name));
    fs::write(&module_path, module_src).expect("write module");
    fs::write(&cfg_path, cfg_src).expect("write cfg");
    (module_path, cfg_path)
}

fn run_spec(
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let (module_path, cfg_path) = write_spec(&dir, module_name, module_src, cfg_src);
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
    // Use a unique per-test work_dir inside the tempdir so parallel tests
    // do not collide on `./.tlapp`.
    engine_cfg.work_dir = dir.path().join("work");

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

#[test]
#[serial]
fn wf_vars_next_with_explicit_stutter_disjunct_does_not_falsely_violate_fairness() {
    let module_src = r#"---- MODULE WrapperNextFairnessInline ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Step == /\ x < 3 /\ x' = x + 1
Done == /\ x = 3 /\ UNCHANGED vars

Next == Step \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
====
"#;
    let cfg_src = "SPECIFICATION Spec\n";

    let outcome = run_spec("WrapperNextFairnessInline", module_src, cfg_src);

    // We expect to enumerate exactly 4 distinct states (x = 0..3).
    assert_eq!(
        outcome.stats.states_distinct, 4,
        "expected 4 distinct states (x = 0..3), got {}",
        outcome.stats.states_distinct
    );

    // Critically: no liveness/fairness violation should be reported.
    if let Some(v) = &outcome.violation {
        if v.property_type == PropertyType::Liveness {
            panic!(
                "T1.3 regression: WF_vars(Next) on an explicit-stutter spec must \
                 NOT report a fairness violation, got: {}",
                v.message
            );
        }
    }
    assert!(
        outcome.violation.is_none(),
        "expected no violation, got: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn wf_vars_named_subaction_still_detects_real_violation() {
    // True-positive sanity check: if the user declares fairness on a
    // *named subaction* that genuinely never fires inside an SCC,
    // we must STILL report the violation. The fix only relaxes the
    // wrapper-Next case, not subaction fairness.
    //
    // Spec: x toggles between 0 and 1 forever via `Toggle`. We declare
    // `WF_vars(NeverFires)` for an action that requires x = 99 (never
    // reachable). The fairness checker uses the conservative
    // assumption that any non-trivial SCC that lacks the named action
    // is a violation, so we expect a Liveness violation here.
    let module_src = r#"---- MODULE NamedSubactionFairness ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Toggle == \/ /\ x = 0 /\ x' = 1
            \/ /\ x = 1 /\ x' = 0

NeverFires == /\ x = 99 /\ x' = 100

Next == Toggle \/ NeverFires

Spec == Init /\ [][Next]_vars /\ WF_vars(NeverFires)
====
"#;
    let cfg_src = "SPECIFICATION Spec\n";

    let outcome = run_spec("NamedSubactionFairness", module_src, cfg_src);

    // The state space cycles 0 -> 1 -> 0 -> ... forming a 2-state SCC.
    assert_eq!(
        outcome.stats.states_distinct, 2,
        "expected exactly 2 distinct states (Toggle cycle)"
    );

    // We expect a Liveness violation for `NeverFires`.
    let violation = outcome
        .violation
        .as_ref()
        .expect("expected fairness violation for WF_vars(NeverFires)");
    assert_eq!(
        violation.property_type,
        PropertyType::Liveness,
        "expected Liveness property type, got {:?}",
        violation.property_type
    );
    assert!(
        violation.message.contains("NeverFires"),
        "expected violation message to mention 'NeverFires', got: {}",
        violation.message
    );
}
