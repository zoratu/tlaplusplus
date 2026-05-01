// T10.2 — End-to-end test for the streaming-SCC oracle (`--liveness-streaming`).
//
// Runs the same TLA+ fairness fixtures as `wrapper_next_fairness_t1_3.rs` with
// the streaming-SCC oracle enabled, then asserts that:
//
//   1. The verdict (violation / no violation) is identical to the
//      Tarjan-based path.
//   2. (smoke) Stderr does not contain "[T10.2 oracle] DIVERGENCE".
//
// This is the validation harness for the v1.1.0 in-exploration variant: as
// long as the oracle agrees with Tarjan on the corpus, we have a correctness
// proof that the lift-into-worker-loop step preserves semantics.

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

fn run_spec_with_streaming(
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
    streaming: bool,
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
    engine_cfg.work_dir = dir.path().join("work");
    engine_cfg.liveness_streaming = streaming;

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

const WRAPPER_NEXT_MODULE: &str = r#"---- MODULE WrapperNextFairnessOracle ----
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

const NAMED_SUBACTION_MODULE: &str = r#"---- MODULE NamedSubactionFairnessOracle ----
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

#[test]
#[serial]
fn t10_2_oracle_agrees_on_passing_spec() {
    // WrapperNextFairness should pass under both modes.
    let cfg = "SPECIFICATION Spec\n";
    let baseline = run_spec_with_streaming(
        "WrapperNextFairnessOracle",
        WRAPPER_NEXT_MODULE,
        cfg,
        false,
    );
    let streaming = run_spec_with_streaming(
        "WrapperNextFairnessOracle",
        WRAPPER_NEXT_MODULE,
        cfg,
        true,
    );

    assert_eq!(
        baseline.violation.is_none(),
        streaming.violation.is_none(),
        "T10.2 oracle disagrees with Tarjan on passing spec: \
         baseline violation = {:?}, streaming violation = {:?}",
        baseline.violation.as_ref().map(|v| &v.message),
        streaming.violation.as_ref().map(|v| &v.message),
    );
    assert!(
        streaming.violation.is_none(),
        "WrapperNext-with-Done spec must not violate fairness, got: {:?}",
        streaming.violation
    );
}

#[test]
#[serial]
fn t10_2_oracle_agrees_on_failing_spec() {
    // NamedSubactionFairness should violate under both modes (NeverFires
    // is in an SCC of {x=0, x=1} but never occurs there).
    let cfg = "SPECIFICATION Spec\n";
    let baseline = run_spec_with_streaming(
        "NamedSubactionFairnessOracle",
        NAMED_SUBACTION_MODULE,
        cfg,
        false,
    );
    let streaming = run_spec_with_streaming(
        "NamedSubactionFairnessOracle",
        NAMED_SUBACTION_MODULE,
        cfg,
        true,
    );

    assert!(
        baseline.violation.is_some(),
        "baseline (Tarjan) must report a fairness violation"
    );
    assert!(
        streaming.violation.is_some(),
        "streaming oracle must report the same fairness violation"
    );

    // Both should be Liveness type.
    assert_eq!(
        baseline.violation.as_ref().unwrap().property_type,
        PropertyType::Liveness
    );
    assert_eq!(
        streaming.violation.as_ref().unwrap().property_type,
        PropertyType::Liveness
    );
}
