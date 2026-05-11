// T10.2 stage 3 — gate-7: parity test for `--liveness-streaming-exploration`.
//
// Runs the same TLA+ fairness fixtures as `wrapper_next_fairness_t1_3.rs` and
// `streaming_scc_oracle.rs` with the *page-aligned color-map* nested-DFS path
// enabled, then asserts that:
//
//   1. The verdict (violation / no violation) is identical to the
//      Tarjan-based path AND identical to the v1.1.0 hash-backed oracle.
//   2. State counts match across all three modes.
//
// Stage 3 of T10.2 phase 2 (`docs/T10.2-phase2-refined.md`) lands the
// production `PageAlignedColorMap` data structure end-to-end. The full
// in-exploration BFS→DFS hot-loop lift remains future work; what this
// test gates is "color map + nested DFS produce the same fairness
// verdict as the existing Tarjan path on real fairness specs."

use serial_test::serial;
use std::fs;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

#[derive(Copy, Clone, Debug)]
enum LivenessMode {
    Baseline,
    Streaming,
    StreamingExploration,
}

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

fn run_spec_with_mode(
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
    mode: LivenessMode,
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
    match mode {
        LivenessMode::Baseline => {
            engine_cfg.liveness_streaming = false;
            engine_cfg.liveness_streaming_exploration = false;
        }
        LivenessMode::Streaming => {
            engine_cfg.liveness_streaming = true;
            engine_cfg.liveness_streaming_exploration = false;
        }
        LivenessMode::StreamingExploration => {
            engine_cfg.liveness_streaming = false;
            engine_cfg.liveness_streaming_exploration = true;
        }
    }

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

const WRAPPER_NEXT_MODULE: &str = r#"---- MODULE WrapperNextFairnessExploration ----
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

const NAMED_SUBACTION_MODULE: &str = r#"---- MODULE NamedSubactionFairnessExploration ----
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
fn t10_2_stage3_parity_on_passing_spec() {
    // WrapperNext-with-Done passes under all three modes.
    let cfg = "SPECIFICATION Spec\n";
    let baseline = run_spec_with_mode(
        "WrapperNextFairnessExploration",
        WRAPPER_NEXT_MODULE,
        cfg,
        LivenessMode::Baseline,
    );
    let streaming = run_spec_with_mode(
        "WrapperNextFairnessExploration",
        WRAPPER_NEXT_MODULE,
        cfg,
        LivenessMode::Streaming,
    );
    let exploration = run_spec_with_mode(
        "WrapperNextFairnessExploration",
        WRAPPER_NEXT_MODULE,
        cfg,
        LivenessMode::StreamingExploration,
    );

    assert_eq!(
        baseline.stats.states_distinct, streaming.stats.states_distinct,
        "baseline vs streaming state count mismatch"
    );
    assert_eq!(
        baseline.stats.states_distinct, exploration.stats.states_distinct,
        "baseline vs streaming-exploration state count mismatch"
    );
    assert_eq!(
        baseline.violation.is_some(),
        streaming.violation.is_some(),
        "baseline vs streaming verdict mismatch"
    );
    assert_eq!(
        baseline.violation.is_some(),
        exploration.violation.is_some(),
        "baseline vs streaming-exploration verdict mismatch"
    );
    assert!(
        exploration.violation.is_none(),
        "WrapperNext-with-Done must not violate fairness under streaming-exploration: {:?}",
        exploration.violation
    );
}

#[test]
#[serial]
fn t10_2_stage3_parity_on_failing_spec() {
    // NamedSubaction with WF on NeverFires must violate under all three modes.
    let cfg = "SPECIFICATION Spec\n";
    let baseline = run_spec_with_mode(
        "NamedSubactionFairnessExploration",
        NAMED_SUBACTION_MODULE,
        cfg,
        LivenessMode::Baseline,
    );
    let streaming = run_spec_with_mode(
        "NamedSubactionFairnessExploration",
        NAMED_SUBACTION_MODULE,
        cfg,
        LivenessMode::Streaming,
    );
    let exploration = run_spec_with_mode(
        "NamedSubactionFairnessExploration",
        NAMED_SUBACTION_MODULE,
        cfg,
        LivenessMode::StreamingExploration,
    );

    assert_eq!(
        baseline.stats.states_distinct, streaming.stats.states_distinct,
        "baseline vs streaming state count mismatch"
    );
    assert_eq!(
        baseline.stats.states_distinct, exploration.stats.states_distinct,
        "baseline vs streaming-exploration state count mismatch"
    );

    assert!(
        baseline.violation.is_some(),
        "baseline must report a fairness violation"
    );
    assert!(
        streaming.violation.is_some(),
        "streaming oracle must report a fairness violation"
    );
    assert!(
        exploration.violation.is_some(),
        "streaming-exploration oracle must report a fairness violation"
    );

    assert_eq!(
        exploration.violation.as_ref().unwrap().property_type,
        PropertyType::Liveness,
        "streaming-exploration violation must be Liveness"
    );
    assert!(
        exploration
            .violation
            .as_ref()
            .unwrap()
            .message
            .contains("NeverFires"),
        "expected NeverFires in violation message, got: {}",
        exploration.violation.as_ref().unwrap().message,
    );
}

#[test]
#[serial]
fn t10_2_stage3_default_off_does_not_perturb_safety_spec() {
    // Sanity: a non-fairness spec must be completely unaffected by the
    // new flag (because the page-aligned color-map path is gated on
    // `model.has_fairness_constraints()`, which is false here).
    let cfg = "SPECIFICATION Spec\n";
    let module = r#"---- MODULE NoFairnessExploration ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0
Next == /\ x < 5 /\ x' = x + 1

Spec == Init /\ [][Next]_vars
====
"#;

    let baseline = run_spec_with_mode(
        "NoFairnessExploration",
        module,
        cfg,
        LivenessMode::Baseline,
    );
    let exploration = run_spec_with_mode(
        "NoFairnessExploration",
        module,
        cfg,
        LivenessMode::StreamingExploration,
    );
    assert_eq!(
        baseline.stats.states_distinct, exploration.stats.states_distinct,
        "non-fairness safety spec must produce identical state counts \
         regardless of --liveness-streaming-exploration"
    );
    // Both deadlock at x=5 (Next is conjunctive); allow either no
    // violation (deadlock disabled) or a deadlock violation, but the
    // verdicts must match.
    assert_eq!(
        baseline.violation.is_some(),
        exploration.violation.is_some(),
        "verdicts must match on non-fairness spec"
    );
}
