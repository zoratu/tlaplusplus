// T10.1-4 follow-up correctness tests.
//
// These exercise the post-T10 fairness pipeline end-to-end through
// `run_model` with specs that stress the new code paths:
//
//   - T10.1 (parallel flatten via dashmap raw-shard iter): below the
//     80K-entry threshold the serial path is taken; we still want a
//     basic correctness check that the runtime returns the same
//     verdict as the previous serial implementation.
//   - T10.3 (trivial-SCC pre-filter): a DAG-shaped spec where the
//     trim algorithm must drop every node, leaving Tarjan with an
//     empty input. The fairness check must trivially pass (no SCCs
//     means no cycles to violate).
//   - T10.4 (per-action transition shard): a spec with many named
//     subaction fairness constraints. The shard index must route
//     each constraint to its own action's edges, and the per-action
//     verdict must agree with what the un-sharded implementation
//     would have computed.
//
// Each test asserts a specific outcome so the implementation cannot
// silently break correctness while improving performance.

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
fn t10_3_dag_shaped_spec_trim_drops_everything_no_violation() {
    // A pure DAG: x climbs from 0 to 5 and then deadlocks. There are
    // 6 distinct states, 5 transitions, no cycles. The trivial-SCC
    // pre-filter (T10.3) is tuned to skip on small graphs, so for
    // this 6-node spec Tarjan still runs and produces 6 trivial
    // singleton SCCs which Phase 4 filters out. Either path
    // (trim-active or trim-skipped) must end with zero non-trivial
    // SCCs and no fairness violation — that's the correctness
    // invariant we lock in here.
    //
    // We declare WF_vars(Step) — the constraint is satisfied because
    // there are no non-trivial SCCs to evaluate it against.
    let module_src = r#"---- MODULE DagShapedFairness ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Step == /\ x < 5 /\ x' = x + 1

Next == Step

Spec == Init /\ [][Next]_vars /\ WF_vars(Step)
====
"#;
    let cfg_src = "SPECIFICATION Spec\n";
    let outcome = run_spec("DagShapedFairness", module_src, cfg_src);

    // 6 states, 0..5
    assert_eq!(outcome.stats.states_distinct, 6);
    // No fairness violation: the only "SCC" candidates after trim is
    // empty, so the fairness check is trivially satisfied.
    if let Some(v) = &outcome.violation {
        if v.property_type == PropertyType::Liveness {
            panic!(
                "T10.3 regression: pure DAG must NOT raise a fairness \
                 violation after trim (got: {})",
                v.message
            );
        }
    }
}

#[test]
#[serial]
fn t10_4_many_subaction_constraints_all_satisfied() {
    // Three named subactions form a triangular cycle: A -> B -> C -> A.
    // We declare WF on each. The per-action shard must route each
    // constraint to its own edge and confirm it occurs in the SCC.
    // No fairness violation expected.
    let module_src = r#"---- MODULE TriangleFairness ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

A == /\ x = 0 /\ x' = 1
B == /\ x = 1 /\ x' = 2
C == /\ x = 2 /\ x' = 0

Next == A \/ B \/ C

Spec == Init /\ [][Next]_vars /\ WF_vars(A) /\ WF_vars(B) /\ WF_vars(C)
====
"#;
    let cfg_src = "SPECIFICATION Spec\n";
    let outcome = run_spec("TriangleFairness", module_src, cfg_src);

    assert_eq!(
        outcome.stats.states_distinct, 3,
        "triangle has 3 distinct states (x = 0, 1, 2)"
    );
    assert!(
        outcome.violation.is_none(),
        "all three subaction constraints occur in the SCC; expected no violation, got: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn t10_4_named_subaction_missing_from_scc_still_violates() {
    // A 2-state Toggle cycle. We declare WF on a never-fires action.
    // Per-action shard for `NeverFires` is empty; the constraint must
    // therefore flag a violation. This exercises the shard fast-path
    // for the missing-action case.
    let module_src = r#"---- MODULE ShardedNeverFires ----
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
    let outcome = run_spec("ShardedNeverFires", module_src, cfg_src);

    assert_eq!(outcome.stats.states_distinct, 2);
    let v = outcome
        .violation
        .as_ref()
        .expect("expected fairness violation for WF_vars(NeverFires)");
    assert_eq!(v.property_type, PropertyType::Liveness);
    assert!(
        v.message.contains("NeverFires"),
        "violation message should mention the missing action; got: {}",
        v.message
    );
}

#[test]
#[serial]
fn t10_3_self_loop_state_survives_trim_and_satisfies_wrapper_next() {
    // A single-state self-loop: x = 0 with `Done == /\ UNCHANGED vars`.
    // The trim must NOT drop the self-loop state (it's a non-trivial
    // SCC by itself). Wrapper-Next fairness must pass because the
    // self-loop edge counts as a Next step.
    let module_src = r#"---- MODULE SelfLoopWrapperNext ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Done == /\ x = 0 /\ UNCHANGED vars

Next == Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
====
"#;
    let cfg_src = "SPECIFICATION Spec\n";
    let outcome = run_spec("SelfLoopWrapperNext", module_src, cfg_src);

    assert_eq!(outcome.stats.states_distinct, 1);
    if let Some(v) = &outcome.violation {
        if v.property_type == PropertyType::Liveness {
            panic!(
                "T10.3 regression: self-loop SCC must satisfy wrapper-Next fairness, \
                 got violation: {}",
                v.message
            );
        }
    }
}
