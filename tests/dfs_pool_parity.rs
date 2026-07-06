//! T10.2 stage 5 — gate-8 multi-worker DFS pool parity.
//!
//! Asserts that the same fairness specs run with `dfs_workers = 1` and
//! with `dfs_workers > 1` produce:
//!
//! 1. Identical fairness verdicts (violation present/absent + property
//!    type + violation message contains the same subaction name).
//! 2. Identical `states_distinct` counts. Cross-partition routing must
//!    not lose or duplicate states; the BFS fp_store-dedup invariant is
//!    preserved by the pool's `partition_for_fp(fp) → owner` routing
//!    where each fingerprint has exactly one home worker.
//!
//! These tests are the gate-8 deliverable referenced in the Stage 5
//! task brief.

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

fn run_pool(
    module_name: &str,
    module_src: &str,
    cfg_src: &str,
    dfs_workers: usize,
) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let (module_path, cfg_path) = write_spec(&dir, module_name, module_src, cfg_src);
    let mut model =
        TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model loads");
    model.allow_deadlock = true;

    let mut engine_cfg = EngineConfig::default();
    // Need workers >= dfs_workers because the pool size is clamped to the
    // BFS fleet size. Use 4 BFS workers; pool will run with `dfs_workers`
    // (1, 2, 3, or 4).
    engine_cfg.workers = 4;
    engine_cfg.enforce_cgroups = false;
    engine_cfg.numa_pinning = false;
    engine_cfg.fp_expected_items = 4096;
    engine_cfg.checkpoint_on_exit = false;
    engine_cfg.enable_fp_persistence = false;
    engine_cfg.work_dir = dir.path().join("work");
    engine_cfg.liveness_streaming = false;
    engine_cfg.liveness_streaming_exploration = true;
    engine_cfg.dfs_workers = dfs_workers;

    run_model(model, engine_cfg).expect("engine runs cleanly")
}

const PASSING_FAIRNESS_SPEC: &str = r#"---- MODULE DfsPoolPassingWrapperNext ----
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

const FAILING_FAIRNESS_SPEC: &str = r#"---- MODULE DfsPoolFailingNamedSubaction ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Toggle == \/ /\ x = 0 /\ x' = 1
            \/ /\ x = 1 /\ x' = 0

NeverFires == /\ x = 99 /\ x' = 100

Next == Toggle \/ NeverFires

Spec == Init /\ [][Next]_vars /\ WF_vars(NeverFires)

\* Liveness property to verify. Because `NeverFires` is never enabled from
\* any reachable state (`x` toggles in {0,1}, never reaching 99), the fair
\* subaction never fires and `Reaches100` can never hold — a genuine
\* fairness/liveness violation that TLC also reports. The PROPERTY is
\* required: fairness constraints alone (with no liveness property to
\* verify) are assumptions, not checkable properties — TLC reports SAFE for
\* `SPECIFICATION Spec` + only `WF_vars(...)` and no PROPERTIES.
Reaches100 == <>(x = 100)
====
"#;

#[test]
#[serial]
fn pool_state_count_matches_single_worker_on_passing_spec() {
    let cfg = "SPECIFICATION Spec\n";
    let one = run_pool("DfsPoolPassingWrapperNext", PASSING_FAIRNESS_SPEC, cfg, 1);
    let four = run_pool("DfsPoolPassingWrapperNext", PASSING_FAIRNESS_SPEC, cfg, 4);

    assert_eq!(
        one.stats.states_distinct, four.stats.states_distinct,
        "DFS pool state count must be identical to single-worker DFS \
         (1 worker => {}, 4 workers => {})",
        one.stats.states_distinct, four.stats.states_distinct
    );
    assert!(
        one.violation.is_none() && four.violation.is_none(),
        "passing fairness spec must not violate under either pool size; \
         got 1-worker={:?}, 4-worker={:?}",
        one.violation,
        four.violation,
    );
}

#[test]
#[serial]
fn pool_verdict_matches_single_worker_on_failing_spec() {
    let cfg = "SPECIFICATION Spec\nPROPERTIES Reaches100\n";
    let one = run_pool("DfsPoolFailingNamedSubaction", FAILING_FAIRNESS_SPEC, cfg, 1);
    let four = run_pool("DfsPoolFailingNamedSubaction", FAILING_FAIRNESS_SPEC, cfg, 4);

    assert_eq!(
        one.stats.states_distinct, four.stats.states_distinct,
        "DFS pool state count must be identical to single-worker DFS \
         on failing spec (1 worker => {}, 4 workers => {})",
        one.stats.states_distinct, four.stats.states_distinct
    );

    let one_v = one
        .violation
        .as_ref()
        .expect("1-worker pool must report fairness violation");
    let four_v = four
        .violation
        .as_ref()
        .expect("4-worker pool must report fairness violation");
    assert_eq!(one_v.property_type, PropertyType::Liveness);
    assert_eq!(four_v.property_type, PropertyType::Liveness);
    assert!(
        one_v.message.contains("NeverFires"),
        "1-worker violation must mention NeverFires, got: {}",
        one_v.message
    );
    assert!(
        four_v.message.contains("NeverFires"),
        "4-worker violation must mention NeverFires, got: {}",
        four_v.message
    );
}

#[test]
#[serial]
fn pool_state_count_invariant_across_pool_sizes() {
    // Three-state cycle from gate-7. Run with 1, 2, 3, 4 pool workers
    // and assert the state count is identical across all four
    // configurations. Any drift means cross-partition routing is losing
    // or duplicating states.
    let module_src = r#"---- MODULE DfsPoolThreeCycle ----
EXTENDS Naturals

VARIABLES x

vars == <<x>>

Init == x = 0

Cycle == \/ /\ x = 0 /\ x' = 1
           \/ /\ x = 1 /\ x' = 2
           \/ /\ x = 2 /\ x' = 0

Idle == /\ x = 999 /\ UNCHANGED vars

Next == Cycle \/ Idle

Spec == Init /\ [][Next]_vars /\ WF_vars(Idle)
====
"#;
    let cfg = "SPECIFICATION Spec\n";
    let baseline = run_pool("DfsPoolThreeCycle", module_src, cfg, 1);
    for n in [2usize, 3, 4].iter() {
        let outcome = run_pool("DfsPoolThreeCycle", module_src, cfg, *n);
        assert_eq!(
            outcome.stats.states_distinct, baseline.stats.states_distinct,
            "DFS pool n={} produced state count {} but n=1 produced {}",
            n, outcome.stats.states_distinct, baseline.stats.states_distinct
        );
        assert_eq!(
            outcome.violation.is_some(),
            baseline.violation.is_some(),
            "DFS pool n={} verdict (violation present? {}) disagrees with n=1 ({})",
            n,
            outcome.violation.is_some(),
            baseline.violation.is_some(),
        );
        if let Some(v) = outcome.violation.as_ref() {
            assert_eq!(v.property_type, PropertyType::Liveness);
            assert!(
                v.message.contains("Idle"),
                "DFS pool n={} violation must mention Idle, got: {}",
                n,
                v.message
            );
        }
    }
}
