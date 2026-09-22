// Terminal-state deadlock detection (roadmap #1).
//
// TLC reports a reachable state with no successors (no action enabled) as a
// deadlock, unless deadlock checking is disabled (`-deadlock` /
// `CHECK_DEADLOCK FALSE`). tlaplusplus previously had NO terminal-deadlock
// detection at all: the `--allow-deadlock` flag only gated the eval-error
// branch, and a quiescent state was silently accepted as a leaf. These tests
// exercise the runtime worker's deadlock detection end-to-end via `run_model`:
//   1. A state that reaches a dead end IS reported as a Deadlock violation
//      when checking is enabled (default).
//   2. The same spec with `allow_deadlock = true` reports NO violation and
//      explores the full state space.
//   3. A spec whose terminal state self-loops is NOT a deadlock (it has a
//      successor), so no false positive.
//   4. An initial state with no successors is itself a deadlock.

use serial_test::serial;
use std::fs;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

/// Build + run a spec, choosing whether deadlock checking is enabled.
/// `allow_deadlock = true` disables the check (TLC's `-deadlock`).
fn run_spec_dl(
    name: &str,
    module_src: &str,
    cfg_src: &str,
    allow_deadlock: bool,
) -> RunOutcome<<TlaModel as Model>::State> {
    let dir = TempDir::new().expect("tempdir");
    let module_path = dir.path().join(format!("{name}.tla"));
    let cfg_path = dir.path().join(format!("{name}.cfg"));
    fs::write(&module_path, module_src).expect("write module");
    fs::write(&cfg_path, cfg_src).expect("write cfg");
    let mut model =
        TlaModel::from_files(&module_path, Some(&cfg_path), None, None).expect("model loads");
    model.allow_deadlock = allow_deadlock;

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

// A chain 0 -> 1 -> 2 -> 3 where x=3 has no enabled action (x < 3 is false).
const CHAIN_DEADLOCK: &str = r#"---- MODULE ChainDeadlock ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
====
"#;

// Same chain, but x=3 self-loops, so it is NOT a deadlock.
const CHAIN_SELFLOOP: &str = r#"---- MODULE ChainSelfLoop ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == \/ (x < 3 /\ x' = x + 1)
        \/ (x = 3 /\ x' = 3)
====
"#;

// The initial state itself has no successor (x < 0 is false at x=0).
const INIT_DEADLOCK: &str = r#"---- MODULE InitDeadlock ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x < 0 /\ x' = x - 1
====
"#;

const CFG: &str = "INIT Init\nNEXT Next\n";

#[test]
#[serial]
fn terminal_state_is_reported_as_deadlock() {
    let outcome = run_spec_dl("ChainDeadlock", CHAIN_DEADLOCK, CFG, /*allow_deadlock=*/ false);

    let v = outcome
        .violation
        .expect("a reachable state with no successors must be reported as a deadlock");
    assert_eq!(
        v.property_type,
        PropertyType::Deadlock,
        "expected a Deadlock property type, got {:?}: {}",
        v.property_type,
        v.message
    );
    assert!(
        v.message.contains("Deadlock reached"),
        "unexpected violation message: {}",
        v.message
    );
    // All four states (x = 0,1,2,3) are discovered before the dead end at x=3
    // is popped and found to have no successors.
    assert_eq!(
        outcome.stats.states_distinct, 4,
        "expected 4 distinct states (x = 0..3), got {}",
        outcome.stats.states_distinct
    );
}

#[test]
#[serial]
fn allow_deadlock_suppresses_the_report() {
    let outcome = run_spec_dl("ChainDeadlock", CHAIN_DEADLOCK, CFG, /*allow_deadlock=*/ true);

    assert!(
        outcome.violation.is_none(),
        "with --allow-deadlock the dead end must not be reported, got: {:?}",
        outcome.violation
    );
    assert_eq!(
        outcome.stats.states_distinct, 4,
        "expected 4 distinct states (x = 0..3), got {}",
        outcome.stats.states_distinct
    );
}

#[test]
#[serial]
fn self_looping_terminal_state_is_not_a_deadlock() {
    // x=3 has a successor (itself), so no action is "disabled" — not a deadlock,
    // even with deadlock checking enabled.
    let outcome = run_spec_dl("ChainSelfLoop", CHAIN_SELFLOOP, CFG, /*allow_deadlock=*/ false);

    assert!(
        outcome.violation.is_none(),
        "a self-looping terminal state must not be a deadlock, got: {:?}",
        outcome.violation
    );
    assert_eq!(
        outcome.stats.states_distinct, 4,
        "expected 4 distinct states (x = 0..3), got {}",
        outcome.stats.states_distinct
    );
}

#[test]
#[serial]
fn initial_state_with_no_successors_is_a_deadlock() {
    let outcome = run_spec_dl("InitDeadlock", INIT_DEADLOCK, CFG, /*allow_deadlock=*/ false);

    let v = outcome
        .violation
        .expect("an initial state with no successors must be reported as a deadlock");
    assert_eq!(v.property_type, PropertyType::Deadlock);
    assert_eq!(
        outcome.stats.states_distinct, 1,
        "expected the single initial state, got {}",
        outcome.stats.states_distinct
    );
}
