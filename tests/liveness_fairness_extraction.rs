//! Bug A regression: fairness written in a NAMED operator or a QUANTIFIED
//! `\A n \in N : WF_v(A(n))` must be extracted from the SPECIFICATION, not
//! silently dropped. Dropping it made infinite stuttering in `Init` look fair,
//! yielding a false `<>[]Q` violation (found cross-checking Fermion's DQS specs
//! against Java TLC).
//!
//! These run `run_model` directly (no Java/TLC needed) and assert the verdict.
//! The companion diff_tlc corpus entries (fairness_named_wf, fairness_quantified_wf,
//! fairness_inline_wf) cross-check the same specs against TLC in CI.

use std::fs;

use serial_test::serial;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, RunOutcome, run_model};

fn run_spec(module_name: &str, module_src: &str, cfg_src: &str) -> RunOutcome<<TlaModel as Model>::State> {
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

const NAMED_WF: &str = r#"---- MODULE ReproNamedWF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fairness == WF_x(Next)
Spec == Init /\ [][Next]_x /\ Fairness
EvTwo == <>[](x = 2)
====
"#;

const QUANT_WF: &str = r#"---- MODULE ReproQuantWF ----
CONSTANT N
VARIABLE x
Init == x = [n \in N |-> 0]
Set(n) == x[n] = 0 /\ x' = [x EXCEPT ![n] = 1]
Next == \E n \in N : Set(n)
Spec == Init /\ [][Next]_x /\ \A n \in N : WF_x(Set(n))
AllSet == <>[](\A n \in N : x[n] = 1)
====
"#;

const INLINE_WF: &str = r#"---- MODULE ReproFairWF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Spec == Init /\ [][Next]_x /\ WF_x(Next)
EvTwo == <>[](x = 2)
====
"#;

#[test]
#[serial]
fn named_operator_wf_is_extracted() {
    // `Fairness == WF_x(Next)`: under WF every fair behavior reaches x=2, so
    // `<>[](x=2)` HOLDS. Before the fix the named WF was dropped and tlapp
    // reported a false violation from stutter-at-Init.
    let outcome = run_spec("ReproNamedWF", NAMED_WF, "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nPROPERTIES\n  EvTwo\n");
    assert!(
        outcome.violation.is_none(),
        "named-operator WF must be extracted so <>[](x=2) holds; got a false violation: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn quantified_wf_is_expanded() {
    // `\A n \in N : WF_x(Set(n))` over N={a,b}: under fairness every node is set,
    // so `<>[](\A n : x[n]=1)` HOLDS. Before the fix the quantified WF was dropped.
    let outcome = run_spec("ReproQuantWF", QUANT_WF, "CONSTANT N = {a, b}\nSPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nPROPERTIES\n  AllSet\n");
    assert!(
        outcome.violation.is_none(),
        "quantified WF must be expanded over N so <>[](all set) holds; got a false violation: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn inline_wf_still_works() {
    // Control: the pre-existing inline-WF path must keep working after the rewrite.
    let outcome = run_spec("ReproFairWF", INLINE_WF, "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nPROPERTIES\n  EvTwo\n");
    assert!(
        outcome.violation.is_none(),
        "inline WF_x(Next) must still be extracted; got a false violation: {:?}",
        outcome.violation
    );
}
