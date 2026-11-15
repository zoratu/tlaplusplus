//! Bug A regression: fairness written in a NAMED operator (including an alias
//! chain), or as strong fairness `SF_`, must be extracted from the SPECIFICATION,
//! not silently dropped. Dropping it made infinite stuttering in `Init` look
//! fair, yielding a false `<>[]Q` violation (found cross-checking Fermion's DQS
//! specs against Java TLC).
//!
//! Each spec is constructed so the property is FALSE without fairness (stutter in
//! Init) and TRUE with it — so a passing "no violation" result genuinely depends
//! on the fairness being extracted, not on the checker being lenient.
//!
//! Runs `run_model` directly (no Java/TLC needed). The companion diff_tlc corpus
//! entries (fairness_named_wf, fairness_inline_wf) cross-check against TLC in CI.
//!
//! NOTE: quantified fairness `\A n \in N : WF_x(Set(n))` is intentionally NOT
//! covered here — it needs per-instance fairness and is deferred to the Bug B/C
//! fairness-filter rewrite (see extract_spec_fairness docs and PR #212 review).

use std::fs;

use serial_test::serial;
use tempfile::TempDir;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::{EngineConfig, Model, PropertyType, RunOutcome, run_model};

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

const CFG: &str = "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nPROPERTIES\n  EvTwo\n";

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

// Alias chain: Spec references `Fair1`, which is `Fair2`, which is the WF.
// Exercises transitive named-operator unfolding (Codex review #5).
const ALIAS_CHAIN_WF: &str = r#"---- MODULE ReproAliasWF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fair2 == WF_x(Next)
Fair1 == Fair2
Spec == Init /\ [][Next]_x /\ Fair1
EvTwo == <>[](x = 2)
====
"#;

// Strong fairness inside a named operator — exercises the SF_ extraction path.
const NAMED_SF: &str = r#"---- MODULE ReproNamedSF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fairness == SF_x(Next)
Spec == Init /\ [][Next]_x /\ Fairness
EvTwo == <>[](x = 2)
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

// Whole Spec body wrapped in parens — exercises strip-before-split (Codex re-review #1).
const PAREN_NAMED_WF: &str = r#"---- MODULE ReproParenWF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fairness == WF_x(Next)
Spec == (Init /\ [][Next]_x /\ Fairness)
EvTwo == <>[](x = 2)
====
"#;

// Negative control: NO fairness, so x may stutter at 0 forever and `<>[](x=2)`
// is VIOLATED. Proves the "no violation" results above genuinely depend on
// fairness being extracted (Codex re-review suggestion).
const NO_FAIRNESS: &str = r#"---- MODULE ReproNoFair ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Spec == Init /\ [][Next]_x
EvTwo == <>[](x = 2)
====
"#;

#[test]
#[serial]
fn named_operator_wf_is_extracted() {
    let outcome = run_spec("ReproNamedWF", NAMED_WF, CFG);
    assert!(
        outcome.violation.is_none(),
        "named-operator WF must be extracted so <>[](x=2) holds; got a false violation: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn alias_chain_named_wf_is_extracted() {
    let outcome = run_spec("ReproAliasWF", ALIAS_CHAIN_WF, CFG);
    assert!(
        outcome.violation.is_none(),
        "transitive (alias-chain) named WF must be unfolded so <>[](x=2) holds; got: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn named_operator_sf_is_extracted() {
    let outcome = run_spec("ReproNamedSF", NAMED_SF, CFG);
    assert!(
        outcome.violation.is_none(),
        "named-operator SF must be extracted so <>[](x=2) holds; got a false violation: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn inline_wf_still_works() {
    // Control: the pre-existing inline-WF path must keep working after the rewrite.
    let outcome = run_spec("ReproFairWF", INLINE_WF, CFG);
    assert!(
        outcome.violation.is_none(),
        "inline WF_x(Next) must still be extracted; got a false violation: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn parenthesized_spec_named_wf_is_extracted() {
    let outcome = run_spec("ReproParenWF", PAREN_NAMED_WF, CFG);
    assert!(
        outcome.violation.is_none(),
        "named WF inside a fully-parenthesized Spec must be extracted; got: {:?}",
        outcome.violation
    );
}

#[test]
#[serial]
fn no_fairness_violates() {
    // The baseline: with NO fairness, stutter-at-Init violates <>[](x=2). This
    // confirms the positive tests' "no violation" is caused by extraction, not by
    // the checker failing to detect the violation at all.
    let outcome = run_spec("ReproNoFair", NO_FAIRNESS, CFG);
    let v = outcome.violation.as_ref().expect(
        "with no fairness, <>[](x=2) must be VIOLATED (stutter at Init); got no violation",
    );
    assert_eq!(
        v.property_type,
        PropertyType::Liveness,
        "the no-fairness violation must be a liveness-property violation, not something else"
    );
}

const NESTED_GROUP_WF: &str = r#"---- MODULE ReproNestedGroupWF ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
Fairness == WF_x(Next)
Spec == (Init /\ [][Next]_x) /\ Fairness
EvTwo == <>[](x = 2)
====
"#;

#[test]
#[serial]
fn nested_group_spec_loads_and_extracts() {
    // `Spec == (Init /\ [][Next]_x) /\ Fairness`: the parenthesized group must be
    // re-split by BOTH extractors (Init/Next and fairness), or the model fails to
    // load or drops the fairness. With both consistent, <>[](x=2) HOLDS.
    let outcome = run_spec("ReproNestedGroupWF", NESTED_GROUP_WF, CFG);
    assert!(
        outcome.violation.is_none(),
        "nested parenthesized group must be re-split by both extractors; got: {:?}",
        outcome.violation
    );
}
