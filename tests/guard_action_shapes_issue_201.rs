//! Regression matrix for issue #201: a guard followed by an action in every
//! degenerate parenthesization must keep its successor, for both operator-call
//! and inline primed actions. Each test checks that, from state (pc="a", ph="ok")
//! where only the guarded branch is enabled, the sole successor has pc'="b".
//! A dropped successor (empty) or an error marks a regression.
//!
//! Pre-fix history: the operator-call shapes were fixed in the first #201 pass;
//! the inline `((IA))` (double-paren) and `(IA \/ FALSE)` shapes were fixed by
//! the `classify_clause` top-level-junction guard (see src/tla/formula.rs).

use tlaplusplus::tla::{TlaValue, evaluate_next_states, parse_tla_module_text, tla_state};

const PREAMBLE: &str = r#"VARIABLES pc, ph
Init == pc = "a" /\ ph = "pend"
Arm  == ph = "pend" /\ ph' = "ok" /\ UNCHANGED pc
Body == pc = "a" /\ pc' = "b" /\ UNCHANGED ph
"#;

fn pcs_from_a_ok(next_body: &str) -> Result<Vec<String>, String> {
    let src = format!("---- MODULE Probe ----\n{PREAMBLE}Next == {next_body}\n====\n");
    let module = parse_tla_module_text(&src).map_err(|e| e.to_string())?;
    let next = module.definitions.get("Next").ok_or("no Next")?;
    let state = tla_state([
        ("pc", TlaValue::String("a".to_string())),
        ("ph", TlaValue::String("ok".to_string())),
    ]);
    let succ = evaluate_next_states(&next.body, &module.definitions, &state)
        .map_err(|e| e.to_string())?;
    Ok(succ
        .iter()
        .filter_map(|s| match s.get("pc") {
            Some(TlaValue::String(x)) => Some(x.clone()),
            _ => None,
        })
        .collect())
}

fn check(name: &str, next_body: &str) {
    let r = pcs_from_a_ok(next_body);
    let ok = matches!(&r, Ok(v) if v == &vec!["b".to_string()]);
    eprintln!("[{}] body={:<55} => {:?}  {}", if ok {"PASS"} else {"FAIL"}, next_body, r, name);
    assert!(ok, "{name}: expected [b], got {r:?}  (body: {next_body})");
}

// ---- operator-call action shapes ----
#[test] fn opcall_bare()          { check("g /\\ OpBody (control)",      "Arm \\/ (ph = \"ok\" /\\ Body)"); }
#[test] fn opcall_paren()         { check("g /\\ (OpBody)",              "Arm \\/ (ph = \"ok\" /\\ (Body))"); }
#[test] fn opcall_double_paren()  { check("g /\\ ((OpBody))",            "Arm \\/ (ph = \"ok\" /\\ ((Body)))"); }
#[test] fn opcall_or_false()      { check("g /\\ (OpBody \\/ FALSE)",    "Arm \\/ (ph = \"ok\" /\\ (Body \\/ FALSE))"); }

// ---- inline primed action shapes ----
const IA: &str = "pc' = \"b\" /\\ UNCHANGED ph";
#[test] fn inline_bare()          { check("g /\\ IA",       &format!("Arm \\/ (ph = \"ok\" /\\ {IA})")); }
#[test] fn inline_paren()         { check("g /\\ (IA)",     &format!("Arm \\/ (ph = \"ok\" /\\ ({IA}))")); }
#[test] fn inline_double_paren()  { check("g /\\ ((IA))",   &format!("Arm \\/ (ph = \"ok\" /\\ (({IA})))")); }
#[test] fn inline_or_false()      { check("g /\\ (IA \\/ FALSE)", &format!("Arm \\/ (ph = \"ok\" /\\ ({IA} \\/ FALSE))")); }
#[test] fn inline_bulleted()      { check("g /\\ (\\/ IA)", &format!("Arm \\/ (ph = \"ok\" /\\ (\\/ {IA}))")); }
