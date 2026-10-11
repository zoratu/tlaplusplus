//! Regression for GitHub issue #201: a guard followed by a parenthesized or
//! single-branch disjunction of *operator-call* actions dropped successors
//! (missed-violation / under-exploration).
//!
//! The crux: `Arm`/`Body` are named operators, so the prime (`pc' = "b"`) lives
//! inside `Body`'s definition and is invisible in the `Next` body text. The
//! compiled guard path classified the clause `(Body \/ FALSE)` / `\/ Body` as a
//! boolean, evaluated `Body` in value context (dropping its prime), and produced
//! no successor. See `try_eval_guard_disjunction_as_action` in
//! `src/tla/compiled_eval.rs`.
//!
//! These tests drive `evaluate_next_states` directly from the state `(pc="a",
//! ph="ok")`, where only the guarded disjunction branch is enabled: `Arm`'s
//! guard `ph = "pend"` is false, so the sole successor must come from `Body`
//! (`pc' = "b"`). Pre-fix this yielded 0 successors; TLC yields 1.

use tlaplusplus::tla::{TlaValue, evaluate_next_states, parse_tla_module_text, tla_state};

fn body_successor_count_from_a_ok(module_src: &str) -> Vec<String> {
    let module = parse_tla_module_text(module_src).expect("inline #201 spec parses");
    let next = module
        .definitions
        .get("Next")
        .expect("Next definition present");
    // State (pc="a", ph="ok"): Arm is disabled (ph != "pend"); only the guarded
    // disjunction branch can fire, and it must step pc -> "b".
    let state = tla_state([
        ("pc", TlaValue::String("a".to_string())),
        ("ph", TlaValue::String("ok".to_string())),
    ]);
    let successors = evaluate_next_states(&next.body, &module.definitions, &state)
        .expect("guard-disjunction Next must evaluate cleanly");
    successors
        .iter()
        .map(|s| {
            s.get("pc")
                .and_then(|v| match v {
                    TlaValue::String(x) => Some(x.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        })
        .collect()
}

const PREAMBLE: &str = r#"VARIABLES pc, ph
Init == pc = "a" /\ ph = "pend"
Arm  == ph = "pend" /\ ph' = "ok" /\ UNCHANGED pc
Body == pc = "a" /\ pc' = "b" /\ UNCHANGED ph
"#;

#[test]
fn guard_then_parenthesized_disjunction_with_false_keeps_body_successor() {
    // Row 2 of the issue's table (the headline shape): `(Body \/ FALSE)`.
    let src = format!(
        "---- MODULE Issue201Paren ----\n{PREAMBLE}\
         Next == Arm \\/ (ph = \"ok\" /\\ (Body \\/ FALSE))\n====\n"
    );
    let pcs = body_successor_count_from_a_ok(&src);
    assert_eq!(
        pcs,
        vec!["b".to_string()],
        "guard /\\ (Body \\/ FALSE) must yield exactly the Body successor (pc'=\"b\"), got {pcs:?}"
    );
}

#[test]
fn guard_then_single_bulleted_disjunction_keeps_body_successor() {
    // Row 4 of the issue's table: a bulleted one-item inner list `/\ \/ Body`.
    let src = format!(
        "---- MODULE Issue201Bullet ----\n{PREAMBLE}\
         Next == \\/ Arm\n        \\/ /\\ ph = \"ok\"\n           /\\ \\/ Body\n====\n"
    );
    let pcs = body_successor_count_from_a_ok(&src);
    assert_eq!(
        pcs,
        vec!["b".to_string()],
        "guard /\\ (\\/ Body) must yield exactly the Body successor (pc'=\"b\"), got {pcs:?}"
    );
}

#[test]
fn guard_then_bare_action_still_works_control() {
    // Row 1 control (already worked pre-fix): `g /\ Body`, no inner disjunction.
    let src = format!(
        "---- MODULE Issue201Bare ----\n{PREAMBLE}\
         Next == Arm \\/ (ph = \"ok\" /\\ Body)\n====\n"
    );
    let pcs = body_successor_count_from_a_ok(&src);
    assert_eq!(
        pcs,
        vec!["b".to_string()],
        "control: guard /\\ Body must yield the Body successor, got {pcs:?}"
    );
}
