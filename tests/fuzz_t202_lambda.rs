//! T202 — regression test for the compiled-vs-interpreted Ok-vs-different-value
//! divergence surfaced by `fuzz/fuzz_targets/fuzz_tla_swarm.rs` on a 35-byte
//! input whose `==`-defined body began with `LAMBDA\t\t...`.
//!
//! Pre-fix:
//!   - Compiler: `expr.starts_with("LAMBDA ")` (literal space) — missed
//!     `LAMBDA\t...`, fell through to the comparison parser, returned
//!     `Bool(false)` from `=I` chains in the body.
//!   - Interpreter: `starts_with_keyword(expr, "LAMBDA")` — accepted any
//!     non-word boundary, took the LAMBDA branch, returned `TlaValue::Lambda`.
//!
//! Same family as the closed T101.1 set — compiler being more lenient on
//! malformed-but-keyword-prefixed input than the interpreter. Fix
//! (`src/tla/compiled_expr.rs`):
//!   1. `starts_with_lambda_keyword` mirrors `eval::starts_with_keyword`.
//!   2. `strip_lambda_keyword` is the matching extractor used by
//!      `try_parse_lambda`.
//!   3. When the LAMBDA prefix is recognised but the body shape is rejected
//!      (e.g. params that are not identifiers), the compiler emits
//!      `Unparsed(expr)` so the interpreter handles the input — guaranteeing
//!      result-parity on every LAMBDA-prefixed input.

use tlaplusplus::tla::module::parse_tla_module_text;
use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

/// Run both evaluators against an empty state and assert they either both
/// Err or both Ok with equal values. Mirrors `fuzz_tla_swarm::assert_equivalent`.
fn assert_equivalent(expr: &str) {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr(expr, &ctx);
    let compi = eval_compiled(&compile_expr(expr), &ctx);
    match (&interp, &compi) {
        (Ok(a), Ok(b)) => assert_eq!(
            a, b,
            "DIVERGENCE on `{expr}`:\n  interpreter -> {a:?}\n  compiler    -> {b:?}"
        ),
        // Ok-vs-Err is tolerated by the fuzz harness; only Ok-with-different-
        // values is a hard divergence.
        _ => {}
    }
}

/// The original 35-byte fuzz artefact — verbatim. The first two bytes are
/// the swarm config flags; the rest is the TLA+ module text.
const T202_BYTES: &[u8] = &[
    0x47, 0x00, 0x47, 0x00, 0x04, 0x00, 0x00, 0x47, 0x00, 0x00, 0x02, 0x5b, 0x04, 0x3d, 0x3d, 0x4c,
    0x41, 0x4d, 0x42, 0x44, 0x41, 0x09, 0x09, 0x3d, 0x3d, 0x49, 0x01, 0x00, 0x00, 0x3a, 0x09, 0x49,
    0x09, 0x3d, 0x49,
];

#[test]
fn t202_lambda_tab_keyword_boundary_via_module() {
    // Walk the same path as the swarm fuzz target: parse the module text,
    // then evaluate every definition body with both evaluators.
    let text = std::str::from_utf8(&T202_BYTES[2..]).expect("utf-8");
    let module = parse_tla_module_text(text).expect("parse should succeed");
    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    for (_name, def) in &module.definitions {
        if def.body.is_empty() {
            continue;
        }
        let interp = eval_expr(&def.body, &ctx);
        let compi = eval_compiled(&compile_expr(&def.body), &ctx);
        if let (Ok(a), Ok(b)) = (&interp, &compi) {
            assert_eq!(
                a, b,
                "T202 DIVERGENCE on body {:?}:\n  interpreter -> {a:?}\n  compiler    -> {b:?}",
                def.body
            );
        }
    }
}

#[test]
fn t202_lambda_followed_by_tab() {
    // Direct repro of the failing body: `LAMBDA\t...` should take the LAMBDA
    // branch on both sides. Pre-fix: interpreter -> Lambda, compiler -> Bool.
    assert_equivalent("LAMBDA\tx: x");
}

#[test]
fn t202_lambda_followed_by_newline() {
    // The `LAMBDA\n` prefix is the other interpreter-accepted shape that
    // the literal-space check would have missed.
    assert_equivalent("LAMBDA\nx: x");
}

#[test]
fn t202_lambda_with_non_identifier_param_falls_back_to_interpreter() {
    // When the LAMBDA prefix is present but the params are not identifiers
    // (e.g. `==I` from the original artefact), the compiler used to fall
    // through to the comparison parser and silently produce `Bool(...)`.
    // After the fix the compiler emits `Unparsed(expr)` so the interpreter
    // handles the input and the two paths agree.
    assert_equivalent("LAMBDA\t\t==I  :\tI\t=I");
}

#[test]
fn t202_well_formed_lambda_still_compiles() {
    // Sanity: the compiler must still take the fast LAMBDA path on
    // well-formed input — we are tightening the prefix detector, not
    // disabling the compiled lambda evaluator.
    assert_equivalent("LAMBDA x: x + 1");
    assert_equivalent("LAMBDA x, y: x + y");
}
