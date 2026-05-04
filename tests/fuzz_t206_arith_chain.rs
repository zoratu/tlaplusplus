//! T206: regression — chained binary `-`/`+` arithmetic with `^^` XOR
//! produced different answers in interpreter vs compiler due to splitter-
//! cascade associativity divergences. Fix: compiler emits Unparsed for
//! arithmetic chains with 3+ binary `-`/`+` occurrences, so the
//! interpreter (the reference) parses them. This test pins the headline
//! divergence (interp -56 vs compiler -48 on the original fuzz input)
//! and a few related shapes.

use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

fn assert_equiv(expr: &str) {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr(expr, &ctx);
    let compi = eval_compiled(&compile_expr(expr), &ctx);
    match (&interp, &compi) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "DIVERGENCE on `{expr}`: interp={a:?} compi={b:?}"),
        (Err(_), Err(_)) => {}
        _ => panic!("Ok-vs-Err mismatch on `{expr}`: interp={interp:?} compi={compi:?}"),
    }
}

#[test]
fn t206_original_fuzz_repro_no_divergence() {
    assert_equiv("0-2--442-0--0-4-2-0-0-442-0--0-44^^4");
}

#[test]
fn t206_chained_subtraction_no_divergence() {
    assert_equiv("10-3-2-1");
    assert_equiv("100-50-25-12-6-3-1");
}

#[test]
fn t206_chained_with_xor_no_divergence() {
    assert_equiv("10-2-1^^4");
    assert_equiv("0-1-2^^3-4");
}

#[test]
fn t206_simple_shapes_still_compile_to_structured_ast() {
    use tlaplusplus::tla::compiled_expr::compile_expr;
    use tlaplusplus::tla::CompiledExpr;
    // `a - b` — single binary minus, NOT a chain — should stay structured
    assert!(matches!(compile_expr("a - b"), CompiledExpr::Sub(_, _)),
            "single binary minus should stay structured");
    // `x * -3` — unary minus, no chain — should stay structured
    assert!(matches!(compile_expr("x * -3"), CompiledExpr::Mul(_, _)),
            "single multiply with unary literal should stay structured");
}
