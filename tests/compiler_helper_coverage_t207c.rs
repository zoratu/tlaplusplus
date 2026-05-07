//! T207 iteration 3: deep-recursion stress. Targets the ~330 missed
//! mutants of the form `depth + 1 -> depth * 1` and `depth + 1 -> depth - 1`
//! in eval_compiled_inner / eval_compiled_opcall recursive call sites.
//!
//! How it works:
//!   - Original code increments `depth` by 1 on every recursive eval call.
//!     At MAX_DEPTH (256) the check `if depth > MAX_DEPTH` returns Err,
//!     gracefully aborting the recursion.
//!   - Mutated `depth + 1 -> depth * 1`: depth stays at 0 → the depth
//!     check never fires → recursion finishes successfully with a value.
//!     We assert Err: any Ok result fails the test, catching the mutation.
//!   - Mutated `depth + 1 -> depth - 1`: depth underflows on first call
//!     (usize wraps) → `depth > MAX_DEPTH` fires immediately → Err on
//!     ANY input, even shallow ones. Shallow tests asserting Ok catch it.

use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled};

/// Deep nesting count — must exceed MAX_DEPTH = 256 in compiled_eval.rs.
const DEEP: usize = 280;

fn deep_eval(expr: &str) -> anyhow::Result<TlaValue> {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    eval_compiled(&compile_expr(expr), &ctx)
}

/// Assert that evaluating `expr` returns Err with a recursion-related
/// message. Catches the `depth + 1 -> depth * 1` mutation: the mutation
/// keeps depth at 0, the depth check never fires, and a finite-but-deep
/// expression evaluates successfully (Ok) instead of erroring out.
fn assert_recursion_err(expr: &str, ctx_label: &str) {
    let r = deep_eval(expr);
    match r {
        Err(e)
            if e.to_string().contains("depth")
                || e.to_string().contains("recursion") =>
        {
            // Original: depth check fires at MAX_DEPTH+1 → Err. Pass.
        }
        Ok(v) => panic!(
            "{ctx_label}: expected recursion-depth Err but got Ok({v:?}). \
             Indicates the depth counter never incremented (mutated `+1` to `*1`/`*0`)."
        ),
        Err(e) => panic!(
            "{ctx_label}: expected recursion-depth Err but got non-recursion Err: {e}"
        ),
    }
}

// ============================================================
// Deep recursion → assert Err. These catch `depth + 1 -> depth * 1`:
// without depth tracking, the finite (but deep) expression completes
// successfully; with depth tracking, the check fires at MAX_DEPTH+1.
// ============================================================

#[test]
fn t207c_deep_if_then_must_hit_depth_limit() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("IF TRUE THEN ({expr}) ELSE 0");
    }
    assert_recursion_err(&expr, "deep IF/THEN");
}

#[test]
fn t207c_deep_if_else_must_hit_depth_limit() {
    let mut expr = "99".to_string();
    for _ in 0..DEEP {
        expr = format!("IF FALSE THEN 0 ELSE ({expr})");
    }
    assert_recursion_err(&expr, "deep IF/ELSE");
}

#[test]
fn t207c_deep_let_must_hit_depth_limit() {
    let mut expr = "7".to_string();
    for i in 0..DEEP {
        expr = format!("LET a{i} == 0 IN ({expr})");
    }
    assert_recursion_err(&expr, "deep LET");
}

#[test]
fn t207c_deep_not_must_hit_depth_limit() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("~({expr})");
    }
    assert_recursion_err(&expr, "deep NOT");
}

#[test]
fn t207c_deep_nested_and_must_hit_depth_limit() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE /\\ ({expr}))");
    }
    assert_recursion_err(&expr, "deep AND");
}

#[test]
fn t207c_deep_nested_or_must_hit_depth_limit() {
    let mut expr = "FALSE".to_string();
    for _ in 0..DEEP {
        expr = format!("(FALSE \\/ ({expr}))");
    }
    assert_recursion_err(&expr, "deep OR");
}

#[test]
fn t207c_deep_implies_must_hit_depth_limit() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE => ({expr}))");
    }
    assert_recursion_err(&expr, "deep =>");
}

#[test]
fn t207c_deep_iff_must_hit_depth_limit() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE <=> ({expr}))");
    }
    assert_recursion_err(&expr, "deep <=>");
}

#[test]
fn t207c_deep_record_chain_must_hit_depth_limit() {
    let mut expr = "1".to_string();
    for _ in 0..(DEEP / 2) {
        expr = format!("[a |-> {expr}].a");
    }
    assert_recursion_err(&expr, "deep record");
}

#[test]
fn t207c_deep_choose_must_hit_depth_limit() {
    let mut expr = "0".to_string();
    for _ in 0..DEEP {
        expr = format!("CHOOSE x \\in {{{expr}}} : TRUE");
    }
    assert_recursion_err(&expr, "deep CHOOSE");
}

#[test]
fn t207c_deep_addition_pairs_must_hit_depth_limit() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(1 + ({expr}))");
    }
    assert_recursion_err(&expr, "deep addition");
}

#[test]
fn t207c_deep_operator_call_must_hit_depth_limit() {
    let mut expr = "{1}".to_string();
    for _ in 0..(DEEP / 2) {
        expr = format!("Cardinality({{{expr}}})");
    }
    assert_recursion_err(&expr, "deep Cardinality");
}

// ============================================================
// Parens are stripped at compile time, no recursion at eval. This test
// asserts ALWAYS Ok — catches mutations of strip_outer_parens.
// ============================================================

#[test]
fn t207c_deep_parens_compile_to_leaf() {
    let mut expr = "42".to_string();
    for _ in 0..DEEP {
        expr = format!("({expr})");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(42));
}

// ============================================================
// Sub-MAX_DEPTH success cases — assert Ok. Catch `depth + 1 -> depth - 1`
// (depth underflows for usize → check fires immediately on first call).
// ============================================================

#[test]
fn t207c_shallow_if_chain_must_succeed() {
    let mut expr = "1".to_string();
    for _ in 0..50 {
        expr = format!("IF TRUE THEN ({expr}) ELSE 0");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(1));
}

#[test]
fn t207c_shallow_let_chain_must_succeed() {
    let mut expr = "7".to_string();
    for i in 0..50 {
        expr = format!("LET v{i} == 0 IN ({expr})");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(7));
}

#[test]
fn t207c_shallow_not_chain_must_succeed() {
    let mut expr = "TRUE".to_string();
    for _ in 0..50 {
        expr = format!("~({expr})");
    }
    // 50 NOTs of TRUE = TRUE (even count)
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Bool(true));
}

#[test]
fn t207c_shallow_record_chain_must_succeed() {
    let mut expr = "42".to_string();
    for _ in 0..50 {
        expr = format!("[a |-> {expr}].a");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(42));
}

#[test]
fn t207c_shallow_choose_chain_must_succeed() {
    let mut expr = "5".to_string();
    for _ in 0..30 {
        expr = format!("CHOOSE x \\in {{{expr}}} : TRUE");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(5));
}

#[test]
fn t207c_shallow_append_chain_must_succeed() {
    let mut expr = "<<>>".to_string();
    for i in 0..30 {
        expr = format!("Append({expr}, {i})");
    }
    let r = deep_eval(&expr).unwrap();
    if let TlaValue::Seq(s) = &r {
        assert_eq!(s.len(), 30);
    } else {
        panic!("expected Seq, got {r:?}");
    }
}
