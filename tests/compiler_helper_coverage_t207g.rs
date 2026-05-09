//! T207 iteration 8: deep-recursion stress through relop and arithmetic
//! dispatch arms in eval_compiled_inner. Targets ~50+ missed mutations
//! of the form `depth + 1 -> depth * 1` on lines 239–265 (Eq, Neq, Lt,
//! Le, Gt, Ge), 271–280 (In/NotIn), and 294+ (Add, Sub, Mul, Div, Mod).
//!
//! Strategy: chain N nested binary ops where each level drives the
//! corresponding dispatch arm via the left operand. With MAX_DEPTH=256
//! and DEEP=280, recursion via the outer chain drives the depth check
//! to fire on the original code; mutated `depth + 1 -> depth * 1` keeps
//! depth at 0 and the finite tree completes successfully (Ok).

use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled};

const DEEP: usize = 280;

fn deep_eval(expr: &str) -> anyhow::Result<TlaValue> {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    eval_compiled(&compile_expr(expr), &ctx)
}

fn assert_recursion_err(expr: &str, ctx_label: &str) {
    let r = deep_eval(expr);
    match r {
        Err(e)
            if e.to_string().contains("depth")
                || e.to_string().contains("recursion") => {}
        Ok(v) => panic!(
            "{ctx_label}: expected recursion-depth Err but got Ok({v:?})"
        ),
        Err(e) => panic!(
            "{ctx_label}: expected recursion-depth Err but got: {e}"
        ),
    }
}

// ============================================================
// Relops: Eq, Neq, Lt, Le, Gt, Ge
// Build a left-leaning tree of nested comparisons. Each level adds
// one Eq/Lt/etc. dispatch on the path.
// ============================================================

#[test]
fn t207g_deep_eq_chain() {
    // ((((TRUE = TRUE) = TRUE) = TRUE) ... ) = TRUE
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("({expr}) = TRUE");
    }
    assert_recursion_err(&expr, "deep Eq");
}

#[test]
fn t207g_deep_neq_chain() {
    let mut expr = "FALSE".to_string();
    for _ in 0..DEEP {
        expr = format!("({expr}) # FALSE");
    }
    assert_recursion_err(&expr, "deep Neq");
}

#[test]
fn t207g_deep_lt_chain() {
    // Build a deeply-nested expression that ends with `... < N` where each
    // wrap is `IF e < N THEN 1 ELSE 0` to keep returning Int.
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) < 100 THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep Lt");
}

#[test]
fn t207g_deep_le_chain() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) <= 100 THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep Le");
}

#[test]
fn t207g_deep_gt_chain() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) > 0 THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep Gt");
}

#[test]
fn t207g_deep_ge_chain() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) >= 0 THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep Ge");
}

// ============================================================
// In / NotIn
// ============================================================

#[test]
fn t207g_deep_in_chain() {
    // Wrap each level in `IF (val \in {val}) THEN val ELSE val`
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) \\in {{1}} THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep \\in");
}

#[test]
fn t207g_deep_notin_chain() {
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(IF ({expr}) \\notin {{99}} THEN 1 ELSE 0)");
    }
    assert_recursion_err(&expr, "deep \\notin");
}

// ============================================================
// Arithmetic: Add, Sub, Mul, Div, Mod
// (Add/Sub need to avoid 3+ chains that trigger T206 Unparsed delegation.)
// ============================================================

#[test]
fn t207g_deep_add_pairs() {
    // (1 + (1 + (... 1 + 1))) — each level adds one Add dispatch.
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(1 + ({expr}))");
    }
    assert_recursion_err(&expr, "deep Add");
}

#[test]
fn t207g_deep_sub_pairs() {
    let mut expr = "1000000".to_string();
    for _ in 0..DEEP {
        expr = format!("(({expr}) - 1)");
    }
    assert_recursion_err(&expr, "deep Sub");
}

#[test]
fn t207g_deep_mul_pairs() {
    // 1 * 1 * ... * 1 = 1 (no overflow). Use (1 * inner) to nest.
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(1 * ({expr}))");
    }
    assert_recursion_err(&expr, "deep Mul");
}

#[test]
fn t207g_deep_div_pairs() {
    // (val \div 1) leaves val unchanged.
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(({expr}) \\div 1)");
    }
    assert_recursion_err(&expr, "deep Div");
}

#[test]
fn t207g_deep_mod_pairs() {
    // (val % 100) → eventually yields val % 100. Nest.
    let mut expr = "5".to_string();
    for _ in 0..DEEP {
        expr = format!("(({expr}) % 100)");
    }
    assert_recursion_err(&expr, "deep Mod");
}

// ============================================================
// Shallow versions of the above — assert Ok. Catch `+1 -> -1` mutations
// that would underflow depth on the first call.
// ============================================================

#[test]
fn t207g_shallow_eq_chain_ok() {
    let mut expr = "TRUE".to_string();
    for _ in 0..30 {
        expr = format!("({expr}) = TRUE");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Bool(true));
}

#[test]
fn t207g_shallow_lt_chain_ok() {
    let mut expr = "1".to_string();
    for _ in 0..30 {
        expr = format!("(IF ({expr}) < 100 THEN 1 ELSE 0)");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(1));
}

#[test]
fn t207g_shallow_in_chain_ok() {
    let mut expr = "1".to_string();
    for _ in 0..30 {
        expr = format!("(IF ({expr}) \\in {{1}} THEN 1 ELSE 0)");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(1));
}

#[test]
fn t207g_shallow_add_pairs_ok() {
    let mut expr = "0".to_string();
    for _ in 0..30 {
        expr = format!("(1 + ({expr}))");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(30));
}

#[test]
fn t207g_shallow_sub_pairs_ok() {
    let mut expr = "100".to_string();
    for _ in 0..30 {
        expr = format!("(({expr}) - 1)");
    }
    assert_eq!(deep_eval(&expr).unwrap(), TlaValue::Int(70));
}
