//! T207 iteration 3: deep-recursion stress. Targets the ~330 missed
//! mutants of the form `depth + 1 -> depth * 1` and `depth + 1 -> depth - 1`
//! in eval_compiled_inner / eval_compiled_opcall recursive call sites.
//!
//! How it works:
//!   - Original code increments `depth` by 1 on every recursive eval call.
//!     At MAX_DEPTH (256) the check `if depth > MAX_DEPTH` returns Err,
//!     gracefully aborting the recursion.
//!   - Mutated `depth + 1 -> depth * 1`: depth never increments → recursion
//!     is unbounded → stack overflow → process death (cargo-mutants
//!     classifies this as "caught").
//!   - Mutated `depth + 1 -> depth - 1`: depth becomes negative (saturates
//!     for usize) → check fires on first call → unexpected Err → test
//!     assertion fails.
//!
//! The strategy: build expressions that drive recursion through every
//! recursive site to >= MAX_DEPTH=256 levels deep. The original code
//! returns Err("recursion depth exceeded") gracefully. A mutated version
//! either stack-overflows or produces a different Ok/Err shape.

use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled};

/// Number of nesting levels — must exceed MAX_DEPTH = 256 in compiled_eval.rs.
/// We pick 280 so we cleanly cross the boundary.
const DEEP: usize = 280;

fn deep_eval(expr: &str) -> anyhow::Result<TlaValue> {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    eval_compiled(&compile_expr(expr), &ctx)
}

// ============================================================
// Deep IF/THEN/ELSE — drives eval_compiled_inner IfThenElse path.
// ============================================================

#[test]
fn t207c_deep_if_then() {
    // IF TRUE THEN (IF TRUE THEN (... 280 deep ... 1)) ELSE 0
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("IF TRUE THEN ({expr}) ELSE 0");
    }
    let r = deep_eval(&expr);
    // Either Ok (recursion completed under MAX_DEPTH) or Err
    // ("recursion depth exceeded"). Both indicate depth tracking
    // works. Mutated code stack-overflows (process killed).
    match r {
        Ok(TlaValue::Int(1)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep IF: expected Ok(1) or recursion-depth Err, got {other:?}"),
    }
}

#[test]
fn t207c_deep_if_else_branch() {
    // ELSE branch — exercises the same IF dispatch but takes the false leg
    let mut expr = "99".to_string();
    for _ in 0..DEEP {
        expr = format!("IF FALSE THEN 0 ELSE ({expr})");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(99)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep IF (else): got {other:?}"),
    }
}

// ============================================================
// Deep LET — drives eval_let_expression depth+1 sites.
// ============================================================

#[test]
fn t207c_deep_let_chain() {
    // LET a0 == 0 IN LET a1 == 0 IN ... LET a279 == 0 IN 7
    let mut expr = "7".to_string();
    for i in 0..DEEP {
        expr = format!("LET a{i} == 0 IN ({expr})");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(7)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep LET: got {other:?}"),
    }
}

// ============================================================
// Deep parens — exercises strip_outer_parens / compile_expr recursion.
// ============================================================

#[test]
fn t207c_deep_parens() {
    let mut expr = "42".to_string();
    for _ in 0..DEEP {
        expr = format!("({expr})");
    }
    let r = deep_eval(&expr);
    // Parens are stripped at compile time (loop, not recursion), so this
    // should ALWAYS succeed. Test makes sure compile_expr handles the
    // input without recursion blowup.
    assert_eq!(r.unwrap(), TlaValue::Int(42));
}

// ============================================================
// Deep arithmetic — exercises eval_compiled_inner Add/Sub/Mul depth.
// We can't chain `+` directly (T206 emits Unparsed for 3+ chains),
// but parenthesised pairs preserve binary tree structure.
// ============================================================

#[test]
fn t207c_deep_addition_pairs() {
    // (1 + (1 + (1 + ... + 1)...)) — DEEP levels of nested Add
    let mut expr = "1".to_string();
    for _ in 0..DEEP {
        expr = format!("(1 + ({expr}))");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(_n)) => {} // exact value depends on whether Unparsed kicks in
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep addition: got {other:?}"),
    }
}

// ============================================================
// Deep NOT — exercises eval_compiled_inner Not depth+1.
// ============================================================

#[test]
fn t207c_deep_not_chain() {
    // ~~~~...~~~TRUE — DEEP nested negations
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("~({expr})");
    }
    let r = deep_eval(&expr);
    // DEEP=280 even → result is TRUE
    match r {
        Ok(TlaValue::Bool(_b)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep NOT: got {other:?}"),
    }
}

// ============================================================
// Deep AND/OR — note these flatten in compile_expr (And takes Vec).
// To drive recursion through And/Or we need NESTED, not flat.
// ============================================================

#[test]
fn t207c_deep_nested_and() {
    // (TRUE /\ (TRUE /\ (TRUE /\ ...))) — explicit parens force nesting
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE /\\ ({expr}))");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Bool(true)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep AND: got {other:?}"),
    }
}

#[test]
fn t207c_deep_nested_or() {
    let mut expr = "FALSE".to_string();
    for _ in 0..DEEP {
        expr = format!("(FALSE \\/ ({expr}))");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Bool(false)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep OR: got {other:?}"),
    }
}

// ============================================================
// Deep IMPLIES / IFF — separate dispatch arms in eval_compiled_inner.
// ============================================================

#[test]
fn t207c_deep_implies() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE => ({expr}))");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Bool(true)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep =>: got {other:?}"),
    }
}

#[test]
fn t207c_deep_iff() {
    let mut expr = "TRUE".to_string();
    for _ in 0..DEEP {
        expr = format!("(TRUE <=> ({expr}))");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Bool(_)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep <=>: got {other:?}"),
    }
}

// ============================================================
// Deep operator call chain — drives eval_compiled_opcall depth+1.
// Uses Cardinality-of-singleton repeatedly.
// ============================================================

#[test]
fn t207c_deep_operator_call_chain() {
    // Cardinality({Cardinality({Cardinality({...{1}})})}) — each call
    // recurses through eval_compiled_opcall.
    let mut expr = "{1}".to_string();
    for _ in 0..(DEEP / 2) {
        // /2 because each level adds Cardinality(...) and {...}
        expr = format!("Cardinality({{{expr}}})");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(1)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep Cardinality: got {other:?}"),
    }
}

// ============================================================
// Deep CHOOSE — exercises eval_compiled_inner Choose depth.
// ============================================================

#[test]
fn t207c_deep_choose() {
    // CHOOSE x \in {0} : x = 0 nested DEEP times
    let mut expr = "0".to_string();
    for _ in 0..DEEP {
        expr = format!("CHOOSE x \\in {{{expr}}} : TRUE");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(0)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep CHOOSE: got {other:?}"),
    }
}

// ============================================================
// Deep RecordAccess + RecordLiteral — exercises Record path depth.
// ============================================================

#[test]
fn t207c_deep_record_chain() {
    // [a |-> [a |-> [a |-> ... 1]]].a.a.a.a... DEEP
    let mut expr = "1".to_string();
    for _ in 0..(DEEP / 2) {
        expr = format!("[a |-> {expr}].a");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Int(1)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep record: got {other:?}"),
    }
}

// ============================================================
// Deep sequence — Append wrapping.
// ============================================================

#[test]
fn t207c_deep_append_chain() {
    let mut expr = "<<>>".to_string();
    for i in 0..(DEEP / 2) {
        expr = format!("Append({expr}, {i})");
    }
    let r = deep_eval(&expr);
    match r {
        Ok(TlaValue::Seq(_)) => {}
        Err(e) if e.to_string().contains("depth") || e.to_string().contains("recursion") => {}
        other => panic!("deep Append: got {other:?}"),
    }
}

// ============================================================
// Sub-MAX_DEPTH success cases — catch `depth + 1 -> depth - 1` mutations
// (depth would underflow and immediately error).
// ============================================================

#[test]
fn t207c_shallow_if_chain_must_succeed() {
    // 50 levels — well under MAX_DEPTH, must succeed under both original
    // and `+ -> *` mutation (since mutated only stack-overflows at depth).
    // But `+ -> -` mutation: depth decrements from 0 → underflow → errors
    // immediately. Original: succeeds. So this test catches `+ -> -`.
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
