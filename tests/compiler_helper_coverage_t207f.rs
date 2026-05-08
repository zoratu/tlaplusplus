//! T207 iteration 7: target the user-defined-operator recursive
//! dispatch (line 1909 in compiled_eval.rs's eval_compiled_opcall_user_defined),
//! which currently has missed `depth + 1 -> depth * 1` mutations.
//!
//! Strategy: define a RECURSIVE operator that recurses through itself
//! many times, driving the dispatch path that calls eval_compiled_inner
//! with depth + 1. With the mutation, depth never increments per
//! operator-call level → recursion is unbounded → stack overflow.
//! Without the mutation, depth check fires at MAX_DEPTH=256.

use tlaplusplus::tla::module::parse_tla_module_text;
use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled};

/// Use a recursive operator that decrements N each call. Original code
/// errors at MAX_DEPTH=256 (recursion depth exceeded); mutated code
/// either stack-overflows or produces a different trace.
#[test]
fn t207f_recursive_operator_hits_depth_limit() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
RECURSIVE Count(_)
Count(n) == IF n <= 0 THEN 0 ELSE Count(n - 1) + 1
T == Count(500)
====
"#,
    )
    .unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx);

    // Original: depth check fires after MAX_DEPTH levels of recursion → Err.
    // Mutated `depth + 1 -> depth * 1`: depth stays at outer call level, so
    // recursion is bounded only by the structural depth (Count(500) → 500
    // levels of recursion). On 2MB stack, 500 deep with each frame ~5KB
    // would stack-overflow → process death → cargo-mutants catches.
    // Either Ok or Err with depth-related message indicates working tracking.
    match r {
        Err(e) if e.to_string().contains("depth")
            || e.to_string().contains("recursion") =>
        {
            // Original: depth check fires gracefully.
        }
        Ok(_) => panic!(
            "Count(500) returned Ok — depth check never fired, indicates \
             depth + 1 mutation kept depth at 0 (or original returned a value \
             without hitting MAX_DEPTH = unexpected for n=500)"
        ),
        Err(e) => panic!("Count(500) errored with non-depth message: {e}"),
    }
}

/// A shallower recursive call to ensure original SUCCEEDS.
/// Catches `depth + 1 -> depth - 1` mutations that would underflow on
/// the first call.
#[test]
fn t207f_shallow_recursive_operator_must_succeed() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
RECURSIVE Count(_)
Count(n) == IF n <= 0 THEN 0 ELSE Count(n - 1) + 1
T == Count(20)
====
"#,
    )
    .unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();

    // Count(20) should return 20.
    assert_eq!(r, TlaValue::Int(20));
}

/// Recursive Fibonacci — exercises the same path with branching recursion.
#[test]
fn t207f_recursive_fibonacci_small() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
RECURSIVE Fib(_)
Fib(n) == IF n <= 1 THEN n ELSE Fib(n - 1) + Fib(n - 2)
T == Fib(10)
====
"#,
    )
    .unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();

    // Fib(10) = 55.
    assert_eq!(r, TlaValue::Int(55));
}

/// Mutual recursion — A calls B, B calls A. Drives the user-defined op
/// dispatch through TWO operators.
#[test]
fn t207f_mutual_recursion_small() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
RECURSIVE IsEven(_), IsOdd(_)
IsEven(n) == IF n = 0 THEN TRUE ELSE IsOdd(n - 1)
IsOdd(n) == IF n = 0 THEN FALSE ELSE IsEven(n - 1)
T1 == IsEven(10)
T2 == IsOdd(7)
====
"#,
    )
    .unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let t1 = &module.definitions.get("T1").unwrap().body;
    let t2 = &module.definitions.get("T2").unwrap().body;

    assert_eq!(
        eval_compiled(&compile_expr(t1), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr(t2), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
}

/// User-defined non-recursive operator chain — N operators each calling
/// the next. Drives line 1909 through N levels.
#[test]
fn t207f_operator_chain_must_succeed() {
    // Chain f1 → f2 → ... → f10 → terminal value.
    let mut defs = String::new();
    for i in 1..=10 {
        if i == 10 {
            defs.push_str(&format!("f{i}(x) == x + 1\n"));
        } else {
            defs.push_str(&format!("f{i}(x) == f{}(x)\n", i + 1));
        }
    }

    let module_text = format!(
        "---- MODULE M ----\n{defs}T == f1(99)\n====\n"
    );
    let module = parse_tla_module_text(&module_text).unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();

    assert_eq!(r, TlaValue::Int(100));
}

/// Deep operator chain — drive recursion past MAX_DEPTH.
/// Catches `depth + 1 -> depth * 1` at the user-defined op site.
#[test]
fn t207f_deep_operator_chain_hits_depth_limit() {
    // Chain f1 → f2 → ... → f300 → terminal.
    // Each call enters eval_compiled_opcall_user_defined at line 1909.
    let mut defs = String::new();
    let n = 300;
    for i in 1..=n {
        if i == n {
            defs.push_str(&format!("f{i}(x) == x\n"));
        } else {
            defs.push_str(&format!("f{i}(x) == f{}(x)\n", i + 1));
        }
    }

    let module_text = format!(
        "---- MODULE M ----\n{defs}T == f1(0)\n====\n"
    );
    let module = parse_tla_module_text(&module_text).unwrap();

    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx);

    // Must error gracefully (MAX_DEPTH exceeded), or stack-overflow on
    // the mutated version. Either way: no Ok(_) without hitting limits.
    match r {
        Err(e) if e.to_string().contains("depth")
            || e.to_string().contains("recursion") => {}
        Ok(v) => panic!(
            "deep operator chain (300 deep): expected Err but got Ok({v:?}). \
             Mutation `depth + 1 -> depth * 1` lets recursion proceed without \
             tracking, so finite chain finishes successfully."
        ),
        Err(e) => panic!("expected depth Err, got: {e}"),
    }
}
