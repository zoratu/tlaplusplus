//! T207 iteration 6: deep-recursion stress tests through every built-in
//! that has a `depth + 1` call site, plus targeted shape coverage for
//! membership and arity dispatch paths still missing direct tests.
//!
//! After iter5, ~161 `+ -> *` mutations on `depth + 1` sites remain
//! missed. Most are in eval_compiled_opcall built-in dispatch arms that
//! my t207c didn't drive. This file adds one deep-recursion path per
//! remaining built-in, plus shape coverage for compiled_membership_contains.

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
// Deep recursion through every built-in operator path that has
// `depth + 1` in eval_compiled_opcall dispatch.
// ============================================================

#[test]
fn t207e_deep_cardinality_chain() {
    let mut expr = "{1}".to_string();
    for _ in 0..(DEEP / 2) {
        expr = format!("{{Cardinality({expr})}}");
    }
    assert_recursion_err(&format!("Cardinality({expr})"), "deep Cardinality");
}

// Len/Append doubling test removed — same exponential explosion.

// Head/Tail deep test removed — `Append(Tail(s), Head(s))` doubles
// expression size per level, OOMs the test process.

// SubSeq deep test removed — duplicates expr in `Len({expr})`.

#[test]
fn t207e_deep_domain_chain() {
    // [a |-> [a |-> ... ]]; DOMAIN at each level recurses through
    // record dispatch.
    let mut expr = "1".to_string();
    for _ in 0..(DEEP / 3) {
        expr = format!("Cardinality(DOMAIN [a |-> {expr}])");
    }
    assert_recursion_err(&expr, "deep DOMAIN");
}

#[test]
fn t207e_deep_range_chain() {
    let mut expr = "[a |-> 1]".to_string();
    for _ in 0..(DEEP / 3) {
        expr = format!("[a |-> Cardinality(Range({expr}))]");
    }
    assert_recursion_err(&expr, "deep Range");
}

// Note: SUBSET / UNION / quantifier / set-comprehension deep tests
// removed — they OOM on the original code (SUBSET grows 2^N, set
// comprehension allocates intermediate sets, quantifiers iterate
// per-recursion). Coverage of the recursive-call sites in those
// dispatch arms is left to the parity / shape tests below.

#[test]
fn t207e_deep_func_constructor_chain() {
    // [x \in S |-> body] — body recurses
    let mut expr = "1".to_string();
    for _ in 0..(DEEP / 3) {
        expr = format!("[x \\in {{1}} |-> ({expr})][1]");
    }
    assert_recursion_err(&expr, "deep function constructor");
}

#[test]
fn t207e_deep_func_apply_chain() {
    // f[x] — chain function applications
    let mut expr = "[i \\in 1..3 |-> i + 1]".to_string();
    for _ in 0..(DEEP / 3) {
        expr = format!("[i \\in 1..3 |-> Cardinality(DOMAIN ({expr}))]");
    }
    assert_recursion_err(&format!("({expr})[1]"), "deep func apply");
}

#[test]
fn t207e_deep_record_access_chain() {
    let mut expr = "1".to_string();
    for _ in 0..(DEEP / 2) {
        expr = format!("[a |-> ({expr})].a");
    }
    assert_recursion_err(&expr, "deep record access");
}

// Record EXCEPT deep test removed — substitution duplicates expr.

#[test]
fn t207e_deep_lambda_application_chain() {
    // LAMBDA application — recurses through user-defined operator path
    let mut expr = "1".to_string();
    for i in 0..(DEEP / 2) {
        expr = format!("LET f{i}(x) == x IN f{i}({expr})");
    }
    assert_recursion_err(&expr, "deep LAMBDA app");
}

// ToString deep test removed — accumulates string length per level.

// ============================================================
// Membership exhaustive shapes — compiled_membership_contains paths.
// ============================================================

#[test]
fn t207e_membership_int_in_int_set() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("0 \\in {0}"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("0 \\in {1}"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207e_membership_negative_int() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("-3 \\in {-5, -3, -1}"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-2 \\in {-5, -3, -1}"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207e_membership_in_negative_range() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("-5 \\in -5..-3"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-3 \\in -5..-3"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-2 \\in -5..-3"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-6 \\in -5..-3"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207e_membership_zero_in_range_with_zero() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("0 \\in 0..0"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("0 \\in -1..1"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
}

#[test]
fn t207e_membership_in_empty_range() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // 5..3 is empty
    assert_eq!(
        eval_compiled(&compile_expr("4 \\in 5..3"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207e_membership_string_set_exhaustive() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // String set membership
    let exprs = [
        ("\"a\" \\in {\"a\", \"b\", \"c\"}", true),
        ("\"d\" \\in {\"a\", \"b\", \"c\"}", false),
        ("\"a\" \\in {\"a\"}", true),
        ("\"\" \\in {\"\"}", true),
        ("\"\" \\in {\"a\"}", false),
    ];
    for (expr, expected) in exprs {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(r, TlaValue::Bool(expected), "for {expr}");
    }
}

#[test]
fn t207e_membership_record_set_exhaustive() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // [a: S, b: T] membership
    let exprs = [
        ("[a |-> 1, b |-> 1] \\in [a: {1}, b: {1, 2}]", true),
        ("[a |-> 1, b |-> 3] \\in [a: {1}, b: {1, 2}]", false),
        ("[a |-> 9, b |-> 1] \\in [a: {1}, b: {1, 2}]", false),
    ];
    for (expr, expected) in exprs {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(r, TlaValue::Bool(expected), "for {expr}");
    }
}

#[test]
fn t207e_membership_in_nat() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // Nat is the set of natural numbers
    assert_eq!(
        eval_compiled(&compile_expr("5 \\in Nat"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("0 \\in Nat"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-1 \\in Nat"), &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207e_membership_in_int() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("5 \\in Int"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("-5 \\in Int"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
}

#[test]
fn t207e_membership_in_boolean() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    assert_eq!(
        eval_compiled(&compile_expr("TRUE \\in BOOLEAN"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_compiled(&compile_expr("FALSE \\in BOOLEAN"), &ctx).unwrap(),
        TlaValue::Bool(true)
    );
}

#[test]
fn t207e_notin_negation_exhaustive() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let exprs = [
        ("1 \\notin {2, 3}", true),
        ("1 \\notin {1, 2, 3}", false),
        ("\"a\" \\notin {\"b\"}", true),
        ("\"a\" \\notin {\"a\"}", false),
    ];
    for (expr, expected) in exprs {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(r, TlaValue::Bool(expected), "for {expr}");
    }
}

// ============================================================
// Built-in arity exhaustive — covering remaining match arms.
// Each test exercises the `arg_values.len() == N && !user_defined_shadow`
// guard. Calls with wrong arity → guard fails, falls through.
// ============================================================

#[test]
fn t207e_isfinite_set_arity_via_text_eval() {
    // IsFiniteSet may not be a built-in; this exercises the fall-through.
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // Whatever the result is, both interp and compiled should agree.
    let i = tlaplusplus::tla::eval_expr("IsFiniteSet({1, 2, 3})", &ctx);
    let c = eval_compiled(&compile_expr("IsFiniteSet({1, 2, 3})"), &ctx);
    match (i, c) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("IsFiniteSet shape diverge: {a:?} vs {b:?}"),
    }
}

#[test]
fn t207e_select_seq_arity() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // SelectSeq with right arity (2) and lambda predicate
    let r = eval_compiled(
        &compile_expr("SelectSeq(<<1, 2, 3, 4, 5>>, LAMBDA x: x > 2)"),
        &ctx,
    )
    .unwrap();
    if let TlaValue::Seq(seq) = r {
        assert_eq!(seq.len(), 3);
    } else {
        panic!("expected Seq, got {r:?}");
    }
}

#[test]
fn t207e_funasseq_negative_b_errors() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    // FunAsSeq(f, a, b) requires b >= 0
    let r = eval_compiled(
        &compile_expr("FunAsSeq([i \\in 1..3 |-> i], 1, -1)"),
        &ctx,
    );
    assert!(r.is_err(), "FunAsSeq with b<0 should error, got {r:?}");
}

#[test]
fn t207e_funasseq_zero_b_returns_empty() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let r = eval_compiled(
        &compile_expr("FunAsSeq([i \\in 1..3 |-> i], 1, 0)"),
        &ctx,
    )
    .unwrap();
    assert_eq!(r, TlaValue::Seq(std::sync::Arc::new(vec![])));
}

#[test]
fn t207e_subseq_n_zero_with_m_one() {
    // SubSeq(s, 1, 0) — m=1 (valid), n=0 → end clamped to 0 → empty.
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let r = eval_compiled(&compile_expr("SubSeq(<<1, 2, 3>>, 1, 0)"), &ctx).unwrap();
    assert_eq!(r, TlaValue::Seq(std::sync::Arc::new(vec![])));
}

#[test]
fn t207e_subseq_full_seq() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let r = eval_compiled(&compile_expr("SubSeq(<<1, 2, 3>>, 1, 3)"), &ctx).unwrap();
    if let TlaValue::Seq(seq) = r {
        assert_eq!(seq.len(), 3);
    } else {
        panic!("expected Seq");
    }
}

// ============================================================
// String operations — Asc / Chr / IF/THEN-IN-string contexts.
// ============================================================

#[test]
fn t207e_concat_strings_via_concatenation() {
    // String concatenation `\o` if supported
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let i = tlaplusplus::tla::eval_expr("<<1, 2>> \\o <<3, 4>>", &ctx);
    let c = eval_compiled(&compile_expr("<<1, 2>> \\o <<3, 4>>"), &ctx);
    match (i, c) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("\\o shape diverge: {a:?} vs {b:?}"),
    }
}

// ============================================================
// Exhaustive comparison operator exercising — split_first_top_level_op
// missed mutations on every relop dispatch.
// ============================================================

#[test]
fn t207e_relops_exhaustive_int() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let cases = [
        ("3 = 3", true),
        ("3 = 4", false),
        ("3 # 4", true),
        ("3 # 3", false),
        ("3 < 4", true),
        ("3 < 3", false),
        ("3 > 2", true),
        ("3 > 3", false),
        ("3 <= 3", true),
        ("3 <= 4", true),
        ("3 <= 2", false),
        ("3 >= 3", true),
        ("3 >= 2", true),
        ("3 >= 4", false),
        ("3 /= 4", true),
        ("3 /= 3", false),
    ];
    for (expr, expected) in cases {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(r, TlaValue::Bool(expected), "for {expr}");
    }
}

// ============================================================
// Set algebra exhaustive — exercises every set-op dispatch.
// ============================================================

#[test]
fn t207e_set_algebra_exhaustive() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let exprs = [
        ("Cardinality({1, 2, 3} \\cup {3, 4})", TlaValue::Int(4)),
        ("Cardinality({1, 2, 3} \\cap {2, 3, 4})", TlaValue::Int(2)),
        ("Cardinality({1, 2, 3} \\ {2})", TlaValue::Int(2)),
        ("Cardinality(SUBSET {1, 2})", TlaValue::Int(4)),
        ("Cardinality(UNION {{1}, {2, 3}})", TlaValue::Int(3)),
    ];
    for (expr, expected) in exprs {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(r, expected, "for {expr}");
    }
}

// ============================================================
// Sequence operations exhaustive.
// ============================================================

#[test]
fn t207e_sequence_ops_exhaustive() {
    let s = TlaState::new();
    let ctx = EvalContext::new(&s);
    let cases: &[(&str, TlaValue)] = &[
        ("Len(<<1, 2, 3>>)", TlaValue::Int(3)),
        ("Len(<<>>)", TlaValue::Int(0)),
        ("Head(<<1, 2, 3>>)", TlaValue::Int(1)),
        ("Len(Tail(<<1, 2, 3>>))", TlaValue::Int(2)),
        ("Len(Append(<<1, 2>>, 3))", TlaValue::Int(3)),
        (
            "Head(Append(<<1, 2>>, 3))",
            TlaValue::Int(1),
        ),
        ("Len(SubSeq(<<1, 2, 3, 4>>, 2, 3))", TlaValue::Int(2)),
    ];
    for (expr, expected) in cases {
        let r = eval_compiled(&compile_expr(expr), &ctx).unwrap();
        assert_eq!(&r, expected, "for {expr}");
    }
}
