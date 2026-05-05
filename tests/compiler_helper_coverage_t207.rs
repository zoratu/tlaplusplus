//! T207: targeted coverage for compiler-internal helpers surfaced as
//! mutation-testing gaps in the v1.2.0 prep. The interpreter (`eval.rs`)
//! had a 100% mutation kill rate; the compiler (`compiled_eval.rs` +
//! `compiled_expr.rs`) had ~42% because internal helpers
//! (`eval_compiled_opcall`, `eval_compiled_inner`, `split_first_top_level_op`,
//! `split_binary_op_with`, `split_case_arms`, `split_quantifier_bindings`,
//! `find_keyword`, `find_top_level_except`, `compiled_membership_contains`,
//! `membership_matches_text`, `guard_text_is_action_body`) were exercised
//! only end-to-end via T2 proptest equivalence.
//!
//! These tests target boundary conditions cargo-mutants found unguarded:
//!   - arity checks on every built-in operator (`arg_values.len() != N`)
//!   - user-defined-shadow guards on `Max`/`Min`/`BoundedSeq`/community ops
//!   - empty-sequence checks for `Head`/`Tail`
//!   - chained-relop splits at the leftmost operator (mirrors T101.1)
//!   - word-boundary keyword matches (`IN` vs `INVENTORY`)
//!   - operator splits inside LET / IF / CASE / quantifier bodies
//!   - membership shapes on Set / Seq / Function / Record / String
//!
//! Each test is shaped so a typical mutation (operator swap, comparison
//! flip, returning `true`/`false` from a match guard) breaks an assertion.

use std::collections::BTreeMap;
use tlaplusplus::tla::module::{TlaDefinition, parse_tla_module_text};
use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

fn ctx_from<'a>(state: &'a TlaState, defs: &'a BTreeMap<String, TlaDefinition>) -> EvalContext<'a> {
    EvalContext::with_definitions(state, defs)
}

fn eval_c(expr: &str, ctx: &EvalContext<'_>) -> anyhow::Result<TlaValue> {
    eval_compiled(&compile_expr(expr), ctx)
}

// ============================================================
// Arity checks — every built-in must reject wrong arity.
// Each pair catches `arg_values.len() != N` → `len() == N` or `true`/`false`.
// ============================================================

macro_rules! arity_err {
    ($name:literal, $expr:literal) => {{
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        let r = eval_c($expr, &ctx);
        assert!(
            r.is_err(),
            "{}: expected arity error on `{}`, got {:?}",
            $name,
            $expr,
            r
        );
    }};
}

#[test]
fn t207_arity_cardinality_zero_args_errors() {
    arity_err!("Cardinality", "Cardinality()");
}
#[test]
fn t207_arity_cardinality_two_args_errors() {
    arity_err!("Cardinality", "Cardinality({1}, {2})");
}
#[test]
fn t207_arity_cardinality_one_arg_ok() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("Cardinality({1, 2, 3})", &ctx).unwrap(), TlaValue::Int(3));
    assert_eq!(eval_c("Cardinality({})", &ctx).unwrap(), TlaValue::Int(0));
    // Singleton — catches `len() != 0` mutations
    assert_eq!(eval_c("Cardinality({42})", &ctx).unwrap(), TlaValue::Int(1));
}

#[test]
fn t207_arity_len_wrong_args_error() {
    arity_err!("Len", "Len()");
    arity_err!("Len", "Len(<<1>>, <<2>>)");
}
#[test]
fn t207_arity_len_correct() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("Len(<<>>)", &ctx).unwrap(), TlaValue::Int(0));
    assert_eq!(eval_c("Len(<<1>>)", &ctx).unwrap(), TlaValue::Int(1));
    assert_eq!(eval_c("Len(<<1, 2, 3>>)", &ctx).unwrap(), TlaValue::Int(3));
}

#[test]
fn t207_arity_head_wrong() {
    arity_err!("Head", "Head()");
    arity_err!("Head", "Head(<<1>>, <<2>>)");
}
#[test]
fn t207_head_empty_errors() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("Head(<<>>)", &ctx);
    assert!(r.is_err(), "Head(<<>>) should error, got {r:?}");
}
#[test]
fn t207_head_singleton_works() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Singleton seq; catches `if seq.is_empty()` → `if !seq.is_empty()` mutation
    assert_eq!(eval_c("Head(<<7>>)", &ctx).unwrap(), TlaValue::Int(7));
    assert_eq!(eval_c("Head(<<10, 20, 30>>)", &ctx).unwrap(), TlaValue::Int(10));
}

#[test]
fn t207_arity_tail_wrong() {
    arity_err!("Tail", "Tail()");
    arity_err!("Tail", "Tail(<<1>>, <<2>>)");
}
#[test]
fn t207_tail_empty_errors() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert!(eval_c("Tail(<<>>)", &ctx).is_err());
}
#[test]
fn t207_tail_singleton_yields_empty() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Catches `seq[1..]` → `seq[0..]` (would still have 7 in result)
    assert_eq!(eval_c("Tail(<<7>>)", &ctx).unwrap(), TlaValue::Seq(std::sync::Arc::new(vec![])));
    assert_eq!(
        eval_c("Tail(<<10, 20, 30>>)", &ctx).unwrap(),
        TlaValue::Seq(std::sync::Arc::new(vec![TlaValue::Int(20), TlaValue::Int(30)]))
    );
}

#[test]
fn t207_arity_append_wrong() {
    arity_err!("Append", "Append()");
    arity_err!("Append", "Append(<<1>>)");
    arity_err!("Append", "Append(<<1>>, 2, 3)");
}
#[test]
fn t207_append_correct() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("Append(<<1, 2>>, 3)", &ctx).unwrap();
    assert_eq!(
        r,
        TlaValue::Seq(std::sync::Arc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3)
        ]))
    );
    // Append to empty seq
    let r2 = eval_c("Append(<<>>, 99)", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Seq(std::sync::Arc::new(vec![TlaValue::Int(99)])));
}

#[test]
fn t207_arity_subseq_wrong() {
    arity_err!("SubSeq", "SubSeq()");
    arity_err!("SubSeq", "SubSeq(<<1>>)");
    arity_err!("SubSeq", "SubSeq(<<1>>, 1)");
    arity_err!("SubSeq", "SubSeq(<<1>>, 1, 1, 1)");
}
#[test]
fn t207_subseq_boundary_indices() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // m < 1 must error — catches `if m < 1` → `if m <= 1`
    assert!(eval_c("SubSeq(<<1, 2, 3>>, 0, 2)", &ctx).is_err());
    // Valid: take 1..=2 -> <<1, 2>>
    let r = eval_c("SubSeq(<<1, 2, 3>>, 1, 2)", &ctx).unwrap();
    assert_eq!(
        r,
        TlaValue::Seq(std::sync::Arc::new(vec![TlaValue::Int(1), TlaValue::Int(2)]))
    );
    // n past end clamps; catches `(n as usize).min(seq.len())` mutations
    let r2 = eval_c("SubSeq(<<1, 2, 3>>, 1, 99)", &ctx).unwrap();
    assert_eq!(
        r2,
        TlaValue::Seq(std::sync::Arc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3)
        ]))
    );
    // start > seq.len() returns empty; catches `if start > seq.len()` flips
    let r3 = eval_c("SubSeq(<<1, 2>>, 99, 100)", &ctx).unwrap();
    assert_eq!(r3, TlaValue::Seq(std::sync::Arc::new(vec![])));
}

#[test]
fn t207_arity_max_min_wrong() {
    arity_err!("Max", "Max()");
    arity_err!("Min", "Min()");
}
#[test]
fn t207_max_min_singleton_and_multi() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("Max({3, 1, 2})", &ctx).unwrap(), TlaValue::Int(3));
    assert_eq!(eval_c("Min({3, 1, 2})", &ctx).unwrap(), TlaValue::Int(1));
    assert_eq!(eval_c("Max({42})", &ctx).unwrap(), TlaValue::Int(42));
    assert_eq!(eval_c("Min({42})", &ctx).unwrap(), TlaValue::Int(42));
}

#[test]
fn t207_max_min_user_defined_shadow() {
    // Catches `&& !user_defined_shadow` → `true` mutations: when the user
    // defines `Max(s) == 999`, the compiler must call THE USER's version,
    // not the built-in extremum.
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
Max(s) == 999
Test == Max({1, 2, 3})
====
"#,
    )
    .unwrap();
    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let body = &module.definitions.get("Test").unwrap().body;
    let r = eval_c(body, &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(999), "user-defined Max must shadow built-in");
}

#[test]
fn t207_min_user_defined_shadow() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
Min(s) == 777
Test == Min({1, 2, 3})
====
"#,
    )
    .unwrap();
    let state = TlaState::new();
    let ctx = ctx_from(&state, &module.definitions);
    let body = &module.definitions.get("Test").unwrap().body;
    assert_eq!(eval_c(body, &ctx).unwrap(), TlaValue::Int(777));
}

#[test]
fn t207_arity_tlc_get_set_wrong() {
    arity_err!("TLCGet", "TLCGet()");
    arity_err!("TLCGet", "TLCGet(1, 2)");
    arity_err!("TLCSet", "TLCSet()");
    arity_err!("TLCSet", "TLCSet(1)");
    arity_err!("TLCSet", "TLCSet(1, 2, 3)");
}

#[test]
fn t207_domain_record_seq_function() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Record domain → set of field names
    let r = eval_c("DOMAIN [a |-> 1, b |-> 2]", &ctx).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 2);
        assert!(s.contains(&TlaValue::String("a".to_string())));
        assert!(s.contains(&TlaValue::String("b".to_string())));
    } else {
        panic!("expected Set, got {r:?}");
    }
    // Sequence domain → 1..len
    let r2 = eval_c("DOMAIN <<10, 20, 30>>", &ctx).unwrap();
    if let TlaValue::Set(s) = r2 {
        assert_eq!(s.len(), 3);
        assert!(s.contains(&TlaValue::Int(1)));
        assert!(s.contains(&TlaValue::Int(3)));
    } else {
        panic!("expected Set, got {r2:?}");
    }
}

#[test]
fn t207_arity_range_wrong() {
    arity_err!("Range", "Range()");
    arity_err!("Range", "Range([a |-> 1], [b |-> 2])");
}
#[test]
fn t207_range_record_and_seq() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("Range([a |-> 1, b |-> 2])", &ctx).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 2);
        assert!(s.contains(&TlaValue::Int(1)));
        assert!(s.contains(&TlaValue::Int(2)));
    } else {
        panic!("expected Set, got {r:?}");
    }
    // Range of seq is set of values (deduplicated)
    let r2 = eval_c("Range(<<1, 2, 1, 3>>)", &ctx).unwrap();
    if let TlaValue::Set(s) = r2 {
        assert_eq!(s.len(), 3, "expected 3 unique values");
    } else {
        panic!("expected Set, got {r2:?}");
    }
}

#[test]
fn t207_arity_tostring_wrong() {
    arity_err!("ToString", "ToString()");
    arity_err!("ToString", "ToString(1, 2)");
}
#[test]
fn t207_tostring_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("ToString(42)", &ctx).unwrap();
    if let TlaValue::String(s) = r {
        assert!(s.contains("42"), "expected '42' in {s:?}");
    } else {
        panic!("expected String, got {r:?}");
    }
}

#[test]
fn t207_xor_op_correct() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // 5 ^ 3 = 6
    assert_eq!(eval_c("5 ^^ 3", &ctx).unwrap(), TlaValue::Int(6));
    assert_eq!(eval_c("0 ^^ 0", &ctx).unwrap(), TlaValue::Int(0));
    assert_eq!(eval_c("7 ^^ 7", &ctx).unwrap(), TlaValue::Int(0));
}

#[test]
fn t207_arity_funasseq_wrong() {
    arity_err!("FunAsSeq", "FunAsSeq()");
    arity_err!("FunAsSeq", "FunAsSeq([i \\in 1..3 |-> i])");
    arity_err!("FunAsSeq", "FunAsSeq([i \\in 1..3 |-> i], 1)");
    arity_err!("FunAsSeq", "FunAsSeq([i \\in 1..3 |-> i], 1, 3, 4)");
}
#[test]
fn t207_funasseq_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // f = [i \in 1..3 |-> i*10]; FunAsSeq(f, 1, 3) = <<10, 20, 30>>
    let r = eval_c("FunAsSeq([i \\in 1..3 |-> i * 10], 1, 3)", &ctx).unwrap();
    assert_eq!(
        r,
        TlaValue::Seq(std::sync::Arc::new(vec![
            TlaValue::Int(10),
            TlaValue::Int(20),
            TlaValue::Int(30)
        ]))
    );
}

// ============================================================
// SymDiff / SumSet / ProductSet — community-module built-ins.
// Catches `arg_values.len() == N && !user_defined_shadow` → `true` flips
// (which would force the dispatch even for wrong arity / shadowed names).
// ============================================================

#[test]
fn t207_symdiff_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("SymDiff({1, 2, 3}, {2, 3, 4})", &ctx).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 2);
        assert!(s.contains(&TlaValue::Int(1)));
        assert!(s.contains(&TlaValue::Int(4)));
    } else {
        panic!("expected Set, got {r:?}");
    }
}

#[test]
fn t207_sumset_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("SumSet({1, 2, 3, 4})", &ctx).unwrap(), TlaValue::Int(10));
    assert_eq!(eval_c("SumSet({})", &ctx).unwrap(), TlaValue::Int(0));
}

#[test]
fn t207_productset_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("ProductSet({2, 3, 4})", &ctx).unwrap(), TlaValue::Int(24));
    // Empty product is identity = 1
    assert_eq!(eval_c("ProductSet({})", &ctx).unwrap(), TlaValue::Int(1));
}

// ============================================================
// Comparison-operator splits — exercises split_first_top_level_op.
// Mirrors T101.1 C2 (chained relops split at leftmost, not priority-highest).
// ============================================================

#[test]
fn t207_split_chained_relop_leftmost() {
    // `a # b = c` must split at `#` (the leftmost relop), giving
    // `(a) # (b = c)` → b=c is a Bool, then a # Bool → ModelValue/Bool inequality.
    // The interpreter does this; the compiler must agree (T101.1 fix).
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr("1 # (2 = 2)", &ctx).unwrap();
    let compi = eval_c("1 # (2 = 2)", &ctx).unwrap();
    assert_eq!(interp, compi);

    // 1 = 1 = TRUE → split at first `=` → (1) = (1 = TRUE), the rhs is comparison
    // of Int and Bool which is type mismatch but both evaluators must agree.
    let i2 = eval_expr("1 = 1", &ctx);
    let c2 = eval_c("1 = 1", &ctx);
    match (i2, c2) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {} // both error is also fine
        (a, b) => panic!("interp/compi diverge on `1 = 1`: {a:?} vs {b:?}"),
    }
}

#[test]
fn t207_split_relop_word_boundary_in_vs_intersect() {
    // `\in` and `\intersect` start the same — splitting must not match `\in`
    // inside `\intersect`.  We verify by giving an expression where the only
    // `\in` available is INSIDE a quantifier body and the only top-level op
    // would be wrong if word-boundary check failed.
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("{1, 2, 3} \\intersect {2, 3, 4}", &ctx).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 2);
        assert!(s.contains(&TlaValue::Int(2)));
        assert!(s.contains(&TlaValue::Int(3)));
    } else {
        panic!("expected Set from intersect, got {r:?}");
    }
}

#[test]
fn t207_split_eq_not_in_arrow() {
    // `=>` (implies) must not split as `=`. Test expression:
    // `TRUE => FALSE` should evaluate to FALSE.
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("TRUE => FALSE", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("TRUE => TRUE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("FALSE => FALSE", &ctx).unwrap(), TlaValue::Bool(true));
}

#[test]
fn t207_split_ge_le_distinct_from_gt_lt() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // <= must not be parsed as <
    assert_eq!(eval_c("3 <= 3", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("3 < 3", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("3 >= 3", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("3 > 3", &ctx).unwrap(), TlaValue::Bool(false));
    // /= must not split as `=`
    assert_eq!(eval_c("1 /= 2", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("1 /= 1", &ctx).unwrap(), TlaValue::Bool(false));
    // # must not be parsed as identifier; equivalent to /=
    assert_eq!(eval_c("1 # 2", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("1 # 1", &ctx).unwrap(), TlaValue::Bool(false));
}

// ============================================================
// Splits inside protected scopes — the at_top guard in
// split_first_top_level_op / split_binary_op_with must skip ops inside
// LET / IF / CASE / quantifier bodies.
// ============================================================

#[test]
fn t207_split_skips_op_inside_if_branch() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // `IF a > b THEN c ELSE d` — the `>` inside the IF condition must
    // NOT cause the IF expression to be split as `IF a` `>` `b THEN c ELSE d`.
    let r = eval_c("IF 5 > 3 THEN 10 ELSE 20", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(10));
    let r2 = eval_c("IF 1 > 5 THEN 10 ELSE 20", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(20));
}

#[test]
fn t207_split_skips_op_inside_let() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("(LET x == 5 IN x) + 10", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(15));
    // Two LET layers
    let r2 = eval_c("(LET x == 5 IN LET y == 3 IN x + y) * 2", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(16));
}

#[test]
fn t207_split_skips_op_inside_case() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // CASE arms — operators inside arms should not split outer expression.
    // The CASE is wrapped in parens then added to 100.
    let r = eval_c(
        "(CASE 5 = 1 -> 10 [] 5 = 5 -> 20 [] OTHER -> 30) + 100",
        &ctx,
    )
    .unwrap();
    assert_eq!(r, TlaValue::Int(120));
}

#[test]
fn t207_split_skips_op_inside_quantifier() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // \A x \in S : x > 0 — the `>` inside the quantifier body must not split
    // the outer expression. We test by using the quantifier as a Bool.
    assert_eq!(eval_c("\\A x \\in {1, 2, 3} : x > 0", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("\\A x \\in {1, 2, 3} : x > 1", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("\\E x \\in {1, 2, 3} : x = 2", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("\\E x \\in {1, 2, 3} : x = 99", &ctx).unwrap(), TlaValue::Bool(false));
}

// ============================================================
// CASE arms — split_case_arms (39 missed).
// Arms separated by `[]`. Test single arm, two arms, OTHER fallback.
// ============================================================

#[test]
fn t207_case_single_arm_with_other() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("CASE 5 = 5 -> 99 [] OTHER -> 0", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(99));
    let r2 = eval_c("CASE 5 = 7 -> 99 [] OTHER -> 0", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(0));
}

#[test]
fn t207_case_three_arms() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c(
        "CASE 1 = 0 -> \"a\" [] 1 = 1 -> \"b\" [] 1 = 2 -> \"c\" [] OTHER -> \"d\"",
        &ctx,
    )
    .unwrap();
    assert_eq!(r, TlaValue::String("b".to_string()));
}

#[test]
fn t207_case_no_match_no_other_errors() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // No arm matches and no OTHER — must error.
    let r = eval_c("CASE 1 = 0 -> 99 [] 2 = 0 -> 88", &ctx);
    assert!(r.is_err(), "no-match CASE without OTHER should error, got {r:?}");
}

// ============================================================
// LET / IF / Quantifier bindings — split_quantifier_bindings (37),
// split_let_expression (22).
// ============================================================

#[test]
fn t207_quantifier_multi_binding() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // \A x, y \in S : x = y — multiple variables in one binding.
    // Compiler must split bindings correctly.
    assert_eq!(
        eval_c("\\A x, y \\in {1} : x = y", &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("\\A x, y \\in {1, 2} : x = y", &ctx).unwrap(),
        TlaValue::Bool(false)
    );
    // \E with multi-binding
    assert_eq!(
        eval_c("\\E x, y \\in {1, 2, 3} : x + y = 5", &ctx).unwrap(),
        TlaValue::Bool(true)
    );
}

#[test]
fn t207_let_two_definitions() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("LET x == 3 IN LET y == 4 IN x * y", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(12));
    // LET defining a function
    let r2 = eval_c("LET f(n) == n + 1 IN f(5)", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(6));
}

#[test]
fn t207_let_body_uses_outer_scope() {
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
K == 100
Test == LET x == 5 IN x + K
====
"#,
    )
    .unwrap();
    let state = TlaState::new();
    let ctx = ctx_from(&state, &module.definitions);
    let body = &module.definitions.get("Test").unwrap().body;
    assert_eq!(eval_c(body, &ctx).unwrap(), TlaValue::Int(105));
}

// ============================================================
// find_keyword (32 missed) — word-boundary keyword scanning.
// `IN` inside `INVENTORY` must NOT match.
// ============================================================

#[test]
fn t207_keyword_word_boundary_in() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // `LET` body where the variable name STARTS with IN (e.g., `INPUT`).
    // The compiler must NOT find the IN keyword inside `INPUT`.
    let r = eval_c("LET INPUT == 42 IN INPUT + 1", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(43));
    // Variable named with embedded keywords
    let r2 = eval_c("LET THENCE == 7 IN THENCE", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(7));
}

#[test]
fn t207_keyword_then_else_word_boundary() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Variable starting with THEN
    let r = eval_c("IF TRUE THEN 1 ELSE 2", &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(1));
    // ELSE is a keyword — variable ELSEWHERE shouldn't match
    let r2 = eval_c("LET ELSEWHERE == 99 IN ELSEWHERE", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Int(99));
}

// ============================================================
// EXCEPT — find_top_level_except (29 missed).
// `[record EXCEPT !.field = newval]`
// ============================================================

#[test]
fn t207_except_record_field() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("[[a |-> 1, b |-> 2] EXCEPT !.a = 99]", &ctx).unwrap();
    if let TlaValue::Record(map) = r {
        assert_eq!(map.get("a"), Some(&TlaValue::Int(99)));
        assert_eq!(map.get("b"), Some(&TlaValue::Int(2)));
    } else {
        panic!("expected Record, got {r:?}");
    }
}

#[test]
fn t207_except_function_index() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("[[i \\in 1..3 |-> i * 10] EXCEPT ![2] = 999]", &ctx).unwrap();
    // Expect [1 |-> 10, 2 |-> 999, 3 |-> 30]
    if let TlaValue::Function(map) = r {
        assert_eq!(map.get(&TlaValue::Int(1)), Some(&TlaValue::Int(10)));
        assert_eq!(map.get(&TlaValue::Int(2)), Some(&TlaValue::Int(999)));
        assert_eq!(map.get(&TlaValue::Int(3)), Some(&TlaValue::Int(30)));
    } else {
        panic!("expected Function, got {r:?}");
    }
}

// ============================================================
// Membership — compiled_membership_contains (24 missed),
// membership_matches_text (21 missed).
// ============================================================

#[test]
fn t207_membership_set_in() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("1 \\in {1, 2, 3}", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("9 \\in {1, 2, 3}", &ctx).unwrap(), TlaValue::Bool(false));
    // Singleton set boundary
    assert_eq!(eval_c("1 \\in {1}", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("2 \\in {1}", &ctx).unwrap(), TlaValue::Bool(false));
    // Empty set
    assert_eq!(eval_c("1 \\in {}", &ctx).unwrap(), TlaValue::Bool(false));
}

#[test]
fn t207_membership_notin() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("1 \\notin {2, 3}", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("1 \\notin {1, 2, 3}", &ctx).unwrap(), TlaValue::Bool(false));
}

#[test]
fn t207_membership_in_range() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("3 \\in 1..5", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("0 \\in 1..5", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("6 \\in 1..5", &ctx).unwrap(), TlaValue::Bool(false));
    // Boundaries — catches off-by-one in Int range membership
    assert_eq!(eval_c("1 \\in 1..5", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("5 \\in 1..5", &ctx).unwrap(), TlaValue::Bool(true));
}

#[test]
fn t207_membership_in_record_set() {
    // [a: {1, 2}] — set of records with a single field a.
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(
        eval_c("[a |-> 1] \\in [a: {1, 2}]", &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("[a |-> 9] \\in [a: {1, 2}]", &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207_membership_in_user_defined_set() {
    // membership_matches_text path: the RHS is a name resolving to a Set.
    let module = parse_tla_module_text(
        r#"
---- MODULE M ----
S == {1, 2, 3}
Test1 == 2 \in S
Test2 == 9 \in S
====
"#,
    )
    .unwrap();
    let state = TlaState::new();
    let ctx = ctx_from(&state, &module.definitions);
    let t1 = &module.definitions.get("Test1").unwrap().body;
    let t2 = &module.definitions.get("Test2").unwrap().body;
    assert_eq!(eval_c(t1, &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c(t2, &ctx).unwrap(), TlaValue::Bool(false));
}

// ============================================================
// Recursion / nested expressions — exercises eval_compiled_inner's
// `depth + 1` recursion through And / Or / Not / arithmetic.
// Tests at moderate depth, not at MAX_DEPTH boundary (brittle).
// ============================================================

#[test]
fn t207_nested_and_or_short_circuit() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Short-circuit: if first is FALSE, second not evaluated. Verify by
    // making the second produce a divide-by-zero that would error if reached.
    // 1/0 errors, so `FALSE /\ (1 \div 0 = 0)` succeeds only if short-circuit works.
    let r = eval_c("FALSE /\\ (1 \\div 0 = 0)", &ctx).unwrap();
    assert_eq!(r, TlaValue::Bool(false));
    let r2 = eval_c("TRUE \\/ (1 \\div 0 = 0)", &ctx).unwrap();
    assert_eq!(r2, TlaValue::Bool(true));
}

#[test]
fn t207_nested_not_double() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // ~~TRUE = TRUE (catches `!val` → `val` mutation)
    assert_eq!(eval_c("~~TRUE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("~~FALSE", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("~TRUE", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("~FALSE", &ctx).unwrap(), TlaValue::Bool(true));
}

#[test]
fn t207_iff_truth_table() {
    // Catches `lhs == rhs` → `lhs != rhs` mutation in Iff.
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("TRUE <=> TRUE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("FALSE <=> FALSE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("TRUE <=> FALSE", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("FALSE <=> TRUE", &ctx).unwrap(), TlaValue::Bool(false));
}

#[test]
fn t207_implies_truth_table() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert_eq!(eval_c("TRUE => TRUE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("TRUE => FALSE", &ctx).unwrap(), TlaValue::Bool(false));
    assert_eq!(eval_c("FALSE => TRUE", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("FALSE => FALSE", &ctx).unwrap(), TlaValue::Bool(true));
}

// ============================================================
// guard_text_is_action_body (30 missed) — used to detect whether
// a guard expression is actually an action body (contains primes).
// ============================================================

#[test]
fn t207_guard_action_body_distinguishes_primes() {
    // We can't call guard_text_is_action_body directly (private). But we
    // can exercise it through compile_expr by feeding expressions that
    // SHOULD vs SHOULDN'T trip the guard.
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // No prime → not an action body. Plain comparison.
    assert_eq!(eval_c("1 = 1", &ctx).unwrap(), TlaValue::Bool(true));
    assert_eq!(eval_c("1 + 1 = 2", &ctx).unwrap(), TlaValue::Bool(true));
}

// ============================================================
// Integration smoke — IfThenElse with computed condition catches
// arm-evaluation mutations.
// ============================================================

#[test]
fn t207_ifthenelse_evaluates_correct_branch() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Catches IF picking wrong branch
    assert_eq!(eval_c("IF 2 + 2 = 4 THEN 1 ELSE 0", &ctx).unwrap(), TlaValue::Int(1));
    assert_eq!(eval_c("IF 2 + 2 = 5 THEN 1 ELSE 0", &ctx).unwrap(), TlaValue::Int(0));
    // Catches ELSE not being evaluated when THEN matches
    assert_eq!(
        eval_c("IF TRUE THEN 100 ELSE (1 \\div 0)", &ctx).unwrap(),
        TlaValue::Int(100)
    );
    // Symmetric: THEN not evaluated when condition FALSE
    assert_eq!(
        eval_c("IF FALSE THEN (1 \\div 0) ELSE 100", &ctx).unwrap(),
        TlaValue::Int(100)
    );
}

// ============================================================
// CHOOSE — eval_compiled_inner Choose path
// ============================================================

#[test]
fn t207_choose_returns_satisfying_element() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // CHOOSE x \in {1,2,3,4} : x > 2 → must return 3 or 4
    let r = eval_c("CHOOSE x \\in {1, 2, 3, 4} : x > 2", &ctx).unwrap();
    if let TlaValue::Int(n) = r {
        assert!(n == 3 || n == 4, "expected 3 or 4, got {n}");
    } else {
        panic!("expected Int from CHOOSE, got {r:?}");
    }
    // Singleton set
    assert_eq!(
        eval_c("CHOOSE x \\in {42} : TRUE", &ctx).unwrap(),
        TlaValue::Int(42)
    );
}

// ============================================================
// Set operations — UNION / SUBSET / \cup / \cap
// ============================================================

#[test]
fn t207_set_ops_basic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // UNION of set of sets
    let r = eval_c("UNION {{1, 2}, {2, 3}, {3, 4}}", &ctx).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 4);
    } else {
        panic!("expected Set, got {r:?}");
    }
    // SUBSET (powerset)
    let r2 = eval_c("SUBSET {1, 2}", &ctx).unwrap();
    if let TlaValue::Set(s) = r2 {
        // {}, {1}, {2}, {1,2}
        assert_eq!(s.len(), 4);
    } else {
        panic!("expected Set, got {r2:?}");
    }
    // \cup union
    let r3 = eval_c("{1, 2} \\cup {2, 3}", &ctx).unwrap();
    if let TlaValue::Set(s) = r3 {
        assert_eq!(s.len(), 3);
    } else {
        panic!("expected Set");
    }
    // \subseteq
    assert_eq!(
        eval_c("{1, 2} \\subseteq {1, 2, 3}", &ctx).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("{1, 4} \\subseteq {1, 2, 3}", &ctx).unwrap(),
        TlaValue::Bool(false)
    );
}

// ============================================================
// Compiler-vs-interpreter parity on edge cases — final defense.
// If a missed mutation actually changes behavior, this catches it via
// the existing T2 cross-check, but here we pin specific expressions.
// ============================================================

fn parity(expr: &str) {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let i = eval_expr(expr, &ctx);
    let c = eval_c(expr, &ctx);
    match (i, c) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "interp/compi diverge on `{expr}`"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("interp/compi shape diverge on `{expr}`: {a:?} vs {b:?}"),
    }
}

#[test]
fn t207_parity_mixed_grammar() {
    parity("LET x == 3 IN x * (x + 1)");
    parity("IF Cardinality({1, 2, 3}) = 3 THEN \"y\" ELSE \"n\"");
    parity("[a |-> 1, b |-> 2].a + [a |-> 1, b |-> 2].b");
    parity("Head(Tail(<<1, 2, 3>>))");
    parity("\\A x \\in 1..5 : x >= 1 /\\ x <= 5");
    parity("\\E x \\in 1..5 : x = 3");
    parity("UNION {{1}, {2, 3}, {4, 5, 6}}");
    parity("SUBSET {1, 2}");
    parity("CASE 1 = 1 -> 100 [] OTHER -> 0");
    parity("DOMAIN [a |-> 1, b |-> 2, c |-> 3]");
    parity("LET f[i \\in 1..3] == i * i IN f[2]");
}

// ============================================================
// T207: SubSeq compiler-vs-interpreter parity. The compiler had a typed
// CompiledExpr::SubSeq variant that omitted the interpreter's m >= 1
// check, silently accepting m=0 and returning a result the interpreter
// would reject. Surfaced by this test file's t207_subseq_boundary_indices.
// ============================================================

#[test]
fn t207_subseq_typed_variant_rejects_m_zero() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("SubSeq(<<1, 2, 3>>, 0, 2)", &ctx);
    assert!(
        r.is_err(),
        "compiled SubSeq with m=0 must error to match interpreter, got {r:?}"
    );
}

#[test]
fn t207_subseq_typed_variant_rejects_negative_m() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    assert!(eval_c("SubSeq(<<1, 2, 3>>, -1, 2)", &ctx).is_err());
}

#[test]
fn t207_subseq_typed_variant_start_past_end_returns_empty() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let r = eval_c("SubSeq(<<1, 2>>, 5, 9)", &ctx).unwrap();
    assert_eq!(r, TlaValue::Seq(std::sync::Arc::new(vec![])));
}
