//! T207 iteration 2: extend compiler-helper coverage to address mutations
//! still missed after the first T207 pass. Targets:
//!
//!   * `eval_compiled_opcall` arity guards on community-module operators
//!     (SymDiff, SumSet, ProductSet, FoldSet, MapThenFoldSet,
//!     FoldFunctionOnSet, IsUndirectedGraph, IsLoopFreeUndirectedGraph,
//!     ConnectedComponents, AreConnectedIn, IsStronglyConnected,
//!     Quantify, FlattenSet, kSubset, ChooseUnique, SeqOf,
//!     IsFiniteSet, FunAsSeq).
//!   * `has_chained_top_level_arithmetic` boundary — exactly 2 vs exactly 3
//!     binary +/- chains, depth-protected zones (parens, braces, brackets,
//!     angle), unary-vs-binary distinction, string-literal skip.
//!   * `compiled_membership_contains` and `membership_matches_text` — more
//!     RHS shapes (SUBSET, UNION, named-operator references, mixed types).
//!
//! The first T207 pass lifted the compiler kill rate from 42.4% → 55.7%
//! (mutation testing on the v1.2.0 surface). This file targets the
//! highest-density missed categories from that run.

use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{CompiledExpr, EvalContext, compile_expr, eval_compiled, eval_expr};

fn ctx<'a>(state: &'a TlaState) -> EvalContext<'a> {
    EvalContext::new(state)
}

fn eval_c(expr: &str, ctx: &EvalContext<'_>) -> anyhow::Result<TlaValue> {
    eval_compiled(&compile_expr(expr), ctx)
}

// ============================================================
// has_chained_top_level_arithmetic — `binary_minus_or_plus >= 3` boundary.
// Compiler emits `CompiledExpr::Unparsed` when this returns true; we can
// observe via the AST shape.
// ============================================================

fn is_unparsed(expr: &str) -> bool {
    matches!(compile_expr(expr), CompiledExpr::Unparsed(_))
}

#[test]
fn t207b_chained_arith_one_binary_op_not_chained() {
    // 1 binary op — counter at 1, threshold 3 → not chained → structured AST.
    assert!(!is_unparsed("1 + 2"));
    assert!(!is_unparsed("5 - 3"));
}

#[test]
fn t207b_chained_arith_two_binary_ops_not_chained() {
    // 2 binary ops — counter at 2, threshold 3 → not chained.
    // Catches `>= 3` → `>= 2` mutation.
    assert!(!is_unparsed("1 + 2 + 3"));
    assert!(!is_unparsed("1 - 2 - 3"));
    assert!(!is_unparsed("1 + 2 - 3"));
}

#[test]
fn t207b_chained_arith_three_binary_ops_is_chained() {
    // 3 binary ops — counter hits 3, threshold 3 → chained → Unparsed.
    // Catches `>= 3` → `>= 4` and `>= 3` → `> 3` mutations.
    assert!(is_unparsed("1 + 2 + 3 + 4"));
    assert!(is_unparsed("1 - 2 - 3 - 4"));
    assert!(is_unparsed("1 - 2 + 3 - 4"));
}

#[test]
fn t207b_chained_arith_inside_parens_ignored() {
    // Arithmetic ops inside parens/braces/brackets should NOT count
    // toward the top-level chain count. Catches `depth_paren == 0` →
    // `!= 0` mutations. (compile_expr strips matched outer parens, so
    // we use shapes that don't trigger that — outer expression is
    // not pure-parens-wrapped.)
    // 1 binary + at top level, 4 +'s inside parens → only 1 counted → not chained
    assert!(!is_unparsed("1 + (2 + 3 + 4 + 5)"));
    // Set construction with arithmetic inside the set elements
    assert!(!is_unparsed("Cardinality({1 + 2 + 3 + 4 + 5})"));
    // Sequence with arithmetic — brackets protect
    assert!(!is_unparsed("Len(<<1 + 2 + 3 + 4 + 5>>)"));
}

#[test]
fn t207b_chained_arith_inside_string_ignored() {
    // String "1 - 2 - 3 - 4" — arithmetic inside string ignored.
    // Catches `in_string = !in_string` mutation.
    assert!(!is_unparsed("\"1 - 2 - 3 - 4\""));
}

#[test]
fn t207b_chained_arith_unary_minus_not_counted() {
    // `5 + -3` — the `-` here is unary (preceded by `+`, not a value-end).
    // So binary count is 1 (the `+`), not 2.
    assert!(!is_unparsed("5 + -3"));
    // `-1 + 2` — leading unary minus, then 1 binary op.
    assert!(!is_unparsed("-1 + 2"));
    // `5 - -3 - -4` — only one binary `-` (between 5 and -3 if at all,
    // depends on prev_non_ws). Construct so that we have only 1-2 binary.
    // Tricky; rely on simpler unary cases above.
}

#[test]
fn t207b_chained_arith_value_end_after_paren_close() {
    // `(1) + (2) + (3) + (4)` — value-end is `)`. Counter increments at
    // each top-level `+`. 3 ops total → chained.
    // Catches `p == b')'` → `!= b')'` mutations in the value-end check.
    assert!(is_unparsed("(1) + (2) + (3) + (4)"));
}

// ============================================================
// Built-in arity — community-module operators. Each has a guard like:
//     "OpName" if arg_values.len() == N && !user_defined_shadow => ...
// Calling with wrong arity should fall through (delegate to interpreter
// or fail). Calling with right arity should succeed.
// ============================================================

#[test]
fn t207b_seqof_arity_correct() {
    let s = TlaState::new();
    let c = ctx(&s);
    // SeqOf(S, n) → set of all sequences of length up to n.
    // Cardinality({}, 2) == 1 (empty seq); {1}, 2 == 1 + 1 + 1 = 3 (<<>>, <<1>>, <<1,1>>).
    let r = eval_c("SeqOf({1}, 2)", &c).unwrap();
    if let TlaValue::Set(s) = r {
        assert_eq!(s.len(), 3, "expected 3 sequences");
    } else {
        panic!("expected Set, got {r:?}");
    }
}

#[test]
fn t207b_quantify_arity_correct() {
    // Quantify(set, predicate) - count elements satisfying predicate
    let m = tlaplusplus::tla::module::parse_tla_module_text(
        r#"
---- MODULE M ----
EXTENDS FiniteSetsExt, Naturals
T == Quantify({1, 2, 3, 4, 5}, LAMBDA x: x > 2)
====
"#,
    )
    .unwrap();
    let s = TlaState::new();
    let ctx = EvalContext::with_definitions(&s, &m.definitions);
    let body = &m.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(3));
}

#[test]
fn t207b_flattenset_arity_correct() {
    let m = tlaplusplus::tla::module::parse_tla_module_text(
        r#"
---- MODULE M ----
EXTENDS FiniteSetsExt
T == FlattenSet({{1, 2}, {2, 3}, {3, 4}})
====
"#,
    )
    .unwrap();
    let s = TlaState::new();
    let ctx = EvalContext::with_definitions(&s, &m.definitions);
    let body = &m.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 4);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_ksubset_arity_correct() {
    let m = tlaplusplus::tla::module::parse_tla_module_text(
        r#"
---- MODULE M ----
EXTENDS FiniteSetsExt
T == kSubset(2, {1, 2, 3})
====
"#,
    )
    .unwrap();
    let s = TlaState::new();
    let ctx = EvalContext::with_definitions(&s, &m.definitions);
    let body = &m.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();
    if let TlaValue::Set(set) = r {
        // C(3,2) = 3 subsets of size 2
        assert_eq!(set.len(), 3);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_chooseunique_arity_correct() {
    let m = tlaplusplus::tla::module::parse_tla_module_text(
        r#"
---- MODULE M ----
EXTENDS FiniteSetsExt
T == ChooseUnique({1, 2, 3, 4}, LAMBDA x: x = 3)
====
"#,
    )
    .unwrap();
    let s = TlaState::new();
    let ctx = EvalContext::with_definitions(&s, &m.definitions);
    let body = &m.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();
    assert_eq!(r, TlaValue::Int(3));
}

#[test]
fn t207b_undirected_graph_ops() {
    // IsUndirectedGraph(G) — 1 arg — community module
    let m = tlaplusplus::tla::module::parse_tla_module_text(
        r#"
---- MODULE M ----
EXTENDS UndirectedGraphs
G == [node |-> {1, 2, 3}, edge |-> {{1, 2}, {2, 3}}]
T == IsUndirectedGraph(G)
====
"#,
    )
    .unwrap();
    let s = TlaState::new();
    let ctx = EvalContext::with_definitions(&s, &m.definitions);
    let body = &m.definitions.get("T").unwrap().body;
    let r = eval_compiled(&compile_expr(body), &ctx).unwrap();
    assert_eq!(r, TlaValue::Bool(true));
}

// ============================================================
// SymDiff edge cases — 2-arg arity, set semantics.
// ============================================================

#[test]
fn t207b_symdiff_disjoint_sets() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c("SymDiff({1, 2}, {3, 4})", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 4);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_symdiff_identical_sets() {
    let s = TlaState::new();
    let c = ctx(&s);
    // SymDiff(A, A) = empty
    let r = eval_c("SymDiff({1, 2, 3}, {1, 2, 3})", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 0);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_symdiff_one_empty() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c("SymDiff({}, {1, 2, 3})", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 3);
    } else {
        panic!("expected Set");
    }
}

// ============================================================
// SumSet / ProductSet boundary cases.
// ============================================================

#[test]
fn t207b_sumset_singleton() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("SumSet({42})", &c).unwrap(), TlaValue::Int(42));
}

#[test]
fn t207b_sumset_negative_elements() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("SumSet({-1, 2, -3, 4})", &c).unwrap(), TlaValue::Int(2));
}

#[test]
fn t207b_productset_singleton() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("ProductSet({7})", &c).unwrap(), TlaValue::Int(7));
}

#[test]
fn t207b_productset_with_zero() {
    let s = TlaState::new();
    let c = ctx(&s);
    // 0 in product → 0
    assert_eq!(eval_c("ProductSet({0, 5, 10})", &c).unwrap(), TlaValue::Int(0));
}

// ============================================================
// Membership patterns — compiled_membership_contains (24) and
// membership_matches_text (21). More shapes than t207 covered.
// ============================================================

#[test]
fn t207b_membership_in_subset() {
    // x \in SUBSET S — x is itself a set
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(
        eval_c("{1, 2} \\in SUBSET {1, 2, 3}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("{1, 9} \\in SUBSET {1, 2, 3}", &c).unwrap(),
        TlaValue::Bool(false)
    );
    // Empty set is in any powerset
    assert_eq!(
        eval_c("{} \\in SUBSET {1, 2, 3}", &c).unwrap(),
        TlaValue::Bool(true)
    );
}

#[test]
fn t207b_membership_in_union() {
    let s = TlaState::new();
    let c = ctx(&s);
    // x \in UNION S — element of any set in S
    assert_eq!(
        eval_c("2 \\in UNION {{1, 2}, {3, 4}}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("9 \\in UNION {{1, 2}, {3, 4}}", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207b_membership_in_set_comprehension() {
    let s = TlaState::new();
    let c = ctx(&s);
    // 4 \in {x * 2 : x \in 1..3} = {2, 4, 6} → true
    assert_eq!(
        eval_c("4 \\in {x * 2 : x \\in 1..3}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("3 \\in {x * 2 : x \\in 1..3}", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207b_membership_string_in_set() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(
        eval_c("\"foo\" \\in {\"foo\", \"bar\"}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("\"baz\" \\in {\"foo\", \"bar\"}", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207b_membership_seq_in_set() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(
        eval_c("<<1, 2>> \\in {<<1, 2>>, <<3, 4>>}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("<<1, 9>> \\in {<<1, 2>>, <<3, 4>>}", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207b_membership_record_in_set() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(
        eval_c("[a |-> 1] \\in {[a |-> 1], [a |-> 2]}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("[a |-> 9] \\in {[a |-> 1], [a |-> 2]}", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

#[test]
fn t207b_membership_in_function_set() {
    // [S -> T] is the set of all functions from S to T
    let s = TlaState::new();
    let c = ctx(&s);
    // [a \in {1} |-> 5] is in [{1} -> {5, 6, 7}]
    assert_eq!(
        eval_c("[a \\in {1} |-> 5] \\in [{1} -> {5, 6, 7}]", &c).unwrap(),
        TlaValue::Bool(true)
    );
    assert_eq!(
        eval_c("[a \\in {1} |-> 9] \\in [{1} -> {5, 6, 7}]", &c).unwrap(),
        TlaValue::Bool(false)
    );
}

// ============================================================
// Set algebra parity — exercises split_first_top_level_op + binary op
// dispatch on every set operator. Catches `==` → `!=`, `&&` → `||`
// mutations in op recognition.
// ============================================================

#[test]
fn t207b_set_intersection_basic() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c("{1, 2, 3} \\cap {2, 3, 4}", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 2);
        assert!(set.contains(&TlaValue::Int(2)));
        assert!(set.contains(&TlaValue::Int(3)));
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_set_difference() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c("{1, 2, 3} \\ {2}", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 2);
        assert!(set.contains(&TlaValue::Int(1)));
        assert!(set.contains(&TlaValue::Int(3)));
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_subseteq_strict_proper_relation() {
    let s = TlaState::new();
    let c = ctx(&s);
    // \subseteq allows equality
    assert_eq!(
        eval_c("{1, 2} \\subseteq {1, 2}", &c).unwrap(),
        TlaValue::Bool(true)
    );
    // Empty subset of any set
    assert_eq!(eval_c("{} \\subseteq {1}", &c).unwrap(), TlaValue::Bool(true));
    // Empty subset of empty
    assert_eq!(eval_c("{} \\subseteq {}", &c).unwrap(), TlaValue::Bool(true));
}

// ============================================================
// Arithmetic edge cases — modulo, integer division, negation.
// Many missed mutations are in the dispatch path for these.
// ============================================================

#[test]
fn t207b_arith_modulo() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("10 % 3", &c).unwrap(), TlaValue::Int(1));
    assert_eq!(eval_c("9 % 3", &c).unwrap(), TlaValue::Int(0));
    assert_eq!(eval_c("0 % 5", &c).unwrap(), TlaValue::Int(0));
}

#[test]
fn t207b_arith_div() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("10 \\div 3", &c).unwrap(), TlaValue::Int(3));
    assert_eq!(eval_c("9 \\div 3", &c).unwrap(), TlaValue::Int(3));
    assert_eq!(eval_c("0 \\div 5", &c).unwrap(), TlaValue::Int(0));
    // Division by zero must error
    assert!(eval_c("5 \\div 0", &c).is_err());
}

#[test]
fn t207b_arith_neg() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("-7", &c).unwrap(), TlaValue::Int(-7));
    assert_eq!(eval_c("-(-7)", &c).unwrap(), TlaValue::Int(7));
    // Subtraction with negative operand
    assert_eq!(eval_c("5 - -3", &c).unwrap(), TlaValue::Int(8));
}

#[test]
fn t207b_arith_exponentiation() {
    let s = TlaState::new();
    let c = ctx(&s);
    assert_eq!(eval_c("2 ^ 3", &c).unwrap(), TlaValue::Int(8));
    assert_eq!(eval_c("5 ^ 0", &c).unwrap(), TlaValue::Int(1));
    assert_eq!(eval_c("3 ^ 4", &c).unwrap(), TlaValue::Int(81));
}

// ============================================================
// Range operator boundary — `m..n`. Exercises split + range construction.
// ============================================================

#[test]
fn t207b_range_singleton_and_empty() {
    let s = TlaState::new();
    let c = ctx(&s);
    // 5..5 = {5}
    let r = eval_c("5..5", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 1);
        assert!(set.contains(&TlaValue::Int(5)));
    } else {
        panic!("expected Set");
    }
    // 5..3 = {} (empty)
    let r2 = eval_c("5..3", &c).unwrap();
    if let TlaValue::Set(set) = r2 {
        assert_eq!(set.len(), 0);
    } else {
        panic!("expected Set");
    }
}

#[test]
fn t207b_range_with_negative() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c("-2..2", &c).unwrap();
    if let TlaValue::Set(set) = r {
        assert_eq!(set.len(), 5);
        assert!(set.contains(&TlaValue::Int(-2)));
        assert!(set.contains(&TlaValue::Int(0)));
        assert!(set.contains(&TlaValue::Int(2)));
    } else {
        panic!("expected Set");
    }
}

// ============================================================
// Nested IF / CASE — exercises eval_compiled_inner recursion
// (depth + 1 mutations) at moderate depth.
// ============================================================

#[test]
fn t207b_nested_if_depth_5() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c(
        "IF TRUE THEN \
            IF TRUE THEN \
                IF TRUE THEN \
                    IF TRUE THEN \
                        IF TRUE THEN 100 ELSE 0 \
                    ELSE 0 \
                ELSE 0 \
            ELSE 0 \
        ELSE 0",
        &c,
    )
    .unwrap();
    assert_eq!(r, TlaValue::Int(100));
}

#[test]
fn t207b_nested_let_depth_5() {
    let s = TlaState::new();
    let c = ctx(&s);
    let r = eval_c(
        "LET a == 1 IN LET b == 2 IN LET c == 3 IN LET d == 4 IN LET e == 5 IN a + b + c + d + e",
        &c,
    )
    .unwrap();
    // 1+2+3+4+5 = 15. But this has 4 binary +/- which triggers Unparsed,
    // so this also exercises that path. Both paths should produce 15.
    assert_eq!(r, TlaValue::Int(15));
}

// ============================================================
// String operators — Asc, Chr, ToString variants.
// ============================================================

#[test]
fn t207b_tostring_for_each_value_kind() {
    let s = TlaState::new();
    let c = ctx(&s);
    // Bool
    if let TlaValue::String(s1) = eval_c("ToString(TRUE)", &c).unwrap() {
        assert!(s1.contains("TRUE") || s1 == "true");
    }
    // Set
    let r2 = eval_c("ToString({1, 2, 3})", &c).unwrap();
    assert!(matches!(r2, TlaValue::String(_)));
    // String passthrough
    let r3 = eval_c("ToString(\"hello\")", &c).unwrap();
    assert!(matches!(r3, TlaValue::String(_)));
}

// ============================================================
// CHOOSE x \in S : P — exercises eval_compiled_inner CHOOSE branch.
// ============================================================

#[test]
fn t207b_choose_no_satisfying_element_errors() {
    let s = TlaState::new();
    let c = ctx(&s);
    // No element satisfies — CHOOSE must error
    let r = eval_c("CHOOSE x \\in {1, 2, 3} : x > 100", &c);
    assert!(
        r.is_err(),
        "CHOOSE with no satisfying element should error, got {r:?}"
    );
}

#[test]
fn t207b_choose_filters_correctly() {
    let s = TlaState::new();
    let c = ctx(&s);
    // CHOOSE x \in {1,2,3,4} : x % 2 = 0 → 2 or 4
    let r = eval_c("CHOOSE x \\in {1, 2, 3, 4} : x % 2 = 0", &c).unwrap();
    if let TlaValue::Int(n) = r {
        assert!(n == 2 || n == 4, "expected 2 or 4, got {n}");
    } else {
        panic!("expected Int");
    }
}

// ============================================================
// Cross-check parity — for each shape, interp and compi must agree.
// Catches mutations that don't change errs/oks but change values.
// ============================================================

fn parity(expr: &str) {
    let s = TlaState::new();
    let c = ctx(&s);
    let i = eval_expr(expr, &c);
    let cm = eval_c(expr, &c);
    match (i, cm) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "interp/compi differ on `{expr}`"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("shape diverge on `{expr}`: {a:?} vs {b:?}"),
    }
}

#[test]
fn t207b_parity_arithmetic_chains() {
    parity("1 + 2 * 3");
    parity("10 - 4 - 2");
    parity("(5 + 3) * 2");
    parity("100 \\div 10 + 5");
    parity("(7 % 3) * 2");
    parity("2 ^ 3 + 1");
}

#[test]
fn t207b_parity_set_chains() {
    parity("({1,2} \\cup {3}) \\cap {2,3}");
    parity("{1,2,3} \\ ({2} \\cup {})");
    parity("Cardinality(SUBSET {1,2})");
    parity("UNION {{1}, {2,3}, {3,4}}");
}

#[test]
fn t207b_parity_seq_chains() {
    parity("Append(Append(<<>>, 1), 2)");
    parity("Tail(Tail(<<1, 2, 3>>))");
    parity("Len(Append(<<1>>, 2))");
    parity("Head(<<10, 20>>) + Head(Tail(<<10, 20>>))");
}

#[test]
fn t207b_parity_record_chains() {
    parity("[a |-> 1, b |-> 2].a + [a |-> 1, b |-> 2].b");
    parity("[[a |-> 1] EXCEPT !.a = 99].a");
    parity("Cardinality(DOMAIN [a |-> 1, b |-> 2, c |-> 3])");
}

#[test]
fn t207b_parity_quantifiers() {
    parity("\\A x \\in 1..10 : x > 0");
    parity("\\E x \\in 1..10 : x = 7");
    parity("\\A x \\in {1,2,3} : \\E y \\in {1,2,3} : x = y");
    parity("\\E x \\in {} : x > 0");
}
