//! T2 — Property-based equivalence harness between the interpreted and
//! compiled TLA+ expression evaluators.
//!
//! Generates random *well-typed-with-high-probability* TLA+ expressions and
//! asserts that `eval_expr` (interpreter, in `src/tla/eval.rs`) and
//! `eval_compiled(compile_expr(...))` (compiler, in `src/tla/compiled_*.rs`)
//! agree on every input. If either side errors, both must error.
//!
//! Phase-1 fixes (T1.1, T1.4, T1.5) all involved silent drift between these
//! two paths; this harness is the regression gate that catches future drift.
//!
//! Per the T2 working rules: when a divergence is found we DO NOT patch it
//! here. The proptest is allowed to fail (panic with the minimised
//! counter-example), the seed/expression goes into `RELEASE_1.0.0_LOG.md`
//! under T2.N follow-ups, and triage continues separately.
//!
//! `PROPTEST_CASES` controls iteration count. CI pins it (see
//! `.github/workflows/diff-tlc.yml`); local devs can crank it higher via the
//! environment variable.

use proptest::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

use tlaplusplus::tla::{
    EvalContext, TlaDefinition, TlaState, TlaValue, compile_expr, eval_compiled, eval_expr,
    tla_state,
};

// ---------------------------------------------------------------------------
// Test environment: a small fixed state + a few definitions.
// ---------------------------------------------------------------------------

/// Build the shared evaluation context. We use:
///   x = 3, y = 7  (Ints)
///   b = TRUE      (Bool)
///   S = {1, 2, 3} (Set of Int)
///   T = {2, 3, 4} (Set of Int)
///   sq = <<10, 20, 30>> (Seq of Int)
///   r = [a |-> 1, b |-> 2] (Record)
///   f = (1 :> 100 @@ 2 :> 200) (Function: Int -> Int)
///
/// Plus operator definitions:
///   Inc(n) == n + 1
///   Add2(a, b) == a + b
///   IsPos(n) == n > 0
///
/// Names are intentionally short so the generator can spell them as identifiers.
fn build_state() -> TlaState {
    use std::collections::{BTreeMap, BTreeSet};

    let mut s = BTreeSet::new();
    s.insert(TlaValue::Int(1));
    s.insert(TlaValue::Int(2));
    s.insert(TlaValue::Int(3));

    let mut t = BTreeSet::new();
    t.insert(TlaValue::Int(2));
    t.insert(TlaValue::Int(3));
    t.insert(TlaValue::Int(4));

    let sq = vec![TlaValue::Int(10), TlaValue::Int(20), TlaValue::Int(30)];

    let mut rec = BTreeMap::new();
    rec.insert("a".to_string(), TlaValue::Int(1));
    rec.insert("b".to_string(), TlaValue::Int(2));

    let mut func = BTreeMap::new();
    func.insert(TlaValue::Int(1), TlaValue::Int(100));
    func.insert(TlaValue::Int(2), TlaValue::Int(200));

    tla_state([
        ("x", TlaValue::Int(3)),
        ("y", TlaValue::Int(7)),
        ("b", TlaValue::Bool(true)),
        ("S", TlaValue::Set(Arc::new(s))),
        ("T", TlaValue::Set(Arc::new(t))),
        ("sq", TlaValue::Seq(Arc::new(sq))),
        ("r", TlaValue::Record(Arc::new(rec))),
        ("f", TlaValue::Function(Arc::new(func))),
    ])
}

fn build_definitions() -> BTreeMap<String, TlaDefinition> {
    let mut defs = BTreeMap::new();
    defs.insert(
        "Inc".to_string(),
        TlaDefinition {
            name: "Inc".to_string(),
            params: vec!["n".to_string()],
            body: "n + 1".to_string(),
            is_recursive: false,
        },
    );
    defs.insert(
        "Add2".to_string(),
        TlaDefinition {
            name: "Add2".to_string(),
            params: vec!["a".to_string(), "b".to_string()],
            body: "a + b".to_string(),
            is_recursive: false,
        },
    );
    defs.insert(
        "IsPos".to_string(),
        TlaDefinition {
            name: "IsPos".to_string(),
            params: vec!["n".to_string()],
            body: "n > 0".to_string(),
            is_recursive: false,
        },
    );
    defs
}

// ---------------------------------------------------------------------------
// Typed expression generators.
//
// We track a *type tag* with each generated expression so that we can
// compose them well-typedly. This keeps the well-typed proportion high
// without throwing away tons of cases via `prop_assume!`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Typed {
    Int(String),
    Bool(String),
    /// Set of Int.
    SetInt(String),
    /// Sequence of Int.
    SeqInt(String),
    /// Record with fields {a, b} both Int.
    RecAB(String),
    /// String literal.
    Str(String),
}

impl Typed {
    fn text(&self) -> &str {
        match self {
            Typed::Int(s)
            | Typed::Bool(s)
            | Typed::SetInt(s)
            | Typed::SeqInt(s)
            | Typed::RecAB(s)
            | Typed::Str(s) => s,
        }
    }
}

// Leaf generators ----------------------------------------------------------

fn arb_int_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        // small literals (favour zero, ±1, ±2 to surface boundary bugs)
        (-5i64..=5).prop_map(|n| Typed::Int(format!("{}", n))),
        // state variables
        Just(Typed::Int("x".to_string())),
        Just(Typed::Int("y".to_string())),
        // constants from definitions
        Just(Typed::Int("Inc(0)".to_string())),
    ]
}

fn arb_bool_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        Just(Typed::Bool("TRUE".to_string())),
        Just(Typed::Bool("FALSE".to_string())),
        Just(Typed::Bool("b".to_string())),
    ]
}

fn arb_set_int_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        Just(Typed::SetInt("S".to_string())),
        Just(Typed::SetInt("T".to_string())),
        Just(Typed::SetInt("{}".to_string())),
        Just(Typed::SetInt("{1, 2, 3}".to_string())),
        Just(Typed::SetInt("{0}".to_string())),
        // small range
        Just(Typed::SetInt("1..3".to_string())),
        Just(Typed::SetInt("0..2".to_string())),
    ]
}

fn arb_seq_int_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        Just(Typed::SeqInt("sq".to_string())),
        Just(Typed::SeqInt("<<>>".to_string())),
        Just(Typed::SeqInt("<<1, 2>>".to_string())),
        Just(Typed::SeqInt("<<5>>".to_string())),
    ]
}

fn arb_rec_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        Just(Typed::RecAB("r".to_string())),
        Just(Typed::RecAB("[a |-> 1, b |-> 2]".to_string())),
        Just(Typed::RecAB("[a |-> 0, b |-> 0]".to_string())),
    ]
}

fn arb_str_leaf() -> impl Strategy<Value = Typed> {
    prop_oneof![
        Just(Typed::Str("\"hello\"".to_string())),
        Just(Typed::Str("\"world\"".to_string())),
        Just(Typed::Str("\"\"".to_string())),
    ]
}

// Recursive composition ----------------------------------------------------

fn arb_int(depth: u32) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_int_leaf().boxed();
    }
    let leaf = arb_int_leaf();
    let arith = (arb_int(depth - 1), arb_int(depth - 1)).prop_flat_map(|(a, b)| {
        prop_oneof![
            Just(Typed::Int(format!("({} + {})", a.text(), b.text()))),
            Just(Typed::Int(format!("({} - {})", a.text(), b.text()))),
            Just(Typed::Int(format!("({} * {})", a.text(), b.text()))),
            // \div / % only when divisor is a non-zero literal to avoid
            // div-by-zero asymmetries (both eval paths error, but error
            // *texts* may differ — we treat both-error as equal so this is
            // mostly fine, but pinning the divisor reduces noise).
            Just(Typed::Int(format!("({} \\div 2)", a.text()))),
            Just(Typed::Int(format!("({} % 3)", a.text()))),
        ]
    });
    let neg = arb_int(depth - 1).prop_map(|a| Typed::Int(format!("(-({}))", a.text())));
    let if_int =
        (arb_bool(depth - 1), arb_int(depth - 1), arb_int(depth - 1)).prop_map(|(c, t, e)| {
            Typed::Int(format!(
                "(IF {} THEN {} ELSE {})",
                c.text(),
                t.text(),
                e.text()
            ))
        });
    let let_int = (arb_int(depth - 1), arb_int(depth - 1)).prop_map(|(v, body)| {
        // shadow with __t1 to dodge collisions with state vars
        Typed::Int(format!(
            "(LET __t1 == {} IN ({} + __t1))",
            v.text(),
            body.text()
        ))
    });
    let opcall = arb_int(depth - 1).prop_map(|a| Typed::Int(format!("Inc({})", a.text())));
    let opcall2 = (arb_int(depth - 1), arb_int(depth - 1))
        .prop_map(|(a, b)| Typed::Int(format!("Add2({}, {})", a.text(), b.text())));
    // sequence Len / Head
    let len = arb_seq_int(depth - 1).prop_map(|s| Typed::Int(format!("Len({})", s.text())));
    let head = arb_seq_int(depth - 1).prop_map(|s| Typed::Int(format!("Head({})", s.text())));
    // record access
    let rec_a = arb_rec(depth - 1).prop_map(|r| Typed::Int(format!("({}).a", r.text())));
    let rec_b = arb_rec(depth - 1).prop_map(|r| Typed::Int(format!("({}).b", r.text())));
    // function application f[1] / f[2]
    let func_app = prop_oneof![
        Just(Typed::Int("f[1]".to_string())),
        Just(Typed::Int("f[2]".to_string())),
    ];
    // case
    let case =
        (arb_bool(depth - 1), arb_int(depth - 1), arb_int(depth - 1)).prop_map(|(c, a, b)| {
            Typed::Int(format!(
                "(CASE {} -> {} [] OTHER -> {})",
                c.text(),
                a.text(),
                b.text()
            ))
        });

    prop_oneof![
        4 => leaf,
        3 => arith,
        1 => neg,
        2 => if_int,
        1 => let_int,
        2 => opcall,
        1 => opcall2,
        1 => len,
        1 => head,
        1 => rec_a,
        1 => rec_b,
        1 => func_app,
        1 => case,
    ]
    .boxed()
}

fn arb_bool(depth: u32) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_bool_leaf().boxed();
    }
    let leaf = arb_bool_leaf();
    let conn = (arb_bool(depth - 1), arb_bool(depth - 1)).prop_flat_map(|(a, b)| {
        prop_oneof![
            Just(Typed::Bool(format!("({} /\\ {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} \\/ {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} => {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} <=> {})", a.text(), b.text()))),
        ]
    });
    let not = arb_bool(depth - 1).prop_map(|a| Typed::Bool(format!("(~ {})", a.text())));
    let cmp_int = (arb_int(depth - 1), arb_int(depth - 1)).prop_flat_map(|(a, b)| {
        prop_oneof![
            Just(Typed::Bool(format!("({} = {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} # {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} < {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} > {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} <= {})", a.text(), b.text()))),
            Just(Typed::Bool(format!("({} >= {})", a.text(), b.text()))),
        ]
    });
    let in_set = (arb_int(depth - 1), arb_set_int(depth - 1)).prop_flat_map(|(e, s)| {
        prop_oneof![
            Just(Typed::Bool(format!("({} \\in {})", e.text(), s.text()))),
            Just(Typed::Bool(format!("({} \\notin {})", e.text(), s.text()))),
        ]
    });
    let subseteq = (arb_set_int(depth - 1), arb_set_int(depth - 1))
        .prop_map(|(a, b)| Typed::Bool(format!("({} \\subseteq {})", a.text(), b.text())));
    // Quantifiers over fixed small set 1..3 (using a safe binder name).
    let exists = arb_bool(depth - 1).prop_map(|body| {
        // body may freely reference the state var x, y, etc., but we add a
        // fresh binder __q so the shape exercises the binding mechanism.
        Typed::Bool(format!(
            "(\\E __q \\in 1..3 : ({} \\/ (__q > 0)))",
            body.text()
        ))
    });
    let forall = arb_bool(depth - 1).prop_map(|body| {
        Typed::Bool(format!(
            "(\\A __q \\in 1..3 : ({} \\/ (__q > 0)))",
            body.text()
        ))
    });
    let if_bool = (
        arb_bool(depth - 1),
        arb_bool(depth - 1),
        arb_bool(depth - 1),
    )
        .prop_map(|(c, t, e)| {
            Typed::Bool(format!(
                "(IF {} THEN {} ELSE {})",
                c.text(),
                t.text(),
                e.text()
            ))
        });
    let opcall = arb_int(depth - 1).prop_map(|a| Typed::Bool(format!("IsPos({})", a.text())));

    prop_oneof![
        4 => leaf,
        3 => conn,
        1 => not,
        3 => cmp_int,
        2 => in_set,
        1 => subseteq,
        1 => exists,
        1 => forall,
        1 => if_bool,
        1 => opcall,
    ]
    .boxed()
}

fn arb_set_int(depth: u32) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_set_int_leaf().boxed();
    }
    let leaf = arb_set_int_leaf();
    let union = (arb_set_int(depth - 1), arb_set_int(depth - 1))
        .prop_map(|(a, b)| Typed::SetInt(format!("({} \\union {})", a.text(), b.text())));
    let inter = (arb_set_int(depth - 1), arb_set_int(depth - 1))
        .prop_map(|(a, b)| Typed::SetInt(format!("({} \\intersect {})", a.text(), b.text())));
    let minus = (arb_set_int(depth - 1), arb_set_int(depth - 1))
        .prop_map(|(a, b)| Typed::SetInt(format!("({} \\ {})", a.text(), b.text())));
    let lit = (arb_int(depth - 1), arb_int(depth - 1))
        .prop_map(|(a, b)| Typed::SetInt(format!("{{{}, {}}}", a.text(), b.text())));
    let filter = arb_bool(depth - 1).prop_map(|p| {
        // {__e \in 1..4 : P(__e)} — the body may reference outer vars but
        // we anchor the predicate in __e to keep the filter meaningful.
        Typed::SetInt(format!("{{__e \\in 1..4 : ({}) \\/ (__e > 0)}}", p.text()))
    });
    let map = arb_int(depth - 1).prop_map(|e| {
        // {__e + 1 : __e \in 1..3}
        Typed::SetInt(format!("{{({}) + __e : __e \\in 1..3}}", e.text()))
    });

    prop_oneof![
        4 => leaf,
        2 => union,
        2 => inter,
        2 => minus,
        2 => lit,
        1 => filter,
        1 => map,
    ]
    .boxed()
}

fn arb_seq_int(depth: u32) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_seq_int_leaf().boxed();
    }
    let leaf = arb_seq_int_leaf();
    let append = (arb_seq_int(depth - 1), arb_int(depth - 1))
        .prop_map(|(s, e)| Typed::SeqInt(format!("Append({}, {})", s.text(), e.text())));
    let tail = arb_seq_int(depth - 1).prop_map(|s| Typed::SeqInt(format!("Tail({})", s.text())));
    let lit = (arb_int(depth - 1), arb_int(depth - 1))
        .prop_map(|(a, b)| Typed::SeqInt(format!("<<{}, {}>>", a.text(), b.text())));

    prop_oneof![
        3 => leaf,
        2 => append,
        2 => tail,
        2 => lit,
    ]
    .boxed()
}

fn arb_rec(depth: u32) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_rec_leaf().boxed();
    }
    let leaf = arb_rec_leaf();
    let lit = (arb_int(depth - 1), arb_int(depth - 1))
        .prop_map(|(a, b)| Typed::RecAB(format!("[a |-> {}, b |-> {}]", a.text(), b.text())));
    let except_a = (arb_rec(depth - 1), arb_int(depth - 1))
        .prop_map(|(r, v)| Typed::RecAB(format!("[{} EXCEPT !.a = {}]", r.text(), v.text())));
    let except_b = (arb_rec(depth - 1), arb_int(depth - 1))
        .prop_map(|(r, v)| Typed::RecAB(format!("[{} EXCEPT !.b = {}]", r.text(), v.text())));

    prop_oneof![
        3 => leaf,
        2 => lit,
        1 => except_a,
        1 => except_b,
    ]
    .boxed()
}

/// Top-level expression of any type. Each iteration picks a type and recurses.
fn arb_expr() -> impl Strategy<Value = Typed> {
    let depth = 3u32;
    prop_oneof![
        arb_int(depth),
        arb_bool(depth),
        arb_set_int(depth),
        arb_seq_int(depth),
        arb_rec(depth),
        arb_str_leaf().boxed(),
    ]
}

// ---------------------------------------------------------------------------
// Equivalence assertion.
// ---------------------------------------------------------------------------

/// Compare two evaluation results. Both-error is treated as equal (so we
/// don't surface noise from differing error texts on e.g. div-by-zero).
fn compare(
    expr: &str,
    interp: anyhow::Result<TlaValue>,
    compi: anyhow::Result<TlaValue>,
) -> Result<(), String> {
    match (interp, compi) {
        (Ok(a), Ok(b)) => {
            if a == b {
                Ok(())
            } else {
                Err(format!(
                    "DIVERGENCE on `{expr}`:\n  interpreter -> {a:?}\n  compiler    -> {b:?}"
                ))
            }
        }
        (Err(_), Err(_)) => Ok(()),
        (Ok(a), Err(e)) => Err(format!(
            "DIVERGENCE on `{expr}`:\n  interpreter -> Ok({a:?})\n  compiler    -> Err({e})"
        )),
        (Err(e), Ok(b)) => Err(format!(
            "DIVERGENCE on `{expr}`:\n  interpreter -> Err({e})\n  compiler    -> Ok({b:?})"
        )),
    }
}

fn check_equivalence(expr: &str) -> Result<(), String> {
    // Catch panics — either evaluator panicking is itself a divergence we
    // want to surface (and not bring down the whole proptest run with).
    let state = build_state();
    let defs = build_definitions();
    let ctx = EvalContext::with_definitions(&state, &defs);

    let interp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_expr(expr, &ctx)));
    let compi = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let compiled = compile_expr(expr);
        eval_compiled(&compiled, &ctx)
    }));

    match (interp, compi) {
        (Ok(i), Ok(c)) => compare(expr, i, c),
        (Err(_), Err(_)) => Ok(()), // both panicked - symmetric
        (Ok(_), Err(_)) => Err(format!(
            "DIVERGENCE on `{expr}`: compiler panicked, interpreter did not"
        )),
        (Err(_), Ok(_)) => Err(format!(
            "DIVERGENCE on `{expr}`: interpreter panicked, compiler did not"
        )),
    }
}

// ---------------------------------------------------------------------------
// proptest entry points.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        // CI pins via PROPTEST_CASES env var; default 256 keeps local runs cheap.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256),
        // Cap shrinking iterations — the generator is recursive enough that
        // shrinking can spend a long time without finding a much smaller
        // counter-example.
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    /// The single big property: for any generated expression, the
    /// interpreter and compiler agree (both produce the same Ok value, or
    /// both produce an error).
    #[test]
    fn compiled_matches_interpreted(expr in arb_expr()) {
        let text = expr.text().to_string();
        // Skip empty-text or trivially-malformed cases (shouldn't happen
        // with our generators, but defence-in-depth).
        prop_assume!(!text.is_empty());
        if let Err(msg) = check_equivalence(&text) {
            return Err(TestCaseError::fail(msg));
        }
    }
}

// ---------------------------------------------------------------------------
// Sanity tests — quick smoke checks that the harness itself works correctly.
// These run in the regular `cargo test` step alongside the proptest.
// ---------------------------------------------------------------------------

#[test]
fn sanity_int_literal() {
    check_equivalence("42").unwrap();
}

#[test]
fn sanity_arith() {
    check_equivalence("(2 + 3) * 4").unwrap();
}

#[test]
fn sanity_var_reference() {
    check_equivalence("x + y").unwrap();
}

#[test]
fn sanity_set_membership() {
    check_equivalence("3 \\in S").unwrap();
}

#[test]
fn sanity_quantifier() {
    check_equivalence("\\E __q \\in 1..3 : __q > 1").unwrap();
}

#[test]
fn sanity_record_access() {
    check_equivalence("r.a + r.b").unwrap();
}

#[test]
fn sanity_seq_ops() {
    check_equivalence("Len(Append(sq, 99))").unwrap();
}

#[test]
fn sanity_let() {
    check_equivalence("LET __t1 == x + 1 IN __t1 * 2").unwrap();
}

#[test]
fn sanity_opcall() {
    check_equivalence("Add2(Inc(2), 3)").unwrap();
}

#[test]
fn sanity_div_by_zero_both_error() {
    // Both eval paths should error on this. compare() treats both-error as equal.
    check_equivalence("5 \\div 0").unwrap();
}
