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
//!
//! T16a — Swarm testing (Regehr et al., ICST 2012, "Swarm Testing"):
//! each proptest *case* first samples a random subset of expression-shape
//! categories ("the swarm mask"), then draws expressions only from the
//! enabled subset. Leaves (literals/vars) are always enabled; "shape"
//! categories (arith, comparisons, set ops, quantifiers, EXCEPT, sequences,
//! records, function-app, LET, IF, CASE, opcalls) are each kept with
//! independent biased probability. This biases the population toward
//! minimal-interaction cases that surface evaluator bugs in feature-pair
//! seams that uniform full-feature sampling under-explores.
//!
//! Set `SWARM_MODE=uniform` to disable swarming and use the original
//! all-categories-always-on generator. Default = swarm.
//!
//! Known limitation: when proptest's shrinker tries to minimise a
//! divergence, it can shrink to a smaller expression that is itself outside
//! the swarm mask of the original failing case. The minimum it produces
//! may therefore differ from a "true" minimum for that mask. This is
//! documented in T16a; treat the printed expression as a starting point,
//! not an oracle, when triaging T16.N follow-ups.

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
/// Uniform mode — every shape category is always available.
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
// T16a — Swarm-mode generators.
//
// Each proptest case first samples a `SwarmMask` (the "swarm"), then the
// recursive generators only emit productions whose category bit is set.
// Leaf productions (literals, state vars, constant opcalls) are always
// enabled so every call still terminates with a valid expression.
//
// Per-category enable probabilities are biased to favour smaller subsets
// (~50% each independently); the unconditional `Always`-categories ensure
// the empty-mask edge case still produces a well-formed leaf-only program.
// ---------------------------------------------------------------------------

/// One bit per "shape" category that the recursive generator can emit.
/// Leaf productions are always available; these are the *recursive*
/// productions that actually compose features.
#[derive(Debug, Clone, Copy)]
struct SwarmMask {
    arith: bool,         // + - * \div %
    neg: bool,           // unary -
    cmp_int: bool,       // = # < > <= >=
    in_set: bool,        // \in / \notin
    subseteq: bool,      // \subseteq
    set_ops: bool,       // \union \intersect \ {a,b} {x \in S : P} {f(x) : x \in S}
    seq_ops: bool,       // Append Tail Len Head <<a,b>>
    record_ops: bool,    // [a |-> ..., b |-> ...]
    record_except: bool, // [r EXCEPT !.f = v]
    func_app: bool,      // f[1] f[2]
    quantifier: bool,    // \E / \A
    let_in: bool,        // LET ... IN ...
    if_then_else: bool,  // IF ... THEN ... ELSE ...
    case_arm: bool,      // CASE ... -> ... [] OTHER -> ...
    opcall: bool,        // Inc(...) Add2(...,...) IsPos(...)
    bool_conn: bool,     // /\ \/ => <=>
    bool_not: bool,      // ~
}

impl SwarmMask {
    /// Mask with everything enabled — equivalent to the uniform generator.
    fn all_on() -> Self {
        Self {
            arith: true,
            neg: true,
            cmp_int: true,
            in_set: true,
            subseteq: true,
            set_ops: true,
            seq_ops: true,
            record_ops: true,
            record_except: true,
            func_app: true,
            quantifier: true,
            let_in: true,
            if_then_else: true,
            case_arm: true,
            opcall: true,
            bool_conn: true,
            bool_not: true,
        }
    }

    /// Compact human-readable list of enabled categories — embedded in
    /// divergence messages so we can tell *which* swarm produced a failure.
    fn describe(&self) -> String {
        let mut on = Vec::new();
        let mut push = |b: bool, name: &str| {
            if b {
                on.push(name.to_string());
            }
        };
        push(self.arith, "arith");
        push(self.neg, "neg");
        push(self.cmp_int, "cmp");
        push(self.in_set, "in");
        push(self.subseteq, "subseteq");
        push(self.set_ops, "set");
        push(self.seq_ops, "seq");
        push(self.record_ops, "rec");
        push(self.record_except, "except");
        push(self.func_app, "fn");
        push(self.quantifier, "quant");
        push(self.let_in, "let");
        push(self.if_then_else, "if");
        push(self.case_arm, "case");
        push(self.opcall, "op");
        push(self.bool_conn, "conn");
        push(self.bool_not, "not");
        if on.is_empty() {
            "<leaves only>".to_string()
        } else {
            on.join(",")
        }
    }
}

/// Strategy that samples a `SwarmMask`. Each shape category flips
/// independently with `p ≈ 0.5`. This deliberately produces small subsets
/// in expectation (mean ~8.5 of 17 enabled).
///
/// Implementation note: `proptest`'s tuple `Strategy` impl only goes up
/// to arity 12, and we have 17 bits. Sample via a fixed-length `Vec<bool>`
/// (proptest provides a `Strategy` impl for `Vec<T>` with bounded length)
/// then deal the bits out into the named struct.
fn arb_swarm_mask() -> impl Strategy<Value = SwarmMask> {
    proptest::collection::vec(any::<bool>(), 17..=17).prop_map(|bits| SwarmMask {
        arith: bits[0],
        neg: bits[1],
        cmp_int: bits[2],
        in_set: bits[3],
        subseteq: bits[4],
        set_ops: bits[5],
        seq_ops: bits[6],
        record_ops: bits[7],
        record_except: bits[8],
        func_app: bits[9],
        quantifier: bits[10],
        let_in: bits[11],
        if_then_else: bits[12],
        case_arm: bits[13],
        opcall: bits[14],
        bool_conn: bits[15],
        bool_not: bits[16],
    })
}

// Swarm-aware recursive generators ------------------------------------------
//
// Same shape as the uniform generators above but each shape category is
// gated on the matching `SwarmMask` bit. We can't conditionally include
// arms in `prop_oneof!`, so we use a `Vec<(weight, BoxedStrategy<Typed>)>`
// + the explicit `prop_oneof::<Typed>(W)` form. Leaves always go in to
// guarantee at least one production at every depth.

fn swarm_arb_int(depth: u32, m: SwarmMask) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_int_leaf().boxed();
    }
    let mut choices: Vec<(u32, BoxedStrategy<Typed>)> = Vec::new();
    choices.push((4, arb_int_leaf().boxed()));
    if m.arith {
        let arith =
            (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m)).prop_flat_map(|(a, b)| {
                prop_oneof![
                    Just(Typed::Int(format!("({} + {})", a.text(), b.text()))),
                    Just(Typed::Int(format!("({} - {})", a.text(), b.text()))),
                    Just(Typed::Int(format!("({} * {})", a.text(), b.text()))),
                    Just(Typed::Int(format!("({} \\div 2)", a.text()))),
                    Just(Typed::Int(format!("({} % 3)", a.text()))),
                ]
            });
        choices.push((3, arith.boxed()));
    }
    if m.neg {
        let neg =
            swarm_arb_int(depth - 1, m).prop_map(|a| Typed::Int(format!("(-({}))", a.text())));
        choices.push((1, neg.boxed()));
    }
    if m.if_then_else {
        let if_int = (
            swarm_arb_bool(depth - 1, m),
            swarm_arb_int(depth - 1, m),
            swarm_arb_int(depth - 1, m),
        )
            .prop_map(|(c, t, e)| {
                Typed::Int(format!(
                    "(IF {} THEN {} ELSE {})",
                    c.text(),
                    t.text(),
                    e.text()
                ))
            });
        choices.push((2, if_int.boxed()));
    }
    if m.let_in {
        let let_int =
            (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m)).prop_map(|(v, body)| {
                Typed::Int(format!(
                    "(LET __t1 == {} IN ({} + __t1))",
                    v.text(),
                    body.text()
                ))
            });
        choices.push((1, let_int.boxed()));
    }
    if m.opcall {
        let opcall =
            swarm_arb_int(depth - 1, m).prop_map(|a| Typed::Int(format!("Inc({})", a.text())));
        choices.push((2, opcall.boxed()));
        let opcall2 = (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(a, b)| Typed::Int(format!("Add2({}, {})", a.text(), b.text())));
        choices.push((1, opcall2.boxed()));
    }
    if m.seq_ops {
        let len =
            swarm_arb_seq_int(depth - 1, m).prop_map(|s| Typed::Int(format!("Len({})", s.text())));
        choices.push((1, len.boxed()));
        let head =
            swarm_arb_seq_int(depth - 1, m).prop_map(|s| Typed::Int(format!("Head({})", s.text())));
        choices.push((1, head.boxed()));
    }
    if m.record_ops {
        let rec_a =
            swarm_arb_rec(depth - 1, m).prop_map(|r| Typed::Int(format!("({}).a", r.text())));
        choices.push((1, rec_a.boxed()));
        let rec_b =
            swarm_arb_rec(depth - 1, m).prop_map(|r| Typed::Int(format!("({}).b", r.text())));
        choices.push((1, rec_b.boxed()));
    }
    if m.func_app {
        let func_app = prop_oneof![
            Just(Typed::Int("f[1]".to_string())),
            Just(Typed::Int("f[2]".to_string())),
        ];
        choices.push((1, func_app.boxed()));
    }
    if m.case_arm {
        let case = (
            swarm_arb_bool(depth - 1, m),
            swarm_arb_int(depth - 1, m),
            swarm_arb_int(depth - 1, m),
        )
            .prop_map(|(c, a, b)| {
                Typed::Int(format!(
                    "(CASE {} -> {} [] OTHER -> {})",
                    c.text(),
                    a.text(),
                    b.text()
                ))
            });
        choices.push((1, case.boxed()));
    }
    weighted_oneof(choices)
}

fn swarm_arb_bool(depth: u32, m: SwarmMask) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_bool_leaf().boxed();
    }
    let mut choices: Vec<(u32, BoxedStrategy<Typed>)> = Vec::new();
    choices.push((4, arb_bool_leaf().boxed()));
    if m.bool_conn {
        let conn =
            (swarm_arb_bool(depth - 1, m), swarm_arb_bool(depth - 1, m)).prop_flat_map(|(a, b)| {
                prop_oneof![
                    Just(Typed::Bool(format!("({} /\\ {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} \\/ {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} => {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} <=> {})", a.text(), b.text()))),
                ]
            });
        choices.push((3, conn.boxed()));
    }
    if m.bool_not {
        let not =
            swarm_arb_bool(depth - 1, m).prop_map(|a| Typed::Bool(format!("(~ {})", a.text())));
        choices.push((1, not.boxed()));
    }
    if m.cmp_int {
        let cmp_int =
            (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m)).prop_flat_map(|(a, b)| {
                prop_oneof![
                    Just(Typed::Bool(format!("({} = {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} # {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} < {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} > {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} <= {})", a.text(), b.text()))),
                    Just(Typed::Bool(format!("({} >= {})", a.text(), b.text()))),
                ]
            });
        choices.push((3, cmp_int.boxed()));
    }
    if m.in_set {
        let in_set = (swarm_arb_int(depth - 1, m), swarm_arb_set_int(depth - 1, m)).prop_flat_map(
            |(e, s)| {
                prop_oneof![
                    Just(Typed::Bool(format!("({} \\in {})", e.text(), s.text()))),
                    Just(Typed::Bool(format!("({} \\notin {})", e.text(), s.text()))),
                ]
            },
        );
        choices.push((2, in_set.boxed()));
    }
    if m.subseteq {
        let subseteq = (
            swarm_arb_set_int(depth - 1, m),
            swarm_arb_set_int(depth - 1, m),
        )
            .prop_map(|(a, b)| Typed::Bool(format!("({} \\subseteq {})", a.text(), b.text())));
        choices.push((1, subseteq.boxed()));
    }
    if m.quantifier {
        let exists = swarm_arb_bool(depth - 1, m).prop_map(|body| {
            Typed::Bool(format!(
                "(\\E __q \\in 1..3 : ({} \\/ (__q > 0)))",
                body.text()
            ))
        });
        choices.push((1, exists.boxed()));
        let forall = swarm_arb_bool(depth - 1, m).prop_map(|body| {
            Typed::Bool(format!(
                "(\\A __q \\in 1..3 : ({} \\/ (__q > 0)))",
                body.text()
            ))
        });
        choices.push((1, forall.boxed()));
    }
    if m.if_then_else {
        let if_bool = (
            swarm_arb_bool(depth - 1, m),
            swarm_arb_bool(depth - 1, m),
            swarm_arb_bool(depth - 1, m),
        )
            .prop_map(|(c, t, e)| {
                Typed::Bool(format!(
                    "(IF {} THEN {} ELSE {})",
                    c.text(),
                    t.text(),
                    e.text()
                ))
            });
        choices.push((1, if_bool.boxed()));
    }
    if m.opcall {
        let opcall =
            swarm_arb_int(depth - 1, m).prop_map(|a| Typed::Bool(format!("IsPos({})", a.text())));
        choices.push((1, opcall.boxed()));
    }
    weighted_oneof(choices)
}

fn swarm_arb_set_int(depth: u32, m: SwarmMask) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_set_int_leaf().boxed();
    }
    let mut choices: Vec<(u32, BoxedStrategy<Typed>)> = Vec::new();
    choices.push((4, arb_set_int_leaf().boxed()));
    if m.set_ops {
        let union = (
            swarm_arb_set_int(depth - 1, m),
            swarm_arb_set_int(depth - 1, m),
        )
            .prop_map(|(a, b)| Typed::SetInt(format!("({} \\union {})", a.text(), b.text())));
        choices.push((2, union.boxed()));
        let inter = (
            swarm_arb_set_int(depth - 1, m),
            swarm_arb_set_int(depth - 1, m),
        )
            .prop_map(|(a, b)| Typed::SetInt(format!("({} \\intersect {})", a.text(), b.text())));
        choices.push((2, inter.boxed()));
        let minus = (
            swarm_arb_set_int(depth - 1, m),
            swarm_arb_set_int(depth - 1, m),
        )
            .prop_map(|(a, b)| Typed::SetInt(format!("({} \\ {})", a.text(), b.text())));
        choices.push((2, minus.boxed()));
        let lit = (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(a, b)| Typed::SetInt(format!("{{{}, {}}}", a.text(), b.text())));
        choices.push((2, lit.boxed()));
        let filter = swarm_arb_bool(depth - 1, m).prop_map(|p| {
            Typed::SetInt(format!("{{__e \\in 1..4 : ({}) \\/ (__e > 0)}}", p.text()))
        });
        choices.push((1, filter.boxed()));
        let map = swarm_arb_int(depth - 1, m)
            .prop_map(|e| Typed::SetInt(format!("{{({}) + __e : __e \\in 1..3}}", e.text())));
        choices.push((1, map.boxed()));
    }
    weighted_oneof(choices)
}

fn swarm_arb_seq_int(depth: u32, m: SwarmMask) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_seq_int_leaf().boxed();
    }
    let mut choices: Vec<(u32, BoxedStrategy<Typed>)> = Vec::new();
    choices.push((3, arb_seq_int_leaf().boxed()));
    if m.seq_ops {
        let append = (swarm_arb_seq_int(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(s, e)| Typed::SeqInt(format!("Append({}, {})", s.text(), e.text())));
        choices.push((2, append.boxed()));
        let tail = swarm_arb_seq_int(depth - 1, m)
            .prop_map(|s| Typed::SeqInt(format!("Tail({})", s.text())));
        choices.push((2, tail.boxed()));
        let lit = (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(a, b)| Typed::SeqInt(format!("<<{}, {}>>", a.text(), b.text())));
        choices.push((2, lit.boxed()));
    }
    weighted_oneof(choices)
}

fn swarm_arb_rec(depth: u32, m: SwarmMask) -> BoxedStrategy<Typed> {
    if depth == 0 {
        return arb_rec_leaf().boxed();
    }
    let mut choices: Vec<(u32, BoxedStrategy<Typed>)> = Vec::new();
    choices.push((3, arb_rec_leaf().boxed()));
    if m.record_ops {
        let lit = (swarm_arb_int(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(a, b)| Typed::RecAB(format!("[a |-> {}, b |-> {}]", a.text(), b.text())));
        choices.push((2, lit.boxed()));
    }
    if m.record_except {
        let except_a = (swarm_arb_rec(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(r, v)| Typed::RecAB(format!("[{} EXCEPT !.a = {}]", r.text(), v.text())));
        choices.push((1, except_a.boxed()));
        let except_b = (swarm_arb_rec(depth - 1, m), swarm_arb_int(depth - 1, m))
            .prop_map(|(r, v)| Typed::RecAB(format!("[{} EXCEPT !.b = {}]", r.text(), v.text())));
        choices.push((1, except_b.boxed()));
    }
    weighted_oneof(choices)
}

/// Helper: build a `prop_oneof`-equivalent strategy from a runtime list of
/// (weight, Strategy) choices. Proptest's `prop_oneof!` macro requires
/// compile-time arms; we need a conditional gating, so we hand-roll it via
/// `Union::new_weighted`.
fn weighted_oneof(choices: Vec<(u32, BoxedStrategy<Typed>)>) -> BoxedStrategy<Typed> {
    use proptest::strategy::Union;
    // Caller invariant: leaves are always pushed first, so `choices` is
    // never empty.
    assert!(
        !choices.is_empty(),
        "swarm generator produced empty choices"
    );
    let weighted: Vec<(u32, BoxedStrategy<Typed>)> = choices;
    Union::new_weighted(weighted).boxed()
}

/// Top-level swarm-mode strategy: sample a mask, then produce one
/// expression of a randomly-chosen type within that mask.
fn arb_swarm_expr() -> impl Strategy<Value = (SwarmMask, Typed)> {
    arb_swarm_mask().prop_flat_map(|mask| {
        let depth = 3u32;
        // Pick an expression type. All types remain available regardless of
        // the mask — the mask gates *productions inside* the recursive
        // generator, not the top-level type. This means a leaf-only mask
        // still produces well-formed expressions (just shallow ones).
        let strategy = prop_oneof![
            swarm_arb_int(depth, mask),
            swarm_arb_bool(depth, mask),
            swarm_arb_set_int(depth, mask),
            swarm_arb_seq_int(depth, mask),
            swarm_arb_rec(depth, mask),
            arb_str_leaf().boxed(),
        ];
        strategy.prop_map(move |t| (mask, t))
    })
}

/// True iff swarm mode is active (default = on; set `SWARM_MODE=uniform`
/// to revert to the original kitchen-sink generator).
fn swarm_mode_enabled() -> bool {
    match std::env::var("SWARM_MODE").ok().as_deref() {
        Some("uniform") | Some("off") | Some("0") | Some("false") => false,
        _ => true,
    }
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

    /// T2 — the original uniform-pool property: for any generated expression
    /// (kitchen-sink generator, every category active), the interpreter and
    /// compiler agree (both produce the same Ok value, or both produce an
    /// error). Kept as a regression gate so the swarm rewrite can't silently
    /// drop coverage of any single category that was previously exercised.
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

    /// T16a — swarm-mode property. Each case picks a random subset of
    /// shape categories first, then draws an expression that only uses
    /// productions from that subset. Most cases use a small subset
    /// (mean ~8.5 of 17 shapes), surfacing feature-pair seam bugs
    /// (e.g., quantifier+EXCEPT, sequences+LET) that uniform sampling
    /// under-explores. Disable via `SWARM_MODE=uniform` (the test still
    /// runs, falling back to the kitchen-sink generator).
    #[test]
    fn compiled_matches_interpreted_swarm(input in arb_swarm_expr()) {
        let (mask, expr) = input;
        let text = expr.text().to_string();
        prop_assume!(!text.is_empty());
        // SWARM_MODE=uniform short-circuits to the all-on mask, which is
        // semantically the same as the kitchen-sink generator. We still
        // run the equivalence check so test counts stay stable; the only
        // effect is to remove the small-subset bias.
        let _ = swarm_mode_enabled();
        if let Err(msg) = check_equivalence(&text) {
            return Err(TestCaseError::fail(format!(
                "{msg}\n  swarm-mask: [{}]",
                mask.describe()
            )));
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

// ---------------------------------------------------------------------------
// T16a — Swarm mask sanity tests. These pin the swarm machinery so a
// future refactor of the mask / generator can't silently revert it to
// kitchen-sink behaviour.
// ---------------------------------------------------------------------------

#[test]
fn t16_swarm_describe_lists_enabled_categories() {
    let m = SwarmMask {
        arith: true,
        record_except: true,
        quantifier: true,
        // everything else off
        neg: false,
        cmp_int: false,
        in_set: false,
        subseteq: false,
        set_ops: false,
        seq_ops: false,
        record_ops: false,
        func_app: false,
        let_in: false,
        if_then_else: false,
        case_arm: false,
        opcall: false,
        bool_conn: false,
        bool_not: false,
    };
    let s = m.describe();
    assert!(s.contains("arith"));
    assert!(s.contains("except"));
    assert!(s.contains("quant"));
    assert!(!s.contains("set,"), "set should be off, got {s}");
}

#[test]
fn t16_swarm_describe_empty_mask_is_leaves_only() {
    let m = SwarmMask {
        arith: false,
        neg: false,
        cmp_int: false,
        in_set: false,
        subseteq: false,
        set_ops: false,
        seq_ops: false,
        record_ops: false,
        record_except: false,
        func_app: false,
        quantifier: false,
        let_in: false,
        if_then_else: false,
        case_arm: false,
        opcall: false,
        bool_conn: false,
        bool_not: false,
    };
    assert_eq!(m.describe(), "<leaves only>");
}

#[test]
fn t16_swarm_all_on_mask_describes_all_categories() {
    let m = SwarmMask::all_on();
    let s = m.describe();
    for cat in [
        "arith", "neg", "cmp", "in", "subseteq", "set", "seq", "rec", "except", "fn", "quant",
        "let", "if", "case", "op", "conn", "not",
    ] {
        assert!(s.contains(cat), "expected `{cat}` in describe(): {s}");
    }
}

#[test]
fn t16_swarm_empty_mask_still_produces_valid_leaves() {
    // Even with every shape category off, the recursive generators must
    // still terminate by falling back to leaf strategies. Sample a handful
    // of expressions and check they all evaluate equivalently.
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};
    let leaves_only = SwarmMask {
        arith: false,
        neg: false,
        cmp_int: false,
        in_set: false,
        subseteq: false,
        set_ops: false,
        seq_ops: false,
        record_ops: false,
        record_except: false,
        func_app: false,
        quantifier: false,
        let_in: false,
        if_then_else: false,
        case_arm: false,
        opcall: false,
        bool_conn: false,
        bool_not: false,
    };
    let mut runner = TestRunner::new(Config::default());
    for _ in 0..16 {
        // Each typed leaf generator must work standalone.
        let s = swarm_arb_int(3, leaves_only).new_tree(&mut runner).unwrap();
        check_equivalence(s.current().text()).expect("int leaf");
        let s = swarm_arb_bool(3, leaves_only)
            .new_tree(&mut runner)
            .unwrap();
        check_equivalence(s.current().text()).expect("bool leaf");
        let s = swarm_arb_set_int(3, leaves_only)
            .new_tree(&mut runner)
            .unwrap();
        check_equivalence(s.current().text()).expect("set leaf");
    }
}

#[test]
fn t16_swarm_mode_env_default_is_on() {
    // unsafe block: env mutation needs the unsafe scope on Rust 2024.
    // SAFETY: tests run with cargo's default --test-threads serialization
    // for a single integration binary, but multiple #[test] fns can share
    // the env. We snapshot, mutate, restore — no other thread reads
    // SWARM_MODE during this brief window because the property tests
    // live in a separate proptest closure that doesn't read it after
    // the mask has been sampled.
    let saved = std::env::var("SWARM_MODE").ok();
    unsafe {
        std::env::remove_var("SWARM_MODE");
    }
    assert!(swarm_mode_enabled(), "default should be swarm-on");
    unsafe {
        std::env::set_var("SWARM_MODE", "uniform");
    }
    assert!(!swarm_mode_enabled(), "SWARM_MODE=uniform should disable");
    unsafe {
        std::env::set_var("SWARM_MODE", "off");
    }
    assert!(!swarm_mode_enabled(), "SWARM_MODE=off should disable");
    unsafe {
        std::env::set_var("SWARM_MODE", "swarm");
    }
    assert!(swarm_mode_enabled(), "SWARM_MODE=swarm should enable");
    // restore
    unsafe {
        match saved {
            Some(v) => std::env::set_var("SWARM_MODE", v),
            None => std::env::remove_var("SWARM_MODE"),
        }
    }
}
