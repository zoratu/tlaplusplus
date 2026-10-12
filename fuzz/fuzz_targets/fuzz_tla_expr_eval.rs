#![no_main]

//! Structure-aware fuzzer for the TLA+ expression **parser + evaluator**
//! pipeline (`eval_expr`), as opposed to the byte-level `fuzz_tla_module`
//! target.
//!
//! ## Why a second, structured target
//!
//! `fuzz_tla_module` feeds raw bytes to `parse_tla_module_text`. That is great
//! for lexer/parser robustness, but random UTF-8 almost never forms a
//! well-typed expression, so the coverage-guided fuzzer spends nearly all its
//! budget bouncing off the lexer and rarely reaches the *evaluator* — the part
//! that does set algebra, quantifier expansion, function application, record
//! projection, arithmetic, and `IF/THEN/ELSE`. This target closes that gap: it
//! GENERATES syntactically-plausible TLA+ expressions from a small grammar
//! (Bug Bash 2026 "structure-aware fuzzing" strategy — a typed `arbitrary`
//! generator rendered to source), so the fuzzer explores deep evaluator paths
//! instead of the lexer boundary.
//!
//! ## Contracts checked
//!
//!   1. No panic / no abort / no unwind across either the string→AST parse
//!      (via `eval_expr`, which parses then evaluates) or the evaluation
//!      itself. `eval_expr` returning `Err` is a *normal* outcome (type error,
//!      unknown name, budget exhausted) and is fine — only a panic is a bug.
//!   2. Termination. A generated expression could otherwise drive an expensive
//!      set comprehension (`{ f[x] : x \in BigSet }`) or a deep quantifier
//!      nest; a per-evaluation **eval budget** (`set_active_eval_budget`) caps
//!      the number of evaluation steps so a hostile/large expression returns
//!      `Err(budget exceeded)` rather than hanging the fuzzer. This exercises
//!      the same budget path the model checker uses for untrusted specs.
//!
//! The state is a small fixed environment (`x`, `y` ints, `S` a set, `r` a
//! record) so generated identifiers frequently resolve and evaluation proceeds
//! past name lookup into real computation.

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use tlaplusplus::tla::hashed_arc::HashedArc;
use tlaplusplus::tla::{
    EvalContext, TlaValue, eval_expr, restore_eval_budget, set_active_eval_budget, tla_state,
};

/// A generated TLA+ value/predicate expression. Rendered to source and fed to
/// `eval_expr`. Kept close to the subset the evaluator supports so most
/// generated strings reach evaluation rather than erroring at parse.
#[derive(Arbitrary, Debug)]
enum TExpr {
    // --- leaves (reference the fixed environment, below) ---
    /// A small integer literal.
    IntLit(u8),
    /// `TRUE` / `FALSE`.
    BoolLit(bool),
    /// One of the in-scope variable names: x, y, S, r.
    VarRef(VarName),

    // --- arithmetic / comparison ---
    Add(Box<TExpr>, Box<TExpr>),
    Sub(Box<TExpr>, Box<TExpr>),
    Mul(Box<TExpr>, Box<TExpr>),
    Lt(Box<TExpr>, Box<TExpr>),
    Eq(Box<TExpr>, Box<TExpr>),

    // --- boolean ---
    And(Box<TExpr>, Box<TExpr>),
    Or(Box<TExpr>, Box<TExpr>),
    Not(Box<TExpr>),
    Implies(Box<TExpr>, Box<TExpr>),
    IfThenElse(Box<TExpr>, Box<TExpr>, Box<TExpr>),

    // --- sets ---
    SetEnum(Vec<TExpr>),
    Union(Box<TExpr>, Box<TExpr>),
    Inter(Box<TExpr>, Box<TExpr>),
    In(Box<TExpr>, Box<TExpr>),
    Subset(Box<TExpr>, Box<TExpr>),
    Range(Box<TExpr>, Box<TExpr>),
    Cardinality(Box<TExpr>),
    /// `{ e : v \in set }` — a comprehension that can be made expensive, so a
    /// good stress on the eval budget.
    SetMap(Box<TExpr>, Box<TExpr>),
    /// `{ v \in set : pred }` — filtered comprehension.
    SetFilter(Box<TExpr>, Box<TExpr>),

    // --- quantifiers ---
    Exists(Box<TExpr>, Box<TExpr>),
    Forall(Box<TExpr>, Box<TExpr>),

    // --- records / functions ---
    /// `r.f` — record field projection.
    RecField,
    /// `[a |-> e1, b |-> e2]` — record constructor.
    RecCons(Box<TExpr>, Box<TExpr>),
    /// `f[e]` where f is the record r used as a function (exercises the
    /// function-application path, usually erroring — which is fine).
    FnApply(Box<TExpr>),
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum VarName {
    X,
    Y,
    S,
    R,
}

impl VarName {
    fn as_str(self) -> &'static str {
        match self {
            VarName::X => "x",
            VarName::Y => "y",
            VarName::S => "S",
            VarName::R => "r",
        }
    }
}

impl TExpr {
    /// Render to a TLA+ source string, bounding recursion so a deeply nested
    /// `Arbitrary` value degrades to a leaf rather than overflowing the
    /// generator's own stack (a generator-side overflow would be a false
    /// positive). All comprehensions/quantifiers bind the same name `i`; nested
    /// ones simply shadow, which is legal TLA+ and keeps the generator simple.
    fn render(&self, depth: u32, out: &mut String) {
        if depth == 0 {
            out.push('1');
            return;
        }
        let d = depth - 1;
        match self {
            TExpr::IntLit(n) => out.push_str(&(*n % 10).to_string()),
            TExpr::BoolLit(b) => out.push_str(if *b { "TRUE" } else { "FALSE" }),
            TExpr::VarRef(v) => out.push_str(v.as_str()),
            TExpr::Add(a, b) => bin(a, "+", b, d, out),
            TExpr::Sub(a, b) => bin(a, "-", b, d, out),
            TExpr::Mul(a, b) => bin(a, "*", b, d, out),
            TExpr::Lt(a, b) => bin(a, "<", b, d, out),
            TExpr::Eq(a, b) => bin(a, "=", b, d, out),
            TExpr::And(a, b) => bin(a, "/\\", b, d, out),
            TExpr::Or(a, b) => bin(a, "\\/", b, d, out),
            TExpr::Implies(a, b) => bin(a, "=>", b, d, out),
            TExpr::Not(a) => {
                out.push_str("~(");
                a.render(d, out);
                out.push(')');
            }
            TExpr::IfThenElse(c, t, e) => {
                out.push_str("IF ");
                c.render(d, out);
                out.push_str(" THEN ");
                t.render(d, out);
                out.push_str(" ELSE ");
                e.render(d, out);
            }
            TExpr::SetEnum(items) => {
                out.push('{');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    it.render(d, out);
                }
                out.push('}');
            }
            TExpr::Union(a, b) => bin(a, "\\union", b, d, out),
            TExpr::Inter(a, b) => bin(a, "\\intersect", b, d, out),
            TExpr::In(a, b) => bin(a, "\\in", b, d, out),
            TExpr::Subset(a, b) => bin(a, "\\subseteq", b, d, out),
            TExpr::Range(a, b) => bin(a, "..", b, d, out),
            TExpr::Cardinality(a) => {
                out.push_str("Cardinality(");
                a.render(d, out);
                out.push(')');
            }
            TExpr::SetMap(elem, set) => {
                out.push_str("{ (");
                elem.render(d, out);
                out.push_str(") : i \\in ");
                set.render(d, out);
                out.push_str(" }");
            }
            TExpr::SetFilter(set, pred) => {
                out.push_str("{ i \\in ");
                set.render(d, out);
                out.push_str(" : ");
                pred.render(d, out);
                out.push_str(" }");
            }
            TExpr::Exists(set, pred) => {
                out.push_str("\\E i \\in ");
                set.render(d, out);
                out.push_str(" : ");
                pred.render(d, out);
            }
            TExpr::Forall(set, pred) => {
                out.push_str("\\A i \\in ");
                set.render(d, out);
                out.push_str(" : ");
                pred.render(d, out);
            }
            TExpr::RecField => out.push_str("r.a"),
            TExpr::RecCons(a, b) => {
                out.push_str("[a |-> ");
                a.render(d, out);
                out.push_str(", b |-> ");
                b.render(d, out);
                out.push(']');
            }
            TExpr::FnApply(idx) => {
                out.push_str("r[");
                idx.render(d, out);
                out.push(']');
            }
        }
    }
}

fn bin(a: &TExpr, op: &str, b: &TExpr, depth: u32, out: &mut String) {
    out.push('(');
    a.render(depth, out);
    out.push(' ');
    out.push_str(op);
    out.push(' ');
    b.render(depth, out);
    out.push(')');
}

/// Cap on renderer recursion depth. Deep enough to build interesting nested
/// comprehensions/quantifiers, shallow enough to keep the rendered string and
/// the generator stack bounded.
const MAX_DEPTH: u32 = 10;

/// Per-evaluation step budget. Large enough that typical expressions evaluate
/// to completion, small enough that a pathological comprehension over a range
/// (e.g. `Cardinality({ i*i : i \in 1..9 })` nested deep) returns
/// `Err(budget)` promptly instead of hanging the fuzzer.
const EVAL_BUDGET: usize = 200_000;

fuzz_target!(|expr: TExpr| {
    let mut src = String::new();
    expr.render(MAX_DEPTH, &mut src);

    // Fixed small environment so generated identifiers resolve and evaluation
    // proceeds into real computation rather than stopping at name lookup.
    // S = {1, 2, 3, 4} as a small set for comprehensions/quantifiers.
    let s_set = TlaValue::Set(HashedArc::new(BTreeSet::from([
        TlaValue::Int(1),
        TlaValue::Int(2),
        TlaValue::Int(3),
        TlaValue::Int(4),
    ])));
    // r = [a |-> 1, b |-> 2]: a record, also usable as a function.
    let r_rec = TlaValue::Record(HashedArc::new(BTreeMap::from([
        ("a".to_string(), TlaValue::Int(1)),
        ("b".to_string(), TlaValue::Int(2)),
    ])));
    let state = tla_state([
        ("x", TlaValue::Int(3)),
        ("y", TlaValue::Int(5)),
        ("S", s_set),
        ("r", r_rec),
    ]);

    // Bound evaluation cost BEFORE building the context: `EvalContext::new`
    // captures the active thread-local budget at construction time, so the
    // budget must be installed first for the context to inherit it. Restore the
    // previous budget afterward (there is none here, but this is the correct
    // scoping discipline and mirrors the model checker's usage).
    let prev = set_active_eval_budget(EVAL_BUDGET);
    let ctx = EvalContext::new(&state);
    // Panic is the only failure mode we assert against; Err (type error,
    // unknown name, budget exceeded) is an expected, acceptable outcome.
    let _ = eval_expr(&src, &ctx);
    restore_eval_budget(prev);
});
