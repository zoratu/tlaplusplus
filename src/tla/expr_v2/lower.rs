//! Lower the `expr_v2` structural [`Expr`] AST to the existing [`CompiledExpr`].
//!
//! Only the STRUCTURE (junctions, `=>`/`<=>`, quantifiers, `LET`, `IF`, and the
//! Pratt-level binary/unary operators) is owned by v2. Every opaque `Atom` leaf
//! is lowered by delegating to the *existing* `compile_expr`, so leaf semantics
//! are byte-for-byte identical to the old parser.

use super::ast::*;
// NOTE: Atom leaves lower via `compile_expr_v1` (the original string-based
// compiler), NOT the public `compile_expr` wrapper. Routing leaves back through
// the wrapper would re-enter v2 (a pure leaf parses as a single Atom) and
// recurse forever when the `TLAPLUS_EXPR_PARSER=v2` flag is on. Using v1 also
// guarantees leaf semantics are byte-for-byte identical to the old parser.
use crate::tla::compiled_expr::{compile_expr_v1, CompiledExpr};

/// Lower a whole AST to a `CompiledExpr`.
pub fn lower(expr: &Expr) -> CompiledExpr {
    match expr {
        Expr::Atom { text, .. } => compile_expr_v1(text),

        Expr::Paren { inner, .. } => lower(inner),

        Expr::Not { operand, .. } => CompiledExpr::Not(Box::new(lower(operand))),

        Expr::Junction { op, items, .. } => {
            let lowered: Vec<CompiledExpr> = items.iter().map(lower).collect();
            match op {
                JunctionOp::And => CompiledExpr::And(lowered),
                JunctionOp::Or => CompiledExpr::Or(lowered),
            }
        }

        Expr::Binary { op, lhs, rhs, .. } => lower_binary(*op, lhs, rhs),

        Expr::Quant { kind, bounds, body, .. } => lower_quant(*kind, bounds, body),

        Expr::Let { defs, body, .. } => lower_let(defs, body),

        Expr::If { cond, then_, else_, .. } => CompiledExpr::If {
            cond: Box::new(lower(cond)),
            then_branch: Box::new(lower(then_)),
            else_branch: Box::new(lower(else_)),
        },
    }
}

fn lower_binary(op: BinOp, lhs: &Expr, rhs: &Expr) -> CompiledExpr {
    let l = Box::new(lower(lhs));
    let r = Box::new(lower(rhs));
    match op {
        BinOp::Implies => CompiledExpr::Implies(l, r),
        BinOp::Iff => CompiledExpr::Iff(l, r),
        BinOp::Eq => CompiledExpr::Eq(l, r),
        BinOp::Neq => CompiledExpr::Neq(l, r),
        BinOp::Lt => CompiledExpr::Lt(l, r),
        BinOp::Le => CompiledExpr::Le(l, r),
        BinOp::Gt => CompiledExpr::Gt(l, r),
        BinOp::Ge => CompiledExpr::Ge(l, r),
        BinOp::In => CompiledExpr::In(l, r),
        BinOp::NotIn => CompiledExpr::NotIn(l, r),
        BinOp::Add => CompiledExpr::Add(l, r),
        BinOp::Sub => CompiledExpr::Sub(l, r),
        BinOp::Mul => CompiledExpr::Mul(l, r),
        BinOp::Div => CompiledExpr::Div(l, r),
        BinOp::Mod => CompiledExpr::Mod(l, r),
        BinOp::Pow => CompiledExpr::Pow(l, r),
        BinOp::Union => CompiledExpr::Union(l, r),
        BinOp::Intersect => CompiledExpr::Intersect(l, r),
        BinOp::SetMinus => CompiledExpr::SetMinus(l, r),
        BinOp::Concat => CompiledExpr::Concat(l, r),
    }
}

/// Lower a quantifier, expanding multi-variable / multi-bound heads into nested
/// single-variable `Forall`/`Exists` (which is how `CompiledExpr` models them).
/// `\A x, y \in S : B`  ->  `\A x \in S : \A y \in S : B`.
/// `\A x \in S, y \in T : B`  ->  `\A x \in S : \A y \in T : B`.
fn lower_quant(kind: QuantKind, bounds: &[QuantBound], body: &Expr) -> CompiledExpr {
    // Flatten to a list of (var, domain) pairs, preserving order.
    let mut pairs: Vec<(String, CompiledExpr)> = Vec::new();
    for b in bounds {
        let dom = lower_domain(&b.domain);
        for v in &b.vars {
            pairs.push((v.clone(), dom.clone()));
        }
    }
    let mut acc = lower(body);
    for (v, dom) in pairs.into_iter().rev() {
        acc = match kind {
            QuantKind::Forall => CompiledExpr::Forall {
                var: v,
                domain: Box::new(dom),
                body: Box::new(acc),
            },
            QuantKind::Exists => CompiledExpr::Exists {
                var: v,
                domain: Box::new(dom),
                body: Box::new(acc),
            },
        };
    }
    acc
}

/// Domain lowering: an unbounded quantifier's placeholder domain lowers to an
/// `Unparsed("")` sentinel (matches the old parser's handling of bare `\A x :`).
fn lower_domain(domain: &Expr) -> CompiledExpr {
    if let Expr::Atom { text, .. } = domain {
        if text == "__UNBOUNDED__" {
            return CompiledExpr::Unparsed(String::new());
        }
    }
    lower(domain)
}

fn lower_let(defs: &[LetDef], body: &Expr) -> CompiledExpr {
    // Fix #3: parameterized operator defs (`Op(a,b) == ..`) and function defs
    // (`f[x] == ..`) are REJECTED at parse time (see `parser::parse_let`), so
    // every def reaching here is a plain `name == value` that `CompiledExpr::Let`
    // (whose bindings are just `(name, value)`) can represent faithfully. Lower
    // each value structurally.
    let bindings = defs
        .iter()
        .map(|d| {
            debug_assert!(
                d.params.is_empty() && d.func_args.is_empty(),
                "parameterized/function LET should have been rejected in the parser"
            );
            (d.name.clone(), lower(&d.value))
        })
        .collect();
    CompiledExpr::Let {
        bindings,
        body: Box::new(lower(body)),
    }
}
