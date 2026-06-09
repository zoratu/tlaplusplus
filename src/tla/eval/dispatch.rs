//! Pre-computed dispatch shape for `eval_expr_inner`.
//!
//! `eval_expr_inner` (in `expr.rs`) is the top-of-grammar precedence cascade.
//! Historically it has been a flat sequence of `let X_parts = split_top_level_*(expr); if X_parts.len() > 1 { ... }`
//! tests, in a precise precedence order. That worked but ran the splitter cascade ~25 times per call.
//!
//! This module introduces a seam: [`Op`] is the structured result of the
//! cascade, and [`classify_op`] runs the cascade once and returns the chosen
//! variant carrying its parts. `eval_expr_inner` now dispatches on the `Op`
//! variant instead of repeating the cascade inline.
//!
//! **No semantic change in this commit.** `classify_op` walks the cascade
//! in the same precedence order as the old inline code and returns the
//! variant the inline code would have entered. The bodies in `eval_expr_inner`
//! consume the parts from the `Op` variant exactly as before.
//!
//! The seam unlocks two follow-ons that this commit does NOT do:
//!
//! 1. **Caching** `classify_op(expr)` keyed by the trimmed expression string,
//!    so the cascade runs once per unique expression rather than once per
//!    evaluation. Most of `eval_expr_inner`'s cost on hot specs is repeated
//!    evaluation of the same predicate text under different binder values
//!    (see PR #80 + the MCBinarySearch profile).
//! 2. **Replacing the cascade body** of `classify_op` with a single
//!    precedence-aware scan over the expression.
//!
//! Both stages plug into this seam without changing `eval_expr_inner`.
//!
//! # Soundness invariants
//!
//! - **Precedence order is load-bearing** (see the `expr.rs` module header
//!   comment about T1.6). The cascade in `classify_op` MUST mirror the order
//!   in the old inline code; reordering changes parse semantics.
//! - The single-disjunct / single-conjunct fallbacks (e.g. `\/ x > 0`
//!   treated as `x > 0`) are now their own `Op` variants
//!   (`OrPrefixSingle` / `AndPrefixSingle`), surfaced when the corresponding
//!   `>1` test fails but the expression starts with the operator.
//! - `Op::DefinedInfix` is ctx-dependent (the operator set lives on
//!   `EvalContext`). When this seam grows a cache (stage 2), the cache key
//!   must include the ctx's defined-infix-operator set, OR `Op::DefinedInfix`
//!   must remain uncached. The simpler choice is the latter: have
//!   `classify_op` skip the defined-infix probe and let the dispatcher run
//!   it as a non-cached fallback before falling through to `Op::Atom`.

use super::{
    split_indented_top_level_boolean, split_once_top_level, split_top_level_additive,
    split_top_level_comparison, split_top_level_keyword, split_top_level_multiplicative,
    split_top_level_range, split_top_level_set_minus, split_top_level_symbol,
};

/// One of the cascade dispatch outcomes from `eval_expr_inner`. Each variant
/// carries the parts the corresponding branch in the old inline code computed
/// via its own `split_top_level_*` call.
///
/// The `String` payloads are owned because the existing branch bodies consume
/// `Vec<String>` (the splitter return shape) and the cache (stage 2) hands
/// out clones of the cached `Op`.
#[derive(Clone)]
pub(super) enum Op {
    /// Indented `/\` conjunction (e.g. `Init`-style). Distinct from the
    /// non-indented `/\` (handled by `And`) because
    /// `split_indented_top_level_boolean` parses differently.
    IndentedAnd(Vec<String>),
    /// Indented `\/` disjunction. See `IndentedAnd`.
    IndentedOr(Vec<String>),
    /// `<=>` (biconditional / iff). Lower precedence than `=>` — split first.
    Iff(Vec<String>),
    /// `=>` (implication).
    Implies(Vec<String>),
    /// `\/` (disjunction, top-level). `>1` parts.
    Or(Vec<String>),
    /// Expression starts with `\/` but had only one part after splitting on
    /// `\/`. The dispatcher strips the leading `\/` and re-evaluates the rest.
    /// (Old inline code: lines 156-164.)
    OrPrefixSingle,
    /// `/\` (conjunction, top-level). `>1` parts.
    And(Vec<String>),
    /// `/\` prefix with only one conjunct — same fallback shape as
    /// `OrPrefixSingle`.
    AndPrefixSingle,
    /// `~p` (negation).
    Not,
    /// `\cdot` (action composition).
    Cdot(Vec<String>),
    /// Comparison operator: `=`, `/=`, `#`, `<`, `<=`, `=<`, `\leq`, `>`,
    /// `>=`, `\geq`, `\in`, `\notin`, `\subseteq`. `split_top_level_comparison`
    /// returns the op as a `&'static str`; we own a `String` here so the
    /// cached `Op` is `'static`.
    Comparison { lhs: String, op: String, rhs: String },
    /// `\union` / `\cup` (set union). `\union` aliases `\cup`; old code tries
    /// `\union` first, then `\cup`; we collapse both into `Union`.
    Union(Vec<String>),
    /// `\intersect` / `\cap` (set intersection). Same alias pattern as `Union`.
    Intersect(Vec<String>),
    /// Set difference (`\` not followed by a known operator). `>1` parts.
    SetMinus(Vec<String>),
    /// Sequence concatenation `\o` / `\circ`.
    Concat(Vec<String>),
    /// Cartesian product `\X` / `\times`.
    Cartesian(Vec<String>),
    /// Function override `@@`.
    FuncOverride(Vec<String>),
    /// Function pair constructor `a :> b`.
    FuncPair { lhs: String, rhs: String },
    /// Bitwise xor `a ^^ b`.
    Xor { lhs: String, rhs: String },
    /// Range `a..b`.
    Range { lhs: String, rhs: String },
    /// Additive `a + b` / `a - b`.
    Additive { lhs: String, op: char, rhs: String },
    /// Multiplicative `a * b` / `a \div b` / `a % b`.
    Multiplicative { lhs: String, op: String, rhs: String },
    /// Exponent `a ^ b`.
    Exp { lhs: String, rhs: String },
    /// No cached cascade variant matched. The dispatcher falls through to the
    /// remaining inline branches: `split_top_level_defined_infix` (ctx-
    /// dependent, deliberately not cached), `UNCHANGED`/`SUBSET`/`UNION`
    /// keyword shortcuts, unary `-`, and atom evaluation.
    Atom,
}

/// Run the cascade once and return the chosen `Op`. Mirrors the order and
/// predicates of the old inline cascade in `eval_expr_inner` exactly.
///
/// Pre-conditions enforced by the caller (`eval_expr_inner`):
/// - `expr` has already been trimmed and had outer parens stripped.
/// - The string-equality early-exits (`TRUE`, `FALSE`, `BOOLEAN`, `@`) and
///   keyword-prefix dispatches (`LET`, `IF`, `CASE`, `\E`, `\A`, `CHOOSE`,
///   `LAMBDA`) have already not matched.
pub(super) fn classify_op(expr: &str) -> Op {
    // ---- indented boolean: /\ / \/ ----
    if expr.starts_with("/\\")
        && let Some(parts) = split_indented_top_level_boolean(expr, "/\\")
        && parts.len() > 1
    {
        return Op::IndentedAnd(parts);
    }
    if expr.starts_with("\\/")
        && let Some(parts) = split_indented_top_level_boolean(expr, "\\/")
        && parts.len() > 1
    {
        return Op::IndentedOr(parts);
    }

    // ---- <=> before => (T1.6 precedence) ----
    let iff_parts = split_top_level_symbol(expr, "<=>");
    if iff_parts.len() > 1 {
        return Op::Iff(iff_parts);
    }

    let implies_parts = split_top_level_symbol(expr, "=>");
    if implies_parts.len() > 1 {
        return Op::Implies(implies_parts);
    }

    // ---- \/ then \/-prefix-single fallback ----
    let or_parts = split_top_level_symbol(expr, "\\/");
    if or_parts.len() > 1 {
        return Op::Or(or_parts);
    }
    if expr.trim().starts_with("\\/") {
        return Op::OrPrefixSingle;
    }

    // ---- /\ then /\-prefix-single fallback ----
    let and_parts = split_top_level_symbol(expr, "/\\");
    if and_parts.len() > 1 {
        return Op::And(and_parts);
    }
    if expr.trim().starts_with("/\\") {
        return Op::AndPrefixSingle;
    }

    // ---- ~ (unary negation) ----
    if expr.starts_with('~') {
        return Op::Not;
    }

    // ---- \cdot ----
    let cdot_parts = split_top_level_keyword(expr, "\\cdot");
    if cdot_parts.len() > 1 {
        return Op::Cdot(cdot_parts);
    }

    // ---- comparison: =, /=, #, <, <=, =<, \leq, >, >=, \geq, \in, \notin, \subseteq ----
    if let Some((lhs, op, rhs)) = split_top_level_comparison(expr) {
        return Op::Comparison {
            lhs: lhs.to_string(),
            op: op.to_string(),
            rhs: rhs.to_string(),
        };
    }

    // Defined-infix is intentionally NOT classified here — it depends on `ctx`
    // (the set of user-defined infix operators registered in scope). The
    // dispatcher in `eval_expr_inner` probes it separately with the live ctx.
    // The cache key (the trimmed expression) wouldn't be sound otherwise.

    // ---- set ops: \union, \cup, \intersect, \cap, set-minus ----
    let union_parts = split_top_level_keyword(expr, "\\union");
    if union_parts.len() > 1 {
        return Op::Union(union_parts);
    }
    let cup_parts = split_top_level_keyword(expr, "\\cup");
    if cup_parts.len() > 1 {
        return Op::Union(cup_parts);
    }
    let intersect_parts = split_top_level_keyword(expr, "\\intersect");
    if intersect_parts.len() > 1 {
        return Op::Intersect(intersect_parts);
    }
    let cap_parts = split_top_level_keyword(expr, "\\cap");
    if cap_parts.len() > 1 {
        return Op::Intersect(cap_parts);
    }
    let minus_parts = split_top_level_set_minus(expr);
    if minus_parts.len() > 1 {
        return Op::SetMinus(minus_parts);
    }

    // ---- sequence concat: \o / \circ ----
    let mut concat_parts = split_top_level_keyword(expr, "\\o");
    if concat_parts.len() == 1 {
        concat_parts = split_top_level_keyword(expr, "\\circ");
    }
    if concat_parts.len() > 1 {
        return Op::Concat(concat_parts);
    }

    // ---- cartesian product: \X / \times ----
    let mut cartesian_parts = split_top_level_keyword(expr, "\\X");
    if cartesian_parts.len() == 1 {
        cartesian_parts = split_top_level_keyword(expr, "\\times");
    }
    if cartesian_parts.len() > 1 {
        return Op::Cartesian(cartesian_parts);
    }

    // ---- @@ (function override) ----
    let func_override_parts = split_top_level_symbol(expr, "@@");
    if func_override_parts.len() > 1 {
        return Op::FuncOverride(func_override_parts);
    }

    // ---- :> (function pair) ----
    if let Some((lhs, rhs)) = split_once_top_level(expr, ":>") {
        return Op::FuncPair {
            lhs: lhs.trim().to_string(),
            rhs: rhs.trim().to_string(),
        };
    }

    // ---- ^^ (bitwise xor) ----
    if let Some((lhs, rhs)) = split_once_top_level(expr, "^^") {
        return Op::Xor {
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        };
    }

    // ---- .. (range) ----
    if let Some((lhs, rhs)) = split_top_level_range(expr) {
        return Op::Range {
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        };
    }

    // ---- additive: + / - ----
    if let Some((lhs, op, rhs)) = split_top_level_additive(expr) {
        return Op::Additive {
            lhs: lhs.to_string(),
            op,
            rhs: rhs.to_string(),
        };
    }

    // ---- multiplicative: *, \div, % ----
    if let Some((lhs, op, rhs)) = split_top_level_multiplicative(expr) {
        return Op::Multiplicative {
            lhs: lhs.to_string(),
            op: op.to_string(),
            rhs: rhs.to_string(),
        };
    }

    // ---- ^ (exponent) ----
    if let Some((lhs, rhs)) = split_once_top_level(expr, "^") {
        return Op::Exp {
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        };
    }

    // No cached cascade match — caller handles defined-infix, UNCHANGED,
    // SUBSET, UNION keyword, unary -, and atom evaluation.
    Op::Atom
}

// =============================================================================
// Stage 2: thread-local cache for `classify_op` results.
// =============================================================================
//
// `eval_expr_inner` is called millions of times per spec on the same predicate
// strings (e.g. inner bodies of `\A i \in 1..n: ...` loops). Without caching,
// `classify_op` re-runs the splitter cascade for each call. With a cache, the
// cascade runs once per unique trimmed expression.
//
// The cache is THREAD-LOCAL so it needs no locking. Each worker thread warms
// its own cache; under heavy parallelism this duplicates the cache across
// threads but avoids contention. Capacity is bounded (`CACHE_CAP`) so memory
// stays predictable; on overflow we simply clear and rebuild (good enough for
// the workload — eval expressions reach a steady state quickly).
//
// Cache key: the trimmed expression `&str` (cloned to owned `String` on miss).
// Cache value: cloned `Op` (the variant data is small: a `Vec<String>` of
// already-trimmed sub-expressions, plus the unit-tag variants).
//
// Correctness: `classify_op` is a pure function of the trimmed expression
// (see Soundness invariants in the module docs). The only non-pure dispatch
// step in `eval_expr_inner` is `split_top_level_defined_infix(expr, ctx)`,
// which is NOT part of `Op` — it lives downstream in `eval_expr_inner`'s
// remaining inline cascade. So caching `Op` is sound.

const CACHE_CAP: usize = 8192;

thread_local! {
    static OP_CACHE: std::cell::RefCell<std::collections::HashMap<String, Op>> =
        std::cell::RefCell::new(std::collections::HashMap::with_capacity(CACHE_CAP));
}

/// Cached entrypoint mirroring `classify_op`. Looks up the trimmed `expr` in
/// the thread-local cache; on miss, runs `classify_op` and inserts the result.
/// Returns a cloned `Op`.
pub(super) fn classify_op_cached(expr: &str) -> Op {
    OP_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(op) = cache.get(expr) {
            return op.clone();
        }
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        let op = classify_op(expr);
        cache.insert(expr.to_string(), op.clone());
        op
    })
}

/// True if `op` is one of the "low" cascade variants (set ops, sequence
/// concat, cartesian, function override, :>, ^^, range, additive,
/// multiplicative, exp). Used by the dispatcher to inject the ctx-dependent
/// `split_top_level_defined_infix` probe between the high (boolean +
/// comparison) and low (algebraic) tiers — matching the original cascade's
/// `Comparison → DefinedInfix → SetOps → ...` precedence order.
pub(super) fn is_low_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Union(_)
            | Op::Intersect(_)
            | Op::SetMinus(_)
            | Op::Concat(_)
            | Op::Cartesian(_)
            | Op::FuncOverride(_)
            | Op::FuncPair { .. }
            | Op::Xor { .. }
            | Op::Range { .. }
            | Op::Additive { .. }
            | Op::Multiplicative { .. }
            | Op::Exp { .. }
    )
}
