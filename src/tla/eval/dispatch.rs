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

/// Phase 3: layout-aware boolean/junction grouping via the `expr_v2` AST.
///
/// This is the interpreted-path twin of Phase 2's compiled-path routing. It
/// replaces ONLY the top-level boolean grouping decision (`/\`/`\/`/`=>`/`<=>`)
/// that the string splitter cascade in `classify_op` performs — the same
/// string-splitting disease Phase 2 fixed for the compiled path. Everything
/// else (comparison, arithmetic, set ops, ...) still runs through the existing
/// cascade unchanged.
///
/// Returns:
/// - `None` if v2 is disabled (`TLAPLUS_EXPR_PARSER=v1|off`) or the v2 parse
///   errors → the existing splitter cascade runs, unchanged (always safe).
/// - `None` if the TOP node is anything other than an And/Or junction with
///   `>= 2` items or a top-level `=>`/`<=>` binary → fall through to the
///   existing cascade (which handles comparison/arith/set/quant/etc., or the
///   prefix-single fallbacks for degenerate 1-item junctions).
/// - `Some(Op)` for the boolean shapes we take over.
///
/// # Semantic-mapping evidence (read `eval_expr.rs` arms ~130-197)
/// - `Op::IndentedAnd(parts)` and `Op::And(parts)` evaluate IDENTICALLY: both
///   fold `&&` over the parts, short-circuiting on the first `false`. Likewise
///   `Op::IndentedOr` and `Op::Or` both fold `||`, short-circuiting on `true`.
///   So mapping every And-junction → `IndentedAnd` and Or-junction →
///   `IndentedOr` is behaviour-preserving; we pick the `Indented*` variants
///   because a v2 `Junction` is precisely the layout-grouped bulleted list.
/// - `Op::Implies(vec![lhs, rhs])`: `eval_implies_parts` on a 2-element slice
///   evaluates lhs; if false returns true, else evaluates rhs — exactly
///   `lhs => rhs`. For a right-assoc chain `A => B => C`, v2 gives
///   `Implies(A, Implies(B, C))`, so `rhs = "B => C"` and the recursive
///   re-classify re-parses it correctly. We do NOT flatten.
/// - `Op::Iff(vec![lhs, rhs])`: the arm folds pairwise `==`; on 2 parts that is
///   `lhs == rhs`, i.e. the biconditional.
///
/// # Span-offset invariant
/// v2 spans are byte offsets 0-based into the exact `&str` passed to
/// `expr_v2::parse_ast` (the lexer starts `pos = 0` on `expr.as_bytes()`; see
/// `expr_v2::lexer::Lexer::new`). So `expr[child.span.start .. child.span.end]`
/// is the original source substring for that child. A debug assertion below
/// guards the bounds.
///
/// # Cache soundness
/// This is a pure function of `expr` (`v2_enabled()` is process-constant), so
/// the thread-local `classify_op` cache keyed on the trimmed `expr` stays sound.
fn classify_boolean_v2(expr: &str) -> Option<Op> {
    use crate::tla::expr_v2::{self, ast};

    if !expr_v2::v2_enabled() {
        return None;
    }
    let tree = expr_v2::parse_ast(expr).ok()?;

    // Take the original source substring for a child expr. Spans are 0-based
    // byte offsets into `expr` (see the doc-comment invariant above).
    let slice = |child: &ast::Expr| -> Option<String> {
        let sp = child.span();
        debug_assert!(
            sp.start <= sp.end && sp.end <= expr.len(),
            "expr_v2 span out of bounds for classify_boolean_v2: start={}, end={}, len={}",
            sp.start,
            sp.end,
            expr.len()
        );
        // Defensive (release): if the span is inverted, somehow out of bounds,
        // or not a char boundary, bail to the fallback rather than panic-slicing.
        // The `sp.start > sp.end` check mirrors the debug_assert above so an
        // inverted span (even with both endpoints on char boundaries) bails to
        // `None`→fallback instead of panicking on `expr[sp.start..sp.end]`.
        if sp.start > sp.end
            || sp.end > expr.len()
            || !expr.is_char_boundary(sp.start)
            || !expr.is_char_boundary(sp.end)
        {
            return None;
        }
        Some(expr[sp.start..sp.end].trim().to_string())
    };

    match &tree {
        ast::Expr::Junction { op, items, .. } => {
            // Degenerate 1-item (or empty) junctions: fall through to the
            // existing `*PrefixSingle` fallback, which matches the historical
            // semantics exactly. Only take over `>= 2`-item lists.
            if items.len() < 2 {
                return None;
            }
            let mut parts = Vec::with_capacity(items.len());
            for it in items {
                parts.push(slice(it)?);
            }
            match op {
                ast::JunctionOp::And => Some(Op::IndentedAnd(parts)),
                ast::JunctionOp::Or => Some(Op::IndentedOr(parts)),
            }
        }
        ast::Expr::Binary { op: ast::BinOp::Implies, lhs, rhs, .. } => {
            Some(Op::Implies(vec![slice(lhs)?, slice(rhs)?]))
        }
        ast::Expr::Binary { op: ast::BinOp::Iff, lhs, rhs, .. } => {
            Some(Op::Iff(vec![slice(lhs)?, slice(rhs)?]))
        }
        // Any other top node (comparison, arithmetic, set-op, Quant, Let, If,
        // Not, Paren, containers, Atom) → fall through to the existing cascade.
        _ => None,
    }
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
    // ---- ~ applied to a bulleted list: `~ /\ A /\ B` / `~ \/ A \/ B` ----
    //
    // In TLA+, `~` binds tighter than `/\`/`\/`/`=>`, so `~A /\ B` is
    // `(~A) /\ B` and is correctly handled by the And/Or splits below (the
    // first conjunct `~A` is a well-formed operand). The ONE case those
    // splits get wrong is when the `~` prefixes a *bulleted conjunction/
    // disjunction* — `~ /\ A /\ B`. There the negation applies to the whole
    // list (`~(A /\ B)`), but `split_top_level_symbol(_, "/\\")` slices off a
    // bare `~` as the first "conjunct", which then evaluates to the empty
    // expression and errors. Intercept that shape here and route it to
    // `Op::Not` so the operand `/\ A /\ B` is evaluated as one bulleted list.
    // (TCommit's `TCConsistent == \A rm1,rm2 : ~ /\ ... /\ ...` is exactly
    // this shape and previously false-violated the initial state.)
    if let Some(after_tilde) = expr.strip_prefix('~') {
        let t = after_tilde.trim_start();
        if t.starts_with("/\\") || t.starts_with("\\/") {
            return Op::Not;
        }
    }

    // ---- Phase 3: layout-aware boolean grouping via expr_v2 ----
    //
    // Take over ONLY the top-level boolean grouping decision (And/Or junctions,
    // `=>`, `<=>`) using the layout-aware v2 AST. This fixes the interpreted-
    // path twin of the Phase-2.1 MCPaxos bug where the string splitter let an
    // indented `/\` swallow a lower-precedence `=>`. Returns `None` (→ the
    // splitter cascade below runs unchanged) whenever v2 is disabled, the parse
    // errors, or the top node is not a boolean shape we own. See
    // `classify_boolean_v2` for the full mapping + semantic evidence.
    if let Some(op) = classify_boolean_v2(expr) {
        return op;
    }

    // ---- indented boolean: /\ / \/ (fallback: v2 disabled / parse err / non-boolean top) ----
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

// Use ahash for the cache's HashMap because std's default SipHash is ~3x
// slower per lookup. ahash::AHasher::default() uses a per-process random
// seed — fine here because this cache is thread-local scratch space that
// never escapes the process (so the seed doesn't need to be deterministic
// across processes; the determinism rule applies only to `Model::fingerprint`
// — see `crate::model::fingerprint_hasher`).
// Cache values are wrapped in `Rc` so a cache hit is a refcount bump rather
// than a deep clone of the `Op`'s owned `Vec<String>` / `String` payloads.
// The dispatcher in `expr.rs` consumes the `Op` purely by reference (every
// arm passes `&str` slices to `eval_expr_inner`), so it never needs to own
// the strings — borrowing through the `Rc` is sufficient. `Rc` (not `Arc`)
// because `OP_CACHE` is thread-local and the returned handle never crosses
// threads.
type CacheMap = std::collections::HashMap<String, std::rc::Rc<Op>, ahash::RandomState>;

thread_local! {
    static OP_CACHE: std::cell::RefCell<CacheMap> = std::cell::RefCell::new(
        CacheMap::with_capacity_and_hasher(CACHE_CAP, ahash::RandomState::new())
    );
}

/// Cached entrypoint mirroring `classify_op`. Looks up the trimmed `expr` in
/// the thread-local cache; on miss, runs `classify_op` and inserts the result.
/// Returns a shared `Rc<Op>` — cloning it on a hit is a refcount bump, not a
/// reallocation of the cascade parts.
pub(super) fn classify_op_cached(expr: &str) -> std::rc::Rc<Op> {
    OP_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(op) = cache.get(expr) {
            return std::rc::Rc::clone(op);
        }
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        let op = std::rc::Rc::new(classify_op(expr));
        cache.insert(expr.to_string(), std::rc::Rc::clone(&op));
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

#[cfg(test)]
mod phase3_boolean_tests {
    use super::*;

    // Helper: assert classify_boolean_v2 returns the expected part-slices for a
    // given boolean top shape. Also implicitly exercises the span-offset
    // invariant (a wrong offset would yield wrong / out-of-bounds slices, and
    // the debug_assert in `slice` would fire under `cargo test`).
    fn parts_of(op: &Op) -> Vec<String> {
        match op {
            Op::IndentedAnd(p) | Op::IndentedOr(p) | Op::Implies(p) | Op::Iff(p) => p.clone(),
            other => panic!("unexpected op variant: {:?}", variant_name(other)),
        }
    }
    fn variant_name(op: &Op) -> &'static str {
        match op {
            Op::IndentedAnd(_) => "IndentedAnd",
            Op::IndentedOr(_) => "IndentedOr",
            Op::Implies(_) => "Implies",
            Op::Iff(_) => "Iff",
            _ => "other",
        }
    }

    #[test]
    fn and_junction_maps_to_indented_and() {
        let e = "/\\ a > 0\n/\\ b > 1";
        let op = classify_boolean_v2(e).expect("should classify as boolean");
        assert!(matches!(op, Op::IndentedAnd(_)));
        assert_eq!(parts_of(&op), vec!["a > 0", "b > 1"]);
    }

    #[test]
    fn or_junction_maps_to_indented_or() {
        let e = "\\/ a > 0\n\\/ b > 1";
        let op = classify_boolean_v2(e).expect("should classify as boolean");
        assert!(matches!(op, Op::IndentedOr(_)));
        assert_eq!(parts_of(&op), vec!["a > 0", "b > 1"]);
    }

    #[test]
    fn implies_maps_to_two_element_implies() {
        let e = "a > 0 => b > 1";
        let op = classify_boolean_v2(e).expect("should classify as implies");
        assert!(matches!(op, Op::Implies(_)));
        assert_eq!(parts_of(&op), vec!["a > 0", "b > 1"]);
    }

    #[test]
    fn implies_chain_is_right_assoc_two_element() {
        // A => B => C  =>  Implies(A, "B => C") — recursion re-parses the rhs.
        let e = "a => b => c";
        let op = classify_boolean_v2(e).expect("should classify as implies");
        let p = parts_of(&op);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], "a");
        assert_eq!(p[1], "b => c");
    }

    #[test]
    fn iff_maps_to_two_element_iff() {
        let e = "a > 0 <=> b > 1";
        let op = classify_boolean_v2(e).expect("should classify as iff");
        assert!(matches!(op, Op::Iff(_)));
        assert_eq!(parts_of(&op), vec!["a > 0", "b > 1"]);
    }

    #[test]
    fn implies_looser_than_indented_and() {
        // The interpreted-path twin of the MCPaxos bug: a bulleted /\ under a
        // quantifier body followed by => must group as Implies(And[...], rhs),
        // NOT let the /\ swallow the =>. v2 gives Implies at top.
        let e = "/\\ p\n/\\ q\n=> r";
        // Depending on layout this parses as either Implies at top or None;
        // the key assertion is we NEVER return an IndentedAnd that swallows =>.
        if let Some(op) = classify_boolean_v2(e) {
            assert!(
                matches!(op, Op::Implies(_)),
                "must not let /\\ swallow => ; got {}",
                variant_name(&op)
            );
        }
    }

    #[test]
    fn single_item_junction_falls_through() {
        // `\/ X` single-bullet → None so the OrPrefixSingle fallback handles it,
        // preserving historical semantics.
        let e = "\\/ a > 0";
        assert!(classify_boolean_v2(e).is_none());
    }

    #[test]
    fn non_boolean_top_falls_through() {
        // Comparison / arithmetic / quantifier / set tops → None (existing
        // cascade handles them).
        assert!(classify_boolean_v2("a > 0").is_none());
        assert!(classify_boolean_v2("a + b").is_none());
        assert!(classify_boolean_v2("\\A x \\in S : P(x)").is_none());
        assert!(classify_boolean_v2("{1, 2, 3}").is_none());
        assert!(classify_boolean_v2("f[x]").is_none());
    }

    #[test]
    fn slices_are_valid_substrings_of_input() {
        // Span-offset invariant: every returned part is a trimmed substring of
        // the original input (guards against off-by-one span bugs).
        let e = "/\\ foo = bar\n/\\ baz \\in Nat";
        let op = classify_boolean_v2(e).unwrap();
        for part in parts_of(&op) {
            assert!(
                e.contains(part.trim()),
                "part {:?} is not a substring of input {:?}",
                part,
                e
            );
        }
    }

    // NOTE: the v1 escape-hatch behavior of `classify_boolean_v2` (returning
    // `None` when `TLAPLUS_EXPR_PARSER=v1`) is deliberately NOT tested here by
    // mutating the process-global env var: `std::env::set_var` is `unsafe` and
    // parallel-racy under the default (multi-threaded) test harness, and can
    // poison other tests that read the var concurrently. The pure config
    // decision underneath the env layer is unit-tested directly in
    // `crate::tla::expr_v2::config_tests` (via `v2_enabled_for`), and the
    // end-to-end escape hatch is covered at integration level by
    // `TLAPLUS_EXPR_PARSER=v1 scripts/diff_tlc.sh`.

    #[test]
    fn nested_child_junction_slice_reclassifies_or_falls_back_safely() {
        // Codex-requested nested-layout slice test. Shape:
        //     /\ \/ A
        //        \/ B
        //     /\ C
        // (1) The whole expr must classify as IndentedAnd with 2 parts: the
        //     inner `\/ A ... \/ B` disjunction source and `C`.
        // (2) Recursively re-classifying part[0] must yield the A-or-B
        //     disjunction — either directly via v2 (IndentedOr/Or) or via the
        //     v1 fallback (OrPrefixSingle etc.). Whichever path, the EVALUATION
        //     must be the disjunction of A and B, never a mis-grouped single
        //     atom. If the child slice shifts continuation-bullet columns so v2
        //     would mis-group, the code returns None → v1 fallback, which still
        //     evaluates correctly.
        let e = "/\\ \\/ A\n   \\/ B\n/\\ C";
        let op = classify_op(e);
        assert!(
            matches!(op, Op::IndentedAnd(_)),
            "whole expr must be IndentedAnd; got {}",
            variant_name(&op)
        );
        let parts = parts_of(&op);
        assert_eq!(parts.len(), 2, "expected 2 conjuncts, got {:?}", parts);
        // part[1] is the trailing `C` conjunct.
        assert_eq!(parts[1].trim(), "C");
        // part[0] is the inner disjunction source; it must contain both bullets.
        let inner = parts[0].as_str();
        assert!(
            inner.contains("\\/ A") && inner.contains("\\/ B"),
            "part[0] should be the A/B disjunction source; got {:?}",
            inner
        );
        // Recursively classify the child slice: it must resolve to an OR of the
        // two branches (directly via v2 IndentedOr/Or, or via the v1
        // OrPrefixSingle fallback), NOT a mis-grouped IndentedAnd/Implies/Iff.
        let child_op = classify_op(inner);
        let child_parts: Vec<String> = match &child_op {
            Op::IndentedOr(p) | Op::Or(p) => p.clone(),
            // OrPrefixSingle: the dispatcher strips the leading `\/` and
            // re-evaluates the remainder — still an OR-shaped, correct
            // evaluation (safe fallback), so accept it.
            Op::OrPrefixSingle => vec![inner.to_string()],
            other => panic!(
                "part[0] must re-classify as an OR of A and B (v2 or fallback); \
                 got a mis-grouping that is not OR-shaped: {}",
                variant_name(other)
            ),
        };
        // Both branches must survive the re-parse (or, in the single-strip
        // fallback, both bullets remain in the residual source to be evaluated).
        let joined = child_parts.join(" ");
        assert!(
            joined.contains('A') && joined.contains('B'),
            "child re-classify must retain both A and B branches; got {:?}",
            child_parts
        );
    }
}
