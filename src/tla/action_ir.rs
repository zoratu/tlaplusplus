use crate::tla::{ClauseKind, TlaDefinition, classify_clause, split_top_level};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionClause {
    PrimedAssignment {
        var: String,
        expr: String,
    },
    PrimedMembership {
        var: String,
        set_expr: String,
    },
    Unchanged {
        vars: Vec<String>,
    },
    Guard {
        expr: String,
    },
    Exists {
        binders: String,
        body: String,
    },
    /// LET expression that contains primed assignments in its IN body
    LetWithPrimes {
        expr: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionIr {
    pub name: String,
    pub params: Vec<String>,
    pub clauses: Vec<ActionClause>,
}

/// Phase 4: route top-level action-body conjunct splitting through the
/// layout-aware `expr_v2` AST when the body's ROOT is an `And` junction.
///
/// Returns `Some(items)` — one string per top-level conjunct, each the exact
/// source extent of that junction child (per-clause normalized, ready for
/// `classify_clause`) — ONLY when ALL of:
///   1. `expr_v2::parse_ast` succeeds. (`parse_ast` == `parser::parse`, whose
///      entry `parse_expr_top` already enforces FULL CONSUMPTION: it errors with
///      "trailing tokens after expr" unless the parse reaches `Eof`. So a
///      successful `Ok` guarantees no leftover/unparsed input — the requirement
///      from the Phase-4 design is satisfied by the parser itself. A lex error
///      also yields `Err`. No recovered/partial nodes exist in this parser.)
///   2. The ROOT node is `Junction { op: And, items }` with `items.len() >= 2`.
///      We do NOT unwrap through `Quant`/`Let`/`If`/`Binary`/etc. — a body that
///      is a bare `\E`/`\A`/`LET`/`IF`/`\/` has a non-`And`-junction root, so we
///      return `None` and the existing whole-body special-cases + string logic
///      run unchanged.
///   3. Every child span slices cleanly out of the SAME input string
///      (span-safety-guarded exactly like Phase 3's `classify_boolean_v2`:
///      bail to `None` on inverted / out-of-bounds / non-char-boundary spans).
///
/// CRITICAL — COLUMN CONSISTENCY: we parse the source AFTER
/// `normalize_multiline_action_indentation`. That normalization is NOT the
/// string splitter's line-flattening (which is what loses layout and what the
/// Phase-4 design warns against) — it is a UNIFORM dedent that strips the common
/// leading indentation while PRESERVING the relative columns of every line.
/// That is exactly what the layout-aware parser needs. It is required because a
/// bare `expr.trim()` only dedents the FIRST line: a body extracted mid-
/// definition (or a Rust raw-string test literal) then has its first bullet at
/// column 0 but every continuation bullet at the original indent (e.g. 16), so
/// the junction fence set by the first bullet is shallower than the siblings and
/// the second item's body SWALLOWS the trailing siblings (verified: the raw
/// `.trim()` of `keeps_quantified_action_bodies_together` yields 2 items where
/// the normalized form correctly yields 4). Normalization preserves relative
/// columns, so the root `And` still fences quantifier / `IF` / nested-junction
/// bodies correctly. Per-clause normalization AFTER slicing is then idempotent
/// (already-uniform slices dedent by 0), matching what the fallback does to each
/// `part`.
///
/// The v2 items are already correctly bounded, so NONE of the re-glue /
/// `open_quant_or_let` / `bound_trailing_let_in_body` compensation logic runs on
/// this path — that machinery exists only to patch the string splitter's
/// mis-grouping of `\E`/`\A`/`LET`/`IF` body extents, which expr_v2 owns.
///
/// CONJUNCTS ONLY. Disjunct splitting (`\/`, `\E`-over-disjunction) has
/// binder-scope semantics that raw inner-`Or` slices would break, and is left
/// entirely to the string path (possible Phase 5).
fn split_action_conjuncts_v2(expr: &str) -> Option<Vec<String>> {
    use crate::tla::expr_v2::{self, ast};

    // FIX D1 — parse a GENUINELY layout-preserved (uniform-dedent) source.
    //
    // `normalize_multiline_action_indentation` is NOT a true uniform dedent: it
    // leaves line 1 UNCHANGED and dedents only the continuation lines by *their
    // own* min-indent. That does NOT preserve the relative column between line 1
    // and later lines. For a shape like
    //     /\ A /\ \E x \in S :
    //            /\ P(x)
    //            /\ Q(x)
    // (no later line is an outer sibling) the continuation bullets `P`/`Q` get
    // collapsed toward line-1's column, so expr_v2 sees a top-level `And` where
    // `P(x)`/`Q(x)` were actually INSIDE the `\E` body → wrong conjunct grouping
    // → wrong successors. A TRUE uniform dedent — subtract the SAME min-indent
    // (computed across ALL non-blank lines, INCLUDING line 1) from EVERY line —
    // preserves every relative column and puts the shallowest line at column 0,
    // so expr_v2's junction fences match the original layout. We slice item
    // spans out of exactly this `src`, so the byte offsets stay valid.
    let dedented = uniform_dedent(expr);
    let src = dedented.trim();
    if src.is_empty() {
        return None;
    }
    // `parse_ast` (== `parser::parse` → `parse_expr_top`) errors on any trailing
    // tokens, so `Ok` == full consumption. `.ok()?` → fallback on parse/lex error.
    let tree = expr_v2::parse_ast(src).ok()?;

    // Root gate: only take the v2 path when the ROOT is an `And`-junction. Any
    // other root (Or-junction, Quant, Let, If, Binary, Not, containers, Atom, …)
    // → fallback, preserving the `\E`/`\A`/`IF`/`\/` whole-body special-cases and
    // the disjunct path.
    let ast::Expr::Junction { op: ast::JunctionOp::And, col: root_col, .. } = &tree else {
        return None;
    };

    // DISJUNCTION-BODY GUARD (soundness — EWD998PCal `node`, PlusCal
    // `Op(self) == \/ ... \/ ...`). A body whose FIRST logical line begins with a
    // `\/` bullet is semantically a DISJUNCTION, never a top-level conjunction.
    // But when that leading `\/` sits at a SHALLOWER column than its sibling `\/`
    // bullets — the classic `node(self) == \/ /\ A` shape, where line 1's `\/`
    // trails the `== ` at col 0 while the continuation `\/`s are indented — the
    // uniform-dedented source has inconsistent `\/` columns. v2 then fences the
    // shallow leading `\/` as a single (collapsing) disjunct and lets the LAST
    // quantifier/junction body greedily swallow the deeper sibling `\/` disjuncts,
    // so `parse_ast` returns a ROOT `And` (`/\ guard /\ \E x : <swallowed rest>`).
    // Conjunct-splitting THAT And shreds the whole multi-disjunct action into two
    // leaves whose second leaf is a `\E` body that absorbed every other disjunct
    // → the action loses all but a few successors (EWD998PCal N=3: 321,370 → ~11k
    // distinct, a false-safe under-exploration). The trusted string splitter
    // groups this shape correctly, so when the first logical line is `\/`-led we
    // return None and fall back. (A genuine `/\`-led body with an INLINE `\/`
    // inside a conjunct does not trip this: its first logical line starts with
    // `/\`, handled by the layout guard below.)
    if crate::tla::text_util::first_logical_line_skipping_comments(src).starts_with("\\/") {
        return None;
    }

    // LAYOUT-CONSISTENCY GUARD (D1 companion). A uniform dedent preserves every
    // relative column, but it does NOT repair a body whose top-level bullets sit
    // at INCONSISTENT columns — e.g. a raw-string / mid-definition extraction
    // where line 1's bullet is at col 0 but its top-level SIBLINGS are all
    // indented (col 12), with a nested `IF`/quantifier body even deeper. There
    // the parser fences the root junction on the DEEPER sibling column and a
    // trailing sibling gets swallowed into an `IF`-ELSE / quantifier body
    // (`split_action_body_clauses_keeps_parser_shaped_if_then_else_together`:
    // `/\ UNCHANGED tmState` absorbed into the ELSE branch). We detect this by
    // comparing line-1's LEADING bullet column against the root junction's fence
    // column `root_col`: when line 1 begins with a `/\` bullet, that bullet MUST
    // be the root fence. If it is SHALLOWER than `root_col`, the layout is
    // inconsistent for v2's fence model → return None and let the string
    // fallback own it (verified: the fallback groups every such shape — including
    // Codex's `/\ A /\ \E x : <bulleted body>` D1 case — correctly).
    //
    // The guard applies only to MULTI-LINE bodies: a single-line inline junction
    // (`/\ a' = 1 /\ b' = 2`) has no cross-line layout to be inconsistent, and
    // its root `col` is the SECOND (mid-line) bullet's column — never line-1's
    // leading-bullet column — so comparing them would spuriously reject every
    // legitimate inline conjunction.
    if src.contains('\n') {
        let first_line = src.lines().next().unwrap_or("");
        let first_trim = first_line.trim_start();
        if first_trim.starts_with("/\\") {
            let first_bullet_col = (first_line.len() - first_trim.len()) as u32;
            if first_bullet_col != *root_col {
                return None;
            }
        }
    }

    // FIX D2 — AST-based flattening instead of the string inline-splitter.
    //
    // The old v2 path ran the string `split_inline_action_conjuncts` on each AST
    // item, re-introducing string-level `/\` splitting INSIDE AST-bounded items.
    // That breaks binder/branch scope for shapes like
    //     /\ b' = \E x \in S : P(x) /\ Q(x)   (the /\ is the \E body's)
    //     /\ b' = IF c THEN P /\ Q ELSE R     (the /\ is the IF branch's)
    // — string-splitting that inner `/\` loses/invents successors.
    //
    // Instead we recursively flatten `Junction{And}` nodes: a conjunct is a
    // MAXIMAL subtree whose root is NOT `Junction{And}`. We unwrap nested
    // And-junctions into their children and STOP at any non-And node
    // (`Binary`/`Quant`/`Let`/`If`/`Atom`/containers/`Or`-junction/…), emitting
    // THAT node's source slice as one conjunct. This correctly splits inline
    // `guard /\ x'=e` (both are And-junction children) WITHOUT splitting the
    // `/\` inside a `\E`/`LET`/`IF` body (that `/\` lives under a Quant/Let/If
    // node, never unwrapped).
    let mut leaves: Vec<&ast::Expr> = Vec::new();
    flatten_and_leaves(&tree, &mut leaves);
    // A single flattened leaf → not really a >=2-conjunct action → fallback.
    if leaves.len() < 2 {
        return None;
    }

    let mut out = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        let sp = leaf.span();
        // Span-safety guard (mirrors `classify_boolean_v2`): bail to fallback
        // rather than panic-slicing on an inverted / OOB / non-boundary span.
        if sp.start > sp.end
            || sp.end > src.len()
            || !src.is_char_boundary(sp.start)
            || !src.is_char_boundary(sp.end)
        {
            return None;
        }
        // Leaf spans start AFTER any leading `/\` bullet (`parse_junction`
        // `advance()`s past the bullet before parsing the item), so the slice
        // never includes an outer bullet. Apply the SAME per-clause
        // normalization the fallback applies to each conjunct.
        let raw = &src[sp.start..sp.end];
        let clause = normalize_multiline_action_indentation(raw.trim())
            .trim()
            .to_string();
        if clause.is_empty() {
            return None;
        }
        out.push(clause);
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// FIX D2 — recursively collect the conjunct LEAVES of an action-body AST.
///
/// A conjunct is a maximal subtree whose root is NOT `Junction{And}`. This walk
/// unwraps nested `Junction{And}` nodes into their children and STOPS at every
/// other node kind — `Binary`, `Quant`, `Let`, `If`, `Atom`, `Paren`,
/// containers, and crucially a `Junction{Or}` (a disjunctive sub-action stays
/// ONE conjunct). The `/\` inside a `\E`/`\A`/`LET`/`IF` body is never unwrapped
/// because it lives under a `Quant`/`Let`/`If` node, which is a leaf here.
fn flatten_and_leaves<'a>(
    expr: &'a crate::tla::expr_v2::ast::Expr,
    out: &mut Vec<&'a crate::tla::expr_v2::ast::Expr>,
) {
    use crate::tla::expr_v2::ast;
    match expr {
        ast::Expr::Junction { op: ast::JunctionOp::And, items, .. } => {
            for it in items {
                flatten_and_leaves(it, out);
            }
        }
        // Same-connective Paren unwrap: `(A /\ B)` is a parenthesized conjunction
        // whose leaves are conjuncts, so descend into it. We unwrap ONLY when the
        // inner node is itself an `And`-junction — a `Paren` wrapping a `Quant`,
        // `Let`, `If`, `Or`-junction, `Binary`, etc. is a single conjunct leaf and
        // must stay whole (its inner `/\`, if any, is scoped by that node). This
        // closes the conservative `Paren(And)` case: previously `(A /\ B)` stayed
        // one leaf; now it flattens to `A`, `B`.
        ast::Expr::Paren { inner, .. }
            if matches!(
                inner.as_ref(),
                ast::Expr::Junction { op: ast::JunctionOp::And, .. }
            ) =>
        {
            flatten_and_leaves(inner, out);
        }
        // Any non-And node is a conjunct leaf, kept whole.
        other => out.push(other),
    }
}

/// The 4-state result of classifying an action body for DISJUNCT splitting via
/// expr_v2. A plain `Option<Vec<String>>` is insufficient here: a body whose
/// root is a SCOPED node (`Let`/`If`/`\A`/`Paren`-non-Or/`Quant`) may contain an
/// inner `\/` whose binder/branch scope spans both sides. Returning `None`
/// (fallback) for such a body would let the blind `split_top_level(_, "\\/")`
/// wrongly split that inner disjunction and orphan a binder or drop a guard. The
/// 4 states let the caller distinguish "root Or → split" from "fully parsed,
/// root is NOT a splittable Or → keep the body whole" from "genuine parse
/// failure → let the string logic decide".
enum DisjunctSplitV2 {
    /// Root is an `Or`-junction and the Or layout guard passed → these are the
    /// top-level disjuncts.
    Split(Vec<String>),
    /// Full parse succeeded but the root is NOT an `Or`-junction (`Quant`/`Let`/
    /// `If`/`And`/`Binary`/`Paren`-non-Or/`Atom`/containers/…). Do NOT blind-split
    /// on `\/`: any inner `\/` is scoped by the root node.
    ParsedNonRootOr,
    /// Root IS an `Or`-junction but the Or layout guard failed (inconsistent
    /// bullet columns for a raw/mid-definition extraction). Do NOT blind-split;
    /// keep the body whole conservatively.
    RootOrLayoutMismatch,
    /// Parse failed or did not fully consume the input → let the existing string
    /// logic own the body (pre-Phase-5 behavior).
    Fallback,
}

/// Phase 5 — classify an action body for top-level DISJUNCT splitting via the
/// layout-aware expr_v2 AST. See `DisjunctSplitV2` for the 4-state contract.
///
/// Mirrors `split_action_conjuncts_v2` (Phase 4): uniform-dedent so the parser's
/// coordinate system matches the source, `parse_ast` (full-consume == `Ok`),
/// then an Or-fence layout guard + `flatten_or_leaves` slice extraction with the
/// same span-safety and per-clause normalization.
fn classify_action_disjunct_v2(expr: &str) -> DisjunctSplitV2 {
    use crate::tla::expr_v2::{self, ast};

    // TRUE uniform dedent (preserves every relative column; see `uniform_dedent`).
    // We slice leaf spans out of exactly this `src`, so byte offsets stay valid.
    let dedented = uniform_dedent(expr);
    let src = dedented.trim();
    if src.is_empty() {
        return DisjunctSplitV2::Fallback;
    }

    // `Ok` == full consumption (trailing tokens error). Parse/lex error → fallback.
    let Ok(tree) = expr_v2::parse_ast(src) else {
        return DisjunctSplitV2::Fallback;
    };

    // Root gate. Any non-`Or` root is a fully-parsed scoped/leaf node: return
    // `ParsedNonRootOr` so the caller keeps the body whole (protecting root
    // `Let`/`If`/`\A`/`\E`/`And`/`Paren`-non-Or bodies with an inner `\/`).
    let ast::Expr::Junction { op: ast::JunctionOp::Or, col: root_col, .. } = &tree else {
        return DisjunctSplitV2::ParsedNonRootOr;
    };

    // CONJUNCTION-BODY GUARD (soundness — Peterson `a3`, 2PC guard branches).
    // A `/\`-led action body is a CONJUNCTION, never a top-level disjunction. But
    // expr_v2 fences an INLINE boolean `\/` (e.g. `~c[Other(self)] \/ turn=self`
    // sitting at/left of the `/\` bullet column) as the LOOSER outer operator, so
    // it parses `/\ A /\ (B \/ C) /\ D` with a ROOT `Or` that splits the bulleted
    // `/\` list across the inner boolean `\/`. Taking the v2 `Split` here bypasses
    // the existing string `/\`-conjunction guard (`conjunctive_clauses.len() > 1
    // => whole body`) and shreds the conjunction into disjuncts that cross a
    // guard with another branch's effect — inventing successors (Peterson's
    // spurious `a0 -> cs`, a false mutual-exclusion violation). If the first
    // logical line begins with a `/\` bullet, the semantic root is an `And`, so
    // this v2 root-`Or` is a mis-fence: return `RootOrLayoutMismatch` and let the
    // string path's conjunction guard keep the whole body as one action.
    {
        // FIX B (Phase 5.1): locate the first LOGICAL line, skipping not only
        // blank lines and `\*` line comments but also leading `(* ... *)` block
        // comments (single- OR multi-line). A leading block comment before a
        // `/\`-led body would otherwise bypass this guard and recreate the
        // Peterson mis-fence (a `/\`-led conjunction shredded into disjuncts →
        // false mutual-exclusion violation).
        let first_logical = crate::tla::text_util::first_logical_line_skipping_comments(src);
        if first_logical.starts_with("/\\") {
            return DisjunctSplitV2::RootOrLayoutMismatch;
        }
    }

    // OR LAYOUT-CONSISTENCY GUARD (mirrors the Phase-4 And guard). A uniform
    // dedent preserves relative columns but cannot repair a body whose top-level
    // bullets sit at inconsistent columns (raw / mid-definition extraction). For
    // a MULTI-LINE body whose first logical line begins with a `\/` bullet, that
    // bullet MUST be the root Or fence column. If it is shallower, v2's fence
    // model does not match the layout → `RootOrLayoutMismatch` (keep whole).
    // Single-line inline `\/` bodies have no cross-line layout: `root_col` is a
    // mid-line bullet column, never line-1's leading-bullet column, so comparing
    // them would spuriously reject every legitimate inline disjunction.
    if src.contains('\n') {
        let first_line = src.lines().next().unwrap_or("");
        let first_trim = first_line.trim_start();
        if first_trim.starts_with("\\/") {
            let first_bullet_col = (first_line.len() - first_trim.len()) as u32;
            if first_bullet_col != *root_col {
                return DisjunctSplitV2::RootOrLayoutMismatch;
            }
        }
    }

    // Flatten the root `Or` into its maximal non-Or leaves.
    let mut leaves: Vec<&ast::Expr> = Vec::new();
    flatten_or_leaves(&tree, &mut leaves);
    // Fewer than 2 leaves → not really a >=2-disjunct body → fallback.
    if leaves.len() < 2 {
        return DisjunctSplitV2::Fallback;
    }

    let mut out = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        let sp = leaf.span();
        // Span-safety guard: bail to fallback rather than panic-slice on an
        // inverted / OOB / non-char-boundary span.
        if sp.start > sp.end
            || sp.end > src.len()
            || !src.is_char_boundary(sp.start)
            || !src.is_char_boundary(sp.end)
        {
            return DisjunctSplitV2::Fallback;
        }
        // Leaf spans start AFTER the enclosing `\/` bullet (`parse_junction`
        // advances past the bullet before parsing the item). But when a disjunct
        // branch is written `\/ /\ <conjuncts>`, the leaf's OWN leading `/\`
        // bullet can also be excluded from the span:
        //   - an `And`-junction leaf's span starts at its FIRST item when that
        //     item is inline right after the `\/` (Peterson's `a3`, 2PC guards);
        //   - a SINGLE-conjunct branch (`\/ /\ \E S : ...`) collapses in
        //     `parse_junction` (`items.len()==1` → the item), so the leaf is the
        //     `\E`/`Atom` and the `/\` is likewise dropped (byzpaxos `leader`).
        // Dropping that `/\` (and re-normalizing the now-unwrapped body) shreds a
        // conjunction's guard (Peterson: false mutual-exclusion violation) or
        // changes a quantifier body's extent (byzpaxos: state-space blow-up) —
        // successors invented or lost. The trusted string splitter KEEPS the
        // `/\`. So EXTEND the slice backward to recapture a leading `/\` bullet
        // that sits between the `\/` bullet and this leaf (skipping only
        // whitespace/newlines, and never crossing the `\/` bullet). This makes
        // every disjunct's source text match the string path exactly, for every
        // leaf kind.
        let start = recapture_leading_and_bullet(src, sp.start);
        let raw = &src[start..sp.end];
        let clause = normalize_multiline_action_indentation(raw.trim())
            .trim()
            .to_string();
        if clause.is_empty() {
            return DisjunctSplitV2::Fallback;
        }
        out.push(clause);
    }
    if out.is_empty() {
        return DisjunctSplitV2::Fallback;
    }
    DisjunctSplitV2::Split(out)
}

/// Phase 5 — extend a disjunct-leaf slice start backward to recapture a leading
/// `/\` bullet that the parser dropped.
///
/// Given `src` and the leaf's span start, scan backward over ASCII whitespace
/// (spaces / tabs / newlines) only. If the two bytes immediately before that run
/// are a `/\` AndBullet, return the index of that `/\` (so the slice includes
/// it). Otherwise return `leaf_start` unchanged. The scan never crosses a `\/`
/// OrBullet: we stop looking as soon as the preceding non-whitespace is anything
/// other than the `\` of a `/\`. This is intentionally minimal — it recovers the
/// single `/\` a `\/ /\ <body>` branch dropped, and does nothing for a genuine
/// non-bulleted leaf (Atom / bare `\E` with no preceding `/\`).
/// Phase 5.1 (FIX C) — take ONE backward step over a run of whitespace, a `\*`
/// line comment, or a `(* ... *)` block comment ending immediately before `end`.
/// Returns the index of the byte just before the skipped run, or `end` unchanged
/// if the byte before `end` is neither whitespace nor the close of a comment.
/// Called in a loop until it stops making progress.
fn skip_whitespace_and_comments_backward(bytes: &[u8], end: usize) -> usize {
    let mut i = end;
    // 1. Whitespace run.
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\n' | b'\r') {
        i -= 1;
    }
    if i < end {
        return i;
    }
    // 2. Block comment `(* ... *)` closing at `i` (i.e. bytes[i-2..i] == "*)").
    if i >= 2 && bytes[i - 2] == b'*' && bytes[i - 1] == b')' {
        // Scan back to the matching `(*`, honoring nesting.
        let mut depth = 1usize;
        let mut j = i - 2;
        while j >= 2 {
            if bytes[j - 2] == b'*' && bytes[j - 1] == b')' {
                depth += 1;
                j -= 2;
            } else if bytes[j - 2] == b'(' && bytes[j - 1] == b'*' {
                depth -= 1;
                j -= 2;
                if depth == 0 {
                    return j;
                }
            } else {
                j -= 1;
            }
        }
        // Unbalanced (no matching `(*`): don't skip past — leave unchanged so the
        // recapture conservatively fails rather than over-absorbing.
        return end;
    }
    // 3. `\*` line comment: it runs to end-of-line. A trailing line comment sits
    // BEFORE a newline, so after the whitespace step above we'd be at the `\n`
    // that precedes the comment's line only if there was one; a same-line `\*`
    // comment ending at `i` means bytes[..i] ends inside the comment text. Walk
    // back to the `\*` opener on this line if present (bounded by line start).
    let line_start = {
        let mut k = i;
        while k > 0 && bytes[k - 1] != b'\n' {
            k -= 1;
        }
        k
    };
    let mut k = i;
    while k >= line_start + 2 {
        if bytes[k - 2] == b'\\' && bytes[k - 1] == b'*' {
            return k - 2;
        }
        k -= 1;
    }
    end
}

fn recapture_leading_and_bullet(src: &str, leaf_start: usize) -> usize {
    let bytes = src.as_bytes();
    // Walk back over whitespace AND comments. FIX C (Phase 5.1): a `\*` line
    // comment or `(* ... *)` block comment can sit between the `/\` bullet and
    // the leaf (`\/ /\ (* c *) guard`); without skipping it the backward scan
    // stops at the comment's closing byte, fails to find the `/\`, and DROPS the
    // conjunction bullet — shredding the disjunct's guard. Skip such comments so
    // the `/\` is still recaptured.
    let mut i = leaf_start;
    loop {
        let before = skip_whitespace_and_comments_backward(bytes, i);
        if before == i {
            break;
        }
        i = before;
    }
    // Need at least two bytes `/\` immediately before the whitespace/comment run.
    if i >= 2 && bytes[i - 2] == b'/' && bytes[i - 1] == b'\\' {
        let bullet_start = i - 2;
        // Guard: the `/\` must itself be a bullet — i.e. preceded by start-of-
        // input, whitespace, or a `\/` OrBullet (`\/ /\ ...`). If the byte before
        // the `/\` is a non-space, non-`/` token char, this `/\` is part of some
        // larger token/expression and must NOT be absorbed.
        let ok_before = bullet_start == 0
            || matches!(bytes[bullet_start - 1], b' ' | b'\t' | b'\n' | b'\r' | b'/');
        if ok_before && src.is_char_boundary(bullet_start) {
            return bullet_start;
        }
    }
    leaf_start
}

/// Phase 5 — recursively collect the DISJUNCT leaves of an action-body AST.
///
/// A disjunct is a maximal subtree whose root is NOT `Junction{Or}`. This walk
/// unwraps nested `Junction{Or}` nodes into their children and STOPS at every
/// other node kind — `Junction{And}` (a `\/ /\ guard /\ post` branch is ONE
/// disjunct that CONTAINS a conjunction), `Quant`, `Let`, `If`, `Binary`,
/// `Atom`, containers. It NEVER enters a quantifier / LET / IF / And node, so a
/// `\/` scoped inside any of those (e.g. `\E x : A(x) \/ B(x)` — root `Quant`)
/// is never split here.
fn flatten_or_leaves<'a>(
    expr: &'a crate::tla::expr_v2::ast::Expr,
    out: &mut Vec<&'a crate::tla::expr_v2::ast::Expr>,
) {
    use crate::tla::expr_v2::ast;
    match expr {
        ast::Expr::Junction { op: ast::JunctionOp::Or, items, .. } => {
            for it in items {
                flatten_or_leaves(it, out);
            }
        }
        // Same-connective Paren unwrap: `(A \/ B)` is a parenthesized disjunction
        // whose leaves are disjuncts, so descend. Unwrap ONLY when the inner node
        // is itself an `Or`-junction — a `Paren` wrapping an `And`/`Quant`/`Let`/
        // `If`/`Binary` is a single disjunct leaf and stays whole.
        ast::Expr::Paren { inner, .. }
            if matches!(
                inner.as_ref(),
                ast::Expr::Junction { op: ast::JunctionOp::Or, .. }
            ) =>
        {
            flatten_or_leaves(inner, out);
        }
        // Any non-Or node is a disjunct leaf, kept whole.
        other => out.push(other),
    }
}

/// FIX D1 — a TRUE uniform dedent: compute the minimum leading indentation
/// across ALL non-blank lines (INCLUDING the first) and strip that SAME amount
/// from EVERY line. This PRESERVES the relative column of every line, so the
/// layout-aware expr_v2 parser sees a coordinate system that matches the
/// original source exactly.
///
/// Unlike `normalize_multiline_action_indentation` (which leaves line 1
/// untouched and dedents continuations by *their own* min-indent — collapsing a
/// nested `\E`/`IF` body toward line-1's column when line 1 is the shallowest
/// line and all continuations are that body), a uniform dedent never changes
/// any line's column RELATIVE to any other. When line 1 is already the
/// shallowest line at column 0 this is a no-op; when line 1 is itself indented
/// it rebases the whole block so the shallowest line lands at column 0 while
/// keeping every relative offset intact.
pub(crate) fn uniform_dedent(expr: &str) -> String {
    let min_indent = expr
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                None
            } else {
                Some(line.len() - trimmed.len())
            }
        })
        .min()
        .unwrap_or(0);

    if min_indent == 0 {
        return expr.to_string();
    }

    let mut out = String::with_capacity(expr.len());
    for (i, line) in expr.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if line.trim_start().is_empty() {
            // Blank / whitespace-only line: keep empty (no column to preserve).
            continue;
        }
        // Every non-blank line has indent >= min_indent by construction, so this
        // slice is always in bounds and on a char boundary (leading spaces/tabs
        // are single-byte ASCII).
        out.push_str(&line[min_indent..]);
    }
    out
}

pub(crate) fn split_action_body_clauses(expr: &str) -> Vec<String> {
    // ---- Phase 4: layout-aware action-conjunct splitting via expr_v2 ----
    // Route the top-level conjunct split through the v2 AST when the body's ROOT
    // is an `And` junction. Returns `None` (→ the entire string-based body below
    // runs UNCHANGED as fallback) whenever v2 is disabled, the parse errors, or
    // the root is not a >=2-item `And` junction. See `split_action_conjuncts_v2`.
    if crate::tla::expr_v2::v2_enabled()
        && let Some(clauses) = split_action_conjuncts_v2(expr)
    {
        return clauses;
    }

    let original = expr.trim();
    let normalized = normalize_multiline_action_indentation(original);
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if let Some(body) = strip_leading_comment_only_lines(trimmed)
        && body.starts_with("\\/")
    {
        return vec![trimmed.to_string()];
    }
    if trimmed.starts_with("\\E") || trimmed.starts_with("\\A") {
        return vec![trimmed.to_string()];
    }
    // A body that *is* a bare `IF cond THEN ... ELSE ...` is one clause, not a
    // conjunction. Crucially, its condition can be a multi-line bulleted
    // conjunction — `IF /\ A\n   /\ B THEN ...` — and the indentation-based
    // conjunct splitter below (`split_indented_action_conjuncts`) would treat
    // the condition's `/\ B` continuation line as a *top-level* conjunct,
    // shredding the IF into `["IF /\ A", "B\nTHEN", "<then>\nELSE", "<else>"]`.
    // That mis-split silently drops the action's successors (the shredded
    // first clause `IF /\ A` has no THEN, so it compiles to a dead branch).
    // Nested IFs inside an outer IF's THEN/ELSE branch hit exactly this shape
    // (e.g. DiningPhilosophers' `Loop`, PlusCal-generated elsif chains).
    // `\E`/`\A`/comment-`\/` above are guarded for the same reason; `IF` was
    // simply missing. Keep the whole IF together and let the recursive
    // `compile_action_clause_text` / `parse_action_if` handle its structure.
    if trimmed.starts_with("IF")
        && trimmed[2..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_')
    {
        return vec![trimmed.to_string()];
    }

    let raw = split_indented_action_conjuncts(original)
        .unwrap_or_else(|| split_top_level(trimmed, "/\\"));
    let expanded = raw
        .into_iter()
        .flat_map(|part| split_inline_action_conjuncts(&part))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "verus"))]
    let merged_cap = expanded.len().max(1);
    #[cfg(feature = "verus")]
    let merged_cap = crate::storage::verus_smoke::max_usize(expanded.len(), 1);
    let mut merged = Vec::with_capacity(merged_cap);
    let mut idx = 0usize;
    while idx < expanded.len() {
        let part = normalize_multiline_action_indentation(expanded[idx].trim())
            .trim()
            .to_string();
        if part.is_empty() {
            idx += 1;
            continue;
        }
        let starts_quant = part.starts_with("\\E") || part.starts_with("\\A");
        let incomplete_quantifier = starts_quant && !has_complete_quantifier_body(&part);
        let open_quant_or_let = incomplete_quantifier
            || (starts_quant && (part.ends_with(':') || part.ends_with("IN")))
            || (part.starts_with("LET") && part.ends_with("IN"));
        if open_quant_or_let {
            // Re-glue the trailing conjuncts of an open quantifier/LET body with
            // ` /\ `. A conjunct whose value is `LHS = LET ... IN <body>` (or a
            // bare trailing `LET ... IN <body>`) is hazardous here: a LET body has
            // no closing token, so once flattened onto one line the next ` /\ `
            // glue is swallowed *into* that body — e.g.
            //   received' = LET sb == ... IN [n \in Node |-> ...]
            //   /\ UNCHANGED distributedLedger
            // becomes `received' = LET ... IN ([func] /\ UNCHANGED ...)`, and the
            // primed assignment's RHS evaluates to a Function used as a Boolean
            // (NanoBlockchain's Create{Send,Receive,ChangeRep}Block). Bound the
            // LET body with explicit parentheses so the glue conjoins at the
            // intended (outer) level.
            let mut combined = bound_trailing_let_in_body(&part);
            for rest in expanded.iter().skip(idx + 1) {
                let rest = normalize_multiline_action_indentation(rest.trim())
                    .trim()
                    .to_string();
                if rest.is_empty() {
                    continue;
                }
                combined.push_str(" /\\ ");
                combined.push_str(&bound_trailing_let_in_body(&rest));
            }
            merged.push(combined);
            break;
        }

        merged.push(part);
        idx += 1;
    }

    merged
}

fn has_complete_quantifier_body(expr: &str) -> bool {
    let trimmed = expr.trim();
    let rest = if let Some(rest) = trimmed.strip_prefix("\\E") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("\\A") {
        rest
    } else {
        return false;
    };
    if !rest.starts_with(char::is_whitespace) && !rest.starts_with('(') {
        return false;
    }
    let rest = rest.trim_start();
    let Some(colon_idx) = find_action_char(rest, ':') else {
        return false;
    };
    !rest[colon_idx + 1..].trim().is_empty()
}

fn split_inline_action_conjuncts(part: &str) -> Vec<String> {
    let normalized = normalize_multiline_action_indentation(part.trim());
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if trimmed.starts_with("LET")
        || trimmed.starts_with("\\E")
        || trimmed.starts_with("\\A")
        || trimmed.starts_with("IF")
        || trimmed.starts_with("CASE")
        || trimmed.starts_with("\\/")
    {
        return vec![trimmed.to_string()];
    }

    let split = split_top_level(trimmed, "/\\");
    if split.len() <= 1 {
        vec![trimmed.to_string()]
    } else {
        split
    }
}

pub fn split_action_body_disjuncts(expr: &str) -> Vec<String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // ---- Phase 5: layout-aware action-DISJUNCT splitting via expr_v2 ----
    // Compute the 4-state v2 classification ONCE up front (gated on v2_enabled).
    //   - `Split(v)`             → return the v2 disjuncts immediately.
    //   - `ParsedNonRootOr`      → full parse, root is a scoped/leaf node
    //                              (`Let`/`If`/`\A`/`\E`/`And`/`Binary`/…): the
    //                              existing string guards below give the right
    //                              answer for the shapes they cover, and the
    //                              blind `split_top_level(_, "\\/")` fall-through
    //                              is REPLACED with "return the whole body as one
    //                              disjunct" (see the guard just before it). This
    //                              protects root `LET`/`IF`/`\A`/`Paren`-non-Or
    //                              bodies with an inner `\/` that the string
    //                              guards don't cover.
    //   - `RootOrLayoutMismatch` → same conservative protection (whole body).
    //   - `Fallback`             → parse genuinely failed: pre-Phase-5 behavior,
    //                              the string logic (incl. the final blind split)
    //                              owns the body unchanged.
    let disjunct_v2 = if crate::tla::expr_v2::v2_enabled() {
        classify_action_disjunct_v2(trimmed)
    } else {
        DisjunctSplitV2::Fallback
    };
    if let DisjunctSplitV2::Split(clauses) = &disjunct_v2 {
        return clauses.clone();
    }

    if split_outer_let(trimmed).is_some() {
        if let Some(disjuncts) = split_outer_let_body_disjuncts(trimmed) {
            return disjuncts;
        }
        return vec![trimmed.to_string()];
    }

    if parse_action_if(trimmed).is_some() {
        return vec![trimmed.to_string()];
    }

    if let Some(rest) = trimmed.strip_prefix("/\\").map(str::trim_start)
        && rest.starts_with("\\/")
    {
        return split_action_body_disjuncts(rest);
    }

    if let Some(disjuncts) = split_indented_action_disjuncts(trimmed) {
        return disjuncts;
    }

    // When the expression is a single-line `\E var \in Set : Body1 \/ Body2`,
    // `split_top_level` would strip the quantifier prefix from disjuncts after
    // the first.  Detect this case and repeat the `\E ... :` prefix on every
    // branch so that bound variables remain in scope.
    if let Some((binders, body)) = parse_action_exists(trimmed) {
        // SOUNDNESS: if the exists body is itself a quantifier (e.g.
        // `\E p \in P : \E q \in Q : A(p,q) \/ B(p,q)`), the `\/` lives *inside*
        // the inner `\E q` scope — its bound variable spans both disjuncts.
        // Splitting on `\/` here would yield `\E p : \E q : A(p,q)` and
        // `\E p : B(p,q)` (the inner `\E q :` prefix silently dropped from the
        // second branch), leaving `q` unbound so that disjunct produces zero
        // successors — a false-safe under-exploration bug (Lamport's mutex spec).
        // Keep the whole quantified body as one disjunct and let the recursive
        // per-branch exists expansion (`eval_exists_action_multi`) handle the
        // inner `\/` with `q` in scope.
        let body_head = body.trim_start();
        let body_is_nested_quantifier = body_head.starts_with("\\E ")
            || body_head.starts_with("\\A ")
            || body_head.starts_with("\\E\n")
            || body_head.starts_with("\\A\n");
        if body_is_nested_quantifier {
            // The inner quantifier's scope covers any `\/` in `body`, so this
            // whole `\E <binders> : <inner quantifier>` is a single disjunct.
            // Returning it here also prevents the generic `split_top_level` fall-
            // through below from splitting on that inner `\/` and orphaning the
            // inner bound variable.
            return vec![trimmed.to_string()];
        }
        // SOUNDNESS: if the quantifier body is a top-level *conjunction* whose
        // final conjunct is a nested disjunction —
        //   \E i \in Proc :
        //      /\ moved[i] = "NO"
        //      /\ procPause' = [procPause EXCEPT ![i] = 0]
        //      /\ \/ /\ moved' = [moved EXCEPT ![i] = "PREDICT"] /\ ...
        //         \/ /\ moved' = [moved EXCEPT ![i] = "RECEIVE"] /\ ...
        // — the `\/` lives *inside* the last `/\` conjunct, NOT at the body's
        // top level. `split_top_level(body, "\\/")` below is blind to the `/\`
        // structure and would shred the disjunction into independent `\E i : ...`
        // branches, dropping the surrounding guards (`moved[i] = "NO"`) and the
        // shared primed conjuncts (`procPause'`, `UNCHANGED`). Those orphaned
        // branches then fire from states where the guard is false and later read
        // an unstaged prime — a spurious-partial-branch bug that both
        // over-generates successors and, when a trailing constraint reads the
        // dropped prime, panics (EnvironmentController's `ProcTick`; ACP's
        // `parProgNB`). The conjunction guard at the end of this function catches
        // the non-quantified analogue, but it inspects `trimmed`, which starts
        // with `\E` here and so `split_action_body_clauses` returns a single
        // clause — the check must run on the *body*. Keep the whole
        // `\E <binders> : <conjunction>` as one disjunct; downstream IR
        // compilation / `eval_exists_action_multi` expands the inner `\/` with
        // the guards and binder in scope.
        if split_action_body_clauses(body).len() > 1 {
            return vec![trimmed.to_string()];
        }
        let body_disjuncts = split_top_level(body, "\\/");
        if body_disjuncts.len() > 1 || body.trim_start().starts_with("\\/") {
            return body_disjuncts
                .into_iter()
                .filter_map(|part| {
                    let part = part.trim().to_string();
                    if part.is_empty() {
                        None
                    } else {
                        Some(format!("\\E {} :\n{}", binders, part))
                    }
                })
                .collect();
        }
    }

    // SOUNDNESS: If the body parses as a top-level conjunction (e.g.
    //   /\ guard
    //   /\ \/ A
    //      \/ B
    //   /\ shared_post
    // ), the outer connective is `/\`, NOT `\/`. Splitting on `\/` here would
    // (a) fabricate spurious successors that drop the surrounding guards and
    // shared post-conditions, and (b) mangle the structure with a stutter
    // half-clause from the first `/\`. The whole body is one action — return
    // it as a single disjunct and let downstream IR compilation
    // (`compile_action_ir_branches` / `apply_compiled_action_ir_multi`) expand
    // the nested disjunction with the shared clauses correctly.
    let conjunctive_clauses = split_action_body_clauses(trimmed);
    if conjunctive_clauses.len() > 1 {
        return vec![trimmed.to_string()];
    }

    // SOUNDNESS: the body reduces to a *single* conjunctive clause that is a
    // quantified action (`[/\] \E x \in S : <body-with-nested-\/>`). The `\/`
    // then lives *inside* the quantifier body, so its bound variable spans
    // every disjunct. This is the conjunctive analogue of the nested-quantifier
    // guard on the `parse_action_exists` branch above. A `\E`-led conjunct such
    // as
    //   /\ \E block \in received[node] :
    //          /\ \/ A(block) \/ B(block)
    //          /\ post(block)
    // parses to one `\E block : ...` clause, but the `split_top_level(trimmed,
    // "\\/")` fall-through below is blind to the quantifier body and to the
    // inner `/\` nesting — it would shred the disjunction into `\/`-branches
    // with `block` unbound (each resolving to `ModelValue("block")`), silently
    // dropping every successor (NanoBlockchain's `ProcessBlock`, whose
    // multi-line body loses its relative indentation during action-body
    // normalization so the nested `\/` land at the top level). Keep the whole
    // quantified clause as one disjunct and let `eval_exists_action_multi`
    // expand the inner `\/` with the binder in scope.
    if let [clause] = conjunctive_clauses.as_slice() {
        let head = clause.trim_start();
        let is_quantified = ["\\E ", "\\A ", "\\E\n", "\\A\n", "\\E(", "\\A("]
            .iter()
            .any(|prefix| head.starts_with(prefix));
        if is_quantified {
            return vec![trimmed.to_string()];
        }
    }

    // Phase 5 — IMMEDIATELY before the blind `split_top_level(_, "\\/")`: if v2
    // fully parsed the body and its root is NOT a splittable top-level `Or`
    // (`ParsedNonRootOr`), OR the root was an `Or` whose layout guard failed
    // (`RootOrLayoutMismatch`), do NOT blind-split. For a body with NO leading
    // `\/`, return the WHOLE body as one disjunct — this protects root `LET d ==
    // e IN A \/ B`, `IF c THEN A \/ B ELSE D`, `\A x : A(x) \/ B(x)`, `\E`-scoped
    // and `Paren`-non-Or bodies whose inner `\/` the string guards above do not
    // cover. For a body WITH a single leading `\/` bullet (a collapsed
    // single-disjunct `\/ <scoped item>`), strip that one bullet and return the
    // remainder whole (FIX A below) — same protection, minus the F3 self-loop.
    // Either way the inner disjunction is expanded downstream with its
    // binder/branch scope intact. Only a `Fallback` (genuine v2 parse failure)
    // reaches the blind split with pre-Phase-5 behavior. `Split` already
    // returned at the top of the function.
    //
    // FIX A (Phase 5.1 BLOCKER): a `\/`-led body that v2 fully parsed to a
    // non-root-Or / layout-mismatch shape (e.g. `\/ \E x \in S : A(x) \/ B(x)`,
    // `\/ LET d == e IN A \/ B`) MUST NOT reach the blind `split_top_level(_,
    // "\\/")` below. A single-disjunct `\/ X` collapses in `parse_junction`
    // (`items.len()==1` → the item), so v2 reports the root as that scoped item
    // node (`Quant`/`Let`/`Atom`/…) → `ParsedNonRootOr` (or a layout mismatch).
    // The blind split would re-split the inner scoped `\/` of `X`, leaking the
    // binder / LET scope (Codex blocker). Instead, strip EXACTLY the ONE leading
    // `\/` bullet and return the remainder as ONE disjunct — the downstream `\E`
    // / LET expansion then handles the inner `\/` with `x` / `d` in scope.
    //
    // Why strip the `\/` (rather than keep it as pre-Phase-5): keeping the
    // leading `\/` re-triggers the F3 fixpoint self-loop — every recursive caller
    // that re-feeds a returned disjunct
    // (`collect_action_param_samples_from_expr`, `split_nested_action_disjuncts`,
    // …) branches on `trimmed.starts_with("\\/")` and recurses on its own input
    // → unbounded self-recursion. Stripping the one bullet breaks the loop while
    // preserving the scoped inner `\/` unsplit. A genuine multi-branch `\/ A \/ B
    // \/ C` is a root `Or` → already returned as `Split` at the top of the
    // function, so it never reaches here.
    if matches!(
        disjunct_v2,
        DisjunctSplitV2::ParsedNonRootOr | DisjunctSplitV2::RootOrLayoutMismatch
    ) {
        if let Some(stripped) = trimmed.strip_prefix("\\/") {
            // Strip exactly the ONE leading `\/` bullet; keep the remainder whole
            // (do NOT blind-split — the inner `\/` is scoped by the item node).
            let stripped = stripped.trim_start();
            if !stripped.is_empty() {
                return vec![stripped.to_string()];
            }
        }
        // Non-`\/`-led scoped body (root `LET`/`IF`/`\A`/`\E`/`Paren`-non-Or):
        // keep the whole body as one disjunct.
        return vec![trimmed.to_string()];
    }

    // Only a `Fallback` (genuine v2 parse failure, or v2 disabled) reaches the
    // blind string split below with pre-Phase-5 behavior. No parsed scoped body
    // — `\/`-led or otherwise — can reach it anymore.
    split_top_level(trimmed, "\\/")
        .into_iter()
        .filter_map(|part| {
            let part = part.trim().to_string();
            if part.is_empty() { None } else { Some(part) }
        })
        .collect()
}

fn split_outer_let_body_disjuncts(expr: &str) -> Option<Vec<String>> {
    let (defs, body) = split_outer_let(expr)?;
    let body_clauses = split_action_body_clauses(body);
    let clauses = if body_clauses.is_empty() {
        vec![body.trim().to_string()]
    } else {
        body_clauses
    };

    let mut clause_options = Vec::with_capacity(clauses.len());
    let mut has_branching = false;
    for clause in &clauses {
        let options = split_nested_action_disjuncts(clause);
        has_branching |= options.len() > 1;
        clause_options.push(options);
    }
    if !has_branching {
        return None;
    }

    let mut branch_bodies = vec![Vec::<String>::new()];
    for options in clause_options {
        let mut next = Vec::new();
        for existing in &branch_bodies {
            for option in &options {
                let mut branch = existing.clone();
                branch.push(option.clone());
                next.push(branch);
            }
        }
        branch_bodies = next;
    }

    Some(
        branch_bodies
            .into_iter()
            .map(|clauses| {
                let body = format_action_clause_sequence(&clauses);
                format!("LET {defs} IN\n{body}")
            })
            .collect(),
    )
}

fn split_nested_action_disjuncts(expr: &str) -> Vec<String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if let Some(rest) = trimmed.strip_prefix("/\\").map(str::trim_start)
        && rest.starts_with("\\/")
    {
        return split_action_body_disjuncts(rest);
    }
    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() > 1 || trimmed.starts_with("\\/") {
        disjuncts
    } else {
        vec![trimmed.to_string()]
    }
}

fn format_action_clause_sequence(clauses: &[String]) -> String {
    match clauses {
        [] => String::new(),
        [single] => single.trim().to_string(),
        _ => clauses
            .iter()
            .filter(|clause| !clause.trim().is_empty())
            .map(|clause| format_branch_clause(clause))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn split_outer_let(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("LET")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    let mut depth = 1usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_word(rest, i) {
        match word {
            "LET" => depth += 1,
            "IN" => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let defs = rest[..start].trim();
                    let body = rest[end..].trim();
                    if !defs.is_empty() && !body.is_empty() {
                        return Some((defs, body));
                    }
                    return None;
                }
            }
            _ => {}
        }
        i = end;
    }

    None
}

fn next_word(text: &str, start: usize) -> Option<(&str, usize, usize)> {
    let mut word_start = None;
    for (idx, ch) in text.char_indices().skip_while(|(idx, _)| *idx < start) {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            word_start.get_or_insert(idx);
        } else if let Some(begin) = word_start {
            return Some((&text[begin..idx], begin, idx));
        }
    }
    word_start.map(|begin| (&text[begin..], begin, text.len()))
}

fn split_indented_action_conjuncts(expr: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_action_indentation(expr);
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut base_indent = None;
    let mut saw_top_level = false;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed.starts_with("/\\") {
            let top_level_indent = *base_indent.get_or_insert(indent);
            // A `/\` at the base indent normally starts a new top-level
            // conjunct — UNLESS the conjunct we're building is an as-yet
            // unterminated `IF` whose *condition* is a multi-line bulleted
            // list. In `IF /\ A` / `   /\ B` / `   THEN ...`, the `/\ B` line
            // is a continuation of the IF condition, not a sibling conjunct.
            // Because `normalize_multiline_action_indentation` dedents the
            // condition-continuation lines down to the base indent, they would
            // otherwise be mis-split, shredding the IF (dropping the action's
            // successors — DiningPhilosophers `Loop`, PlusCal elsif chains).
            // Guard with `has_open_if_condition`: true iff `current` has more
            // top-level `IF`s than `THEN`s so far.
            if indent == top_level_indent && !has_open_if_condition(&current) {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(trimmed.trim_start_matches("/\\").trim_start());
                saw_top_level = true;
                continue;
            }
        }

        if !saw_top_level {
            return None;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() > 1 {
        Some(clauses)
    } else {
        None
    }
}

fn split_indented_action_disjuncts(expr: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_action_indentation(expr);
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut prefix_lines = Vec::new();
    let mut candidate_indents = Vec::new();

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("\\/") {
            let indent = line.len().saturating_sub(trimmed.len());
            candidate_indents.push(indent);
        }
    }

    let Some(base_indent) = candidate_indents.into_iter().min() else {
        return None;
    };
    let mut saw_top_level = false;
    let mut shared_prefix: Option<String> = None;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed.starts_with("\\/") {
            if indent == base_indent {
                let branch_head = trimmed.trim_start_matches("\\/").trim_start();
                if !saw_top_level {
                    let prefix = prefix_lines.join("\n").trim().to_string();
                    let prefix_is_comment_only = is_comment_only_text(&prefix);
                    shared_prefix = repeated_disjunct_prefix(&prefix);
                    if shared_prefix.is_none() && !prefix.is_empty() && !prefix_is_comment_only {
                        return None;
                    }
                    if let Some(prefix) = shared_prefix.as_ref() {
                        current.push_str(prefix);
                    } else if !prefix.is_empty() && !prefix_is_comment_only {
                        let inline_prefix_disjuncts = split_top_level(&prefix, "\\/");
                        if inline_prefix_disjuncts.len() > 1 || prefix.starts_with("\\/") {
                            for disjunct in inline_prefix_disjuncts {
                                let disjunct = disjunct.trim();
                                if !disjunct.is_empty() {
                                    clauses.push(disjunct.to_string());
                                }
                            }
                        } else {
                            clauses.push(prefix);
                        }
                    }
                } else if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                if let Some(prefix) = shared_prefix.as_ref()
                    && current.is_empty()
                {
                    current.push_str(prefix);
                }
                if !current.trim().is_empty() && !branch_head.is_empty() {
                    current.push('\n');
                }
                current.push_str(branch_head);
                saw_top_level = true;
                continue;
            }
        }

        if !saw_top_level {
            prefix_lines.push(line.to_string());
            continue;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() > 1 {
        Some(clauses)
    } else {
        None
    }
}

fn is_comment_only_text(text: &str) -> bool {
    let mut saw_nonempty = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        saw_nonempty = true;
        if !trimmed.starts_with("\\*") {
            return false;
        }
    }
    saw_nonempty
}

fn strip_leading_comment_only_lines(text: &str) -> Option<&str> {
    let mut offset = 0usize;
    let mut saw_comment = false;

    for line in text.lines() {
        let trimmed = line.trim();
        let next_offset = offset + line.len() + 1;
        if trimmed.is_empty() {
            offset = next_offset;
            continue;
        }
        if trimmed.starts_with("\\*") {
            saw_comment = true;
            offset = next_offset;
            continue;
        }
        return if saw_comment {
            Some(text[offset..].trim_start())
        } else {
            Some(text.trim_start())
        };
    }

    None
}

fn repeated_disjunct_prefix(prefix: &str) -> Option<String> {
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.ends_with(':')
        || trimmed.ends_with("THEN")
        || trimmed.ends_with("ELSE")
        || trimmed.ends_with("IN")
    {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn normalize_multiline_action_indentation(expr: &str) -> String {
    let mut lines = expr.lines();
    let Some(first) = lines.next() else {
        return String::new();
    };

    let rest: Vec<&str> = lines.collect();
    if rest.is_empty() {
        return expr.to_string();
    }

    let dedent = rest
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                None
            } else {
                Some(line.len().saturating_sub(trimmed.len()))
            }
        })
        .min()
        .unwrap_or(0);

    if dedent == 0 {
        return expr.to_string();
    }

    let mut normalized = String::with_capacity(expr.len());
    normalized.push_str(first);
    for line in rest {
        normalized.push('\n');
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let indent = line.len().saturating_sub(trimmed.len());
        let keep = indent.saturating_sub(dedent);
        normalized.push_str(&" ".repeat(keep));
        normalized.push_str(trimmed);
    }
    normalized
}

/// A LET body has no closing token, so when a conjunct of the form
/// `<lhs> = LET <defs> IN <body>` (or a bare trailing `LET <defs> IN <body>`)
/// is flattened onto a single line and re-glued with a following ` /\ <next>`,
/// the `<next>` is parsed *inside* `<body>` rather than as a sibling conjunct.
/// This wraps the trailing `LET ... IN <body>` in explicit parentheses so the
/// following glue conjoins at the intended outer level. Only fires when the
/// LET has a genuine (non-empty) IN body — a dangling `... IN` head (the open
/// quantifier/LET prefix that the merge loop stitches onto) is left untouched.
///
/// Note: `find_action_keyword` skips `LET`/`IN` when scanning for a plain
/// keyword, so we locate the LET boundaries via `matches_keyword_at` directly.
fn bound_trailing_let_in_body(part: &str) -> String {
    let chars: Vec<char> = part.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_start: Option<usize> = None; // char index of outermost top-level LET
    let mut let_depth = 0usize;
    let mut in_end: Option<usize> = None; // char index just past the matching top-level IN

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }
        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && matches_keyword_at(&chars, i, "LET") {
            if let_depth == 0 {
                let_start = Some(i);
                in_end = None;
            }
            let_depth += 1;
            i += 3;
            continue;
        }
        if at_top && let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
            let_depth -= 1;
            if let_depth == 0 {
                in_end = Some(i + 2);
            }
            i += 2;
            continue;
        }
        i += 1;
    }

    // The LET body starts just past the matching top-level `IN`. Bound it at the
    // first top-level `/\`/`\/` that follows (the LET body is that single leading
    // expression; anything after the operator is a sibling conjunct/disjunct that
    // must stay *outside* the parentheses). If the IN body is empty the `... IN`
    // is a dangling head (the open quantifier/LET prefix) — leave it untouched.
    if let (Some(ls), Some(ie)) = (let_start, in_end) {
        let body_tail: String = chars[ie..].iter().collect();
        let body_trimmed = body_tail.trim_start();
        if body_trimmed.is_empty() {
            return part.to_string(); // dangling `... IN` head — nothing to bound
        }
        // If the IN body *itself* is a bulleted `/\`/`\/` block, the LET body is
        // that whole boolean block (an action-LET like `LET b == v IN /\ A /\ B`),
        // not a value expression followed by a sibling conjunct. Wrapping here
        // would either enclose the entire action or (if we stopped at the first
        // `/\`) leave an empty LET body. Leave such LETs to the action evaluator.
        if body_trimmed.starts_with("/\\") || body_trimmed.starts_with("\\/") {
            return part.to_string();
        }
        // Scan from the IN body for the first top-level boolean connective.
        let (mut p, mut b, mut br, mut a, mut ld) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut body_end = chars.len();
        let mut j = ie;
        while j < chars.len() {
            let c = chars[j];
            let next = chars.get(j + 1).copied();
            match c {
                '(' => p += 1,
                ')' => p = p.saturating_sub(1),
                '[' => b += 1,
                ']' => b = b.saturating_sub(1),
                '{' => br += 1,
                '}' => br = br.saturating_sub(1),
                '<' if next == Some('<') => {
                    a += 1;
                    j += 2;
                    continue;
                }
                '>' if next == Some('>') => {
                    a = a.saturating_sub(1);
                    j += 2;
                    continue;
                }
                _ => {}
            }
            let at_top = p == 0 && b == 0 && br == 0 && a == 0;
            // Track nested LET inside the body so an inner `\/`/`/\` there does
            // not prematurely close the outer body.
            if at_top && matches_keyword_at(&chars, j, "LET") {
                ld += 1;
                j += 3;
                continue;
            }
            if at_top && ld > 0 && matches_keyword_at(&chars, j, "IN") {
                ld -= 1;
                j += 2;
                continue;
            }
            // `/\` is `'/'` then `'\\'`; `\/` is `'\\'` then `'/'`.
            let is_conjunct = c == '/' && next == Some('\\');
            let is_disjunct = c == '\\' && next == Some('/');
            if at_top && ld == 0 && (is_conjunct || is_disjunct) && j > ie {
                body_end = j;
                break;
            }
            j += 1;
        }

        let before: String = chars[..ls].iter().collect();
        let let_expr: String = chars[ls..body_end].iter().collect();
        let after: String = chars[body_end..].iter().collect();
        let let_expr = let_expr.trim_end();
        let after = after.trim_start();
        if after.is_empty() {
            return format!("{before}({let_expr})");
        }
        return format!("{before}({let_expr}) {after}");
    }

    part.to_string()
}

pub(crate) fn parse_action_if(expr: &str) -> Option<(&str, &str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("IF")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    let then_idx = find_action_keyword(rest, "THEN")?;
    let condition = rest[..then_idx].trim();
    let after_then = rest[then_idx + "THEN".len()..].trim();

    let else_idx = find_action_keyword(after_then, "ELSE")?;
    let then_branch = after_then[..else_idx].trim();
    let else_branch = after_then[else_idx + "ELSE".len()..].trim();

    Some((condition, then_branch, else_branch))
}

pub fn parse_action_exists(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("\\E")?;
    if !rest.starts_with(char::is_whitespace) && !rest.starts_with('(') {
        return None;
    }
    let rest = rest.trim_start();
    let colon_idx = find_action_char(rest, ':')?;
    let binders = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();
    if binders.is_empty() || body.is_empty() {
        return None;
    }
    Some((binders, body))
}

pub fn compile_action_ir(def: &TlaDefinition) -> ActionIr {
    let conjuncts = split_action_body_clauses(&def.body);
    #[cfg(not(feature = "verus"))]
    let clauses_cap = conjuncts.len().max(1);
    #[cfg(feature = "verus")]
    let clauses_cap = crate::storage::verus_smoke::max_usize(conjuncts.len(), 1);
    let mut clauses = Vec::with_capacity(clauses_cap);

    if conjuncts.is_empty() {
        clauses.push(ActionClause::Guard {
            expr: def.body.trim().to_string(),
        });
    } else {
        for part in conjuncts {
            if let Some((binders, body)) = parse_action_exists(&part) {
                clauses.push(ActionClause::Exists {
                    binders: binders.to_string(),
                    body: body.to_string(),
                });
                continue;
            }
            match classify_clause(&part) {
                ClauseKind::PrimedAssignment { var, expr } => {
                    clauses.push(ActionClause::PrimedAssignment { var, expr });
                }
                ClauseKind::PrimedMembership { var, set_expr } => {
                    clauses.push(ActionClause::PrimedMembership { var, set_expr });
                }
                ClauseKind::Unchanged { vars } => {
                    clauses.push(ActionClause::Unchanged { vars });
                }
                _ => {
                    // Check if this is a LET expression with primed assignments
                    let trimmed = part.trim();
                    if trimmed.starts_with("LET") && trimmed.contains('\'') {
                        clauses.push(ActionClause::LetWithPrimes {
                            expr: trimmed.to_string(),
                        });
                    } else {
                        clauses.push(ActionClause::Guard {
                            expr: trimmed.to_string(),
                        });
                    }
                }
            }
        }
    }

    ActionIr {
        name: def.name.clone(),
        params: def.params.clone(),
        clauses,
    }
}

/// Compile an action into one or more IR branches for expression probing.
///
/// `analyze-tla` evaluates action clauses independently. For actions written as
/// a top-level disjunction of conjunctive branches, we need to probe each branch
/// separately instead of feeding raw `\/` separators into `compile_action_ir`.
pub fn compile_action_ir_branches(def: &TlaDefinition) -> Vec<ActionIr> {
    let trimmed = def.body.trim();
    let clauses = split_action_body_clauses(trimmed);
    let clause_options: Vec<Vec<String>> = if clauses.is_empty() {
        vec![vec![trimmed.to_string()]]
    } else {
        clauses
            .iter()
            .map(|clause| {
                let disjuncts = split_action_body_disjuncts(clause);
                if disjuncts.len() > 1 || clause.trim().starts_with("\\/") {
                    disjuncts
                        .into_iter()
                        .map(|disjunct| normalize_branch_clause(&disjunct))
                        .collect()
                } else {
                    vec![clause.trim().to_string()]
                }
            })
            .collect()
    };

    let mut branch_bodies = vec![Vec::<String>::new()];
    for options in clause_options {
        let mut next = Vec::new();
        for existing in &branch_bodies {
            for option in &options {
                let mut branch = existing.clone();
                branch.push(option.clone());
                next.push(branch);
            }
        }
        branch_bodies = next;
    }

    branch_bodies
        .into_iter()
        .map(|clauses| {
            if clauses.len() == 1 {
                clauses[0].trim().to_string()
            } else {
                clauses
                    .into_iter()
                    .filter(|clause| !clause.trim().is_empty())
                    .map(|clause| format_branch_clause(&clause))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        })
        .filter_map(|body| {
            let body = body.trim().to_string();
            if body.is_empty() {
                return None;
            }

            Some(compile_action_ir(&TlaDefinition {
                name: def.name.clone(),
                params: def.params.clone(),
                body,
                is_recursive: def.is_recursive,
            }))
        })
        .collect()
}

fn normalize_branch_clause(clause: &str) -> String {
    clause
        .trim()
        .strip_prefix("/\\")
        .map(str::trim_start)
        .unwrap_or_else(|| clause.trim())
        .to_string()
}

fn format_branch_clause(clause: &str) -> String {
    let mut lines = clause
        .trim()
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty());
    let Some(first) = lines.next() else {
        return String::new();
    };

    let mut formatted = String::new();
    formatted.push_str("/\\ ");
    formatted.push_str(first.trim_start());

    for line in lines {
        formatted.push('\n');
        formatted.push_str("   ");
        formatted.push_str(line);
    }

    formatted
}

pub fn looks_like_action(def: &TlaDefinition) -> bool {
    // Filter out THEOREM/LEMMA/proof definitions that contain primes
    // in proof steps but are not actual action definitions
    let name_upper = def.name.to_uppercase();
    if name_upper.starts_with("THEOREM")
        || name_upper.starts_with("LEMMA")
        || name_upper.starts_with("COROLLARY")
        || name_upper.starts_with("PROPOSITION")
    {
        return false;
    }
    // Proof bodies contain step labels like <1>, BY DEF, QED, SUFFICES
    if def.body.contains("BY DEF ")
        || def.body.contains(" QED")
        || def.body.contains("SUFFICES")
        || def.body.contains("<1>")
    {
        return false;
    }
    let body = strip_double_quoted_strings(&def.body);
    body.contains('\'') || body.contains("UNCHANGED")
}

fn strip_double_quoted_strings(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;

    for ch in text.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }
        out.push(ch);
    }

    out
}

fn find_action_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();

        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top {
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                i += 3;
                continue;
            }
            if let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
                let_depth = let_depth.saturating_sub(1);
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "ELSE") {
                if if_depth == 0 {
                    if keyword == "ELSE" {
                        let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                        return Some(byte_offset);
                    }
                } else {
                    if_depth = if_depth.saturating_sub(1);
                }
                i += 4;
                continue;
            }
        }

        if at_top && let_depth == 0 && if_depth == 0 && matches_keyword_at(&chars, i, keyword) {
            let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            return Some(byte_offset);
        }

        i += 1;
    }

    None
}

fn find_action_char(expr: &str, target: char) -> Option<usize> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();

        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && c == target {
            if target == ':' && next == Some('>') {
                i += 1;
                continue;
            }
            let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            return Some(byte_offset);
        }

        i += 1;
    }

    None
}

fn matches_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (j, kc) in kw_chars.iter().enumerate() {
        if chars[i + j] != *kc {
            return false;
        }
    }
    if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        return false;
    }
    let after = i + kw_chars.len();
    if after < chars.len() && (chars[after].is_alphanumeric() || chars[after] == '_') {
        return false;
    }
    true
}

/// True iff `text` contains an `IF` at top level (outside parens/brackets/
/// braces/angles) that has not yet been closed by a matching `THEN` — i.e. we
/// are still inside an IF *condition*. Used by `split_indented_action_conjuncts`
/// so a base-indent `/\` continuation line of a multi-line IF condition
/// (`IF /\ A\n/\ B\nTHEN ...`) is not mis-read as a sibling top-level conjunct.
/// Nested IFs are counted with a depth counter; each `THEN` closes the most
/// recent `IF` condition. Bracket/angle depth tracking mirrors
/// `find_action_keyword`.
fn has_open_if_condition(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut open_if = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            if matches_keyword_at(&chars, i, "IF") {
                open_if += 1;
                i += 2;
                continue;
            }
            if open_if > 0 && matches_keyword_at(&chars, i, "THEN") {
                open_if -= 1;
                i += 4;
                continue;
            }
        }

        i += 1;
    }

    open_if > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_action_clauses() {
        let def = TlaDefinition {
            name: "Tick".to_string(),
            params: vec![],
            body: "/\\ x' = x + 1 /\\ UNCHANGED <<y>> /\\ x < 10".to_string(),
            is_recursive: false,
        };

        let ir = compile_action_ir(&def);
        assert_eq!(ir.clauses.len(), 3);
        assert!(matches!(
            &ir.clauses[0],
            ActionClause::PrimedAssignment { var, .. } if var == "x"
        ));
        assert!(matches!(ir.clauses[1], ActionClause::Unchanged { .. }));
        assert!(matches!(ir.clauses[2], ActionClause::Guard { .. }));
    }

    #[test]
    fn parses_exists_action_clause() {
        let clause = parse_action_exists(
            "\\E targets \\in SUBSET staleSlots : /\\ Cardinality(targets) = needSlots /\\ writeTargets' = targets",
        )
        .expect("expected existential action clause");
        assert_eq!(clause.0, "targets \\in SUBSET staleSlots");
        assert!(clause.1.contains("writeTargets' = targets"));
    }

    #[test]
    fn compiles_top_level_disjunctive_actions_as_probe_branches() {
        let def = TlaDefinition {
            name: "Receive".to_string(),
            params: vec!["i".to_string()],
            body: r#"
  \/ /\ pc[i] = "SENT"
     /\ x' = x + 1
     /\ UNCHANGED <<y>>
  \/ /\ pc[i] = "SENT"
     /\ y' = y + 1
     /\ UNCHANGED <<x>>
"#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 2);

        for branch in branches {
            assert!(!branch.clauses.is_empty());
            assert!(!branch.clauses.iter().any(
                |clause| matches!(clause, ActionClause::Guard { expr } if expr.trim() == "\\/")
            ));
        }
    }

    #[test]
    fn split_action_body_clauses_keeps_quantified_action_bodies_together() {
        let clauses = split_action_body_clauses(
            r#"
                /\ ready
                /\ \A call \in ActiveCalls :
                    /\ Service(call) =>
                        /\ \E worker \in Workers :
                            /\ worker /= self
                            /\ ServiceBy(worker, call)
                /\ nextFloor \in Floor
                /\ state' = nextState
            "#,
        );

        assert_eq!(clauses.len(), 4);
        assert!(clauses[1].contains(r"\A call \in ActiveCalls"));
        assert!(clauses[1].contains("ServiceBy(worker, call)"));
        assert!(!clauses[1].contains("nextFloor \\in Floor"));
        assert_eq!(clauses[2], "nextFloor \\in Floor");
        assert_eq!(clauses[3], "state' = nextState");
    }

    #[test]
    fn split_action_body_clauses_keeps_exists_bodies_with_nested_conjuncts_together() {
        let clauses = split_action_body_clauses(
            r#"
                /\ \E i \in unchecked[self]:
                    /\ unchecked' = [unchecked EXCEPT ![self] = unchecked[self] \ {i}]
                    /\ IF num[i] > max[self]
                          THEN /\ max' = [max EXCEPT ![self] = num[i]]
                          ELSE /\ TRUE
                               /\ max' = max
                /\ pc' = [pc EXCEPT ![self] = "e2"]
            "#,
        );

        assert_eq!(clauses.len(), 2, "{clauses:#?}");
        assert!(clauses[0].starts_with(r"\E i \in unchecked[self]:"));
        assert!(
            clauses[0]
                .contains(r#"unchecked' = [unchecked EXCEPT ![self] = unchecked[self] \ {i}]"#)
        );
        assert!(clauses[0].contains(r#"max' = [max EXCEPT ![self] = num[i]]"#));
        assert!(clauses[0].contains(r#"max' = max"#));
        assert_eq!(clauses[1], r#"pc' = [pc EXCEPT ![self] = "e2"]"#);
    }

    #[test]
    fn split_action_body_clauses_keeps_set_filter_quantifier_bodies_together() {
        let body = r#"
            /\ maxBal[self] =< b
            /\ \E v \in {vv \in Value :
                          \E Q \in ByzQuorum :
                             \A aa \in Q :
                                \E m \in sentMsgs("2av", b) : /\ m.val = vv
                                                              /\ m.acc = aa}:
                    /\ bmsgs' = (bmsgs \cup {([type |-> "2b", acc |-> self, bal |-> b, val |-> v])})
                    /\ maxVVal' = [maxVVal EXCEPT ![self] = v]
            /\ maxBal' = [maxBal EXCEPT ![self] = b]
        "#;

        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 3, "{clauses:#?}");
        assert_eq!(clauses[0], "maxBal[self] =< b");
        assert!(clauses[1].starts_with(r#"\E v \in {vv \in Value :"#));
        assert!(clauses[1].contains(r#"/\ m.acc = aa}:"#));
        assert!(clauses[1].contains(r#"maxVVal' = [maxVVal EXCEPT ![self] = v]"#));
        assert_eq!(clauses[2], r#"maxBal' = [maxBal EXCEPT ![self] = b]"#);

        let def = TlaDefinition {
            name: "Phase2b".to_string(),
            params: vec!["self".to_string(), "b".to_string()],
            body: body.to_string(),
            is_recursive: false,
        };
        let ir = compile_action_ir(&def);
        assert!(
            matches!(ir.clauses[1], ActionClause::Exists { .. }),
            "{ir:?}"
        );
    }

    #[test]
    fn split_action_body_clauses_ignores_leading_empty_conjuncts() {
        let clauses = split_action_body_clauses(
            r#"
                /\ \/ /\ rmState[self] = "working"
                      /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                   \/ /\ tmState = "commit"
                      /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                /\ pc' = [pc EXCEPT ![self] = "RS"]
            "#,
        );

        assert_eq!(clauses.len(), 2);
        assert!(!clauses.iter().any(|clause| clause.trim().is_empty()));
        assert!(clauses[0].starts_with(r#"\/ /\ rmState[self] = "working""#));
        assert_eq!(clauses[1], r#"pc' = [pc EXCEPT ![self] = "RS"]"#);
    }

    #[test]
    fn split_action_body_clauses_reindents_trimmed_if_then_branches() {
        let clauses = split_action_body_clauses(
            r#"/\ \/ /\ rmState[self] = "working"
                             /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                          \/ /\ \/ /\ tmState="commit"
                                   /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                \/ /\ rmState[self]="working" \/ tmState="abort"
                                   /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                          \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                   THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                   ELSE /\ TRUE
                                        /\ UNCHANGED rmState
                       /\ pc' = [pc EXCEPT ![self] = "RS"]"#,
        );

        assert_eq!(clauses.len(), 2);
        assert!(clauses[0].starts_with(r#"\/ /\ rmState[self] = "working""#));
        assert!(clauses[0].contains(r#"IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed""#));
        assert_eq!(clauses[1], r#"pc' = [pc EXCEPT ![self] = "RS"]"#);
        assert!(!clauses.iter().any(|clause| clause.trim() == r#"\/"#));
    }

    #[test]
    fn split_action_body_clauses_keeps_parser_shaped_if_then_else_together() {
        let clauses = split_action_body_clauses(
            r#"/\ pc[self] = "RS"
            /\ IF rmState[self] \in {"working", "prepared"}
                  THEN /\ \/ /\ rmState[self] = "working"
                             /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                          \/ /\ \/ /\ tmState="commit"
                                   /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                \/ /\ rmState[self]="working" \/ tmState="abort"
                                   /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                          \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                   THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                   ELSE /\ TRUE
                                        /\ UNCHANGED rmState
                       /\ pc' = [pc EXCEPT ![self] = "RS"]
                  ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                       /\ UNCHANGED rmState
            /\ UNCHANGED tmState"#,
        );

        assert_eq!(clauses.len(), 3, "{clauses:#?}");
        assert_eq!(clauses[0], r#"pc[self] = "RS""#);
        assert!(clauses[1].starts_with(r#"IF rmState[self] \in {"working", "prepared"}"#));
        assert!(clauses[1].contains(r#"ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]"#));
        assert_eq!(clauses[2], r#"UNCHANGED tmState"#);
    }

    #[test]
    fn split_action_body_clauses_separates_let_assignment_from_unchanged() {
        let clauses = split_action_body_clauses(
            r#"
                /\ c \in ActiveElevatorCalls
                /\ ElevatorState' =
                    LET closest == CHOOSE e \in stationary \cup approaching :
                        /\ \A e2 \in stationary \cup approaching :
                            /\ GetDistance[ElevatorState[e].floor, c.floor] <= GetDistance[ElevatorState[e2].floor, c.floor]
                    IN
                    IF closest \in stationary
                    THEN [ElevatorState EXCEPT ![closest] = [@ EXCEPT !.floor = c.floor, !.direction = c.direction]]
                    ELSE ElevatorState
                /\ UNCHANGED <<PersonState, ActiveElevatorCalls>>
            "#,
        );

        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], r#"c \in ActiveElevatorCalls"#);
        assert!(clauses[1].starts_with("ElevatorState' ="));
        assert!(clauses[1].contains("LET closest =="));
        assert_eq!(
            clauses[2],
            r#"UNCHANGED <<PersonState, ActiveElevatorCalls>>"#
        );
    }

    #[test]
    fn split_action_body_clauses_keeps_top_level_let_body_conjuncts_together() {
        let clauses = split_action_body_clauses(
            r#"
                /\ x < 5
                /\ y >= -950
                /\ y <= 950
                /\ LET newX == x + 1
                   IN /\ x' = newX
                      /\ y' = y + newX
            "#,
        );

        assert_eq!(clauses.len(), 4, "{clauses:?}");
        assert!(clauses[3].starts_with("LET newX == x + 1"));
        assert!(clauses[3].contains("x' = newX"));
        assert!(clauses[3].contains("y' = y + newX"));
    }

    #[test]
    fn split_action_body_clauses_keeps_comment_prefixed_top_level_disjunction_intact() {
        let clauses = split_action_body_clauses(
            r#"
                \* Need an artificial initial state.
                \* Otherwise the first flip will always be fair.
                \/ /\ state = "init"
                   /\ state' = "s0"
                \/ /\ state # "init"
                   /\ state' = Transition[state][flip]
            "#,
        );

        assert_eq!(clauses.len(), 1, "{clauses:#?}");
        assert!(clauses[0].contains(r#"\/ /\ state = "init""#));
        assert!(clauses[0].contains(r#"\/ /\ state # "init""#));
    }

    #[test]
    fn split_action_body_clauses_splits_guard_before_nested_inline_let() {
        let clauses = split_action_body_clauses(
            r#"
                /\ reqMargin = price
                /\ LET newTrade == [buyer |-> bot, seller |-> s, asset |-> aa, price |-> price]
                       newDeploy == [owner |-> bot, seller |-> s, asset |-> aa, price |-> price]
                   IN
                   /\ ccpTrades' = ccpTrades \union {newTrade}
                   /\ ccpPositions' = [ccpPositions EXCEPT ![bot][aa] = @ + 1, ![s][aa] = @ - 1]
                   /\ deployments' = deployments \union {newDeploy}
                   /\ actionCount' = [actionCount EXCEPT ![bot] = @ + 1]
            "#,
        );

        assert_eq!(clauses.len(), 2, "{clauses:#?}");
        assert_eq!(clauses[0], "reqMargin = price");
        assert!(clauses[1].starts_with("LET newTrade =="), "{clauses:#?}");
        assert!(clauses[1].contains("ccpTrades' = ccpTrades \\union {newTrade}"));
        assert!(clauses[1].contains("actionCount' = [actionCount EXCEPT ![bot] = @ + 1]"));
    }

    #[test]
    fn split_action_body_disjuncts_preserves_boolean_or_inside_branch_guards() {
        let disjuncts = split_action_body_disjuncts(
            r#"/\ \/ /\ tmState="commit"
                 /\ rmState' = [rmState EXCEPT ![self] = "committed"]
              \/ /\ rmState[self]="working" \/ tmState="abort"
                 /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
              \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                       THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                       ELSE /\ TRUE
                            /\ UNCHANGED rmState"#,
        );

        assert_eq!(disjuncts.len(), 3);
        assert!(disjuncts[0].starts_with(r#"/\ tmState="commit""#));
        assert!(disjuncts[0].contains(r#"rmState' = [rmState EXCEPT ![self] = "committed"]"#));
        assert!(disjuncts[1].contains(r#"rmState[self]="working" \/ tmState="abort""#));
        assert!(disjuncts[1].contains(r#"rmState' = [rmState EXCEPT ![self] = "aborted"]"#));
        assert!(
            disjuncts[2].starts_with(r#"/\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed""#)
        );
    }

    #[test]
    fn split_action_body_disjuncts_keeps_nested_quantifier_branches_grouped() {
        let disjuncts = split_action_body_disjuncts(
            r#"\/ \E proc \in Proc, reg \in Reg :
                    \/ \E req \in Request : IssueRequest(proc, req, reg)
                    \/ RespondToRd(proc, reg)
                    \/ RespondToWr(proc, reg)
               \/ Internal"#,
        );

        assert_eq!(disjuncts.len(), 2, "{disjuncts:?}");
        assert!(disjuncts[0].starts_with(r#"\E proc \in Proc, reg \in Reg :"#));
        assert!(disjuncts[0].contains("IssueRequest(proc, req, reg)"));
        assert!(disjuncts[0].contains("RespondToRd(proc, reg)"));
        assert!(disjuncts[0].contains("RespondToWr(proc, reg)"));
        assert_eq!(disjuncts[1], "Internal");
    }

    #[test]
    fn split_action_body_disjuncts_keeps_inner_exists_disjunction_grouped() {
        // Regression (Lamport mutex): `\E p : \E q : A(p,q) \/ B(q,p)` — the
        // `\/` is inside the inner `\E q` scope, so the whole thing is ONE
        // disjunct. A prior bug split it into `\E p : \E q : A(p,q)` and
        // `\E p : B(q,p)` (the inner `\E q :` prefix dropped from the second
        // branch), leaving `q` unbound so that branch produced zero successors.
        let disjuncts = split_action_body_disjuncts(
            r#"\E p \in Proc : \E q \in Proc \ {p} : Recv(p, q) \/ Recv(q, p)"#,
        );
        assert_eq!(disjuncts.len(), 1, "{disjuncts:?}");
        assert!(disjuncts[0].contains("\\E q \\in Proc \\ {p}"));
        assert!(disjuncts[0].contains("Recv(p, q)"));
        assert!(disjuncts[0].contains("Recv(q, p)"));
    }

    #[test]
    fn split_action_body_disjuncts_keeps_exists_conjunction_with_trailing_nested_disjunction() {
        // Regression (EnvironmentController `ProcTick`, ACP `parProgNB`): a
        // `\E`-quantified body that is a top-level CONJUNCTION whose final
        // conjunct is a nested disjunction. The `\/` lives inside the last `/\`
        // conjunct, so the whole `\E i : /\ ... /\ (\/ A \/ B)` is ONE disjunct.
        // A prior bug split it into `\E i : /\ guards...`, `\E i : A`, `\E i : B`
        // — dropping the surrounding guards + shared primed conjuncts, so the
        // orphaned A/B branches fired from every state (even when the guard was
        // false) and later read an unstaged prime, panicking.
        let disjuncts = split_action_body_disjuncts(
            r#"\E i \in Proc :
                  /\ failed[i] = FALSE
                  /\ moved[i] = "NO"
                  /\ procPause' = [procPause EXCEPT ![i] = 0]
                  /\ UNCHANGED << failed >>
                  /\ \/ /\ moved' = [moved EXCEPT ![i] = "PREDICT"]
                        /\ UNCHANGED << inTransit, inDelivery >>
                     \/ /\ moved' = [moved EXCEPT ![i] = "RECEIVE"]
                        /\ CommChan!Deliver(i)"#,
        );
        assert_eq!(
            disjuncts.len(),
            1,
            "exists body that is a conjunction with a nested trailing disjunction must stay one disjunct, got {}: {disjuncts:#?}",
            disjuncts.len()
        );
        assert!(disjuncts[0].contains("moved[i] = \"NO\""), "guard preserved: {}", disjuncts[0]);
        assert!(
            disjuncts[0].contains("procPause' = [procPause EXCEPT ![i] = 0]"),
            "shared primed conjunct preserved: {}",
            disjuncts[0]
        );
    }

    #[test]
    fn split_action_body_disjuncts_splits_two_top_level_disjuncts_with_inner_exists_disjunction() {
        // The top-level `\/` list still splits; only the *inner* `\E q`-scoped
        // `\/` stays grouped. This is the exact Lamport `Next` shape.
        let disjuncts = split_action_body_disjuncts(
            "\\/ \\E p \\in Proc : Request(p)\n\\/ \\E p \\in Proc : \\E q \\in Proc \\ {p} : Recv(p, q) \\/ Recv(q, p)",
        );
        assert_eq!(disjuncts.len(), 2, "{disjuncts:?}");
        assert!(disjuncts[1].contains("\\E q \\in Proc \\ {p}"));
        assert!(disjuncts[1].contains("Recv(p, q)"));
        assert!(disjuncts[1].contains("Recv(q, p)"));
    }

    #[test]
    fn split_action_body_disjuncts_repeats_quantifier_prefix_for_each_branch() {
        let disjuncts = split_action_body_disjuncts(
            r#"\E a \in Acceptor, b \in Ballot :
                  \/ IncreaseMaxBal(a, b)
                  \/ \E v \in Value : VoteFor(a, b, v)"#,
        );

        assert_eq!(disjuncts.len(), 2, "{disjuncts:?}");
        assert_eq!(
            disjuncts[0],
            "\\E a \\in Acceptor, b \\in Ballot :\nIncreaseMaxBal(a, b)"
        );
        assert_eq!(
            disjuncts[1],
            "\\E a \\in Acceptor, b \\in Ballot :\n\\E v \\in Value : VoteFor(a, b, v)"
        );
    }

    #[test]
    fn split_action_body_disjuncts_repeats_outer_let_bindings_for_guard_branches() {
        let disjuncts = split_action_body_disjuncts(
            r#"
                LET eState == ElevatorState[e] IN
                /\ ~eState.doorsOpen
                /\  \/ \E call \in ActiveElevatorCalls : CanServiceCall[e, call]
                    \/ eState.floor \in eState.buttonsPressed
                /\ ElevatorState' = [ElevatorState EXCEPT ![e] = [@ EXCEPT !.doorsOpen = TRUE]]
                /\ UNCHANGED <<PersonState>>
            "#,
        );

        assert_eq!(disjuncts.len(), 2, "{disjuncts:#?}");
        for disjunct in &disjuncts {
            assert!(disjunct.starts_with("LET eState == ElevatorState[e] IN"));
            assert!(disjunct.contains("~eState.doorsOpen"));
            assert!(disjunct.contains("ElevatorState' = [ElevatorState EXCEPT ![e]"));
        }
        assert!(
            disjuncts
                .iter()
                .any(|disjunct| disjunct.contains(r#"\E call \in ActiveElevatorCalls"#))
        );
        assert!(
            disjuncts
                .iter()
                .any(|disjunct| disjunct.contains(r#"eState.floor \in eState.buttonsPressed"#))
        );
    }

    #[test]
    fn split_action_body_disjuncts_ignores_comment_only_prefix_lines() {
        let disjuncts = split_action_body_disjuncts(
            r#"
                \* Need an artificial initial state.
                \* The first flip would otherwise always be fair.
                \/ /\ state = "init"
                   /\ state' = "s0"
                \/ /\ state # "init"
                   /\ state' = Transition[state][flip]
            "#,
        );

        assert_eq!(disjuncts.len(), 2, "{disjuncts:#?}");
        assert!(disjuncts[0].starts_with(r#"/\ state = "init""#));
        assert!(disjuncts[1].starts_with(r#"/\ state # "init""#));
    }

    #[test]
    fn split_action_body_disjuncts_does_not_split_disjunctions_inside_let_definitions() {
        let disjuncts = split_action_body_disjuncts(
            r#"
                LET
                  stationary == {e \in Elevator : ElevatorState[e].direction = "Stationary"}
                  approaching == {e \in Elevator :
                    /\ ElevatorState[e].direction = c.direction
                    /\  \/ ElevatorState[e].floor = c.floor
                        \/ GetDirection[ElevatorState[e].floor, c.floor] = c.direction }
                IN
                /\ c \in ActiveElevatorCalls
                /\ stationary \cup approaching /= {}
                /\ UNCHANGED <<PersonState>>
            "#,
        );

        assert_eq!(disjuncts.len(), 1, "{disjuncts:#?}");
        assert!(disjuncts[0].contains(r#"approaching == {e \in Elevator :"#));
        assert!(disjuncts[0].contains(r#"GetDirection[ElevatorState[e].floor, c.floor]"#));
    }

    #[test]
    fn split_action_body_disjuncts_keeps_if_then_else_clause_intact() {
        let disjuncts = split_action_body_disjuncts(
            r#"IF rmState[self] \in {"working", "prepared"}
               THEN /\ \/ /\ rmState[self] = "working"
                          /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                       \/ /\ tmState = "commit"
                          /\ rmState' = [rmState EXCEPT ![self] = "committed"]
               ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                    /\ UNCHANGED rmState"#,
        );

        assert_eq!(disjuncts.len(), 1, "{disjuncts:?}");
        assert!(disjuncts[0].starts_with("IF rmState[self] \\in"));
        assert!(disjuncts[0].contains("ELSE /\\ pc' = [pc EXCEPT ![self] = \"Done\"]"));
    }

    #[test]
    fn split_action_body_disjuncts_does_not_split_function_constructor_if_bodies() {
        let disjuncts = split_action_body_disjuncts(
            r#"grid' = [p \in Pos |-> IF \/  (grid[p] /\ score(p) \in {2, 3})
                                  \/ (~grid[p] /\ score(p) = 3)
                                THEN TRUE
                                ELSE FALSE]"#,
        );

        assert_eq!(disjuncts.len(), 1, "{disjuncts:?}");
        assert!(disjuncts[0].starts_with("grid' = [p \\in Pos |-> IF"));
        assert!(disjuncts[0].contains(r#"(~grid[p] /\ score(p) = 3)"#));
        assert!(disjuncts[0].ends_with("ELSE FALSE]"));
    }

    #[test]
    fn split_action_body_disjuncts_does_not_split_nested_disjunction_inside_outer_conjunction() {
        // Regression for T1.5 soundness bug. The body is a top-level conjunction
        //   /\ guard
        //   /\ \/ A
        //      \/ B
        //   /\ shared_post
        // where the inner `\/` is NOT the outer connective. Splitting on `\/` here
        // would (a) fabricate a stutter half-clause (`/\` with no body), (b) drop the
        // surrounding `guard` from one branch, and (c) drop the `shared_post`
        // from another. The whole body must be returned as ONE disjunct so
        // downstream IR compilation can expand the nested disjunction with the
        // shared clauses correctly applied.
        let body = r#"
            /\ x + y < 15
            /\ \/ /\ x < 10
                  /\ x' = x + 1
                  /\ y' = y
               \/ /\ y < 10
                  /\ y' = y + 1
                  /\ x' = x
            /\ timestamp' = timestamp + 1
        "#;

        let disjuncts = split_action_body_disjuncts(body);
        assert_eq!(
            disjuncts.len(),
            1,
            "expected exactly one disjunct (the whole conjunctive action), \
             got {} disjuncts: {disjuncts:#?}",
            disjuncts.len()
        );

        let only = &disjuncts[0];
        assert!(
            only.contains("x + y < 15"),
            "the outer guard must be preserved in the single disjunct: {only}"
        );
        assert!(
            only.contains("timestamp' = timestamp + 1"),
            "the shared post-condition must be preserved in the single disjunct: {only}"
        );
        assert!(
            only.contains("x' = x + 1"),
            "the first inner disjunct's primed assignment must be preserved: {only}"
        );
        assert!(
            only.contains("y' = y + 1"),
            "the second inner disjunct's primed assignment must be preserved: {only}"
        );
        assert!(
            !only.trim().is_empty(),
            "the single disjunct must not collapse to empty: {only:?}"
        );

        // And via the IR pipeline, this body should expand into exactly two
        // branches (one per inner disjunct), each carrying both the outer guard
        // and the shared post-condition.
        let def = TlaDefinition {
            name: "Next".to_string(),
            params: vec![],
            body: body.to_string(),
            is_recursive: false,
        };
        let branches = compile_action_ir_branches(&def);
        assert_eq!(
            branches.len(),
            2,
            "expected two compiled branches from inner disjunction, got {}: {branches:#?}",
            branches.len()
        );
        for branch in &branches {
            let exprs: Vec<String> = branch
                .clauses
                .iter()
                .map(|clause| match clause {
                    ActionClause::Guard { expr }
                    | ActionClause::PrimedAssignment { expr, .. }
                    | ActionClause::PrimedMembership { set_expr: expr, .. }
                    | ActionClause::LetWithPrimes { expr } => expr.clone(),
                    ActionClause::Exists { body, .. } => body.clone(),
                    ActionClause::Unchanged { vars } => vars.join(","),
                })
                .collect();
            assert!(
                exprs.iter().any(|e| e.contains("x + y < 15")),
                "every branch must carry the outer guard: {exprs:?}"
            );
            assert!(
                exprs.iter().any(|e| e.contains("timestamp + 1")),
                "every branch must carry the shared post-condition: {exprs:?}"
            );
        }
    }

    #[test]
    fn split_action_body_disjuncts_keeps_quantified_conjunctive_body_intact() {
        // Regression for NanoBlockchain `ProcessBlock`. The body is a single
        // conjunct that is a `\E`-quantified action whose body is itself a
        // conjunction with a *nested* disjunction:
        //   /\ \E block \in received[node] :
        //          /\ \/ ProcessOpenBlock(node, block)
        //             \/ ProcessSendBlock(node, block)
        //          /\ received' = [received EXCEPT ![node] = @ \ {block}]
        // The `\/` lives inside the `\E block` scope; splitting on it orphans the
        // `block` binder (it resolves to `ModelValue("block")` → record access on a
        // non-record → the whole ProcessBlock branch is silently dropped). The
        // whole quantified clause must stay ONE disjunct so the per-branch exists
        // expansion binds `block` before evaluating the inner disjunction.
        let body = r#"
            /\ \E block \in received[node] :
                /\  \/ ProcessOpenBlock(node, block)
                    \/ ProcessSendBlock(node, block)
                    \/ ProcessReceiveBlock(node, block)
                    \/ ProcessChangeRepBlock(node, block)
                /\ received' = [received EXCEPT ![node] = @ \ {block}]
        "#;
        let disjuncts = split_action_body_disjuncts(body);
        assert_eq!(
            disjuncts.len(),
            1,
            "quantified conjunctive body must stay one disjunct, got {}: {disjuncts:#?}",
            disjuncts.len()
        );
        assert!(
            disjuncts[0].contains("\\E block \\in received[node]"),
            "the quantifier binder must be preserved: {}",
            disjuncts[0]
        );
        assert!(
            disjuncts[0].contains("received' = [received EXCEPT ![node] = @ \\ {block}]"),
            "the shared post-condition must be preserved: {}",
            disjuncts[0]
        );
    }

    #[test]
    fn bound_trailing_let_in_body_wraps_assignment_rhs_let() {
        // A primed-assignment RHS that is `LET ... IN <value>` followed by a
        // sibling conjunct must have its LET body parenthesized so the sibling
        // `/\` is not swallowed into the (closing-token-less) LET body.
        let part =
            "received' = LET sb == [block |-> b] IN [n \\in Node |-> received[n] \\cup {sb}] \
             /\\ UNCHANGED distributedLedger";
        let bounded = bound_trailing_let_in_body(part);
        assert_eq!(
            bounded,
            "received' = (LET sb == [block |-> b] IN [n \\in Node |-> received[n] \\cup {sb}]) \
             /\\ UNCHANGED distributedLedger"
        );
    }

    #[test]
    fn bound_trailing_let_in_body_leaves_action_let_untouched() {
        // A `LET x == v IN /\ A /\ B` is an action-LET whose IN body is the whole
        // bulleted block — it must NOT be parenthesized (doing so would either
        // enclose the whole action or leave an empty LET body).
        let part = "LET nb == [previous |-> p] IN /\\ ValidateSendBlock(l, nb) /\\ received' = nb";
        assert_eq!(bound_trailing_let_in_body(part), part);
    }

    #[test]
    fn bound_trailing_let_in_body_leaves_dangling_in_head_untouched() {
        // The open quantifier/LET prefix the merge loop stitches onto ends with a
        // bare `... IN` (no body yet) — nothing to bound.
        let part = "\\E x \\in S : LET nb == [a |-> 1] IN";
        assert_eq!(bound_trailing_let_in_body(part), part);
    }

    #[test]
    fn compile_action_ir_branches_expands_nested_disjunctions_with_shared_clauses() {
        let def = TlaDefinition {
            name: "e1".to_string(),
            params: vec!["self".to_string()],
            body: r#"
                /\ pc[self] = "e1"
                /\ \/ /\ flag' = [flag EXCEPT ![self] = ~ flag[self]]
                      /\ pc' = [pc EXCEPT ![self] = "e1"]
                   \/ /\ flag' = [flag EXCEPT ![self] = TRUE]
                      /\ unchecked' = [unchecked EXCEPT ![self] = Procs \ {self}]
                      /\ pc' = [pc EXCEPT ![self] = "e2"]
                /\ UNCHANGED num
            "#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 2);
        for branch in branches {
            let texts: Vec<String> = branch
                .clauses
                .into_iter()
                .map(|clause| match clause {
                    ActionClause::Guard { expr }
                    | ActionClause::PrimedAssignment { expr, .. }
                    | ActionClause::PrimedMembership { set_expr: expr, .. }
                    | ActionClause::LetWithPrimes { expr } => expr,
                    ActionClause::Exists { body, .. } => body,
                    ActionClause::Unchanged { vars } => vars.join(","),
                })
                .collect();
            assert!(texts.iter().any(|expr| expr.contains(r#"pc[self] = "e1""#)));
            assert!(
                texts
                    .iter()
                    .any(|expr| expr.contains(r#"pc EXCEPT ![self]"#))
            );
        }
    }

    #[test]
    fn compile_action_ir_branches_ignores_comment_only_lines_before_top_level_disjuncts() {
        let def = TlaDefinition {
            name: "SimNext".to_string(),
            params: vec![],
            body: r#"
                \* Need an artificial initial state to be able to model a crooked coin.
                \* Otherwise the first flip will always be fair.
                \/ /\ state = "init"
                   /\ state' = "s0"
                   /\ UNCHANGED p
                \/ /\ state # "init"
                   /\ state' = Transition[state][flip]
                   /\ p' = Half(p)
            "#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 2, "{branches:#?}");
        for branch in branches {
            assert!(
                branch.clauses.iter().all(|clause| match clause {
                    ActionClause::Guard { expr } => !expr.trim().starts_with("\\*"),
                    _ => true,
                }),
                "{branch:#?}"
            );
        }
    }

    #[test]
    fn compile_action_ir_branches_reindents_multiline_let_clauses() {
        let def = TlaDefinition {
            name: "Increment".to_string(),
            params: vec![],
            body: r#"
                /\ x < 5
                /\ y >= -950
                /\ y <= 950
                /\ LET newX == x + 1
                   IN /\ x' = newX
                      /\ y' = y + newX
            "#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].clauses.len(), 4);
        match &branches[0].clauses[3] {
            ActionClause::LetWithPrimes { expr } => {
                assert!(expr.contains("LET newX == x + 1"));
                assert!(expr.contains("x' = newX"));
                assert!(expr.contains("y' = y + newX"));
            }
            other => panic!("expected LET action clause, got {other:?}"),
        }
    }

    #[test]
    fn compile_action_ir_branches_keeps_multiline_if_clause_and_following_siblings_separate() {
        let def = TlaDefinition {
            name: "CounterAction".to_string(),
            params: vec!["p".to_string()],
            body: r#"
              /\ p = DesignatedCounter
              /\ IF light = "on"
                 THEN
                   /\ light' = "off"
                   /\ count' = count + 1
                 ELSE
                   UNCHANGED <<light, count>>
              /\ announced' = (count' >= VictoryThreshold)
              /\ UNCHANGED <<signalled>>
            "#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].clauses.len(), 4);
        match &branches[0].clauses[1] {
            ActionClause::Guard { expr } => {
                assert!(expr.contains(r#"IF light = "on""#));
                assert!(expr.contains(r#"count' = count + 1"#));
                assert!(expr.contains(r#"UNCHANGED <<light, count>>"#));
                assert!(!expr.contains(r#"announced' = (count' >= VictoryThreshold)"#));
            }
            other => panic!("expected IF guard clause, got {other:?}"),
        }
        match &branches[0].clauses[2] {
            ActionClause::PrimedAssignment { var, expr } => {
                assert_eq!(var, "announced");
                assert_eq!(expr, "(count' >= VictoryThreshold)");
            }
            other => panic!("expected announced assignment, got {other:?}"),
        }
    }

    #[test]
    fn looks_like_action_ignores_apostrophes_inside_strings() {
        let def = TlaDefinition {
            name: "Defs".to_string(),
            params: vec![],
            body: "\"<defs><marker id='arrow'></marker></defs>\"".to_string(),
            is_recursive: false,
        };

        assert!(!looks_like_action(&def));
    }

    #[test]
    fn looks_like_action_still_detects_real_primes_outside_strings() {
        let def = TlaDefinition {
            name: "NowNext".to_string(),
            params: vec![],
            body: "/\\ now' \\in 0..10 /\\ UNCHANGED hr".to_string(),
            is_recursive: false,
        };

        assert!(looks_like_action(&def));
    }

    // ---- Phase 4: split_action_conjuncts_v2 (expr_v2 action-conjunct split) ----

    // v2 is the default parser under test (TLAPLUS_EXPR_PARSER unset). These
    // tests call `split_action_conjuncts_v2` directly to assert the ROOT-`And`
    // grouping, and `split_action_body_clauses` (the wired entry) to assert the
    // v2/fallback contract end-to-end.

    #[test]
    fn v2_nanoblockchain_let_then_unchanged_two_siblings() {
        // THE CRUX (NanoBlockchain #148): a primed-assignment whose RHS is a
        // (closing-token-less) `LET ... IN <func>` followed by `/\ UNCHANGED`.
        // The end result MUST be TWO sibling conjuncts — the LET body must NOT
        // swallow the UNCHANGED.
        //
        // FINDING: v2 does NOT own this shape. The v2 LET-body extent is greedy
        // (`x' = LET .. IN [func]` reaches to end-of-input), so `parse_ast` fails
        // its full-consumption check with "trailing tokens after expr near Let"
        // → `split_action_conjuncts_v2` returns `None` → the STRING FALLBACK
        // runs, whose `bound_trailing_let_in_body` hack correctly parenthesizes
        // the LET body so the trailing `/\` conjoins at the outer level. So v2's
        // contract here is "do no harm": return `None` and let the existing
        // (correct) fallback own it. The two-sibling result is asserted
        // end-to-end via `split_action_body_clauses`.
        let body = "/\\ received' = LET sb == [block |-> b] IN [n \\in Node |-> received[n] \\cup {sb}]\n\
                    /\\ UNCHANGED distributedLedger";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "v2 does not own the greedy-LET-body shape; must fall back to the string path"
        );
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 2, "expected two siblings end-to-end, got: {clauses:#?}");
        assert!(
            clauses[0].starts_with("received' =") && clauses[0].contains("IN"),
            "first conjunct must keep its LET body intact: {}",
            clauses[0]
        );
        assert!(
            clauses[1].contains("UNCHANGED distributedLedger")
                && !clauses[1].contains("received'"),
            "second conjunct must be the standalone UNCHANGED: {}",
            clauses[1]
        );
    }

    #[test]
    fn v2_bare_if_is_not_an_and_root_falls_back_whole() {
        // `IF /\ A /\ B THEN ... ELSE ...` as one clause: root is `If`, not
        // `And` → v2 returns None → the whole body stays one clause via the
        // existing IF special-case.
        let body = "IF /\\ A /\\ B THEN x' = 1 ELSE x' = 2";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "bare IF root must not be split by v2"
        );
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 1, "bare IF must remain one clause: {clauses:#?}");
        assert!(clauses[0].starts_with("IF"));
    }

    #[test]
    fn v2_exists_body_does_not_leak_two_siblings() {
        // `/\ \E x \in S : /\ P(x) /\ Q(x)` newline `/\ y' = 2` — two conjuncts;
        // the `\E`'s 2-conjunct body must stay inside the existential.
        let body = "/\\ \\E x \\in S : /\\ P(x)\n\
                    \x20                  /\\ Q(x)\n\
                    /\\ y' = 2";
        let parts = split_action_conjuncts_v2(body).expect("root should be an And junction");
        assert_eq!(parts.len(), 2, "expected two siblings, got: {parts:#?}");
        assert!(
            parts[0].starts_with("\\E x \\in S")
                && parts[0].contains("P(x)")
                && parts[0].contains("Q(x)"),
            "the \\E body must not leak: {}",
            parts[0]
        );
        assert!(parts[1].contains("y' = 2") && !parts[1].contains("Q(x)"), "{}", parts[1]);
    }

    #[test]
    fn v2_bare_exists_root_falls_back() {
        // `\E x \in S : A(x) \/ B(x)` alone — root is Quant → None → fallback
        // (must not orphan the binder `x`).
        let body = "\\E x \\in S : A(x) \\/ B(x)";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "bare \\E root must not be split by v2"
        );
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 1, "bare \\E must remain one clause: {clauses:#?}");
        assert!(clauses[0].starts_with("\\E x \\in S"));
    }

    #[test]
    fn v2_inner_disjunction_stays_one_conjunct() {
        // `/\ guard /\ \/ A \/ B /\ post` — the `\/ A \/ B` stays as ONE
        // conjunct (a disjunctive sub-action); guard and post are siblings.
        let body = "/\\ guard\n\
                    /\\ \\/ A\n\
                    \x20  \\/ B\n\
                    /\\ post";
        let parts = split_action_conjuncts_v2(body).expect("root should be an And junction");
        assert_eq!(parts.len(), 3, "expected [guard, (\\/A \\/B), post], got: {parts:#?}");
        assert_eq!(parts[0], "guard");
        assert!(
            parts[1].contains("\\/ A") && parts[1].contains("\\/ B"),
            "the inner disjunction must stay one conjunct: {}",
            parts[1]
        );
        assert_eq!(parts[2], "post");
    }

    #[test]
    fn v2_simple_leaf_conjuncts_split() {
        // Leaves: primed-assignment, membership, EXCEPT, UNCHANGED, guard.
        let body = "/\\ x' = [f EXCEPT ![k] = v] /\\ y' \\in S /\\ UNCHANGED <<z, w>> /\\ x < 10";
        let parts = split_action_conjuncts_v2(body).expect("root should be an And junction");
        assert_eq!(parts.len(), 4, "got: {parts:#?}");
        assert_eq!(parts[0], "x' = [f EXCEPT ![k] = v]");
        assert_eq!(parts[1], "y' \\in S");
        assert_eq!(parts[2], "UNCHANGED <<z, w>>");
        assert_eq!(parts[3], "x < 10");
    }

    #[test]
    fn v2_single_conjunct_body_falls_back() {
        // A single-conjunct (or bulletless) body is not a >=2-item And root.
        assert!(split_action_conjuncts_v2("x' = x + 1").is_none());
        assert!(split_action_conjuncts_v2("/\\ x' = x + 1").is_none());
    }

    #[test]
    fn v2_leading_bullet_stripped_no_double_bullet() {
        // Confirm the sliced item does NOT retain a leading `/\` bullet (item
        // spans start after the bullet in `parse_junction`).
        let body = "/\\ a' = 1 /\\ b' = 2";
        let parts = split_action_conjuncts_v2(body).unwrap();
        for p in &parts {
            assert!(!p.starts_with("/\\"), "conjunct retained a bullet: {p}");
        }
        assert_eq!(parts, vec!["a' = 1".to_string(), "b' = 2".to_string()]);
    }

    #[test]
    fn v2_falls_back_on_parse_error_matches_string_path() {
        // A body v2 cannot parse must yield None (→ identical to pre-Phase-4).
        // An unterminated block comment forces a lex error → Err → None.
        let body = "/\\ x' = 1 (* unterminated";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "lex/parse error must fall back to None"
        );
    }

    #[test]
    fn v2_or_led_shallow_leading_bullet_body_falls_back() {
        // SOUNDNESS REGRESSION (EWD998PCal `node`, PlusCal `Op(self) == \/ ...`).
        // A `\/`-led disjunction whose leading `\/` sits SHALLOWER than its sibling
        // `\/` bullets (the `node(self) == \/ /\ A` shape: line-1 `\/` trails the
        // stripped `== ` at col 0, continuation `\/`s indented) makes v2 mis-fence
        // — the shallow leading `\/` collapses and the last `\E` body swallows the
        // deeper sibling disjuncts, so `parse_ast` returns a ROOT `And`. Conjunct-
        // splitting THAT And shreds the multi-disjunct action (EWD998PCal N=3
        // under-explored 321,370 → ~11k distinct, a false-safe). The disjunction-
        // body guard must return None so the string splitter (correct) owns it.
        let body = "\\/ /\\ active[self]\n   /\\ \\E to \\in Node \\ {self}:\n        network' = f(to)\n   /\\ counter' = g\n              \\/ /\\ \\E msg \\in P:\n                 /\\ network' = h(msg)\n              \\/ /\\ active' = FALSE\n                 /\\ UNCHANGED net";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "\\/-led body must fall back to the string splitter, not be conjunct-split"
        );
        // And end-to-end: the disjunct splitter must recover EXACTLY the three
        // top-level `\/` branches — the action's real successors — not collapse
        // into a swallowed \E body (under-split) and not shred an inner grouped
        // clause into extra siblings (over-split). Assert the exact count AND the
        // distinguishing leading text of each branch so a wrong boundary fails.
        let disj = split_action_body_disjuncts(body);
        assert_eq!(
            disj.len(),
            3,
            "\\/-led body must split into its 3 real disjuncts, got {}: {disj:#?}",
            disj.len()
        );
        // The leading `\/` bullet is stripped; each branch's body opens with `/\`.
        // Match on the first distinguishing token of each branch.
        assert!(
            disj[0].contains("active[self]") && disj[0].contains("network' = f(to)"),
            "branch 0 must be the active-node send branch: {}",
            disj[0]
        );
        assert!(
            disj[1].contains("\\E msg \\in P") && disj[1].contains("network' = h(msg)"),
            "branch 1 must be the receive branch: {}",
            disj[1]
        );
        assert!(
            disj[2].contains("active' = FALSE") && disj[2].contains("UNCHANGED net"),
            "branch 2 must be the deactivate branch: {}",
            disj[2]
        );
        // No branch may leak content from a sibling (over-split / mis-fence guard).
        assert!(
            !disj[0].contains("network' = h(msg)") && !disj[0].contains("active' = FALSE"),
            "branch 0 swallowed a sibling disjunct: {}",
            disj[0]
        );
    }

    // ---- Phase 4.1: D1 (uniform dedent) + D2 (AST And-flatten) ----

    #[test]
    fn v2_d1_single_bulleted_quantified_body_stays_inside_exists() {
        // FIX D1 — Codex's motivating shape. Line 1 (`/\ A /\ \E x \in S :`) is
        // at col 0 while the `\E` body bullets `P`/`Q` sit at col 7 (no top-level
        // sibling follows them). This is a layout where line-1's leading bullet
        // (col 0) is SHALLOWER than the root junction fence the parser installs,
        // so the LAYOUT-CONSISTENCY GUARD routes it to the string fallback — which
        // groups it correctly. The Codex requirement (P/Q NOT hoisted out of the
        // `\E`) is asserted end-to-end via `split_action_body_clauses`: EXACTLY
        // two conjuncts, `A` and the whole `\E x \in S : /\ P(x) /\ Q(x)`.
        let body = "/\\ A /\\ \\E x \\in S :\n\
                    \x20      /\\ P(x)\n\
                    \x20      /\\ Q(x)";
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 2, "P/Q must NOT be hoisted to top level: {clauses:#?}");
        assert_eq!(clauses[0], "A");
        assert!(
            clauses[1].starts_with("\\E x \\in S")
                && clauses[1].contains("P(x)")
                && clauses[1].contains("Q(x)"),
            "the \\E body must keep P/Q: {}",
            clauses[1]
        );
    }

    #[test]
    fn v2_d1_indented_line1_uniform_dedent_preserves_layout() {
        // FIX D1 — line 1 is itself INDENTED (mid-definition extraction / raw
        // string literal). `uniform_dedent` rebases the whole block by the shared
        // min-indent (8) so the shallowest line lands at col 0 while every
        // relative offset is preserved — the layout the parser needs. As in the
        // col-0 variant, line-1's leading bullet is shallower than the root fence,
        // so the consistency guard defers to the (correct) string fallback.
        // Assert end-to-end: two conjuncts with P/Q inside the `\E`.
        let body = "        /\\ A /\\ \\E x \\in S :\n\
                    \x20              /\\ P(x)\n\
                    \x20              /\\ Q(x)";
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 2, "{clauses:#?}");
        assert_eq!(clauses[0], "A");
        assert!(
            clauses[1].starts_with("\\E x \\in S") && clauses[1].contains("Q(x)"),
            "{}",
            clauses[1]
        );
    }

    #[test]
    fn v2_d2_assignment_rhs_exists_inline_and_not_split() {
        // FIX D2 — `/\ b' = \E x \in S : P(x) /\ Q(x)` newline `/\ y' = y + 1`.
        // The inline `/\` after `P(x)` belongs to the `\E` body. v2 does NOT own
        // this shape: the greedy `\E`-body extent reaches to end-of-input, so
        // `parse_ast` fails its full-consumption check ("trailing tokens after
        // expr near Exists") → `split_action_conjuncts_v2` returns None → the
        // STRING FALLBACK owns it. The end-to-end contract (asserted via
        // `split_action_body_clauses`) is exactly what Codex requires: EXACTLY
        // two conjuncts, the `b' = \E ...` intact with its inline `/\` NOT split.
        let body = "/\\ b' = \\E x \\in S : P(x) /\\ Q(x)\n\
                    /\\ y' = y + 1";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "v2 must not own the greedy-\\E-body RHS shape; fall back to string path"
        );
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 2, "inline \\E-body /\\ must NOT split: {clauses:#?}");
        assert_eq!(clauses[0], "b' = \\E x \\in S : P(x) /\\ Q(x)");
        assert_eq!(clauses[1], "y' = y + 1");
    }

    #[test]
    fn v2_d2_assignment_rhs_if_inline_and_not_split() {
        // FIX D2 — `/\ b' = IF c THEN P /\ Q ELSE R` newline `/\ y' = 2`. The
        // `/\` in the THEN branch belongs to the `IF` body. Like the `\E` case,
        // v2's greedy IF-body extent fails full consumption ("trailing tokens
        // after expr near If") → None → the string fallback owns it. End-to-end
        // (via `split_action_body_clauses`) the two conjuncts are correct and the
        // IF-branch inline `/\` stays intact.
        let body = "/\\ b' = IF c THEN P /\\ Q ELSE R\n\
                    /\\ y' = 2";
        assert!(
            split_action_conjuncts_v2(body).is_none(),
            "v2 must not own the greedy-IF-body RHS shape; fall back to string path"
        );
        let clauses = split_action_body_clauses(body);
        assert_eq!(clauses.len(), 2, "IF-branch /\\ must NOT split: {clauses:#?}");
        assert_eq!(clauses[0], "b' = IF c THEN P /\\ Q ELSE R");
        assert_eq!(clauses[1], "y' = 2");
    }

    #[test]
    fn v2_d2_inline_guard_and_assignment_still_split() {
        // FIX D2 (the MOTIVATING split that MUST still happen) — a guard AND a
        // primed assignment on the SAME line: `/\ S # {} /\ unsat' = [...]`.
        // Both `S # {}` and `unsat' = [...]` are And-junction children (the top
        // `/\` is a real conjunction, not a sub-expression body), so AST
        // flattening splits them into TWO conjuncts. (If they were NOT split,
        // `unsat'` would be misread as `unsat' = (S # {} /\ [...])`, type-error,
        // and every successor would be silently dropped — SimpleAllocator 1 vs
        // the correct 400.)
        let body = "/\\ S # {} /\\ unsat' = [x \\in S |-> 0]";
        let parts = split_action_conjuncts_v2(body).expect("root should be an And junction");
        assert_eq!(parts.len(), 2, "guard and assignment must be SEPARATE conjuncts: {parts:#?}");
        assert_eq!(parts[0], "S # {}");
        assert_eq!(parts[1], "unsat' = [x \\in S |-> 0]");
    }

    #[test]
    fn v2_uniform_dedent_is_noop_when_first_line_at_col0() {
        // A def body whose line 1 is already at col 0 with indented
        // continuations dedents by 0 — the shared min-indent is 0.
        let body = "/\\ a' = 1\n       /\\ b' = 2";
        assert_eq!(uniform_dedent(body), body);
    }

    #[test]
    fn v2_uniform_dedent_rebases_by_shared_min_indent() {
        // Every line indented by >= 4; uniform_dedent strips exactly 4 from all,
        // preserving the relative offset of the deeper continuation.
        let input = "    /\\ A\n        /\\ B";
        assert_eq!(uniform_dedent(input), "/\\ A\n    /\\ B");
    }

    // ---- Phase 5: classify_action_disjunct_v2 (expr_v2 action-DISJUNCT split) ----

    // Small matchers to keep the 4-state assertions terse.
    fn disj_v2_is_parsed_non_root_or(expr: &str) -> bool {
        matches!(classify_action_disjunct_v2(expr), DisjunctSplitV2::ParsedNonRootOr)
    }
    // "Conservative, does not split": either `ParsedNonRootOr` (root is a
    // scoped/leaf node) or `RootOrLayoutMismatch` (root Or but a layout /
    // conjunction-body mis-fence was rejected). Both keep the body whole.
    fn disj_v2_conservative(expr: &str) -> bool {
        matches!(
            classify_action_disjunct_v2(expr),
            DisjunctSplitV2::ParsedNonRootOr | DisjunctSplitV2::RootOrLayoutMismatch
        )
    }
    fn disj_v2_split(expr: &str) -> Option<Vec<String>> {
        match classify_action_disjunct_v2(expr) {
            DisjunctSplitV2::Split(v) => Some(v),
            _ => None,
        }
    }

    #[test]
    fn v2_disj_bare_root_or_splits_three() {
        // `\/ A \/ B \/ C` → root Or → 3 disjuncts via v2.
        let body = "\\/ A\n\\/ B\n\\/ C";
        let parts = disj_v2_split(body).expect("root should be an Or junction");
        assert_eq!(parts, vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        // End-to-end through the wired entry.
        assert_eq!(split_action_body_disjuncts(body), vec!["A", "B", "C"]);
    }

    #[test]
    fn v2_disj_inline_root_or_splits() {
        // Single-line inline `A \/ B \/ C` → root Or, no layout guard reject.
        let body = "A \\/ B \\/ C";
        let parts = disj_v2_split(body).expect("root should be an Or junction");
        assert_eq!(parts, vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    #[test]
    fn v2_disj_nested_exists_is_parsed_non_root_or() {
        // Lamport-mutex nested exists: root is `Quant` → ParsedNonRootOr → NOT
        // split by v2. The existing `\E` path keeps the inner `q` in scope.
        let body = "\\E p \\in P : \\E q \\in Q : A(p, q) \\/ B(p, q)";
        assert!(disj_v2_is_parsed_non_root_or(body));
        // End-to-end: nested-quantifier guard keeps it whole (binder not orphaned).
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "nested \\E must stay one disjunct: {out:#?}");
        assert!(out[0].starts_with("\\E p \\in P"));
    }

    #[test]
    fn v2_disj_bare_exists_is_parsed_non_root_or() {
        // `\E x \in S : A(x) \/ B(x)` → root Quant → ParsedNonRootOr. The
        // existing `\E`-prefix-repeat path gives 2 branches each with `x`.
        let body = "\\E x \\in S : A(x) \\/ B(x)";
        assert!(disj_v2_is_parsed_non_root_or(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 2, "expected 2 \\E-prefixed branches: {out:#?}");
        assert!(out[0].starts_with("\\E x \\in S") && out[0].contains("A(x)"));
        assert!(out[1].starts_with("\\E x \\in S") && out[1].contains("B(x)"));
    }

    #[test]
    fn v2_disj_conjunction_root_stays_whole() {
        // `/\ guard /\ \/ A \/ B /\ post` — a `/\`-led CONJUNCTION. v2 classifies
        // it conservatively (root `And` → `ParsedNonRootOr`, or an inline-`\/`
        // mis-fence caught by the conjunction-body guard → `RootOrLayoutMismatch`)
        // — either way it never `Split`s, and the `/\`-conjunction guard keeps it
        // ONE action (the inner `\/` is one conjunct).
        let body = "/\\ guard\n/\\ \\/ A\n   \\/ B\n/\\ post";
        assert!(disj_v2_conservative(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "conjunction must stay one action: {out:#?}");
    }

    #[test]
    fn v2_disj_conjunction_with_inline_boolean_or_guard_stays_whole() {
        // SOUNDNESS REGRESSION GUARD (Peterson `a3`): a `/\`-led conjunction whose
        // SECOND conjunct is an inline boolean `\/` (`~c[Other] \/ turn = self`).
        // expr_v2 fences that inline `\/` as the looser OUTER operator, parsing a
        // ROOT `Or` that would shred the bulleted `/\` list across the boolean
        // `\/` — crossing conjunct #1's guard with a later branch's effect and
        // inventing successors (Peterson's phantom `a0 -> cs`, a false
        // mutual-exclusion violation). The conjunction-body guard must reject the
        // v2 `Split` (first logical line starts with `/\`) → whole body, one
        // action.
        let body = "/\\ pc[self] = \"a3\"\n\
                    /\\ ~c[Other(self)] \\/ turn = self\n\
                    /\\ pc' = [pc EXCEPT ![self] = \"cs\"]\n\
                    /\\ UNCHANGED << c, turn >>";
        assert!(disj_v2_split(body).is_none(), "must NOT split the a3 conjunction");
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "a3 conjunction must stay one action: {out:#?}");
        assert!(out[0].contains("pc[self] = \"a3\""));
        assert!(out[0].contains("~c[Other(self)] \\/ turn = self"));
        assert!(out[0].contains("pc' = [pc EXCEPT ![self] = \"cs\"]"));
    }

    #[test]
    fn v2_disj_root_let_classifies_non_root_or() {
        // root `LET d == e IN A \/ B` → v2 root is `Let` → `ParsedNonRootOr`
        // (v2 never blind-splits the inner `\/`). End-to-end, the EXISTING
        // `split_outer_let_body_disjuncts` path (which runs BEFORE the Phase-5
        // whole-body protection) intentionally DISTRIBUTES the LET over the two
        // disjuncts — `LET d == e IN A` / `LET d == e IN B` — with the binder in
        // scope on both. That pre-existing behavior is unchanged by Phase 5 (v1
        // and v2 agree); the ParsedNonRootOr classification just guarantees the
        // v2 fast-path does not itself split the body.
        let body = "LET d == e IN A \\/ B";
        assert!(disj_v2_is_parsed_non_root_or(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 2, "LET distributes over its 2 disjuncts: {out:#?}");
        assert!(out.iter().all(|d| d.starts_with("LET d == e IN")));
        assert!(out[0].trim_end().ends_with('A'));
        assert!(out[1].trim_end().ends_with('B'));
    }

    #[test]
    fn v2_disj_root_if_with_inner_or_stays_whole() {
        // root `IF c THEN A \/ B ELSE D` → root If → ParsedNonRootOr → whole.
        let body = "IF c THEN A \\/ B ELSE D";
        assert!(disj_v2_is_parsed_non_root_or(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "root IF must stay one disjunct: {out:#?}");
        assert!(out[0].starts_with("IF c THEN"));
    }

    #[test]
    fn v2_disj_root_forall_with_inner_or_stays_whole() {
        // `\A x \in S : A(x) \/ B(x)` → root Quant(\A) → ParsedNonRootOr → whole.
        let body = "\\A x \\in S : A(x) \\/ B(x)";
        assert!(disj_v2_is_parsed_non_root_or(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "root \\A must stay one disjunct: {out:#?}");
        assert!(out[0].starts_with("\\A x \\in S"));
    }

    #[test]
    fn v2_disj_root_paren_or_stays_whole() {
        // A ROOT `(A \/ B)` parses to `Paren`, not `Junction{Or}`, so the root
        // gate returns `ParsedNonRootOr` → the whole parenthesized disjunction is
        // kept as ONE disjunct (conservative and safe; the parens make it a
        // single grouped action). The same-connective Paren unwrap in
        // `flatten_or_leaves` only applies to a Paren(Or) nested UNDER a root Or
        // — see `v2_flatten_or_paren_or_unwraps_disjuncts` and
        // `v2_flatten_or_paren_and_stays_one_leaf`.
        let body = "(A \\/ B)";
        assert!(disj_v2_is_parsed_non_root_or(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "root Paren(Or) stays one disjunct: {out:#?}");
    }

    #[test]
    fn v2_disj_paren_and_stays_one_disjunct() {
        // `(A /\ B)` → root Paren(And) → ParsedNonRootOr (Paren-non-Or) → the
        // flatten_or Paren unwrap is same-connective ONLY, so this stays whole.
        let body = "(A /\\ B)";
        assert!(disj_v2_is_parsed_non_root_or(body));
    }

    #[test]
    fn v2_flatten_and_paren_and_unwraps_conjuncts() {
        // Phase-4 improvement: `(A /\ B)` as a conjunct-flatten root now unwraps
        // to two conjuncts. `flatten_and_leaves` unwraps Paren ONLY when inner is
        // an And-junction.
        use crate::tla::expr_v2;
        let tree = expr_v2::parse_ast("(A /\\ B)").expect("parse");
        let mut leaves = Vec::new();
        flatten_and_leaves(&tree, &mut leaves);
        assert_eq!(leaves.len(), 2, "Paren(And) must flatten to two conjunct leaves");
    }

    #[test]
    fn v2_flatten_or_paren_or_unwraps_disjuncts() {
        use crate::tla::expr_v2;
        let tree = expr_v2::parse_ast("(A \\/ B)").expect("parse");
        let mut leaves = Vec::new();
        flatten_or_leaves(&tree, &mut leaves);
        assert_eq!(leaves.len(), 2, "Paren(Or) must flatten to two disjunct leaves");
    }

    #[test]
    fn v2_flatten_or_paren_and_stays_one_leaf() {
        // A Paren(And) inside an Or-flatten is NOT same-connective → one leaf.
        use crate::tla::expr_v2;
        let tree = expr_v2::parse_ast("(A /\\ B) \\/ C").expect("parse");
        let mut leaves = Vec::new();
        flatten_or_leaves(&tree, &mut leaves);
        assert_eq!(leaves.len(), 2, "expected [(A /\\ B), C]: {}", leaves.len());
    }

    #[test]
    fn v2_disj_branch_with_conjunction_stays_one_disjunct() {
        // `\/ /\ guard /\ x' = 1  \/ /\ ~guard /\ x' = 2` — each `\/` branch is a
        // conjunction (root And under the Or), so flatten_or STOPS at each And →
        // 2 disjuncts, each containing its full conjunction (guard NOT dropped).
        let body = "\\/ /\\ guard\n   /\\ x' = 1\n\\/ /\\ ~guard\n   /\\ x' = 2";
        let parts = disj_v2_split(body).expect("root should be an Or junction");
        assert_eq!(parts.len(), 2, "expected 2 conjunctive disjuncts: {parts:#?}");
        assert!(parts[0].contains("guard") && parts[0].contains("x' = 1"));
        assert!(parts[1].contains("~guard") && parts[1].contains("x' = 2"));
    }

    #[test]
    fn v2_disj_nanoblockchain_process_block_stays_whole() {
        // `/\ \E block \in S : /\ \/ A(block) \/ B(block) /\ post` → root And →
        // ParsedNonRootOr → the `/\`-conjunction / single-\E-conjunct guards keep
        // it ONE disjunct (binder `block` in scope over the inner `\/`).
        let body = "/\\ \\E block \\in S :\n\
                    \x20      /\\ \\/ A(block)\n\
                    \x20         \\/ B(block)\n\
                    \x20      /\\ post(block)";
        assert!(disj_v2_conservative(body));
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "ProcessBlock shape must stay one disjunct: {out:#?}");
    }

    #[test]
    fn v2_disj_conjunction_led_disjunction_prefix_still_works() {
        // `/\ \/ A \/ B` (a `/\`-led body whose sole conjunct is a disjunction)
        // routes via the `strip_prefix("/\\")` branch. root here (after strip) is
        // an Or; ensure the v2 classification of the FULL `/\ \/`-body is a
        // consistent And root → ParsedNonRootOr (kept whole is wrong here, but the
        // dedicated `/\ \/` strip branch runs BEFORE the whole-body protection).
        let body = "/\\ \\/ A\n   \\/ B";
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 2, "leading /\\ over a bare \\/ splits into 2: {out:#?}");
    }

    #[test]
    fn v2_disj_single_bulleted_disjunct_strips_bullet_no_self_loop() {
        // REGRESSION GUARD (probe `parsed_braf_write_at_most_...` stack overflow):
        // a single-disjunct `\/ X` collapses in parse_junction to the item, so v2
        // classifies its root as the item (`ParsedNonRootOr`). The Phase-5
        // whole-body guard must NOT return `["\/ X"]` (keeping the bullet) — that
        // makes recursive re-feeders (`collect_action_param_samples_from_expr`,
        // `split_nested_action_disjuncts`) loop on `\/ X == input`. A `\/`-led
        // body must fall through to the string split, which STRIPS the `\/`.
        let out = split_action_body_disjuncts("\\/ Write1(symbol)");
        assert_eq!(out.len(), 1, "{out:#?}");
        assert!(!out[0].starts_with("\\/"), "leading \\/ must be stripped: {}", out[0]);
        assert_eq!(out[0], "Write1(symbol)");
    }

    #[test]
    fn v2_disj_falls_back_on_parse_error() {
        // Unterminated block comment → lex error → Fallback → the string path
        // (blind split reachable). Just assert the classification is Fallback.
        let body = "\\/ A (* unterminated \\/ B";
        assert!(matches!(
            classify_action_disjunct_v2(body),
            DisjunctSplitV2::Fallback
        ));
    }

    // ---- Phase 5.1 (Codex NO-GO fixes A/B/C) ----

    #[test]
    fn v2_disj_fix_a_leading_or_scoped_exists_returns_one_disjunct_binder_in_scope() {
        // FIX A (BLOCKER): a `\/`-led body whose scoped item is a quantifier
        // (`\/ \E x \in S : A(x) \/ B(x)`) collapses to a `Quant` root
        // (`ParsedNonRootOr`). The inner `\/` lives INSIDE the `\E x` scope, so it
        // must NOT be split. Strip exactly the one leading `\/` and return the
        // remainder as ONE disjunct; the `\E` prefix stays intact.
        let out = split_action_body_disjuncts("\\/ \\E x \\in S : A(x) \\/ B(x)");
        assert_eq!(out.len(), 1, "inner \\/ must NOT be split: {out:#?}");
        assert!(out[0].starts_with("\\E x \\in S :"), "\\E prefix intact: {out:#?}");
        assert!(out[0].contains("A(x)"), "{out:#?}");
        assert!(out[0].contains("B(x)"), "{out:#?}");
        assert!(!out[0].starts_with("\\/"), "leading \\/ stripped: {out:#?}");
    }

    #[test]
    fn v2_disj_fix_a_leading_or_scoped_let_returns_one_disjunct() {
        // FIX A: `\/ LET d == e IN A \/ B` — the `\/` is inside the LET body
        // scope. One disjunct, LET-bound `d` in scope for both A and B.
        let out = split_action_body_disjuncts("\\/ LET d == e IN A \\/ B");
        assert_eq!(out.len(), 1, "LET body \\/ must NOT be split: {out:#?}");
        assert!(out[0].starts_with("LET d == e IN"), "{out:#?}");
        assert!(out[0].contains("A \\/ B"), "{out:#?}");
        assert!(!out[0].starts_with("\\/"), "leading \\/ stripped: {out:#?}");
    }

    #[test]
    fn v2_disj_fix_b_block_comment_before_conjunction_stays_one_action() {
        // FIX B: a leading `(* ... *)` block comment before a `/\`-led body must
        // NOT be shredded into `\/` branches (Peterson false mutual-exclusion
        // violation). The body is a conjunction (one action). The `first_logical`
        // scan must skip the block comment so the `/\`-conjunction guard fires;
        // whether the protection is reported as `RootOrLayoutMismatch` (parser
        // mis-fences to root-Or) or `ParsedNonRootOr` (parser fences the `/\` as
        // a root-And), the END-TO-END guarantee is the same: the body stays ONE
        // disjunct.
        let body = "(* hdr *)\n/\\ A \\/ B\n/\\ C";
        assert!(
            matches!(
                classify_action_disjunct_v2(body),
                DisjunctSplitV2::RootOrLayoutMismatch | DisjunctSplitV2::ParsedNonRootOr
            ),
            "block-comment-prefixed /\\ body must be kept whole, not root-Or-split"
        );
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "conjunction stays one disjunct: {out:#?}");
    }

    #[test]
    fn v2_disj_fix_b_multiline_block_comment_before_conjunction_stays_one_action() {
        // FIX B: multi-line block comment variant — same one-action guarantee.
        let body = "(* line one\n   line two *)\n/\\ A \\/ B\n/\\ C";
        assert!(
            matches!(
                classify_action_disjunct_v2(body),
                DisjunctSplitV2::RootOrLayoutMismatch | DisjunctSplitV2::ParsedNonRootOr
            ),
            "multi-line block-comment /\\ body must be kept whole"
        );
        let out = split_action_body_disjuncts(body);
        assert_eq!(out.len(), 1, "conjunction stays one disjunct: {out:#?}");
    }

    #[test]
    fn v2_disj_fix_b_misfenced_root_or_block_comment_is_layout_mismatch() {
        // FIX B (direct guard exercise): construct a body where the parser DOES
        // mis-fence a `/\`-led conjunction to a root `Or` (inline `\/` at/left of
        // the `/\` bullet column), then prepend a block comment. Without the
        // comment-skipping in the `first_logical` scan the guard would be bypassed
        // and the guard-crossing shred would happen. With FIX B it stays
        // `RootOrLayoutMismatch` (kept whole).
        let no_comment = "/\\ A\n\\/ B\n/\\ C";
        let with_comment = "(* c *)\n/\\ A\n\\/ B\n/\\ C";
        // Both must classify identically (the comment must not change the verdict).
        let a = classify_action_disjunct_v2(no_comment);
        let b = classify_action_disjunct_v2(with_comment);
        assert_eq!(
            std::mem::discriminant(&a),
            std::mem::discriminant(&b),
            "leading block comment must not change the mis-fence classification"
        );
        // And neither is shredded.
        assert_eq!(split_action_body_disjuncts(with_comment).len(), 1);
    }

    #[test]
    fn v2_disj_fix_c_commented_branch_recaptures_and_bullet() {
        // FIX C: a commented `\/` branch (`\/ (* c *) /\ guard /\ post`) must keep
        // its `/\` conjunction INSIDE the disjunct — the comment in the recapture
        // gap must not cause the `/\` to be dropped.
        let out = split_action_body_disjuncts(
            "\\/ (* c *) /\\ guard\n           /\\ post\n\\/ D",
        );
        assert_eq!(out.len(), 2, "{out:#?}");
        // The first disjunct retains its `/\` conjunction (guard AND post).
        assert!(out[0].contains("guard"), "{out:#?}");
        assert!(out[0].contains("post"), "{out:#?}");
        assert!(out[0].contains("/\\"), "the /\\ conjunction must be kept: {out:#?}");
        assert_eq!(out[1].trim(), "D", "{out:#?}");
    }

    #[test]
    fn skip_whitespace_and_comments_backward_handles_block_comment() {
        // `/\ (* c *) guard` — the backward whitespace+comment skip from the leaf
        // must step back PAST the `(* c *)` block comment (and its surrounding
        // whitespace) so that `recapture_leading_and_bullet` can then see the `/\`
        // bullet. The skip helper stops just after the `/\` (leaving the two
        // bullet bytes for recapture's own `bytes[i-2..i]` check), so the
        // end-to-end contract is: recapture returns the index of the `/\`.
        let s = "/\\ (* c *) guard";
        let leaf = s.find("guard").unwrap();
        // The skip must make progress across the comment (not stall at `)`).
        let after_skip = {
            let bytes = s.as_bytes();
            let mut i = leaf;
            loop {
                let b = skip_whitespace_and_comments_backward(bytes, i);
                if b == i {
                    break;
                }
                i = b;
            }
            i
        };
        assert!(
            after_skip <= 2,
            "skip must cross the block comment back to the /\\ region, got {after_skip}"
        );
        // The real contract: recapture recovers the `/\` bullet at index 0.
        let recaptured = recapture_leading_and_bullet(s, leaf);
        assert_eq!(recaptured, 0, "recapture must land on the /\\ bullet");
        assert_eq!(&s[recaptured..recaptured + 2], "/\\");
    }
}
