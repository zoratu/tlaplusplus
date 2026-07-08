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
        // Any non-And node is a conjunct leaf, kept whole.
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
fn uniform_dedent(expr: &str) -> String {
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
}
