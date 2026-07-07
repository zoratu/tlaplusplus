//! Top-level expression splitters and lexical helpers used by the
//! TLA+ expression evaluator.
//!
//! These helpers operate on raw `&str` and only know about TLA+ syntax
//! (parens / brackets / strings / quantifier keywords). They are pure
//! lexical utilities — no evaluation or value construction.
//!
//! `split_top_level_defined_infix` is the one exception: it needs the
//! parent module's `EvalContext` so it can scan user-defined infix
//! operators registered in the local/global definition maps. That call
//! also needs the `is_user_defined_infix_operator` helper which still
//! lives in the parent module.

use anyhow::{Result, anyhow};
use std::collections::BTreeSet;

use super::{EvalContext, is_user_defined_infix_operator};

/// Check if a string is a valid TLA+ identifier
pub(crate) fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

pub(super) fn strip_outer_parens(expr: &str) -> &str {
    let mut current = expr.trim();
    while is_wrapped_by(current, '(', ')') {
        current = current[1..current.len() - 1].trim();
    }
    current
}

pub(super) fn is_wrapped_by(expr: &str, open: char, close: char) -> bool {
    if !expr.starts_with(open) || !expr.ends_with(close) {
        return false;
    }

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in expr.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }

        if ch == open {
            depth += 1;
            continue;
        }
        if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 && idx + ch.len_utf8() < expr.len() {
                return false;
            }
        }
    }

    depth == 0
}

pub(super) fn starts_with_keyword(expr: &str, kw: &str) -> bool {
    if let Some(rest) = expr.strip_prefix(kw) {
        return rest
            .chars()
            .next()
            .map(|c| !is_word_char(c))
            .unwrap_or(true);
    }
    false
}

pub(super) fn take_keyword_prefix<'a>(expr: &'a str, kw: &str) -> Option<(&'a str, &'a str)> {
    let trimmed = expr.trim_start();
    let prefix_ws_len = expr.len() - trimmed.len();
    let before = &expr[..prefix_ws_len];
    let rest = trimmed.strip_prefix(kw)?;
    if rest.chars().next().map(is_word_char).unwrap_or(false) {
        return None;
    }
    Some((before, rest.trim_start()))
}

pub(crate) fn split_top_level_symbol(expr: &str, delim: &str) -> Vec<String> {
    split_top_level(expr, delim, false)
}

pub(crate) fn split_top_level_keyword(expr: &str, delim: &str) -> Vec<String> {
    split_top_level(expr, delim, true)
}

pub(super) fn split_top_level(expr: &str, delim: &str, keyword: bool) -> Vec<String> {
    // Stale debug instrumentation removed — the previous `if expr.contains("PositionLimits")`
    // branch sliced `&expr[..60]` by byte index, which panicked when byte 60
    // landed inside a multi-byte char (a fuzz-discoverable UTF-8 boundary
    // bug). The eprintln was leftover from one-off model debugging; safe to
    // drop entirely. (T101.1 fuzz pass surfaced this; same bug class as the
    // T101 char-boundary fixes in compiled_expr.rs / module.rs.)

    // Fast path: `split_top_level` only ever returns >1 part — or a non-
    // trivial single part — by matching `delim` at top level, or by matching a
    // bare `\` (set-minus) at top level (see the `ch == '\\'` branch below). If
    // the string contains NEITHER byte-sequence anywhere, neither can fire, so
    // the full stateful scan would just return `vec![expr.trim().to_string()]`.
    // Short-circuit that with two cheap substring scans and skip the per-call
    // `Vec` allocation + character walk. This is the dominant hot path: the
    // string evaluator tests every atomic sub-expression (e.g. `i < j`,
    // `ss[i] =< ss[j]`) against `=>`, `<=>`, `\/`, `/\`, `\cdot` in turn while
    // enumerating large filtered sets — none of those occur in such atoms.
    // (When `delim` itself contains `\`, the backslash check conservatively
    // forces the full scan, which is always correct, just not maximally fast.)
    if !expr.contains(delim) && !expr.as_bytes().contains(&b'\\') {
        return vec![expr.trim().to_string()];
    }

    let mut out = Vec::new();
    let mut start = 0usize;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut quantifier_depth = 0usize; // Count of \A or \E seen
    let mut seen_colon_for_quantifier = 0usize; // Count of : seen to close quantifiers
    let mut let_depth = 0usize; // Count of LET keywords seen (increases on LET, decreases on IN)
    let mut if_depth = 0usize; // Count of IF keywords seen (increases on IF, decreases on ELSE)
    let mut case_depth = 0usize; // Count of CASE keywords seen
    let mut else_branch_uses_delimiter = false; // True if ELSE branch starts with the delimiter
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        // Track LET depth as we scan through the expression:
        // - LET increases depth (we're inside a LET expression)
        // - We DON'T decrement on IN - the LET body continues after IN
        // - We only reset let_depth when we perform an actual split
        // This treats the entire "LET defs IN body" as one atomic unit
        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            // Check if we're at the start of LET keyword
            if expr[i..].starts_with("LET") {
                let after_let = &expr[i + 3..];
                // invariant: is_empty() check above guarantees chars().next() is Some
                if after_let.is_empty()
                    || !after_let
                        .chars()
                        .next()
                        .expect("non-empty string has no first char")
                        .is_alphanumeric()
                {
                    let_depth += 1;
                }
            }
        }

        // Track IF-THEN-ELSE: IF increases depth, ELSE decreases it
        // This ensures we don't split inside an IF-THEN-ELSE expression
        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            if expr[i..].starts_with("IF") {
                let after_if = &expr[i + 2..];
                if after_if.is_empty()
                    || after_if
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    if_depth += 1;
                }
            } else if expr[i..].starts_with("ELSE") {
                let after_else = &expr[i + 4..];
                if after_else.is_empty()
                    || after_else
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    // Check if ELSE is followed by the delimiter (indicating ELSE branch uses delimiter)
                    let remaining = after_else.trim_start();
                    if remaining.starts_with(delim) {
                        // ELSE branch starts with delimiter - don't decrement if_depth
                        else_branch_uses_delimiter = true;
                    } else {
                        // ELSE branch doesn't start with delimiter - can decrement
                        if_depth = if_depth.saturating_sub(1);
                    }
                }
            }

            // Track CASE expressions - don't split inside CASE
            if expr[i..].starts_with("CASE") {
                let after_case = &expr[i + 4..];
                if after_case.is_empty()
                    || after_case
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    case_depth += 1;
                }
            }
        }

        // Track quantifiers: \A or \E increases depth
        if ch == '\\'
            && (expr[i..].starts_with("\\A") || expr[i..].starts_with("\\E"))
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            quantifier_depth += 1;
        } else if ch == ':'
            && quantifier_depth > seen_colon_for_quantifier
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            // This : starts a quantifier body
            seen_colon_for_quantifier += 1;
        }

        // We're truly at top level if:
        // 1. No nested parens/brackets
        // 2. Either no quantifiers, OR we've seen all the colons (we're in the body)
        // If we're in a quantifier body and see a top-level /\ or \/, that ENDS the quantifier
        let in_quantifier_body = seen_colon_for_quantifier > 0 && quantifier_depth > 0;
        let brackets_at_zero = paren == 0 && bracket == 0 && brace == 0 && angle == 0;

        // Check if this is our delimiter at a potential split point
        if brackets_at_zero && expr[i..].starts_with(delim) {
            let delim_end = i + delim.len();
            let is_word_ok = !keyword || has_word_boundaries(expr, i, delim_end);

            if is_word_ok {
                let is_conjunction = delim == "/\\" || delim == "\\/";
                // Determine if we should split:
                // 1. If we're in a LET definition section (between LET and IN), don't split on /\ or \/
                // 2. If we're before the : in a quantifier (in binders), don't split on /\ or \/
                // 3. If we're in a quantifier body, check if this /\ is followed by another quantifier
                //    - If yes: this /\ ENDS the current quantifier - split!
                //    - If no: this /\ is part of the quantifier body - don't split
                // 4. If we're inside IF-THEN (before ELSE), don't split on /\ or \/
                // 5. Otherwise, split normally
                let should_split = if let_depth > 0 && is_conjunction {
                    // We're inside a LET expression, don't split on conjunctions
                    false
                } else if (if_depth > 0 || else_branch_uses_delimiter) && is_conjunction {
                    // We're inside IF-THEN (before ELSE), don't split on conjunctions
                    false
                } else if case_depth > 0 && is_conjunction {
                    // We're inside a CASE expression, don't split on conjunctions
                    false
                } else if quantifier_depth > 0
                    && quantifier_depth > seen_colon_for_quantifier
                    && is_conjunction
                {
                    // We're in quantifier binders (before :), don't split on conjunctions
                    false
                } else if in_quantifier_body
                    && (delim == "/\\" || delim == "\\/" || delim == "=>" || delim == "<=>")
                {
                    // A quantifier body `\A x \in S : <body>` (or `\E`, `CHOOSE`)
                    // extends as far to the right as possible: the body consumes
                    // every lower-precedence boolean operator — conjunction,
                    // disjunction, implication, and biconditional — to the end
                    // of the surrounding grouping. So `X /\ \A y \in S : P => Q`
                    // parses as `X /\ (\A y \in S : (P => Q))`, NOT as
                    // `(X /\ \A y \in S : P) => Q`. Suppressing `/\`/`\/` here but
                    // NOT `=>`/`<=>` (the historical behaviour) mis-lifted an
                    // implication inside a quantifier body to the top level,
                    // re-associating `LHS /\ \A y : P => Q` into an implication
                    // whose antecedent `(LHS /\ \A y : P)` was false — making the
                    // whole predicate vacuously TRUE. That silently mis-evaluated
                    // e.g. `ChooseOne(S, P) == CHOOSE x \in S : P(x) /\ \A y \in S
                    // : P(y) => y = x` (SlidingPuzzles), which returned the wrong
                    // element and collapsed the reachable state space.
                    false
                } else {
                    // All other cases: split normally
                    true
                };

                if should_split {
                    let part = expr[start..i].trim();
                    if !part.is_empty() {
                        out.push(part.to_string());
                    }
                    // Reset quantifier, LET, IF, CASE, and else-branch tracking for next part
                    quantifier_depth = 0;
                    seen_colon_for_quantifier = 0;
                    let_depth = 0;
                    if_depth = 0;
                    case_depth = 0;
                    else_branch_uses_delimiter = false;
                    start = delim_end;
                    i = delim_end;
                    continue;
                }
            }
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        i += ch_len;
    }

    let tail = expr[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    let result = if out.is_empty() {
        vec![expr.trim().to_string()]
    } else {
        out
    };

    result
}

/// True if `clause` ends with a binary/binding operator that still expects a
/// following operand or body: `=>` / `<=>` (implication / biconditional whose
/// right side follows) or a LET/quantifier `IN` (whose body follows). Used by
/// `split_indented_top_level_boolean` to avoid mis-splitting a following
/// `/\`/`\/` bullet — the operator's missing right operand — into a spurious
/// sibling junction item once upstream indentation has been flattened.
///
/// Only matches a *trailing* operator (the clause is left dangling), so it
/// never affects a well-formed junction item like `a => b` on one line.
fn ends_with_dangling_binder(clause: &str) -> bool {
    let c = clause.trim_end();
    if c.ends_with("=>") || c.ends_with("<=>") {
        return true;
    }
    // A trailing `IN` keyword (word-boundary) — the LET/quantifier body follows.
    if let Some(prefix) = c.strip_suffix("IN") {
        // Must be a standalone `IN` token: preceded by whitespace (or nothing).
        return prefix.is_empty() || prefix.ends_with(|ch: char| ch.is_whitespace());
    }
    false
}

pub(super) fn split_indented_top_level_boolean(expr: &str, delim: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_boolean_indentation(expr);
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
        if trimmed.starts_with(delim) {
            let top_level_indent = *base_indent.get_or_insert(indent);
            // A `delim` bullet at the top-level indent is normally a NEW sibling
            // junction item. But if the clause accumulated so far ends with a
            // *dangling* binary operator that still expects a right operand
            // (`=>`, `<=>`) or opens a scope whose body follows (`IN`), then this
            // bullet IS that missing operand / body — NOT a sibling. Keep it
            // attached to `current`.
            //
            // This shape arises once upstream indentation has been flattened by
            // nested LET/quantifier body extraction (each level `.trim()`s its
            // sub-expressions). NanoBlockchain's `CryptographicInvariant`:
            //
            //     /\ signedBlock /= NoBlock =>
            //         LET publicKey == PublicKeyOf(ledger, hash) IN
            //         /\ ValidateSignature(...)
            //
            // flattens to (all at column 0)
            //
            //     signedBlock /= NoBlock => LET publicKey == ... IN
            //     /\ ValidateSignature(...)
            //
            // so the `/\ ValidateSignature` (the `=>`-consequent's LET body)
            // would wrongly become a sibling conjunct of the antecedent — its
            // record access (`signedBlock.signature`) is then evaluated
            // UNCONDITIONALLY, past the `=>` short-circuit, producing a spurious
            // `record access on non-record value NoBlockVal` violation on the
            // initial state where every `signedBlock = NoBlock`. Deferring the
            // bullet keeps the whole `... => LET ... IN /\ ...` as one clause so
            // the implication's `=>` correctly gates the consequent.
            if indent == top_level_indent && !ends_with_dangling_binder(current.trim_end()) {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(trimmed.trim_start_matches(delim).trim_start());
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
        current.push_str(trimmed);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() <= 1 {
        return None;
    }

    let mut merged = Vec::with_capacity(clauses.len());
    let mut idx = 0usize;
    while idx < clauses.len() {
        let part = clauses[idx].trim().to_string();
        if part.is_empty() {
            idx += 1;
            continue;
        }

        let open_quant_or_let = ((part.starts_with("\\A") || part.starts_with("\\E"))
            && (part.ends_with(':') || part.ends_with("IN")))
            || (part.starts_with("LET") && part.ends_with("IN"));
        if open_quant_or_let {
            let mut combined = part;
            for rest in clauses.iter().skip(idx + 1) {
                if rest.trim().is_empty() {
                    continue;
                }
                combined.push(' ');
                combined.push_str(delim);
                combined.push(' ');
                combined.push_str(rest.trim());
            }
            merged.push(combined);
            break;
        }

        merged.push(part);
        idx += 1;
    }

    if merged.len() > 1 { Some(merged) } else { None }
}

pub(super) fn normalize_multiline_boolean_indentation(expr: &str) -> String {
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
        normalized.push_str(&line[..keep]);
        normalized.push_str(trimmed);
    }

    normalized
}

pub(super) fn split_top_level_set_minus(expr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        // A `\` immediately followed by an ASCII letter is a NAMED operator
        // token — either a built-in (`\in`, `\cup`, ...) or a USER-DEFINED
        // infix operator (`\preceq`, `\sqsubseteq`, ...). Only the latter is
        // not in `starts_with_tla_backslash_operator`'s fixed list, so without
        // this guard `a \preceq b` is mis-split as set-minus `a \ (preceq b)`,
        // corrupting the expression (invariant/guard eval then errors, which
        // surfaces as a spurious violation + truncated exploration). Genuine
        // set difference `A \ B` has the `\` followed by whitespace or an
        // operand-opening token, never a letter.
        let followed_by_named_op = expr[i + ch_len..]
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic())
            .unwrap_or(false);
        if at_top
            && ch == '\\'
            && !followed_by_named_op
            && !starts_with_tla_backslash_operator(&expr[i..])
        {
            let prev = expr[..i].chars().next_back();
            let next_char = expr[i + ch_len..].chars().next();
            let ws_before = prev.map(|c| c.is_whitespace()).unwrap_or(false);
            let ws_after = next_char.map(|c| c.is_whitespace()).unwrap_or(false);
            if ws_before || ws_after || next_char.is_some() {
                let part = expr[start..i].trim();
                if !part.is_empty() {
                    out.push(part.to_string());
                }
                start = i + ch_len;
            }
        }

        i += ch_len;
    }

    let tail = expr[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    if out.is_empty() {
        vec![expr.trim().to_string()]
    } else {
        out
    }
}

pub(super) fn starts_with_tla_backslash_operator(expr: &str) -> bool {
    [
        "\\A",
        "\\E",
        "\\in",
        "\\notin",
        "\\union",
        "\\intersect",
        "\\cup",
        "\\cap",
        "\\subseteq",
        "\\supseteq",
        "\\div",
        "\\X",
        "\\times",
        "\\o",
        "\\circ",
        "\\/",
        "\\*",
        "\\leq",
        "\\geq",
    ]
    .iter()
    .any(|op| expr.starts_with(op))
}

pub(super) fn split_top_level_comparison(expr: &str) -> Option<(&str, &'static str, &str)> {
    // NOTE: `..` is intentionally NOT in this list. The TLA+ range operator
    // `..` has precedence 9-9 in the official TLA+ grammar (Lamport's
    // "Specifying Systems", cheat sheet), which is *tighter* than the
    // relational/comparison tier (5-5: `\subseteq`, `\in`, `=`, `<`, ...) and
    // tighter than the set-op tier (8-8: `\union`, `\intersect`, `\` ...).
    // Splitting on `..` here would cause `1..3 \subseteq S` to parse as
    // `1 .. (3 \subseteq S)` — a soundness bug that silently misparses any
    // spec writing `n..m \subseteq S` (or `\union`, `\intersect`, `\\`)
    // without paren-wrapping the range. `..` is split separately below in
    // `split_top_level_range`, slotted between the set-op tier and the
    // additive tier (10-10) in the operator-precedence cascade.
    let patterns = [
        "\\subseteq",
        "\\notin",
        "\\in",
        "\\leq",
        "\\geq",
        "=<",
        "<=",
        ">=",
        "/=",
        "#",
        "=",
        "<",
        ">",
    ];

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize; // Track IF...THEN nesting (protects conditions from split)
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        // Check for keywords at word boundaries when at bracket top level
        if let Some((word, start, word_end)) = next_word(expr, i)
            && start == i
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            match word {
                "LET" => {
                    let_depth += 1;
                    i = word_end;
                    continue;
                }
                "IN" if let_depth > 0 => {
                    let_depth -= 1;
                    i = word_end;
                    continue;
                }
                "IF" => {
                    if_depth += 1;
                    i = word_end;
                    continue;
                }
                "THEN" if if_depth > 0 => {
                    if_depth -= 1;
                    i = word_end;
                    continue;
                }
                _ => {}
            }
        }

        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && let_depth == 0
            && if_depth == 0;
        if at_top {
            for pattern in patterns {
                if !expr[i..].starts_with(pattern) {
                    continue;
                }

                let end = i + pattern.len();

                if pattern == "=" {
                    let prev = expr[..i].chars().next_back();
                    let next_char = expr[end..].chars().next();
                    if prev == Some('=')
                        || prev == Some('<')
                        || prev == Some('>')
                        || prev == Some('/')
                    {
                        continue;
                    }
                    if next_char == Some('>') || next_char == Some('=') {
                        continue;
                    }
                }

                if pattern == "<" {
                    let next_char = expr[end..].chars().next();
                    if next_char == Some('=') || next_char == Some('<') {
                        continue;
                    }
                }

                if pattern == ">" {
                    let prev_char = expr[..i].chars().next_back();
                    let next_char = expr[end..].chars().next();
                    // Skip >= and >> operators
                    if next_char == Some('=') || next_char == Some('>') {
                        continue;
                    }
                    // Skip :> (TLC function pair operator)
                    if prev_char == Some(':') {
                        continue;
                    }
                }

                if pattern.starts_with('\\') && !has_word_boundaries(expr, i, end) {
                    continue;
                }

                let lhs = expr[..i].trim();
                let rhs = expr[end..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    continue;
                }
                return Some((lhs, pattern, rhs));
            }
        }

        i += ch_len;
    }

    None
}

/// Split a top-level `..` range operator (TLA+ precedence 9-9). Looks for the
/// first `..` at top bracket depth (not inside parens/brackets/braces/angles)
/// and not inside a string literal. Care taken to:
///   - skip leading `..` if it appears at the very start (e.g. inside `[..` --
///     defensive; nothing legal starts with `..`),
///   - require both sides non-empty after trimming.
///
/// `..` is left-associative but rarely chained; we split at the *first*
/// occurrence so `a..b..c` (uncommon) parses as `a .. (b..c)`. Either choice
/// is fine; the recursive call on `b..c` will then split there.
pub(super) fn split_top_level_range(expr: &str) -> Option<(&str, &str)> {
    let bytes = expr.as_bytes();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i + 1 < bytes.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && ch == '.' && next == Some('.') && i > 0 {
            // Avoid mistaking `...` (e.g. in record `[f : Set, ...]`) for `..`.
            // TLA+ doesn't really use `...` but be defensive.
            let after = expr.get(i + 2..).and_then(|s| s.chars().next());
            if after == Some('.') {
                i += ch_len;
                continue;
            }
            let lhs = expr[..i].trim();
            let rhs = expr[i + 2..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                return Some((lhs, rhs));
            }
        }

        i += ch_len;
    }

    None
}

pub(super) fn split_top_level_additive(expr: &str) -> Option<(&str, char, &str)> {
    let mut last: Option<(usize, char)> = None;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && (ch == '+' || ch == '-') && is_binary_operator(expr, i, ch_len) {
            last = Some((i, ch));
        }

        i += ch_len;
    }

    let (idx, op) = last?;
    let lhs = expr[..idx].trim();
    let rhs = expr[idx + op.len_utf8()..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        return None;
    }
    Some((lhs, op, rhs))
}

pub(super) fn split_top_level_multiplicative(expr: &str) -> Option<(&str, &'static str, &str)> {
    let mut star_idx: Option<usize> = None;
    let mut div_idx: Option<usize> = None;
    let mut mod_idx: Option<usize> = None;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top {
            if ch == '*' && is_binary_operator(expr, i, ch_len) {
                star_idx = Some(i);
            }
            if expr[i..].starts_with("\\div") {
                let end = i + "\\div".len();
                if has_word_boundaries(expr, i, end) {
                    div_idx = Some(i);
                }
            }
            if ch == '%' && is_binary_operator(expr, i, ch_len) {
                mod_idx = Some(i);
            }
        }

        i += ch_len;
    }

    // Find the rightmost operator for left-to-right associativity
    let mut candidates: Vec<(usize, &str, usize)> = Vec::new(); // (idx, op_str, op_len)
    if let Some(s) = star_idx {
        candidates.push((s, "*", 1));
    }
    if let Some(d) = div_idx {
        candidates.push((d, "\\div", "\\div".len()));
    }
    if let Some(m) = mod_idx {
        candidates.push((m, "%", 1));
    }
    // Pick the rightmost (largest index)
    candidates.sort_by_key(|(idx, _, _)| *idx);
    let (idx, op, op_len) = candidates.last()?;
    let lhs = expr[..*idx].trim();
    let rhs = expr[idx + op_len..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        None
    } else {
        Some((lhs, *op, rhs))
    }
}

pub(crate) fn split_once_top_level<'a>(expr: &'a str, delim: &str) -> Option<(&'a str, &'a str)> {
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && expr[i..].starts_with(delim) {
            return Some((expr[..i].trim(), expr[i + delim.len()..].trim()));
        }

        i += ch_len;
    }

    None
}

pub(super) fn split_top_level_defined_infix<'a>(
    expr: &'a str,
    ctx: &EvalContext<'_>,
) -> Option<(&'a str, String, &'a str)> {
    let mut operators = BTreeSet::new();
    for defs in [Some(ctx.local_definitions.as_ref()), ctx.definitions] {
        let Some(defs) = defs else {
            continue;
        };
        for (name, def) in defs.iter() {
            if def.params.len() == 2 && is_user_defined_infix_operator(name) {
                operators.insert(name.clone());
            }
        }
    }
    if operators.is_empty() {
        return None;
    }

    let mut operators: Vec<String> = operators.into_iter().collect();
    operators.sort_by_key(|op| std::cmp::Reverse(op.len()));

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            for op in &operators {
                if !expr[i..].starts_with(op) {
                    continue;
                }
                let lhs = expr[..i].trim();
                let rhs = expr[i + op.len()..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    continue;
                }
                return Some((lhs, op.clone(), rhs));
            }
        }

        i += ch_len;
    }

    None
}

pub(crate) fn find_top_level_keyword_index(expr: &str, keyword: &str) -> Option<usize> {
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && expr[i..].starts_with(keyword)
            && has_keyword_boundaries(expr, i, i + keyword.len(), keyword)
        {
            return Some(i);
        }

        i += ch_len;
    }

    None
}

pub(super) fn contains_top_level_keyword(expr: &str, keyword: &str) -> bool {
    find_top_level_keyword_index(expr, keyword).is_some()
}

pub(crate) fn find_top_level_char(expr: &str, target: char) -> Option<usize> {
    find_top_level_char_from(expr, target, 0)
}

pub(super) fn find_top_level_char_from(expr: &str, target: char, start_at: usize) -> Option<usize> {
    let mut i = start_at;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && ch == target {
            // When searching for ':', skip ':>' (TLC function pair operator)
            if ch == ':' && next == Some('>') {
                i += ch_len;
                continue;
            }
            return Some(i);
        }

        i += ch_len;
    }

    None
}

pub(super) fn take_bracket_group(expr: &str, open: char, close: char) -> Result<(&str, &str)> {
    if !expr.starts_with(open) {
        return Err(anyhow!("expected '{open}'"));
    }

    let mut depth = 0usize;
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let inner = &expr[open.len_utf8()..i];
                let rest = &expr[i + ch_len..];
                return Ok((inner, rest));
            }
        }

        i += ch_len;
    }

    Err(anyhow!("missing closing '{close}' in expression: {expr}"))
}

pub(super) fn take_angle_group(expr: &str) -> Result<(&str, &str)> {
    if !expr.starts_with("<<") {
        return Err(anyhow!("expected '<<'"));
    }

    let mut depth = 0usize;
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            depth += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            depth = depth.saturating_sub(1);
            i += 2;
            if depth == 0 {
                let inner = &expr[2..i - 2];
                let rest = &expr[i..];
                return Ok((inner, rest));
            }
            continue;
        }

        i += ch_len;
    }

    Err(anyhow!("missing closing '>>' in expression: {expr}"))
}

pub(super) fn parse_identifier_prefix(expr: &str) -> Option<(String, &str)> {
    let mut chars = expr.char_indices();
    let (first_idx, first) = chars.next()?;
    if first_idx != 0 {
        return None;
    }
    if !(first.is_alphanumeric() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    let mut saw_identifier_marker = first.is_alphabetic() || first == '_';
    for (idx, c) in chars {
        if c.is_alphanumeric() || c == '_' || c == '\'' {
            end = idx + c.len_utf8();
            saw_identifier_marker |= c.is_alphabetic() || c == '_';
        } else {
            break;
        }
    }

    if !saw_identifier_marker {
        return None;
    }

    Some((expr[..end].to_string(), &expr[end..]))
}

pub(super) fn parse_string_literal_prefix(expr: &str) -> Result<Option<(String, &str)>> {
    if !expr.starts_with('"') {
        return Ok(None);
    }

    let mut out = String::new();
    let mut escaped = false;

    let mut i = 1usize;
    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();

        if escaped {
            out.push(ch);
            escaped = false;
            i += ch_len;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            i += ch_len;
            continue;
        }

        if ch == '"' {
            return Ok(Some((out, &expr[i + ch_len..])));
        }

        out.push(ch);
        i += ch_len;
    }

    Err(anyhow!("unterminated string literal in expression: {expr}"))
}

pub(super) fn parse_int_prefix(expr: &str) -> Option<(i64, &str)> {
    let mut end = 0usize;
    for (idx, c) in expr.char_indices() {
        if c.is_ascii_digit() {
            end = idx + c.len_utf8();
        } else {
            break;
        }
    }

    if end == 0 {
        return None;
    }

    let rest = &expr[end..];
    if rest
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '\'')
    {
        return None;
    }

    let value = expr[..end].parse::<i64>().ok()?;
    Some((value, rest))
}

pub(super) fn has_word_boundaries(expr: &str, start: usize, end: usize) -> bool {
    let prev = expr[..start].chars().next_back();
    let next = expr[end..].chars().next();

    let prev_ok = prev.map(|c| !is_word_char(c)).unwrap_or(true);
    let next_ok = next.map(|c| !is_word_char(c)).unwrap_or(true);

    prev_ok && next_ok
}

pub(super) fn has_keyword_boundaries(expr: &str, start: usize, end: usize, keyword: &str) -> bool {
    if keyword.starts_with('\\') {
        let next = expr[end..].chars().next();
        return next.map(|c| !c.is_alphabetic()).unwrap_or(true);
    }
    has_word_boundaries(expr, start, end)
}

pub(super) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub(super) fn is_binary_operator(expr: &str, idx: usize, len: usize) -> bool {
    let prev = expr[..idx].chars().rev().find(|c| !c.is_whitespace());
    let next = expr[idx + len..].chars().find(|c| !c.is_whitespace());

    let Some(prev) = prev else {
        return false;
    };
    let Some(next) = next else {
        return false;
    };

    let prev_is_operator = matches!(
        prev,
        '(' | '[' | '{' | ',' | ':' | '+' | '-' | '*' | '/' | '\\' | '=' | '<' | '>' | '#'
    );
    let next_is_operator = matches!(next, ')' | ']' | '}' | ',' | ':');

    !prev_is_operator && !next_is_operator
}

pub(super) fn next_word(input: &str, from: usize) -> Option<(&str, usize, usize)> {
    let mut i = from;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if ch.is_alphabetic() || ch == '_' {
            break;
        }
        i += len;
    }

    if i >= input.len() {
        return None;
    }

    let start = i;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if !(ch.is_alphanumeric() || ch == '_') {
            break;
        }
        i += len;
    }

    Some((&input[start..i], start, i))
}

pub(super) fn find_outer_then(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "THEN" if nested_if == 0 => return Some(start),
            "ELSE" if nested_if > 0 => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

pub(super) fn find_outer_else(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "ELSE" if nested_if == 0 => return Some(start),
            "ELSE" => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

pub(super) fn find_top_level_definition_eqs(expr: &str) -> Vec<usize> {
    let mut out = Vec::new();

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut let_depth = 0usize;

    while i < expr.len() {
        if let Some((word, start, _end)) = next_word(expr, i)
            && start == i
        {
            match word {
                "LET" => let_depth += 1,
                "IN" if let_depth > 0 => let_depth -= 1,
                _ => {}
            }
        }

        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && let_depth == 0
            && ch == '='
            && next == Some('=')
        {
            out.push(i);
            i += 2;
            continue;
        }

        i += ch_len;
    }

    out
}

pub(super) fn line_start_before(input: &str, idx: usize) -> usize {
    match input[..idx].rfind('\n') {
        Some(pos) => pos + 1,
        None => 0,
    }
}

pub(super) fn skip_leading_ws(input: &str, mut idx: usize) -> usize {
    while idx < input.len() {
        let ch = input[idx..].chars().next().expect("char at byte index");
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_top_level_keeps_simple_quantified_conjunctions_intact() {
        let parts =
            split_top_level_symbol(r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#, "/\\");
        assert_eq!(
            parts,
            vec![r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#]
        );
    }

    #[test]
    fn split_top_level_keeps_simple_quantified_disjunctions_intact() {
        let parts = split_top_level_symbol(r"\E a \in Acceptor : Phase1b(a) \/ Phase2b(a)", "\\/");
        assert_eq!(parts, vec![r"\E a \in Acceptor : Phase1b(a) \/ Phase2b(a)"]);
    }

    #[test]
    fn split_top_level_keeps_implication_inside_quantifier_body_intact() {
        // SlidingPuzzles missed-violation regression. A quantifier body extends
        // as far right as possible and consumes lower-precedence boolean
        // operators — including `=>`. `X /\ \A y \in S : P => Q` must stay
        // grouped as `X /\ (\A y \in S : (P => Q))`; splitting the `=>` at the
        // top level re-associated it into `(X /\ \A y : P) => Q`, whose false
        // antecedent made the whole predicate vacuously TRUE. That corrupted
        // `ChooseOne(S, P) == CHOOSE x \in S : P(x) /\ \A y \in S : P(y) => y = x`
        // (it returned the wrong element), collapsing SlidingPuzzles to 2 states
        // and hiding the KlotskiGoal violation.
        let expr = r"x = 3 /\ \A y \in S : (y = 3) => y = x";
        assert_eq!(split_top_level_symbol(expr, "=>"), vec![expr]);
        // The outer `/\` still splits (the quantifier body ends where `x = 3`'s
        // conjunct begins, i.e. the `/\` is *before* the `\A`, at top level).
        assert_eq!(
            split_top_level_symbol(expr, "/\\"),
            vec![r"x = 3", r"\A y \in S : (y = 3) => y = x"]
        );
    }

    #[test]
    fn split_top_level_keeps_biconditional_inside_quantifier_body_intact() {
        // Same rule for `<=>`: a quantifier body consumes it too.
        let expr = r"P /\ \A y \in S : Q(y) <=> R(y)";
        assert_eq!(split_top_level_symbol(expr, "<=>"), vec![expr]);
    }

    #[test]
    fn indented_bool_keeps_implies_consequent_let_body_attached() {
        // NanoBlockchain `CryptographicInvariant` missed-soundness regression.
        // After nested LET/quantifier body extraction flattens indentation, the
        // `=>`-consequent's LET body (`/\ ValidateSignature(...)`) sits at the
        // same column as the antecedent's leading `/\`. The indented-boolean
        // splitter must NOT treat the consequent bullet as a *sibling* conjunct
        // — that would evaluate `signedBlock.signature` unconditionally, past the
        // `=>` short-circuit, and false-violate the initial state (every
        // `signedBlock = NoBlock`). The whole `... => LET ... IN /\ ...` stays
        // one clause so `=>` gates the consequent.
        let flat = "/\\ signedBlock /= NoBlock => LET publicKey == PublicKeyOf(ledger, hash) IN\n/\\ ValidateSignature(signedBlock.signature, publicKey, hash)";
        // Single top-level clause (the implication) — no sibling split.
        assert_eq!(split_indented_top_level_boolean(flat, "/\\"), None);

        // dangling-binder detector: trailing `=>`, `<=>`, `IN` expect an operand.
        assert!(ends_with_dangling_binder("a /= b =>"));
        assert!(ends_with_dangling_binder("LET x == y IN"));
        assert!(ends_with_dangling_binder("p <=>"));
        // A well-formed one-line item is NOT dangling.
        assert!(!ends_with_dangling_binder("a => b"));
        assert!(!ends_with_dangling_binder("x = 1"));
        // Identifier ending in "IN" must not be mistaken for the keyword.
        assert!(!ends_with_dangling_binder("MAIN"));
    }

    #[test]
    fn indented_bool_still_splits_genuine_siblings_after_this_fix() {
        // Guard: the dangling-binder gate must NOT suppress normal sibling
        // splits. A conjunct that does NOT end in `=>`/`<=>`/`IN` still splits.
        let siblings = "/\\ x = 1\n/\\ y = 2";
        assert_eq!(
            split_indented_top_level_boolean(siblings, "/\\"),
            Some(vec!["x = 1".to_string(), "y = 2".to_string()])
        );
        // A one-line `a => b` conjunct followed by a sibling still splits (the
        // first clause does not END with a dangling `=>`).
        let impl_then_sib = "/\\ a => b\n/\\ c = 2";
        assert_eq!(
            split_indented_top_level_boolean(impl_then_sib, "/\\"),
            Some(vec!["a => b".to_string(), "c = 2".to_string()])
        );
    }

    #[test]
    fn find_outer_else_handles_newlines() {
        // Test that find_outer_else correctly finds ELSE across newlines
        assert_eq!(find_outer_else("something\nELSE other"), Some(10));
        assert_eq!(find_outer_else("something ELSE other"), Some(10));

        // Nested IF should not confuse it
        let nested = "IF inner THEN a ELSE b\nELSE outer";
        assert_eq!(find_outer_else(nested), Some(23));
    }

    #[test]
    fn t4_split_top_level_set_minus_distinguishes_set_difference_from_backslash_operators() {
        // Kills eval.rs:5681:.. (`!starts_with_tla_backslash_operator` mutants
        // and `is_whitespace`/`!=` mutants in split_top_level_set_minus).
        // The set-minus splitter walks the string char-by-char and must
        // distinguish a bare `\` (set difference) from a `\` that introduces a
        // TLA+ operator like `\union` / `\subseteq` / `\E`. No prior test
        // exercised whitespace-boundary detection.

        let parts = split_top_level_set_minus("S \\ T");
        assert_eq!(
            parts.len(),
            2,
            "S \\ T must split into [S, T], got {parts:?}"
        );
        assert_eq!(parts[0].trim(), "S");
        assert_eq!(parts[1].trim(), "T");

        // \union must NOT trigger a set-minus split.
        let parts = split_top_level_set_minus("S \\union T");
        assert_eq!(
            parts,
            vec!["S \\union T".to_string()],
            "\\union must not be split as set-minus"
        );
        // \subseteq must NOT trigger.
        let parts = split_top_level_set_minus("S \\subseteq T");
        assert_eq!(parts, vec!["S \\subseteq T".to_string()]);
        // \E quantifier must NOT trigger.
        let parts = split_top_level_set_minus("\\E i \\in S : P(i)");
        assert_eq!(parts, vec!["\\E i \\in S : P(i)".to_string()]);
        // No backslash at all => single element.
        let parts = split_top_level_set_minus("S \\union T \\union U");
        assert_eq!(parts.len(), 1);
        // Set difference inside a set literal is *not* top-level (kills the
        // `match ch '{' / '}' brace-tracker mutants).
        let parts = split_top_level_set_minus("{1 \\ 2}");
        assert_eq!(parts, vec!["{1 \\ 2}".to_string()]);

        // A USER-DEFINED backslash infix operator (not in the fixed built-in
        // list) must NOT be mis-split as set-minus. `\preceq` / `\sqsubseteq`
        // are followed immediately by a letter, so the `\` is a named-operator
        // token, not set difference. Regression for the false-violation +
        // undercount on LeastCircularSubstring (`rotation \preceq other.seq`).
        let parts = split_top_level_set_minus("a \\preceq b");
        assert_eq!(parts, vec!["a \\preceq b".to_string()]);
        let parts = split_top_level_set_minus("s0 \\sqsubseteq s1");
        assert_eq!(parts, vec!["s0 \\sqsubseteq s1".to_string()]);
    }

    #[test]
    fn t4_split_top_level_range_distinguishes_dotdot_from_dotdotdot() {
        // Kills mutants in eval.rs:5935 split_top_level_range — particularly
        // the `i > 0` check guarding a non-empty LHS, and the `after == Some('.')`
        // check that prevents `...` from being parsed as `..`.
        let r = split_top_level_range("1..5").expect("simple range must split");
        assert_eq!(r, ("1", "5"));

        // Range with whitespace.
        let r = split_top_level_range("a + 1 .. b - 1").expect("range with spaces must split");
        assert_eq!(r.0.trim(), "a + 1");
        assert_eq!(r.1.trim(), "b - 1");

        // No `..` => None.
        assert!(split_top_level_range("a + b").is_none());
        // `..` inside parens does not split at top level.
        assert!(
            split_top_level_range("(1..3)").is_none(),
            "`..` inside parens must not be top-level"
        );
        // `..` inside braces does not split at top level.
        assert!(
            split_top_level_range("{x \\in 1..5 : x > 2}").is_none(),
            "`..` inside braces must not be top-level"
        );
        // `..` inside square brackets does not split.
        assert!(
            split_top_level_range("[i \\in 1..3 |-> i*i]").is_none(),
            "`..` inside `[]` must not be top-level"
        );
    }

    #[test]
    fn t4_split_top_level_comparison_recognises_subseteq_and_distinguishes_set_ops() {
        // Kills mutants in eval.rs:5736 split_top_level_comparison — most
        // importantly the `pattern == "="` / `pattern.starts_with('\\')` /
        // `has_word_boundaries` mutants. T2.3 fixed `..` precedence relative
        // to set ops; this test pins that fix so it can't regress via mutation.
        let (lhs, op, rhs) = split_top_level_comparison("1..3 \\subseteq S")
            .expect("range \\subseteq set must split");
        assert_eq!(lhs.trim(), "1..3");
        assert_eq!(op, "\\subseteq");
        assert_eq!(rhs.trim(), "S");

        // `=` must not trigger inside `==` (although TLA+ doesn't use `==`,
        // the disambiguation guard at 5871 protects against `<>` / `>=`).
        let (_lhs, op, _rhs) = split_top_level_comparison("a >= b").expect("a >= b must split");
        assert_eq!(op, ">=");

        // `<=` and `=<` are both accepted (TLA+ leq).
        let (_lhs, op, _rhs) = split_top_level_comparison("a <= b").expect("a <= b must split");
        assert_eq!(op, "<=");
        let (_lhs, op, _rhs) = split_top_level_comparison("a =< b").expect("a =< b must split");
        assert_eq!(op, "=<");

        // `\notin` distinct from `\in`.
        let (_lhs, op, _rhs) =
            split_top_level_comparison("x \\notin S").expect("\\notin must split");
        assert_eq!(op, "\\notin");

        // Membership inside a function-application bracket is NOT top-level.
        // f[1] = 2  parses as f[1] = 2 (the `=` is at top-level despite the
        // `[]` wrapper around `1`).
        let (_lhs, op, _rhs) =
            split_top_level_comparison("f[1] = 2").expect("f[1] = 2 must split on top-level `=`");
        assert_eq!(op, "=");
    }
}
