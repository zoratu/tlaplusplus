//! Top-of-grammar dispatcher: `eval_expr_inner` is the ~492-line
//! precedence ladder that implements TLA+ expression evaluation.
//!
//! It hand-rolls the precedence by sequentially trying every splitter in
//! a precise order (`split_top_level_symbol("<=>")` before `=>`,
//! `split_top_level_keyword("\\union")` before `\cup` aliases, etc.).
//! **Reordering any pair of branches changes parse semantics.** The
//! T1.6 (FingerprintStoreResize) regression hit exactly that — splitting
//! `<=>` after `=>` mis-parsed `(seqlock % 2 = 1) <=> resizing` because
//! `<=>` contains `=>`. The function therefore lives in one piece; it is
//! not internally splittable without introducing an explicit precedence
//! table, which is a separate refactor.

use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, MAX_EVAL_DEPTH, eval_atom_with_postfix, eval_case_expression,
    eval_choose_expression, eval_if_expression, eval_implies_parts, eval_lambda_expression,
    eval_let_expression, eval_operator_call, eval_quantifier_expression, matches_membership_expr,
    powerset, seq_or_string_concat, split_indented_top_level_boolean, split_once_top_level,
    split_top_level_additive, split_top_level_comparison, split_top_level_defined_infix,
    split_top_level_keyword, split_top_level_multiplicative, split_top_level_range,
    split_top_level_set_minus, split_top_level_symbol, starts_with_keyword, strip_outer_parens,
};

pub(super) fn eval_expr_inner(raw_expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    // DEBUG: Track expression evaluation path when depth is high
    if std::env::var("TLAPP_TRACE_EVAL").is_ok() && depth >= 3 {
        let preview_len = if raw_expr.starts_with("LET") {
            1000
        } else {
            100
        };
        eprintln!(
            "TRACE depth={}: expr={:?}",
            depth,
            &raw_expr[..preview_len.min(raw_expr.len())]
        );
    }

    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("max expression recursion depth exceeded"));
    }

    let mut expr = raw_expr.trim();
    if expr.is_empty() {
        return Err(anyhow!("empty expression"));
    }

    expr = strip_outer_parens(expr);

    if expr == "TRUE" {
        return Ok(TlaValue::Bool(true));
    }
    if expr == "FALSE" {
        return Ok(TlaValue::Bool(false));
    }
    if expr == "BOOLEAN" {
        // BOOLEAN is the set {TRUE, FALSE}
        let set: BTreeSet<TlaValue> = [TlaValue::Bool(false), TlaValue::Bool(true)]
            .into_iter()
            .collect();
        return Ok(TlaValue::Set(Arc::new(set)));
    }
    if expr == "@" {
        return ctx
            .runtime_value("@")
            .ok_or_else(|| anyhow!("'@' used outside EXCEPT update"));
    }

    if starts_with_keyword(expr, "LET") {
        return eval_let_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "IF") {
        return eval_if_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "CASE") {
        return eval_case_expression(expr, ctx, depth + 1);
    }
    if expr.starts_with("\\E") || expr.starts_with("\\A") {
        return eval_quantifier_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "CHOOSE") {
        return eval_choose_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "LAMBDA") {
        return eval_lambda_expression(expr, ctx, depth + 1);
    }

    if expr.starts_with("/\\")
        && let Some(and_parts) = split_indented_top_level_boolean(expr, "/\\")
        && and_parts.len() > 1
    {
        for part in and_parts {
            if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(false));
            }
        }
        return Ok(TlaValue::Bool(true));
    }

    if expr.starts_with("\\/")
        && let Some(or_parts) = split_indented_top_level_boolean(expr, "\\/")
        && or_parts.len() > 1
    {
        for part in or_parts {
            if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(true));
            }
        }
        return Ok(TlaValue::Bool(false));
    }

    // Logical equivalence (biconditional): a <=> b
    // Lower precedence than =>, so split first; otherwise the `=>` splitter
    // below would match the `=>` *inside* `<=>` and silently mis-parse
    // `(seqlock % 2 = 1) <=> resizing` as `((seqlock % 2 = 1) <) => resizing`.
    // T1.6 (FingerprintStoreResize) regression: see RELEASE_1.0.0_LOG.md.
    let iff_parts = split_top_level_symbol(expr, "<=>");
    if iff_parts.len() > 1 {
        // <=> is right-associative-equivalent (a <=> b <=> c is rare; treat as fold).
        // We evaluate left-to-right reducing pairs by ==.
        let mut acc: Option<bool> = None;
        for part in &iff_parts {
            let v = eval_expr_inner(part, ctx, depth + 1)?.as_bool()?;
            acc = Some(match acc {
                None => v,
                Some(prev) => prev == v,
            });
        }
        return Ok(TlaValue::Bool(acc.expect("iff_parts non-empty")));
    }

    let implies_parts = split_top_level_symbol(expr, "=>");
    if implies_parts.len() > 1 {
        return Ok(TlaValue::Bool(eval_implies_parts(
            &implies_parts,
            ctx,
            depth + 1,
        )?));
    }

    let or_parts = split_top_level_symbol(expr, "\\/");
    if or_parts.len() > 1 {
        for part in or_parts {
            if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(true));
            }
        }
        return Ok(TlaValue::Bool(false));
    }
    // Handle case where expression starts with \/ but has only one disjunct
    // e.g., "\/ x > 0" should be treated as just "x > 0"
    if expr.trim().starts_with("\\/") && or_parts.len() == 1 {
        let rest = expr.trim().trim_start_matches("\\/").trim_start();
        if rest.is_empty() {
            return Err(anyhow!("empty disjunction"));
        }
        return eval_expr_inner(rest, ctx, depth + 1);
    }

    let and_parts = split_top_level_symbol(expr, "/\\");
    if and_parts.len() > 1 {
        for part in and_parts {
            if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(false));
            }
        }
        return Ok(TlaValue::Bool(true));
    }
    // Handle case where expression starts with /\ but has only one conjunct
    // e.g., "/\ x > 0" should be treated as just "x > 0"
    if expr.trim().starts_with("/\\") && and_parts.len() == 1 {
        let rest = expr.trim().trim_start_matches("/\\").trim_start();
        if rest.is_empty() {
            return Err(anyhow!("empty conjunction"));
        }
        return eval_expr_inner(rest, ctx, depth + 1);
    }

    if let Some(rest) = expr.strip_prefix('~') {
        return Ok(TlaValue::Bool(
            !eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_bool()?,
        ));
    }

    // Action composition operator \cdot: A \cdot B
    // For expression evaluation purposes (probing), treat as conjunction —
    // both sides must evaluate successfully. Return the result of the last part.
    let cdot_parts = split_top_level_keyword(expr, "\\cdot");
    if cdot_parts.len() > 1 {
        let mut result = TlaValue::Bool(true);
        for part in &cdot_parts {
            result = eval_expr_inner(part, ctx, depth + 1)?;
        }
        return Ok(result);
    }

    if let Some((lhs, op, rhs)) = split_top_level_comparison(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;

        return match op {
            "=" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                Ok(TlaValue::Bool(left == right))
            }
            "/=" | "#" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                Ok(TlaValue::Bool(left != right))
            }
            "<" | "<=" | "=<" | "\\leq" | ">" | ">=" | "\\geq" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let cmp = match op {
                    "<" => left.as_int()? < right.as_int()?,
                    "<=" | "=<" | "\\leq" => left.as_int()? <= right.as_int()?,
                    ">" => left.as_int()? > right.as_int()?,
                    ">=" | "\\geq" => left.as_int()? >= right.as_int()?,
                    _ => unreachable!(),
                };
                Ok(TlaValue::Bool(cmp))
            }
            "\\in" => {
                let rhs_trimmed = rhs.trim();
                Ok(TlaValue::Bool(matches_membership_expr(
                    &left,
                    rhs_trimmed,
                    ctx,
                    depth + 1,
                )?))
            }
            "\\notin" => {
                let rhs_trimmed = rhs.trim();
                Ok(TlaValue::Bool(!matches_membership_expr(
                    &left,
                    rhs_trimmed,
                    ctx,
                    depth + 1,
                )?))
            }
            "\\subseteq" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let lhs_set = left.as_set()?;
                let rhs_set = right.as_set()?;
                Ok(TlaValue::Bool(lhs_set.iter().all(|v| rhs_set.contains(v))))
            }
            _ => Err(anyhow!("unsupported comparison operator {op}")),
        };
    }

    if let Some((lhs, op, rhs)) = split_top_level_defined_infix(expr, ctx) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        return eval_operator_call(&op, vec![left, right], ctx, depth);
    }

    let union_parts = split_top_level_keyword(expr, "\\union");
    if union_parts.len() > 1 {
        let mut out = eval_expr_inner(&union_parts[0], ctx, depth + 1)?;
        for part in &union_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }
    let union_parts = split_top_level_keyword(expr, "\\cup");
    if union_parts.len() > 1 {
        let mut out = eval_expr_inner(&union_parts[0], ctx, depth + 1)?;
        for part in &union_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let cup_parts = split_top_level_keyword(expr, "\\cup");
    if cup_parts.len() > 1 {
        let mut out = eval_expr_inner(&cup_parts[0], ctx, depth + 1)?;
        for part in &cup_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let intersect_parts = split_top_level_keyword(expr, "\\intersect");
    if intersect_parts.len() > 1 {
        let mut out = eval_expr_inner(&intersect_parts[0], ctx, depth + 1)?;
        for part in &intersect_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }
    let intersect_parts = split_top_level_keyword(expr, "\\cap");
    if intersect_parts.len() > 1 {
        let mut out = eval_expr_inner(&intersect_parts[0], ctx, depth + 1)?;
        for part in &intersect_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let cap_parts = split_top_level_keyword(expr, "\\cap");
    if cap_parts.len() > 1 {
        let mut out = eval_expr_inner(&cap_parts[0], ctx, depth + 1)?;
        for part in &cap_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let minus_parts = split_top_level_set_minus(expr);
    if minus_parts.len() > 1 {
        let mut out = eval_expr_inner(&minus_parts[0], ctx, depth + 1)?;
        for part in &minus_parts[1..] {
            out = out.set_minus(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let mut concat_parts = split_top_level_keyword(expr, "\\o");
    if concat_parts.len() == 1 {
        concat_parts = split_top_level_keyword(expr, "\\circ");
    }
    if concat_parts.len() > 1 {
        let mut out = eval_expr_inner(&concat_parts[0], ctx, depth + 1)?;
        for part in &concat_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            out = seq_or_string_concat(out, rhs)?;
        }
        return Ok(out);
    }

    // Handle cartesian product: \X and \times are synonyms
    let mut cartesian_parts = split_top_level_keyword(expr, "\\X");
    if cartesian_parts.len() == 1 {
        // Try \times as an alternative syntax
        cartesian_parts = split_top_level_keyword(expr, "\\times");
    }
    if cartesian_parts.len() > 1 {
        let mut result = eval_expr_inner(&cartesian_parts[0], ctx, depth + 1)?;
        for part in &cartesian_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            let lhs_set = result.as_set()?;
            let rhs_set = rhs.as_set()?;
            ctx.check_budget(lhs_set.len() * rhs_set.len())?;
            let mut product = BTreeSet::new();
            for lhs_val in lhs_set {
                for rhs_val in rhs_set {
                    let tuple = TlaValue::Seq(Arc::new(vec![lhs_val.clone(), rhs_val.clone()]));
                    product.insert(tuple);
                }
            }
            result = TlaValue::Set(Arc::new(product));
        }
        return Ok(result);
    }

    // TLC module: Function override operator @@
    // f @@ g merges two functions, with g taking precedence on overlapping keys
    let func_override_parts = split_top_level_symbol(expr, "@@");
    if func_override_parts.len() > 1 {
        let mut out = eval_expr_inner(&func_override_parts[0], ctx, depth + 1)?;
        for part in &func_override_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            let left_func = out.as_function()?;
            let right_func = rhs.as_function()?;
            let mut result = left_func.clone();
            for (k, v) in right_func.iter() {
                result.insert(k.clone(), v.clone());
            }
            out = TlaValue::Function(Arc::new(result));
        }
        return Ok(out);
    }

    // TLC module: Function pair constructor :>
    // a :> b creates a function mapping a to b (single mapping {a -> b})
    if let Some((lhs, rhs)) = split_once_top_level(expr, ":>") {
        let key = eval_expr_inner(lhs.trim(), ctx, depth + 1)?;
        let val = eval_expr_inner(rhs.trim(), ctx, depth + 1)?;
        let mut func = BTreeMap::new();
        func.insert(key, val);
        return Ok(TlaValue::Function(Arc::new(func)));
    }

    if let Some((lhs, rhs)) = split_once_top_level(expr, "^^") {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return Ok(TlaValue::Int(left ^ right));
    }

    // Range operator `a..b` (precedence 9-9 in TLA+). Slotted between the
    // set-op tier (8-8: `\union`, `\intersect`, `\\`) above and the additive
    // tier (10-10: `+`, `-`) below — `..` binds tighter than set ops/relops
    // but looser than `+`/`-`. So `1+2..n*3 \subseteq S` parses as
    // `((1+2) .. (n*3)) \subseteq S`.
    if let Some((lhs, rhs)) = split_top_level_range(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        let start = left.as_int()?;
        let end = right.as_int()?;
        let range_set: BTreeSet<TlaValue> = if start <= end {
            // T201: charge the range size against the eval budget BEFORE
            // allocating. Without this, fuzz / probe paths that hit a giant
            // range like `1..1_000_000_000` would OOM before any per-element
            // check fires.
            let span = (end - start).saturating_add(1);
            let cost = if span <= 0 || span as u64 > usize::MAX as u64 {
                usize::MAX
            } else {
                span as usize
            };
            ctx.check_budget(cost)?;
            (start..=end).map(TlaValue::Int).collect()
        } else {
            BTreeSet::new()
        };
        return Ok(TlaValue::Set(Arc::new(range_set)));
    }

    if let Some((lhs, op, rhs)) = split_top_level_additive(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return match op {
            // T101.1: use checked_add/sub so overflow becomes Err instead
            // of a Rust panic. Mirror in `compiled_eval.rs`.
            '+' => left
                .checked_add(right)
                .map(TlaValue::Int)
                .ok_or_else(|| anyhow!("integer overflow: {} + {}", left, right)),
            '-' => left
                .checked_sub(right)
                .map(TlaValue::Int)
                .ok_or_else(|| anyhow!("integer overflow: {} - {}", left, right)),
            _ => Err(anyhow!("unsupported additive operator '{op}'")),
        };
    }

    if let Some((lhs, op, rhs)) = split_top_level_multiplicative(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return match op {
            // T101.1: use checked_mul so overflow becomes an Err instead of
            // a Rust panic — the fuzz pass surfaced one case where a
            // 3-term `*` chain overflowed `i64` and aborted the process
            // (`attempt to multiply with overflow`). Mirror in
            // `compiled_eval.rs::CompiledExpr::Mul`.
            "*" => left
                .checked_mul(right)
                .map(TlaValue::Int)
                .ok_or_else(|| anyhow!("integer overflow: {} * {}", left, right)),
            "\\div" => {
                if right == 0 {
                    Err(anyhow!("division by zero"))
                } else {
                    // i64::MIN / -1 also overflows; use checked_div.
                    left.checked_div(right)
                        .map(TlaValue::Int)
                        .ok_or_else(|| anyhow!("integer overflow: {} \\div {}", left, right))
                }
            }
            "%" => {
                if right == 0 {
                    Err(anyhow!("modulo by zero"))
                } else {
                    left.checked_rem(right)
                        .map(TlaValue::Int)
                        .ok_or_else(|| anyhow!("integer overflow: {} % {}", left, right))
                }
            }
            _ => Err(anyhow!("unsupported multiplicative operator {op}")),
        };
    }

    if let Some((lhs, rhs)) = split_once_top_level(expr, "^") {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        if right < 0 {
            return Err(anyhow!("exponent must be non-negative, got {right}"));
        }
        let value = left
            .checked_pow(right as u32)
            .ok_or_else(|| anyhow!("integer exponent overflow: {left}^{right}"))?;
        return Ok(TlaValue::Int(value));
    }

    // Handle UNCHANGED in expression/guard context.
    // When UNCHANGED appears inside a disjunction that is evaluated as a guard
    // (e.g., \/ Action1(self) \/ Action2(self) \/ UNCHANGED vars), the action
    // semantics (keeping variables at their current values) are handled by the
    // action IR layer. In the expression evaluation context, UNCHANGED simply
    // represents an always-enabled stuttering step, so we return TRUE.
    if starts_with_keyword(expr, "UNCHANGED") {
        return Ok(TlaValue::Bool(true));
    }

    if starts_with_keyword(expr, "SUBSET") {
        let rest = expr["SUBSET".len()..].trim();
        let set = eval_expr_inner(rest, ctx, depth + 1)?;
        return Ok(TlaValue::Set(Arc::new(powerset(set.as_set()?, ctx)?)));
    }

    if starts_with_keyword(expr, "UNION") {
        let rest = expr["UNION".len()..].trim();
        let mut union = BTreeSet::new();
        for set in eval_expr_inner(rest, ctx, depth + 1)?.as_set()? {
            union.extend(set.as_set()?.iter().cloned());
        }
        return Ok(TlaValue::Set(Arc::new(union)));
    }

    if let Some(rest) = expr.strip_prefix('-')
        && !rest.trim().is_empty()
    {
        return Ok(TlaValue::Int(
            -eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_int()?,
        ));
    }

    eval_atom_with_postfix(expr, ctx, depth + 1)
}
