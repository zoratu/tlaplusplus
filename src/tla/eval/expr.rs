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

    // -- Dispatch the cascade via the `Op` seam (see `dispatch.rs` module docs). --
    //
    // `classify_op_cached` walks the splitter cascade once and returns the
    // chosen variant. Bodies below are unchanged. The seam is split into TWO
    // halves with defined-infix in between because defined-infix is
    // ctx-dependent (can't be cached): the original cascade ordering is
    // Comparison → DefinedInfix → SetOps → ... so we preserve that exact
    // precedence here.
    use super::dispatch::{Op, classify_op_cached, is_low_op};
    let op = classify_op_cached(expr);
    // PRECEDENCE: the original cascade probed defined-infix BETWEEN the
    // comparison arm and the set-op arms. Defined-infix is ctx-dependent so
    // it can't live in the cached classify. We preserve the precedence by
    // running it here, BEFORE dispatching a "low" Op variant.
    if is_low_op(&op)
        && let Some((lhs, op_name, rhs)) = split_top_level_defined_infix(expr, ctx)
    {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        return eval_operator_call(&op_name, vec![left, right], ctx, depth);
    }
    match op {
        Op::IndentedAnd(parts) => {
            for part in parts {
                if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(false));
                }
            }
            return Ok(TlaValue::Bool(true));
        }
        Op::IndentedOr(parts) => {
            for part in parts {
                if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(true));
                }
            }
            return Ok(TlaValue::Bool(false));
        }
        Op::Iff(parts) => {
            // <=> is right-associative-equivalent (a <=> b <=> c is rare; treat as fold).
            // We evaluate left-to-right reducing pairs by ==.
            let mut acc: Option<bool> = None;
            for part in &parts {
                let v = eval_expr_inner(part, ctx, depth + 1)?.as_bool()?;
                acc = Some(match acc {
                    None => v,
                    Some(prev) => prev == v,
                });
            }
            return Ok(TlaValue::Bool(acc.expect("iff parts non-empty")));
        }
        Op::Implies(parts) => {
            return Ok(TlaValue::Bool(eval_implies_parts(
                &parts,
                ctx,
                depth + 1,
            )?));
        }
        Op::Or(parts) => {
            for part in parts {
                if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(true));
                }
            }
            return Ok(TlaValue::Bool(false));
        }
        Op::OrPrefixSingle => {
            // Expression starts with \/ but has only one disjunct
            // e.g., "\/ x > 0" should be treated as just "x > 0"
            let rest = expr.trim().trim_start_matches("\\/").trim_start();
            if rest.is_empty() {
                return Err(anyhow!("empty disjunction"));
            }
            return eval_expr_inner(rest, ctx, depth + 1);
        }
        Op::And(parts) => {
            for part in parts {
                if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(false));
                }
            }
            return Ok(TlaValue::Bool(true));
        }
        Op::AndPrefixSingle => {
            let rest = expr.trim().trim_start_matches("/\\").trim_start();
            if rest.is_empty() {
                return Err(anyhow!("empty conjunction"));
            }
            return eval_expr_inner(rest, ctx, depth + 1);
        }
        Op::Not => {
            // safe: `Op::Not` is returned only when expr.starts_with('~')
            let rest = expr.strip_prefix('~').expect("Op::Not implies ~ prefix");
            return Ok(TlaValue::Bool(
                !eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_bool()?,
            ));
        }
        Op::Cdot(parts) => {
            // Action composition operator \cdot: A \cdot B
            // For expression evaluation purposes (probing), treat as conjunction —
            // both sides must evaluate successfully. Return the result of the last part.
            let mut result = TlaValue::Bool(true);
            for part in &parts {
                result = eval_expr_inner(part, ctx, depth + 1)?;
            }
            return Ok(result);
        }
        Op::Comparison { lhs, op, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?;
            return match op.as_str() {
                "=" => {
                    let right = eval_expr_inner(&rhs, ctx, depth + 1)?;
                    Ok(TlaValue::Bool(left == right))
                }
                "/=" | "#" => {
                    let right = eval_expr_inner(&rhs, ctx, depth + 1)?;
                    Ok(TlaValue::Bool(left != right))
                }
                "<" | "<=" | "=<" | "\\leq" | ">" | ">=" | "\\geq" => {
                    let right = eval_expr_inner(&rhs, ctx, depth + 1)?;
                    let cmp = match op.as_str() {
                        "<" => left.as_int()? < right.as_int()?,
                        "<=" | "=<" | "\\leq" => left.as_int()? <= right.as_int()?,
                        ">" => left.as_int()? > right.as_int()?,
                        ">=" | "\\geq" => left.as_int()? >= right.as_int()?,
                        _ => unreachable!(),
                    };
                    Ok(TlaValue::Bool(cmp))
                }
                "\\in" => Ok(TlaValue::Bool(matches_membership_expr(
                    &left,
                    rhs.trim(),
                    ctx,
                    depth + 1,
                )?)),
                "\\notin" => Ok(TlaValue::Bool(!matches_membership_expr(
                    &left,
                    rhs.trim(),
                    ctx,
                    depth + 1,
                )?)),
                "\\subseteq" => {
                    let right = eval_expr_inner(&rhs, ctx, depth + 1)?;
                    let lhs_set = left.as_set()?;
                    let rhs_set = right.as_set()?;
                    Ok(TlaValue::Bool(lhs_set.iter().all(|v| rhs_set.contains(v))))
                }
                _ => Err(anyhow!("unsupported comparison operator {op}")),
            };
        }
        Op::Union(parts) => {
            let mut out = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
                out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
            }
            return Ok(out);
        }
        Op::Intersect(parts) => {
            let mut out = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
                out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
            }
            return Ok(out);
        }
        Op::SetMinus(parts) => {
            let mut out = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
                out = out.set_minus(&eval_expr_inner(part, ctx, depth + 1)?)?;
            }
            return Ok(out);
        }
        Op::Concat(parts) => {
            let mut out = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
                let rhs = eval_expr_inner(part, ctx, depth + 1)?;
                out = seq_or_string_concat(out, rhs)?;
            }
            return Ok(out);
        }
        Op::Cartesian(parts) => {
            let mut result = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
                let rhs = eval_expr_inner(part, ctx, depth + 1)?;
                let lhs_set = result.as_set()?;
                let rhs_set = rhs.as_set()?;
                ctx.check_budget(lhs_set.len() * rhs_set.len())?;
                let mut product = BTreeSet::new();
                for lhs_val in lhs_set {
                    for rhs_val in rhs_set {
                        let tuple =
                            TlaValue::Seq(Arc::new(vec![lhs_val.clone(), rhs_val.clone()]));
                        product.insert(tuple);
                    }
                }
                result = TlaValue::Set(Arc::new(product));
            }
            return Ok(result);
        }
        Op::FuncOverride(parts) => {
            let mut out = eval_expr_inner(&parts[0], ctx, depth + 1)?;
            for part in &parts[1..] {
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
        Op::FuncPair { lhs, rhs } => {
            let key = eval_expr_inner(&lhs, ctx, depth + 1)?;
            let val = eval_expr_inner(&rhs, ctx, depth + 1)?;
            let mut func = BTreeMap::new();
            func.insert(key, val);
            return Ok(TlaValue::Function(Arc::new(func)));
        }
        Op::Xor { lhs, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?.as_int()?;
            let right = eval_expr_inner(&rhs, ctx, depth + 1)?.as_int()?;
            return Ok(TlaValue::Int(left ^ right));
        }
        Op::Range { lhs, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?;
            let right = eval_expr_inner(&rhs, ctx, depth + 1)?;
            let start = left.as_int()?;
            let end = right.as_int()?;
            let range_set: BTreeSet<TlaValue> = if start <= end {
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
        Op::Additive { lhs, op, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?.as_int()?;
            let right = eval_expr_inner(&rhs, ctx, depth + 1)?.as_int()?;
            return match op {
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
        Op::Multiplicative { lhs, op, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?.as_int()?;
            let right = eval_expr_inner(&rhs, ctx, depth + 1)?.as_int()?;
            return match op.as_str() {
                "*" => left
                    .checked_mul(right)
                    .map(TlaValue::Int)
                    .ok_or_else(|| anyhow!("integer overflow: {} * {}", left, right)),
                "\\div" => {
                    if right == 0 {
                        Err(anyhow!("division by zero"))
                    } else {
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
        Op::Exp { lhs, rhs } => {
            let left = eval_expr_inner(&lhs, ctx, depth + 1)?.as_int()?;
            let right = eval_expr_inner(&rhs, ctx, depth + 1)?.as_int()?;
            if right < 0 {
                return Err(anyhow!("exponent must be non-negative, got {right}"));
            }
            let value = left
                .checked_pow(right as u32)
                .ok_or_else(|| anyhow!("integer exponent overflow: {left}^{right}"))?;
            return Ok(TlaValue::Int(value));
        }
        Op::Atom => {
            // Fall through to: defined infix (ctx-dependent, can't cache),
            // UNCHANGED / SUBSET / UNION keyword shortcuts, unary -, atom.
        }
    }

    // Comparison handled via Op::Comparison (cached). Defined-infix CANNOT
    // be cached (ctx-dependent — see dispatch.rs module docs), so its probe
    // remains inline here.
    if let Some((lhs, op, rhs)) = split_top_level_defined_infix(expr, ctx) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        return eval_operator_call(&op, vec![left, right], ctx, depth);
    }

    // Set ops, sequence concat, cartesian product, function override, :>, ^^,
    // range, additive, multiplicative, ^ are now all dispatched via Op (above).

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
