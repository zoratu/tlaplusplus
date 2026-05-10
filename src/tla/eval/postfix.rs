//! Postfix operator and atomic expression parsing/evaluation.
//!
//! `parse_atom_with_postfix` consumes a base expression plus any number
//! of postfix selectors (`.field`, `[...]`, `'`). `eval_atom_with_postfix`
//! is the evaluating wrapper. `eval_primed_postfix_expr` handles primed
//! variables in transition contexts. `parse_base` is the leaf parser that
//! recognises literals, identifiers, parenthesised expressions, set/seq
//! literals, IF/CASE/LET — anything that can stand alone as a value.
//!
//! All four delegate back into `super::eval_expr_inner` for any
//! sub-expression evaluation; `parse_base` additionally calls
//! `super::eval_set_expression`, `super::eval_bracket_expression`,
//! `super::eval_module_instance_*`, `super::eval_enabled`, and
//! `super::eval_operator_call`.

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, apply_value, eval_bracket_expression, eval_bracket_index_key,
    eval_enabled, eval_expr_inner, eval_module_instance_call, eval_module_instance_ref,
    eval_operator_call, eval_set_expression, is_word_char, parse_argument_list,
    parse_identifier_prefix, parse_int_prefix, parse_string_literal_prefix,
    split_top_level, split_top_level_symbol, take_angle_group, take_bracket_group,
};

pub(super) fn parse_atom_with_postfix<'a>(
    expr: &'a str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<(TlaValue, &'a str)> {
    let (mut value, mut rest) = parse_base(expr, ctx, depth + 1)?;

    loop {
        let trimmed = rest.trim_start();
        if trimmed.is_empty() {
            break;
        }

        if let Some(after_dot) = trimmed.strip_prefix('.') {
            let (field, next_rest) = parse_identifier_prefix(after_dot)
                .ok_or_else(|| anyhow!("expected field after '.' in expression: {expr}"))?;
            value = value.select_key(&field)?.clone();
            rest = next_rest;
            continue;
        }

        if trimmed.starts_with('[') {
            let (inside, next_rest) = take_bracket_group(trimmed, '[', ']')?;
            let key = eval_bracket_index_key(inside, ctx, depth + 1)?;

            // Handle Lambda application differently from regular function application
            match &value {
                TlaValue::Lambda { .. } => {
                    value = apply_value(&value, vec![key], ctx, depth + 1)?;
                }
                _ => {
                    value = value.apply(&key)?.clone();
                }
            }

            rest = next_rest;
            continue;
        }

        break;
    }

    Ok((value, rest))
}

pub(super) fn eval_atom_with_postfix(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (value, rest) = parse_atom_with_postfix(expr, ctx, depth + 1)?;
    let trimmed_rest = rest.trim_start();
    if trimmed_rest == "'" {
        let base_len = expr.len().saturating_sub(rest.len());
        let base_expr = expr[..base_len].trim_end();
        return eval_primed_postfix_expr(base_expr, ctx, depth + 1);
    }
    // Handle \cdot action composition as trailing operator:
    // "A \cdot B" parses A as atom, leaving "\cdot B" as rest.
    if let Some(cdot_rest) = trimmed_rest.strip_prefix("\\cdot") {
        let rhs = cdot_rest.trim();
        if !rhs.is_empty() {
            // Evaluate the right-hand side and return its result.
            // For probing purposes both sides need to evaluate successfully.
            let rhs_value = eval_expr_inner(rhs, ctx, depth + 1)?;
            // Return the rhs value (action composition applies B after A).
            return Ok(rhs_value);
        }
    }
    if !trimmed_rest.is_empty() {
        return Err(anyhow!("unexpected trailing tokens in expr: {expr}"));
    }
    Ok(value)
}

pub(super) fn eval_primed_postfix_expr(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let s = expr.trim_start();
    let (name, mut rest) = parse_identifier_prefix(s)
        .ok_or_else(|| anyhow!("expected identifier in primed expression: {expr}'"))?;
    let mut value = {
        let trimmed = rest.trim_start();
        if trimmed.starts_with('(') {
            let (args_text, next_rest) = take_bracket_group(trimmed, '(', ')')?;
            let args = if args_text.trim().is_empty() {
                Vec::new()
            } else {
                split_top_level(args_text, ",", false)
                    .into_iter()
                    .map(|arg| eval_expr_inner(arg.trim(), ctx, depth + 1))
                    .collect::<Result<Vec<_>>>()?
            };
            rest = next_rest;
            let primed_ctx = ctx.with_primed_state_shadow_bindings();
            eval_operator_call(&name, args, &primed_ctx, depth + 1)?
        } else {
            ctx.resolve_identifier(&format!("{name}'"), depth + 1)?
        }
    };

    loop {
        let trimmed = rest.trim_start();
        if trimmed.is_empty() {
            break;
        }

        if let Some(after_dot) = trimmed.strip_prefix('.') {
            let (field, next_rest) = parse_identifier_prefix(after_dot)
                .ok_or_else(|| anyhow!("expected field after '.' in expression: {expr}'"))?;
            value = value.select_key(&field)?.clone();
            rest = next_rest;
            continue;
        }

        if trimmed.starts_with('[') {
            let (inside, next_rest) = take_bracket_group(trimmed, '[', ']')?;
            let key = eval_bracket_index_key(inside, ctx, depth + 1)?;
            value = value.apply(&key)?.clone();
            rest = next_rest;
            continue;
        }

        return Err(anyhow!(
            "unexpected trailing tokens in primed expr: {expr}'"
        ));
    }

    Ok(value)
}

pub(super) fn parse_base<'a>(
    expr: &'a str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<(TlaValue, &'a str)> {
    let s = expr.trim_start();
    if s.is_empty() {
        return Err(anyhow!("unexpected end of expression"));
    }

    if let Some(rest) = s.strip_prefix("TRUE")
        && !rest.chars().next().map(is_word_char).unwrap_or(false)
    {
        return Ok((TlaValue::Bool(true), rest));
    }
    if let Some(rest) = s.strip_prefix("FALSE")
        && !rest.chars().next().map(is_word_char).unwrap_or(false)
    {
        return Ok((TlaValue::Bool(false), rest));
    }

    if let Some((text, rest)) = parse_string_literal_prefix(s)? {
        return Ok((TlaValue::String(text), rest));
    }

    if let Some((value, rest)) = parse_int_prefix(s) {
        // Range operator a..b is now handled as a binary operator in comparison section
        return Ok((TlaValue::Int(value), rest));
    }

    if s.starts_with("<<") {
        let (inner, rest) = take_angle_group(s)?;
        let parts = split_top_level_symbol(inner, ",");
        let mut out = Vec::new();
        for part in parts {
            if part.trim().is_empty() {
                continue;
            }
            out.push(eval_expr_inner(&part, ctx, depth + 1)?);
        }
        return Ok((TlaValue::Seq(Arc::new(out)), rest));
    }

    if s.starts_with('{') {
        let (inner, rest) = take_bracket_group(s, '{', '}')?;
        let value = eval_set_expression(inner, ctx, depth + 1)?;
        return Ok((value, rest));
    }

    if s.starts_with('[') {
        let (inner, rest) = take_bracket_group(s, '[', ']')?;
        let value = eval_bracket_expression(inner, ctx, depth + 1)?;
        return Ok((value, rest));
    }

    if s.starts_with('(') {
        let (inner, rest) = take_bracket_group(s, '(', ')')?;
        let value = eval_expr_inner(inner, ctx, depth + 1)?;
        return Ok((value, rest));
    }

    if let Some((name, rest_after_name)) = parse_identifier_prefix(s) {
        let mut rest = rest_after_name;

        // Strip PlusCal label prefix: "label:: expr" or "label(params):: expr"
        // Labels are documentation-only in TLC — they don't affect evaluation.
        {
            let mut after_params = rest;
            // Skip optional parameter list
            if after_params.trim_start().starts_with('(') {
                if let Ok((_, tail)) = take_bracket_group(after_params.trim_start(), '(', ')') {
                    after_params = tail;
                }
            }
            if let Some(body) = after_params.trim_start().strip_prefix("::") {
                let body = body.trim_start();
                if !body.is_empty() {
                    // Re-parse from the body after the label
                    return parse_base(body, ctx, depth);
                }
            }
        }

        let has_runtime_value = ctx.runtime_value(&name).is_some();
        let operator_param_count = ctx.definition(&name).map(|def| def.params.len());

        // Check for module instance operator: Alias!Operator
        if rest.trim_start().starts_with('!') {
            let after_bang = rest.trim_start()[1..].trim_start();
            if let Some((operator_name, rest_after_op)) = parse_identifier_prefix(after_bang) {
                let trimmed_rest = rest_after_op.trim_start();

                // Check if this is a function call: Alias!Operator(args)
                if trimmed_rest.starts_with('(') {
                    let (args_text, next_rest) = take_bracket_group(trimmed_rest, '(', ')')?;
                    let args = parse_argument_list(args_text, ctx, depth + 1)?;
                    let value =
                        eval_module_instance_call(&name, &operator_name, args, ctx, depth + 1)?;
                    return Ok((value, next_rest));
                }

                // Otherwise it's a module instance reference to a constant/definition
                let value = eval_module_instance_ref(&name, &operator_name, ctx, depth + 1)?;
                return Ok((value, rest_after_op));
            }
        }

        if rest.trim_start().starts_with('(') {
            let trimmed_rest = rest.trim_start();
            let (args_text, next_rest) = take_bracket_group(trimmed_rest, '(', ')')?;
            let args = parse_argument_list(args_text, ctx, depth + 1)?;
            let value = eval_operator_call(&name, args, ctx, depth + 1)?;
            return Ok((value, next_rest));
        }

        if !has_runtime_value
            && operator_param_count.unwrap_or(0) > 0
            && rest.trim_start().starts_with('[')
        {
            let trimmed_rest = rest.trim_start();
            let (args_text, next_rest) = take_bracket_group(trimmed_rest, '[', ']')?;
            let args = parse_argument_list(args_text, ctx, depth + 1)?;
            let value = eval_operator_call(&name, args, ctx, depth + 1)?;
            return Ok((value, next_rest));
        }

        // Handle prefix operators like DOMAIN, UNION, and ENABLED that don't use parentheses
        if matches!(name.as_str(), "DOMAIN" | "UNION") && !has_runtime_value {
            // `DOMAIN ReplicatedLog[node]` should apply after postfix indexing, not
            // as `(DOMAIN ReplicatedLog)[node]`.
            let (arg_value, next_rest) =
                parse_atom_with_postfix(rest_after_name.trim_start(), ctx, depth + 1)?;
            let value = eval_operator_call(&name, vec![arg_value], ctx, depth + 1)?;
            return Ok((value, next_rest));
        }

        // Handle ENABLED operator: ENABLED ActionName
        if matches!(name.as_str(), "ENABLED") && !has_runtime_value {
            let action_name_part = rest_after_name.trim_start();
            if let Some((action_name, next_rest)) = parse_identifier_prefix(action_name_part) {
                let mut rest = next_rest.trim_start();
                let mut args = Vec::new();
                if rest.starts_with('(') {
                    let (args_text, tail) = take_bracket_group(rest, '(', ')')?;
                    args = parse_argument_list(args_text, ctx, depth + 1)?;
                    rest = tail;
                }
                let value = eval_enabled(&action_name, args, ctx, depth + 1)?;
                return Ok((value, rest));
            } else {
                return Err(anyhow!("ENABLED expects an action name"));
            }
        }

        let value = ctx.resolve_identifier(&name, depth + 1)?;
        rest = rest_after_name;
        return Ok((value, rest));
    }

    Err(anyhow!("unsupported expression atom: {expr}"))
}
