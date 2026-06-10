//! Bracket-form expressions: function literals `[d -> r]`, record
//! literals `[f1 |-> v1, ...]`, function/record application via the
//! generalized `[a, b]` syntax, and `EXCEPT` updates.
//!
//! Also hosts `apply_value` — the surface for one-step function/record
//! application called from compiled-eval — and `PathSegment` plus
//! `get_path_value`/`set_path_value` for navigating nested record/seq
//! paths in EXCEPT.

use anyhow::{Context, Result, anyhow};
use crate::tla::hashed_arc::HashedArc;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, bind_param_value, collect_function_mapping,
    contains_top_level_keyword, eval_expr_inner, find_top_level_char,
    find_top_level_keyword_index, parse_binders, parse_identifier_prefix,
    parse_string_literal_prefix, record_key_from_value, split_once_top_level,
    split_top_level_symbol, take_bracket_group, tla_to_string, try_eval_record_set,
};

pub(super) fn eval_bracket_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    if let Some(idx) = find_top_level_keyword_index(expr, "EXCEPT") {
        let base_text = expr[..idx].trim();
        let updates_text = expr[idx + "EXCEPT".len()..].trim();
        return eval_except_expression(base_text, updates_text, ctx, depth + 1);
    }

    if let Some((lhs, rhs)) = split_once_top_level(expr, "|->") {
        let lhs = lhs.trim();
        let rhs = rhs.trim();

        if contains_top_level_keyword(lhs, "\\in") {
            let binders = parse_binders(lhs, ctx, depth + 1)?;
            let mut assignments = BTreeMap::new();
            let mut map = BTreeMap::new();
            collect_function_mapping(0, &binders, rhs, &mut assignments, &mut map, ctx, depth + 1)?;
            return Ok(TlaValue::Function(HashedArc::new(map)));
        }

        let mut record = BTreeMap::new();
        for entry in split_top_level_symbol(expr, ",") {
            let (key_text, value_text) = split_once_top_level(&entry, "|->")
                .ok_or_else(|| anyhow!("invalid record entry: {entry}"))?;
            let key = parse_record_key(key_text.trim())?;
            let value = eval_expr_inner(value_text.trim(), ctx, depth + 1)?;
            record.insert(key, value);
        }
        return Ok(TlaValue::Record(HashedArc::new(record)));
    }

    // Check for function set: [Domain -> Range]
    if let Some((domain_text, range_text)) = split_once_top_level(expr, "->") {
        let domain_text = domain_text.trim();
        let range_text = range_text.trim();

        let domain_val = eval_expr_inner(domain_text, ctx, depth + 1)?;
        let range_val = eval_expr_inner(range_text, ctx, depth + 1)?;

        let domain_set = domain_val
            .as_set()
            .with_context(|| format!("function set domain is not a set: {domain_text}"))?;
        let range_set = range_val
            .as_set()
            .with_context(|| format!("function set range is not a set: {range_text}"))?;

        // Generate all possible functions from domain to range
        let domain_elems: Vec<TlaValue> = domain_set.iter().cloned().collect();
        let range_elems: Vec<TlaValue> = range_set.iter().cloned().collect();

        if domain_elems.is_empty() {
            // Only one function from empty domain: the empty function
            let empty_func = BTreeMap::new();
            let mut result = BTreeSet::new();
            result.insert(TlaValue::Function(HashedArc::new(empty_func)));
            return Ok(TlaValue::Set(HashedArc::new(result)));
        }

        if range_elems.is_empty() {
            // No functions from non-empty domain to empty range
            return Ok(TlaValue::Set(HashedArc::new(BTreeSet::new())));
        }

        let n = domain_elems.len();
        let m = range_elems.len();

        // Limit the size to avoid memory explosion
        let max_functions = 1_000_000usize;
        let total = (m as u64).saturating_pow(n as u32);
        if total > max_functions as u64 {
            return Err(anyhow!(
                "function set [D -> R] too large: {} elements in domain, {} in range = {} functions (max {})",
                n,
                m,
                total,
                max_functions
            ));
        }
        ctx.check_budget(total as usize)?;

        // Generate all functions by iterating through all combinations
        let mut result = BTreeSet::new();
        let mut indices = vec![0usize; n];

        loop {
            // Build function for current indices
            let mut func: BTreeMap<TlaValue, TlaValue> = BTreeMap::new();
            for (i, d) in domain_elems.iter().enumerate() {
                func.insert(d.clone(), range_elems[indices[i]].clone());
            }
            result.insert(TlaValue::Function(HashedArc::new(func)));

            // Increment indices (like counting in base m)
            let mut carry = true;
            for idx in indices.iter_mut() {
                if carry {
                    *idx += 1;
                    if *idx >= m {
                        *idx = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        return Ok(TlaValue::Set(HashedArc::new(result)));
    }

    // Check for record set construction: [field: Set, field2: Set, ...]
    // This creates the set of all records where each field takes values from its corresponding set
    // For example: [a: {1,2}, b: {3,4}] = {[a |-> 1, b |-> 3], [a |-> 1, b |-> 4], [a |-> 2, b |-> 3], [a |-> 2, b |-> 4]}
    if let Some(record_set) = try_eval_record_set(expr, ctx, depth + 1)? {
        return Ok(record_set);
    }

    Err(anyhow!("unsupported bracket expression: [{expr}]"))
}

/// Try to evaluate a record set construction: [field: Set, field2: Set, ...]
/// This creates the Cartesian product of all field/set pairs as records.

pub(super) fn eval_except_expression(
    base_text: &str,
    updates_text: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    let mut current = eval_expr_inner(base_text, ctx, depth + 1)?;

    for raw_update in split_top_level_symbol(updates_text, ",") {
        let update = raw_update.trim();
        if update.is_empty() {
            continue;
        }
        let update = update
            .strip_prefix('!')
            .ok_or_else(|| anyhow!("EXCEPT update must start with '!': {update}"))?;

        let eq_idx = find_top_level_char(update, '=')
            .ok_or_else(|| anyhow!("EXCEPT update missing '=': {update}"))?;
        let path_text = update[..eq_idx].trim();
        let rhs_text = update[eq_idx + 1..].trim();

        let path = parse_except_path(path_text, ctx, depth + 1)?;
        let old_value = get_path_value(&current, &path)?;
        let rhs_ctx = ctx.with_local_value("@", old_value);
        let new_value = eval_expr_inner(rhs_text, &rhs_ctx, depth + 1)?;
        current = set_path_value(&current, &path, new_value)?;
    }

    Ok(current)
}

pub(crate) fn apply_value(
    func: &TlaValue,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    match func {
        TlaValue::Lambda {
            params,
            body,
            captured_locals,
        } => {
            if params.len() != args.len() {
                return Err(anyhow!(
                    "Lambda arity mismatch: expected {}, got {}",
                    params.len(),
                    args.len()
                ));
            }

            // Create a new context with captured locals
            let mut lambda_ctx = ctx.clone();
            lambda_ctx.locals = Rc::new((**captured_locals).clone());

            // Bind arguments to parameters
            {
                let locals_mut = Rc::make_mut(&mut lambda_ctx.locals);
                for (param, arg) in params.iter().zip(args.into_iter()) {
                    bind_param_value(locals_mut, param, arg)?;
                }
            }

            // Evaluate the lambda body
            eval_expr_inner(body, &lambda_ctx, depth + 1)
        }
        TlaValue::Function(map) => {
            if args.len() != 1 {
                return Err(anyhow!("Function application expects exactly 1 argument"));
            }
            map.get(&args[0]).cloned().ok_or_else(|| {
                // Build helpful error message with key and domain
                let key_str = tla_to_string(&args[0]);
                let domain_keys: Vec<String> = map.keys().take(10).map(tla_to_string).collect();
                let domain_str = if map.len() > 10 {
                    format!("{{{},...}} ({} keys)", domain_keys.join(", "), map.len())
                } else {
                    format!("{{{}}}", domain_keys.join(", "))
                };
                anyhow!(
                    "function application failed: key {} not in domain {}",
                    key_str,
                    domain_str
                )
            })
        }
        _ => Err(anyhow!("Cannot apply non-function value: {func:?}")),
    }
}


#[derive(Debug, Clone)]
pub(super) enum PathSegment {
    Field(String),
    Index(TlaValue),
}

pub(super) fn parse_except_path(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<PathSegment>> {
    let mut out = Vec::new();
    let mut rest = expr.trim();

    while !rest.is_empty() {
        if let Some(after_dot) = rest.strip_prefix('.') {
            let (field, next_rest) = parse_identifier_prefix(after_dot)
                .ok_or_else(|| anyhow!("invalid EXCEPT field path segment: {expr}"))?;
            out.push(PathSegment::Field(field));
            rest = next_rest.trim_start();
            continue;
        }

        if rest.starts_with('[') {
            let (inside, next_rest) = take_bracket_group(rest, '[', ']')?;
            let key = eval_bracket_index_key(inside, ctx, depth + 1)?;
            out.push(PathSegment::Index(key));
            rest = next_rest.trim_start();
            continue;
        }

        return Err(anyhow!("invalid EXCEPT path segment in '{expr}'"));
    }

    if out.is_empty() {
        return Err(anyhow!("EXCEPT path is empty"));
    }

    Ok(out)
}

fn get_path_value(base: &TlaValue, path: &[PathSegment]) -> Result<TlaValue> {
    if path.is_empty() {
        return Ok(base.clone());
    }

    match &path[0] {
        PathSegment::Field(name) => match base {
            TlaValue::Record(map) => {
                let next = map.get(name).cloned().unwrap_or(TlaValue::Undefined);
                get_path_value(&next, &path[1..])
            }
            _ => Err(anyhow!("field access on non-record value {base:?}")),
        },
        PathSegment::Index(key) => match base {
            TlaValue::Function(map) => {
                let next = map.get(key).cloned().unwrap_or(TlaValue::Undefined);
                get_path_value(&next, &path[1..])
            }
            TlaValue::Record(map) => {
                let k = record_key_from_value(key)?;
                let next = map.get(&k).cloned().unwrap_or(TlaValue::Undefined);
                get_path_value(&next, &path[1..])
            }
            TlaValue::Seq(values) => {
                let idx = key.as_int()?;
                if idx <= 0 {
                    return Err(anyhow!("sequence index must be >= 1, got {idx}"));
                }
                let zero = (idx - 1) as usize;
                let next = values
                    .get(zero)
                    .cloned()
                    .ok_or_else(|| anyhow!("sequence index out of range: {idx}"))?;
                get_path_value(&next, &path[1..])
            }
            _ => Err(anyhow!("index access on unsupported value {base:?}")),
        },
    }
}

fn set_path_value(base: &TlaValue, path: &[PathSegment], new_value: TlaValue) -> Result<TlaValue> {
    if path.is_empty() {
        return Ok(new_value);
    }

    match &path[0] {
        PathSegment::Field(name) => match base {
            TlaValue::Record(map) => {
                let current = map.get(name).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = (**map).clone();
                next.insert(name.clone(), updated);
                Ok(TlaValue::Record(HashedArc::new(next)))
            }
            _ => Err(anyhow!("field update on non-record value {base:?}")),
        },
        PathSegment::Index(key) => match base {
            TlaValue::Function(map) => {
                let current = map.get(key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = (**map).clone();
                next.insert(key.clone(), updated);
                Ok(TlaValue::Function(HashedArc::new(next)))
            }
            TlaValue::Record(map) => {
                let record_key = record_key_from_value(key)?;
                let current = map.get(&record_key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = (**map).clone();
                next.insert(record_key, updated);
                Ok(TlaValue::Record(HashedArc::new(next)))
            }
            TlaValue::Seq(values) => {
                let idx = key.as_int()?;
                if idx <= 0 {
                    return Err(anyhow!("sequence index must be >= 1, got {idx}"));
                }
                let zero = (idx - 1) as usize;
                if zero >= values.len() {
                    return Err(anyhow!("sequence index out of range: {idx}"));
                }

                let updated = set_path_value(&values[zero], &path[1..], new_value)?;
                let mut next = (**values).clone();
                next[zero] = updated;
                Ok(TlaValue::Seq(HashedArc::new(next)))
            }
            _ => Err(anyhow!("index update on unsupported value {base:?}")),
        },
    }
}

pub(super) fn parse_argument_list(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<TlaValue>> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let mut args = Vec::new();
    for part in split_top_level_symbol(trimmed, ",") {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        args.push(eval_expr_inner(part, ctx, depth + 1)?);
    }
    Ok(args)
}

pub(super) fn eval_bracket_index_key(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let args = parse_argument_list(expr, ctx, depth + 1)?;
    match args.len() {
        0 => Err(anyhow!("empty bracket index")),
        1 => Ok(args.into_iter().next().expect("single arg exists")),
        _ => Ok(TlaValue::Seq(HashedArc::new(args))),
    }
}


/// Parse a record key — accepts string literals (`"foo"`) and bare
/// identifiers (`foo`). Used by both record-literal evaluation and
/// EXCEPT-path destructuring; lives here because both call sites are in
/// `bracket.rs`.
fn parse_record_key(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('"') {
        let (s, rest) = parse_string_literal_prefix(trimmed)?
            .ok_or_else(|| anyhow!("invalid string record key: {trimmed}"))?;
        if !rest.trim().is_empty() {
            return Err(anyhow!("invalid trailing tokens in record key: {trimmed}"));
        }
        return Ok(s);
    }

    if trimmed
        .chars()
        .next()
        .map(|c| c.is_alphabetic() || c == '_')
        .unwrap_or(false)
        && trimmed.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        return Ok(trimmed.to_string());
    }

    Err(anyhow!("unsupported record key syntax: {trimmed}"))
}
