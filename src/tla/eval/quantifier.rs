//! Quantifier and binder evaluation: `\A`, `\E`, `CHOOSE`,
//! `LAMBDA`, plus the `BinderSpec` infrastructure used to evaluate
//! sets, function literals, and choose-bound variables.
//!
//! Most helpers stay private to this submodule. The ones widened to
//! `pub(super)` are the entry points that `expr::eval_expr_inner`
//! (still in eval.rs) and the bracket / set submodules call.

use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, eval_expr_inner, find_top_level_char, find_top_level_char_from,
    find_top_level_keyword_index, is_valid_identifier, normalize_param_name,
    split_top_level_comparison, split_top_level_symbol, take_keyword_prefix,
};

pub(super) fn eval_implies_parts(parts: &[String], ctx: &EvalContext<'_>, depth: usize) -> Result<bool> {
    if parts.is_empty() {
        return Err(anyhow!("empty implication"));
    }
    if parts.len() == 1 {
        return eval_expr_inner(&parts[0], ctx, depth)?.as_bool();
    }

    let lhs = eval_expr_inner(&parts[0], ctx, depth)?.as_bool()?;
    if !lhs {
        return Ok(true);
    }

    eval_implies_parts(&parts[1..], ctx, depth + 1)
}

pub(super) fn eval_quantifier_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let trimmed = expr.trim();
    let (is_forall, after_quant) = if let Some(rest) = trimmed.strip_prefix("\\A") {
        (true, rest.trim_start())
    } else if let Some(rest) = trimmed.strip_prefix("\\E") {
        (false, rest.trim_start())
    } else {
        return Err(anyhow!("not a quantifier expression: {expr}"));
    };

    let colon_idx = find_top_level_char(after_quant, ':')
        .ok_or_else(|| anyhow!("quantifier is missing ':' in expression: {expr}"))?;
    let binders = after_quant[..colon_idx].trim();
    let body = after_quant[colon_idx + 1..].trim();

    if body.is_empty() {
        eprintln!("\n=== QUANTIFIER WITH EMPTY BODY ===");
        eprintln!("is_forall: {}", is_forall);
        eprintln!("expr (first 500 chars): {:?}", &expr[..500.min(expr.len())]);
        eprintln!(
            "after_quant (first 500 chars): {:?}",
            &after_quant[..500.min(after_quant.len())]
        );
        eprintln!("colon_idx: {}", colon_idx);
        eprintln!("binders: {:?}", binders);
        eprintln!("body after colon: {:?}", &after_quant[colon_idx + 1..]);
        eprintln!("body trimmed: {:?}", body);
    }

    let domains = parse_binders(binders, ctx, depth + 1)?;
    let mut assignments = BTreeMap::new();

    let result = if is_forall {
        evaluate_forall(0, &domains, body, &mut assignments, ctx, depth + 1)?
    } else {
        evaluate_exists(0, &domains, body, &mut assignments, ctx, depth + 1)?
    };

    Ok(TlaValue::Bool(result))
}

pub(super) fn eval_choose_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (_, after_choose) = take_keyword_prefix(expr.trim(), "CHOOSE")
        .ok_or_else(|| anyhow!("expected CHOOSE expression"))?;

    let colon_idx = find_top_level_char(after_choose, ':')
        .ok_or_else(|| anyhow!("CHOOSE is missing ':' in expression: {expr}"))?;

    let binders = after_choose[..colon_idx].trim();
    let predicate = after_choose[colon_idx + 1..].trim();

    if find_top_level_keyword_index(binders, "\\in").is_none() {
        let var = binders.trim();
        if !is_valid_identifier(var) {
            return Err(anyhow!(
                "CHOOSE without a domain currently expects a single identifier: {expr}"
            ));
        }

        for value in choose_candidates_without_domain(var, predicate, ctx, depth + 1)? {
            let child = ctx.with_local_value(var, value.clone());
            if eval_expr_inner(predicate, &child, depth + 1)?.as_bool()? {
                return Ok(value);
            }
        }

        return Err(anyhow!("CHOOSE found no matching value"));
    }

    let parsed = parse_binders(binders, ctx, depth + 1)?;
    if parsed.len() != 1 {
        return Err(anyhow!("CHOOSE currently expects exactly one binder"));
    }

    let binder = &parsed[0];
    for value in &binder.domain {
        let mut child = ctx.clone();
        assign_binder_value(std::rc::Rc::make_mut(&mut child.locals), binder, value)?;
        if eval_expr_inner(predicate, &child, depth + 1)?.as_bool()? {
            return Ok(value.clone());
        }
    }

    Err(anyhow!("CHOOSE found no matching value"))
}

fn choose_candidates_without_domain(
    var: &str,
    predicate: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<Vec<TlaValue>> {
    let mut candidates = Vec::new();

    if let Some((lhs, op, rhs)) = split_top_level_comparison(predicate)
        && lhs.trim() == var
    {
        match op {
            "\\notin" => {
                if let Ok(set_val) = eval_expr_inner(rhs, ctx, depth + 1)
                    && let TlaValue::Set(values) = &set_val
                    && let Some(sample) = values.iter().next()
                {
                    candidates.push(sample.clone());
                }
            }
            "/=" | "#" => {
                if let Ok(value) = eval_expr_inner(rhs, ctx, depth + 1) {
                    candidates.push(value);
                }
            }
            _ => {}
        }
    }

    for idx in 0..4usize {
        candidates.push(stable_choose_model_value(var, predicate, idx));
    }
    candidates.push(TlaValue::Set(Arc::new(BTreeSet::new())));
    candidates.push(TlaValue::Seq(Arc::new(Vec::new())));
    candidates.push(TlaValue::Bool(false));
    candidates.push(TlaValue::Int(0));

    let mut deduped = Vec::with_capacity(candidates.len());
    for value in candidates {
        if deduped.iter().all(|existing| existing != &value) {
            deduped.push(value);
        }
    }

    Ok(deduped)
}

fn stable_choose_model_value(var: &str, predicate: &str, idx: usize) -> TlaValue {
    // Must be deterministic across processes: the minted ModelValue name
    // becomes part of the state, so a per-process random seed would make
    // the same CHOOSE produce divergent state content on different cluster
    // nodes. Use the fixed-seed fingerprint hasher.
    let mut hasher = crate::model::fingerprint_hasher();
    var.hash(&mut hasher);
    predicate.trim().hash(&mut hasher);
    let digest = hasher.finish();
    let base = format!("__choose_{var}_{digest:016x}");
    let name = if idx == 0 {
        base
    } else {
        format!("{base}_{idx}")
    };
    TlaValue::ModelValue(name)
}

pub(super) fn eval_lambda_expression(expr: &str, ctx: &EvalContext<'_>, _depth: usize) -> Result<TlaValue> {
    let (_, after_lambda) = take_keyword_prefix(expr.trim(), "LAMBDA")
        .ok_or_else(|| anyhow!("expected LAMBDA expression"))?;

    let colon_idx = find_top_level_char(after_lambda, ':')
        .ok_or_else(|| anyhow!("LAMBDA is missing ':' in expression: {expr}"))?;

    let params_text = after_lambda[..colon_idx].trim();
    let body = after_lambda[colon_idx + 1..].trim();

    // Parse parameter names (comma-separated)
    let params: Vec<String> = params_text
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if params.is_empty() {
        return Err(anyhow!("LAMBDA must have at least one parameter"));
    }

    // Capture current local context
    let captured_locals = (*ctx.locals).clone();

    Ok(TlaValue::Lambda {
        params: Arc::new(params),
        body: body.to_string(),
        captured_locals: Arc::new(captured_locals),
    })
}

#[derive(Debug, Clone)]
pub(super) struct BinderSpec {
    vars: Vec<String>,
    domain: Vec<TlaValue>,
}

fn parse_binder_pattern_vars(pattern: &str) -> Result<Vec<String>> {
    let trimmed = pattern.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("binder pattern is empty"));
    }

    if trimmed.starts_with("<<") && trimmed.ends_with(">>") {
        let inner = &trimmed[2..trimmed.len() - 2];
        let vars = split_top_level_symbol(inner, ",")
            .into_iter()
            .map(|var| var.trim().to_string())
            .filter(|var| !var.is_empty())
            .collect::<Vec<_>>();
        if vars.is_empty() {
            return Err(anyhow!("tuple binder pattern is empty: {pattern}"));
        }
        return Ok(vars);
    }

    Ok(vec![trimmed.to_string()])
}

pub(super) fn parse_binders(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<BinderSpec>> {
    let mut bindings = Vec::new();
    let mut rest = expr.trim();

    while !rest.is_empty() {
        let in_idx = find_top_level_keyword_index(rest, "\\in")
            .ok_or_else(|| anyhow!("binder segment missing \\in: {rest}"))?;
        let vars_text = rest[..in_idx].trim();
        let after_in = rest[in_idx + "\\in".len()..].trim_start();

        let mut split_idx: Option<usize> = None;
        let mut search_from = 0usize;
        while let Some(comma_rel) = find_top_level_char_from(after_in, ',', search_from) {
            let tail = after_in[comma_rel + 1..].trim_start();
            if find_top_level_keyword_index(tail, "\\in").is_some() {
                split_idx = Some(comma_rel);
                break;
            }
            search_from = comma_rel + 1;
        }

        let (domain_text, next_rest) = match split_idx {
            Some(idx) => (&after_in[..idx], &after_in[idx + 1..]),
            None => (after_in, ""),
        };

        let domain_value = eval_expr_inner(domain_text.trim(), ctx, depth + 1)?;
        let domain = domain_value.as_set()?.iter().cloned().collect::<Vec<_>>();

        for pattern in split_top_level_symbol(vars_text, ",") {
            let pattern = pattern.trim();
            if pattern.is_empty() {
                continue;
            }
            bindings.push(BinderSpec {
                vars: parse_binder_pattern_vars(pattern)?,
                domain: domain.clone(),
            });
        }

        rest = next_rest.trim_start();
    }

    Ok(bindings)
}

fn assign_binder_value(
    assignments: &mut BTreeMap<String, TlaValue>,
    binder: &BinderSpec,
    value: &TlaValue,
) -> Result<()> {
    if binder.vars.len() == 1 {
        assignments.insert(binder.vars[0].clone(), value.clone());
        return Ok(());
    }

    let parts = match value {
        TlaValue::Seq(parts) if parts.len() == binder.vars.len() => parts,
        TlaValue::Seq(parts) => {
            return Err(anyhow!(
                "tuple binder arity mismatch: expected {}, got {}",
                binder.vars.len(),
                parts.len()
            ));
        }
        other => {
            return Err(anyhow!(
                "tuple binder expects a sequence value, got {:?}",
                other
            ));
        }
    };

    for (name, part) in binder.vars.iter().zip(parts.iter()) {
        assignments.insert(name.clone(), part.clone());
    }

    Ok(())
}

fn remove_binder_assignments(assignments: &mut BTreeMap<String, TlaValue>, binder: &BinderSpec) {
    for name in &binder.vars {
        assignments.remove(name);
    }
}

pub(super) fn bind_param_value(
    assignments: &mut BTreeMap<String, TlaValue>,
    param: &str,
    value: TlaValue,
) -> Result<()> {
    let pattern = param.trim();
    if pattern.starts_with("<<") && pattern.ends_with(">>") {
        let binder = BinderSpec {
            vars: parse_binder_pattern_vars(pattern)?,
            domain: Vec::new(),
        };
        return assign_binder_value(assignments, &binder, &value);
    }

    assignments.insert(normalize_param_name(pattern).to_string(), value);
    Ok(())
}

fn evaluate_exists(
    idx: usize,
    domains: &[BinderSpec],
    body: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<bool> {
    if idx >= domains.len() {
        let mut child = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
            for (k, v) in assignments.iter() {
                locals_mut.insert(k.clone(), v.clone());
            }
        }
        return eval_expr_inner(body, &child, depth + 1)?.as_bool();
    }

    let binder = &domains[idx];
    let values = binder.domain.clone();
    for value in values {
        assign_binder_value(assignments, binder, &value)?;
        if evaluate_exists(idx + 1, domains, body, assignments, ctx, depth + 1)? {
            remove_binder_assignments(assignments, binder);
            return Ok(true);
        }
    }
    remove_binder_assignments(assignments, binder);
    Ok(false)
}

fn evaluate_forall(
    idx: usize,
    domains: &[BinderSpec],
    body: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<bool> {
    if idx >= domains.len() {
        let mut child = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
            for (k, v) in assignments.iter() {
                locals_mut.insert(k.clone(), v.clone());
            }
        }
        return eval_expr_inner(body, &child, depth + 1)?.as_bool();
    }

    let binder = &domains[idx];
    let values = binder.domain.clone();
    for value in values {
        assign_binder_value(assignments, binder, &value)?;
        if !evaluate_forall(idx + 1, domains, body, assignments, ctx, depth + 1)? {
            remove_binder_assignments(assignments, binder);
            return Ok(false);
        }
    }
    remove_binder_assignments(assignments, binder);
    Ok(true)
}

pub(super) fn collect_function_mapping(
    idx: usize,
    binders: &[BinderSpec],
    body: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeMap<TlaValue, TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
        // T201: charge one budget unit per inner iteration so
        // `[x \in <huge> |-> body]` cannot allocate without bound.
        ctx.check_budget(1)?;
        let mut child = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
            for (k, v) in assignments.iter() {
                locals_mut.insert(k.clone(), v.clone());
            }
        }
        let value = eval_expr_inner(body, &child, depth + 1)?;
        let key = binder_key(assignments, binders)?;
        out.insert(key, value);
        return Ok(());
    }

    let binder = &binders[idx];
    let values = binder.domain.clone();
    for value in values {
        assign_binder_value(assignments, binder, &value)?;
        collect_function_mapping(idx + 1, binders, body, assignments, out, ctx, depth + 1)?;
    }
    remove_binder_assignments(assignments, binder);

    Ok(())
}

pub(super) fn collect_binder_filter_set(
    idx: usize,
    binders: &[BinderSpec],
    predicate: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeSet<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
        // T201: charge one element per inner iteration to keep
        // `{ x \in <huge> : <pred> }` from allocating without bound
        // when the budget is set (e.g., from fuzz harnesses or probes).
        ctx.check_budget(1)?;
        let mut child = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
            for (k, v) in assignments.iter() {
                locals_mut.insert(k.clone(), v.clone());
            }
        }

        if eval_expr_inner(predicate, &child, depth + 1)?.as_bool()? {
            out.insert(binder_key(assignments, binders)?);
        }
        return Ok(());
    }

    let binder = &binders[idx];
    let values = binder.domain.clone();
    for value in values {
        assign_binder_value(assignments, binder, &value)?;
        collect_binder_filter_set(
            idx + 1,
            binders,
            predicate,
            assignments,
            out,
            ctx,
            depth + 1,
        )?;
    }
    remove_binder_assignments(assignments, binder);

    Ok(())
}

pub(super) fn collect_binder_map_set(
    idx: usize,
    binders: &[BinderSpec],
    element_expr: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeSet<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
        // T201: charge ONE element against the eval budget per inner
        // iteration. Without this, set comprehensions of shape
        // `{ <expr> : x \in <huge> }` allocated values in a loop with no
        // per-element accounting and were a known OOM hot path under fuzz.
        ctx.check_budget(1)?;
        let mut child = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
            for (k, v) in assignments.iter() {
                locals_mut.insert(k.clone(), v.clone());
            }
        }
        out.insert(eval_expr_inner(element_expr, &child, depth + 1)?);
        return Ok(());
    }

    let binder = &binders[idx];
    let values = binder.domain.clone();
    for value in values {
        assign_binder_value(assignments, binder, &value)?;
        collect_binder_map_set(
            idx + 1,
            binders,
            element_expr,
            assignments,
            out,
            ctx,
            depth + 1,
        )?;
    }
    remove_binder_assignments(assignments, binder);

    Ok(())
}

fn binder_key(
    assignments: &BTreeMap<String, TlaValue>,
    binders: &[BinderSpec],
) -> Result<TlaValue> {
    let total_vars = binders
        .iter()
        .map(|binder| binder.vars.len())
        .sum::<usize>();
    if total_vars == 1 {
        let name = binders
            .iter()
            .flat_map(|binder| binder.vars.iter())
            .next()
            .ok_or_else(|| anyhow!("missing binder assignment"))?;
        assignments
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("missing binder assignment for {name}"))
    } else {
        let mut items = Vec::with_capacity(total_vars);
        for binder in binders {
            for name in &binder.vars {
                items.push(
                    assignments
                        .get(name)
                        .cloned()
                        .ok_or_else(|| anyhow!("missing binder assignment for {name}"))?,
                );
            }
        }
        Ok(TlaValue::Seq(Arc::new(items)))
    }
}
