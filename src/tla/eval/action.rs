//! Action-IR evaluation: turning a parsed `ActionIr` (or raw action body
//! text) into the set of successor `TlaState`s.
//!
//! This module owns the multi-branch action machinery
//! (`apply_action_ir_with_context_multi`, `eval_action_body_multi`,
//! `eval_let_action_multi`, `parse_action_call_expr`,
//! `parse_action_binder_specs`, `matches_membership_expr`, ...). The
//! interpreter routes every successor-generation path through here:
//!
//! - `apply_action_ir{,_with_context,_with_context_multi}` — entry points
//!   used by the runtime (`action_exec.rs`) and the compiler fallback path.
//! - `eval_action_body_multi` / `eval_let_action_multi` — text-level
//!   entry points used by `compiled_eval.rs` and `cli/probe.rs` when the
//!   compiler can't fully encode an action and falls back to interpretation.
//! - `parse_action_call_expr` / `parse_action_binder_specs` — small
//!   parsers used by the compiler to detect action shape.
//! - `matches_membership_expr` — `\in`-style membership check; called
//!   by `eval_expr_inner` (in `expr.rs`) for `\in` and `\notin`
//!   comparisons. `pub(super)` so the sibling can reach it.
//!
//! `seed_implicit_instance_constant_bindings` and
//! `bind_instance_substitutions` live here too because they share the
//! definition-fallback machinery with `expand_action_call_multi`. The
//! `instance.rs` module reaches them via `super::NAME` (re-exported in
//! the parent `eval` module).

use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::tla::action_ir::{
    parse_action_exists, parse_action_if, split_action_body_clauses, split_action_body_disjuncts,
};
use crate::tla::{
    ActionClause, ActionIr, ClauseKind, TlaDefinition, TlaState, TlaValue, classify_clause,
    looks_like_action, parse_stuttering_action_expr,
};

use super::{
    EvalContext, bind_param_value, effective_instance_scope, eval_expr, eval_expr_inner,
    eval_guard, find_top_level_char, find_top_level_char_from, find_top_level_keyword_index,
    parse_let_definitions, split_outer_let, split_top_level_symbol, take_bracket_group,
};

#[derive(Debug, Clone, Default)]
pub(super) struct ActionEvalBranch {
    staged: BTreeMap<String, TlaValue>,
    unchanged_vars: Vec<String>,
}

/// Evaluate a LET expression that contains primed assignments.
///
/// This wrapper preserves the previous single-successor shape for legacy callers.
#[allow(dead_code)]
pub(crate) fn eval_let_action(
    expr: &str,
    ctx: &EvalContext<'_>,
    staged: &mut BTreeMap<String, TlaValue>,
    unchanged_vars: &mut Vec<String>,
) -> Result<()> {
    let mut branches = eval_let_action_multi(expr, ctx, staged, unchanged_vars)?.into_iter();
    let Some((next_staged, next_unchanged)) = branches.next() else {
        return Err(anyhow!("LET action produced no successor branches"));
    };
    if branches.next().is_some() {
        return Err(anyhow!("LET action produced multiple successor branches"));
    }
    *staged = next_staged;
    *unchanged_vars = next_unchanged;
    Ok(())
}

pub fn eval_action_body_multi(
    body: &str,
    ctx: &EvalContext<'_>,
    staged: &BTreeMap<String, TlaValue>,
) -> Result<Vec<(BTreeMap<String, TlaValue>, Vec<String>)>> {
    Ok(eval_action_body_text_multi(
        body,
        ctx,
        ActionEvalBranch {
            staged: staged.clone(),
            unchanged_vars: Vec::new(),
        },
    )?
    .into_iter()
    .map(|branch| (branch.staged, branch.unchanged_vars))
    .collect())
}

pub fn eval_let_action_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    staged: &BTreeMap<String, TlaValue>,
    unchanged_vars: &[String],
) -> Result<Vec<(BTreeMap<String, TlaValue>, Vec<String>)>> {
    let initial = ActionEvalBranch {
        staged: staged.clone(),
        unchanged_vars: unchanged_vars.to_vec(),
    };
    Ok(eval_let_action_multi_branch(expr, ctx, initial)?
        .into_iter()
        .map(|branch| (branch.staged, branch.unchanged_vars))
        .collect())
}

fn eval_let_action_multi_branch(
    expr: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    let (defs_text, body_text) =
        split_outer_let(expr).ok_or_else(|| anyhow!("invalid LET expression in action: {expr}"))?;
    let defs = parse_let_definitions(defs_text)?;
    // T203: charge budget for clone-on-write.
    ctx.check_budget(ctx.local_definitions.len().saturating_add(defs.len()))?;
    let child_ctx = ctx.with_local_definitions(defs);
    eval_action_body_text_multi(body_text, &child_ctx, branch)
}

pub fn apply_action_ir(action: &ActionIr, current: &TlaState) -> Result<Option<TlaState>> {
    let ctx = EvalContext::new(current);
    Ok(apply_action_ir_with_context_multi(action, current, &ctx)?
        .into_iter()
        .next())
}

pub(crate) fn apply_action_ir_with_context_multi(
    action: &ActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Vec<TlaState>> {
    let branches =
        eval_action_clauses_multi(&action.clauses, ctx, vec![ActionEvalBranch::default()])?;
    let mut out = Vec::with_capacity(branches.len());
    for branch in branches {
        let mut next = current.clone();
        // Mirror of `apply_compiled_action_ir_multi` — the loop B insert
        // below already covers unchanged vars because every Unchanged
        // handler `or_insert`s into `branch.staged`. See the longer note
        // on the compiled-side function for the full rationale.
        for (var, value) in branch.staged {
            // Update in place to reuse the existing key Arc and skip the
            // BTreeMap insert-rebalance — see the longer note on
            // `apply_compiled_action_ir_multi` (compiled_eval.rs).
            if let Some(slot) = next.get_mut(var.as_str()) {
                *slot = value;
            } else {
                next.insert(Arc::from(var), value);
            }
        }
        out.push(next);
    }
    Ok(out)
}

pub fn apply_action_ir_with_context(
    action: &ActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Option<TlaState>> {
    Ok(apply_action_ir_with_context_multi(action, current, ctx)?
        .into_iter()
        .next())
}

fn eval_action_clauses_multi(
    clauses: &[ActionClause],
    ctx: &EvalContext<'_>,
    branches: Vec<ActionEvalBranch>,
) -> Result<Vec<ActionEvalBranch>> {
    let mut branches = branches;
    for clause in clauses {
        let mut next_branches = Vec::new();
        for branch in branches {
            next_branches.extend(eval_action_clause_to_branch(clause, ctx, branch)?);
        }
        branches = next_branches;
        if branches.is_empty() {
            break;
        }
    }
    Ok(branches)
}

fn eval_action_clause_to_branch(
    clause: &ActionClause,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    let eval_ctx = ctx_with_staged_primes(ctx, &branch.staged);
    match clause {
        ActionClause::Guard { expr } => eval_action_clause_text_multi(expr, ctx, branch),
        ActionClause::PrimedAssignment { var, expr } => {
            let mut branch = branch;
            branch
                .staged
                .insert(var.clone(), eval_expr(expr, &eval_ctx)?);
            Ok(vec![branch])
        }
        ActionClause::PrimedMembership { var, set_expr } => {
            // If this variable was already assigned by a prior clause,
            // treat as a guard (membership check) not nondeterministic assignment
            if branch.staged.contains_key(var) {
                let existing = branch.staged.get(var).unwrap();
                let domain = eval_expr(set_expr, &eval_ctx)?;
                let set = domain.as_set()?;
                if set.contains(existing) {
                    Ok(vec![branch])
                } else {
                    Ok(Vec::new()) // Guard failed
                }
            } else {
                let domain = eval_expr(set_expr, &eval_ctx)?;
                let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    let mut branch = branch.clone();
                    branch.staged.insert(var.clone(), value);
                    out.push(branch);
                }
                Ok(out)
            }
        }
        ActionClause::Unchanged { vars } => {
            let mut branch = branch;
            for var in vars {
                branch.unchanged_vars.push(var.clone());
                if let Some(value) = ctx.state.get(var.as_str()) {
                    branch
                        .staged
                        .entry(var.clone())
                        .or_insert_with(|| value.clone());
                }
            }
            Ok(vec![branch])
        }
        ActionClause::Exists { binders, body } => {
            eval_exists_action_multi(binders, body, ctx, branch)
        }
        ActionClause::LetWithPrimes { expr } => eval_let_action_multi_branch(expr, ctx, branch),
    }
}

fn eval_action_body_text_multi(
    body: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    if let Some((action, vars)) = parse_stuttering_action_expr(body) {
        return eval_stuttering_action_multi(&action, &vars, ctx, branch);
    }
    if let Some(result) = eval_disjunctive_action_body_multi(body, ctx, branch.clone()) {
        return result;
    }

    let mut branches = vec![branch];
    for clause in split_action_body_clauses(body) {
        let mut next_branches = Vec::new();
        for branch in branches {
            next_branches.extend(eval_action_clause_text_multi(&clause, ctx, branch)?);
        }
        branches = next_branches;
        if branches.is_empty() {
            break;
        }
    }
    Ok(branches)
}

fn eval_disjunctive_action_body_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Option<Result<Vec<ActionEvalBranch>>> {
    let trimmed = expr.trim();
    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() <= 1 {
        return None;
    }

    let mut out = Vec::new();
    let mut first_err = None;
    let mut saw_clean_empty = false;
    for disjunct in disjuncts {
        match eval_action_body_text_multi(&disjunct, ctx, branch.clone()) {
            Ok(mut branches) => {
                if branches.is_empty() {
                    saw_clean_empty = true;
                } else {
                    out.append(&mut branches);
                }
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
    }

    if !out.is_empty() {
        Some(Ok(out))
    } else if first_err.is_none() && saw_clean_empty {
        Some(Ok(Vec::new()))
    } else {
        Some(Err(first_err.unwrap_or_else(|| {
            anyhow!("no disjunctive action branch produced a successor")
        })))
    }
}

fn eval_stuttering_action_multi(
    action: &str,
    vars: &[String],
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    let mut out = match eval_action_body_text_multi(action, ctx, branch.clone()) {
        Ok(branches) => branches,
        Err(_) => Vec::new(),
    };

    let mut stutter = branch;
    for var in vars {
        stutter.unchanged_vars.push(var.clone());
        if let Some(value) = ctx.state.get(var.as_str()) {
            stutter
                .staged
                .entry(var.clone())
                .or_insert_with(|| value.clone());
        }
    }
    out.push(stutter);
    Ok(out)
}

pub(crate) fn parse_action_call_expr(expr: &str) -> Option<(String, Vec<String>)> {
    let trimmed = expr.trim_start();
    let mut chars = trimmed.char_indices();
    let (_, first) = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '!' {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }

    let name = trimmed[..end].to_string();
    let rest = trimmed[end..].trim_start();
    if rest.is_empty() {
        return Some((name, Vec::new()));
    }
    if !rest.starts_with('(') {
        return None;
    }

    let (args_text, tail) = take_bracket_group(rest, '(', ')').ok()?;
    if !tail.trim().is_empty() {
        return None;
    }

    let args = if args_text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level_symbol(args_text, ",")
            .into_iter()
            .map(|arg| arg.trim().to_string())
            .filter(|arg| !arg.is_empty())
            .collect()
    };

    Some((name, args))
}

pub(super) fn seed_implicit_instance_constant_bindings(
    instance: &crate::tla::module::TlaModuleInstance,
    module: &crate::tla::module::TlaModule,
    parent_ctx: &EvalContext<'_>,
    child_ctx: &mut EvalContext<'_>,
) {
    let defs_mut = std::rc::Rc::make_mut(&mut child_ctx.local_definitions);
    seed_parent_definition_fallbacks(module, parent_ctx, defs_mut);
    let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);

    for constant in &module.constants {
        if instance.substitutions.contains_key(constant) {
            continue;
        }
        if let Some(def) = parent_ctx.definition(constant) {
            defs_mut.insert(constant.clone(), def.clone());
            continue;
        }
        if let Ok(value) = eval_expr(constant, parent_ctx) {
            locals_mut.insert(constant.clone(), value);
        }
    }
}

pub(super) fn bind_instance_substitutions(
    instance: &crate::tla::module::TlaModuleInstance,
    parent_ctx: &EvalContext<'_>,
    locals_mut: &mut BTreeMap<String, TlaValue>,
) -> Result<()> {
    for (param, value_expr) in &instance.substitutions {
        let trimmed = value_expr.trim();
        let value = eval_expr(trimmed, parent_ctx)?;
        locals_mut.insert(param.clone(), value);

        if let Some(primed_value) = resolved_instance_primed_substitution_value(trimmed, parent_ctx)
        {
            locals_mut.insert(format!("{param}'"), primed_value);
        }
    }

    Ok(())
}

fn resolved_instance_primed_substitution_value(
    trimmed: &str,
    parent_ctx: &EvalContext<'_>,
) -> Option<TlaValue> {
    let primed_expr = format!("{trimmed}'");
    let primed_value = eval_expr(&primed_expr, parent_ctx).ok()?;
    match &primed_value {
        TlaValue::ModelValue(name) if name == &primed_expr => None,
        _ => Some(primed_value),
    }
}

fn seed_parent_definition_fallbacks(
    module: &crate::tla::module::TlaModule,
    parent_ctx: &EvalContext<'_>,
    defs_mut: &mut BTreeMap<String, TlaDefinition>,
) {
    for (name, def) in parent_ctx.local_definitions.iter() {
        if !module.definitions.contains_key(name) {
            defs_mut.entry(name.clone()).or_insert_with(|| def.clone());
        }
    }
    if let Some(parent_defs) = parent_ctx.definitions {
        for (name, def) in parent_defs {
            if !module.definitions.contains_key(name) {
                defs_mut.entry(name.clone()).or_insert_with(|| def.clone());
            }
        }
    }
}

/// Decide whether calling operator `def` is (possibly transitively) an action.
///
/// `looks_like_action` only detects *direct* primes / `UNCHANGED`. Wrapper
/// operators like `coordProgB == makeDecision \/ \E i : coordProgA(i)` hide
/// their primes one call deeper, so a disjunct `... \/ coordProgB` was being
/// treated as a boolean guard and silently dropping every successor (ACP
/// specs). We follow operator-call identifiers in the body up to a small depth
/// bound so such wrappers are correctly expanded as actions.
fn def_is_transitively_action(def: &TlaDefinition, ctx: &EvalContext<'_>, depth: usize) -> bool {
    const MAX_DEPTH: usize = 8;
    if looks_like_action(def) {
        return true;
    }
    if depth >= MAX_DEPTH {
        return false;
    }
    for ident in extract_body_call_idents(&def.body) {
        if ident == def.name {
            continue;
        }
        if let Some(inner) = ctx.definition(&ident) {
            if def_is_transitively_action(&inner, ctx, depth + 1) {
                return true;
            }
        }
    }
    false
}

/// Over-approximate list of identifier tokens in a body that might be operator
/// calls (outside string literals). Non-operators simply fail to resolve.
fn extract_body_call_idents(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == b'\\' {
                escaped = true;
            } else if ch == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(body[start..i].to_string());
            continue;
        }
        i += 1;
    }
    out
}

fn expand_action_call_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Option<Result<Vec<ActionEvalBranch>>> {
    let (name, arg_exprs) = parse_action_call_expr(expr)?;

    if let Some((alias, operator_name)) = name.split_once('!') {
        let instances = ctx.instances?;
        let instance = instances.get(alias)?;
        let module = instance.module.as_ref()?;
        let def = module.definitions.get(operator_name)?.clone();
        let mut probe_ctx = ctx.clone();
        probe_ctx.definitions = Some(&module.definitions);
        if !def_is_transitively_action(&def, &probe_ctx, 0) || def.body.trim() == expr.trim() {
            return None;
        }
        if def.params.len() != arg_exprs.len() {
            return Some(Err(anyhow!(
                "operator '{alias}!{operator_name}' arity mismatch: expected {}, got {}",
                def.params.len(),
                arg_exprs.len()
            )));
        }

        let mut instance_ctx = ctx.clone();
        instance_ctx.definitions = Some(&module.definitions);
        instance_ctx.instances = effective_instance_scope(&module.instances, ctx.instances);
        seed_implicit_instance_constant_bindings(instance, module, ctx, &mut instance_ctx);
        {
            let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
            if let Err(err) = bind_instance_substitutions(instance, ctx, locals_mut) {
                return Some(Err(err));
            }
            for (param, arg_expr) in def.params.iter().zip(arg_exprs.iter()) {
                let value = match eval_expr(arg_expr, ctx) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if let Err(err) = bind_param_value(locals_mut, param, value) {
                    return Some(Err(err));
                }
            }
        }

        return Some(eval_action_body_text_multi(
            &def.body,
            &instance_ctx,
            branch,
        ));
    }

    let def = ctx.definition(&name)?;
    if !def_is_transitively_action(&def, ctx, 0) || def.body.trim() == expr.trim() {
        return None;
    }
    if def.params.len() != arg_exprs.len() {
        return Some(Err(anyhow!(
            "operator '{name}' arity mismatch: expected {}, got {}",
            def.params.len(),
            arg_exprs.len()
        )));
    }

    let mut child_ctx = ctx.clone();
    {
        let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
        for (param, arg_expr) in def.params.iter().zip(arg_exprs.iter()) {
            let value = match eval_expr(arg_expr, ctx) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            if let Err(err) = bind_param_value(locals_mut, param, value) {
                return Some(Err(err));
            }
        }
    }

    Some(eval_action_body_text_multi(&def.body, &child_ctx, branch))
}

fn eval_action_clause_text_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty action clause"));
    }

    let eval_ctx = ctx_with_staged_primes(ctx, &branch.staged);
    if let Some((condition, then_branch, else_branch)) = parse_action_if(trimmed) {
        let branch_body = if eval_guard(condition, &eval_ctx)? {
            then_branch
        } else {
            else_branch
        };
        return eval_action_body_text_multi(branch_body, ctx, branch);
    }
    if let Some((binders, body)) = parse_action_exists(trimmed) {
        return eval_exists_action_multi(binders, body, ctx, branch);
    }
    if let Some((action, vars)) = parse_stuttering_action_expr(trimmed) {
        return eval_stuttering_action_multi(&action, &vars, ctx, branch);
    }
    if let Some(result) = eval_disjunctive_action_body_multi(trimmed, ctx, branch.clone()) {
        return result;
    }
    if let Some(result) = expand_action_call_multi(trimmed, ctx, branch.clone()) {
        return result;
    }

    match classify_clause(trimmed) {
        ClauseKind::PrimedAssignment { var, expr: rhs } => {
            let mut branch = branch;
            branch.staged.insert(var, eval_expr(&rhs, &eval_ctx)?);
            Ok(vec![branch])
        }
        ClauseKind::PrimedMembership { var, set_expr } => {
            // If this variable was already assigned by a prior clause,
            // treat as a guard (membership check) not nondeterministic assignment
            if branch.staged.contains_key(&var) {
                let existing = branch.staged.get(&var).unwrap();
                let domain = eval_expr(&set_expr, &eval_ctx)?;
                let set = domain.as_set()?;
                if set.contains(existing) {
                    Ok(vec![branch])
                } else {
                    Ok(Vec::new()) // Guard failed
                }
            } else {
                let domain = eval_expr(&set_expr, &eval_ctx)?;
                let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    let mut branch = branch.clone();
                    branch.staged.insert(var.clone(), value);
                    out.push(branch);
                }
                Ok(out)
            }
        }
        ClauseKind::Unchanged { vars } => {
            let mut branch = branch;
            for var in vars {
                branch.unchanged_vars.push(var.clone());
                if let Some(value) = ctx.state.get(var.as_str()) {
                    branch.staged.entry(var).or_insert_with(|| value.clone());
                }
            }
            Ok(vec![branch])
        }
        ClauseKind::UnprimedEquality { .. }
        | ClauseKind::UnprimedMembership { .. }
        | ClauseKind::Other => {
            if trimmed.starts_with("LET") {
                eval_let_action_multi_branch(trimmed, ctx, branch)
            } else if eval_guard(trimmed, &eval_ctx)? {
                Ok(vec![branch])
            } else {
                Ok(Vec::new())
            }
        }
    }
}

fn eval_exists_action_multi(
    binders: &str,
    body: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    let binder_specs = parse_action_binder_specs(binders)?;
    expand_action_exists_branches(&binder_specs, 0, ctx, body, branch)
}

fn expand_action_exists_branches(
    binder_specs: &[(String, String)],
    idx: usize,
    ctx: &EvalContext<'_>,
    body: &str,
    branch: ActionEvalBranch,
) -> Result<Vec<ActionEvalBranch>> {
    if idx >= binder_specs.len() {
        return eval_action_body_text_multi(body, ctx, branch);
    }

    let (name, domain_text) = &binder_specs[idx];
    let eval_ctx = ctx_with_staged_primes(ctx, &branch.staged);
    let domain = eval_expr(domain_text, &eval_ctx)?;
    let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
    let mut out = Vec::new();
    for value in values {
        let child_ctx = ctx.with_local_value(name.clone(), value);
        out.extend(expand_action_exists_branches(
            binder_specs,
            idx + 1,
            &child_ctx,
            body,
            branch.clone(),
        )?);
    }
    Ok(out)
}

/// Return a context with each `var` in `staged` bound as the primed local
/// `var'`. Clones the `locals` BTreeMap **once** and inserts every primed
/// entry into that single clone, rather than folding `with_local_value` over
/// the map — which re-cloned the whole (growing) locals map once per staged
/// var, i.e. O(staged * locals) allocations per action clause on the hot eval
/// path. Inserting `var'` entries is order-independent (distinct `var` names
/// yield distinct keys; values come from `staged`, not from each other), so
/// the result is identical to the fold.
fn ctx_with_staged_primes<'a>(
    ctx: &EvalContext<'a>,
    staged: &BTreeMap<String, TlaValue>,
) -> EvalContext<'a> {
    if staged.is_empty() {
        return ctx.clone();
    }
    let mut new_locals = (*ctx.locals).clone();
    for (var, value) in staged {
        // `var'` without a format! allocation.
        let mut key = String::with_capacity(var.len() + 1);
        key.push_str(var);
        key.push('\'');
        new_locals.insert(key, value.clone());
    }
    EvalContext {
        state: ctx.state,
        locals: std::rc::Rc::new(new_locals),
        lexical_locals: std::rc::Rc::clone(&ctx.lexical_locals),
        local_definitions: std::rc::Rc::clone(&ctx.local_definitions),
        definitions: ctx.definitions,
        instances: ctx.instances,
        eval_budget: ctx.eval_budget.clone(),
    }
}

pub(crate) fn parse_action_binder_specs(expr: &str) -> Result<Vec<(String, String)>> {
    let mut binders = Vec::new();
    let mut rest = expr.trim();

    while !rest.is_empty() {
        let in_idx = find_top_level_keyword_index(rest, "\\in")
            .ok_or_else(|| anyhow!("binder segment missing \\in: {rest}"))?;

        let vars_text = rest[..in_idx].trim();
        let after_in = rest[in_idx + "\\in".len()..].trim_start();

        let mut split_idx = None;
        let mut search_from = 0usize;
        while let Some(comma_idx) = find_top_level_char_from(after_in, ',', search_from) {
            let tail = after_in[comma_idx + 1..].trim_start();
            if find_top_level_keyword_index(tail, "\\in").is_some() {
                split_idx = Some(comma_idx);
                break;
            }
            search_from = comma_idx + 1;
        }

        let (domain_text, next_rest) = match split_idx {
            Some(idx) => (&after_in[..idx], &after_in[idx + 1..]),
            None => (after_in, ""),
        };

        for var in crate::tla::split_top_level(vars_text, ",") {
            let name = var.trim();
            if !name.is_empty() {
                binders.push((name.to_string(), domain_text.trim().to_string()));
            }
        }

        rest = next_rest.trim_start();
    }

    if binders.is_empty() {
        return Err(anyhow!("exists binder list is empty: {expr}"));
    }

    Ok(binders)
}

pub(super) fn matches_membership_expr(
    value: &TlaValue,
    expr: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<bool> {
    let rhs_trimmed = expr.trim();
    match rhs_trimmed {
        "Nat" => Ok(matches!(value.as_int(), Ok(n) if n >= 0)),
        "Int" => Ok(value.as_int().is_ok()),
        "BOOLEAN" => Ok(matches!(value, TlaValue::Bool(_))),
        _ => {
            if let Some(runtime_value) = ctx.runtime_value(rhs_trimmed) {
                return runtime_value.contains(value);
            }

            if let Some(def) = ctx.definition(rhs_trimmed)
                && def.params.is_empty()
            {
                return matches_membership_expr(value, &def.body, ctx, depth + 1);
            }

            if let Some(inner) = rhs_trimmed.strip_prefix("Seq(") {
                if let Some(set_expr) = inner.strip_suffix(")") {
                    return match value {
                        TlaValue::Seq(seq) => {
                            for elem in seq.iter() {
                                if !matches_membership_expr(elem, set_expr, ctx, depth + 1)? {
                                    return Ok(false);
                                }
                            }
                            Ok(true)
                        }
                        TlaValue::Function(func)
                            if crate::tla::compiled_eval::func_is_sequence_shaped(func) =>
                        {
                            for val in func.values() {
                                if !matches_membership_expr(val, set_expr, ctx, depth + 1)? {
                                    return Ok(false);
                                }
                            }
                            Ok(true)
                        }
                        _ => Ok(false),
                    };
                }
            }

            // Filter-set comprehension fast path: `x \in {y \in S : P(y)}`
            // semantically equals `x \in S /\ P[y := x]`. Without this the
            // fallthrough at the bottom materializes the whole filter set
            // (potentially millions of elements) before testing membership.
            // See `project_mcbinarysearch_stall_root_cause_2026-06-11.md`
            // for the motivating workload — SortedSeqs on a 488K-candidate
            // domain.
            if rhs_trimmed.starts_with('{')
                && rhs_trimmed.ends_with('}')
                && let inner = &rhs_trimmed[1..rhs_trimmed.len() - 1]
                && let Some(colon_idx) = crate::tla::compiled_expr::find_top_level_colon(inner)
            {
                let lhs = inner[..colon_idx].trim();
                let filter_expr = inner[colon_idx + 1..].trim();
                if let Some(in_idx) =
                    super::splitter::find_top_level_keyword_index(lhs, "\\in")
                {
                    let var = lhs[..in_idx].trim();
                    let domain_expr = lhs[in_idx + "\\in".len()..].trim();
                    // Only treat as filter form when the LHS binder is a
                    // simple identifier (not a tuple destructure like
                    // `<<a, b>> \in S` and not a comma-list binder). Those
                    // would need different rebinding semantics.
                    if !var.is_empty()
                        && !var.contains(',')
                        && !var.contains('<')
                        && !var.contains(' ')
                    {
                        if !matches_membership_expr(value, domain_expr, ctx, depth + 1)? {
                            return Ok(false);
                        }
                        let scoped = ctx.with_local_value(var, value.clone());
                        return Ok(eval_expr_inner(filter_expr, &scoped, depth + 1)?.as_bool()?);
                    }
                }
            }

            if rhs_trimmed.starts_with('[') && rhs_trimmed.ends_with(']') {
                let inner = &rhs_trimmed[1..rhs_trimmed.len() - 1];

                if let Some(arrow_idx) = inner.find("->") {
                    let domain_expr = inner[..arrow_idx].trim();
                    let codomain_expr = inner[arrow_idx + 2..].trim();
                    return match value {
                        TlaValue::Function(func) => {
                            let domain_val = eval_expr_inner(domain_expr, ctx, depth + 1)?;
                            let domain_set = match domain_val.as_set() {
                                Ok(set) => set,
                                Err(_) => return Ok(false),
                            };
                            let func_domain: BTreeSet<TlaValue> = func.keys().cloned().collect();
                            if func_domain != *domain_set {
                                return Ok(false);
                            }
                            for item in func.values() {
                                if !matches_membership_expr(item, codomain_expr, ctx, depth + 1)? {
                                    return Ok(false);
                                }
                            }
                            Ok(true)
                        }
                        _ => Ok(false),
                    };
                }

                if inner.contains(':') {
                    return match value {
                        TlaValue::Record(rec) => {
                            // Use top-level split to handle sets like {1, 2} correctly
                            let field_specs = split_top_level_symbol(inner, ",");
                            let mut expected_fields = std::collections::HashSet::<String>::new();
                            for spec in field_specs {
                                let spec = spec.trim();
                                // Find the first colon at top level for field:type separation
                                if let Some(colon_idx) = find_top_level_char(spec, ':') {
                                    let field_name = spec[..colon_idx].trim();
                                    let field_type = spec[colon_idx + 1..].trim();
                                    expected_fields.insert(field_name.to_string());
                                    let Some(field_value) = rec.get(field_name) else {
                                        return Ok(false);
                                    };
                                    if !matches_membership_expr(
                                        field_value,
                                        field_type,
                                        ctx,
                                        depth + 1,
                                    )? {
                                        return Ok(false);
                                    }
                                }
                            }
                            Ok(rec.len() == expected_fields.len())
                        }
                        _ => Ok(false),
                    };
                }
            }

            let set_val = eval_expr_inner(rhs_trimmed, ctx, depth + 1)?;
            set_val.contains(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::tla_state;
    use std::collections::BTreeSet;

    #[test]
    fn applies_action_ir() {
        let state = tla_state([("x", TlaValue::Int(1)), ("y", TlaValue::Int(2))]);
        let action = ActionIr {
            name: "Tick".to_string(),
            params: vec![],
            clauses: vec![
                ActionClause::Guard {
                    expr: "x < 5".to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "x".to_string(),
                    expr: "x + 1".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["y".to_string()],
                },
            ],
        };

        let next = apply_action_ir(&action, &state)
            .expect("action should evaluate")
            .expect("guard is true");
        assert_eq!(next.get("x"), Some(&TlaValue::Int(2)));
        assert_eq!(next.get("y"), Some(&TlaValue::Int(2)));
    }

    #[test]
    fn applies_action_ir_with_nested_action_references() {
        let state = tla_state([
            ("cat_box", TlaValue::Int(2)),
            ("observed_box", TlaValue::Int(2)),
            ("direction", TlaValue::String("right".to_string())),
        ]);

        let mut definitions = BTreeMap::new();
        definitions.insert(
            "Move_Cat".to_string(),
            TlaDefinition {
                name: "Move_Cat".to_string(),
                params: vec![],
                body: "/\\ cat_box' = cat_box + 1\n/\\ cat_box' \\in 1..6".to_string(),
                is_recursive: false,
            },
        );
        definitions.insert(
            "Observe_Box".to_string(),
            TlaDefinition {
                name: "Observe_Box".to_string(),
                params: vec![],
                body: "LET next_box == IF direction = \"right\"\n                  THEN observed_box + 1\n                  ELSE observed_box - 1\nIN \\/ /\\ next_box \\in 2..5\n      /\\ observed_box' = next_box\n      /\\ UNCHANGED direction\n   \\/ /\\ next_box \\notin 2..5\n      /\\ direction' = CHOOSE d \\in {\"left\", \"right\"}: d /= direction\n      /\\ UNCHANGED observed_box".to_string(),
                is_recursive: false,
            },
        );

        let action = ActionIr {
            name: "Next".to_string(),
            params: vec![],
            clauses: vec![
                ActionClause::Guard {
                    expr: "Move_Cat".to_string(),
                },
                ActionClause::Guard {
                    expr: "Observe_Box".to_string(),
                },
            ],
        };

        let ctx = EvalContext::with_definitions(&state, &definitions);
        let next = apply_action_ir_with_context(&action, &state, &ctx)
            .expect("nested actions should evaluate")
            .expect("nested actions should produce a successor");
        assert_eq!(next.get("cat_box"), Some(&TlaValue::Int(3)));
        assert_eq!(next.get("observed_box"), Some(&TlaValue::Int(3)));
        assert_eq!(
            next.get("direction"),
            Some(&TlaValue::String("right".to_string()))
        );
    }

    #[test]
    fn applies_action_ir_with_primed_membership_generates_all_choices() {
        let state = tla_state([("flip", TlaValue::String("H".to_string()))]);
        let action = ActionIr {
            name: "TossCoin".to_string(),
            params: vec![],
            clauses: vec![ActionClause::PrimedMembership {
                var: "flip".to_string(),
                set_expr: "{\"H\", \"T\"}".to_string(),
            }],
        };

        let next = apply_action_ir_with_context_multi(&action, &state, &EvalContext::new(&state))
            .expect("primed membership should enumerate successors");
        assert_eq!(next.len(), 2, "{next:?}");
        assert!(
            next.iter()
                .any(|st| st.get("flip") == Some(&TlaValue::String("H".to_string())))
        );
        assert!(
            next.iter()
                .any(|st| st.get("flip") == Some(&TlaValue::String("T".to_string())))
        );
    }

    #[test]
    fn action_guard_can_block_transition() {
        let state = tla_state([("x", TlaValue::Int(10))]);
        let action = ActionIr {
            name: "Blocked".to_string(),
            params: vec![],
            clauses: vec![ActionClause::Guard {
                expr: "x < 5".to_string(),
            }],
        };
        let out = apply_action_ir(&action, &state).expect("evaluation should succeed");
        assert!(out.is_none());
    }

    #[test]
    fn applies_action_ir_with_conditional_branches() {
        let state = tla_state([
            ("flag", TlaValue::Bool(false)),
            ("x", TlaValue::Int(7)),
            ("y", TlaValue::Int(9)),
        ]);
        let action = ActionIr {
            name: "Conditional".to_string(),
            params: vec![],
            clauses: vec![ActionClause::Guard {
                expr: "IF flag THEN /\\ x' = x + 1 /\\ UNCHANGED y ELSE /\\ UNCHANGED x /\\ y' = y + 1"
                    .to_string(),
            }],
        };

        let next = apply_action_ir(&action, &state)
            .expect("conditional action should evaluate")
            .expect("branch should succeed");
        assert_eq!(next.get("x"), Some(&TlaValue::Int(7)));
        assert_eq!(next.get("y"), Some(&TlaValue::Int(10)));
    }

    #[test]
    fn applies_action_ir_with_unchanged_primes_referenced_later() {
        let state = tla_state([
            ("flag", TlaValue::Bool(false)),
            ("count", TlaValue::Int(7)),
            ("announced", TlaValue::Bool(false)),
        ]);
        let action = ActionIr {
            name: "Counter".to_string(),
            params: vec![],
            clauses: vec![
                ActionClause::Guard {
                    expr: "IF flag THEN /\\ count' = count + 1 ELSE /\\ UNCHANGED count"
                        .to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "announced".to_string(),
                    expr: "count' >= 7".to_string(),
                },
            ],
        };

        let next = apply_action_ir(&action, &state)
            .expect("conditional action should evaluate")
            .expect("branch should succeed");
        assert_eq!(next.get("count"), Some(&TlaValue::Int(7)));
        assert_eq!(next.get("announced"), Some(&TlaValue::Bool(true)));
    }

    #[test]
    fn applies_let_action_with_body_starting_with_disjunction() {
        let state = tla_state([
            ("observed_box", TlaValue::Int(2)),
            ("direction", TlaValue::String("right".to_string())),
        ]);
        let action = ActionIr {
            name: "ObserveBox".to_string(),
            params: vec![],
            clauses: vec![ActionClause::LetWithPrimes {
                expr: "LET next_box == IF direction = \"right\"\n                  THEN observed_box + 1\n                  ELSE observed_box - 1\n  IN \\/ /\\ next_box \\in {2, 3, 4}\n        /\\ observed_box' = next_box\n        /\\ UNCHANGED direction\n     \\/ /\\ next_box \\notin {2, 3, 4}\n        /\\ direction' = CHOOSE d \\in {\"left\", \"right\"}: d /= direction\n        /\\ UNCHANGED observed_box".to_string(),
            }],
        };

        let next = apply_action_ir(&action, &state)
            .expect("LET action should evaluate")
            .expect("one branch should succeed");
        assert_eq!(next.get("observed_box"), Some(&TlaValue::Int(3)));
        assert_eq!(
            next.get("direction"),
            Some(&TlaValue::String("right".to_string()))
        );
    }

    #[test]
    fn quantified_let_action_generates_multiple_successors() {
        let state = tla_state([("x", TlaValue::Int(0)), ("y", TlaValue::Int(9))]);
        let ctx = EvalContext::new(&state);
        let action = ActionIr {
            name: "Pick".to_string(),
            params: vec![],
            clauses: vec![ActionClause::LetWithPrimes {
                expr: "LET choices == {1, 2} IN /\\ \\E pick \\in choices : /\\ x' = pick /\\ UNCHANGED <<y>>".to_string(),
            }],
        };

        let next_states = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("action should branch");
        assert_eq!(next_states.len(), 2);
        let next_xs: BTreeSet<i64> = next_states
            .iter()
            .map(|next| next.get("x").unwrap().as_int().unwrap())
            .collect();
        assert_eq!(next_xs, BTreeSet::from([1, 2]));
        for next in next_states {
            assert_eq!(next.get("y"), Some(&TlaValue::Int(9)));
        }
    }

    #[test]
    fn eval_action_body_supports_box_stuttering_formulas() {
        let state = tla_state([("x", TlaValue::Int(0))]);
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let branches =
            eval_action_body_multi("[x' = x + 1]_x", &ctx, &BTreeMap::new()).expect("box action");
        assert_eq!(branches.len(), 2);
        assert!(
            branches
                .iter()
                .any(|(staged, _)| staged.get("x") == Some(&TlaValue::Int(1)))
        );
        assert!(
            branches
                .iter()
                .any(|(staged, _)| staged.get("x") == Some(&TlaValue::Int(0)))
        );
    }

}
