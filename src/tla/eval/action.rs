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
    // Strip a redundant outer paren wrapper so a *parenthesized* disjunction of
    // actions -- `( \/ b1 \/ b2 )`, as in nbacc_ray97's `UponSent`:
    //   /\ pc[self] = "SENT"
    //   /\ ( \/ (/\ fd'[self] = TRUE  /\ pc' = ... )
    //        \/ (/\ fd'[self] = FALSE /\ ... /\ pc' = ...) )
    //   /\ sent' = sent
    // -- is split into its branches. Without this the `\/` is hidden inside the
    // parens, `split_action_body_disjuncts` returns a single disjunct, and the
    // clause falls through to boolean evaluation, dropping every successor (so
    // `UponSent` never fired and COMMIT/ABORT were unreachable).
    let trimmed = crate::tla::eval::splitter::strip_outer_parens(expr.trim());
    let disjuncts = split_action_body_disjuncts(trimmed);
    if disjuncts.len() <= 1 {
        return None;
    }

    let mut out = Vec::new();
    let mut first_err = None;
    let mut saw_clean_empty = false;
    for disjunct in disjuncts {
        // Each split disjunct keeps its own paren wrapper -- `( /\ g /\ x' = ... )`
        // -- which must be stripped so its inner `/\` conjuncts split (otherwise
        // the whole disjunct is read as one boolean clause and drops its primed
        // assignment).
        let disjunct = crate::tla::eval::splitter::strip_outer_parens(disjunct.trim());
        match eval_action_body_text_multi(disjunct, ctx, branch.clone()) {
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

/// Detect whether an argument expression is (or contains) a primed reference,
/// i.e. a `'` that primes a variable at the top level (outside string literals
/// and outside bracketed sub-expressions).
///
/// This is the signal that a defined operator called with such an argument is
/// *acting as an action* even though its own body has no syntactic prime — the
/// canonical Specifying-Systems shape is a CONSTANT operator override like
/// `Send(p,d,oldMemInt,newMemInt) == newMemInt = <<p,d>>` invoked as
/// `Send(p, req, memInt, memInt')`. There, `newMemInt` binds to `memInt'`, so
/// the body `newMemInt = <<p,d>>` is really the primed assignment
/// `memInt' = <<p,req>>`. `looks_like_action` inspects only the body text, so
/// it misses this; the "actionness" comes from the call site.
fn arg_is_primed_expr(arg: &str) -> bool {
    let bytes = arg.as_bytes();
    let mut in_string = false;
    let mut escaped = false;
    let mut depth: i32 = 0;
    for (i, &ch) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == b'\\' {
                escaped = true;
            } else if ch == b'"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            b'"' => in_string = true,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b'\'' if depth == 0 => {
                // A prime must follow an identifier / closing bracket, not be
                // part of a string or a `\/`-style token. Preceding an
                // alphanumeric/`_`/`]`/`)` is the reliable signal.
                if let Some(prev) = i.checked_sub(1).map(|j| bytes[j]) {
                    if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b']' || prev == b')'
                    {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Textually substitute call arguments for parameters in an operator body,
/// preserving word boundaries. Used when a call passes a primed argument so the
/// substituted body (e.g. `memInt' = <<p,req>>`) surfaces the primed assignment
/// that value-binding a next-state variable cannot express.
pub(crate) fn substitute_params_text(body: &str, params: &[String], args: &[String]) -> String {
    let mut result = body.to_string();
    for (param, arg) in params.iter().zip(args.iter()) {
        let arg = arg.trim();
        // Parenthesize compound arguments to preserve precedence, but leave
        // atoms (simple names, primed names, numbers) bare. A bare primed name
        // is essential: wrapping `m'` as `(m')` would make the substituted
        // `(m') = <<..>>` fail primed-assignment classification (the LHS no
        // longer ends in `'`), silently dropping the successor.
        let replacement = if arg_is_atom(arg) {
            arg.to_string()
        } else {
            format!("({arg})")
        };
        result = replace_identifier_word(&result, param, &replacement);
    }
    result
}

/// A substitution argument is an "atom" (needs no protective parentheses) when
/// it is a simple identifier, a primed identifier, or a numeric literal.
fn arg_is_atom(arg: &str) -> bool {
    let core = arg.strip_suffix('\'').unwrap_or(arg);
    if core.is_empty() {
        return false;
    }
    let mut chars = core.chars();
    let first = chars.next().unwrap();
    if first.is_ascii_digit() {
        return core.chars().all(|c| c.is_ascii_digit());
    }
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    core.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn replace_identifier_word(expr: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(expr.len());
    let mut word = String::new();
    let mut in_string = false;
    let mut escaped = false;
    let flush = |word: &mut String, result: &mut String| {
        if !word.is_empty() {
            if word == from {
                result.push_str(to);
            } else {
                result.push_str(word);
            }
            word.clear();
        }
    };
    for ch in expr.chars() {
        if in_string {
            result.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch.is_alphanumeric() || ch == '_' {
            word.push(ch);
        } else {
            flush(&mut word, &mut result);
            if ch == '"' {
                in_string = true;
            }
            result.push(ch);
        }
    }
    flush(&mut word, &mut result);
    result
}

fn expand_action_call_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    branch: ActionEvalBranch,
) -> Option<Result<Vec<ActionEvalBranch>>> {
    let (name, arg_exprs) = parse_action_call_expr(expr)?;
    let has_primed_arg = arg_exprs.iter().any(|a| arg_is_primed_expr(a));

    if let Some((alias, operator_name)) = name.split_once('!') {
        let instances = ctx.instances?;
        let instance = instances.get(alias)?;
        let module = instance.module.as_ref()?;
        let def = module.definitions.get(operator_name)?.clone();
        let mut probe_ctx = ctx.clone();
        probe_ctx.definitions = Some(&module.definitions);
        let looks_action = def_is_transitively_action(&def, &probe_ctx, 0);
        if (!looks_action && !has_primed_arg) || def.body.trim() == expr.trim() {
            return None;
        }
        // A CONSTANT-operator override invoked with a primed argument (e.g.
        // `Send(p, req, memInt, memInt')`) is an action only via that argument.
        // Substitute the args into the body textually so the primed assignment
        // surfaces, then evaluate the substituted body as an action.
        if !looks_action && has_primed_arg && def.params.len() == arg_exprs.len() {
            let substituted =
                substitute_params_text(&def.body, &def.params, &arg_exprs);
            let mut instance_ctx = ctx.clone();
            instance_ctx.definitions = Some(&module.definitions);
            instance_ctx.instances =
                effective_instance_scope(&module.instances, ctx.instances);
            seed_implicit_instance_constant_bindings(instance, module, ctx, &mut instance_ctx);
            {
                let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
                if let Err(err) = bind_instance_substitutions(instance, ctx, locals_mut) {
                    return Some(Err(err));
                }
            }
            return Some(eval_action_body_text_multi(&substituted, &instance_ctx, branch));
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
    let looks_action = def_is_transitively_action(&def, ctx, 0);
    if (!looks_action && !has_primed_arg) || def.body.trim() == expr.trim() {
        return None;
    }
    if def.params.len() != arg_exprs.len() {
        return Some(Err(anyhow!(
            "operator '{name}' arity mismatch: expected {}, got {}",
            def.params.len(),
            arg_exprs.len()
        )));
    }

    // A primed argument (e.g. `lastHash'`) names a *next-state variable*, not a
    // value: it has no defined value until the action assigns it. So a body
    // clause referencing that parameter (`newLastHash = hash`) is meant to
    // become a primed *assignment* (`lastHash' = hash`), which only a textual
    // substitution of `param -> arg` can express. Value-binding a primed arg
    // instead evaluates `lastHash'` (unassigned) and turns the clause into a
    // guard against a bogus value, silently dropping the successor.
    //
    // Substitute the *primed* args textually so their assignments surface, then
    // value-bind the remaining non-primed args (each is a genuine value and
    // safe/cheaper to bind). This subsumes the earlier `!looks_action` case: a
    // CONSTANT-operator override invoked with a primed arg — e.g. Nano's
    // `CalculateHash(genesisBlock, lastHash, lastHash') <- CalculateHashImpl` —
    // is transitively an action (`looks_action == true`) yet still needs its
    // `lastHash'` arg threaded textually so `newLastHash = hash` becomes the
    // primed assignment `lastHash' = hash`.
    if has_primed_arg {
        let mut substituted = def.body.clone();
        let mut value_params: Vec<&String> = Vec::new();
        let mut value_args: Vec<&String> = Vec::new();
        for (param, arg_expr) in def.params.iter().zip(arg_exprs.iter()) {
            if arg_is_primed_expr(arg_expr) {
                substituted = substitute_params_text(
                    &substituted,
                    std::slice::from_ref(param),
                    std::slice::from_ref(arg_expr),
                );
            } else {
                value_params.push(param);
                value_args.push(arg_expr);
            }
        }

        let mut child_ctx = ctx.clone();
        {
            let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
            for (param, arg_expr) in value_params.iter().zip(value_args.iter()) {
                let value = match eval_expr(arg_expr, ctx) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                if let Err(err) = bind_param_value(locals_mut, param, value) {
                    return Some(Err(err));
                }
            }
        }
        return Some(eval_action_body_text_multi(&substituted, &child_ctx, branch));
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

/// Evaluate a `CASE`-of-actions clause. Returns `Ok(None)` when `expr` is not a
/// CASE expression (caller falls through to the other clause handlers) and
/// `Ok(Some(branches))` when it is — firing every guard-satisfied arm as an
/// action branch (TLA+'s nondeterministic CASE). `OTHER`/default arms fire only
/// when no explicit guard held.
fn eval_case_action_multi(
    expr: &str,
    ctx: &EvalContext<'_>,
    eval_ctx: &EvalContext<'_>,
    branch: &ActionEvalBranch,
) -> Result<Option<Vec<ActionEvalBranch>>> {
    let trimmed = expr.trim();
    let Some(after_case) = trimmed.strip_prefix("CASE") else {
        return Ok(None);
    };
    // Require a word boundary after `CASE` (not `CASEFOO`).
    if after_case
        .chars()
        .next()
        .is_some_and(|c| c.is_alphanumeric() || c == '_')
    {
        return Ok(None);
    }
    let arms = split_top_level_symbol(after_case, "[]");
    if arms.is_empty() {
        return Ok(None);
    }

    let mut out = Vec::new();
    let mut default_arm: Option<String> = None;
    let mut any_guard_matched = false;
    let mut parsed_any_arm = false;
    for arm in arms {
        let arm = arm.trim();
        if arm.is_empty() {
            continue;
        }
        let Some((cond, rhs)) = crate::tla::eval::splitter::split_once_top_level(arm, "->") else {
            // Not a well-formed CASE arm — this isn't a CASE-of-actions after
            // all; let the caller's value-based guard path handle it.
            return Ok(None);
        };
        parsed_any_arm = true;
        if cond.trim() == "OTHER" {
            default_arm = Some(rhs.trim().to_string());
            continue;
        }
        if eval_guard(cond.trim(), eval_ctx)? {
            any_guard_matched = true;
            out.extend(eval_action_body_text_multi(rhs.trim(), ctx, branch.clone())?);
        }
    }

    if !parsed_any_arm {
        return Ok(None);
    }
    if !any_guard_matched {
        if let Some(default_expr) = default_arm {
            return Ok(Some(eval_action_body_text_multi(
                &default_expr,
                ctx,
                branch.clone(),
            )?));
        }
        // No arm matched and no OTHER: the CASE is a disabled action (like a
        // failed guard) — no successors.
        return Ok(Some(Vec::new()));
    }
    Ok(Some(out))
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
    // A `CASE cond1 -> Action1 [] cond2 -> Action2 [] OTHER -> Action3` whose
    // arms are *actions* (they assign primed variables, directly or via an
    // operator call) must be evaluated as an action, firing each arm whose guard
    // holds — not as a value expression. `eval_guard` would evaluate the whole
    // CASE as a boolean, calling the winning arm's action in value context where
    // its primed assignment errors, so the clause produces zero successors and
    // the enclosing action never fires (ReadersWriters' `ReadOrWrite`). Mirror
    // TLA+'s nondeterministic CASE: expand every guard-satisfied arm as an
    // action branch.
    if let Some(branches) = eval_case_action_multi(trimmed, ctx, &eval_ctx, &branch)? {
        return Ok(branches);
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

/// Top-level binary set operator recognized by `matches_membership_expr`
/// for structural (non-materializing) membership testing.
#[derive(Clone, Copy)]
enum SetMemberOp {
    Union,
    Intersect,
    Difference,
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
            // `value \in SUBSET S` holds iff `value` is a set and every element
            // of it is a member of `S`. Materializing `SUBSET S` (the powerset)
            // is exponential and trips the eval budget on record-set codomains
            // like Nano's `received \in [Node -> SUBSET SignedBlock]` (SignedBlock
            // is a large record set, so `SUBSET SignedBlock` is 2^|SignedBlock|).
            // Evaluate it structurally instead — this mirrors the compiled path
            // (`compiled_eval::compiled_membership_contains` / `membership_matches_text`,
            // commit 459d3aa) and TLC's `x \subseteq S` handling. Without it the
            // interpreted function-set codomain recursion below erroneously
            // materialized the powerset and reported a false invariant violation
            // on the initial state (NanoBlockchain MCNano{Small,Medium}).
            if let Some(set_expr) = rhs_trimmed.strip_prefix("SUBSET ") {
                return match value {
                    TlaValue::Set(elems) => {
                        for e in elems.iter() {
                            if !matches_membership_expr(e, set_expr, ctx, depth + 1)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                };
            }

            if let Some(runtime_value) = ctx.runtime_value(rhs_trimmed) {
                // A constant substituted to an infinite built-in set (e.g.
                // `INSTANCE M WITH Values <- Int`) is bound as the runtime
                // value `ModelValue("Int")`, since bare `Int`/`Nat`/`BOOLEAN`
                // have no enumerable set value. Recognize those names and
                // apply the built-in membership test rather than calling
                // `.contains()` on the model value (which errors "expected
                // Set, got ModelValue"). This is what makes the Disruptor
                // RingBuffer `TypeOk` codomain `Values \union {NULL}` (with
                // `Values <- Int`) evaluate correctly.
                if let TlaValue::ModelValue(name) = &runtime_value {
                    match name.as_str() {
                        "Nat" => return Ok(matches!(value.as_int(), Ok(n) if n >= 0)),
                        "Int" => return Ok(value.as_int().is_ok()),
                        "BOOLEAN" => return Ok(matches!(value, TlaValue::Bool(_))),
                        _ => {}
                    }
                }
                return runtime_value.contains(value);
            }

            if let Some(def) = ctx.definition(rhs_trimmed)
                && def.params.is_empty()
            {
                return matches_membership_expr(value, &def.body, ctx, depth + 1);
            }

            // A parameterized defined operator whose body denotes a set --
            // e.g. `ArrayOfAnyLength(T) == [elems: Seq(T)]`. Substitute the
            // argument text into the body and test membership structurally, so
            // an infinite codomain like `Seq(T)` is never materialized (which
            // would fail with "unknown operator/function 'Seq'"). Mirrors the
            // nullary case above. Built-ins like `Seq(S)`/`SUBSET` are handled
            // by their own arms below, since they have no user definition.
            if let Some((op_name, arg_texts)) =
                crate::tla::compiled_expr::parse_op_call(rhs_trimmed)
                && let Some(def) = ctx.definition(op_name)
                && !def.params.is_empty()
                && def.params.len() == arg_texts.len()
            {
                let substituted =
                    substitute_params_text(&def.body, &def.params, &arg_texts);
                return matches_membership_expr(value, &substituted, ctx, depth + 1);
            }

            // `value \in UNION SS`  <=>  `\E S \in SS : value \in S`. When the
            // outer set `SS` is an explicit set literal `{ E1, E2, ... }` we can
            // test membership structurally against each element without
            // materializing any element (which matters when an element is an
            // infinite function set like `[D -> Int \union {NULL}]`). This is
            // the `UNION { [0..N -> Values \union {NULL}] }` idiom used by the
            // Disruptor RingBuffer TypeOk invariant. Non-literal `UNION` args
            // fall through to the generic evaluator below.
            if let Some(rest) = rhs_trimmed.strip_prefix("UNION")
                && rest
                    .chars()
                    .next()
                    .is_none_or(|c| c.is_whitespace() || c == '{' || c == '(')
            {
                let inner = crate::tla::eval::splitter::strip_outer_parens(rest.trim());
                if inner.starts_with('{') && inner.ends_with('}') {
                    let body = inner[1..inner.len() - 1].trim();
                    // Only the explicit-set-literal form (no `\in`/`:`, i.e. not
                    // a comprehension) is safe to distribute element-wise.
                    if !body.is_empty()
                        && crate::tla::eval::splitter::find_top_level_keyword_index(body, "\\in")
                            .is_none()
                        && crate::tla::compiled_expr::find_top_level_colon(body).is_none()
                    {
                        for element in split_top_level_symbol(body, ",") {
                            let element = element.trim();
                            if !element.is_empty()
                                && matches_membership_expr(value, element, ctx, depth + 1)?
                            {
                                return Ok(true);
                            }
                        }
                        return Ok(false);
                    }
                }
            }

            // Structural handling of top-level binary set operators so that
            // membership tests against expressions involving infinite sets
            // (e.g. `Int`, `Nat`) never need to materialize an operand.
            //   value \in (A \union B)     <=>  value \in A  \/  value \in B
            //   value \in (A \intersect B) <=>  value \in A  /\  value \in B
            //   value \in (A \ B)          <=>  value \in A  /\  ~(value \in B)
            // Without this, `value \in (Int \union {NULL})` falls through to
            // `eval_expr_inner`, which tries to `\union` a bare `Int`
            // (evaluated as ModelValue) with a set and errors out
            // ("expected Set, got ModelValue(\"Int\")"). This surfaces in the
            // Disruptor RingBuffer TypeOk invariant, whose slot codomain is
            // `Values \union {NULL}` with `Values <- Int`.
            {
                let stripped = crate::tla::eval::splitter::strip_outer_parens(rhs_trimmed);
                for (kw, op) in [
                    ("\\union", SetMemberOp::Union),
                    ("\\cup", SetMemberOp::Union),
                    ("\\intersect", SetMemberOp::Intersect),
                    ("\\cap", SetMemberOp::Intersect),
                    ("\\", SetMemberOp::Difference),
                ] {
                    if let Some(idx) =
                        crate::tla::eval::splitter::find_top_level_keyword_index(stripped, kw)
                    {
                        // `\` must be the standalone set-difference operator,
                        // not the leading backslash of another keyword such as
                        // `\union` / `\in` / `\/`.
                        if kw == "\\" {
                            let after = stripped[idx + 1..].trim_start_matches(' ');
                            if after
                                .chars()
                                .next()
                                .is_none_or(|c| c.is_alphabetic() || c == '/')
                            {
                                continue;
                            }
                        }
                        let lhs = stripped[..idx].trim();
                        let rhs = stripped[idx + kw.len()..].trim();
                        if lhs.is_empty() || rhs.is_empty() {
                            continue;
                        }
                        return Ok(match op {
                            SetMemberOp::Union => {
                                matches_membership_expr(value, lhs, ctx, depth + 1)?
                                    || matches_membership_expr(value, rhs, ctx, depth + 1)?
                            }
                            SetMemberOp::Intersect => {
                                matches_membership_expr(value, lhs, ctx, depth + 1)?
                                    && matches_membership_expr(value, rhs, ctx, depth + 1)?
                            }
                            SetMemberOp::Difference => {
                                matches_membership_expr(value, lhs, ctx, depth + 1)?
                                    && !matches_membership_expr(value, rhs, ctx, depth + 1)?
                            }
                        });
                    }
                }
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

                // Distinguish a function set `[D -> R]` from a record set
                // `[f1: S1, ...]` using a *top-level* arrow. A plain
                // `inner.find("->")` would match an arrow nested inside a
                // field type (e.g. `[slots: UNION { [0..N -> R] }, ...]`) and
                // misclassify the record set as a function set — testing a
                // Record value against a function-set shape then returns false,
                // spuriously failing the invariant (Disruptor RingBuffer
                // TypeOk).
                if let Some((domain_expr, codomain_expr)) =
                    crate::tla::eval::splitter::split_once_top_level(inner, "->")
                {
                    let domain_expr = domain_expr.trim();
                    let codomain_expr = codomain_expr.trim();
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
                        // A sequence `<<e1,...,en>>` is the function `[1..n -> R]`;
                        // it is a member of `[D -> R]` when `D = 1..n` and every
                        // element is in `R`. Sequences are stored as a distinct
                        // `Seq` value, so accept them here too (mirrors the
                        // compiled FunctionSet arm).
                        TlaValue::Seq(seq) => {
                            let domain_val = eval_expr_inner(domain_expr, ctx, depth + 1)?;
                            let domain_set = match domain_val.as_set() {
                                Ok(set) => set,
                                Err(_) => return Ok(false),
                            };
                            let seq_domain: BTreeSet<TlaValue> =
                                (1..=seq.len() as i64).map(TlaValue::Int).collect();
                            if seq_domain != *domain_set {
                                return Ok(false);
                            }
                            for item in seq.iter() {
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
    fn case_of_actions_fires_matching_arm_as_action() {
        // Regression (ReadersWriters `ReadOrWrite`): a `CASE cond -> Action []
        // ...` whose arms are actions must fire the matching arm as an action,
        // staging its primed assignment — not evaluate the CASE as a boolean.
        let state = tla_state([
            ("mode", TlaValue::String("read".to_string())),
            ("out", TlaValue::Int(0)),
        ]);
        let mut definitions = BTreeMap::new();
        definitions.insert(
            "DoRead".to_string(),
            TlaDefinition {
                name: "DoRead".to_string(),
                params: vec![],
                body: "out' = 1".to_string(),
                is_recursive: false,
            },
        );
        definitions.insert(
            "DoWrite".to_string(),
            TlaDefinition {
                name: "DoWrite".to_string(),
                params: vec![],
                body: "out' = 2".to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &definitions);
        let branches = eval_action_body_multi(
            "CASE mode = \"read\" -> DoRead [] mode = \"write\" -> DoWrite",
            &ctx,
            &BTreeMap::new(),
        )
        .expect("CASE-of-actions should evaluate");
        assert_eq!(branches.len(), 1, "{branches:?}");
        assert_eq!(branches[0].0.get("out"), Some(&TlaValue::Int(1)));
    }

    #[test]
    fn operator_called_with_primed_argument_stages_assignment() {
        // Regression (Specifying-Systems memory specs): a defined operator whose
        // body has no prime but is called with a primed argument acts as an
        // action — `Send(p, req, memInt, memInt')` with
        // `Send(p,d,om,nm) == nm = <<p,d>>` means `memInt' = <<p,req>>`.
        let state = tla_state([("memInt", TlaValue::Int(0))]);
        let mut definitions = BTreeMap::new();
        definitions.insert(
            "Send".to_string(),
            TlaDefinition {
                name: "Send".to_string(),
                params: vec![
                    "p".to_string(),
                    "d".to_string(),
                    "om".to_string(),
                    "nm".to_string(),
                ],
                body: "nm = <<p, d>>".to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &definitions);
        let branches = eval_action_body_multi("Send(7, 9, memInt, memInt')", &ctx, &BTreeMap::new())
            .expect("operator with primed arg should evaluate as an action");
        assert_eq!(branches.len(), 1, "{branches:?}");
        let memint = branches[0].0.get("memInt").expect("memInt' staged");
        let expected = eval_expr("<<7, 9>>", &ctx).expect("tuple evaluates");
        assert_eq!(memint, &expected);
    }

    #[test]
    fn action_operator_called_with_primed_argument_substitutes_it_textually() {
        // Regression (NanoBlockchain `CalculateHash <- CalculateHashImpl`). An
        // operator whose body *is* an action (assigns a prime) but which is ALSO
        // called with a primed argument must still substitute that primed arg
        // textually — value-binding it evaluates the (unassigned) next-state
        // variable and turns the intended primed assignment `newV = w` into a
        // guard against a bogus value, dropping the successor.
        //   Set(v, nv) == /\ hits' = hits + 1 /\ nv = v
        //   Set(5, cursor')  ==>  cursor' = 5  /\  hits' = hits + 1
        let state = tla_state([("hits", TlaValue::Int(0)), ("cursor", TlaValue::Int(0))]);
        let mut definitions = BTreeMap::new();
        definitions.insert(
            "Set".to_string(),
            TlaDefinition {
                name: "Set".to_string(),
                params: vec!["v".to_string(), "nv".to_string()],
                body: "/\\ hits' = hits + 1 /\\ nv = v".to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &definitions);
        let branches = eval_action_body_multi("Set(5, cursor')", &ctx, &BTreeMap::new())
            .expect("action operator with a primed arg should still fire");
        assert_eq!(branches.len(), 1, "{branches:?}");
        assert_eq!(branches[0].0.get("cursor"), Some(&TlaValue::Int(5)));
        assert_eq!(branches[0].0.get("hits"), Some(&TlaValue::Int(1)));
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
    fn subset_membership_is_structural_not_powerset_materialized() {
        use crate::tla::hashed_arc::HashedArc;

        // Regression for the NanoBlockchain MCNano{Small,Medium} false-violation:
        // `received \in [Node -> SUBSET SignedBlock]` in Nano's TypeInvariant.
        // At the initial state `received[n] = {}`, so the codomain membership is
        // `{} \in SUBSET SignedBlock`, which must be TRUE without materializing
        // the exponential powerset of the (large) record set `SignedBlock`.
        let state = tla_state([]);
        let mut defs = BTreeMap::new();
        // A "large" record set standing in for SignedBlock — enumerating its
        // powerset would be 2^(3*3) = 512 here, but for the real spec it is
        // 2^(hundreds), which trips the eval budget. The structural path never
        // enumerates it.
        defs.insert(
            "SignedBlock".to_string(),
            TlaDefinition {
                name: "SignedBlock".to_string(),
                params: vec![],
                body: "[block : {\"a\", \"b\", \"c\"}, sig : {\"x\", \"y\", \"z\"}]".to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        // {} \in SUBSET SignedBlock  ==>  true (empty set is a subset of anything).
        let empty = TlaValue::Set(HashedArc::new(BTreeSet::new()));
        assert!(
            matches_membership_expr(&empty, "SUBSET SignedBlock", &ctx, 0)
                .expect("SUBSET membership must not error"),
            "empty set must be a member of SUBSET SignedBlock"
        );

        // A set whose element IS a SignedBlock record is a member.
        let good_elem = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("block".to_string(), TlaValue::String("a".to_string())),
            ("sig".to_string(), TlaValue::String("x".to_string())),
        ])));
        let good = TlaValue::Set(HashedArc::new(BTreeSet::from([good_elem])));
        assert!(
            matches_membership_expr(&good, "SUBSET SignedBlock", &ctx, 0)
                .expect("SUBSET membership must not error"),
            "{{signedBlock}} must be a member of SUBSET SignedBlock"
        );

        // A set containing a non-member element is NOT a member.
        let bad_elem = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("block".to_string(), TlaValue::String("zzz".to_string())),
            ("sig".to_string(), TlaValue::String("x".to_string())),
        ])));
        let bad = TlaValue::Set(HashedArc::new(BTreeSet::from([bad_elem])));
        assert!(
            !matches_membership_expr(&bad, "SUBSET SignedBlock", &ctx, 0)
                .expect("SUBSET membership must not error"),
            "a set with a non-SignedBlock element is not in SUBSET SignedBlock"
        );

        // A non-set value is never a member of a powerset.
        assert!(
            !matches_membership_expr(&TlaValue::Int(3), "SUBSET SignedBlock", &ctx, 0)
                .expect("SUBSET membership must not error"),
        );
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
