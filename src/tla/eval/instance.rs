//! Module-instance and ENABLED operator evaluation.
//!
//! `eval_module_instance_call` resolves `Alias!Operator(args)` against an
//! INSTANCE substitution map and dispatches into the operator body.
//! `eval_module_instance_ref` is the nullary sugar (`Alias!Constant`).
//! `eval_enabled` implements TLA+'s `ENABLED A` operator.

use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::tla::TlaValue;

use super::{
    EvalContext, MAX_EVAL_DEPTH, apply_action_ir_with_context_multi,
    bind_instance_substitutions, bind_param_value, eval_expr_inner,
    seed_implicit_instance_constant_bindings,
};

/// Evaluate a module instance operator call: Alias!Operator(args)
pub(super) fn eval_module_instance_call(
    alias: &str,
    operator_name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!(
            "module instance recursion depth exceeded at {alias}!{operator_name}"
        ));
    }

    // Module-level instance (`Alias == INSTANCE M` at module scope) first.
    let instance = ctx.instances.and_then(|instances| instances.get(alias));

    let Some(instance) = instance else {
        // Fallback: an INLINE instance bound in a LET, i.e.
        // `LET alias == INSTANCE M [WITH ...] IN ... alias!Op(...) ...`.
        // The LET binding is recorded as a local definition whose body is
        // `INSTANCE M ...`; resolve `alias!Op` against it. Without this,
        // the inline instance binding errors, silently swallowing the
        // operator that referenced it (the MCCheckpointCoordination
        // ShouldReplaceLease-override soundness bug).
        if let Some(def) = ctx.local_definitions.get(alias) {
            if let Some(rest) = def.body.trim_start().strip_prefix("INSTANCE") {
                if rest.starts_with(char::is_whitespace) {
                    return eval_inline_instance_call(
                        rest.trim_start(),
                        operator_name,
                        args,
                        ctx,
                        depth,
                    );
                }
            }
        }
        return Err(anyhow!("module instance '{}' not found", alias));
    };

    // Get the module
    let module = instance.module.as_ref().ok_or_else(|| {
        anyhow!(
            "module '{}' not loaded for instance '{}'",
            instance.module_name,
            alias
        )
    })?;

    // Look up the operator in the instance module
    let operator_def = module.definitions.get(operator_name).ok_or_else(|| {
        anyhow!(
            "operator '{}' not found in module '{}'",
            operator_name,
            instance.module_name
        )
    })?;

    // Check arity
    if operator_def.params.len() != args.len() {
        return Err(anyhow!(
            "operator '{alias}!{operator_name}' arity mismatch: expected {}, got {}",
            operator_def.params.len(),
            args.len()
        ));
    }

    // Create a new context with the instance module's definitions and instances
    let mut instance_ctx = ctx.clone();
    instance_ctx.definitions = Some(&module.definitions);
    instance_ctx.instances = effective_instance_scope(&module.instances, ctx.instances);
    seed_implicit_instance_constant_bindings(instance, module, ctx, &mut instance_ctx);

    // Apply substitutions: replace constants in the context
    {
        let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
        bind_instance_substitutions(instance, ctx, locals_mut)?;
    }

    let mut child_ctx = instance_ctx;
    {
        let locals_mut = std::rc::Rc::make_mut(&mut child_ctx.locals);
        for (param, arg) in operator_def.params.iter().zip(args.into_iter()) {
            bind_param_value(locals_mut, param, arg)?;
        }
    }

    // Evaluate the operator body
    eval_expr_inner(&operator_def.body, &child_ctx, depth + 1)
}

/// Evaluate `alias!Op(args)` where `alias` is an INLINE instance bound by a
/// `LET alias == INSTANCE M [WITH p <- v, ...] IN ...` rather than a
/// module-level `INSTANCE` declaration.
///
/// `instance_spec` is the text after `INSTANCE` in the LET binding body, e.g.
/// `"CheckpointCoordination"` or `"CheckpointCoordination WITH N <- Nodes"`.
///
/// The instantiated module is the one the enclosing module `EXTENDS`, so its
/// definitions are already flattened into the evaluation scope. The one
/// subtlety is config operator-overrides: `M!Op` must reference M's *original*
/// `Op`, not a `CONSTANT Op <- ...` / definition override applied to the
/// instantiating module. `inject_constants_into_definitions` preserves the
/// pre-override definition under `__Original_<Op>__`, so we prefer that.
fn eval_inline_instance_call(
    instance_spec: &str,
    operator_name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!(
            "inline instance recursion depth exceeded at !{operator_name}"
        ));
    }

    // Resolve the operator, preferring the pre-override original.
    let backup = format!("__Original_{operator_name}__");
    let def = ctx
        .definition(&backup)
        .or_else(|| ctx.definition(operator_name))
        .ok_or_else(|| {
            anyhow!("operator '{operator_name}' not found for inline INSTANCE '{instance_spec}'")
        })?;

    if def.params.len() != args.len() {
        return Err(anyhow!(
            "inline instance operator '{operator_name}' arity mismatch: expected {}, got {}",
            def.params.len(),
            args.len()
        ));
    }

    let mut child = ctx.clone();
    {
        let locals_mut = std::rc::Rc::make_mut(&mut child.locals);
        // Apply any `WITH param <- expr` substitutions, evaluated in the
        // caller's context, before binding the operator's own parameters.
        for (param, sub_expr) in parse_instance_with_substitutions(instance_spec) {
            let value = eval_expr_inner(&sub_expr, ctx, depth + 1)?;
            bind_param_value(locals_mut, &param, value)?;
        }
        for (param, arg) in def.params.iter().zip(args.into_iter()) {
            bind_param_value(locals_mut, param, arg)?;
        }
    }

    eval_expr_inner(&def.body, &child, depth + 1)
}

/// Parse the `WITH p1 <- e1, p2 <- e2, ...` clause of an INSTANCE spec into
/// `(param, expr)` pairs. Returns empty if there is no `WITH`.
fn parse_instance_with_substitutions(instance_spec: &str) -> Vec<(String, String)> {
    let Some(with_idx) = super::find_top_level_keyword_index(instance_spec, "WITH") else {
        return Vec::new();
    };
    let subs_text = &instance_spec[with_idx + "WITH".len()..];
    let mut out = Vec::new();
    for part in crate::tla::split_top_level(subs_text, ",") {
        if let Some((param, expr)) = part.split_once("<-") {
            let param = param.trim();
            let expr = expr.trim();
            if !param.is_empty() && !expr.is_empty() {
                out.push((param.to_string(), expr.to_string()));
            }
        }
    }
    out
}

pub(super) fn effective_instance_scope<'a>(
    module_instances: &'a BTreeMap<String, crate::tla::module::TlaModuleInstance>,
    parent_instances: Option<&'a BTreeMap<String, crate::tla::module::TlaModuleInstance>>,
) -> Option<&'a BTreeMap<String, crate::tla::module::TlaModuleInstance>> {
    if module_instances.is_empty() {
        parent_instances
    } else {
        Some(module_instances)
    }
}

/// Evaluate a module instance reference: Alias!Constant
pub(super) fn eval_module_instance_ref(
    alias: &str,
    name: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!(
            "module instance recursion depth exceeded at {alias}!{name}"
        ));
    }

    // Try to evaluate as a nullary operator call
    eval_module_instance_call(alias, name, vec![], ctx, depth)
}

/// Evaluate ENABLED operator: check if an action is enabled in the current state.
pub(super) fn eval_enabled(
    action_name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("ENABLED recursion depth exceeded at {action_name}"));
    }

    // Look up the action definition
    let action_def = ctx
        .definition(action_name)
        .ok_or_else(|| anyhow!("action '{}' not found for ENABLED", action_name))?;

    // When arguments are provided but arity mismatches, that's an error.
    // When no arguments are provided for a parameterized action (bare `ENABLED Send`),
    // return TRUE as a safe over-approximation — the action *may* be enabled for some
    // parameter values.  TLC would existentially quantify over the parameter domains,
    // but discovering those domains is not always possible in the evaluator.
    if action_def.params.len() != args.len() {
        if args.is_empty() && !action_def.params.is_empty() {
            // Bare ENABLED on a parameterized action — safe approximation
            return Ok(TlaValue::Bool(true));
        }
        return Err(anyhow!(
            "ENABLED action '{}' arity mismatch: expected {}, got {}",
            action_name,
            action_def.params.len(),
            args.len()
        ));
    }

    let mut enabled_ctx = ctx.clone();
    if !args.is_empty() {
        let locals_mut = Rc::make_mut(&mut enabled_ctx.locals);
        for (param, arg) in action_def.params.iter().zip(args.into_iter()) {
            bind_param_value(locals_mut, param, arg)?;
        }
    }

    // Compile the action to IR
    let action_ir = crate::tla::compile_action_ir(&action_def);

    // Try to apply the action to see if it's enabled
    // An action is enabled if it can produce a successor state
    match apply_action_ir_with_context_multi(&action_ir, ctx.state, &enabled_ctx) {
        Ok(next_states) => Ok(TlaValue::Bool(!next_states.is_empty())),
        Err(_) => {
            // Evaluation error - action is not enabled
            // (This can happen with complex expressions that aren't fully supported)
            Ok(TlaValue::Bool(false))
        }
    }
}
