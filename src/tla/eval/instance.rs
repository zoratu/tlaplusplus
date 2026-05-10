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

    // Get the module instance
    let instances = ctx
        .instances
        .ok_or_else(|| anyhow!("no module instances available in context"))?;

    let instance = instances
        .get(alias)
        .ok_or_else(|| anyhow!("module instance '{}' not found", alias))?;

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
