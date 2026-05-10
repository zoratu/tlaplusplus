//! Transition-context evaluation: action-constraint checking that
//! reads both pre-state and post-state variables (`x` vs `x'`).
//!
//! `eval_action_constraint` is the public entry point. It builds a
//! `TransitionContext` (declared in the parent module) and dispatches
//! to `eval_transition_expr`, which today only handles identifiers and
//! simple binary comparisons — full primed-variable evaluation goes
//! through the main interpreter via
//! `eval_action_clause_text_multi`.

use anyhow::{Result, anyhow};
use std::collections::BTreeMap;

use crate::tla::{TlaDefinition, TlaState, TlaValue};

use super::{MAX_EVAL_DEPTH, TransitionContext};

/// Evaluate an action constraint over a state transition
///
/// Action constraints can reference both current-state variables (x) and
/// next-state variables (x'). Returns true if the constraint is satisfied.
pub fn eval_action_constraint(
    expr: &str,
    current: &TlaState,
    next: &TlaState,
    definitions: Option<&BTreeMap<String, TlaDefinition>>,
) -> Result<bool> {
    let ctx = if let Some(defs) = definitions {
        TransitionContext::with_definitions(current, next, defs)
    } else {
        TransitionContext::new(current, next)
    };

    eval_transition_expr(expr, &ctx, 0)?.as_bool()
}

/// Evaluate an expression in a transition context (supporting primed variables)
fn eval_transition_expr(expr: &str, ctx: &TransitionContext<'_>, depth: usize) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("eval depth limit exceeded at {}", MAX_EVAL_DEPTH));
    }

    let expr = expr.trim();

    // Handle primed and unprimed identifiers
    if is_identifier(expr) {
        if let Some(val) = ctx.resolve_variable(expr) {
            return Ok(val);
        }

        // Try to resolve as definition
        if let Some(def) = ctx.definition(expr) {
            if def.params.is_empty() {
                // For now, we can't fully evaluate definitions in transition context
                // This would require a full transition-aware evaluator
                // Just return a model value
                return Ok(TlaValue::ModelValue(expr.to_string()));
            }
        }

        return Ok(TlaValue::ModelValue(expr.to_string()));
    }

    // For now, only support simple comparisons
    // Full implementation would need to recursively handle all expressions
    // with primed variable support throughout

    // Try simple comparison: x' >= x
    if let Some((left, op, right)) = parse_simple_comparison(expr) {
        let left_val = eval_transition_expr(left.trim(), ctx, depth + 1)?;
        let right_val = eval_transition_expr(right.trim(), ctx, depth + 1)?;

        return Ok(TlaValue::Bool(apply_comparison(&left_val, op, &right_val)?));
    }

    // Deferred: full primed-variable expression evaluation is not needed for
    // action constraint checking — the main evaluator in eval_action_clause_text_multi
    // handles primed variables via staged assignments and classify_clause().
    // This fallback only fires for eval_action_constraint() which is used for
    // post-hoc constraint validation, not successor generation.
    Err(anyhow!(
        "complex action constraints not yet fully supported: {}",
        expr
    ))
}

pub(super) fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // Allow primed identifiers (x')
    let base = s.strip_suffix('\'').unwrap_or(s);
    let mut chars = base.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_alphanumeric() || first == '_') {
        return false;
    }

    let mut saw_identifier_marker = first.is_alphabetic() || first == '_';
    for c in chars {
        if !(c.is_alphanumeric() || c == '_') {
            return false;
        }
        saw_identifier_marker |= c.is_alphabetic() || c == '_';
    }

    saw_identifier_marker
}

fn parse_simple_comparison(expr: &str) -> Option<(&str, &str, &str)> {
    for op in [">=", "<=", "=", ">", "<", "/="] {
        if let Some(idx) = expr.find(op) {
            let left = &expr[..idx];
            let right = &expr[idx + op.len()..];
            return Some((left, op, right));
        }
    }
    None
}

fn apply_comparison(left: &TlaValue, op: &str, right: &TlaValue) -> Result<bool> {
    match (left, right) {
        (TlaValue::Int(a), TlaValue::Int(b)) => Ok(match op {
            ">=" => a >= b,
            "<=" => a <= b,
            "=" => a == b,
            ">" => a > b,
            "<" => a < b,
            "/=" => a != b,
            _ => return Err(anyhow!("unknown operator: {}", op)),
        }),
        (TlaValue::Bool(a), TlaValue::Bool(b)) => Ok(match op {
            "=" => a == b,
            "/=" => a != b,
            _ => return Err(anyhow!("invalid operator for bool: {}", op)),
        }),
        _ => Err(anyhow!(
            "comparison not supported for types: {:?} {} {:?}",
            left,
            op,
            right
        )),
    }
}
