//! Transition-context evaluation: action-constraint checking that
//! reads both pre-state and post-state variables (`x` vs `x'`).
//!
//! `eval_action_constraint` is the public entry point. It stages the
//! post-state (`next`) variables into an `EvalContext` as primes
//! (`var'`) alongside the pre-state (`current`) and dispatches to the
//! full expression interpreter (`eval_expr`), so any constraint shape —
//! `\in` over a range, set/record expressions, definition calls — is
//! evaluated the same way successor generation evaluates guards.

use anyhow::Result;
use std::collections::BTreeMap;

use crate::tla::{TlaDefinition, TlaState};

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
    // Evaluate through the FULL expression evaluator with `next`'s values staged
    // as primes (`var'`), so an action constraint like `step' \in step..(step+1)`
    // -- `\in` over a range, unprimed `step` from `current`, primed `step'` from
    // `next` -- is handled correctly. The bespoke `eval_transition_expr` below
    // only supported simple comparisons and mis-evaluated such constraints to
    // false, pruning every successor (LanguageFeatureMatrix: 1 state vs TLC's 21).
    let mut ctx = match definitions {
        Some(defs) => super::EvalContext::with_definitions(current, defs),
        None => super::EvalContext::new(current),
    };
    {
        let locals = std::rc::Rc::make_mut(&mut ctx.locals);
        for (name, value) in next.iter() {
            let mut key = String::with_capacity(name.len() + 1);
            key.push_str(name);
            key.push('\'');
            locals.insert(key, value.clone());
        }
    }
    super::eval_expr(expr, &ctx)?.as_bool()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::{TlaValue, tla_state};

    #[test]
    fn t4_apply_comparison_kills_int_match_arm_deletion() {
        // Kills eval.rs:457:9 (`delete match arm (TlaValue::Int(a), TlaValue::Int(b))`),
        // 458:23 (`>=` -> `<`) and the surrounding integer comparison ops.
        // apply_comparison is the transition-context comparison evaluator used
        // by eval_action_constraint(); a soundness regression there silently
        // mis-evaluates action constraints in temporal/fairness checks.
        let current = tla_state([("x", TlaValue::Int(0)), ("y", TlaValue::Int(0))]);
        let next = tla_state([("x", TlaValue::Int(5)), ("y", TlaValue::Int(3))]);

        // x' >= x  : 5 >= 0  -> true (kills `>= -> <`).
        assert!(
            eval_action_constraint("x' >= x", &current, &next, None)
                .expect("primed >= unprimed must evaluate"),
            "primed integer comparison `x' >= x` must succeed when 5 >= 0"
        );
        // x' < x  : 5 < 0  -> false (kills `< -> >`).
        assert!(
            !eval_action_constraint("x' < x", &current, &next, None)
                .expect("primed < unprimed must evaluate"),
            "primed integer comparison `x' < x` must be false when 5 < 0"
        );
        // y' = y  : 3 = 0  -> false (kills the integer-arm deletion mutant).
        assert!(
            !eval_action_constraint("y' = y", &current, &next, None)
                .expect("primed = must evaluate for ints"),
            "primed integer equality `y' = y` must be false when 3 != 0"
        );

        // Equal values, equal comparison.
        let current = tla_state([("z", TlaValue::Int(7))]);
        let next = tla_state([("z", TlaValue::Int(7))]);
        assert!(
            eval_action_constraint("z' = z", &current, &next, None)
                .expect("primed = must evaluate for equal ints"),
            "z' = z must be true when both are 7"
        );
        assert!(
            eval_action_constraint("z' >= z", &current, &next, None)
                .expect("primed >= must evaluate for equal ints"),
            "z' >= z must be true when both are 7"
        );
        assert!(
            !eval_action_constraint("z' > z", &current, &next, None)
                .expect("primed > must evaluate for equal ints"),
            "z' > z must be false when both are 7"
        );
    }

    #[test]
    fn action_constraint_membership_over_range() {
        // Regression for LanguageFeatureMatrix (ours=1 vs TLC=21): the action
        // constraint `step' \in step..(step+1)` is ALWAYS satisfiable (a step
        // that advances by one lands in the range), but the old bespoke
        // `eval_transition_expr` only understood simple binary comparisons and
        // errored/returned false on the `\in`-over-range shape, pruning every
        // successor. The full-interpreter path must evaluate it correctly.
        let current = tla_state([("step", TlaValue::Int(3))]);

        // step' = step + 1  -> 4 \in 3..4  -> true
        let next = tla_state([("step", TlaValue::Int(4))]);
        assert!(
            eval_action_constraint("step' \\in step..(step+1)", &current, &next, None)
                .expect("membership-over-range constraint must evaluate"),
            "4 \\in 3..4 must satisfy the action constraint"
        );

        // step' = step  -> 3 \in 3..4  -> true (in range)
        let next_same = tla_state([("step", TlaValue::Int(3))]);
        assert!(
            eval_action_constraint("step' \\in step..(step+1)", &current, &next_same, None)
                .expect("membership-over-range constraint must evaluate"),
            "3 \\in 3..4 must satisfy the action constraint"
        );

        // step' = step + 5  -> 8 \in 3..4  -> false (out of range: constraint prunes)
        let next_far = tla_state([("step", TlaValue::Int(8))]);
        assert!(
            !eval_action_constraint("step' \\in step..(step+1)", &current, &next_far, None)
                .expect("membership-over-range constraint must evaluate"),
            "8 \\in 3..4 must NOT satisfy the action constraint"
        );
    }

    #[test]
    fn t4_is_identifier_rejects_non_identifier_starts() {
        // Kills eval.rs:429:8 (`delete !` in `!(first.is_alphanumeric() || first == '_')`)
        // and 438:31 (`|=` -> `&=` in `saw_identifier_marker |= ...`).
        // is_identifier gates the transition evaluator's atom recognition;
        // accepting a bare digit or symbol as an identifier silently coerces
        // numeric literals into ModelValues during constraint checking.
        assert!(is_identifier("x"));
        assert!(is_identifier("foo_bar"));
        assert!(is_identifier("x'"), "primed identifier must be accepted");
        assert!(is_identifier("_internal"));

        // Bare digits are NOT identifiers — kills the `delete !` mutant which
        // would invert the leading-char check and accept "1" as an identifier.
        assert!(!is_identifier("1"));
        assert!(!is_identifier("123"));
        assert!(!is_identifier("9'"));
        // All-digit-with-underscore is also not an identifier — kills the
        // `|=` -> `&=` mutant, because saw_identifier_marker must remain
        // monotonically true once any letter / `_` is seen.
        // "x1" must succeed (`x` sets the marker; `1` doesn't clear it under
        // `|=` but would clear it under `&=`).
        assert!(
            is_identifier("x1"),
            "alphanumeric identifier must be accepted; `|=` -> `&=` mutant breaks this"
        );
        assert!(is_identifier("a1b2c3"));

        // Empty string => false.
        assert!(!is_identifier(""));
        // Symbol-leading => false.
        assert!(!is_identifier("+x"));
        assert!(!is_identifier("?"));
    }
}
