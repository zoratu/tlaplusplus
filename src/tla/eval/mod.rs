use crate::tla::{TlaDefinition, TlaState, TlaValue};
use crate::tla::hashed_arc::HashedArc;
// `ActionClause`, `ActionIr`, and `ClauseKind` are reused by the trailing
// `#[cfg(test)] mod tests` block (which constructs literal action IRs to
// drive `apply_action_ir` and friends through `use super::*;`).
#[cfg(test)]
use crate::tla::{ActionClause, ActionIr};
use anyhow::{Result, anyhow};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

mod budget;
mod splitter;
mod control;
mod transition;
mod bracket;
mod instance;
mod postfix;
mod quantifier;
mod set;
mod operator;
mod action;
mod dispatch;
mod expr;

// pub(crate) re-export of operator::eval_operator_call preserves the
// `crate::tla::eval::eval_operator_call` path used by compiled_eval.rs.
pub(crate) use operator::eval_operator_call;

// Sibling-submodule items reused inside this file (eval_expr_inner) AND
// items that other submodules reach via super::NAME.
use operator::{
    is_user_defined_infix_operator, record_key_from_value, seq_or_string_concat, tla_to_string,
};

// Sibling submodule items reused inside this file AND items that
// sibling submodules reach via `super::NAME`. The latter wouldn't
// resolve if these `use` lines didn't bring them into eval.rs's
// namespace first.
use quantifier::{
    bind_param_value, collect_binder_filter_set, collect_binder_map_set,
    collect_function_mapping, eval_choose_expression, eval_implies_parts,
    eval_lambda_expression, eval_quantifier_expression, parse_binders,
};
use set::{
    eval_set_expression, generate_permutations, powerset, try_eval_record_set,
};

// pub(crate) re-exports preserve the existing crate::tla::eval::NAME paths
// used by symbolic_init.rs (and tests).
#[cfg(feature = "symbolic-init")]
pub(crate) use set::{
    try_destructure_function_set_comprehension, try_resolve_funasseq_permutation_set,
    try_resolve_sequence_domain,
};

// Sibling submodule items reused inside this file.
use bracket::{eval_bracket_expression, eval_bracket_index_key, parse_argument_list};
use instance::{
    effective_instance_scope, eval_enabled, eval_module_instance_call,
    eval_module_instance_ref,
};
use postfix::eval_atom_with_postfix;

// pub(crate) re-export of bracket::apply_value preserves the existing
// `crate::tla::eval::apply_value` path used by compiled_eval.rs.
pub(crate) use bracket::apply_value;

// Re-export public transition-context entry point.
pub use transition::eval_action_constraint;

// Re-export action-evaluation entry points. The names below preserve the
// `crate::tla::eval::NAME` paths used by `compiled_eval.rs`,
// `action_exec.rs`, `models/tla_native.rs`, and `cli/probe.rs`. The
// `pub(super) use` items make `bind_instance_substitutions`,
// `seed_implicit_instance_constant_bindings`, and `matches_membership_expr`
// reachable as `super::NAME` from `instance.rs` / the eval_expr_inner
// dispatcher.
pub use action::{
    apply_action_ir, apply_action_ir_with_context, eval_action_body_multi, eval_let_action_multi,
};
pub(crate) use action::{
    apply_action_ir_with_context_multi, parse_action_binder_specs, parse_action_call_expr,
};
// `eval_let_action` is the legacy single-successor wrapper; kept
// `pub(crate)` for parity with the pre-split surface but currently
// unused outside the action module.
#[allow(unused_imports)]
pub(crate) use action::eval_let_action;
// Sibling submodule items (private to eval). Brought into eval.rs's
// namespace so siblings can resolve them via `super::NAME`. Also used
// by `eval_expr_inner` (matches_membership_expr).
use action::{
    bind_instance_substitutions, matches_membership_expr,
    seed_implicit_instance_constant_bindings,
};

// `eval_expr_inner` is the top-of-grammar dispatcher; lives in expr.rs.
// Imported here so the public `eval_expr` can delegate to it AND so
// sibling submodules can reach it via `super::eval_expr_inner`.
use expr::eval_expr_inner;

// Sibling submodule items reused inside this file.
use control::{
    eval_case_expression, eval_if_expression, eval_let_expression, parse_let_definitions,
    split_outer_let,
};
// transition.rs items reused only by the existing `#[cfg(test)] mod tests`
// block at the bottom of this file. Surfacing them in the lib build would
// emit a dead-import warning since no non-test code calls them.
#[cfg(test)]
use transition::is_identifier;

// Re-export bare names used throughout this file so call sites stay
// unchanged. The submodules own the implementations; this file is the
// umbrella module while the rest of the refactor lands.
//
// `pub use` / `pub(crate) use` here doubles as a re-export — preserves
// `crate::tla::eval::NAME` paths used by external callers.
pub use budget::{normalize_param_name, restore_eval_budget, set_active_eval_budget};
pub(crate) use splitter::{
    find_top_level_char, find_top_level_keyword_index, is_valid_identifier,
    split_once_top_level, split_top_level_keyword, split_top_level_symbol,
};
use budget::{MAX_EVAL_DEPTH, get_active_eval_budget};
use splitter::{
    contains_top_level_keyword, find_outer_else, find_outer_then,
    find_top_level_char_from, find_top_level_definition_eqs,
    has_keyword_boundaries, has_word_boundaries, is_binary_operator,
    is_word_char, is_wrapped_by, line_start_before, next_word,
    normalize_multiline_boolean_indentation, parse_identifier_prefix, parse_int_prefix,
    parse_string_literal_prefix, skip_leading_ws, split_indented_top_level_boolean,
    split_top_level, split_top_level_additive, split_top_level_comparison,
    split_top_level_defined_infix, split_top_level_multiplicative, split_top_level_range,
    split_top_level_set_minus, starts_with_keyword, starts_with_tla_backslash_operator,
    strip_outer_parens, take_angle_group, take_bracket_group, take_keyword_prefix,
};

/// Context for evaluating expressions on a single state
/// Uses Rc for copy-on-write semantics to avoid cloning entire context
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    pub state: &'a TlaState,
    pub locals: Rc<BTreeMap<String, TlaValue>>,
    pub local_definitions: Rc<BTreeMap<String, TlaDefinition>>,
    pub definitions: Option<&'a BTreeMap<String, TlaDefinition>>,
    pub instances: Option<&'a BTreeMap<String, crate::tla::module::TlaModuleInstance>>,
    /// Budget for set construction. Each element added to a set/sequence
    /// decrements the budget. When it reaches 0, evaluation returns an error.
    /// None = unlimited (used during model checking).
    pub eval_budget: Option<Rc<Cell<usize>>>,
}

/// Context for evaluating action constraints over two states (current -> next)
#[derive(Debug, Clone)]
pub struct TransitionContext<'a> {
    pub current_state: &'a TlaState,
    pub next_state: &'a TlaState,
    pub locals: BTreeMap<String, TlaValue>,
    pub local_definitions: BTreeMap<String, TlaDefinition>,
    pub definitions: Option<&'a BTreeMap<String, TlaDefinition>>,
}

impl<'a> EvalContext<'a> {
    pub fn new(state: &'a TlaState) -> Self {
        Self {
            state,
            locals: Rc::new(BTreeMap::new()),
            local_definitions: Rc::new(BTreeMap::new()),
            definitions: None,
            instances: None,
            eval_budget: get_active_eval_budget(),
        }
    }

    pub fn with_definitions(
        state: &'a TlaState,
        definitions: &'a BTreeMap<String, TlaDefinition>,
    ) -> Self {
        Self {
            state,
            locals: Rc::new(BTreeMap::new()),
            local_definitions: Rc::new(BTreeMap::new()),
            definitions: Some(definitions),
            instances: None,
            eval_budget: get_active_eval_budget(),
        }
    }

    pub fn with_definitions_and_instances(
        state: &'a TlaState,
        definitions: &'a BTreeMap<String, TlaDefinition>,
        instances: &'a BTreeMap<String, crate::tla::module::TlaModuleInstance>,
    ) -> Self {
        Self {
            state,
            locals: Rc::new(BTreeMap::new()),
            local_definitions: Rc::new(BTreeMap::new()),
            definitions: Some(definitions),
            instances: Some(instances),
            eval_budget: get_active_eval_budget(),
        }
    }

    pub(crate) fn with_local_value(&self, name: impl Into<String>, value: TlaValue) -> Self {
        // Copy-on-write: only clone the locals map, reuse the rest
        let mut new_locals = (*self.locals).clone();
        new_locals.insert(name.into(), value);
        Self {
            state: self.state,
            locals: Rc::new(new_locals),
            local_definitions: Rc::clone(&self.local_definitions),
            definitions: self.definitions,
            instances: self.instances,
            eval_budget: self.eval_budget.clone(),
        }
    }

    #[allow(dead_code)]
    fn with_local_values(&self, values: &[(&str, TlaValue)]) -> Self {
        // Copy-on-write: only clone the locals map, reuse the rest
        let mut new_locals = (*self.locals).clone();
        for (k, v) in values {
            new_locals.insert((*k).to_string(), v.clone());
        }
        Self {
            state: self.state,
            locals: Rc::new(new_locals),
            local_definitions: Rc::clone(&self.local_definitions),
            definitions: self.definitions,
            instances: self.instances,
            eval_budget: self.eval_budget.clone(),
        }
    }

    fn with_local_definitions(&self, defs: BTreeMap<String, TlaDefinition>) -> Self {
        // Copy-on-write: only clone the local_definitions map, reuse the rest
        let mut new_defs = (*self.local_definitions).clone();
        // A LET binding must shadow any outer binding of the same name — including
        // a dynamic `locals` entry. `local_definitions` is resolved AFTER `locals`
        // (see runtime_value / resolve_identifier), so if a LET-bound name also
        // exists in `locals` the stale local would win and the LET would silently
        // fail to shadow. This bites when an operator's body re-`LET`s a name that
        // leaked into `locals` via a callee Lambda's captured_locals: e.g. both
        // `MoveElevator` and `CanServiceCall` use `LET eState == ElevatorState[e]`,
        // and `CanServiceCall[e2, ...]` would otherwise read the caller's `eState`
        // (the wrong elevator) instead of its own. Drop shadowed names from
        // `locals` so the LET definitions take effect.
        let mut locals = Rc::clone(&self.locals);
        if defs.keys().any(|name| self.locals.contains_key(name.as_str())) {
            let owned = Rc::make_mut(&mut locals);
            for name in defs.keys() {
                owned.remove(name.as_str());
            }
        }
        for (name, def) in defs {
            new_defs.insert(name, def);
        }
        Self {
            state: self.state,
            locals,
            local_definitions: Rc::new(new_defs),
            definitions: self.definitions,
            instances: self.instances,
            eval_budget: self.eval_budget.clone(),
        }
    }

    /// Charge `cost` elements against the evaluation budget.
    /// Returns Ok(()) if no budget is set or the budget has sufficient remaining capacity.
    /// Returns an error if the budget would be exceeded.
    pub fn check_budget(&self, cost: usize) -> Result<()> {
        if let Some(ref budget) = self.eval_budget {
            let remaining = budget.get();
            if cost > remaining {
                return Err(anyhow!(
                    "evaluation budget exceeded ({} elements requested, {} remaining)",
                    cost,
                    remaining
                ));
            }
            budget.set(remaining - cost);
        }
        Ok(())
    }

    pub fn runtime_value(&self, name: &str) -> Option<TlaValue> {
        if let Some(v) = self.locals.get(name) {
            return Some(v.clone());
        }
        self.state.get(name).cloned()
    }

    pub(crate) fn definition(&self, name: &str) -> Option<TlaDefinition> {
        if let Some(def) = self.local_definitions.get(name) {
            return Some(def.clone());
        }
        self.definitions.and_then(|defs| defs.get(name).cloned())
    }

    fn eval_primed_zero_arg_definition(
        &self,
        name: &str,
        depth: usize,
    ) -> Option<Result<TlaValue>> {
        let base_name = name.strip_suffix('\'')?;
        let def = self.definition(base_name)?;
        if !def.params.is_empty() {
            return None;
        }

        let primed_ctx = self.with_primed_state_shadow_bindings();

        Some(eval_operator_call(
            base_name,
            Vec::new(),
            &primed_ctx,
            depth + 1,
        ))
    }

    fn with_primed_state_shadow_bindings(&self) -> Self {
        let mut primed_ctx = self.clone();
        let primed_bindings: Vec<(String, TlaValue)> = self
            .state
            .keys()
            .filter_map(|var| {
                self.locals
                    .get(&format!("{var}'"))
                    .cloned()
                    .map(|value| (var.to_string(), value))
            })
            .collect();
        if !primed_bindings.is_empty() {
            let locals_mut = Rc::make_mut(&mut primed_ctx.locals);
            for (var, value) in primed_bindings {
                locals_mut.insert(var, value);
            }
        }
        primed_ctx
    }

    pub(crate) fn resolve_identifier(&self, name: &str, depth: usize) -> Result<TlaValue> {
        if let Some(v) = self.runtime_value(name) {
            return Ok(v);
        }

        if let Some(result) = self.eval_primed_zero_arg_definition(name, depth) {
            return result;
        }

        if let Some(def) = self.definition(name) {
            if def.params.is_empty() {
                return eval_operator_call(name, Vec::new(), self, depth);
            }
            // A module-level operator is lexically scoped to the module and must
            // NOT capture the caller's dynamic locals (that would let a free
            // variable in its body bind to a same-named caller local). Only a
            // LET-local operator may reference an enclosing bound variable.
            let empty = BTreeMap::new();
            let captured = if self.local_definitions.contains_key(name) {
                self.locals.as_ref()
            } else {
                &empty
            };
            return Ok(definition_as_lambda(&def, captured));
        }

        // Check for known zero-arg built-in operators used as bare identifiers
        // (e.g., IOEnv, EmptyBag — TLA+ uses them without parentheses)
        match name {
            "IOEnv" | "EmptyBag" => {
                return eval_operator_call(name, Vec::new(), self, depth);
            }
            _ => {}
        }

        Ok(TlaValue::ModelValue(name.to_string()))
    }
}

impl<'a> TransitionContext<'a> {
    pub fn new(current_state: &'a TlaState, next_state: &'a TlaState) -> Self {
        Self {
            current_state,
            next_state,
            locals: BTreeMap::new(),
            local_definitions: BTreeMap::new(),
            definitions: None,
        }
    }

    pub fn with_definitions(
        current_state: &'a TlaState,
        next_state: &'a TlaState,
        definitions: &'a BTreeMap<String, TlaDefinition>,
    ) -> Self {
        Self {
            current_state,
            next_state,
            locals: BTreeMap::new(),
            local_definitions: BTreeMap::new(),
            definitions: Some(definitions),
        }
    }

    /// Resolve a variable name, handling primed (x') and unprimed (x) variables
    pub fn resolve_variable(&self, name: &str) -> Option<TlaValue> {
        // Check if it's a primed variable (x')
        if let Some(unprimed_name) = name.strip_suffix('\'') {
            // Primed variable - look in next_state
            return self.next_state.get(unprimed_name).cloned();
        }

        // Unprimed variable - check locals first, then current_state
        if let Some(v) = self.locals.get(name) {
            return Some(v.clone());
        }

        self.current_state.get(name).cloned()
    }

    pub fn definition(&self, name: &str) -> Option<TlaDefinition> {
        if let Some(def) = self.local_definitions.get(name) {
            return Some(def.clone());
        }
        self.definitions.and_then(|defs| defs.get(name).cloned())
    }
}

pub fn eval_expr(expr: &str, ctx: &EvalContext<'_>) -> Result<TlaValue> {
    eval_expr_inner(expr, ctx, 0)
}

pub(super) fn definition_as_lambda(
    def: &TlaDefinition,
    captured_locals: &BTreeMap<String, TlaValue>,
) -> TlaValue {
    TlaValue::Lambda {
        params: Arc::new(def.params.clone()),
        body: def.body.clone(),
        captured_locals: Arc::new(captured_locals.clone()),
    }
}

pub fn eval_guard(expr: &str, ctx: &EvalContext<'_>) -> Result<bool> {
    eval_expr(expr, ctx)?.as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::tla_state;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    /// Regression: a module-level operator is lexically scoped and must NOT see
    /// the caller's dynamic locals. Here `Op(y) == z + y` references the free
    /// name `z` (a module-level definition = 5); called from `\A z \in {7} : ...`
    /// the `z` bound by the `\A` must NOT capture into `Op`, so `Op(10) = 15`.
    /// Before the fix, operator application captured the caller's locals, so
    /// `z` resolved to the loop's 7 and `Op(10)` was 17. Unlike the LET-shadow
    /// case, `Op` does not rebind `z`, so only removing the capture fixes it.
    #[test]
    fn module_operator_does_not_capture_caller_locals() {
        use crate::tla::{compile_expr, eval_compiled};
        let defs = std::collections::BTreeMap::from([
            (
                "z".to_string(),
                TlaDefinition {
                    name: "z".to_string(),
                    params: vec![],
                    body: "5".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Op".to_string(),
                TlaDefinition {
                    name: "Op".to_string(),
                    params: vec!["y".to_string()],
                    body: "z + y".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let state = tla_state([]);
        let ctx = EvalContext::with_definitions(&state, &defs);
        let expr = "\\A z \\in {7} : Op(10) = 15";
        assert_eq!(eval_expr(expr, &ctx).unwrap(), TlaValue::Bool(true));
        assert_eq!(
            eval_compiled(&compile_expr(expr), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    /// Regression: a `LET` binding inside an applied operator's body must
    /// shadow a same-named variable that leaked into `locals` (here via the
    /// enclosing `\A` binder + operator application capturing caller locals).
    /// `local_definitions` (where the LET goes) is resolved AFTER `locals`, so
    /// before the fix the LET silently failed to shadow and `Inner(2)` returned
    /// the outer bound `x` (7) instead of 2. This is the mechanism behind the
    /// MultiCarElevator vs TLC state-space divergence: `CanServiceCall`'s
    /// `LET eState` vs `MoveElevator`'s leaked `eState`.
    #[test]
    fn let_binding_shadows_same_named_leaked_local() {
        use crate::tla::{compile_expr, eval_compiled};
        let defs = std::collections::BTreeMap::from([(
            "Inner".to_string(),
            TlaDefinition {
                name: "Inner".to_string(),
                params: vec!["y".to_string()],
                body: "LET x == y IN x".to_string(),
                is_recursive: false,
            },
        )]);
        let state = tla_state([]);
        let ctx = EvalContext::with_definitions(&state, &defs);
        // `\A x \in {7} : Inner(2) = 2` binds x=7 in locals; Inner(2)'s
        // `LET x == y(=2)` must shadow it, so Inner(2) = 2 and the formula holds.
        let expr = "\\A x \\in {7} : Inner(2) = 2";
        assert_eq!(eval_expr(expr, &ctx).unwrap(), TlaValue::Bool(true));
        assert_eq!(
            eval_compiled(&compile_expr(expr), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn normalizes_binder_and_higher_order_param_names() {
        assert_eq!(normalize_param_name("leader \\in Node"), "leader");
        assert_eq!(normalize_param_name("Op(_, _)"), "Op");
        assert_eq!(normalize_param_name("HostOf"), "HostOf");
    }

    #[test]
    fn evals_arithmetic_and_boolean() {
        let state = tla_state([("x", TlaValue::Int(4)), ("y", TlaValue::Int(2))]);
        let ctx = EvalContext::new(&state);
        assert_eq!(
            eval_expr("x + y * 3", &ctx).expect("expr should evaluate"),
            TlaValue::Int(10)
        );
        assert_eq!(
            eval_expr("(x = 4) /\\ (y # 3)", &ctx).expect("expr should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn evals_cup_and_cap_aliases() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        assert_eq!(
            eval_expr("{1} \\cup {2}", &ctx).expect("union alias should evaluate"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2)
            ])))
        );
        assert_eq!(
            eval_expr("{1, 2} \\cap {2, 3}", &ctx).expect("intersection alias should evaluate"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(2)])))
        );
    }

    #[test]
    fn evals_cartesian_product_times_alias() {
        // Test that \times works as an alias for \X (cartesian product)
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        // Test \X (standard syntax)
        let result_x = eval_expr("{1, 2} \\X {3}", &ctx).expect("\\X should evaluate");

        // Test \times (alternate syntax)
        let result_times = eval_expr("{1, 2} \\times {3}", &ctx).expect("\\times should evaluate");

        // Both should produce the same result
        assert_eq!(result_x, result_times);

        // Verify the result is correct: {<<1, 3>>, <<2, 3>>}
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(1), TlaValue::Int(3)])),
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(2), TlaValue::Int(3)])),
        ])));
        assert_eq!(result_times, expected);
    }

    #[test]
    fn resolves_named_operator_identifiers_as_higher_order_values() {
        let state = tla_state([(
            "waiting",
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Seq(HashedArc::new(vec![
                    TlaValue::String("read".to_string()),
                    TlaValue::Int(1),
                ])),
                TlaValue::Seq(HashedArc::new(vec![
                    TlaValue::String("write".to_string()),
                    TlaValue::Int(2),
                ])),
            ])),
        )]);
        let defs = BTreeMap::from([(
            "read".to_string(),
            TlaDefinition {
                name: "read".to_string(),
                params: vec!["s".to_string()],
                body: r#"s[1] = "read""#.to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("SelectSeq(waiting, read)", &ctx)
                .expect("higher-order operator should apply"),
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Seq(HashedArc::new(vec![
                TlaValue::String("read".to_string()),
                TlaValue::Int(1),
            ]))]))
        );
    }

    /// T1.6 regression: the interpreter must split `<=>` BEFORE `=>`.
    /// Pre-fix, `split_top_level_symbol(expr, "=>")` matched the `=>` inside
    /// `<=>`, so `(seqlock % 2 = 1) <=> resizing` evaluated against the
    /// FingerprintStoreResize initial state errored with
    /// `expected Int, got Bool(false)` (the bogus LHS `(seqlock % 2 = 1) <`
    /// fell through to a malformed `<` comparison).
    #[test]
    fn evaluates_iff_biconditional_t1_6() {
        // Mirror FingerprintStoreResize initial state: seqlock = 0, resizing = FALSE.
        let state = tla_state([
            ("seqlock", TlaValue::Int(0)),
            ("resizing", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state);

        // (0 % 2 = 1)  =>  FALSE; resizing = FALSE; FALSE <=> FALSE = TRUE.
        assert_eq!(
            eval_expr("(seqlock % 2 = 1) <=> resizing", &ctx)
                .expect("biconditional should evaluate without type-error"),
            TlaValue::Bool(true),
            "FingerprintStoreResize SeqlockConsistent invariant must hold at init"
        );

        // Sanity: TRUE <=> TRUE = TRUE, TRUE <=> FALSE = FALSE, FALSE <=> FALSE = TRUE.
        assert_eq!(
            eval_expr("TRUE <=> TRUE", &ctx).unwrap(),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("TRUE <=> FALSE", &ctx).unwrap(),
            TlaValue::Bool(false)
        );
        assert_eq!(
            eval_expr("FALSE <=> FALSE", &ctx).unwrap(),
            TlaValue::Bool(true)
        );

        // `=>` (without `<`) must still evaluate as implication.
        assert_eq!(
            eval_expr("FALSE => TRUE", &ctx).unwrap(),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("TRUE => FALSE", &ctx).unwrap(),
            TlaValue::Bool(false)
        );

        // Mixed precedence: `a => b <=> c` parses as `(a => b) <=> c`
        // (=> binds tighter than <=>). With a = FALSE, b = anything, the
        // implication is vacuously TRUE; then TRUE <=> resizing = FALSE.
        let state2 = tla_state([
            ("a", TlaValue::Bool(false)),
            ("b", TlaValue::Bool(true)),
            ("c", TlaValue::Bool(false)),
        ]);
        let ctx2 = EvalContext::new(&state2);
        assert_eq!(
            eval_expr("a => b <=> c", &ctx2).unwrap(),
            TlaValue::Bool(false),
            "(a => b) <=> c with a=FALSE,b=TRUE,c=FALSE should be (TRUE) <=> FALSE = FALSE"
        );
    }

    #[test]
    fn evaluates_user_defined_infix_operator_definitions() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            r"\prec".to_string(),
            TlaDefinition {
                name: r"\prec".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                body: "a[1] < b[1]".to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("<<1, 2>> \\prec <<2, 1>>", &ctx).expect("infix operator should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn evaluates_circ_as_sequence_concatenation_alias() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("<<1, 2>> \\circ <<3, 4>>", &ctx).expect("circ alias should evaluate"),
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
                TlaValue::Int(4),
            ]))
        );
    }

    proptest! {
        #[test]
        fn bitwise_xor_is_self_inverse(a in any::<i64>(), b in any::<i64>()) {
            let state = tla_state([
                ("a", TlaValue::Int(a)),
                ("b", TlaValue::Int(b)),
            ]);
            let ctx = EvalContext::new(&state);

            prop_assert_eq!(
                eval_expr("(a ^^ b) ^^ b", &ctx).expect("xor expression should evaluate"),
                TlaValue::Int(a)
            );
        }
    }

    #[test]
    fn evaluates_let_and_operator_calls() {
        let defs = BTreeMap::from([
            (
                "CanAct".to_string(),
                TlaDefinition {
                    name: "CanAct".to_string(),
                    params: vec!["p".to_string()],
                    body: "actionCount[p] < MaxActionsPerTick".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "MaxActionsPerTick".to_string(),
                TlaDefinition {
                    name: "MaxActionsPerTick".to_string(),
                    params: vec![],
                    body: "3".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = tla_state([(
            "actionCount",
            TlaValue::Function(HashedArc::new(BTreeMap::from([(
                TlaValue::ModelValue("bot1".to_string()),
                TlaValue::Int(1),
            )]))),
        )]);

        let ctx = EvalContext::with_definitions(&state, &defs);
        let expr = "LET x == 1 IN CanAct(bot1) /\\ x = 1";
        assert_eq!(
            eval_expr(expr, &ctx).expect("expression should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn evaluates_except_updates() {
        let state = tla_state([(
            "actionCount",
            TlaValue::Function(HashedArc::new(BTreeMap::from([(
                TlaValue::ModelValue("bot1".to_string()),
                TlaValue::Int(1),
            )]))),
        )]);
        let ctx = EvalContext::new(&state);
        let updated = eval_expr("[actionCount EXCEPT ![bot1] = @ + 1]", &ctx)
            .expect("EXCEPT update should evaluate");

        let TlaValue::Function(map) = updated else {
            panic!("expected function value");
        };
        assert_eq!(
            map.get(&TlaValue::ModelValue("bot1".to_string())),
            Some(&TlaValue::Int(2))
        );
    }

    #[test]
    fn evaluates_let_with_comment_separated_bindings_and_nested_inner_let() {
        let asset = TlaValue::ModelValue("asset1".to_string());
        let alice = TlaValue::ModelValue("alice".to_string());
        let bob = TlaValue::ModelValue("bob".to_string());
        let state = tla_state([
            (
                "Participants",
                TlaValue::Set(HashedArc::new(BTreeSet::from([alice.clone(), bob.clone()]))),
            ),
            (
                "referencePrice",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    asset.clone(),
                    TlaValue::Int(15),
                )]))),
            ),
            (
                "positions",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (alice.clone(), TlaValue::Int(1)),
                    (bob.clone(), TlaValue::Int(-1)),
                ]))),
            ),
            (
                "balances",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (alice.clone(), TlaValue::Int(100)),
                    (bob.clone(), TlaValue::Int(100)),
                ]))),
            ),
        ]);
        let future = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("asset".to_string(), asset),
            ("price".to_string(), TlaValue::Int(10)),
        ])));
        let ctx = EvalContext::new(&state).with_local_value("f", future);

        let expr = r#"LET
    priceDiff == referencePrice[f.asset] - f.price
    \* For each participant, calculate their P&L
    settlementPayments == { <<p, positions[p] * priceDiff>> : p \in Participants }
    \* Check that all participants can cover their losses
    canSettle == \A <<p, pnl>> \in settlementPayments : pnl >= 0 \/ balances[p] >= -pnl
IN
    IF canSettle
    THEN [p \in Participants |-> LET pnl == positions[p] * priceDiff IN balances[p] + pnl]
    ELSE balances"#;

        let result = eval_expr(expr, &ctx).expect("comment-separated LET should evaluate");
        let TlaValue::Function(map) = result else {
            panic!("expected function value");
        };
        assert_eq!(map.get(&alice), Some(&TlaValue::Int(105)));
        assert_eq!(map.get(&bob), Some(&TlaValue::Int(95)));
    }

    #[test]
    fn action_let_preserves_quantified_scope_in_nested_except_updates() {
        let bot = TlaValue::ModelValue("bot1".to_string());
        let seller = TlaValue::ModelValue("seller1".to_string());
        let asset = TlaValue::ModelValue("asset1".to_string());
        let state = tla_state([
            (
                "Sellers",
                TlaValue::Set(HashedArc::new(BTreeSet::from([seller.clone()]))),
            ),
            (
                "Assets",
                TlaValue::Set(HashedArc::new(BTreeSet::from([asset.clone()]))),
            ),
            ("ccpTrades", TlaValue::Set(HashedArc::new(BTreeSet::new()))),
            (
                "ccpPositions",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (
                        bot.clone(),
                        TlaValue::Function(HashedArc::new(BTreeMap::from([(
                            asset.clone(),
                            TlaValue::Int(0),
                        )]))),
                    ),
                    (
                        seller.clone(),
                        TlaValue::Function(HashedArc::new(BTreeMap::from([(
                            asset.clone(),
                            TlaValue::Int(3),
                        )]))),
                    ),
                ]))),
            ),
            ("deployments", TlaValue::Set(HashedArc::new(BTreeSet::new()))),
            (
                "actionCount",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(bot.clone(), TlaValue::Int(0))]))),
            ),
        ]);
        let ctx = EvalContext::new(&state).with_local_value("bot", bot.clone());
        let action = ActionIr {
            name: "BotNovateSpotTrade".to_string(),
            params: vec![],
            clauses: vec![ActionClause::Guard {
                expr: r#"\E s \in Sellers, aa \in Assets :
    /\ \E price \in {10} :
        /\ LET reqMargin == price
           IN
           /\ reqMargin = price
           /\ LET newTrade == [buyer |-> bot, seller |-> s, asset |-> aa, price |-> price]
                  newDeploy == [owner |-> bot, seller |-> s, asset |-> aa, price |-> price]
              IN
              /\ ccpTrades' = ccpTrades \union {newTrade}
              /\ ccpPositions' = [ccpPositions EXCEPT ![bot][aa] = @ + 1, ![s][aa] = @ - 1]
              /\ deployments' = deployments \union {newDeploy}
              /\ actionCount' = [actionCount EXCEPT ![bot] = @ + 1]"#
                    .to_string(),
            }],
        };

        let next_states = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("nested LET action should evaluate");
        assert_eq!(next_states.len(), 1);
        let next = &next_states[0];

        let TlaValue::Function(ccp_positions) = next
            .get("ccpPositions")
            .cloned()
            .expect("next state should include ccpPositions")
        else {
            panic!("expected nested function value");
        };
        let TlaValue::Function(bot_positions) = ccp_positions
            .get(&bot)
            .cloned()
            .expect("bot position should exist")
        else {
            panic!("expected bot sub-function");
        };
        let TlaValue::Function(seller_positions) = ccp_positions
            .get(&seller)
            .cloned()
            .expect("seller position should exist")
        else {
            panic!("expected seller sub-function");
        };
        assert_eq!(bot_positions.get(&asset), Some(&TlaValue::Int(1)));
        assert_eq!(seller_positions.get(&asset), Some(&TlaValue::Int(2)));

        let TlaValue::Function(action_count) = next
            .get("actionCount")
            .cloned()
            .expect("next state should include actionCount")
        else {
            panic!("expected function value");
        };
        assert_eq!(action_count.get(&bot), Some(&TlaValue::Int(1)));

        let TlaValue::Set(ccp_trades) = next
            .get("ccpTrades")
            .cloned()
            .expect("next state should include ccpTrades")
        else {
            panic!("expected set value");
        };
        let trade = ccp_trades.iter().next().expect("trade should be inserted");
        let TlaValue::Record(trade) = trade else {
            panic!("expected record trade");
        };
        assert_eq!(trade.get("buyer"), Some(&bot));
        assert_eq!(trade.get("seller"), Some(&seller));
        assert_eq!(trade.get("asset"), Some(&asset));
    }

    #[test]
    fn action_let_guard_keeps_local_sets_available_in_disjunctive_bodies() {
        let a1 = TlaValue::ModelValue("a1".to_string());
        let a2 = TlaValue::ModelValue("a2".to_string());
        let v1 = TlaValue::ModelValue("v1".to_string());
        let quorum = TlaValue::Set(HashedArc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let msg_a1 = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a1.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(1)),
            ("mval".to_string(), v1.clone()),
        ])));
        let msg_a2 = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a2.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(0)),
            ("mval".to_string(), v1.clone()),
        ])));
        let state = tla_state([
            (
                "msgs",
                TlaValue::Set(HashedArc::new(BTreeSet::from([msg_a1, msg_a2]))),
            ),
            ("Quorum", TlaValue::Set(HashedArc::new(BTreeSet::from([quorum])))),
            ("sent", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state)
            .with_local_values(&[("b", TlaValue::Int(1)), ("v", v1.clone())]);
        let action = ActionIr {
            name: "Phase2a".to_string(),
            params: vec![],
            clauses: vec![
                ActionClause::Guard {
                    expr: r#"\E Q \in Quorum :
        LET Q1b == {m \in msgs : /\ m.type = "1b"
                                 /\ m.acc \in Q
                                 /\ m.bal = b}
            Q1bv == {m \in Q1b : m.mbal >= 0}
        IN  /\ \A a \in Q : \E m \in Q1b : m.acc = a
            /\ \/ Q1bv = {}
               \/ \E m \in Q1bv :
                    /\ m.mval = v
                    /\ \A mm \in Q1bv : m.mbal >= mm.mbal"#
                        .to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "sent".to_string(),
                    expr: "TRUE".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["msgs".to_string()],
                },
            ],
        };

        let next_states = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("LET guard with local quantified set should evaluate");
        assert_eq!(next_states.len(), 1);
        let next = &next_states[0];
        assert_eq!(next.get("sent"), Some(&TlaValue::Bool(true)));
        assert_eq!(next.get("msgs"), state.get("msgs"));
    }

    #[test]
    fn disjunctive_action_guard_can_be_cleanly_disabled() {
        let a1 = TlaValue::ModelValue("a1".to_string());
        let a2 = TlaValue::ModelValue("a2".to_string());
        let v1 = TlaValue::ModelValue("v1".to_string());
        let quorum = TlaValue::Set(HashedArc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let state = tla_state([
            ("msgs", TlaValue::Set(HashedArc::new(BTreeSet::new()))),
            ("Quorum", TlaValue::Set(HashedArc::new(BTreeSet::from([quorum])))),
            ("sent", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state).with_local_values(&[("b", TlaValue::Int(0)), ("v", v1)]);
        let action = ActionIr {
            name: "Phase2a".to_string(),
            params: vec![],
            clauses: vec![
                ActionClause::Guard {
                    expr: r#"\E Q \in Quorum :
        LET Q1b == {m \in msgs : /\ m.type = "1b"
                                 /\ m.acc \in Q
                                 /\ m.bal = b}
            Q1bv == {m \in Q1b : m.mbal >= 0}
        IN  /\ \A a \in Q : \E m \in Q1b : m.acc = a
            /\ \/ Q1bv = {}
               \/ \E m \in Q1bv :
                    /\ m.mval = v
                    /\ \A mm \in Q1bv : m.mbal >= mm.mbal"#
                        .to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "sent".to_string(),
                    expr: "TRUE".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["msgs".to_string()],
                },
            ],
        };

        let next_states = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("disabled disjunctive guard should not error");
        assert!(next_states.is_empty());
    }

    #[test]
    fn compiled_action_ir_handles_phase2a_style_let_disjunctions() {
        let a1 = TlaValue::ModelValue("a1".to_string());
        let a2 = TlaValue::ModelValue("a2".to_string());
        let v1 = TlaValue::ModelValue("v1".to_string());
        let quorum = TlaValue::Set(HashedArc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let msg_a1 = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a1.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(1)),
            ("mval".to_string(), v1.clone()),
        ])));
        let msg_a2 = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a2.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(0)),
            ("mval".to_string(), v1.clone()),
        ])));
        let state = tla_state([
            (
                "msgs",
                TlaValue::Set(HashedArc::new(BTreeSet::from([msg_a1, msg_a2]))),
            ),
            ("Quorum", TlaValue::Set(HashedArc::new(BTreeSet::from([quorum])))),
            ("sent", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state).with_local_values(&[("b", TlaValue::Int(1)), ("v", v1)]);
        let def = TlaDefinition {
            name: "Phase2a".to_string(),
            params: vec![],
            body: r#"
                /\ \E Q \in Quorum :
                      LET Q1b == {m \in msgs : /\ m.type = "1b"
                                               /\ m.acc \in Q
                                               /\ m.bal = b}
                          Q1bv == {m \in Q1b : m.mbal >= 0}
                      IN  /\ \A a \in Q : \E m \in Q1b : m.acc = a
                          /\ \/ Q1bv = {}
                             \/ \E m \in Q1bv :
                                  /\ m.mval = v
                                  /\ \A mm \in Q1bv : m.mbal >= mm.mbal
                /\ sent' = TRUE
                /\ UNCHANGED <<msgs>>
            "#
            .to_string(),
            is_recursive: false,
        };
        let action = crate::tla::compile_action_ir(&def);

        let next_states = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("compiled action IR should preserve LET-local quantified sets");
        assert_eq!(next_states.len(), 1);
        let next = &next_states[0];
        assert_eq!(next.get("sent"), Some(&TlaValue::Bool(true)));
        assert_eq!(next.get("msgs"), state.get("msgs"));
    }

    #[test]
    fn evaluates_tuple_index_function_access() {
        let tuple_key = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::ModelValue("n1".to_string()),
            TlaValue::ModelValue("n2".to_string()),
        ]));
        let state = tla_state([
            (
                "NetworkPath",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    tuple_key,
                    TlaValue::Bool(true),
                )]))),
            ),
            ("src", TlaValue::ModelValue("n1".to_string())),
            ("dst", TlaValue::ModelValue("n2".to_string())),
        ]);

        let ctx = EvalContext::new(&state);
        assert_eq!(
            eval_expr("NetworkPath[src, dst]", &ctx).expect("tuple lookup should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn evaluates_tuple_index_except_updates() {
        let tuple_key = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::ModelValue("n1".to_string()),
            TlaValue::ModelValue("n2".to_string()),
        ]));
        let state = tla_state([
            (
                "NetworkPath",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    tuple_key,
                    TlaValue::Bool(true),
                )]))),
            ),
            ("src", TlaValue::ModelValue("n1".to_string())),
            ("dst", TlaValue::ModelValue("n2".to_string())),
        ]);

        let ctx = EvalContext::new(&state);
        let updated = eval_expr("[NetworkPath EXCEPT ![src, dst] = FALSE]", &ctx)
            .expect("tuple EXCEPT update should evaluate");

        let TlaValue::Function(map) = updated else {
            panic!("expected function value");
        };
        let tuple_key = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::ModelValue("n1".to_string()),
            TlaValue::ModelValue("n2".to_string()),
        ]));
        assert_eq!(map.get(&tuple_key), Some(&TlaValue::Bool(false)));
    }

    #[test]
    fn domain_accepts_indexed_function_values() {
        let state = tla_state([
            (
                "ReplicatedLog",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("n1".to_string()),
                    TlaValue::Function(HashedArc::new(BTreeMap::from([
                        (TlaValue::Int(1), TlaValue::ModelValue("a".to_string())),
                        (TlaValue::Int(2), TlaValue::ModelValue("b".to_string())),
                    ]))),
                )]))),
            ),
            ("node", TlaValue::ModelValue("n1".to_string())),
        ]);
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("DOMAIN ReplicatedLog[node]", &ctx)
                .expect("DOMAIN should bind after postfix indexing"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2)
            ])))
        );
    }

    #[test]
    fn union_prefix_accepts_set_of_sets() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("UNION {{1, 2}, {2, 3}}", &ctx).expect("UNION should evaluate"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3)
            ])))
        );
    }

    #[test]
    fn set_minus_accepts_compact_rhs_without_spaces() {
        let state = tla_state([
            (
                "Nodes",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                    TlaValue::Int(3),
                ]))),
            ),
            ("n", TlaValue::Int(2)),
        ]);
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("Nodes\\{n}", &ctx).expect("compact set minus should evaluate"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(3)
            ])))
        );
    }

    #[test]
    fn set_minus_accepts_union_prefix_rhs() {
        let state = tla_state([
            (
                "Pos",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                    TlaValue::Int(3),
                ]))),
            ),
            (
                "board",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
                    TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(3)]))),
                ]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("Pos \\ UNION board", &ctx).expect("set minus with UNION rhs should work"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(2)])))
        );
    }

    #[test]
    fn random_element_picks_a_representative_member() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("RandomElement({2, 1})", &ctx).expect("RandomElement should evaluate"),
            TlaValue::Int(1)
        );
    }

    #[test]
    fn evaluates_quantifier_and_choose() {
        let state = tla_state([(
            "S",
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))),
        )]);
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("\\A x \\in S : x > 0", &ctx).expect("forall should evaluate"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("CHOOSE x \\in S : x > 1", &ctx).expect("choose should evaluate"),
            TlaValue::Int(2)
        );
    }

    #[test]
    fn evaluates_choose_without_domain_using_stable_model_value() {
        let state = tla_state([(
            "SignedBlock",
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::ModelValue("b1".to_string()),
                TlaValue::ModelValue("b2".to_string()),
            ]))),
        )]);
        let ctx = EvalContext::new(&state);

        let value = eval_expr("CHOOSE b : b \\notin SignedBlock", &ctx)
            .expect("domainless choose should evaluate");

        assert!(matches!(value, TlaValue::ModelValue(_)));
        assert!(
            !state["SignedBlock"]
                .contains(&value)
                .expect("SignedBlock should be a set"),
            "stable choose value should not collide with the existing set"
        );

        let state_with_alias = tla_state([
            ("SignedBlock", state["SignedBlock"].clone()),
            ("remembered", value.clone()),
        ]);
        let aliased_ctx = EvalContext::new(&state_with_alias);
        let repeated = eval_expr("CHOOSE b : b \\notin SignedBlock", &aliased_ctx)
            .expect("domainless choose should stay stable");

        assert_eq!(value, repeated);
    }

    #[test]
    fn choose_without_domain_stays_equal_across_repeated_definition_evaluation() {
        let state = tla_state([
            (
                "Hash",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::ModelValue("h1".to_string()),
                    TlaValue::ModelValue("h2".to_string()),
                ]))),
            ),
            ("SignedBlock", TlaValue::Set(HashedArc::new(BTreeSet::new()))),
        ]);
        let definitions = BTreeMap::from([
            (
                "NoBlock".to_string(),
                TlaDefinition {
                    name: "NoBlock".to_string(),
                    params: vec![],
                    body: "CHOOSE b : b \\notin SignedBlock".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "hashFunction".to_string(),
                TlaDefinition {
                    name: "hashFunction".to_string(),
                    params: vec![],
                    body: "[hash \\in Hash |-> NoBlock]".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let base_ctx = EvalContext::with_definitions(&state, &definitions);
        let hash_function = eval_expr("hashFunction", &base_ctx).expect("function should evaluate");

        let state_with_function = tla_state([
            ("Hash", state["Hash"].clone()),
            ("SignedBlock", state["SignedBlock"].clone()),
            ("hashFunction", hash_function),
        ]);
        let ctx = EvalContext::with_definitions(&state_with_function, &definitions);

        let chosen = eval_expr("CHOOSE hash \\in Hash : hashFunction[hash] = NoBlock", &ctx)
            .expect("matching hash should exist");

        assert!(matches!(
            chosen,
            TlaValue::ModelValue(ref name) if name == "h1" || name == "h2"
        ));
    }

    #[test]
    fn quantifier_body_preserves_implication_rhs_conjunctions() {
        let defs = BTreeMap::from([
            (
                "CanService".to_string(),
                TlaDefinition {
                    name: "CanService".to_string(),
                    params: vec!["e".to_string(), "call".to_string()],
                    body: "TRUE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Elevator".to_string(),
                TlaDefinition {
                    name: "Elevator".to_string(),
                    params: vec![],
                    body: "{1, 2}".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let state = tla_state([(
            "Calls",
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs)
            .with_local_values(&[("e", TlaValue::Int(1))]);

        let expr = "\\A call \\in Calls : /\\ CanService[e, call] => /\\ \\E other \\in Elevator : /\\ other /= e /\\ CanService[other, call]";
        assert_eq!(
            eval_expr(expr, &ctx).expect("quantified implication should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn implication_short_circuits_false_lhs_before_rhs_record_access() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("FALSE => (NoVal.adr = 1)", &ctx)
                .expect("false implication antecedent should short-circuit"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn implication_remains_right_associative() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("FALSE => TRUE => FALSE", &ctx).expect("implication chain should evaluate"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn evaluates_primed_postfix_function_access() {
        let key = TlaValue::ModelValue("p1".to_string());
        let current =
            TlaValue::Function(HashedArc::new(BTreeMap::from([(key.clone(), TlaValue::Int(0))])));
        let next = TlaValue::Function(HashedArc::new(BTreeMap::from([(key.clone(), TlaValue::Int(1))])));
        let state = tla_state([("weight", current)]);
        let ctx = EvalContext::new(&state).with_local_values(&[("p", key), ("weight'", next)]);

        assert_eq!(
            eval_expr("weight[p]'", &ctx).expect("primed postfix function access should evaluate"),
            TlaValue::Int(1)
        );
    }

    #[test]
    fn evaluates_case_expression() {
        let state = tla_state([("d", TlaValue::Int(2))]);
        let ctx = EvalContext::new(&state);
        let expr = "CASE d = 1 -> 10 [] d = 2 -> 20 [] OTHER -> 0";
        assert_eq!(
            eval_expr(expr, &ctx).expect("CASE should evaluate"),
            TlaValue::Int(20)
        );
    }

    #[test]
    fn evaluates_lambda_expressions() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        // Test simple lambda
        let lambda_expr = "LAMBDA x: x + 1";
        let lambda = eval_expr(lambda_expr, &ctx).expect("LAMBDA should evaluate");
        match lambda {
            TlaValue::Lambda { params, body, .. } => {
                assert_eq!(*params, vec!["x".to_string()]);
                assert_eq!(body, "x + 1");
            }
            _ => panic!("Expected Lambda value"),
        }

        // Test lambda application with bracket notation
        assert_eq!(
            eval_expr("(LAMBDA x: x + 1)[5]", &ctx).expect("lambda application should work"),
            TlaValue::Int(6)
        );

        // Test lambda with boolean result
        assert_eq!(
            eval_expr("(LAMBDA x: x > 2)[5]", &ctx).expect("lambda boolean should work"),
            TlaValue::Bool(true)
        );

        // Test lambda with captured context
        let state_with_y = tla_state([("y", TlaValue::Int(10))]);
        let ctx_with_y = EvalContext::new(&state_with_y);
        assert_eq!(
            eval_expr("(LAMBDA x: x + y)[5]", &ctx_with_y)
                .expect("lambda with closure should work"),
            TlaValue::Int(15)
        );
    }

    #[test]
    fn higher_order_operator_parameters() {
        // Test operator with P(_) syntax for higher-order parameter
        // Similar to CigaretteSmokers example: ChooseOne(S, P(_))
        let state = tla_state([(
            "items",
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))),
        )]);

        let defs = BTreeMap::from([(
            "FindOne".to_string(),
            TlaDefinition {
                name: "FindOne".to_string(),
                params: vec!["S".to_string(), "P(_)".to_string()],
                body: "CHOOSE x \\in S : P(x)".to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Call FindOne with a lambda
        let expr = "FindOne(items, LAMBDA x: x > 1)";
        let result = eval_expr(expr, &ctx).expect("higher-order operator should work");

        // Result should be 2 or 3 (both satisfy x > 1)
        match result {
            TlaValue::Int(n) => assert!(n > 1, "result should be > 1, got {}", n),
            _ => panic!("Expected Int, got {:?}", result),
        }
    }

    #[test]
    fn higher_order_operator_values_preserve_tuple_binder_params() {
        let defs = BTreeMap::from([
            (
                "Sum".to_string(),
                TlaDefinition {
                    name: "Sum".to_string(),
                    params: vec!["f".to_string(), "S".to_string()],
                    body: r#"IF S = {} THEN 0
                             ELSE LET x == CHOOSE x \in S : TRUE
                                  IN f[x] + Sum(f, S \ {x})"#
                        .to_string(),
                    is_recursive: true,
                },
            ),
            (
                "Score".to_string(),
                TlaDefinition {
                    name: "Score".to_string(),
                    params: vec!["<<x, y>>".to_string()],
                    body: "x + y".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("Sum(Score, {<<1, 2>>, <<3, 4>>})", &ctx)
                .expect("higher-order tuple-binder operator should work"),
            TlaValue::Int(10)
        );
    }

    #[test]
    fn membership_prefers_runtime_value_over_shadowed_local_definition() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("LET pc == {1} IN (LAMBDA pc: 2 \\in pc)[{2}]", &ctx)
                .expect("lambda parameter should shadow local definition in membership"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn higher_order_lambda_predicate_can_reference_let_bound_value_without_recursing() {
        let piece_a = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(1), TlaValue::Int(1)])),
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(1), TlaValue::Int(2)])),
        ])));
        let piece_b = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(2), TlaValue::Int(1)])),
            TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(2), TlaValue::Int(2)])),
        ])));
        let board = TlaValue::Set(HashedArc::new(BTreeSet::from([piece_a, piece_b.clone()])));
        let state = tla_state([("board", board)]);
        let defs = BTreeMap::from([
            (
                "ChooseOne".to_string(),
                TlaDefinition {
                    name: "ChooseOne".to_string(),
                    params: vec!["S".to_string(), "P(_)".to_string()],
                    body: "CHOOSE x \\in S : P(x)".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "MovePiece".to_string(),
                TlaDefinition {
                    name: "MovePiece".to_string(),
                    params: vec!["p".to_string(), "d".to_string()],
                    body: "LET s == <<p[1] + d[1], p[2] + d[2]>>\n                  pc == ChooseOne(board, LAMBDA pc : s \\in pc)\n              IN pc".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("MovePiece(<<1, 1>>, <<1, 0>>)", &ctx)
                .expect("higher-order LET should evaluate without recursion"),
            piece_b
        );
    }

    #[test]
    fn tuple_binders_destructure_sequence_members_in_set_filters() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr(
                "LET moved == {<<{1}, {2, 3}>>, <<{4}, {5}>>} IN {<<pc, m>> \\in moved : 2 \\in m}",
                &ctx,
            )
            .expect("tuple binder filter should evaluate"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::Int(2),
                    TlaValue::Int(3)
                ]))),
            ]))])))
        );
    }

    #[test]
    fn selectseq_with_lambda_predicate() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "SelectSeq".to_string(),
            TlaDefinition {
                name: "SelectSeq".to_string(),
                params: vec!["s".to_string(), "Test".to_string()],
                body: "SelectSeq(s, Test)".to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Test SelectSeq with lambda predicate
        let expr = "SelectSeq(<<1, 2, 3, 4, 5>>, LAMBDA x: x > 2)";
        let result = eval_expr(expr, &ctx).expect("SelectSeq with lambda should work");

        assert_eq!(
            result,
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Int(3),
                TlaValue::Int(4),
                TlaValue::Int(5)
            ]))
        );

        // Test SelectSeq with different predicate
        let expr2 = "SelectSeq(<<1, 2, 3, 4, 5>>, LAMBDA x: x # 3)";
        let result2 = eval_expr(expr2, &ctx).expect("SelectSeq with lambda should work");

        assert_eq!(
            result2,
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(4),
                TlaValue::Int(5)
            ]))
        );
    }

    #[test]
    fn split_top_level_preserves_quantified_conjunction_bodies() {
        let expr = r#"
            /\ eState.direction /= "Stationary"
            /\ ~eState.doorsOpen
            /\ eState.floor \notin eState.buttonsPressed
            /\ \A call \in ActiveElevatorCalls :
                /\ CanServiceCall[e, call] =>
                    /\ \E e2 \in Elevator :
                        /\ e /= e2
                        /\ CanServiceCall[e2, call]
            /\ nextFloor \in Floor
        "#;

        let parts = split_top_level_symbol(expr, "/\\");
        assert_eq!(parts.len(), 4);
        assert!(parts[3].contains(r"\A call \in ActiveElevatorCalls"));
        assert!(parts[3].contains("CanServiceCall[e2, call]"));
        assert!(parts[3].contains("nextFloor \\in Floor"));
    }

    #[test]
    fn split_indented_top_level_boolean_keeps_nested_disjunctions_inside_conjuncts() {
        let expr = r#"
            /\ ElevatorState[e].direction = c.direction
            /\  \/ ElevatorState[e].floor = c.floor
                \/ GetDirection[ElevatorState[e].floor, c.floor] = c.direction
        "#;

        let parts = split_indented_top_level_boolean(expr, "/\\")
            .expect("indented conjunction splitter should recognize the top-level clauses");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "ElevatorState[e].direction = c.direction");
        assert!(parts[1].starts_with(r#"\/ ElevatorState[e].floor = c.floor"#));
        assert!(
            parts[1].contains(r#"GetDirection[ElevatorState[e].floor, c.floor] = c.direction"#)
        );
    }

    #[test]
    fn split_indented_top_level_boolean_keeps_quantifier_body_with_header() {
        let expr = r#"
            /\ \A e2 \in stationary \cup approaching :
                /\ GetDistance[ElevatorState[e].floor, c.floor] <= GetDistance[ElevatorState[e2].floor, c.floor]
            /\ c \in ActiveElevatorCalls
        "#;

        let parts = split_indented_top_level_boolean(expr, "/\\")
            .expect("indented conjunction splitter should recognize the top-level clauses");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].starts_with(r#"\A e2 \in stationary \cup approaching :"#));
        assert!(parts[0].contains(r#"/\ GetDistance[ElevatorState[e].floor, c.floor] <="#));
        assert_eq!(parts[1], r#"c \in ActiveElevatorCalls"#);
    }

    #[test]
    fn split_indented_top_level_boolean_keeps_quantifier_body_after_nested_let_in() {
        let expr = r#"
            /\ \E repPublicKey \in {"pub"} :
                /\ \E srcHash \in {"hash"} :
                    LET newOpenBlock == srcHash
                    IN
                    /\ newOpenBlock = "hash"
                    /\ repPublicKey = "pub"
            /\ TRUE
        "#;

        let parts = split_indented_top_level_boolean(expr, "/\\")
            .expect("indented conjunction splitter should preserve quantifier LET body");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].starts_with(r#"\E repPublicKey \in {"pub"} :"#));
        assert!(parts[0].contains(r#"LET newOpenBlock == srcHash"#));
        assert!(parts[0].contains(r#"/\ newOpenBlock = "hash""#));
        assert!(parts[0].contains(r#"/\ repPublicKey = "pub""#));
        assert_eq!(parts[1], "TRUE");
    }

    #[test]
    fn evaluates_multiline_set_filter_with_nested_disjunction() {
        let elevator_1 = TlaValue::ModelValue("e1".to_string());
        let elevator_2 = TlaValue::ModelValue("e2".to_string());
        let up = TlaValue::String("Up".to_string());
        let stationary = TlaValue::String("Stationary".to_string());

        let state = tla_state([
            (
                "Elevator",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    elevator_1.clone(),
                    elevator_2.clone(),
                ]))),
            ),
            (
                "ElevatorState",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (
                        elevator_1.clone(),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([
                            ("floor".to_string(), TlaValue::Int(1)),
                            ("direction".to_string(), up.clone()),
                        ]))),
                    ),
                    (
                        elevator_2.clone(),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([
                            ("floor".to_string(), TlaValue::Int(3)),
                            ("direction".to_string(), stationary),
                        ]))),
                    ),
                ]))),
            ),
            (
                "c",
                TlaValue::Record(HashedArc::new(BTreeMap::from([
                    ("floor".to_string(), TlaValue::Int(2)),
                    ("direction".to_string(), up),
                ]))),
            ),
        ]);
        let defs = BTreeMap::from([(
            "GetDirection".to_string(),
            TlaDefinition {
                name: "GetDirection".to_string(),
                params: vec!["from".to_string(), "to".to_string()],
                body: r#"IF from < to THEN "Up" ELSE IF from > to THEN "Down" ELSE "Stationary""#
                    .to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        let expr = r#"
            {e \in Elevator :
                /\ ElevatorState[e].direction = c.direction
                /\  \/ ElevatorState[e].floor = c.floor
                    \/ GetDirection[ElevatorState[e].floor, c.floor] = c.direction
            }
        "#;

        let value = eval_expr(expr, &ctx).expect("set filter should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([elevator_1])));
        assert_eq!(value, expected);
    }

    #[test]
    fn evals_empty_universal_quantifier_with_record_body() {
        let state = tla_state([
            (
                "ActiveElevatorCalls",
                TlaValue::Set(HashedArc::new(BTreeSet::new())),
            ),
            (
                "Elevator",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::ModelValue("e1".to_string()),
                    TlaValue::ModelValue("e2".to_string()),
                ]))),
            ),
            ("e", TlaValue::ModelValue("e1".to_string())),
            ("nextFloor", TlaValue::Int(1)),
            (
                "Floor",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
        ]);
        let defs = BTreeMap::from([(
            "CanServiceCall".to_string(),
            TlaDefinition {
                name: "CanServiceCall".to_string(),
                params: vec!["e".to_string(), "c".to_string()],
                body: r#"
                    /\ c.floor = 1
                    /\ c.direction = "Up"
                "#
                .to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        let value = eval_expr(
            r#"
                \A call \in ActiveElevatorCalls :
                    /\ CanServiceCall[e, call] =>
                        /\ \E e2 \in Elevator :
                            /\ e /= e2
                            /\ CanServiceCall[e2, call]
                    /\ nextFloor \in Floor
            "#,
            &ctx,
        )
        .expect("empty universal quantifier should evaluate");

        assert_eq!(value, TlaValue::Bool(true));
    }

    #[test]
    fn subseq_accepts_sequence_like_functions() {
        let state = tla_state([(
            "log",
            TlaValue::Function(HashedArc::new(BTreeMap::from([
                (TlaValue::Int(1), TlaValue::String("a".to_string())),
                (TlaValue::Int(2), TlaValue::String("b".to_string())),
                (TlaValue::Int(3), TlaValue::String("c".to_string())),
            ]))),
        )]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("SubSeq(log, 1, 2)", &ctx)
            .expect("SubSeq should accept sequence-like functions");

        assert_eq!(
            result,
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::String("a".to_string()),
                TlaValue::String("b".to_string()),
            ]))
        );
    }

    #[test]
    fn concat_accepts_sequence_like_functions() {
        let state = tla_state([
            (
                "lhs",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (TlaValue::Int(1), TlaValue::Int(1)),
                    (TlaValue::Int(2), TlaValue::Int(2)),
                ]))),
            ),
            (
                "rhs",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::Int(1),
                    TlaValue::Int(3),
                )]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        let result =
            eval_expr("lhs \\o rhs", &ctx).expect("function-backed sequences should concat");

        assert_eq!(
            result,
            TlaValue::Seq(HashedArc::new(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))
        );
    }

    #[test]
    fn if_with_equality_in_condition() {
        let state = tla_state([("opts", TlaValue::Set(HashedArc::new(BTreeSet::new())))]);
        let ctx = EvalContext::new(&state);

        // IF expression with equality in condition should not be split at =
        let expr = "IF opts = {} THEN 0 ELSE 1";
        let result = eval_expr(expr, &ctx).expect("IF expression should evaluate");
        assert_eq!(
            result,
            TlaValue::Int(0),
            "opts is empty set, so should return 0"
        );

        // Same test with non-empty set
        let state2 = tla_state([(
            "opts",
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
        )]);
        let ctx2 = EvalContext::new(&state2);
        let result2 = eval_expr(expr, &ctx2).expect("IF expression should evaluate");
        assert_eq!(
            result2,
            TlaValue::Int(1),
            "opts is non-empty, so should return 1"
        );
    }

    #[test]
    fn if_with_nested_let_in_else() {
        let state = tla_state([("S", TlaValue::Set(HashedArc::new(BTreeSet::new())))]);
        let ctx = EvalContext::new(&state);

        // IF with nested LET in ELSE branch
        let expr = "IF S = {} THEN 0 ELSE LET x == 1 IN x + 2";
        let result = eval_expr(expr, &ctx).expect("IF with nested LET should evaluate");
        assert_eq!(
            result,
            TlaValue::Int(0),
            "S is empty set, so should return 0"
        );

        // Same test with non-empty set to exercise the ELSE branch
        let state2 = tla_state([(
            "S",
            TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
        )]);
        let ctx2 = EvalContext::new(&state2);
        let result2 = eval_expr(expr, &ctx2).expect("IF with nested LET should evaluate");
        assert_eq!(
            result2,
            TlaValue::Int(3),
            "S is non-empty, so should evaluate LET and return 3"
        );
    }

    #[test]
    fn split_top_level_comparison_respects_if_then() {
        // split_top_level_comparison should NOT split inside IF...THEN
        let result = split_top_level_comparison("IF x = 5 THEN y ELSE z");
        assert!(
            result.is_none(),
            "split_top_level_comparison should not split inside IF condition: {:?}",
            result
        );

        // After THEN, it should be able to split
        let result = split_top_level_comparison("IF x THEN a = b ELSE c");
        assert!(
            result.is_some(),
            "split_top_level_comparison should split after THEN"
        );
        let (left, op, right) = result.unwrap();
        assert_eq!(left, "IF x THEN a");
        assert_eq!(op, "=");
        assert_eq!(right, "b ELSE c");
    }

    #[test]
    fn split_top_level_comparison_respects_let_in() {
        // split_top_level_comparison should NOT split inside LET...IN
        let result = split_top_level_comparison("LET x = 5 IN y");
        assert!(
            result.is_none(),
            "split_top_level_comparison should not split inside LET: {:?}",
            result
        );

        // After IN, it should be able to split
        let result = split_top_level_comparison("LET x IN a = b");
        assert!(
            result.is_some(),
            "split_top_level_comparison should split after IN"
        );
    }

    #[test]
    fn evaluates_cup_alias() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);

        let result = eval_expr("{1, 2} \\cup {2, 3}", &ctx).expect("\\cup should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn evaluates_recursive_factorial() {
        // Define a recursive Factorial operator
        let defs = BTreeMap::from([(
            "Factorial".to_string(),
            TlaDefinition {
                name: "Factorial".to_string(),
                params: vec!["n".to_string()],
                body: "IF n <= 1 THEN 1 ELSE n * Factorial(n - 1)".to_string(),
                is_recursive: true,
            },
        )]);

        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Test various factorial values
        assert_eq!(
            eval_expr("Factorial(0)", &ctx).expect("Factorial(0)"),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_expr("Factorial(1)", &ctx).expect("Factorial(1)"),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_expr("Factorial(2)", &ctx).expect("Factorial(2)"),
            TlaValue::Int(2)
        );
        assert_eq!(
            eval_expr("Factorial(3)", &ctx).expect("Factorial(3)"),
            TlaValue::Int(6)
        );
        assert_eq!(
            eval_expr("Factorial(4)", &ctx).expect("Factorial(4)"),
            TlaValue::Int(24)
        );
        assert_eq!(
            eval_expr("Factorial(5)", &ctx).expect("Factorial(5)"),
            TlaValue::Int(120)
        );
    }

    #[test]
    fn evaluates_recursive_sum_seq() {
        // Define a recursive SumSeq operator
        let defs = BTreeMap::from([(
            "SumSeq".to_string(),
            TlaDefinition {
                name: "SumSeq".to_string(),
                params: vec!["s".to_string()],
                body: "IF s = <<>> THEN 0 ELSE Head(s) + SumSeq(Tail(s))".to_string(),
                is_recursive: true,
            },
        )]);

        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Test summing sequences
        assert_eq!(
            eval_expr("SumSeq(<<>>)", &ctx).expect("SumSeq empty"),
            TlaValue::Int(0)
        );
        assert_eq!(
            eval_expr("SumSeq(<<1>>)", &ctx).expect("SumSeq single"),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_expr("SumSeq(<<1, 2>>)", &ctx).expect("SumSeq two"),
            TlaValue::Int(3)
        );
        assert_eq!(
            eval_expr("SumSeq(<<1, 2, 3>>)", &ctx).expect("SumSeq three"),
            TlaValue::Int(6)
        );
        assert_eq!(
            eval_expr("SumSeq(<<1, 2, 3, 4, 5>>)", &ctx).expect("SumSeq five"),
            TlaValue::Int(15)
        );
    }

    #[test]
    fn recursive_operator_respects_depth_limit() {
        // Run the recursive evaluation on a dedicated thread with an explicit
        // 8 MB stack. The eval recursion is allowed to climb to MAX_EVAL_DEPTH
        // (256) before the depth-limit error fires, and each frame on x86_64
        // Linux release-with-debug builds can exceed what fits in GitHub
        // Actions' default 2 MB thread stack. A larger fixed budget makes the
        // test self-contained on any host (T12.1).
        let handle = std::thread::Builder::new()
            .name("recursive_operator_depth_limit_test".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                // Define a recursive operator that never terminates (no base case)
                let defs = BTreeMap::from([(
                    "Forever".to_string(),
                    TlaDefinition {
                        name: "Forever".to_string(),
                        params: vec!["n".to_string()],
                        body: "Forever(n + 1)".to_string(),
                        is_recursive: true,
                    },
                )]);

                let state = TlaState::new();
                let ctx = EvalContext::with_definitions(&state, &defs);

                // This should fail with a depth limit error, not hang forever
                let result = eval_expr("Forever(0)", &ctx);
                assert!(
                    result.is_err(),
                    "Forever should fail with recursion depth limit"
                );
                let err_msg = result.unwrap_err().to_string();
                assert!(
                    err_msg.contains("recursion depth") || err_msg.contains("depth limit"),
                    "Error should mention recursion depth: {}",
                    err_msg
                );
            })
            .expect("spawn recursive-depth-limit test thread");
        handle.join().expect("recursive-depth-limit test thread");
    }

    #[test]
    fn evaluates_mutually_recursive_operators() {
        // Define mutually recursive operators: IsEven and IsOdd
        // IsEven(n) == IF n = 0 THEN TRUE ELSE IsOdd(n - 1)
        // IsOdd(n) == IF n = 0 THEN FALSE ELSE IsEven(n - 1)
        let defs = BTreeMap::from([
            (
                "IsEven".to_string(),
                TlaDefinition {
                    name: "IsEven".to_string(),
                    params: vec!["n".to_string()],
                    body: "IF n = 0 THEN TRUE ELSE IsOdd(n - 1)".to_string(),
                    is_recursive: true,
                },
            ),
            (
                "IsOdd".to_string(),
                TlaDefinition {
                    name: "IsOdd".to_string(),
                    params: vec!["n".to_string()],
                    body: "IF n = 0 THEN FALSE ELSE IsEven(n - 1)".to_string(),
                    is_recursive: true,
                },
            ),
        ]);

        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("IsEven(0)", &ctx).expect("IsEven(0)"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("IsEven(1)", &ctx).expect("IsEven(1)"),
            TlaValue::Bool(false)
        );
        assert_eq!(
            eval_expr("IsEven(2)", &ctx).expect("IsEven(2)"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("IsEven(4)", &ctx).expect("IsEven(4)"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("IsOdd(0)", &ctx).expect("IsOdd(0)"),
            TlaValue::Bool(false)
        );
        assert_eq!(
            eval_expr("IsOdd(1)", &ctx).expect("IsOdd(1)"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("IsOdd(3)", &ctx).expect("IsOdd(3)"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn conjunction_with_nested_quantifier_conjunction() {
        // This test reproduces a bug where parsing:
        //   /\ \A i \in Inodes :
        //       /\ inodeState[i].readers >= 0
        //       /\ inodeState[i].writers >= 0
        // would incorrectly split and result in "/" being parsed as an expression atom

        let state = tla_state([
            (
                "Inodes",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(HashedArc::new(BTreeMap::from([
                        ("readers".to_string(), TlaValue::Int(0)),
                        ("writers".to_string(), TlaValue::Int(0)),
                    ]))),
                )]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // Test simple nested conjunction inside quantifier
        let expr = r#"/\ \A i \in Inodes :
        /\ inodeState[i].readers >= 0
        /\ inodeState[i].writers >= 0"#;

        let result = eval_expr(expr, &ctx);
        assert!(
            result.is_ok(),
            "Nested conjunction in quantifier should not fail with: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), TlaValue::Bool(true));
    }

    #[test]
    fn two_quantifiers_conjunction() {
        // Test the CoherentIO TypeOK pattern:
        //   /\ \A i \in Inodes : ...
        //   /\ \A c \in Clients, i \in Inodes : ...

        let state = tla_state([
            (
                "Inodes",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "Clients",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
                    "c1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(HashedArc::new(BTreeMap::from([
                        ("readers".to_string(), TlaValue::Int(0)),
                        ("writers".to_string(), TlaValue::Int(0)),
                        ("dataVerifier".to_string(), TlaValue::Int(0)),
                    ]))),
                )]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Function(HashedArc::new(BTreeMap::from([(
                        TlaValue::ModelValue("c1".to_string()),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "clientCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("c1".to_string()),
                    TlaValue::Function(HashedArc::new(BTreeMap::from([(
                        TlaValue::ModelValue("i1".to_string()),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::String("NONE".to_string()),
                    TlaValue::String("READ".to_string()),
                    TlaValue::String("WRITE".to_string()),
                ]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // TypeOK pattern with two quantifiers
        let expr = r#"/\ \A i \in Inodes :
        /\ inodeState[i].readers >= 0
        /\ inodeState[i].writers >= 0
        /\ inodeState[i].dataVerifier >= 0
    /\ \A c \in Clients, i \in Inodes :
        /\ serverCharters[i][c].givenAccess \in AccessLevel
        /\ clientCharters[c][i].givenAccess \in AccessLevel"#;

        let result = eval_expr(expr, &ctx);
        assert!(
            result.is_ok(),
            "Two quantifiers conjunction should not fail with: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), TlaValue::Bool(true));
    }

    #[test]
    fn two_quantifiers_compiled() {
        use crate::tla::{compile_expr, eval_compiled};

        // Test the CoherentIO TypeOK pattern with compiled expressions

        let state = tla_state([
            (
                "Inodes",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "Clients",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
                    "c1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(HashedArc::new(BTreeMap::from([
                        ("readers".to_string(), TlaValue::Int(0)),
                        ("writers".to_string(), TlaValue::Int(0)),
                        ("dataVerifier".to_string(), TlaValue::Int(0)),
                    ]))),
                )]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Function(HashedArc::new(BTreeMap::from([(
                        TlaValue::ModelValue("c1".to_string()),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "clientCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from([(
                    TlaValue::ModelValue("c1".to_string()),
                    TlaValue::Function(HashedArc::new(BTreeMap::from([(
                        TlaValue::ModelValue("i1".to_string()),
                        TlaValue::Record(HashedArc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(HashedArc::new(BTreeSet::from([
                    TlaValue::String("NONE".to_string()),
                    TlaValue::String("READ".to_string()),
                    TlaValue::String("WRITE".to_string()),
                ]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // TypeOK pattern with two quantifiers
        let expr = r#"/\ \A i \in Inodes :
        /\ inodeState[i].readers >= 0
        /\ inodeState[i].writers >= 0
        /\ inodeState[i].dataVerifier >= 0
    /\ \A c \in Clients, i \in Inodes :
        /\ serverCharters[i][c].givenAccess \in AccessLevel
        /\ clientCharters[c][i].givenAccess \in AccessLevel"#;

        let compiled = compile_expr(expr);
        let result = eval_compiled(&compiled, &ctx);
        assert!(
            result.is_ok(),
            "Compiled two quantifiers conjunction should not fail with: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), TlaValue::Bool(true));
    }

    /// Test multi-variable quantifier with different domains
    /// Pattern: \A c \in Clients, i \in Inodes : body
    /// This should compile to nested Forall without any Unparsed nodes.
    #[test]
    fn test_multi_var_quantifier_different_domains() {
        use crate::tla::CompiledExpr;
        use crate::tla::compile_expr;

        // Multi-variable quantifier with different domains
        let expr = r#"\A c \in Clients, i \in Inodes :
    /\ serverCharters[i][c].givenAccess \in AccessLevel
    /\ clientCharters[c][i].givenAccess \in AccessLevel"#;

        let compiled = compile_expr(expr);

        // Should compile to nested Forall: \A c \in Clients : \A i \in Inodes : body
        match &compiled {
            CompiledExpr::Forall {
                var: var1,
                body: body1,
                ..
            } => {
                assert_eq!(var1, "c", "First binding should be 'c'");
                match body1.as_ref() {
                    CompiledExpr::Forall {
                        var: var2,
                        body: body2,
                        ..
                    } => {
                        assert_eq!(var2, "i", "Second binding should be 'i'");
                        // Body should be an And with 2 In expressions
                        match body2.as_ref() {
                            CompiledExpr::And(parts) => {
                                assert_eq!(parts.len(), 2, "Body should have 2 conjuncts");
                                assert!(
                                    matches!(&parts[0], CompiledExpr::In(_, _)),
                                    "First part should be In, got: {:?}",
                                    parts[0]
                                );
                                assert!(
                                    matches!(&parts[1], CompiledExpr::In(_, _)),
                                    "Second part should be In, got: {:?}",
                                    parts[1]
                                );
                            }
                            other => panic!("Body should be And, got: {:?}", other),
                        }
                    }
                    other => panic!(
                        "First Forall body should be nested Forall, got: {:?}",
                        other
                    ),
                }
            }
            other => panic!("Should compile to Forall, got: {:?}", other),
        }
    }

    #[test]
    fn test_multivar_quantifier_full_typeok_pattern() {
        use crate::tla::compile_expr;
        use crate::tla::compiled_eval::eval_compiled;

        // Exact pattern from failing TypeOK - this is the full TypeOK with
        // both single-var and multi-var quantifiers with different domains
        let state = tla_state([
            (
                "inodeState",
                TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                    (
                        TlaValue::Int(1),
                        TlaValue::Record(HashedArc::new(BTreeMap::from_iter([
                            ("readers".to_string(), TlaValue::Int(0)),
                            ("writers".to_string(), TlaValue::Int(0)),
                            ("dataVerifier".to_string(), TlaValue::Int(0)),
                        ]))),
                    ),
                    (
                        TlaValue::Int(2),
                        TlaValue::Record(HashedArc::new(BTreeMap::from_iter([
                            ("readers".to_string(), TlaValue::Int(0)),
                            ("writers".to_string(), TlaValue::Int(0)),
                            ("dataVerifier".to_string(), TlaValue::Int(0)),
                        ]))),
                    ),
                ]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                    (
                        TlaValue::Int(1),
                        TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                            (
                                TlaValue::String("a".to_string()),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Read".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::String("b".to_string()),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                    (
                        TlaValue::Int(2),
                        TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                            (
                                TlaValue::String("a".to_string()),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::String("b".to_string()),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Write".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                ]))),
            ),
            (
                "clientCharters",
                TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                    (
                        TlaValue::String("a".to_string()),
                        TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                            (
                                TlaValue::Int(1),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Read".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::Int(2),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                    (
                        TlaValue::String("b".to_string()),
                        TlaValue::Function(HashedArc::new(BTreeMap::from_iter([
                            (
                                TlaValue::Int(1),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::Int(2),
                                TlaValue::Record(HashedArc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Write".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                ]))),
            ),
            (
                "Inodes",
                TlaValue::Set(HashedArc::new(BTreeSet::from_iter([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
            (
                "Clients",
                TlaValue::Set(HashedArc::new(BTreeSet::from_iter([
                    TlaValue::String("a".to_string()),
                    TlaValue::String("b".to_string()),
                ]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(HashedArc::new(BTreeSet::from_iter([
                    TlaValue::String("Read".to_string()),
                    TlaValue::String("Write".to_string()),
                    TlaValue::String("None".to_string()),
                ]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // Full TypeOK with both single and multi-var quantifiers
        // IMPORTANT: Both quantifiers must be at the same indentation level (column 0)
        // to be recognized as siblings at the top level
        let expr = r#"/\ \A i \in Inodes :
    /\ inodeState[i].readers >= 0
    /\ inodeState[i].writers >= 0
    /\ inodeState[i].dataVerifier >= 0
/\ \A c \in Clients, i \in Inodes :
    /\ serverCharters[i][c].givenAccess \in AccessLevel
    /\ clientCharters[c][i].givenAccess \in AccessLevel"#;

        let compiled = compile_expr(expr);
        println!("Full TypeOK compiled: {:#?}", compiled);

        let result = eval_compiled(&compiled, &ctx);
        assert!(
            result.is_ok(),
            "Full TypeOK should not fail with: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), TlaValue::Bool(true));
    }

    #[test]
    fn test_funasseq_basic() {
        // FunAsSeq(f, a, b) == [i \in 1..b |-> f[a + i - 1]]
        // Create a function mapping 1->10, 2->20, 3->30
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        // FunAsSeq(f, 1, 3) should produce <<10, 20, 30>>
        let result = eval_expr("FunAsSeq(f, 1, 3)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::Int(10),
            TlaValue::Int(20),
            TlaValue::Int(30),
        ]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_offset() {
        // FunAsSeq(f, n, m) == [i \in 1..m |-> f[i]]
        // n is the domain bound (unused in eval), m is the output length
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
            (TlaValue::Int(4), TlaValue::Int(40)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        // FunAsSeq(f, 4, 2) should produce <<f[1], f[2]>> = <<10, 20>>
        let result = eval_expr("FunAsSeq(f, 4, 2)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(10), TlaValue::Int(20)]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_empty() {
        // FunAsSeq(f, 1, 0) should produce empty sequence
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([(
            TlaValue::Int(1),
            TlaValue::Int(10),
        )])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("FunAsSeq(f, 1, 0)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(HashedArc::new(vec![]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_compiled() {
        use crate::tla::{compile_expr, eval_compiled};

        // Test compiled version
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(100)),
            (TlaValue::Int(2), TlaValue::Int(200)),
            (TlaValue::Int(3), TlaValue::Int(300)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let compiled = compile_expr("FunAsSeq(f, 1, 3)");
        let result = eval_compiled(&compiled, &ctx).expect("Compiled FunAsSeq should evaluate");
        let expected = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::Int(100),
            TlaValue::Int(200),
            TlaValue::Int(300),
        ]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_with_sequence() {
        // TLA+ sequences are functions with domain 1..n
        // Test FunAsSeq on a sequence (which is a function)
        let seq = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
            TlaValue::Int(4),
        ]));

        let state = tla_state([("s", seq)]);
        let ctx = EvalContext::new(&state);

        // FunAsSeq(s, 4, 2) should produce <<s[1], s[2]>> = <<1, 2>>
        let result =
            eval_expr("FunAsSeq(s, 4, 2)", &ctx).expect("FunAsSeq should evaluate on sequence");
        let expected = TlaValue::Seq(HashedArc::new(vec![TlaValue::Int(1), TlaValue::Int(2)]));
        assert_eq!(result, expected);
    }

    #[test]
    fn evals_multiline_if_then_else() {
        // Test IF-THEN-ELSE with newlines (as would come from a parsed module)
        let state = tla_state([("condition", TlaValue::Bool(false))]);
        let ctx = EvalContext::new(&state);

        // Expression with newlines as stored in TlaDefinition body
        let expr = "IF condition\nTHEN 1\nELSE 2";
        let result = eval_expr(expr, &ctx).expect("multiline IF should evaluate");
        assert_eq!(
            result,
            TlaValue::Int(2),
            "condition is false, should return ELSE branch"
        );

        // Test with condition=true
        let state_true = tla_state([("condition", TlaValue::Bool(true))]);
        let ctx_true = EvalContext::new(&state_true);
        let result_true = eval_expr(expr, &ctx_true).expect("multiline IF should evaluate");
        assert_eq!(
            result_true,
            TlaValue::Int(1),
            "condition is true, should return THEN branch"
        );
    }

    #[test]
    fn evals_multiline_else_with_conjunction() {
        // Test ELSE branch with conjunction spanning multiple lines
        let state = tla_state([("condition", TlaValue::Bool(false))]);
        let ctx = EvalContext::new(&state);

        // Expression like DiningPhilosophers with multiline ELSE containing /\
        let expr = "IF condition\nTHEN TRUE\nELSE /\\ TRUE\n     /\\ TRUE";
        let result = eval_expr(expr, &ctx).expect("multiline IF with conjunction should evaluate");
        assert_eq!(
            result,
            TlaValue::Bool(true),
            "ELSE branch conjunction should evaluate to TRUE"
        );
    }

    #[test]
    fn evals_nested_multiline_if_then_else() {
        // Test nested IF-THEN-ELSE where outer ELSE contains another IF
        let state = tla_state([
            ("outer", TlaValue::Bool(false)),
            ("inner", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state);

        let expr = "IF outer\nTHEN 1\nELSE IF inner\n     THEN 2\n     ELSE 3";
        let result = eval_expr(expr, &ctx).expect("nested multiline IF should evaluate");
        assert_eq!(
            result,
            TlaValue::Int(3),
            "outer=false, inner=false should return 3"
        );

        // Test inner=true case
        let state_inner = tla_state([
            ("outer", TlaValue::Bool(false)),
            ("inner", TlaValue::Bool(true)),
        ]);
        let ctx_inner = EvalContext::new(&state_inner);
        let result_inner =
            eval_expr(expr, &ctx_inner).expect("nested multiline IF should evaluate");
        assert_eq!(
            result_inner,
            TlaValue::Int(2),
            "outer=false, inner=true should return 2"
        );
    }

    #[test]
    fn evals_else_starting_with_conjunction_and_nested_if() {
        // Test the exact pattern from DiningPhilosophers:
        // ELSE /\ IF nested_cond THEN ... ELSE ...
        let state = tla_state([
            ("outer", TlaValue::Bool(false)),
            ("inner", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state);

        let expr = "IF outer\nTHEN TRUE\nELSE /\\ IF inner\n        THEN FALSE\n        ELSE TRUE\n     /\\ TRUE";
        let result = eval_expr(expr, &ctx).expect("complex nested IF should evaluate");
        assert_eq!(
            result,
            TlaValue::Bool(true),
            "outer=false, inner=false: /\\ TRUE /\\ TRUE = TRUE"
        );
    }

    #[test]
    fn evals_let_body_starting_with_disjunction() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        let expr = "LET next_box == 3 IN \\/ /\\ next_box \\in {2, 3, 4}\n                         /\\ TRUE\n                      \\/ /\\ next_box \\notin {2, 3, 4}\n                         /\\ FALSE";

        let result =
            eval_expr(expr, &ctx).expect("LET body with leading disjunction should evaluate");
        assert_eq!(result, TlaValue::Bool(true));
    }

    #[test]
    fn test_range_function() {
        // Range(f) returns the set of all values in the range of f
        // For f = [x \in {1, 2, 3} |-> x * 10], Range(f) = {10, 20, 30}
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(f)", &ctx).expect("Range should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(10),
            TlaValue::Int(20),
            TlaValue::Int(30),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_sequence() {
        // Range of a sequence returns the set of all its elements
        let seq = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(2), // Duplicate to test set semantics
            TlaValue::Int(3),
        ]));

        let state = tla_state([("s", seq)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(s)", &ctx).expect("Range should evaluate on sequence");
        // Duplicates are removed since Range returns a set
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_record() {
        // Range of a record returns the set of all its field values
        let rec = TlaValue::Record(HashedArc::new(BTreeMap::from([
            ("a".to_string(), TlaValue::Int(10)),
            ("b".to_string(), TlaValue::Int(20)),
            ("c".to_string(), TlaValue::Int(10)), // Duplicate value
        ])));

        let state = tla_state([("r", rec)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(r)", &ctx).expect("Range should evaluate on record");
        // Duplicates are removed since Range returns a set
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(10),
            TlaValue::Int(20),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_empty_function() {
        // Range of an empty function is the empty set
        let func = TlaValue::Function(HashedArc::new(BTreeMap::new()));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(f)", &ctx).expect("Range should evaluate on empty function");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::new()));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_compiled() {
        use crate::tla::{compile_expr, eval_compiled};

        // Test compiled version
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(100)),
            (TlaValue::Int(2), TlaValue::Int(200)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let compiled = compile_expr("Range(f)");
        let result = eval_compiled(&compiled, &ctx).expect("Compiled Range should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(100),
            TlaValue::Int(200),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn evals_module_instance_operator_call() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        // Create a helper module with some operators
        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.definitions.insert(
            "Double".to_string(),
            TlaDefinition {
                name: "Double".to_string(),
                params: vec!["n".to_string()],
                body: "n * 2".to_string(),
                is_recursive: false,
            },
        );
        helper_module.definitions.insert(
            "AddConst".to_string(),
            TlaDefinition {
                name: "AddConst".to_string(),
                params: vec!["n".to_string()],
                body: "n + Const".to_string(), // Uses a constant from substitution
                is_recursive: false,
            },
        );

        // Create an instance with Const <- 10 substitution
        let instance = TlaModuleInstance {
            alias: "H".to_string(),
            module_name: "Helper".to_string(),
            substitutions: BTreeMap::from([("Const".to_string(), "10".to_string())]),
            is_local: false,
            module: Some(Box::new(helper_module)),
        };

        // Create instances map
        let mut instances = BTreeMap::new();
        instances.insert("H".to_string(), instance);

        // Create state and context
        let state = TlaState::new();
        let definitions = BTreeMap::new();
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances);

        // Test H!Double(5) = 10
        let result = eval_expr("H!Double(5)", &ctx).expect("H!Double(5) should evaluate");
        assert_eq!(result, TlaValue::Int(10));

        // Test H!AddConst(7) = 17 (7 + 10)
        let result = eval_expr("H!AddConst(7)", &ctx).expect("H!AddConst(7) should evaluate");
        assert_eq!(result, TlaValue::Int(17));
    }

    #[test]
    fn module_instance_substitutions_bind_primed_values() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["contents".to_string()];
        helper_module.definitions.insert(
            "NextValue".to_string(),
            TlaDefinition {
                name: "NextValue".to_string(),
                params: vec![],
                body: "contents'[1]".to_string(),
                is_recursive: false,
            },
        );

        let instance = TlaModuleInstance {
            alias: "D".to_string(),
            module_name: "Helper".to_string(),
            substitutions: BTreeMap::from([("contents".to_string(), "c1".to_string())]),
            is_local: false,
            module: Some(Box::new(helper_module)),
        };

        let instances = BTreeMap::from([("D".to_string(), instance)]);
        let current = TlaValue::Function(HashedArc::new(BTreeMap::from([(
            TlaValue::Int(1),
            TlaValue::Int(0),
        )])));
        let next = TlaValue::Function(HashedArc::new(BTreeMap::from([(
            TlaValue::Int(1),
            TlaValue::Int(3),
        )])));
        let state = tla_state([("c1", current)]);
        let definitions = BTreeMap::new();
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances)
            .with_local_values(&[("c1'", next)]);

        assert_eq!(
            eval_expr("D!NextValue", &ctx)
                .expect("instance substitution should expose primed value"),
            TlaValue::Int(3)
        );
    }

    #[test]
    fn module_instances_inherit_same_named_outer_constant_definitions() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["KeyPair".to_string()];
        helper_module.definitions.insert(
            "UseKeyPair".to_string(),
            TlaDefinition {
                name: "UseKeyPair".to_string(),
                params: vec!["priv".to_string()],
                body: "KeyPair[priv]".to_string(),
                is_recursive: false,
            },
        );

        let instance = TlaModuleInstance {
            alias: "H".to_string(),
            module_name: "Helper".to_string(),
            substitutions: BTreeMap::new(),
            is_local: false,
            module: Some(Box::new(helper_module)),
        };

        let definitions = BTreeMap::from([(
            "KeyPair".to_string(),
            TlaDefinition {
                name: "KeyPair".to_string(),
                params: vec![],
                body: "[prv1 |-> pub1]".to_string(),
                is_recursive: false,
            },
        )]);
        let instances = BTreeMap::from([("H".to_string(), instance)]);
        let state = TlaState::new();
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances);

        let result =
            eval_expr("H!UseKeyPair(prv1)", &ctx).expect("instance should inherit KeyPair");
        assert_eq!(result, TlaValue::ModelValue("pub1".to_string()));
    }

    #[test]
    fn inline_if_actions_expand_module_instance_calls_with_outer_constants() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["KeyPair".to_string()];
        helper_module.definitions.insert(
            "Next".to_string(),
            TlaDefinition {
                name: "Next".to_string(),
                params: vec![],
                body: "/\\ x' = KeyPair[prv1]".to_string(),
                is_recursive: false,
            },
        );

        let instances = BTreeMap::from([(
            "H".to_string(),
            TlaModuleInstance {
                alias: "H".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::new(),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let definitions = BTreeMap::from([(
            "KeyPair".to_string(),
            TlaDefinition {
                name: "KeyPair".to_string(),
                params: vec![],
                body: "[prv1 |-> pub1]".to_string(),
                is_recursive: false,
            },
        )]);
        let state = tla_state([("x", TlaValue::ModelValue("old".to_string()))]);
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances);
        let action = crate::tla::compile_action_ir(&TlaDefinition {
            name: "RootNext".to_string(),
            params: vec![],
            body: "IF TRUE THEN H!Next ELSE /\\ UNCHANGED x".to_string(),
            is_recursive: false,
        });

        let next = apply_action_ir_with_context_multi(&action, &state, &ctx)
            .expect("if action should evaluate through instance call");

        assert_eq!(next.len(), 1);
        assert_eq!(
            next[0].get("x"),
            Some(&TlaValue::ModelValue("pub1".to_string()))
        );
    }

    #[test]
    fn module_instances_can_use_outer_operator_ref_helpers() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["Hasher".to_string()];
        helper_module.definitions.insert(
            "UseHasher".to_string(),
            TlaDefinition {
                name: "UseHasher".to_string(),
                params: vec!["n".to_string()],
                body: "Hasher(n)".to_string(),
                is_recursive: false,
            },
        );

        let instances = BTreeMap::from([(
            "H".to_string(),
            TlaModuleInstance {
                alias: "H".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::new(),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let definitions = BTreeMap::from([
            (
                "Hasher".to_string(),
                TlaDefinition {
                    name: "Hasher".to_string(),
                    params: vec!["n".to_string()],
                    body: "HashImpl(n)".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "HashImpl".to_string(),
                TlaDefinition {
                    name: "HashImpl".to_string(),
                    params: vec!["n".to_string()],
                    body: "n * 2".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let state = TlaState::new();
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances);

        let result =
            eval_expr("H!UseHasher(5)", &ctx).expect("instance should resolve outer helper");
        assert_eq!(result, TlaValue::Int(10));
    }

    #[test]
    fn module_instances_can_use_outer_helpers_that_reference_outer_instances() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut inner_module = TlaModule::default();
        inner_module.name = "Inner".to_string();
        inner_module.definitions.insert(
            "Double".to_string(),
            TlaDefinition {
                name: "Double".to_string(),
                params: vec!["n".to_string()],
                body: "n * 2".to_string(),
                is_recursive: false,
            },
        );

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["Hasher".to_string()];
        helper_module.definitions.insert(
            "UseHasher".to_string(),
            TlaDefinition {
                name: "UseHasher".to_string(),
                params: vec!["n".to_string()],
                body: "Hasher(n)".to_string(),
                is_recursive: false,
            },
        );

        let instances = BTreeMap::from([
            (
                "N".to_string(),
                TlaModuleInstance {
                    alias: "N".to_string(),
                    module_name: "Inner".to_string(),
                    substitutions: BTreeMap::new(),
                    is_local: false,
                    module: Some(Box::new(inner_module)),
                },
            ),
            (
                "H".to_string(),
                TlaModuleInstance {
                    alias: "H".to_string(),
                    module_name: "Helper".to_string(),
                    substitutions: BTreeMap::new(),
                    is_local: false,
                    module: Some(Box::new(helper_module)),
                },
            ),
        ]);
        let definitions = BTreeMap::from([
            (
                "Hasher".to_string(),
                TlaDefinition {
                    name: "Hasher".to_string(),
                    params: vec!["n".to_string()],
                    body: "HashImpl(n)".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "HashImpl".to_string(),
                TlaDefinition {
                    name: "HashImpl".to_string(),
                    params: vec!["n".to_string()],
                    body: "N!Double(n)".to_string(),
                    is_recursive: false,
                },
            ),
        ]);
        let state = TlaState::new();
        let ctx = EvalContext::with_definitions_and_instances(&state, &definitions, &instances);

        let result = eval_expr("H!UseHasher(5)", &ctx)
            .expect("instance should retain outer instances for helper calls");
        assert_eq!(result, TlaValue::Int(10));
    }

    #[test]
    fn parsed_module_instances_can_invoke_child_operators_from_loaded_files() {
        use crate::tla::module::parse_tla_module_file;
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-instance-operator-file-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let child = tmp.join("RingBuffer.tla");
        fs::write(
            &child,
            r#"
---- MODULE RingBuffer ----
LOCAL INSTANCE Naturals
CONSTANT Size
IndexOf(sequence) == sequence % Size
====
"#,
        )
        .expect("child module should be written");

        let parent = tmp.join("MC.tla");
        fs::write(
            &parent,
            r#"
---- MODULE MC ----
EXTENDS Integers
CONSTANT Size
Buffer == INSTANCE RingBuffer
====
"#,
        )
        .expect("parent module should be written");

        let module = parse_tla_module_file(&parent).expect("parent should parse");
        let buffer = module
            .instances
            .get("Buffer")
            .expect("instance should load");
        let buffer_module = buffer
            .module
            .as_ref()
            .expect("child module should be available");
        assert!(
            buffer_module.definitions.contains_key("IndexOf"),
            "instance child should retain IndexOf"
        );

        let state = tla_state([("Size", TlaValue::Int(3))]);
        let ctx = EvalContext::with_definitions_and_instances(
            &state,
            &module.definitions,
            &module.instances,
        );
        assert_eq!(
            eval_expr("Buffer!IndexOf(4)", &ctx).expect("instance operator should evaluate"),
            TlaValue::Int(1)
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn choose_can_match_instance_defined_sentinel_values_in_functions() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut nano_module = TlaModule::default();
        nano_module.name = "Nano".to_string();
        nano_module.constants = vec!["SignedBlock".to_string()];
        nano_module.definitions.insert(
            "NoBlock".to_string(),
            TlaDefinition {
                name: "NoBlock".to_string(),
                params: vec![],
                body: "CHOOSE b : b \\notin SignedBlock".to_string(),
                is_recursive: false,
            },
        );

        let instances = BTreeMap::from([(
            "N".to_string(),
            TlaModuleInstance {
                alias: "N".to_string(),
                module_name: "Nano".to_string(),
                substitutions: BTreeMap::new(),
                is_local: false,
                module: Some(Box::new(nano_module)),
            },
        )]);

        let signed_block = TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::ModelValue(
            "signed".to_string(),
        )])));
        let seed_state = tla_state([("SignedBlock", signed_block.clone())]);
        let seed_defs = BTreeMap::new();
        let seed_ctx =
            EvalContext::with_definitions_and_instances(&seed_state, &seed_defs, &instances);
        let no_block = eval_expr("N!NoBlock", &seed_ctx).expect("sentinel should resolve");

        let hash_1 = TlaValue::ModelValue("h1".to_string());
        let hash_2 = TlaValue::ModelValue("h2".to_string());
        let state = tla_state([
            ("SignedBlock", signed_block),
            (
                "Hash",
                TlaValue::Set(HashedArc::new(BTreeSet::from([hash_1.clone(), hash_2.clone()]))),
            ),
            (
                "hashFunction",
                TlaValue::Function(HashedArc::new(BTreeMap::from([
                    (hash_1.clone(), no_block.clone()),
                    (hash_2.clone(), no_block.clone()),
                ]))),
            ),
        ]);
        let defs = BTreeMap::from([(
            "HashOf".to_string(),
            TlaDefinition {
                name: "HashOf".to_string(),
                params: vec!["block".to_string()],
                body: r#"IF \E hash \in Hash : hashFunction[hash] = block
                         THEN CHOOSE hash \in Hash : hashFunction[hash] = block
                         ELSE CHOOSE hash \in Hash : hashFunction[hash] = N!NoBlock"#
                    .to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions_and_instances(&state, &defs, &instances);

        let chosen =
            eval_expr("HashOf(targetBlock)", &ctx).expect("HashOf should choose an unused hash");
        assert!(
            chosen == hash_1 || chosen == hash_2,
            "unexpected hash {chosen:?}"
        );
    }

    #[test]
    fn test_unchanged_in_expression_context() {
        // UNCHANGED in expression/guard context should evaluate to TRUE.
        // This handles cases like: \/ Action1(self) \/ UNCHANGED vars
        // where the disjunction is evaluated as a guard expression.
        let state = TlaState::new();
        let definitions = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &definitions);

        // Single variable form
        let result = eval_expr("UNCHANGED x", &ctx).expect("UNCHANGED x should evaluate");
        assert_eq!(result, TlaValue::Bool(true));

        // Tuple form
        let result =
            eval_expr("UNCHANGED <<x, y>>", &ctx).expect("UNCHANGED <<x, y>> should evaluate");
        assert_eq!(result, TlaValue::Bool(true));

        // Inside a disjunction (the motivating use case)
        let result = eval_expr("TRUE \\/ UNCHANGED vars", &ctx)
            .expect("disjunction with UNCHANGED should evaluate");
        assert_eq!(result, TlaValue::Bool(true));

        let result = eval_expr("FALSE \\/ UNCHANGED <<x, y>>", &ctx)
            .expect("disjunction with UNCHANGED tuple should evaluate");
        assert_eq!(result, TlaValue::Bool(true));
    }

    #[test]
    fn enabled_supports_parameterized_action_calls() {
        let mut state = TlaState::new();
        state.insert(Arc::from("x"), TlaValue::Int(1));
        state.insert(Arc::from("y"), TlaValue::Int(2));

        let mut definitions = BTreeMap::new();
        definitions.insert(
            "Step".to_string(),
            TlaDefinition {
                name: "Step".to_string(),
                params: vec!["target".to_string()],
                body: "/\\ x = target /\\ x' = x /\\ UNCHANGED <<y>>".to_string(),
                is_recursive: false,
            },
        );

        let ctx = EvalContext::with_definitions(&state, &definitions);

        let enabled = eval_expr("ENABLED Step(1)", &ctx).expect("ENABLED Step(1)");
        assert_eq!(enabled, TlaValue::Bool(true));

        let disabled = eval_expr("ENABLED Step(2)", &ctx).expect("ENABLED Step(2)");
        assert_eq!(disabled, TlaValue::Bool(false));
    }

    #[test]
    fn numeric_prefixed_identifiers_evaluate_as_identifiers() {
        let state = tla_state([
            (
                "1bMessage",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(1)]))),
            ),
            (
                "2bMessage",
                TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(2)]))),
            ),
        ]);
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("1bMessage \\cup 2bMessage", &ctx).expect("numeric-prefixed names"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
            ])))
        );
    }

    #[test]
    fn max_min_and_power_builtins_work() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("Max({1, 4, 2})", &ctx).expect("Max should work"),
            TlaValue::Int(4)
        );
        assert_eq!(
            eval_expr("Min({1, 4, 2})", &ctx).expect("Min should work"),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_expr("2^5", &ctx).expect("power should work"),
            TlaValue::Int(32)
        );
        assert_eq!(
            eval_expr("2^5 - 1", &ctx).expect("power precedence should work"),
            TlaValue::Int(31)
        );
        assert_eq!(
            eval_expr("Cardinality(BoundedSeq({1, 2}, 2))", &ctx).expect("BoundedSeq should work"),
            TlaValue::Int(7)
        );
    }

    #[test]
    fn user_defined_max_can_shadow_builtin_max() {
        let defs = BTreeMap::from([(
            "Max".to_string(),
            TlaDefinition {
                name: "Max".to_string(),
                params: vec!["S".to_string()],
                body: "42".to_string(),
                is_recursive: false,
            },
        )]);
        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("Max({1, 4, 2})", &ctx).expect("user-defined Max should win"),
            TlaValue::Int(42)
        );
    }

    #[test]
    fn tlc_builtins_provide_analysis_compatibility_defaults() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("TLCGet(\"level\")", &ctx).expect("TLCGet level should work"),
            TlaValue::Int(0)
        );
        assert_eq!(
            eval_expr("TLCGet(2)", &ctx).expect("TLCGet slot should work"),
            TlaValue::Int(999)
        );
        assert_eq!(
            eval_expr("TLCSet(2, 17)", &ctx).expect("TLCSet should work"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("TLCGet(\"config\").mode = \"bfs\"", &ctx)
                .expect("TLCGet config record should work"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn zero_arg_operator_primes_use_staged_next_state_bindings() {
        let state = tla_state([("x", TlaValue::Int(1)), ("y", TlaValue::Int(2))]);
        let definitions = BTreeMap::from([
            (
                "PairSum".to_string(),
                TlaDefinition {
                    name: "PairSum".to_string(),
                    params: vec![],
                    body: "x + y".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "PairSumPositive".to_string(),
                TlaDefinition {
                    name: "PairSumPositive".to_string(),
                    params: vec![],
                    body: "PairSum > 0".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let mut ctx = EvalContext::with_definitions(&state, &definitions);
        {
            let locals_mut = Rc::make_mut(&mut ctx.locals);
            locals_mut.insert("x'".to_string(), TlaValue::Int(10));
            locals_mut.insert("y'".to_string(), TlaValue::Int(-3));
        }

        assert_eq!(
            eval_expr("PairSum'", &ctx).expect("primed zero-arg operator should resolve"),
            TlaValue::Int(7)
        );
        assert_eq!(
            eval_expr("PairSumPositive'", &ctx)
                .expect("derived primed zero-arg operator should resolve"),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn parameterized_operator_primes_use_staged_next_state_bindings() {
        let state = tla_state([("x", TlaValue::Int(1))]);
        let definitions = BTreeMap::from([(
            "neutral".to_string(),
            TlaDefinition {
                name: "neutral".to_string(),
                params: vec!["p".to_string()],
                body: "p = x".to_string(),
                is_recursive: false,
            },
        )]);

        let mut ctx = EvalContext::with_definitions(&state, &definitions);
        Rc::make_mut(&mut ctx.locals).insert("x'".to_string(), TlaValue::Int(2));

        assert_eq!(
            eval_expr("neutral(2)'", &ctx).expect("primed operator call should resolve"),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_expr("neutral(2)", &ctx).expect("unprimed operator call should resolve"),
            TlaValue::Bool(false)
        );
    }

    #[test]
    fn multiplicative_ops_return_errors_for_zero_divisors() {
        let state = TlaState::new();
        let definitions = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &definitions);

        let div_err = eval_expr("5 \\div 0", &ctx).expect_err("division by zero should error");
        assert!(div_err.to_string().contains("division by zero"));

        let mod_err = eval_expr("5 % 0", &ctx).expect_err("modulo by zero should error");
        assert!(mod_err.to_string().contains("modulo by zero"));
    }

    #[test]
    fn parses_compact_quantifier_binders_without_space_before_in() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "Proc".to_string(),
            TlaDefinition {
                name: "Proc".to_string(),
                params: vec![],
                body: "{1, 2}".to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);
        let result = eval_expr("\\A p\\in Proc : p \\in Proc", &ctx)
            .expect("compact binder should evaluate");
        assert_eq!(result, TlaValue::Bool(true));
    }

    #[test]
    fn bracket_applies_zero_arg_operator_result_as_function() {
        let state = TlaState::new();
        let defs = BTreeMap::from([(
            "Transition".to_string(),
            TlaDefinition {
                name: "Transition".to_string(),
                params: vec![],
                body: "[\"s0\" |-> [\"H\" |-> \"1\", \"T\" |-> \"2\"]]".to_string(),
                is_recursive: false,
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);
        let value = eval_expr("Transition[\"s0\"][\"H\"]", &ctx)
            .expect("zero-arg operator result should be indexable");
        assert_eq!(value, TlaValue::String("1".to_string()));
    }

    // T2.3 SOUNDNESS regression: `..` (TLA+ precedence 9-9) must bind tighter
    // than the set-op tier (8-8: `\union`, `\intersect`, `\\`) and tighter
    // than the relational/comparison tier (5-5: `\subseteq`, `=`, etc.).
    // Previously `1..3 \subseteq S` was misparsed as `1 .. (3 \subseteq S)`,
    // typically surfacing as `Err(expected Set, got Int(3))`. This silently
    // misparses any spec writing `n..m \subseteq S` (or with `\union`,
    // `\intersect`, `\\`) without paren-wrapping the range.
    //
    // These tests exercise both forms (paren-wrapped and bare) for each
    // set-op pairing. All must succeed and return the same value.
    #[test]
    fn t2_3_dotdot_binds_tighter_than_subseteq() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let paren = eval_expr("(1..3) \\subseteq {0, 0}", &ctx)
            .expect("paren-wrapped range \\subseteq set should evaluate");
        assert_eq!(paren, TlaValue::Bool(false));

        let bare = eval_expr("1..3 \\subseteq {0, 0}", &ctx)
            .expect("bare range \\subseteq set should evaluate (not type-error)");
        assert_eq!(bare, TlaValue::Bool(false));

        let true_case = eval_expr("1..3 \\subseteq {1, 2, 3, 4}", &ctx)
            .expect("range subset of larger set should be true");
        assert_eq!(true_case, TlaValue::Bool(true));
    }

    #[test]
    fn t2_3_dotdot_binds_tighter_than_union() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let paren = eval_expr("(1..3) \\union {5}", &ctx)
            .expect("paren-wrapped range \\union set should evaluate");
        let bare =
            eval_expr("1..3 \\union {5}", &ctx).expect("bare range \\union set should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
            TlaValue::Int(5),
        ])));
        assert_eq!(paren, expected);
        assert_eq!(bare, expected);
    }

    #[test]
    fn t2_3_dotdot_binds_tighter_than_intersect() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let paren = eval_expr("(1..3) \\intersect {2, 7}", &ctx)
            .expect("paren-wrapped range \\intersect set should evaluate");
        let bare = eval_expr("1..3 \\intersect {2, 7}", &ctx)
            .expect("bare range \\intersect set should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([TlaValue::Int(2)])));
        assert_eq!(paren, expected);
        assert_eq!(bare, expected);
    }

    #[test]
    fn t2_3_dotdot_binds_tighter_than_set_minus() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let paren = eval_expr("(1..5) \\ {2, 4}", &ctx)
            .expect("paren-wrapped range \\ set should evaluate");
        let bare = eval_expr("1..5 \\ {2, 4}", &ctx).expect("bare range \\ set should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(3),
            TlaValue::Int(5),
        ])));
        assert_eq!(paren, expected);
        assert_eq!(bare, expected);
    }

    #[test]
    fn t2_3_dotdot_binds_tighter_than_equality() {
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let paren = eval_expr("(1..3) = {1, 2, 3}", &ctx)
            .expect("paren-wrapped range = set should evaluate");
        let bare = eval_expr("1..3 = {1, 2, 3}", &ctx).expect("bare range = set should evaluate");
        assert_eq!(paren, TlaValue::Bool(true));
        assert_eq!(bare, TlaValue::Bool(true));
    }

    #[test]
    fn t2_3_dotdot_still_looser_than_addition() {
        // Sanity check the OTHER direction: `..` is at precedence 9, `+`/`-` at
        // 10, so `+`/`-` binds tighter. `0 .. N-1` must parse as `0 .. (N-1)`,
        // NOT `(0 .. N) - 1`. Same for `+`.
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        let v = eval_expr("0 .. 5-1", &ctx).expect("0 .. 5-1 should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(0),
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
            TlaValue::Int(4),
        ])));
        assert_eq!(v, expected);

        let v = eval_expr("1+1 .. 2+2", &ctx).expect("1+1 .. 2+2 should evaluate");
        let expected = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(2),
            TlaValue::Int(3),
            TlaValue::Int(4),
        ])));
        assert_eq!(v, expected);
    }

    #[test]
    fn t2_3_dotdot_inside_filter_set_body() {
        // The original T2.3 minimised case from the proptest harness, exercised
        // nested in a filter-set body: `({__e \in 1..4 : ((1..3 \subseteq S))
        // \/ (__e > 0)} \intersect {0, 0})`. With `S = {1, 2, 3}` the inner
        // `1..3 \subseteq S` is TRUE so the filter body is TRUE for every
        // element, yielding `{1, 2, 3, 4}`; intersect with `{0, 0}` = `{}`.
        let state = TlaState::new();
        let defs = BTreeMap::new();
        let mut state = state;
        state.insert(
            Arc::from("S"),
            TlaValue::Set(HashedArc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))),
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        let v = eval_expr(
            "({__e \\in 1..4 : ((1..3 \\subseteq S)) \\/ (__e > 0)} \\intersect {0, 0})",
            &ctx,
        )
        .expect("nested range \\subseteq inside filter body should evaluate");
        assert_eq!(v, TlaValue::Set(HashedArc::new(BTreeSet::new())));
    }

    // --- T4 mutation-kill tests -------------------------------------------
    //
    // The tests below close consequential coverage gaps surfaced by the
    // cargo-mutants audit (RELEASE_1.0.0_LOG.md, T4). Each test is annotated
    // with the line(s) of the mutant(s) it kills.

    #[test]
    fn t4_check_budget_decrements_not_increments() {
        // Kills eval.rs:191:34 (`-` -> `+` in EvalContext::check_budget).
        // The eval budget protects against exponential blowup in set
        // construction. Mutating the decrement to an increment makes the
        // budget unbounded, which would let probes / Init enumeration spend
        // unbounded CPU on bad inputs.
        use std::cell::Cell;
        use std::rc::Rc;

        let state = TlaState::new();
        let budget = Rc::new(Cell::new(10usize));
        let ctx = EvalContext {
            state: &state,
            locals: Rc::new(BTreeMap::new()),
            local_definitions: Rc::new(BTreeMap::new()),
            definitions: None,
            instances: None,
            eval_budget: Some(Rc::clone(&budget)),
        };

        // First call: spend 5, leaving 5.
        ctx.check_budget(5).expect("budget 5 of 10 should succeed");
        assert_eq!(budget.get(), 5, "budget must decrement, not increment");
        // Second: spend 5 more, leaving 0.
        ctx.check_budget(5).expect("budget 5 of 5 should succeed");
        assert_eq!(budget.get(), 0, "budget must hit zero after exact spend");
        // Third: any non-zero spend must fail.
        let err = ctx
            .check_budget(1)
            .expect_err("budget exhausted must error");
        let msg = err.to_string();
        assert!(
            msg.contains("budget") || msg.contains("Budget"),
            "exhausted-budget error must mention 'budget', got {msg}"
        );
    }

}
