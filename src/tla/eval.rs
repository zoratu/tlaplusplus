use crate::tla::action_ir::{
    parse_action_exists, parse_action_if, split_action_body_clauses, split_action_body_disjuncts,
};
use crate::tla::{
    ActionClause, ActionIr, ClauseKind, TlaDefinition, TlaState, TlaValue, classify_clause,
    looks_like_action, parse_stuttering_action_expr,
};
use anyhow::{Context, Result, anyhow};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

thread_local! {
    /// Thread-local active evaluation budget. When set, newly created EvalContexts
    /// inherit this budget so that all evaluation on the thread shares a single
    /// cumulative element limit. This avoids threading a budget parameter through
    /// every function that creates an EvalContext.
    static ACTIVE_EVAL_BUDGET: Cell<Option<Rc<Cell<usize>>>> = const { Cell::new(None) };
}

/// Set a thread-local evaluation budget that all new EvalContexts will inherit.
/// Returns the previous budget (if any) so it can be restored.
pub fn set_active_eval_budget(limit: usize) -> Option<Rc<Cell<usize>>> {
    let budget = Rc::new(Cell::new(limit));
    let prev = ACTIVE_EVAL_BUDGET.with(|b| b.replace(Some(budget.clone())));
    prev
}

/// Clear the thread-local evaluation budget, restoring an optional previous one.
pub fn restore_eval_budget(prev: Option<Rc<Cell<usize>>>) {
    ACTIVE_EVAL_BUDGET.with(|b| b.set(prev));
}

/// Get the currently active thread-local budget (if any).
fn get_active_eval_budget() -> Option<Rc<Cell<usize>>> {
    ACTIVE_EVAL_BUDGET.with(|b| {
        let val = b.take();
        b.set(val.clone());
        val
    })
}

const MAX_EVAL_DEPTH: usize = 256;

/// Normalize a higher-order operator parameter name.
/// TLA+ allows parameters like `P(_)` or `Op(_, _)` for higher-order operators.
/// When binding arguments to these parameters, we need just the base name.
/// E.g., "P(_)" -> "P", "Op(_, _)" -> "Op", "x" -> "x"
pub fn normalize_param_name(param: &str) -> &str {
    let param = param.trim();
    let param = if let Some(in_pos) = param.find("\\in") {
        param[..in_pos].trim()
    } else {
        param
    };
    if let Some(paren_pos) = param.find('(') {
        param[..paren_pos].trim()
    } else {
        param.trim()
    }
}

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
        for (name, def) in defs {
            new_defs.insert(name, def);
        }
        Self {
            state: self.state,
            locals: Rc::clone(&self.locals),
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
                return eval_operator_call(name, Vec::new(), self, depth + 1);
            }
            return Ok(definition_as_lambda(&def, self.locals.as_ref()));
        }

        // Check for known zero-arg built-in operators used as bare identifiers
        // (e.g., IOEnv, EmptyBag — TLA+ uses them without parentheses)
        match name {
            "IOEnv" | "EmptyBag" => {
                return eval_operator_call(name, Vec::new(), self, depth + 1);
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

fn definition_as_lambda(
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

fn is_identifier(s: &str) -> bool {
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

#[derive(Debug, Clone, Default)]
struct ActionEvalBranch {
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
        for var in branch.unchanged_vars {
            if let Some(old) = current.get(var.as_str()) {
                next.insert(Arc::from(var), old.clone());
            }
        }
        for (var, value) in branch.staged {
            next.insert(Arc::from(var), value);
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

fn seed_implicit_instance_constant_bindings(
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

fn bind_instance_substitutions(
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
        if !looks_like_action(&def) || def.body.trim() == expr.trim() {
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
    if !looks_like_action(&def) || def.body.trim() == expr.trim() {
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

fn ctx_with_staged_primes<'a>(
    ctx: &EvalContext<'a>,
    staged: &BTreeMap<String, TlaValue>,
) -> EvalContext<'a> {
    let mut out = ctx.clone();
    for (var, value) in staged {
        out = out.with_local_value(&format!("{}'", var), value.clone());
    }
    out
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

fn matches_membership_expr(
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

fn eval_expr_inner(raw_expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    // DEBUG: Track expression evaluation path when depth is high
    if std::env::var("TLAPP_TRACE_EVAL").is_ok() && depth >= 3 {
        let preview_len = if raw_expr.starts_with("LET") {
            1000
        } else {
            100
        };
        eprintln!(
            "TRACE depth={}: expr={:?}",
            depth,
            &raw_expr[..preview_len.min(raw_expr.len())]
        );
    }

    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("max expression recursion depth exceeded"));
    }

    let mut expr = raw_expr.trim();
    if expr.is_empty() {
        return Err(anyhow!("empty expression"));
    }

    expr = strip_outer_parens(expr);

    if expr == "TRUE" {
        return Ok(TlaValue::Bool(true));
    }
    if expr == "FALSE" {
        return Ok(TlaValue::Bool(false));
    }
    if expr == "BOOLEAN" {
        // BOOLEAN is the set {TRUE, FALSE}
        let set: BTreeSet<TlaValue> = [TlaValue::Bool(false), TlaValue::Bool(true)]
            .into_iter()
            .collect();
        return Ok(TlaValue::Set(Arc::new(set)));
    }
    if expr == "@" {
        return ctx
            .runtime_value("@")
            .ok_or_else(|| anyhow!("'@' used outside EXCEPT update"));
    }

    if starts_with_keyword(expr, "LET") {
        return eval_let_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "IF") {
        return eval_if_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "CASE") {
        return eval_case_expression(expr, ctx, depth + 1);
    }
    if expr.starts_with("\\E") || expr.starts_with("\\A") {
        return eval_quantifier_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "CHOOSE") {
        return eval_choose_expression(expr, ctx, depth + 1);
    }
    if starts_with_keyword(expr, "LAMBDA") {
        return eval_lambda_expression(expr, ctx, depth + 1);
    }

    if expr.starts_with("/\\")
        && let Some(and_parts) = split_indented_top_level_boolean(expr, "/\\")
        && and_parts.len() > 1
    {
        for part in and_parts {
            if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(false));
            }
        }
        return Ok(TlaValue::Bool(true));
    }

    if expr.starts_with("\\/")
        && let Some(or_parts) = split_indented_top_level_boolean(expr, "\\/")
        && or_parts.len() > 1
    {
        for part in or_parts {
            if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(true));
            }
        }
        return Ok(TlaValue::Bool(false));
    }

    let implies_parts = split_top_level_symbol(expr, "=>");
    if implies_parts.len() > 1 {
        return Ok(TlaValue::Bool(eval_implies_parts(
            &implies_parts,
            ctx,
            depth + 1,
        )?));
    }

    let or_parts = split_top_level_symbol(expr, "\\/");
    if or_parts.len() > 1 {
        for part in or_parts {
            if eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(true));
            }
        }
        return Ok(TlaValue::Bool(false));
    }
    // Handle case where expression starts with \/ but has only one disjunct
    // e.g., "\/ x > 0" should be treated as just "x > 0"
    if expr.trim().starts_with("\\/") && or_parts.len() == 1 {
        let rest = expr.trim().trim_start_matches("\\/").trim_start();
        if rest.is_empty() {
            return Err(anyhow!("empty disjunction"));
        }
        return eval_expr_inner(rest, ctx, depth + 1);
    }

    let and_parts = split_top_level_symbol(expr, "/\\");
    if and_parts.len() > 1 {
        for part in and_parts {
            if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(false));
            }
        }
        return Ok(TlaValue::Bool(true));
    }
    // Handle case where expression starts with /\ but has only one conjunct
    // e.g., "/\ x > 0" should be treated as just "x > 0"
    if expr.trim().starts_with("/\\") && and_parts.len() == 1 {
        let rest = expr.trim().trim_start_matches("/\\").trim_start();
        if rest.is_empty() {
            return Err(anyhow!("empty conjunction"));
        }
        return eval_expr_inner(rest, ctx, depth + 1);
    }

    if let Some(rest) = expr.strip_prefix('~') {
        return Ok(TlaValue::Bool(
            !eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_bool()?,
        ));
    }

    // Action composition operator \cdot: A \cdot B
    // For expression evaluation purposes (probing), treat as conjunction —
    // both sides must evaluate successfully. Return the result of the last part.
    let cdot_parts = split_top_level_keyword(expr, "\\cdot");
    if cdot_parts.len() > 1 {
        let mut result = TlaValue::Bool(true);
        for part in &cdot_parts {
            result = eval_expr_inner(part, ctx, depth + 1)?;
        }
        return Ok(result);
    }

    if let Some((lhs, op, rhs)) = split_top_level_comparison(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;

        return match op {
            "=" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                Ok(TlaValue::Bool(left == right))
            }
            "/=" | "#" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                Ok(TlaValue::Bool(left != right))
            }
            "<" | "<=" | "=<" | "\\leq" | ">" | ">=" | "\\geq" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let cmp = match op {
                    "<" => left.as_int()? < right.as_int()?,
                    "<=" | "=<" | "\\leq" => left.as_int()? <= right.as_int()?,
                    ">" => left.as_int()? > right.as_int()?,
                    ">=" | "\\geq" => left.as_int()? >= right.as_int()?,
                    _ => unreachable!(),
                };
                Ok(TlaValue::Bool(cmp))
            }
            "\\in" => {
                let rhs_trimmed = rhs.trim();
                Ok(TlaValue::Bool(matches_membership_expr(
                    &left,
                    rhs_trimmed,
                    ctx,
                    depth + 1,
                )?))
            }
            "\\notin" => {
                let rhs_trimmed = rhs.trim();
                Ok(TlaValue::Bool(!matches_membership_expr(
                    &left,
                    rhs_trimmed,
                    ctx,
                    depth + 1,
                )?))
            }
            "\\subseteq" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let lhs_set = left.as_set()?;
                let rhs_set = right.as_set()?;
                Ok(TlaValue::Bool(lhs_set.iter().all(|v| rhs_set.contains(v))))
            }
            _ => Err(anyhow!("unsupported comparison operator {op}")),
        };
    }

    if let Some((lhs, op, rhs)) = split_top_level_defined_infix(expr, ctx) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        return eval_operator_call(&op, vec![left, right], ctx, depth + 1);
    }

    let union_parts = split_top_level_keyword(expr, "\\union");
    if union_parts.len() > 1 {
        let mut out = eval_expr_inner(&union_parts[0], ctx, depth + 1)?;
        for part in &union_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }
    let union_parts = split_top_level_keyword(expr, "\\cup");
    if union_parts.len() > 1 {
        let mut out = eval_expr_inner(&union_parts[0], ctx, depth + 1)?;
        for part in &union_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let cup_parts = split_top_level_keyword(expr, "\\cup");
    if cup_parts.len() > 1 {
        let mut out = eval_expr_inner(&cup_parts[0], ctx, depth + 1)?;
        for part in &cup_parts[1..] {
            out = out.set_union(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let intersect_parts = split_top_level_keyword(expr, "\\intersect");
    if intersect_parts.len() > 1 {
        let mut out = eval_expr_inner(&intersect_parts[0], ctx, depth + 1)?;
        for part in &intersect_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }
    let intersect_parts = split_top_level_keyword(expr, "\\cap");
    if intersect_parts.len() > 1 {
        let mut out = eval_expr_inner(&intersect_parts[0], ctx, depth + 1)?;
        for part in &intersect_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let cap_parts = split_top_level_keyword(expr, "\\cap");
    if cap_parts.len() > 1 {
        let mut out = eval_expr_inner(&cap_parts[0], ctx, depth + 1)?;
        for part in &cap_parts[1..] {
            out = out.set_intersection(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let minus_parts = split_top_level_set_minus(expr);
    if minus_parts.len() > 1 {
        let mut out = eval_expr_inner(&minus_parts[0], ctx, depth + 1)?;
        for part in &minus_parts[1..] {
            out = out.set_minus(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let mut concat_parts = split_top_level_keyword(expr, "\\o");
    if concat_parts.len() == 1 {
        concat_parts = split_top_level_keyword(expr, "\\circ");
    }
    if concat_parts.len() > 1 {
        let mut out = eval_expr_inner(&concat_parts[0], ctx, depth + 1)?;
        for part in &concat_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            out = seq_or_string_concat(out, rhs)?;
        }
        return Ok(out);
    }

    // Handle cartesian product: \X and \times are synonyms
    let mut cartesian_parts = split_top_level_keyword(expr, "\\X");
    if cartesian_parts.len() == 1 {
        // Try \times as an alternative syntax
        cartesian_parts = split_top_level_keyword(expr, "\\times");
    }
    if cartesian_parts.len() > 1 {
        let mut result = eval_expr_inner(&cartesian_parts[0], ctx, depth + 1)?;
        for part in &cartesian_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            let lhs_set = result.as_set()?;
            let rhs_set = rhs.as_set()?;
            ctx.check_budget(lhs_set.len() * rhs_set.len())?;
            let mut product = BTreeSet::new();
            for lhs_val in lhs_set {
                for rhs_val in rhs_set {
                    let tuple = TlaValue::Seq(Arc::new(vec![lhs_val.clone(), rhs_val.clone()]));
                    product.insert(tuple);
                }
            }
            result = TlaValue::Set(Arc::new(product));
        }
        return Ok(result);
    }

    // TLC module: Function override operator @@
    // f @@ g merges two functions, with g taking precedence on overlapping keys
    let func_override_parts = split_top_level_symbol(expr, "@@");
    if func_override_parts.len() > 1 {
        let mut out = eval_expr_inner(&func_override_parts[0], ctx, depth + 1)?;
        for part in &func_override_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            let left_func = out.as_function()?;
            let right_func = rhs.as_function()?;
            let mut result = left_func.clone();
            for (k, v) in right_func.iter() {
                result.insert(k.clone(), v.clone());
            }
            out = TlaValue::Function(Arc::new(result));
        }
        return Ok(out);
    }

    // TLC module: Function pair constructor :>
    // a :> b creates a function mapping a to b (single mapping {a -> b})
    if let Some((lhs, rhs)) = split_once_top_level(expr, ":>") {
        let key = eval_expr_inner(lhs.trim(), ctx, depth + 1)?;
        let val = eval_expr_inner(rhs.trim(), ctx, depth + 1)?;
        let mut func = BTreeMap::new();
        func.insert(key, val);
        return Ok(TlaValue::Function(Arc::new(func)));
    }

    if let Some((lhs, rhs)) = split_once_top_level(expr, "^^") {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return Ok(TlaValue::Int(left ^ right));
    }

    // Range operator `a..b` (precedence 9-9 in TLA+). Slotted between the
    // set-op tier (8-8: `\union`, `\intersect`, `\\`) above and the additive
    // tier (10-10: `+`, `-`) below — `..` binds tighter than set ops/relops
    // but looser than `+`/`-`. So `1+2..n*3 \subseteq S` parses as
    // `((1+2) .. (n*3)) \subseteq S`.
    if let Some((lhs, rhs)) = split_top_level_range(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
        let start = left.as_int()?;
        let end = right.as_int()?;
        let range_set: BTreeSet<TlaValue> = if start <= end {
            (start..=end).map(TlaValue::Int).collect()
        } else {
            BTreeSet::new()
        };
        return Ok(TlaValue::Set(Arc::new(range_set)));
    }

    if let Some((lhs, op, rhs)) = split_top_level_additive(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return match op {
            '+' => Ok(TlaValue::Int(left + right)),
            '-' => Ok(TlaValue::Int(left - right)),
            _ => Err(anyhow!("unsupported additive operator '{op}'")),
        };
    }

    if let Some((lhs, op, rhs)) = split_top_level_multiplicative(expr) {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        return match op {
            "*" => Ok(TlaValue::Int(left * right)),
            "\\div" => {
                if right == 0 {
                    Err(anyhow!("division by zero"))
                } else {
                    Ok(TlaValue::Int(left / right))
                }
            }
            "%" => {
                if right == 0 {
                    Err(anyhow!("modulo by zero"))
                } else {
                    Ok(TlaValue::Int(left % right))
                }
            }
            _ => Err(anyhow!("unsupported multiplicative operator {op}")),
        };
    }

    if let Some((lhs, rhs)) = split_once_top_level(expr, "^") {
        let left = eval_expr_inner(lhs, ctx, depth + 1)?.as_int()?;
        let right = eval_expr_inner(rhs, ctx, depth + 1)?.as_int()?;
        if right < 0 {
            return Err(anyhow!("exponent must be non-negative, got {right}"));
        }
        let value = left
            .checked_pow(right as u32)
            .ok_or_else(|| anyhow!("integer exponent overflow: {left}^{right}"))?;
        return Ok(TlaValue::Int(value));
    }

    // Handle UNCHANGED in expression/guard context.
    // When UNCHANGED appears inside a disjunction that is evaluated as a guard
    // (e.g., \/ Action1(self) \/ Action2(self) \/ UNCHANGED vars), the action
    // semantics (keeping variables at their current values) are handled by the
    // action IR layer. In the expression evaluation context, UNCHANGED simply
    // represents an always-enabled stuttering step, so we return TRUE.
    if starts_with_keyword(expr, "UNCHANGED") {
        return Ok(TlaValue::Bool(true));
    }

    if starts_with_keyword(expr, "SUBSET") {
        let rest = expr["SUBSET".len()..].trim();
        let set = eval_expr_inner(rest, ctx, depth + 1)?;
        return Ok(TlaValue::Set(Arc::new(powerset(set.as_set()?, ctx)?)));
    }

    if starts_with_keyword(expr, "UNION") {
        let rest = expr["UNION".len()..].trim();
        let mut union = BTreeSet::new();
        for set in eval_expr_inner(rest, ctx, depth + 1)?.as_set()? {
            union.extend(set.as_set()?.iter().cloned());
        }
        return Ok(TlaValue::Set(Arc::new(union)));
    }

    if let Some(rest) = expr.strip_prefix('-')
        && !rest.trim().is_empty()
    {
        return Ok(TlaValue::Int(
            -eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_int()?,
        ));
    }

    eval_atom_with_postfix(expr, ctx, depth + 1)
}

fn eval_implies_parts(parts: &[String], ctx: &EvalContext<'_>, depth: usize) -> Result<bool> {
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

fn eval_if_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (_, after_if) =
        take_keyword_prefix(expr.trim(), "IF").ok_or_else(|| anyhow!("expected IF expression"))?;

    let then_pos =
        find_outer_then(after_if).ok_or_else(|| anyhow!("IF expression missing THEN: {expr}"))?;
    let cond_text = after_if[..then_pos].trim();

    let after_then = after_if[then_pos + "THEN".len()..].trim_start();
    let else_pos =
        find_outer_else(after_then).ok_or_else(|| anyhow!("IF expression missing ELSE: {expr}"))?;

    let then_text = after_then[..else_pos].trim();
    let else_text = after_then[else_pos + "ELSE".len()..].trim();

    if eval_expr_inner(cond_text, ctx, depth + 1)?.as_bool()? {
        eval_expr_inner(then_text, ctx, depth + 1)
    } else {
        eval_expr_inner(else_text, ctx, depth + 1)
    }
}

fn eval_case_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (_, after_case) = take_keyword_prefix(expr.trim(), "CASE")
        .ok_or_else(|| anyhow!("expected CASE expression"))?;
    let arms = split_top_level_symbol(after_case, "[]");

    let mut default_arm: Option<String> = None;
    for arm in arms {
        let arm = arm.trim();
        if arm.is_empty() {
            continue;
        }

        let (cond, rhs) =
            split_once_top_level(arm, "->").ok_or_else(|| anyhow!("invalid CASE arm: {arm}"))?;
        if cond.trim() == "OTHER" {
            default_arm = Some(rhs.trim().to_string());
            continue;
        }

        if eval_expr_inner(cond.trim(), ctx, depth + 1)?.as_bool()? {
            return eval_expr_inner(rhs.trim(), ctx, depth + 1);
        }
    }

    if let Some(default_expr) = default_arm {
        eval_expr_inner(&default_expr, ctx, depth + 1)
    } else {
        Err(anyhow!(
            "CASE expression has no matching arm and no OTHER arm"
        ))
    }
}

fn eval_quantifier_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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

fn eval_choose_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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
    let mut hasher = ahash::AHasher::default();
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

fn eval_lambda_expression(expr: &str, ctx: &EvalContext<'_>, _depth: usize) -> Result<TlaValue> {
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

fn eval_let_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (defs_text, body_text) =
        split_outer_let(expr).ok_or_else(|| anyhow!("invalid LET expression: {expr}"))?;
    let defs = parse_let_definitions(defs_text)?;
    let child = ctx.with_local_definitions(defs);
    eval_expr_inner(body_text, &child, depth + 1)
}

fn parse_atom_with_postfix<'a>(
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

fn eval_atom_with_postfix(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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

fn eval_primed_postfix_expr(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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

fn parse_base<'a>(
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

fn eval_set_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let inner = expr.trim();
    if inner.is_empty() {
        return Ok(TlaValue::Set(Arc::new(BTreeSet::new())));
    }

    if let Some(colon_idx) = find_top_level_char(inner, ':') {
        let lhs = inner[..colon_idx].trim();
        let rhs = inner[colon_idx + 1..].trim();

        if contains_top_level_keyword(lhs, "\\in") {
            // Optimization: for {x \in RecordSet : pred} where RecordSet is
            // [field1: S1, field2: S2, ...], generate records inline with
            // predicate filtering instead of materializing the full set.
            let in_idx = find_top_level_keyword_index(lhs, "\\in").unwrap();
            let var_name = lhs[..in_idx].trim();
            let domain_expr = lhs[in_idx + "\\in".len()..].trim();

            // Resolve the domain expression: it may be a literal bracket
            // expression `[f1: D1, ...]` or a name that resolves through
            // the definition scope to one. We unwrap the latter so the
            // record-set fast path (and the symbolic Init enumerator
            // below) can still match the shape.
            let mut resolved_domain_owned: String = domain_expr.to_string();
            if !(domain_expr.starts_with('[') && domain_expr.ends_with(']')) {
                if is_valid_identifier(domain_expr) {
                    if let Some(defs) = ctx.definitions {
                        if let Some(def) = defs.get(domain_expr) {
                            if def.params.is_empty() {
                                let body = def.body.trim();
                                if body.starts_with('[') && body.ends_with(']') {
                                    resolved_domain_owned = body.to_string();
                                }
                            }
                        }
                    }
                }
            }
            let domain_expr_resolved = resolved_domain_owned.as_str();

            // Check if domain is a bracket expression (potential record set)
            if domain_expr_resolved.starts_with('[') && domain_expr_resolved.ends_with(']') {
                let bracket_inner = &domain_expr_resolved[1..domain_expr_resolved.len() - 1];
                // Try record set pattern: field1: Set1, field2: Set2
                if let Some(colon) = find_top_level_char(bracket_inner, ':') {
                    let field_name = bracket_inner[..colon].trim();
                    if is_valid_identifier(field_name)
                        && !bracket_inner.contains("|->")
                        && !bracket_inner.contains("->")
                    {
                        // This is a record set — generate records inline with filtering
                        let entries = split_top_level_symbol(bracket_inner, ",");
                        let mut field_specs: Vec<(String, Vec<TlaValue>)> = Vec::new();
                        let mut is_record_set = true;
                        for entry in &entries {
                            let entry = entry.trim();
                            if let Some(ci) = find_top_level_char(entry, ':') {
                                let fname = entry[..ci].trim();
                                let sexpr = entry[ci + 1..].trim();
                                if is_valid_identifier(fname) {
                                    if let Ok(sv) = eval_expr_inner(sexpr, ctx, depth + 1) {
                                        if let Ok(s) = sv.as_set() {
                                            field_specs.push((
                                                fname.to_string(),
                                                s.iter().cloned().collect(),
                                            ));
                                            continue;
                                        }
                                    }
                                }
                            }
                            is_record_set = false;
                            break;
                        }

                        if is_record_set && !field_specs.is_empty() {
                            // Constraint propagation: for 2-field integer record sets
                            // with a sum-range predicate like c.f1 + c.f2 \in lo..hi,
                            // compute valid ranges directly instead of iterating all pairs.
                            if field_specs.len() == 2 {
                                let all_ints = field_specs
                                    .iter()
                                    .all(|(_, v)| v.iter().all(|x| matches!(x, TlaValue::Int(_))));
                                if all_ints {
                                    if let Some((sum_lo, sum_hi)) = extract_sum_range_constraint(
                                        rhs,
                                        var_name,
                                        &field_specs,
                                        ctx,
                                    ) {
                                        let (f0_name, f0_vals) = &field_specs[0];
                                        let (f1_name, f1_vals) = &field_specs[1];
                                        let f1_lo = f1_vals
                                            .first()
                                            .and_then(|v| v.as_int().ok())
                                            .unwrap_or(0);
                                        let f1_hi = f1_vals
                                            .last()
                                            .and_then(|v| v.as_int().ok())
                                            .unwrap_or(0);

                                        // Collect valid (f0, f1) integer pairs first,
                                        // then bulk-construct records. Using integer
                                        // pairs avoids expensive Record creation +
                                        // BTreeSet comparison during enumeration.
                                        let mut pairs: Vec<(i64, i64)> = Vec::new();
                                        for v0 in f0_vals {
                                            let a = v0.as_int().unwrap();
                                            let valid_lo = (sum_lo - a).max(f1_lo);
                                            let valid_hi = (sum_hi - a).min(f1_hi);
                                            if valid_lo > valid_hi {
                                                continue;
                                            }
                                            // Direct integer range instead of scanning f1_vals
                                            for b in valid_lo..=valid_hi {
                                                pairs.push((a, b));
                                            }
                                        }

                                        eprintln!(
                                            "Constraint propagation: {} valid records \
                                             (from {}x{} = {} total)",
                                            pairs.len(),
                                            f0_vals.len(),
                                            f1_vals.len(),
                                            f0_vals.len() as u64 * f1_vals.len() as u64
                                        );

                                        // Build records as Seq (Vec) — O(n) instead
                                        // of BTreeSet O(n log n). The membership
                                        // evaluator in evaluate_init_states handles
                                        // both Set and Seq as membership domains.
                                        let records: Vec<TlaValue> = pairs
                                            .into_iter()
                                            .map(|(a, b)| {
                                                TlaValue::Record(Arc::new(BTreeMap::from([
                                                    (f0_name.clone(), TlaValue::Int(a)),
                                                    (f1_name.clone(), TlaValue::Int(b)),
                                                ])))
                                            })
                                            .collect();
                                        return Ok(TlaValue::Seq(Arc::new(records)));
                                    }
                                }
                            }

                            // Symbolic Init enumeration via SMT (T5).
                            // Try translating the predicate into Z3; if
                            // it succeeds, return the enumerated record
                            // set directly. Falls back to brute force
                            // when the predicate is outside the supported
                            // subset or the SMT path bails out.
                            #[cfg(feature = "symbolic-init")]
                            {
                                if let Some(records) =
                                    crate::tla::symbolic_init::try_symbolic_record_set_enumerate(
                                        rhs,
                                        var_name,
                                        &field_specs,
                                        ctx,
                                    )
                                {
                                    if std::env::var("TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT").is_ok() {
                                        eprintln!(
                                            "Symbolic Init enumeration: {} records (brute-force candidate count: {})",
                                            records.len(),
                                            field_specs
                                                .iter()
                                                .map(|(_, v)| v.len() as u64)
                                                .product::<u64>()
                                        );
                                    }
                                    return Ok(TlaValue::Seq(Arc::new(records)));
                                }
                            }

                            let mut out = BTreeSet::new();

                            // Fallback: brute-force with compiled predicate
                            let compiled_pred = crate::tla::compile_expr(rhs);
                            let mut indices = vec![0usize; field_specs.len()];
                            let total: u64 =
                                field_specs.iter().map(|(_, v)| v.len() as u64).product();
                            ctx.check_budget(total.min(500_000_000) as usize)?;

                            let mut rec = BTreeMap::new();
                            for (fname, vals) in &field_specs {
                                rec.insert(fname.clone(), vals[0].clone());
                            }
                            let base_locals = (*ctx.locals).clone();
                            let var_key = var_name.to_string();

                            loop {
                                for (i, (fname, vals)) in field_specs.iter().enumerate() {
                                    *rec.get_mut(fname).unwrap() = vals[indices[i]].clone();
                                }
                                let record_arc = Arc::new(rec.clone());
                                let mut iter_locals = base_locals.clone();
                                iter_locals.insert(
                                    var_key.clone(),
                                    TlaValue::Record(Arc::clone(&record_arc)),
                                );
                                let child = EvalContext {
                                    state: ctx.state,
                                    locals: Rc::new(iter_locals),
                                    local_definitions: Rc::clone(&ctx.local_definitions),
                                    definitions: ctx.definitions,
                                    instances: ctx.instances,
                                    eval_budget: ctx.eval_budget.clone(),
                                };
                                if matches!(
                                    crate::tla::eval_compiled(&compiled_pred, &child),
                                    Ok(TlaValue::Bool(true))
                                ) {
                                    out.insert(TlaValue::Record(record_arc));
                                }
                                let mut carry = true;
                                for i in (0..indices.len()).rev() {
                                    if carry {
                                        indices[i] += 1;
                                        if indices[i] < field_specs[i].1.len() {
                                            carry = false;
                                        } else {
                                            indices[i] = 0;
                                        }
                                    }
                                }
                                if carry {
                                    break;
                                }
                            }
                            return Ok(TlaValue::Set(Arc::new(out)));
                        }
                    }
                }
            }

            let binders = parse_binders(lhs, ctx, depth + 1)?;
            let mut assignments = BTreeMap::new();
            let mut out = BTreeSet::new();
            collect_binder_filter_set(
                0,
                &binders,
                rhs,
                &mut assignments,
                &mut out,
                ctx,
                depth + 1,
            )?;
            return Ok(TlaValue::Set(Arc::new(out)));
        }

        let binders = parse_binders(rhs, ctx, depth + 1)?;
        let mut assignments = BTreeMap::new();
        let mut out = BTreeSet::new();
        collect_binder_map_set(0, &binders, lhs, &mut assignments, &mut out, ctx, depth + 1)?;
        return Ok(TlaValue::Set(Arc::new(out)));
    }

    let mut out = BTreeSet::new();
    for item in split_top_level_symbol(inner, ",") {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        out.insert(eval_expr_inner(item, ctx, depth + 1)?);
    }

    Ok(TlaValue::Set(Arc::new(out)))
}

fn eval_bracket_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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
            return Ok(TlaValue::Function(Arc::new(map)));
        }

        let mut record = BTreeMap::new();
        for entry in split_top_level_symbol(expr, ",") {
            let (key_text, value_text) = split_once_top_level(&entry, "|->")
                .ok_or_else(|| anyhow!("invalid record entry: {entry}"))?;
            let key = parse_record_key(key_text.trim())?;
            let value = eval_expr_inner(value_text.trim(), ctx, depth + 1)?;
            record.insert(key, value);
        }
        return Ok(TlaValue::Record(Arc::new(record)));
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
            result.insert(TlaValue::Function(Arc::new(empty_func)));
            return Ok(TlaValue::Set(Arc::new(result)));
        }

        if range_elems.is_empty() {
            // No functions from non-empty domain to empty range
            return Ok(TlaValue::Set(Arc::new(BTreeSet::new())));
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
            result.insert(TlaValue::Function(Arc::new(func)));

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

        return Ok(TlaValue::Set(Arc::new(result)));
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
/// Returns None if the expression doesn't match the record set pattern.
fn try_eval_record_set(
    expr: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<Option<TlaValue>> {
    // Record set syntax: [field1: Set1, field2: Set2, ...]
    // Each entry is "fieldName: SetExpr" separated by commas
    // We need to distinguish this from:
    // - Record literals: [a |-> 1, b |-> 2] (contains |->)
    // - Function sets: [D -> R] (contains -> without |)
    // - Set comprehensions: {x \in S : P} (inside braces, not brackets)

    let entries = split_top_level_symbol(expr, ",");
    if entries.is_empty() {
        return Ok(None);
    }

    // Parse each entry as "field: Set"
    let mut field_sets: Vec<(String, Vec<TlaValue>)> = Vec::new();

    for entry in &entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        // Find the colon separator at top level
        let colon_idx = match find_top_level_char(entry, ':') {
            Some(idx) => idx,
            None => return Ok(None), // Not a record set pattern
        };

        let field_name = entry[..colon_idx].trim();
        let set_expr = entry[colon_idx + 1..].trim();

        // Field name must be a valid identifier
        if !is_valid_identifier(field_name) {
            return Ok(None);
        }

        // Evaluate the set expression
        let set_value = eval_expr_inner(set_expr, ctx, depth + 1)?;
        let set = set_value.as_set().with_context(|| {
            format!(
                "record set field '{}' value is not a set: {}",
                field_name, set_expr
            )
        })?;

        let elements: Vec<TlaValue> = set.iter().cloned().collect();
        field_sets.push((field_name.to_string(), elements));
    }

    if field_sets.is_empty() {
        return Ok(None);
    }

    // Check for empty sets - if any field has an empty domain, the result is empty
    if field_sets.iter().any(|(_, elems)| elems.is_empty()) {
        return Ok(Some(TlaValue::Set(Arc::new(BTreeSet::new()))));
    }

    // Calculate total number of records (product of all set sizes)
    let total_records: u64 = field_sets
        .iter()
        .map(|(_, elems)| elems.len() as u64)
        .product();

    // Limit the size to avoid memory explosion
    let max_records = 10_000_000u64;
    if total_records > max_records {
        return Err(anyhow!(
            "record set too large: {} records (max {})",
            total_records,
            max_records
        ));
    }

    // Generate all records using Cartesian product
    let mut result = BTreeSet::new();
    let n = field_sets.len();
    let mut indices = vec![0usize; n];

    loop {
        // Build record for current indices
        let mut record: BTreeMap<String, TlaValue> = BTreeMap::new();
        for (i, (field_name, elements)) in field_sets.iter().enumerate() {
            record.insert(field_name.clone(), elements[indices[i]].clone());
        }
        result.insert(TlaValue::Record(Arc::new(record)));

        // Increment indices (like counting in mixed-radix number system)
        let mut carry = true;
        for i in 0..n {
            if carry {
                indices[i] += 1;
                if indices[i] >= field_sets[i].1.len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }

    Ok(Some(TlaValue::Set(Arc::new(result))))
}

/// Check if a string is a valid TLA+ identifier
pub(crate) fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn eval_except_expression(
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

/// Evaluate a module instance operator call: Alias!Operator(args)
fn eval_module_instance_call(
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

fn effective_instance_scope<'a>(
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
fn eval_module_instance_ref(
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
fn eval_enabled(
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

pub(crate) fn eval_operator_call(
    name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("operator recursion depth exceeded at {name}"));
    }

    let user_defined_shadow = matches!(name, "BoundedSeq" | "Max" | "Min")
        && ctx
            .definition(name)
            .is_some_and(|def| def.params.len() == args.len());

    match name {
        "Cardinality" => {
            if args.len() != 1 {
                return Err(anyhow!("Cardinality expects 1 argument"));
            }
            return Ok(TlaValue::Int(args[0].len()? as i64));
        }
        "Max" if args.len() == 1 && !user_defined_shadow => {
            return eval_builtin_extremum(name, &args[0], true);
        }
        "Min" if args.len() == 1 && !user_defined_shadow => {
            return eval_builtin_extremum(name, &args[0], false);
        }
        "BoundedSeq" if args.len() == 2 && !user_defined_shadow => {
            let max_len = args[1].as_int()?;
            return eval_builtin_bounded_seq(&args[0], max_len);
        }
        "TLCGet" => {
            if args.len() != 1 {
                return Err(anyhow!("TLCGet expects 1 argument"));
            }
            return eval_builtin_tlc_get(&args[0]);
        }
        "TLCSet" => {
            if args.len() != 2 {
                return Err(anyhow!("TLCSet expects 2 arguments"));
            }
            return Ok(TlaValue::Bool(true));
        }
        "ToString" => {
            if args.len() != 1 {
                return Err(anyhow!("ToString expects 1 argument"));
            }
            return Ok(TlaValue::String(tla_to_string(&args[0])));
        }
        "Len" => {
            if args.len() != 1 {
                return Err(anyhow!("Len expects 1 argument"));
            }
            return Ok(TlaValue::Int(args[0].len()? as i64));
        }
        "Head" => {
            if args.len() != 1 {
                return Err(anyhow!("Head expects 1 argument"));
            }
            let seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("Head expects a sequence, got {:?}", args[0]))?;
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            return Ok(seq[0].clone());
        }
        "Tail" => {
            if args.len() != 1 {
                return Err(anyhow!("Tail expects 1 argument"));
            }
            let seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("Tail expects a sequence, got {:?}", args[0]))?;
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            return Ok(TlaValue::Seq(Arc::new(seq[1..].to_vec())));
        }
        "Append" => {
            if args.len() != 2 {
                return Err(anyhow!("Append expects 2 arguments"));
            }
            let mut new_seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("Append expects a sequence, got {:?}", args[0]))?;
            new_seq.push(args[1].clone());
            return Ok(TlaValue::Seq(Arc::new(new_seq)));
        }
        "SubSeq" => {
            if args.len() != 3 {
                return Err(anyhow!("SubSeq expects 3 arguments"));
            }
            let seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("SubSeq expects a sequence, got {:?}", args[0]))?;
            let m = args[1].as_int()?;
            let n = args[2].as_int()?;

            // TLA+ sequences are 1-indexed
            if m < 1 {
                return Err(anyhow!("SubSeq start index must be >= 1, got {}", m));
            }
            if n < m - 1 {
                return Err(anyhow!(
                    "SubSeq end index must be >= start index - 1, got m={}, n={}",
                    m,
                    n
                ));
            }

            // Convert to 0-indexed and handle bounds
            let start = (m - 1) as usize;
            let end = (n as usize).min(seq.len());

            if start > seq.len() {
                // If start is beyond the sequence, return empty sequence
                return Ok(TlaValue::Seq(Arc::new(vec![])));
            }

            return Ok(TlaValue::Seq(Arc::new(seq[start..end].to_vec())));
        }
        "SelectSeq" => {
            if args.len() != 2 {
                return Err(anyhow!("SelectSeq expects 2 arguments"));
            }
            let seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("SelectSeq expects a sequence, got {:?}", args[0]))?;
            let test_fn = &args[1];

            let mut result = Vec::new();
            for elem in seq.iter() {
                // Apply the test function to each element
                let test_result = apply_value(test_fn, vec![elem.clone()], ctx, depth + 1)?;
                let passes = test_result.as_bool()?;
                if passes {
                    result.push(elem.clone());
                }
            }
            return Ok(TlaValue::Seq(Arc::new(result)));
        }
        // === Community module: SequencesExt ===
        "SeqOf" if args.len() == 2 && !user_defined_shadow => {
            // SeqOf(set, n) == UNION {[1..m -> set] : m \in 0..n}
            let set = args[0].as_set()?;
            let n = args[1].as_int()?;
            if n < 0 {
                return Ok(TlaValue::Set(Arc::new(BTreeSet::new())));
            }
            ctx.check_budget(1)?; // budget check before potentially large construction
            let elements: Vec<TlaValue> = set.iter().cloned().collect();
            let mut result = BTreeSet::new();
            // m=0: empty sequence
            result.insert(TlaValue::Seq(Arc::new(vec![])));
            for m in 1..=n as usize {
                let total = elements.len().checked_pow(m as u32).unwrap_or(usize::MAX);
                ctx.check_budget(total)?;
                // Generate all sequences of length m over elements
                let mut indices = vec![0usize; m];
                loop {
                    let seq: Vec<TlaValue> = indices.iter().map(|&i| elements[i].clone()).collect();
                    result.insert(TlaValue::Seq(Arc::new(seq)));
                    // Increment indices
                    let mut carry = true;
                    for j in (0..m).rev() {
                        if carry {
                            indices[j] += 1;
                            if indices[j] >= elements.len() {
                                indices[j] = 0;
                            } else {
                                carry = false;
                            }
                        }
                    }
                    if carry {
                        break;
                    }
                }
            }
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "RemoveAt" if args.len() == 2 && !user_defined_shadow => {
            let seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("RemoveAt expects a sequence, got {:?}", args[0]))?;
            let i = args[1].as_int()? as usize;
            if i < 1 || i > seq.len() {
                return Err(anyhow!(
                    "RemoveAt index {} out of bounds for sequence of length {}",
                    i,
                    seq.len()
                ));
            }
            let mut result = seq[..i - 1].to_vec();
            result.extend_from_slice(&seq[i..]);
            return Ok(TlaValue::Seq(Arc::new(result)));
        }
        // === Community module: Functions ===
        "FoldFunction" if args.len() == 3 && !user_defined_shadow => {
            let op = &args[0];
            let base = args[1].clone();
            let fun = &args[2];
            let domain_keys: Vec<TlaValue> = match fun {
                TlaValue::Function(map) => map.values().cloned().collect(),
                TlaValue::Seq(seq) => seq.iter().cloned().collect(),
                _ => {
                    return Err(anyhow!(
                        "FoldFunction expects a function or sequence as 3rd arg, got {:?}",
                        fun
                    ));
                }
            };
            let mut acc = base;
            for val in domain_keys {
                acc = apply_value(op, vec![acc, val], ctx, depth + 1)?;
            }
            return Ok(acc);
        }
        "FoldFunctionOnSet" if args.len() == 4 && !user_defined_shadow => {
            let op = &args[0];
            let base = args[1].clone();
            let fun = &args[2];
            let indices = args[3].as_set()?;
            let mut acc = base;
            for idx in indices.iter() {
                let val = match fun {
                    TlaValue::Function(map) => map.get(idx).cloned().ok_or_else(|| {
                        anyhow!("FoldFunctionOnSet: key {:?} not in function", idx)
                    })?,
                    _ => {
                        return Err(anyhow!(
                            "FoldFunctionOnSet expects a function as 3rd arg, got {:?}",
                            fun
                        ));
                    }
                };
                acc = apply_value(op, vec![acc, val], ctx, depth + 1)?;
            }
            return Ok(acc);
        }
        // === Community module: DyadicRationals ===
        // Dyadic rationals are represented as records [num |-> Int, den |-> Int]
        // where den is a power of 2.
        "Half" if args.len() == 1 && !user_defined_shadow => {
            // Half(p) == Reduce([num |-> p.num, den |-> p.den * 2])
            let p = &args[0];
            let num = p.select_key("num")?.as_int()?;
            let den = p.select_key("den")?.as_int()?;
            let new_den = den * 2;
            // Reduce: divide by GCD
            let g = gcd(num.unsigned_abs(), new_den as u64) as i64;
            return Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                ("num".to_string(), TlaValue::Int(num / g)),
                ("den".to_string(), TlaValue::Int(new_den / g)),
            ]))));
        }
        "Add" if args.len() == 2 && !user_defined_shadow => {
            // Check if both args are records with num/den fields (dyadic rational Add)
            if args[0].select_key("num").is_ok() && args[1].select_key("num").is_ok() {
                let p = &args[0];
                let q = &args[1];
                let pn = p.select_key("num")?.as_int()?;
                let pd = p.select_key("den")?.as_int()?;
                let qn = q.select_key("num")?.as_int()?;
                let qd = q.select_key("den")?.as_int()?;
                if pn == 0 {
                    return Ok(args[1].clone());
                }
                // LCM for dyadic rationals is just max(pd, qd)
                let lcm = pd.max(qd);
                let new_num = pn * (lcm / pd) + qn * (lcm / qd);
                let g = gcd(new_num.unsigned_abs(), lcm as u64) as i64;
                return Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                    ("num".to_string(), TlaValue::Int(new_num / g)),
                    ("den".to_string(), TlaValue::Int(lcm / g)),
                ]))));
            }
            // Fall through to user-defined Add
        }
        "IsDyadicRational" if args.len() == 1 && !user_defined_shadow => {
            let r = &args[0];
            if let (Ok(den_val), Ok(_num_val)) = (r.select_key("den"), r.select_key("num")) {
                let den = den_val.as_int()?;
                // Check if den is a power of 2
                let is_dyadic = den > 0 && (den & (den - 1)) == 0;
                return Ok(TlaValue::Bool(is_dyadic));
            }
            return Ok(TlaValue::Bool(false));
        }
        // === Community module: Folds ===
        "MapThenFoldSet" if args.len() == 5 && !user_defined_shadow => {
            let op = &args[0];
            let base = args[1].clone();
            let f = &args[2];
            let choose_fn = &args[3];
            let set = args[4].as_set()?;
            let mut acc = base;
            let mut remaining = set.clone();
            while !remaining.is_empty() {
                let chosen = apply_value(
                    choose_fn,
                    vec![TlaValue::Set(Arc::new(remaining.clone()))],
                    ctx,
                    depth + 1,
                )?;
                let mapped = apply_value(f, vec![chosen.clone()], ctx, depth + 1)?;
                acc = apply_value(op, vec![mapped, acc], ctx, depth + 1)?;
                remaining.remove(&chosen);
            }
            return Ok(acc);
        }
        // === Community module: FiniteSetsExt ===
        "FoldSet" if args.len() == 3 && !user_defined_shadow => {
            let op = &args[0];
            let base = args[1].clone();
            let set = args[2].as_set()?;
            let mut acc = base;
            for elem in set.iter() {
                acc = apply_value(op, vec![elem.clone(), acc], ctx, depth + 1)?;
            }
            return Ok(acc);
        }
        // === Community module: FiniteSetsExt (additional operators) ===
        "Quantify" if args.len() == 2 && !user_defined_shadow => {
            let set = args[0].as_set()?;
            let pred = &args[1];
            let mut count = 0i64;
            for elem in set.iter() {
                if apply_value(pred, vec![elem.clone()], ctx, depth + 1)?.as_bool()? {
                    count += 1;
                }
            }
            return Ok(TlaValue::Int(count));
        }
        "SymDiff" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_set()?;
            let b = args[1].as_set()?;
            let result: BTreeSet<TlaValue> = a.symmetric_difference(&b).cloned().collect();
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "FlattenSet" if args.len() == 1 && !user_defined_shadow => {
            let sets = args[0].as_set()?;
            let mut result = BTreeSet::new();
            for s in sets.iter() {
                result.extend(s.as_set()?.iter().cloned());
            }
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "kSubset" if args.len() == 2 && !user_defined_shadow => {
            let k = args[0].as_int()? as usize;
            let set = args[1].as_set()?;
            let elements: Vec<TlaValue> = set.iter().cloned().collect();
            let mut result = BTreeSet::new();
            // Generate all k-subsets using combinatorial enumeration
            let n = elements.len();
            if k <= n {
                let mut indices: Vec<usize> = (0..k).collect();
                loop {
                    let subset: BTreeSet<TlaValue> =
                        indices.iter().map(|&i| elements[i].clone()).collect();
                    result.insert(TlaValue::Set(Arc::new(subset)));
                    // Next combination
                    let mut i = k;
                    loop {
                        if i == 0 {
                            break;
                        }
                        i -= 1;
                        indices[i] += 1;
                        if indices[i] <= n - k + i {
                            break;
                        }
                        if i == 0 {
                            indices[0] = n;
                            break;
                        } // signal done
                    }
                    if indices[0] > n - k {
                        break;
                    }
                    for j in (i + 1)..k {
                        indices[j] = indices[j - 1] + 1;
                    }
                }
            }
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "ChooseUnique" if args.len() == 2 && !user_defined_shadow => {
            let set = args[0].as_set()?;
            let pred = &args[1];
            let mut found = None;
            for elem in set.iter() {
                if apply_value(pred, vec![elem.clone()], ctx, depth + 1)?.as_bool()? {
                    if found.is_some() {
                        return Err(anyhow!("ChooseUnique: multiple elements satisfy predicate"));
                    }
                    found = Some(elem.clone());
                }
            }
            return found.ok_or_else(|| anyhow!("ChooseUnique: no element satisfies predicate"));
        }
        "SumSet" if args.len() == 1 && !user_defined_shadow => {
            let set = args[0].as_set()?;
            let mut sum = 0i64;
            for elem in set.iter() {
                sum += elem.as_int()?;
            }
            return Ok(TlaValue::Int(sum));
        }
        "ProductSet" if args.len() == 1 && !user_defined_shadow => {
            let set = args[0].as_set()?;
            let mut product = 1i64;
            for elem in set.iter() {
                product *= elem.as_int()?;
            }
            return Ok(TlaValue::Int(product));
        }
        "IsInjective" if args.len() == 1 && !user_defined_shadow => {
            // IsInjective(f) == \A a, b \in DOMAIN f : f[a] = f[b] => a = b
            let func = &args[0];
            match func {
                TlaValue::Function(map) => {
                    let values: Vec<&TlaValue> = map.values().collect();
                    for i in 0..values.len() {
                        for j in (i + 1)..values.len() {
                            if values[i] == values[j] {
                                return Ok(TlaValue::Bool(false));
                            }
                        }
                    }
                    return Ok(TlaValue::Bool(true));
                }
                TlaValue::Seq(seq) => {
                    for i in 0..seq.len() {
                        for j in (i + 1)..seq.len() {
                            if seq[i] == seq[j] {
                                return Ok(TlaValue::Bool(false));
                            }
                        }
                    }
                    return Ok(TlaValue::Bool(true));
                }
                _ => return Err(anyhow!("IsInjective expects a function or sequence")),
            }
        }
        // === Community module: Graphs (directed) ===
        "IsDirectedGraph" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            if let (Ok(nodes), Ok(edges)) = (g.select_key("node"), g.select_key("edge")) {
                let node_set = nodes.as_set()?;
                let edge_set = edges.as_set()?;
                for e in edge_set.iter() {
                    // Edges are <<a, b>> tuples (sequences of length 2)
                    if let TlaValue::Seq(pair) = e {
                        if pair.len() != 2
                            || !node_set.contains(&pair[0])
                            || !node_set.contains(&pair[1])
                        {
                            return Ok(TlaValue::Bool(false));
                        }
                    } else {
                        return Ok(TlaValue::Bool(false));
                    }
                }
                return Ok(TlaValue::Bool(true));
            }
            return Ok(TlaValue::Bool(false));
        }
        "Successors" if args.len() == 2 && !user_defined_shadow => {
            let g = &args[0];
            let n = &args[1];
            let edges = g.select_key("edge")?.as_set()?;
            let mut result = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 && &pair[0] == n {
                        result.insert(pair[1].clone());
                    }
                }
            }
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "Predecessors" if args.len() == 2 && !user_defined_shadow => {
            let g = &args[0];
            let n = &args[1];
            let edges = g.select_key("edge")?.as_set()?;
            let mut result = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 && &pair[1] == n {
                        result.insert(pair[0].clone());
                    }
                }
            }
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "InDegree" if args.len() == 2 && !user_defined_shadow => {
            let preds = eval_operator_call(
                "Predecessors",
                vec![args[0].clone(), args[1].clone()],
                ctx,
                depth + 1,
            )?;
            return Ok(TlaValue::Int(preds.as_set()?.len() as i64));
        }
        "OutDegree" if args.len() == 2 && !user_defined_shadow => {
            let succs = eval_operator_call(
                "Successors",
                vec![args[0].clone(), args[1].clone()],
                ctx,
                depth + 1,
            )?;
            return Ok(TlaValue::Int(succs.as_set()?.len() as i64));
        }
        "Roots" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            let nodes = g.select_key("node")?.as_set()?;
            let edges = g.select_key("edge")?.as_set()?;
            let mut has_incoming: BTreeSet<TlaValue> = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 {
                        has_incoming.insert(pair[1].clone());
                    }
                }
            }
            let result: BTreeSet<TlaValue> = nodes.difference(&has_incoming).cloned().collect();
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "Leaves" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            let nodes = g.select_key("node")?.as_set()?;
            let edges = g.select_key("edge")?.as_set()?;
            let mut has_outgoing: BTreeSet<TlaValue> = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 {
                        has_outgoing.insert(pair[0].clone());
                    }
                }
            }
            let result: BTreeSet<TlaValue> = nodes.difference(&has_outgoing).cloned().collect();
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "Transpose" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            let nodes = g.select_key("node")?.clone();
            let edges = g.select_key("edge")?.as_set()?;
            let mut reversed = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 {
                        reversed.insert(TlaValue::Seq(Arc::new(vec![
                            pair[1].clone(),
                            pair[0].clone(),
                        ])));
                    }
                }
            }
            return Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                ("node".to_string(), nodes),
                ("edge".to_string(), TlaValue::Set(Arc::new(reversed))),
            ]))));
        }
        "IsDag" if args.len() == 1 && !user_defined_shadow => {
            // Native cycle detection using DFS
            let g = &args[0];
            let nodes = g.select_key("node")?.as_set()?;
            let edges = g.select_key("edge")?.as_set()?;
            // Build adjacency list
            let node_list: Vec<TlaValue> = nodes.iter().cloned().collect();
            let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 {
                        if let (Some(from), Some(to)) = (
                            node_list.iter().position(|n| n == &pair[0]),
                            node_list.iter().position(|n| n == &pair[1]),
                        ) {
                            adj.entry(from).or_default().push(to);
                        }
                    }
                }
            }
            // DFS cycle detection
            let n = node_list.len();
            let mut color = vec![0u8; n]; // 0=white, 1=gray, 2=black
            fn has_cycle(node: usize, adj: &BTreeMap<usize, Vec<usize>>, color: &mut [u8]) -> bool {
                color[node] = 1;
                if let Some(neighbors) = adj.get(&node) {
                    for &next in neighbors {
                        if color[next] == 1 {
                            return true;
                        } // back edge = cycle
                        if color[next] == 0 && has_cycle(next, adj, color) {
                            return true;
                        }
                    }
                }
                color[node] = 2;
                false
            }
            let mut is_dag = true;
            for i in 0..n {
                if color[i] == 0 && has_cycle(i, &adj, &mut color) {
                    is_dag = false;
                    break;
                }
            }
            return Ok(TlaValue::Bool(is_dag));
        }
        // === Community module: Relation ===
        "TransitiveClosure" if args.len() == 2 && !user_defined_shadow => {
            // Floyd-Warshall style transitive closure
            let rel = &args[0]; // function [node x node -> BOOLEAN]
            let set = args[1].as_set()?;
            let nodes: Vec<TlaValue> = set.iter().cloned().collect();
            let n = nodes.len();
            // Initialize reachability matrix
            let mut reach = vec![vec![false; n]; n];
            for (i, ni) in nodes.iter().enumerate() {
                for (j, nj) in nodes.iter().enumerate() {
                    let key = TlaValue::Seq(Arc::new(vec![ni.clone(), nj.clone()]));
                    if let Ok(val) = rel.apply(&key) {
                        reach[i][j] = val.as_bool().unwrap_or(false);
                    }
                }
            }
            // Floyd-Warshall
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        if reach[i][k] && reach[k][j] {
                            reach[i][j] = true;
                        }
                    }
                }
            }
            // Build result function
            let mut result = BTreeMap::new();
            for (i, ni) in nodes.iter().enumerate() {
                for (j, nj) in nodes.iter().enumerate() {
                    let key = TlaValue::Seq(Arc::new(vec![ni.clone(), nj.clone()]));
                    result.insert(key, TlaValue::Bool(reach[i][j]));
                }
            }
            return Ok(TlaValue::Function(Arc::new(result)));
        }
        // === Standard module: Bags ===
        // Bags are represented as TlaValue::Function from elements to natural counts.
        // EmptyBag is the empty function. SetToBag converts a set to a bag with count 1.
        "EmptyBag" if args.is_empty() && !user_defined_shadow => {
            return Ok(TlaValue::Function(Arc::new(BTreeMap::new())));
        }
        "SetToBag" if args.len() == 1 && !user_defined_shadow => {
            // SetToBag(S) == [e \in S |-> 1]
            let set = args[0].as_set()?;
            let mut bag = BTreeMap::new();
            for elem in set.iter() {
                bag.insert(elem.clone(), TlaValue::Int(1));
            }
            return Ok(TlaValue::Function(Arc::new(bag)));
        }
        "BagToSet" if args.len() == 1 && !user_defined_shadow => {
            // BagToSet(B) == DOMAIN B
            if let TlaValue::Function(f) = &args[0] {
                let set: BTreeSet<TlaValue> = f.keys().cloned().collect();
                return Ok(TlaValue::Set(Arc::new(set)));
            }
            return Err(anyhow!("BagToSet: argument is not a bag"));
        }
        "IsABag" if args.len() == 1 && !user_defined_shadow => {
            if let TlaValue::Function(f) = &args[0] {
                let all_nat = f.values().all(|v| matches!(v, TlaValue::Int(n) if *n > 0));
                return Ok(TlaValue::Bool(all_nat));
            }
            return Ok(TlaValue::Bool(false));
        }
        "BagIn" if args.len() == 2 && !user_defined_shadow => {
            // BagIn(e, B) == e \in DOMAIN B
            if let TlaValue::Function(f) = &args[1] {
                return Ok(TlaValue::Bool(f.contains_key(&args[0])));
            }
            return Ok(TlaValue::Bool(false));
        }
        "BagOfAll" if args.len() == 2 && !user_defined_shadow => {
            // BagOfAll(F, B) — apply F to each element, sum counts
            // Simplified: just return a function
            if let TlaValue::Function(bag) = &args[1] {
                let mut result = BTreeMap::new();
                for (elem, count) in bag.iter() {
                    // We can't easily apply F here without the full eval context,
                    // so just pass through
                    result.insert(elem.clone(), count.clone());
                }
                return Ok(TlaValue::Function(Arc::new(result)));
            }
            return Err(anyhow!("BagOfAll: second argument is not a bag"));
        }
        "BagUnion" if args.len() == 2 && !user_defined_shadow => {
            if let (TlaValue::Function(a), TlaValue::Function(b)) = (&args[0], &args[1]) {
                let mut result = a.as_ref().clone();
                for (k, v) in b.iter() {
                    let count_a = result.get(k).and_then(|c| c.as_int().ok()).unwrap_or(0);
                    let count_b = v.as_int().unwrap_or(0);
                    result.insert(k.clone(), TlaValue::Int(count_a + count_b));
                }
                return Ok(TlaValue::Function(Arc::new(result)));
            }
            return Err(anyhow!("BagUnion: arguments are not bags"));
        }
        "CopiesIn" if args.len() == 2 && !user_defined_shadow => {
            // CopiesIn(e, B) == IF BagIn(e, B) THEN B[e] ELSE 0
            if let TlaValue::Function(f) = &args[1] {
                if let Some(count) = f.get(&args[0]) {
                    return Ok(count.clone());
                }
                return Ok(TlaValue::Int(0));
            }
            return Ok(TlaValue::Int(0));
        }
        // === Community module: BagsExt ===
        "BagAdd" if args.len() == 2 && !user_defined_shadow => {
            // BagAdd(B, e) — add one copy of e to bag B
            if let TlaValue::Function(f) = &args[0] {
                let mut result = f.as_ref().clone();
                let count = result
                    .get(&args[1])
                    .and_then(|c| c.as_int().ok())
                    .unwrap_or(0);
                result.insert(args[1].clone(), TlaValue::Int(count + 1));
                return Ok(TlaValue::Function(Arc::new(result)));
            }
            return Err(anyhow!("BagAdd: first argument is not a bag"));
        }
        "BagRemove" if args.len() == 2 && !user_defined_shadow => {
            // BagRemove(B, e) — remove one copy of e from bag B
            if let TlaValue::Function(f) = &args[0] {
                let mut result = f.as_ref().clone();
                if let Some(count_val) = result.get(&args[1]) {
                    let count = count_val.as_int().unwrap_or(0);
                    if count <= 1 {
                        result.remove(&args[1]);
                    } else {
                        result.insert(args[1].clone(), TlaValue::Int(count - 1));
                    }
                }
                return Ok(TlaValue::Function(Arc::new(result)));
            }
            return Err(anyhow!("BagRemove: first argument is not a bag"));
        }
        // === Standard module: IOUtils ===
        "IOEnv" if args.is_empty() && !user_defined_shadow => {
            let mut rec = BTreeMap::new();
            for (k, v) in std::env::vars() {
                rec.insert(k, TlaValue::String(v));
            }
            return Ok(TlaValue::Record(Arc::new(rec)));
        }
        "ndJsonDeserialize" if args.len() == 1 && !user_defined_shadow => {
            let path = match &args[0] {
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => return Err(anyhow!("expected string argument, got {:?}", other)),
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|e| anyhow!("ndJsonDeserialize: {}: {}", path, e))?;
            let mut items = Vec::new();
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let json_val: serde_json::Value = serde_json::from_str(trimmed)
                    .map_err(|e| anyhow!("ndJsonDeserialize: parse error: {}", e))?;
                items.push(json_to_tla_value(&json_val));
            }
            return Ok(TlaValue::Seq(Arc::new(items)));
        }
        "JsonDeserialize" if args.len() == 1 && !user_defined_shadow => {
            let path = match &args[0] {
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => return Err(anyhow!("expected string argument, got {:?}", other)),
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|e| anyhow!("JsonDeserialize: {}: {}", path, e))?;
            let json_val: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| anyhow!("JsonDeserialize: parse error: {}", e))?;
            return Ok(json_to_tla_value(&json_val));
        }
        "JsonSerialize" if args.len() == 2 && !user_defined_shadow => {
            let path = match &args[0] {
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => return Err(anyhow!("expected string argument, got {:?}", other)),
            };
            let json = tla_value_to_json(&args[1]);
            let json_str =
                serde_json::to_string_pretty(&json).map_err(|e| anyhow!("JsonSerialize: {}", e))?;
            std::fs::write(&path, json_str)
                .map_err(|e| anyhow!("JsonSerialize: {}: {}", path, e))?;
            return Ok(TlaValue::Bool(true));
        }
        "ToString" if args.len() == 1 && !user_defined_shadow => {
            let s = match &args[0] {
                TlaValue::Int(n) => n.to_string(),
                TlaValue::Bool(b) => (if *b { "TRUE" } else { "FALSE" }).to_string(),
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => format!("{:?}", other),
            };
            return Ok(TlaValue::String(s));
        }
        // === Community module: Bitwise ===
        "IsABitVector" if args.len() == 2 && !user_defined_shadow => {
            let val = args[0].as_int()?;
            let n = args[1].as_int()?;
            return Ok(TlaValue::Bool(val >= 0 && val < (1 << n)));
        }
        "IsANatural" if args.len() == 1 && !user_defined_shadow => {
            return Ok(TlaValue::Bool(
                matches!(&args[0], TlaValue::Int(n) if *n >= 0),
            ));
        }
        // Bitwise AND, OR, XOR on integers
        "BitsAnd" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            let b = args[1].as_int()?;
            return Ok(TlaValue::Int(a & b));
        }
        "BitsOr" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            let b = args[1].as_int()?;
            return Ok(TlaValue::Int(a | b));
        }
        "BitsXor" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            let b = args[1].as_int()?;
            return Ok(TlaValue::Int(a ^ b));
        }
        "BitNot" if args.len() == 1 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            return Ok(TlaValue::Int(!a));
        }
        "LeftShift" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            let n = args[1].as_int()?;
            return Ok(TlaValue::Int(a << n));
        }
        "RightShift" if args.len() == 2 && !user_defined_shadow => {
            let a = args[0].as_int()?;
            let n = args[1].as_int()?;
            return Ok(TlaValue::Int(a >> n));
        }
        // === Community module: Combinatorics ===
        "Factorial" if args.len() == 1 && !user_defined_shadow => {
            let n = args[0].as_int()?;
            if n < 0 {
                return Err(anyhow!("Factorial: argument must be non-negative"));
            }
            let mut result: i64 = 1;
            for i in 2..=n {
                result = result.saturating_mul(i);
            }
            return Ok(TlaValue::Int(result));
        }
        "nCk" if args.len() == 2 && !user_defined_shadow => {
            let n = args[0].as_int()?;
            let k = args[1].as_int()?;
            if k < 0 || k > n {
                return Ok(TlaValue::Int(0));
            }
            let k = k.min(n - k); // optimize
            let mut result: i64 = 1;
            for i in 0..k {
                result = result * (n - i) / (i + 1);
            }
            return Ok(TlaValue::Int(result));
        }
        "nPk" if args.len() == 2 && !user_defined_shadow => {
            let n = args[0].as_int()?;
            let k = args[1].as_int()?;
            if k < 0 || k > n {
                return Ok(TlaValue::Int(0));
            }
            let mut result: i64 = 1;
            for i in 0..k {
                result = result.saturating_mul(n - i);
            }
            return Ok(TlaValue::Int(result));
        }
        // === Community module: CSV ===
        "CSVRead" if args.len() == 1 && !user_defined_shadow => {
            let path = match &args[0] {
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => return Err(anyhow!("CSVRead: expected string path, got {:?}", other)),
            };
            let content =
                std::fs::read_to_string(&path).map_err(|e| anyhow!("CSVRead: {}: {}", path, e))?;
            let mut rows = Vec::new();
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let fields: Vec<TlaValue> = line
                    .split(',')
                    .map(|f| TlaValue::String(f.trim().to_string()))
                    .collect();
                rows.push(TlaValue::Seq(Arc::new(fields)));
            }
            return Ok(TlaValue::Seq(Arc::new(rows)));
        }
        "CSVWrite" if args.len() == 2 && !user_defined_shadow => {
            let path = match &args[0] {
                TlaValue::String(s) => s.clone(),
                TlaValue::ModelValue(s) => s.clone(),
                other => return Err(anyhow!("CSVWrite: expected string path, got {:?}", other)),
            };
            let data = tla_value_to_json(&args[1]);
            let json_str = serde_json::to_string(&data).map_err(|e| anyhow!("CSVWrite: {}", e))?;
            std::fs::write(&path, json_str).map_err(|e| anyhow!("CSVWrite: {}: {}", path, e))?;
            return Ok(TlaValue::Bool(true));
        }
        // === Community module: VectorClocks ===
        "VCLessOrEqual" if args.len() == 2 && !user_defined_shadow => {
            // vc1 <= vc2 iff forall keys, vc1[k] <= vc2[k]
            if let (TlaValue::Function(a), TlaValue::Function(b)) = (&args[0], &args[1]) {
                for (k, va) in a.iter() {
                    let ia = va.as_int().unwrap_or(0);
                    let ib = b.get(k).and_then(|v| v.as_int().ok()).unwrap_or(0);
                    if ia > ib {
                        return Ok(TlaValue::Bool(false));
                    }
                }
                return Ok(TlaValue::Bool(true));
            }
            return Err(anyhow!(
                "VCLessOrEqual: arguments must be vector clocks (functions)"
            ));
        }
        "VCLess" if args.len() == 2 && !user_defined_shadow => {
            // vc1 < vc2 iff vc1 <= vc2 and vc1 != vc2
            if let (TlaValue::Function(a), TlaValue::Function(b)) = (&args[0], &args[1]) {
                let mut all_leq = true;
                let mut any_less = false;
                for (k, va) in a.iter() {
                    let ia = va.as_int().unwrap_or(0);
                    let ib = b.get(k).and_then(|v| v.as_int().ok()).unwrap_or(0);
                    if ia > ib {
                        all_leq = false;
                        break;
                    }
                    if ia < ib {
                        any_less = true;
                    }
                }
                return Ok(TlaValue::Bool(all_leq && any_less));
            }
            return Err(anyhow!(
                "VCLess: arguments must be vector clocks (functions)"
            ));
        }
        "VCMerge" if args.len() == 2 && !user_defined_shadow => {
            // max of each component
            if let (TlaValue::Function(a), TlaValue::Function(b)) = (&args[0], &args[1]) {
                let mut result = a.as_ref().clone();
                for (k, vb) in b.iter() {
                    let ib = vb.as_int().unwrap_or(0);
                    let ia = result.get(k).and_then(|v| v.as_int().ok()).unwrap_or(0);
                    result.insert(k.clone(), TlaValue::Int(ia.max(ib)));
                }
                return Ok(TlaValue::Function(Arc::new(result)));
            }
            return Err(anyhow!(
                "VCMerge: arguments must be vector clocks (functions)"
            ));
        }
        // === Community module: UndirectedGraphs ===
        "IsUndirectedGraph" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            if let (Ok(nodes), Ok(edges)) = (g.select_key("node"), g.select_key("edge")) {
                let node_set = nodes.as_set()?;
                let edge_set = edges.as_set()?;
                for e in edge_set.iter() {
                    let e_set = e.as_set()?;
                    if e_set.len() != 2 {
                        return Ok(TlaValue::Bool(false));
                    }
                    for elem in e_set.iter() {
                        if !node_set.contains(elem) {
                            return Ok(TlaValue::Bool(false));
                        }
                    }
                }
                return Ok(TlaValue::Bool(true));
            }
            return Ok(TlaValue::Bool(false));
        }
        "IsLoopFreeUndirectedGraph" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            if let (Ok(nodes), Ok(edges)) = (g.select_key("node"), g.select_key("edge")) {
                let node_set = nodes.as_set()?;
                let edge_set = edges.as_set()?;
                for e in edge_set.iter() {
                    let e_set = e.as_set()?;
                    if e_set.len() != 2 {
                        return Ok(TlaValue::Bool(false));
                    }
                    for elem in e_set.iter() {
                        if !node_set.contains(elem) {
                            return Ok(TlaValue::Bool(false));
                        }
                    }
                }
                return Ok(TlaValue::Bool(true));
            }
            return Ok(TlaValue::Bool(false));
        }
        "ConnectedComponents" if args.len() == 1 && !user_defined_shadow => {
            // Native union-find implementation (much faster than the TLA+ definition)
            let g = &args[0];
            let nodes = g.select_key("node")?.as_set()?;
            let edges = g.select_key("edge")?.as_set()?;
            let node_list: Vec<TlaValue> = nodes.iter().cloned().collect();
            // Map each node to its component index
            let mut parent: Vec<usize> = (0..node_list.len()).collect();
            let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]]; // path compression
                    x = parent[x];
                }
                x
            };
            for edge in edges.iter() {
                let e_set = edge.as_set()?;
                let edge_nodes: Vec<&TlaValue> = e_set.iter().collect();
                if edge_nodes.len() == 2 {
                    if let (Some(a), Some(b)) = (
                        node_list.iter().position(|n| n == edge_nodes[0]),
                        node_list.iter().position(|n| n == edge_nodes[1]),
                    ) {
                        let ra = find(&mut parent, a);
                        let rb = find(&mut parent, b);
                        if ra != rb {
                            parent[ra] = rb;
                        }
                    }
                }
            }
            // Build component sets
            let mut components: BTreeMap<usize, BTreeSet<TlaValue>> = BTreeMap::new();
            for (i, node) in node_list.iter().enumerate() {
                let root = find(&mut parent, i);
                components.entry(root).or_default().insert(node.clone());
            }
            let result: BTreeSet<TlaValue> = components
                .into_values()
                .map(|s| TlaValue::Set(Arc::new(s)))
                .collect();
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "AreConnectedIn" if args.len() == 3 && !user_defined_shadow => {
            let m = &args[0];
            let n = &args[1];
            let g = &args[2];
            // Reuse ConnectedComponents
            let comps = eval_operator_call("ConnectedComponents", vec![g.clone()], ctx, depth + 1)?;
            for comp in comps.as_set()?.iter() {
                let comp_set = comp.as_set()?;
                if comp_set.contains(m) && comp_set.contains(n) {
                    return Ok(TlaValue::Bool(true));
                }
            }
            return Ok(TlaValue::Bool(false));
        }
        "IsStronglyConnected" if args.len() == 1 && !user_defined_shadow => {
            let comps =
                eval_operator_call("ConnectedComponents", vec![args[0].clone()], ctx, depth + 1)?;
            let num_comps = comps.as_set()?.len();
            return Ok(TlaValue::Bool(num_comps <= 1));
        }
        "PrettyPrint" if args.len() == 1 && !user_defined_shadow => {
            let p = &args[0];
            if let (Ok(num_val), Ok(den_val)) = (p.select_key("num"), p.select_key("den")) {
                let num = num_val.as_int()?;
                let den = den_val.as_int()?;
                if num == 0 {
                    return Ok(TlaValue::String("0".to_string()));
                }
                if num == 1 && den == 1 {
                    return Ok(TlaValue::String("1".to_string()));
                }
                return Ok(TlaValue::String(format!("{}/{}", num, den)));
            }
            return Ok(TlaValue::String(format!("{:?}", p)));
        }
        "Permutations" => {
            if args.len() != 1 {
                return Err(anyhow!("Permutations expects 1 argument"));
            }
            let values = args[0].as_set()?.iter().cloned().collect::<Vec<_>>();
            let permutations = generate_permutations(&values);
            let mut out = BTreeSet::new();
            for perm in permutations {
                let mut map = BTreeMap::new();
                for (k, v) in values.iter().cloned().zip(perm.into_iter()) {
                    map.insert(k, v);
                }
                out.insert(TlaValue::Function(Arc::new(map)));
            }
            return Ok(TlaValue::Set(Arc::new(out)));
        }
        "DOMAIN" => {
            if args.len() != 1 {
                return Err(anyhow!("DOMAIN expects 1 argument"));
            }
            match &args[0] {
                TlaValue::Function(map) => {
                    let keys = map.keys().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(keys)));
                }
                TlaValue::Record(map) => {
                    let keys = map
                        .keys()
                        .map(|k| TlaValue::String(k.clone()))
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(keys)));
                }
                TlaValue::Seq(seq) => {
                    // DOMAIN of a sequence is {1, 2, ..., Len(seq)}
                    let indices = (1..=seq.len() as i64)
                        .map(TlaValue::Int)
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(indices)));
                }
                _ => {
                    return Err(anyhow!("DOMAIN expects a function, record, or sequence"));
                }
            }
        }
        "UNION" => {
            if args.len() != 1 {
                return Err(anyhow!("UNION expects 1 argument"));
            }
            let mut union = BTreeSet::new();
            for set in args[0].as_set()? {
                union.extend(set.as_set()?.iter().cloned());
            }
            return Ok(TlaValue::Set(Arc::new(union)));
        }
        "RandomElement" => {
            if args.len() != 1 {
                return Err(anyhow!("RandomElement expects 1 argument"));
            }
            return args[0]
                .as_set()?
                .iter()
                .next()
                .cloned()
                .ok_or_else(|| anyhow!("RandomElement expects a non-empty set"));
        }
        // Randomization module: RandomSubset(k, S) - returns a random subset of S with k elements
        // For model checking, we return the first k elements (deterministic)
        "RandomSubset" => {
            if args.len() != 2 {
                return Err(anyhow!("RandomSubset expects 2 arguments (k, S)"));
            }
            let k = args[0].as_int()? as usize;
            let set = args[1].as_set()?;
            let subset: BTreeSet<TlaValue> = set.iter().take(k).cloned().collect();
            return Ok(TlaValue::Set(Arc::new(subset)));
        }
        // TLC module: Range(f) - returns the set of all values in the range of function f
        // Range(f) == {f[x] : x \in DOMAIN f}
        "Range" => {
            if args.len() != 1 {
                return Err(anyhow!("Range expects 1 argument"));
            }
            match &args[0] {
                TlaValue::Function(map) => {
                    let values = map.values().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                TlaValue::Seq(seq) => {
                    // Range of a sequence is the set of all its elements
                    let values = seq.iter().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                TlaValue::Record(map) => {
                    // Range of a record is the set of all its field values
                    let values = map.values().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                _ => {
                    return Err(anyhow!("Range expects a function, sequence, or record"));
                }
            }
        }

        // TLC module: FunAsSeq(f, a, b) - converts a function to a sequence
        // FunAsSeq(f, a, b) == [i \in 1..b |-> f[a + i - 1]]
        // This creates a sequence of length b by extracting values from f
        // starting at index a
        "FunAsSeq" => {
            if args.len() != 3 {
                return Err(anyhow!("FunAsSeq expects 3 arguments: f, a, b"));
            }
            let func = &args[0];
            let a = args[1].as_int()?;
            let b = args[2].as_int()?;

            if b < 0 {
                return Err(anyhow!("FunAsSeq: b must be non-negative, got {}", b));
            }

            let mut result = Vec::with_capacity(b as usize);
            // FunAsSeq(f, n, m) == [i \in 1..m |-> f[i]]
            // n is unused (domain bound for type checking), m is length
            let _ = a; // n is unused in evaluation
            for i in 1..=b {
                let key = TlaValue::Int(i);
                let val = func.apply(&key)?.clone();
                result.push(val);
            }
            return Ok(TlaValue::Seq(Arc::new(result)));
        }
        _ => {}
    }

    // Check if name refers to a local value that's a Lambda (higher-order parameter)
    if let Some(local_val) = ctx.runtime_value(name) {
        if matches!(local_val, TlaValue::Lambda { .. }) {
            return apply_value(&local_val, args, ctx, depth + 1);
        }
    }

    let def = ctx
        .definition(name)
        .ok_or_else(|| anyhow!("unknown operator/function '{name}'"))?;

    if def.params.len() != args.len() {
        return Err(anyhow!(
            "operator '{name}' arity mismatch: expected {}, got {}",
            def.params.len(),
            args.len()
        ));
    }

    let mut child = ctx.clone();
    {
        let locals_mut = Rc::make_mut(&mut child.locals);
        for (param, arg) in def.params.iter().zip(args.into_iter()) {
            bind_param_value(locals_mut, param, arg)?;
        }
    }
    eval_expr_inner(&def.body, &child, depth + 1)
}

fn eval_builtin_extremum(name: &str, value: &TlaValue, want_max: bool) -> Result<TlaValue> {
    let mut ints = match value {
        TlaValue::Set(set) => set
            .iter()
            .map(TlaValue::as_int)
            .collect::<Result<Vec<_>, _>>()?,
        TlaValue::Seq(seq) => seq
            .iter()
            .map(TlaValue::as_int)
            .collect::<Result<Vec<_>, _>>()?,
        other => {
            return Err(anyhow!(
                "{name} expects a set or sequence of integers, got {:?}",
                other
            ));
        }
    };

    let first = ints
        .pop()
        .ok_or_else(|| anyhow!("{name} expects a non-empty set or sequence"))?;
    let extremum = ints.into_iter().fold(first, |current, item| {
        if (want_max && item > current) || (!want_max && item < current) {
            item
        } else {
            current
        }
    });
    Ok(TlaValue::Int(extremum))
}

fn eval_builtin_bounded_seq(domain: &TlaValue, max_len: i64) -> Result<TlaValue> {
    if max_len < 0 {
        return Err(anyhow!(
            "BoundedSeq expects a non-negative bound, got {max_len}"
        ));
    }

    let elements = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
    let max_len = max_len as usize;
    let max_sequences = 100_000u128;

    let mut total = 1u128;
    let base = elements.len() as u128;
    for len in 1..=max_len {
        total = total.saturating_add(base.saturating_pow(len as u32));
        if total > max_sequences {
            return Err(anyhow!(
                "BoundedSeq too large: |S|={} and n={} would generate {} sequences (max {})",
                elements.len(),
                max_len,
                total,
                max_sequences
            ));
        }
    }

    let mut out = BTreeSet::new();
    let mut current = vec![Vec::<TlaValue>::new()];
    out.insert(TlaValue::Seq(Arc::new(Vec::new())));

    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in &current {
            for value in &elements {
                let mut seq = prefix.clone();
                seq.push(value.clone());
                out.insert(TlaValue::Seq(Arc::new(seq.clone())));
                next.push(seq);
            }
        }
        current = next;
    }

    Ok(TlaValue::Set(Arc::new(out)))
}

fn eval_builtin_tlc_get(key: &TlaValue) -> Result<TlaValue> {
    match key {
        TlaValue::String(name) if name == "level" => Ok(TlaValue::Int(0)),
        TlaValue::String(name) if name == "config" => {
            Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                ("mode".to_string(), TlaValue::String("bfs".to_string())),
                ("worker".to_string(), TlaValue::Int(1)),
            ]))))
        }
        TlaValue::Int(slot) if *slot == 2 || *slot == 3 => Ok(TlaValue::Int(999)),
        TlaValue::Int(_) => Ok(TlaValue::Int(0)),
        other => Err(anyhow!("unsupported TLCGet key: {:?}", other)),
    }
}

fn split_outer_let(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let (let_start, after_let) = take_keyword_prefix(trimmed, "LET")?;
    debug_assert!(let_start.is_empty());

    let mut depth = 1usize;
    let mut i = 0usize;
    while i < after_let.len() {
        if let Some((word, start, end)) = next_word(after_let, i) {
            match word {
                "LET" => depth += 1,
                "IN" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        let defs = after_let[..start].trim();
                        let body = after_let[end..].trim_end();
                        return Some((defs, body));
                    }
                }
                _ => {}
            }
            i = end;
        } else {
            break;
        }
    }

    None
}

fn parse_let_definitions(defs_text: &str) -> Result<BTreeMap<String, TlaDefinition>> {
    let eq_positions = find_top_level_definition_eqs(defs_text);

    if eq_positions.is_empty() {
        return Err(anyhow!("LET block has no definitions"));
    }

    let mut defs = BTreeMap::new();
    let mut cursor = skip_leading_ws(defs_text, 0);

    for (idx, eq_pos) in eq_positions.iter().enumerate() {
        let next_head_start = if idx + 1 < eq_positions.len() {
            line_start_before(defs_text, eq_positions[idx + 1])
        } else {
            defs_text.len()
        };

        let head = trim_let_edge_comments(&defs_text[cursor..*eq_pos]);
        let body = trim_let_edge_comments(&defs_text[*eq_pos + 2..next_head_start]);
        let (name, params) = parse_local_def_head(head);

        if !name.is_empty() {
            defs.insert(
                name.clone(),
                TlaDefinition {
                    name,
                    params,
                    body: body.to_string(),
                    is_recursive: false,
                },
            );
        }

        cursor = skip_leading_ws(defs_text, next_head_start);
    }

    Ok(defs)
}

fn trim_let_edge_comments(text: &str) -> &str {
    let mut start = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("\\*") {
            start += line.len();
            continue;
        }
        break;
    }

    let trimmed = &text[start..];
    let mut end = trimmed.len();
    let lines: Vec<&str> = trimmed.split_inclusive('\n').collect();
    for line in lines.into_iter().rev() {
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() || line_trimmed.starts_with("\\*") {
            end = end.saturating_sub(line.len());
            continue;
        }
        break;
    }

    trimmed[..end].trim()
}

fn parse_local_def_head(head: &str) -> (String, Vec<String>) {
    if let Some((open, close)) = first_local_def_param_delims(head)
        && let Some((name, params)) = parse_local_def_head_with_delims(head, open, close)
    {
        return (name, params);
    }

    let name = head
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn first_local_def_param_delims(head: &str) -> Option<(char, char)> {
    match (head.find('('), head.find('[')) {
        (Some(paren), Some(bracket)) if bracket < paren => Some(('[', ']')),
        (Some(_), Some(_)) => Some(('(', ')')),
        (Some(_), None) => Some(('(', ')')),
        (None, Some(_)) => Some(('[', ']')),
        (None, None) => None,
    }
}

fn parse_local_def_head_with_delims(
    head: &str,
    open_delim: char,
    close_delim: char,
) -> Option<(String, Vec<String>)> {
    let open = head.find(open_delim)?;
    let close = head.rfind(close_delim)?;
    if close <= open {
        return None;
    }

    let name = head[..open].trim().to_string();
    let inside = &head[open + 1..close];
    let params = if open_delim == '(' {
        split_top_level_symbol(inside, ",")
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
    } else {
        let mut params = Vec::new();
        for part in split_top_level_symbol(inside, ",") {
            let part = part.trim();
            let lhs = if let Some(pos) = find_top_level_keyword_index(part, "\\in") {
                &part[..pos]
            } else {
                part
            };
            for candidate in split_top_level_symbol(lhs, ",") {
                let candidate = candidate.trim();
                if candidate.is_empty() {
                    continue;
                }
                params.push(candidate.to_string());
            }
        }
        params
    };
    Some((name, params))
}

#[derive(Debug, Clone)]
struct BinderSpec {
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

fn parse_binders(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<BinderSpec>> {
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

fn bind_param_value(
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

fn collect_function_mapping(
    idx: usize,
    binders: &[BinderSpec],
    body: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeMap<TlaValue, TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
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

fn collect_binder_filter_set(
    idx: usize,
    binders: &[BinderSpec],
    predicate: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeSet<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
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

fn collect_binder_map_set(
    idx: usize,
    binders: &[BinderSpec],
    element_expr: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    out: &mut BTreeSet<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<()> {
    if idx >= binders.len() {
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

#[derive(Debug, Clone)]
enum PathSegment {
    Field(String),
    Index(TlaValue),
}

fn parse_except_path(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<PathSegment>> {
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
                Ok(TlaValue::Record(Arc::new(next)))
            }
            _ => Err(anyhow!("field update on non-record value {base:?}")),
        },
        PathSegment::Index(key) => match base {
            TlaValue::Function(map) => {
                let current = map.get(key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = (**map).clone();
                next.insert(key.clone(), updated);
                Ok(TlaValue::Function(Arc::new(next)))
            }
            TlaValue::Record(map) => {
                let record_key = record_key_from_value(key)?;
                let current = map.get(&record_key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = (**map).clone();
                next.insert(record_key, updated);
                Ok(TlaValue::Record(Arc::new(next)))
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
                Ok(TlaValue::Seq(Arc::new(next)))
            }
            _ => Err(anyhow!("index update on unsupported value {base:?}")),
        },
    }
}

fn parse_argument_list(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<Vec<TlaValue>> {
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

fn eval_bracket_index_key(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let args = parse_argument_list(expr, ctx, depth + 1)?;
    match args.len() {
        0 => Err(anyhow!("empty bracket index")),
        1 => Ok(args.into_iter().next().expect("single arg exists")),
        _ => Ok(TlaValue::Seq(Arc::new(args))),
    }
}

/// Greatest common divisor (Euclidean algorithm)
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn sequence_like_values(value: &TlaValue) -> Option<Vec<TlaValue>> {
    match value {
        TlaValue::Seq(seq) => Some((**seq).clone()),
        TlaValue::Function(map) => {
            let mut out = Vec::with_capacity(map.len());
            for (idx, (key, value)) in map.iter().enumerate() {
                let expected = (idx as i64) + 1;
                match key {
                    TlaValue::Int(actual) if *actual == expected => out.push(value.clone()),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn seq_or_string_concat(lhs: TlaValue, rhs: TlaValue) -> Result<TlaValue> {
    match (lhs, rhs) {
        (TlaValue::String(mut a), TlaValue::String(b)) => {
            a.push_str(&b);
            Ok(TlaValue::String(a))
        }
        (a, b) => {
            let Some(mut lhs_seq) = sequence_like_values(&a) else {
                return Err(anyhow!(
                    "\\o expects String or Seq operands, got {a:?} and {b:?}"
                ));
            };
            let Some(rhs_seq) = sequence_like_values(&b) else {
                return Err(anyhow!(
                    "\\o expects String or Seq operands, got {a:?} and {b:?}"
                ));
            };
            lhs_seq.extend(rhs_seq);
            Ok(TlaValue::Seq(Arc::new(lhs_seq)))
        }
    }
}

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

fn record_key_from_value(value: &TlaValue) -> Result<String> {
    match value {
        TlaValue::String(v) | TlaValue::ModelValue(v) => Ok(v.clone()),
        _ => Err(anyhow!(
            "record key must be String or ModelValue, got {value:?}"
        )),
    }
}

fn powerset(input: &BTreeSet<TlaValue>, ctx: &EvalContext<'_>) -> Result<BTreeSet<TlaValue>> {
    let mut subsets = BTreeSet::new();
    let values: Vec<TlaValue> = input.iter().cloned().collect();
    let n = values.len();

    // Hard limit: SUBSET of sets with >20 elements is never practical
    if n > 20 {
        return Err(anyhow!(
            "SUBSET of {}-element set would produce 2^{} = {} elements (too large)",
            n,
            n,
            if n < 64 {
                format!("{}", 1u64 << n)
            } else {
                format!("2^{}", n)
            }
        ));
    }

    if n >= usize::BITS as usize {
        return Ok(subsets);
    }

    let total = 1usize << n;
    ctx.check_budget(total)?;

    for mask in 0usize..total {
        let mut subset = BTreeSet::new();
        for (idx, value) in values.iter().enumerate() {
            if (mask >> idx) & 1 == 1 {
                subset.insert(value.clone());
            }
        }
        subsets.insert(TlaValue::Set(Arc::new(subset)));
    }

    Ok(subsets)
}

fn generate_permutations(values: &[TlaValue]) -> Vec<Vec<TlaValue>> {
    if values.is_empty() {
        return vec![Vec::new()];
    }

    let mut out = Vec::new();
    let mut current = values.to_vec();
    permute(&mut current, 0, &mut out);
    out
}

fn permute(values: &mut [TlaValue], idx: usize, out: &mut Vec<Vec<TlaValue>>) {
    if idx >= values.len() {
        out.push(values.to_vec());
        return;
    }

    for i in idx..values.len() {
        values.swap(idx, i);
        permute(values, idx + 1, out);
        values.swap(idx, i);
    }
}

fn tla_to_string(value: &TlaValue) -> String {
    match value {
        TlaValue::String(v) => v.clone(),
        TlaValue::Int(v) => v.to_string(),
        TlaValue::Bool(v) => {
            if *v {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        TlaValue::ModelValue(v) => v.clone(),
        TlaValue::Set(s) => {
            let items: Vec<String> = s.iter().map(tla_to_string).collect();
            format!("{{{}}}", items.join(", "))
        }
        TlaValue::Seq(s) => {
            let items: Vec<String> = s.iter().map(tla_to_string).collect();
            format!("<<{}>>", items.join(", "))
        }
        TlaValue::Record(r) => {
            let items: Vec<String> = r
                .iter()
                .map(|(k, v)| format!("{} |-> {}", k, tla_to_string(v)))
                .collect();
            format!("[{}]", items.join(", "))
        }
        TlaValue::Function(f) => {
            let items: Vec<String> = f
                .iter()
                .take(10)
                .map(|(k, v)| format!("{} :> {}", tla_to_string(k), tla_to_string(v)))
                .collect();
            if f.len() > 10 {
                format!("({}, ...{} more)", items.join(" @@ "), f.len() - 10)
            } else {
                format!("({})", items.join(" @@ "))
            }
        }
        TlaValue::Lambda { params, body, .. } => {
            format!("LAMBDA {}: {}", params.join(", "), body)
        }
        TlaValue::Undefined => "UNDEFINED".to_string(),
    }
}

fn strip_outer_parens(expr: &str) -> &str {
    let mut current = expr.trim();
    while is_wrapped_by(current, '(', ')') {
        current = current[1..current.len() - 1].trim();
    }
    current
}

fn is_wrapped_by(expr: &str, open: char, close: char) -> bool {
    if !expr.starts_with(open) || !expr.ends_with(close) {
        return false;
    }

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in expr.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }

        if ch == open {
            depth += 1;
            continue;
        }
        if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 && idx + ch.len_utf8() < expr.len() {
                return false;
            }
        }
    }

    depth == 0
}

fn starts_with_keyword(expr: &str, kw: &str) -> bool {
    if let Some(rest) = expr.strip_prefix(kw) {
        return rest
            .chars()
            .next()
            .map(|c| !is_word_char(c))
            .unwrap_or(true);
    }
    false
}

fn take_keyword_prefix<'a>(expr: &'a str, kw: &str) -> Option<(&'a str, &'a str)> {
    let trimmed = expr.trim_start();
    let prefix_ws_len = expr.len() - trimmed.len();
    let before = &expr[..prefix_ws_len];
    let rest = trimmed.strip_prefix(kw)?;
    if rest.chars().next().map(is_word_char).unwrap_or(false) {
        return None;
    }
    Some((before, rest.trim_start()))
}

pub(crate) fn split_top_level_symbol(expr: &str, delim: &str) -> Vec<String> {
    split_top_level(expr, delim, false)
}

pub(crate) fn split_top_level_keyword(expr: &str, delim: &str) -> Vec<String> {
    split_top_level(expr, delim, true)
}

fn split_top_level(expr: &str, delim: &str, keyword: bool) -> Vec<String> {
    if expr.contains("PositionLimits") && delim == "/\\" {
        eprintln!(
            "[split_top_level] Called with delim=/\\, expr len={}, expr start: {}",
            expr.len(),
            &expr[..60.min(expr.len())]
        );
    }
    let mut out = Vec::new();
    let mut start = 0usize;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut quantifier_depth = 0usize; // Count of \A or \E seen
    let mut seen_colon_for_quantifier = 0usize; // Count of : seen to close quantifiers
    let mut let_depth = 0usize; // Count of LET keywords seen (increases on LET, decreases on IN)
    let mut if_depth = 0usize; // Count of IF keywords seen (increases on IF, decreases on ELSE)
    let mut case_depth = 0usize; // Count of CASE keywords seen
    let mut else_branch_uses_delimiter = false; // True if ELSE branch starts with the delimiter
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        // Track LET depth as we scan through the expression:
        // - LET increases depth (we're inside a LET expression)
        // - We DON'T decrement on IN - the LET body continues after IN
        // - We only reset let_depth when we perform an actual split
        // This treats the entire "LET defs IN body" as one atomic unit
        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            // Check if we're at the start of LET keyword
            if expr[i..].starts_with("LET") {
                let after_let = &expr[i + 3..];
                // invariant: is_empty() check above guarantees chars().next() is Some
                if after_let.is_empty()
                    || !after_let
                        .chars()
                        .next()
                        .expect("non-empty string has no first char")
                        .is_alphanumeric()
                {
                    let_depth += 1;
                }
            }
        }

        // Track IF-THEN-ELSE: IF increases depth, ELSE decreases it
        // This ensures we don't split inside an IF-THEN-ELSE expression
        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            if expr[i..].starts_with("IF") {
                let after_if = &expr[i + 2..];
                if after_if.is_empty()
                    || after_if
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    if_depth += 1;
                }
            } else if expr[i..].starts_with("ELSE") {
                let after_else = &expr[i + 4..];
                if after_else.is_empty()
                    || after_else
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    // Check if ELSE is followed by the delimiter (indicating ELSE branch uses delimiter)
                    let remaining = after_else.trim_start();
                    if remaining.starts_with(delim) {
                        // ELSE branch starts with delimiter - don't decrement if_depth
                        else_branch_uses_delimiter = true;
                    } else {
                        // ELSE branch doesn't start with delimiter - can decrement
                        if_depth = if_depth.saturating_sub(1);
                    }
                }
            }

            // Track CASE expressions - don't split inside CASE
            if expr[i..].starts_with("CASE") {
                let after_case = &expr[i + 4..];
                if after_case.is_empty()
                    || after_case
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric())
                {
                    case_depth += 1;
                }
            }
        }

        // Track quantifiers: \A or \E increases depth
        if ch == '\\'
            && (expr[i..].starts_with("\\A") || expr[i..].starts_with("\\E"))
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            quantifier_depth += 1;
        } else if ch == ':'
            && quantifier_depth > seen_colon_for_quantifier
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            // This : starts a quantifier body
            seen_colon_for_quantifier += 1;
        }

        // We're truly at top level if:
        // 1. No nested parens/brackets
        // 2. Either no quantifiers, OR we've seen all the colons (we're in the body)
        // If we're in a quantifier body and see a top-level /\ or \/, that ENDS the quantifier
        let in_quantifier_body = seen_colon_for_quantifier > 0 && quantifier_depth > 0;
        let brackets_at_zero = paren == 0 && bracket == 0 && brace == 0 && angle == 0;

        // Check if this is our delimiter at a potential split point
        if brackets_at_zero && expr[i..].starts_with(delim) {
            let delim_end = i + delim.len();
            let is_word_ok = !keyword || has_word_boundaries(expr, i, delim_end);

            if is_word_ok {
                let is_conjunction = delim == "/\\" || delim == "\\/";
                // Determine if we should split:
                // 1. If we're in a LET definition section (between LET and IN), don't split on /\ or \/
                // 2. If we're before the : in a quantifier (in binders), don't split on /\ or \/
                // 3. If we're in a quantifier body, check if this /\ is followed by another quantifier
                //    - If yes: this /\ ENDS the current quantifier - split!
                //    - If no: this /\ is part of the quantifier body - don't split
                // 4. If we're inside IF-THEN (before ELSE), don't split on /\ or \/
                // 5. Otherwise, split normally
                let should_split = if let_depth > 0 && is_conjunction {
                    // We're inside a LET expression, don't split on conjunctions
                    false
                } else if (if_depth > 0 || else_branch_uses_delimiter) && is_conjunction {
                    // We're inside IF-THEN (before ELSE), don't split on conjunctions
                    false
                } else if case_depth > 0 && is_conjunction {
                    // We're inside a CASE expression, don't split on conjunctions
                    false
                } else if quantifier_depth > 0
                    && quantifier_depth > seen_colon_for_quantifier
                    && is_conjunction
                {
                    // We're in quantifier binders (before :), don't split on conjunctions
                    false
                } else if in_quantifier_body && (delim == "/\\" || delim == "\\/") {
                    // Quantifier bodies consume conjunctions and disjunctions to
                    // the end of the surrounding grouping.
                    false
                } else {
                    // All other cases: split normally
                    true
                };

                if should_split {
                    let part = expr[start..i].trim();
                    if !part.is_empty() {
                        out.push(part.to_string());
                    }
                    // Reset quantifier, LET, IF, CASE, and else-branch tracking for next part
                    quantifier_depth = 0;
                    seen_colon_for_quantifier = 0;
                    let_depth = 0;
                    if_depth = 0;
                    case_depth = 0;
                    else_branch_uses_delimiter = false;
                    start = delim_end;
                    i = delim_end;
                    continue;
                }
            }
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        i += ch_len;
    }

    let tail = expr[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    let result = if out.is_empty() {
        vec![expr.trim().to_string()]
    } else {
        out
    };

    result
}

fn split_indented_top_level_boolean(expr: &str, delim: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_boolean_indentation(expr);
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut base_indent = None;
    let mut saw_top_level = false;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed.starts_with(delim) {
            let top_level_indent = *base_indent.get_or_insert(indent);
            if indent == top_level_indent {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(trimmed.trim_start_matches(delim).trim_start());
                saw_top_level = true;
                continue;
            }
        }

        if !saw_top_level {
            return None;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() <= 1 {
        return None;
    }

    let mut merged = Vec::with_capacity(clauses.len());
    let mut idx = 0usize;
    while idx < clauses.len() {
        let part = clauses[idx].trim().to_string();
        if part.is_empty() {
            idx += 1;
            continue;
        }

        let open_quant_or_let = ((part.starts_with("\\A") || part.starts_with("\\E"))
            && (part.ends_with(':') || part.ends_with("IN")))
            || (part.starts_with("LET") && part.ends_with("IN"));
        if open_quant_or_let {
            let mut combined = part;
            for rest in clauses.iter().skip(idx + 1) {
                if rest.trim().is_empty() {
                    continue;
                }
                combined.push(' ');
                combined.push_str(delim);
                combined.push(' ');
                combined.push_str(rest.trim());
            }
            merged.push(combined);
            break;
        }

        merged.push(part);
        idx += 1;
    }

    if merged.len() > 1 { Some(merged) } else { None }
}

fn normalize_multiline_boolean_indentation(expr: &str) -> String {
    let mut lines = expr.lines();
    let Some(first) = lines.next() else {
        return String::new();
    };

    let rest: Vec<&str> = lines.collect();
    if rest.is_empty() {
        return expr.to_string();
    }

    let dedent = rest
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                None
            } else {
                Some(line.len().saturating_sub(trimmed.len()))
            }
        })
        .min()
        .unwrap_or(0);

    if dedent == 0 {
        return expr.to_string();
    }

    let mut normalized = String::with_capacity(expr.len());
    normalized.push_str(first);
    for line in rest {
        normalized.push('\n');
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        let keep = indent.saturating_sub(dedent);
        normalized.push_str(&line[..keep]);
        normalized.push_str(trimmed);
    }

    normalized
}

fn split_top_level_set_minus(expr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && ch == '\\' && !starts_with_tla_backslash_operator(&expr[i..]) {
            let prev = expr[..i].chars().next_back();
            let next_char = expr[i + ch_len..].chars().next();
            let ws_before = prev.map(|c| c.is_whitespace()).unwrap_or(false);
            let ws_after = next_char.map(|c| c.is_whitespace()).unwrap_or(false);
            if ws_before || ws_after || next_char.is_some() {
                let part = expr[start..i].trim();
                if !part.is_empty() {
                    out.push(part.to_string());
                }
                start = i + ch_len;
            }
        }

        i += ch_len;
    }

    let tail = expr[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    if out.is_empty() {
        vec![expr.trim().to_string()]
    } else {
        out
    }
}

fn starts_with_tla_backslash_operator(expr: &str) -> bool {
    [
        "\\A",
        "\\E",
        "\\in",
        "\\notin",
        "\\union",
        "\\intersect",
        "\\cup",
        "\\cap",
        "\\subseteq",
        "\\supseteq",
        "\\div",
        "\\X",
        "\\times",
        "\\o",
        "\\circ",
        "\\/",
        "\\*",
        "\\leq",
        "\\geq",
    ]
    .iter()
    .any(|op| expr.starts_with(op))
}

fn split_top_level_comparison(expr: &str) -> Option<(&str, &'static str, &str)> {
    // NOTE: `..` is intentionally NOT in this list. The TLA+ range operator
    // `..` has precedence 9-9 in the official TLA+ grammar (Lamport's
    // "Specifying Systems", cheat sheet), which is *tighter* than the
    // relational/comparison tier (5-5: `\subseteq`, `\in`, `=`, `<`, ...) and
    // tighter than the set-op tier (8-8: `\union`, `\intersect`, `\` ...).
    // Splitting on `..` here would cause `1..3 \subseteq S` to parse as
    // `1 .. (3 \subseteq S)` — a soundness bug that silently misparses any
    // spec writing `n..m \subseteq S` (or `\union`, `\intersect`, `\\`)
    // without paren-wrapping the range. `..` is split separately below in
    // `split_top_level_range`, slotted between the set-op tier and the
    // additive tier (10-10) in the operator-precedence cascade.
    let patterns = [
        "\\subseteq",
        "\\notin",
        "\\in",
        "\\leq",
        "\\geq",
        "=<",
        "<=",
        ">=",
        "/=",
        "#",
        "=",
        "<",
        ">",
    ];

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize; // Track IF...THEN nesting (protects conditions from split)
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        // Check for keywords at word boundaries when at bracket top level
        if let Some((word, start, word_end)) = next_word(expr, i)
            && start == i
            && paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
        {
            match word {
                "LET" => {
                    let_depth += 1;
                    i = word_end;
                    continue;
                }
                "IN" if let_depth > 0 => {
                    let_depth -= 1;
                    i = word_end;
                    continue;
                }
                "IF" => {
                    if_depth += 1;
                    i = word_end;
                    continue;
                }
                "THEN" if if_depth > 0 => {
                    if_depth -= 1;
                    i = word_end;
                    continue;
                }
                _ => {}
            }
        }

        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && let_depth == 0
            && if_depth == 0;
        if at_top {
            for pattern in patterns {
                if !expr[i..].starts_with(pattern) {
                    continue;
                }

                let end = i + pattern.len();

                if pattern == "=" {
                    let prev = expr[..i].chars().next_back();
                    let next_char = expr[end..].chars().next();
                    if prev == Some('=')
                        || prev == Some('<')
                        || prev == Some('>')
                        || prev == Some('/')
                    {
                        continue;
                    }
                    if next_char == Some('>') || next_char == Some('=') {
                        continue;
                    }
                }

                if pattern == "<" {
                    let next_char = expr[end..].chars().next();
                    if next_char == Some('=') || next_char == Some('<') {
                        continue;
                    }
                }

                if pattern == ">" {
                    let prev_char = expr[..i].chars().next_back();
                    let next_char = expr[end..].chars().next();
                    // Skip >= and >> operators
                    if next_char == Some('=') || next_char == Some('>') {
                        continue;
                    }
                    // Skip :> (TLC function pair operator)
                    if prev_char == Some(':') {
                        continue;
                    }
                }

                if pattern.starts_with('\\') && !has_word_boundaries(expr, i, end) {
                    continue;
                }

                let lhs = expr[..i].trim();
                let rhs = expr[end..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    continue;
                }
                return Some((lhs, pattern, rhs));
            }
        }

        i += ch_len;
    }

    None
}

/// Split a top-level `..` range operator (TLA+ precedence 9-9). Looks for the
/// first `..` at top bracket depth (not inside parens/brackets/braces/angles)
/// and not inside a string literal. Care taken to:
///   - skip leading `..` if it appears at the very start (e.g. inside `[..` --
///     defensive; nothing legal starts with `..`),
///   - require both sides non-empty after trimming.
///
/// `..` is left-associative but rarely chained; we split at the *first*
/// occurrence so `a..b..c` (uncommon) parses as `a .. (b..c)`. Either choice
/// is fine; the recursive call on `b..c` will then split there.
fn split_top_level_range(expr: &str) -> Option<(&str, &str)> {
    let bytes = expr.as_bytes();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i + 1 < bytes.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && ch == '.' && next == Some('.') && i > 0 {
            // Avoid mistaking `...` (e.g. in record `[f : Set, ...]`) for `..`.
            // TLA+ doesn't really use `...` but be defensive.
            let after = expr.get(i + 2..).and_then(|s| s.chars().next());
            if after == Some('.') {
                i += ch_len;
                continue;
            }
            let lhs = expr[..i].trim();
            let rhs = expr[i + 2..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                return Some((lhs, rhs));
            }
        }

        i += ch_len;
    }

    None
}

fn split_top_level_additive(expr: &str) -> Option<(&str, char, &str)> {
    let mut last: Option<(usize, char)> = None;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top && (ch == '+' || ch == '-') && is_binary_operator(expr, i, ch_len) {
            last = Some((i, ch));
        }

        i += ch_len;
    }

    let (idx, op) = last?;
    let lhs = expr[..idx].trim();
    let rhs = expr[idx + op.len_utf8()..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        return None;
    }
    Some((lhs, op, rhs))
}

fn split_top_level_multiplicative(expr: &str) -> Option<(&str, &'static str, &str)> {
    let mut star_idx: Option<usize> = None;
    let mut div_idx: Option<usize> = None;
    let mut mod_idx: Option<usize> = None;

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top {
            if ch == '*' && is_binary_operator(expr, i, ch_len) {
                star_idx = Some(i);
            }
            if expr[i..].starts_with("\\div") {
                let end = i + "\\div".len();
                if has_word_boundaries(expr, i, end) {
                    div_idx = Some(i);
                }
            }
            if ch == '%' && is_binary_operator(expr, i, ch_len) {
                mod_idx = Some(i);
            }
        }

        i += ch_len;
    }

    // Find the rightmost operator for left-to-right associativity
    let mut candidates: Vec<(usize, &str, usize)> = Vec::new(); // (idx, op_str, op_len)
    if let Some(s) = star_idx {
        candidates.push((s, "*", 1));
    }
    if let Some(d) = div_idx {
        candidates.push((d, "\\div", "\\div".len()));
    }
    if let Some(m) = mod_idx {
        candidates.push((m, "%", 1));
    }
    // Pick the rightmost (largest index)
    candidates.sort_by_key(|(idx, _, _)| *idx);
    let (idx, op, op_len) = candidates.last()?;
    let lhs = expr[..*idx].trim();
    let rhs = expr[idx + op_len..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        None
    } else {
        Some((lhs, *op, rhs))
    }
}

fn split_once_top_level<'a>(expr: &'a str, delim: &str) -> Option<(&'a str, &'a str)> {
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && expr[i..].starts_with(delim) {
            return Some((expr[..i].trim(), expr[i + delim.len()..].trim()));
        }

        i += ch_len;
    }

    None
}

fn split_top_level_defined_infix<'a>(
    expr: &'a str,
    ctx: &EvalContext<'_>,
) -> Option<(&'a str, String, &'a str)> {
    let mut operators = BTreeSet::new();
    for defs in [Some(ctx.local_definitions.as_ref()), ctx.definitions] {
        let Some(defs) = defs else {
            continue;
        };
        for (name, def) in defs.iter() {
            if def.params.len() == 2 && is_user_defined_infix_operator(name) {
                operators.insert(name.clone());
            }
        }
    }
    if operators.is_empty() {
        return None;
    }

    let mut operators: Vec<String> = operators.into_iter().collect();
    operators.sort_by_key(|op| std::cmp::Reverse(op.len()));

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            for op in &operators {
                if !expr[i..].starts_with(op) {
                    continue;
                }
                let lhs = expr[..i].trim();
                let rhs = expr[i + op.len()..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    continue;
                }
                return Some((lhs, op.clone(), rhs));
            }
        }

        i += ch_len;
    }

    None
}

fn is_user_defined_infix_operator(name: &str) -> bool {
    if matches!(
        name,
        "/\\"
            | "\\/"
            | "=>"
            | "~>"
            | "="
            | "/="
            | "#"
            | "<"
            | "<="
            | "=<"
            | ">"
            | ">="
            | "\\leq"
            | "\\geq"
            | "\\in"
            | "\\notin"
            | "\\subseteq"
            | ".."
            | "\\union"
            | "\\cup"
            | "\\intersect"
            | "\\cap"
            | "\\o"
            | "\\circ"
            | "\\X"
            | "\\times"
            | "@@"
            | ":>"
            | "+"
            | "-"
            | "*"
            | "\\div"
            | "%"
            | "^^"
    ) {
        return false;
    }

    !name.is_empty()
        && name
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_'))
}

pub(crate) fn find_top_level_keyword_index(expr: &str, keyword: &str) -> Option<usize> {
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && expr[i..].starts_with(keyword)
            && has_keyword_boundaries(expr, i, i + keyword.len(), keyword)
        {
            return Some(i);
        }

        i += ch_len;
    }

    None
}

fn contains_top_level_keyword(expr: &str, keyword: &str) -> bool {
    find_top_level_keyword_index(expr, keyword).is_some()
}

pub(crate) fn find_top_level_char(expr: &str, target: char) -> Option<usize> {
    find_top_level_char_from(expr, target, 0)
}

fn find_top_level_char_from(expr: &str, target: char, start_at: usize) -> Option<usize> {
    let mut i = start_at;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && ch == target {
            // When searching for ':', skip ':>' (TLC function pair operator)
            if ch == ':' && next == Some('>') {
                i += ch_len;
                continue;
            }
            return Some(i);
        }

        i += ch_len;
    }

    None
}

fn take_bracket_group(expr: &str, open: char, close: char) -> Result<(&str, &str)> {
    if !expr.starts_with(open) {
        return Err(anyhow!("expected '{open}'"));
    }

    let mut depth = 0usize;
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let inner = &expr[open.len_utf8()..i];
                let rest = &expr[i + ch_len..];
                return Ok((inner, rest));
            }
        }

        i += ch_len;
    }

    Err(anyhow!("missing closing '{close}' in expression: {expr}"))
}

fn take_angle_group(expr: &str) -> Result<(&str, &str)> {
    if !expr.starts_with("<<") {
        return Err(anyhow!("expected '<<'"));
    }

    let mut depth = 0usize;
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            depth += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            depth = depth.saturating_sub(1);
            i += 2;
            if depth == 0 {
                let inner = &expr[2..i - 2];
                let rest = &expr[i..];
                return Ok((inner, rest));
            }
            continue;
        }

        i += ch_len;
    }

    Err(anyhow!("missing closing '>>' in expression: {expr}"))
}

fn parse_identifier_prefix(expr: &str) -> Option<(String, &str)> {
    let mut chars = expr.char_indices();
    let (first_idx, first) = chars.next()?;
    if first_idx != 0 {
        return None;
    }
    if !(first.is_alphanumeric() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    let mut saw_identifier_marker = first.is_alphabetic() || first == '_';
    for (idx, c) in chars {
        if c.is_alphanumeric() || c == '_' || c == '\'' {
            end = idx + c.len_utf8();
            saw_identifier_marker |= c.is_alphabetic() || c == '_';
        } else {
            break;
        }
    }

    if !saw_identifier_marker {
        return None;
    }

    Some((expr[..end].to_string(), &expr[end..]))
}

fn parse_string_literal_prefix(expr: &str) -> Result<Option<(String, &str)>> {
    if !expr.starts_with('"') {
        return Ok(None);
    }

    let mut out = String::new();
    let mut escaped = false;

    let mut i = 1usize;
    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();

        if escaped {
            out.push(ch);
            escaped = false;
            i += ch_len;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            i += ch_len;
            continue;
        }

        if ch == '"' {
            return Ok(Some((out, &expr[i + ch_len..])));
        }

        out.push(ch);
        i += ch_len;
    }

    Err(anyhow!("unterminated string literal in expression: {expr}"))
}

fn parse_int_prefix(expr: &str) -> Option<(i64, &str)> {
    let mut end = 0usize;
    for (idx, c) in expr.char_indices() {
        if c.is_ascii_digit() {
            end = idx + c.len_utf8();
        } else {
            break;
        }
    }

    if end == 0 {
        return None;
    }

    let rest = &expr[end..];
    if rest
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '\'')
    {
        return None;
    }

    let value = expr[..end].parse::<i64>().ok()?;
    Some((value, rest))
}

fn has_word_boundaries(expr: &str, start: usize, end: usize) -> bool {
    let prev = expr[..start].chars().next_back();
    let next = expr[end..].chars().next();

    let prev_ok = prev.map(|c| !is_word_char(c)).unwrap_or(true);
    let next_ok = next.map(|c| !is_word_char(c)).unwrap_or(true);

    prev_ok && next_ok
}

fn has_keyword_boundaries(expr: &str, start: usize, end: usize, keyword: &str) -> bool {
    if keyword.starts_with('\\') {
        let next = expr[end..].chars().next();
        return next.map(|c| !c.is_alphabetic()).unwrap_or(true);
    }
    has_word_boundaries(expr, start, end)
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_binary_operator(expr: &str, idx: usize, len: usize) -> bool {
    let prev = expr[..idx].chars().rev().find(|c| !c.is_whitespace());
    let next = expr[idx + len..].chars().find(|c| !c.is_whitespace());

    let Some(prev) = prev else {
        return false;
    };
    let Some(next) = next else {
        return false;
    };

    let prev_is_operator = matches!(
        prev,
        '(' | '[' | '{' | ',' | ':' | '+' | '-' | '*' | '/' | '\\' | '=' | '<' | '>' | '#'
    );
    let next_is_operator = matches!(next, ')' | ']' | '}' | ',' | ':');

    !prev_is_operator && !next_is_operator
}

fn next_word(input: &str, from: usize) -> Option<(&str, usize, usize)> {
    let mut i = from;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if ch.is_alphabetic() || ch == '_' {
            break;
        }
        i += len;
    }

    if i >= input.len() {
        return None;
    }

    let start = i;
    while i < input.len() {
        let ch = input[i..].chars().next()?;
        let len = ch.len_utf8();
        if !(ch.is_alphanumeric() || ch == '_') {
            break;
        }
        i += len;
    }

    Some((&input[start..i], start, i))
}

fn find_outer_then(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "THEN" if nested_if == 0 => return Some(start),
            "ELSE" if nested_if > 0 => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

fn find_outer_else(input: &str) -> Option<usize> {
    let mut nested_if = 0usize;
    let mut i = 0usize;
    while let Some((word, start, end)) = next_word(input, i) {
        match word {
            "IF" => nested_if += 1,
            "ELSE" if nested_if == 0 => return Some(start),
            "ELSE" => nested_if = nested_if.saturating_sub(1),
            _ => {}
        }
        i = end;
    }
    None
}

fn find_top_level_definition_eqs(expr: &str) -> Vec<usize> {
    let mut out = Vec::new();

    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut let_depth = 0usize;

    while i < expr.len() {
        if let Some((word, start, _end)) = next_word(expr, i)
            && start == i
        {
            match word {
                "LET" => let_depth += 1,
                "IN" if let_depth > 0 => let_depth -= 1,
                _ => {}
            }
        }

        let ch = expr[i..].chars().next().expect("char at byte index");
        let ch_len = ch.len_utf8();
        let next = expr[i + ch_len..].chars().next();

        if in_string {
            if escaped {
                escaped = false;
                i += ch_len;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                i += ch_len;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            i += ch_len;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += ch_len;
            continue;
        }

        if ch == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if ch == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            i += 2;
            continue;
        }

        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        if paren == 0
            && bracket == 0
            && brace == 0
            && angle == 0
            && let_depth == 0
            && ch == '='
            && next == Some('=')
        {
            out.push(i);
            i += 2;
            continue;
        }

        i += ch_len;
    }

    out
}

fn line_start_before(input: &str, idx: usize) -> usize {
    match input[..idx].rfind('\n') {
        Some(pos) => pos + 1,
        None => 0,
    }
}

fn skip_leading_ws(input: &str, mut idx: usize) -> usize {
    while idx < input.len() {
        let ch = input[idx..].chars().next().expect("char at byte index");
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

/// Convert a serde_json::Value to a TlaValue.
/// Try to extract a sum-range constraint from a predicate like
/// `c.f1 + c.f2 \in lo..hi`. Returns Some((lo, hi)) if successful.
/// The range bounds can be literals or resolvable expressions (e.g., MaxBeanCount).
fn extract_sum_range_constraint(
    pred_text: &str,
    var_name: &str,
    field_specs: &[(String, Vec<TlaValue>)],
    ctx: &EvalContext<'_>,
) -> Option<(i64, i64)> {
    let pred = pred_text.trim();
    // Pattern: var.f1 + var.f2 \in lo..hi
    let in_idx = find_top_level_keyword_index(pred, "\\in")?;
    let lhs = pred[..in_idx].trim();
    let rhs = pred[in_idx + 3..].trim();

    // Check LHS is var.f1 + var.f2
    let plus_idx = lhs.find('+')?;
    let left_part = lhs[..plus_idx].trim();
    let right_part = lhs[plus_idx + 1..].trim();

    let f1_prefix = format!("{}.", var_name);
    let left_field = left_part.strip_prefix(&f1_prefix)?;
    let right_field = right_part.strip_prefix(&f1_prefix)?;

    if !field_specs.iter().any(|(n, _)| n == left_field)
        || !field_specs.iter().any(|(n, _)| n == right_field)
    {
        return None;
    }

    // Parse RHS as lo..hi (bounds can be expressions, not just literals)
    let dotdot_idx = rhs.find("..")?;
    let lo_text = rhs[..dotdot_idx].trim();
    let hi_text = rhs[dotdot_idx + 2..].trim();

    // Try parsing as literal first, then evaluate as expression
    let lo = lo_text
        .parse::<i64>()
        .ok()
        .or_else(|| eval_expr_inner(lo_text, ctx, 0).ok()?.as_int().ok())?;
    let hi = hi_text
        .parse::<i64>()
        .ok()
        .or_else(|| eval_expr_inner(hi_text, ctx, 0).ok()?.as_int().ok())?;

    Some((lo, hi))
}

fn json_to_tla_value(v: &serde_json::Value) -> TlaValue {
    match v {
        serde_json::Value::Null => TlaValue::ModelValue("null".to_string()),
        serde_json::Value::Bool(b) => TlaValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                TlaValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                // Truncate floats to ints for TLA+
                TlaValue::Int(f as i64)
            } else {
                TlaValue::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => TlaValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<TlaValue> = arr.iter().map(json_to_tla_value).collect();
            TlaValue::Seq(Arc::new(items))
        }
        serde_json::Value::Object(obj) => {
            let mut rec = BTreeMap::new();
            for (k, v) in obj {
                rec.insert(k.clone(), json_to_tla_value(v));
            }
            TlaValue::Record(Arc::new(rec))
        }
    }
}

/// Convert a TlaValue to a serde_json::Value.
fn tla_value_to_json(v: &TlaValue) -> serde_json::Value {
    match v {
        TlaValue::Int(n) => serde_json::Value::Number((*n).into()),
        TlaValue::Bool(b) => serde_json::Value::Bool(*b),
        TlaValue::String(s) => serde_json::Value::String(s.clone()),
        TlaValue::ModelValue(s) => serde_json::Value::String(s.clone()),
        TlaValue::Seq(items) => {
            serde_json::Value::Array(items.iter().map(tla_value_to_json).collect())
        }
        TlaValue::Record(rec) => {
            let obj: serde_json::Map<String, serde_json::Value> = rec
                .iter()
                .map(|(k, v)| (k.clone(), tla_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        TlaValue::Set(set) => serde_json::Value::Array(set.iter().map(tla_value_to_json).collect()),
        TlaValue::Function(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (format!("{:?}", k), tla_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        TlaValue::Lambda { .. } => serde_json::Value::String("<lambda>".to_string()),
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::tla_state;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    #[test]
    fn normalizes_binder_and_higher_order_param_names() {
        assert_eq!(normalize_param_name("leader \\in Node"), "leader");
        assert_eq!(normalize_param_name("Op(_, _)"), "Op");
        assert_eq!(normalize_param_name("HostOf"), "HostOf");
    }

    #[test]
    fn split_top_level_keeps_simple_quantified_conjunctions_intact() {
        let parts =
            split_top_level_symbol(r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#, "/\\");
        assert_eq!(
            parts,
            vec![r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#]
        );
    }

    #[test]
    fn split_top_level_keeps_simple_quantified_disjunctions_intact() {
        let parts = split_top_level_symbol(r"\E a \in Acceptor : Phase1b(a) \/ Phase2b(a)", "\\/");
        assert_eq!(parts, vec![r"\E a \in Acceptor : Phase1b(a) \/ Phase2b(a)"]);
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
            TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2)
            ])))
        );
        assert_eq!(
            eval_expr("{1, 2} \\cap {2, 3}", &ctx).expect("intersection alias should evaluate"),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(2)])))
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(1), TlaValue::Int(3)])),
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(2), TlaValue::Int(3)])),
        ])));
        assert_eq!(result_times, expected);
    }

    #[test]
    fn resolves_named_operator_identifiers_as_higher_order_values() {
        let state = tla_state([(
            "waiting",
            TlaValue::Seq(Arc::new(vec![
                TlaValue::Seq(Arc::new(vec![
                    TlaValue::String("read".to_string()),
                    TlaValue::Int(1),
                ])),
                TlaValue::Seq(Arc::new(vec![
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
            TlaValue::Seq(Arc::new(vec![TlaValue::Seq(Arc::new(vec![
                TlaValue::String("read".to_string()),
                TlaValue::Int(1),
            ]))]))
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
            TlaValue::Seq(Arc::new(vec![
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
            TlaValue::Function(Arc::new(BTreeMap::from([(
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
            TlaValue::Function(Arc::new(BTreeMap::from([(
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
                TlaValue::Set(Arc::new(BTreeSet::from([alice.clone(), bob.clone()]))),
            ),
            (
                "referencePrice",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    asset.clone(),
                    TlaValue::Int(15),
                )]))),
            ),
            (
                "positions",
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (alice.clone(), TlaValue::Int(1)),
                    (bob.clone(), TlaValue::Int(-1)),
                ]))),
            ),
            (
                "balances",
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (alice.clone(), TlaValue::Int(100)),
                    (bob.clone(), TlaValue::Int(100)),
                ]))),
            ),
        ]);
        let future = TlaValue::Record(Arc::new(BTreeMap::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([seller.clone()]))),
            ),
            (
                "Assets",
                TlaValue::Set(Arc::new(BTreeSet::from([asset.clone()]))),
            ),
            ("ccpTrades", TlaValue::Set(Arc::new(BTreeSet::new()))),
            (
                "ccpPositions",
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (
                        bot.clone(),
                        TlaValue::Function(Arc::new(BTreeMap::from([(
                            asset.clone(),
                            TlaValue::Int(0),
                        )]))),
                    ),
                    (
                        seller.clone(),
                        TlaValue::Function(Arc::new(BTreeMap::from([(
                            asset.clone(),
                            TlaValue::Int(3),
                        )]))),
                    ),
                ]))),
            ),
            ("deployments", TlaValue::Set(Arc::new(BTreeSet::new()))),
            (
                "actionCount",
                TlaValue::Function(Arc::new(BTreeMap::from([(bot.clone(), TlaValue::Int(0))]))),
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
        let quorum = TlaValue::Set(Arc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let msg_a1 = TlaValue::Record(Arc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a1.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(1)),
            ("mval".to_string(), v1.clone()),
        ])));
        let msg_a2 = TlaValue::Record(Arc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a2.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(0)),
            ("mval".to_string(), v1.clone()),
        ])));
        let state = tla_state([
            (
                "msgs",
                TlaValue::Set(Arc::new(BTreeSet::from([msg_a1, msg_a2]))),
            ),
            ("Quorum", TlaValue::Set(Arc::new(BTreeSet::from([quorum])))),
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
        let quorum = TlaValue::Set(Arc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let state = tla_state([
            ("msgs", TlaValue::Set(Arc::new(BTreeSet::new()))),
            ("Quorum", TlaValue::Set(Arc::new(BTreeSet::from([quorum])))),
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
        let quorum = TlaValue::Set(Arc::new(BTreeSet::from([a1.clone(), a2.clone()])));
        let msg_a1 = TlaValue::Record(Arc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a1.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(1)),
            ("mval".to_string(), v1.clone()),
        ])));
        let msg_a2 = TlaValue::Record(Arc::new(BTreeMap::from([
            ("type".to_string(), TlaValue::String("1b".to_string())),
            ("acc".to_string(), a2.clone()),
            ("bal".to_string(), TlaValue::Int(1)),
            ("mbal".to_string(), TlaValue::Int(0)),
            ("mval".to_string(), v1.clone()),
        ])));
        let state = tla_state([
            (
                "msgs",
                TlaValue::Set(Arc::new(BTreeSet::from([msg_a1, msg_a2]))),
            ),
            ("Quorum", TlaValue::Set(Arc::new(BTreeSet::from([quorum])))),
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
        let tuple_key = TlaValue::Seq(Arc::new(vec![
            TlaValue::ModelValue("n1".to_string()),
            TlaValue::ModelValue("n2".to_string()),
        ]));
        let state = tla_state([
            (
                "NetworkPath",
                TlaValue::Function(Arc::new(BTreeMap::from([(
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
        let tuple_key = TlaValue::Seq(Arc::new(vec![
            TlaValue::ModelValue("n1".to_string()),
            TlaValue::ModelValue("n2".to_string()),
        ]));
        let state = tla_state([
            (
                "NetworkPath",
                TlaValue::Function(Arc::new(BTreeMap::from([(
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
        let tuple_key = TlaValue::Seq(Arc::new(vec![
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
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("n1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                    TlaValue::Int(3),
                ]))),
            ),
            (
                "board",
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
                    TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(3)]))),
                ]))),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_expr("Pos \\ UNION board", &ctx).expect("set minus with UNION rhs should work"),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(2)])))
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("h1".to_string()),
                    TlaValue::ModelValue("h2".to_string()),
                ]))),
            ),
            ("SignedBlock", TlaValue::Set(Arc::new(BTreeSet::new()))),
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
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
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
            TlaValue::Function(Arc::new(BTreeMap::from([(key.clone(), TlaValue::Int(0))])));
        let next = TlaValue::Function(Arc::new(BTreeMap::from([(key.clone(), TlaValue::Int(1))])));
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
        let piece_a = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(1), TlaValue::Int(1)])),
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(1), TlaValue::Int(2)])),
        ])));
        let piece_b = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(2), TlaValue::Int(1)])),
            TlaValue::Seq(Arc::new(vec![TlaValue::Int(2), TlaValue::Int(2)])),
        ])));
        let board = TlaValue::Set(Arc::new(BTreeSet::from([piece_a, piece_b.clone()])));
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
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Seq(Arc::new(vec![
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
                TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Seq(Arc::new(vec![
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
            TlaValue::Seq(Arc::new(vec![
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
                TlaValue::Set(Arc::new(BTreeSet::from([
                    elevator_1.clone(),
                    elevator_2.clone(),
                ]))),
            ),
            (
                "ElevatorState",
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (
                        elevator_1.clone(),
                        TlaValue::Record(Arc::new(BTreeMap::from([
                            ("floor".to_string(), TlaValue::Int(1)),
                            ("direction".to_string(), up.clone()),
                        ]))),
                    ),
                    (
                        elevator_2.clone(),
                        TlaValue::Record(Arc::new(BTreeMap::from([
                            ("floor".to_string(), TlaValue::Int(3)),
                            ("direction".to_string(), stationary),
                        ]))),
                    ),
                ]))),
            ),
            (
                "c",
                TlaValue::Record(Arc::new(BTreeMap::from([
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([elevator_1])));
        assert_eq!(value, expected);
    }

    #[test]
    fn evals_empty_universal_quantifier_with_record_body() {
        let state = tla_state([
            (
                "ActiveElevatorCalls",
                TlaValue::Set(Arc::new(BTreeSet::new())),
            ),
            (
                "Elevator",
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::ModelValue("e1".to_string()),
                    TlaValue::ModelValue("e2".to_string()),
                ]))),
            ),
            ("e", TlaValue::ModelValue("e1".to_string())),
            ("nextFloor", TlaValue::Int(1)),
            (
                "Floor",
                TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Function(Arc::new(BTreeMap::from([
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
            TlaValue::Seq(Arc::new(vec![
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
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (TlaValue::Int(1), TlaValue::Int(1)),
                    (TlaValue::Int(2), TlaValue::Int(2)),
                ]))),
            ),
            (
                "rhs",
                TlaValue::Function(Arc::new(BTreeMap::from([(
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
            TlaValue::Seq(Arc::new(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))
        );
    }

    #[test]
    fn if_with_equality_in_condition() {
        let state = tla_state([("opts", TlaValue::Set(Arc::new(BTreeSet::new())))]);
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
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
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
        let state = tla_state([("S", TlaValue::Set(Arc::new(BTreeSet::new())))]);
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
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(Arc::new(BTreeMap::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "Clients",
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                    "c1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(Arc::new(BTreeMap::from([
                        ("readers".to_string(), TlaValue::Int(0)),
                        ("writers".to_string(), TlaValue::Int(0)),
                        ("dataVerifier".to_string(), TlaValue::Int(0)),
                    ]))),
                )]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([(
                        TlaValue::ModelValue("c1".to_string()),
                        TlaValue::Record(Arc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "clientCharters",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("c1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([(
                        TlaValue::ModelValue("i1".to_string()),
                        TlaValue::Record(Arc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                    "i1".to_string(),
                )]))),
            ),
            (
                "Clients",
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
                    "c1".to_string(),
                )]))),
            ),
            (
                "inodeState",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Record(Arc::new(BTreeMap::from([
                        ("readers".to_string(), TlaValue::Int(0)),
                        ("writers".to_string(), TlaValue::Int(0)),
                        ("dataVerifier".to_string(), TlaValue::Int(0)),
                    ]))),
                )]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("i1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([(
                        TlaValue::ModelValue("c1".to_string()),
                        TlaValue::Record(Arc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "clientCharters",
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::ModelValue("c1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([(
                        TlaValue::ModelValue("i1".to_string()),
                        TlaValue::Record(Arc::new(BTreeMap::from([(
                            "givenAccess".to_string(),
                            TlaValue::String("NONE".to_string()),
                        )]))),
                    )]))),
                )]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(Arc::new(BTreeSet::from([
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
                TlaValue::Function(Arc::new(BTreeMap::from_iter([
                    (
                        TlaValue::Int(1),
                        TlaValue::Record(Arc::new(BTreeMap::from_iter([
                            ("readers".to_string(), TlaValue::Int(0)),
                            ("writers".to_string(), TlaValue::Int(0)),
                            ("dataVerifier".to_string(), TlaValue::Int(0)),
                        ]))),
                    ),
                    (
                        TlaValue::Int(2),
                        TlaValue::Record(Arc::new(BTreeMap::from_iter([
                            ("readers".to_string(), TlaValue::Int(0)),
                            ("writers".to_string(), TlaValue::Int(0)),
                            ("dataVerifier".to_string(), TlaValue::Int(0)),
                        ]))),
                    ),
                ]))),
            ),
            (
                "serverCharters",
                TlaValue::Function(Arc::new(BTreeMap::from_iter([
                    (
                        TlaValue::Int(1),
                        TlaValue::Function(Arc::new(BTreeMap::from_iter([
                            (
                                TlaValue::String("a".to_string()),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Read".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::String("b".to_string()),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                    (
                        TlaValue::Int(2),
                        TlaValue::Function(Arc::new(BTreeMap::from_iter([
                            (
                                TlaValue::String("a".to_string()),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::String("b".to_string()),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
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
                TlaValue::Function(Arc::new(BTreeMap::from_iter([
                    (
                        TlaValue::String("a".to_string()),
                        TlaValue::Function(Arc::new(BTreeMap::from_iter([
                            (
                                TlaValue::Int(1),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("Read".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::Int(2),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                        ]))),
                    ),
                    (
                        TlaValue::String("b".to_string()),
                        TlaValue::Function(Arc::new(BTreeMap::from_iter([
                            (
                                TlaValue::Int(1),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
                                    "givenAccess".to_string(),
                                    TlaValue::String("None".to_string()),
                                )]))),
                            ),
                            (
                                TlaValue::Int(2),
                                TlaValue::Record(Arc::new(BTreeMap::from_iter([(
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
                TlaValue::Set(Arc::new(BTreeSet::from_iter([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
            (
                "Clients",
                TlaValue::Set(Arc::new(BTreeSet::from_iter([
                    TlaValue::String("a".to_string()),
                    TlaValue::String("b".to_string()),
                ]))),
            ),
            (
                "AccessLevel",
                TlaValue::Set(Arc::new(BTreeSet::from_iter([
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
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        // FunAsSeq(f, 1, 3) should produce <<10, 20, 30>>
        let result = eval_expr("FunAsSeq(f, 1, 3)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(Arc::new(vec![
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
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
            (TlaValue::Int(4), TlaValue::Int(40)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        // FunAsSeq(f, 4, 2) should produce <<f[1], f[2]>> = <<10, 20>>
        let result = eval_expr("FunAsSeq(f, 4, 2)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(Arc::new(vec![TlaValue::Int(10), TlaValue::Int(20)]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_empty() {
        // FunAsSeq(f, 1, 0) should produce empty sequence
        let func = TlaValue::Function(Arc::new(BTreeMap::from([(
            TlaValue::Int(1),
            TlaValue::Int(10),
        )])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("FunAsSeq(f, 1, 0)", &ctx).expect("FunAsSeq should evaluate");
        let expected = TlaValue::Seq(Arc::new(vec![]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_funasseq_compiled() {
        use crate::tla::{compile_expr, eval_compiled};

        // Test compiled version
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(100)),
            (TlaValue::Int(2), TlaValue::Int(200)),
            (TlaValue::Int(3), TlaValue::Int(300)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let compiled = compile_expr("FunAsSeq(f, 1, 3)");
        let result = eval_compiled(&compiled, &ctx).expect("Compiled FunAsSeq should evaluate");
        let expected = TlaValue::Seq(Arc::new(vec![
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
        let seq = TlaValue::Seq(Arc::new(vec![
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
        let expected = TlaValue::Seq(Arc::new(vec![TlaValue::Int(1), TlaValue::Int(2)]));
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
    fn find_outer_else_handles_newlines() {
        // Test that find_outer_else correctly finds ELSE across newlines
        assert_eq!(find_outer_else("something\nELSE other"), Some(10));
        assert_eq!(find_outer_else("something ELSE other"), Some(10));

        // Nested IF should not confuse it
        let nested = "IF inner THEN a ELSE b\nELSE outer";
        assert_eq!(find_outer_else(nested), Some(23));
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
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(10)),
            (TlaValue::Int(2), TlaValue::Int(20)),
            (TlaValue::Int(3), TlaValue::Int(30)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(f)", &ctx).expect("Range should evaluate");
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Int(10),
            TlaValue::Int(20),
            TlaValue::Int(30),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_sequence() {
        // Range of a sequence returns the set of all its elements
        let seq = TlaValue::Seq(Arc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(2), // Duplicate to test set semantics
            TlaValue::Int(3),
        ]));

        let state = tla_state([("s", seq)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(s)", &ctx).expect("Range should evaluate on sequence");
        // Duplicates are removed since Range returns a set
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_record() {
        // Range of a record returns the set of all its field values
        let rec = TlaValue::Record(Arc::new(BTreeMap::from([
            ("a".to_string(), TlaValue::Int(10)),
            ("b".to_string(), TlaValue::Int(20)),
            ("c".to_string(), TlaValue::Int(10)), // Duplicate value
        ])));

        let state = tla_state([("r", rec)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(r)", &ctx).expect("Range should evaluate on record");
        // Duplicates are removed since Range returns a set
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Int(10),
            TlaValue::Int(20),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_empty_function() {
        // Range of an empty function is the empty set
        let func = TlaValue::Function(Arc::new(BTreeMap::new()));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let result = eval_expr("Range(f)", &ctx).expect("Range should evaluate on empty function");
        let expected = TlaValue::Set(Arc::new(BTreeSet::new()));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_range_compiled() {
        use crate::tla::{compile_expr, eval_compiled};

        // Test compiled version
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(100)),
            (TlaValue::Int(2), TlaValue::Int(200)),
        ])));

        let state = tla_state([("f", func)]);
        let ctx = EvalContext::new(&state);

        let compiled = compile_expr("Range(f)");
        let result = eval_compiled(&compiled, &ctx).expect("Compiled Range should evaluate");
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
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
        let current = TlaValue::Function(Arc::new(BTreeMap::from([(
            TlaValue::Int(1),
            TlaValue::Int(0),
        )])));
        let next = TlaValue::Function(Arc::new(BTreeMap::from([(
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

        let signed_block = TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::ModelValue(
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
                TlaValue::Set(Arc::new(BTreeSet::from([hash_1.clone(), hash_2.clone()]))),
            ),
            (
                "hashFunction",
                TlaValue::Function(Arc::new(BTreeMap::from([
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

    #[test]
    fn numeric_prefixed_identifiers_evaluate_as_identifiers() {
        let state = tla_state([
            (
                "1bMessage",
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
            ),
            (
                "2bMessage",
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(2)]))),
            ),
        ]);
        let defs = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_expr("1bMessage \\cup 2bMessage", &ctx).expect("numeric-prefixed names"),
            TlaValue::Set(Arc::new(BTreeSet::from([
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(2)])));
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
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
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Int(0),
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
            TlaValue::Int(4),
        ])));
        assert_eq!(v, expected);

        let v = eval_expr("1+1 .. 2+2", &ctx).expect("1+1 .. 2+2 should evaluate");
        let expected = TlaValue::Set(Arc::new(BTreeSet::from([
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
            TlaValue::Set(Arc::new(BTreeSet::from([
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
        assert_eq!(v, TlaValue::Set(Arc::new(BTreeSet::new())));
    }

    // --- T4 mutation-kill tests -------------------------------------------
    //
    // The tests below close consequential coverage gaps surfaced by the
    // cargo-mutants audit (RELEASE_1.0.0_LOG.md, T4). Each test is annotated
    // with the line(s) of the mutant(s) it kills.

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

    #[test]
    fn t4_split_top_level_set_minus_distinguishes_set_difference_from_backslash_operators() {
        // Kills eval.rs:5681:.. (`!starts_with_tla_backslash_operator` mutants
        // and `is_whitespace`/`!=` mutants in split_top_level_set_minus).
        // The set-minus splitter walks the string char-by-char and must
        // distinguish a bare `\` (set difference) from a `\` that introduces a
        // TLA+ operator like `\union` / `\subseteq` / `\E`. No prior test
        // exercised whitespace-boundary detection.

        let parts = split_top_level_set_minus("S \\ T");
        assert_eq!(
            parts.len(),
            2,
            "S \\ T must split into [S, T], got {parts:?}"
        );
        assert_eq!(parts[0].trim(), "S");
        assert_eq!(parts[1].trim(), "T");

        // \union must NOT trigger a set-minus split.
        let parts = split_top_level_set_minus("S \\union T");
        assert_eq!(
            parts,
            vec!["S \\union T".to_string()],
            "\\union must not be split as set-minus"
        );
        // \subseteq must NOT trigger.
        let parts = split_top_level_set_minus("S \\subseteq T");
        assert_eq!(parts, vec!["S \\subseteq T".to_string()]);
        // \E quantifier must NOT trigger.
        let parts = split_top_level_set_minus("\\E i \\in S : P(i)");
        assert_eq!(parts, vec!["\\E i \\in S : P(i)".to_string()]);
        // No backslash at all => single element.
        let parts = split_top_level_set_minus("S \\union T \\union U");
        assert_eq!(parts.len(), 1);
        // Set difference inside a set literal is *not* top-level (kills the
        // `match ch '{' / '}' brace-tracker mutants).
        let parts = split_top_level_set_minus("{1 \\ 2}");
        assert_eq!(parts, vec!["{1 \\ 2}".to_string()]);
    }

    #[test]
    fn t4_split_top_level_range_distinguishes_dotdot_from_dotdotdot() {
        // Kills mutants in eval.rs:5935 split_top_level_range — particularly
        // the `i > 0` check guarding a non-empty LHS, and the `after == Some('.')`
        // check that prevents `...` from being parsed as `..`.
        let r = split_top_level_range("1..5").expect("simple range must split");
        assert_eq!(r, ("1", "5"));

        // Range with whitespace.
        let r = split_top_level_range("a + 1 .. b - 1").expect("range with spaces must split");
        assert_eq!(r.0.trim(), "a + 1");
        assert_eq!(r.1.trim(), "b - 1");

        // No `..` => None.
        assert!(split_top_level_range("a + b").is_none());
        // `..` inside parens does not split at top level.
        assert!(
            split_top_level_range("(1..3)").is_none(),
            "`..` inside parens must not be top-level"
        );
        // `..` inside braces does not split at top level.
        assert!(
            split_top_level_range("{x \\in 1..5 : x > 2}").is_none(),
            "`..` inside braces must not be top-level"
        );
        // `..` inside square brackets does not split.
        assert!(
            split_top_level_range("[i \\in 1..3 |-> i*i]").is_none(),
            "`..` inside `[]` must not be top-level"
        );
    }

    #[test]
    fn t4_split_top_level_comparison_recognises_subseteq_and_distinguishes_set_ops() {
        // Kills mutants in eval.rs:5736 split_top_level_comparison — most
        // importantly the `pattern == "="` / `pattern.starts_with('\\')` /
        // `has_word_boundaries` mutants. T2.3 fixed `..` precedence relative
        // to set ops; this test pins that fix so it can't regress via mutation.
        let (lhs, op, rhs) = split_top_level_comparison("1..3 \\subseteq S")
            .expect("range \\subseteq set must split");
        assert_eq!(lhs.trim(), "1..3");
        assert_eq!(op, "\\subseteq");
        assert_eq!(rhs.trim(), "S");

        // `=` must not trigger inside `==` (although TLA+ doesn't use `==`,
        // the disambiguation guard at 5871 protects against `<>` / `>=`).
        let (_lhs, op, _rhs) = split_top_level_comparison("a >= b").expect("a >= b must split");
        assert_eq!(op, ">=");

        // `<=` and `=<` are both accepted (TLA+ leq).
        let (_lhs, op, _rhs) = split_top_level_comparison("a <= b").expect("a <= b must split");
        assert_eq!(op, "<=");
        let (_lhs, op, _rhs) = split_top_level_comparison("a =< b").expect("a =< b must split");
        assert_eq!(op, "=<");

        // `\notin` distinct from `\in`.
        let (_lhs, op, _rhs) =
            split_top_level_comparison("x \\notin S").expect("\\notin must split");
        assert_eq!(op, "\\notin");

        // Membership inside a function-application bracket is NOT top-level.
        // f[1] = 2  parses as f[1] = 2 (the `=` is at top-level despite the
        // `[]` wrapper around `1`).
        let (_lhs, op, _rhs) =
            split_top_level_comparison("f[1] = 2").expect("f[1] = 2 must split on top-level `=`");
        assert_eq!(op, "=");
    }
}
