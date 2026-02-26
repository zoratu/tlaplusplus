use crate::tla::{
    ActionClause, ActionIr, ClauseKind, TlaDefinition, TlaState, TlaValue, classify_clause,
};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

const MAX_EVAL_DEPTH: usize = 256;

/// Context for evaluating expressions on a single state
/// Uses Rc for copy-on-write semantics to avoid cloning entire context
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    pub state: &'a TlaState,
    pub locals: Rc<BTreeMap<String, TlaValue>>,
    pub local_definitions: Rc<BTreeMap<String, TlaDefinition>>,
    pub definitions: Option<&'a BTreeMap<String, TlaDefinition>>,
    pub instances: Option<&'a BTreeMap<String, crate::tla::module::TlaModuleInstance>>,
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
        }
    }

    fn with_local_value(&self, name: impl Into<String>, value: TlaValue) -> Self {
        // Copy-on-write: only clone the locals map, reuse the rest
        let mut new_locals = (*self.locals).clone();
        new_locals.insert(name.into(), value);
        Self {
            state: self.state,
            locals: Rc::new(new_locals),
            local_definitions: Rc::clone(&self.local_definitions),
            definitions: self.definitions,
            instances: self.instances,
        }
    }

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
        }
    }

    pub fn runtime_value(&self, name: &str) -> Option<TlaValue> {
        if let Some(v) = self.locals.get(name) {
            return Some(v.clone());
        }
        self.state.get(name).cloned()
    }

    fn definition(&self, name: &str) -> Option<TlaDefinition> {
        if let Some(def) = self.local_definitions.get(name) {
            return Some(def.clone());
        }
        self.definitions.and_then(|defs| defs.get(name).cloned())
    }

    fn resolve_identifier(&self, name: &str, depth: usize) -> Result<TlaValue> {
        if let Some(v) = self.runtime_value(name) {
            return Ok(v);
        }

        if let Some(def) = self.definition(name)
            && def.params.is_empty()
        {
            return eval_operator_call(name, Vec::new(), self, depth + 1);
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

    // TODO: Add support for full expression evaluation with primed variables
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
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }

    chars.all(|c| c.is_alphanumeric() || c == '_')
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

/// Evaluate a LET expression that contains primed assignments
fn eval_let_action(
    expr: &str,
    ctx: &EvalContext<'_>,
    staged: &mut BTreeMap<String, TlaValue>,
    unchanged_vars: &mut Vec<String>,
) -> Result<()> {
    let (defs_text, body_text) =
        split_outer_let(expr).ok_or_else(|| anyhow!("invalid LET expression in action: {expr}"))?;
    let defs = parse_let_definitions(defs_text)?;
    let child_ctx = ctx.with_local_definitions(defs);

    // The body should be a conjunction of primed assignments and guards
    let conjuncts = split_top_level_symbol(body_text, "/\\");
    for conjunct in conjuncts {
        let trimmed = conjunct.trim();
        match classify_clause(trimmed) {
            ClauseKind::PrimedAssignment { var, expr: rhs } => {
                let value = eval_expr(&rhs, &child_ctx)?;
                staged.insert(var, value);
            }
            ClauseKind::Unchanged { vars } => {
                unchanged_vars.extend(vars);
            }
            _ => {
                // This is a guard - evaluate it
                if !eval_guard(trimmed, &child_ctx)? {
                    return Err(anyhow!("LET action guard failed: {}", trimmed));
                }
            }
        }
    }

    Ok(())
}

pub fn apply_action_ir(action: &ActionIr, current: &TlaState) -> Result<Option<TlaState>> {
    let ctx = EvalContext::new(current);
    apply_action_ir_with_context(action, current, &ctx)
}

pub fn apply_action_ir_with_context(
    action: &ActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Option<TlaState>> {
    let mut staged: BTreeMap<String, TlaValue> = BTreeMap::new();
    let mut unchanged_vars: Vec<String> = Vec::new();

    for clause in &action.clauses {
        match clause {
            ActionClause::Guard { expr } => {
                if !eval_guard(expr, ctx)? {
                    return Ok(None);
                }
            }
            ActionClause::PrimedAssignment { var, expr } => {
                let value = eval_expr(expr, ctx)?;
                staged.insert(var.clone(), value);
            }
            ActionClause::Unchanged { vars } => {
                unchanged_vars.extend(vars.iter().cloned());
            }
            ActionClause::LetWithPrimes { expr } => {
                // Evaluate LET expression which should produce assignments
                // The LET IN body should be a conjunction of primed assignments
                eval_let_action(expr, ctx, &mut staged, &mut unchanged_vars)?;
            }
        }
    }

    let mut next = current.clone();
    for var in unchanged_vars {
        if let Some(old) = current.get(&var) {
            next.insert(var, old.clone());
        }
    }
    for (var, value) in staged {
        next.insert(var, value);
    }

    Ok(Some(next))
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
        // Get caller information to debug where empty expressions come from
        eprintln!("\n=== EMPTY EXPRESSION AT DEPTH {} ===", depth);
        eprintln!("raw_expr: {:?}", raw_expr);
        eprintln!("locals: {:?}", ctx.locals.keys().collect::<Vec<_>>());

        // Print a simplified backtrace
        let bt = std::backtrace::Backtrace::force_capture();
        let bt_str = format!("{:?}", bt);
        eprintln!("\nBacktrace (showing tlaplusplus frames):");
        for line in bt_str
            .lines()
            .filter(|l| l.contains("tlaplusplus::tla") && !l.contains("rust_begin_unwind"))
            .take(15)
        {
            eprintln!("  {}", line.trim());
        }

        return Err(anyhow!("empty expression (raw: {raw_expr:?})"));
    }

    expr = strip_outer_parens(expr);

    if expr == "TRUE" {
        return Ok(TlaValue::Bool(true));
    }
    if expr == "FALSE" {
        return Ok(TlaValue::Bool(false));
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

    let implies_parts = split_top_level_symbol(expr, "=>");
    if implies_parts.len() > 1 {
        let mut rhs =
            eval_expr_inner(implies_parts.last().expect("non-empty"), ctx, depth + 1)?.as_bool()?;
        for i in (0..implies_parts.len() - 1).rev() {
            let lhs = eval_expr_inner(&implies_parts[i], ctx, depth + 1)?.as_bool()?;
            rhs = (!lhs) || rhs;
        }
        return Ok(TlaValue::Bool(rhs));
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

    let and_parts = split_top_level_symbol(expr, "/\\");
    if and_parts.len() > 1 {
        for part in and_parts {
            if !eval_expr_inner(&part, ctx, depth + 1)?.as_bool()? {
                return Ok(TlaValue::Bool(false));
            }
        }
        return Ok(TlaValue::Bool(true));
    }

    if let Some(rest) = expr.strip_prefix('~') {
        return Ok(TlaValue::Bool(
            !eval_expr_inner(rest.trim(), ctx, depth + 1)?.as_bool()?,
        ));
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
            "<" | "<=" | ">" | ">=" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let cmp = match op {
                    "<" => left.as_int()? < right.as_int()?,
                    "<=" => left.as_int()? <= right.as_int()?,
                    ">" => left.as_int()? > right.as_int()?,
                    ">=" => left.as_int()? >= right.as_int()?,
                    _ => unreachable!(),
                };
                Ok(TlaValue::Bool(cmp))
            }
            "\\in" => {
                // Special handling for infinite built-in sets
                let rhs_trimmed = rhs.trim();
                match rhs_trimmed {
                    "Nat" => {
                        let val = left.as_int()?;
                        Ok(TlaValue::Bool(val >= 0))
                    }
                    "Int" => {
                        // Any integer is in Int
                        let _ = left.as_int()?; // Verify it's an integer
                        Ok(TlaValue::Bool(true))
                    }
                    "BOOLEAN" => {
                        // Check if it's a boolean
                        match left {
                            TlaValue::Bool(_) => Ok(TlaValue::Bool(true)),
                            _ => Ok(TlaValue::Bool(false)),
                        }
                    }
                    _ => {
                        // Check for Seq(S) - sequence membership
                        if let Some(inner) = rhs_trimmed.strip_prefix("Seq(") {
                            if let Some(set_expr) = inner.strip_suffix(")") {
                                // x \in Seq(S) means x is a sequence and all elements are in S
                                match &left {
                                    TlaValue::Seq(seq) => {
                                        let set_val = eval_expr_inner(set_expr, ctx, depth + 1)?;
                                        let all_in_set = seq
                                            .iter()
                                            .all(|elem| set_val.contains(elem).unwrap_or(false));
                                        return Ok(TlaValue::Bool(all_in_set));
                                    }
                                    _ => return Ok(TlaValue::Bool(false)), // Not a sequence
                                }
                            }
                        }

                        // Check for [S -> T] or [field: Type, ...] - function/record set membership
                        if rhs_trimmed.starts_with('[') && rhs_trimmed.ends_with(']') {
                            let inner = &rhs_trimmed[1..rhs_trimmed.len() - 1];

                            // Check if it's a function type [S -> T]
                            if let Some(arrow_idx) = inner.find("->") {
                                let domain_expr = inner[..arrow_idx].trim();
                                let codomain_expr = inner[arrow_idx + 2..].trim();

                                // f \in [S -> T] means f is a function, DOMAIN f = S, and all values in T
                                match &left {
                                    TlaValue::Function(func) => {
                                        let domain_set =
                                            eval_expr_inner(domain_expr, ctx, depth + 1)?;
                                        let domain_set = domain_set.as_set()?;

                                        // Check that function domain equals the expected domain
                                        let func_domain: BTreeSet<TlaValue> =
                                            func.keys().cloned().collect();
                                        if func_domain != *domain_set {
                                            return Ok(TlaValue::Bool(false));
                                        }

                                        // Check all function values are in codomain
                                        // Handle special case for Nat
                                        if codomain_expr == "Nat" {
                                            let all_nat = func
                                                .values()
                                                .all(|v| matches!(v.as_int(), Ok(n) if n >= 0));
                                            return Ok(TlaValue::Bool(all_nat));
                                        }

                                        let codomain_set =
                                            eval_expr_inner(codomain_expr, ctx, depth + 1)?;
                                        let all_in_codomain = func
                                            .values()
                                            .all(|v| codomain_set.contains(v).unwrap_or(false));
                                        return Ok(TlaValue::Bool(all_in_codomain));
                                    }
                                    _ => return Ok(TlaValue::Bool(false)), // Not a function
                                }
                            }

                            // Check if it's a record type [field: Type, ...]
                            if inner.contains(':') {
                                match &left {
                                    TlaValue::Record(rec) => {
                                        // Parse field: Type specifications
                                        let field_specs: Vec<&str> = inner.split(',').collect();
                                        let mut expected_fields =
                                            std::collections::HashSet::<String>::new();

                                        for spec in field_specs {
                                            let parts: Vec<&str> = spec.split(':').collect();
                                            if parts.len() != 2 {
                                                continue;
                                            }
                                            let field_name = parts[0].trim();
                                            let field_type = parts[1].trim();
                                            expected_fields.insert(field_name.to_string());

                                            // Check if record has this field
                                            if let Some(field_value) = rec.get(field_name) {
                                                // Check if value matches type
                                                let matches_type = match field_type {
                                                    "Nat" => {
                                                        matches!(field_value.as_int(), Ok(n) if n >= 0)
                                                    }
                                                    "Int" => field_value.as_int().is_ok(),
                                                    "BOOLEAN" => {
                                                        matches!(field_value, TlaValue::Bool(_))
                                                    }
                                                    _ => {
                                                        // Evaluate type expression and check membership
                                                        if let Ok(type_val) = eval_expr_inner(
                                                            field_type,
                                                            ctx,
                                                            depth + 1,
                                                        ) {
                                                            type_val
                                                                .contains(field_value)
                                                                .unwrap_or(false)
                                                        } else {
                                                            false
                                                        }
                                                    }
                                                };
                                                if !matches_type {
                                                    return Ok(TlaValue::Bool(false));
                                                }
                                            } else {
                                                // Field missing
                                                return Ok(TlaValue::Bool(false));
                                            }
                                        }

                                        // Check that record has no extra fields
                                        if rec.len() != expected_fields.len() {
                                            return Ok(TlaValue::Bool(false));
                                        }

                                        return Ok(TlaValue::Bool(true));
                                    }
                                    _ => return Ok(TlaValue::Bool(false)), // Not a record
                                }
                            }
                        }

                        // Regular set membership
                        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                        Ok(TlaValue::Bool(right.contains(&left)?))
                    }
                }
            }
            "\\notin" => {
                // Not in operator - opposite of \in
                let rhs_trimmed = rhs.trim();
                match rhs_trimmed {
                    "Nat" => {
                        let val = left.as_int()?;
                        Ok(TlaValue::Bool(val < 0))
                    }
                    "Int" => {
                        // No integer is not in Int
                        let _ = left.as_int()?;
                        Ok(TlaValue::Bool(false))
                    }
                    "BOOLEAN" => match left {
                        TlaValue::Bool(_) => Ok(TlaValue::Bool(false)),
                        _ => Ok(TlaValue::Bool(true)),
                    },
                    _ => {
                        // Check for Seq(S) - sequence membership
                        if let Some(inner) = rhs_trimmed.strip_prefix("Seq(") {
                            if let Some(set_expr) = inner.strip_suffix(")") {
                                // x \notin Seq(S) means x is not a sequence or some element is not in S
                                match &left {
                                    TlaValue::Seq(seq) => {
                                        let set_val = eval_expr_inner(set_expr, ctx, depth + 1)?;
                                        let all_in_set = seq
                                            .iter()
                                            .all(|elem| set_val.contains(elem).unwrap_or(false));
                                        return Ok(TlaValue::Bool(!all_in_set));
                                    }
                                    _ => return Ok(TlaValue::Bool(true)), // Not a sequence
                                }
                            }
                        }

                        // Check for [S -> T] or [field: Type, ...] - function/record set membership
                        if rhs_trimmed.starts_with('[') && rhs_trimmed.ends_with(']') {
                            let inner = &rhs_trimmed[1..rhs_trimmed.len() - 1];

                            // Check if it's a function type [S -> T]
                            if let Some(arrow_idx) = inner.find("->") {
                                let domain_expr = inner[..arrow_idx].trim();
                                let codomain_expr = inner[arrow_idx + 2..].trim();

                                // f \notin [S -> T] is the opposite of \in
                                match &left {
                                    TlaValue::Function(func) => {
                                        let domain_set =
                                            eval_expr_inner(domain_expr, ctx, depth + 1)?;
                                        let domain_set = domain_set.as_set()?;

                                        let func_domain: BTreeSet<TlaValue> =
                                            func.keys().cloned().collect();
                                        if func_domain != *domain_set {
                                            return Ok(TlaValue::Bool(true));
                                        }

                                        if codomain_expr == "Nat" {
                                            let all_nat = func
                                                .values()
                                                .all(|v| matches!(v.as_int(), Ok(n) if n >= 0));
                                            return Ok(TlaValue::Bool(!all_nat));
                                        }

                                        let codomain_set =
                                            eval_expr_inner(codomain_expr, ctx, depth + 1)?;
                                        let all_in_codomain = func
                                            .values()
                                            .all(|v| codomain_set.contains(v).unwrap_or(false));
                                        return Ok(TlaValue::Bool(!all_in_codomain));
                                    }
                                    _ => return Ok(TlaValue::Bool(true)), // Not a function
                                }
                            }

                            // Check if it's a record type [field: Type, ...]
                            if inner.contains(':') {
                                match &left {
                                    TlaValue::Record(rec) => {
                                        let field_specs: Vec<&str> = inner.split(',').collect();
                                        let mut expected_fields =
                                            std::collections::HashSet::<String>::new();

                                        for spec in field_specs {
                                            let parts: Vec<&str> = spec.split(':').collect();
                                            if parts.len() != 2 {
                                                continue;
                                            }
                                            let field_name = parts[0].trim();
                                            let field_type = parts[1].trim();
                                            expected_fields.insert(field_name.to_string());

                                            if let Some(field_value) = rec.get(field_name) {
                                                let matches_type = match field_type {
                                                    "Nat" => {
                                                        matches!(field_value.as_int(), Ok(n) if n >= 0)
                                                    }
                                                    "Int" => field_value.as_int().is_ok(),
                                                    "BOOLEAN" => {
                                                        matches!(field_value, TlaValue::Bool(_))
                                                    }
                                                    _ => {
                                                        if let Ok(type_val) = eval_expr_inner(
                                                            field_type,
                                                            ctx,
                                                            depth + 1,
                                                        ) {
                                                            type_val
                                                                .contains(field_value)
                                                                .unwrap_or(false)
                                                        } else {
                                                            false
                                                        }
                                                    }
                                                };
                                                if !matches_type {
                                                    return Ok(TlaValue::Bool(true));
                                                }
                                            } else {
                                                return Ok(TlaValue::Bool(true));
                                            }
                                        }

                                        if rec.len() != expected_fields.len() {
                                            return Ok(TlaValue::Bool(true));
                                        }

                                        return Ok(TlaValue::Bool(false));
                                    }
                                    _ => return Ok(TlaValue::Bool(true)), // Not a record
                                }
                            }
                        }

                        let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                        Ok(TlaValue::Bool(!right.contains(&left)?))
                    }
                }
            }
            "\\subseteq" => {
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let lhs_set = left.as_set()?;
                let rhs_set = right.as_set()?;
                Ok(TlaValue::Bool(lhs_set.iter().all(|v| rhs_set.contains(v))))
            }
            ".." => {
                // Range operator: a..b creates set {a, a+1, ..., b}
                let right = eval_expr_inner(rhs, ctx, depth + 1)?;
                let start = left.as_int()?;
                let end = right.as_int()?;
                let range_set: BTreeSet<TlaValue> = (start..=end).map(TlaValue::Int).collect();
                Ok(TlaValue::Set(range_set))
            }
            _ => Err(anyhow!("unsupported comparison operator {op}")),
        };
    }

    let union_parts = split_top_level_keyword(expr, "\\union");
    if union_parts.len() > 1 {
        let mut out = eval_expr_inner(&union_parts[0], ctx, depth + 1)?;
        for part in &union_parts[1..] {
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

    let minus_parts = split_top_level_set_minus(expr);
    if minus_parts.len() > 1 {
        let mut out = eval_expr_inner(&minus_parts[0], ctx, depth + 1)?;
        for part in &minus_parts[1..] {
            out = out.set_minus(&eval_expr_inner(part, ctx, depth + 1)?)?;
        }
        return Ok(out);
    }

    let concat_parts = split_top_level_keyword(expr, "\\o");
    if concat_parts.len() > 1 {
        let mut out = eval_expr_inner(&concat_parts[0], ctx, depth + 1)?;
        for part in &concat_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            out = seq_or_string_concat(out, rhs)?;
        }
        return Ok(out);
    }

    let cartesian_parts = split_top_level_keyword(expr, "\\X");
    if cartesian_parts.len() > 1 {
        let mut result = eval_expr_inner(&cartesian_parts[0], ctx, depth + 1)?;
        for part in &cartesian_parts[1..] {
            let rhs = eval_expr_inner(part, ctx, depth + 1)?;
            let lhs_set = result.as_set()?;
            let rhs_set = rhs.as_set()?;
            let mut product = BTreeSet::new();
            for lhs_val in lhs_set {
                for rhs_val in rhs_set {
                    let tuple = TlaValue::Seq(vec![lhs_val.clone(), rhs_val.clone()]);
                    product.insert(tuple);
                }
            }
            result = TlaValue::Set(product);
        }
        return Ok(result);
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
            "\\div" => Ok(TlaValue::Int(left / right)),
            _ => Err(anyhow!("unsupported multiplicative operator {op}")),
        };
    }

    if starts_with_keyword(expr, "SUBSET") {
        let rest = expr["SUBSET".len()..].trim();
        let set = eval_expr_inner(rest, ctx, depth + 1)?;
        return Ok(TlaValue::Set(powerset(set.as_set()?)));
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

    let parsed = parse_binders(binders, ctx, depth + 1)?;
    if parsed.len() != 1 {
        return Err(anyhow!("CHOOSE currently expects exactly one binder"));
    }

    let (var, domain) = &parsed[0];
    for value in domain {
        let child = ctx.with_local_value(var.clone(), value.clone());
        if eval_expr_inner(predicate, &child, depth + 1)?.as_bool()? {
            return Ok(value.clone());
        }
    }

    Err(anyhow!("CHOOSE found no matching value"))
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
        params,
        body: body.to_string(),
        captured_locals,
    })
}

fn eval_let_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (defs_text, body_text) =
        split_outer_let(expr).ok_or_else(|| anyhow!("invalid LET expression: {expr}"))?;
    let defs = parse_let_definitions(defs_text)?;
    let child = ctx.with_local_definitions(defs);
    eval_expr_inner(body_text, &child, depth + 1)
}

fn eval_atom_with_postfix(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
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
            let key = eval_expr_inner(inside, ctx, depth + 1)?;

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

        return Err(anyhow!("unexpected trailing tokens in expr: {expr}"));
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
        return Ok((TlaValue::Seq(out), rest));
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
        let has_runtime_value = ctx.runtime_value(&name).is_some();
        let has_operator = ctx.definition(&name).is_some();

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

        if !has_runtime_value && has_operator && rest.trim_start().starts_with('[') {
            let trimmed_rest = rest.trim_start();
            let (args_text, next_rest) = take_bracket_group(trimmed_rest, '[', ']')?;
            let args = parse_argument_list(args_text, ctx, depth + 1)?;
            let value = eval_operator_call(&name, args, ctx, depth + 1)?;
            return Ok((value, next_rest));
        }

        // Handle prefix operators like DOMAIN and ENABLED that don't use parentheses
        if matches!(name.as_str(), "DOMAIN") && !has_runtime_value {
            // Parse the argument as the next atom
            let (arg_value, next_rest) = parse_base(rest_after_name.trim_start(), ctx, depth + 1)?;
            let value = eval_operator_call(&name, vec![arg_value], ctx, depth + 1)?;
            return Ok((value, next_rest));
        }

        // Handle ENABLED operator: ENABLED ActionName
        if matches!(name.as_str(), "ENABLED") && !has_runtime_value {
            let action_name_part = rest_after_name.trim_start();
            // Parse the action name (simple identifier)
            if let Some((action_name, next_rest)) = parse_identifier_prefix(action_name_part) {
                let value = eval_enabled(&action_name, ctx, depth + 1)?;
                return Ok((value, next_rest));
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
        return Ok(TlaValue::Set(BTreeSet::new()));
    }

    if let Some(colon_idx) = find_top_level_char(inner, ':') {
        let lhs = inner[..colon_idx].trim();
        let rhs = inner[colon_idx + 1..].trim();

        if contains_top_level_keyword(lhs, "\\in") {
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
            return Ok(TlaValue::Set(out));
        }

        let binders = parse_binders(rhs, ctx, depth + 1)?;
        let mut assignments = BTreeMap::new();
        let mut out = BTreeSet::new();
        collect_binder_map_set(0, &binders, lhs, &mut assignments, &mut out, ctx, depth + 1)?;
        return Ok(TlaValue::Set(out));
    }

    let mut out = BTreeSet::new();
    for item in split_top_level_symbol(inner, ",") {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        out.insert(eval_expr_inner(item, ctx, depth + 1)?);
    }

    Ok(TlaValue::Set(out))
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
            return Ok(TlaValue::Function(map));
        }

        let mut record = BTreeMap::new();
        for entry in split_top_level_symbol(expr, ",") {
            let (key_text, value_text) = split_once_top_level(&entry, "|->")
                .ok_or_else(|| anyhow!("invalid record entry: {entry}"))?;
            let key = parse_record_key(key_text.trim())?;
            let value = eval_expr_inner(value_text.trim(), ctx, depth + 1)?;
            record.insert(key, value);
        }
        return Ok(TlaValue::Record(record));
    }

    Err(anyhow!("unsupported bracket expression: [{expr}]"))
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

fn apply_value(
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
            lambda_ctx.locals = std::rc::Rc::new(captured_locals.clone());

            // Bind arguments to parameters
            {
                let locals_mut = std::rc::Rc::make_mut(&mut lambda_ctx.locals);
                for (param, arg) in params.iter().zip(args.into_iter()) {
                    locals_mut.insert(param.clone(), arg);
                }
            }

            // Evaluate the lambda body
            eval_expr_inner(body, &lambda_ctx, depth + 1)
        }
        TlaValue::Function(map) => {
            if args.len() != 1 {
                return Err(anyhow!("Function application expects exactly 1 argument"));
            }
            map.get(&args[0])
                .cloned()
                .ok_or_else(|| anyhow!("function missing key {:?}", args[0]))
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
    instance_ctx.instances = Some(&module.instances);

    // Apply substitutions: replace constants in the context
    {
        let locals_mut = std::rc::Rc::make_mut(&mut instance_ctx.locals);
        for (param, value_expr) in &instance.substitutions {
            // Evaluate the substitution value in the original context
            let value = eval_expr(value_expr, ctx)?;
            locals_mut.insert(param.clone(), value);
        }
    }

    // Bind operator parameters
    let mut bound = Vec::with_capacity(operator_def.params.len());
    for (param, arg) in operator_def.params.iter().zip(args.into_iter()) {
        bound.push((param.as_str(), arg));
    }

    let child_ctx = instance_ctx.with_local_values(&bound);

    // Evaluate the operator body
    eval_expr_inner(&operator_def.body, &child_ctx, depth + 1)
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

/// Evaluate ENABLED operator: check if an action is enabled in the current state
fn eval_enabled(action_name: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("ENABLED recursion depth exceeded at {action_name}"));
    }

    // Look up the action definition
    let action_def = ctx
        .definition(action_name)
        .ok_or_else(|| anyhow!("action '{}' not found for ENABLED", action_name))?;

    // Check if the action has parameters (should be 0 for actions)
    if !action_def.params.is_empty() {
        return Err(anyhow!(
            "ENABLED expects a nullary action, but '{}' has {} parameters",
            action_name,
            action_def.params.len()
        ));
    }

    // Compile the action to IR
    let action_ir = crate::tla::compile_action_ir(&action_def);

    // Try to apply the action to see if it's enabled
    // An action is enabled if it can produce a successor state
    match crate::tla::apply_action_ir_with_context(&action_ir, ctx.state, ctx) {
        Ok(Some(_next_state)) => {
            // Action produced a successor - it's enabled
            Ok(TlaValue::Bool(true))
        }
        Ok(None) => {
            // Action did not produce a successor (guard failed) - not enabled
            Ok(TlaValue::Bool(false))
        }
        Err(_) => {
            // Evaluation error - action is not enabled
            // (This can happen with complex expressions that aren't fully supported)
            Ok(TlaValue::Bool(false))
        }
    }
}

fn eval_operator_call(
    name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_EVAL_DEPTH {
        return Err(anyhow!("operator recursion depth exceeded at {name}"));
    }

    match name {
        "Cardinality" => {
            if args.len() != 1 {
                return Err(anyhow!("Cardinality expects 1 argument"));
            }
            return Ok(TlaValue::Int(args[0].len()? as i64));
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
            let seq = match &args[0] {
                TlaValue::Seq(v) => v,
                _ => return Err(anyhow!("Head expects a sequence, got {:?}", args[0])),
            };
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            return Ok(seq[0].clone());
        }
        "Tail" => {
            if args.len() != 1 {
                return Err(anyhow!("Tail expects 1 argument"));
            }
            let seq = match &args[0] {
                TlaValue::Seq(v) => v,
                _ => return Err(anyhow!("Tail expects a sequence, got {:?}", args[0])),
            };
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            return Ok(TlaValue::Seq(seq[1..].to_vec()));
        }
        "Append" => {
            if args.len() != 2 {
                return Err(anyhow!("Append expects 2 arguments"));
            }
            let seq = match &args[0] {
                TlaValue::Seq(v) => v,
                _ => return Err(anyhow!("Append expects a sequence, got {:?}", args[0])),
            };
            let mut new_seq = seq.clone();
            new_seq.push(args[1].clone());
            return Ok(TlaValue::Seq(new_seq));
        }
        "SubSeq" => {
            if args.len() != 3 {
                return Err(anyhow!("SubSeq expects 3 arguments"));
            }
            let seq = match &args[0] {
                TlaValue::Seq(v) => v,
                _ => return Err(anyhow!("SubSeq expects a sequence, got {:?}", args[0])),
            };
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
                return Ok(TlaValue::Seq(vec![]));
            }

            return Ok(TlaValue::Seq(seq[start..end].to_vec()));
        }
        "SelectSeq" => {
            if args.len() != 2 {
                return Err(anyhow!("SelectSeq expects 2 arguments"));
            }
            let seq = match &args[0] {
                TlaValue::Seq(v) => v,
                _ => return Err(anyhow!("SelectSeq expects a sequence, got {:?}", args[0])),
            };
            let test_fn = &args[1];

            let mut result = Vec::new();
            for elem in seq {
                // Apply the test function to each element
                let test_result = apply_value(test_fn, vec![elem.clone()], ctx, depth + 1)?;
                let passes = test_result.as_bool()?;
                if passes {
                    result.push(elem.clone());
                }
            }
            return Ok(TlaValue::Seq(result));
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
                out.insert(TlaValue::Function(map));
            }
            return Ok(TlaValue::Set(out));
        }
        "DOMAIN" => {
            if args.len() != 1 {
                return Err(anyhow!("DOMAIN expects 1 argument"));
            }
            match &args[0] {
                TlaValue::Function(map) => {
                    let keys = map.keys().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(keys));
                }
                TlaValue::Record(map) => {
                    let keys = map
                        .keys()
                        .map(|k| TlaValue::String(k.clone()))
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(keys));
                }
                TlaValue::Seq(seq) => {
                    // DOMAIN of a sequence is {1, 2, ..., Len(seq)}
                    let indices = (1..=seq.len() as i64)
                        .map(TlaValue::Int)
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(indices));
                }
                _ => {
                    return Err(anyhow!("DOMAIN expects a function, record, or sequence"));
                }
            }
        }
        _ => {}
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

    let mut bound = Vec::with_capacity(def.params.len());
    for (param, arg) in def.params.iter().zip(args.into_iter()) {
        bound.push((param.as_str(), arg));
    }
    let child = ctx.with_local_values(&bound);
    eval_expr_inner(&def.body, &child, depth + 1)
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
                        let body = after_let[end..].trim();
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

        let head = defs_text[cursor..*eq_pos].trim();
        let body = defs_text[*eq_pos + 2..next_head_start].trim();
        let (name, params) = parse_local_def_head(head);

        if !name.is_empty() {
            defs.insert(
                name.clone(),
                TlaDefinition {
                    name,
                    params,
                    body: body.to_string(),
                },
            );
        }

        cursor = skip_leading_ws(defs_text, next_head_start);
    }

    Ok(defs)
}

fn parse_local_def_head(head: &str) -> (String, Vec<String>) {
    if let Some(open) = head.find('(')
        && let Some(close) = head.rfind(')')
        && close > open
    {
        let name = head[..open].trim().to_string();
        let params = split_top_level_symbol(&head[open + 1..close], ",")
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>();
        return (name, params);
    }

    if let Some(open) = head.find('[')
        && let Some(close) = head.rfind(']')
        && close > open
    {
        let name = head[..open].trim().to_string();
        let inside = &head[open + 1..close];
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
        return (name, params);
    }

    let name = head
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn parse_binders(
    expr: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<Vec<(String, Vec<TlaValue>)>> {
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

        for var in split_top_level_symbol(vars_text, ",") {
            let name = var.trim();
            if name.is_empty() {
                continue;
            }
            bindings.push((name.to_string(), domain.clone()));
        }

        rest = next_rest.trim_start();
    }

    Ok(bindings)
}

fn evaluate_exists(
    idx: usize,
    domains: &[(String, Vec<TlaValue>)],
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

    let (name, values) = &domains[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
        if evaluate_exists(idx + 1, domains, body, assignments, ctx, depth + 1)? {
            assignments.remove(name);
            return Ok(true);
        }
    }
    assignments.remove(name);
    Ok(false)
}

fn evaluate_forall(
    idx: usize,
    domains: &[(String, Vec<TlaValue>)],
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

    let (name, values) = &domains[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
        if !evaluate_forall(idx + 1, domains, body, assignments, ctx, depth + 1)? {
            assignments.remove(name);
            return Ok(false);
        }
    }
    assignments.remove(name);
    Ok(true)
}

fn collect_function_mapping(
    idx: usize,
    binders: &[(String, Vec<TlaValue>)],
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

    let (name, values) = &binders[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
        collect_function_mapping(idx + 1, binders, body, assignments, out, ctx, depth + 1)?;
    }
    assignments.remove(name);

    Ok(())
}

fn collect_binder_filter_set(
    idx: usize,
    binders: &[(String, Vec<TlaValue>)],
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

    let (name, values) = &binders[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
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
    assignments.remove(name);

    Ok(())
}

fn collect_binder_map_set(
    idx: usize,
    binders: &[(String, Vec<TlaValue>)],
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

    let (name, values) = &binders[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
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
    assignments.remove(name);

    Ok(())
}

fn binder_key(
    assignments: &BTreeMap<String, TlaValue>,
    binders: &[(String, Vec<TlaValue>)],
) -> Result<TlaValue> {
    if binders.len() == 1 {
        assignments
            .get(&binders[0].0)
            .cloned()
            .ok_or_else(|| anyhow!("missing binder assignment"))
    } else {
        let mut items = Vec::with_capacity(binders.len());
        for (name, _) in binders {
            items.push(
                assignments
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("missing binder assignment for {name}"))?,
            );
        }
        Ok(TlaValue::Seq(items))
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
            let key = eval_expr_inner(inside, ctx, depth + 1)?;
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
                let mut next = map.clone();
                next.insert(name.clone(), updated);
                Ok(TlaValue::Record(next))
            }
            _ => Err(anyhow!("field update on non-record value {base:?}")),
        },
        PathSegment::Index(key) => match base {
            TlaValue::Function(map) => {
                let current = map.get(key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = map.clone();
                next.insert(key.clone(), updated);
                Ok(TlaValue::Function(next))
            }
            TlaValue::Record(map) => {
                let record_key = record_key_from_value(key)?;
                let current = map.get(&record_key).cloned().unwrap_or(TlaValue::Undefined);
                let updated = set_path_value(&current, &path[1..], new_value)?;
                let mut next = map.clone();
                next.insert(record_key, updated);
                Ok(TlaValue::Record(next))
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
                let mut next = values.clone();
                next[zero] = updated;
                Ok(TlaValue::Seq(next))
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

fn seq_or_string_concat(lhs: TlaValue, rhs: TlaValue) -> Result<TlaValue> {
    match (lhs, rhs) {
        (TlaValue::String(mut a), TlaValue::String(b)) => {
            a.push_str(&b);
            Ok(TlaValue::String(a))
        }
        (TlaValue::Seq(mut a), TlaValue::Seq(b)) => {
            a.extend(b);
            Ok(TlaValue::Seq(a))
        }
        (a, b) => Err(anyhow!(
            "\\o expects String or Seq operands, got {a:?} and {b:?}"
        )),
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

fn powerset(input: &BTreeSet<TlaValue>) -> BTreeSet<TlaValue> {
    let mut subsets = BTreeSet::new();
    let values: Vec<TlaValue> = input.iter().cloned().collect();
    let n = values.len();

    if n >= usize::BITS as usize {
        return subsets;
    }

    for mask in 0usize..(1usize << n) {
        let mut subset = BTreeSet::new();
        for (idx, value) in values.iter().enumerate() {
            if (mask >> idx) & 1 == 1 {
                subset.insert(value.clone());
            }
        }
        subsets.insert(TlaValue::Set(subset));
    }

    subsets
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
        other => format!("{other:?}"),
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

fn split_top_level_symbol(expr: &str, delim: &str) -> Vec<String> {
    split_top_level(expr, delim, false)
}

fn split_top_level_keyword(expr: &str, delim: &str) -> Vec<String> {
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
                if after_let.is_empty() || !after_let.chars().next().unwrap().is_alphanumeric() {
                    let_depth += 1;
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
                // 4. Otherwise, split normally
                let should_split = if let_depth > 0 && is_conjunction {
                    // We're inside a LET expression, don't split on conjunctions
                    false
                } else if quantifier_depth > 0
                    && quantifier_depth > seen_colon_for_quantifier
                    && is_conjunction
                {
                    // We're in quantifier binders (before :), don't split on conjunctions
                    false
                } else if in_quantifier_body && is_conjunction {
                    // We're in a quantifier body. Check what follows this /\
                    let after_delim = &expr[delim_end..].trim_start();
                    let next_is_quantifier =
                        after_delim.starts_with("\\A") || after_delim.starts_with("\\E");
                    // Split only if followed by another quantifier (this ends the current quantifier)
                    next_is_quantifier
                } else {
                    // All other cases: split normally
                    true
                };

                if should_split {
                    let part = expr[start..i].trim();
                    if !part.is_empty() {
                        out.push(part.to_string());
                    }
                    // Reset quantifier and LET tracking for next part
                    quantifier_depth = 0;
                    seen_colon_for_quantifier = 0;
                    let_depth = 0;
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
        if at_top && ch == '\\' {
            let prev = expr[..i].chars().next_back();
            let next_char = expr[i + ch_len..].chars().next();
            let ws_before = prev.map(|c| c.is_whitespace()).unwrap_or(false);
            let ws_after = next_char.map(|c| c.is_whitespace()).unwrap_or(false);
            if ws_before && ws_after {
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

fn split_top_level_comparison(expr: &str) -> Option<(&str, &'static str, &str)> {
    let patterns = [
        "\\subseteq",
        "\\notin",
        "\\in",
        "..",
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
                    let next_char = expr[end..].chars().next();
                    if next_char == Some('=') || next_char == Some('>') {
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
        }

        i += ch_len;
    }

    match (star_idx, div_idx) {
        (None, None) => None,
        (Some(s), None) => {
            let lhs = expr[..s].trim();
            let rhs = expr[s + 1..].trim();
            if lhs.is_empty() || rhs.is_empty() {
                None
            } else {
                Some((lhs, "*", rhs))
            }
        }
        (None, Some(d)) => {
            let lhs = expr[..d].trim();
            let rhs = expr[d + "\\div".len()..].trim();
            if lhs.is_empty() || rhs.is_empty() {
                None
            } else {
                Some((lhs, "\\div", rhs))
            }
        }
        (Some(s), Some(d)) => {
            if s > d {
                let lhs = expr[..s].trim();
                let rhs = expr[s + 1..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    None
                } else {
                    Some((lhs, "*", rhs))
                }
            } else {
                let lhs = expr[..d].trim();
                let rhs = expr[d + "\\div".len()..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    None
                } else {
                    Some((lhs, "\\div", rhs))
                }
            }
        }
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

fn find_top_level_keyword_index(expr: &str, keyword: &str) -> Option<usize> {
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
            && has_word_boundaries(expr, i, i + keyword.len())
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

fn find_top_level_char(expr: &str, target: char) -> Option<usize> {
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
    if !(first.is_alphabetic() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, c) in chars {
        if c.is_alphanumeric() || c == '_' || c == '\'' {
            end = idx + c.len_utf8();
        } else {
            break;
        }
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

    let value = expr[..end].parse::<i64>().ok()?;
    Some((value, &expr[end..]))
}

fn has_word_boundaries(expr: &str, start: usize, end: usize) -> bool {
    let prev = expr[..start].chars().next_back();
    let next = expr[end..].chars().next();

    let prev_ok = prev.map(|c| !is_word_char(c)).unwrap_or(true);
    let next_ok = next.map(|c| !is_word_char(c)).unwrap_or(true);

    prev_ok && next_ok
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evals_arithmetic_and_boolean() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(4)),
            ("y".to_string(), TlaValue::Int(2)),
        ]);
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
    fn applies_action_ir() {
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(1)),
            ("y".to_string(), TlaValue::Int(2)),
        ]);
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
    fn action_guard_can_block_transition() {
        let state = TlaState::from([("x".to_string(), TlaValue::Int(10))]);
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
    fn evaluates_let_and_operator_calls() {
        let defs = BTreeMap::from([
            (
                "CanAct".to_string(),
                TlaDefinition {
                    name: "CanAct".to_string(),
                    params: vec!["p".to_string()],
                    body: "actionCount[p] < MaxActionsPerTick".to_string(),
                },
            ),
            (
                "MaxActionsPerTick".to_string(),
                TlaDefinition {
                    name: "MaxActionsPerTick".to_string(),
                    params: vec![],
                    body: "3".to_string(),
                },
            ),
        ]);

        let state = TlaState::from([(
            "actionCount".to_string(),
            TlaValue::Function(BTreeMap::from([(
                TlaValue::ModelValue("bot1".to_string()),
                TlaValue::Int(1),
            )])),
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
        let state = TlaState::from([(
            "actionCount".to_string(),
            TlaValue::Function(BTreeMap::from([(
                TlaValue::ModelValue("bot1".to_string()),
                TlaValue::Int(1),
            )])),
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
    fn evaluates_quantifier_and_choose() {
        let state = TlaState::from([(
            "S".to_string(),
            TlaValue::Set(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ])),
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
    fn evaluates_case_expression() {
        let state = TlaState::from([("d".to_string(), TlaValue::Int(2))]);
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
                assert_eq!(params, vec!["x".to_string()]);
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
        let state_with_y = TlaState::from([("y".to_string(), TlaValue::Int(10))]);
        let ctx_with_y = EvalContext::new(&state_with_y);
        assert_eq!(
            eval_expr("(LAMBDA x: x + y)[5]", &ctx_with_y)
                .expect("lambda with closure should work"),
            TlaValue::Int(15)
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
            },
        )]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        // Test SelectSeq with lambda predicate
        let expr = "SelectSeq(<<1, 2, 3, 4, 5>>, LAMBDA x: x > 2)";
        let result = eval_expr(expr, &ctx).expect("SelectSeq with lambda should work");

        assert_eq!(
            result,
            TlaValue::Seq(vec![TlaValue::Int(3), TlaValue::Int(4), TlaValue::Int(5)])
        );

        // Test SelectSeq with different predicate
        let expr2 = "SelectSeq(<<1, 2, 3, 4, 5>>, LAMBDA x: x # 3)";
        let result2 = eval_expr(expr2, &ctx).expect("SelectSeq with lambda should work");

        assert_eq!(
            result2,
            TlaValue::Seq(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(4),
                TlaValue::Int(5)
            ])
        );
    }
}
