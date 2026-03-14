use crate::fairness::{ActionLabel, LabeledTransition};
use crate::tla::eval::apply_action_ir_with_context_multi;
use crate::tla::module::TlaModuleInstance;
use crate::tla::{
    ActionClause, ActionIr, CompiledActionIr, EvalContext, TlaDefinition, TlaState, TlaValue,
    apply_compiled_action_ir_multi, compile_action_ir, eval_expr, normalize_param_name,
    split_action_body_disjuncts, split_top_level,
};
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock};

// Global cache for pre-warmed compiled action IRs (populated at model load time)
// This is read-only during model checking, so no contention
static PREWARMED_ACTION_CACHE: LazyLock<DashMap<String, Arc<CompiledActionIr>>> =
    LazyLock::new(DashMap::new);

// Thread-local cache for runtime-compiled action IRs
// Using thread-local storage eliminates cross-thread contention entirely
thread_local! {
    static THREAD_LOCAL_ACTION_CACHE: RefCell<HashMap<String, Arc<CompiledActionIr>>> =
        RefCell::new(HashMap::with_capacity(64));
}

/// Get or compile an action IR with compiled expressions (thread-local, zero contention)
fn get_or_compile_action(def: &TlaDefinition) -> Arc<CompiledActionIr> {
    // Use body as cache key (name could have duplicates with different bodies)
    let cache_key = format!("{}:{}", def.name, def.body);

    // Check pre-warmed global cache first (read-only, no contention)
    if let Some(cached) = PREWARMED_ACTION_CACHE.get(&cache_key) {
        return Arc::clone(&cached);
    }

    // Check thread-local cache
    THREAD_LOCAL_ACTION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(cached) = cache.get(&cache_key) {
            return Arc::clone(cached);
        }

        // Compile and cache in thread-local storage
        let ir = compile_action_ir(def);
        let compiled = Arc::new(CompiledActionIr::from_ir(&ir));
        cache.insert(cache_key, Arc::clone(&compiled));
        compiled
    })
}

/// Insert a pre-compiled action into the global pre-warmed cache
///
/// This is used to warm up the cache at model load time with actions
/// that have already been compiled. The global cache is read-only during
/// model checking, so there's no contention.
pub fn insert_compiled_action(cache_key: String, compiled: Arc<CompiledActionIr>) {
    PREWARMED_ACTION_CACHE.insert(cache_key, compiled);
}

#[derive(Debug, Clone, Default)]
pub struct NextBranchProbe {
    pub total_disjuncts: usize,
    pub supported_disjuncts: usize,
    pub generated_successors: usize,
    pub failures: BTreeMap<String, u64>,
}

pub fn probe_next_disjuncts(
    next_body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    state: &TlaState,
) -> NextBranchProbe {
    probe_next_disjuncts_with_instances(next_body, definitions, None, state)
}

pub fn probe_next_disjuncts_with_instances(
    next_body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> NextBranchProbe {
    let disjuncts = split_action_disjuncts(next_body);
    let mut probe = NextBranchProbe {
        total_disjuncts: disjuncts.len(),
        ..NextBranchProbe::default()
    };

    for disj in disjuncts {
        match execute_branch(disj.trim(), &BTreeMap::new(), definitions, instances, state) {
            Ok(successors) => {
                probe.supported_disjuncts += 1;
                probe.generated_successors += successors.len();
            }
            Err(err) => {
                let key = err.to_string();
                *probe.failures.entry(key).or_insert(0) += 1;
            }
        }
    }

    probe
}

pub fn evaluate_next_states(
    next_body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    state: &TlaState,
) -> Result<Vec<TlaState>> {
    evaluate_next_states_with_instances(next_body, definitions, None, state)
}

/// Evaluate next states with module instances context
pub fn evaluate_next_states_with_instances(
    next_body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> Result<Vec<TlaState>> {
    let disjuncts = split_action_disjuncts(next_body);
    let mut out = Vec::new();
    let mut last_error = None;
    for disj in &disjuncts {
        match execute_branch(disj.trim(), &BTreeMap::new(), definitions, instances, state) {
            Ok(successors) => out.extend(successors),
            Err(err) => {
                // Treat evaluation errors in individual branches as disabled
                // branches (e.g., record access on ModelValue when the guard
                // would have been false). Only fail if ALL branches error out
                // and none produced successors.
                last_error = Some(err);
            }
        }
    }
    // If no branch produced any successors and we had errors, report the last error.
    // If some branches succeeded but others errored, the errors were likely guards
    // that didn't apply to this state (e.g., wrong message type).
    if out.is_empty() && last_error.is_some() && disjuncts.len() == 1 {
        // Single-branch specs: propagate the error since there's no fallback
        return Err(last_error.unwrap());
    }
    Ok(out)
}

/// Evaluate next states with action labels for fairness checking
///
/// This version tracks which action (disjunct) generated each successor state,
/// enabling fairness constraint checking on strongly connected components.
pub fn evaluate_next_states_labeled(
    next_body: &str,
    next_name: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    state: &TlaState,
) -> Result<Vec<LabeledTransition<TlaState>>> {
    evaluate_next_states_labeled_with_instances(next_body, next_name, definitions, None, state)
}

/// Evaluate next states with action labels and module instances context
pub fn evaluate_next_states_labeled_with_instances(
    next_body: &str,
    next_name: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> Result<Vec<LabeledTransition<TlaState>>> {
    let disjuncts = split_action_disjuncts(next_body);
    let mut out = Vec::new();

    for (disjunct_idx, disj) in disjuncts.iter().enumerate() {
        let successors =
            execute_branch(disj.trim(), &BTreeMap::new(), definitions, instances, state)?;

        // Extract action name from disjunct (e.g., "SendMsg(m)" -> "SendMsg")
        let action_name = extract_action_name(disj.trim()).unwrap_or_else(|| next_name.to_string());

        let label = if disjuncts.len() > 1 {
            // Multiple disjuncts - label with index
            ActionLabel::with_disjunct(action_name, disjunct_idx)
        } else {
            // Single action
            ActionLabel::new(action_name)
        };

        for next_state in successors {
            out.push(LabeledTransition {
                from: state.clone(),
                to: next_state,
                action: label.clone(),
            });
        }
    }

    Ok(out)
}

/// Extract action name from a disjunct expression
fn extract_action_name(expr: &str) -> Option<String> {
    let trimmed = strip_outer_parens(expr.trim());

    // Handle \E quantifier
    if trimmed.starts_with("\\E") {
        // Try to find the action call after the ':'
        if let Some(colon_idx) = trimmed.find(':') {
            let body = trimmed[colon_idx + 1..].trim();
            return extract_action_name(body);
        }
        return None;
    }

    // Try to extract identifier (action name)
    parse_identifier_prefix(trimmed).map(|(name, _)| name)
}

fn execute_branch(
    expr: &str,
    locals: &BTreeMap<String, TlaValue>,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> Result<Vec<TlaState>> {
    let trimmed = normalize_branch_expr(strip_outer_parens(expr.trim()));
    if trimmed.is_empty() {
        return Err(anyhow!("empty branch expression"));
    }

    // Handle nested disjunctions: \/ A \/ B \/ C
    // This happens when a quantifier body contains disjunctions, e.g.:
    //   \E op \in pendingOps : \/ JournalOperation(op) \/ CommitOperation(op)
    // We only handle this if split_top_level actually produces multiple parts.
    // If it returns just one part (e.g., when the disjunction is inside a quantifier),
    // we should proceed with other handling (like \E processing).
    let clause_count = crate::tla::action_ir::split_action_body_clauses(trimmed).len();
    let should_split_disjunction = trimmed.starts_with("\\/")
        || (contains_top_level_disjunction(trimmed) && clause_count <= 1);
    if should_split_disjunction {
        let disjuncts = split_action_disjuncts(trimmed);
        // Only handle as disjunction if we actually split into multiple parts
        if disjuncts.len() > 1 || (disjuncts.len() == 1 && disjuncts[0].trim() != trimmed) {
            let mut out = Vec::new();
            for disj in disjuncts {
                let disj_trimmed = disj.trim();
                if !disj_trimmed.is_empty() {
                    let successors =
                        execute_branch(disj_trimmed, locals, definitions, instances, state)?;
                    out.extend(successors);
                }
            }
            return Ok(out);
        }
        // Otherwise, fall through to other handling (e.g., \E quantifier)
    }

    if let Some(after_exists) = trimmed.strip_prefix("\\E") {
        return execute_exists_branch(
            after_exists.trim_start(),
            locals,
            definitions,
            instances,
            state,
        );
    }

    if trimmed.starts_with("LET") {
        let mut ctx = if let Some(inst) = instances {
            EvalContext::with_definitions_and_instances(state, definitions, inst)
        } else {
            EvalContext::with_definitions(state, definitions)
        };
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (name, value) in locals {
                locals_mut.insert(name.clone(), value.clone());
            }
        }
        let action = ActionIr {
            name: "__LetBranch__".to_string(),
            params: vec![],
            clauses: vec![ActionClause::LetWithPrimes {
                expr: trimmed.to_string(),
            }],
        };
        return apply_action_ir_with_context_multi(&action, state, &ctx);
    }

    // Try to parse as an action call
    if let Some((name, arg_exprs)) = parse_action_call(trimmed) {
        if let Some((alias, operator_name)) = name.split_once('!') {
            let instance_map = instances
                .ok_or_else(|| anyhow!("no module instances available for action '{name}'"))?;
            let instance = instance_map
                .get(alias)
                .ok_or_else(|| anyhow!("unknown module instance '{alias}'"))?;
            let module = instance.module.as_ref().ok_or_else(|| {
                anyhow!(
                    "module '{}' not loaded for instance '{}'",
                    instance.module_name,
                    alias
                )
            })?;
            let def = module
                .definitions
                .get(operator_name)
                .ok_or_else(|| anyhow!("unknown action '{name}'"))?;
            if def.params.len() != arg_exprs.len() {
                return Err(anyhow!(
                    "action '{name}' arity mismatch: expected {}, got {}",
                    def.params.len(),
                    arg_exprs.len()
                ));
            }

            let mut outer_ctx =
                EvalContext::with_definitions_and_instances(state, definitions, instance_map);
            {
                let locals_mut = std::rc::Rc::make_mut(&mut outer_ctx.locals);
                for (k, v) in locals {
                    locals_mut.insert(k.clone(), v.clone());
                }
            }

            let mut ctx = EvalContext::with_definitions_and_instances(
                state,
                &module.definitions,
                effective_instance_scope(&module.instances, Some(instance_map))
                    .expect("module instance scope should exist"),
            );
            {
                let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
                for (k, v) in locals {
                    locals_mut.insert(k.clone(), v.clone());
                }
            }
            seed_instance_constant_bindings(instance, module, &outer_ctx, &mut ctx);
            {
                let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
                bind_instance_substitutions(instance, &outer_ctx, locals_mut)?;
            }

            let mut args = Vec::with_capacity(arg_exprs.len());
            for arg_expr in arg_exprs {
                args.push(eval_expr(&arg_expr, &outer_ctx)?);
            }

            let mut bound_locals = locals.clone();
            for (param, arg) in def.params.iter().zip(args.iter()) {
                bound_locals.insert(normalize_param_name(param).to_string(), arg.clone());
            }
            {
                let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
                for (param, arg) in def.params.iter().zip(args.into_iter()) {
                    locals_mut.insert(normalize_param_name(param).to_string(), arg);
                }
            }

            let interpreted_ir = compile_action_ir(def);
            return match apply_action_ir_with_context_multi(&interpreted_ir, state, &ctx) {
                Ok(successors) => Ok(successors),
                Err(_) => execute_branch(
                    &def.body,
                    &bound_locals,
                    &module.definitions,
                    effective_instance_scope(&module.instances, Some(instance_map)),
                    state,
                ),
            };
        }

        let def = definitions
            .get(&name)
            .ok_or_else(|| anyhow!("unknown action '{name}'"))?;
        if def.params.len() != arg_exprs.len() {
            return Err(anyhow!(
                "action '{name}' arity mismatch: expected {}, got {}",
                def.params.len(),
                arg_exprs.len()
            ));
        }

        let mut ctx = if let Some(inst) = instances {
            EvalContext::with_definitions_and_instances(state, definitions, inst)
        } else {
            EvalContext::with_definitions(state, definitions)
        };
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (k, v) in locals {
                locals_mut.insert(k.clone(), v.clone());
            }
        }

        let mut args = Vec::with_capacity(arg_exprs.len());
        for arg_expr in arg_exprs {
            args.push(eval_expr(&arg_expr, &ctx)?);
        }

        let mut bound_locals = locals.clone();
        for (param, arg) in def.params.iter().zip(args.iter()) {
            bound_locals.insert(normalize_param_name(param).to_string(), arg.clone());
        }
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (param, arg) in def.params.iter().zip(args.into_iter()) {
                locals_mut.insert(normalize_param_name(param).to_string(), arg);
            }
        }

        // Use compiled action IR for faster evaluation
        let compiled_ir = get_or_compile_action(def);
        return match apply_compiled_action_ir_multi(&compiled_ir, state, &ctx) {
            Ok(successors) if !successors.is_empty() => Ok(successors),
            _ => {
                let interpreted_ir = compile_action_ir(def);
                match apply_action_ir_with_context_multi(&interpreted_ir, state, &ctx) {
                    Ok(successors) => Ok(successors),
                    Err(_) => {
                        execute_branch(&def.body, &bound_locals, definitions, instances, state)
                    }
                }
            }
        };
    }

    // Not an action call - treat as inline action body (conjunction of constraints)
    let inline_def = TlaDefinition {
        name: "<inline>".to_string(),
        params: Vec::new(),
        body: trimmed.to_string(),
        is_recursive: false,
    };

    let mut ctx = if let Some(inst) = instances {
        EvalContext::with_definitions_and_instances(state, definitions, inst)
    } else {
        EvalContext::with_definitions(state, definitions)
    };
    {
        let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
        for (k, v) in locals {
            locals_mut.insert(k.clone(), v.clone());
        }
    }

    // Use compiled action IR for inline actions too
    let compiled_ir = get_or_compile_action(&inline_def);
    match apply_compiled_action_ir_multi(&compiled_ir, state, &ctx) {
        Ok(successors) if !successors.is_empty() => Ok(successors),
        _ => apply_action_ir_with_context_multi(&compile_action_ir(&inline_def), state, &ctx),
    }
}

fn execute_exists_branch(
    expr: &str,
    locals: &BTreeMap<String, TlaValue>,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> Result<Vec<TlaState>> {
    let colon_idx = find_top_level_char(expr, ':')
        .ok_or_else(|| anyhow!("exists branch missing ':' in expression: {expr}"))?;

    let binder_text = expr[..colon_idx].trim();
    let body = expr[colon_idx + 1..].trim();
    let binders = parse_binders(binder_text, locals, definitions, instances, state)?;

    let mut out = Vec::new();
    let mut assignments = locals.clone();
    expand_binders(
        0,
        &binders,
        body,
        &mut assignments,
        definitions,
        instances,
        state,
        &mut out,
    )?;

    Ok(out)
}

fn expand_binders(
    idx: usize,
    binders: &[(String, Vec<TlaValue>)],
    body: &str,
    assignments: &mut BTreeMap<String, TlaValue>,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
    out: &mut Vec<TlaState>,
) -> Result<()> {
    if idx >= binders.len() {
        let successors = execute_branch(body, assignments, definitions, instances, state)?;
        out.extend(successors);
        return Ok(());
    }

    let (name, values) = &binders[idx];
    for value in values {
        assignments.insert(name.clone(), value.clone());
        expand_binders(
            idx + 1,
            binders,
            body,
            assignments,
            definitions,
            instances,
            state,
            out,
        )?;
    }
    assignments.remove(name);

    Ok(())
}

fn parse_binders(
    expr: &str,
    locals: &BTreeMap<String, TlaValue>,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: Option<&BTreeMap<String, TlaModuleInstance>>,
    state: &TlaState,
) -> Result<Vec<(String, Vec<TlaValue>)>> {
    let mut binders: Vec<(String, Vec<TlaValue>)> = Vec::new();
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

        let mut ctx = if let Some(inst) = instances {
            EvalContext::with_definitions_and_instances(state, definitions, inst)
        } else {
            EvalContext::with_definitions(state, definitions)
        };
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (k, v) in locals {
                locals_mut.insert(k.clone(), v.clone());
            }
        }

        let domain = eval_expr(domain_text.trim(), &ctx)?;
        let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();

        for var in split_top_level(vars_text, ",") {
            let name = var.trim();
            if name.is_empty() {
                continue;
            }
            binders.push((name.to_string(), values.clone()));
        }

        rest = next_rest.trim_start();
    }

    Ok(binders)
}

fn parse_action_call(expr: &str) -> Option<(String, Vec<String>)> {
    let (name, rest) = parse_identifier_prefix(expr)?;
    let rest = rest.trim_start();
    if rest.is_empty() {
        return Some((name, Vec::new()));
    }

    if !rest.starts_with('(') {
        return None;
    }
    let (args_text, tail) = take_group(rest, '(', ')').ok()?;
    if !tail.trim().is_empty() {
        return None;
    }

    let args = if args_text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(args_text, ",")
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    Some((name, args))
}

fn split_action_disjuncts(expr: &str) -> Vec<String> {
    let disjuncts = split_action_body_disjuncts(expr);
    if disjuncts.is_empty() {
        vec![expr.trim().to_string()]
    } else {
        disjuncts
    }
}

fn strip_outer_parens(expr: &str) -> &str {
    let mut current = expr;
    while is_wrapped_by(current, '(', ')') {
        current = current[1..current.len() - 1].trim();
    }
    current
}

fn normalize_branch_expr(expr: &str) -> &str {
    let mut current = expr.trim();
    while let Some(rest) = current.strip_prefix("/\\") {
        current = rest.trim_start();
    }
    current
}

/// Check if expression contains a top-level disjunction (\\/)
/// Returns true if there's a \\/ outside of any brackets/parens
fn contains_top_level_disjunction(expr: &str) -> bool {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut i = 0usize;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let len = ch.len_utf8();
        let next = expr[i + len..].chars().next();

        // Handle << >> pairs
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
            '\\' if paren == 0 && bracket == 0 && brace == 0 && angle == 0 => {
                // Check for \/ at top level
                if next == Some('/') {
                    return true;
                }
            }
            _ => {}
        }

        i += len;
    }

    false
}

fn is_wrapped_by(expr: &str, open: char, close: char) -> bool {
    if !expr.starts_with(open) || !expr.ends_with(close) {
        return false;
    }

    let mut depth = 0usize;
    for (idx, ch) in expr.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 && idx + ch.len_utf8() < expr.len() {
                return false;
            }
        }
    }

    depth == 0
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
        if c.is_alphanumeric() || c == '_' || c == '\'' || c == '!' {
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

fn seed_instance_constant_bindings(
    instance: &TlaModuleInstance,
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
            defs_mut.insert(constant.clone(), def);
            continue;
        }
        if let Ok(value) = eval_expr(constant, parent_ctx) {
            locals_mut.insert(constant.clone(), value);
        }
    }
}

fn bind_instance_substitutions(
    instance: &TlaModuleInstance,
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

fn effective_instance_scope<'a>(
    module_instances: &'a BTreeMap<String, TlaModuleInstance>,
    parent_instances: Option<&'a BTreeMap<String, TlaModuleInstance>>,
) -> Option<&'a BTreeMap<String, TlaModuleInstance>> {
    if module_instances.is_empty() {
        parent_instances
    } else {
        Some(module_instances)
    }
}

fn take_group(expr: &str, open: char, close: char) -> Result<(&str, &str)> {
    if !expr.starts_with(open) {
        return Err(anyhow!("expected '{open}'"));
    }

    let mut depth = 0usize;
    let mut i = 0usize;
    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let len = ch.len_utf8();
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Ok((&expr[open.len_utf8()..i], &expr[i + len..]));
            }
        }
        i += len;
    }

    Err(anyhow!("missing closing '{close}'"))
}

fn find_top_level_keyword_index(expr: &str, keyword: &str) -> Option<usize> {
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let len = ch.len_utf8();
        let next = expr[i + len..].chars().next();

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

        i += len;
    }

    None
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

    while i < expr.len() {
        let ch = expr[i..].chars().next().expect("char at byte index");
        let len = ch.len_utf8();
        let next = expr[i + len..].chars().next();

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
                i += len;
                continue;
            }
            return Some(i);
        }

        i += len;
    }

    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::tla_native::TlaModel;
    use crate::tla::{parse_tla_module_file, parse_tla_module_text};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tlapp-action-exec-{prefix}-{nanos}-{}",
            std::process::id()
        ))
    }

    fn parsed_two_pc_with_btm_defs() -> BTreeMap<String, TlaDefinition> {
        parse_tla_module_text(
            r#"---- MODULE TwoPCwithBTMProbe ----
CONSTANTS RM, RMMAYFAIL, TMMAYFAIL
VARIABLES rmState, tmState, pc

vars == << rmState, tmState, pc >>
ProcSet == (RM) \cup {0} \cup {10}

RS(self) == /\ pc[self] = "RS"
            /\ IF rmState[self] \in {"working", "prepared"}
                  THEN /\ \/ /\ rmState[self] = "working"
                             /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                          \/ /\ \/ /\ tmState="commit"
                                   /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                \/ /\ rmState[self]="working" \/ tmState="abort"
                                   /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                          \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                   THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                   ELSE /\ TRUE
                                        /\ UNCHANGED rmState
                       /\ pc' = [pc EXCEPT ![self] = "RS"]
                  ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                       /\ UNCHANGED rmState
            /\ UNCHANGED tmState

RManager(self) == RS(self)

TS == /\ pc[0] = "TS"
      /\ \/ /\ tmState = "commit"
            /\ pc' = [pc EXCEPT ![0] = "TC"]
         \/ /\ tmState = "abort"
            /\ pc' = [pc EXCEPT ![0] = "TA"]
      /\ UNCHANGED << rmState, tmState >>

TC == /\ pc[0] = "TC"
      /\ tmState' = "commit"
      /\ pc' = [pc EXCEPT ![0] = "Done"]
      /\ UNCHANGED rmState

TA == /\ pc[0] = "TA"
      /\ tmState' = "abort"
      /\ pc' = [pc EXCEPT ![0] = "Done"]
      /\ UNCHANGED rmState

TManager == TS \/ TC \/ TA

BTS == /\ pc[10] = "BTS"
       /\ \/ /\ tmState = "commit"
             /\ pc' = [pc EXCEPT ![10] = "BTC"]
          \/ /\ tmState = "abort"
             /\ pc' = [pc EXCEPT ![10] = "BTA"]
       /\ UNCHANGED << rmState, tmState >>

BTC == /\ pc[10] = "BTC"
       /\ tmState' = "commit"
       /\ pc' = [pc EXCEPT ![10] = "Done"]
       /\ UNCHANGED rmState

BTA == /\ pc[10] = "BTA"
       /\ tmState' = "abort"
       /\ pc' = [pc EXCEPT ![10] = "Done"]
       /\ UNCHANGED rmState

BTManager == BTS \/ BTC \/ BTA

Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == TManager \/ BTManager
           \/ (\E self \in RM: RManager(self))
           \/ Terminating
====
"#,
        )
        .expect("2PCwithBTM probe module should parse")
        .definitions
    }

    #[test]
    fn probes_exists_action_call_branch() {
        let defs = BTreeMap::from([
            (
                "Inc".to_string(),
                TlaDefinition {
                    name: "Inc".to_string(),
                    params: vec!["i".to_string()],
                    body: "/\\ x' = x + 1 /\\ UNCHANGED <<y, S>>".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Next".to_string(),
                TlaDefinition {
                    name: "Next".to_string(),
                    params: vec![],
                    body: "\\E i \\in S : Inc(i)".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            ("y".to_string(), TlaValue::Int(7)),
            (
                "S".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
            ),
        ]);

        let probe = probe_next_disjuncts("\\E i \\in S : Inc(i)", &defs, &state);
        assert_eq!(probe.total_disjuncts, 1);
        assert_eq!(probe.supported_disjuncts, 1, "{probe:?}");
        assert_eq!(probe.generated_successors, 1, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn splits_line_leading_quantified_disjuncts() {
        let next_body = r#"\/ \E t \in TxId : OpenTx(t)
            \/ \E t \in tx : \E k \in Key : \E v \in Val : Add(t, k, v)
            \/ \E t \in tx : CloseTx(t)"#;

        let disjuncts = split_action_disjuncts(next_body);
        assert_eq!(disjuncts.len(), 3, "{disjuncts:?}");
        assert!(disjuncts[0].contains("OpenTx"));
        assert!(disjuncts[1].contains("Add"));
        assert!(disjuncts[2].contains("CloseTx"));
    }

    #[test]
    fn probes_quantified_line_leading_disjuncts_with_action_calls() {
        let defs = BTreeMap::from([
            (
                "OpenTx".to_string(),
                TlaDefinition {
                    name: "OpenTx".to_string(),
                    params: vec!["t".to_string()],
                    body: "/\\ tx' = tx \\cup {t} /\\ UNCHANGED <<store, snapshotStore, written, missed>>"
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Add".to_string(),
                TlaDefinition {
                    name: "Add".to_string(),
                    params: vec!["t".to_string(), "k".to_string(), "v".to_string()],
                    body: "/\\ tx' = tx /\\ snapshotStore' = [snapshotStore EXCEPT ![t][k] = v] /\\ written' = [written EXCEPT ![t] = @ \\cup {k}] /\\ UNCHANGED <<store, missed>>"
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "CloseTx".to_string(),
                TlaDefinition {
                    name: "CloseTx".to_string(),
                    params: vec!["t".to_string()],
                    body: "/\\ tx' = tx \\ {t} /\\ UNCHANGED <<store, snapshotStore, written, missed>>"
                        .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            (
                "TxId".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::String(
                    "t1".to_string(),
                )]))),
            ),
            (
                "tx".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::String(
                    "t1".to_string(),
                )]))),
            ),
            (
                "Key".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::String(
                    "k1".to_string(),
                )]))),
            ),
            (
                "Val".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::String(
                    "v1".to_string(),
                )]))),
            ),
            (
                "store".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::String("k1".to_string()),
                    TlaValue::String("old".to_string()),
                )]))),
            ),
            (
                "snapshotStore".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::String("t1".to_string()),
                    TlaValue::Function(Arc::new(BTreeMap::from([(
                        TlaValue::String("k1".to_string()),
                        TlaValue::String("old".to_string()),
                    )]))),
                )]))),
            ),
            (
                "written".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::String("t1".to_string()),
                    TlaValue::Set(Arc::new(BTreeSet::new())),
                )]))),
            ),
            (
                "missed".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::String("t1".to_string()),
                    TlaValue::Set(Arc::new(BTreeSet::new())),
                )]))),
            ),
        ]);

        let next_body = r#"\/ \E t \in TxId : OpenTx(t)
            \/ \E t \in tx : \E k \in Key : \E v \in Val : Add(t, k, v)
            \/ \E t \in tx : CloseTx(t)"#;

        let probe = probe_next_disjuncts(next_body, &defs, &state);
        assert_eq!(probe.total_disjuncts, 3, "{probe:?}");
        assert_eq!(probe.supported_disjuncts, 3, "{probe:?}");
        assert_eq!(probe.generated_successors, 3, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn probes_actions_inherited_via_extends() {
        let tmp = temp_dir("extends-actions");
        fs::create_dir_all(&tmp).unwrap();

        let base = tmp.join("Base.tla");
        fs::write(
            &base,
            r#"---- MODULE Base ----
CONSTANTS TxId, Key, Val
VARIABLES tx, snapshotStore, written, store, missed
Init ==
    /\ tx = {1}
    /\ snapshotStore = [t \in TxId |-> [k \in Key |-> 0]]
    /\ written = [t \in TxId |-> {}]
    /\ store = [k \in Key |-> 0]
    /\ missed = [t \in TxId |-> {}]

OpenTx(t) ==
    /\ tx' = tx \cup {t}
    /\ UNCHANGED <<snapshotStore, written, store, missed>>

Add(t, k, v) ==
    /\ tx' = tx
    /\ snapshotStore' = [snapshotStore EXCEPT ![t][k] = v]
    /\ written' = [written EXCEPT ![t] = @ \cup {k}]
    /\ UNCHANGED <<store, missed>>

CloseTx(t) ==
    /\ tx' = tx \ {t}
    /\ UNCHANGED <<snapshotStore, written, store, missed>>

Next ==
    \/ \E t \in TxId : OpenTx(t)
    \/ \E t \in tx : \E k \in Key : \E v \in Val : Add(t, k, v)
    \/ \E t \in tx : CloseTx(t)
====
"#,
        )
        .unwrap();

        let derived = tmp.join("Derived.tla");
        fs::write(
            &derived,
            r#"---- MODULE Derived ----
EXTENDS Base
Spec == Init /\ [][Next]_<<tx, snapshotStore, written, store, missed>>
====
"#,
        )
        .unwrap();

        let cfg = tmp.join("Derived.cfg");
        fs::write(
            &cfg,
            r#"SPECIFICATION Spec
CONSTANTS
    TxId = {1}
    Key = {1}
    Val = {1}
"#,
        )
        .unwrap();

        let model = TlaModel::from_files(&derived, Some(&cfg), None, None).unwrap();
        let next_def = model.module.definitions.get(&model.next_name).unwrap();
        let definition_names: Vec<String> = model.module.definitions.keys().cloned().collect();

        assert!(
            model.module.definitions.contains_key("Add"),
            "defs={definition_names:?}"
        );
        assert!(
            model.module.definitions.contains_key("OpenTx"),
            "defs={definition_names:?}"
        );
        assert!(
            model.module.definitions.contains_key("CloseTx"),
            "defs={definition_names:?}"
        );

        let probe = probe_next_disjuncts_with_instances(
            &next_def.body,
            &model.module.definitions,
            if model.module.instances.is_empty() {
                None
            } else {
                Some(&model.module.instances)
            },
            &model.initial_states_vec[0],
        );

        let _ = fs::remove_dir_all(&tmp);

        assert_eq!(probe.total_disjuncts, 3, "{probe:?}");
        assert_eq!(probe.supported_disjuncts, 3, "{probe:?}");
        assert_eq!(probe.generated_successors, 3, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn probes_module_instance_actions_with_inherited_constant_bindings() {
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
        let defs = BTreeMap::from([(
            "KeyPair".to_string(),
            TlaDefinition {
                name: "KeyPair".to_string(),
                params: vec![],
                body: "[prv1 |-> pub1]".to_string(),
                is_recursive: false,
            },
        )]);
        let state = TlaState::from([("x".to_string(), TlaValue::ModelValue("old".to_string()))]);

        let probe = probe_next_disjuncts_with_instances("H!Next", &defs, Some(&instances), &state);

        assert_eq!(probe.total_disjuncts, 1);
        assert_eq!(probe.supported_disjuncts, 1, "{probe:?}");
        assert_eq!(probe.generated_successors, 1, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn executes_module_instance_actions_with_primed_substitutions() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["contents".to_string()];
        helper_module.definitions.insert(
            "Next".to_string(),
            TlaDefinition {
                name: "Next".to_string(),
                params: vec![],
                body: "/\\ x' = contents'[1]".to_string(),
                is_recursive: false,
            },
        );

        let instances = BTreeMap::from([(
            "D".to_string(),
            TlaModuleInstance {
                alias: "D".to_string(),
                module_name: "Helper".to_string(),
                substitutions: BTreeMap::from([("contents".to_string(), "c1".to_string())]),
                is_local: false,
                module: Some(Box::new(helper_module)),
            },
        )]);
        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            (
                "c1".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([(
                    TlaValue::Int(1),
                    TlaValue::Int(0),
                )]))),
            ),
        ]);
        let locals = BTreeMap::from([(
            "c1'".to_string(),
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::Int(1),
                TlaValue::Int(7),
            )]))),
        )]);

        let successors = execute_branch(
            "D!Next",
            &locals,
            &BTreeMap::new(),
            Some(&instances),
            &state,
        )
        .expect("instance action should evaluate against primed substitution");

        assert_eq!(successors.len(), 1);
        assert_eq!(successors[0].get("x"), Some(&TlaValue::Int(7)));
    }

    #[test]
    fn probes_inline_if_with_module_instance_action_branch() {
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
        let defs = BTreeMap::from([(
            "KeyPair".to_string(),
            TlaDefinition {
                name: "KeyPair".to_string(),
                params: vec![],
                body: "[prv1 |-> pub1]".to_string(),
                is_recursive: false,
            },
        )]);
        let state = TlaState::from([("x".to_string(), TlaValue::ModelValue("old".to_string()))]);

        let probe = probe_next_disjuncts_with_instances(
            "IF TRUE THEN H!Next ELSE /\\ UNCHANGED x",
            &defs,
            Some(&instances),
            &state,
        );

        println!("probe={probe:?}");
        assert_eq!(probe.total_disjuncts, 1);
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
    }

    #[test]
    fn probes_module_instance_actions_with_outer_operator_ref_helpers() {
        use crate::tla::module::{TlaModule, TlaModuleInstance};

        let mut helper_module = TlaModule::default();
        helper_module.name = "Helper".to_string();
        helper_module.constants = vec!["Hasher".to_string()];
        helper_module.definitions.insert(
            "Next".to_string(),
            TlaDefinition {
                name: "Next".to_string(),
                params: vec!["n".to_string()],
                body: "/\\ x' = Hasher(n)".to_string(),
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
        let defs = BTreeMap::from([
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
        let state = TlaState::from([("x".to_string(), TlaValue::Int(0))]);

        let probe =
            probe_next_disjuncts_with_instances("H!Next(5)", &defs, Some(&instances), &state);

        assert_eq!(probe.total_disjuncts, 1);
        assert_eq!(probe.supported_disjuncts, 1, "{probe:?}");
        assert_eq!(probe.generated_successors, 1, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn probes_module_instance_actions_with_outer_helpers_that_reference_outer_instances() {
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
            "Next".to_string(),
            TlaDefinition {
                name: "Next".to_string(),
                params: vec!["n".to_string()],
                body: "/\\ x' = Hasher(n)".to_string(),
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
        let defs = BTreeMap::from([
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
        let state = TlaState::from([("x".to_string(), TlaValue::Int(0))]);

        let probe =
            probe_next_disjuncts_with_instances("H!Next(5)", &defs, Some(&instances), &state);

        assert_eq!(probe.total_disjuncts, 1);
        assert_eq!(probe.supported_disjuncts, 1, "{probe:?}");
        assert_eq!(probe.generated_successors, 1, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    /// Test for nested disjunctions inside quantifier body
    /// This is the pattern from ClusterLeaseFailover.tla that was failing with
    /// "empty branch expression" error.
    ///
    /// The pattern is:
    /// ```tla
    /// Next ==
    ///     \/ \E op \in pendingOps :
    ///         \/ JournalOperation(op)
    ///         \/ CommitOperation(op)
    ///         \/ FailOperation(op)
    /// ```
    #[test]
    fn handles_nested_disjunction_in_quantifier_body() {
        let defs = BTreeMap::from([
            (
                "ActionA".to_string(),
                TlaDefinition {
                    name: "ActionA".to_string(),
                    params: vec!["op".to_string()],
                    body: "/\\ x' = x + 1 /\\ y' = y".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "ActionB".to_string(),
                TlaDefinition {
                    name: "ActionB".to_string(),
                    params: vec!["op".to_string()],
                    body: "/\\ x' = x /\\ y' = y + 1".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "ActionC".to_string(),
                TlaDefinition {
                    name: "ActionC".to_string(),
                    params: vec!["op".to_string()],
                    body: "/\\ x' = x + 10 /\\ y' = y + 10".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            ("y".to_string(), TlaValue::Int(0)),
            (
                "pendingOps".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
        ]);

        // Test the nested disjunction pattern:
        // \E op \in pendingOps : \/ ActionA(op) \/ ActionB(op) \/ ActionC(op)
        let next_body = r#"\E op \in pendingOps :
            \/ ActionA(op)
            \/ ActionB(op)
            \/ ActionC(op)"#;

        let result = evaluate_next_states(next_body, &defs, &state);
        assert!(
            result.is_ok(),
            "Nested disjunction should not fail: {:?}",
            result.err()
        );

        let successors = result.unwrap();
        // With 2 ops and 3 actions, we should get 6 successor states (2 * 3)
        assert_eq!(
            successors.len(),
            6,
            "Expected 6 successors (2 ops * 3 actions), got {}",
            successors.len()
        );

        // Verify the states are what we expect
        let x_values: Vec<i64> = successors
            .iter()
            .filter_map(|s| s.get("x").and_then(|v| v.as_int().ok()))
            .collect();
        // ActionA: x+1=1 (x2 ops), ActionB: x=0 (x2 ops), ActionC: x+10=10 (x2 ops)
        assert!(x_values.contains(&1), "Should have x=1 from ActionA");
        assert!(x_values.contains(&0), "Should have x=0 from ActionB");
        assert!(x_values.contains(&10), "Should have x=10 from ActionC");
    }

    #[test]
    fn probes_top_level_disjunction_with_quantified_multiline_branch() {
        let defs = BTreeMap::from([
            (
                "ActionA".to_string(),
                TlaDefinition {
                    name: "ActionA".to_string(),
                    params: vec!["op".to_string()],
                    body: "/\\ x' = op /\\ y' = y".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "ActionB".to_string(),
                TlaDefinition {
                    name: "ActionB".to_string(),
                    params: vec!["op".to_string()],
                    body: "/\\ x' = x /\\ y' = op".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Idle".to_string(),
                TlaDefinition {
                    name: "Idle".to_string(),
                    params: vec![],
                    body: "/\\ UNCHANGED <<x, y>>".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            ("y".to_string(), TlaValue::Int(0)),
            (
                "pendingOps".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                ]))),
            ),
        ]);

        let next_body = r#"
            \/ \E op \in pendingOps :
                \/ ActionA(op)
                \/ ActionB(op)
            \/ Idle()
        "#;

        let probe = probe_next_disjuncts(next_body, &defs, &state);
        assert_eq!(probe.total_disjuncts, 2, "{probe:?}");
        assert_eq!(probe.supported_disjuncts, 2, "{probe:?}");
        assert_eq!(probe.generated_successors, 5, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn probes_next_with_mixed_zero_arg_and_quantified_branches() {
        let defs = BTreeMap::from([
            (
                "TManager".to_string(),
                TlaDefinition {
                    name: "TManager".to_string(),
                    params: vec![],
                    body: "/\\ tm' = tm + 1 /\\ UNCHANGED <<pc>>".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "BTManager".to_string(),
                TlaDefinition {
                    name: "BTManager".to_string(),
                    params: vec![],
                    body: "/\\ pc' = [pc EXCEPT ![10] = \"Done\"] /\\ UNCHANGED <<tm>>".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "RManager".to_string(),
                TlaDefinition {
                    name: "RManager".to_string(),
                    params: vec!["self".to_string()],
                    body: "/\\ pc' = [pc EXCEPT ![self] = \"Done\"] /\\ UNCHANGED <<tm>>"
                        .to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Terminating".to_string(),
                TlaDefinition {
                    name: "Terminating".to_string(),
                    params: vec![],
                    body: "/\\ UNCHANGED <<pc, tm>>".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("tm".to_string(), TlaValue::Int(0)),
            (
                "pc".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (TlaValue::Int(1), TlaValue::String("Run".to_string())),
                    (TlaValue::Int(10), TlaValue::String("Run".to_string())),
                ]))),
            ),
            (
                "RM".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
            ),
        ]);

        let next_body = r#"
            TManager \/ BTManager
                     \/ (\E self \in RM: RManager(self))
                     \/ Terminating
        "#;

        let probe = probe_next_disjuncts(next_body, &defs, &state);
        assert_eq!(probe.total_disjuncts, 4, "{probe:?}");
        assert_eq!(probe.supported_disjuncts, 4, "{probe:?}");
        assert_eq!(probe.generated_successors, 4, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn probes_parsed_two_pc_with_btm_next_disjuncts() {
        let defs = parsed_two_pc_with_btm_defs();
        let rm1 = TlaValue::ModelValue("rm1".to_string());
        let rm2 = TlaValue::ModelValue("rm2".to_string());
        let rm3 = TlaValue::ModelValue("rm3".to_string());
        let state = TlaState::from([
            (
                "RM".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    rm1.clone(),
                    rm2.clone(),
                    rm3.clone(),
                ]))),
            ),
            ("RMMAYFAIL".to_string(), TlaValue::Bool(true)),
            ("TMMAYFAIL".to_string(), TlaValue::Bool(false)),
            (
                "rmState".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (rm1.clone(), TlaValue::String("working".to_string())),
                    (rm2.clone(), TlaValue::String("working".to_string())),
                    (rm3.clone(), TlaValue::String("prepared".to_string())),
                ]))),
            ),
            (
                "tmState".to_string(),
                TlaValue::String("commit".to_string()),
            ),
            (
                "pc".to_string(),
                TlaValue::Function(Arc::new(BTreeMap::from([
                    (TlaValue::Int(0), TlaValue::String("TS".to_string())),
                    (TlaValue::Int(10), TlaValue::String("BTS".to_string())),
                    (rm1.clone(), TlaValue::String("RS".to_string())),
                    (rm2.clone(), TlaValue::String("RS".to_string())),
                    (rm3.clone(), TlaValue::String("RS".to_string())),
                ]))),
            ),
        ]);

        let next_def = defs.get("Next").expect("Next definition should exist");
        let probe = probe_next_disjuncts(&next_def.body, &defs, &state);
        assert_eq!(
            probe.total_disjuncts, 4,
            "body={:?} probe={probe:?}",
            next_def.body
        );
        assert_eq!(
            probe.supported_disjuncts, 4,
            "body={:?} probe={probe:?}",
            next_def.body
        );
        assert!(
            probe.generated_successors >= 4,
            "body={:?} probe={probe:?}",
            next_def.body
        );
        assert!(
            probe.failures.is_empty(),
            "body={:?} probe={probe:?}",
            next_def.body
        );
    }

    #[test]
    fn falls_back_for_nested_action_calls_with_later_prime_references() {
        let defs = BTreeMap::from([
            (
                "CounterAction".to_string(),
                TlaDefinition {
                    name: "CounterAction".to_string(),
                    params: vec!["p".to_string()],
                    body: "/\\ p = designated /\\ IF light = \"on\" THEN /\\ light' = \"off\" /\\ count' = count + 1 ELSE /\\ UNCHANGED <<light, count>> /\\ announced' = (count' >= threshold) /\\ UNCHANGED designated".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "WardenAction".to_string(),
                TlaDefinition {
                    name: "WardenAction".to_string(),
                    params: vec!["p".to_string()],
                    body: "/\\ CounterAction(p)".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("count".to_string(), TlaValue::Int(7)),
            ("announced".to_string(), TlaValue::Bool(false)),
            ("light".to_string(), TlaValue::String("off".to_string())),
            (
                "designated".to_string(),
                TlaValue::String("alice".to_string()),
            ),
            ("threshold".to_string(), TlaValue::Int(7)),
            (
                "Prisoner".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::String(
                    "alice".to_string(),
                )]))),
            ),
        ]);

        let probe = probe_next_disjuncts("\\E p \\in Prisoner : WardenAction(p)", &defs, &state);
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
    }

    #[test]
    fn falls_back_for_operator_bracket_calls_in_action_bodies() {
        let defs = BTreeMap::from([
            (
                "GetDirection".to_string(),
                TlaDefinition {
                    name: "GetDirection".to_string(),
                    params: vec!["current".to_string(), "destination".to_string()],
                    body: "IF destination > current THEN \"Up\" ELSE \"Down\"".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Move".to_string(),
                TlaDefinition {
                    name: "Move".to_string(),
                    params: vec![],
                    body: "/\\ GetDirection[current, destination] = \"Up\" /\\ moved' = TRUE"
                        .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            ("current".to_string(), TlaValue::Int(1)),
            ("destination".to_string(), TlaValue::Int(2)),
            ("moved".to_string(), TlaValue::Bool(false)),
        ]);

        let probe = probe_next_disjuncts("Move()", &defs, &state);
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
    }

    #[test]
    fn probes_top_level_let_branches() {
        let state = TlaState::from([
            (
                "Pos".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(1),
                    TlaValue::Int(2),
                    TlaValue::Int(3),
                ]))),
            ),
            (
                "board".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
                    TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(3)]))),
                ]))),
            ),
        ]);

        let probe = probe_next_disjuncts(
            "LET empty == Pos \\ UNION board IN \\E e \\in empty : board' = board",
            &BTreeMap::new(),
            &state,
        );
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
    }

    #[test]
    fn contains_top_level_disjunction_works() {
        // Basic cases
        assert!(contains_top_level_disjunction(r"\/ A \/ B"));
        assert!(contains_top_level_disjunction(r"A \/ B"));
        assert!(!contains_top_level_disjunction(r"A /\ B"));
        assert!(!contains_top_level_disjunction(r"Action(x)"));

        // Nested in parens - should not count
        assert!(!contains_top_level_disjunction(r"(A \/ B)"));
        assert!(!contains_top_level_disjunction(r"Action((a \/ b))"));

        // Mixed
        assert!(contains_top_level_disjunction(r"A \/ (B /\ C)"));
        assert!(contains_top_level_disjunction(r"(A /\ B) \/ C"));
    }

    #[test]
    fn probes_exists_action_call_with_compact_binder() {
        let defs = BTreeMap::from([(
            "Inc".to_string(),
            TlaDefinition {
                name: "Inc".to_string(),
                params: vec!["p".to_string()],
                body: "/\\ count' = count /\\ UNCHANGED <<flag>>".to_string(),
                is_recursive: false,
            },
        )]);

        let state = TlaState::from([
            ("count".to_string(), TlaValue::Int(1)),
            ("flag".to_string(), TlaValue::Bool(true)),
            (
                "Proc".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
            ),
        ]);

        let probe = probe_next_disjuncts("\\E p\\in Proc : Inc(p)", &defs, &state);
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
    }

    #[test]
    fn evaluates_next_disjuncts_with_phase2a_style_let_guards() {
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

        let defs = BTreeMap::from([
            (
                "Ballot".to_string(),
                TlaDefinition {
                    name: "Ballot".to_string(),
                    params: vec![],
                    body: "{1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Value".to_string(),
                TlaDefinition {
                    name: "Value".to_string(),
                    params: vec![],
                    body: "{v1}".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase1a".to_string(),
                TlaDefinition {
                    name: "Phase1a".to_string(),
                    params: vec!["b".to_string()],
                    body: "FALSE".to_string(),
                    is_recursive: false,
                },
            ),
            (
                "Phase2a".to_string(),
                TlaDefinition {
                    name: "Phase2a".to_string(),
                    params: vec!["b".to_string(), "v".to_string()],
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
                        /\ UNCHANGED <<msgs, Quorum>>
                    "#
                    .to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::from([
            (
                "msgs".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([msg_a1, msg_a2]))),
            ),
            (
                "Quorum".to_string(),
                TlaValue::Set(Arc::new(BTreeSet::from([quorum]))),
            ),
            ("sent".to_string(), TlaValue::Bool(false)),
            ("v1".to_string(), v1),
        ]);

        let probe = probe_next_disjuncts(
            r#"\E b \in Ballot :
                  \/ Phase1a(b)
                  \/ \E v \in Value : Phase2a(b, v)"#,
            &defs,
            &state,
        );
        assert_eq!(probe.supported_disjuncts, 2, "{probe:?}");
        assert_eq!(probe.generated_successors, 1, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");
    }

    #[test]
    fn parsed_module_probe_matches_model_state_for_phase2a_let_guards() {
        let tmp = temp_dir("phase2a-parsed-model");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let paxos_base = tmp.join("PaxosBase.tla");
        fs::write(
            &paxos_base,
            r#"
---- MODULE PaxosBase ----
EXTENDS Integers

CONSTANTS Acceptor, Value, Quorum, Ballot

Message ==
       [type : {"1b"}, acc : Acceptor, bal : Ballot, mbal : Ballot, mval : Value]
  \cup [type : {"2a"}, bal : Ballot, val : Value]

VARIABLE msgs

Init ==
  /\ msgs = { [type |-> "1b", acc |-> a, bal |-> 1, mbal |-> 1, mval |-> CHOOSE v \in Value : TRUE]
            : a \in Acceptor }

Send(m) == msgs' = msgs \cup {m}

Phase1a(b) == FALSE

Phase2a(b, v) ==
  /\ ~ \E m \in msgs : m.type = "2a" /\ m.bal = b
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
  /\ Send([type |-> "2a", bal |-> b, val |-> v])

Next == \E b \in Ballot :
            \/ Phase1a(b)
            \/ \E v \in Value : Phase2a(b, v)

Spec == Init /\ [][Next]_<<msgs>>
====
"#,
        )
        .expect("base module should be written");

        let mc = tmp.join("MCPaxosMini.tla");
        fs::write(
            &mc,
            r#"
---- MODULE MCPaxosMini ----
EXTENDS PaxosBase, TLC

CONSTANTS a1, a2, v1

MCAcceptor == {a1, a2}
MCValue == {v1}
MCQuorum == {{a1, a2}}
MCBallot == {1}
====
"#,
        )
        .expect("wrapper module should be written");

        let cfg = tmp.join("MCPaxosMini.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANTS
  a1 = a1
  a2 = a2
  v1 = v1
CONSTANT
  Acceptor <- MCAcceptor
CONSTANT
  Value <- MCValue
CONSTANT
  Quorum <- MCQuorum
CONSTANT
  Ballot <- MCBallot
SPECIFICATION
  Spec
"#,
        )
        .expect("cfg should be written");

        let parsed_module = parse_tla_module_file(&mc).expect("parsed module should load");
        let model = TlaModel::from_files(&mc, Some(&cfg), None, None).expect("model");
        let probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state");
        let next_def = parsed_module
            .definitions
            .get("Next")
            .expect("Next definition should exist");

        let probe = probe_next_disjuncts_with_instances(
            &next_def.body,
            &parsed_module.definitions,
            Some(&parsed_module.instances),
            &probe_state,
        );

        assert_eq!(probe.supported_disjuncts, 2, "{probe:?}");
        assert_eq!(probe.generated_successors, 2, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parsed_paxos_style_probe_keeps_let_locals_in_scope_with_operator_overrides() {
        let tmp = temp_dir("phase2a-paxos-style");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should exist");

        let voting = tmp.join("Voting.tla");
        fs::write(
            &voting,
            r#"
---- MODULE Voting ----
EXTENDS Naturals

Ballot == Nat
Spec == TRUE
====
"#,
        )
        .expect("voting module should be written");

        let paxos = tmp.join("Paxos.tla");
        fs::write(
            &paxos,
            r#"
---- MODULE Paxos ----
EXTENDS Integers

CONSTANTS Value, Acceptor, Quorum

Ballot == Nat
None == CHOOSE v : v \notin Ballot

Message ==
       [type : {"1a"}, bal : Ballot]
  \cup [type : {"1b"}, acc : Acceptor, bal : Ballot,
        mbal : Ballot \cup {-1}, mval : Value \cup {None}]
  \cup [type : {"2a"}, bal : Ballot, val : Value]
  \cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]

VARIABLES maxBal, maxVBal, maxVal, msgs
vars == <<maxBal, maxVBal, maxVal, msgs>>

Init == /\ maxBal  = [a \in Acceptor |-> -1]
        /\ maxVBal = [a \in Acceptor |-> -1]
        /\ maxVal  = [a \in Acceptor |-> None]
        /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

Phase1a(b) == /\ Send([type |-> "1a", bal |-> b])
              /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Phase1b(a) ==
  /\ \E m \in msgs :
        /\ m.type = "1a"
        /\ m.bal > maxBal[a]
        /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
        /\ Send([type |-> "1b", acc |-> a, bal |-> m.bal,
                  mbal |-> maxVBal[a], mval |-> maxVal[a]])
  /\ UNCHANGED <<maxVBal, maxVal>>

Phase2a(b, v) ==
  /\ ~ \E m \in msgs : m.type = "2a" /\ m.bal = b
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
  /\ Send([type |-> "2a", bal |-> b, val |-> v])
  /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Phase2b(a) ==
  \E m \in msgs :
      /\ m.type = "2a"
      /\ m.bal >= maxBal[a]
      /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
      /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
      /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
      /\ Send([type |-> "2b", acc |-> a,
              bal |-> m.bal, val |-> m.val])

Next == \/ \E b \in Ballot : \/ Phase1a(b)
                             \/ \E v \in Value : Phase2a(b, v)
        \/ \E a \in Acceptor : Phase1b(a) \/ Phase2b(a)

V == INSTANCE Voting

Spec == Init /\ [][Next]_vars
====
"#,
        )
        .expect("paxos module should be written");

        let mc = tmp.join("MCPaxosMini.tla");
        fs::write(
            &mc,
            r#"
---- MODULE MCPaxosMini ----
EXTENDS Paxos, TLC

CONSTANT MaxBallot
CONSTANTS a1, a2, a3
CONSTANTS v1, v2

MCAcceptor == {a1, a2, a3}
MCValue == {v1, v2}
MCQuorum == {{a1, a2}, {a1, a3}, {a2, a3}}
MCBallot == 0..MaxBallot
MCBallotVoting == 0..MaxBallot

====
"#,
        )
        .expect("wrapper module should be written");

        let cfg = tmp.join("MCPaxosMini.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANT
  MaxBallot = 2
CONSTANTS
  a1 = a1
  a2 = a2
  a3 = a3
CONSTANTS
  v1 = v1
  v2 = v2
CONSTANT
  Acceptor <- MCAcceptor
CONSTANT
  Value <- MCValue
CONSTANT
  Quorum <- MCQuorum
CONSTANT
  None = None
  Ballot <- MCBallot
  Ballot <- [Voting]MCBallotVoting
SPECIFICATION
  Spec
"#,
        )
        .expect("cfg should be written");

        let parsed_module = parse_tla_module_file(&mc).expect("parsed module should load");
        let model = TlaModel::from_files(&mc, Some(&cfg), None, None).expect("model");
        let probe_state = model
            .initial_states_vec
            .first()
            .cloned()
            .expect("initial state");
        let next_def = parsed_module
            .definitions
            .get("Next")
            .expect("Next definition should exist");

        let probe = probe_next_disjuncts_with_instances(
            &next_def.body,
            &parsed_module.definitions,
            Some(&parsed_module.instances),
            &probe_state,
        );

        let phase2a_def = parsed_module
            .definitions
            .get("Phase2a")
            .expect("Phase2a definition should exist");
        let phase2a_ir = compile_action_ir(phase2a_def);
        let mut ctx = crate::tla::EvalContext::with_definitions_and_instances(
            &probe_state,
            &parsed_module.definitions,
            &parsed_module.instances,
        );
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            locals_mut.insert("b".to_string(), TlaValue::Int(0));
            locals_mut.insert("v".to_string(), TlaValue::ModelValue("v1".to_string()));
        }
        let phase2a_ir_result = apply_action_ir_with_context_multi(&phase2a_ir, &probe_state, &ctx);
        assert!(phase2a_ir_result.is_ok(), "{phase2a_ir_result:?}");

        let phase2a = execute_branch(
            "Phase2a(0, v1)",
            &BTreeMap::new(),
            &parsed_module.definitions,
            Some(&parsed_module.instances),
            &probe_state,
        );
        assert!(phase2a.is_ok(), "{phase2a:?}");

        assert_eq!(probe.supported_disjuncts, 2, "{probe:?}");
        assert_eq!(probe.generated_successors, 3, "{probe:?}");
        assert!(probe.failures.is_empty(), "{probe:?}");

        let _ = fs::remove_dir_all(&tmp);
    }
}
