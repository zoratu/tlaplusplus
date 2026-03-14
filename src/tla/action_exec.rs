use crate::fairness::{ActionLabel, LabeledTransition};
use crate::tla::module::TlaModuleInstance;
use crate::tla::{
    CompiledActionIr, EvalContext, TlaDefinition, TlaState, TlaValue,
    apply_compiled_action_ir_multi, compile_action_ir, eval_expr, normalize_param_name,
    split_top_level,
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
    let disjuncts = split_top_level(next_body, "\\/");
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
    let disjuncts = split_top_level(next_body, "\\/");
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
    let disjuncts = split_top_level(next_body, "\\/");
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
    let trimmed = strip_outer_parens(expr.trim());
    if trimmed.is_empty() {
        return Err(anyhow!("empty branch expression"));
    }

    // Handle nested disjunctions: \/ A \/ B \/ C
    // This happens when a quantifier body contains disjunctions, e.g.:
    //   \E op \in pendingOps : \/ JournalOperation(op) \/ CommitOperation(op)
    // We only handle this if split_top_level actually produces multiple parts.
    // If it returns just one part (e.g., when the disjunction is inside a quantifier),
    // we should proceed with other handling (like \E processing).
    if trimmed.starts_with("\\/") || contains_top_level_disjunction(trimmed) {
        let disjuncts = split_top_level(trimmed, "\\/");
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

    // Try to parse as an action call
    if let Some((name, arg_exprs)) = parse_action_call(trimmed) {
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
        {
            let locals_mut = std::rc::Rc::make_mut(&mut ctx.locals);
            for (param, arg) in def.params.iter().zip(args.into_iter()) {
                locals_mut.insert(normalize_param_name(param).to_string(), arg);
            }
        }

        // Use compiled action IR for faster evaluation
        let compiled_ir = get_or_compile_action(def);
        return apply_compiled_action_ir_multi(&compiled_ir, state, &ctx);
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
    apply_compiled_action_ir_multi(&compiled_ir, state, &ctx)
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

fn strip_outer_parens(expr: &str) -> &str {
    let mut current = expr;
    while is_wrapped_by(current, '(', ')') {
        current = current[1..current.len() - 1].trim();
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
            && has_word_boundaries(expr, i, i + keyword.len())
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

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

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
        assert_eq!(probe.supported_disjuncts, 1);
        assert_eq!(probe.generated_successors, 1);
        assert!(probe.failures.is_empty());
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
}
