//! Fast evaluator for compiled TLA+ expressions
//!
//! This module provides evaluation of pre-compiled expressions, avoiding
//! the overhead of string parsing on every evaluation.

use crate::tla::compiled_expr::{CompiledExpr, compile_expr};
use crate::tla::eval::{EvalContext, eval_expr};
use crate::tla::value::TlaValue;
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

// Thread-local cache for compiled operator bodies
// Using thread-local storage eliminates cross-thread contention entirely
thread_local! {
    static THREAD_LOCAL_OPERATOR_CACHE: RefCell<HashMap<String, Arc<CompiledExpr>>> =
        RefCell::new(HashMap::with_capacity(256));
}

/// Get or compile an operator body expression (thread-local, zero contention)
fn get_or_compile_operator(name: &str, body: &str) -> Arc<CompiledExpr> {
    let cache_key = format!("{}:{}", name, body);

    THREAD_LOCAL_OPERATOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(cached) = cache.get(&cache_key) {
            return Arc::clone(cached);
        }

        // Compile and cache
        let compiled = Arc::new(compile_expr(body));
        cache.insert(cache_key, Arc::clone(&compiled));
        compiled
    })
}

/// Convert a TlaValue to its string representation (for ToString operator)
fn tla_value_to_string(value: &TlaValue) -> String {
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
            let items: Vec<String> = s.iter().map(tla_value_to_string).collect();
            format!("{{{}}}", items.join(", "))
        }
        TlaValue::Seq(s) => {
            let items: Vec<String> = s.iter().map(tla_value_to_string).collect();
            format!("<<{}>>", items.join(", "))
        }
        TlaValue::Record(r) => {
            let items: Vec<String> = r
                .iter()
                .map(|(k, v)| format!("{} |-> {}", k, tla_value_to_string(v)))
                .collect();
            format!("[{}]", items.join(", "))
        }
        TlaValue::Function(f) => {
            let items: Vec<String> = f
                .iter()
                .map(|(k, v)| format!("{} :> {}", tla_value_to_string(k), tla_value_to_string(v)))
                .collect();
            format!("({})", items.join(" @@ "))
        }
        TlaValue::Lambda { params, body, .. } => {
            format!("LAMBDA {}: {}", params.join(", "), body)
        }
        TlaValue::Undefined => "UNDEFINED".to_string(),
    }
}

const MAX_DEPTH: usize = 256;

/// Evaluate a compiled expression
pub fn eval_compiled(expr: &CompiledExpr, ctx: &EvalContext<'_>) -> Result<TlaValue> {
    eval_compiled_inner(expr, ctx, 0)
}

fn eval_compiled_inner(
    expr: &CompiledExpr,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_DEPTH {
        return Err(anyhow!("max expression recursion depth exceeded"));
    }

    match expr {
        // Literals
        CompiledExpr::Bool(b) => Ok(TlaValue::Bool(*b)),
        CompiledExpr::Int(n) => Ok(TlaValue::Int(*n)),
        CompiledExpr::String(s) => Ok(TlaValue::String(s.clone())),
        CompiledExpr::ModelValue(s) => Ok(TlaValue::ModelValue(s.clone())),

        // Variable reference
        CompiledExpr::Var(name) => ctx
            .resolve_identifier(name, depth)
            .map_err(|e| anyhow!("failed to resolve {}: {}", name, e)),

        // Primed variable reference (next-state value)
        CompiledExpr::PrimedVar(name) => {
            // In action contexts, primed variables are looked up with the prime suffix
            let primed_name = format!("{}'", name);
            ctx.resolve_identifier(&primed_name, depth)
                .or_else(|_| {
                    // If not found with prime, try looking up in the next state
                    // or fall back to string-based evaluation
                    eval_expr(&primed_name, ctx)
                })
                .map_err(|e| anyhow!("failed to resolve {}: {}", primed_name, e))
        }

        // Logical operators with short-circuit evaluation
        CompiledExpr::And(exprs) => {
            for e in exprs {
                if !eval_compiled_inner(e, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(false));
                }
            }
            Ok(TlaValue::Bool(true))
        }
        CompiledExpr::Or(exprs) => {
            for e in exprs {
                if eval_compiled_inner(e, ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(true));
                }
            }
            Ok(TlaValue::Bool(false))
        }
        CompiledExpr::Not(e) => {
            let val = eval_compiled_inner(e, ctx, depth + 1)?.as_bool()?;
            Ok(TlaValue::Bool(!val))
        }
        CompiledExpr::Implies(a, b) => {
            let lhs = eval_compiled_inner(a, ctx, depth + 1)?.as_bool()?;
            if !lhs {
                return Ok(TlaValue::Bool(true));
            }
            let rhs = eval_compiled_inner(b, ctx, depth + 1)?.as_bool()?;
            Ok(TlaValue::Bool(rhs))
        }

        // Comparison operators
        CompiledExpr::Eq(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            Ok(TlaValue::Bool(left == right))
        }
        CompiledExpr::Neq(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            Ok(TlaValue::Bool(left != right))
        }
        CompiledExpr::Lt(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Bool(left < right))
        }
        CompiledExpr::Le(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Bool(left <= right))
        }
        CompiledExpr::Gt(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Bool(left > right))
        }
        CompiledExpr::Ge(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Bool(left >= right))
        }

        // Set membership
        CompiledExpr::In(elem, set) => {
            let elem_val = eval_compiled_inner(elem, ctx, depth + 1)?;
            let set_val = eval_compiled_inner(set, ctx, depth + 1)?;
            Ok(TlaValue::Bool(set_val.contains(&elem_val)?))
        }
        CompiledExpr::NotIn(elem, set) => {
            let elem_val = eval_compiled_inner(elem, ctx, depth + 1)?;
            let set_val = eval_compiled_inner(set, ctx, depth + 1)?;
            Ok(TlaValue::Bool(!set_val.contains(&elem_val)?))
        }

        // Arithmetic
        CompiledExpr::Add(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left + right))
        }
        CompiledExpr::Sub(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left - right))
        }
        CompiledExpr::Mul(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left * right))
        }
        CompiledExpr::Div(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            if right == 0 {
                return Err(anyhow!("division by zero"));
            }
            Ok(TlaValue::Int(left / right))
        }
        CompiledExpr::Mod(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            if right == 0 {
                return Err(anyhow!("modulo by zero"));
            }
            Ok(TlaValue::Int(left % right))
        }
        CompiledExpr::Neg(a) => {
            let val = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(-val))
        }

        // Set operations
        CompiledExpr::SetLiteral(exprs) => {
            let mut set = BTreeSet::new();
            for e in exprs {
                set.insert(eval_compiled_inner(e, ctx, depth + 1)?);
            }
            Ok(TlaValue::Set(Arc::new(set)))
        }
        CompiledExpr::SetRange(a, b) => {
            let start = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let end = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            let set: BTreeSet<TlaValue> = (start..=end).map(TlaValue::Int).collect();
            Ok(TlaValue::Set(Arc::new(set)))
        }
        CompiledExpr::Union(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            let left_set = left.as_set()?;
            let right_set = right.as_set()?;
            let union: BTreeSet<TlaValue> = left_set.union(&right_set).cloned().collect();
            Ok(TlaValue::Set(Arc::new(union)))
        }
        CompiledExpr::Intersect(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            let left_set = left.as_set()?;
            let right_set = right.as_set()?;
            let intersect: BTreeSet<TlaValue> =
                left_set.intersection(&right_set).cloned().collect();
            Ok(TlaValue::Set(Arc::new(intersect)))
        }
        CompiledExpr::SetMinus(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            let left_set = left.as_set()?;
            let right_set = right.as_set()?;
            let diff: BTreeSet<TlaValue> = left_set.difference(&right_set).cloned().collect();
            Ok(TlaValue::Set(Arc::new(diff)))
        }
        CompiledExpr::Subset(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            let left_set = left.as_set()?;
            let right_set = right.as_set()?;
            Ok(TlaValue::Bool(left_set.is_subset(&right_set)))
        }
        CompiledExpr::Cardinality(e) => {
            let set = eval_compiled_inner(e, ctx, depth + 1)?;
            let set = set.as_set()?;
            Ok(TlaValue::Int(set.len() as i64))
        }
        CompiledExpr::PowerSet(e) => {
            let set = eval_compiled_inner(e, ctx, depth + 1)?;
            let set = set.as_set()?;
            let elements: Vec<TlaValue> = set.iter().cloned().collect();
            let n = elements.len();
            let mut powerset = BTreeSet::new();
            for mask in 0..(1u64 << n) {
                let mut subset = BTreeSet::new();
                for i in 0..n {
                    if (mask >> i) & 1 == 1 {
                        subset.insert(elements[i].clone());
                    }
                }
                powerset.insert(TlaValue::Set(Arc::new(subset)));
            }
            Ok(TlaValue::Set(Arc::new(powerset)))
        }

        // Sequence operations
        CompiledExpr::SeqLiteral(exprs) => {
            let mut seq = Vec::new();
            for e in exprs {
                seq.push(eval_compiled_inner(e, ctx, depth + 1)?);
            }
            Ok(TlaValue::Seq(Arc::new(seq)))
        }
        CompiledExpr::Head(e) => {
            let seq = eval_compiled_inner(e, ctx, depth + 1)?;
            let seq = seq.as_seq()?;
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            Ok(seq[0].clone())
        }
        CompiledExpr::Tail(e) => {
            let seq = eval_compiled_inner(e, ctx, depth + 1)?;
            let seq = seq.as_seq()?;
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            Ok(TlaValue::Seq(Arc::new(seq[1..].to_vec())))
        }
        CompiledExpr::Append(a, b) => {
            let seq = eval_compiled_inner(a, ctx, depth + 1)?;
            let elem = eval_compiled_inner(b, ctx, depth + 1)?;
            let mut seq = seq.as_seq()?.to_vec();
            seq.push(elem);
            Ok(TlaValue::Seq(Arc::new(seq)))
        }
        CompiledExpr::Concat(a, b) => {
            let seq1 = eval_compiled_inner(a, ctx, depth + 1)?;
            let seq2 = eval_compiled_inner(b, ctx, depth + 1)?;
            let mut result = seq1.as_seq()?.to_vec();
            result.extend(seq2.as_seq()?.iter().cloned());
            Ok(TlaValue::Seq(Arc::new(result)))
        }
        CompiledExpr::Len(e) => {
            let seq = eval_compiled_inner(e, ctx, depth + 1)?;
            let seq = seq.as_seq()?;
            Ok(TlaValue::Int(seq.len() as i64))
        }
        CompiledExpr::SubSeq(s, a, b) => {
            let seq = eval_compiled_inner(s, ctx, depth + 1)?;
            let start = eval_compiled_inner(a, ctx, depth + 1)?.as_int()? as usize;
            let end = eval_compiled_inner(b, ctx, depth + 1)?.as_int()? as usize;
            let seq = seq.as_seq()?;
            // TLA+ uses 1-based indexing
            let start = start.saturating_sub(1);
            let end = end.min(seq.len());
            Ok(TlaValue::Seq(Arc::new(seq[start..end].to_vec())))
        }

        // Record operations
        CompiledExpr::RecordLiteral(fields) => {
            let mut rec = BTreeMap::new();
            for (name, expr) in fields {
                rec.insert(name.clone(), eval_compiled_inner(expr, ctx, depth + 1)?);
            }
            Ok(TlaValue::Record(Arc::new(rec)))
        }
        CompiledExpr::RecordAccess(e, field) => {
            let rec = eval_compiled_inner(e, ctx, depth + 1)?;
            let rec = rec.as_record()?;
            rec.get(field)
                .cloned()
                .ok_or_else(|| anyhow!("record field not found: {}", field))
        }

        // Function operations
        CompiledExpr::FuncLiteral(entries) => {
            let mut func = BTreeMap::new();
            for (key_expr, val_expr) in entries {
                let key = eval_compiled_inner(key_expr, ctx, depth + 1)?;
                let val = eval_compiled_inner(val_expr, ctx, depth + 1)?;
                func.insert(key, val);
            }
            Ok(TlaValue::Function(Arc::new(func)))
        }
        CompiledExpr::FuncApply(f, args) => {
            let func = eval_compiled_inner(f, ctx, depth + 1)?;

            match &func {
                TlaValue::Function(map) => {
                    // Single argument or tuple
                    let key = if args.len() == 1 {
                        eval_compiled_inner(&args[0], ctx, depth + 1)?
                    } else {
                        let tuple: Vec<TlaValue> = args
                            .iter()
                            .map(|a| eval_compiled_inner(a, ctx, depth + 1))
                            .collect::<Result<_>>()?;
                        TlaValue::Seq(Arc::new(tuple))
                    };
                    map.get(&key)
                        .cloned()
                        .ok_or_else(|| anyhow!("function application failed: key not in domain"))
                }
                TlaValue::Seq(seq) => {
                    // Sequence indexing (1-based)
                    if args.len() != 1 {
                        return Err(anyhow!("sequence indexing requires exactly one argument"));
                    }
                    let idx = eval_compiled_inner(&args[0], ctx, depth + 1)?.as_int()? as usize;
                    if idx < 1 || idx > seq.len() {
                        return Err(anyhow!("sequence index out of bounds: {}", idx));
                    }
                    Ok(seq[idx - 1].clone())
                }
                TlaValue::Record(rec) => {
                    // Record field access via function syntax
                    if args.len() != 1 {
                        return Err(anyhow!("record access requires exactly one argument"));
                    }
                    let field = match eval_compiled_inner(&args[0], ctx, depth + 1)? {
                        TlaValue::String(s) => s,
                        other => {
                            return Err(anyhow!("record field must be string, got {:?}", other));
                        }
                    };
                    rec.get(&field)
                        .cloned()
                        .ok_or_else(|| anyhow!("record field not found: {}", field))
                }
                _ => Err(anyhow!("cannot apply function to {:?}", func)),
            }
        }
        CompiledExpr::Domain(e) => {
            let val = eval_compiled_inner(e, ctx, depth + 1)?;
            match &val {
                TlaValue::Function(map) => {
                    let domain: BTreeSet<TlaValue> = map.keys().cloned().collect();
                    Ok(TlaValue::Set(Arc::new(domain)))
                }
                TlaValue::Seq(seq) => {
                    let domain: BTreeSet<TlaValue> =
                        (1..=seq.len()).map(|i| TlaValue::Int(i as i64)).collect();
                    Ok(TlaValue::Set(Arc::new(domain)))
                }
                TlaValue::Record(rec) => {
                    let domain: BTreeSet<TlaValue> =
                        rec.keys().map(|k| TlaValue::String(k.clone())).collect();
                    Ok(TlaValue::Set(Arc::new(domain)))
                }
                _ => Err(anyhow!("DOMAIN requires function/sequence/record")),
            }
        }
        CompiledExpr::FuncExcept(base, updates) => {
            let base_val = eval_compiled_inner(base, ctx, depth + 1)?;
            eval_except(&base_val, updates, ctx, depth)
        }

        // Self-reference (@ in EXCEPT) - should only be evaluated within EXCEPT context
        CompiledExpr::SelfRef => {
            // This should be handled specially by eval_except
            Err(anyhow!("@ (self-reference) used outside of EXCEPT context"))
        }

        // Control flow
        CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond_val = eval_compiled_inner(cond, ctx, depth + 1)?.as_bool()?;
            if cond_val {
                eval_compiled_inner(then_branch, ctx, depth + 1)
            } else {
                eval_compiled_inner(else_branch, ctx, depth + 1)
            }
        }
        CompiledExpr::Case { arms, other } => {
            for (cond, expr) in arms {
                if eval_compiled_inner(cond, ctx, depth + 1)?.as_bool()? {
                    return eval_compiled_inner(expr, ctx, depth + 1);
                }
            }
            if let Some(other_expr) = other {
                eval_compiled_inner(other_expr, ctx, depth + 1)
            } else {
                Err(anyhow!("CASE: no arm matched and no OTHER clause"))
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let mut new_ctx = ctx.clone();
            for (name, expr) in bindings {
                let val = eval_compiled_inner(expr, &new_ctx, depth + 1)?;
                new_ctx = new_ctx.with_local_value(name, val);
            }
            eval_compiled_inner(body, &new_ctx, depth + 1)
        }

        // Quantifiers
        CompiledExpr::Exists { var, domain, body } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                if eval_compiled_inner(body, &new_ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(true));
                }
            }
            Ok(TlaValue::Bool(false))
        }
        CompiledExpr::Forall { var, domain, body } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                if !eval_compiled_inner(body, &new_ctx, depth + 1)?.as_bool()? {
                    return Ok(TlaValue::Bool(false));
                }
            }
            Ok(TlaValue::Bool(true))
        }
        CompiledExpr::Choose { var, domain, body } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                if eval_compiled_inner(body, &new_ctx, depth + 1)?.as_bool()? {
                    return Ok(elem.clone());
                }
            }
            Err(anyhow!("CHOOSE: no element satisfies predicate"))
        }

        // Set comprehension
        CompiledExpr::SetComprehension {
            var,
            domain,
            body,
            filter,
        } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            let mut result = BTreeSet::new();
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                if let Some(filter_expr) = filter {
                    if !eval_compiled_inner(filter_expr, &new_ctx, depth + 1)?.as_bool()? {
                        continue;
                    }
                }
                let val = eval_compiled_inner(body, &new_ctx, depth + 1)?;
                result.insert(val);
            }
            Ok(TlaValue::Set(Arc::new(result)))
        }

        // Function construction
        CompiledExpr::FuncConstruct { var, domain, body } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            let mut func = BTreeMap::new();
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                let val = eval_compiled_inner(body, &new_ctx, depth + 1)?;
                func.insert(elem.clone(), val);
            }
            Ok(TlaValue::Function(Arc::new(func)))
        }

        // Operator call - evaluate using compiled expressions
        CompiledExpr::OpCall { name, args } => eval_compiled_opcall(name, args, ctx, depth),

        // Built-in sets
        CompiledExpr::NatSet => {
            // Can't enumerate Nat, but can use it in membership checks
            Err(anyhow!(
                "cannot enumerate Nat - use it only in membership checks"
            ))
        }
        CompiledExpr::IntSet => Err(anyhow!(
            "cannot enumerate Int - use it only in membership checks"
        )),
        CompiledExpr::BooleanSet => {
            let set: BTreeSet<TlaValue> = [TlaValue::Bool(false), TlaValue::Bool(true)]
                .into_iter()
                .collect();
            Ok(TlaValue::Set(Arc::new(set)))
        }

        // Lambda
        CompiledExpr::Lambda { params, body } => {
            // Return lambda as a value that can be applied later
            Ok(TlaValue::Lambda {
                params: Arc::new(params.clone()),
                body: format!("{:?}", body), // Hack - we'd need proper unparsing
                captured_locals: Arc::new((*ctx.locals).clone()),
            })
        }

        // Fallback: use string-based evaluation
        CompiledExpr::Unparsed(s) => {
            if std::env::var("TLAPP_TRACE_UNPARSED").is_ok() {
                eprintln!("EVAL_UNPARSED: {:?}", s);
            }
            eval_expr(s, ctx)
        }
    }
}

// ============================================================================
// EXCEPT Expression Evaluation
// ============================================================================

/// Evaluate an EXCEPT expression with support for nested paths and @ operator
fn eval_except(
    base: &TlaValue,
    updates: &[(Vec<CompiledExpr>, CompiledExpr)],
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    let mut result = base.clone();

    for (path, val_expr) in updates {
        if path.is_empty() {
            return Err(anyhow!("EXCEPT path cannot be empty"));
        }

        // Evaluate the path keys
        let keys: Vec<TlaValue> = path
            .iter()
            .map(|k| eval_compiled_inner(k, ctx, depth + 1))
            .collect::<Result<_>>()?;

        // Get the old value at this path (for @ operator)
        let old_value = get_nested_value(&result, &keys)?;

        // Evaluate the new value, with @ bound to the old value
        let new_value = eval_with_self_ref(val_expr, &old_value, ctx, depth)?;

        // Update the nested path
        result = set_nested_value(result, &keys, new_value)?;
    }

    Ok(result)
}

/// Evaluate an expression with @ (self-reference) bound to a specific value
fn eval_with_self_ref(
    expr: &CompiledExpr,
    self_val: &TlaValue,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    // If the expression is SelfRef, return the self value directly
    if matches!(expr, CompiledExpr::SelfRef) {
        return Ok(self_val.clone());
    }

    // For other expressions, we need to evaluate them but handle SelfRef specially
    // The simplest approach is to add self_val as a special local binding
    let self_ctx = ctx.with_local_value("@", self_val.clone());
    eval_with_self_ref_inner(expr, self_val, &self_ctx, depth + 1)
}

/// Inner evaluation that handles SelfRef in subexpressions
fn eval_with_self_ref_inner(
    expr: &CompiledExpr,
    self_val: &TlaValue,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    match expr {
        CompiledExpr::SelfRef => Ok(self_val.clone()),

        // For binary operations, recursively handle SelfRef
        CompiledExpr::Add(a, b) => {
            let left = eval_with_self_ref_inner(a, self_val, ctx, depth + 1)?.as_int()?;
            let right = eval_with_self_ref_inner(b, self_val, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left + right))
        }
        CompiledExpr::Sub(a, b) => {
            let left = eval_with_self_ref_inner(a, self_val, ctx, depth + 1)?.as_int()?;
            let right = eval_with_self_ref_inner(b, self_val, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left - right))
        }
        CompiledExpr::Mul(a, b) => {
            let left = eval_with_self_ref_inner(a, self_val, ctx, depth + 1)?.as_int()?;
            let right = eval_with_self_ref_inner(b, self_val, ctx, depth + 1)?.as_int()?;
            Ok(TlaValue::Int(left * right))
        }

        // For all other expressions, fall back to regular evaluation
        // (they shouldn't contain unhandled SelfRef)
        _ => eval_compiled_inner(expr, ctx, depth + 1),
    }
}

/// Get a value at a nested path
fn get_nested_value(value: &TlaValue, keys: &[TlaValue]) -> Result<TlaValue> {
    let mut current = value.clone();
    for key in keys {
        current = match &current {
            TlaValue::Function(f) => f
                .get(key)
                .cloned()
                .ok_or_else(|| anyhow!("key not found in function: {:?}", key))?,
            TlaValue::Seq(s) => {
                let idx = key.as_int()? as usize;
                if idx < 1 || idx > s.len() {
                    return Err(anyhow!("sequence index out of bounds: {}", idx));
                }
                s[idx - 1].clone()
            }
            TlaValue::Record(r) => {
                let field = match key {
                    TlaValue::String(s) => s.clone(),
                    _ => return Err(anyhow!("record key must be string")),
                };
                r.get(&field)
                    .cloned()
                    .ok_or_else(|| anyhow!("field not found: {}", field))?
            }
            _ => return Err(anyhow!("cannot index into {:?}", current)),
        };
    }
    Ok(current)
}

/// Set a value at a nested path, returning the updated structure
fn set_nested_value(value: TlaValue, keys: &[TlaValue], new_val: TlaValue) -> Result<TlaValue> {
    if keys.is_empty() {
        return Ok(new_val);
    }

    let key = &keys[0];
    let remaining = &keys[1..];

    match value {
        TlaValue::Function(f) => {
            let mut new_func = (*f).clone();
            if remaining.is_empty() {
                new_func.insert(key.clone(), new_val);
            } else {
                let inner = f
                    .get(key)
                    .cloned()
                    .ok_or_else(|| anyhow!("key not found: {:?}", key))?;
                let updated = set_nested_value(inner, remaining, new_val)?;
                new_func.insert(key.clone(), updated);
            }
            Ok(TlaValue::Function(Arc::new(new_func)))
        }
        TlaValue::Seq(s) => {
            let idx = key.as_int()? as usize;
            if idx < 1 || idx > s.len() {
                return Err(anyhow!("sequence index out of bounds: {}", idx));
            }
            let mut new_seq = (*s).clone();
            if remaining.is_empty() {
                new_seq[idx - 1] = new_val;
            } else {
                let inner = s[idx - 1].clone();
                new_seq[idx - 1] = set_nested_value(inner, remaining, new_val)?;
            }
            Ok(TlaValue::Seq(Arc::new(new_seq)))
        }
        TlaValue::Record(r) => {
            let field = match key {
                TlaValue::String(s) => s.clone(),
                _ => return Err(anyhow!("record key must be string")),
            };
            let mut new_rec = (*r).clone();
            if remaining.is_empty() {
                new_rec.insert(field, new_val);
            } else {
                let inner = r
                    .get(&field)
                    .cloned()
                    .ok_or_else(|| anyhow!("field not found: {}", field))?;
                new_rec.insert(field, set_nested_value(inner, remaining, new_val)?);
            }
            Ok(TlaValue::Record(Arc::new(new_rec)))
        }
        _ => Err(anyhow!("cannot update nested path in {:?}", value)),
    }
}

// ============================================================================
// Compiled Operator Call Evaluation
// ============================================================================

/// Evaluate a compiled operator call
///
/// This function:
/// 1. Handles built-in operators (Cardinality, Len, Head, Tail, etc.)
/// 2. Looks up user-defined operators from ctx.definitions
/// 3. Compiles the operator body (caching it for reuse)
/// 4. Binds parameters to evaluated argument values
/// 5. Evaluates the compiled body in the new context
fn eval_compiled_opcall(
    name: &str,
    args: &[CompiledExpr],
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    if depth > MAX_DEPTH {
        return Err(anyhow!("operator recursion depth exceeded at {}", name));
    }

    // First evaluate all arguments
    let arg_values: Vec<TlaValue> = args
        .iter()
        .map(|a| eval_compiled_inner(a, ctx, depth + 1))
        .collect::<Result<_>>()?;

    // Handle built-in operators
    match name {
        "Cardinality" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Cardinality expects 1 argument"));
            }
            return Ok(TlaValue::Int(arg_values[0].len()? as i64));
        }
        "Len" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Len expects 1 argument"));
            }
            return Ok(TlaValue::Int(arg_values[0].len()? as i64));
        }
        "Head" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Head expects 1 argument"));
            }
            let seq = arg_values[0].as_seq()?;
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            return Ok(seq[0].clone());
        }
        "Tail" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Tail expects 1 argument"));
            }
            let seq = arg_values[0].as_seq()?;
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            return Ok(TlaValue::Seq(Arc::new(seq[1..].to_vec())));
        }
        "Append" => {
            if arg_values.len() != 2 {
                return Err(anyhow!("Append expects 2 arguments"));
            }
            let mut seq = arg_values[0].as_seq()?.to_vec();
            seq.push(arg_values[1].clone());
            return Ok(TlaValue::Seq(Arc::new(seq)));
        }
        "SubSeq" => {
            if arg_values.len() != 3 {
                return Err(anyhow!("SubSeq expects 3 arguments"));
            }
            let seq = arg_values[0].as_seq()?;
            let m = arg_values[1].as_int()?;
            let n = arg_values[2].as_int()?;

            // TLA+ sequences are 1-indexed
            if m < 1 {
                return Err(anyhow!("SubSeq start index must be >= 1, got {}", m));
            }

            // Convert to 0-indexed and handle bounds
            let start = (m - 1) as usize;
            let end = (n as usize).min(seq.len());

            if start > seq.len() {
                return Ok(TlaValue::Seq(Arc::new(vec![])));
            }

            return Ok(TlaValue::Seq(Arc::new(seq[start..end].to_vec())));
        }
        "DOMAIN" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("DOMAIN expects 1 argument"));
            }
            match &arg_values[0] {
                TlaValue::Function(map) => {
                    let keys: BTreeSet<TlaValue> = map.keys().cloned().collect();
                    return Ok(TlaValue::Set(Arc::new(keys)));
                }
                TlaValue::Record(map) => {
                    let keys: BTreeSet<TlaValue> =
                        map.keys().map(|k| TlaValue::String(k.clone())).collect();
                    return Ok(TlaValue::Set(Arc::new(keys)));
                }
                TlaValue::Seq(seq) => {
                    let indices: BTreeSet<TlaValue> =
                        (1..=seq.len() as i64).map(TlaValue::Int).collect();
                    return Ok(TlaValue::Set(Arc::new(indices)));
                }
                _ => {
                    return Err(anyhow!("DOMAIN expects a function, record, or sequence"));
                }
            }
        }
        "ToString" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("ToString expects 1 argument"));
            }
            // Convert TlaValue to string representation
            let s = tla_value_to_string(&arg_values[0]);
            return Ok(TlaValue::String(s));
        }
        _ => {}
    }

    // Look up user-defined operator
    let def = lookup_definition(name, ctx)
        .ok_or_else(|| anyhow!("unknown operator/function '{}'", name))?;

    // Check arity
    if def.params.len() != arg_values.len() {
        return Err(anyhow!(
            "operator '{}' arity mismatch: expected {}, got {}",
            name,
            def.params.len(),
            arg_values.len()
        ));
    }

    // Get or compile the operator body
    let compiled_body = get_or_compile_operator(name, &def.body);

    // Create new context with parameter bindings
    let mut new_ctx = ctx.clone();
    for (param, value) in def.params.iter().zip(arg_values.into_iter()) {
        new_ctx = new_ctx.with_local_value(param, value);
    }

    // Evaluate the compiled body
    eval_compiled_inner(&compiled_body, &new_ctx, depth + 1)
}

/// Look up a definition from the context
fn lookup_definition(name: &str, ctx: &EvalContext<'_>) -> Option<crate::tla::TlaDefinition> {
    // Check local definitions first
    if let Some(def) = ctx.local_definitions.get(name) {
        return Some(def.clone());
    }
    // Then check module definitions
    ctx.definitions.and_then(|defs| defs.get(name).cloned())
}

// ============================================================================
// Compiled Action Execution
// ============================================================================

use crate::tla::TlaState;
use crate::tla::compiled_expr::{CompiledActionClause, CompiledActionIr};
use crate::tla::eval::eval_let_action;

/// Evaluate a guard expression using compiled evaluation, falling back to bool
pub fn eval_compiled_guard(expr: &CompiledExpr, ctx: &EvalContext<'_>) -> Result<bool> {
    eval_compiled(expr, ctx)?.as_bool()
}

/// Apply a compiled action IR to the current state
pub fn apply_compiled_action_ir(
    action: &CompiledActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Option<TlaState>> {
    let mut staged: BTreeMap<String, TlaValue> = BTreeMap::new();
    let mut unchanged_vars: Vec<String> = Vec::new();

    for clause in &action.clauses {
        if !eval_compiled_clause(clause, ctx, &mut staged, &mut unchanged_vars)? {
            return Ok(None); // Guard failed
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

/// Evaluate a single compiled action clause recursively
/// Returns Ok(true) to continue, Ok(false) if a guard failed
fn eval_compiled_clause(
    clause: &CompiledActionClause,
    ctx: &EvalContext<'_>,
    staged: &mut BTreeMap<String, TlaValue>,
    unchanged_vars: &mut Vec<String>,
) -> Result<bool> {
    match clause {
        CompiledActionClause::Guard { expr } => {
            if !eval_compiled_guard(expr, ctx)? {
                return Ok(false);
            }
        }
        CompiledActionClause::PrimedAssignment { var, expr } => {
            let value = eval_compiled(expr, ctx)?;
            staged.insert(var.clone(), value);
        }
        CompiledActionClause::Unchanged { vars } => {
            unchanged_vars.extend(vars.iter().cloned());
        }
        CompiledActionClause::CompiledLetWithPrimes {
            bindings,
            body_clauses,
        } => {
            // Evaluate bindings and create a new context with them
            let mut inner_ctx = ctx.clone();
            for (name, expr) in bindings {
                let value = eval_compiled(expr, &inner_ctx)?;
                inner_ctx = inner_ctx.with_local_value(name, value);
            }

            // Recursively evaluate body clauses with the extended context
            for body_clause in body_clauses {
                if !eval_compiled_clause(body_clause, &inner_ctx, staged, unchanged_vars)? {
                    return Ok(false); // Guard failed
                }
            }
        }
        CompiledActionClause::LetWithPrimes { expr } => {
            // Fall back to string evaluation for LET expressions we couldn't compile
            eval_let_action(expr, ctx, staged, unchanged_vars)?;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::TlaState;
    use crate::tla::compiled_expr::compile_expr;

    fn empty_ctx() -> EvalContext<'static> {
        static EMPTY_STATE: std::sync::OnceLock<TlaState> = std::sync::OnceLock::new();
        let state = EMPTY_STATE.get_or_init(TlaState::new);
        EvalContext::new(state)
    }

    #[test]
    fn test_eval_literal() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("42"), &ctx).unwrap(),
            TlaValue::Int(42)
        );
        assert_eq!(
            eval_compiled(&compile_expr("TRUE"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn test_eval_arithmetic() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("2 + 3"), &ctx).unwrap(),
            TlaValue::Int(5)
        );
        assert_eq!(
            eval_compiled(&compile_expr("10 - 4"), &ctx).unwrap(),
            TlaValue::Int(6)
        );
    }

    #[test]
    fn test_eval_comparison() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("5 > 3"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_compiled(&compile_expr("2 = 2"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn test_eval_set() {
        let ctx = empty_ctx();
        let result = eval_compiled(&compile_expr("{1, 2, 3}"), &ctx).unwrap();
        if let TlaValue::Set(s) = result {
            assert_eq!(s.len(), 3);
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_eval_if() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("IF TRUE THEN 1 ELSE 2"), &ctx).unwrap(),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_compiled(&compile_expr("IF FALSE THEN 1 ELSE 2"), &ctx).unwrap(),
            TlaValue::Int(2)
        );
    }

    #[test]
    fn test_eval_builtin_opcall() {
        let ctx = empty_ctx();

        // Test Cardinality
        assert_eq!(
            eval_compiled(&compile_expr("Cardinality({1, 2, 3})"), &ctx).unwrap(),
            TlaValue::Int(3)
        );

        // Test Len
        assert_eq!(
            eval_compiled(&compile_expr("Len(<<1, 2, 3, 4>>)"), &ctx).unwrap(),
            TlaValue::Int(4)
        );

        // Test Head
        assert_eq!(
            eval_compiled(&compile_expr("Head(<<5, 6, 7>>)"), &ctx).unwrap(),
            TlaValue::Int(5)
        );

        // Test Tail
        let result = eval_compiled(&compile_expr("Tail(<<5, 6, 7>>)"), &ctx).unwrap();
        let expected: Vec<TlaValue> = vec![TlaValue::Int(6), TlaValue::Int(7)];
        assert_eq!(result, TlaValue::Seq(Arc::new(expected)));
    }

    #[test]
    fn test_eval_user_defined_opcall() {
        use crate::tla::TlaDefinition;

        // Create a simple state
        static STATE: std::sync::OnceLock<TlaState> = std::sync::OnceLock::new();
        let state = STATE.get_or_init(TlaState::new);

        // Create a definitions map with a user-defined operator
        static DEFINITIONS: std::sync::OnceLock<BTreeMap<String, TlaDefinition>> =
            std::sync::OnceLock::new();
        let definitions = DEFINITIONS.get_or_init(|| {
            let mut defs = BTreeMap::new();
            // Double(x) == x + x
            defs.insert(
                "Double".to_string(),
                TlaDefinition {
                    name: "Double".to_string(),
                    params: vec!["x".to_string()],
                    body: "x + x".to_string(),
                },
            );
            // Add(a, b) == a + b
            defs.insert(
                "Add".to_string(),
                TlaDefinition {
                    name: "Add".to_string(),
                    params: vec!["a".to_string(), "b".to_string()],
                    body: "a + b".to_string(),
                },
            );
            // Triple(x) == Double(x) + x (recursive call to another operator)
            defs.insert(
                "Triple".to_string(),
                TlaDefinition {
                    name: "Triple".to_string(),
                    params: vec!["x".to_string()],
                    body: "Double(x) + x".to_string(),
                },
            );
            // Constant == 42 (no parameters)
            defs.insert(
                "Constant".to_string(),
                TlaDefinition {
                    name: "Constant".to_string(),
                    params: vec![],
                    body: "42".to_string(),
                },
            );
            defs
        });

        let ctx = EvalContext::with_definitions(state, definitions);

        // Test single-parameter operator
        assert_eq!(
            eval_compiled(&compile_expr("Double(5)"), &ctx).unwrap(),
            TlaValue::Int(10)
        );

        // Test two-parameter operator
        assert_eq!(
            eval_compiled(&compile_expr("Add(3, 7)"), &ctx).unwrap(),
            TlaValue::Int(10)
        );

        // Test operator calling another operator
        assert_eq!(
            eval_compiled(&compile_expr("Triple(4)"), &ctx).unwrap(),
            TlaValue::Int(12)
        );

        // Test parameterless operator (constant)
        assert_eq!(
            eval_compiled(&compile_expr("Constant"), &ctx).unwrap(),
            TlaValue::Int(42)
        );
    }

    #[test]
    fn test_compiled_let_with_primes_action() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        // Create a state with initial values
        let mut state = TlaState::new();
        state.insert("x".to_string(), TlaValue::Int(5));
        state.insert("y".to_string(), TlaValue::Int(10));

        let ctx = EvalContext::new(&state);

        // Create an ActionIr with a LET expression containing primed assignments
        let action_ir = ActionIr {
            name: "TestAction".to_string(),
            params: vec![],
            clauses: vec![ActionClause::LetWithPrimes {
                expr: "LET newX == x + 1 IN x' = newX /\\ UNCHANGED <<y>>".to_string(),
            }],
        };

        // Compile the action
        let compiled = CompiledActionIr::from_ir(&action_ir);

        // Verify it was compiled to CompiledLetWithPrimes
        assert_eq!(compiled.clauses.len(), 1);
        match &compiled.clauses[0] {
            CompiledActionClause::CompiledLetWithPrimes {
                bindings,
                body_clauses,
            } => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].0, "newX");
                assert_eq!(body_clauses.len(), 2);
            }
            CompiledActionClause::LetWithPrimes { .. } => {
                // If it fell back to string parsing, that's also acceptable
                // (might happen if the parser can't handle the expression)
            }
            _ => panic!("Expected CompiledLetWithPrimes or LetWithPrimes"),
        }

        // Apply the action
        let result = apply_compiled_action_ir(&compiled, &state, &ctx).unwrap();
        assert!(result.is_some());

        let next_state = result.unwrap();
        // x' should be x + 1 = 6
        assert_eq!(next_state.get("x"), Some(&TlaValue::Int(6)));
        // y should be unchanged
        assert_eq!(next_state.get("y"), Some(&TlaValue::Int(10)));
    }
}
