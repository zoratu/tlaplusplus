//! Fast evaluator for compiled TLA+ expressions
//!
//! This module provides evaluation of pre-compiled expressions, avoiding
//! the overhead of string parsing on every evaluation.

use crate::tla::compiled_expr::{CompiledExpr, compile_expr};
use crate::tla::eval::{EvalContext, eval_expr};
use crate::tla::value::TlaValue;
use anyhow::{Result, anyhow};
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
            Ok(TlaValue::Bool(compiled_membership_contains(
                &elem_val,
                set,
                ctx,
                depth + 1,
            )?))
        }
        CompiledExpr::NotIn(elem, set) => {
            let elem_val = eval_compiled_inner(elem, ctx, depth + 1)?;
            Ok(TlaValue::Bool(!compiled_membership_contains(
                &elem_val,
                set,
                ctx,
                depth + 1,
            )?))
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
            let lhs = eval_compiled_inner(a, ctx, depth + 1)?;
            let rhs = eval_compiled_inner(b, ctx, depth + 1)?;
            // Handle both string and sequence concatenation
            match (lhs, rhs) {
                (TlaValue::String(mut a), TlaValue::String(b)) => {
                    a.push_str(&b);
                    Ok(TlaValue::String(a))
                }
                (TlaValue::Seq(a), TlaValue::Seq(b)) => {
                    let mut result = (*a).clone();
                    result.extend(b.iter().cloned());
                    Ok(TlaValue::Seq(Arc::new(result)))
                }
                (a, b) => Err(anyhow!(
                    "\\o expects String or Seq operands, got {:?} and {:?}",
                    a,
                    b
                )),
            }
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

        // TLC module: Function pair constructor (a :> b creates {a -> b})
        CompiledExpr::FuncPair(key_expr, val_expr) => {
            let key = eval_compiled_inner(key_expr, ctx, depth + 1)?;
            let val = eval_compiled_inner(val_expr, ctx, depth + 1)?;
            let mut func = BTreeMap::new();
            func.insert(key, val);
            Ok(TlaValue::Function(Arc::new(func)))
        }

        // TLC module: Function override (f @@ g merges functions, g takes precedence)
        CompiledExpr::FuncOverride(left, right) => {
            let left_val = eval_compiled_inner(left, ctx, depth + 1)?;
            let right_val = eval_compiled_inner(right, ctx, depth + 1)?;

            // Both operands must be functions
            let left_func = left_val.as_function()?;
            let right_func = right_val.as_function()?;

            // Merge: start with left, override with right
            let mut result = left_func.clone();
            for (k, v) in right_func.iter() {
                result.insert(k.clone(), v.clone());
            }
            Ok(TlaValue::Function(Arc::new(result)))
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

        // Function set: [Domain -> Range] - set of all functions from Domain to Range
        CompiledExpr::FunctionSet { domain, range } => {
            let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
            let range_val = eval_compiled_inner(range, ctx, depth + 1)?;
            let domain_set = domain_val.as_set()?;
            let range_set = range_val.as_set()?;

            // Generate all possible functions from domain to range
            // For domain of size n and range of size m, there are m^n functions
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
            // m^n can be huge; limit to reasonable size
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

            // Generate all functions by iterating through all combinations
            // Each function is a mapping from each domain element to some range element
            let mut result = BTreeSet::new();

            // indices[i] is the index into range_elems for domain_elems[i]
            let mut indices = vec![0usize; n];

            loop {
                // Build function for current indices
                let mut func = BTreeMap::new();
                for (i, d) in domain_elems.iter().enumerate() {
                    func.insert(d.clone(), range_elems[indices[i]].clone());
                }
                result.insert(TlaValue::Function(Arc::new(func)));

                // Increment indices (like counting in base m)
                let mut carry = true;
                for i in 0..n {
                    if carry {
                        indices[i] += 1;
                        if indices[i] >= m {
                            indices[i] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
                if carry {
                    // All combinations exhausted
                    break;
                }
            }

            Ok(TlaValue::Set(Arc::new(result)))
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
        CompiledExpr::Unparsed(s) => eval_expr(s, ctx),
    }
}

fn compiled_membership_contains(
    value: &TlaValue,
    set_expr: &CompiledExpr,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<bool> {
    match set_expr {
        CompiledExpr::NatSet => Ok(matches!(value.as_int(), Ok(n) if n >= 0)),
        CompiledExpr::IntSet => Ok(value.as_int().is_ok()),
        CompiledExpr::BooleanSet => Ok(matches!(value, TlaValue::Bool(_))),
        CompiledExpr::Var(name) => {
            if let Some(def) = ctx.definition(name)
                && def.params.is_empty()
            {
                return membership_matches_text(value, &def.body, ctx, depth + 1);
            }
            let set_val = eval_compiled_inner(set_expr, ctx, depth + 1)?;
            set_val.contains(value)
        }
        CompiledExpr::Unparsed(text) => membership_matches_text(value, text, ctx, depth + 1),
        _ => {
            let set_val = eval_compiled_inner(set_expr, ctx, depth + 1)?;
            set_val.contains(value)
        }
    }
}

fn membership_matches_text(
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
            if let Some(def) = ctx.definition(rhs_trimmed)
                && def.params.is_empty()
            {
                return membership_matches_text(value, &def.body, ctx, depth + 1);
            }

            if let Some(inner) = rhs_trimmed.strip_prefix("Seq(") {
                if let Some(set_expr) = inner.strip_suffix(")") {
                    return match value {
                        TlaValue::Seq(seq) => {
                            let mut all_in_set = true;
                            for elem in seq.iter() {
                                if !membership_matches_text(elem, set_expr, ctx, depth + 1)? {
                                    all_in_set = false;
                                    break;
                                }
                            }
                            Ok(all_in_set)
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
                            let domain_val = crate::tla::eval::eval_expr(domain_expr, ctx)?;
                            let domain_set = match domain_val.as_set() {
                                Ok(set) => set,
                                Err(_) => return Ok(false),
                            };
                            let func_domain: std::collections::BTreeSet<TlaValue> =
                                func.keys().cloned().collect();
                            if func_domain != *domain_set {
                                return Ok(false);
                            }
                            for item in func.values() {
                                if !membership_matches_text(item, codomain_expr, ctx, depth + 1)? {
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
                            let field_specs: Vec<&str> = inner.split(',').collect();
                            let mut expected_fields = std::collections::HashSet::<String>::new();
                            for spec in field_specs {
                                let parts: Vec<&str> = spec.split(':').collect();
                                if parts.len() != 2 {
                                    continue;
                                }
                                let field_name = parts[0].trim();
                                let field_type = parts[1].trim();
                                expected_fields.insert(field_name.to_string());
                                let Some(field_value) = rec.get(field_name) else {
                                    return Ok(false);
                                };
                                if !membership_matches_text(
                                    field_value,
                                    field_type,
                                    ctx,
                                    depth + 1,
                                )? {
                                    return Ok(false);
                                }
                            }
                            Ok(rec.len() == expected_fields.len())
                        }
                        _ => Ok(false),
                    };
                }
            }

            let set_val = crate::tla::eval::eval_expr(rhs_trimmed, ctx)?;
            set_val.contains(value)
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
use crate::tla::eval::{eval_let_action_multi, parse_action_binder_specs};

/// Evaluate a guard expression using compiled evaluation, falling back to bool
pub fn eval_compiled_guard(expr: &CompiledExpr, ctx: &EvalContext<'_>) -> Result<bool> {
    eval_compiled(expr, ctx)?.as_bool()
}

#[derive(Debug, Clone)]
struct CompiledActionBranch<'a> {
    ctx: EvalContext<'a>,
    staged: BTreeMap<String, TlaValue>,
    unchanged_vars: Vec<String>,
}

/// Apply a compiled action IR to the current state.
///
/// This wrapper preserves the older single-successor shape for legacy callers.
pub fn apply_compiled_action_ir(
    action: &CompiledActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Option<TlaState>> {
    Ok(apply_compiled_action_ir_multi(action, current, ctx)?
        .into_iter()
        .next())
}

pub fn apply_compiled_action_ir_multi(
    action: &CompiledActionIr,
    current: &TlaState,
    ctx: &EvalContext<'_>,
) -> Result<Vec<TlaState>> {
    let branches = eval_compiled_clause_sequence(
        &action.clauses,
        vec![CompiledActionBranch {
            ctx: ctx.clone(),
            staged: BTreeMap::new(),
            unchanged_vars: Vec::new(),
        }],
    )?;
    let mut out = Vec::with_capacity(branches.len());
    for branch in branches {
        let mut next = current.clone();
        for var in branch.unchanged_vars {
            if let Some(old) = current.get(&var) {
                next.insert(var, old.clone());
            }
        }
        for (var, value) in branch.staged {
            next.insert(var, value);
        }
        out.push(next);
    }
    Ok(out)
}

fn eval_compiled_clause_sequence<'a>(
    clauses: &[CompiledActionClause],
    branches: Vec<CompiledActionBranch<'a>>,
) -> Result<Vec<CompiledActionBranch<'a>>> {
    let mut branches = branches;
    for clause in clauses {
        let mut next_branches = Vec::new();
        for branch in branches {
            next_branches.extend(eval_compiled_clause_to_branch(clause, branch)?);
        }
        branches = next_branches;
        if branches.is_empty() {
            break;
        }
    }
    Ok(branches)
}

fn eval_compiled_clause_to_branch<'a>(
    clause: &CompiledActionClause,
    branch: CompiledActionBranch<'a>,
) -> Result<Vec<CompiledActionBranch<'a>>> {
    let eval_ctx = ctx_with_staged_primes(&branch.ctx, &branch.staged);
    match clause {
        CompiledActionClause::Guard { expr } => {
            if eval_compiled_guard(expr, &eval_ctx)? {
                Ok(vec![branch])
            } else {
                Ok(Vec::new())
            }
        }
        CompiledActionClause::Exists {
            binders,
            body_clauses,
        } => {
            let binder_specs = parse_action_binder_specs(binders)?;
            let outer_ctx = branch.ctx.clone();
            let branches = expand_compiled_exists_branches(
                &binder_specs,
                0,
                &branch.ctx,
                &branch.staged,
                &branch.unchanged_vars,
                body_clauses,
            )?;
            Ok(branches
                .into_iter()
                .map(|inner| CompiledActionBranch {
                    ctx: outer_ctx.clone(),
                    staged: inner.staged,
                    unchanged_vars: inner.unchanged_vars,
                })
                .collect())
        }
        CompiledActionClause::Conditional {
            condition,
            then_clauses,
            else_clauses,
        } => {
            let selected = if eval_compiled_guard(condition, &eval_ctx)? {
                then_clauses
            } else {
                else_clauses
            };
            eval_compiled_clause_sequence(selected, vec![branch])
        }
        CompiledActionClause::PrimedAssignment { var, expr } => {
            let mut branch = branch;
            let value = eval_compiled(expr, &eval_ctx)?;
            branch.staged.insert(var.clone(), value);
            Ok(vec![branch])
        }
        CompiledActionClause::Unchanged { vars } => {
            let mut branch = branch;
            branch.unchanged_vars.extend(vars.iter().cloned());
            Ok(vec![branch])
        }
        CompiledActionClause::CompiledLetWithPrimes {
            bindings,
            body_clauses,
        } => {
            let mut inner_ctx = eval_ctx.clone();
            for (name, expr) in bindings {
                let value = eval_compiled(expr, &inner_ctx)?;
                inner_ctx = inner_ctx.with_local_value(name, value);
            }
            let outer_ctx = branch.ctx.clone();
            let branches = eval_compiled_clause_sequence(
                body_clauses,
                vec![CompiledActionBranch {
                    ctx: inner_ctx,
                    staged: branch.staged,
                    unchanged_vars: branch.unchanged_vars,
                }],
            )?;
            Ok(branches
                .into_iter()
                .map(|inner| CompiledActionBranch {
                    ctx: outer_ctx.clone(),
                    staged: inner.staged,
                    unchanged_vars: inner.unchanged_vars,
                })
                .collect())
        }
        CompiledActionClause::LetWithPrimes { expr } => {
            Ok(
                eval_let_action_multi(expr, &eval_ctx, &branch.staged, &branch.unchanged_vars)?
                    .into_iter()
                    .map(|(staged, unchanged_vars)| CompiledActionBranch {
                        ctx: branch.ctx.clone(),
                        staged,
                        unchanged_vars,
                    })
                    .collect(),
            )
        }
    }
}

fn expand_compiled_exists_branches<'a>(
    binder_specs: &[(String, String)],
    idx: usize,
    ctx: &EvalContext<'a>,
    staged: &BTreeMap<String, TlaValue>,
    unchanged_vars: &[String],
    body_clauses: &[CompiledActionClause],
) -> Result<Vec<CompiledActionBranch<'a>>> {
    if idx >= binder_specs.len() {
        return eval_compiled_clause_sequence(
            body_clauses,
            vec![CompiledActionBranch {
                ctx: ctx.clone(),
                staged: staged.clone(),
                unchanged_vars: unchanged_vars.to_vec(),
            }],
        );
    }

    let (name, domain_text) = &binder_specs[idx];
    let eval_ctx = ctx_with_staged_primes(ctx, staged);
    let domain = eval_expr(domain_text, &eval_ctx)?;
    let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
    let mut out = Vec::new();
    for value in values {
        let child_ctx = ctx.with_local_value(name.clone(), value);
        out.extend(expand_compiled_exists_branches(
            binder_specs,
            idx + 1,
            &child_ctx,
            staged,
            unchanged_vars,
            body_clauses,
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

    #[test]
    fn test_compiled_conditional_action() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        let state = TlaState::from([
            ("flag".to_string(), TlaValue::Bool(false)),
            ("x".to_string(), TlaValue::Int(7)),
            ("y".to_string(), TlaValue::Int(9)),
        ]);
        let ctx = EvalContext::new(&state);

        let action_ir = ActionIr {
            name: "Conditional".to_string(),
            params: vec![],
            clauses: vec![ActionClause::Guard {
                expr: "IF flag THEN /\\ x' = x + 1 /\\ UNCHANGED y ELSE /\\ UNCHANGED x /\\ y' = y + 1"
                    .to_string(),
            }],
        };

        let compiled = CompiledActionIr::from_ir(&action_ir);
        let result = apply_compiled_action_ir(&compiled, &state, &ctx).unwrap();
        let next_state = result.expect("conditional branch should succeed");
        assert_eq!(next_state.get("x"), Some(&TlaValue::Int(7)));
        assert_eq!(next_state.get("y"), Some(&TlaValue::Int(10)));
    }

    #[test]
    fn test_compiled_quantified_let_action_generates_multiple_successors() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        let state = TlaState::from([
            ("x".to_string(), TlaValue::Int(0)),
            ("y".to_string(), TlaValue::Int(9)),
        ]);
        let ctx = EvalContext::new(&state);

        let action_ir = ActionIr {
            name: "Pick".to_string(),
            params: vec![],
            clauses: vec![ActionClause::LetWithPrimes {
                expr: "LET choices == {1, 2} IN /\\ \\E pick \\in choices : /\\ x' = pick /\\ UNCHANGED <<y>>".to_string(),
            }],
        };

        let compiled = CompiledActionIr::from_ir(&action_ir);
        let next_states = apply_compiled_action_ir_multi(&compiled, &state, &ctx).unwrap();
        assert_eq!(next_states.len(), 2);
        let next_xs: std::collections::BTreeSet<i64> = next_states
            .iter()
            .map(|next| next.get("x").unwrap().as_int().unwrap())
            .collect();
        assert_eq!(next_xs, std::collections::BTreeSet::from([1, 2]));
        for next in next_states {
            assert_eq!(next.get("y"), Some(&TlaValue::Int(9)));
        }
    }
}

#[cfg(test)]
mod forall_tests {
    use super::*;
    use crate::tla::TlaState;
    use crate::tla::compiled_expr::compile_expr;
    use std::sync::Arc;

    #[test]
    fn test_forall_with_record_access_evaluation() {
        // Create a state with 'listings' as a set of records
        let mut state = TlaState::new();

        // Create a set of record values
        let mut rec1 = BTreeMap::new();
        rec1.insert("price".to_string(), TlaValue::Int(10));
        rec1.insert("id".to_string(), TlaValue::Int(1));

        let mut rec2 = BTreeMap::new();
        rec2.insert("price".to_string(), TlaValue::Int(15));
        rec2.insert("id".to_string(), TlaValue::Int(2));

        let listings_set: BTreeSet<TlaValue> = [
            TlaValue::Record(Arc::new(rec1)),
            TlaValue::Record(Arc::new(rec2)),
        ]
        .into_iter()
        .collect();

        state.insert(
            "listings".to_string(),
            TlaValue::Set(Arc::new(listings_set)),
        );
        state.insert("MinPrice".to_string(), TlaValue::Int(5));

        // Create context
        let ctx = EvalContext::new(&state);

        // Compile and evaluate
        let expr = compile_expr("\\A l \\in listings : l.price >= MinPrice");
        println!("Evaluating: {:?}", expr);

        let result = eval_compiled(&expr, &ctx);
        println!("Result: {:?}", result);

        assert!(result.is_ok(), "Evaluation failed: {:?}", result);
        assert_eq!(result.unwrap(), TlaValue::Bool(true));
    }
}

#[test]
fn test_full_price_bands_invariant() {
    // Replicate the full PriceBandsRespected invariant
    let expr_str = r#"\A l \in listings : l.price >= MinPrice /\ l.price <= MaxPrice"#;

    let expr = compile_expr(expr_str);
    println!("Full invariant compiled: {:?}", expr);

    // Make sure it's a Forall with And body
    match &expr {
        CompiledExpr::Forall { var, domain, body } => {
            println!("var: {}, domain: {:?}", var, domain);
            println!("body: {:?}", body);

            // Check body is And with two Ge/Le
            match body.as_ref() {
                CompiledExpr::And(parts) => {
                    println!("And with {} parts", parts.len());
                    for (i, part) in parts.iter().enumerate() {
                        println!("Part {}: {:?}", i, part);
                    }
                }
                _ => {
                    println!("Body is not And: {:?}", body);
                }
            }
        }
        _ => {
            panic!("Expected Forall, got: {:?}", expr);
        }
    }

    // Now test evaluation
    let mut state = TlaState::new();

    // Create a set of record values
    let mut rec1 = BTreeMap::new();
    rec1.insert("price".to_string(), TlaValue::Int(10));

    let listings_set: BTreeSet<TlaValue> = [TlaValue::Record(Arc::new(rec1))].into_iter().collect();

    state.insert(
        "listings".to_string(),
        TlaValue::Set(Arc::new(listings_set)),
    );
    state.insert("MinPrice".to_string(), TlaValue::Int(5));
    state.insert("MaxPrice".to_string(), TlaValue::Int(20));

    let ctx = EvalContext::new(&state);

    let result = eval_compiled(&expr, &ctx);
    println!("Result: {:?}", result);
    assert!(result.is_ok(), "Evaluation failed: {:?}", result);
    assert_eq!(result.unwrap(), TlaValue::Bool(true));
}

#[test]
fn test_set_comprehension_with_except() {
    // Regression test for: {[w EXCEPT !.state = "FAILED"] : w \in pendingWrites}
    // This should apply EXCEPT to each record in the set, not treat 'w' as a ModelValue
    use crate::tla::compiled_expr::compile_expr;

    // Create a state with pendingWrites as a set of records
    let mut state = TlaState::new();

    // Create a set of record values
    let mut rec1 = BTreeMap::new();
    rec1.insert("id".to_string(), TlaValue::Int(1));
    rec1.insert("state".to_string(), TlaValue::String("PENDING".to_string()));

    let mut rec2 = BTreeMap::new();
    rec2.insert("id".to_string(), TlaValue::Int(2));
    rec2.insert("state".to_string(), TlaValue::String("PENDING".to_string()));

    let pending_writes: BTreeSet<TlaValue> = [
        TlaValue::Record(Arc::new(rec1)),
        TlaValue::Record(Arc::new(rec2)),
    ]
    .into_iter()
    .collect();

    state.insert(
        "pendingWrites".to_string(),
        TlaValue::Set(Arc::new(pending_writes)),
    );

    let ctx = EvalContext::new(&state);

    // First, let's test that the expression compiles correctly
    let expr_str = r#"{[w EXCEPT !.state = "FAILED"] : w \in pendingWrites}"#;
    let expr = compile_expr(expr_str);
    println!("Compiled expression: {:#?}", expr);

    // Verify it's a SetComprehension with FuncExcept body
    match &expr {
        CompiledExpr::SetComprehension { var, body, .. } => {
            assert_eq!(var, "w");
            match body.as_ref() {
                CompiledExpr::FuncExcept(base, _) => {
                    // The base should be Var("w"), not ModelValue("w")
                    match base.as_ref() {
                        CompiledExpr::Var(name) => assert_eq!(name, "w"),
                        other => panic!("Expected Var(\"w\") as EXCEPT base, got: {:?}", other),
                    }
                }
                other => panic!("Expected FuncExcept body, got: {:?}", other),
            }
        }
        other => panic!("Expected SetComprehension, got: {:?}", other),
    }

    // Now evaluate the expression
    let result = eval_compiled(&expr, &ctx);
    println!("Evaluation result: {:?}", result);

    // The evaluation should succeed
    assert!(result.is_ok(), "Evaluation failed: {:?}", result);

    let result_set = result.unwrap();
    let set = result_set.as_set().expect("result should be a set");

    // Each record in the result should have state = "FAILED"
    assert_eq!(set.len(), 2, "Expected 2 records in result");
    for record_val in set.iter() {
        let record = record_val.as_record().expect("element should be a record");
        assert_eq!(
            record.get("state"),
            Some(&TlaValue::String("FAILED".to_string())),
            "Record state should be FAILED: {:?}",
            record
        );
    }
}
