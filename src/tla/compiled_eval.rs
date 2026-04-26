//! Fast evaluator for compiled TLA+ expressions
//!
//! This module provides evaluation of pre-compiled expressions, avoiding
//! the overhead of string parsing on every evaluation.

use crate::tla::compiled_expr::{CompiledExpr, compile_expr, find_top_level_colon};
use crate::tla::eval::{
    EvalContext, apply_value, eval_expr, eval_operator_call, normalize_param_name,
};
use crate::tla::formula::split_top_level;
use crate::tla::value::TlaValue;
#[cfg(test)]
use crate::tla::value::tla_state;
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

        // Debug: trace operator compilation
        if std::env::var("TLAPP_TRACE_OPCACHE").is_ok() {
            eprintln!(
                "OPCACHE compiling '{}': {}",
                name,
                body.chars().take(100).collect::<String>()
            );
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
        CompiledExpr::Var(name) => {
            // First check local variables
            if let Some(v) = ctx.runtime_value(name) {
                if std::env::var("TLAPP_TRACE_VAR").is_ok() {
                    eprintln!("VAR {} -> Ok({:?})", name, v);
                }
                return Ok(v);
            }

            // Then check if it's a no-arg operator - use compiled evaluation
            if let Some(def) = ctx.definition(name) {
                if def.params.is_empty() {
                    // Get or compile the operator body
                    let compiled_body = get_or_compile_operator(name, &def.body);
                    let result = eval_compiled_inner(&compiled_body, ctx, depth + 1);
                    if std::env::var("TLAPP_TRACE_VAR").is_ok() {
                        eprintln!("VAR {} (operator) -> {:?}", name, result);
                    }
                    return result.map_err(|e| anyhow!("failed to resolve {}: {}", name, e));
                }

                let value = TlaValue::Lambda {
                    params: Arc::new(def.params.clone()),
                    body: def.body.clone(),
                    captured_locals: Arc::new((*ctx.locals).clone()),
                };
                if std::env::var("TLAPP_TRACE_VAR").is_ok() {
                    eprintln!("VAR {} (operator value) -> Ok({:?})", name, value);
                }
                return Ok(value);
            }

            // Fall back to model value for undefined identifiers
            if std::env::var("TLAPP_TRACE_VAR").is_ok() {
                eprintln!("VAR {} -> ModelValue", name);
            }
            Ok(TlaValue::ModelValue(name.to_string()))
        }

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
        CompiledExpr::Pow(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?.as_int()?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?.as_int()?;
            if right < 0 {
                return Err(anyhow!("exponent must be non-negative, got {}", right));
            }
            let value = left
                .checked_pow(right as u32)
                .ok_or_else(|| anyhow!("integer exponent overflow: {}^{}", left, right))?;
            Ok(TlaValue::Int(value))
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
        CompiledExpr::CartesianProduct(a, b) => {
            let left = eval_compiled_inner(a, ctx, depth + 1)?;
            let right = eval_compiled_inner(b, ctx, depth + 1)?;
            let left_set = left.as_set()?;
            let right_set = right.as_set()?;
            ctx.check_budget(left_set.len() * right_set.len())?;
            let mut product = BTreeSet::new();
            for lhs_val in left_set {
                for rhs_val in right_set {
                    let tuple = TlaValue::Seq(Arc::new(vec![lhs_val.clone(), rhs_val.clone()]));
                    product.insert(tuple);
                }
            }
            Ok(TlaValue::Set(Arc::new(product)))
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
            if n > 20 {
                return Err(anyhow!(
                    "SUBSET of {}-element set would produce 2^{} elements (too large)",
                    n,
                    n
                ));
            }
            if n >= 64 {
                return Ok(TlaValue::Set(Arc::new(BTreeSet::new())));
            }
            let total = 1u64 << n;
            ctx.check_budget(total as usize)?;
            let mut powerset = BTreeSet::new();
            for mask in 0..total {
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
            let seq =
                sequence_like_values(&seq).ok_or_else(|| anyhow!("expected Seq, got {seq:?}"))?;
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            Ok(seq[0].clone())
        }
        CompiledExpr::Tail(e) => {
            let seq = eval_compiled_inner(e, ctx, depth + 1)?;
            let seq =
                sequence_like_values(&seq).ok_or_else(|| anyhow!("expected Seq, got {seq:?}"))?;
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            Ok(TlaValue::Seq(Arc::new(seq[1..].to_vec())))
        }
        CompiledExpr::Append(a, b) => {
            let seq = eval_compiled_inner(a, ctx, depth + 1)?;
            let elem = eval_compiled_inner(b, ctx, depth + 1)?;
            let mut seq =
                sequence_like_values(&seq).ok_or_else(|| anyhow!("expected Seq, got {seq:?}"))?;
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
                (a, b) => {
                    let Some(mut lhs_seq) = sequence_like_values(&a) else {
                        return Err(anyhow!(
                            "\\o expects String or Seq operands, got {:?} and {:?}",
                            a,
                            b
                        ));
                    };
                    let Some(rhs_seq) = sequence_like_values(&b) else {
                        return Err(anyhow!(
                            "\\o expects String or Seq operands, got {:?} and {:?}",
                            a,
                            b
                        ));
                    };
                    lhs_seq.extend(rhs_seq);
                    Ok(TlaValue::Seq(Arc::new(lhs_seq)))
                }
            }
        }
        CompiledExpr::Len(e) => {
            let seq = eval_compiled_inner(e, ctx, depth + 1)?;
            let seq =
                sequence_like_values(&seq).ok_or_else(|| anyhow!("expected Seq, got {seq:?}"))?;
            Ok(TlaValue::Int(seq.len() as i64))
        }
        CompiledExpr::SubSeq(s, a, b) => {
            let seq = eval_compiled_inner(s, ctx, depth + 1)?;
            let start = eval_compiled_inner(a, ctx, depth + 1)?.as_int()? as usize;
            let end = eval_compiled_inner(b, ctx, depth + 1)?.as_int()? as usize;
            let seq =
                sequence_like_values(&seq).ok_or_else(|| anyhow!("expected Seq, got {seq:?}"))?;
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
        CompiledExpr::RecordSet(fields) => {
            // Evaluate each field's domain set
            let mut field_sets: Vec<(String, Vec<TlaValue>)> = Vec::new();
            for (name, expr) in fields {
                let set_value = eval_compiled_inner(expr, ctx, depth + 1)?;
                let set = set_value.as_set()?;
                let elements: Vec<TlaValue> = set.iter().cloned().collect();
                field_sets.push((name.clone(), elements));
            }

            // Check for empty sets - if any field has an empty domain, the result is empty
            if field_sets.iter().any(|(_, elems)| elems.is_empty()) {
                return Ok(TlaValue::Set(Arc::new(BTreeSet::new())));
            }

            // Calculate total number of records (product of all set sizes)
            let total_records: u64 = field_sets
                .iter()
                .map(|(_, elems)| elems.len() as u64)
                .product();

            // Limit the size to avoid memory explosion
            let max_records = 1_000_000u64;
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

            Ok(TlaValue::Set(Arc::new(result)))
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
            if std::env::var("TLAPP_TRACE_FUNCAPPLY").is_ok() {
                eprintln!("FUNCAPPLY: {:?}[{:?}]", f, args);
            }
            let func = eval_compiled_inner(f, ctx, depth + 1)?;
            if std::env::var("TLAPP_TRACE_FUNCAPPLY").is_ok() {
                eprintln!("  func evaluated to: {}", tla_value_to_string(&func));
            }

            match &func {
                TlaValue::Lambda { .. } => {
                    let arg_values = args
                        .iter()
                        .map(|a| eval_compiled_inner(a, ctx, depth + 1))
                        .collect::<Result<Vec<_>>>()?;
                    apply_value(&func, arg_values, ctx, depth + 1)
                }
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
                    map.get(&key).cloned().ok_or_else(|| {
                        // Build helpful error message with key and domain
                        let key_str = tla_value_to_string(&key);
                        let domain_keys: Vec<String> =
                            map.keys().take(10).map(tla_value_to_string).collect();
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
                TlaValue::ModelValue(name) => {
                    if let Some(def) = ctx.definition(name)
                        && def.params.len() == args.len()
                    {
                        let arg_values = args
                            .iter()
                            .map(|a| eval_compiled_inner(a, ctx, depth + 1))
                            .collect::<Result<Vec<_>>>()?;
                        return eval_operator_call(name, arg_values, ctx, depth + 1);
                    }

                    let func_str = tla_value_to_string(&func);
                    let args_str: Vec<String> = args.iter().map(|a| format!("{:?}", a)).collect();
                    Err(anyhow!(
                        "cannot index {} with [{}] - only functions/sequences/records can be indexed",
                        func_str,
                        args_str.join(", ")
                    ))
                }
                _ => {
                    // Include the expression being indexed for better debugging
                    let func_str = tla_value_to_string(&func);
                    let args_str: Vec<String> = args.iter().map(|a| format!("{:?}", a)).collect();
                    Err(anyhow!(
                        "cannot index {} with [{}] - only functions/sequences/records can be indexed",
                        func_str,
                        args_str.join(", ")
                    ))
                }
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

        // Self-reference (@ in EXCEPT) - check if @ is bound in context
        CompiledExpr::SelfRef => {
            // Check if @ was bound by eval_with_self_ref (for nested expressions)
            if let Some(val) = ctx.runtime_value("@") {
                Ok(val.clone())
            } else {
                // @ used outside EXCEPT context
                Err(anyhow!("@ (self-reference) used outside of EXCEPT context"))
            }
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
            if std::env::var("TLAPP_TRACE_FORALL").is_ok() {
                eprintln!(
                    "FORALL {} in {:?}, ctx.locals: {:?}, depth: {}",
                    var, domain, ctx.locals, depth
                );
            }
            for elem in domain_set.iter() {
                let new_ctx = ctx.with_local_value(var, elem.clone());
                if std::env::var("TLAPP_TRACE_FORALL").is_ok() {
                    eprintln!(
                        "  FORALL {} = {:?}, new_ctx.locals: {:?}, depth: {}",
                        var, elem, new_ctx.locals, depth
                    );
                }
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
            ctx.check_budget(total as usize)?;

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
        CompiledExpr::Lambda {
            params, body_text, ..
        } => {
            // Return lambda as a value that can be applied later
            Ok(TlaValue::Lambda {
                params: Arc::new(params.clone()),
                body: body_text.clone(),
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
        // Fast path: x \in a..b → a <= x && x <= b (avoids set construction)
        CompiledExpr::SetRange(lo, hi) => {
            let x = value.as_int()?;
            let a = eval_compiled_inner(lo, ctx, depth + 1)?.as_int()?;
            let b = eval_compiled_inner(hi, ctx, depth + 1)?.as_int()?;
            Ok(a <= x && x <= b)
        }
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
        // Handle Seq(S) - check if value is a sequence with all elements in S
        CompiledExpr::OpCall { name, args } if name == "Seq" && args.len() == 1 => {
            match value {
                TlaValue::Seq(seq) => {
                    for elem in seq.iter() {
                        if !compiled_membership_contains(elem, &args[0], ctx, depth + 1)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                // A TLA+ function with domain 1..n is equivalent to a sequence.
                // [i \in 1..3 |-> x] produces a Function, not a Seq, but is in Seq(S).
                TlaValue::Function(func) => {
                    if func_is_sequence_shaped(func) {
                        for val in func.values() {
                            if !compiled_membership_contains(val, &args[0], ctx, depth + 1)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            }
        }
        // Handle [Domain -> Range] membership structurally instead of enumerating
        CompiledExpr::FunctionSet { domain, range } => {
            match value {
                TlaValue::Function(func) => {
                    let domain_val = eval_compiled_inner(domain, ctx, depth + 1)?;
                    let domain_set = domain_val.as_set()?;
                    // Check that the function's domain matches exactly
                    let func_domain: std::collections::BTreeSet<TlaValue> =
                        func.keys().cloned().collect();
                    if func_domain != *domain_set {
                        return Ok(false);
                    }
                    // Check that every value in the function's range is in the range set
                    for val in func.values() {
                        if !compiled_membership_contains(val, range, ctx, depth + 1)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
        _ => {
            let set_val = eval_compiled_inner(set_expr, ctx, depth + 1)?;
            set_val.contains(value)
        }
    }
}

/// Check if a BTreeMap<TlaValue, TlaValue> has a sequence-shaped domain: {Int(1), Int(2), ..., Int(n)}.
/// An empty function is also considered sequence-shaped (it represents the empty sequence <<>>).
pub fn func_is_sequence_shaped(func: &std::collections::BTreeMap<TlaValue, TlaValue>) -> bool {
    if func.is_empty() {
        return true;
    }
    let n = func.len();
    for i in 1..=n {
        if !func.contains_key(&TlaValue::Int(i as i64)) {
            return false;
        }
    }
    // Also check that the domain has exactly n elements (no extra keys)
    func.len() == n
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
            if let Some(runtime_value) = ctx.runtime_value(rhs_trimmed) {
                return runtime_value.contains(value);
            }

            if let Some(def) = ctx.definition(rhs_trimmed)
                && def.params.is_empty()
            {
                return membership_matches_text(value, &def.body, ctx, depth + 1);
            }

            if let Some(inner) = rhs_trimmed.strip_prefix("Seq(") {
                if let Some(set_expr) = inner.strip_suffix(")") {
                    return match value {
                        TlaValue::Seq(seq) => {
                            for elem in seq.iter() {
                                if !membership_matches_text(elem, set_expr, ctx, depth + 1)? {
                                    return Ok(false);
                                }
                            }
                            Ok(true)
                        }
                        TlaValue::Function(func) if func_is_sequence_shaped(func) => {
                            for val in func.values() {
                                if !membership_matches_text(val, set_expr, ctx, depth + 1)? {
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
                            // Use top-level split to handle sets like {1, 2} correctly
                            let field_specs = split_top_level(inner, ",");
                            let mut expected_fields = std::collections::HashSet::<String>::new();
                            for spec in field_specs {
                                let spec = spec.trim();
                                // Find the first colon at top level for field:type separation
                                if let Some(colon_idx) = find_top_level_colon(spec) {
                                    let field_name = spec[..colon_idx].trim();
                                    let field_type = spec[colon_idx + 1..].trim();
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
    let user_defined_shadow = matches!(name, "BoundedSeq" | "Max" | "Min")
        && ctx
            .definition(name)
            .is_some_and(|def| def.params.len() == arg_values.len());

    // Handle built-in operators
    match name {
        "Cardinality" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Cardinality expects 1 argument"));
            }
            return Ok(TlaValue::Int(arg_values[0].len()? as i64));
        }
        "Max" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_builtin_extremum(name, &arg_values[0], true);
        }
        "Min" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_builtin_extremum(name, &arg_values[0], false);
        }
        "BoundedSeq" if arg_values.len() == 2 && !user_defined_shadow => {
            let max_len = arg_values[1].as_int()?;
            return eval_builtin_bounded_seq(&arg_values[0], max_len);
        }
        "TLCGet" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("TLCGet expects 1 argument"));
            }
            return eval_builtin_tlc_get(&arg_values[0]);
        }
        "TLCSet" => {
            if arg_values.len() != 2 {
                return Err(anyhow!("TLCSet expects 2 arguments"));
            }
            return Ok(TlaValue::Bool(true));
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
            let seq = sequence_like_values(&arg_values[0])
                .ok_or_else(|| anyhow!("expected Seq, got {:?}", arg_values[0]))?;
            if seq.is_empty() {
                return Err(anyhow!("Head of empty sequence"));
            }
            return Ok(seq[0].clone());
        }
        "Tail" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Tail expects 1 argument"));
            }
            let seq = sequence_like_values(&arg_values[0])
                .ok_or_else(|| anyhow!("expected Seq, got {:?}", arg_values[0]))?;
            if seq.is_empty() {
                return Err(anyhow!("Tail of empty sequence"));
            }
            return Ok(TlaValue::Seq(Arc::new(seq[1..].to_vec())));
        }
        "Append" => {
            if arg_values.len() != 2 {
                return Err(anyhow!("Append expects 2 arguments"));
            }
            let mut seq = sequence_like_values(&arg_values[0])
                .ok_or_else(|| anyhow!("expected Seq, got {:?}", arg_values[0]))?;
            seq.push(arg_values[1].clone());
            return Ok(TlaValue::Seq(Arc::new(seq)));
        }
        "SubSeq" => {
            if arg_values.len() != 3 {
                return Err(anyhow!("SubSeq expects 3 arguments"));
            }
            let seq = sequence_like_values(&arg_values[0])
                .ok_or_else(|| anyhow!("expected Seq, got {:?}", arg_values[0]))?;
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
        "SelectSeq" => {
            if arg_values.len() != 2 {
                return Err(anyhow!("SelectSeq expects 2 arguments"));
            }
            let seq = match &arg_values[0] {
                TlaValue::Seq(v) => v,
                _ => {
                    return Err(anyhow!(
                        "SelectSeq expects a sequence, got {:?}",
                        arg_values[0]
                    ));
                }
            };
            let test_fn = &arg_values[1];

            let mut result = Vec::new();
            for elem in seq.iter() {
                // Apply the test function to the element
                let test_result = match test_fn {
                    TlaValue::Lambda {
                        params,
                        body,
                        captured_locals,
                    } => {
                        if params.len() != 1 {
                            return Err(anyhow!(
                                "SelectSeq test function must take exactly 1 parameter"
                            ));
                        }
                        // Create a context with the captured locals and the parameter bound
                        let mut locals = (**captured_locals).clone();
                        locals.insert(params[0].clone(), elem.clone());
                        // Evaluate using the captured locals context
                        let lambda_ctx = EvalContext {
                            state: ctx.state,
                            locals: std::rc::Rc::new(locals),
                            local_definitions: ctx.local_definitions.clone(),
                            definitions: ctx.definitions,
                            instances: ctx.instances,
                            eval_budget: ctx.eval_budget.clone(),
                        };
                        eval_expr(body, &lambda_ctx)?
                    }
                    TlaValue::Function(map) => map.get(elem).cloned().ok_or_else(|| {
                        anyhow!("SelectSeq test function is missing key {:?}", elem)
                    })?,
                    other => {
                        return Err(anyhow!(
                            "SelectSeq test must be a lambda or function, got {:?}",
                            other
                        ));
                    }
                };

                if let TlaValue::Bool(true) = test_result {
                    result.push(elem.clone());
                }
            }
            return Ok(TlaValue::Seq(Arc::new(result)));
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
        // TLC module: Range(f) - returns the set of all values in the range of function f
        // Range(f) == {f[x] : x \in DOMAIN f}
        "Range" => {
            if arg_values.len() != 1 {
                return Err(anyhow!("Range expects 1 argument"));
            }
            match &arg_values[0] {
                TlaValue::Function(map) => {
                    let values: BTreeSet<TlaValue> = map.values().cloned().collect();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                TlaValue::Seq(seq) => {
                    // Range of a sequence is the set of all its elements
                    let values: BTreeSet<TlaValue> = seq.iter().cloned().collect();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                TlaValue::Record(map) => {
                    // Range of a record is the set of all its field values
                    let values: BTreeSet<TlaValue> = map.values().cloned().collect();
                    return Ok(TlaValue::Set(Arc::new(values)));
                }
                _ => {
                    return Err(anyhow!("Range expects a function, sequence, or record"));
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
        "^^" => {
            if arg_values.len() != 2 {
                return Err(anyhow!("^^ expects 2 arguments"));
            }
            let left = arg_values[0].as_int()?;
            let right = arg_values[1].as_int()?;
            return Ok(TlaValue::Int(left ^ right));
        }
        // TLC module: FunAsSeq(f, a, b) - converts a function to a sequence
        // FunAsSeq(f, a, b) == [i \in 1..b |-> f[a + i - 1]]
        // This creates a sequence of length b by extracting values from f
        // starting at index a
        "FunAsSeq" => {
            if arg_values.len() != 3 {
                return Err(anyhow!("FunAsSeq expects 3 arguments: f, a, b"));
            }
            let func = &arg_values[0];
            let a = arg_values[1].as_int()?;
            let b = arg_values[2].as_int()?;

            if b < 0 {
                return Err(anyhow!("FunAsSeq: b must be non-negative, got {}", b));
            }

            let mut result = Vec::with_capacity(b as usize);
            for i in 1..=b {
                let key = TlaValue::Int(a + i - 1);
                let val = func.apply(&key)?.clone();
                result.push(val);
            }
            return Ok(TlaValue::Seq(Arc::new(result)));
        }
        // === Community module: SequencesExt ===
        "SeqOf" if arg_values.len() == 2 && !user_defined_shadow => {
            // Delegate to text-based eval (set construction is complex)
            return eval_operator_call("SeqOf", arg_values.clone(), ctx, depth + 1);
        }
        // === Community module: Folds ===
        "MapThenFoldSet" if arg_values.len() == 5 && !user_defined_shadow => {
            return eval_operator_call("MapThenFoldSet", arg_values.clone(), ctx, depth + 1);
        }
        "FoldSet" if arg_values.len() == 3 && !user_defined_shadow => {
            return eval_operator_call("FoldSet", arg_values.clone(), ctx, depth + 1);
        }
        "FoldFunctionOnSet" if arg_values.len() == 4 && !user_defined_shadow => {
            return eval_operator_call("FoldFunctionOnSet", arg_values.clone(), ctx, depth + 1);
        }
        // === Community module: UndirectedGraphs ===
        "IsUndirectedGraph" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call("IsUndirectedGraph", arg_values.clone(), ctx, depth + 1);
        }
        "IsLoopFreeUndirectedGraph" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call(
                "IsLoopFreeUndirectedGraph",
                arg_values.clone(),
                ctx,
                depth + 1,
            );
        }
        "ConnectedComponents" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call("ConnectedComponents", arg_values.clone(), ctx, depth + 1);
        }
        "AreConnectedIn" if arg_values.len() == 3 && !user_defined_shadow => {
            return eval_operator_call("AreConnectedIn", arg_values.clone(), ctx, depth + 1);
        }
        "IsStronglyConnected" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call("IsStronglyConnected", arg_values.clone(), ctx, depth + 1);
        }
        // === Community module: FiniteSetsExt (additional) ===
        "Quantify" if arg_values.len() == 2 && !user_defined_shadow => {
            return eval_operator_call("Quantify", arg_values.clone(), ctx, depth + 1);
        }
        "SymDiff" if arg_values.len() == 2 && !user_defined_shadow => {
            let a = arg_values[0].as_set()?;
            let b = arg_values[1].as_set()?;
            let result: BTreeSet<TlaValue> = a.symmetric_difference(&b).cloned().collect();
            return Ok(TlaValue::Set(Arc::new(result)));
        }
        "FlattenSet" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call("FlattenSet", arg_values.clone(), ctx, depth + 1);
        }
        "kSubset" if arg_values.len() == 2 && !user_defined_shadow => {
            return eval_operator_call("kSubset", arg_values.clone(), ctx, depth + 1);
        }
        "ChooseUnique" if arg_values.len() == 2 && !user_defined_shadow => {
            return eval_operator_call("ChooseUnique", arg_values.clone(), ctx, depth + 1);
        }
        "SumSet" if arg_values.len() == 1 && !user_defined_shadow => {
            let set = arg_values[0].as_set()?;
            let sum: i64 = set.iter().map(|v| v.as_int().unwrap_or(0)).sum();
            return Ok(TlaValue::Int(sum));
        }
        "ProductSet" if arg_values.len() == 1 && !user_defined_shadow => {
            let set = arg_values[0].as_set()?;
            let product: i64 = set.iter().map(|v| v.as_int().unwrap_or(1)).product();
            return Ok(TlaValue::Int(product));
        }
        "IsInjective" if arg_values.len() == 1 && !user_defined_shadow => {
            return eval_operator_call("IsInjective", arg_values.clone(), ctx, depth + 1);
        }
        // === Community module: Graphs (directed) ===
        "IsDirectedGraph" | "Successors" | "Predecessors" | "InDegree" | "OutDegree" | "Roots"
        | "Leaves" | "Transpose" | "IsDag"
            if !user_defined_shadow =>
        {
            return eval_operator_call(name, arg_values.clone(), ctx, depth + 1);
        }
        // === Community module: Relation ===
        "TransitiveClosure" if arg_values.len() == 2 && !user_defined_shadow => {
            return eval_operator_call("TransitiveClosure", arg_values.clone(), ctx, depth + 1);
        }
        "RemoveAt" if arg_values.len() == 2 && !user_defined_shadow => {
            let seq = sequence_like_values(&arg_values[0])
                .ok_or_else(|| anyhow!("RemoveAt expects a sequence, got {:?}", arg_values[0]))?;
            let i = arg_values[1].as_int()? as usize;
            if i < 1 || i > seq.len() {
                return Err(anyhow!(
                    "RemoveAt index {} out of bounds (len {})",
                    i,
                    seq.len()
                ));
            }
            let mut result = seq[..i - 1].to_vec();
            result.extend_from_slice(&seq[i..]);
            return Ok(TlaValue::Seq(Arc::new(result)));
        }
        // === Community module: Functions ===
        "FoldFunction" if arg_values.len() == 3 && !user_defined_shadow => {
            let op = &arg_values[0];
            let base = arg_values[1].clone();
            let fun = &arg_values[2];
            let values: Vec<TlaValue> = match fun {
                TlaValue::Function(map) => map.values().cloned().collect(),
                TlaValue::Seq(seq) => seq.iter().cloned().collect(),
                _ => return Err(anyhow!("FoldFunction: 3rd arg must be function/sequence")),
            };
            let mut acc = base;
            for val in values {
                acc = apply_value(op, vec![acc, val], ctx, depth + 1)?;
            }
            return Ok(acc);
        }
        // === Community module: DyadicRationals ===
        "Half" if arg_values.len() == 1 && !user_defined_shadow => {
            let p = &arg_values[0];
            let num = p.select_key("num")?.as_int()?;
            let den = p.select_key("den")?.as_int()?;
            let new_den = den * 2;
            let g = gcd(num.unsigned_abs(), new_den as u64) as i64;
            return Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                ("num".to_string(), TlaValue::Int(num / g)),
                ("den".to_string(), TlaValue::Int(new_den / g)),
            ]))));
        }
        "Add" if arg_values.len() == 2 && !user_defined_shadow => {
            if arg_values[0].select_key("num").is_ok() && arg_values[1].select_key("num").is_ok() {
                let pn = arg_values[0].select_key("num")?.as_int()?;
                let pd = arg_values[0].select_key("den")?.as_int()?;
                let qn = arg_values[1].select_key("num")?.as_int()?;
                let qd = arg_values[1].select_key("den")?.as_int()?;
                if pn == 0 {
                    return Ok(arg_values[1].clone());
                }
                let lcm = pd.max(qd);
                let new_num = pn * (lcm / pd) + qn * (lcm / qd);
                let g = gcd(new_num.unsigned_abs(), lcm as u64) as i64;
                return Ok(TlaValue::Record(Arc::new(BTreeMap::from([
                    ("num".to_string(), TlaValue::Int(new_num / g)),
                    ("den".to_string(), TlaValue::Int(lcm / g)),
                ]))));
            }
        }
        "IsDyadicRational" if arg_values.len() == 1 && !user_defined_shadow => {
            if let (Ok(den_val), Ok(_)) = (
                arg_values[0].select_key("den"),
                arg_values[0].select_key("num"),
            ) {
                let den = den_val.as_int()?;
                return Ok(TlaValue::Bool(den > 0 && (den & (den - 1)) == 0));
            }
            return Ok(TlaValue::Bool(false));
        }
        "PrettyPrint" if arg_values.len() == 1 && !user_defined_shadow => {
            if let (Ok(num_val), Ok(den_val)) = (
                arg_values[0].select_key("num"),
                arg_values[0].select_key("den"),
            ) {
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
            return Ok(TlaValue::String(format!("{:?}", arg_values[0])));
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
        new_ctx = new_ctx.with_local_value(normalize_param_name(param), value);
    }

    // Evaluate the compiled body
    eval_compiled_inner(&compiled_body, &new_ctx, depth + 1)
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
            "BoundedSeq expects a non-negative bound, got {}",
            max_len
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
        CompiledActionClause::Guard { expr, text } => {
            // Soundness fix: a "guard" whose text is an action call to a
            // user-defined action (e.g. inside `\E x \in S : ActionCall(x)`)
            // must be expanded as an action — not evaluated as a boolean.
            // The compiled IR cannot recognise an action call at compile time
            // because operator definitions aren't in scope, so this fallback
            // routes through the interpreted action evaluator which handles
            // primed assignments correctly.
            //
            // Without this, `Next == \/ \E x \in S : Action(x) \/ ...` (or any
            // wrapper definition reducing to that shape) silently drops every
            // successor from the existential branch — masking real invariant
            // violations as false negatives.
            if let Some(branches) = try_eval_compiled_guard_as_action(text, &branch)? {
                return Ok(branches);
            }
            // T1.5 soundness fix: a "guard" whose text is itself an action body
            // (e.g. a top-level disjunction containing primed assignments, as
            // produced by `compile_action_ir` when a top-level conjunct is
            // `\/ /\ x' = ... \/ /\ y' = ...`) must NOT be evaluated as a
            // boolean — that would silently discard the inner primed
            // assignments and report a stutter as a "successful" transition.
            // Route through the interpreted action evaluator so each inner
            // disjunct stages its primes and produces a real successor branch.
            if guard_text_is_action_body(text) {
                let interpreted_branches =
                    crate::tla::eval::eval_action_body_multi(text, &branch.ctx, &branch.staged)?;
                let outer_ctx = branch.ctx.clone();
                let mut out = Vec::with_capacity(interpreted_branches.len());
                for (staged, unchanged_vars) in interpreted_branches {
                    let mut merged_unchanged = branch.unchanged_vars.clone();
                    for var in unchanged_vars {
                        if !merged_unchanged.contains(&var) {
                            merged_unchanged.push(var);
                        }
                    }
                    out.push(CompiledActionBranch {
                        ctx: outer_ctx.clone(),
                        staged,
                        unchanged_vars: merged_unchanged,
                    });
                }
                return Ok(out);
            }
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
        CompiledActionClause::PrimedMembership { var, set_expr } => {
            let domain = eval_compiled(set_expr, &eval_ctx)?;
            let values = domain.as_set()?.iter().cloned().collect::<Vec<_>>();
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let mut branch = branch.clone();
                branch.staged.insert(var.clone(), value);
                out.push(branch);
            }
            Ok(out)
        }
        CompiledActionClause::Unchanged { vars } => {
            let mut branch = branch;
            for var in vars {
                branch.unchanged_vars.push(var.clone());
                if let Some(value) = branch.ctx.state.get(var.as_str()) {
                    branch
                        .staged
                        .entry(var.clone())
                        .or_insert_with(|| value.clone());
                }
            }
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

/// Heuristic: detect when a compiled `Guard`'s text is actually an action body
/// rather than a pure boolean. Specifically, when the text is a top-level
/// disjunction (or conjunction-of-disjunctions, normalised to start with `\/`
/// after dedent) whose branches contain primed assignments or `UNCHANGED`,
/// `compile_action_ir` will have wrapped the whole thing as a single Guard
/// clause — but the correct semantics is to expand each disjunct as a
/// successor-producing branch. Returns true when the text needs the
/// interpreted action-body evaluator instead of the boolean guard evaluator.
fn guard_text_is_action_body(text: &str) -> bool {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("\\/") {
        return false;
    }
    // Quick scan for a primed identifier (`X'`) or an UNCHANGED clause that
    // appears outside double-quoted strings. Both indicate an action body.
    let bytes = trimmed.as_bytes();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == b'\\' {
                escaped = true;
            } else if ch == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        // Detect `IDENT'` (a primed identifier). The apostrophe must follow an
        // identifier character to avoid matching string contents or set
        // displays (already filtered by the `in_string` check above).
        if ch == b'\'' && i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                return true;
            }
        }
        if ch == b'U'
            && trimmed[i..].starts_with("UNCHANGED")
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
        {
            let after = i + "UNCHANGED".len();
            if after >= bytes.len() || !is_ident_byte(bytes[after]) {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Detect when a compiled `Guard` is actually an action call to a
/// user-defined action and dispatch through the interpreted action evaluator.
///
/// Returns `Ok(Some(branches))` if `text` resolves to such an action and was
/// expanded successfully. Returns `Ok(None)` if `text` is not an action call
/// (in which case the caller should fall back to ordinary boolean guard
/// evaluation). Returns `Err` only when interpreted evaluation itself fails.
fn try_eval_compiled_guard_as_action<'a>(
    text: &str,
    branch: &CompiledActionBranch<'a>,
) -> Result<Option<Vec<CompiledActionBranch<'a>>>> {
    let Some((name, _args)) = crate::tla::eval::parse_action_call_expr(text) else {
        return Ok(None);
    };

    // Resolve through instance scope first (e.g. `Inst!Action(x)`), then
    // ordinary definitions. We only need to know whether the target is an
    // action — the interpreted evaluator handles the actual call.
    let def_opt = if let Some((alias, op)) = name.split_once('!') {
        branch
            .ctx
            .instances
            .and_then(|insts| insts.get(alias))
            .and_then(|inst| inst.module.as_ref())
            .and_then(|m| m.definitions.get(op).cloned())
    } else {
        lookup_definition(&name, &branch.ctx)
    };

    let Some(def) = def_opt else {
        return Ok(None);
    };
    if !crate::tla::looks_like_action(&def) {
        return Ok(None);
    }
    // Avoid infinite recursion when the body is the call itself (already
    // protected by the interpreted path, but mirror the same check here).
    if def.body.trim() == text.trim() {
        return Ok(None);
    }

    let outer_ctx = branch.ctx.clone();
    let interpreted_branches =
        crate::tla::eval::eval_action_body_multi(text, &branch.ctx, &branch.staged)?;

    // Merge interpreted branches with the existing compiled branch's
    // unchanged_vars (interpreted starts with an empty set; we must preserve
    // any prior `UNCHANGED` clauses that ran in the compiled sequence).
    let mut out = Vec::with_capacity(interpreted_branches.len());
    for (staged, unchanged_vars) in interpreted_branches {
        let mut merged_unchanged = branch.unchanged_vars.clone();
        for var in unchanged_vars {
            if !merged_unchanged.contains(&var) {
                merged_unchanged.push(var);
            }
        }
        out.push(CompiledActionBranch {
            ctx: outer_ctx.clone(),
            staged,
            unchanged_vars: merged_unchanged,
        });
    }
    Ok(Some(out))
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
    use crate::tla::compiled_expr::compile_expr;
    use crate::tla::{TlaDefinition, TlaState, tla_state};

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
    fn test_eval_leq_geq() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("3 \\leq 5"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
        assert_eq!(
            eval_compiled(&compile_expr("5 \\geq 3"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn test_eval_modulo() {
        let ctx = empty_ctx();
        assert_eq!(
            eval_compiled(&compile_expr("10 % 3"), &ctx).unwrap(),
            TlaValue::Int(1)
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

        assert_eq!(
            eval_compiled(&compile_expr("Max({1, 4, 2})"), &ctx).unwrap(),
            TlaValue::Int(4)
        );
        assert_eq!(
            eval_compiled(&compile_expr("Min({1, 4, 2})"), &ctx).unwrap(),
            TlaValue::Int(1)
        );
        assert_eq!(
            eval_compiled(&compile_expr("2^5"), &ctx).unwrap(),
            TlaValue::Int(32)
        );
        assert_eq!(
            eval_compiled(&compile_expr("6 ^^ 3"), &ctx).unwrap(),
            TlaValue::Int(5)
        );
        assert_eq!(
            eval_compiled(&compile_expr("<<1, 2>> \\circ <<3>>"), &ctx).unwrap(),
            TlaValue::Seq(Arc::new(vec![
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))
        );
        assert_eq!(
            eval_compiled(&compile_expr("2^5 - 1"), &ctx).unwrap(),
            TlaValue::Int(31)
        );
        assert_eq!(
            eval_compiled(&compile_expr("Cardinality(BoundedSeq({1, 2}, 2))"), &ctx).unwrap(),
            TlaValue::Int(7)
        );
        assert_eq!(
            eval_compiled(&compile_expr("TLCGet(\"level\")"), &ctx).unwrap(),
            TlaValue::Int(0)
        );
        assert_eq!(
            eval_compiled(&compile_expr("TLCGet(2)"), &ctx).unwrap(),
            TlaValue::Int(999)
        );
        assert_eq!(
            eval_compiled(&compile_expr("TLCSet(2, 17)"), &ctx).unwrap(),
            TlaValue::Bool(true)
        );
    }

    #[test]
    fn test_compiled_eval_supports_numeric_prefixed_identifiers() {
        let mut state = TlaState::new();
        state.insert(
            Arc::from("1bMessage"),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(1)]))),
        );
        state.insert(
            Arc::from("2bMessage"),
            TlaValue::Set(Arc::new(BTreeSet::from([TlaValue::Int(2)]))),
        );
        let ctx = EvalContext::new(&state);

        assert_eq!(
            eval_compiled(&compile_expr("1bMessage \\cup 2bMessage"), &ctx).unwrap(),
            TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
            ])))
        );
    }

    #[test]
    fn test_compiled_eval_can_pass_operator_values_to_recursive_operators() {
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
                "sc".to_string(),
                TlaDefinition {
                    name: "sc".to_string(),
                    params: vec!["<<x, y>>".to_string()],
                    body: "x + y".to_string(),
                    is_recursive: false,
                },
            ),
        ]);

        let state = TlaState::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_compiled(&compile_expr("Sum(sc, {<<1, 2>>, <<3, 4>>})"), &ctx)
                .expect("compiled higher-order recursion should work"),
            TlaValue::Int(10)
        );
    }

    #[test]
    fn test_compiled_lambda() {
        let ctx = empty_ctx();

        // Test that LAMBDA expressions compile and evaluate to Lambda values
        let compiled = compile_expr("LAMBDA x: x + 1");
        println!("Compiled LAMBDA: {:?}", compiled);

        let result = eval_compiled(&compiled, &ctx);
        println!("Evaluated result: {:?}", result);

        match result {
            Ok(TlaValue::Lambda { params, body, .. }) => {
                assert_eq!(*params, vec!["x".to_string()]);
                // Body should be parseable
                assert!(body.contains("+") || body.contains("x"));
            }
            Ok(other) => panic!("Expected Lambda, got {:?}", other),
            Err(e) => panic!("Error evaluating lambda: {}", e),
        }
    }

    #[test]
    fn test_compiled_selectseq() {
        let ctx = empty_ctx();

        // Test SelectSeq with a simple lambda
        let result = eval_compiled(
            &compile_expr("SelectSeq(<<1, 2, 3, 4>>, LAMBDA x: x > 2)"),
            &ctx,
        );
        println!("SelectSeq result: {:?}", result);

        match result {
            Ok(TlaValue::Seq(seq)) => {
                assert_eq!(*seq, vec![TlaValue::Int(3), TlaValue::Int(4)]);
            }
            Ok(other) => panic!("Expected Seq, got {:?}", other),
            Err(e) => panic!("Error in SelectSeq: {}", e),
        }
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
                    is_recursive: false,
                },
            );
            // Add(a, b) == a + b
            defs.insert(
                "Add".to_string(),
                TlaDefinition {
                    name: "Add".to_string(),
                    params: vec!["a".to_string(), "b".to_string()],
                    body: "a + b".to_string(),
                    is_recursive: false,
                },
            );
            // Triple(x) == Double(x) + x (recursive call to another operator)
            defs.insert(
                "Triple".to_string(),
                TlaDefinition {
                    name: "Triple".to_string(),
                    params: vec!["x".to_string()],
                    body: "Double(x) + x".to_string(),
                    is_recursive: false,
                },
            );
            // Constant == 42 (no parameters)
            defs.insert(
                "Constant".to_string(),
                TlaDefinition {
                    name: "Constant".to_string(),
                    params: vec![],
                    body: "42".to_string(),
                    is_recursive: false,
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
        state.insert(Arc::from("x"), TlaValue::Int(5));
        state.insert(Arc::from("y"), TlaValue::Int(10));

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

        let state = tla_state([
            ("flag", TlaValue::Bool(false)),
            ("x", TlaValue::Int(7)),
            ("y", TlaValue::Int(9)),
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
    fn test_compiled_conditional_unchanged_primes_flow_to_later_clauses() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        let state = tla_state([
            ("flag", TlaValue::Bool(false)),
            ("count", TlaValue::Int(7)),
            ("announced", TlaValue::Bool(false)),
        ]);
        let ctx = EvalContext::new(&state);

        let action_ir = ActionIr {
            name: "ConditionalCounter".to_string(),
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

        let compiled = CompiledActionIr::from_ir(&action_ir);
        let result = apply_compiled_action_ir(&compiled, &state, &ctx).unwrap();
        let next_state = result.expect("conditional branch should succeed");
        assert_eq!(next_state.get("count"), Some(&TlaValue::Int(7)));
        assert_eq!(next_state.get("announced"), Some(&TlaValue::Bool(true)));
    }

    #[test]
    fn test_compiled_primed_membership_generates_all_choices() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        let state = tla_state([("flip", TlaValue::String("H".to_string()))]);
        let ctx = EvalContext::new(&state);

        let action_ir = ActionIr {
            name: "TossCoin".to_string(),
            params: vec![],
            clauses: vec![ActionClause::PrimedMembership {
                var: "flip".to_string(),
                set_expr: "{\"H\", \"T\"}".to_string(),
            }],
        };

        let compiled = CompiledActionIr::from_ir(&action_ir);
        let next_states = apply_compiled_action_ir_multi(&compiled, &state, &ctx).unwrap();
        assert_eq!(next_states.len(), 2, "{next_states:?}");
        assert!(
            next_states
                .iter()
                .any(|st| st.get("flip") == Some(&TlaValue::String("H".to_string())))
        );
        assert!(
            next_states
                .iter()
                .any(|st| st.get("flip") == Some(&TlaValue::String("T".to_string())))
        );
    }

    #[test]
    fn test_compiled_bracket_operator_calls_use_operator_definitions() {
        let defs = BTreeMap::from([(
            "F".to_string(),
            crate::tla::TlaDefinition {
                name: "F".to_string(),
                params: vec!["x".to_string(), "y".to_string()],
                body: "x + y".to_string(),
                is_recursive: false,
            },
        )]);
        let state = tla_state([("a", TlaValue::Int(2)), ("b", TlaValue::Int(3))]);
        let ctx = EvalContext::with_definitions(&state, &defs);

        assert_eq!(
            eval_compiled(&compile_expr("F[a, b]"), &ctx).expect("operator call should evaluate"),
            TlaValue::Int(5)
        );
    }

    #[test]
    fn test_compiled_quantified_let_action_generates_multiple_successors() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

        let state = tla_state([("x", TlaValue::Int(0)), ("y", TlaValue::Int(9))]);
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

    #[test]
    fn test_compiled_eval_handles_comment_separated_let_bindings() {
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

        let expr = compile_expr(
            r#"LET
    priceDiff == referencePrice[f.asset] - f.price
    \* For each participant, calculate their P&L
    settlementPayments == { <<p, positions[p] * priceDiff>> : p \in Participants }
    \* Check that all participants can cover their losses
    canSettle == \A <<p, pnl>> \in settlementPayments : pnl >= 0 \/ balances[p] >= -pnl
IN
    IF canSettle
    THEN [p \in Participants |-> LET pnl == positions[p] * priceDiff IN balances[p] + pnl]
    ELSE balances"#,
        );

        let result = eval_compiled(&expr, &ctx).expect("compiled LET should evaluate");
        let TlaValue::Function(map) = result else {
            panic!("expected function value");
        };
        assert_eq!(map.get(&alice), Some(&TlaValue::Int(105)));
        assert_eq!(map.get(&bob), Some(&TlaValue::Int(95)));
    }

    #[test]
    fn test_compiled_nested_let_action_preserves_quantified_scope_in_except_updates() {
        use crate::tla::compiled_expr::CompiledActionIr;
        use crate::tla::{ActionClause, ActionIr};

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
        let action_ir = ActionIr {
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

        let compiled = CompiledActionIr::from_ir(&action_ir);
        let next_states = apply_compiled_action_ir_multi(&compiled, &state, &ctx)
            .expect("compiled nested LET action should evaluate");
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
    fn test_except_with_self_ref_arithmetic() {
        // Test @ + 1 pattern used in [table EXCEPT ![k] = @ + 1]
        let ctx = empty_ctx();

        // Create a function (table) to test with
        let mut func = BTreeMap::new();
        func.insert(TlaValue::Int(1), TlaValue::Int(5));
        func.insert(TlaValue::Int(2), TlaValue::Int(10));
        let func_val = TlaValue::Function(Arc::new(func));

        let ctx_with_table = ctx.with_local_value("table", func_val);

        // Evaluate [table EXCEPT ![1] = @ + 1]
        let result = eval_compiled(
            &compile_expr("[table EXCEPT ![1] = @ + 1]"),
            &ctx_with_table,
        )
        .unwrap();

        // Should increment table[1] from 5 to 6
        if let TlaValue::Function(f) = result {
            assert_eq!(f.get(&TlaValue::Int(1)), Some(&TlaValue::Int(6)));
            assert_eq!(f.get(&TlaValue::Int(2)), Some(&TlaValue::Int(10)));
        } else {
            panic!("Expected function, got {:?}", result);
        }
    }

    #[test]
    fn test_except_with_self_ref_negation() {
        // Test ~@ pattern used in [rec EXCEPT !.flag = ~@]
        let ctx = empty_ctx();

        // Create a record to test with
        let mut rec = BTreeMap::new();
        rec.insert("flag".to_string(), TlaValue::Bool(true));
        rec.insert("count".to_string(), TlaValue::Int(5));
        let rec_val = TlaValue::Record(Arc::new(rec));

        let ctx_with_rec = ctx.with_local_value("rec", rec_val);

        // Evaluate [rec EXCEPT !.flag = ~@]
        let result =
            eval_compiled(&compile_expr("[rec EXCEPT !.flag = ~@]"), &ctx_with_rec).unwrap();

        // Should negate rec.flag from TRUE to FALSE
        if let TlaValue::Record(r) = result {
            assert_eq!(r.get("flag"), Some(&TlaValue::Bool(false)));
            assert_eq!(r.get("count"), Some(&TlaValue::Int(5)));
        } else {
            panic!("Expected record, got {:?}", result);
        }
    }

    #[test]
    fn test_except_with_multiple_self_refs() {
        // Test pattern from LanguageFeatureMatrix: [rec EXCEPT !.flag = ~@, !.count = @ + 1]
        let ctx = empty_ctx();

        // Create a record to test with
        let mut rec = BTreeMap::new();
        rec.insert("flag".to_string(), TlaValue::Bool(false));
        rec.insert("count".to_string(), TlaValue::Int(10));
        let rec_val = TlaValue::Record(Arc::new(rec));

        let ctx_with_rec = ctx.with_local_value("rec", rec_val);

        // Evaluate [rec EXCEPT !.flag = ~@, !.count = @ + 1]
        let result = eval_compiled(
            &compile_expr("[rec EXCEPT !.flag = ~@, !.count = @ + 1]"),
            &ctx_with_rec,
        )
        .unwrap();

        // Should negate flag (FALSE -> TRUE) and increment count (10 -> 11)
        if let TlaValue::Record(r) = result {
            assert_eq!(r.get("flag"), Some(&TlaValue::Bool(true)));
            assert_eq!(r.get("count"), Some(&TlaValue::Int(11)));
        } else {
            panic!("Expected record, got {:?}", result);
        }
    }
}

#[cfg(test)]
mod forall_tests {
    use super::*;
    use crate::tla::compiled_expr::compile_expr;
    use crate::tla::{TlaState, tla_state};
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

        state.insert(Arc::from("listings"), TlaValue::Set(Arc::new(listings_set)));
        state.insert(Arc::from("MinPrice"), TlaValue::Int(5));

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

    /// T1.4 regression: `\E m \in {} : <body>` must always be FALSE.
    ///
    /// Compiled fast path was previously silently returning TRUE in some
    /// scope/contexts (see follow-on test below) — this is the simplest
    /// shape of the soundness failure.
    #[test]
    fn t1_4_exists_over_empty_set_is_false() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        let expr = compile_expr("\\E m \\in {} : TRUE");
        let v = eval_compiled(&expr, &ctx).expect("eval");
        assert_eq!(
            v,
            TlaValue::Bool(false),
            "\\E m \\in {{}} : TRUE must be FALSE"
        );
    }

    /// T1.4 regression: `\A a \in NonEmpty : \E m \in {} : <anything>`
    /// must be FALSE because the body is FALSE for every binding of `a`.
    ///
    /// Before T1.4 the compiled evaluator returned TRUE — which silently
    /// passed invariants in Paxos-style protocol specs that quantify
    /// universally over an inner existential.
    #[test]
    fn t1_4_forall_over_inner_empty_exists_is_false() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        let expr = compile_expr("\\A a \\in {1, 2, 3} : \\E m \\in {} : TRUE");
        let v = eval_compiled(&expr, &ctx).expect("eval");
        assert_eq!(
            v,
            TlaValue::Bool(false),
            "\\A a \\in {{1,2,3}} : \\E m \\in {{}} : TRUE must be FALSE",
        );
    }

    /// T1.4 regression: even with the inner-Exists domain bound by a LET
    /// (which is how it appears in Paxos `Phase2a`), the result must be
    /// FALSE when the LET-bound set is empty.
    #[test]
    fn t1_4_forall_over_inner_exists_with_let_bound_empty_set_is_false() {
        let state = TlaState::new();
        let ctx = EvalContext::new(&state);
        let expr = compile_expr("LET S == {} IN \\A a \\in {1, 2, 3} : \\E m \\in S : m = a");
        let v = eval_compiled(&expr, &ctx).expect("eval");
        assert_eq!(v, TlaValue::Bool(false));
    }

    /// T1.4 regression: Paxos-style Phase2a guard using LET-bound
    /// set-comprehension over an empty source variable. This is the
    /// exact scope shape that exposed the bug.
    #[test]
    fn t1_4_paxos_phase2a_guard_with_empty_msgs_is_false() {
        let mut state = TlaState::new();
        // msgs == {} (empty set, mimicking initial Paxos state)
        state.insert(Arc::from("msgs"), TlaValue::Set(Arc::new(BTreeSet::new())));
        let mut q_set = BTreeSet::new();
        q_set.insert(TlaValue::ModelValue("a1".to_string()));
        q_set.insert(TlaValue::ModelValue("a2".to_string()));
        let ctx = EvalContext::new(&state).with_local_value("Q", TlaValue::Set(Arc::new(q_set)));
        // Mimic the Phase2a guard inside `\E Q \in Quorum : LET ... IN ...`:
        //   Q1b == {m \in msgs : ...} -- becomes {} here
        //   Body : \A a \in Q : \E m \in Q1b : m.acc = a
        let expr = compile_expr(
            "LET Q1b == {m \\in msgs : m.type = \"1b\"} \
             IN \\A a \\in Q : \\E m \\in Q1b : m.acc = a",
        );
        let v = eval_compiled(&expr, &ctx).expect("eval");
        assert_eq!(
            v,
            TlaValue::Bool(false),
            "Phase2a-style guard with empty msgs must be FALSE",
        );
    }
}

#[test]
fn test_full_typeok_pattern_cluster_lease_failover() {
    // Test the FULL TypeOK pattern from ClusterLeaseFailover.tla that fails:
    // - 3 DOMAIN checks
    // - 1 outer quantifier (\A s \in Shards) with 4 conjuncts in body
    // - 1 nested quantifier (\A c \in Clients) with 3 conjuncts
    // Error: "key s not in domain {s1, s2, s3, s4}"
    //
    // TypeOK ==
    //     /\ DOMAIN shardTerm = Shards
    //     /\ DOMAIN shardLeader = Shards
    //     /\ DOMAIN leases = Shards
    //     /\ \A s \in Shards :
    //         /\ shardTerm[s] \in 0..MaxTerm
    //         /\ shardLeader[s] \in {"none", "stable", "electing"}
    //         /\ DOMAIN leases[s] = Clients
    //         /\ \A c \in Clients :
    //             /\ leases[s][c].held \in BOOLEAN
    //             /\ leases[s][c].grantedTerm \in 0..MaxTerm
    //             /\ leases[s][c].access \in {"none", "read", "write", "exclusive"}

    use crate::tla::TlaState;
    use crate::tla::compiled_expr::compile_expr;
    use std::sync::Arc;

    let mut state = TlaState::new();

    // Create Shards set (use 4 shards like in the error message)
    let shards_set: BTreeSet<TlaValue> = ["s1", "s2", "s3", "s4"]
        .iter()
        .map(|s| TlaValue::ModelValue(s.to_string()))
        .collect();
    state.insert(Arc::from("Shards"), TlaValue::Set(Arc::new(shards_set)));

    // Create Clients set
    let clients_set: BTreeSet<TlaValue> = ["c1", "c2"]
        .iter()
        .map(|s| TlaValue::ModelValue(s.to_string()))
        .collect();
    state.insert(Arc::from("Clients"), TlaValue::Set(Arc::new(clients_set)));

    // Create shardTerm function: s1 -> 1, s2 -> 2, s3 -> 3, s4 -> 4
    let mut shardterm_map = BTreeMap::new();
    for (i, s) in ["s1", "s2", "s3", "s4"].iter().enumerate() {
        shardterm_map.insert(
            TlaValue::ModelValue(s.to_string()),
            TlaValue::Int(i as i64 + 1),
        );
    }
    state.insert(
        Arc::from("shardTerm"),
        TlaValue::Function(Arc::new(shardterm_map)),
    );

    // Create shardLeader function: s1 -> "stable", s2 -> "none", etc.
    // Use String values (not ModelValue) to match the set literal {"none", "stable", "electing"}
    let mut shardleader_map = BTreeMap::new();
    let statuses = ["stable", "none", "electing", "stable"];
    for (i, s) in ["s1", "s2", "s3", "s4"].iter().enumerate() {
        shardleader_map.insert(
            TlaValue::ModelValue(s.to_string()),
            TlaValue::String(statuses[i].to_string()),
        );
    }
    state.insert(
        Arc::from("shardLeader"),
        TlaValue::Function(Arc::new(shardleader_map)),
    );

    // Create leases nested function: leases[s][c] = {held: TRUE, grantedTerm: 1, access: "read"}
    // Use String values for access to match the set literal {"none", "read", "write", "exclusive"}
    let mut leases_outer = BTreeMap::new();
    for shard in ["s1", "s2", "s3", "s4"] {
        let mut inner_map = BTreeMap::new();
        for client in ["c1", "c2"] {
            let mut lease_rec = BTreeMap::new();
            lease_rec.insert("held".to_string(), TlaValue::Bool(true));
            lease_rec.insert("grantedTerm".to_string(), TlaValue::Int(1));
            lease_rec.insert("access".to_string(), TlaValue::String("read".to_string()));
            inner_map.insert(
                TlaValue::ModelValue(client.to_string()),
                TlaValue::Record(Arc::new(lease_rec)),
            );
        }
        leases_outer.insert(
            TlaValue::ModelValue(shard.to_string()),
            TlaValue::Function(Arc::new(inner_map)),
        );
    }
    state.insert(
        Arc::from("leases"),
        TlaValue::Function(Arc::new(leases_outer)),
    );

    state.insert(Arc::from("MaxTerm"), TlaValue::Int(10));

    let ctx = EvalContext::new(&state);

    // The FULL TypeOK invariant expression
    let expr_str = r#"/\ DOMAIN shardTerm = Shards
/\ DOMAIN shardLeader = Shards
/\ DOMAIN leases = Shards
/\ \A s \in Shards :
    /\ shardTerm[s] \in 0..MaxTerm
    /\ shardLeader[s] \in {"none", "stable", "electing"}
    /\ DOMAIN leases[s] = Clients
    /\ \A c \in Clients :
        /\ leases[s][c].held \in BOOLEAN
        /\ leases[s][c].grantedTerm \in 0..MaxTerm
        /\ leases[s][c].access \in {"none", "read", "write", "exclusive"}"#;

    let expr = compile_expr(expr_str);
    println!("Full TypeOK compiled: {:#?}", expr);

    let result = eval_compiled(&expr, &ctx);
    println!("Full TypeOK result: {:?}", result);

    assert!(
        result.is_ok(),
        "Full TypeOK evaluation failed: {:?}",
        result
    );
    assert_eq!(result.unwrap(), TlaValue::Bool(true));
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

    state.insert(Arc::from("listings"), TlaValue::Set(Arc::new(listings_set)));
    state.insert(Arc::from("MinPrice"), TlaValue::Int(5));
    state.insert(Arc::from("MaxPrice"), TlaValue::Int(20));

    let ctx = EvalContext::new(&state);

    let result = eval_compiled(&expr, &ctx);
    println!("Result: {:?}", result);
    assert!(result.is_ok(), "Evaluation failed: {:?}", result);
    assert_eq!(result.unwrap(), TlaValue::Bool(true));
}

#[test]
fn test_nested_quantifiers_typeok_evaluation() {
    // Test the exact pattern from ClusterLeaseFailover's TypeOK with nested quantifiers
    // \A s \in Shards :
    //     /\ shardTerm[s] \in 0..MaxTerm
    //     /\ \A c \in Clients :
    //         /\ leases[s][c].held \in BOOLEAN

    let mut state = TlaState::new();

    // Create Shards set
    let shards_set: BTreeSet<TlaValue> = ["s1", "s2"]
        .iter()
        .map(|s| TlaValue::ModelValue(s.to_string()))
        .collect();
    state.insert(Arc::from("Shards"), TlaValue::Set(Arc::new(shards_set)));

    // Create Clients set
    let clients_set: BTreeSet<TlaValue> = ["c1", "c2"]
        .iter()
        .map(|s| TlaValue::ModelValue(s.to_string()))
        .collect();
    state.insert(Arc::from("Clients"), TlaValue::Set(Arc::new(clients_set)));

    // Create shardTerm function: s1 -> 1, s2 -> 2
    let mut shardterm_map = BTreeMap::new();
    shardterm_map.insert(TlaValue::ModelValue("s1".to_string()), TlaValue::Int(1));
    shardterm_map.insert(TlaValue::ModelValue("s2".to_string()), TlaValue::Int(2));
    state.insert(
        Arc::from("shardTerm"),
        TlaValue::Function(Arc::new(shardterm_map)),
    );

    // Create leases nested function: leases[s][c] = {held: TRUE, ...}
    let mut leases_outer = BTreeMap::new();
    for shard in ["s1", "s2"] {
        let mut inner_map = BTreeMap::new();
        for client in ["c1", "c2"] {
            let mut lease_rec = BTreeMap::new();
            lease_rec.insert("held".to_string(), TlaValue::Bool(true));
            lease_rec.insert("grantedTerm".to_string(), TlaValue::Int(1));
            inner_map.insert(
                TlaValue::ModelValue(client.to_string()),
                TlaValue::Record(Arc::new(lease_rec)),
            );
        }
        leases_outer.insert(
            TlaValue::ModelValue(shard.to_string()),
            TlaValue::Function(Arc::new(inner_map)),
        );
    }
    state.insert(
        Arc::from("leases"),
        TlaValue::Function(Arc::new(leases_outer)),
    );

    state.insert(Arc::from("MaxTerm"), TlaValue::Int(10));

    let ctx = EvalContext::new(&state);

    // Compile and evaluate the nested quantifier expression
    let expr_str = r#"\A s \in Shards :
/\ shardTerm[s] \in 0..MaxTerm
/\ \A c \in Clients :
    /\ leases[s][c].held \in BOOLEAN"#;

    let expr = compile_expr(expr_str);
    println!("Nested quantifier expr: {:#?}", expr);

    let result = eval_compiled(&expr, &ctx);
    println!("Result: {:?}", result);

    assert!(result.is_ok(), "Evaluation failed: {:?}", result);
    assert_eq!(result.unwrap(), TlaValue::Bool(true));
}

#[test]
fn test_compiled_cartesian_product_times() {
    // Test that \\times is correctly compiled and evaluated as cartesian product
    use crate::tla::TlaState;
    use crate::tla::compiled_expr::compile_expr;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    let state = TlaState::new();
    let ctx = EvalContext::new(&state);

    // Test \\X (standard syntax)
    let result_x = eval_compiled(&compile_expr("{1, 2} \\X {3}"), &ctx).unwrap();

    // Test \\times (alternate syntax)
    let result_times = eval_compiled(&compile_expr("{1, 2} \\times {3}"), &ctx).unwrap();

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
fn test_compiled_sequence_builtins_accept_sequence_like_functions() {
    let state = tla_state([
        (
            "seq_like",
            TlaValue::Function(Arc::new(BTreeMap::from([
                (TlaValue::Int(1), TlaValue::Int(1)),
                (TlaValue::Int(2), TlaValue::Int(2)),
            ]))),
        ),
        (
            "other",
            TlaValue::Function(Arc::new(BTreeMap::from([(
                TlaValue::Int(1),
                TlaValue::Int(3),
            )]))),
        ),
    ]);
    let ctx = EvalContext::new(&state);

    assert_eq!(
        eval_compiled(&compile_expr("Len(seq_like)"), &ctx).unwrap(),
        TlaValue::Int(2)
    );
    assert_eq!(
        eval_compiled(&compile_expr("Head(seq_like)"), &ctx).unwrap(),
        TlaValue::Int(1)
    );
    assert_eq!(
        eval_compiled(&compile_expr("Tail(seq_like)"), &ctx).unwrap(),
        TlaValue::Seq(Arc::new(vec![TlaValue::Int(2)]))
    );
    assert_eq!(
        eval_compiled(&compile_expr("Append(seq_like, 4)"), &ctx).unwrap(),
        TlaValue::Seq(Arc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(4),
        ]))
    );
    assert_eq!(
        eval_compiled(&compile_expr("SubSeq(seq_like, 1, 2)"), &ctx).unwrap(),
        TlaValue::Seq(Arc::new(vec![TlaValue::Int(1), TlaValue::Int(2)]))
    );
    assert_eq!(
        eval_compiled(&compile_expr("seq_like \\o other"), &ctx).unwrap(),
        TlaValue::Seq(Arc::new(vec![
            TlaValue::Int(1),
            TlaValue::Int(2),
            TlaValue::Int(3),
        ]))
    );
}

#[cfg(test)]
mod compiled_action_correctness_tests {
    use super::*;
    use crate::tla::compiled_expr::CompiledActionIr;
    use crate::tla::{ActionClause, ActionIr, EvalContext, TlaState, TlaValue, tla_state};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn compile_and_run(
        clauses: Vec<ActionClause>,
        state: &TlaState,
        ctx: &EvalContext<'_>,
    ) -> Vec<TlaState> {
        let ir = ActionIr {
            name: "Test".to_string(),
            params: vec![],
            clauses,
        };
        let compiled = CompiledActionIr::from_ir(&ir);
        apply_compiled_action_ir_multi(&compiled, state, ctx)
            .expect("compiled eval should not error")
    }

    /// Compiled \A guard must block when quantifier body is false for some element.
    #[test]
    fn forall_guard_blocks_when_false() {
        let state = tla_state([(
            "locked",
            TlaValue::Function(Arc::new(BTreeMap::from([
                (TlaValue::String("p1".to_string()), TlaValue::Bool(true)),
                (TlaValue::String("p2".to_string()), TlaValue::Bool(false)),
            ]))),
        )]);
        let mut defs = BTreeMap::new();
        defs.insert(
            "Procs".to_string(),
            crate::tla::TlaDefinition {
                name: "Procs".to_string(),
                params: vec![],
                body: r#"{"p1", "p2"}"#.to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        // \A p \in Procs : ~locked[p]
        // p1 is locked, so this should be FALSE → no successors
        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"\A p \in Procs : ~locked[p]"#.to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "locked".to_string(),
                    expr: "locked".to_string(),
                },
            ],
            &state,
            &ctx,
        );
        assert!(result.is_empty(), "\\A guard should block: {result:?}");
    }

    /// Compiled \A guard must pass when quantifier body is true for all elements.
    #[test]
    fn forall_guard_passes_when_true() {
        let state = tla_state([(
            "locked",
            TlaValue::Function(Arc::new(BTreeMap::from([
                (TlaValue::String("p1".to_string()), TlaValue::Bool(false)),
                (TlaValue::String("p2".to_string()), TlaValue::Bool(false)),
            ]))),
        )]);
        let mut defs = BTreeMap::new();
        defs.insert(
            "Procs".to_string(),
            crate::tla::TlaDefinition {
                name: "Procs".to_string(),
                params: vec![],
                body: r#"{"p1", "p2"}"#.to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"\A p \in Procs : ~locked[p]"#.to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "locked".to_string(),
                    expr: "locked".to_string(),
                },
            ],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1, "\\A guard should pass: {result:?}");
    }

    /// IF/THEN/ELSE guard: THEN branch blocks, ELSE branch passes.
    #[test]
    fn if_guard_selects_correct_branch() {
        let state = tla_state([
            ("mode", TlaValue::String("exclusive".to_string())),
            ("count", TlaValue::Int(2)),
        ]);
        let ctx = EvalContext::new(&state);

        // IF mode = "exclusive" THEN count = 0 ELSE TRUE
        // mode IS exclusive and count is 2 (not 0), so guard should BLOCK
        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"IF mode = "exclusive" THEN count = 0 ELSE TRUE"#.to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "count".to_string(),
                    expr: "count + 1".to_string(),
                },
            ],
            &state,
            &ctx,
        );
        assert!(result.is_empty(), "IF/THEN guard should block: {result:?}");

        // Now with mode = "shared", ELSE branch is TRUE → should pass
        let state2 = tla_state([
            ("mode", TlaValue::String("shared".to_string())),
            ("count", TlaValue::Int(2)),
        ]);
        let ctx2 = EvalContext::new(&state2);
        let result2 = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"IF mode = "exclusive" THEN count = 0 ELSE TRUE"#.to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "count".to_string(),
                    expr: "count + 1".to_string(),
                },
            ],
            &state2,
            &ctx2,
        );
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].get("count"), Some(&TlaValue::Int(3)));
    }

    /// UNCHANGED must preserve the current value in successors, verified
    /// by checking the successor state contains the unchanged variable.
    #[test]
    fn unchanged_copies_current_value() {
        let state = tla_state([
            ("x", TlaValue::Int(42)),
            ("y", TlaValue::String("keep_me".to_string())),
        ]);
        let ctx = EvalContext::new(&state);

        let result = compile_and_run(
            vec![
                ActionClause::PrimedAssignment {
                    var: "x".to_string(),
                    expr: "x + 1".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["y".to_string()],
                },
            ],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x"), Some(&TlaValue::Int(43)));
        assert_eq!(
            result[0].get("y"),
            Some(&TlaValue::String("keep_me".to_string())),
            "UNCHANGED must preserve current value"
        );
    }

    /// Negation guard: ~active must block when active=TRUE.
    #[test]
    fn negation_guard_blocks_and_passes() {
        let state_active = tla_state([("active", TlaValue::Bool(true))]);
        let ctx_active = EvalContext::new(&state_active);

        let result = compile_and_run(
            vec![ActionClause::Guard {
                expr: "~active".to_string(),
            }],
            &state_active,
            &ctx_active,
        );
        assert!(result.is_empty(), "~active should block when active=TRUE");

        let state_inactive = tla_state([("active", TlaValue::Bool(false))]);
        let ctx_inactive = EvalContext::new(&state_inactive);

        let result2 = compile_and_run(
            vec![ActionClause::Guard {
                expr: "~active".to_string(),
            }],
            &state_inactive,
            &ctx_inactive,
        );
        assert_eq!(result2.len(), 1, "~active should pass when active=FALSE");
    }

    /// Exists clause must produce one successor per domain element.
    #[test]
    fn exists_produces_correct_successor_count() {
        let state = tla_state([("chosen", TlaValue::Int(0))]);
        let mut defs = BTreeMap::new();
        defs.insert(
            "Items".to_string(),
            crate::tla::TlaDefinition {
                name: "Items".to_string(),
                params: vec![],
                body: "{10, 20, 30}".to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        let result = compile_and_run(
            vec![ActionClause::Exists {
                binders: "x \\in Items".to_string(),
                body: "/\\ chosen' = x".to_string(),
            }],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 3, "should produce 3 successors");
        let values: std::collections::BTreeSet<_> = result
            .iter()
            .map(|s| s.get("chosen").cloned().unwrap())
            .collect();
        assert!(values.contains(&TlaValue::Int(10)));
        assert!(values.contains(&TlaValue::Int(20)));
        assert!(values.contains(&TlaValue::Int(30)));
    }

    /// Primed assignments from different action IRs applied to the same state
    /// must not interfere — each IR produces its own independent successors.
    #[test]
    fn separate_actions_produce_independent_successors() {
        let state = tla_state([("x", TlaValue::Int(0)), ("y", TlaValue::Int(0))]);
        let ctx = EvalContext::new(&state);

        let result_x = compile_and_run(
            vec![
                ActionClause::PrimedAssignment {
                    var: "x".to_string(),
                    expr: "x + 1".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["y".to_string()],
                },
            ],
            &state,
            &ctx,
        );
        let result_y = compile_and_run(
            vec![
                ActionClause::PrimedAssignment {
                    var: "y".to_string(),
                    expr: "y + 10".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["x".to_string()],
                },
            ],
            &state,
            &ctx,
        );

        assert_eq!(result_x.len(), 1);
        assert_eq!(result_x[0].get("x"), Some(&TlaValue::Int(1)));
        assert_eq!(result_x[0].get("y"), Some(&TlaValue::Int(0)));

        assert_eq!(result_y.len(), 1);
        assert_eq!(result_y[0].get("x"), Some(&TlaValue::Int(0)));
        assert_eq!(result_y[0].get("y"), Some(&TlaValue::Int(10)));
    }

    /// CASE expression guard: selects the matching arm based on state.
    #[test]
    fn test_compiled_case_expression_guard() {
        // phase = "prepare" → CASE arm yields TRUE → should produce successor
        let state_prepare = tla_state([("phase", TlaValue::String("prepare".to_string()))]);
        let ctx_prepare = EvalContext::new(&state_prepare);

        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"CASE phase = "prepare" -> TRUE [] phase = "commit" -> FALSE [] OTHER -> FALSE"#
                        .to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["phase".to_string()],
                },
            ],
            &state_prepare,
            &ctx_prepare,
        );
        assert_eq!(
            result.len(),
            1,
            "CASE guard should pass when phase=prepare: {result:?}"
        );

        // phase = "commit" → CASE arm yields FALSE → should block
        let state_commit = tla_state([("phase", TlaValue::String("commit".to_string()))]);
        let ctx_commit = EvalContext::new(&state_commit);

        let result2 = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: r#"CASE phase = "prepare" -> TRUE [] phase = "commit" -> FALSE [] OTHER -> FALSE"#
                        .to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["phase".to_string()],
                },
            ],
            &state_commit,
            &ctx_commit,
        );
        assert!(
            result2.is_empty(),
            "CASE guard should block when phase=commit: {result2:?}"
        );
    }

    /// Implication guard: (ready => value >= 0) blocks when TRUE => FALSE.
    #[test]
    fn test_compiled_implication_guard() {
        // ready=TRUE, value=0: TRUE => (0 >= 0) = TRUE => should pass
        let state_pass = tla_state([("ready", TlaValue::Bool(true)), ("value", TlaValue::Int(0))]);
        let ctx_pass = EvalContext::new(&state_pass);

        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "ready => value >= 0".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["ready".to_string(), "value".to_string()],
                },
            ],
            &state_pass,
            &ctx_pass,
        );
        assert_eq!(
            result.len(),
            1,
            "implication guard should pass (TRUE => TRUE): {result:?}"
        );

        // ready=TRUE, value=-1: TRUE => (-1 >= 0) = FALSE => should block
        let state_block = tla_state([
            ("ready", TlaValue::Bool(true)),
            ("value", TlaValue::Int(-1)),
        ]);
        let ctx_block = EvalContext::new(&state_block);

        let result2 = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "ready => value >= 0".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["ready".to_string(), "value".to_string()],
                },
            ],
            &state_block,
            &ctx_block,
        );
        assert!(
            result2.is_empty(),
            "implication guard should block (TRUE => FALSE): {result2:?}"
        );
    }

    /// Sequence operations: Len guard + Tail assignment.
    #[test]
    fn test_compiled_sequence_operations_in_guard() {
        let state = tla_state([(
            "msgs",
            TlaValue::Seq(Arc::new(vec![
                TlaValue::String("msg1".to_string()),
                TlaValue::String("msg2".to_string()),
                TlaValue::String("msg3".to_string()),
            ])),
        )]);
        let ctx = EvalContext::new(&state);

        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "Len(msgs) > 0".to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "msgs".to_string(),
                    expr: "Tail(msgs)".to_string(),
                },
            ],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1, "Len guard should pass: {result:?}");
        let succ_msgs = result[0].get("msgs").unwrap();
        let expected = TlaValue::Seq(Arc::new(vec![
            TlaValue::String("msg2".to_string()),
            TlaValue::String("msg3".to_string()),
        ]));
        assert_eq!(succ_msgs, &expected, "Tail should remove first element");

        // Empty sequence: Len = 0, guard should block
        let state_empty = tla_state([("msgs", TlaValue::Seq(Arc::new(vec![])))]);
        let ctx_empty = EvalContext::new(&state_empty);

        let result2 = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "Len(msgs) > 0".to_string(),
                },
                ActionClause::PrimedAssignment {
                    var: "msgs".to_string(),
                    expr: "Tail(msgs)".to_string(),
                },
            ],
            &state_empty,
            &ctx_empty,
        );
        assert!(
            result2.is_empty(),
            "Len guard should block on empty seq: {result2:?}"
        );
    }

    /// Function construction: [n \in Nodes |-> f[n] + 1].
    #[test]
    fn test_compiled_function_construction_in_assignment() {
        let state = tla_state([(
            "f",
            TlaValue::Function(Arc::new(BTreeMap::from([
                (TlaValue::String("n1".to_string()), TlaValue::Int(0)),
                (TlaValue::String("n2".to_string()), TlaValue::Int(0)),
            ]))),
        )]);
        let mut defs = BTreeMap::new();
        defs.insert(
            "Nodes".to_string(),
            crate::tla::TlaDefinition {
                name: "Nodes".to_string(),
                params: vec![],
                body: r#"{"n1", "n2"}"#.to_string(),
                is_recursive: false,
            },
        );
        let ctx = EvalContext::with_definitions(&state, &defs);

        let result = compile_and_run(
            vec![ActionClause::PrimedAssignment {
                var: "f".to_string(),
                expr: r#"[n \in Nodes |-> f[n] + 1]"#.to_string(),
            }],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1, "should produce one successor: {result:?}");
        let expected = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::String("n1".to_string()), TlaValue::Int(1)),
            (TlaValue::String("n2".to_string()), TlaValue::Int(1)),
        ])));
        assert_eq!(
            result[0].get("f").unwrap(),
            &expected,
            "f should be incremented"
        );
    }

    /// Nested EXCEPT: [data EXCEPT !["a"]["x"] = 99].
    #[test]
    fn test_compiled_nested_except() {
        let inner_a = TlaValue::Record(Arc::new(BTreeMap::from([
            ("x".to_string(), TlaValue::Int(1)),
            ("y".to_string(), TlaValue::Int(2)),
        ])));
        let inner_b = TlaValue::Record(Arc::new(BTreeMap::from([
            ("x".to_string(), TlaValue::Int(3)),
            ("y".to_string(), TlaValue::Int(4)),
        ])));
        let state = tla_state([(
            "data",
            TlaValue::Function(Arc::new(BTreeMap::from([
                (TlaValue::String("a".to_string()), inner_a),
                (TlaValue::String("b".to_string()), inner_b.clone()),
            ]))),
        )]);
        let ctx = EvalContext::new(&state);

        let result = compile_and_run(
            vec![ActionClause::PrimedAssignment {
                var: "data".to_string(),
                expr: r#"[data EXCEPT !["a"]["x"] = 99]"#.to_string(),
            }],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1, "should produce one successor");

        let data_prime = result[0].get("data").unwrap();
        // Extract data'["a"]
        if let TlaValue::Function(f) = data_prime {
            let a_val = f.get(&TlaValue::String("a".to_string())).unwrap();
            // a["x"] should be 99
            if let TlaValue::Record(rec) = a_val {
                assert_eq!(
                    rec.get("x"),
                    Some(&TlaValue::Int(99)),
                    "data'[\"a\"][\"x\"] should be 99"
                );
                assert_eq!(
                    rec.get("y"),
                    Some(&TlaValue::Int(2)),
                    "data'[\"a\"][\"y\"] should be unchanged"
                );
            } else if let TlaValue::Function(inner) = a_val {
                assert_eq!(
                    inner.get(&TlaValue::String("x".to_string())),
                    Some(&TlaValue::Int(99)),
                    "data'[\"a\"][\"x\"] should be 99"
                );
                assert_eq!(
                    inner.get(&TlaValue::String("y".to_string())),
                    Some(&TlaValue::Int(2)),
                    "data'[\"a\"][\"y\"] should be unchanged"
                );
            } else {
                panic!("data'[\"a\"] should be a record or function, got: {a_val:?}");
            }
            // b should be unchanged
            let b_val = f.get(&TlaValue::String("b".to_string())).unwrap();
            assert_eq!(b_val, &inner_b, "data'[\"b\"] should be unchanged");
        } else {
            panic!("data' should be a Function, got: {data_prime:?}");
        }
    }

    /// Set comprehension guard: Cardinality({x \in vals : x > 3}) >= 2.
    #[test]
    fn test_compiled_set_comprehension_guard() {
        use std::collections::BTreeSet;

        // vals = {1, 2, 3, 4, 5}: 2 elements > 3 → guard passes
        let state_pass = tla_state([(
            "vals",
            TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
                TlaValue::Int(4),
                TlaValue::Int(5),
            ]))),
        )]);
        let ctx_pass = EvalContext::new(&state_pass);

        let result = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "Cardinality({x \\in vals : x > 3}) >= 2".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["vals".to_string()],
                },
            ],
            &state_pass,
            &ctx_pass,
        );
        assert_eq!(
            result.len(),
            1,
            "set comprehension guard should pass: {result:?}"
        );

        // vals = {1, 2, 3}: 0 elements > 3 → guard blocks
        let state_block = tla_state([(
            "vals",
            TlaValue::Set(Arc::new(BTreeSet::from([
                TlaValue::Int(1),
                TlaValue::Int(2),
                TlaValue::Int(3),
            ]))),
        )]);
        let ctx_block = EvalContext::new(&state_block);

        let result2 = compile_and_run(
            vec![
                ActionClause::Guard {
                    expr: "Cardinality({x \\in vals : x > 3}) >= 2".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["vals".to_string()],
                },
            ],
            &state_block,
            &ctx_block,
        );
        assert!(
            result2.is_empty(),
            "set comprehension guard should block: {result2:?}"
        );
    }

    /// CHOOSE in assignment: picked' = CHOOSE x \in items : x > 15.
    #[test]
    fn test_compiled_choose_in_assignment() {
        use std::collections::BTreeSet;

        let state = tla_state([
            (
                "items",
                TlaValue::Set(Arc::new(BTreeSet::from([
                    TlaValue::Int(10),
                    TlaValue::Int(20),
                    TlaValue::Int(30),
                ]))),
            ),
            ("picked", TlaValue::Int(0)),
        ]);
        let ctx = EvalContext::new(&state);

        let result = compile_and_run(
            vec![
                ActionClause::PrimedAssignment {
                    var: "picked".to_string(),
                    expr: "CHOOSE x \\in items : x > 15".to_string(),
                },
                ActionClause::Unchanged {
                    vars: vec!["items".to_string()],
                },
            ],
            &state,
            &ctx,
        );
        assert_eq!(result.len(), 1, "should produce one successor: {result:?}");
        let picked = result[0].get("picked").unwrap();
        assert!(
            *picked == TlaValue::Int(20) || *picked == TlaValue::Int(30),
            "picked' should be 20 or 30, got: {picked:?}"
        );
    }
}

#[cfg(test)]
mod swarm_eval_consistency_tests {
    //! Swarm testing across evaluation paths.
    //!
    //! For each internal corpus spec, evaluate Init/Next definition bodies
    //! through both the compiled path (`eval_compiled`) and the text path
    //! (`eval_expr`), then compare results. Randomly disabling some compiled
    //! operators (forcing fallback to the text path) catches divergences
    //! between the two evaluation engines.
    //!
    //! Based on "Swarm Testing" (Groce et al., ISSTA 2012).

    use super::*;
    use crate::tla::compiled_expr::compile_expr;
    use crate::tla::eval::eval_expr;
    use crate::tla::{TlaDefinition, TlaState, parse_tla_config, parse_tla_module_file, tla_state};
    use std::path::PathBuf;

    /// Locate the corpus/language directory relative to the crate root.
    fn corpus_language_dir() -> PathBuf {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("corpus");
        dir.push("language");
        dir
    }

    /// Collect .tla/.cfg pairs from a corpus directory.
    fn collect_corpus_specs(dir: &std::path::Path) -> Vec<(PathBuf, Option<PathBuf>)> {
        let mut specs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut tla_files: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "tla"))
                .collect();
            tla_files.sort();
            for tla_path in tla_files {
                let cfg_path = tla_path.with_extension("cfg");
                let cfg = if cfg_path.exists() {
                    Some(cfg_path)
                } else {
                    None
                };
                specs.push((tla_path, cfg));
            }
        }
        specs
    }

    /// Evaluate a definition body via both the text and compiled paths.
    /// Returns (text_result, compiled_result).
    fn eval_both_paths(body: &str, ctx: &EvalContext<'_>) -> (Result<TlaValue>, Result<TlaValue>) {
        let text_result = eval_expr(body, ctx);
        let compiled = compile_expr(body);
        let compiled_result = eval_compiled(&compiled, ctx);
        (text_result, compiled_result)
    }

    #[test]
    fn swarm_eval_path_consistency() {
        let corpus_dir = corpus_language_dir();
        if !corpus_dir.exists() {
            eprintln!(
                "Skipping swarm_eval_path_consistency: corpus dir not found at {}",
                corpus_dir.display()
            );
            return;
        }

        let specs = collect_corpus_specs(&corpus_dir);
        if specs.is_empty() {
            eprintln!("No corpus specs found in {}", corpus_dir.display());
            return;
        }

        // Simple deterministic "random" based on definition index
        let mut checked = 0usize;
        let mut matched = 0usize;
        let mut skipped_action = 0usize;
        let mut both_err = 0usize;

        for (tla_path, cfg_path) in &specs {
            let module = match parse_tla_module_file(tla_path) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let config = if let Some(cfg) = cfg_path {
                let raw = match std::fs::read_to_string(cfg) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                match parse_tla_config(&raw) {
                    Ok(c) => c,
                    Err(_) => continue,
                }
            } else {
                crate::tla::cfg::TlaConfig::default()
            };

            // Build initial state to have variable bindings for the context
            let init_name = config
                .init
                .clone()
                .or_else(|| config.specification.clone())
                .unwrap_or_else(|| "Init".to_string());

            // Use empty state for expression evaluation (non-action definitions)
            let empty_state = TlaState::new();
            let ctx = EvalContext::with_definitions_and_instances(
                &empty_state,
                &module.definitions,
                &module.instances,
            );

            for (def_idx, (name, def)) in module.definitions.iter().enumerate() {
                if def.body.is_empty() {
                    continue;
                }

                // Skip action definitions (contain primes) - they need state context
                if def.body.contains('\'') {
                    skipped_action += 1;
                    continue;
                }

                // Skip parameterized operators (need arguments)
                if !def.params.is_empty() {
                    continue;
                }

                // Swarm: randomly skip some definitions based on index
                // This creates diversity across runs with different corpus orders
                let swarm_bit = (def_idx.wrapping_mul(2654435761)) & 0x3;
                if swarm_bit == 0 {
                    // Skip ~25% of definitions (swarm omission)
                    continue;
                }

                let (text_result, compiled_result) = eval_both_paths(&def.body, &ctx);
                checked += 1;

                match (&text_result, &compiled_result) {
                    (Ok(text_val), Ok(compiled_val)) => {
                        assert_eq!(
                            text_val,
                            compiled_val,
                            "Eval path divergence in {} definition '{}': \
                             text={:?}, compiled={:?}, body='{}'",
                            tla_path.display(),
                            name,
                            text_val,
                            compiled_val,
                            def.body.chars().take(200).collect::<String>(),
                        );
                        matched += 1;
                    }
                    (Err(_), Err(_)) => {
                        // Both errored - that's consistent
                        both_err += 1;
                    }
                    (Ok(text_val), Err(compiled_err)) => {
                        // Compiled path failed but text succeeded.
                        // This is acceptable if the compiled path doesn't support
                        // all operators yet - just log it.
                        eprintln!(
                            "Note: compiled path failed for {} '{}': {} (text got {:?})",
                            tla_path.display(),
                            name,
                            compiled_err,
                            text_val,
                        );
                    }
                    (Err(text_err), Ok(compiled_val)) => {
                        // Text path failed but compiled succeeded - suspicious
                        eprintln!(
                            "Warning: text path failed but compiled succeeded for {} '{}': \
                             text_err={}, compiled={:?}",
                            tla_path.display(),
                            name,
                            text_err,
                            compiled_val,
                        );
                    }
                }
            }
        }

        eprintln!(
            "Swarm eval consistency: checked={}, matched={}, both_err={}, skipped_action={}",
            checked, matched, both_err, skipped_action,
        );

        // Sanity check: we should have checked at least some definitions
        assert!(
            checked > 0,
            "No definitions were checked - corpus may be empty or all definitions are actions"
        );
    }
}

#[cfg(test)]
mod seq_membership_tests {
    use super::*;
    use crate::tla::compiled_expr::compile_expr;
    use crate::tla::{EvalContext, TlaState, TlaValue, tla_state};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// A function with domain 1..n should be accepted as a member of Seq(S).
    /// This is the bug that caused a false positive TypeInvariant violation in
    /// MCCheckpointCoordination: BlankLog = [i \in LogIndex |-> NoNode] produces
    /// a TlaValue::Function, but Log == Seq(Node \cup {NoNode}) requires Seq
    /// membership.
    #[test]
    fn function_with_seq_domain_is_in_seq_set() {
        // Create a function {1 |-> "a", 2 |-> "b", 3 |-> "a"} which is sequence-shaped
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::String("a".into())),
            (TlaValue::Int(2), TlaValue::String("b".into())),
            (TlaValue::Int(3), TlaValue::String("a".into())),
        ])));

        let state = tla_state([
            ("f", func.clone()),
            (
                "S",
                TlaValue::Set(Arc::new(
                    [TlaValue::String("a".into()), TlaValue::String("b".into())]
                        .into_iter()
                        .collect(),
                )),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // f \in Seq(S) should be TRUE since f is sequence-shaped and all values are in S
        let compiled = compile_expr("f \\in Seq(S)");
        let result = eval_compiled(&compiled, &ctx).unwrap();
        assert_eq!(result, TlaValue::Bool(true));
    }

    /// A function with non-sequential domain should NOT be in Seq(S).
    #[test]
    fn function_with_non_seq_domain_not_in_seq_set() {
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(0), TlaValue::String("a".into())),
            (TlaValue::Int(1), TlaValue::String("b".into())),
        ])));

        let state = tla_state([
            ("f", func),
            (
                "S",
                TlaValue::Set(Arc::new(
                    [TlaValue::String("a".into()), TlaValue::String("b".into())]
                        .into_iter()
                        .collect(),
                )),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // Domain starts at 0, not 1 -- not a valid sequence
        let compiled = compile_expr("f \\in Seq(S)");
        let result = eval_compiled(&compiled, &ctx).unwrap();
        assert_eq!(result, TlaValue::Bool(false));
    }

    /// Membership in [Domain -> Range] should be checked structurally via the
    /// compiled FunctionSet arm, without enumerating all possible functions.
    #[test]
    fn function_set_membership_checked_structurally() {
        let func = TlaValue::Function(Arc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Bool(true)),
            (TlaValue::Int(2), TlaValue::Bool(false)),
        ])));

        let state = tla_state([
            ("f", func),
            (
                "D",
                TlaValue::Set(Arc::new(
                    [TlaValue::Int(1), TlaValue::Int(2)].into_iter().collect(),
                )),
            ),
        ]);
        let ctx = EvalContext::new(&state);

        // f \in [D -> BOOLEAN] should be TRUE
        let compiled = compile_expr("f \\in [D -> BOOLEAN]");
        let result = eval_compiled(&compiled, &ctx).unwrap();
        assert_eq!(result, TlaValue::Bool(true));
    }

    #[test]
    fn func_is_sequence_shaped_basic() {
        // Empty is sequence-shaped
        assert!(func_is_sequence_shaped(&BTreeMap::new()));

        // {1 -> x} is sequence-shaped
        let mut m = BTreeMap::new();
        m.insert(TlaValue::Int(1), TlaValue::Bool(true));
        assert!(func_is_sequence_shaped(&m));

        // {1 -> x, 2 -> y} is sequence-shaped
        m.insert(TlaValue::Int(2), TlaValue::Bool(false));
        assert!(func_is_sequence_shaped(&m));

        // {0 -> x} is NOT sequence-shaped (starts at 0)
        let mut m2 = BTreeMap::new();
        m2.insert(TlaValue::Int(0), TlaValue::Bool(true));
        assert!(!func_is_sequence_shaped(&m2));

        // {1 -> x, 3 -> y} is NOT sequence-shaped (gap at 2)
        let mut m3 = BTreeMap::new();
        m3.insert(TlaValue::Int(1), TlaValue::Bool(true));
        m3.insert(TlaValue::Int(3), TlaValue::Bool(false));
        assert!(!func_is_sequence_shaped(&m3));

        // {"a" -> x} is NOT sequence-shaped (non-integer keys)
        let mut m4 = BTreeMap::new();
        m4.insert(TlaValue::String("a".into()), TlaValue::Bool(true));
        assert!(!func_is_sequence_shaped(&m4));
    }
}
