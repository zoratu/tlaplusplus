//! Set-comprehension and set-construction evaluators.
//!
//! `eval_set_expression` is the dispatcher for everything inside `{...}`
//! braces — set literals, filtered comprehensions, mapped
//! comprehensions, and a few special-case fast paths
//! (record-set construction, `{ FunAsSeq(p, n, n) : p \in ... }`).
//!
//! `try_eval_record_set` recognises bracket-form record-set syntax
//! `[f1: S1, f2: S2, ...]` and is reachable from bracket.rs's
//! `eval_bracket_expression`.
//!
//! Several helpers carry `#[cfg(feature = "symbolic-init")]` —
//! they support T5's Z3-backed symbolic Init enumeration.

use anyhow::{Context, Result, anyhow};
use crate::tla::hashed_arc::HashedArc;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, collect_binder_filter_set, collect_binder_map_set,
    contains_top_level_keyword, eval_expr_inner, find_top_level_char,
    find_top_level_keyword_index, is_valid_identifier, parse_binders,
    split_top_level_symbol,
};
#[cfg(feature = "symbolic-init")]
use super::split_once_top_level;

pub(super) fn eval_set_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let inner = expr.trim();
    if inner.is_empty() {
        return Ok(TlaValue::Set(HashedArc::new(BTreeSet::new())));
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
                                                TlaValue::Record(HashedArc::new(BTreeMap::from([
                                                    (f0_name.clone(), TlaValue::Int(a)),
                                                    (f1_name.clone(), TlaValue::Int(b)),
                                                ])))
                                            })
                                            .collect();
                                        return Ok(TlaValue::Seq(HashedArc::new(records)));
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
                                    return Ok(TlaValue::Seq(HashedArc::new(records)));
                                }
                            }

                            let mut out = BTreeSet::new();

                            // Fallback: brute-force with compiled predicate
                            let compiled_pred = crate::tla::compile_expr(rhs);
                            let mut indices = vec![0usize; field_specs.len()];
                            let total: u64 =
                                field_specs.iter().map(|(_, v)| v.len() as u64).product();
                            #[cfg(not(feature = "verus"))]
                            let capped_total = total.min(500_000_000) as usize;
                            #[cfg(feature = "verus")]
                            let capped_total = crate::storage::verus_smoke::min_u64(total, 500_000_000) as usize;
                            ctx.check_budget(capped_total)?;

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
                                let record_arc = HashedArc::new(rec.clone());
                                let mut iter_locals = base_locals.clone();
                                iter_locals.insert(
                                    var_key.clone(),
                                    TlaValue::Record(record_arc.clone()),
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
                            return Ok(TlaValue::Set(HashedArc::new(out)));
                        }
                    }
                }
            }

            // T5.1 — composed FunAsSeq wrapper. If `domain_expr` resolves
            // through definitions to a FunAsSeq-wrapped function-set
            // comprehension (Einstein shape: `Permutation(S)`), rewrite
            // the outer comprehension as a single function-set
            // comprehension and route to the symbolic enumerator.
            #[cfg(feature = "symbolic-init")]
            if is_valid_identifier(var_name) {
                if let Some((inner_var, inner_dom, inner_range, inner_pred)) =
                    try_resolve_funasseq_permutation_set(domain_expr, ctx)
                {
                    if let Some((seq_len, range_vals)) =
                        try_resolve_sequence_domain(&inner_dom, &inner_range, ctx, depth + 1)
                    {
                        // Substitute outer var_name -> inner_var in `rhs`.
                        let outer_pred = substitute_identifier_owned(rhs, var_name, &inner_var);
                        let combined_pred = format!("({}) /\\ ({})", inner_pred, outer_pred);
                        if let Some(seqs) =
                            crate::tla::symbolic_init::try_symbolic_function_set_enumerate(
                                &combined_pred,
                                &inner_var,
                                seq_len,
                                &range_vals,
                                ctx,
                            )
                        {
                            if std::env::var("TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT").is_ok() {
                                eprintln!(
                                    "Symbolic Init (FunAsSeq+filter): {} sequences (len={}, range={})",
                                    seqs.len(),
                                    seq_len,
                                    range_vals.len()
                                );
                            }
                            let set: BTreeSet<TlaValue> = seqs.into_iter().collect();
                            return Ok(TlaValue::Set(HashedArc::new(set)));
                        }
                    }
                }
            }

            // T5.1 — Sequence-set / function-set Init.
            // Recognize `{ var \in [Domain -> Range] : pred(var) }` where
            // Domain == 1..n. This is the Einstein/SortedSeqs shape (when
            // unwrapped from FunAsSeq, see below). We only attempt the
            // symbolic path when the var name is a simple identifier and
            // the predicate references positions like var[i].
            #[cfg(feature = "symbolic-init")]
            if domain_expr_resolved.starts_with('[')
                && domain_expr_resolved.ends_with(']')
                && is_valid_identifier(var_name)
            {
                let bracket_inner = &domain_expr_resolved[1..domain_expr_resolved.len() - 1];
                if let Some((dom_text, range_text)) = split_once_top_level(bracket_inner, "->") {
                    if !dom_text.contains("|") {
                        if let Some((seq_len, range_vals)) = try_resolve_sequence_domain(
                            dom_text.trim(),
                            range_text.trim(),
                            ctx,
                            depth + 1,
                        ) {
                            if let Some(seqs) =
                                crate::tla::symbolic_init::try_symbolic_function_set_enumerate(
                                    rhs,
                                    var_name,
                                    seq_len,
                                    &range_vals,
                                    ctx,
                                )
                            {
                                if std::env::var("TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT").is_ok() {
                                    eprintln!(
                                        "Symbolic Init (function-set): {} sequences (len={}, range={})",
                                        seqs.len(),
                                        seq_len,
                                        range_vals.len()
                                    );
                                }
                                let set: BTreeSet<TlaValue> = seqs.into_iter().collect();
                                return Ok(TlaValue::Set(HashedArc::new(set)));
                            }
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
            return Ok(TlaValue::Set(HashedArc::new(out)));
        }

        // T5.1 — FunAsSeq wrapper recognition.
        // Pattern: `{ FunAsSeq(p, n, n) : p \in <inner> }`. When `<inner>`
        // is itself a function-set comprehension `{ p \in [Dom -> Range] : pred }`
        // we route directly through the symbolic enumerator (the FunAsSeq
        // call simply re-views a function on 1..n as a Seq, which is what
        // the symbolic path produces natively).
        #[cfg(feature = "symbolic-init")]
        if let Some(seqs) = try_funasseq_wrapper_symbolic(lhs, rhs, ctx, depth + 1) {
            if std::env::var("TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT").is_ok() {
                eprintln!("Symbolic Init (FunAsSeq wrapper): {} sequences", seqs.len());
            }
            return Ok(TlaValue::Set(HashedArc::new(seqs.into_iter().collect())));
        }

        let binders = parse_binders(rhs, ctx, depth + 1)?;
        let mut assignments = BTreeMap::new();
        let mut out = BTreeSet::new();
        collect_binder_map_set(0, &binders, lhs, &mut assignments, &mut out, ctx, depth + 1)?;
        return Ok(TlaValue::Set(HashedArc::new(out)));
    }

    let mut out = BTreeSet::new();
    for item in split_top_level_symbol(inner, ",") {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        out.insert(eval_expr_inner(item, ctx, depth + 1)?);
    }

    Ok(TlaValue::Set(HashedArc::new(out)))
}

/// T5.1 helper. Resolve `dom_text -> range_text` to (seq_len, range_values)
/// where `dom_text` evaluates to `1..n` for some positive n and
/// `range_text` evaluates to a finite set. Returns None if the shape
/// doesn't match.
#[cfg(feature = "symbolic-init")]
pub(crate) fn try_resolve_sequence_domain(
    dom_text: &str,
    range_text: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Option<(usize, Vec<TlaValue>)> {
    // Evaluate the domain. It must be a contiguous integer range starting
    // at 1.
    let dom_val = eval_expr_inner(dom_text, ctx, depth + 1).ok()?;
    let dom_set = dom_val.as_set().ok()?;
    if dom_set.is_empty() {
        return None;
    }
    let mut elems: Vec<i64> = Vec::with_capacity(dom_set.len());
    for v in dom_set.iter() {
        let n = v.as_int().ok()?;
        elems.push(n);
    }
    elems.sort_unstable();
    if *elems.first().unwrap() != 1 {
        return None;
    }
    // Must be contiguous 1..n.
    for (i, &v) in elems.iter().enumerate() {
        if v != (i as i64) + 1 {
            return None;
        }
    }
    let seq_len = elems.len();
    // Cap the sequence length to a sane upper bound to avoid solver blowup.
    if seq_len > 32 {
        return None;
    }
    let range_val = eval_expr_inner(range_text, ctx, depth + 1).ok()?;
    let range_set = range_val.as_set().ok()?;
    let range_vals: Vec<TlaValue> = range_set.iter().cloned().collect();
    Some((seq_len, range_vals))
}

/// T5.1 — recognize `{ FunAsSeq(p, n, n) : p \in <inner> }` map-set form
/// where <inner> resolves to a function-set comprehension with predicate.
/// On match, returns `Some(seqs)`; otherwise `None`.
#[cfg(feature = "symbolic-init")]
fn try_funasseq_wrapper_symbolic(
    lhs: &str,
    rhs: &str,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Option<Vec<TlaValue>> {
    // lhs must be `FunAsSeq(IDENT, N, N)`.
    let lhs_t = lhs.trim();
    let after_name = lhs_t.strip_prefix("FunAsSeq")?.trim_start();
    let inner = after_name.strip_prefix('(')?.strip_suffix(')')?;
    let parts = split_top_level_symbol(inner, ",");
    if parts.len() != 3 {
        return None;
    }
    let p_name = parts[0].trim();
    if !is_valid_identifier(p_name) {
        return None;
    }
    let arg2 = parts[1].trim();
    let arg3 = parts[2].trim();
    let n2 = eval_expr_inner(arg2, ctx, depth + 1).ok()?.as_int().ok()?;
    let n3 = eval_expr_inner(arg3, ctx, depth + 1).ok()?.as_int().ok()?;
    if n2 != n3 || n2 < 1 {
        return None;
    }
    let seq_len = n2 as usize;
    if seq_len > 32 {
        return None;
    }

    // rhs must be `<binder> \in <set>` where binder == p_name.
    let in_idx = find_top_level_keyword_index(rhs, "\\in")?;
    let bind_var = rhs[..in_idx].trim();
    if bind_var != p_name {
        return None;
    }
    let inner_set_expr = rhs[in_idx + "\\in".len()..].trim();

    // The inner set must itself be a comprehension `{ p \in [Dom -> Range] : pred }`.
    // It can be either a literal brace-comprehension or a name resolving to one.
    let (inner_var, inner_dom_text, inner_range_text, inner_pred) =
        try_destructure_function_set_comprehension(inner_set_expr, ctx)?;
    if inner_var != p_name {
        return None;
    }
    let (resolved_seq_len, range_vals) =
        try_resolve_sequence_domain(&inner_dom_text, &inner_range_text, ctx, depth + 1)?;
    if resolved_seq_len != seq_len {
        return None;
    }
    crate::tla::symbolic_init::try_symbolic_function_set_enumerate(
        &inner_pred,
        &inner_var,
        seq_len,
        &range_vals,
        ctx,
    )
}

/// Destructure a set comprehension expression of the form
/// `{ p \in [Dom -> Range] : pred }`. The expression may be:
///   - Literal `{...}`
///   - A name resolving to a parameterless definition whose body is `{...}`
///   - A parameterless definition body that itself resolves to that shape
///
/// Returns `(p_name, dom_text, range_text, pred_text)` on success.
#[cfg(feature = "symbolic-init")]
pub(crate) fn try_destructure_function_set_comprehension(
    expr: &str,
    ctx: &EvalContext<'_>,
) -> Option<(String, String, String, String)> {
    let mut text: String = expr.trim().to_string();
    // Resolve parameterless name references up to a small fixed depth.
    for _ in 0..4 {
        let t = text.trim().to_string();
        if t.starts_with('{') && t.ends_with('}') {
            text = t;
            break;
        }
        // Identifier (possibly with parentheses-call to a 1-arg operator).
        // Try strict identifier first.
        if is_valid_identifier(t.as_str()) {
            if let Some(defs) = ctx.definitions {
                if let Some(def) = defs.get(&t) {
                    if def.params.is_empty() {
                        text = def.body.trim().to_string();
                        continue;
                    }
                }
            }
            return None;
        }
        // Try operator call form `Name(args)` and substitute literal args.
        if let Some(open) = t.find('(') {
            if t.ends_with(')') {
                let name = t[..open].trim();
                if is_valid_identifier(name) {
                    if let Some(defs) = ctx.definitions {
                        if let Some(def) = defs.get(name) {
                            let args_text = &t[open + 1..t.len() - 1];
                            let args = split_top_level_symbol(args_text, ",");
                            if def.params.len() == args.len() {
                                let mut body = def.body.trim().to_string();
                                for (param, arg) in def.params.iter().zip(args.iter()) {
                                    body = substitute_identifier_owned(&body, param, arg.trim());
                                }
                                text = body;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        return None;
    }
    if !(text.starts_with('{') && text.ends_with('}')) {
        return None;
    }
    let inner = &text[1..text.len() - 1];
    let colon_idx = find_top_level_char(inner, ':')?;
    let lhs = inner[..colon_idx].trim();
    let pred = inner[colon_idx + 1..].trim().to_string();
    // Could itself be a `FunAsSeq(...)` wrapper. We handle both forms.
    // Here, we want lhs to be `<var> \in [Dom -> Range]` form.
    let in_idx = find_top_level_keyword_index(lhs, "\\in")?;
    let var_name = lhs[..in_idx].trim().to_string();
    if !is_valid_identifier(&var_name) {
        return None;
    }
    let dom_expr = lhs[in_idx + "\\in".len()..].trim();
    if !(dom_expr.starts_with('[') && dom_expr.ends_with(']')) {
        return None;
    }
    let bracket_inner = &dom_expr[1..dom_expr.len() - 1];
    let arrow_split = split_once_top_level(bracket_inner, "->")?;
    let (dom_text, range_text) = arrow_split;
    if dom_text.contains("|") {
        return None;
    }
    Some((
        var_name,
        dom_text.trim().to_string(),
        range_text.trim().to_string(),
        pred,
    ))
}

/// T5.1 — try to resolve `domain_expr` (a name or operator call like
/// `Permutation(S)`, or a literal `{ FunAsSeq(p, n, n) : p \in INNER }`)
/// to a tuple `(p_name, dom_text, range_text, pred_text)` describing the
/// underlying function-set comprehension.
///
/// On success the caller may treat `{ x \in <domain_expr> : outer_pred(x) }`
/// as `{ FunAsSeq(p, n, n) : p \in {p \in [Dom -> Range] : pred /\ outer_pred[x:=p]}}`.
#[cfg(feature = "symbolic-init")]
pub(crate) fn try_resolve_funasseq_permutation_set(
    domain_expr: &str,
    ctx: &EvalContext<'_>,
) -> Option<(String, String, String, String)> {
    let mut text = domain_expr.trim().to_string();
    // Resolve names / operator calls up to a fixed depth.
    for _ in 0..4 {
        let t = text.trim().to_string();
        // Direct match: `{ FunAsSeq(p, n, n) : p \in <inner> }`.
        if t.starts_with('{') && t.ends_with('}') {
            if let Some(parsed) = parse_funasseq_comprehension(&t, ctx) {
                return Some(parsed);
            }
            return None;
        }
        if is_valid_identifier(t.as_str()) {
            if let Some(defs) = ctx.definitions {
                if let Some(def) = defs.get(&t) {
                    if def.params.is_empty() {
                        text = def.body.trim().to_string();
                        continue;
                    }
                }
            }
            return None;
        }
        if let Some(open) = t.find('(') {
            if t.ends_with(')') {
                let name = t[..open].trim();
                if is_valid_identifier(name) {
                    if let Some(defs) = ctx.definitions {
                        if let Some(def) = defs.get(name) {
                            let args_text = &t[open + 1..t.len() - 1];
                            let args = split_top_level_symbol(args_text, ",");
                            if def.params.len() == args.len() {
                                let mut body = def.body.trim().to_string();
                                for (param, arg) in def.params.iter().zip(args.iter()) {
                                    body = substitute_identifier_owned(&body, param, arg.trim());
                                }
                                text = body;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        return None;
    }
    None
}

/// Parse a literal `{ FunAsSeq(p, n, n) : p \in <inner> }` expression.
#[cfg(feature = "symbolic-init")]
fn parse_funasseq_comprehension(
    text: &str,
    ctx: &EvalContext<'_>,
) -> Option<(String, String, String, String)> {
    let inner = text.trim().strip_prefix('{')?.strip_suffix('}')?;
    let colon_idx = find_top_level_char(inner, ':')?;
    let lhs = inner[..colon_idx].trim();
    let rhs = inner[colon_idx + 1..].trim();
    // lhs must be `FunAsSeq(p, n, n)`.
    let after_name = lhs.strip_prefix("FunAsSeq")?.trim_start();
    let inner_args = after_name.strip_prefix('(')?.strip_suffix(')')?;
    let parts = split_top_level_symbol(inner_args, ",");
    if parts.len() != 3 {
        return None;
    }
    let p_name = parts[0].trim();
    if !is_valid_identifier(p_name) {
        return None;
    }
    let n2 = eval_expr_inner(parts[1].trim(), ctx, 0)
        .ok()?
        .as_int()
        .ok()?;
    let n3 = eval_expr_inner(parts[2].trim(), ctx, 0)
        .ok()?
        .as_int()
        .ok()?;
    if n2 != n3 || n2 < 1 {
        return None;
    }
    // rhs: `<binder> \in <inner-set>`.
    let in_idx = find_top_level_keyword_index(rhs, "\\in")?;
    let bind_var = rhs[..in_idx].trim();
    if bind_var != p_name {
        return None;
    }
    let inner_set_expr = rhs[in_idx + "\\in".len()..].trim();
    let (inner_var, dom_text, range_text, pred_text) =
        try_destructure_function_set_comprehension(inner_set_expr, ctx)?;
    if inner_var != p_name {
        return None;
    }
    Some((inner_var, dom_text, range_text, pred_text))
}

/// Whole-word identifier substitution returning an owned String.
#[cfg(feature = "symbolic-init")]
fn substitute_identifier_owned(source: &str, name: &str, replacement: &str) -> String {
    let bytes = source.as_bytes();
    let needle = name.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + needle.len() <= bytes.len() && &bytes[i..i + needle.len()] == needle {
            let prev_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let next_idx = i + needle.len();
            let next_ok = next_idx == bytes.len()
                || !(bytes[next_idx].is_ascii_alphanumeric() || bytes[next_idx] == b'_');
            if prev_ok && next_ok {
                out.push_str(replacement);
                i += needle.len();
                continue;
            }
        }
        let ch = source[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Returns None if the expression doesn't match the record set pattern.
pub(super) fn try_eval_record_set(
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
        return Ok(Some(TlaValue::Set(HashedArc::new(BTreeSet::new()))));
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
        result.insert(TlaValue::Record(HashedArc::new(record)));

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

    Ok(Some(TlaValue::Set(HashedArc::new(result))))
}

pub(super) fn powerset(input: &BTreeSet<TlaValue>, ctx: &EvalContext<'_>) -> Result<BTreeSet<TlaValue>> {
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
        subsets.insert(TlaValue::Set(HashedArc::new(subset)));
    }

    Ok(subsets)
}

pub(super) fn generate_permutations(values: &[TlaValue]) -> Vec<Vec<TlaValue>> {
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
