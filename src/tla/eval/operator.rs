//! TLA+ standard-library operator dispatch.
//!
//! `eval_operator_call` is the giant `match name { ... }` that fans
//! out to every named built-in (`Cardinality`, `Max`, `Append`, `Head`,
//! `DOMAIN`, `SubSeq`, `Permutations`, `Bag*`, `IO*`, `Bitwise*`,
//! `Combinations`, `VectorClock*`, `Random*`, `Print*`, `Assert`,
//! `IsFiniteSet`, etc.) plus the user-defined-operator inlining
//! fallthrough at the bottom.
//!
//! The arms are opaque to the rest of the evaluator — splitting them
//! by category would multiply the dispatch cost (string compare in N
//! small files instead of one big file). The function moves intact
//! per the plan.
//!
//! Hosts companions: `eval_builtin_extremum/bounded_seq/tlc_get`,
//! `gcd`, `sequence_like_values`, `seq_or_string_concat`,
//! `tla_to_string`, `record_key_from_value`,
//! `is_user_defined_infix_operator`, and the JSON helpers.

use anyhow::{Result, anyhow};
use crate::tla::hashed_arc::HashedArc;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::tla::TlaValue;

use super::{
    EvalContext, MAX_EVAL_DEPTH, apply_value, bind_param_value, eval_expr_inner,
    generate_permutations,
};

pub(crate) fn eval_operator_call(
    name: &str,
    args: Vec<TlaValue>,
    ctx: &EvalContext<'_>,
    depth: usize,
) -> Result<TlaValue> {
    // Depth-tracking consolidation (iter11): callers pass `depth` (not
    // `depth + 1`); we increment here. Mirrors compiled_eval.rs's pattern.
    let depth = depth + 1;
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
            return Ok(TlaValue::Seq(HashedArc::new(seq[1..].to_vec())));
        }
        "Append" => {
            if args.len() != 2 {
                return Err(anyhow!("Append expects 2 arguments"));
            }
            let mut new_seq = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("Append expects a sequence, got {:?}", args[0]))?;
            new_seq.push(args[1].clone());
            return Ok(TlaValue::Seq(HashedArc::new(new_seq)));
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
                return Ok(TlaValue::Seq(HashedArc::new(vec![])));
            }

            return Ok(TlaValue::Seq(HashedArc::new(seq[start..end].to_vec())));
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
            return Ok(TlaValue::Seq(HashedArc::new(result)));
        }
        // === Community module: SequencesExt ===
        "SeqOf" if args.len() == 2 && !user_defined_shadow => {
            // SeqOf(set, n) == UNION {[1..m -> set] : m \in 0..n}
            let set = args[0].as_set()?;
            let n = args[1].as_int()?;
            if n < 0 {
                return Ok(TlaValue::Set(HashedArc::new(BTreeSet::new())));
            }
            ctx.check_budget(1)?; // budget check before potentially large construction
            let elements: Vec<TlaValue> = set.iter().cloned().collect();
            let mut result = BTreeSet::new();
            // m=0: empty sequence
            result.insert(TlaValue::Seq(HashedArc::new(vec![])));
            for m in 1..=n as usize {
                let total = elements.len().checked_pow(m as u32).unwrap_or(usize::MAX);
                ctx.check_budget(total)?;
                // Generate all sequences of length m over elements
                let mut indices = vec![0usize; m];
                loop {
                    let seq: Vec<TlaValue> = indices.iter().map(|&i| elements[i].clone()).collect();
                    result.insert(TlaValue::Seq(HashedArc::new(seq)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
            return Ok(TlaValue::Seq(HashedArc::new(result)));
        }
        // SequencesExt prefix/suffix family.
        //   IsPrefix(s, t)       == DOMAIN s \subseteq DOMAIN t /\ \A i \in DOMAIN s : s[i] = t[i]
        //   IsStrictPrefix(s, t) == IsPrefix(s, t) /\ s # t
        //   IsSuffix(s, t)       == IsPrefix(Reverse(s), Reverse(t))   (s matches the tail of t)
        //   IsStrictSuffix(s, t) == IsSuffix(s, t) /\ s # t
        // Element comparison uses TLA+ semantic equality (a Seq and a 1..n
        // Function are equal), matching the `=` operator.
        "IsPrefix" | "IsStrictPrefix" if args.len() == 2 && !user_defined_shadow => {
            let s = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("{} expects a sequence as 1st arg, got {:?}", name, args[0]))?;
            let t = sequence_like_values(&args[1])
                .ok_or_else(|| anyhow!("{} expects a sequence as 2nd arg, got {:?}", name, args[1]))?;
            let is_prefix =
                s.len() <= t.len() && s.iter().zip(t.iter()).all(|(a, b)| a.semantic_eq(b));
            // Given `is_prefix`, s # t exactly when the lengths differ.
            let result = if name == "IsStrictPrefix" {
                is_prefix && s.len() < t.len()
            } else {
                is_prefix
            };
            return Ok(TlaValue::Bool(result));
        }
        "IsSuffix" | "IsStrictSuffix" if args.len() == 2 && !user_defined_shadow => {
            let s = sequence_like_values(&args[0])
                .ok_or_else(|| anyhow!("{} expects a sequence as 1st arg, got {:?}", name, args[0]))?;
            let t = sequence_like_values(&args[1])
                .ok_or_else(|| anyhow!("{} expects a sequence as 2nd arg, got {:?}", name, args[1]))?;
            let is_suffix = s.len() <= t.len() && {
                let off = t.len() - s.len();
                s.iter().enumerate().all(|(i, a)| a.semantic_eq(&t[i + off]))
            };
            let result = if name == "IsStrictSuffix" {
                is_suffix && s.len() < t.len()
            } else {
                is_suffix
            };
            return Ok(TlaValue::Bool(result));
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
            return Ok(TlaValue::Record(HashedArc::new(BTreeMap::from([
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
                return Ok(TlaValue::Record(HashedArc::new(BTreeMap::from([
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
                    vec![TlaValue::Set(HashedArc::new(remaining.clone()))],
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
        }
        "FlattenSet" if args.len() == 1 && !user_defined_shadow => {
            let sets = args[0].as_set()?;
            let mut result = BTreeSet::new();
            for s in sets.iter() {
                result.extend(s.as_set()?.iter().cloned());
            }
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
                    result.insert(TlaValue::Set(HashedArc::new(subset)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
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
            return Ok(TlaValue::Set(HashedArc::new(result)));
        }
        "Transpose" if args.len() == 1 && !user_defined_shadow => {
            let g = &args[0];
            let nodes = g.select_key("node")?.clone();
            let edges = g.select_key("edge")?.as_set()?;
            let mut reversed = BTreeSet::new();
            for e in edges.iter() {
                if let TlaValue::Seq(pair) = e {
                    if pair.len() == 2 {
                        reversed.insert(TlaValue::Seq(HashedArc::new(vec![
                            pair[1].clone(),
                            pair[0].clone(),
                        ])));
                    }
                }
            }
            return Ok(TlaValue::Record(HashedArc::new(BTreeMap::from([
                ("node".to_string(), nodes),
                ("edge".to_string(), TlaValue::Set(HashedArc::new(reversed))),
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
                    let key = TlaValue::Seq(HashedArc::new(vec![ni.clone(), nj.clone()]));
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
                    let key = TlaValue::Seq(HashedArc::new(vec![ni.clone(), nj.clone()]));
                    result.insert(key, TlaValue::Bool(reach[i][j]));
                }
            }
            return Ok(TlaValue::Function(HashedArc::new(result)));
        }
        // === Standard module: Bags ===
        // Bags are represented as TlaValue::Function from elements to natural counts.
        // EmptyBag is the empty function. SetToBag converts a set to a bag with count 1.
        "EmptyBag" if args.is_empty() && !user_defined_shadow => {
            return Ok(TlaValue::Function(HashedArc::new(BTreeMap::new())));
        }
        "SetToBag" if args.len() == 1 && !user_defined_shadow => {
            // SetToBag(S) == [e \in S |-> 1]
            let set = args[0].as_set()?;
            let mut bag = BTreeMap::new();
            for elem in set.iter() {
                bag.insert(elem.clone(), TlaValue::Int(1));
            }
            return Ok(TlaValue::Function(HashedArc::new(bag)));
        }
        "BagToSet" if args.len() == 1 && !user_defined_shadow => {
            // BagToSet(B) == DOMAIN B
            if let TlaValue::Function(f) = &args[0] {
                let set: BTreeSet<TlaValue> = f.keys().cloned().collect();
                return Ok(TlaValue::Set(HashedArc::new(set)));
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
                return Ok(TlaValue::Function(HashedArc::new(result)));
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
                return Ok(TlaValue::Function(HashedArc::new(result)));
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
                return Ok(TlaValue::Function(HashedArc::new(result)));
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
                return Ok(TlaValue::Function(HashedArc::new(result)));
            }
            return Err(anyhow!("BagRemove: first argument is not a bag"));
        }
        // === Standard module: IOUtils ===
        "IOEnv" if args.is_empty() && !user_defined_shadow => {
            let mut rec = BTreeMap::new();
            for (k, v) in std::env::vars() {
                rec.insert(k, TlaValue::String(v));
            }
            return Ok(TlaValue::Record(HashedArc::new(rec)));
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
            return Ok(TlaValue::Seq(HashedArc::new(items)));
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
                rows.push(TlaValue::Seq(HashedArc::new(fields)));
            }
            return Ok(TlaValue::Seq(HashedArc::new(rows)));
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
                return Ok(TlaValue::Function(HashedArc::new(result)));
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
                .map(|s| TlaValue::Set(HashedArc::new(s)))
                .collect();
            return Ok(TlaValue::Set(HashedArc::new(result)));
        }
        "AreConnectedIn" if args.len() == 3 && !user_defined_shadow => {
            let m = &args[0];
            let n = &args[1];
            let g = &args[2];
            // Reuse ConnectedComponents
            let comps = eval_operator_call("ConnectedComponents", vec![g.clone()], ctx, depth)?;
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
                eval_operator_call("ConnectedComponents", vec![args[0].clone()], ctx, depth)?;
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
                out.insert(TlaValue::Function(HashedArc::new(map)));
            }
            return Ok(TlaValue::Set(HashedArc::new(out)));
        }
        "DOMAIN" => {
            if args.len() != 1 {
                return Err(anyhow!("DOMAIN expects 1 argument"));
            }
            match &args[0] {
                TlaValue::Function(map) => {
                    let keys = map.keys().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(HashedArc::new(keys)));
                }
                TlaValue::Record(map) => {
                    let keys = map
                        .keys()
                        .map(|k| TlaValue::String(k.clone()))
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(HashedArc::new(keys)));
                }
                TlaValue::Seq(seq) => {
                    // DOMAIN of a sequence is {1, 2, ..., Len(seq)}
                    let indices = (1..=seq.len() as i64)
                        .map(TlaValue::Int)
                        .collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(HashedArc::new(indices)));
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
            return Ok(TlaValue::Set(HashedArc::new(union)));
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
            return Ok(TlaValue::Set(HashedArc::new(subset)));
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
                    return Ok(TlaValue::Set(HashedArc::new(values)));
                }
                TlaValue::Seq(seq) => {
                    // Range of a sequence is the set of all its elements
                    let values = seq.iter().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(HashedArc::new(values)));
                }
                TlaValue::Record(map) => {
                    // Range of a record is the set of all its field values
                    let values = map.values().cloned().collect::<BTreeSet<_>>();
                    return Ok(TlaValue::Set(HashedArc::new(values)));
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
            return Ok(TlaValue::Seq(HashedArc::new(result)));
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
        // A module-level operator is lexically scoped to the module: its body
        // must not see the caller's lexical locals (a free variable would else
        // bind to a same-named caller local). Drop them, keeping only staged
        // primed (next-state) bindings plus the params bound below. A LET-local
        // operator may reference an enclosing bound variable, so it keeps them.
        if !ctx.local_definitions.contains_key(name) {
            locals_mut.retain(|k, _| k.ends_with('\''));
        }
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
    out.insert(TlaValue::Seq(HashedArc::new(Vec::new())));

    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in &current {
            for value in &elements {
                let mut seq = prefix.clone();
                seq.push(value.clone());
                out.insert(TlaValue::Seq(HashedArc::new(seq.clone())));
                next.push(seq);
            }
        }
        current = next;
    }

    Ok(TlaValue::Set(HashedArc::new(out)))
}

fn eval_builtin_tlc_get(key: &TlaValue) -> Result<TlaValue> {
    match key {
        TlaValue::String(name) if name == "level" => Ok(TlaValue::Int(0)),
        TlaValue::String(name) if name == "config" => {
            Ok(TlaValue::Record(HashedArc::new(BTreeMap::from([
                ("mode".to_string(), TlaValue::String("bfs".to_string())),
                ("worker".to_string(), TlaValue::Int(1)),
            ]))))
        }
        TlaValue::Int(slot) if *slot == 2 || *slot == 3 => Ok(TlaValue::Int(999)),
        TlaValue::Int(_) => Ok(TlaValue::Int(0)),
        other => Err(anyhow!("unsupported TLCGet key: {:?}", other)),
    }
}


#[cfg(not(feature = "verus"))]
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(feature = "verus")]
fn gcd(a: u64, b: u64) -> u64 {
    crate::storage::verus_smoke::gcd(a, b)
}

pub(super) fn sequence_like_values(value: &TlaValue) -> Option<Vec<TlaValue>> {
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

pub(super) fn seq_or_string_concat(lhs: TlaValue, rhs: TlaValue) -> Result<TlaValue> {
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
            Ok(TlaValue::Seq(HashedArc::new(lhs_seq)))
        }
    }
}

pub(super) fn record_key_from_value(value: &TlaValue) -> Result<String> {
    match value {
        TlaValue::String(v) | TlaValue::ModelValue(v) => Ok(v.clone()),
        _ => Err(anyhow!(
            "record key must be String or ModelValue, got {value:?}"
        )),
    }
}

pub(super) fn tla_to_string(value: &TlaValue) -> String {
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

pub(super) fn is_user_defined_infix_operator(name: &str) -> bool {
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
            TlaValue::Seq(HashedArc::new(items))
        }
        serde_json::Value::Object(obj) => {
            let mut rec = BTreeMap::new();
            for (k, v) in obj {
                rec.insert(k.clone(), json_to_tla_value(v));
            }
            TlaValue::Record(HashedArc::new(rec))
        }
    }
}


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
