//! Symbolic Init enumeration via SMT (Z3).
//!
//! For Init expressions of the shape
//!
//! ```text
//! { tup \in [f1: D1, f2: D2, ..., fN: DN] : Predicate(tup) }
//! ```
//!
//! where each `Di` is a finite domain of integers / strings / model values
//! and `Predicate` uses only a supported subset of TLA+ operators, this
//! module translates the predicate into a Z3 query and enumerates all
//! satisfying assignments via the standard block-and-resolve loop.
//!
//! Supported subset:
//!   - Boolean ops: `/\`, `\/`, `~`, `=>`, `<=>`, `TRUE`, `FALSE`
//!   - Equality: `=`, `#`, `/=`
//!   - Integer comparisons: `<`, `<=`, `>=`, `>`
//!   - Integer arithmetic on field projections: `+`, `-`, `*` (constant fold OK)
//!   - Set membership: `var.f \in {literal,...}` and `var.f \in lo..hi`
//!   - Field access: `var.f` (returns the field's symbolic value)
//!   - Quantifiers expanded over finite literal domains (`\A x \in {a,b,c} : P(x)`)
//!   - Pre-evaluable subexpressions are constant-folded via the existing evaluator
//!
//! Anything outside the subset returns `None` so the caller falls back to
//! brute-force enumeration. Correctness gate: the caller should compare
//! results against brute force on small inputs (see `tests/symbolic_init_*`).
//!
//! # Feature flag
//!
//! Enabled with `--features symbolic-init`. When the feature is off, all
//! entry points return `None` and there is no z3 build dependency.

#[cfg(feature = "symbolic-init")]
use std::collections::BTreeMap;
#[cfg(feature = "symbolic-init")]
use std::sync::Arc;

use crate::tla::eval::EvalContext;
use crate::tla::value::TlaValue;

/// Hard cap on how many solutions we'll enumerate via SMT before bailing out.
/// At ~10M solutions, brute force is probably comparable, so the symbolic
/// path is no longer a clear win; we'd rather fall back than risk pinning
/// the solver in a pathological case.
pub const SYMBOLIC_ENUM_HARD_CAP: usize = 10_000_000;

/// Try to enumerate all records `tup` satisfying the predicate
/// `pred_text` in the record-set comprehension
///
/// ```text
/// { tup \in [f1: D1, ..., fN: DN] : pred_text }
/// ```
///
/// Returns `Some(records)` if the translation succeeds end-to-end
/// (including SAT-loop completion), or `None` if any part of the
/// predicate is outside the supported subset.
///
/// Each `field_specs` entry is `(field_name, possible_values)` already
/// evaluated by the caller. The values may be any `TlaValue` variant the
/// translator can encode (`Int`, `Bool`, `String`, `ModelValue`).
#[cfg_attr(not(feature = "symbolic-init"), allow(unused_variables))]
pub fn try_symbolic_record_set_enumerate(
    pred_text: &str,
    var_name: &str,
    field_specs: &[(String, Vec<TlaValue>)],
    ctx: &EvalContext<'_>,
) -> Option<Vec<TlaValue>> {
    #[cfg(not(feature = "symbolic-init"))]
    {
        None
    }
    #[cfg(feature = "symbolic-init")]
    {
        backend::try_symbolic_record_set_enumerate_z3(pred_text, var_name, field_specs, ctx)
    }
}

/// Try to enumerate all sequences `var` satisfying the predicate
/// `pred_text` in the function-set comprehension
///
/// ```text
/// { var \in [Domain -> Range] : pred_text }
/// ```
///
/// Where `Domain` is `1..n` (positive contiguous integers starting at 1)
/// and `Range` is a finite set of values. Sequences are encoded as `n`
/// per-position Int variables in Z3. The predicate may use
/// - `var[i]` field projections (i a literal Int)
/// - `var[i] = const`, `var[i] # var[j]`, `var[i] \in S`, `var[i] \in S \ {var[j], ...}`
/// - All boolean operators (`/\`, `\/`, `~`, `=>`, `<=>`)
/// - Quantifiers expanded over finite literal domains (e.g. `\E i \in 1..4 : ...`)
/// - Pre-evaluable subexpressions (constant-folded via the outer evaluator)
///
/// T5.2: distinctness shapes — `\A i, j \in Dom : i # j => var[i] # var[j]`,
/// or chains like `var[2] \in S \ {var[1]}`, `var[3] \in S \ {var[1], var[2]}`,
/// etc. — are detected as a permutation predicate and translated via Z3's
/// `Distinct` constraint, which is exponentially smaller than O(n^2)
/// pairwise inequalities.
///
/// Returns `Some(seqs)` on success (each TlaValue is a `Seq` of length `n`),
/// or `None` if the predicate or domain shape is outside the supported subset.
#[cfg_attr(not(feature = "symbolic-init"), allow(unused_variables))]
pub fn try_symbolic_function_set_enumerate(
    pred_text: &str,
    var_name: &str,
    seq_len: usize,
    range: &[TlaValue],
    ctx: &EvalContext<'_>,
) -> Option<Vec<TlaValue>> {
    #[cfg(not(feature = "symbolic-init"))]
    {
        None
    }
    #[cfg(feature = "symbolic-init")]
    {
        backend::try_symbolic_function_set_enumerate_z3(pred_text, var_name, seq_len, range, ctx)
    }
}

/// Build a record `TlaValue` from a field assignment vector.
#[cfg(feature = "symbolic-init")]
pub(crate) fn build_record(field_assignments: Vec<(String, TlaValue)>) -> TlaValue {
    let mut rec: BTreeMap<String, TlaValue> = BTreeMap::new();
    for (k, v) in field_assignments {
        rec.insert(k, v);
    }
    TlaValue::Record(Arc::new(rec))
}

/// Build a Seq `TlaValue` from a Vec of element values.
#[cfg(feature = "symbolic-init")]
pub(crate) fn build_seq(elems: Vec<TlaValue>) -> TlaValue {
    TlaValue::Seq(Arc::new(elems))
}

#[cfg(feature = "symbolic-init")]
mod backend {
    use super::*;
    use crate::tla::eval::{
        EvalContext, find_top_level_char, find_top_level_keyword_index, is_valid_identifier,
        split_top_level_keyword, split_top_level_symbol,
    };
    use crate::tla::eval_expr;
    use std::collections::HashMap;
    use z3::ast::{Ast, Bool, Int};
    use z3::{Config, Context, SatResult, Solver};

    /// Encoding of one record field: either an Int value or an enum-coded
    /// non-integer (string / model value / bool). For enum-coded fields we
    /// keep the original `TlaValue` per integer code and translate equality
    /// constraints into integer equality on the code.
    enum FieldEncoding {
        IntDomain {
            /// Sorted list of allowed integer values.
            values: Vec<i64>,
        },
        EnumDomain {
            /// `code -> original TlaValue`. Always 0..len exclusive.
            values: Vec<TlaValue>,
        },
    }

    struct Translator<'ctx, 'a> {
        z3: &'ctx Context,
        var_name: &'a str,
        /// `field_name -> (Z3 Int variable, encoding, ordered field index)`
        fields: HashMap<&'a str, (Int<'ctx>, &'a FieldEncoding, usize)>,
        /// Outer evaluator for constant folding of subexpressions that
        /// don't reference the bound record variable.
        eval_ctx: &'a EvalContext<'a>,
    }

    pub(super) fn try_symbolic_record_set_enumerate_z3(
        pred_text: &str,
        var_name: &str,
        field_specs: &[(String, Vec<TlaValue>)],
        ctx: &EvalContext<'_>,
    ) -> Option<Vec<TlaValue>> {
        if field_specs.is_empty() {
            return None;
        }

        // Build per-field encodings.
        let mut encodings: Vec<FieldEncoding> = Vec::with_capacity(field_specs.len());
        for (_, vals) in field_specs {
            if vals.is_empty() {
                // Empty domain → no records.
                return Some(Vec::new());
            }
            let all_ints = vals.iter().all(|v| matches!(v, TlaValue::Int(_)));
            if all_ints {
                let mut ints: Vec<i64> = vals.iter().map(|v| v.as_int().unwrap()).collect();
                ints.sort_unstable();
                ints.dedup();
                encodings.push(FieldEncoding::IntDomain { values: ints });
            } else {
                let all_codable = vals.iter().all(|v| {
                    matches!(
                        v,
                        TlaValue::Int(_)
                            | TlaValue::Bool(_)
                            | TlaValue::String(_)
                            | TlaValue::ModelValue(_)
                    )
                });
                if !all_codable {
                    return None;
                }
                let mut deduped: Vec<TlaValue> = Vec::new();
                for v in vals {
                    if !deduped.contains(v) {
                        deduped.push(v.clone());
                    }
                }
                encodings.push(FieldEncoding::EnumDomain { values: deduped });
            }
        }

        let z3_cfg = Config::new();
        let z3_ctx = Context::new(&z3_cfg);
        let solver = Solver::new(&z3_ctx);

        // Create symbolic Int variables for each field and assert the
        // domain constraints. For Int domains we enforce range or set
        // membership; for enum domains we enforce 0 <= code < len.
        let mut field_map: HashMap<&str, (Int<'_>, &FieldEncoding, usize)> = HashMap::new();
        let mut z3_vars: Vec<Int<'_>> = Vec::with_capacity(field_specs.len());
        for (idx, (fname, _)) in field_specs.iter().enumerate() {
            let var = Int::new_const(&z3_ctx, format!("{}_{}", var_name, fname));
            let enc = &encodings[idx];
            assert_domain_constraint(&z3_ctx, &solver, &var, enc);
            field_map.insert(fname.as_str(), (var.clone(), enc, idx));
            z3_vars.push(var);
        }

        let translator = Translator {
            z3: &z3_ctx,
            var_name,
            fields: field_map,
            eval_ctx: ctx,
        };

        let pred_z3 = translator.translate_bool(pred_text)?;
        solver.assert(&pred_z3);

        // Block-and-resolve enumeration loop.
        let mut results: Vec<TlaValue> = Vec::new();
        loop {
            match solver.check() {
                SatResult::Sat => {}
                SatResult::Unsat => break,
                SatResult::Unknown => {
                    // Solver gave up — bail out to brute force so we
                    // don't silently miss states.
                    return None;
                }
            }

            if results.len() >= SYMBOLIC_ENUM_HARD_CAP {
                // Too many solutions — fall back so the caller can use
                // its tuned brute-force loop.
                return None;
            }

            let model = solver.get_model()?;

            // Extract values, build the record, and emit a blocking clause.
            let mut field_assignments: Vec<(String, TlaValue)> =
                Vec::with_capacity(field_specs.len());
            let mut block_disjuncts: Vec<Bool<'_>> = Vec::with_capacity(field_specs.len());

            for (fname, _) in field_specs {
                let (var, enc, _idx) = translator.fields.get(fname.as_str())?;
                let code = model.eval(var, true)?.as_i64()?;
                let value = match enc {
                    FieldEncoding::IntDomain { .. } => TlaValue::Int(code),
                    FieldEncoding::EnumDomain { values } => {
                        if code < 0 || (code as usize) >= values.len() {
                            return None;
                        }
                        values[code as usize].clone()
                    }
                };
                field_assignments.push((fname.clone(), value));
                let lit = Int::from_i64(&z3_ctx, code);
                block_disjuncts.push(var._eq(&lit).not());
            }

            results.push(build_record(field_assignments));

            // Block this exact assignment: assert the disjunction of
            // "at least one field differs".
            let refs: Vec<&Bool<'_>> = block_disjuncts.iter().collect();
            let block = Bool::or(&z3_ctx, &refs);
            solver.assert(&block);
        }

        Some(results)
    }

    fn assert_domain_constraint<'ctx>(
        z3: &'ctx Context,
        solver: &Solver<'ctx>,
        var: &Int<'ctx>,
        enc: &FieldEncoding,
    ) {
        match enc {
            FieldEncoding::IntDomain { values } => {
                if values.is_empty() {
                    let f = Bool::from_bool(z3, false);
                    solver.assert(&f);
                    return;
                }
                // Detect a contiguous range; emit lo <= x <= hi.
                let lo = *values.first().unwrap();
                let hi = *values.last().unwrap();
                let contiguous = (hi - lo) as usize + 1 == values.len();
                if contiguous {
                    let lo_e = Int::from_i64(z3, lo);
                    let hi_e = Int::from_i64(z3, hi);
                    solver.assert(&var.ge(&lo_e));
                    solver.assert(&var.le(&hi_e));
                } else {
                    // Enumerate as a disjunction of equalities.
                    let eqs: Vec<Bool<'_>> = values
                        .iter()
                        .map(|&v| var._eq(&Int::from_i64(z3, v)))
                        .collect();
                    let refs: Vec<&Bool<'_>> = eqs.iter().collect();
                    solver.assert(&Bool::or(z3, &refs));
                }
            }
            FieldEncoding::EnumDomain { values } => {
                let lo = Int::from_i64(z3, 0);
                let hi = Int::from_i64(z3, values.len() as i64 - 1);
                solver.assert(&var.ge(&lo));
                solver.assert(&var.le(&hi));
            }
        }
    }

    impl<'ctx, 'a> Translator<'ctx, 'a> {
        /// Translate a TLA+ boolean expression into a Z3 Bool.
        fn translate_bool(&self, expr: &str) -> Option<Bool<'ctx>> {
            let expr = strip_redundant_parens(expr.trim());

            // TRUE / FALSE literals.
            if expr.eq_ignore_ascii_case("TRUE") {
                return Some(Bool::from_bool(self.z3, true));
            }
            if expr.eq_ignore_ascii_case("FALSE") {
                return Some(Bool::from_bool(self.z3, false));
            }

            // <=> (equivalence). Lower precedence than =>.
            if let Some(parts) = split_two(expr, "<=>") {
                let a = self.translate_bool(&parts.0)?;
                let b = self.translate_bool(&parts.1)?;
                return Some(a.iff(&b));
            }

            // => (implication).
            if let Some(parts) = split_two(expr, "=>") {
                let a = self.translate_bool(&parts.0)?;
                let b = self.translate_bool(&parts.1)?;
                return Some(a.implies(&b));
            }

            // \/ (disjunction). Lower precedence than /\.
            let dj = split_top_level_keyword(expr, "\\/");
            if dj.len() > 1 {
                let mut bools: Vec<Bool<'ctx>> = Vec::with_capacity(dj.len());
                for part in &dj {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    bools.push(self.translate_bool(trimmed)?);
                }
                let refs: Vec<&Bool<'_>> = bools.iter().collect();
                return Some(Bool::or(self.z3, &refs));
            }

            // /\ (conjunction).
            let cj = split_top_level_keyword(expr, "/\\");
            if cj.len() > 1 {
                let mut bools: Vec<Bool<'ctx>> = Vec::with_capacity(cj.len());
                for part in &cj {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    bools.push(self.translate_bool(trimmed)?);
                }
                let refs: Vec<&Bool<'_>> = bools.iter().collect();
                return Some(Bool::and(self.z3, &refs));
            }

            // Negation.
            if let Some(rest) = expr.strip_prefix('~') {
                let inner = self.translate_bool(rest.trim())?;
                return Some(inner.not());
            }
            if let Some(rest) = expr.strip_prefix("\\lnot") {
                let inner = self.translate_bool(rest.trim())?;
                return Some(inner.not());
            }

            // \A binder : body  -- expand if domain is a finite literal set.
            if let Some(stripped) = strip_quantifier_keyword(expr, "\\A") {
                return self.translate_quantifier(stripped, /*forall=*/ true);
            }
            if let Some(stripped) = strip_quantifier_keyword(expr, "\\E") {
                return self.translate_quantifier(stripped, /*forall=*/ false);
            }

            // Comparisons / membership / equality.
            if let Some(b) = self.translate_atomic_bool(expr) {
                return Some(b);
            }

            // Try constant folding via the outer evaluator.
            if !mentions_record_var(expr, self.var_name) {
                if let Ok(TlaValue::Bool(b)) = eval_expr(expr, self.eval_ctx) {
                    return Some(Bool::from_bool(self.z3, b));
                }
            }

            None
        }

        fn translate_quantifier(&self, body_text: &str, forall: bool) -> Option<Bool<'ctx>> {
            // body_text is `var \in domain : body`
            let colon = find_top_level_char(body_text, ':')?;
            let binder = body_text[..colon].trim();
            let body = body_text[colon + 1..].trim();
            let in_idx = find_top_level_keyword_index(binder, "\\in")?;
            let v = binder[..in_idx].trim();
            if !is_valid_identifier(v) {
                return None;
            }
            let domain_expr = binder[in_idx + 3..].trim();

            // Domain must be a literal set or range over constants we can
            // enumerate eagerly. Use the outer evaluator (no record-var
            // reference allowed).
            if mentions_record_var(domain_expr, self.var_name) {
                return None;
            }
            let domain_val = eval_expr(domain_expr, self.eval_ctx).ok()?;
            let domain_set = domain_val.as_set().ok()?;
            // Substitute v textually with each domain element's TLA+
            // literal form and translate. That keeps the translator simple
            // (no nested name lookup) and works for the shapes we care
            // about: small literal domains in puzzle predicates.
            let mut sub_bools: Vec<Bool<'ctx>> = Vec::with_capacity(domain_set.len());
            for elem in domain_set.iter() {
                let lit = tla_value_to_literal(elem)?;
                let substituted = substitute_identifier(body, v, &lit);
                sub_bools.push(self.translate_bool(&substituted)?);
            }
            if sub_bools.is_empty() {
                return Some(Bool::from_bool(self.z3, forall));
            }
            let refs: Vec<&Bool<'_>> = sub_bools.iter().collect();
            if forall {
                Some(Bool::and(self.z3, &refs))
            } else {
                Some(Bool::or(self.z3, &refs))
            }
        }

        /// Translate equality / membership / comparison atoms.
        fn translate_atomic_bool(&self, expr: &str) -> Option<Bool<'ctx>> {
            // Membership: lhs \in rhs
            if let Some(in_idx) = find_top_level_keyword_index(expr, "\\in") {
                let lhs = expr[..in_idx].trim();
                let rhs = expr[in_idx + 3..].trim();
                return self.translate_membership(lhs, rhs);
            }
            // Non-membership: lhs \notin rhs
            if let Some(notin_idx) = find_top_level_keyword_index(expr, "\\notin") {
                let lhs = expr[..notin_idx].trim();
                let rhs = expr[notin_idx + "\\notin".len()..].trim();
                return Some(self.translate_membership(lhs, rhs)?.not());
            }
            // Inequality: lhs # rhs / lhs /= rhs
            if let Some(idx) = find_top_level_op(expr, "#") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                return Some(self.translate_eq(lhs, rhs)?.not());
            }
            if let Some(idx) = find_top_level_str(expr, "/=") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 2..].trim());
                return Some(self.translate_eq(lhs, rhs)?.not());
            }

            // Comparisons: <=, >=, <, >. Detect <= / >= before < / >.
            for op in ["<=", ">="] {
                if let Some(idx) = find_top_level_str(expr, op) {
                    let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 2..].trim());
                    let l = self.translate_int(lhs)?;
                    let r = self.translate_int(rhs)?;
                    return Some(if op == "<=" { l.le(&r) } else { l.ge(&r) });
                }
            }
            for op in ['<', '>'] {
                if let Some(idx) = find_top_level_op(expr, &op.to_string()) {
                    let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                    let l = self.translate_int(lhs)?;
                    let r = self.translate_int(rhs)?;
                    return Some(if op == '<' { l.lt(&r) } else { l.gt(&r) });
                }
            }

            // Equality (handled last because = appears in many contexts).
            if let Some(idx) = find_top_level_op(expr, "=") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                return self.translate_eq(lhs, rhs);
            }

            None
        }

        fn translate_membership(&self, lhs: &str, rhs: &str) -> Option<Bool<'ctx>> {
            // lhs is var.f or an integer expression we can translate;
            // rhs must be a constant set we can enumerate.
            let lhs_field = self.lookup_field(lhs);

            // rhs as range a..b
            if let Some(dotdot) = rhs.find("..") {
                let lo_text = rhs[..dotdot].trim();
                let hi_text = rhs[dotdot + 2..].trim();
                let lo = const_eval_int(lo_text, self.eval_ctx)?;
                let hi = const_eval_int(hi_text, self.eval_ctx)?;
                let lo_e = Int::from_i64(self.z3, lo);
                let hi_e = Int::from_i64(self.z3, hi);
                if let Some((var, enc, _)) = lhs_field {
                    if let FieldEncoding::IntDomain { .. } = enc {
                        return Some(Bool::and(self.z3, &[&var.ge(&lo_e), &var.le(&hi_e)]));
                    }
                    return None;
                }
                // Fallback: lhs is an integer expression (e.g. var.f1 + var.f2).
                if let Some(int_expr) = self.translate_int(lhs) {
                    return Some(Bool::and(
                        self.z3,
                        &[&int_expr.ge(&lo_e), &int_expr.le(&hi_e)],
                    ));
                }
                return None;
            }

            // rhs as a set literal {e1, e2, ...}
            if let Some(inner) = rhs.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let parts = split_top_level_symbol(inner, ",");
                let (var, enc, _) = lhs_field?;
                let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(parts.len());
                for part in &parts {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let elem_val = eval_expr(trimmed, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem_val, enc)?;
                    eqs.push(var._eq(&Int::from_i64(self.z3, coded)));
                }
                if eqs.is_empty() {
                    return Some(Bool::from_bool(self.z3, false));
                }
                let refs: Vec<&Bool<'_>> = eqs.iter().collect();
                return Some(Bool::or(self.z3, &refs));
            }

            // rhs as a name evaluable to a finite set
            if !mentions_record_var(rhs, self.var_name) {
                let val = eval_expr(rhs, self.eval_ctx).ok()?;
                let set = val.as_set().ok()?;
                let (var, enc, _) = lhs_field?;
                let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(set.len());
                for elem in set.iter() {
                    let coded = encode_value_as_int(elem, enc)?;
                    eqs.push(var._eq(&Int::from_i64(self.z3, coded)));
                }
                if eqs.is_empty() {
                    return Some(Bool::from_bool(self.z3, false));
                }
                let refs: Vec<&Bool<'_>> = eqs.iter().collect();
                return Some(Bool::or(self.z3, &refs));
            }

            None
        }

        fn translate_eq(&self, lhs: &str, rhs: &str) -> Option<Bool<'ctx>> {
            // Both sides may be field projections or constants. Decide
            // whether to use Int equality (when both encodable as ints)
            // or treat as enum equality.
            let lhs_field = self.lookup_field(lhs);
            let rhs_field = self.lookup_field(rhs);

            match (lhs_field, rhs_field) {
                (Some((lv, lenc, _)), Some((rv, renc, _))) => {
                    // Field-to-field equality: only meaningful if both
                    // encodings agree. For two enum domains, codes only
                    // match for shared values, so reduce to enumerated
                    // pairs.
                    if matches!(lenc, FieldEncoding::IntDomain { .. })
                        && matches!(renc, FieldEncoding::IntDomain { .. })
                    {
                        return Some(lv._eq(&rv));
                    }
                    if let (
                        FieldEncoding::EnumDomain { values: lvals },
                        FieldEncoding::EnumDomain { values: rvals },
                    ) = (lenc, renc)
                    {
                        // Build OR of (lv == lcode AND rv == rcode) for
                        // every shared value.
                        let mut pairs: Vec<Bool<'_>> = Vec::new();
                        for (li, lval) in lvals.iter().enumerate() {
                            if let Some(ri) = rvals.iter().position(|v| v == lval) {
                                let l_eq = lv._eq(&Int::from_i64(self.z3, li as i64));
                                let r_eq = rv._eq(&Int::from_i64(self.z3, ri as i64));
                                pairs.push(Bool::and(self.z3, &[&l_eq, &r_eq]));
                            }
                        }
                        if pairs.is_empty() {
                            return Some(Bool::from_bool(self.z3, false));
                        }
                        let refs: Vec<&Bool<'_>> = pairs.iter().collect();
                        return Some(Bool::or(self.z3, &refs));
                    }
                    None
                }
                (Some((var, enc, _)), None) => {
                    if let FieldEncoding::IntDomain { .. } = enc {
                        if let Some(r) = self.translate_int(rhs) {
                            return Some(var._eq(&r));
                        }
                    }
                    let elem = eval_expr(rhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, Some((var, enc, _))) => {
                    if let FieldEncoding::IntDomain { .. } = enc {
                        if let Some(l) = self.translate_int(lhs) {
                            return Some(var._eq(&l));
                        }
                    }
                    let elem = eval_expr(lhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, None) => {
                    // Try integer arithmetic equality (e.g. `tup.a + tup.b = 7`).
                    if let (Some(l), Some(r)) = (self.translate_int(lhs), self.translate_int(rhs)) {
                        return Some(l._eq(&r));
                    }
                    // Constant equality — fold via outer evaluator.
                    if !mentions_record_var(lhs, self.var_name)
                        && !mentions_record_var(rhs, self.var_name)
                    {
                        let l = eval_expr(lhs, self.eval_ctx).ok()?;
                        let r = eval_expr(rhs, self.eval_ctx).ok()?;
                        return Some(Bool::from_bool(self.z3, l == r));
                    }
                    None
                }
            }
        }

        fn translate_int(&self, expr: &str) -> Option<Int<'ctx>> {
            let expr = strip_redundant_parens(expr.trim());

            // Field projection.
            if let Some((var, enc, _)) = self.lookup_field(expr) {
                if matches!(enc, FieldEncoding::IntDomain { .. }) {
                    return Some(var.clone());
                }
                return None;
            }

            // Integer literal.
            if let Ok(n) = expr.parse::<i64>() {
                return Some(Int::from_i64(self.z3, n));
            }

            // Negation literal.
            if let Some(rest) = expr.strip_prefix('-') {
                let r = self.translate_int(rest.trim())?;
                let zero = Int::from_i64(self.z3, 0);
                return Some(Int::sub(self.z3, &[&zero, &r]));
            }

            // Addition / subtraction (left-associative; split on the
            // last top-level + or -).
            if let Some((l_text, op, r_text)) = split_top_level_additive_local(expr) {
                let l = self.translate_int(&l_text)?;
                let r = self.translate_int(&r_text)?;
                return Some(match op {
                    '+' => Int::add(self.z3, &[&l, &r]),
                    '-' => Int::sub(self.z3, &[&l, &r]),
                    _ => unreachable!(),
                });
            }

            // Multiplication.
            if let Some((l_text, r_text)) = split_top_level_mul_local(expr) {
                let l = self.translate_int(&l_text)?;
                let r = self.translate_int(&r_text)?;
                return Some(Int::mul(self.z3, &[&l, &r]));
            }

            // Constant fold.
            if !mentions_record_var(expr, self.var_name) {
                if let Ok(TlaValue::Int(n)) = eval_expr(expr, self.eval_ctx) {
                    return Some(Int::from_i64(self.z3, n));
                }
            }

            None
        }

        fn lookup_field(&self, expr: &str) -> Option<(Int<'ctx>, &FieldEncoding, usize)> {
            let prefix = format!("{}.", self.var_name);
            let stripped = expr.strip_prefix(&prefix)?;
            // Field name must be a valid identifier and have nothing else.
            if !is_valid_identifier(stripped) {
                return None;
            }
            self.fields
                .get(stripped)
                .map(|(v, e, i)| (v.clone(), *e, *i))
        }
    }

    fn encode_value_as_int(value: &TlaValue, enc: &FieldEncoding) -> Option<i64> {
        match enc {
            FieldEncoding::IntDomain { values } => {
                let n = value.as_int().ok()?;
                if values.binary_search(&n).is_ok() {
                    Some(n)
                } else {
                    // Out-of-domain literal — equality with this element
                    // is statically false. Encode as a code that no
                    // domain assertion can take. Use any int outside the
                    // [lo, hi] range.
                    let lo = *values.first().unwrap();
                    Some(lo - 1)
                }
            }
            FieldEncoding::EnumDomain { values } => {
                if let Some(idx) = values.iter().position(|v| v == value) {
                    Some(idx as i64)
                } else {
                    // Out-of-domain literal — encode as a code outside
                    // the legal range so the domain constraint blocks it.
                    Some(values.len() as i64)
                }
            }
        }
    }

    fn const_eval_int(expr: &str, ctx: &EvalContext<'_>) -> Option<i64> {
        if let Ok(n) = expr.parse::<i64>() {
            return Some(n);
        }
        eval_expr(expr, ctx).ok().and_then(|v| v.as_int().ok())
    }

    fn mentions_record_var(expr: &str, var_name: &str) -> bool {
        // Simple identifier-boundary scan.
        let bytes = expr.as_bytes();
        let needle = var_name.as_bytes();
        if needle.is_empty() || bytes.len() < needle.len() {
            return false;
        }
        let mut i = 0;
        while i + needle.len() <= bytes.len() {
            if &bytes[i..i + needle.len()] == needle {
                let prev_ok =
                    i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                let next_idx = i + needle.len();
                let next_ok = next_idx == bytes.len()
                    || !(bytes[next_idx].is_ascii_alphanumeric() || bytes[next_idx] == b'_');
                if prev_ok && next_ok {
                    return true;
                }
            }
            i += 1;
        }
        false
    }

    /// Render a TlaValue as a TLA+ literal text (limited subset: Int,
    /// String, ModelValue, Bool). Returns None for unsupported variants.
    fn tla_value_to_literal(v: &TlaValue) -> Option<String> {
        match v {
            TlaValue::Int(n) => Some(n.to_string()),
            TlaValue::Bool(true) => Some("TRUE".to_string()),
            TlaValue::Bool(false) => Some("FALSE".to_string()),
            TlaValue::String(s) => Some(format!("\"{}\"", s)),
            TlaValue::ModelValue(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Replace standalone `name` with `replacement` in `source` (whole-word).
    fn substitute_identifier(source: &str, name: &str, replacement: &str) -> String {
        let bytes = source.as_bytes();
        let needle = name.as_bytes();
        let mut out = String::with_capacity(source.len());
        let mut i = 0;
        while i < bytes.len() {
            if i + needle.len() <= bytes.len() && &bytes[i..i + needle.len()] == needle {
                let prev_ok =
                    i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                let next_idx = i + needle.len();
                let next_ok = next_idx == bytes.len()
                    || !(bytes[next_idx].is_ascii_alphanumeric() || bytes[next_idx] == b'_');
                if prev_ok && next_ok {
                    out.push_str(replacement);
                    i += needle.len();
                    continue;
                }
            }
            // Push current char.
            let ch = source[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
        out
    }

    fn strip_redundant_parens(expr: &str) -> &str {
        let mut s = expr.trim();
        loop {
            if s.starts_with('(') && s.ends_with(')') && balanced_outer_paren(s) {
                s = s[1..s.len() - 1].trim();
            } else {
                break;
            }
        }
        s
    }

    fn balanced_outer_paren(s: &str) -> bool {
        let bytes = s.as_bytes();
        if bytes.len() < 2 || bytes[0] != b'(' {
            return false;
        }
        let mut depth = 0i32;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            if depth == 0 && i + 1 < bytes.len() {
                return false;
            }
        }
        depth == 0
    }

    fn strip_quantifier_keyword<'b>(expr: &'b str, kw: &str) -> Option<&'b str> {
        let stripped = expr.strip_prefix(kw)?;
        if !stripped.starts_with(' ') && !stripped.starts_with('\t') {
            return None;
        }
        Some(stripped.trim_start())
    }

    fn split_two(expr: &str, kw: &str) -> Option<(String, String)> {
        let parts = split_top_level_symbol(expr, kw);
        if parts.len() == 2 {
            Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
        } else {
            None
        }
    }

    fn find_top_level_op(expr: &str, op: &str) -> Option<usize> {
        // Find an op character that isn't part of a longer operator
        // (e.g. don't match '=' inside '<=', '>=', '/=' or '|->').
        let bytes = expr.as_bytes();
        let needle = op.as_bytes();
        if needle.is_empty() {
            return None;
        }
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut brace = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        let mut i = 0;
        while i + needle.len() <= bytes.len() {
            let b = bytes[i];
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            match b {
                b'"' => {
                    in_string = true;
                }
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => bracket += 1,
                b']' => bracket -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
            if paren == 0 && bracket == 0 && brace == 0 && &bytes[i..i + needle.len()] == needle {
                let prev = if i > 0 { bytes[i - 1] } else { 0 };
                let next = if i + needle.len() < bytes.len() {
                    bytes[i + needle.len()]
                } else {
                    0
                };
                let bad_prev = matches!(prev, b'<' | b'>' | b'/' | b'!' | b'=' | b':' | b'|');
                let bad_next = matches!(next, b'=' | b'>' | b'<' | b'|');
                if !bad_prev && !bad_next && needle == b"=" {
                    return Some(i);
                }
                if needle != b"=" && !bad_prev && !bad_next {
                    return Some(i);
                }
            }
            i += 1;
        }
        None
    }

    fn find_top_level_str(expr: &str, op: &str) -> Option<usize> {
        let bytes = expr.as_bytes();
        let needle = op.as_bytes();
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut brace = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        let mut i = 0;
        while i + needle.len() <= bytes.len() {
            let b = bytes[i];
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => bracket += 1,
                b']' => bracket -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
            if paren == 0 && bracket == 0 && brace == 0 && &bytes[i..i + needle.len()] == needle {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn split_top_level_additive_local(expr: &str) -> Option<(String, char, String)> {
        // Find the rightmost top-level + or - that isn't a unary prefix
        // and isn't part of "..".
        let bytes = expr.as_bytes();
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut brace = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        let mut last: Option<(usize, char)> = None;
        for (i, &b) in bytes.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => bracket += 1,
                b']' => bracket -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
            if paren == 0 && bracket == 0 && brace == 0 && (b == b'+' || b == b'-') {
                // Skip unary at start.
                if i == 0 {
                    continue;
                }
                let prev = bytes[i - 1];
                if matches!(
                    prev,
                    b'+' | b'-'
                        | b'*'
                        | b'/'
                        | b'<'
                        | b'>'
                        | b'='
                        | b'('
                        | b'['
                        | b'{'
                        | b','
                        | b':'
                ) || prev == b' '
                    && i > 1
                    && matches!(
                        bytes[i - 2],
                        b'+' | b'-'
                            | b'*'
                            | b'/'
                            | b'<'
                            | b'>'
                            | b'='
                            | b'('
                            | b'['
                            | b'{'
                            | b','
                            | b':'
                    )
                {
                    // Likely unary or part of e.g. "f(- x)"; skip.
                    if i > 1 && bytes[i - 1].is_ascii_whitespace() {
                        // Could still be binary if previous non-space is operand-like.
                        let mut j = i;
                        while j > 0 && bytes[j - 1].is_ascii_whitespace() {
                            j -= 1;
                        }
                        if j > 0
                            && (bytes[j - 1].is_ascii_alphanumeric()
                                || bytes[j - 1] == b')'
                                || bytes[j - 1] == b']')
                        {
                            // Binary case.
                            last = Some((i, b as char));
                        }
                        continue;
                    }
                    continue;
                }
                // Don't match '..' as range delimiter.
                if b == b'.'
                    || (i + 1 < bytes.len() && (bytes[i + 1] == b'.' || bytes[i + 1] == b'>'))
                {
                    continue;
                }
                last = Some((i, b as char));
            }
        }
        let (idx, op) = last?;
        Some((
            expr[..idx].trim().to_string(),
            op,
            expr[idx + 1..].trim().to_string(),
        ))
    }

    fn split_top_level_mul_local(expr: &str) -> Option<(String, String)> {
        let bytes = expr.as_bytes();
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut brace = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (i, &b) in bytes.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => bracket += 1,
                b']' => bracket -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
            if paren == 0 && bracket == 0 && brace == 0 && b == b'*' && i > 0 {
                return Some((
                    expr[..i].trim().to_string(),
                    expr[i + 1..].trim().to_string(),
                ));
            }
        }
        None
    }

    // ========================================================================
    // T5.1 / T5.2 — Sequence-set Init enumeration with permutation support
    // ========================================================================

    /// Encoding for a sequence-set translator: the bound variable is a
    /// length-`n` sequence whose elements come from `range`. We encode it
    /// as `n` Z3 Int variables (one per position), each constrained to the
    /// range's enum codes.
    struct SeqTranslator<'ctx, 'a> {
        z3: &'ctx Context,
        var_name: &'a str,
        seq_len: usize,
        /// Z3 Int variable for each position 1..=n (vars[i] holds position i+1).
        position_vars: Vec<Int<'ctx>>,
        /// Range encoding (always EnumDomain for now — the translator
        /// reuses the existing FieldEncoding machinery to share code with
        /// the record-set path).
        range_enc: FieldEncoding,
        eval_ctx: &'a EvalContext<'a>,
    }

    pub(super) fn try_symbolic_function_set_enumerate_z3(
        pred_text: &str,
        var_name: &str,
        seq_len: usize,
        range: &[TlaValue],
        ctx: &EvalContext<'_>,
    ) -> Option<Vec<TlaValue>> {
        if seq_len == 0 {
            // Empty sequence — only one possible sequence, the empty one;
            // predicate must be evaluable as a constant. Skip and let
            // brute-force handle it.
            return None;
        }
        if range.is_empty() {
            // Empty range, non-empty domain → no functions/sequences.
            return Some(Vec::new());
        }

        // Build the range encoding. For sequences we always use enum
        // coding so that strings / model values / integers are all handled
        // uniformly via integer codes.
        let range_enc = build_range_encoding(range)?;

        let z3_cfg = Config::new();
        let z3_ctx = Context::new(&z3_cfg);
        let solver = Solver::new(&z3_ctx);

        // Per-position Z3 Int variables.
        let mut position_vars: Vec<Int<'_>> = Vec::with_capacity(seq_len);
        for i in 1..=seq_len {
            let v = Int::new_const(&z3_ctx, format!("{}_{}", var_name, i));
            assert_domain_constraint(&z3_ctx, &solver, &v, &range_enc);
            position_vars.push(v);
        }

        let translator = SeqTranslator {
            z3: &z3_ctx,
            var_name,
            seq_len,
            position_vars,
            range_enc,
            eval_ctx: ctx,
        };

        // T5.2 — detect explicit permutation predicates and emit Z3
        // Distinct as a structural shortcut. The full predicate is also
        // translated; the Distinct is added as an additional assertion.
        let pred_z3 = translator.translate_bool(pred_text)?;
        solver.assert(&pred_z3);

        // Extra: if `seq_len == range.len()` AND the predicate textually
        // contains a "permutation indicator" (chained `\ {var[i], ...}` or
        // pairwise `var[i] # var[j]`), assert Distinct over all positions.
        // This is sound (it's implied by the predicate) and dramatically
        // shrinks the search.
        if seq_len == translator.range_size()
            && contains_permutation_indicator(pred_text, var_name, seq_len)
        {
            let var_refs: Vec<&Int<'_>> = translator.position_vars.iter().collect();
            let distinct = z3::ast::Ast::distinct(&z3_ctx, &var_refs);
            solver.assert(&distinct);
        }

        // Block-and-resolve enumeration.
        let mut results: Vec<TlaValue> = Vec::new();
        loop {
            match solver.check() {
                SatResult::Sat => {}
                SatResult::Unsat => break,
                SatResult::Unknown => return None,
            }

            if results.len() >= SYMBOLIC_ENUM_HARD_CAP {
                return None;
            }

            let model = solver.get_model()?;
            let mut elems: Vec<TlaValue> = Vec::with_capacity(seq_len);
            let mut block_disjuncts: Vec<Bool<'_>> = Vec::with_capacity(seq_len);

            for v in &translator.position_vars {
                let code = model.eval(v, true)?.as_i64()?;
                let value = match &translator.range_enc {
                    FieldEncoding::IntDomain { .. } => TlaValue::Int(code),
                    FieldEncoding::EnumDomain { values } => {
                        if code < 0 || (code as usize) >= values.len() {
                            return None;
                        }
                        values[code as usize].clone()
                    }
                };
                elems.push(value);
                let lit = Int::from_i64(&z3_ctx, code);
                block_disjuncts.push(v._eq(&lit).not());
            }

            results.push(crate::tla::symbolic_init::build_seq(elems));

            let refs: Vec<&Bool<'_>> = block_disjuncts.iter().collect();
            let block = Bool::or(&z3_ctx, &refs);
            solver.assert(&block);
        }

        Some(results)
    }

    fn build_range_encoding(range: &[TlaValue]) -> Option<FieldEncoding> {
        let all_ints = range.iter().all(|v| matches!(v, TlaValue::Int(_)));
        if all_ints {
            let mut ints: Vec<i64> = range.iter().map(|v| v.as_int().unwrap()).collect();
            ints.sort_unstable();
            ints.dedup();
            return Some(FieldEncoding::IntDomain { values: ints });
        }
        let all_codable = range.iter().all(|v| {
            matches!(
                v,
                TlaValue::Int(_)
                    | TlaValue::Bool(_)
                    | TlaValue::String(_)
                    | TlaValue::ModelValue(_)
            )
        });
        if !all_codable {
            return None;
        }
        let mut deduped: Vec<TlaValue> = Vec::new();
        for v in range {
            if !deduped.contains(v) {
                deduped.push(v.clone());
            }
        }
        Some(FieldEncoding::EnumDomain { values: deduped })
    }

    /// Cheap textual scan for permutation indicators. Conservative — only
    /// returns true if it sees one of the canonical Einstein-shape patterns.
    /// False negatives are fine (we just miss the Distinct shortcut and
    /// fall back to pairwise inequalities expressed in the predicate).
    fn contains_permutation_indicator(pred: &str, var_name: &str, seq_len: usize) -> bool {
        // Pattern 1: `var[i] \in S \ {var[j], ...}` — Einstein's
        // distinctness clauses.
        // Pattern 2: chained `var[i] # var[j]` for all (i,j) pairs.
        let needle1 = format!("{}[", var_name);
        if !pred.contains(&needle1) {
            return false;
        }
        // Cheap heuristic: look for `\ {var[` (set-difference with var-indexed
        // singletons), which only appears in distinctness chains.
        if pred.contains("\\ {") || pred.contains("\\{") {
            // crude count of how often we see `var[` inside set-difference.
            let var_index_marker = format!("{}[", var_name);
            // If the pred mentions n-1 distinct var[k] indices in difference
            // contexts, it's almost certainly a distinctness chain.
            let occurrences = pred.matches(var_index_marker.as_str()).count();
            if occurrences >= seq_len {
                return true;
            }
        }
        // Pairwise: at least n*(n-1)/2 `#` operators between var[i]/var[j].
        let hash_count = pred.matches('#').count();
        let needed_pairs = seq_len * (seq_len - 1) / 2;
        if hash_count >= needed_pairs && hash_count >= 1 {
            // Rough heuristic only. Even if wrong, asserting Distinct is
            // unsound only if the predicate didn't actually require
            // distinctness — we add an extra constraint that filters out
            // legal solutions. To stay safe we additionally require that
            // the seq_len equals range_size (already checked at call site).
            // Combined with seq_len == range_size, Distinct becomes a
            // permutation requirement; if the user's predicate doesn't
            // require this, we'd miss solutions. Be more conservative:
            // require that the variable name appears on both sides of every
            // `#`. We approximate by counting `var[..] # var[..]` patterns.
            let pat = format!("{}[", var_name);
            let combined_needle = format!("{} # {}", pat, pat).replace("{}[", &pat);
            // Cheap: count adjacent occurrences via splits — best-effort.
            let _ = combined_needle;
            // Be conservative — only emit Distinct if we ALSO see the
            // set-difference pattern above. Otherwise return false here.
            return false;
        }
        false
    }

    impl<'ctx, 'a> SeqTranslator<'ctx, 'a> {
        fn range_size(&self) -> usize {
            match &self.range_enc {
                FieldEncoding::IntDomain { values } => values.len(),
                FieldEncoding::EnumDomain { values } => values.len(),
            }
        }

        /// Translate a TLA+ boolean expression where `var_name[i]` denotes
        /// the i-th sequence element.
        fn translate_bool(&self, expr: &str) -> Option<Bool<'ctx>> {
            let expr = strip_redundant_parens(expr.trim());

            if expr.eq_ignore_ascii_case("TRUE") {
                return Some(Bool::from_bool(self.z3, true));
            }
            if expr.eq_ignore_ascii_case("FALSE") {
                return Some(Bool::from_bool(self.z3, false));
            }

            if let Some(parts) = split_two(expr, "<=>") {
                let a = self.translate_bool(&parts.0)?;
                let b = self.translate_bool(&parts.1)?;
                return Some(a.iff(&b));
            }

            if let Some(parts) = split_two(expr, "=>") {
                let a = self.translate_bool(&parts.0)?;
                let b = self.translate_bool(&parts.1)?;
                return Some(a.implies(&b));
            }

            let dj = split_top_level_keyword(expr, "\\/");
            if dj.len() > 1 {
                let mut bools: Vec<Bool<'ctx>> = Vec::with_capacity(dj.len());
                for part in &dj {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    bools.push(self.translate_bool(trimmed)?);
                }
                let refs: Vec<&Bool<'_>> = bools.iter().collect();
                return Some(Bool::or(self.z3, &refs));
            }

            let cj = split_top_level_keyword(expr, "/\\");
            if cj.len() > 1 {
                let mut bools: Vec<Bool<'ctx>> = Vec::with_capacity(cj.len());
                for part in &cj {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    bools.push(self.translate_bool(trimmed)?);
                }
                let refs: Vec<&Bool<'_>> = bools.iter().collect();
                return Some(Bool::and(self.z3, &refs));
            }

            if let Some(rest) = expr.strip_prefix('~') {
                let inner = self.translate_bool(rest.trim())?;
                return Some(inner.not());
            }
            if let Some(rest) = expr.strip_prefix("\\lnot") {
                let inner = self.translate_bool(rest.trim())?;
                return Some(inner.not());
            }

            // \A / \E binders over finite literal domains.
            if let Some(stripped) = strip_quantifier_keyword(expr, "\\A") {
                return self.translate_quantifier(stripped, true);
            }
            if let Some(stripped) = strip_quantifier_keyword(expr, "\\E") {
                return self.translate_quantifier(stripped, false);
            }

            if let Some(b) = self.translate_atomic_bool(expr) {
                return Some(b);
            }

            // Constant fold.
            if !mentions_record_var(expr, self.var_name) {
                if let Ok(TlaValue::Bool(b)) = crate::tla::eval_expr(expr, self.eval_ctx) {
                    return Some(Bool::from_bool(self.z3, b));
                }
            }

            None
        }

        fn translate_quantifier(&self, body_text: &str, forall: bool) -> Option<Bool<'ctx>> {
            let colon = find_top_level_char(body_text, ':')?;
            let binder = body_text[..colon].trim();
            let body = body_text[colon + 1..].trim();
            let in_idx = find_top_level_keyword_index(binder, "\\in")?;
            let v = binder[..in_idx].trim();
            if !is_valid_identifier(v) {
                return None;
            }
            let domain_expr = binder[in_idx + 3..].trim();
            if mentions_record_var(domain_expr, self.var_name) {
                return None;
            }
            let domain_val = crate::tla::eval_expr(domain_expr, self.eval_ctx).ok()?;
            let domain_set = domain_val.as_set().ok()?;
            let mut sub_bools: Vec<Bool<'ctx>> = Vec::with_capacity(domain_set.len());
            for elem in domain_set.iter() {
                let lit = tla_value_to_literal(elem)?;
                let substituted = substitute_identifier(body, v, &lit);
                sub_bools.push(self.translate_bool(&substituted)?);
            }
            if sub_bools.is_empty() {
                return Some(Bool::from_bool(self.z3, forall));
            }
            let refs: Vec<&Bool<'_>> = sub_bools.iter().collect();
            if forall {
                Some(Bool::and(self.z3, &refs))
            } else {
                Some(Bool::or(self.z3, &refs))
            }
        }

        fn translate_atomic_bool(&self, expr: &str) -> Option<Bool<'ctx>> {
            // Membership: lhs \in rhs
            if let Some(in_idx) = find_top_level_keyword_index(expr, "\\in") {
                let lhs = expr[..in_idx].trim();
                let rhs = expr[in_idx + 3..].trim();
                return self.translate_membership(lhs, rhs);
            }
            // Non-membership: lhs \notin rhs
            if let Some(notin_idx) = find_top_level_keyword_index(expr, "\\notin") {
                let lhs = expr[..notin_idx].trim();
                let rhs = expr[notin_idx + "\\notin".len()..].trim();
                return Some(self.translate_membership(lhs, rhs)?.not());
            }

            if let Some(idx) = find_top_level_op(expr, "#") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                return Some(self.translate_eq(lhs, rhs)?.not());
            }
            if let Some(idx) = find_top_level_str(expr, "/=") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 2..].trim());
                return Some(self.translate_eq(lhs, rhs)?.not());
            }

            for op in ["<=", ">="] {
                if let Some(idx) = find_top_level_str(expr, op) {
                    let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 2..].trim());
                    let l = self.translate_int(lhs)?;
                    let r = self.translate_int(rhs)?;
                    return Some(if op == "<=" { l.le(&r) } else { l.ge(&r) });
                }
            }
            for op in ['<', '>'] {
                if let Some(idx) = find_top_level_op(expr, &op.to_string()) {
                    let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                    let l = self.translate_int(lhs)?;
                    let r = self.translate_int(rhs)?;
                    return Some(if op == '<' { l.lt(&r) } else { l.gt(&r) });
                }
            }

            if let Some(idx) = find_top_level_op(expr, "=") {
                let (lhs, rhs) = (expr[..idx].trim(), expr[idx + 1..].trim());
                return self.translate_eq(lhs, rhs);
            }

            None
        }

        fn translate_membership(&self, lhs: &str, rhs: &str) -> Option<Bool<'ctx>> {
            // lhs is var[i] (returns the position's symbolic Int).
            let lhs_pos = self.lookup_position(lhs);

            // rhs as range a..b
            if let Some(dotdot) = rhs.find("..") {
                let lo_text = rhs[..dotdot].trim();
                let hi_text = rhs[dotdot + 2..].trim();
                let lo = const_eval_int(lo_text, self.eval_ctx)?;
                let hi = const_eval_int(hi_text, self.eval_ctx)?;
                let lo_e = Int::from_i64(self.z3, lo);
                let hi_e = Int::from_i64(self.z3, hi);
                if let Some(var) = lhs_pos {
                    if matches!(self.range_enc, FieldEncoding::IntDomain { .. }) {
                        return Some(Bool::and(self.z3, &[&var.ge(&lo_e), &var.le(&hi_e)]));
                    }
                    return None;
                }
                return None;
            }

            // rhs as a set literal {e1, e2, ...} OR as a set difference
            // expression `S \ {var[i], ...}` (T5.2 Einstein distinctness pattern).
            // Detect set difference at top level.
            if let Some(diff_idx) = find_top_level_set_diff(rhs) {
                let base_text = rhs[..diff_idx].trim();
                let removed_text = rhs[diff_idx + 1..].trim_start();
                // `removed_text` must be a brace literal whose contents are
                // var[k] references (possibly mixed with constants).
                if let Some(removed_inner) = removed_text
                    .strip_prefix('{')
                    .and_then(|s| s.strip_suffix('}'))
                {
                    // Translate as: lhs \in base AND for each removed term,
                    // lhs # removed_term.
                    let var = lhs_pos?;
                    let mut clauses: Vec<Bool<'ctx>> = Vec::new();

                    // Membership in `base` (a constant set).
                    let base_clause = self.constant_set_membership(&var, base_text)?;
                    clauses.push(base_clause);

                    let removed_parts = split_top_level_symbol(removed_inner, ",");
                    for r in &removed_parts {
                        let r = r.trim();
                        if r.is_empty() {
                            continue;
                        }
                        // r may be var[k] (another position) or a constant.
                        if let Some(other) = self.lookup_position(r) {
                            clauses.push(var._eq(&other).not());
                        } else if let Ok(elem_val) =
                            crate::tla::eval_expr(r, self.eval_ctx)
                        {
                            let coded = encode_value_as_int(&elem_val, &self.range_enc)?;
                            clauses.push(var._eq(&Int::from_i64(self.z3, coded)).not());
                        } else {
                            return None;
                        }
                    }
                    let refs: Vec<&Bool<'_>> = clauses.iter().collect();
                    return Some(Bool::and(self.z3, &refs));
                }
                return None;
            }

            // rhs as a brace-set literal {e1, e2, ...}
            if let Some(inner) = rhs.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let var = lhs_pos?;
                return self.brace_set_membership(&var, inner);
            }

            // rhs as a name evaluable to a finite set.
            if !mentions_record_var(rhs, self.var_name) {
                let var = lhs_pos?;
                return self.constant_set_membership(&var, rhs);
            }

            None
        }

        fn constant_set_membership(
            &self,
            var: &Int<'ctx>,
            set_text: &str,
        ) -> Option<Bool<'ctx>> {
            let val = crate::tla::eval_expr(set_text, self.eval_ctx).ok()?;
            let set = val.as_set().ok()?;
            let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(set.len());
            for elem in set.iter() {
                let coded = encode_value_as_int(elem, &self.range_enc)?;
                eqs.push(var._eq(&Int::from_i64(self.z3, coded)));
            }
            if eqs.is_empty() {
                return Some(Bool::from_bool(self.z3, false));
            }
            let refs: Vec<&Bool<'_>> = eqs.iter().collect();
            Some(Bool::or(self.z3, &refs))
        }

        fn brace_set_membership(
            &self,
            var: &Int<'ctx>,
            inner: &str,
        ) -> Option<Bool<'ctx>> {
            let parts = split_top_level_symbol(inner, ",");
            let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(parts.len());
            for part in &parts {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Could be another var[k] or a constant.
                if let Some(other) = self.lookup_position(trimmed) {
                    eqs.push(var._eq(&other));
                    continue;
                }
                let elem_val = crate::tla::eval_expr(trimmed, self.eval_ctx).ok()?;
                let coded = encode_value_as_int(&elem_val, &self.range_enc)?;
                eqs.push(var._eq(&Int::from_i64(self.z3, coded)));
            }
            if eqs.is_empty() {
                return Some(Bool::from_bool(self.z3, false));
            }
            let refs: Vec<&Bool<'_>> = eqs.iter().collect();
            Some(Bool::or(self.z3, &refs))
        }

        fn translate_eq(&self, lhs: &str, rhs: &str) -> Option<Bool<'ctx>> {
            let lhs_pos = self.lookup_position(lhs);
            let rhs_pos = self.lookup_position(rhs);
            match (lhs_pos, rhs_pos) {
                (Some(l), Some(r)) => Some(l._eq(&r)),
                (Some(var), None) => {
                    if matches!(self.range_enc, FieldEncoding::IntDomain { .. }) {
                        if let Some(r) = self.translate_int(rhs) {
                            return Some(var._eq(&r));
                        }
                    }
                    let elem = crate::tla::eval_expr(rhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, &self.range_enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, Some(var)) => {
                    if matches!(self.range_enc, FieldEncoding::IntDomain { .. }) {
                        if let Some(l) = self.translate_int(lhs) {
                            return Some(var._eq(&l));
                        }
                    }
                    let elem = crate::tla::eval_expr(lhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, &self.range_enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, None) => {
                    if let (Some(l), Some(r)) =
                        (self.translate_int(lhs), self.translate_int(rhs))
                    {
                        return Some(l._eq(&r));
                    }
                    if !mentions_record_var(lhs, self.var_name)
                        && !mentions_record_var(rhs, self.var_name)
                    {
                        let l = crate::tla::eval_expr(lhs, self.eval_ctx).ok()?;
                        let r = crate::tla::eval_expr(rhs, self.eval_ctx).ok()?;
                        return Some(Bool::from_bool(self.z3, l == r));
                    }
                    None
                }
            }
        }

        fn translate_int(&self, expr: &str) -> Option<Int<'ctx>> {
            let expr = strip_redundant_parens(expr.trim());

            if let Some(var) = self.lookup_position(expr) {
                if matches!(self.range_enc, FieldEncoding::IntDomain { .. }) {
                    return Some(var);
                }
                return None;
            }

            if let Ok(n) = expr.parse::<i64>() {
                return Some(Int::from_i64(self.z3, n));
            }

            if let Some(rest) = expr.strip_prefix('-') {
                let r = self.translate_int(rest.trim())?;
                let zero = Int::from_i64(self.z3, 0);
                return Some(Int::sub(self.z3, &[&zero, &r]));
            }

            if let Some((l_text, op, r_text)) = split_top_level_additive_local(expr) {
                let l = self.translate_int(&l_text)?;
                let r = self.translate_int(&r_text)?;
                return Some(match op {
                    '+' => Int::add(self.z3, &[&l, &r]),
                    '-' => Int::sub(self.z3, &[&l, &r]),
                    _ => unreachable!(),
                });
            }

            if let Some((l_text, r_text)) = split_top_level_mul_local(expr) {
                let l = self.translate_int(&l_text)?;
                let r = self.translate_int(&r_text)?;
                return Some(Int::mul(self.z3, &[&l, &r]));
            }

            if !mentions_record_var(expr, self.var_name) {
                if let Ok(TlaValue::Int(n)) = crate::tla::eval_expr(expr, self.eval_ctx) {
                    return Some(Int::from_i64(self.z3, n));
                }
            }

            None
        }

        /// Recognize `var_name[K]` where K is a positive integer literal in
        /// 1..=seq_len. Returns the corresponding position variable.
        fn lookup_position(&self, expr: &str) -> Option<Int<'ctx>> {
            let expr = strip_redundant_parens(expr.trim());
            let prefix = format!("{}[", self.var_name);
            let stripped = expr.strip_prefix(&prefix)?;
            let inner = stripped.strip_suffix(']')?;
            // Inner could be `i` (literal) or `i + k`, etc. Try const-eval.
            let idx = if let Ok(n) = inner.trim().parse::<i64>() {
                n
            } else {
                // try as const arithmetic against the eval context.
                let v = crate::tla::eval_expr(inner.trim(), self.eval_ctx).ok()?;
                v.as_int().ok()?
            };
            if idx < 1 || (idx as usize) > self.seq_len {
                return None;
            }
            Some(self.position_vars[(idx as usize) - 1].clone())
        }
    }

    /// Find the top-level `\` operator (set difference) in `expr`. The
    /// `\` operator must be followed by a brace `{` (set literal) or a
    /// space + identifier; we use a simple rule: it's at top level (not
    /// inside parens/braces/brackets), it is preceded by a space or
    /// closing bracket, and the next non-space character is `{` (we only
    /// care about `S \ {...}` shapes for permutation distinctness).
    fn find_top_level_set_diff(expr: &str) -> Option<usize> {
        let bytes = expr.as_bytes();
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut brace = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (i, &b) in bytes.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => bracket += 1,
                b']' => bracket -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
            if paren == 0 && bracket == 0 && brace == 0 && b == b'\\' {
                // Check this isn't part of `\in`, `\notin`, `\A`, `\E`,
                // `\/`, `\union`, `\subseteq`, `\lnot`, etc. The set
                // difference operator is just `\` followed by a space or `{`.
                let after = if i + 1 < bytes.len() {
                    bytes[i + 1]
                } else {
                    return None;
                };
                if after == b' ' || after == b'\t' || after == b'{' {
                    // Confirm next non-space is `{`.
                    let mut j = i + 1;
                    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b'{' {
                        // Make sure the `\` is preceded by a value (not the
                        // start of expression).
                        if i > 0 {
                            return Some(i);
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(all(test, feature = "symbolic-init"))]
mod tests {
    use super::*;
    use crate::tla::eval::EvalContext;
    use crate::tla::module::TlaModuleInstance;
    use std::collections::BTreeMap;

    fn make_ctx<'a>(
        state: &'a crate::tla::value::TlaState,
        defs: &'a BTreeMap<String, crate::tla::module::TlaDefinition>,
        instances: &'a BTreeMap<String, TlaModuleInstance>,
    ) -> EvalContext<'a> {
        EvalContext::with_definitions_and_instances(state, defs, instances)
    }

    #[test]
    fn empty_field_specs_returns_none() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let result = try_symbolic_record_set_enumerate("TRUE", "tup", &[], &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn single_int_field_with_true_predicate_enumerates_full_domain() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let fields = vec![(
            "x".to_string(),
            (0..5).map(TlaValue::Int).collect::<Vec<_>>(),
        )];
        let result = try_symbolic_record_set_enumerate("TRUE", "tup", &fields, &ctx);
        let result = result.expect("symbolic enumeration should succeed");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn two_int_fields_with_sum_constraint_matches_brute_force() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let fields = vec![
            (
                "a".to_string(),
                (0..10).map(TlaValue::Int).collect::<Vec<_>>(),
            ),
            (
                "b".to_string(),
                (0..10).map(TlaValue::Int).collect::<Vec<_>>(),
            ),
        ];
        let pred = "tup.a + tup.b = 7";
        let result = try_symbolic_record_set_enumerate(pred, "tup", &fields, &ctx).expect("ok");
        // Brute force: 8 pairs (0,7) (1,6) ... (7,0).
        assert_eq!(result.len(), 8);
        // Verify each record satisfies the constraint.
        for rec in &result {
            let r = rec.as_record().unwrap();
            let a = r.get("a").unwrap().as_int().unwrap();
            let b = r.get("b").unwrap().as_int().unwrap();
            assert_eq!(a + b, 7);
        }
    }

    #[test]
    fn distinctness_constraint_three_fields() {
        // Mini-Einstein: 3 fields, each in 1..3, all distinct.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let dom: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let fields = vec![
            ("a".to_string(), dom.clone()),
            ("b".to_string(), dom.clone()),
            ("c".to_string(), dom.clone()),
        ];
        let pred = "tup.a # tup.b /\\ tup.a # tup.c /\\ tup.b # tup.c";
        let result = try_symbolic_record_set_enumerate(pred, "tup", &fields, &ctx).expect("ok");
        // 3! = 6 permutations.
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn enum_domain_with_equality() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let colors: Vec<TlaValue> = ["RED", "GREEN", "BLUE"]
            .iter()
            .map(|s| TlaValue::ModelValue((*s).to_string()))
            .collect();
        let fields = vec![
            ("color".to_string(), colors.clone()),
            ("backup".to_string(), colors.clone()),
        ];
        let pred = "tup.color = RED";
        let result = try_symbolic_record_set_enumerate(pred, "tup", &fields, &ctx).expect("ok");
        // color must be RED; backup ranges over 3 colors → 3 records.
        assert_eq!(result.len(), 3);
        for rec in &result {
            let r = rec.as_record().unwrap();
            assert_eq!(
                r.get("color").unwrap(),
                &TlaValue::ModelValue("RED".to_string())
            );
        }
    }

    #[test]
    fn unsupported_predicate_returns_none() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let fields = vec![(
            "x".to_string(),
            (0..3).map(TlaValue::Int).collect::<Vec<_>>(),
        )];
        // Cardinality of a synthesized set isn't part of the supported subset.
        let pred = "Cardinality({tup.x, 1, 2}) = 3";
        let result = try_symbolic_record_set_enumerate(pred, "tup", &fields, &ctx);
        assert!(result.is_none(), "unsupported predicate must fall back");
    }

    // ====================================================================
    // T5.1 — sequence-set Init translator tests
    // ====================================================================

    #[test]
    fn function_set_int_range_with_position_filter() {
        // Sequence p of length 3 over {1,2,3,4} where p[2] = 2.
        // Expected: 4 * 1 * 4 = 16 sequences.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=4).map(TlaValue::Int).collect();
        let result = try_symbolic_function_set_enumerate("p[2] = 2", "p", 3, &range, &ctx)
            .expect("symbolic enumeration succeeds");
        assert_eq!(result.len(), 16);
        for seq in &result {
            let s = seq.as_seq().unwrap();
            assert_eq!(s.len(), 3);
            assert_eq!(s[1], TlaValue::Int(2));
        }
    }

    #[test]
    fn function_set_enum_range_permutation_distinctness() {
        // Mini-Einstein: sequence of length 3 over {RED, GREEN, BLUE},
        // distinctness constraint via chained set differences. Should
        // yield 3! = 6 permutations.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = ["RED", "GREEN", "BLUE"]
            .iter()
            .map(|s| TlaValue::ModelValue((*s).to_string()))
            .collect();
        let pred = "/\\ p[2] \\in {RED, GREEN, BLUE} \\ {p[1]} \
                    /\\ p[3] \\in {RED, GREEN, BLUE} \\ {p[1], p[2]}";
        let result = try_symbolic_function_set_enumerate(pred, "p", 3, &range, &ctx)
            .expect("symbolic enumeration succeeds");
        assert_eq!(result.len(), 6);
        // Verify each is a permutation (all elements distinct).
        for seq in &result {
            let s = seq.as_seq().unwrap();
            let mut sorted: Vec<&TlaValue> = s.iter().collect();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), s.len(), "permutation must be distinct");
        }
    }

    #[test]
    fn function_set_with_constant_filter() {
        // Permutation of {1,2,3} with p[1] = 2 should give 2 results.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let pred = "/\\ p[2] \\in {1,2,3} \\ {p[1]} \
                    /\\ p[3] \\in {1,2,3} \\ {p[1], p[2]} \
                    /\\ p[1] = 2";
        let result = try_symbolic_function_set_enumerate(pred, "p", 3, &range, &ctx)
            .expect("symbolic enumeration succeeds");
        assert_eq!(result.len(), 2);
        for seq in &result {
            let s = seq.as_seq().unwrap();
            assert_eq!(s[0], TlaValue::Int(2));
        }
    }

    #[test]
    fn function_set_permutation_string_range_matches_brute_force() {
        // Brute-force-comparable size: 4! = 24 permutations of a string set.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = ["a", "b", "c", "d"]
            .iter()
            .map(|s| TlaValue::String((*s).to_string()))
            .collect();
        let pred = "/\\ p[2] \\in {\"a\",\"b\",\"c\",\"d\"} \\ {p[1]} \
                    /\\ p[3] \\in {\"a\",\"b\",\"c\",\"d\"} \\ {p[1], p[2]} \
                    /\\ p[4] \\in {\"a\",\"b\",\"c\",\"d\"} \\ {p[1], p[2], p[3]}";
        let result = try_symbolic_function_set_enumerate(pred, "p", 4, &range, &ctx)
            .expect("symbolic enumeration succeeds");
        // Brute force comparison: 4! = 24.
        assert_eq!(result.len(), 24);
        // Each must be a permutation.
        let mut all: std::collections::BTreeSet<TlaValue> = std::collections::BTreeSet::new();
        for seq in &result {
            let s = seq.as_seq().unwrap();
            assert_eq!(s.len(), 4);
            let mut elements: Vec<&TlaValue> = s.iter().collect();
            elements.sort();
            elements.dedup();
            assert_eq!(elements.len(), 4, "must be a permutation");
            all.insert(seq.clone());
        }
        assert_eq!(all.len(), 24, "all 24 permutations distinct");
    }

    #[test]
    fn function_set_zero_length_returns_none() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = vec![TlaValue::Int(0)];
        let result = try_symbolic_function_set_enumerate("TRUE", "p", 0, &range, &ctx);
        assert!(result.is_none(), "zero-length seq should fall back");
    }

    #[test]
    fn function_set_empty_range_returns_zero_seqs() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let result = try_symbolic_function_set_enumerate("TRUE", "p", 3, &[], &ctx)
            .expect("empty range should succeed");
        assert!(result.is_empty(), "empty range yields no seqs");
    }

    #[test]
    fn function_set_unsupported_pred_returns_none() {
        // Sequence projection inside Cardinality is not part of the subset.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let pred = "Cardinality({p[1], p[2], p[3]}) = 3";
        let result = try_symbolic_function_set_enumerate(pred, "p", 3, &range, &ctx);
        assert!(result.is_none(), "unsupported pred falls back");
    }

    #[test]
    fn function_set_pairwise_inequality_distinctness() {
        // Permutation expressed via pairwise # operators (no set-difference
        // shortcut). Should still produce 6 results — the translator
        // handles `p[i] # p[j]` directly via translate_eq + not.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let pred = "/\\ p[1] # p[2] /\\ p[1] # p[3] /\\ p[2] # p[3]";
        let result = try_symbolic_function_set_enumerate(pred, "p", 3, &range, &ctx)
            .expect("symbolic enumeration succeeds");
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn function_set_correctness_gate_brute_force_agrees() {
        // Correctness gate: enumerate small sequence-set with a complex
        // predicate via SMT and via brute force, assert sets agree.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=4).map(TlaValue::Int).collect();
        let pred = "/\\ p[1] # p[2] /\\ p[1] + p[2] = 5 /\\ p[3] = 1";
        let symbolic = try_symbolic_function_set_enumerate(pred, "p", 3, &range, &ctx)
            .expect("symbolic succeeds");

        // Brute force the same enumeration.
        let mut brute: Vec<TlaValue> = Vec::new();
        for a in 1..=4 {
            for b in 1..=4 {
                for c in 1..=4 {
                    if a != b && a + b == 5 && c == 1 {
                        brute.push(TlaValue::Seq(Arc::new(vec![
                            TlaValue::Int(a),
                            TlaValue::Int(b),
                            TlaValue::Int(c),
                        ])));
                    }
                }
            }
        }

        let mut sym_sorted = symbolic.clone();
        sym_sorted.sort();
        let mut brute_sorted = brute.clone();
        brute_sorted.sort();
        assert_eq!(
            sym_sorted, brute_sorted,
            "symbolic enumeration must agree with brute force"
        );
    }
}
