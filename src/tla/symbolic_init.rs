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

// ============================================================================
// T5.5 — Joint Init+invariant symbolic encoding.
//
// Recognises Init shapes where ALL state variables are independently
// constrained as filtered sequence sets (the Einstein shape):
//
//   /\ v1 \in { p \in [Dom -> R1] : Pred1(p) }
//   /\ v2 \in { p \in [Dom -> R2] : Pred2(p) }
//   /\ ...
//
// (each `Pred_i` may be a `Permutation(R_i)`-style chain of distinctness
// constraints plus extra per-variable filters). The brute-force path computes
// the Cartesian product of per-variable enumeration sizes and then BFS-checks
// invariants on each resulting state — O(N1 * N2 * ... * Nk) states.
//
// T5.5 fuses the entire problem into a single Z3 query: it asserts all
// per-variable Init predicates AND the *negation* of the conjunction of all
// invariants. SAT means a violating initial state exists; UNSAT means every
// initial state satisfies every invariant.
//
// On the Einstein riddle (`FindSolution == ~Solution` → invariant body is
// `~Solution`, and the user wants to find a violation that pins down the
// answer), the joint encoding solves in <100ms versus the ~44 minutes the
// brute-force cross-product takes.
// ============================================================================

/// Spec for one state variable in the joint Init+invariant encoding.
///
/// Each variable is modelled as a `seq_len`-long sequence whose elements are
/// drawn from `range`. `init_pred` is the predicate body from the Init clause
/// `var_name \in { p \in [Dom -> Range] : <pred> }`, with the bound name
/// already rewritten to `var_name`. If the Init clause is unfiltered (i.e.
/// `var_name \in [Dom -> Range]`), supply `"TRUE"`.
#[cfg_attr(not(feature = "symbolic-init"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct JointVarSpec {
    pub name: String,
    pub seq_len: usize,
    pub range: Vec<TlaValue>,
    pub init_pred: String,
}

/// Result of a joint Init+invariant solve.
#[derive(Clone, Debug)]
pub enum JointInitOutcome {
    /// Z3 proved that every initial state satisfies every invariant.
    NoViolation,
    /// Z3 found a witness initial state that violates at least one invariant.
    /// The vector is `(var_name, Seq<TlaValue>)` for every state variable.
    Violation {
        state: Vec<(String, TlaValue)>,
    },
}

/// Try to solve the joint Init+invariant query symbolically.
///
/// Returns `Some(outcome)` on a successful translation. Returns `None` if
/// any variable's Init predicate or any invariant body falls outside the
/// supported subset, or if the SMT solver returns `Unknown`. The caller must
/// fall back to brute-force Init enumeration + per-state invariant checking
/// in the `None` case (the established T5.4 streaming-Init path).
///
/// Soundness contract: the symbolic translator is a *conservative*
/// approximation. Anything it cannot translate yields `None` — never a
/// silently-wrong result. This is the same guard used by the T5.1/T5.2
/// per-variable enumerator.
#[cfg_attr(not(feature = "symbolic-init"), allow(unused_variables))]
pub fn try_symbolic_init_with_invariants(
    var_specs: &[JointVarSpec],
    invariants: &[(String, String)],
    ctx: &EvalContext<'_>,
) -> Option<JointInitOutcome> {
    #[cfg(not(feature = "symbolic-init"))]
    {
        None
    }
    #[cfg(feature = "symbolic-init")]
    {
        backend::try_symbolic_init_with_invariants_z3(var_specs, invariants, ctx)
    }
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
    /// returns true if it sees one of the canonical Einstein-shape patterns
    /// AND every position 1..=seq_len is constrained to be distinct from the
    /// others. False negatives are fine (we just miss the Distinct shortcut
    /// and fall back to pairwise inequalities expressed in the predicate).
    ///
    /// T5.6 fix: previously this returned `true` whenever `var[` occurred
    /// >= seq_len times alongside any `\ {`. That was unsound for predicates
    /// like `p[3] \in {1,2,3} \ {p[1], p[2]}` where only one position is
    /// constrained — Distinct would wrongly force ALL positions distinct,
    /// dropping legal solutions where p[1] = p[2]. The corrected check
    /// requires evidence that every position 1..=seq_len participates in
    /// the distinctness chain (either as the LHS of a chain entry, or via
    /// pairwise `#` covering all pairs).
    fn contains_permutation_indicator(pred: &str, var_name: &str, seq_len: usize) -> bool {
        let needle1 = format!("{}[", var_name);
        if !pred.contains(&needle1) {
            return false;
        }

        // Pattern 1: chained `var[i] \in S \ {var[1], ..., var[i-1]}` for
        // every i in 1..=seq_len (i=1 may omit the difference clause if the
        // domain alone constrains it). To keep this textual but sound,
        // require that for every i in 2..=seq_len we see *both*:
        //   (a) `var[i]` appearing somewhere as the LHS of `\in` (i.e.
        //       `var[i] \in` substring), and
        //   (b) within that chain entry, a `\ {` set-difference referencing
        //       at least i-1 prior var[k] occurrences.
        // Cheap approximation: scan for each `var[i] \in` substring and
        // walk forward to find the matching `\ {` and count `var[` inside
        // its braces.
        if pred.contains("\\ {") || pred.contains("\\{") {
            let mut chain_ok = true;
            for i in 2..=seq_len {
                let lhs = format!("{}[{}]", var_name, i);
                let lhs_in = format!("{} \\in", lhs);
                let lhs_in_no_space = format!("{}\\in", lhs);
                let start = pred
                    .find(lhs_in.as_str())
                    .or_else(|| pred.find(lhs_in_no_space.as_str()));
                let Some(start) = start else {
                    chain_ok = false;
                    break;
                };
                // Look for a `\ {` after this LHS within a reasonable window
                // (until the next top-level conjunction marker; we use a
                // simple heuristic of scanning until the next `/\` or end).
                let rest = &pred[start..];
                let end = rest
                    .find("/\\")
                    .or_else(|| rest.find("\\/"))
                    .unwrap_or(rest.len());
                let segment = &rest[..end];
                let Some(diff_idx) = segment.find("\\ {").or_else(|| segment.find("\\{")) else {
                    chain_ok = false;
                    break;
                };
                // Find the closing brace and count var[ inside.
                let after_open = &segment[diff_idx..];
                let Some(close_rel) = after_open.find('}') else {
                    chain_ok = false;
                    break;
                };
                let inside = &after_open[..close_rel];
                let inside_count = inside.matches(needle1.as_str()).count();
                if inside_count < i - 1 {
                    chain_ok = false;
                    break;
                }
            }
            if chain_ok && seq_len >= 2 {
                return true;
            }
        }

        // Pattern 2: pairwise `var[i] # var[j]` for every (i,j) pair.
        // Require literal substrings `var[i] # var[j]` for every i<j; this
        // is the only sound textual witness that all positions are pairwise
        // distinct.
        if seq_len >= 2 && pred.contains('#') {
            let mut all_pairs_ok = true;
            'outer: for i in 1..=seq_len {
                for j in (i + 1)..=seq_len {
                    let p1 = format!("{}[{}] # {}[{}]", var_name, i, var_name, j);
                    let p2 = format!("{}[{}] # {}[{}]", var_name, j, var_name, i);
                    if !pred.contains(p1.as_str()) && !pred.contains(p2.as_str()) {
                        all_pairs_ok = false;
                        break 'outer;
                    }
                }
            }
            if all_pairs_ok {
                return true;
            }
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
                        } else if let Ok(elem_val) = crate::tla::eval_expr(r, self.eval_ctx) {
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

        fn constant_set_membership(&self, var: &Int<'ctx>, set_text: &str) -> Option<Bool<'ctx>> {
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

        fn brace_set_membership(&self, var: &Int<'ctx>, inner: &str) -> Option<Bool<'ctx>> {
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
                    if let (Some(l), Some(r)) = (self.translate_int(lhs), self.translate_int(rhs)) {
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

    // ========================================================================
    // T5.5 — Joint Init+invariant translator
    // ========================================================================

    /// Per-variable encoding for the joint translator.
    struct VarEncoding<'ctx> {
        /// Sequence length (one Z3 Int per position, indexed 1..=seq_len).
        seq_len: usize,
        /// Z3 Int variable for each position 1..=seq_len.
        position_vars: Vec<Int<'ctx>>,
        /// Range encoding (always EnumDomain or IntDomain).
        range_enc: FieldEncoding,
    }

    /// Translator that handles boolean expressions referencing multiple
    /// sequence-shaped state variables. Each variable is registered as
    /// `var_name[i]` with its own per-position Z3 vars.
    struct MultiSeqTranslator<'ctx, 'a> {
        z3: &'ctx Context,
        vars: HashMap<String, VarEncoding<'ctx>>,
        eval_ctx: &'a EvalContext<'a>,
    }

    pub(super) fn try_symbolic_init_with_invariants_z3(
        var_specs: &[super::JointVarSpec],
        invariants: &[(String, String)],
        ctx: &EvalContext<'_>,
    ) -> Option<super::JointInitOutcome> {
        if var_specs.is_empty() {
            return None;
        }
        // Defensive: each variable must have a non-empty range and a positive
        // sequence length. An empty range with non-empty domain means there
        // are no initial states at all → vacuously no violation.
        let mut any_empty_range = false;
        for spec in var_specs {
            if spec.seq_len == 0 {
                return None;
            }
            if spec.range.is_empty() {
                any_empty_range = true;
            }
        }
        if any_empty_range {
            return Some(super::JointInitOutcome::NoViolation);
        }

        let z3_cfg = Config::new();
        let z3_ctx = Context::new(&z3_cfg);
        let solver = Solver::new(&z3_ctx);

        // Build per-variable encodings.
        let mut vars: HashMap<String, VarEncoding<'_>> = HashMap::with_capacity(var_specs.len());
        for spec in var_specs {
            if vars.contains_key(&spec.name) {
                // Duplicate variable name — shape unsupported.
                return None;
            }
            let range_enc = build_range_encoding(&spec.range)?;
            let mut position_vars: Vec<Int<'_>> = Vec::with_capacity(spec.seq_len);
            for i in 1..=spec.seq_len {
                let v = Int::new_const(&z3_ctx, format!("{}_{}", spec.name, i));
                assert_domain_constraint(&z3_ctx, &solver, &v, &range_enc);
                position_vars.push(v);
            }
            vars.insert(
                spec.name.clone(),
                VarEncoding {
                    seq_len: spec.seq_len,
                    position_vars,
                    range_enc,
                },
            );
        }

        let translator = MultiSeqTranslator {
            z3: &z3_ctx,
            vars,
            eval_ctx: ctx,
        };

        // Assert each variable's Init predicate.
        for spec in var_specs {
            let pred_text = spec.init_pred.trim();
            // Permutation distinctness shortcut (T5.2): if seq_len ==
            // range.len() and the predicate textually contains a permutation
            // indicator, additionally assert Distinct over all positions.
            // (Does no harm if redundant — Z3 simplifies trivially.)
            let enc = translator.vars.get(&spec.name)?;
            if spec.seq_len == range_size(&enc.range_enc)
                && contains_permutation_indicator(pred_text, &spec.name, spec.seq_len)
            {
                let var_refs: Vec<&Int<'_>> = enc.position_vars.iter().collect();
                let distinct = z3::ast::Ast::distinct(&z3_ctx, &var_refs);
                solver.assert(&distinct);
            }

            // Empty / TRUE predicate is vacuously true.
            if pred_text.is_empty() || pred_text.eq_ignore_ascii_case("TRUE") {
                continue;
            }
            let pred_z3 = translator.translate_bool(pred_text)?;
            solver.assert(&pred_z3);
        }

        // Assert the negation of (invariant_1 /\ invariant_2 /\ ...).
        // SAT == witness violating at least one invariant.
        if !invariants.is_empty() {
            let mut inv_bools: Vec<Bool<'_>> = Vec::with_capacity(invariants.len());
            for (_name, body) in invariants {
                let body = body.trim();
                if body.is_empty() {
                    continue;
                }
                let inv_z3 = translator.translate_bool(body)?;
                inv_bools.push(inv_z3);
            }
            if inv_bools.is_empty() {
                // No invariants to check — vacuously safe.
                return Some(super::JointInitOutcome::NoViolation);
            }
            let inv_refs: Vec<&Bool<'_>> = inv_bools.iter().collect();
            let conj = Bool::and(&z3_ctx, &inv_refs);
            solver.assert(&conj.not());
        } else {
            return Some(super::JointInitOutcome::NoViolation);
        }

        match solver.check() {
            SatResult::Sat => {
                let model = solver.get_model()?;
                let mut state: Vec<(String, TlaValue)> = Vec::with_capacity(var_specs.len());
                for spec in var_specs {
                    let enc = translator.vars.get(&spec.name)?;
                    let mut elems: Vec<TlaValue> = Vec::with_capacity(spec.seq_len);
                    for v in &enc.position_vars {
                        let code = model.eval(v, true)?.as_i64()?;
                        let value = match &enc.range_enc {
                            FieldEncoding::IntDomain { .. } => TlaValue::Int(code),
                            FieldEncoding::EnumDomain { values } => {
                                if code < 0 || (code as usize) >= values.len() {
                                    return None;
                                }
                                values[code as usize].clone()
                            }
                        };
                        elems.push(value);
                    }
                    state.push((spec.name.clone(), super::build_seq(elems)));
                }
                Some(super::JointInitOutcome::Violation { state })
            }
            SatResult::Unsat => Some(super::JointInitOutcome::NoViolation),
            SatResult::Unknown => None,
        }
    }

    fn range_size(enc: &FieldEncoding) -> usize {
        match enc {
            FieldEncoding::IntDomain { values } => values.len(),
            FieldEncoding::EnumDomain { values } => values.len(),
        }
    }

    impl<'ctx, 'a> MultiSeqTranslator<'ctx, 'a> {
        /// Lookup `var_name[K]` where K is a positive integer literal in
        /// `1..=seq_len(var_name)`. Returns the (var_encoding, position var).
        fn lookup_position(&self, expr: &str) -> Option<(&VarEncoding<'ctx>, Int<'ctx>)> {
            let expr = strip_redundant_parens(expr.trim());
            // Find the `[` — must be a registered var name on the LHS.
            let open = expr.find('[')?;
            let name = expr[..open].trim();
            if name.is_empty() {
                return None;
            }
            let enc = self.vars.get(name)?;
            let rest = &expr[open + 1..];
            let close = rest.rfind(']')?;
            // Anything after `]` would mean compound expression — bail.
            if !rest[close + 1..].trim().is_empty() {
                return None;
            }
            let inner = rest[..close].trim();
            let idx = if let Ok(n) = inner.parse::<i64>() {
                n
            } else {
                let v = crate::tla::eval_expr(inner, self.eval_ctx).ok()?;
                v.as_int().ok()?
            };
            if idx < 1 || (idx as usize) > enc.seq_len {
                return None;
            }
            Some((enc, enc.position_vars[(idx as usize) - 1].clone()))
        }

        /// True if `expr` mentions any registered var name as an identifier.
        fn mentions_any_var(&self, expr: &str) -> bool {
            for name in self.vars.keys() {
                if mentions_record_var(expr, name) {
                    return true;
                }
            }
            false
        }

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

            if let Some(stripped) = strip_quantifier_keyword(expr, "\\A") {
                return self.translate_quantifier(stripped, true);
            }
            if let Some(stripped) = strip_quantifier_keyword(expr, "\\E") {
                return self.translate_quantifier(stripped, false);
            }

            if let Some(b) = self.translate_atomic_bool(expr) {
                return Some(b);
            }

            // Operator-name expansion. If `expr` is a bare identifier (or a
            // parameterised operator call `Name(arg1, arg2, ...)`) that
            // resolves to a definition, inline its body and recurse. This
            // is what lets the joint Init+invariant solver translate the
            // top-level invariant `FindSolution`, which expands to
            // `~Solution`, which in turn expands to a giant conjunction of
            // sub-operators (BritLivesInTheRedHouse /\ ...). Each
            // sub-operator's body in turn references state variables that
            // the joint translator does know about, so the recursion
            // terminates at translatable atoms.
            //
            // Only performed if `expr` mentions at least one of our
            // registered state variables — otherwise the constant-fold path
            // below will handle it. (We test mention transitively by
            // expanding the body and checking it.)
            if let Some(expanded) = self.try_expand_definition(expr) {
                if let Some(b) = self.translate_bool(&expanded) {
                    return Some(b);
                }
            }

            // Constant fold for sub-exprs that mention no registered var.
            if !self.mentions_any_var(expr) {
                if let Ok(TlaValue::Bool(b)) = crate::tla::eval_expr(expr, self.eval_ctx) {
                    return Some(Bool::from_bool(self.z3, b));
                }
            }

            None
        }

        /// If `expr` is a bare identifier `Name` or a call `Name(arg1, ...)`
        /// that resolves to a parameterless / matching-arity definition in
        /// the eval context, return the expanded body with arguments
        /// textually substituted for parameters.
        fn try_expand_definition(&self, expr: &str) -> Option<String> {
            let expr = strip_redundant_parens(expr.trim());
            let defs = self.eval_ctx.definitions?;
            // Bare identifier.
            if is_valid_identifier(expr) {
                let def = defs.get(expr)?;
                if def.params.is_empty() {
                    return Some(def.body.clone());
                }
                return None;
            }
            // Operator call `Name(arg1, arg2, ...)`.
            if let Some(open) = expr.find('(') {
                if expr.ends_with(')') {
                    let name = expr[..open].trim();
                    if is_valid_identifier(name) {
                        let def = defs.get(name)?;
                        let args_text = &expr[open + 1..expr.len() - 1];
                        let args = split_top_level_symbol(args_text, ",");
                        if def.params.len() != args.len() {
                            return None;
                        }
                        let mut body = def.body.clone();
                        for (param, arg) in def.params.iter().zip(args.iter()) {
                            body = substitute_identifier(&body, param, arg.trim());
                        }
                        return Some(body);
                    }
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
            // Domain must not mention any registered var.
            if self.mentions_any_var(domain_expr) {
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
            if let Some(in_idx) = find_top_level_keyword_index(expr, "\\in") {
                let lhs = expr[..in_idx].trim();
                let rhs = expr[in_idx + 3..].trim();
                return self.translate_membership(lhs, rhs);
            }
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
            let lhs_pos = self.lookup_position(lhs);

            // rhs as range a..b
            if let Some(dotdot) = rhs.find("..") {
                let lo_text = rhs[..dotdot].trim();
                let hi_text = rhs[dotdot + 2..].trim();
                let lo = const_eval_int(lo_text, self.eval_ctx)?;
                let hi = const_eval_int(hi_text, self.eval_ctx)?;
                let lo_e = Int::from_i64(self.z3, lo);
                let hi_e = Int::from_i64(self.z3, hi);
                if let Some((enc, var)) = lhs_pos {
                    if matches!(enc.range_enc, FieldEncoding::IntDomain { .. }) {
                        return Some(Bool::and(self.z3, &[&var.ge(&lo_e), &var.le(&hi_e)]));
                    }
                    return None;
                }
                return None;
            }

            // rhs as set difference `S \ {var[k], ...}`
            if let Some(diff_idx) = find_top_level_set_diff(rhs) {
                let base_text = rhs[..diff_idx].trim();
                let removed_text = rhs[diff_idx + 1..].trim_start();
                if let Some(removed_inner) = removed_text
                    .strip_prefix('{')
                    .and_then(|s| s.strip_suffix('}'))
                {
                    let (enc, var) = lhs_pos?;
                    let mut clauses: Vec<Bool<'ctx>> = Vec::new();
                    let base_clause = self.constant_set_membership(&var, &enc.range_enc, base_text)?;
                    clauses.push(base_clause);
                    let removed_parts = split_top_level_symbol(removed_inner, ",");
                    for r in &removed_parts {
                        let r = r.trim();
                        if r.is_empty() {
                            continue;
                        }
                        if let Some((_other_enc, other)) = self.lookup_position(r) {
                            clauses.push(var._eq(&other).not());
                        } else if let Ok(elem_val) = crate::tla::eval_expr(r, self.eval_ctx) {
                            let coded = encode_value_as_int(&elem_val, &enc.range_enc)?;
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

            if let Some(inner) = rhs.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let (enc, var) = lhs_pos?;
                return self.brace_set_membership(&var, &enc.range_enc, inner);
            }

            if !self.mentions_any_var(rhs) {
                let (enc, var) = lhs_pos?;
                return self.constant_set_membership(&var, &enc.range_enc, rhs);
            }

            None
        }

        fn constant_set_membership(
            &self,
            var: &Int<'ctx>,
            enc: &FieldEncoding,
            set_text: &str,
        ) -> Option<Bool<'ctx>> {
            let val = crate::tla::eval_expr(set_text, self.eval_ctx).ok()?;
            let set = val.as_set().ok()?;
            let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(set.len());
            for elem in set.iter() {
                let coded = encode_value_as_int(elem, enc)?;
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
            enc: &FieldEncoding,
            inner: &str,
        ) -> Option<Bool<'ctx>> {
            let parts = split_top_level_symbol(inner, ",");
            let mut eqs: Vec<Bool<'ctx>> = Vec::with_capacity(parts.len());
            for part in &parts {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Some((_other_enc, other)) = self.lookup_position(trimmed) {
                    eqs.push(var._eq(&other));
                    continue;
                }
                let elem_val = crate::tla::eval_expr(trimmed, self.eval_ctx).ok()?;
                let coded = encode_value_as_int(&elem_val, enc)?;
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
                (Some((lenc, l)), Some((renc, r))) => {
                    // Cross-variable equality: encodings may differ.
                    if matches!(lenc.range_enc, FieldEncoding::IntDomain { .. })
                        && matches!(renc.range_enc, FieldEncoding::IntDomain { .. })
                    {
                        return Some(l._eq(&r));
                    }
                    if let (
                        FieldEncoding::EnumDomain { values: lvals },
                        FieldEncoding::EnumDomain { values: rvals },
                    ) = (&lenc.range_enc, &renc.range_enc)
                    {
                        // Build OR over shared values: (l == lcode AND r == rcode).
                        let mut pairs: Vec<Bool<'_>> = Vec::new();
                        for (li, lval) in lvals.iter().enumerate() {
                            if let Some(ri) = rvals.iter().position(|v| v == lval) {
                                let l_eq = l._eq(&Int::from_i64(self.z3, li as i64));
                                let r_eq = r._eq(&Int::from_i64(self.z3, ri as i64));
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
                (Some((enc, var)), None) => {
                    if matches!(enc.range_enc, FieldEncoding::IntDomain { .. }) {
                        if let Some(r) = self.translate_int(rhs) {
                            return Some(var._eq(&r));
                        }
                    }
                    let elem = crate::tla::eval_expr(rhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, &enc.range_enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, Some((enc, var))) => {
                    if matches!(enc.range_enc, FieldEncoding::IntDomain { .. }) {
                        if let Some(l) = self.translate_int(lhs) {
                            return Some(var._eq(&l));
                        }
                    }
                    let elem = crate::tla::eval_expr(lhs, self.eval_ctx).ok()?;
                    let coded = encode_value_as_int(&elem, &enc.range_enc)?;
                    Some(var._eq(&Int::from_i64(self.z3, coded)))
                }
                (None, None) => {
                    if let (Some(l), Some(r)) = (self.translate_int(lhs), self.translate_int(rhs)) {
                        return Some(l._eq(&r));
                    }
                    if !self.mentions_any_var(lhs) && !self.mentions_any_var(rhs) {
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

            if let Some((enc, var)) = self.lookup_position(expr) {
                if matches!(enc.range_enc, FieldEncoding::IntDomain { .. }) {
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

            if !self.mentions_any_var(expr) {
                if let Ok(TlaValue::Int(n)) = crate::tla::eval_expr(expr, self.eval_ctx) {
                    return Some(Int::from_i64(self.z3, n));
                }
            }

            None
        }
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

    // ========================================================================
    // T5.5 — joint Init+invariant tests
    // ========================================================================

    #[test]
    fn joint_solve_no_invariants_returns_no_violation() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let specs = vec![JointVarSpec {
            name: "x".to_string(),
            seq_len: 3,
            range: vec![TlaValue::Int(1), TlaValue::Int(2), TlaValue::Int(3)],
            init_pred: "TRUE".to_string(),
        }];
        let outcome = try_symbolic_init_with_invariants(&specs, &[], &ctx)
            .expect("solve should succeed");
        assert!(matches!(outcome, JointInitOutcome::NoViolation));
    }

    #[test]
    fn joint_solve_two_vars_violation_witness() {
        // Two sequence variables; invariant says
        // `\E i \in 1..3 : a[i] = b[i]` (matching positions). Violation
        // means there's some assignment with all positions different.
        // With seq_len 3 and range {1,2,3}, the cross-product is 27*27=729
        // total state pairs; many violate. Witness should satisfy the
        // negated invariant.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let specs = vec![
            JointVarSpec {
                name: "a".to_string(),
                seq_len: 3,
                range: range.clone(),
                init_pred: "TRUE".to_string(),
            },
            JointVarSpec {
                name: "b".to_string(),
                seq_len: 3,
                range,
                init_pred: "TRUE".to_string(),
            },
        ];
        let invariant = (
            "MatchExists".to_string(),
            "\\E i \\in 1..3 : a[i] = b[i]".to_string(),
        );
        let outcome = try_symbolic_init_with_invariants(&specs, &[invariant], &ctx)
            .expect("solve should succeed");
        match outcome {
            JointInitOutcome::Violation { state } => {
                // Witness must have a[i] != b[i] for all i in 1..3.
                let a = state.iter().find(|(n, _)| n == "a").unwrap().1.as_seq().unwrap().clone();
                let b = state.iter().find(|(n, _)| n == "b").unwrap().1.as_seq().unwrap().clone();
                assert_eq!(a.len(), 3);
                assert_eq!(b.len(), 3);
                for i in 0..3 {
                    assert_ne!(a[i], b[i], "witness violates invariant: position {} matches", i + 1);
                }
            }
            JointInitOutcome::NoViolation => panic!("expected violation witness"),
        }
    }

    #[test]
    fn joint_solve_two_vars_no_violation_when_invariant_universally_true() {
        // Both vars pinned to identical sequences [1,2,3]. Invariant says
        // `\A i \in 1..3 : a[i] = b[i]` (every position matches). This holds
        // for every initial state (only one exists: a=b=[1,2,3]). The
        // joint solver should return NoViolation.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let specs = vec![
            JointVarSpec {
                name: "a".to_string(),
                seq_len: 3,
                range: range.clone(),
                init_pred: "/\\ a[1] = 1 /\\ a[2] = 2 /\\ a[3] = 3".to_string(),
            },
            JointVarSpec {
                name: "b".to_string(),
                seq_len: 3,
                range,
                init_pred: "/\\ b[1] = 1 /\\ b[2] = 2 /\\ b[3] = 3".to_string(),
            },
        ];
        let invariant = (
            "AllMatch".to_string(),
            "\\A i \\in 1..3 : a[i] = b[i]".to_string(),
        );
        let outcome = try_symbolic_init_with_invariants(&specs, &[invariant], &ctx)
            .expect("solve should succeed");
        assert!(
            matches!(outcome, JointInitOutcome::NoViolation),
            "with both pinned to identical sequences, every position matches"
        );
    }

    #[test]
    fn joint_solve_returns_none_for_zero_var_specs() {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let outcome = try_symbolic_init_with_invariants(&[], &[], &ctx);
        assert!(outcome.is_none());
    }

    #[test]
    fn joint_solve_empty_range_proves_safe() {
        // Empty range with non-empty seq → no initial state exists at all.
        // Vacuously safe (no violation possible).
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let specs = vec![JointVarSpec {
            name: "x".to_string(),
            seq_len: 3,
            range: vec![],
            init_pred: "TRUE".to_string(),
        }];
        let invariant = ("Always".to_string(), "x[1] = x[2]".to_string());
        let outcome = try_symbolic_init_with_invariants(&specs, &[invariant], &ctx)
            .expect("solve should succeed");
        assert!(matches!(outcome, JointInitOutcome::NoViolation));
    }

    #[test]
    fn joint_solve_witness_satisfies_init_predicates_and_violates_invariant() {
        // Mini-Einstein: a is a 3-permutation of {1,2,3} (distinctness via
        // chained set-difference), b is a 3-permutation of {10,20,30}.
        // Invariant: `\E i \in 1..3 : a[i] = 1 /\ b[i] = 30`.
        // With both free permutations there exist witnesses where position
        // of 1 in a doesn't co-occur with position of 30 in b. So
        // `~invariant` is SAT → solver returns Violation witness.
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range_a: Vec<TlaValue> = (1..=3).map(TlaValue::Int).collect();
        let range_b: Vec<TlaValue> = (1..=3).map(|i| TlaValue::Int(i * 10)).collect();
        let specs = vec![
            JointVarSpec {
                name: "a".to_string(),
                seq_len: 3,
                range: range_a,
                init_pred: "/\\ a[2] \\in {1,2,3} \\ {a[1]}\n/\\ a[3] \\in {1,2,3} \\ {a[1], a[2]}".to_string(),
            },
            JointVarSpec {
                name: "b".to_string(),
                seq_len: 3,
                range: range_b,
                init_pred: "/\\ b[2] \\in {10,20,30} \\ {b[1]}\n/\\ b[3] \\in {10,20,30} \\ {b[1], b[2]}".to_string(),
            },
        ];
        let invariant = (
            "AlwaysCoLocated".to_string(),
            "\\E i \\in 1..3 : a[i] = 1 /\\ b[i] = 30".to_string(),
        );
        let outcome = try_symbolic_init_with_invariants(&specs, &[invariant], &ctx)
            .expect("solve should succeed");
        match outcome {
            JointInitOutcome::Violation { state } => {
                let a = state.iter().find(|(n, _)| n == "a").unwrap().1.as_seq().unwrap().clone();
                let b = state.iter().find(|(n, _)| n == "b").unwrap().1.as_seq().unwrap().clone();
                // Verify a is permutation of {1,2,3} and b of {10,20,30}.
                let mut a_sorted: Vec<i64> = a.iter().map(|v| v.as_int().unwrap()).collect();
                a_sorted.sort();
                assert_eq!(a_sorted, vec![1, 2, 3]);
                let mut b_sorted: Vec<i64> = b.iter().map(|v| v.as_int().unwrap()).collect();
                b_sorted.sort();
                assert_eq!(b_sorted, vec![10, 20, 30]);
                // Verify witness violates invariant: position of 1 in a ≠
                // position of 30 in b.
                let pos_a = a.iter().position(|v| *v == TlaValue::Int(1)).unwrap();
                let pos_b = b.iter().position(|v| *v == TlaValue::Int(30)).unwrap();
                assert_ne!(pos_a, pos_b);
            }
            JointInitOutcome::NoViolation => panic!("expected violation witness"),
        }
    }
}
