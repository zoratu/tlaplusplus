use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Temporal formula representation for TLA+ temporal operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalFormula {
    /// State predicate (boolean expression evaluated on a single state)
    StatePredicate(String),

    /// [] P - Always P (invariance)
    /// True if P holds in all reachable states along the path
    Always(Box<TemporalFormula>),

    /// <> P - Eventually P (liveness)
    /// True if there exists a state in the future where P holds
    Eventually(Box<TemporalFormula>),

    /// P ~> Q - Leads-to (P implies eventually Q)
    /// If P becomes true, then Q must eventually become true
    LeadsTo(Box<TemporalFormula>, Box<TemporalFormula>),

    /// [] <> P - Infinitely often P
    /// P holds infinitely often along any infinite path
    InfinitelyOften(Box<TemporalFormula>),

    /// <> [] P - Eventually always P (stability)
    /// Eventually P becomes true and stays true forever
    EventuallyAlways(Box<TemporalFormula>),

    /// P /\ Q - Conjunction
    And(Box<TemporalFormula>, Box<TemporalFormula>),

    /// P \/ Q - Disjunction
    Or(Box<TemporalFormula>, Box<TemporalFormula>),

    /// P => Q - Temporal implication.
    ///
    /// Only produced when at least one operand is itself TEMPORAL (contains
    /// `<>`/`[]`/`~>`/`WF_`/`SF_`). A pure `state-pred => state-pred` is kept
    /// as a single `StatePredicate` (see the parser guardrail) so box-safety
    /// lowering (`[](A => B)`) and ACP_NB-style `[]`-safety are not broken.
    /// The canonical shape this unlocks is
    /// `[](P => <>[]Q)` ->
    /// `Always(Implies(StatePredicate(P), EventuallyAlways(StatePredicate(Q))))`.
    Implies(Box<TemporalFormula>, Box<TemporalFormula>),

    /// ~P - Negation
    Not(Box<TemporalFormula>),

    /// WF_v(A) - Weak fairness of action A with respect to variables v
    /// If A is enabled infinitely often, then A must occur infinitely often
    WeakFairness { vars: Vec<String>, action: String },

    /// SF_v(A) - Strong fairness of action A with respect to variables v
    /// If A is enabled infinitely often without being disabled, then A must occur infinitely often
    StrongFairness { vars: Vec<String>, action: String },

    /// \AA x: P - Universal temporal quantification
    /// For all values x in the domain, temporal formula P holds
    TemporalForAll {
        var: String,
        domain: String,
        formula: Box<TemporalFormula>,
    },

    /// \EE x: P - Existential temporal quantification
    /// There exists a value x in the domain such that temporal formula P holds
    TemporalExists {
        var: String,
        domain: String,
        formula: Box<TemporalFormula>,
    },
}

impl TemporalFormula {
    /// Check if this formula is a pure safety property (can be checked on individual states)
    pub fn is_safety_property(&self) -> bool {
        match self {
            TemporalFormula::StatePredicate(_) => true,
            TemporalFormula::Always(inner) => inner.is_safety_property(),
            TemporalFormula::And(left, right) => {
                left.is_safety_property() && right.is_safety_property()
            }
            TemporalFormula::Not(inner) => {
                // ~[]P is not a safety property
                !matches!(inner.as_ref(), TemporalFormula::Always(_))
            }
            _ => false,
        }
    }

    /// Check if this formula is a pure liveness property (requires path/trace checking)
    pub fn is_liveness_property(&self) -> bool {
        match self {
            TemporalFormula::Eventually(_)
            | TemporalFormula::LeadsTo(_, _)
            | TemporalFormula::InfinitelyOften(_)
            | TemporalFormula::EventuallyAlways(_)
            | TemporalFormula::WeakFairness { .. }
            | TemporalFormula::StrongFairness { .. } => true,
            TemporalFormula::TemporalForAll { formula, .. }
            | TemporalFormula::TemporalExists { formula, .. } => formula.is_liveness_property(),
            TemporalFormula::And(left, right) => {
                left.is_liveness_property() || right.is_liveness_property()
            }
            TemporalFormula::Or(left, right) => {
                left.is_liveness_property() || right.is_liveness_property()
            }
            // A temporal implication `P => G` is a liveness obligation iff its
            // consequent is one. This is what makes `[](P => <>[]Q)` classify
            // as liveness (Always -> Implies -> EventuallyAlways) so the
            // graph post-processing pass runs for it.
            TemporalFormula::Implies(_, consequent) => consequent.is_liveness_property(),
            // NARROW `Always` arm (blast-radius control): `[]G` is liveness
            // ONLY when `G` is a temporal implication carrying a liveness
            // consequent — i.e. exactly the new `[](P => <>[]Q)` shape. Every
            // other `Always(..)` keeps its prior classification (falls to
            // `_ => false`), so no existing spec's safety/liveness verdict or
            // fairness-pass gating changes.
            TemporalFormula::Always(inner)
                if matches!(inner.as_ref(), TemporalFormula::Implies(_, _)) =>
            {
                inner.is_liveness_property()
            }
            _ => false,
        }
    }

    /// Parse a temporal formula from TLA+ syntax
    ///
    /// Operator precedence (from lowest to highest):
    /// 1. \AA, \EE (temporal quantifiers) - bind variables first
    /// 2. /\ and \/ (conjunction/disjunction)
    /// 3. ~> (leads-to)
    /// 4. [], <>, []<>, <>[], WF, SF (temporal operators) - highest precedence
    pub fn parse(expr: &str) -> Result<Self> {
        let trimmed = expr.trim();

        // Strip outer parentheses if present
        if trimmed.starts_with('(') && trimmed.ends_with(')') {
            let inner = &trimmed[1..trimmed.len() - 1];
            if is_balanced(inner) {
                return Self::parse(inner);
            }
        }

        // Highest priority: Check for temporal quantifiers first (they bind variables)
        // \AA and \EE must be parsed before conjunction because the : separates var from formula
        if let Some(rest) = trimmed.strip_prefix("\\AA") {
            if let Some((var, domain, formula_expr)) = parse_temporal_quantifier(rest.trim()) {
                let formula = Self::parse(&formula_expr)?;
                return Ok(TemporalFormula::TemporalForAll {
                    var,
                    domain,
                    formula: Box::new(formula),
                });
            }
        }

        if let Some(rest) = trimmed.strip_prefix("\\EE") {
            if let Some((var, domain, formula_expr)) = parse_temporal_quantifier(rest.trim()) {
                let formula = Self::parse(&formula_expr)?;
                return Ok(TemporalFormula::TemporalExists {
                    var,
                    domain,
                    formula: Box::new(formula),
                });
            }
        }

        // If the expression contains NO temporal operator at all, it is a pure
        // state predicate — even if it contains top-level `/\`, `\/`, `\A`,
        // `\E`, etc. Decomposing such text into temporal `And`/`Or` is wrong:
        // e.g. `[] \A i,j : (\/ A \/ B)` must stay `Always(StatePredicate(..))`,
        // not `Or(Always("\A i,j :"), Or(A, B))`. Splitting on `\/`/`/\` is only
        // meaningful when the operands are themselves temporal formulas. Bail
        // out to StatePredicate here so the state-predicate evaluator (which
        // understands quantifiers, `#`, `\/`, function application, etc.)
        // handles the whole body atomically. This is what makes a
        // box-safety property like `AC1 == [] \A i,j : ...` parse as
        // `Always(StatePredicate(...))` and thus be lowerable to an invariant.
        if !contains_temporal_operator(trimmed) {
            return Ok(TemporalFormula::StatePredicate(trimmed.to_string()));
        }

        // Temporal-prefix over a pure state predicate. A prefix operator
        // (`[]`, `<>`, `[]<>`, `<>[]`) binds tighter than infix `/\`/`\/`/`~>`,
        // so if the ENTIRE expression is a single prefix applied to a
        // temporal-operator-free body, build the temporal node directly and
        // keep the body atomic. This is the AC1 case:
        //   `[] \A i,j : \/ P \/ Q`  ->  Always(StatePredicate("\A i,j : ..."))
        // Without this, the top-level `\/` inside the (unparenthesized)
        // quantifier body would be mis-split into a temporal `Or`. We only take
        // this path when the body has NO temporal operator, so genuinely mixed
        // formulas like `[] P /\ <> Q` still fall through to the infix split.
        for (prefix, ctor) in [
            ("[]<>", 0usize),
            ("<>[]", 1),
            ("[]", 2),
            ("<>", 3),
        ] {
            if let Some(body) = trimmed.strip_prefix(prefix) {
                let body = body.trim();
                if !contains_temporal_operator(body) && !body.is_empty() {
                    let inner = Box::new(TemporalFormula::StatePredicate(body.to_string()));
                    return Ok(match ctor {
                        0 => TemporalFormula::InfinitelyOften(inner),
                        1 => TemporalFormula::EventuallyAlways(inner),
                        2 => TemporalFormula::Always(inner),
                        _ => TemporalFormula::Eventually(inner),
                    });
                }
            }
        }

        // Temporal implication `P => Q`. In TLA+, `=>` binds LOOSER than
        // `/\`/`\/`, so we split on the FIRST top-level `=>` before the
        // conjunction/disjunction split (a top-level `=>` groups the whole
        // expression as an implication).
        //
        // CRITICAL GUARDRAIL (soundness): only build a temporal `Implies` node
        // when at least ONE operand is genuinely temporal (contains
        // `<>`/`[]`/`~>`/`WF_`/`SF_`/action-box). If BOTH operands are pure
        // state predicates we must NOT decompose — the whole thing is a state
        // predicate `P => Q` that the state-predicate evaluator handles
        // atomically, and decomposing it would break box-safety lowering
        // (`[](A => B)` -> per-state invariant, PR #127) and ACP_NB-style AC1
        // (`[] \A .. : (P => Q)`). Note we only ever reach this point when the
        // *overall* expression contains a temporal operator (the
        // `!contains_temporal_operator` bailout above already returned for
        // fully non-temporal text) — but the temporal operator might live
        // outside this particular `=>`'s operands (e.g. `[] P` sitting in a
        // conjunction), so we re-check each operand independently here.
        //
        // We split on `=>` but not on the `>=`/`<=`/`=<` comparison tokens or
        // the `<=>` equivalence; `find_top_level_implies` handles that.
        if let Some(idx) = find_top_level_implies(trimmed) {
            let lhs = trimmed[..idx].trim();
            let rhs = trimmed[idx + 2..].trim();
            if contains_temporal_operator(lhs) || contains_temporal_operator(rhs) {
                let left = Self::parse(lhs)?;
                let right = Self::parse(rhs)?;
                return Ok(TemporalFormula::Implies(Box::new(left), Box::new(right)));
            }
            // Both operands are pure state predicates: fall through so the
            // whole `P => Q` is kept atomic (StatePredicate) by the default
            // arm at the bottom. We must NOT split on `/\`/`\/` inside a
            // state-predicate implication either, so bail directly.
            return Ok(TemporalFormula::StatePredicate(trimmed.to_string()));
        }

        // Lower precedence: Check for conjunction/disjunction
        if let Some(idx) = find_top_level_operator(trimmed, "/\\") {
            let left = Self::parse(&trimmed[..idx])?;
            let right = Self::parse(&trimmed[idx + 2..])?;
            return Ok(TemporalFormula::And(Box::new(left), Box::new(right)));
        }

        if let Some(idx) = find_top_level_operator(trimmed, "\\/") {
            let left = Self::parse(&trimmed[..idx])?;
            let right = Self::parse(&trimmed[idx + 2..])?;
            return Ok(TemporalFormula::Or(Box::new(left), Box::new(right)));
        }

        // Medium precedence: Check for ~> (leads-to)
        if let Some(idx) = find_top_level_operator(trimmed, "~>") {
            let left = Self::parse(&trimmed[..idx])?;
            let right = Self::parse(&trimmed[idx + 2..])?;
            return Ok(TemporalFormula::LeadsTo(Box::new(left), Box::new(right)));
        }

        // Higher precedence: Check for prefix temporal operators (longest patterns first)
        // Check for []<> pattern
        if let Some(inner) = trimmed.strip_prefix("[]<>") {
            let inner_formula = Self::parse(inner.trim())?;
            return Ok(TemporalFormula::InfinitelyOften(Box::new(inner_formula)));
        }

        // Check for <>[] pattern
        if let Some(inner) = trimmed.strip_prefix("<>[]") {
            let inner_formula = Self::parse(inner.trim())?;
            return Ok(TemporalFormula::EventuallyAlways(Box::new(inner_formula)));
        }

        // Check for [] (always)
        if let Some(inner) = trimmed.strip_prefix("[]") {
            let inner_formula = Self::parse(inner.trim())?;
            return Ok(TemporalFormula::Always(Box::new(inner_formula)));
        }

        // Check for <> (eventually)
        if let Some(inner) = trimmed.strip_prefix("<>") {
            let inner_formula = Self::parse(inner.trim())?;
            return Ok(TemporalFormula::Eventually(Box::new(inner_formula)));
        }

        // Check for WF_vars(action)
        if let Some(rest) = trimmed.strip_prefix("WF_") {
            if let Some((vars_part, action_part)) = parse_fairness_expr(rest) {
                return Ok(TemporalFormula::WeakFairness {
                    vars: parse_var_list(&vars_part),
                    action: action_part.to_string(),
                });
            }
        }

        // Check for SF_vars(action)
        if let Some(rest) = trimmed.strip_prefix("SF_") {
            if let Some((vars_part, action_part)) = parse_fairness_expr(rest) {
                return Ok(TemporalFormula::StrongFairness {
                    vars: parse_var_list(&vars_part),
                    action: action_part.to_string(),
                });
            }
        }

        // Check for negation
        if let Some(inner) = trimmed
            .strip_prefix('~')
            .or_else(|| trimmed.strip_prefix("\\neg"))
        {
            let inner_formula = Self::parse(inner.trim())?;
            return Ok(TemporalFormula::Not(Box::new(inner_formula)));
        }

        // Default: treat as state predicate
        Ok(TemporalFormula::StatePredicate(trimmed.to_string()))
    }
}

/// Parse fairness expression: vars(action) -> (vars, action)
fn parse_fairness_expr(input: &str) -> Option<(String, String)> {
    let input = input.trim();
    let paren_idx = input.find('(')?;
    let vars_part = input[..paren_idx].trim();

    // Find matching closing paren
    let rest = &input[paren_idx + 1..];
    let close_idx = rest.rfind(')')?;
    let action_part = rest[..close_idx].trim();

    Some((vars_part.to_string(), action_part.to_string()))
}

/// Parse variable list: "<<x, y, z>>" or "x" -> vec!["x", "y", "z"]
fn parse_var_list(input: &str) -> Vec<String> {
    let input = input.trim();

    // Handle <<x, y, z>> format
    if let Some(inner) = input.strip_prefix("<<").and_then(|s| s.strip_suffix(">>")) {
        return inner.split(',').map(|s| s.trim().to_string()).collect();
    }

    // Handle single variable
    vec![input.to_string()]
}

/// Returns true if `expr` contains any TLA+ temporal operator, i.e. it is not
/// a pure state predicate. Used to decide whether the temporal parser should
/// decompose `/\`/`\/`/`~>` into temporal `And`/`Or`/`LeadsTo` nodes, or treat
/// the whole text atomically as a `StatePredicate`.
///
/// Detects: `[]` (box), `<>` (diamond), `[A]_v` (action box), `~>` (leads-to),
/// `WF_`/`SF_` (fairness), and `\AA`/`\EE` (temporal quantifiers).
///
/// Note the `[...]_` action-box form: a leading-bracket subterm immediately
/// followed by `_` (subscript) is an action formula and IS temporal, even
/// though it has no `[]`/`<>` prefix.
fn contains_temporal_operator(expr: &str) -> bool {
    if expr.contains("[]")
        || expr.contains("<>")
        || expr.contains("~>")
        || expr.contains("WF_")
        || expr.contains("SF_")
        || expr.contains("\\AA")
        || expr.contains("\\EE")
    {
        return true;
    }
    // Detect `[ ... ]_subscript` action-box forms. Scan for a `]` immediately
    // followed by `_` whose matching `[` is at the same bracket depth — this is
    // the `[A]_v` syntax, distinct from record/function bracketing.
    let bytes = expr.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b']' && i + 1 < bytes.len() && bytes[i + 1] == b'_' {
            return true;
        }
    }
    false
}

/// Find the index of a top-level operator (not inside parentheses)
fn find_top_level_operator(expr: &str, op: &str) -> Option<usize> {
    let mut depth = 0;
    let chars: Vec<char> = expr.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }

        if depth == 0 && expr[i..].starts_with(op) {
            return Some(i);
        }
    }

    None
}

/// Find the index of the first top-level `=>` implication token (not inside
/// parentheses/brackets/braces), skipping the `<=>` equivalence operator.
///
/// TLA+ implication is `=>`. We must not confuse it with:
///   * `<=>` (equivalence) — a `=>` immediately preceded by `<`.
///   * `>=` / `=<` comparisons — these do not contain the two-char `=>`
///     sequence, so a plain substring scan for `=>` already excludes them
///     (`>=` is `>` then `=`; `=<` is `=` then `<`).
///
/// Returns the byte index of the `=` in the matched `=>`.
fn find_top_level_implies(expr: &str) -> Option<usize> {
    let bytes = expr.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'=' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                // Skip `<=>` (equivalence): the `=` is preceded by `<`.
                if i > 0 && bytes[i - 1] == b'<' {
                    i += 2;
                    continue;
                }
                return Some(i);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Check if parentheses/brackets are balanced
fn is_balanced(expr: &str) -> bool {
    let mut depth = 0;
    for ch in expr.chars() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
}

/// Parse temporal quantifier expression: "var: formula" -> (var, domain, formula)
///
/// Examples:
/// - "\AA n: (n \in Node) /\ <>(n \in Visited)" -> ("n", "Node", "(n \in Node) /\ <>(n \in Visited)")
/// - "\EE n: (n \in Node) => []P" -> ("n", "Node", "(n \in Node) => []P")
///
/// The domain is extracted from the first conjunct if it's a membership test,
/// otherwise the full formula is kept as-is and domain is set to a placeholder.
fn parse_temporal_quantifier(input: &str) -> Option<(String, String, String)> {
    let input = input.trim();

    // Find the colon that separates variable from formula
    let colon_idx = find_top_level_operator(input, ":")?;

    let var_part = input[..colon_idx].trim();
    let formula_part = input[colon_idx + 1..].trim();

    // Extract variable name (simple identifier)
    let var = var_part.to_string();

    // Try to extract domain from the formula
    // Look for patterns like "(n \in Domain)" at the start
    let domain =
        extract_domain_from_formula(&var, formula_part).unwrap_or_else(|| "Any".to_string());

    Some((var, domain, formula_part.to_string()))
}

/// Extract domain from a formula like "(n \in Node) /\ P" -> Some("Node")
fn extract_domain_from_formula(var: &str, formula: &str) -> Option<String> {
    let trimmed = formula.trim();

    // Strip outer parens if present
    let inner = if trimmed.starts_with('(') {
        trimmed.trim_start_matches('(').trim_end_matches(')').trim()
    } else {
        trimmed
    };

    // Look for "var \in Domain" at the start (before /\, =>, etc.)
    let membership_pattern = format!("{} \\in ", var);
    if let Some(start_idx) = inner.find(&membership_pattern) {
        if start_idx < 10 {
            // Should be near the start
            let after_in = &inner[start_idx + membership_pattern.len()..];

            // Extract domain name (up to next operator or paren)
            let domain_end = after_in
                .find(|c: char| matches!(c, ')' | '/' | '\\' | '=' | '~' | '<' | '>' | '['))
                .unwrap_or(after_in.len());

            let domain = after_in[..domain_end].trim();
            if !domain.is_empty() {
                return Some(domain.to_string());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_always() {
        let formula = TemporalFormula::parse("[] x > 0").unwrap();
        assert!(matches!(formula, TemporalFormula::Always(_)));
        assert!(formula.is_safety_property());
        assert!(!formula.is_liveness_property());
    }

    #[test]
    fn test_parse_eventually() {
        let formula = TemporalFormula::parse("<> done = TRUE").unwrap();
        assert!(matches!(formula, TemporalFormula::Eventually(_)));
        assert!(!formula.is_safety_property());
        assert!(formula.is_liveness_property());
    }

    #[test]
    fn test_parse_infinitely_often() {
        let formula = TemporalFormula::parse("[]<> Tick").unwrap();
        assert!(matches!(formula, TemporalFormula::InfinitelyOften(_)));
        assert!(formula.is_liveness_property());
    }

    #[test]
    fn test_parse_leads_to() {
        let formula = TemporalFormula::parse("Request ~> Response").unwrap();
        assert!(matches!(formula, TemporalFormula::LeadsTo(_, _)));
        assert!(formula.is_liveness_property());
    }

    #[test]
    fn test_parse_weak_fairness() {
        let formula = TemporalFormula::parse("WF_vars(Next)").unwrap();
        assert!(matches!(formula, TemporalFormula::WeakFairness { .. }));
        assert!(formula.is_liveness_property());
    }

    #[test]
    fn test_parse_conjunction() {
        let formula = TemporalFormula::parse("[] P /\\ <> Q").unwrap();
        assert!(matches!(formula, TemporalFormula::And(_, _)));
        // Mixed: safety + liveness = liveness
        assert!(formula.is_liveness_property());
    }

    #[test]
    fn test_box_over_quantified_disjunction_stays_state_predicate() {
        // AC1 shape: `[] \A i,j : \/ A \/ B` must parse as
        // Always(StatePredicate(..)) — the unparenthesized top-level `\/`
        // inside the quantifier body must NOT be split into a temporal Or.
        let f = TemporalFormula::parse(
            "[] \\A i, j \\in participants :\n  \\/ p[i].decision # commit\n  \\/ p[j].decision # abort",
        )
        .unwrap();
        match &f {
            TemporalFormula::Always(inner) => {
                assert!(matches!(inner.as_ref(), TemporalFormula::StatePredicate(_)));
            }
            other => panic!("expected Always(StatePredicate), got {:?}", other),
        }
        assert!(f.is_safety_property());
        assert!(!f.is_liveness_property());
    }

    #[test]
    fn test_pure_state_predicate_not_decomposed() {
        // A property that is a plain state predicate with top-level `/\`
        // (no temporal operator) must stay a single StatePredicate.
        let f = TemporalFormula::parse("a > 0 /\\ b > 0").unwrap();
        assert!(matches!(f, TemporalFormula::StatePredicate(_)));
    }

    #[test]
    fn test_action_box_still_temporal() {
        // `[][Next]_vars` must remain temporal (not a bare state predicate).
        let f = TemporalFormula::parse("[][Next]_vars").unwrap();
        assert!(matches!(f, TemporalFormula::Always(_)));
        assert!(contains_temporal_operator("[Next]_vars"));
        assert!(contains_temporal_operator("x /\\ [][Next]_vars"));
    }

    #[test]
    fn test_mixed_temporal_still_splits() {
        // `[] P /\ <> Q` has genuine temporal operators on both sides and must
        // still decompose into And(Always, Eventually).
        let f = TemporalFormula::parse("[] P /\\ <> Q").unwrap();
        assert!(matches!(f, TemporalFormula::And(_, _)));
        assert!(f.is_liveness_property());
    }

    #[test]
    fn test_parse_temporal_forall() {
        let formula = TemporalFormula::parse("\\AA n: (n \\in Node) => [](n \\in Node)").unwrap();
        assert!(matches!(formula, TemporalFormula::TemporalForAll { .. }));

        if let TemporalFormula::TemporalForAll {
            var,
            domain,
            formula: inner,
        } = formula
        {
            assert_eq!(var, "n");
            assert_eq!(domain, "Node");
            // The body `(n \in Node) => [](n \in Node)` is a temporal
            // implication (RHS is `[]...`), so it now decomposes into
            // Implies(StatePredicate, Always(StatePredicate)). Previously the
            // `=>` was unhandled and the whole body stayed an (unevaluable)
            // StatePredicate; the decomposed form is more correct.
            match *inner {
                TemporalFormula::Implies(ref ante, ref cons) => {
                    assert!(matches!(ante.as_ref(), TemporalFormula::StatePredicate(_)));
                    assert!(matches!(cons.as_ref(), TemporalFormula::Always(_)));
                }
                other => panic!("expected Implies body, got {:?}", other),
            }
        } else {
            panic!("Expected TemporalForAll");
        }
    }

    #[test]
    fn test_parse_temporal_exists() {
        let formula = TemporalFormula::parse("\\EE n: (n \\in Node) /\\ <>(visited)").unwrap();
        assert!(matches!(formula, TemporalFormula::TemporalExists { .. }));
        assert!(formula.is_liveness_property());

        if let TemporalFormula::TemporalExists {
            var,
            domain,
            formula: inner,
        } = formula
        {
            assert_eq!(var, "n");
            assert_eq!(domain, "Node");
            assert!(matches!(*inner, TemporalFormula::And(_, _)));
        } else {
            panic!("Expected TemporalExists");
        }
    }

    #[test]
    fn test_box_implies_eventually_always_is_liveness() {
        // The RealTime shape: `[]((now # 4) => <>[](now # 4))` must parse to
        // Always(Implies(StatePredicate, EventuallyAlways(StatePredicate))) and
        // classify as LIVENESS so the graph post-processing pass runs.
        let f = TemporalFormula::parse("[]((now # 4) => <>[](now # 4))").unwrap();
        match &f {
            TemporalFormula::Always(inner) => match inner.as_ref() {
                TemporalFormula::Implies(p, q) => {
                    assert!(
                        matches!(p.as_ref(), TemporalFormula::StatePredicate(_)),
                        "antecedent should be a state predicate, got {:?}",
                        p
                    );
                    assert!(
                        matches!(q.as_ref(), TemporalFormula::EventuallyAlways(_)),
                        "consequent should be <>[]Q, got {:?}",
                        q
                    );
                }
                other => panic!("expected Implies inside Always, got {:?}", other),
            },
            other => panic!("expected Always(Implies(..)), got {:?}", other),
        }
        assert!(f.is_liveness_property(), "must classify as liveness");
        assert!(!f.is_safety_property());
    }

    #[test]
    fn test_state_pred_implies_stays_atomic() {
        // `[](A => B)` where A and B are pure state predicates must NOT be
        // decomposed into a temporal Implies — it stays Always(StatePredicate)
        // so box-safety lowering (#127) still fires.
        let f = TemporalFormula::parse("[](x > 0 => y > 0)").unwrap();
        match &f {
            TemporalFormula::Always(inner) => {
                assert!(
                    matches!(inner.as_ref(), TemporalFormula::StatePredicate(_)),
                    "state-pred implication body must stay a single StatePredicate, got {:?}",
                    inner
                );
            }
            other => panic!("expected Always(StatePredicate), got {:?}", other),
        }
        assert!(f.is_safety_property());
        assert!(!f.is_liveness_property());
    }

    #[test]
    fn test_bare_state_pred_implies_not_temporal() {
        // A top-level `A => B` with no temporal operator anywhere stays a
        // single StatePredicate (handled by the non-temporal bailout).
        let f = TemporalFormula::parse("a = 1 => b = 2").unwrap();
        assert!(matches!(f, TemporalFormula::StatePredicate(_)));
    }

    #[test]
    fn test_equivalence_not_split_as_implies() {
        // `<=>` must not be split on its embedded `=>`. `P <=> <>Q` is temporal
        // (RHS has `<>`) but the top-level connective is equivalence, not
        // implication — find_top_level_implies must skip the `<=>`.
        assert_eq!(find_top_level_implies("P <=> Q"), None);
        assert_eq!(find_top_level_implies("a >= 1 /\\ b <= 2"), None);
        // A genuine `=>` is found.
        assert_eq!(find_top_level_implies("P => Q"), Some(2));
        // `=>` inside parens is not top-level.
        assert_eq!(find_top_level_implies("(P => Q) /\\ R"), None);
    }

    #[test]
    fn test_implies_left_temporal_builds_implies() {
        // `<>P => []Q` — antecedent temporal — builds an Implies node.
        // Classification: liveness iff the CONSEQUENT is liveness. Here the
        // consequent `[]Q` is safety, so the whole implication is NOT a pure
        // liveness property (it is `[]¬P \/ []Q`, a safety-ish disjunction).
        let f = TemporalFormula::parse("<>P => []Q").unwrap();
        assert!(matches!(f, TemporalFormula::Implies(_, _)));
        assert!(!f.is_liveness_property());

        // But `[]P => <>Q` (consequent is liveness) IS classified liveness.
        let g = TemporalFormula::parse("[]P => <>Q").unwrap();
        assert!(matches!(g, TemporalFormula::Implies(_, _)));
        assert!(g.is_liveness_property());
    }

    #[test]
    fn test_parse_temporal_exists_with_eventually() {
        let formula = TemporalFormula::parse("\\EE n: (n \\in 1..10) /\\ <>(x = n)").unwrap();
        assert!(matches!(formula, TemporalFormula::TemporalExists { .. }));
        assert!(formula.is_liveness_property());
    }
}
