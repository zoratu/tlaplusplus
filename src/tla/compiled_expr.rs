//! Compiled TLA+ expressions for fast evaluation
//!
//! Instead of parsing expression strings on every evaluation, we compile them
//! once to an AST and then evaluate the AST. This eliminates:
//! - String parsing overhead
//! - String allocation during evaluation
//! - Repeated pattern matching on string prefixes
//!
//! Benchmarks show this can provide 3-5x speedup on expression-heavy workloads.

use std::sync::Arc;

/// Compiled TLA+ expression - parsed once, evaluated many times
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    // Literals
    Bool(bool),
    Int(i64),
    String(String),
    ModelValue(String),

    // Variable/identifier reference
    Var(String),

    // Primed variable reference (e.g., x' - the next-state value)
    PrimedVar(String),

    // Self-reference in EXCEPT expressions (the @ operator)
    // Represents the original value at the current update path
    SelfRef,

    // Logical operators (short-circuit evaluation)
    And(Vec<CompiledExpr>),
    Or(Vec<CompiledExpr>),
    Not(Box<CompiledExpr>),
    Implies(Box<CompiledExpr>, Box<CompiledExpr>),

    // Comparison operators
    Eq(Box<CompiledExpr>, Box<CompiledExpr>),
    Neq(Box<CompiledExpr>, Box<CompiledExpr>),
    Lt(Box<CompiledExpr>, Box<CompiledExpr>),
    Le(Box<CompiledExpr>, Box<CompiledExpr>),
    Gt(Box<CompiledExpr>, Box<CompiledExpr>),
    Ge(Box<CompiledExpr>, Box<CompiledExpr>),

    // Set membership
    In(Box<CompiledExpr>, Box<CompiledExpr>),
    NotIn(Box<CompiledExpr>, Box<CompiledExpr>),

    // Arithmetic
    Add(Box<CompiledExpr>, Box<CompiledExpr>),
    Sub(Box<CompiledExpr>, Box<CompiledExpr>),
    Mul(Box<CompiledExpr>, Box<CompiledExpr>),
    Pow(Box<CompiledExpr>, Box<CompiledExpr>),
    Div(Box<CompiledExpr>, Box<CompiledExpr>),
    Mod(Box<CompiledExpr>, Box<CompiledExpr>),
    Neg(Box<CompiledExpr>),

    // Set operations
    SetLiteral(Vec<CompiledExpr>),
    SetRange(Box<CompiledExpr>, Box<CompiledExpr>),
    Union(Box<CompiledExpr>, Box<CompiledExpr>),
    Intersect(Box<CompiledExpr>, Box<CompiledExpr>),
    SetMinus(Box<CompiledExpr>, Box<CompiledExpr>),
    /// Cartesian product: \X or \times
    CartesianProduct(Box<CompiledExpr>, Box<CompiledExpr>),
    Subset(Box<CompiledExpr>, Box<CompiledExpr>),
    Cardinality(Box<CompiledExpr>),
    PowerSet(Box<CompiledExpr>),

    // Sequence operations
    SeqLiteral(Vec<CompiledExpr>),
    Head(Box<CompiledExpr>),
    Tail(Box<CompiledExpr>),
    Append(Box<CompiledExpr>, Box<CompiledExpr>),
    Concat(Box<CompiledExpr>, Box<CompiledExpr>),
    Len(Box<CompiledExpr>),
    SubSeq(Box<CompiledExpr>, Box<CompiledExpr>, Box<CompiledExpr>),

    // Record operations
    RecordLiteral(Vec<(String, CompiledExpr)>),
    RecordAccess(Box<CompiledExpr>, String),
    /// Record set construction: [field: Set, field2: Set, ...]
    /// Represents the Cartesian product of all field/set pairs as records.
    RecordSet(Vec<(String, CompiledExpr)>),

    // Function operations
    FuncLiteral(Vec<(CompiledExpr, CompiledExpr)>),
    FuncApply(Box<CompiledExpr>, Vec<CompiledExpr>),
    FuncExcept(Box<CompiledExpr>, Vec<(Vec<CompiledExpr>, CompiledExpr)>),
    Domain(Box<CompiledExpr>),
    /// TLC module: Function pair constructor (a :> b creates {a -> b})
    FuncPair(Box<CompiledExpr>, Box<CompiledExpr>),
    /// TLC module: Function override (f @@ g merges functions, g takes precedence)
    FuncOverride(Box<CompiledExpr>, Box<CompiledExpr>),

    // Control flow
    If {
        cond: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Box<CompiledExpr>,
    },
    Case {
        arms: Vec<(CompiledExpr, CompiledExpr)>,
        other: Option<Box<CompiledExpr>>,
    },
    Let {
        bindings: Vec<(String, CompiledExpr)>,
        body: Box<CompiledExpr>,
    },

    // Quantifiers
    Exists {
        var: String,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    Forall {
        var: String,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    Choose {
        var: String,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },

    // Set comprehension: {expr : x \in S}
    SetComprehension {
        var: String,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
        filter: Option<Box<CompiledExpr>>,
    },

    // Function construction: [x \in S |-> expr]
    FuncConstruct {
        var: String,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },

    // Function set: [Domain -> Range] - set of all functions from Domain to Range
    FunctionSet {
        domain: Box<CompiledExpr>,
        range: Box<CompiledExpr>,
    },

    // Operator/definition call
    OpCall {
        name: String,
        args: Vec<CompiledExpr>,
    },

    // Built-in sets
    NatSet,
    IntSet,
    BooleanSet,

    // Lambda (for higher-order operators)
    Lambda {
        params: Vec<String>,
        body: Box<CompiledExpr>,
        body_text: String, // Original text for TlaValue::Lambda conversion
    },

    // Fallback: unparsed expression (for complex cases we haven't compiled yet)
    Unparsed(String),
}

impl CompiledExpr {
    /// Check if this expression is fully compiled (no Unparsed nodes)
    pub fn is_fully_compiled(&self) -> bool {
        match self {
            CompiledExpr::Unparsed(_) => false,
            CompiledExpr::And(exprs) | CompiledExpr::Or(exprs) => {
                exprs.iter().all(|e| e.is_fully_compiled())
            }
            CompiledExpr::Not(e) | CompiledExpr::Neg(e) => e.is_fully_compiled(),
            CompiledExpr::Implies(a, b)
            | CompiledExpr::Eq(a, b)
            | CompiledExpr::Neq(a, b)
            | CompiledExpr::Lt(a, b)
            | CompiledExpr::Le(a, b)
            | CompiledExpr::Gt(a, b)
            | CompiledExpr::Ge(a, b)
            | CompiledExpr::In(a, b)
            | CompiledExpr::NotIn(a, b)
            | CompiledExpr::Add(a, b)
            | CompiledExpr::Sub(a, b)
            | CompiledExpr::Mul(a, b)
            | CompiledExpr::Pow(a, b)
            | CompiledExpr::Div(a, b)
            | CompiledExpr::Mod(a, b)
            | CompiledExpr::Union(a, b)
            | CompiledExpr::Intersect(a, b)
            | CompiledExpr::SetMinus(a, b)
            | CompiledExpr::CartesianProduct(a, b)
            | CompiledExpr::Subset(a, b)
            | CompiledExpr::SetRange(a, b)
            | CompiledExpr::Append(a, b)
            | CompiledExpr::Concat(a, b)
            | CompiledExpr::FuncPair(a, b)
            | CompiledExpr::FuncOverride(a, b)
            | CompiledExpr::FunctionSet {
                domain: a,
                range: b,
            } => a.is_fully_compiled() && b.is_fully_compiled(),
            CompiledExpr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                cond.is_fully_compiled()
                    && then_branch.is_fully_compiled()
                    && else_branch.is_fully_compiled()
            }
            CompiledExpr::Let { bindings, body } => {
                bindings.iter().all(|(_, e)| e.is_fully_compiled()) && body.is_fully_compiled()
            }
            CompiledExpr::Exists { domain, body, .. }
            | CompiledExpr::Forall { domain, body, .. }
            | CompiledExpr::Choose { domain, body, .. }
            | CompiledExpr::FuncConstruct { domain, body, .. } => {
                domain.is_fully_compiled() && body.is_fully_compiled()
            }
            CompiledExpr::SetComprehension {
                domain,
                body,
                filter,
                ..
            } => {
                domain.is_fully_compiled()
                    && body.is_fully_compiled()
                    && filter.as_ref().map_or(true, |f| f.is_fully_compiled())
            }
            CompiledExpr::SetLiteral(exprs) | CompiledExpr::SeqLiteral(exprs) => {
                exprs.iter().all(|e| e.is_fully_compiled())
            }
            CompiledExpr::RecordLiteral(fields) | CompiledExpr::RecordSet(fields) => {
                fields.iter().all(|(_, e)| e.is_fully_compiled())
            }
            CompiledExpr::FuncLiteral(entries) => entries
                .iter()
                .all(|(k, v)| k.is_fully_compiled() && v.is_fully_compiled()),
            CompiledExpr::OpCall { args, .. } => args.iter().all(|e| e.is_fully_compiled()),
            CompiledExpr::FuncApply(f, args) => {
                f.is_fully_compiled() && args.iter().all(|e| e.is_fully_compiled())
            }
            CompiledExpr::Case { arms, other } => {
                arms.iter()
                    .all(|(c, e)| c.is_fully_compiled() && e.is_fully_compiled())
                    && other.as_ref().map_or(true, |e| e.is_fully_compiled())
            }
            CompiledExpr::Lambda { body, .. } => body.is_fully_compiled(),
            CompiledExpr::FuncExcept(base, updates) => {
                base.is_fully_compiled()
                    && updates.iter().all(|(path, val)| {
                        path.iter().all(|p| p.is_fully_compiled()) && val.is_fully_compiled()
                    })
            }
            CompiledExpr::SubSeq(s, a, b) => {
                s.is_fully_compiled() && a.is_fully_compiled() && b.is_fully_compiled()
            }
            CompiledExpr::RecordAccess(e, _)
            | CompiledExpr::Cardinality(e)
            | CompiledExpr::PowerSet(e)
            | CompiledExpr::Head(e)
            | CompiledExpr::Tail(e)
            | CompiledExpr::Len(e)
            | CompiledExpr::Domain(e) => e.is_fully_compiled(),
            // Terminals are always compiled
            CompiledExpr::Bool(_)
            | CompiledExpr::Int(_)
            | CompiledExpr::String(_)
            | CompiledExpr::ModelValue(_)
            | CompiledExpr::Var(_)
            | CompiledExpr::PrimedVar(_)
            | CompiledExpr::SelfRef
            | CompiledExpr::NatSet
            | CompiledExpr::IntSet
            | CompiledExpr::BooleanSet => true,
        }
    }
}

/// Expression compiler - caches compiled expressions
pub struct ExprCompiler {
    cache: std::collections::HashMap<String, Arc<CompiledExpr>>,
}

impl ExprCompiler {
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }

    /// Compile an expression, using cache if available
    pub fn compile(&mut self, expr: &str) -> Arc<CompiledExpr> {
        if let Some(cached) = self.cache.get(expr) {
            return Arc::clone(cached);
        }

        let compiled = Arc::new(compile_expr(expr));
        self.cache.insert(expr.to_string(), Arc::clone(&compiled));
        compiled
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let total = self.cache.len();
        let compiled = self
            .cache
            .values()
            .filter(|e| e.is_fully_compiled())
            .count();
        (compiled, total)
    }
}

impl Default for ExprCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a string literal from the start of an expression
/// Returns (string_content, rest_of_expression) if successful
fn parse_string_literal(expr: &str) -> Option<(String, &str)> {
    if !expr.starts_with('"') {
        return None;
    }

    let mut out = String::new();
    let mut escaped = false;
    let mut i = 1; // Skip opening quote

    let bytes = expr.as_bytes();
    while i < bytes.len() {
        let ch = bytes[i] as char;

        if escaped {
            // Handle escape sequences
            match ch {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                _ => {
                    out.push('\\');
                    out.push(ch);
                }
            }
            escaped = false;
            i += 1;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            i += 1;
            continue;
        }

        if ch == '"' {
            // Found closing quote
            return Some((out, &expr[i + 1..]));
        }

        out.push(ch);
        i += 1;
    }

    // Unterminated string
    None
}

/// Compile a TLA+ expression string to a CompiledExpr
pub fn compile_expr(expr: &str) -> CompiledExpr {
    let expr = expr.trim();

    if expr.is_empty() {
        return CompiledExpr::Unparsed(expr.to_string());
    }

    // Strip outer parentheses
    let expr = strip_outer_parens(expr);

    // Literals
    if expr == "TRUE" {
        return CompiledExpr::Bool(true);
    }
    if expr == "FALSE" {
        return CompiledExpr::Bool(false);
    }
    if expr == "Nat" {
        return CompiledExpr::NatSet;
    }
    if expr == "Int" {
        return CompiledExpr::IntSet;
    }
    if expr == "BOOLEAN" {
        return CompiledExpr::BooleanSet;
    }

    // Integer literal
    if let Ok(n) = expr.parse::<i64>() {
        return CompiledExpr::Int(n);
    }

    // Negative integer
    if expr.starts_with('-') {
        if let Ok(n) = expr[1..].trim().parse::<i64>() {
            return CompiledExpr::Int(-n);
        }
    }

    // String literal - must properly parse to find the matching closing quote
    // The naive check `starts_with('"') && ends_with('"')` is wrong because
    // it matches expressions like `"hello" = "hello"` as a single string
    if expr.starts_with('"') {
        if let Some((string_content, rest)) = parse_string_literal(expr) {
            if rest.is_empty() {
                return CompiledExpr::String(string_content);
            }
        }
    }

    // Self-reference in EXCEPT (the @ operator)
    if expr == "@" {
        return CompiledExpr::SelfRef;
    }

    // Primed variable (e.g., x')
    if expr.ends_with('\'') {
        let base = &expr[..expr.len() - 1];
        if is_identifier(base) {
            return CompiledExpr::PrimedVar(base.to_string());
        }
    }

    // Simple identifier (variable reference)
    if is_identifier(expr) {
        return CompiledExpr::Var(expr.to_string());
    }

    // Quantifiers MUST be parsed FIRST before logical operators
    // because the quantifier body extends to the end of the expression
    // e.g., \A x \in S : P(x) /\ Q(x) should have body = "P(x) /\ Q(x)"
    if expr.starts_with("\\E ") || expr.starts_with("\\E(") {
        if let Some(exists) = try_parse_exists(expr) {
            return exists;
        }
    }
    if expr.starts_with("\\A ") || expr.starts_with("\\A(") {
        if let Some(forall) = try_parse_forall(expr) {
            return forall;
        }
    }

    // CHOOSE also binds loosely
    if expr.starts_with("CHOOSE ") {
        if let Some(choose) = try_parse_choose(expr) {
            return choose;
        }
    }

    // IF-THEN-ELSE (must check before logical operators)
    if expr.starts_with("IF ") || expr.starts_with("IF(") {
        if let Some(if_expr) = try_parse_if(expr) {
            return if_expr;
        }
    }

    // LET-IN (must check before logical operators)
    if expr.starts_with("LET ") || expr.starts_with("LET\n") {
        if let Some(let_expr) = try_parse_let(expr) {
            return let_expr;
        }
    }

    // LAMBDA (must check before logical operators)
    // LAMBDA binds loosely - everything after the colon is the body
    if expr.starts_with("LAMBDA ") {
        if let Some(lambda) = try_parse_lambda(expr) {
            return lambda;
        }
    }

    // UNCHANGED in expression/guard context.
    // When UNCHANGED appears inside a disjunction evaluated as a guard (e.g.,
    // \/ Action1(self) \/ UNCHANGED vars), the action IR layer handles the
    // actual variable-preservation semantics. In expression context, UNCHANGED
    // represents an always-enabled stuttering step, so we compile it as TRUE.
    if expr.starts_with("UNCHANGED ") || expr == "UNCHANGED" {
        return CompiledExpr::Bool(true);
    }

    // Logical operators (in precedence order)

    // Implication: =>
    if let Some((left, right)) = split_binary_op(expr, "=>") {
        return CompiledExpr::Implies(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Disjunction: \/
    let or_parts = split_top_level(expr, "\\/");
    if or_parts.len() > 1 {
        return CompiledExpr::Or(or_parts.into_iter().map(|s| compile_expr(&s)).collect());
    } else if or_parts.len() == 1 && expr.trim_start().starts_with("\\/") {
        // Expression starts with \/ but has only one disjunct - strip the leading \/
        // This handles cases like "\/ single_expr" in quantifier bodies
        return compile_expr(&or_parts[0]);
    }

    // Conjunction: /\
    let and_parts = split_top_level(expr, "/\\");
    if and_parts.len() > 1 {
        return CompiledExpr::And(and_parts.into_iter().map(|s| compile_expr(&s)).collect());
    } else if and_parts.len() == 1 && expr.trim_start().starts_with("/\\") {
        // Expression starts with /\ but has only one conjunct - strip the leading /\
        // This handles cases like "/\ single_expr" in quantifier bodies
        return compile_expr(&and_parts[0]);
    }

    // Negation: ~
    if let Some(rest) = expr.strip_prefix('~') {
        return CompiledExpr::Not(Box::new(compile_expr(rest.trim())));
    }

    // Comparison operators
    if let Some((left, op, right)) = split_comparison(expr) {
        if matches!(op, "\\in" | "\\notin") && is_bracket_type_expr(right) {
            return CompiledExpr::Unparsed(expr.to_string());
        }

        let left_expr = Box::new(compile_expr(left));
        let right_expr = Box::new(compile_expr(right));
        return match op {
            "=" => CompiledExpr::Eq(left_expr, right_expr),
            "/=" | "#" => CompiledExpr::Neq(left_expr, right_expr),
            "<" => CompiledExpr::Lt(left_expr, right_expr),
            "<=" | "=<" | "\\leq" => CompiledExpr::Le(left_expr, right_expr),
            ">" => CompiledExpr::Gt(left_expr, right_expr),
            ">=" | "\\geq" => CompiledExpr::Ge(left_expr, right_expr),
            "\\in" => CompiledExpr::In(left_expr, right_expr),
            "\\notin" => CompiledExpr::NotIn(left_expr, right_expr),
            "\\subseteq" => CompiledExpr::Subset(left_expr, right_expr),
            _ => CompiledExpr::Unparsed(expr.to_string()),
        };
    }

    // Arithmetic operators
    if let Some((left, right)) = split_binary_op(expr, "+") {
        return CompiledExpr::Add(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "-") {
        // Be careful not to match negative numbers
        if !left.is_empty() {
            return CompiledExpr::Sub(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
        }
    }
    if let Some((left, right)) = split_binary_op(expr, "*") {
        return CompiledExpr::Mul(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "\\div") {
        return CompiledExpr::Div(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "%") {
        return CompiledExpr::Mod(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "^") {
        return CompiledExpr::Pow(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // TLC module: Function override operator @@
    // f @@ g merges two functions, with g taking precedence on overlapping keys
    // Lower precedence than :> so parse it first
    if let Some((left, right)) = split_binary_op(expr, "@@") {
        return CompiledExpr::FuncOverride(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }

    // Set operations
    if let Some((left, right)) = split_binary_op(expr, "\\union") {
        return CompiledExpr::Union(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "\\cup") {
        return CompiledExpr::Union(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "\\intersect") {
        return CompiledExpr::Intersect(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }
    if let Some((left, right)) = split_binary_op(expr, "\\cap") {
        return CompiledExpr::Intersect(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }
    // Handle \o (concatenation) BEFORE \ (set minus) since \ would match \o
    if let Some((left, right)) = split_binary_op(expr, "\\o") {
        return CompiledExpr::Concat(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    // Handle cartesian product: \X and \times are synonyms
    // Must be before \ (set minus) since \ would match \X or \times
    if let Some((left, right)) = split_binary_op(expr, "\\X") {
        return CompiledExpr::CartesianProduct(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }
    if let Some((left, right)) = split_binary_op(expr, "\\times") {
        return CompiledExpr::CartesianProduct(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }
    if let Some((left, right)) = split_binary_op(expr, "\\") {
        // Only treat as set minus if right side doesn't start with a known keyword
        // (this avoids incorrectly splitting \union, \cup, \cap, \intersect, \div, etc.)
        let right_trimmed = right.trim();
        if !right_trimmed.starts_with("union")
            && !right_trimmed.starts_with("cup")
            && !right_trimmed.starts_with("cap")
            && !right_trimmed.starts_with("intersect")
            && !right_trimmed.starts_with("div")
            && !right_trimmed.starts_with("in")
            && !right_trimmed.starts_with("notin")
            && !right_trimmed.starts_with("subseteq")
            && !right_trimmed.starts_with("E")
            && !right_trimmed.starts_with("A")
        {
            return CompiledExpr::SetMinus(
                Box::new(compile_expr(left)),
                Box::new(compile_expr(right)),
            );
        }
    }

    // TLC module: Function pair constructor :>
    // a :> b creates a function mapping a to b (single mapping {a -> b})
    // Higher precedence than @@, so parse it after set operations
    if let Some((left, right)) = split_binary_op(expr, ":>") {
        return CompiledExpr::FuncPair(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Set/sequence range: a..b
    if let Some((left, right)) = split_binary_op(expr, "..") {
        return CompiledExpr::SetRange(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Set literal: {a, b, c}
    if expr.starts_with('{') && expr.ends_with('}') {
        let inner = &expr[1..expr.len() - 1];
        if inner.is_empty() {
            return CompiledExpr::SetLiteral(vec![]);
        }

        // Check for set comprehension: {expr : x \in S} or {x \in S : filter}
        if let Some(comp) = try_parse_set_comprehension(inner) {
            return comp;
        }

        // Simple set literal
        let elements = split_top_level(inner, ",");
        return CompiledExpr::SetLiteral(elements.into_iter().map(|s| compile_expr(&s)).collect());
    }

    // Sequence literal: <<a, b, c>>
    if expr.starts_with("<<") && expr.ends_with(">>") {
        let inner = &expr[2..expr.len() - 2];
        if inner.is_empty() {
            return CompiledExpr::SeqLiteral(vec![]);
        }
        let elements = split_top_level(inner, ",");
        return CompiledExpr::SeqLiteral(elements.into_iter().map(|s| compile_expr(&s)).collect());
    }

    // Record literal: [a |-> 1, b |-> 2]
    if expr.starts_with('[') && expr.ends_with(']') {
        let inner = &expr[1..expr.len() - 1];

        // Check for function construction: [x \in S |-> expr]
        if let Some(func) = try_parse_func_construct(inner) {
            return func;
        }

        // Check for function set: [Domain -> Range]
        if let Some(func_set) = try_parse_function_set(inner) {
            return func_set;
        }

        // Check for EXCEPT: [f EXCEPT ![key] = val] or [f EXCEPT ![k1][k2] = val]
        if let Some(except_expr) = try_parse_except(inner) {
            return except_expr;
        }

        // Record literal
        if inner.contains("|->") {
            let entries = split_top_level(inner, ",");
            let mut fields = Vec::new();
            for entry in entries {
                if let Some((field, value)) = entry.split_once("|->") {
                    fields.push((field.trim().to_string(), compile_expr(value.trim())));
                }
            }
            if !fields.is_empty() {
                return CompiledExpr::RecordLiteral(fields);
            }
        }

        // Record set construction: [field: Set, field2: Set, ...]
        // This creates the Cartesian product of all field/set pairs as records.
        if let Some(record_set) = try_parse_record_set(inner) {
            return record_set;
        }
    }

    // Record access: r.field
    if let Some(dot_idx) = find_record_access_dot(expr) {
        let base = &expr[..dot_idx];
        let field = &expr[dot_idx + 1..];
        if is_identifier(field) {
            return CompiledExpr::RecordAccess(Box::new(compile_expr(base)), field.to_string());
        }
    }

    // Prefix operators SUBSET and DOMAIN must be parsed BEFORE function application
    // to handle cases like "DOMAIN f[x]" correctly as Domain(FuncApply(f, [x]))
    // rather than FuncApply(Domain(f), [x])
    if let Some(inner) = expr.strip_prefix("SUBSET ") {
        return CompiledExpr::PowerSet(Box::new(compile_expr(inner)));
    }
    if let Some(inner) = expr.strip_prefix("DOMAIN ") {
        return CompiledExpr::Domain(Box::new(compile_expr(inner)));
    }

    // Function application: f[x] or f[x, y]
    if let Some((func, args)) = parse_func_apply(expr) {
        return CompiledExpr::FuncApply(
            Box::new(compile_expr(func)),
            args.into_iter().map(|s| compile_expr(&s)).collect(),
        );
    }

    // Operator call: Op(a, b)
    if let Some((name, args)) = parse_op_call(expr) {
        return CompiledExpr::OpCall {
            name: name.to_string(),
            args: args.into_iter().map(|s| compile_expr(&s)).collect(),
        };
    }

    // Built-in functions
    if let Some(inner) = expr
        .strip_prefix("Cardinality(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return CompiledExpr::Cardinality(Box::new(compile_expr(inner)));
    }
    if let Some(inner) = expr.strip_prefix("Head(").and_then(|s| s.strip_suffix(')')) {
        return CompiledExpr::Head(Box::new(compile_expr(inner)));
    }
    if let Some(inner) = expr.strip_prefix("Tail(").and_then(|s| s.strip_suffix(')')) {
        return CompiledExpr::Tail(Box::new(compile_expr(inner)));
    }
    if let Some(inner) = expr.strip_prefix("Len(").and_then(|s| s.strip_suffix(')')) {
        return CompiledExpr::Len(Box::new(compile_expr(inner)));
    }

    // Fallback: unparsed
    if std::env::var("TLAPP_TRACE_UNPARSED").is_ok() {
        eprintln!("UNPARSED: {:?}", expr);
    }
    CompiledExpr::Unparsed(expr.to_string())
}

// Helper functions for parsing

fn strip_outer_parens(expr: &str) -> &str {
    let bytes = expr.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'(' || bytes[bytes.len() - 1] != b')' {
        return expr;
    }

    // Check if parens are balanced
    let mut depth = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 && i < bytes.len() - 1 {
                    return expr; // Closing paren in middle, not wrapping
                }
            }
            _ => {}
        }
    }

    if depth == 0 {
        strip_outer_parens(&expr[1..expr.len() - 1])
    } else {
        expr
    }
}

fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_alphanumeric() || first == '_') {
        return false;
    }

    let mut saw_identifier_marker = first.is_alphabetic() || first == '_';
    for c in chars {
        if !(c.is_alphanumeric() || c == '_') {
            return false;
        }
        saw_identifier_marker |= c.is_alphabetic() || c == '_';
    }

    saw_identifier_marker
}

fn split_top_level(expr: &str, delim: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut let_depth: usize = 0; // Track LET...IN nesting
    let mut if_depth: usize = 0; // Track IF...ELSE nesting
    let mut case_depth: usize = 0; // Track CASE expression nesting
    let mut quantifier_depth: usize = 0; // Track nested quantifier depth (not just boolean!)
    let mut quantifier_base_indent: Option<usize> = None; // Indentation of first quantifier's line
    let mut current_col: usize = 0; // Current column position
    let mut line_indent: usize = 0; // Indentation of current line (spaces at start)
    let mut at_line_start = true; // Are we at the start of a line (counting spaces)?
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
    let delim_chars: Vec<char> = delim.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Track column position and line indentation
        if c == '\n' {
            current_col = 0;
            line_indent = 0;
            at_line_start = true;
        } else {
            if at_line_start {
                if c == ' ' || c == '\t' {
                    line_indent += if c == '\t' { 4 } else { 1 };
                } else {
                    at_line_start = false;
                }
            }
            current_col += 1;
        }

        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
            in_string = !in_string;
            current.push(c);
            i += 1;
            continue;
        }

        if in_string {
            current.push(c);
            i += 1;
            continue;
        }

        // Handle << and >> for sequence literals
        if i + 1 < chars.len() {
            if c == '<' && chars[i + 1] == '<' {
                depth += 1;
                current.push(c);
                current.push(chars[i + 1]);
                i += 2;
                current_col += 1;
                continue;
            }
            if c == '>' && chars[i + 1] == '>' {
                depth -= 1;
                current.push(c);
                current.push(chars[i + 1]);
                i += 2;
                current_col += 1;
                continue;
            }
        }

        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }

        // Check for keywords at word boundary when at bracket top level
        if depth == 0 {
            // Track LET...IN nesting
            // NOTE: We increment let_depth on LET but DO NOT decrement on IN.
            // This is because in TLA+ actions, the pattern:
            //   /\ LET x == ... IN /\ expr1 /\ expr2
            // should treat "expr1 /\ expr2" as the body of the LET, not as
            // top-level conjuncts. The LET expression extends to the end of the
            // current grouping, and we only want to split at conjunctions that
            // are truly at the top level (outside any LET).
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                current.push_str("LET");
                i += 3;
                current_col += 2;
                continue;
            }
            // Just push IN without decrementing - the LET body extends to the end
            if let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
                current.push_str("IN");
                i += 2;
                current_col += 1;
                continue;
            }

            // Track IF...ELSE nesting - don't split inside IF-THEN-ELSE
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                current.push_str("IF");
                i += 2;
                current_col += 1;
                continue;
            }
            if if_depth > 0 && matches_keyword_at(&chars, i, "ELSE") {
                if_depth = if_depth.saturating_sub(1);
                current.push_str("ELSE");
                i += 4;
                current_col += 3;
                continue;
            }

            // Track CASE expressions - don't split inside CASE
            if matches_keyword_at(&chars, i, "CASE") {
                case_depth += 1;
                current.push_str("CASE");
                i += 4;
                current_col += 3;
                continue;
            }
        }

        // Check for quantifier start: \A or \E followed by space
        // These bind loosely - everything after : is part of the quantifier body
        // We track nesting depth to handle nested quantifiers properly
        if depth == 0 && c == '\\' && i + 2 < chars.len() {
            let next = chars[i + 1];
            let after = chars[i + 2];
            if (next == 'A' || next == 'E') && (after.is_whitespace() || after == '(') {
                // Found quantifier start - look for the : that ends the binding part
                // and starts the body
                let mut j = i + 2;
                let mut inner_depth = 0;
                while j < chars.len() {
                    match chars[j] {
                        '(' | '[' | '{' => inner_depth += 1,
                        ')' | ']' | '}' => inner_depth -= 1,
                        ':' if inner_depth == 0 => {
                            // Found the colon - everything after this is quantifier body
                            // INCREMENT depth instead of setting boolean - this handles nesting!
                            quantifier_depth += 1;
                            // Record the INDENTATION of the first quantifier's line
                            // This is used to detect sibling quantifiers at the same indentation level
                            if quantifier_base_indent.is_none() {
                                quantifier_base_indent = Some(line_indent);
                            }
                            // Push everything up to and including the colon
                            let chars_pushed = j - i + 1;
                            for k in i..=j {
                                current.push(chars[k]);
                            }
                            i = j + 1;
                            // Adjust current_col for the characters we just pushed
                            current_col += chars_pushed;
                            break;
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if j < chars.len() {
                    continue;
                }
            }
        }

        let at_top = depth == 0 && let_depth == 0 && if_depth == 0 && case_depth == 0;
        if at_top {
            // Check for delimiter
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with(delim) {
                if quantifier_depth > 0 {
                    // We're in a quantifier body - but should we split?
                    // Check if what follows is a new quantifier at the SAME or LESSER
                    // indentation level (indicating a sibling, not nested)
                    let after_delim = remaining[delim.len()..].trim_start();
                    let starts_new_quantifier = after_delim.starts_with("\\A ")
                        || after_delim.starts_with("\\A(")
                        || after_delim.starts_with("\\E ")
                        || after_delim.starts_with("\\E(");

                    if starts_new_quantifier {
                        // Check indentation: if the current line's indentation is at or before
                        // the base quantifier's indentation, this is a sibling quantifier
                        let is_sibling = quantifier_base_indent
                            .map(|base| line_indent <= base)
                            .unwrap_or(false);

                        if is_sibling {
                            // Split here - this is a sibling quantifier
                            quantifier_depth = 0;
                            quantifier_base_indent = None;
                            if !current.trim().is_empty() {
                                parts.push(current.trim().to_string());
                            }
                            current = String::new();
                            case_depth = 0;
                            i += delim_chars.len();
                            continue;
                        }
                    }
                    // Otherwise don't split - we're inside a quantifier body
                    current.push(c);
                    i += 1;
                    continue;
                } else {
                    // Not in a quantifier body, split normally
                    if !current.trim().is_empty() {
                        parts.push(current.trim().to_string());
                    }
                    current = String::new();
                    // Reset case depth when we split
                    case_depth = 0;
                    i += delim_chars.len();
                    continue;
                }
            }
        }

        current.push(c);
        i += 1;
    }

    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    parts
}

/// Check if a keyword appears at position i with word boundaries
fn matches_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();

    // Check if keyword matches
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (j, kc) in kw_chars.iter().enumerate() {
        if chars[i + j] != *kc {
            return false;
        }
    }

    // Check word boundary before (must be start or non-alphanumeric)
    if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        return false;
    }

    // Check word boundary after (must be end or non-alphanumeric)
    let after = i + kw_chars.len();
    if after < chars.len() && (chars[after].is_alphanumeric() || chars[after] == '_') {
        return false;
    }

    true
}

fn split_binary_op<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut bracket_depth = 0i32;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize; // Track IF...THEN nesting (protects conditions from split)
    let mut case_depth = 0usize; // Track CASE...[] nesting (arms contain = and ->)
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
    let op_chars: Vec<char> = op.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Handle strings
        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
            in_string = !in_string;
            i += 1;
            continue;
        }
        if in_string {
            i += 1;
            continue;
        }

        // Handle << and >> for sequence literals
        if i + 1 < chars.len() {
            if c == '<' && chars[i + 1] == '<' {
                bracket_depth += 1;
                i += 2;
                continue;
            }
            if c == '>' && chars[i + 1] == '>' {
                bracket_depth -= 1;
                i += 2;
                continue;
            }
        }

        match c {
            '(' | '[' | '{' => bracket_depth += 1,
            ')' | ']' | '}' => bracket_depth -= 1,
            _ => {}
        }

        // At bracket top level, check for LET, IN, IF, THEN, CASE keywords
        if bracket_depth == 0 {
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                i += 3;
                continue;
            }
            if matches_keyword_at(&chars, i, "IN") && let_depth > 0 {
                let_depth -= 1;
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "THEN") && if_depth > 0 {
                if_depth -= 1;
                i += 4;
                continue;
            }
            // Track CASE expressions - they contain = and -> in their arms
            if matches_keyword_at(&chars, i, "CASE") {
                case_depth += 1;
                i += 4;
                continue;
            }
            // CASE expressions end at the end of the expression or when we hit
            // a conjunction/disjunction at the same bracket level (but we don't
            // track that here - just don't split inside CASE at all)
        }

        // Only split when at top level of all constructs
        let at_top = bracket_depth == 0 && let_depth == 0 && if_depth == 0 && case_depth == 0;
        if at_top && i + op_chars.len() <= chars.len() {
            // Check if operator matches
            let mut matches = true;
            for (j, &oc) in op_chars.iter().enumerate() {
                if chars[i + j] != oc {
                    matches = false;
                    break;
                }
            }
            if matches {
                if op == ">" && i > 0 && chars[i - 1] == ':' {
                    i += 1;
                    continue;
                }
                // For operators that end with letters and could be prefixes of longer operators
                // (e.g., \in could be prefix of \intersect), check that the next char is not a letter
                if op.chars().last().map_or(false, |c| c.is_alphabetic()) {
                    let next_idx = i + op_chars.len();
                    if next_idx < chars.len() && chars[next_idx].is_alphabetic() {
                        // This is a prefix match, not a full operator match
                        i += 1;
                        continue;
                    }
                }
                // Calculate byte offsets
                let left_byte_end: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                let right_byte_start: usize = chars[..i + op_chars.len()]
                    .iter()
                    .map(|c| c.len_utf8())
                    .sum();
                return Some((&expr[..left_byte_end], &expr[right_byte_start..]));
            }
        }
        i += 1;
    }
    None
}

fn split_comparison(expr: &str) -> Option<(&str, &str, &str)> {
    // Order matters: check longer operators first
    for op in &[
        "\\subseteq",
        "\\notin",
        "\\in",
        "\\geq",
        "\\leq",
        ">=",
        "=<",
        "<=",
        "/=",
        "=",
        ">",
        "<",
        "#",
    ] {
        if let Some((left, right)) = split_binary_op(expr, op) {
            return Some((left.trim(), *op, right.trim()));
        }
    }
    None
}

fn find_record_access_dot(expr: &str) -> Option<usize> {
    let mut depth = 0;
    let bytes = expr.as_bytes();

    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' | b']' | b'}' => depth += 1,
            b'(' | b'[' | b'{' => depth -= 1,
            b'.' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn parse_func_apply(expr: &str) -> Option<(&str, Vec<String>)> {
    // CASE expressions use [] as arm separator, not function application
    // Skip them entirely to avoid confusing [] with f[x] syntax
    let trimmed = expr.trim_start();
    if trimmed.starts_with("CASE")
        && (trimmed.len() == 4
            || !trimmed
                .as_bytes()
                .get(4)
                .map_or(false, |c| c.is_ascii_alphanumeric()))
    {
        return None;
    }

    // Find the last [ at depth 0
    let mut depth = 0;
    let bytes = expr.as_bytes();

    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' | b'}' => depth += 1,
            b'(' | b'{' => depth -= 1,
            b']' => depth += 1,
            b'[' if depth == 1 => {
                // Found the opening bracket
                let func = &expr[..i];
                if func.trim().is_empty() {
                    return None;
                }
                let args_str = &expr[i + 1..expr.len() - 1];
                let args = split_top_level(args_str, ",");
                return Some((func, args));
            }
            b'[' => depth -= 1,
            _ => {}
        }
    }
    None
}

fn is_bracket_type_expr(expr: &str) -> bool {
    let trimmed = expr.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return false;
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.contains("|->") || inner.contains(" EXCEPT ") {
        return false;
    }

    inner.contains("->") || inner.contains(':')
}

fn parse_op_call(expr: &str) -> Option<(&str, Vec<String>)> {
    // Find name followed by (args)
    if let Some(paren_idx) = expr.find('(') {
        let name = &expr[..paren_idx];
        if is_identifier(name.trim()) && expr.ends_with(')') {
            let args_str = &expr[paren_idx + 1..expr.len() - 1];
            let args = if args_str.trim().is_empty() {
                vec![]
            } else {
                split_top_level(args_str, ",")
            };
            return Some((name.trim(), args));
        }
    }
    None
}

fn try_parse_set_comprehension(inner: &str) -> Option<CompiledExpr> {
    // Pattern: {expr : x \in S} or {x \in S : filter}
    if let Some(colon_idx) = find_top_level_colon(inner) {
        let left = inner[..colon_idx].trim();
        let right = inner[colon_idx + 1..].trim();

        // Check if right side is "x \in S" pattern
        if let Some((var, domain)) = parse_in_binding(right) {
            // {expr : x \in S}
            return Some(CompiledExpr::SetComprehension {
                var: var.to_string(),
                domain: Box::new(compile_expr(domain)),
                body: Box::new(compile_expr(left)),
                filter: None,
            });
        }

        // Check if left side is "x \in S" pattern
        if let Some((var, domain)) = parse_in_binding(left) {
            // {x \in S : filter}
            return Some(CompiledExpr::SetComprehension {
                var: var.to_string(),
                domain: Box::new(compile_expr(domain)),
                body: Box::new(CompiledExpr::Var(var.to_string())),
                filter: Some(Box::new(compile_expr(right))),
            });
        }
    }
    None
}

fn try_parse_func_construct(inner: &str) -> Option<CompiledExpr> {
    // Pattern: x \in S |-> expr
    if let Some(arrow_idx) = inner.find("|->") {
        let binding = inner[..arrow_idx].trim();
        let body = inner[arrow_idx + 3..].trim();

        if let Some((var, domain)) = parse_in_binding(binding) {
            return Some(CompiledExpr::FuncConstruct {
                var: var.to_string(),
                domain: Box::new(compile_expr(domain)),
                body: Box::new(compile_expr(body)),
            });
        }
    }
    None
}

/// Try to parse a function set: [Domain -> Range]
/// This is the set of all functions from Domain to Range.
fn try_parse_function_set(inner: &str) -> Option<CompiledExpr> {
    // Find " -> " at top level (not inside nested brackets)
    // Must not contain "|->", which would be function construction
    if inner.contains("|->") {
        return None;
    }

    // Find the "->" at top level
    let arrow_idx = find_top_level_arrow(inner)?;

    let domain = inner[..arrow_idx].trim();
    let range = inner[arrow_idx + 2..].trim();

    if domain.is_empty() || range.is_empty() {
        return None;
    }

    Some(CompiledExpr::FunctionSet {
        domain: Box::new(compile_expr(domain)),
        range: Box::new(compile_expr(range)),
    })
}

/// Try to parse a record set construction: [field: Set, field2: Set, ...]
/// This creates the Cartesian product of all field/set pairs as records.
/// For example: [a: {1,2}, b: {3,4}] produces all combinations of a and b values.
fn try_parse_record_set(inner: &str) -> Option<CompiledExpr> {
    let entries = split_top_level(inner, ",");
    if entries.is_empty() {
        return None;
    }

    let mut fields = Vec::new();

    for entry in &entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        // Find colon at top level
        let colon_idx = find_top_level_colon(entry)?;

        let field_name = entry[..colon_idx].trim();
        let set_expr = entry[colon_idx + 1..].trim();

        // Field name must be a valid identifier
        if !is_identifier(field_name) {
            return None;
        }

        fields.push((field_name.to_string(), compile_expr(set_expr)));
    }

    if fields.is_empty() {
        return None;
    }

    Some(CompiledExpr::RecordSet(fields))
}

/// Find " -> " at top level (not inside brackets)
fn find_top_level_arrow(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            b'-' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                // Check it's not "|->"
                if i > 0 && bytes[i - 1] == b'|' {
                    i += 1;
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

/// Try to parse an EXCEPT expression
/// Patterns:
/// - `f EXCEPT ![key] = val`
/// - `f EXCEPT ![k1][k2] = val` (nested path)
/// - `f EXCEPT ![k1] = val, ![k2] = val2` (multiple updates)
/// - `f EXCEPT ![k] = @ + 1` (@ refers to original value)
fn try_parse_except(inner: &str) -> Option<CompiledExpr> {
    // Find "EXCEPT" keyword (may be followed by space or newline)
    // Try " EXCEPT " first, then " EXCEPT\n"
    let (except_pos, keyword_len) = if let Some(pos) = inner.find(" EXCEPT ") {
        (pos, 8) // " EXCEPT " is 8 characters
    } else if let Some(pos) = inner.find(" EXCEPT\n") {
        (pos, 8) // " EXCEPT\n" is also 8 characters
    } else {
        return None;
    };
    let base = inner[..except_pos].trim();
    let updates_str = inner[except_pos + keyword_len..].trim();

    if base.is_empty() || updates_str.is_empty() {
        return None;
    }

    // Parse updates: ![path] = value, ![path2] = value2, ...
    let updates = parse_except_updates(updates_str)?;

    Some(CompiledExpr::FuncExcept(
        Box::new(compile_expr(base)),
        updates,
    ))
}

/// Parse EXCEPT update clauses: ![k1][k2] = val, ![k3] = val2
fn parse_except_updates(s: &str) -> Option<Vec<(Vec<CompiledExpr>, CompiledExpr)>> {
    let mut updates = Vec::new();

    // Split by comma at top level (not inside brackets)
    let parts = split_top_level(s, ",");

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Each part should be: ![k1][k2]... = value
        if !part.starts_with('!') {
            return None;
        }

        // Find the = sign (not inside brackets)
        // Note: eq_pos is relative to part[1..], so we add 1 to get the position in part
        let eq_pos = find_top_level_eq(&part[1..])?;
        let path_str = part[1..1 + eq_pos].trim();
        let value_str = part[1 + eq_pos + 1..].trim();

        // Parse the path: [k1][k2][k3]...
        let path = parse_except_path(path_str)?;
        let value = compile_expr(value_str);

        updates.push((path, value));
    }

    if updates.is_empty() {
        None
    } else {
        Some(updates)
    }
}

/// Parse EXCEPT path: [k1][k2][k3]... or .field.field2... -> vec![k1, k2, k3] or vec!["field", "field2"]
fn parse_except_path(s: &str) -> Option<Vec<CompiledExpr>> {
    let mut path = Vec::new();
    let mut remaining = s.trim();

    while !remaining.is_empty() {
        // Handle bracket notation: [key]
        if remaining.starts_with('[') {
            // Find matching ]
            let mut depth = 0;
            let mut end_idx = None;
            for (i, c) in remaining.char_indices() {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            end_idx = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            let end = end_idx?;
            let key_expr = &remaining[1..end].trim();
            path.push(compile_expr(key_expr));
            remaining = &remaining[end + 1..];
            continue;
        }

        // Handle dot notation: .field
        if remaining.starts_with('.') {
            remaining = &remaining[1..];
            // Parse field name (identifier)
            let mut end_idx = 0;
            for (i, c) in remaining.char_indices() {
                if c.is_alphanumeric() || c == '_' {
                    end_idx = i + c.len_utf8();
                } else {
                    break;
                }
            }
            if end_idx == 0 {
                // No valid field name after dot
                break;
            }
            let field_name = &remaining[..end_idx];
            path.push(CompiledExpr::String(field_name.to_string()));
            remaining = &remaining[end_idx..];
            continue;
        }

        // Neither bracket nor dot - done with path
        break;
    }

    if path.is_empty() { None } else { Some(path) }
}

/// Find = sign at top level (not inside brackets/parens)
fn find_top_level_eq(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '=' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn parse_in_binding(s: &str) -> Option<(&str, &str)> {
    if let Some(in_idx) = s.find("\\in") {
        let var = s[..in_idx].trim();
        let domain = s[in_idx + 3..].trim();
        if is_identifier(var) {
            return Some((var, domain));
        }
    }
    None
}

/// Parse multi-variable binding with same domain: "i, j \in S" -> (["i", "j"], "S")
fn parse_multi_in_binding(s: &str) -> Option<(Vec<String>, &str)> {
    if let Some(in_idx) = s.find("\\in") {
        let vars_part = s[..in_idx].trim();
        let domain = s[in_idx + 3..].trim();

        // Check if there are multiple variables (comma-separated)
        if !vars_part.contains(',') {
            return None; // Single variable, let parse_in_binding handle it
        }

        let vars: Vec<String> = vars_part
            .split(',')
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .collect();

        // All parts must be valid identifiers (for "i, j \in S" pattern)
        // This must NOT match "x \in S, y \in T" pattern
        if vars.len() >= 2 && vars.iter().all(|v| is_identifier(v)) && !domain.contains("\\in") {
            return Some((vars, domain));
        }
    }
    None
}

/// Parse multiple separate bindings: "x \in S, y \in T" -> [("x", "S"), ("y", "T")]
fn parse_multiple_bindings(s: &str) -> Option<Vec<(String, String)>> {
    // Count the number of \in occurrences
    let in_count = s.matches("\\in").count();
    if in_count < 2 {
        return None; // Not multiple bindings
    }

    // Split by comma at top level, then parse each as a binding
    let parts = split_quantifier_bindings(s);
    if parts.len() < 2 {
        return None;
    }

    let mut bindings = Vec::new();
    for part in parts {
        let part = part.trim();
        if let Some((var, domain)) = parse_in_binding(part) {
            bindings.push((var.to_string(), domain.to_string()));
        } else {
            return None; // One of the parts isn't a valid binding
        }
    }

    if bindings.len() >= 2 {
        Some(bindings)
    } else {
        None
    }
}

fn contains_tuple_binder(binding: &str) -> bool {
    binding.contains("<<") && binding.contains(">>")
}

/// Split quantifier bindings by comma, respecting nesting
/// "x \in S, y \in T" -> ["x \in S", "y \in T"]
/// "x \in {1,2}, y \in T" -> ["x \in {1,2}", "y \in T"]
fn split_quantifier_bindings(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        match c {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            '<' if i + 1 < chars.len() && chars[i + 1] == '<' => {
                depth += 1;
                current.push(c);
                current.push(chars[i + 1]);
                i += 1;
            }
            '>' if i + 1 < chars.len() && chars[i + 1] == '>' => {
                depth -= 1;
                current.push(c);
                current.push(chars[i + 1]);
                i += 1;
            }
            ',' if depth == 0 => {
                // Only split on comma if the current part contains \in
                // This ensures "i, j \in S" doesn't get split here
                if current.contains("\\in") {
                    parts.push(current.trim().to_string());
                    current = String::new();
                } else {
                    current.push(c);
                }
            }
            _ => {
                current.push(c);
            }
        }
        i += 1;
    }

    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    parts
}

pub fn find_top_level_colon(expr: &str) -> Option<usize> {
    let mut depth = 0;
    let bytes = expr.as_bytes();

    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b':' if depth == 0 => {
                // Skip :> (TLC function pair operator) - only match standalone :
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    continue;
                }
                return Some(i);
            }
            _ => {}
        }
    }
    None
}

fn try_parse_if(expr: &str) -> Option<CompiledExpr> {
    // IF cond THEN expr1 ELSE expr2
    let rest = expr.strip_prefix("IF ")?.trim();

    let then_idx = find_keyword(rest, "THEN")?;
    let cond = rest[..then_idx].trim();
    let rest = rest[then_idx + 4..].trim();

    let else_idx = find_keyword(rest, "ELSE")?;
    let then_expr = rest[..else_idx].trim();
    let else_expr = rest[else_idx + 4..].trim();

    Some(CompiledExpr::If {
        cond: Box::new(compile_expr(cond)),
        then_branch: Box::new(compile_expr(then_expr)),
        else_branch: Box::new(compile_expr(else_expr)),
    })
}

fn try_parse_let(expr: &str) -> Option<CompiledExpr> {
    // LET defs IN body
    // Handle "LET " or "LET\n" or "LET\t" etc.
    let rest = expr.strip_prefix("LET")?;
    // Must be followed by whitespace
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    let in_idx = find_keyword(rest, "IN")?;
    let defs_str = rest[..in_idx].trim();
    let body_str = rest[in_idx + 2..].trim();

    // Parse definitions properly handling multiline definitions
    // Find all == positions at top level (not inside brackets or nested LET)
    let eq_positions = find_definition_equals(defs_str);
    if eq_positions.is_empty() {
        return None;
    }

    let mut bindings = Vec::new();
    for (idx, eq_pos) in eq_positions.iter().enumerate() {
        // Find the start of this definition (name part)
        let name_start = if idx == 0 {
            0
        } else {
            // Start after the previous definition's body
            // Find the line start before this ==
            find_line_start_before(defs_str, *eq_pos)
        };

        // Find the end of this definition's body
        let body_end = if idx + 1 < eq_positions.len() {
            find_line_start_before(defs_str, eq_positions[idx + 1])
        } else {
            defs_str.len()
        };

        let name = trim_let_edge_comments(&defs_str[name_start..*eq_pos]);
        let value = trim_let_edge_comments(&defs_str[*eq_pos + 2..body_end]);

        // Skip comments (lines starting with \*)
        let name = name.lines().last().unwrap_or("").trim();

        // Name should be a simple identifier (possibly with params in brackets)
        if let Some(_bracket_idx) = name.find('[') {
            // Has parameters like sumF[S \in SUBSET opts] - skip these recursive defs
            continue;
        }
        if !name.is_empty() && is_identifier(name) {
            bindings.push((name.to_string(), compile_expr(value)));
        }
    }

    if bindings.is_empty() {
        // We successfully parsed the LET...IN structure but couldn't handle the definitions
        // (likely recursive function definitions like sumF[S \in SUBSET opts]).
        // Return Unparsed so eval.rs can handle it with string-based evaluation,
        // rather than returning None which would cause the expression to fall through
        // to other parsers (like split_comparison) that might incorrectly split on
        // operators inside the LET body.
        return Some(CompiledExpr::Unparsed(expr.to_string()));
    }

    Some(CompiledExpr::Let {
        bindings,
        body: Box::new(compile_expr(body_str)),
    })
}

fn try_parse_lambda(expr: &str) -> Option<CompiledExpr> {
    // LAMBDA x: body  OR  LAMBDA x, y: body
    let rest = expr.strip_prefix("LAMBDA ")?.trim();

    let colon_idx = find_top_level_colon(rest)?;
    let params_str = rest[..colon_idx].trim();
    let body_str = rest[colon_idx + 1..].trim();

    // Parse parameter names (comma-separated identifiers)
    let params: Vec<String> = params_str
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();

    if params.is_empty() {
        return None;
    }

    // All params should be simple identifiers
    if !params.iter().all(|p| is_identifier(p)) {
        return None;
    }

    Some(CompiledExpr::Lambda {
        params,
        body: Box::new(compile_expr(body_str)),
        body_text: body_str.to_string(),
    })
}

fn try_parse_exists(expr: &str) -> Option<CompiledExpr> {
    // \E x \in S : body  OR  \E i, j \in S : body (multi-variable same domain)
    // OR  \E x \in S, y \in T : body (multiple separate bindings)
    let rest = expr.strip_prefix("\\E ")?.trim();

    let colon_idx = find_top_level_colon(rest)?;
    let binding = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();

    if contains_tuple_binder(binding) {
        return Some(CompiledExpr::Unparsed(expr.to_string()));
    }

    // Try to parse multiple separate bindings: "x \in S, y \in T"
    if let Some(bindings) = parse_multiple_bindings(binding) {
        // Convert to nested exists
        let compiled_body = compile_expr(body);
        let mut result = compiled_body;
        for (var, domain) in bindings.into_iter().rev() {
            result = CompiledExpr::Exists {
                var,
                domain: Box::new(compile_expr(&domain)),
                body: Box::new(result),
            };
        }
        return Some(result);
    }

    // Try multi-variable binding with same domain: "i, j \in S"
    if let Some((vars, domain)) = parse_multi_in_binding(binding) {
        // Convert to nested exists: \E i \in S : \E j \in S : body
        let compiled_body = compile_expr(body);
        let compiled_domain = compile_expr(domain);

        // Build from inside out
        let mut result = compiled_body;
        for var in vars.into_iter().rev() {
            result = CompiledExpr::Exists {
                var,
                domain: Box::new(compiled_domain.clone()),
                body: Box::new(result),
            };
        }
        return Some(result);
    }

    // Single variable binding
    let (var, domain) = parse_in_binding(binding)?;

    Some(CompiledExpr::Exists {
        var: var.to_string(),
        domain: Box::new(compile_expr(domain)),
        body: Box::new(compile_expr(body)),
    })
}

fn try_parse_forall(expr: &str) -> Option<CompiledExpr> {
    // \A x \in S : body  OR  \A i, j \in S : body (multi-variable same domain)
    // OR  \A x \in S, y \in T : body (multiple separate bindings)
    let rest = expr.strip_prefix("\\A ")?.trim();

    let colon_idx = find_top_level_colon(rest)?;
    let binding = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();

    if contains_tuple_binder(binding) {
        return Some(CompiledExpr::Unparsed(expr.to_string()));
    }

    // Try to parse multiple separate bindings: "x \in S, y \in T"
    if let Some(bindings) = parse_multiple_bindings(binding) {
        // Convert to nested forall
        let compiled_body = compile_expr(body);
        let mut result = compiled_body;
        for (var, domain) in bindings.into_iter().rev() {
            result = CompiledExpr::Forall {
                var,
                domain: Box::new(compile_expr(&domain)),
                body: Box::new(result),
            };
        }
        return Some(result);
    }

    // Try multi-variable binding with same domain: "i, j \in S"
    if let Some((vars, domain)) = parse_multi_in_binding(binding) {
        // Convert to nested forall: \A i \in S : \A j \in S : body
        let compiled_body = compile_expr(body);
        let compiled_domain = compile_expr(domain);

        // Build from inside out
        let mut result = compiled_body;
        for var in vars.into_iter().rev() {
            result = CompiledExpr::Forall {
                var,
                domain: Box::new(compiled_domain.clone()),
                body: Box::new(result),
            };
        }
        return Some(result);
    }

    // Single variable binding
    let (var, domain) = parse_in_binding(binding)?;

    Some(CompiledExpr::Forall {
        var: var.to_string(),
        domain: Box::new(compile_expr(domain)),
        body: Box::new(compile_expr(body)),
    })
}

fn try_parse_choose(expr: &str) -> Option<CompiledExpr> {
    // CHOOSE x \in S : body
    let rest = expr.strip_prefix("CHOOSE ")?.trim();

    let colon_idx = find_top_level_colon(rest)?;
    let binding = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();

    if contains_tuple_binder(binding) {
        return Some(CompiledExpr::Unparsed(expr.to_string()));
    }

    let (var, domain) = parse_in_binding(binding)?;

    Some(CompiledExpr::Choose {
        var: var.to_string(),
        domain: Box::new(compile_expr(domain)),
        body: Box::new(compile_expr(body)),
    })
}

fn find_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let mut bracket_depth = 0i32;
    let mut let_depth = 0usize; // Track LET...IN nesting
    let mut if_depth = 0usize; // Track IF...ELSE nesting
    let chars: Vec<char> = expr.chars().collect();
    let _keyword_chars: Vec<char> = keyword.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Track bracket depth
        match c {
            '(' | '[' | '{' => bracket_depth += 1,
            ')' | ']' | '}' => bracket_depth -= 1,
            '<' if i + 1 < chars.len() && chars[i + 1] == '<' => {
                bracket_depth += 1;
                i += 2;
                continue;
            }
            '>' if i + 1 < chars.len() && chars[i + 1] == '>' => {
                bracket_depth -= 1;
                i += 2;
                continue;
            }
            _ => {}
        }

        // At bracket top level, check for LET, IN, IF, ELSE keywords to track nesting
        if bracket_depth == 0 {
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                i += 3;
                continue;
            }
            if matches_keyword_at(&chars, i, "IN") && let_depth > 0 {
                let_depth -= 1;
                i += 2;
                continue;
            }
            // Track IF...ELSE nesting to avoid finding keywords inside IF conditions
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "ELSE") && if_depth > 0 {
                if_depth -= 1;
                i += 4;
                continue;
            }
        }

        // Check if we found the keyword at top level (brackets, LET, and IF all at 0)
        if bracket_depth == 0
            && let_depth == 0
            && if_depth == 0
            && matches_keyword_at(&chars, i, keyword)
        {
            // Return byte offset, not char index
            let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            return Some(byte_offset);
        }

        i += 1;
    }
    None
}

// ============================================================================
// Compiled Action IR - pre-compiled action clauses for fast execution
// ============================================================================

use crate::tla::action_ir::{parse_action_exists, parse_action_if, split_action_body_clauses};
use crate::tla::{ActionClause, ClauseKind, classify_clause};

/// A compiled version of ActionClause that stores pre-parsed expressions
#[derive(Debug, Clone)]
pub enum CompiledActionClause {
    PrimedAssignment {
        var: String,
        expr: CompiledExpr,
    },
    Unchanged {
        vars: Vec<String>,
    },
    Guard {
        expr: CompiledExpr,
    },
    Exists {
        binders: String,
        body_clauses: Vec<CompiledActionClause>,
    },
    Conditional {
        condition: CompiledExpr,
        then_clauses: Vec<CompiledActionClause>,
        else_clauses: Vec<CompiledActionClause>,
    },
    /// Compiled LET expression with primed assignments in the body
    /// The bindings are evaluated first to create local definitions,
    /// then the body's assignments/guards are evaluated with those bindings in scope
    CompiledLetWithPrimes {
        /// Local bindings: name -> compiled expression
        bindings: Vec<(String, CompiledExpr)>,
        /// The body's primed assignments and guards (compiled)
        body_clauses: Vec<CompiledActionClause>,
    },
    /// Fallback: LET expression we couldn't fully compile
    LetWithPrimes {
        expr: String,
    },
}

/// Pre-compiled action IR - compile once, execute many times
#[derive(Debug, Clone)]
pub struct CompiledActionIr {
    pub name: String,
    pub params: Vec<String>,
    pub clauses: Vec<CompiledActionClause>,
}

impl CompiledActionIr {
    /// Compile an ActionIr into a CompiledActionIr with pre-parsed expressions
    pub fn from_ir(ir: &crate::tla::ActionIr) -> Self {
        let clauses = ir.clauses.iter().map(compile_action_clause).collect();

        CompiledActionIr {
            name: ir.name.clone(),
            params: ir.params.clone(),
            clauses,
        }
    }
}

fn compile_action_clause(clause: &ActionClause) -> CompiledActionClause {
    match clause {
        ActionClause::PrimedAssignment { var, expr } => CompiledActionClause::PrimedAssignment {
            var: var.clone(),
            expr: compile_expr(expr),
        },
        ActionClause::Unchanged { vars } => CompiledActionClause::Unchanged { vars: vars.clone() },
        ActionClause::Guard { expr } => compile_action_clause_text(expr),
        ActionClause::Exists { binders, body } => CompiledActionClause::Exists {
            binders: binders.clone(),
            body_clauses: compile_action_body_clauses(body),
        },
        ActionClause::LetWithPrimes { expr } => match compile_let_with_primes(expr) {
            Some(compiled) => compiled,
            None => CompiledActionClause::LetWithPrimes { expr: expr.clone() },
        },
    }
}

fn compile_action_body_clauses(body: &str) -> Vec<CompiledActionClause> {
    split_action_body_clauses(body)
        .into_iter()
        .map(|clause| compile_action_clause_text(&clause))
        .collect()
}

fn compile_action_clause_text(expr: &str) -> CompiledActionClause {
    let trimmed = expr.trim();
    if let Some((condition, then_branch, else_branch)) = parse_action_if(trimmed) {
        let then_clauses = compile_action_body_clauses(then_branch);
        let else_clauses = compile_action_body_clauses(else_branch);
        return CompiledActionClause::Conditional {
            condition: compile_expr(condition),
            then_clauses,
            else_clauses,
        };
    }
    if let Some((binders, body)) = parse_action_exists(trimmed) {
        return CompiledActionClause::Exists {
            binders: binders.to_string(),
            body_clauses: compile_action_body_clauses(body),
        };
    }

    match classify_clause(trimmed) {
        ClauseKind::PrimedAssignment { var, expr } => CompiledActionClause::PrimedAssignment {
            var,
            expr: compile_expr(&expr),
        },
        ClauseKind::Unchanged { vars } => CompiledActionClause::Unchanged { vars },
        ClauseKind::UnprimedEquality { .. }
        | ClauseKind::UnprimedMembership { .. }
        | ClauseKind::Other => {
            if trimmed.starts_with("LET") && trimmed.contains('\'') {
                match compile_let_with_primes(trimmed) {
                    Some(compiled) => compiled,
                    None => CompiledActionClause::LetWithPrimes {
                        expr: trimmed.to_string(),
                    },
                }
            } else {
                CompiledActionClause::Guard {
                    expr: compile_expr(trimmed),
                }
            }
        }
    }
}

/// Try to compile a LET expression with primed assignments
/// Returns None if we can't fully compile it (fallback to string eval)
fn compile_let_with_primes(expr: &str) -> Option<CompiledActionClause> {
    // Parse: LET defs IN body
    let (defs_text, body_text) = split_let_expression(expr)?;

    // Parse bindings: name == expr
    let bindings = parse_let_bindings(defs_text)?;

    // Compile bindings
    let compiled_bindings: Vec<(String, CompiledExpr)> = bindings
        .into_iter()
        .map(|(name, expr_str)| (name, compile_expr(&expr_str)))
        .collect();

    // Parse body as conjunction of clauses
    let body_conjuncts = split_action_body_clauses(body_text);

    // Compile each clause in the body
    let body_clauses = body_conjuncts
        .into_iter()
        .map(|conjunct| compile_action_clause_text(&conjunct))
        .collect();

    Some(CompiledActionClause::CompiledLetWithPrimes {
        bindings: compiled_bindings,
        body_clauses,
    })
}

/// Split a LET expression into definitions and body
/// LET defs IN body -> Some((defs, body))
fn split_let_expression(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let after_let = trimmed.strip_prefix("LET")?.trim_start();

    // Find matching IN keyword, handling nested LETs
    let mut depth = 1usize;
    let bytes = after_let.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        // Check for keywords at word boundaries
        if i + 3 <= bytes.len() && &bytes[i..i + 3] == b"LET" {
            // Check word boundary after
            if i + 3 >= bytes.len() || !bytes[i + 3].is_ascii_alphanumeric() {
                depth += 1;
                i += 3;
                continue;
            }
        }

        if i + 2 <= bytes.len() && &bytes[i..i + 2] == b"IN" {
            // Check word boundaries
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_ok = i + 2 >= bytes.len() || !bytes[i + 2].is_ascii_alphanumeric();
            if before_ok && after_ok {
                depth -= 1;
                if depth == 0 {
                    let defs = after_let[..i].trim();
                    let body = after_let[i + 2..].trim();
                    return Some((defs, body));
                }
            }
        }

        // Skip to next character
        i += 1;
    }

    None
}

/// Parse LET bindings into (name, expr) pairs
/// Handles: name == expr (possibly multi-line with multiple definitions)
fn parse_let_bindings(defs_text: &str) -> Option<Vec<(String, String)>> {
    let mut bindings = Vec::new();

    // Find all == positions at top level
    let eq_positions = find_definition_equals(defs_text);
    if eq_positions.is_empty() {
        return None;
    }

    for (idx, eq_pos) in eq_positions.iter().enumerate() {
        // Find the start of this definition (name part)
        let name_start = if idx == 0 {
            0
        } else {
            // Start after the previous definition's body
            // We need to find where the previous body ends
            // For simplicity, look for the line start before this ==
            find_line_start_before(defs_text, *eq_pos)
        };

        // Find the end of this definition's body
        let body_end = if idx + 1 < eq_positions.len() {
            find_line_start_before(defs_text, eq_positions[idx + 1])
        } else {
            defs_text.len()
        };

        let name = trim_let_edge_comments(&defs_text[name_start..*eq_pos]);
        let body = trim_let_edge_comments(&defs_text[*eq_pos + 2..body_end]);

        // Name should be a simple identifier (possibly with params, but we ignore those for now)
        let name = if let Some(_paren) = name.find('(') {
            // Has parameters - for simplicity, we don't support parameterized local defs yet
            // Return None to fall back to string eval
            return None;
        } else {
            name.to_string()
        };

        if !name.is_empty() && is_identifier(&name) {
            bindings.push((name, body.to_string()));
        }
    }

    if bindings.is_empty() {
        None
    } else {
        Some(bindings)
    }
}

/// Find all == positions at top level (not inside brackets/parens or nested LET)
fn find_definition_equals(text: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut depth: usize = 0; // Bracket depth
    let mut let_depth: usize = 0; // LET...IN nesting depth
    let mut i = 0;

    while i + 1 < chars.len() {
        let c = chars[i];

        // Track bracket depth
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '<' if i + 1 < chars.len() && chars[i + 1] == '<' => {
                depth += 1;
                i += 1;
            }
            '>' if i + 1 < chars.len() && chars[i + 1] == '>' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => {}
        }

        // Track LET...IN nesting (only at bracket top level)
        if depth == 0 && matches_keyword_at(&chars, i, "LET") {
            let_depth += 1;
            i += 3;
            continue;
        }
        if depth == 0 && let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
            let_depth = let_depth.saturating_sub(1);
            i += 2;
            continue;
        }

        // Check for == at true top level (no brackets, no nested LET)
        if c == '=' && i + 1 < chars.len() && chars[i + 1] == '=' && depth == 0 && let_depth == 0 {
            // Found == at top level
            // Calculate byte offset
            let byte_pos: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            positions.push(byte_pos);
            i += 1; // Skip the second =
        }

        i += 1;
    }

    positions
}

fn trim_let_edge_comments(text: &str) -> &str {
    let mut start = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("\\*") {
            start += line.len();
            continue;
        }
        break;
    }

    let trimmed = &text[start..];
    let mut end = trimmed.len();
    let lines: Vec<&str> = trimmed.split_inclusive('\n').collect();
    for line in lines.into_iter().rev() {
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() || line_trimmed.starts_with("\\*") {
            end = end.saturating_sub(line.len());
            continue;
        }
        break;
    }

    trimmed[..end].trim()
}

/// Find the start of the line before a given position
fn find_line_start_before(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.saturating_sub(1);

    // First, go back to find a newline
    while i > 0 && bytes[i] != b'\n' {
        i -= 1;
    }

    // If we found a newline, the line starts after it
    if bytes[i] == b'\n' { i + 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_literals() {
        assert!(matches!(compile_expr("TRUE"), CompiledExpr::Bool(true)));
        assert!(matches!(compile_expr("FALSE"), CompiledExpr::Bool(false)));
        assert!(matches!(compile_expr("42"), CompiledExpr::Int(42)));
        assert!(matches!(compile_expr("-5"), CompiledExpr::Int(-5)));
    }

    #[test]
    fn test_compile_logical() {
        let expr = compile_expr("x /\\ y");
        assert!(matches!(expr, CompiledExpr::And(_)));

        let expr = compile_expr("a \\/ b");
        assert!(matches!(expr, CompiledExpr::Or(_)));

        let expr = compile_expr("~p");
        assert!(matches!(expr, CompiledExpr::Not(_)));
    }

    #[test]
    fn test_compile_comparison() {
        let expr = compile_expr("x = 5");
        assert!(matches!(expr, CompiledExpr::Eq(_, _)));

        let expr = compile_expr("n < 10");
        assert!(matches!(expr, CompiledExpr::Lt(_, _)));
    }

    #[test]
    fn test_compile_leq_geq() {
        let expr = compile_expr("x \\leq 5");
        assert!(matches!(expr, CompiledExpr::Le(_, _)));

        let expr = compile_expr("x \\geq 5");
        assert!(matches!(expr, CompiledExpr::Ge(_, _)));
    }

    #[test]
    fn test_compile_function_pair() {
        let expr = compile_expr("mid :> [x |-> 1]");
        assert!(
            matches!(expr, CompiledExpr::FuncPair(_, _)),
            "function pair should not be parsed as comparison: {:?}",
            expr
        );
    }

    #[test]
    fn test_intersect_not_confused_with_in() {
        // Ensure \intersect is not confused with \in
        // \in should NOT match inside \intersect
        let expr = compile_expr("seen \\intersect Node");
        assert!(
            matches!(expr, CompiledExpr::Intersect(_, _)),
            "\\intersect should be parsed as Intersect, not In: {:?}",
            expr
        );

        // Also test that \in still works on its own
        let expr2 = compile_expr("x \\in S");
        assert!(
            matches!(expr2, CompiledExpr::In(_, _)),
            "\\in should be parsed as In: {:?}",
            expr2
        );
    }

    #[test]
    fn test_compile_set_literal() {
        let expr = compile_expr("{1, 2, 3}");
        if let CompiledExpr::SetLiteral(elems) = expr {
            assert_eq!(elems.len(), 3);
        } else {
            panic!("Expected SetLiteral");
        }
    }

    #[test]
    fn test_compile_if() {
        let expr = compile_expr("IF x > 0 THEN x ELSE -x");
        assert!(matches!(expr, CompiledExpr::If { .. }));
    }

    #[test]
    fn test_compile_quantifier() {
        let expr = compile_expr("\\E x \\in S : x > 0");
        assert!(matches!(expr, CompiledExpr::Exists { .. }));

        let expr = compile_expr("\\A x \\in S : x >= 0");
        assert!(matches!(expr, CompiledExpr::Forall { .. }));
    }

    #[test]
    fn test_bracketed_function_set_parsed_correctly() {
        let expr = compile_expr("[S -> T]");
        match expr {
            CompiledExpr::FunctionSet { domain, range } => {
                assert!(matches!(*domain, CompiledExpr::Var(ref s) if s == "S"));
                assert!(matches!(*range, CompiledExpr::Var(ref s) if s == "T"));
            }
            other => {
                panic!("expected FunctionSet, got: {other:?}")
            }
        }

        // Test with more complex expressions
        let expr = compile_expr("[{1, 2} -> BOOLEAN]");
        assert!(matches!(expr, CompiledExpr::FunctionSet { .. }));

        // Test that function construction is not misparsed as function set
        let expr = compile_expr("[x \\in S |-> x + 1]");
        assert!(matches!(expr, CompiledExpr::FuncConstruct { .. }));
    }

    #[test]
    fn test_split_let_expression() {
        // Simple case
        let result = split_let_expression("LET x == 1 IN x + 1");
        assert!(result.is_some());
        let (defs, body) = result.unwrap();
        assert_eq!(defs, "x == 1");
        assert_eq!(body, "x + 1");

        // Nested LET
        let result = split_let_expression("LET x == 1 IN LET y == 2 IN x + y");
        assert!(result.is_some());
        let (defs, body) = result.unwrap();
        assert_eq!(defs, "x == 1");
        assert_eq!(body, "LET y == 2 IN x + y");
    }

    #[test]
    fn test_parse_let_bindings() {
        // Single binding
        let result = parse_let_bindings("x == 1");
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0], ("x".to_string(), "1".to_string()));

        // Multiple bindings
        let result = parse_let_bindings("x == 1\ny == x + 1");
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[1].0, "y");
    }

    #[test]
    fn test_parse_let_bindings_ignores_interstitial_comment_lines() {
        let defs = r#"priceDiff == referencePrice[f.asset] - f.price
\* For each participant, calculate their P&L
settlementPayments == { <<p, positions[p] * priceDiff>> : p \in Participants }
\* Check that all participants can cover their losses
canSettle == \A <<p, pnl>> \in settlementPayments : pnl >= 0"#;

        let bindings = parse_let_bindings(defs).expect("bindings should parse");
        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].0, "priceDiff");
        assert_eq!(bindings[0].1, "referencePrice[f.asset] - f.price");
        assert_eq!(bindings[1].0, "settlementPayments");
        assert_eq!(
            bindings[1].1,
            "{ <<p, positions[p] * priceDiff>> : p \\in Participants }"
        );
        assert_eq!(bindings[2].0, "canSettle");
    }

    #[test]
    fn test_compile_let_with_comments_between_bindings() {
        let expr = compile_expr(
            r#"LET
    priceDiff == referencePrice[f.asset] - f.price
    \* For each participant, calculate their P&L
    settlementPayments == { <<p, positions[p] * priceDiff>> : p \in Participants }
    \* Check that all participants can cover their losses
    canSettle == \A <<p, pnl>> \in settlementPayments : pnl >= 0
IN
    IF canSettle THEN priceDiff ELSE 0"#,
        );

        let CompiledExpr::Let { bindings, body } = expr else {
            panic!("expected LET expression");
        };
        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].0, "priceDiff");
        assert_eq!(bindings[1].0, "settlementPayments");
        assert_eq!(bindings[2].0, "canSettle");
        assert!(matches!(*body, CompiledExpr::If { .. }));
    }

    #[test]
    fn test_tuple_pattern_quantifier_falls_back_to_unparsed() {
        let expr = compile_expr(r#"\A <<p, pnl>> \in settlementPayments : pnl >= 0"#);
        assert!(
            matches!(expr, CompiledExpr::Unparsed(_)),
            "tuple-pattern quantifier should preserve the original expression for interpreted fallback: {:?}",
            expr
        );
    }

    #[test]
    fn test_compile_let_with_primes() {
        // Simple LET with primed assignment
        let result = compile_let_with_primes("LET x == 1 IN y' = x");
        assert!(result.is_some());
        if let Some(CompiledActionClause::CompiledLetWithPrimes {
            bindings,
            body_clauses,
        }) = result
        {
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0, "x");
            assert_eq!(body_clauses.len(), 1);
            assert!(matches!(
                body_clauses[0],
                CompiledActionClause::PrimedAssignment { .. }
            ));
        } else {
            panic!("Expected CompiledLetWithPrimes");
        }

        // LET with multiple body clauses
        let result = compile_let_with_primes("LET x == 1 IN y' = x /\\ UNCHANGED <<z>>");
        assert!(result.is_some());
        if let Some(CompiledActionClause::CompiledLetWithPrimes {
            bindings,
            body_clauses,
        }) = result
        {
            assert_eq!(bindings.len(), 1);
            assert_eq!(body_clauses.len(), 2);
            assert!(matches!(
                body_clauses[0],
                CompiledActionClause::PrimedAssignment { .. }
            ));
            assert!(matches!(
                body_clauses[1],
                CompiledActionClause::Unchanged { .. }
            ));
        } else {
            panic!("Expected CompiledLetWithPrimes");
        }
    }

    #[test]
    fn test_compile_if_with_equality_in_condition() {
        // IF with equality operator in condition should not be split at =
        let expr = compile_expr("IF opts = {} THEN 0 ELSE 1");
        assert!(
            matches!(expr, CompiledExpr::If { .. }),
            "IF expression with equality should be parsed as If, got: {:?}",
            expr
        );

        // IF with more complex condition containing =
        let expr = compile_expr("IF x = 5 THEN x + 1 ELSE x - 1");
        if let CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } = expr
        {
            // Condition should be an equality check, not Unparsed
            assert!(
                matches!(*cond, CompiledExpr::Eq(_, _)),
                "Condition should be Eq, got: {:?}",
                cond
            );
            // Branches should be compiled
            assert!(
                matches!(*then_branch, CompiledExpr::Add(_, _)),
                "Then branch should be Add, got: {:?}",
                then_branch
            );
            assert!(
                matches!(*else_branch, CompiledExpr::Sub(_, _)),
                "Else branch should be Sub, got: {:?}",
                else_branch
            );
        } else {
            panic!("Expected If expression");
        }
    }

    #[test]
    fn test_compile_if_with_nested_let_in_else() {
        // IF with nested LET in ELSE branch
        let expr = compile_expr("IF S = {} THEN 0 ELSE LET x == 1 IN x + 2");
        if let CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } = expr
        {
            // Condition should be equality check
            assert!(
                matches!(*cond, CompiledExpr::Eq(_, _)),
                "Condition should be Eq, got: {:?}",
                cond
            );
            // Then branch should be integer
            assert!(
                matches!(*then_branch, CompiledExpr::Int(0)),
                "Then branch should be Int(0), got: {:?}",
                then_branch
            );
            // Else branch should be LET
            assert!(
                matches!(*else_branch, CompiledExpr::Let { .. }),
                "Else branch should be Let, got: {:?}",
                else_branch
            );
        } else {
            panic!("Expected If expression");
        }
    }

    #[test]
    fn test_split_binary_op_respects_if_then() {
        // split_binary_op should NOT split inside IF...THEN
        // The = inside "IF x = 5 THEN" should not be found
        let result = split_binary_op("IF x = 5 THEN y ELSE z", "=");
        assert!(
            result.is_none(),
            "split_binary_op should not split inside IF condition: {:?}",
            result
        );

        // But after THEN, it should be able to split
        let result = split_binary_op("IF x THEN a = b ELSE c", "=");
        assert!(result.is_some(), "split_binary_op should split after THEN");
        let (left, right) = result.unwrap();
        assert_eq!(left.trim(), "IF x THEN a");
        assert_eq!(right.trim(), "b ELSE c");
    }

    #[test]
    fn test_split_binary_op_respects_let_in() {
        // split_binary_op should NOT split inside LET...IN
        let result = split_binary_op("LET x = 5 IN y", "=");
        assert!(
            result.is_none(),
            "split_binary_op should not split inside LET: {:?}",
            result
        );

        // After IN, it should be able to split
        let result = split_binary_op("LET x IN a = b", "=");
        assert!(result.is_some(), "split_binary_op should split after IN");
    }

    #[test]
    fn test_except_with_record_literal_value() {
        // Test EXCEPT expression where the replacement value is a record literal
        // This tests the fix for the bug where EXCEPT followed by newline was not recognized
        let expr_with_newline = r#"[f EXCEPT
![key] = [a |-> 1, b |-> 2]]"#;
        let compiled = compile_expr(expr_with_newline);

        // Should be parsed as FuncExcept, not as RecordLiteral
        assert!(
            matches!(compiled, CompiledExpr::FuncExcept(_, _)),
            "EXCEPT with newline should be parsed as FuncExcept, got: {:?}",
            compiled
        );

        // Verify the structure: FuncExcept with one update whose value is a RecordLiteral
        if let CompiledExpr::FuncExcept(_, updates) = &compiled {
            assert_eq!(updates.len(), 1, "Should have exactly one update");
            let (_, value_expr) = &updates[0];
            assert!(
                matches!(value_expr, CompiledExpr::RecordLiteral(_)),
                "Update value should be RecordLiteral, got: {:?}",
                value_expr
            );
            if let CompiledExpr::RecordLiteral(fields) = value_expr {
                assert_eq!(fields.len(), 2, "Record should have 2 fields");
                assert_eq!(fields[0].0, "a");
                assert_eq!(fields[1].0, "b");
            }
        }
    }

    #[test]
    fn test_except_with_multiline_record_literal() {
        // Test the specific pattern from the bug report
        let expr = r#"[serverCharters EXCEPT
![inode][client] = [
    issuedClock |-> newClock,
    givenAccess |-> grantedAccess,
    isRevoked |-> FALSE
]]"#;
        let compiled = compile_expr(expr);

        assert!(
            matches!(compiled, CompiledExpr::FuncExcept(_, _)),
            "Should be parsed as FuncExcept, got: {:?}",
            compiled
        );

        if let CompiledExpr::FuncExcept(base, updates) = &compiled {
            // Base should be the serverCharters variable
            assert!(
                matches!(base.as_ref(), CompiledExpr::Var(name) if name == "serverCharters"),
                "Base should be Var(serverCharters), got: {:?}",
                base
            );

            // Should have exactly one update with a 2-element path
            assert_eq!(updates.len(), 1, "Should have exactly one update");
            let (path, value) = &updates[0];
            assert_eq!(
                path.len(),
                2,
                "Path should have 2 elements (inode and client)"
            );

            // Value should be a record literal with 3 fields
            assert!(
                matches!(value, CompiledExpr::RecordLiteral(fields) if fields.len() == 3),
                "Value should be RecordLiteral with 3 fields, got: {:?}",
                value
            );
        }
    }
}

#[test]
fn test_forall_with_record_access() {
    // This tests the exact pattern from PriceBandsRespected
    let expr = compile_expr("\\A l \\in listings : l.price >= MinPrice");
    println!("Compiled expression: {:?}", expr);

    match &expr {
        CompiledExpr::Forall { var, domain, body } => {
            println!("Forall var: {}", var);
            println!("Forall domain: {:?}", domain);
            println!("Forall body: {:?}", body);
            assert_eq!(var, "l");
            // Check that body has RecordAccess
            match body.as_ref() {
                CompiledExpr::Ge(left, _) => {
                    println!("Left side of Ge: {:?}", left);
                    match left.as_ref() {
                        CompiledExpr::RecordAccess(base, field) => {
                            println!("RecordAccess base: {:?}, field: {}", base, field);
                            assert_eq!(field, "price");
                            match base.as_ref() {
                                CompiledExpr::Var(v) => {
                                    println!("Var name: {}", v);
                                    assert_eq!(v, "l");
                                }
                                _ => panic!("Expected Var for base, got: {:?}", base),
                            }
                        }
                        _ => panic!("Expected RecordAccess, got: {:?}", left),
                    }
                }
                _ => panic!("Expected Ge in body, got: {:?}", body),
            }
        }
        _ => panic!("Expected Forall, got: {:?}", expr),
    }
}

#[test]
fn test_multiple_quantifiers_in_conjunction() {
    // This tests the exact pattern from PriceBandsRespected
    let expr_str = r#"/\ \A l \in listings : l.price >= MinPrice /\ l.price <= MaxPrice
/\ \A o \in options : o.strike >= MinPrice"#;

    let expr = compile_expr(expr_str);
    println!("Multi-quantifier compiled: {:?}", expr);

    // Should be And([Forall{...}, Forall{...}])
    match &expr {
        CompiledExpr::And(parts) => {
            println!("And with {} parts:", parts.len());
            for (i, part) in parts.iter().enumerate() {
                println!("Part {}: {:?}", i, part);
                // Each part should be a Forall
                match part {
                    CompiledExpr::Forall { var, body, .. } => {
                        println!("  Forall var: {}", var);
                        // Make sure the body contains the full conjunction
                        match body.as_ref() {
                            CompiledExpr::And(body_parts) => {
                                println!("  Body has {} And parts", body_parts.len());
                                assert!(body_parts.len() >= 1, "Body should have at least 1 part");
                            }
                            CompiledExpr::Ge(_, _) => {
                                // This is fine for the second quantifier
                                println!("  Body is Ge (single condition)");
                            }
                            _ => {
                                // For first quantifier, body should be And
                                if i == 0 {
                                    panic!("First quantifier body should be And, got: {:?}", body);
                                }
                            }
                        }
                    }
                    _ => panic!("Part {} should be Forall, got: {:?}", i, part),
                }
            }
            assert_eq!(parts.len(), 2, "Should have exactly 2 quantifiers");
        }
        _ => panic!("Expected And, got: {:?}", expr),
    }
}

#[test]
fn test_domain_with_function_application() {
    // Test that DOMAIN is parsed before function application
    // "DOMAIN f[x]" should parse as Domain(FuncApply(f, [x]))
    // NOT as FuncApply(Domain(f), [x]) which would be incorrect
    let expr = compile_expr("DOMAIN leases[s]");
    match &expr {
        CompiledExpr::Domain(inner) => match inner.as_ref() {
            CompiledExpr::FuncApply(func, args) => {
                assert!(
                    matches!(func.as_ref(), CompiledExpr::Var(name) if name == "leases"),
                    "Expected FuncApply of 'leases', got: {:?}",
                    func
                );
                assert_eq!(args.len(), 1, "Expected 1 argument");
                assert!(
                    matches!(&args[0], CompiledExpr::Var(name) if name == "s"),
                    "Expected argument 's', got: {:?}",
                    args[0]
                );
            }
            other => panic!(
                "DOMAIN f[x] should have FuncApply inside Domain, got: {:?}",
                other
            ),
        },
        other => panic!("DOMAIN f[x] should parse as Domain, got: {:?}", other),
    }
}

#[test]
fn test_domain_simple() {
    // Test simple DOMAIN expression
    let expr = compile_expr("DOMAIN leases");
    match &expr {
        CompiledExpr::Domain(inner) => {
            assert!(
                matches!(inner.as_ref(), CompiledExpr::Var(name) if name == "leases"),
                "Expected Var('leases'), got: {:?}",
                inner
            );
        }
        other => panic!("Expected Domain, got: {:?}", other),
    }
}

#[test]
fn test_domain_equality() {
    // Test DOMAIN x = S parses correctly
    let expr = compile_expr("DOMAIN leases = Shards");
    match &expr {
        CompiledExpr::Eq(left, right) => {
            assert!(
                matches!(left.as_ref(), CompiledExpr::Domain(_)),
                "Left side should be Domain, got: {:?}",
                left
            );
            assert!(
                matches!(right.as_ref(), CompiledExpr::Var(name) if name == "Shards"),
                "Right side should be Var('Shards'), got: {:?}",
                right
            );
        }
        other => panic!("Expected Eq, got: {:?}", other),
    }
}

#[test]
fn test_subset_with_function_application() {
    // Similar test for SUBSET prefix operator
    let expr = compile_expr("SUBSET sets[i]");
    match &expr {
        CompiledExpr::PowerSet(inner) => match inner.as_ref() {
            CompiledExpr::FuncApply(func, args) => {
                assert!(
                    matches!(func.as_ref(), CompiledExpr::Var(name) if name == "sets"),
                    "Expected FuncApply of 'sets', got: {:?}",
                    func
                );
                assert_eq!(args.len(), 1);
            }
            other => panic!(
                "SUBSET f[x] should have FuncApply inside PowerSet, got: {:?}",
                other
            ),
        },
        other => panic!("Expected PowerSet, got: {:?}", other),
    }
}

#[test]
fn test_split_forall_body_with_leading_conjunction() {
    // This is the exact body extracted from:
    // \A s \in Shards :
    // /\ shardTerm[s] \in 0..MaxTerm
    // /\ \A c \in Clients :
    //     /\ leases[s][c].held \in BOOLEAN

    let body = r#"/\ shardTerm[s] \in 0..MaxTerm
/\ \A c \in Clients :
    /\ leases[s][c].held \in BOOLEAN"#;

    let parts = split_top_level(body.trim(), "/\\");

    println!("Body: {:?}", body);
    println!("Number of parts: {}", parts.len());
    for (i, part) in parts.iter().enumerate() {
        println!("Part {}: {:?}", i, part);
    }

    // The body should split into 2 parts:
    // 1. shardTerm[s] \in 0..MaxTerm
    // 2. \A c \in Clients : /\ leases[s][c].held \in BOOLEAN
    // NOT 3 parts where the inner /\ is also split
    assert_eq!(parts.len(), 2, "Body should split into exactly 2 parts");

    // Verify the second part contains the inner quantifier's full body
    assert!(
        parts[1].contains("\\A c \\in Clients"),
        "Second part should contain the nested quantifier"
    );
    assert!(
        parts[1].contains("leases[s][c].held"),
        "Second part should contain the nested quantifier's body"
    );
}

#[test]
fn test_typeok_pattern_domain_checks_then_quantifier() {
    // The exact TypeOK pattern that fails:
    // /\ DOMAIN shardTerm = Shards
    // /\ DOMAIN shardLeader = Shards
    // /\ \A s \in Shards :
    //     /\ shardTerm[s] \in 0..MaxTerm
    //     /\ shardLeader[s] \in {"none", "stable", "electing"}

    let expr_str = r#"/\ DOMAIN shardTerm = Shards
/\ DOMAIN shardLeader = Shards
/\ \A s \in Shards :
    /\ shardTerm[s] \in 0..MaxTerm
    /\ shardLeader[s] \in {"none", "stable", "electing"}"#;

    let expr = compile_expr(expr_str);
    println!("TypeOK pattern compiled: {:#?}", expr);

    // Should compile to And([Eq(...), Eq(...), Forall{...}])
    match &expr {
        CompiledExpr::And(parts) => {
            println!("And with {} parts", parts.len());
            assert_eq!(parts.len(), 3, "Should have exactly 3 conjuncts");

            // First two should be Eq (DOMAIN x = S)
            assert!(
                matches!(&parts[0], CompiledExpr::Eq(_, _)),
                "First conjunct should be Eq: {:?}",
                parts[0]
            );
            assert!(
                matches!(&parts[1], CompiledExpr::Eq(_, _)),
                "Second conjunct should be Eq: {:?}",
                parts[1]
            );

            // Third should be Forall
            match &parts[2] {
                CompiledExpr::Forall { var, body, .. } => {
                    assert_eq!(var, "s", "Quantifier variable should be 's'");
                    println!("Forall body: {:#?}", body);

                    // The body should be an And with 2 parts
                    match body.as_ref() {
                        CompiledExpr::And(body_parts) => {
                            assert_eq!(
                                body_parts.len(),
                                2,
                                "Forall body should have 2 conjuncts, got: {:?}",
                                body_parts
                            );
                        }
                        other => panic!("Forall body should be And, got: {:?}", other),
                    }
                }
                other => panic!("Third conjunct should be Forall, got: {:?}", other),
            }
        }
        other => panic!("Expected And, got: {:?}", other),
    }
}

#[test]
fn test_forall_with_leading_conjunction_body() {
    // This is the EXACT pattern from the failing test in compiled_eval.rs
    // The quantifier body starts with /\ on a new line

    let expr_str = r#"\A s \in Shards :
/\ shardTerm[s] \in 0..MaxTerm
/\ \A c \in Clients :
    /\ leases[s][c].held \in BOOLEAN"#;

    let expr = compile_expr(expr_str);

    // Should compile to Forall { var: "s", domain: ..., body: And([In(...), Forall{...}]) }
    match &expr {
        CompiledExpr::Forall { var, body, .. } => {
            assert_eq!(var, "s", "Outer quantifier variable should be 's'");

            // The body should be an And with 2 parts
            match body.as_ref() {
                CompiledExpr::And(body_parts) => {
                    assert_eq!(
                        body_parts.len(),
                        2,
                        "Forall body should have 2 conjuncts, got {} parts: {:?}",
                        body_parts.len(),
                        body_parts
                    );

                    // First part should be In (shardTerm[s] \in 0..MaxTerm)
                    assert!(
                        matches!(&body_parts[0], CompiledExpr::In(_, _)),
                        "First body part should be In, got: {:?}",
                        body_parts[0]
                    );

                    // Second part should be nested Forall
                    match &body_parts[1] {
                        CompiledExpr::Forall {
                            var: inner_var,
                            body: inner_body,
                            ..
                        } => {
                            assert_eq!(inner_var, "c", "Inner quantifier variable should be 'c'");
                            // Inner body should be In (leases[s][c].held \in BOOLEAN)
                            assert!(
                                matches!(inner_body.as_ref(), CompiledExpr::In(_, _)),
                                "Inner body should be In, got: {:?}",
                                inner_body
                            );
                        }
                        other => panic!("Second body part should be Forall, got: {:?}", other),
                    }
                }
                other => panic!("Forall body should be And, got: {:?}", other),
            }
        }
        other => panic!("Expected Forall, got: {:?}", other),
    }
}

#[test]
fn test_single_conjunct_with_leading_conjunction() {
    // Test that "/\ expr" compiles to just "expr" when there's only one conjunct
    let expr = compile_expr("/\\ x > 0");
    assert!(
        matches!(expr, CompiledExpr::Gt(_, _)),
        "Single conjunct should compile to the inner expression, got: {:?}",
        expr
    );

    // Same for disjunction
    let expr = compile_expr("\\/ x > 0");
    assert!(
        matches!(expr, CompiledExpr::Gt(_, _)),
        "Single disjunct should compile to the inner expression, got: {:?}",
        expr
    );
}

#[test]
fn test_compile_unchanged_in_expression_context() {
    // UNCHANGED should compile to Bool(true) in expression context
    let expr = compile_expr("UNCHANGED x");
    assert!(
        matches!(expr, CompiledExpr::Bool(true)),
        "UNCHANGED x should compile to Bool(true), got: {:?}",
        expr
    );

    let expr = compile_expr("UNCHANGED <<x, y>>");
    assert!(
        matches!(expr, CompiledExpr::Bool(true)),
        "UNCHANGED <<x, y>> should compile to Bool(true), got: {:?}",
        expr
    );

    // UNCHANGED inside a disjunction should not cause errors
    let expr = compile_expr("x > 0 \\/ UNCHANGED vars");
    assert!(
        matches!(expr, CompiledExpr::Or(_)),
        "disjunction with UNCHANGED should compile to Or, got: {:?}",
        expr
    );
    if let CompiledExpr::Or(parts) = &expr {
        assert_eq!(parts.len(), 2);
        assert!(
            matches!(parts[1], CompiledExpr::Bool(true)),
            "UNCHANGED disjunct should be Bool(true), got: {:?}",
            parts[1]
        );
    }
}
