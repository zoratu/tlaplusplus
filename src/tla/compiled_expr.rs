//! Compiled TLA+ expressions for fast evaluation
//!
//! Instead of parsing expression strings on every evaluation, we compile them
//! once to an AST and then evaluate the AST. This eliminates:
//! - String parsing overhead
//! - String allocation during evaluation
//! - Repeated pattern matching on string prefixes
//!
//! Benchmarks show this can provide 3-5x speedup on expression-heavy workloads.

use crate::tla::{TlaDefinition, TlaValue};
use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
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
    Div(Box<CompiledExpr>, Box<CompiledExpr>),
    Mod(Box<CompiledExpr>, Box<CompiledExpr>),
    Neg(Box<CompiledExpr>),

    // Set operations
    SetLiteral(Vec<CompiledExpr>),
    SetRange(Box<CompiledExpr>, Box<CompiledExpr>),
    Union(Box<CompiledExpr>, Box<CompiledExpr>),
    Intersect(Box<CompiledExpr>, Box<CompiledExpr>),
    SetMinus(Box<CompiledExpr>, Box<CompiledExpr>),
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

    // Function operations
    FuncLiteral(Vec<(CompiledExpr, CompiledExpr)>),
    FuncApply(Box<CompiledExpr>, Vec<CompiledExpr>),
    FuncExcept(Box<CompiledExpr>, Vec<(Vec<CompiledExpr>, CompiledExpr)>),
    Domain(Box<CompiledExpr>),

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
            | CompiledExpr::Div(a, b)
            | CompiledExpr::Mod(a, b)
            | CompiledExpr::Union(a, b)
            | CompiledExpr::Intersect(a, b)
            | CompiledExpr::SetMinus(a, b)
            | CompiledExpr::Subset(a, b)
            | CompiledExpr::SetRange(a, b)
            | CompiledExpr::Append(a, b)
            | CompiledExpr::Concat(a, b) => a.is_fully_compiled() && b.is_fully_compiled(),
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
            CompiledExpr::RecordLiteral(fields) => {
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

    // String literal
    if expr.starts_with('"') && expr.ends_with('"') && expr.len() >= 2 {
        return CompiledExpr::String(expr[1..expr.len() - 1].to_string());
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

    // Logical operators (in precedence order)

    // Implication: =>
    if let Some((left, right)) = split_binary_op(expr, "=>") {
        return CompiledExpr::Implies(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Disjunction: \/
    let or_parts = split_top_level(expr, "\\/");
    if or_parts.len() > 1 {
        return CompiledExpr::Or(or_parts.into_iter().map(|s| compile_expr(&s)).collect());
    }

    // Conjunction: /\
    let and_parts = split_top_level(expr, "/\\");
    if and_parts.len() > 1 {
        return CompiledExpr::And(and_parts.into_iter().map(|s| compile_expr(&s)).collect());
    }

    // Negation: ~
    if let Some(rest) = expr.strip_prefix('~') {
        return CompiledExpr::Not(Box::new(compile_expr(rest.trim())));
    }

    // Comparison operators
    if let Some((left, op, right)) = split_comparison(expr) {
        let left_expr = Box::new(compile_expr(left));
        let right_expr = Box::new(compile_expr(right));
        return match op {
            "=" => CompiledExpr::Eq(left_expr, right_expr),
            "/=" | "#" => CompiledExpr::Neq(left_expr, right_expr),
            "<" => CompiledExpr::Lt(left_expr, right_expr),
            "<=" => CompiledExpr::Le(left_expr, right_expr),
            ">" => CompiledExpr::Gt(left_expr, right_expr),
            ">=" => CompiledExpr::Ge(left_expr, right_expr),
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
    if let Some((left, right)) = split_binary_op(expr, "\\") {
        return CompiledExpr::SetMinus(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
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
    }

    // Record access: r.field
    if let Some(dot_idx) = find_record_access_dot(expr) {
        let base = &expr[..dot_idx];
        let field = &expr[dot_idx + 1..];
        if is_identifier(field) {
            return CompiledExpr::RecordAccess(Box::new(compile_expr(base)), field.to_string());
        }
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
    if let Some(inner) = expr.strip_prefix("SUBSET ") {
        return CompiledExpr::PowerSet(Box::new(compile_expr(inner)));
    }
    if let Some(inner) = expr.strip_prefix("DOMAIN ") {
        return CompiledExpr::Domain(Box::new(compile_expr(inner)));
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
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn split_top_level(expr: &str, delim: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut let_depth: usize = 0; // Track LET...IN nesting
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
    let delim_chars: Vec<char> = delim.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

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
                continue;
            }
            if c == '>' && chars[i + 1] == '>' {
                depth -= 1;
                current.push(c);
                current.push(chars[i + 1]);
                i += 2;
                continue;
            }
        }

        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }

        // Check for LET keyword at word boundary when at bracket top level
        if depth == 0 && matches_keyword_at(&chars, i, "LET") {
            let_depth += 1;
            current.push_str("LET");
            i += 3;
            continue;
        }

        // Check for IN keyword at word boundary when inside a LET
        if depth == 0 && let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
            let_depth = let_depth.saturating_sub(1);
            current.push_str("IN");
            i += 2;
            continue;
        }

        if depth == 0 && let_depth == 0 {
            // Check for delimiter
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with(delim) {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current = String::new();
                i += delim_chars.len();
                continue;
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
    let mut depth = 0;
    let mut in_string = false;
    let bytes = expr.as_bytes();
    let op_bytes = op.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_string = !in_string;
            i += 1;
            continue;
        }
        if in_string {
            i += 1;
            continue;
        }

        // Handle << and >> for sequence literals
        if i + 1 < bytes.len() {
            if bytes[i] == b'<' && bytes[i + 1] == b'<' {
                depth += 1;
                i += 2;
                continue;
            }
            if bytes[i] == b'>' && bytes[i + 1] == b'>' {
                depth -= 1;
                i += 2;
                continue;
            }
        }

        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }

        if depth == 0 && i + op_bytes.len() <= bytes.len() {
            if &bytes[i..i + op_bytes.len()] == op_bytes {
                return Some((&expr[..i], &expr[i + op_bytes.len()..]));
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
        ">=",
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

/// Try to parse an EXCEPT expression
/// Patterns:
/// - `f EXCEPT ![key] = val`
/// - `f EXCEPT ![k1][k2] = val` (nested path)
/// - `f EXCEPT ![k1] = val, ![k2] = val2` (multiple updates)
/// - `f EXCEPT ![k] = @ + 1` (@ refers to original value)
fn try_parse_except(inner: &str) -> Option<CompiledExpr> {
    // Find "EXCEPT" keyword
    let except_pos = inner.find(" EXCEPT ")?;
    let base = inner[..except_pos].trim();
    let updates_str = inner[except_pos + 8..].trim();

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
        let eq_pos = find_top_level_eq(&part[1..])?;
        let path_str = &part[1..eq_pos + 1].trim();
        let value_str = part[eq_pos + 2..].trim();

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

/// Parse EXCEPT path: [k1][k2][k3]... -> vec![k1, k2, k3]
fn parse_except_path(s: &str) -> Option<Vec<CompiledExpr>> {
    let mut path = Vec::new();
    let mut remaining = s.trim();

    while !remaining.is_empty() {
        if !remaining.starts_with('[') {
            break;
        }

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

fn find_top_level_colon(expr: &str) -> Option<usize> {
    let mut depth = 0;
    let bytes = expr.as_bytes();

    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b':' if depth == 0 => return Some(i),
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

    // Parse definitions (simplified: name == expr)
    let mut bindings = Vec::new();
    for def in split_top_level(defs_str, "\n") {
        let def = def.trim();
        if def.is_empty() {
            continue;
        }
        if let Some(eq_idx) = def.find("==") {
            let name = def[..eq_idx].trim();
            let value = def[eq_idx + 2..].trim();
            if is_identifier(name) {
                bindings.push((name.to_string(), compile_expr(value)));
            }
        }
    }

    Some(CompiledExpr::Let {
        bindings,
        body: Box::new(compile_expr(body_str)),
    })
}

fn try_parse_exists(expr: &str) -> Option<CompiledExpr> {
    // \E x \in S : body  OR  \E i, j \in S : body (multi-variable same domain)
    // OR  \E x \in S, y \in T : body (multiple separate bindings)
    let rest = expr.strip_prefix("\\E ")?.trim();

    let colon_idx = find_top_level_colon(rest)?;
    let binding = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();

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

    let (var, domain) = parse_in_binding(binding)?;

    Some(CompiledExpr::Choose {
        var: var.to_string(),
        domain: Box::new(compile_expr(domain)),
        body: Box::new(compile_expr(body)),
    })
}

fn find_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let mut depth = 0;
    let bytes = expr.as_bytes();
    let keyword_bytes = keyword.as_bytes();

    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ if depth == 0 => {
                if i + keyword_bytes.len() <= bytes.len()
                    && &bytes[i..i + keyword_bytes.len()] == keyword_bytes
                {
                    // Check it's a word boundary
                    let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                    let after_ok = i + keyword_bytes.len() >= bytes.len()
                        || !bytes[i + keyword_bytes.len()].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        return Some(i);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

// ============================================================================
// Compiled Action IR - pre-compiled action clauses for fast execution
// ============================================================================

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
        let clauses = ir
            .clauses
            .iter()
            .map(|clause| match clause {
                ActionClause::PrimedAssignment { var, expr } => {
                    CompiledActionClause::PrimedAssignment {
                        var: var.clone(),
                        expr: compile_expr(expr),
                    }
                }
                ActionClause::Unchanged { vars } => {
                    CompiledActionClause::Unchanged { vars: vars.clone() }
                }
                ActionClause::Guard { expr } => CompiledActionClause::Guard {
                    expr: compile_expr(expr),
                },
                ActionClause::LetWithPrimes { expr } => {
                    // Try to compile the LET expression
                    match compile_let_with_primes(expr) {
                        Some(compiled) => compiled,
                        None => CompiledActionClause::LetWithPrimes { expr: expr.clone() },
                    }
                }
            })
            .collect();

        CompiledActionIr {
            name: ir.name.clone(),
            params: ir.params.clone(),
            clauses,
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
    let body_conjuncts = split_top_level(body_text, "/\\");

    // Compile each clause in the body
    let mut body_clauses = Vec::new();
    for conjunct in body_conjuncts {
        let trimmed = conjunct.trim();
        if trimmed.is_empty() {
            continue;
        }

        match classify_clause(trimmed) {
            ClauseKind::PrimedAssignment { var, expr } => {
                body_clauses.push(CompiledActionClause::PrimedAssignment {
                    var,
                    expr: compile_expr(&expr),
                });
            }
            ClauseKind::Unchanged { vars } => {
                body_clauses.push(CompiledActionClause::Unchanged { vars });
            }
            ClauseKind::UnprimedEquality { .. } | ClauseKind::Other => {
                // Check for nested LET with primes
                if trimmed.starts_with("LET") && trimmed.contains('\'') {
                    match compile_let_with_primes(trimmed) {
                        Some(compiled) => body_clauses.push(compiled),
                        None => return None, // Can't compile nested LET, fall back
                    }
                } else {
                    // Treat as guard
                    body_clauses.push(CompiledActionClause::Guard {
                        expr: compile_expr(trimmed),
                    });
                }
            }
        }
    }

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

        let name = defs_text[name_start..*eq_pos].trim();
        let body = defs_text[*eq_pos + 2..body_end].trim();

        // Name should be a simple identifier (possibly with params, but we ignore those for now)
        let name = if let Some(paren) = name.find('(') {
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

/// Find all == positions at top level (not inside brackets/parens)
fn find_definition_equals(text: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let bytes = text.as_bytes();
    let mut depth: usize = 0;
    let mut i = 0;

    while i + 1 < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'<' => {
                depth += 1;
                i += 1;
            }
            b'>' if i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'=' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'=' => {
                // Found == at top level
                // Make sure it's not inside a string
                positions.push(i);
                i += 1; // Skip the second =
            }
            _ => {}
        }
        i += 1;
    }

    positions
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
}
