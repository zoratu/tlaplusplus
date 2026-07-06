//! Compiled TLA+ expressions for fast evaluation
//!
//! Instead of parsing expression strings on every evaluation, we compile them
//! once to an AST and then evaluate the AST. This eliminates:
//! - String parsing overhead
//! - String allocation during evaluation
//! - Repeated pattern matching on string prefixes
//!
//! Benchmarks show this can provide 3-5x speedup on expression-heavy workloads.

use std::collections::BTreeSet;
use std::sync::Arc;
use crate::tla::hashed_arc::HashedArc;
use crate::tla::value::StateSchema;

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

    // Resolved state-variable reference. The name is the Arc from the schema
    // used during resolution; eval only takes the slot fast path when the
    // runtime state's schema has the exact same Arc at that slot.
    StateVar { slot: u32, name: Arc<str> },

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
    /// Logical equivalence (biconditional): a <=> b. True iff both sides
    /// evaluate to the same Boolean.
    Iff(Box<CompiledExpr>, Box<CompiledExpr>),

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
            | CompiledExpr::Iff(a, b)
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
            | CompiledExpr::StateVar { .. }
            | CompiledExpr::PrimedVar(_)
            | CompiledExpr::SelfRef
            | CompiledExpr::NatSet
            | CompiledExpr::IntSet
            | CompiledExpr::BooleanSet => true,
        }
    }
}

pub fn resolve_state_vars(expr: &mut CompiledExpr, schema: &StateSchema) {
    resolve_state_vars_with_binders(expr, schema, std::iter::empty::<&str>());
}

pub fn resolve_state_vars_with_binders<'a, I>(
    expr: &mut CompiledExpr,
    schema: &StateSchema,
    external_binders: I,
) where
    I: IntoIterator<Item = &'a str>,
{
    let mut binders = BTreeSet::new();
    for binder in external_binders {
        insert_binder_name(&mut binders, binder);
    }
    collect_expr_binders(expr, &mut binders);
    resolve_expr_slots(expr, schema, &binders);
}

pub fn resolve_state_vars_in_action_ir(action: &mut CompiledActionIr, schema: &StateSchema) {
    let mut binders = BTreeSet::new();
    for param in &action.params {
        insert_binder_name(&mut binders, param);
    }
    collect_action_binders(&action.clauses, &mut binders);
    resolve_action_clause_slots(&mut action.clauses, schema, &binders);
}

fn insert_binder_name(binders: &mut BTreeSet<String>, name: &str) {
    let name = normalize_binder_name(name);
    if name.is_empty() {
        return;
    }
    if is_identifier(name) {
        binders.insert(name.to_string());
    } else {
        collect_identifier_names(name, binders);
    }
}

fn normalize_binder_name(name: &str) -> &str {
    let name = name.trim();
    let name = if let Some(in_pos) = name.find("\\in") {
        name[..in_pos].trim()
    } else {
        name
    };
    if let Some(paren_pos) = name.find('(') {
        name[..paren_pos].trim()
    } else {
        name.trim()
    }
}

fn collect_identifier_names(text: &str, binders: &mut BTreeSet<String>) {
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            let ident = &text[s..idx];
            if ident
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
                && is_identifier(ident)
            {
                insert_binder_name(binders, ident);
            }
        }
    }
    if let Some(s) = start {
        let ident = &text[s..];
        if ident
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
            && is_identifier(ident)
        {
            insert_binder_name(binders, ident);
        }
    }
}

fn collect_action_binder_names(text: &str, binders: &mut BTreeSet<String>) {
    for segment in split_top_level(text, ",") {
        let lhs = segment
            .split_once("\\in")
            .map(|(vars, _)| vars)
            .unwrap_or(segment.as_str());
        collect_identifier_names(lhs, binders);
    }
}

fn collect_expr_binders(expr: &CompiledExpr, binders: &mut BTreeSet<String>) {
    match expr {
        CompiledExpr::Bool(_)
        | CompiledExpr::Int(_)
        | CompiledExpr::String(_)
        | CompiledExpr::ModelValue(_)
        | CompiledExpr::Var(_)
        | CompiledExpr::StateVar { .. }
        | CompiledExpr::PrimedVar(_)
        | CompiledExpr::SelfRef
        | CompiledExpr::NatSet
        | CompiledExpr::IntSet
        | CompiledExpr::BooleanSet
        | CompiledExpr::Unparsed(_) => {}
        CompiledExpr::And(exprs)
        | CompiledExpr::Or(exprs)
        | CompiledExpr::SetLiteral(exprs)
        | CompiledExpr::SeqLiteral(exprs) => {
            for expr in exprs {
                collect_expr_binders(expr, binders);
            }
        }
        CompiledExpr::Not(expr)
        | CompiledExpr::Neg(expr)
        | CompiledExpr::Cardinality(expr)
        | CompiledExpr::PowerSet(expr)
        | CompiledExpr::Head(expr)
        | CompiledExpr::Tail(expr)
        | CompiledExpr::Len(expr)
        | CompiledExpr::Domain(expr)
        | CompiledExpr::RecordAccess(expr, _) => collect_expr_binders(expr, binders),
        CompiledExpr::Implies(left, right)
        | CompiledExpr::Iff(left, right)
        | CompiledExpr::Eq(left, right)
        | CompiledExpr::Neq(left, right)
        | CompiledExpr::Lt(left, right)
        | CompiledExpr::Le(left, right)
        | CompiledExpr::Gt(left, right)
        | CompiledExpr::Ge(left, right)
        | CompiledExpr::In(left, right)
        | CompiledExpr::NotIn(left, right)
        | CompiledExpr::Add(left, right)
        | CompiledExpr::Sub(left, right)
        | CompiledExpr::Mul(left, right)
        | CompiledExpr::Pow(left, right)
        | CompiledExpr::Div(left, right)
        | CompiledExpr::Mod(left, right)
        | CompiledExpr::SetRange(left, right)
        | CompiledExpr::Union(left, right)
        | CompiledExpr::Intersect(left, right)
        | CompiledExpr::SetMinus(left, right)
        | CompiledExpr::CartesianProduct(left, right)
        | CompiledExpr::Subset(left, right)
        | CompiledExpr::Append(left, right)
        | CompiledExpr::Concat(left, right)
        | CompiledExpr::FuncPair(left, right)
        | CompiledExpr::FuncOverride(left, right)
        | CompiledExpr::FunctionSet {
            domain: left,
            range: right,
        } => {
            collect_expr_binders(left, binders);
            collect_expr_binders(right, binders);
        }
        CompiledExpr::SubSeq(seq, start, end) => {
            collect_expr_binders(seq, binders);
            collect_expr_binders(start, binders);
            collect_expr_binders(end, binders);
        }
        CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_expr_binders(cond, binders);
            collect_expr_binders(then_branch, binders);
            collect_expr_binders(else_branch, binders);
        }
        CompiledExpr::Case { arms, other } => {
            for (cond, value) in arms {
                collect_expr_binders(cond, binders);
                collect_expr_binders(value, binders);
            }
            if let Some(other) = other {
                collect_expr_binders(other, binders);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (name, value) in bindings {
                insert_binder_name(binders, name);
                collect_expr_binders(value, binders);
            }
            collect_expr_binders(body, binders);
        }
        CompiledExpr::Exists { var, domain, body }
        | CompiledExpr::Forall { var, domain, body }
        | CompiledExpr::Choose { var, domain, body }
        | CompiledExpr::FuncConstruct { var, domain, body } => {
            insert_binder_name(binders, var);
            collect_expr_binders(domain, binders);
            collect_expr_binders(body, binders);
        }
        CompiledExpr::SetComprehension {
            var,
            domain,
            body,
            filter,
        } => {
            insert_binder_name(binders, var);
            collect_expr_binders(domain, binders);
            collect_expr_binders(body, binders);
            if let Some(filter) = filter {
                collect_expr_binders(filter, binders);
            }
        }
        CompiledExpr::RecordLiteral(fields) | CompiledExpr::RecordSet(fields) => {
            for (_, value) in fields {
                collect_expr_binders(value, binders);
            }
        }
        CompiledExpr::FuncLiteral(entries) => {
            for (key, value) in entries {
                collect_expr_binders(key, binders);
                collect_expr_binders(value, binders);
            }
        }
        CompiledExpr::FuncApply(func, args) => {
            collect_expr_binders(func, binders);
            for arg in args {
                collect_expr_binders(arg, binders);
            }
        }
        CompiledExpr::FuncExcept(base, updates) => {
            collect_expr_binders(base, binders);
            for (path, value) in updates {
                for path_expr in path {
                    collect_expr_binders(path_expr, binders);
                }
                collect_expr_binders(value, binders);
            }
        }
        CompiledExpr::OpCall { args, .. } => {
            for arg in args {
                collect_expr_binders(arg, binders);
            }
        }
        CompiledExpr::Lambda { params, body, .. } => {
            for param in params {
                insert_binder_name(binders, param);
            }
            collect_expr_binders(body, binders);
        }
    }
}

fn resolve_expr_slots(expr: &mut CompiledExpr, schema: &StateSchema, binders: &BTreeSet<String>) {
    match expr {
        CompiledExpr::Var(name) => {
            let replacement = if binders.contains(name.as_str()) {
                None
            } else {
                schema.slot_of(name.as_str()).map(|slot| {
                    let schema_name = Arc::clone(
                        schema
                            .name_at(slot)
                            .expect("slot from schema map must have a name"),
                    );
                    (slot, schema_name)
                })
            };
            if let Some((slot, name)) = replacement {
                *expr = CompiledExpr::StateVar { slot, name };
            }
        }
        CompiledExpr::Bool(_)
        | CompiledExpr::Int(_)
        | CompiledExpr::String(_)
        | CompiledExpr::ModelValue(_)
        | CompiledExpr::StateVar { .. }
        | CompiledExpr::PrimedVar(_)
        | CompiledExpr::SelfRef
        | CompiledExpr::NatSet
        | CompiledExpr::IntSet
        | CompiledExpr::BooleanSet
        | CompiledExpr::Unparsed(_) => {}
        CompiledExpr::And(exprs)
        | CompiledExpr::Or(exprs)
        | CompiledExpr::SetLiteral(exprs)
        | CompiledExpr::SeqLiteral(exprs) => {
            for expr in exprs {
                resolve_expr_slots(expr, schema, binders);
            }
        }
        CompiledExpr::Not(expr)
        | CompiledExpr::Neg(expr)
        | CompiledExpr::Cardinality(expr)
        | CompiledExpr::PowerSet(expr)
        | CompiledExpr::Head(expr)
        | CompiledExpr::Tail(expr)
        | CompiledExpr::Len(expr)
        | CompiledExpr::Domain(expr)
        | CompiledExpr::RecordAccess(expr, _) => resolve_expr_slots(expr, schema, binders),
        CompiledExpr::Implies(left, right)
        | CompiledExpr::Iff(left, right)
        | CompiledExpr::Eq(left, right)
        | CompiledExpr::Neq(left, right)
        | CompiledExpr::Lt(left, right)
        | CompiledExpr::Le(left, right)
        | CompiledExpr::Gt(left, right)
        | CompiledExpr::Ge(left, right)
        | CompiledExpr::In(left, right)
        | CompiledExpr::NotIn(left, right)
        | CompiledExpr::Add(left, right)
        | CompiledExpr::Sub(left, right)
        | CompiledExpr::Mul(left, right)
        | CompiledExpr::Pow(left, right)
        | CompiledExpr::Div(left, right)
        | CompiledExpr::Mod(left, right)
        | CompiledExpr::SetRange(left, right)
        | CompiledExpr::Union(left, right)
        | CompiledExpr::Intersect(left, right)
        | CompiledExpr::SetMinus(left, right)
        | CompiledExpr::CartesianProduct(left, right)
        | CompiledExpr::Subset(left, right)
        | CompiledExpr::Append(left, right)
        | CompiledExpr::Concat(left, right)
        | CompiledExpr::FuncPair(left, right)
        | CompiledExpr::FuncOverride(left, right)
        | CompiledExpr::FunctionSet {
            domain: left,
            range: right,
        } => {
            resolve_expr_slots(left, schema, binders);
            resolve_expr_slots(right, schema, binders);
        }
        CompiledExpr::SubSeq(seq, start, end) => {
            resolve_expr_slots(seq, schema, binders);
            resolve_expr_slots(start, schema, binders);
            resolve_expr_slots(end, schema, binders);
        }
        CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            resolve_expr_slots(cond, schema, binders);
            resolve_expr_slots(then_branch, schema, binders);
            resolve_expr_slots(else_branch, schema, binders);
        }
        CompiledExpr::Case { arms, other } => {
            for (cond, value) in arms {
                resolve_expr_slots(cond, schema, binders);
                resolve_expr_slots(value, schema, binders);
            }
            if let Some(other) = other {
                resolve_expr_slots(other, schema, binders);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, value) in bindings {
                resolve_expr_slots(value, schema, binders);
            }
            resolve_expr_slots(body, schema, binders);
        }
        CompiledExpr::Exists { domain, body, .. }
        | CompiledExpr::Forall { domain, body, .. }
        | CompiledExpr::Choose { domain, body, .. }
        | CompiledExpr::FuncConstruct { domain, body, .. } => {
            resolve_expr_slots(domain, schema, binders);
            resolve_expr_slots(body, schema, binders);
        }
        CompiledExpr::SetComprehension {
            domain,
            body,
            filter,
            ..
        } => {
            resolve_expr_slots(domain, schema, binders);
            resolve_expr_slots(body, schema, binders);
            if let Some(filter) = filter {
                resolve_expr_slots(filter, schema, binders);
            }
        }
        CompiledExpr::RecordLiteral(fields) | CompiledExpr::RecordSet(fields) => {
            for (_, value) in fields {
                resolve_expr_slots(value, schema, binders);
            }
        }
        CompiledExpr::FuncLiteral(entries) => {
            for (key, value) in entries {
                resolve_expr_slots(key, schema, binders);
                resolve_expr_slots(value, schema, binders);
            }
        }
        CompiledExpr::FuncApply(func, args) => {
            resolve_expr_slots(func, schema, binders);
            for arg in args {
                resolve_expr_slots(arg, schema, binders);
            }
        }
        CompiledExpr::FuncExcept(base, updates) => {
            resolve_expr_slots(base, schema, binders);
            for (path, value) in updates {
                for path_expr in path {
                    resolve_expr_slots(path_expr, schema, binders);
                }
                resolve_expr_slots(value, schema, binders);
            }
        }
        CompiledExpr::OpCall { args, .. } => {
            for arg in args {
                resolve_expr_slots(arg, schema, binders);
            }
        }
        CompiledExpr::Lambda { body, .. } => resolve_expr_slots(body, schema, binders),
    }
}

fn collect_action_binders(clauses: &[CompiledActionClause], binders: &mut BTreeSet<String>) {
    for clause in clauses {
        match clause {
            CompiledActionClause::PrimedAssignment { expr, .. } => {
                collect_expr_binders(expr, binders);
            }
            CompiledActionClause::PrimedMembership { set_expr, .. } => {
                collect_expr_binders(set_expr, binders);
            }
            CompiledActionClause::Unchanged { .. } => {}
            CompiledActionClause::Guard { expr, .. } => {
                collect_expr_binders(expr, binders);
            }
            CompiledActionClause::Exists {
                binders: action_binders,
                body_clauses,
            } => {
                collect_action_binder_names(action_binders, binders);
                collect_action_binders(body_clauses, binders);
            }
            CompiledActionClause::Conditional {
                condition,
                then_clauses,
                else_clauses,
            } => {
                collect_expr_binders(condition, binders);
                collect_action_binders(then_clauses, binders);
                collect_action_binders(else_clauses, binders);
            }
            CompiledActionClause::CompiledLetWithPrimes {
                bindings,
                body_clauses,
            } => {
                for (name, expr) in bindings {
                    insert_binder_name(binders, name);
                    collect_expr_binders(expr, binders);
                }
                collect_action_binders(body_clauses, binders);
            }
            CompiledActionClause::LetWithPrimes { .. } => {}
        }
    }
}

fn resolve_action_clause_slots(
    clauses: &mut [CompiledActionClause],
    schema: &StateSchema,
    binders: &BTreeSet<String>,
) {
    for clause in clauses {
        match clause {
            CompiledActionClause::PrimedAssignment { expr, .. } => {
                resolve_expr_slots(expr, schema, binders);
            }
            CompiledActionClause::PrimedMembership { set_expr, .. } => {
                resolve_expr_slots(set_expr, schema, binders);
            }
            CompiledActionClause::Unchanged { .. } => {}
            CompiledActionClause::Guard { expr, .. } => {
                resolve_expr_slots(expr, schema, binders);
            }
            CompiledActionClause::Exists { body_clauses, .. } => {
                resolve_action_clause_slots(body_clauses, schema, binders);
            }
            CompiledActionClause::Conditional {
                condition,
                then_clauses,
                else_clauses,
            } => {
                resolve_expr_slots(condition, schema, binders);
                resolve_action_clause_slots(then_clauses, schema, binders);
                resolve_action_clause_slots(else_clauses, schema, binders);
            }
            CompiledActionClause::CompiledLetWithPrimes {
                bindings,
                body_clauses,
            } => {
                for (_, expr) in bindings {
                    resolve_expr_slots(expr, schema, binders);
                }
                resolve_action_clause_slots(body_clauses, schema, binders);
            }
            CompiledActionClause::LetWithPrimes { .. } => {}
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
    let mut i = 1; // Skip opening quote (always 1 byte — '"' is ASCII)

    // T101.1: walk by char (not byte) so non-ASCII content inside the
    // string is preserved code-point-faithfully instead of being torn
    // mid-rune. The previous byte-based loop also only handled the C-style
    // escapes `\n` `\t` `\r` `\\` `\"` and kept `\X` literal for any other
    // X — but the interpreter (`eval.rs::parse_string_literal_prefix`)
    // implements TLA+ escapes more bluntly: every `\X` collapses to just
    // `X`, regardless of whether X is `n`/`t`/etc. (so e.g. `"\\n"` is a
    // two-char string `\n` and `"\n"` is the one-char string `n`). The
    // fuzz pass surfaced four divergence cases on `"...\G..."`-shaped
    // inputs where the compiler kept the leading `\` and the interpreter
    // dropped it. Mirror the interpreter exactly: `\X` → `X` for all X.
    let mut chars = expr[1..].char_indices().peekable();
    while let Some((rel_idx, ch)) = chars.next() {
        let abs_after = 1 + rel_idx + ch.len_utf8();
        i = abs_after;

        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            continue;
        }

        if ch == '"' {
            // Found closing quote — i already points one past it.
            return Some((out, &expr[i..]));
        }

        out.push(ch);
    }

    // Unterminated string
    None
}

/// Compile a TLA+ expression string to a CompiledExpr
/// Remove the common leading whitespace shared by every non-empty line,
/// preserving relative indentation between lines. For a single-line input this
/// is equivalent to `trim_start`. Unlike `trim()`, it never de-indents one line
/// relative to the others — which is exactly what the indentation-sensitive
/// boolean splitter relies on to distinguish a quantifier/junction head from
/// its deeper body.
///
/// Leading and trailing BLANK lines are dropped so the first real line's prefix
/// checks behave like `trim()`'s: after a binary split (e.g. `=>`) the right
/// operand often begins with a newline (`"\n    \A x : ..."`), and callers test
/// `starts_with("\\A ")` etc. on the (re-`compile_expr`'d) result. If a leading
/// blank line survived, those checks would fail and a quantifier/LET head would
/// silently not be recognised — mis-evaluating the expression (observed as a
/// false `SegmentsRecoverable` violation on QueueSegmentSync). Relative indent
/// among the *non-blank* lines is still preserved.
fn dedent_common(expr: &str) -> String {
    let lines: Vec<&str> = expr.lines().collect();
    let Some(start) = lines.iter().position(|l| !l.trim().is_empty()) else {
        return String::new();
    };
    let end = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .unwrap_or(start);
    let body = &lines[start..=end];
    let min_indent = body
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    body.iter()
        .map(|l| {
            let dedented = if l.len() >= min_indent {
                &l[min_indent..]
            } else {
                l.trim_start()
            };
            // Trim trailing whitespace per line so a single-operand result like
            // `"0 "` (e.g. the low bound split out of `0 .. N-1`) re-compiles to
            // `Int(0)`, not `Unparsed("0 ")`. The pre-rewrite `dedent_common`
            // did this via a leading `trim_end()`.
            dedented.trim_end()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn compile_expr(expr: &str) -> CompiledExpr {
    // Re-base the expression while PRESERVING relative indentation: strip the
    // common leading whitespace shared by all non-empty lines instead of
    // `trim()`-ing line 1 to column 0. A bare `trim()` de-indents the first
    // line independently, destroying the column relationship between a
    // junction/quantifier head and its deeper body (the head lands at column 0
    // while its body keeps its original indent) — making an indented body
    // indistinguishable from a sibling junction item. `dedent_common` keeps
    // heads and bodies at consistent relative columns so the indented-boolean
    // splitter can tell siblings from nested bodies.
    let dedented = dedent_common(expr);
    // If the first line is over-indented relative to the rest, strip its leading
    // whitespace. This happens for a quantifier/LET head whose bulleted body
    // aligns to the ENCLOSING bullet, so the body sits LESS indented than the
    // head (e.g. a deeply nested `=> /\ \A n : /\ f /\ g`). dedent_common re-bases
    // by the global min — the body — leaving the head with a leading space, which
    // would break the prefix checks below (starts_with quantifier/LET/operator)
    // and let a trailing subscript `[x]` be mis-read as a function application
    // over the whole quantifier expression (observed: MCCheckpointCoordination
    // SafetyInvariant compiled `\A n : ... ~g[n]` to `FuncApply(Unparsed(...), n)`).
    let dedented = if dedented.starts_with([' ', '\t']) {
        match dedented.split_once('\n') {
            Some((first, rest)) => format!("{}\n{}", first.trim_start(), rest),
            None => dedented.trim_start().to_string(),
        }
    } else {
        dedented
    };
    let expr = dedented.as_str();

    if expr.is_empty() {
        return CompiledExpr::Unparsed(expr.to_string());
    }

    // Strip outer parentheses
    let expr = strip_outer_parens(expr);

    // Strip PlusCal label prefix: "label:: expr" or "label(x):: expr"
    // Labels are documentation-only in TLC and don't affect evaluation.
    let expr = strip_label_prefix(expr);

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

    // Infix operators used as higher-order values: +, -, *, \div, %
    // In TLA+, these can be passed as operator arguments, e.g.
    // FoldFunctionOnSet(+, 0, f, S). We compile them to Lambdas.
    match expr {
        "+" => {
            return CompiledExpr::Lambda {
                params: vec!["__a".to_string(), "__b".to_string()],
                body: Box::new(CompiledExpr::Add(
                    Box::new(CompiledExpr::Var("__a".to_string())),
                    Box::new(CompiledExpr::Var("__b".to_string())),
                )),
                body_text: "__a + __b".to_string(),
            };
        }
        "-" => {
            return CompiledExpr::Lambda {
                params: vec!["__a".to_string(), "__b".to_string()],
                body: Box::new(CompiledExpr::Sub(
                    Box::new(CompiledExpr::Var("__a".to_string())),
                    Box::new(CompiledExpr::Var("__b".to_string())),
                )),
                body_text: "__a - __b".to_string(),
            };
        }
        "*" => {
            return CompiledExpr::Lambda {
                params: vec!["__a".to_string(), "__b".to_string()],
                body: Box::new(CompiledExpr::Mul(
                    Box::new(CompiledExpr::Var("__a".to_string())),
                    Box::new(CompiledExpr::Var("__b".to_string())),
                )),
                body_text: "__a * __b".to_string(),
            };
        }
        _ => {}
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

    // T101.1 fix: bare `UNCHANGED` (with no operand) is an always-enabled
    // stuttering step in TLA+. The interpreter returns Bool(true). The
    // compiler used to fall through to `is_identifier(...)` below and emit
    // `Var("UNCHANGED")`, which `eval_compiled` then resolved to
    // `ModelValue("UNCHANGED")` — diverging from the interpreter on every
    // `UNCHANGED`-only expression body. The keyword check below at line ~550
    // already handles the `UNCHANGED <vars>` form; this special-case covers
    // the bare-token case before the identifier path swallows it.
    if expr == "UNCHANGED" {
        return CompiledExpr::Bool(true);
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
    // LAMBDA binds loosely - everything after the colon is the body.
    // T202: detect LAMBDA via word-boundary match (matching the interpreter's
    // `starts_with_keyword`) so inputs like "LAMBDA\t..." or "LAMBDA\n..."
    // take the LAMBDA branch here too — otherwise the literal " " check
    // missed those cases and the compiler fell through to the comparison
    // parser, returning `Bool(...)` while the interpreter returned `Lambda`.
    // If the LAMBDA prefix is recognised but `try_parse_lambda` rejects the
    // shape (e.g. non-identifier params), return `Unparsed(expr)` so the
    // interpreter handles it and the two paths agree.
    if starts_with_lambda_keyword(expr) {
        if let Some(lambda) = try_parse_lambda(expr) {
            return lambda;
        }
        return CompiledExpr::Unparsed(expr.to_string());
    }

    // CASE expression (must check before logical operators)
    if expr.starts_with("CASE ") || expr.starts_with("CASE\n") {
        if let Some(case_expr) = try_compile_case(expr) {
            return case_expr;
        }
    }

    // UNCHANGED in expression/guard context.
    // When UNCHANGED appears inside a disjunction evaluated as a guard (e.g.,
    // \/ Action1(self) \/ UNCHANGED vars), the action IR layer handles the
    // actual variable-preservation semantics. In expression context, UNCHANGED
    // represents an always-enabled stuttering step, so we compile it as TRUE.
    //
    // T101.1: the `expr == "UNCHANGED"` bare-token case is handled earlier
    // (above the identifier path); only the `UNCHANGED <vars>` form is
    // matched here. Even so, deferring to AFTER the binary / comparison
    // split below would let an outer `=` in `UNCHANGED x = y` win — but
    // that's exactly what the interpreter does, so we MUST defer too.
    // Move the prefix check to the bottom of the precedence cascade.

    // Logical operators (in precedence order)

    // Negation of a *bulleted* junction: `~ /\ A /\ B` / `~ \/ A \/ B`.
    //
    // `~` binds tighter than `/\`/`\/`, so `~A /\ B` = `(~A) /\ B` and is
    // correctly handled by the And/Or splits below (first conjunct `~A` is a
    // well-formed operand). The ONE shape those splits get wrong is a `~`
    // prefixing a bulleted list: `split_top_level("~ /\ A /\ B", "/\\")`
    // slices off a bare `~` as the first "conjunct", which then compiles to
    // `Not(compile_expr(""))` and errors with "empty expression" at eval time.
    // We must intercept it here — BEFORE the And/Or splits — and route the
    // whole junction list to `Not`, matching TLA+ semantics `~(A /\ B)`.
    // (TCommit's `TCConsistent == \A rm1,rm2 : ~ /\ ... /\ ...` compiled to a
    // broken `And(["", ...])` and false-violated every state, including the
    // initial one. The interpreted evaluator's `dispatch::classify_op` has the
    // mirror-image guard for the same reason.)
    if let Some(after_tilde) = expr.strip_prefix('~') {
        let t = after_tilde.trim_start();
        if t.starts_with("/\\") || t.starts_with("\\/") {
            return CompiledExpr::Not(Box::new(compile_expr(t)));
        }
    }

    // INDENTATION-based boolean splitting comes BEFORE infix `=>`/`<=>`.
    //
    // For a multiline TLA+ bulleted junction list, the alignment of the `/\`
    // (or `\/`) bullets is the top-level structure, and each bullet is itself a
    // full expression that may contain `=>`/`<=>` — exactly TLA+'s bulleted-list
    // semantics. So:
    //
    //   /\ a
    //   /\ b => c        parses as   a /\ (b => c)
    //
    // i.e. the bullet split outranks the infix `=>`. `split_indented_top_level_
    // boolean` returns None when line 1 is NOT itself a bullet at the base
    // indent (e.g. `L = N =>` followed by an indented `/\` block), so a genuine
    // top-level `=>` whose consequent is a junction list still defers to the
    // `=>` split below. For single-line input it also returns None (no newline
    // to align on), so ordinary infix precedence applies there too.
    //
    // The body-delimiter-based `formula::split_top_level` cannot distinguish
    // "\\/ inside an indented sub-block" from "\\/ at the top level"; for
    //
    //   /\ \A a \in Q : \E m \in Q1b : m.acc = a
    //   /\ \/ Q1bv = {}
    //      \/ \E m \in Q1bv : ...
    //
    // it would mis-split on the indented `\\/` and produce a flat
    // `Or([Forall, Eq, Exists])` instead of `And([Forall, Or([Eq, Exists])])`.
    // With `Q1bv = {}` true at the initial state, the mis-compiled `Or`
    // evaluated TRUE — silently passing universally-quantified existential
    // guards (Paxos `Phase2a`). (T1.4)
    if let Some(parts) = split_indented_top_level_boolean(expr, "/\\") {
        if parts.len() == 1 {
            // Single `/\ X` wrapper — `X` (re-dedented) is the real expression.
            return compile_expr(&parts[0]);
        }
        return CompiledExpr::And(parts.into_iter().map(|s| compile_expr(&s)).collect());
    }
    if let Some(parts) = split_indented_top_level_boolean(expr, "\\/") {
        if parts.len() == 1 {
            return compile_expr(&parts[0]);
        }
        return CompiledExpr::Or(parts.into_iter().map(|s| compile_expr(&s)).collect());
    }

    // Equivalence (biconditional): <=>
    // MUST be split BEFORE `=>` because `split_binary_op(expr, "=>")` would
    // otherwise match the `=>` *inside* `<=>` (no `<` look-back protection).
    // T1.6 (FingerprintStoreResize) regression: see RELEASE_1.0.0_LOG.md.
    if let Some((left, right)) = split_binary_op(expr, "<=>") {
        return CompiledExpr::Iff(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

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

    // TLC module: Function override operator @@ (precedence 6-6)
    // f @@ g merges two functions, with g taking precedence on overlapping keys
    // Lower precedence than :> so parse it first.
    if let Some((left, right)) = split_binary_op(expr, "@@") {
        return CompiledExpr::FuncOverride(
            Box::new(compile_expr(left)),
            Box::new(compile_expr(right)),
        );
    }

    // TLC module: Function pair constructor :> (precedence 7-7)
    // a :> b creates a function mapping a to b (single mapping {a -> b})
    if let Some((left, right)) = split_binary_op(expr, ":>") {
        return CompiledExpr::FuncPair(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Set operations (TLA+ precedence 8-8 for \union, \cup, \intersect, \cap,
    // \\ set minus). MUST be split before `..` (9-9) and `+`/`-` (10-10) and
    // `*`/`\div`/`%` (13-13) so that `S \union 1..3` parses as
    // `S \union (1..3)` rather than `(S \union 1)..3`. T2.3 fix — see
    // `RELEASE_1.0.0_LOG.md`. Note: `\o`/`\circ` (concat, 13-13) and
    // `\X`/`\times` (cartesian, 10-13) are kept in this block only to avoid
    // `\` set-minus matching `\o`/`\X`/`\times`; `\` itself is a set op at
    // 8-8 so its placement is correct.
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
    // Handle \o (concatenation) BEFORE \ (set minus) since \ would match \o.
    if let Some((left, right)) = split_binary_op(expr, "\\o") {
        return CompiledExpr::Concat(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "\\circ") {
        return CompiledExpr::Concat(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    // Handle cartesian product: \X and \times are synonyms.
    // Must be before \ (set minus) since \ would match \X or \times.
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

    // Set/sequence range: a..b (TLA+ precedence 9-9). MUST be split BEFORE
    // arithmetic (`+`/`-` at 10) so `0 .. N-1` parses as `0 .. (N-1)`
    // rather than `(0 .. N) - 1`. AND MUST be split AFTER set ops (above)
    // so `S \union 1..3` parses as `S \union (1..3)` rather than
    // `(S \union 1)..3`. See `test_dotdot_has_lower_precedence_than_arithmetic`
    // and `test_dotdot_binds_tighter_than_set_ops` for the regression gates.
    if let Some((left, right)) = split_binary_op(expr, "..") {
        return CompiledExpr::SetRange(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Arithmetic operators (TLA+ precedence: + is 10-10, - is 11-11, * is 13-13)
    // T101.1: arithmetic is left-associative, so we use *_last to take the
    // RIGHTMOST top-level operator. This makes `a + b + c` parse as
    // `(a + b) + c`, matching the interpreter (`split_top_level_additive`).
    // T206: forward chained arithmetic to the interpreter. Compiler and
    // interpreter use different splitter cascades; they diverge on
    // associativity for chains of 2+ binary arithmetic ops. The interpreter
    // is the reference. Emit Unparsed here — AFTER all structural keyword
    // handlers (IF/LET/LAMBDA/CASE/quantifiers) so those still parse into
    // structured AST — and BEFORE the binary arithmetic split cascade so
    // chained ops never reach the divergent splitters.
    if has_chained_top_level_arithmetic(expr) {
        return CompiledExpr::Unparsed(expr.to_string());
    }

    if let Some((left, right)) = split_binary_op_last(expr, "+") {
        return CompiledExpr::Add(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    // Binary subtraction `-` is tricky because the same character also
    // serves as unary minus. `split_binary_op` returns the FIRST top-level
    // `-` it encounters, but for inputs like `-1 - (r).a` the FIRST `-` is
    // the leading unary one (left = ""), and the actual binary `-` sits
    // later in the string. T2.4 fix: scan ALL top-level `-` positions and
    // pick the first one where the preceding text ends in a value-producing
    // token. The `left_ends_with_value` guard (T2.1+T2.2) still rules out
    // operator-adjacent `-` like `x * -3`.
    if let Some((left, right)) = find_binary_minus_split(expr) {
        return CompiledExpr::Sub(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    // T101.1: multiplicative ops are left-associative; use *_last so
    // `a * b * c` parses as `(a * b) * c` (matches `split_top_level_multiplicative`).
    if let Some((left, right)) = split_binary_op_last(expr, "*") {
        return CompiledExpr::Mul(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op_last(expr, "\\div") {
        return CompiledExpr::Div(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op_last(expr, "%") {
        return CompiledExpr::Mod(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }
    if let Some((left, right)) = split_binary_op(expr, "^^") {
        return CompiledExpr::OpCall {
            name: "^^".to_string(),
            args: vec![compile_expr(left), compile_expr(right)],
        };
    }
    if let Some((left, right)) = split_binary_op(expr, "^") {
        return CompiledExpr::Pow(Box::new(compile_expr(left)), Box::new(compile_expr(right)));
    }

    // Unary minus on a non-literal subexpression: `-(expr)`, `-Op(...)`,
    // `-x.field`, etc. The `-N` literal case is already handled near the
    // top (line ~456). T2.4 fix: previously `-((LET ... IN ...))` and
    // `-(r).a` fell through every binary-operator split (no binary `-`
    // found because the leading `-` is unary), then through structural
    // matchers, and ended up as `Unparsed`. Wrap as `Neg(rest)`. We only
    // do this when binary subtraction has already been ruled out (above)
    // so we don't mis-compile `1 - 2` as `Neg(2)`.
    if let Some(rest) = expr.strip_prefix('-') {
        let rest = rest.trim_start();
        if !rest.is_empty() {
            return CompiledExpr::Neg(Box::new(compile_expr(rest)));
        }
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
    // For known built-in operators, produce dedicated variants instead of generic OpCall.
    if let Some((name, args)) = parse_op_call(expr) {
        match name {
            "Cardinality" if args.len() == 1 => {
                return CompiledExpr::Cardinality(Box::new(compile_expr(&args[0])));
            }
            "Head" if args.len() == 1 => {
                return CompiledExpr::Head(Box::new(compile_expr(&args[0])));
            }
            "Tail" if args.len() == 1 => {
                return CompiledExpr::Tail(Box::new(compile_expr(&args[0])));
            }
            "Len" if args.len() == 1 => {
                return CompiledExpr::Len(Box::new(compile_expr(&args[0])));
            }
            "Append" if args.len() == 2 => {
                return CompiledExpr::Append(
                    Box::new(compile_expr(&args[0])),
                    Box::new(compile_expr(&args[1])),
                );
            }
            "SubSeq" if args.len() == 3 => {
                return CompiledExpr::SubSeq(
                    Box::new(compile_expr(&args[0])),
                    Box::new(compile_expr(&args[1])),
                    Box::new(compile_expr(&args[2])),
                );
            }
            _ => {
                return CompiledExpr::OpCall {
                    name: name.to_string(),
                    args: args.into_iter().map(|s| compile_expr(&s)).collect(),
                };
            }
        }
    }

    // T101.1: deferred `UNCHANGED <vars>` match. Sits at the bottom of the
    // precedence cascade so an outer comparison / boolean op binds first.
    // Mirrors `eval.rs` where `if starts_with_keyword(expr, "UNCHANGED")`
    // sits below `split_top_level_comparison`. Without this deferral, the
    // compiler matched on the prefix and returned `Bool(true)` for inputs
    // like `UNCHANGED x = y` where the interpreter correctly evaluated the
    // `=` comparison first.
    if expr.starts_with("UNCHANGED ") || expr.starts_with("UNCHANGED\n") {
        return CompiledExpr::Bool(true);
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

/// Strip a PlusCal label prefix from an expression.
///
/// PlusCal-generated TLA+ specs include labels like `P0:: expr` or
/// `lab(x):: expr`. Labels are documentation-only in TLC — they don't
/// affect evaluation. This function strips the label and `::` and
/// returns the remaining expression.
fn strip_label_prefix(expr: &str) -> &str {
    let s = expr.trim_start();
    // A label starts with an identifier character
    let first = match s.chars().next() {
        Some(c) if c.is_alphabetic() || c == '_' => c,
        _ => return expr,
    };
    // Find end of identifier — sum the byte length of the leading char and
    // each subsequent identifier-continuation char so id_end always lands on
    // a UTF-8 boundary (a non-ASCII first char is `len_utf8() > 1`).
    let id_end = first.len_utf8()
        + s[first.len_utf8()..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .map(|c| c.len_utf8())
            .sum::<usize>();
    let after_id = &s[id_end..];
    // Check for optional parameter list: (x) or (x, y)
    let after_params = if after_id.starts_with('(') {
        let mut depth = 0;
        let mut end = 0;
        for (i, c) in after_id.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            return expr;
        }
        &after_id[end..]
    } else {
        after_id
    };
    // Check for ::
    let trimmed = after_params.trim_start();
    if let Some(rest) = trimmed.strip_prefix("::") {
        let body = rest.trim_start();
        if !body.is_empty() {
            return body;
        }
    }
    expr
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
    // Delegate to the formula-level split_top_level which handles quantifier scoping
    // correctly. The compiled_expr version previously used indentation-based sibling
    // detection, but this broke when the expression was trimmed (losing the first
    // line's indent). The formula version uses body-delimiter detection instead,
    // which is more robust.
    crate::tla::formula::split_top_level(expr, delim)
}

/// Split a multi-line boolean expression on its **outermost** vertical
/// `delim` (`/\\` or `\\/`) operators, where "outermost" is determined by
/// indentation: only delimiter tokens that begin a line at the smallest
/// indent level are treated as splits.
///
/// Returns `None` when the expression doesn't span multiple lines, when
/// no line begins with `delim`, or when the apparent split would yield a
/// single clause (handing back to the caller's symbol-based fallback).
///
/// T1.4: this function exists to fix a soundness bug in the compiled
/// expression evaluator. Before adding this pass, multi-line shapes like
///
/// ```text
/// /\ \A a \in Q : \E m \in Q1b : ...
/// /\ \/ Q1bv = {}
///    \/ \E m \in Q1bv : ...
/// ```
///
/// were mis-split by `formula::split_top_level("\\/")`, which doesn't
/// know that the indented `\\/` lines are inside the second top-level
/// `/\\` conjunct. The result was a flat `Or` instead of an
/// `And([Forall, Or])` — silently passing universally-quantified
/// existential guards in Paxos-style specs.
fn split_indented_top_level_boolean(expr: &str, delim: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    // NOTE: we operate on `expr` directly and do NOT re-normalize indentation
    // here. `compile_expr` re-bases every (sub)expression with `dedent_common`
    // at entry, which preserves the relative columns of bullets and their
    // bodies. The previous `normalize_multiline_boolean_indentation` step
    // de-indented line 1 independently of the rest, collapsing an indented
    // body onto its head's column and making nested bodies look like siblings.
    let normalized = expr;

    // Pass 1: find the smallest indent at which a line begins with `delim`.
    let mut min_indent: Option<usize> = None;
    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with(delim) {
            let indent = line.len().saturating_sub(trimmed.len());
            min_indent = Some(min_indent.map_or(indent, |m| m.min(indent)));
        }
    }
    let base_indent = min_indent?;

    // Pass 2: collect the clauses, treating only `delim`-prefixed lines at
    // `base_indent` as split points. All other lines (including indented
    // `delim` lines, which are part of the current clause) are appended to
    // the current clause verbatim, preserving indentation so that nested
    // calls to compile_expr can see the same shape and split it again at
    // the inner indent.
    let mut clauses: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut saw_top_level = false;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let indent = line.len().saturating_sub(trimmed.len());
        if indent == base_indent && trimmed.starts_with(delim) {
            if saw_top_level {
                if !current.trim().is_empty() {
                    clauses.push(current.trim_end().to_string());
                }
                current.clear();
            } else {
                let prefix = current.trim();
                if !prefix.is_empty() {
                    // Content before the first delim that isn't a delim
                    // line means the expression isn't a clean indented
                    // boolean — bail out and let the caller handle it.
                    return None;
                }
                current.clear();
                saw_top_level = true;
            }
            // Column-preserving extraction: replace the leading `delim` with
            // an equal run of spaces so the bullet's body stays at its ORIGINAL
            // column. This keeps a deeper body (e.g. a quantifier's `/\` list)
            // distinguishable from a sibling bullet at the bullet column, after
            // the recursive `compile_expr` re-bases the clause with
            // `dedent_common`. (The old `trim_start().trim_start_matches(delim)`
            // pinned the body to column 0, destroying that distinction.)
            let after = trimmed.strip_prefix(delim).unwrap_or(trimmed);
            let lead_ws = after.len() - after.trim_start().len();
            let col = indent + delim.len() + lead_ws;
            for _ in 0..col {
                current.push(' ');
            }
            current.push_str(after.trim_start());
            continue;
        }
        if !saw_top_level {
            // Haven't yet seen a top-level delim — don't accept this as
            // an indented boolean split.
            return None;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim_end().to_string());
    }

    // A single top-level clause is still a meaningful result: it means the
    // expression is a `delim`-prefixed wrapper around one body (e.g.
    // `/\ \A n : ...` with the quantifier's own `/\` body indented deeper).
    // Returning it lets `compile_expr` strip the leading bullet and recurse on
    // the body, where re-dedenting realigns the inner bullets — instead of
    // falling through to the symbol-based `/\`/`\/` split, which cannot tell
    // the deeper body bullets from top-level siblings. `clauses` is empty only
    // when no base-indent `delim` line was ever consumed.
    if clauses.is_empty() {
        return None;
    }

    Some(clauses)
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

/// Helper for disambiguating a `-` token: returns true if the trimmed
/// left-hand side ends in a character that can plausibly produce a
/// value (so the `-` is a binary subtraction). Returns false if the
/// left ends in a binary operator or open delimiter (so the `-` is a
/// unary minus and we should not split here).
///
/// Used by the binary-`-` split site in `compile_expr`. Without this
/// guard, `(x * -3)` was sliced as `Sub(Mul(x, ""), 3)` because
/// `split_binary_op` happily reported `left = "x * "` and `right = "3"`.
fn left_ends_with_value(left: &str) -> bool {
    let trimmed = left.trim_end();
    let Some(last) = trimmed.chars().last() else {
        return false;
    };
    // Value-producing terminators: identifier chars, digits, closing
    // brackets/parens/braces, and the quote at the end of a string
    // literal. Note: `>>` (sequence end) ends in `>` so we accept `>`.
    matches!(last, ')' | ']' | '}' | '"' | '>') || last.is_alphanumeric() || last == '_'
}

/// Find a binary subtraction `-` split point at top level. Walks past
/// leading unary `-` positions (where `left` is empty or doesn't end in a
/// value-producing token) so that, e.g., `-1 - (r).a` correctly splits at
/// the SECOND `-`, not the first.
///
/// T2.4 fix: previously `compile_expr` called `split_binary_op(expr, "-")`
/// which returns the FIRST top-level `-`; if that first match was a unary
/// minus, the binary split was abandoned entirely (and the expression was
/// later misparsed as e.g. `RecordAccess(Unparsed("-1 - (r)"), "a")`).
/// Now we keep scanning until we find a `-` whose preceding text actually
/// ends in a value (digit, identifier, `)`, `]`, `>>`, `}`, or `"`).
///
/// T101.1 fix: subtraction is **left-associative** in TLA+, so for `a-b-c`
/// we want the LAST top-level binary `-`, not the first. We scan all
/// candidates and return the rightmost one whose left side is a value
/// expression. This makes `0 - 6 - 555555555` parse as
/// `(0 - 6) - 555555555 = -555555561`, matching the interpreter.
fn find_binary_minus_split(expr: &str) -> Option<(&str, &str)> {
    let mut search_from = 0usize;
    let mut last: Option<(usize, usize)> = None;
    loop {
        let candidate = &expr[search_from..];
        let Some((left_rel, right)) = split_binary_op(candidate, "-") else {
            break;
        };
        let left_abs_end = search_from + left_rel.len();
        let right_abs_start = expr.len() - right.len();
        let left = &expr[..left_abs_end];
        if !left.is_empty() && left_ends_with_value(left) {
            last = Some((left_abs_end, right_abs_start));
        }
        // Always advance past this `-` and keep looking for a later one.
        // We use `right_abs_start` (= position after the `-`) rather than
        // `left_abs_end + 1` so we step past the operator regardless of
        // whether this candidate was unary or binary.
        if right_abs_start <= search_from {
            // No forward progress — bail out to avoid an infinite loop.
            break;
        }
        search_from = right_abs_start;
        if search_from >= expr.len() {
            break;
        }
    }
    last.map(|(left_end, right_start)| (&expr[..left_end], &expr[right_start..]))
}

fn split_binary_op<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    split_binary_op_with(expr, op, false)
}

/// Like `split_binary_op`, but returns the LAST top-level match instead of
/// the first. Use for left-associative operators (additive `+`/`-`,
/// multiplicative `*`/`\div`/`%`, range `..`) so that
/// `a OP b OP c` parses as `(a OP b) OP c` instead of `a OP (b OP c)`.
///
/// T101.1 fix: `0 - 6 - 555555555` was returning `Sub(0, Sub(6, 555555555))`
/// (= 555555549) under first-match splitting, while the interpreter (which
/// scans left-to-right and keeps the LAST top-level operator position, see
/// `eval.rs::split_top_level_additive`) correctly returned
/// `Sub(Sub(0, 6), 555555555)` (= -555555561). This was a soundness bug on
/// well-formed input, not just on garbage; any 3+-term arithmetic chain
/// without explicit parens was silently mis-evaluated by the compiler.
fn split_binary_op_last<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    split_binary_op_with(expr, op, true)
}

fn split_binary_op_with<'a>(expr: &'a str, op: &str, prefer_last: bool) -> Option<(&'a str, &'a str)> {
    let mut bracket_depth = 0i32;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize; // Track IF...THEN nesting (protects conditions from split)
    let mut case_depth = 0usize; // Track CASE...[] nesting (arms contain = and ->)
    let mut quantifier_depth = 0usize; // Track \A/\E quantifier body nesting
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
    let op_chars: Vec<char> = op.chars().collect();
    let mut i = 0;
    let mut last_match: Option<(usize, usize)> = None;

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
            // Track quantifier bodies: \A x \in S : body or \E x \in S : body
            // Operators inside the body should not be split at the top level.
            if c == '\\' && i + 2 < chars.len() {
                let next = chars[i + 1];
                let after = chars[i + 2];
                if (next == 'A' || next == 'E') && (after.is_whitespace() || after == '(') {
                    // Found quantifier start - scan forward for the colon
                    let mut j = i + 2;
                    let mut inner_depth = 0i32;
                    while j < chars.len() {
                        match chars[j] {
                            '(' | '[' | '{' => inner_depth += 1,
                            ')' | ']' | '}' => inner_depth -= 1,
                            ':' if inner_depth == 0 => {
                                quantifier_depth += 1;
                                i = j + 1;
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
        }

        // Only split when at top level of all constructs
        let at_top = bracket_depth == 0
            && let_depth == 0
            && if_depth == 0
            && case_depth == 0
            && quantifier_depth == 0;
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
                if prefer_last {
                    last_match = Some((left_byte_end, right_byte_start));
                    // Skip past the matched operator and keep scanning for a
                    // later occurrence. We deliberately don't advance the
                    // bracket-depth tracking inside the operator chars
                    // (single-char ops in the LIST ABOVE — `+`, `-`, `*`,
                    // `%`, `..` — are not parens; multi-char arithmetic
                    // ops like `\\div` are word-bounded and the inner chars
                    // are alphabetic, harmless to skip).
                    #[cfg(not(feature = "verus"))]
                    let step = op_chars.len().max(1);
                    #[cfg(feature = "verus")]
                    let step = crate::storage::verus_smoke::max_usize(op_chars.len(), 1);
                    i += step;
                    continue;
                }
                return Some((&expr[..left_byte_end], &expr[right_byte_start..]));
            }
        }
        i += 1;
    }
    last_match.map(|(left_end, right_start)| (&expr[..left_end], &expr[right_start..]))
}

fn split_comparison(expr: &str) -> Option<(&str, &str, &str)> {
    // T101.1: scan left-to-right and split at the FIRST top-level comparison
    // operator we encounter, regardless of which one. The previous
    // implementation iterated patterns in priority order and ran a
    // whole-expression scan per pattern, which silently preferred the later
    // `=` in `CAE#S = i` over the earlier `#`. The interpreter
    // (`eval.rs::split_top_level_comparison`) walks the string once and
    // returns at the first relop hit, so the compiler must match.
    //
    // Order within `OPS` only matters when two operators share a prefix at
    // the same position (e.g. `<=` vs `<`); we list longer-first so the
    // longest match wins. Once a position is decided, scanning stops.
    const OPS: &[&str] = &[
        "\\subseteq",
        "\\notin",
        "\\in",
        "\\geq",
        "\\leq",
        ">=",
        "=<",
        "<=",
        "/=",
        "#",
        "=",
        ">",
        "<",
    ];
    split_first_top_level_op(expr, OPS).map(|(left, op, right)| (left.trim(), op, right.trim()))
}

/// Walk `expr` left-to-right at top bracket-depth and return the FIRST
/// top-level position at which any of `ops` matches (longest-match-first
/// at that position). Mirrors the interpreter's
/// `split_top_level_comparison` so chained relops like `CAE # S = i` split
/// at the leftmost relop (`#`) instead of jumping ahead to a later `=`.
///
/// The bracket / LET / IF / CASE / quantifier scoping logic is the same as
/// `split_binary_op`'s — kept inlined here rather than refactored because
/// `split_binary_op` returns on the first match of a single operator and
/// we'd otherwise have to call it once per op (each scan is O(n), so the
/// composite was O(n*|ops|); this is O(n*max(|op|))).
fn split_first_top_level_op<'a>(
    expr: &'a str,
    ops: &'static [&'static str],
) -> Option<(&'a str, &'static str, &'a str)> {
    let mut bracket_depth = 0i32;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize;
    let mut case_depth = 0usize;
    let mut quantifier_depth = 0usize;
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
            in_string = !in_string;
            i += 1;
            continue;
        }
        if in_string {
            i += 1;
            continue;
        }

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
            if matches_keyword_at(&chars, i, "CASE") {
                case_depth += 1;
                i += 4;
                continue;
            }
            if c == '\\' && i + 2 < chars.len() {
                let next = chars[i + 1];
                let after = chars[i + 2];
                if (next == 'A' || next == 'E') && (after.is_whitespace() || after == '(') {
                    let mut j = i + 2;
                    let mut inner_depth = 0i32;
                    while j < chars.len() {
                        match chars[j] {
                            '(' | '[' | '{' => inner_depth += 1,
                            ')' | ']' | '}' => inner_depth -= 1,
                            ':' if inner_depth == 0 => {
                                quantifier_depth += 1;
                                i = j + 1;
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
        }

        let at_top = bracket_depth == 0
            && let_depth == 0
            && if_depth == 0
            && case_depth == 0
            && quantifier_depth == 0;

        if at_top {
            for &op in ops {
                let op_chars: Vec<char> = op.chars().collect();
                if i + op_chars.len() > chars.len() {
                    continue;
                }
                if !op_chars.iter().enumerate().all(|(j, &oc)| chars[i + j] == oc) {
                    continue;
                }
                // Word-boundary check for backslash-prefixed names like \in.
                if op.chars().last().map_or(false, |c| c.is_alphabetic()) {
                    let next_idx = i + op_chars.len();
                    if next_idx < chars.len() && chars[next_idx].is_alphabetic() {
                        continue;
                    }
                }
                // Disambiguate `=` from longer ops with `=` somewhere — never
                // split mid-`=>`/`<=>`/`>=`/`<=`/`/=`/`=<`. The longer ops
                // are listed earlier in OPS so they get matched first; this
                // guard catches the case where the longer op is _not_ in the
                // ops list but its prefix coincides with a shorter one.
                if op == "=" {
                    let prev = (i > 0).then(|| chars[i - 1]);
                    let next = chars.get(i + 1).copied();
                    if prev == Some('=')
                        || prev == Some('<')
                        || prev == Some('>')
                        || prev == Some('/')
                    {
                        continue;
                    }
                    if next == Some('>') || next == Some('=') {
                        continue;
                    }
                }
                if op == "<" {
                    let next = chars.get(i + 1).copied();
                    if next == Some('=') || next == Some('<') {
                        continue;
                    }
                }
                if op == ">" {
                    let prev = (i > 0).then(|| chars[i - 1]);
                    let next = chars.get(i + 1).copied();
                    if next == Some('=') || next == Some('>') {
                        continue;
                    }
                    if prev == Some(':') {
                        continue;
                    }
                }

                let left_byte_end: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                let right_byte_start: usize = chars[..i + op_chars.len()]
                    .iter()
                    .map(|c| c.len_utf8())
                    .sum();
                let left = &expr[..left_byte_end];
                let right = &expr[right_byte_start..];
                if left.is_empty() || right.is_empty() {
                    continue;
                }
                return Some((left, op, right));
            }
        }

        i += 1;
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

    // Find the last [ at depth 0. The trailing `]` we want to strip MUST be
    // ASCII for this caller's grammar; if the expression ends in something
    // else (a non-ASCII rune the fuzzer mutated in), bail out instead of
    // panicking on a sub-byte slice.
    let bytes = expr.as_bytes();
    let trim_end = match bytes.last() {
        Some(b']') => bytes.len() - 1,
        _ => return None,
    };

    let mut depth = 0;
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
                let args_str = &expr[i + 1..trim_end];
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
            // MULTI-BINDER MAP FORM: `{ expr : x \in S, y \in T, ... }`.
            // The compiled `SetComprehension` node models only ONE binder.
            // `parse_in_binding(right)` above matched the FIRST `\in` greedily
            // and folded the remaining binders into the domain expression
            // (`domain = "S, y \in T"`), which would then compile to a broken
            // `Membership(compile("S, y"), T)` and fail at eval with
            // "unexpected trailing tokens in expr: S, y" — reported as a false
            // invariant violation on the initial state (btree/kvstore `TypeOK`,
            // `args \in {<<k,v>>: k \in Keys, v \in Vals}`). The interpreter's
            // `eval_set_expression` handles arbitrary multi-binder map
            // comprehensions correctly (via `parse_binders` +
            // `collect_binder_map_set`), so defer the whole set expression to it.
            if parse_multiple_bindings(right).map(|b| b.len()).unwrap_or(1) > 1 {
                return Some(CompiledExpr::Unparsed(format!("{{{inner}}}")));
            }
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
            // Same guard for the filter form `{x \in S : P}`: if the binder
            // side actually carries multiple bindings (`{<<x,y>> \in S : P}`
            // is a tuple binder, but `{x \in S, y \in T : P}` is multi-binder),
            // defer to the interpreter rather than dropping binders.
            if parse_multiple_bindings(left).map(|b| b.len()).unwrap_or(1) > 1 {
                return Some(CompiledExpr::Unparsed(format!("{{{inner}}}")));
            }
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
            #[cfg(not(feature = "verus"))]
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            #[cfg(feature = "verus")]
            b')' | b']' | b'}' => depth = crate::storage::verus_smoke::saturating_dec_i32_to_zero(depth),
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
///
/// **T2.1 + T2.2 fix:** the EXCEPT keyword must be matched at top-level
/// bracket depth. Previously this used `inner.find(" EXCEPT ")`, which
/// also matched EXCEPT keywords nested inside `[... EXCEPT ...]`
/// sub-expressions. That caused two failure modes:
///   - `[[r EXCEPT !.a = 0] EXCEPT !.b = 0]` was sliced as
///     `base = "[r"` + EXCEPT + ... → "missing closing ']'".
///   - `[a |-> 0, b |-> ([r EXCEPT !.a = 0]).a]` matched the inner EXCEPT
///     (inside the record-literal field value), producing a malformed
///     base like `"a |-> 0, b |-> ([r"` → "unexpected trailing tokens".
/// Both surface only on the compiled fast path; the interpreter handles
/// both shapes correctly. See `RELEASE_1.0.0_LOG.md` § T2.1 + T2.2.
fn try_parse_except(inner: &str) -> Option<CompiledExpr> {
    // Find "EXCEPT" keyword at top-level bracket/paren/brace depth, with a
    // mandatory leading space and a trailing space or newline. Anything
    // inside `(...)`, `[...]`, `{...}`, or `<<...>>` is ignored.
    let (except_pos, keyword_len) = find_top_level_except(inner)?;

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

/// Locate the first top-level (bracket-depth-0) occurrence of the EXCEPT
/// keyword surrounded by ` EXCEPT ` or ` EXCEPT\n`. Returns the byte
/// offset of the leading space and the keyword-with-delimiters length
/// (always 8: 1 lead space + "EXCEPT" + 1 trail char).
///
/// Tracks `(`, `[`, `{`, `<<` depth so that EXCEPT keywords inside
/// nested expressions (record literals, sub-EXCEPTs, function
/// applications, etc.) are not mistaken for the outer separator.
fn find_top_level_except(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        // Track `<<` / `>>` as bracket pairs (TLA+ tuple syntax).
        if b == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'<' {
            depth += 1;
            i += 2;
            continue;
        }
        if b == b'>' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            #[cfg(not(feature = "verus"))]
            { depth = (depth - 1).max(0); }
            #[cfg(feature = "verus")]
            { depth = crate::storage::verus_smoke::saturating_dec_i32_to_zero(depth); }
            i += 2;
            continue;
        }
        match b {
            b'(' | b'[' | b'{' => {
                depth += 1;
                i += 1;
                continue;
            }
            b')' | b']' | b'}' => {
                #[cfg(not(feature = "verus"))]
                { depth = (depth - 1).max(0); }
                #[cfg(feature = "verus")]
                { depth = crate::storage::verus_smoke::saturating_dec_i32_to_zero(depth); }
                i += 1;
                continue;
            }
            _ => {}
        }
        if depth == 0 && b == b' ' && i + 8 <= bytes.len() && &bytes[i + 1..i + 7] == b"EXCEPT" {
            let trail = bytes[i + 7];
            if trail == b' ' || trail == b'\n' {
                return Some((i, 8));
            }
        }
        i += 1;
    }
    None
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

fn try_compile_case(expr: &str) -> Option<CompiledExpr> {
    // CASE cond1 -> val1 [] cond2 -> val2 [] OTHER -> default
    let rest = expr.strip_prefix("CASE")?;
    // Must be followed by whitespace
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    // Split on [] at top level. We can't use split_top_level("[]") because
    // that function tracks [ and ] as bracket depth changers. Instead, do a
    // custom scan that looks for the two-char sequence [] at bracket depth 0.
    let raw_arms = split_case_arms(rest);
    let mut arms = Vec::new();
    let mut other = None;

    for arm_str in &raw_arms {
        let arm = arm_str.trim();
        if arm.is_empty() {
            continue;
        }

        // Find the first top-level "->" to split condition from value.
        let arrow_idx = find_top_level_arrow(arm)?;
        let cond = arm[..arrow_idx].trim();
        let value = arm[arrow_idx + 2..].trim();

        if cond == "OTHER" {
            other = Some(Box::new(compile_expr(value)));
        } else {
            arms.push((compile_expr(cond), compile_expr(value)));
        }
    }

    if arms.is_empty() && other.is_none() {
        return None;
    }

    Some(CompiledExpr::Case { arms, other })
}

/// Split CASE expression body on top-level `[]` separators.
/// Unlike `split_top_level`, this treats `[]` as an atomic two-character
/// delimiter and does not adjust bracket depth for it.
fn split_case_arms(expr: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32; // combined bracket depth for ( ) { } [ ] << >>
    let mut in_string = false;
    let chars: Vec<char> = expr.chars().collect();
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

        // Handle << and >>
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

        // Check for [] delimiter at top level BEFORE adjusting depth
        if depth == 0 && c == '[' && i + 1 < chars.len() && chars[i + 1] == ']' {
            // This is the [] arm separator
            if !current.trim().is_empty() {
                parts.push(current.trim().to_string());
            }
            current = String::new();
            i += 2;
            continue;
        }

        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }

        current.push(c);
        i += 1;
    }

    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
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

        // Defensive: malformed input may put two `==` markers right next to
        // each other (e.g. `Op==Op==`), making `body_end < eq_pos + 2`. Skip
        // such definitions instead of panicking.
        let value_start = *eq_pos + 2;
        if value_start > body_end {
            continue;
        }
        let name = trim_let_edge_comments(&defs_str[name_start..*eq_pos]);
        let value = trim_let_edge_comments(&defs_str[value_start..body_end]);

        // Skip comments (lines starting with \*)
        let name = name.lines().last().unwrap_or("").trim();

        // A LET binding whose value is `INSTANCE M [WITH ...]` defines a
        // module instance, not a value. The compiled `Let` evaluator binds
        // each value eagerly, but a bare INSTANCE has no value — it errors,
        // which would silently swallow the whole LET. Hand the entire LET to
        // the interpreted evaluator (via `Unparsed`), which registers the
        // inline instance so `alias!Op(...)` resolves in the body. See
        // `eval::control::eval_let_expression` / `eval::instance`.
        if value.trim_start().starts_with("INSTANCE ")
            || value.trim_start().starts_with("INSTANCE\n")
        {
            return Some(CompiledExpr::Unparsed(expr.to_string()));
        }

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

/// True when `expr` begins with `LAMBDA` followed by a non-word character
/// (or end of string). Mirrors `eval::starts_with_keyword(expr, "LAMBDA")`
/// so the compiler and interpreter agree on what counts as a LAMBDA prefix.
/// T202: pre-fix the compiler used `expr.starts_with("LAMBDA ")` which
/// missed `LAMBDA\t...`, `LAMBDA\n...` and `LAMBDA(...)`, so those inputs
/// fell through to the comparison parser and produced `Bool(...)` while
/// the interpreter produced `Lambda(...)`.
fn starts_with_lambda_keyword(expr: &str) -> bool {
    let Some(rest) = expr.strip_prefix("LAMBDA") else {
        return false;
    };
    rest.chars()
        .next()
        .map(|c| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(true)
}

/// Strip a `LAMBDA` keyword prefix (with word-boundary check) and return
/// the remainder. Counterpart to `starts_with_lambda_keyword`.
fn strip_lambda_keyword(expr: &str) -> Option<&str> {
    let rest = expr.strip_prefix("LAMBDA")?;
    match rest.chars().next() {
        Some(c) if c.is_alphanumeric() || c == '_' => None,
        _ => Some(rest),
    }
}

fn try_parse_lambda(expr: &str) -> Option<CompiledExpr> {
    // LAMBDA x: body  OR  LAMBDA x, y: body
    // T202: accept any non-word boundary after `LAMBDA` (space, tab, newline,
    // `(` etc.) — matches the interpreter's `starts_with_keyword` semantics.
    let rest = strip_lambda_keyword(expr)?.trim();

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
    // Body uses `.trim()` (NOT `dedent_common`). The body is re-`compile_expr`'d,
    // which re-bases it with `dedent_common` anyway, so within-body conjunction
    // bullets still split correctly via the entry dedent + symbol fallback. We
    // deliberately do NOT realign the body here: re-indenting a quantifier body
    // before it reaches the action-IR layer flips the IR's guard-vs-action-
    // existential classification of an inner `\E ... : <pure guard>` (it starts
    // enumerating witnesses as successors). Keeping the de-indenting `.trim()`
    // preserves the shape the action IR expects. Regression: Paxos `Phase2a`
    // generated 2 (duplicate) successors instead of 1.
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
    // Body uses `.trim()` (NOT `dedent_common`); see try_parse_exists for why
    // realigning a quantifier body here breaks action-IR existential
    // classification (Paxos `Phase2a` duplicate-successor regression).
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
    PrimedMembership {
        var: String,
        set_expr: CompiledExpr,
    },
    Unchanged {
        vars: Vec<String>,
    },
    Guard {
        expr: CompiledExpr,
        /// Original source text of the guard expression.
        ///
        /// Retained so the runtime can detect when a "guard" is actually an
        /// action call to a user-defined operator (with primed assignments)
        /// and fall back to the interpreted action evaluator. Without this,
        /// `\E x \in S : ActionCall(x)` silently produces zero successors —
        /// a soundness bug that hides invariant violations.
        text: String,
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
        ActionClause::PrimedMembership { var, set_expr } => {
            CompiledActionClause::PrimedMembership {
                var: var.clone(),
                set_expr: compile_expr(set_expr),
            }
        }
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
        ClauseKind::PrimedMembership { var, set_expr } => CompiledActionClause::PrimedMembership {
            var,
            set_expr: compile_expr(&set_expr),
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
                    text: trimmed.to_string(),
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

        // Defensive: malformed input may pack two `==` markers adjacently,
        // making `body_end < eq_pos + 2`. Skip rather than panic.
        let body_start = *eq_pos + 2;
        if body_start > body_end {
            continue;
        }
        let name = trim_let_edge_comments(&defs_text[name_start..*eq_pos]);
        let body = trim_let_edge_comments(&defs_text[body_start..body_end]);

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

    /// Compact structural shape of a compiled expression, for asserting on the
    /// parse tree of indentation-sensitive boolean shapes without depending on
    /// leaf details. e.g. `And[_,Or[_,_]]`, `Forall(And[_,_])`.
    fn shape(e: &CompiledExpr) -> String {
        match e {
            CompiledExpr::And(xs) => {
                format!("And[{}]", xs.iter().map(shape).collect::<Vec<_>>().join(","))
            }
            CompiledExpr::Or(xs) => {
                format!("Or[{}]", xs.iter().map(shape).collect::<Vec<_>>().join(","))
            }
            CompiledExpr::Implies(a, b) => format!("Implies({},{})", shape(a), shape(b)),
            CompiledExpr::Iff(a, b) => format!("Iff({},{})", shape(a), shape(b)),
            CompiledExpr::Not(a) => format!("Not({})", shape(a)),
            CompiledExpr::Forall { body, .. } => format!("Forall({})", shape(body)),
            CompiledExpr::Exists { body, .. } => format!("Exists({})", shape(body)),
            CompiledExpr::Let { body, .. } => format!("Let({})", shape(body)),
            CompiledExpr::Unparsed(s) if s.trim().is_empty() => "EMPTY".to_string(),
            _ => "_".to_string(),
        }
    }

    /// 3-level nested implication where the innermost `\A` head is MORE indented
    /// than its bulleted body (the body aligns to the enclosing bullet) and the
    /// body ends in a subscript `[n]`. Regression for the MCCheckpointCoordination
    /// `SafetyInvariant` mis-parse: the over-indented `\A` kept a leading space
    /// after `dedent_common`, so `starts_with("\\A ")` failed and the trailing
    /// `[n]` was mis-read as a function application over the whole quantifier
    /// (`FuncApply(Unparsed("\\A n : ... ~g"), [n])`), making the invariant
    /// evaluate to a Function instead of a Boolean.
    #[test]
    fn deeply_nested_implication_quantifier_body_with_trailing_subscript() {
        let e = compile_expr(
            "  /\\ p = 0 =>\n    /\\ q = 1\n    /\\ r = 2 =>\n      /\\ \\A n \\in S :\n        /\\ ~f[n]\n        /\\ ~g[n]",
        );
        assert_eq!(
            shape(&e),
            "Implies(_,And[_,Implies(_,Forall(And[Not(_),Not(_)]))])",
            "deeply nested quantifier body with trailing subscript must not become FuncApply"
        );
    }

    /// Indentation-aware boolean parsing: these shapes must parse by their
    /// TLA+ bulleted-list/quantifier-body structure, NOT by flat symbol
    /// splitting. They pin the two ambiguous cases that a naive
    /// first-line-de-basing splitter conflates:
    ///   - sibling junction items (`\/ A` / `\/ B`, aligned) -> separate items
    ///   - a quantifier/LET head's indented body -> stays inside the head
    /// See the MCCheckpointCoordination SafetyInvariant + Paxos Phase2a specs.
    #[test]
    fn indented_boolean_disambiguates_siblings_from_nested_bodies() {
        // (1) Nested quantifier body: `\A n : /\ a /\ b` — the body is the
        // quantifier's, NOT sibling conjuncts. (The MCCheckpointCoordination
        // SafetyInvariant bug: this flattened to And[Forall(EMPTY),_,_].)
        let q = compile_expr("/\\ \\A n \\in Node :\n      /\\ a = 1\n      /\\ b = 2");
        assert_eq!(shape(&q), "Forall(And[_,_])", "nested quantifier body");

        // (2) Sibling disjunction after a conjunct (Phase2a): the second
        // conjunct's two aligned `\/` items are siblings.
        let p = compile_expr("/\\ x = 1\n/\\ \\/ a = 1\n   \\/ b = 2");
        assert_eq!(shape(&p), "And[_,Or[_,_]]", "sibling disjunction");

        // (3) Disjunct that is a quantifier with an indented body (Phase2a):
        // the `\E m` body stays inside the Exists.
        let d = compile_expr("\\/ a = 1\n\\/ \\E m \\in S :\n     /\\ p = 1\n     /\\ q = 2");
        assert_eq!(shape(&d), "Or[_,Exists(And[_,_])]", "nested exists body");

        // (4) Implication consequent that is an indented conjunction with a
        // nested implication (the SafetyInvariant top shape):
        // `P => /\ A /\ (R => B)`.
        let i = compile_expr(
            "L = N =>\n  /\\ a = 1\n  /\\ c = 2 =>\n    /\\ \\A n \\in Node : d = 1",
        );
        assert_eq!(
            shape(&i),
            "Implies(_,And[_,Implies(_,Forall(_))])",
            "implication consequent as indented conjunction with nested implication"
        );
    }

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

    /// T1.6 (FingerprintStoreResize) regression: `<=>` must compile to
    /// `Iff(...)`, NOT to `Implies((... <), ...)`. The bug pre-fix:
    /// `split_binary_op(expr, "=>")` matched the `=>` *inside* `<=>`, leaving
    /// LHS = `(seqlock % 2 = 1) <` and the resulting compiled tree later
    /// errored with `expected Int, got Bool(false)` at runtime (the trailing
    /// `<` made the LHS unparseable). See RELEASE_1.0.0_LOG.md `### T1.6`.
    #[test]
    fn iff_compiles_as_iff_not_implies_t1_6() {
        let expr = compile_expr("(seqlock % 2 = 1) <=> resizing");
        assert!(
            matches!(expr, CompiledExpr::Iff(_, _)),
            "<=> must compile to Iff, got: {:?}",
            expr
        );

        // Make sure left side is the parenthesised comparison, NOT a malformed
        // `Lt((seqlock % 2 = 1), <empty>)`.
        if let CompiledExpr::Iff(lhs, rhs) = expr {
            assert!(
                matches!(*lhs, CompiledExpr::Eq(_, _)),
                "lhs of <=> should be Eq(seqlock % 2, 1), got: {:?}",
                lhs
            );
            assert!(
                matches!(*rhs, CompiledExpr::Var(_)),
                "rhs of <=> should be Var(resizing), got: {:?}",
                rhs
            );
        }

        // `=>` (without `<`) must still compile to Implies.
        let imp = compile_expr("p => q");
        assert!(
            matches!(imp, CompiledExpr::Implies(_, _)),
            "=> must still compile to Implies, got: {:?}",
            imp
        );

        // Mixed precedence: `a => b <=> c` must parse as `(a => b) <=> c`
        // (=> binds tighter than <=>).
        let mixed = compile_expr("a => b <=> c");
        assert!(
            matches!(mixed, CompiledExpr::Iff(_, _)),
            "a => b <=> c must compile to Iff at the top, got: {:?}",
            mixed
        );
        if let CompiledExpr::Iff(lhs, _) = mixed {
            assert!(
                matches!(*lhs, CompiledExpr::Implies(_, _)),
                "lhs of `a => b <=> c` should be Implies(a, b), got: {:?}",
                lhs
            );
        }
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

    /// T2.1 regression: outer `[... EXCEPT ...]` whose base is itself a
    /// `[r EXCEPT !.f = v]` must be recognised as a nested EXCEPT, not
    /// have its inner EXCEPT keyword swallowed by `try_parse_except`.
    /// Before the fix, `try_parse_except` did `inner.find(" EXCEPT ")`
    /// which matched the FIRST EXCEPT — the one inside the nested
    /// brackets — producing `base = "[r"` and bubbling up
    /// `missing closing ']' in expression: [r`.
    #[test]
    fn test_t2_1_nested_except_compiles_as_funcexcept() {
        let expr = "[[r EXCEPT !.a = 0] EXCEPT !.b = 0]";
        let compiled = compile_expr(expr);
        assert!(
            matches!(compiled, CompiledExpr::FuncExcept(_, _)),
            "nested EXCEPT should compile to FuncExcept, got: {:?}",
            compiled
        );
        // Outer EXCEPT must update field `b`; its base must itself be a
        // FuncExcept that updates field `a`.
        if let CompiledExpr::FuncExcept(base, updates) = &compiled {
            assert_eq!(updates.len(), 1, "outer should have 1 update");
            assert!(
                matches!(base.as_ref(), CompiledExpr::FuncExcept(_, _)),
                "outer base should be the inner FuncExcept, got: {:?}",
                base
            );
            if let CompiledExpr::FuncExcept(inner_base, inner_updates) = base.as_ref() {
                assert!(
                    matches!(inner_base.as_ref(), CompiledExpr::Var(name) if name == "r"),
                    "innermost base should be Var(r), got: {:?}",
                    inner_base
                );
                assert_eq!(inner_updates.len(), 1, "inner should have 1 update");
            }
        }
    }

    /// T2.2 regression: a record literal whose field value contains a
    /// `[... EXCEPT ...]` sub-expression must compile as a RecordLiteral.
    /// Before the fix, the outer `[`-block's `try_parse_except` matched
    /// the EXCEPT inside the nested bracket and tried to slice the whole
    /// `inner` on it, producing a malformed base like
    /// `"a |-> 0, b |-> ([r"` and surfacing as
    /// `unexpected trailing tokens in expr: a |`.
    #[test]
    fn test_t2_2_except_inside_record_field_value() {
        let expr = "[a |-> 0, b |-> ([r EXCEPT !.a = 0]).a]";
        let compiled = compile_expr(expr);
        assert!(
            matches!(compiled, CompiledExpr::RecordLiteral(_)),
            "record literal with EXCEPT in a field value should compile to RecordLiteral, got: {:?}",
            compiled
        );
        if let CompiledExpr::RecordLiteral(fields) = &compiled {
            assert_eq!(fields.len(), 2, "should have 2 fields");
            assert_eq!(fields[0].0, "a");
            assert_eq!(fields[1].0, "b");
            // Field b's value should be a RecordAccess on a FuncExcept.
            assert!(
                matches!(&fields[1].1, CompiledExpr::RecordAccess(inner, fname)
                    if fname == "a" && matches!(inner.as_ref(), CompiledExpr::FuncExcept(_, _))),
                "field b should be (FuncExcept).a, got: {:?}",
                fields[1].1
            );
        }
    }

    /// T2.1 + T2.2 end-to-end: the compiled value must equal the
    /// interpreter's value. Both shapes should evaluate to
    /// `[a |-> 0, b |-> 0]`.
    #[test]
    fn test_t2_1_t2_2_evaluation_matches_interpreter() {
        use crate::tla::eval::{EvalContext, eval_expr};
        use crate::tla::value::TlaValue;
        use crate::tla::{TlaState, tla_state};
        use std::collections::BTreeMap;
        use std::sync::Arc;

        // Build a state with `r = [a |-> 1, b |-> 2]`.
        let mut rec: BTreeMap<String, TlaValue> = BTreeMap::new();
        rec.insert("a".to_string(), TlaValue::Int(1));
        rec.insert("b".to_string(), TlaValue::Int(2));
        let state: TlaState = tla_state([("r", TlaValue::Record(HashedArc::new(rec)))]);
        let defs: BTreeMap<String, crate::tla::TlaDefinition> = BTreeMap::new();
        let ctx = EvalContext::with_definitions(&state, &defs);

        for expr in &[
            "[[r EXCEPT !.a = 0] EXCEPT !.b = 0]",
            "[a |-> 0, b |-> ([r EXCEPT !.a = 0]).a]",
        ] {
            let interp = eval_expr(expr, &ctx).unwrap_or_else(|e| {
                panic!("interpreter failed on `{expr}`: {e}");
            });
            let compiled = compile_expr(expr);
            let comp_val = crate::tla::compiled_eval::eval_compiled(&compiled, &ctx)
                .unwrap_or_else(|e| panic!("compiler failed on `{expr}`: {e}"));
            assert_eq!(
                interp, comp_val,
                "interpreter / compiler disagree on `{expr}`"
            );
        }
    }

    /// EXCEPT inside a function-application argument must not confuse
    /// the outer parser. e.g. `f[[r EXCEPT !.a = 0]]` should be parsed
    /// as `FuncApply(f, [FuncExcept(r, ...)])`, not have the inner
    /// EXCEPT keyword promoted to top level.
    #[test]
    fn test_top_level_except_ignores_func_application_arg() {
        // No top-level EXCEPT here — the only EXCEPT is inside `f[ ... ]`.
        let inner = "f[[r EXCEPT !.a = 0]]";
        // try_parse_except is called on the *inner* of an outer `[...]`,
        // but here we just want to confirm find_top_level_except sees
        // nothing at depth 0.
        assert_eq!(super::find_top_level_except(inner), None);
    }

    // ============================================================
    // T207c: direct unit tests on private helpers, addressing missed
    // mutations that no end-to-end test exercises.
    // ============================================================

    // ---- has_chained_top_level_arithmetic ----

    #[test]
    fn t207c_has_chained_one_op() {
        assert!(!super::has_chained_top_level_arithmetic("1 + 2"));
        assert!(!super::has_chained_top_level_arithmetic("a - b"));
    }
    #[test]
    fn t207c_has_chained_two_ops() {
        // Counter at 2, threshold 3 → not chained.
        assert!(!super::has_chained_top_level_arithmetic("1 + 2 + 3"));
        assert!(!super::has_chained_top_level_arithmetic("1 - 2 - 3"));
        assert!(!super::has_chained_top_level_arithmetic("1 + 2 - 3"));
    }
    #[test]
    fn t207c_has_chained_three_ops_is_chained() {
        // Counter hits 3 at threshold → chained.
        assert!(super::has_chained_top_level_arithmetic("1 + 2 + 3 + 4"));
        assert!(super::has_chained_top_level_arithmetic("1 - 2 - 3 - 4"));
        assert!(super::has_chained_top_level_arithmetic("a + b - c + d"));
    }
    #[test]
    fn t207c_has_chained_inside_parens_protected() {
        // Inner `(1+2+3+4)` has 3 ops but they're at depth_paren==1.
        // Outer scan sees only 1 op → not chained.
        assert!(!super::has_chained_top_level_arithmetic("a + (1 + 2 + 3 + 4)"));
    }
    #[test]
    fn t207c_has_chained_inside_braces_protected() {
        assert!(!super::has_chained_top_level_arithmetic("a + {1 + 2 + 3 + 4}"));
    }
    #[test]
    fn t207c_has_chained_inside_brackets_protected() {
        assert!(!super::has_chained_top_level_arithmetic("a + [1 + 2 + 3 + 4]"));
    }
    #[test]
    fn t207c_has_chained_inside_angle_protected() {
        assert!(!super::has_chained_top_level_arithmetic("a + <<1 + 2 + 3 + 4>>"));
    }
    #[test]
    fn t207c_has_chained_inside_string_protected() {
        // Catches `in_string = !in_string` mutation deletion / flip.
        assert!(!super::has_chained_top_level_arithmetic("\"1 + 2 + 3 + 4\""));
        assert!(!super::has_chained_top_level_arithmetic("a + \"1 + 2 + 3 + 4\""));
    }
    #[test]
    fn t207c_has_chained_unary_minus_after_eq_not_counted() {
        // `=` is not a value-end → following `-` is unary, not counted.
        assert!(!super::has_chained_top_level_arithmetic("x = -1 + 2 + 3"));
    }
    #[test]
    fn t207c_has_chained_value_end_paren_close() {
        // `)` IS a value-end → following `+`/`-` IS binary.
        // Catches `p == b')'` mutations.
        assert!(super::has_chained_top_level_arithmetic("(1) + (2) + (3) + (4)"));
    }
    #[test]
    fn t207c_has_chained_value_end_bracket_close() {
        // `]` IS a value-end.
        assert!(super::has_chained_top_level_arithmetic("a[1] + b[1] + c[1] + d[1]"));
    }
    #[test]
    fn t207c_has_chained_value_end_angle_close() {
        // `>` IS a value-end (e.g. closes `<<...>>`).
        assert!(super::has_chained_top_level_arithmetic("<<1>> + <<2>> + <<3>> + <<4>>"));
    }
    #[test]
    fn t207c_has_chained_value_end_digit() {
        assert!(super::has_chained_top_level_arithmetic("1 + 2 + 3 + 4"));
    }
    #[test]
    fn t207c_has_chained_value_end_letter() {
        assert!(super::has_chained_top_level_arithmetic("a + b + c + d"));
    }
    #[test]
    fn t207c_has_chained_value_end_underscore() {
        // `_` is not is_ascii_alphabetic but the predicate explicitly checks `b'_'`.
        // Catches `p == b'_'` → `!= b'_'` mutations.
        assert!(super::has_chained_top_level_arithmetic("a_ + b_ + c_ + d_"));
    }

    // ---- split_first_top_level_op ----

    #[test]
    fn t207c_split_first_op_finds_leftmost() {
        // Mirrors the interpreter — split at FIRST top-level relop.
        // OPS contains: \\subseteq, \\notin, \\in, \\geq, \\leq, >=, =<, <=, /=, #, =, >, <
        let ops: &[&str] = &["#", "="];
        let r = super::split_first_top_level_op("a # b = c", ops);
        assert!(r.is_some());
        let (l, op, _r) = r.unwrap();
        assert_eq!(op, "#");
        assert_eq!(l.trim(), "a");
    }
    #[test]
    fn t207c_split_first_op_skips_inside_parens() {
        let ops: &[&str] = &["+"];
        // `(a + b) + c` — first top-level `+` is the SECOND one, not the inner one.
        let r = super::split_first_top_level_op("(a + b) + c", ops);
        assert!(r.is_some());
        let (l, _op, r2) = r.unwrap();
        assert_eq!(l.trim(), "(a + b)");
        assert_eq!(r2.trim(), "c");
    }
    #[test]
    fn t207c_split_first_op_word_boundary_in_intersect() {
        // `\in` should not match inside `\intersect`.
        let ops: &[&str] = &["\\in"];
        let r = super::split_first_top_level_op("a \\intersect b", ops);
        // No real `\in` exists — should return None or split at `\intersect`,
        // not at the `\in` prefix of `\intersect`.
        if let Some((_, op, _)) = r {
            assert_ne!(op, "\\in", "must not match \\in inside \\intersect");
        }
    }
    #[test]
    fn t207c_split_first_op_no_match() {
        let ops: &[&str] = &["#"];
        assert!(super::split_first_top_level_op("a + b", ops).is_none());
    }
    #[test]
    fn t207c_split_first_op_empty_left_or_right_skipped() {
        let ops: &[&str] = &["+"];
        // `+a` has empty left → must NOT split (continues looking).
        // No further `+` exists → None.
        assert!(super::split_first_top_level_op("+a", ops).is_none());
        // `a+` has empty right → must NOT split.
        assert!(super::split_first_top_level_op("a+", ops).is_none());
    }

    // ---- split_binary_op_with ----

    #[test]
    fn t207c_split_binary_first_match() {
        // prefer_last=false → return on first match.
        let r = super::split_binary_op_with("1 + 2 + 3", "+", false);
        assert!(r.is_some());
        let (l, _) = r.unwrap();
        assert_eq!(l.trim(), "1");
    }
    #[test]
    fn t207c_split_binary_last_match() {
        // prefer_last=true → keep scanning, return last.
        let r = super::split_binary_op_with("1 + 2 + 3", "+", true);
        assert!(r.is_some());
        let (l, _) = r.unwrap();
        assert_eq!(l.trim(), "1 + 2");
    }
    #[test]
    fn t207c_split_binary_skips_inside_brackets() {
        let r = super::split_binary_op_with("[1 + 2] + 3", "+", false);
        assert!(r.is_some());
        let (l, r2) = r.unwrap();
        assert_eq!(l.trim(), "[1 + 2]");
        assert_eq!(r2.trim(), "3");
    }

    // ---- find_keyword: word-boundary keyword search ----

    #[test]
    fn t207c_find_keyword_word_boundary_skips_substring() {
        // `IN` inside `INVENTORY` must NOT match.
        // The function returns None if no word-bounded `IN` exists.
        assert_eq!(super::find_keyword("INVENTORY", "IN"), None);
    }
    #[test]
    fn t207c_find_keyword_finds_then_with_spaces() {
        // find_keyword tracks LET/IN and IF/ELSE nesting and only returns
        // positions where let_depth == if_depth == 0. So we use a "THEN"
        // outside any IF — caller is responsible for IF tracking.
        // `IF a THEN b ELSE c` has THEN inside IF → not found at top level.
        // Direct `a THEN b` has if_depth = 0 → found.
        let r = super::find_keyword("a THEN b", "THEN");
        assert!(r.is_some(), "should find THEN outside any IF");
    }
    #[test]
    fn t207c_find_keyword_no_match() {
        assert_eq!(super::find_keyword("a + b", "THEN"), None);
    }

    // ---- split_quantifier_bindings ----

    #[test]
    fn t207c_split_quantifier_single_var() {
        let parts = super::split_quantifier_bindings("x \\in S");
        assert_eq!(parts.len(), 1);
        assert!(parts[0].contains("x") && parts[0].contains("S"));
    }
    #[test]
    fn t207c_split_quantifier_multi_var_one_domain() {
        // "x, y \in S" — single binding with two vars
        let parts = super::split_quantifier_bindings("x, y \\in S");
        // Compiler treats this as one binding in some forms; either 1 or 2 entries
        // are valid depending on convention. Test that it doesn't panic and
        // returns nonempty.
        assert!(!parts.is_empty());
    }
    #[test]
    fn t207c_split_quantifier_two_bindings_comma_separated() {
        // "x \in S, y \in T" — two distinct bindings
        let parts = super::split_quantifier_bindings("x \\in S, y \\in T");
        assert_eq!(parts.len(), 2);
    }

    // ---- split_case_arms ----

    #[test]
    fn t207c_split_case_two_arms() {
        let arms = super::split_case_arms(" a -> 1 [] b -> 2");
        assert_eq!(arms.len(), 2);
    }
    #[test]
    fn t207c_split_case_three_arms_with_other() {
        let arms = super::split_case_arms(" a -> 1 [] b -> 2 [] OTHER -> 3");
        assert_eq!(arms.len(), 3);
    }

    // ---- split_let_expression ----

    #[test]
    fn t207c_split_let_simple() {
        let r = super::split_let_expression("LET x == 1 IN x + 1");
        assert!(r.is_some());
        let (defs, body) = r.unwrap();
        assert!(defs.contains("x == 1"));
        assert!(body.contains("x + 1"));
    }
    #[test]
    fn t207c_split_let_nested() {
        let r = super::split_let_expression("LET x == 1 IN LET y == 2 IN x + y");
        assert!(r.is_some());
    }
    #[test]
    fn t207c_split_let_no_let_keyword() {
        assert!(super::split_let_expression("x + 1").is_none());
    }

    // ---- find_definition_equals ----

    #[test]
    fn t207c_find_def_equals_simple() {
        let positions = super::find_definition_equals("x == 1");
        assert_eq!(positions.len(), 1);
    }
    #[test]
    fn t207c_find_def_equals_multi() {
        let positions = super::find_definition_equals("x == 1\ny == 2\nz == 3");
        assert_eq!(positions.len(), 3);
    }
    #[test]
    fn t207c_find_def_equals_skips_eq_op() {
        // Single = is comparison, not definition.
        let positions = super::find_definition_equals("x = 1");
        assert_eq!(positions.len(), 0);
    }

    // ---- find_top_level_except ----

    #[test]
    fn t207c_find_except_simple() {
        let r = super::find_top_level_except("r EXCEPT !.a = 1");
        assert!(r.is_some());
    }
    #[test]
    fn t207c_find_except_no_except() {
        assert_eq!(super::find_top_level_except("r"), None);
    }
    #[test]
    fn t207c_find_except_inside_brackets_skipped() {
        // EXCEPT inside `[...]` should not be top-level.
        let r = super::find_top_level_except("a [r EXCEPT !.x = 0] b");
        assert_eq!(r, None);
    }

    // ---- strip_label_prefix ----

    #[test]
    fn t207c_strip_label_no_label() {
        assert_eq!(super::strip_label_prefix("x + 1"), "x + 1");
    }
    #[test]
    fn t207c_strip_label_simple() {
        // "label:: expr" — strips the label.
        let s = super::strip_label_prefix("Lbl:: x + 1");
        assert_eq!(s.trim(), "x + 1");
    }
    #[test]
    fn t207c_strip_label_with_args() {
        // "label(a, b):: expr"
        let s = super::strip_label_prefix("Lbl(a, b):: x + 1");
        assert_eq!(s.trim(), "x + 1");
    }

    // ============================================================
    // T207d (iter5): targeted boundary tests for high-density miss
    // categories — split_quantifier_bindings, find_top_level_except,
    // find_definition_equals, split_first_top_level_op,
    // split_binary_op_with, split_let_expression, find_keyword.
    // Each test exercises a SPECIFIC byte-position branch so the
    // mutation at that branch flips observable output.
    // ============================================================

    // ---- split_quantifier_bindings: <<>> bracket tracking ----

    #[test]
    fn t207d_split_qbindings_with_tuple_in_domain() {
        // `<<x, y>>` is a tuple — the inner `,` is at depth>0 and must
        // not split. Catches `<` and `+ 1` mutations on lines 2256, 2262.
        let parts = super::split_quantifier_bindings("p \\in <<a, b>>");
        assert_eq!(parts.len(), 1);
        assert!(parts[0].contains("<<a, b>>"));
    }

    #[test]
    fn t207d_split_qbindings_with_set_in_domain() {
        let parts = super::split_quantifier_bindings("p \\in {a, b, c}");
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn t207d_split_qbindings_two_bindings_with_tuples() {
        // Two distinct bindings, each with a tuple domain.
        let parts = super::split_quantifier_bindings("p \\in <<1, 2>>, q \\in <<3, 4>>");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("<<1, 2>>"));
        assert!(parts[1].contains("<<3, 4>>"));
    }

    #[test]
    fn t207d_split_qbindings_three_bindings() {
        let parts = super::split_quantifier_bindings("a \\in S, b \\in T, c \\in U");
        assert_eq!(parts.len(), 3);
    }

    // ---- find_top_level_except: <<>> bracket tracking ----

    #[test]
    fn t207d_find_except_with_tuple_inner_skipped() {
        // EXCEPT inside `<<...>>` is at depth>0 → not top level.
        // Catches `<` and `+ 1` mutations on lines 2015, 2020.
        assert_eq!(
            super::find_top_level_except("<<r EXCEPT !.a = 1>>"),
            None
        );
    }

    #[test]
    fn t207d_find_except_at_top_after_tuple() {
        // <<a, b>> EXCEPT — top-level EXCEPT comes AFTER the tuple closes.
        let r = super::find_top_level_except("<<a, b>> EXCEPT !.x = 0");
        assert!(r.is_some());
    }

    #[test]
    fn t207d_find_except_nested_brace_skipped() {
        assert_eq!(super::find_top_level_except("{r EXCEPT !.a = 1}"), None);
    }

    #[test]
    fn t207d_find_except_nested_paren_skipped() {
        assert_eq!(super::find_top_level_except("(r EXCEPT !.a = 1)"), None);
    }

    // ---- find_definition_equals: scanning for `==` at top level ----

    #[test]
    fn t207d_find_def_eq_skips_inside_paren() {
        // (a == b) — the == is inside parens, not a top-level definition.
        let positions = super::find_definition_equals("(a == b)");
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn t207d_find_def_eq_skips_inside_brace() {
        let positions = super::find_definition_equals("{a == b}");
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn t207d_find_def_eq_skips_inside_tuple() {
        // `<<>>` increments depth → == inside is not top-level.
        let positions = super::find_definition_equals("<<a == b>>");
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn t207d_find_def_eq_skips_inside_let_in() {
        // LET x == 1 IN ... — the inner == is part of LET binding,
        // tracked via let_depth; top-level == count is 0 here.
        let positions = super::find_definition_equals("LET x == 1 IN x");
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn t207d_find_def_eq_after_let_in_resumes() {
        // After IN closes the LET, we're back at top level.
        // x == 1 (top-level) followed by LET y == 2 IN y == 3 (one inner)
        // total top-level: 1.
        let positions = super::find_definition_equals("x == 1\nLET y == 2 IN y");
        assert_eq!(positions.len(), 1);
    }

    #[test]
    fn t207d_find_def_eq_correct_byte_position() {
        // "x == 1" → `==` at byte position 2.
        let positions = super::find_definition_equals("x == 1");
        assert_eq!(positions, vec![2]);
    }

    // ---- split_first_top_level_op: leftmost-match-first ----

    #[test]
    fn t207d_split_first_op_at_position_zero_skipped() {
        // "+a" — left side empty → must continue, no other op → None.
        let ops: &[&str] = &["+"];
        assert!(super::split_first_top_level_op("+a", ops).is_none());
    }

    #[test]
    fn t207d_split_first_op_inside_set_skipped() {
        // {a + b} + c — first top-level + is the SECOND one (outside set).
        let ops: &[&str] = &["+"];
        let r = super::split_first_top_level_op("{a + b} + c", ops);
        assert!(r.is_some());
        let (l, _, r2) = r.unwrap();
        assert_eq!(l.trim(), "{a + b}");
        assert_eq!(r2.trim(), "c");
    }

    #[test]
    fn t207d_split_first_op_inside_tuple_skipped() {
        let ops: &[&str] = &["+"];
        let r = super::split_first_top_level_op("<<a + b>> + c", ops);
        assert!(r.is_some());
        let (l, _, r2) = r.unwrap();
        assert_eq!(l.trim(), "<<a + b>>");
        assert_eq!(r2.trim(), "c");
    }

    #[test]
    fn t207d_split_first_op_disambig_eq_arrow() {
        // `=>` must NOT match as `=`. ops contains both with `=` last.
        let ops: &[&str] = &["="];
        // `a => b` — the `=` here is part of `=>` and must be skipped.
        let r = super::split_first_top_level_op("a => b", ops);
        assert!(r.is_none(), "= should not match inside =>");
    }

    #[test]
    fn t207d_split_first_op_disambig_eq_le() {
        // `<=` contains `=` but must not split as `=`.
        let ops: &[&str] = &["="];
        let r = super::split_first_top_level_op("a <= b", ops);
        assert!(r.is_none(), "= should not match inside <=");
    }

    #[test]
    fn t207d_split_first_op_disambig_lt_le() {
        // `<=` must not match as `<`.
        let ops: &[&str] = &["<"];
        assert!(super::split_first_top_level_op("a <= b", ops).is_none());
    }

    #[test]
    fn t207d_split_first_op_disambig_gt_ge() {
        let ops: &[&str] = &[">"];
        assert!(super::split_first_top_level_op("a >= b", ops).is_none());
    }

    #[test]
    fn t207d_split_first_op_skip_colon_gt() {
        // `:>` should not split as `>` (operator-as-separator pattern).
        let ops: &[&str] = &[">"];
        let r = super::split_first_top_level_op("a :> b", ops);
        assert!(r.is_none() || r.unwrap().1 != ">");
    }

    // ---- split_binary_op_with: prefer_first vs prefer_last ----

    #[test]
    fn t207d_split_binary_outer_plus_after_let() {
        // `(LET x == 1 IN x) + 2` — the `+` is OUTSIDE the LET (let_depth=0
        // when we hit the `+`). split_binary_op_with should split there.
        let r = super::split_binary_op_with("(LET x == 1 IN x) + 2", "+", false);
        assert!(r.is_some());
        let (l, r2) = r.unwrap();
        assert_eq!(l.trim(), "(LET x == 1 IN x)");
        assert_eq!(r2.trim(), "2");
    }

    #[test]
    fn t207d_split_binary_inside_quantifier_body_skipped() {
        // \A x \in S : x + 1 — the `+` is inside quantifier body.
        let r = super::split_binary_op_with("\\A x \\in S : x + 1", "+", false);
        assert!(r.is_none(), "+ inside quantifier body should not split");
    }

    #[test]
    fn t207d_split_binary_inside_if_skipped() {
        // IF a > b THEN ... — the `>` is inside IF condition.
        let r = super::split_binary_op_with("IF a > b THEN c ELSE d", ">", false);
        // The > inside IF condition should not be a top-level split.
        // The IF...THEN/ELSE structure protects it.
        assert!(r.is_none() || {
            let (l, _) = r.unwrap();
            !l.trim().eq("IF a")
        });
    }

    // ---- split_let_expression: nested LET handling ----

    #[test]
    fn t207d_split_let_with_nested_let() {
        // LET a == 1 IN LET b == 2 IN a + b — outer should match first IN.
        let r = super::split_let_expression("LET a == 1 IN LET b == 2 IN a + b");
        assert!(r.is_some());
        let (defs, body) = r.unwrap();
        assert!(defs.contains("a == 1"));
        assert!(body.starts_with("LET b == 2"));
    }

    #[test]
    fn t207d_split_let_must_start_with_let() {
        assert!(super::split_let_expression("a == 1 IN b").is_none());
    }

    #[test]
    fn t207d_split_let_no_in() {
        assert!(super::split_let_expression("LET x == 1").is_none());
    }

    // ---- find_keyword: depth tracking and word boundaries ----

    #[test]
    fn t207d_find_keyword_inside_brackets_skipped() {
        // `[THEN]` — at depth_bracket > 0, should not match.
        assert_eq!(super::find_keyword("[THEN]", "THEN"), None);
    }

    #[test]
    fn t207d_find_keyword_inside_parens_skipped() {
        assert_eq!(super::find_keyword("(THEN)", "THEN"), None);
    }

    #[test]
    fn t207d_find_keyword_inside_braces_skipped() {
        assert_eq!(super::find_keyword("{THEN}", "THEN"), None);
    }

    #[test]
    fn t207d_find_keyword_in_substring_not_matched() {
        // THENCE contains THEN but must not match (word boundary).
        assert_eq!(super::find_keyword("THENCE", "THEN"), None);
    }

    #[test]
    fn t207d_find_keyword_returns_byte_offset() {
        // For ASCII-only input, byte offset == char position.
        let r = super::find_keyword("a THEN b", "THEN");
        assert_eq!(r, Some(2));
    }

    // ---- has_chained_top_level_arithmetic: more boundary cases ----

    #[test]
    fn t207d_has_chained_with_unary_after_open_paren() {
        // `(- a + b + c + d)` — inside parens; outer scan sees no ops.
        // depth_paren guards the inner ops.
        assert!(!super::has_chained_top_level_arithmetic("f(-a + b + c + d)"));
    }

    #[test]
    fn t207d_has_chained_only_unary_minuses() {
        // `- - - - 1` — preceded by spaces (not value-end), so all unary.
        // Counter never increments → not chained.
        assert!(!super::has_chained_top_level_arithmetic("- - - - 1"));
    }

    #[test]
    fn t207d_has_chained_minus_after_op_is_unary() {
        // `1 + -2 + -3 + -4` — first `+` is binary; following `-` are unary
        // (preceded by `+`/space, not value-end). So binary count: 3 (the
        // three `+`s) — chained.
        // But `+` after value-end ` ` is not value-end... let me think.
        // Actually `1` is value-end, then ` `, then `+` (binary, count 1),
        // then ` `, then `-` (preceded by `+`/space — NOT value-end → unary),
        // then `2` (value-end), then ` `, `+` (binary, count 2), `-` (unary),
        // `3`, ` `, `+` (binary, count 3) → chained!
        assert!(super::has_chained_top_level_arithmetic("1 + -2 + -3 + -4"));
    }

    // ---- find_top_level_arrow: -> at top level ----

    #[test]
    fn t207d_find_arrow_inside_paren_skipped() {
        assert_eq!(super::find_top_level_arrow("(a -> b)"), None);
    }

    #[test]
    fn t207d_find_arrow_at_top_level() {
        // For e.g. CASE arms `a -> b`. A simple top-level -> should be found.
        assert!(super::find_top_level_arrow("a -> b").is_some());
    }

    // ---- find_top_level_eq: = at top level ----

    #[test]
    fn t207d_find_top_eq_inside_brackets_skipped() {
        assert_eq!(super::find_top_level_eq("[a = b]"), None);
    }

    #[test]
    fn t207d_find_top_eq_at_top() {
        assert!(super::find_top_level_eq("a = b").is_some());
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

#[test]
fn test_dotdot_has_lower_precedence_than_arithmetic() {
    // "0 .. N-1" must parse as SetRange(0, Sub(N, 1))
    // NOT as Sub(SetRange(0, N), 1)
    let expr = compile_expr("0 .. N-1");
    match &expr {
        CompiledExpr::SetRange(lo, hi) => {
            assert!(
                matches!(**lo, CompiledExpr::Int(0)),
                "low bound should be Int(0), got: {:?}",
                lo
            );
            assert!(
                matches!(**hi, CompiledExpr::Sub(_, _)),
                "high bound should be Sub(N, 1), got: {:?}",
                hi
            );
        }
        other => panic!("expected SetRange, got: {:?}", other),
    }

    // "a+1 .. b-1" must parse as SetRange(Add(a,1), Sub(b,1))
    let expr = compile_expr("a+1 .. b-1");
    assert!(
        matches!(expr, CompiledExpr::SetRange(_, _)),
        "a+1 .. b-1 should be SetRange, got: {:?}",
        expr
    );
    if let CompiledExpr::SetRange(lo, hi) = &expr {
        assert!(
            matches!(**lo, CompiledExpr::Add(_, _)),
            "low bound should be Add, got: {:?}",
            lo
        );
        assert!(
            matches!(**hi, CompiledExpr::Sub(_, _)),
            "high bound should be Sub, got: {:?}",
            hi
        );
    }
}

#[test]
fn t2_3_dotdot_binds_tighter_than_set_ops() {
    // T2.3 SOUNDNESS regression: in TLA+ `..` has precedence 9-9, set ops
    // (`\union`, `\intersect`, `\\`) have precedence 8-8. So `..` binds
    // *tighter*: `S \union 1..3` parses as `S \union (1..3)` -- NOT
    // `(S \union 1)..3`. Pre-fix the compiled-expr cascade had `..` split
    // before `\union`/etc., producing the wrong shape.
    let expr = compile_expr("S \\union 1..3");
    assert!(
        matches!(expr, CompiledExpr::Union(_, _)),
        "S \\union 1..3 should be Union(S, SetRange(1, 3)), got: {:?}",
        expr
    );
    if let CompiledExpr::Union(_lhs, rhs) = &expr {
        assert!(
            matches!(**rhs, CompiledExpr::SetRange(_, _)),
            "RHS of Union should be SetRange(1, 3), got: {:?}",
            rhs
        );
    }

    let expr = compile_expr("S \\intersect 1..n");
    assert!(
        matches!(expr, CompiledExpr::Intersect(_, _)),
        "S \\intersect 1..n should be Intersect(S, SetRange(1, n)), got: {:?}",
        expr
    );
    if let CompiledExpr::Intersect(_lhs, rhs) = &expr {
        assert!(
            matches!(**rhs, CompiledExpr::SetRange(_, _)),
            "RHS of Intersect should be SetRange, got: {:?}",
            rhs
        );
    }

    // Set minus is also at 8-8.
    let expr = compile_expr("S \\ 1..3");
    assert!(
        matches!(expr, CompiledExpr::SetMinus(_, _)),
        "S \\ 1..3 should be SetMinus(S, SetRange(1, 3)), got: {:?}",
        expr
    );
    if let CompiledExpr::SetMinus(_lhs, rhs) = &expr {
        assert!(
            matches!(**rhs, CompiledExpr::SetRange(_, _)),
            "RHS of SetMinus should be SetRange, got: {:?}",
            rhs
        );
    }

    // \subseteq is at the relational/comparison tier (5-5), so `..` binds
    // tighter than that as well. `1..3 \subseteq S` parses as
    // `(1..3) \subseteq S`.
    let expr = compile_expr("1..3 \\subseteq S");
    assert!(
        matches!(expr, CompiledExpr::Subset(_, _)),
        "1..3 \\subseteq S should be Subset(SetRange(1, 3), S), got: {:?}",
        expr
    );
    if let CompiledExpr::Subset(lhs, _rhs) = &expr {
        assert!(
            matches!(**lhs, CompiledExpr::SetRange(_, _)),
            "LHS of Subset should be SetRange, got: {:?}",
            lhs
        );
    }
}

#[test]
fn t2_4_unary_minus_does_not_swallow_binary_minus() {
    // T2.4 SOUNDNESS regression: shapes like `(-1 - (r).a)` were silently
    // mis-compiled. `split_binary_op(expr, "-")` returned the FIRST
    // top-level `-`, which is the leading unary minus (left = ""). The
    // existing `left_ends_with_value` guard rejected that split but the
    // caller didn't keep scanning, so the binary `-` was lost. The whole
    // expression then fell through to `find_record_access_dot`, which
    // sliced off `.a` and produced
    //   `RecordAccess(Unparsed("-1 - (r)"), "a")`
    // — the compiler returned the record `r` itself instead of `-4`.
    //
    // T2.1+T2.2 fixed `(x * -3)`; T2.4 extends the fix so the cascade
    // also handles `unary-minus on the LEFT, binary-minus later`.
    let expr = compile_expr("(-1 - (r).a)");
    match &expr {
        CompiledExpr::Sub(lhs, rhs) => {
            assert!(
                matches!(**lhs, CompiledExpr::Int(-1)),
                "LHS should be Int(-1), got: {:?}",
                lhs
            );
            assert!(
                matches!(**rhs, CompiledExpr::RecordAccess(_, _)),
                "RHS should be RecordAccess, got: {:?}",
                rhs
            );
        }
        other => panic!(
            "(-1 - (r).a) should compile to Sub(Int(-1), RecordAccess(...)), got: {:?}",
            other
        ),
    }

    // Same shape without the outer parens — must still split correctly.
    let expr = compile_expr("-1 - (r).a");
    assert!(
        matches!(expr, CompiledExpr::Sub(_, _)),
        "-1 - (r).a should be Sub(Int(-1), RecordAccess(...)), got: {:?}",
        expr
    );

    // Unary minus on a parenthesized non-literal subexpression. Pre-fix
    // this was Unparsed because there was no rule to wrap `-(expr)` as
    // Neg when `expr` isn't a bare integer literal.
    let expr = compile_expr("-((LET __t1 == 0 IN (0 + __t1)))");
    assert!(
        matches!(expr, CompiledExpr::Neg(_)),
        "-((LET ...)) should be Neg(Let(...)), got: {:?}",
        expr
    );

    // Make sure we did NOT regress the T2.1+T2.2 case: `(x * -3)` must
    // still parse as Mul(Var(x), Int(-3)), not Sub(...).
    let expr = compile_expr("(x * -3)");
    assert!(
        matches!(expr, CompiledExpr::Mul(_, _)),
        "(x * -3) should be Mul(Var(x), Int(-3)), got: {:?}",
        expr
    );

    // And ordinary binary subtraction: `1 - 2` is Sub, not Neg.
    let expr = compile_expr("1 - 2");
    assert!(
        matches!(expr, CompiledExpr::Sub(_, _)),
        "1 - 2 should be Sub(Int(1), Int(2)), got: {:?}",
        expr
    );
}

#[test]
fn test_txlifecycle_body_compiles_to_two_conjuncts() {
    // Regression test: TxLifecycle invariant body from KeyValueStore spec.
    // The body has two forall conjuncts that must be split correctly.
    let body = r"    /\ \A t \in tx :
        \A k \in Key : (store[k] /= snapshotStore[t][k] /\ k \notin written[t]) => k \in missed[t]
    /\ \A t \in TxId \ tx :
        /\ \A k \in Key : snapshotStore[t][k] = NoVal
        /\ written[t] = {}
        /\ missed[t] = {}";
    let expr = compile_expr(body);
    match &expr {
        CompiledExpr::And(parts) => {
            assert_eq!(
                parts.len(),
                2,
                "Should have 2 conjuncts, got {}: {:?}",
                parts.len(),
                parts
            );
            assert!(
                matches!(parts[0], CompiledExpr::Forall { .. }),
                "First conjunct should be Forall, got: {:?}",
                parts[0]
            );
            assert!(
                matches!(parts[1], CompiledExpr::Forall { .. }),
                "Second conjunct should be Forall, got: {:?}",
                parts[1]
            );
        }
        other => panic!("Expected And, got: {:?}", other),
    }
}

#[test]
fn test_negated_bulleted_conjunction_compiles_to_not_of_and() {
    // Regression: `~` prefixing a *bulleted* junction. Root cause of the
    // byihive/Voucher{Issue,Cancel,Redeem,Transfer} false-violation
    // (checker halted at 1 distinct, violation=true). Their VTPConsistent
    // invariant is
    //     \A h \in H, i \in I : /\ ~ /\ hState[h] = "holding"
    //                                 /\ iState[i] = "aborted"
    //                           /\ ~ /\ hState[h] = "aborted"
    //                                 /\ iState[i] = "issued"
    // Before the fix, `~ /\ A /\ B` was fed to the /\-splitter, which sliced
    // a bare `~` as the first "conjunct" -> Not(compile_expr("")) ->
    // "empty expression" reported as a violation on every state (incl. Init).
    // The `~` must bind to the WHOLE bulleted list: `~(A /\ B)`.
    let body = "~ /\\ hState = \"holding\"\n  /\\ iState = \"aborted\"";
    let expr = compile_expr(body);
    match &expr {
        CompiledExpr::Not(inner) => match inner.as_ref() {
            CompiledExpr::And(parts) => assert_eq!(
                parts.len(),
                2,
                "negated junction should be Not(And([A, B])), got parts: {:?}",
                parts
            ),
            other => panic!("expected Not(And(..)), got Not({:?})", other),
        },
        other => panic!("expected Not(And(..)), got: {:?}", other),
    }

    // `~ \/ A \/ B` — disjunction variant, same shape.
    let disj = "~ \\/ a = 1\n  \\/ b = 2";
    match &compile_expr(disj) {
        CompiledExpr::Not(inner) => assert!(
            matches!(inner.as_ref(), CompiledExpr::Or(p) if p.len() == 2),
            "expected Not(Or([_, _])), got Not({:?})",
            inner
        ),
        other => panic!("expected Not(Or(..)), got: {:?}", other),
    }

    // Guardrail: tighter `~` on a plain operand (`~A /\ B`) is UNCHANGED —
    // it must stay `(~A) /\ B`, i.e. a top-level And, NOT Not(And(..)).
    match &compile_expr("~a = 1 /\\ b = 2") {
        CompiledExpr::And(parts) => assert_eq!(parts.len(), 2),
        other => panic!("`~A /\\ B` must be And([~A, B]), got: {:?}", other),
    }
}

/// T206: detect expressions with 3+ top-level binary `-` (or `+`) occurrences,
/// the precise pattern where compiler/interpreter associativity diverges.
/// Such chains return wrong answers (T206 case: `0-2--442-...^^4` →
/// interpreter -56, compiler -48, off by 8). Caller emits `Unparsed` so
/// the interpreter (the reference) parses it.
///
/// We deliberately DO NOT trigger on simple shapes like `x * -3` or
/// `a + b` (≤2 ops including unary) — those compile correctly. The
/// fallback only kicks in for genuine chains.
fn has_chained_top_level_arithmetic(expr: &str) -> bool {
    let bytes = expr.as_bytes();
    let mut depth_paren = 0i32;
    let mut depth_brace = 0i32;
    let mut depth_bracket = 0i32;
    let mut depth_angle = 0i32;
    let mut in_string = false;
    let mut prev_non_ws: Option<u8> = None;
    let mut binary_minus_or_plus = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'"' { in_string = false; }
            i += 1; continue;
        }
        match b {
            b'"' => { in_string = true; }
            b'(' => depth_paren += 1,
            b')' => depth_paren -= 1,
            b'{' => depth_brace += 1,
            b'}' => depth_brace -= 1,
            b'[' => depth_bracket += 1,
            b']' => depth_bracket -= 1,
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'<' => {
                depth_angle += 1; i += 2; continue;
            }
            b'>' if i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                depth_angle -= 1; i += 2; continue;
            }
            b'-' | b'+' if depth_paren == 0
                && depth_brace == 0
                && depth_bracket == 0
                && depth_angle == 0 =>
            {
                // Binary if the previous non-whitespace char is a value-end:
                // digit, identifier-letter, `)`, `]`, `>` (close angle).
                if let Some(p) = prev_non_ws {
                    if p.is_ascii_digit() || p.is_ascii_alphabetic() || p == b'_'
                        || p == b')' || p == b']' || p == b'>'
                    {
                        binary_minus_or_plus += 1;
                        if binary_minus_or_plus >= 3 {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
        if !b.is_ascii_whitespace() {
            prev_non_ws = Some(b);
        }
        i += 1;
    }
    false
}
