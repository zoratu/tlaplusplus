//! AST for the `expr_v2` TLA+ expression parser.
//!
//! This is a *structural* AST: it captures the layout-sensitive skeleton of a
//! TLA+ expression — junctions (`/\`/`\/` bulleted lists), implication/iff,
//! quantifiers, `LET`, `IF`, and a handful of infix/prefix operators — while
//! leaving anything it does not (yet) understand as an opaque [`Expr::Atom`]
//! whose text is lowered via the *existing* `compile_expr`. That keeps leaf
//! semantics byte-for-byte identical to the old parser and lets v2 own only the
//! junction / `=>` / quantifier / `LET` STRUCTURE that the old string-based
//! parser gets wrong.

/// A source span: byte offsets plus line/column (1-based line, 0-based column)
/// of the start and end of a token or sub-expression. Columns are computed on
/// the *original* source so layout fences see real token columns even though
/// comments are stripped lexically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

impl Span {
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            start_line: self.start_line,
            start_col: self.start_col,
            end_line: other.end_line,
            end_col: other.end_col,
        }
    }
}

/// Junction operator: conjunction (`/\`) or disjunction (`\/`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JunctionOp {
    And,
    Or,
}

/// Binary operators handled structurally by v2. Comparison / arithmetic / set
/// operators are handled by the Pratt precedence table; the ones that MUST be
/// structural for correct body attachment are `=>` and `<=>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Implies,
    Iff,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    // Set operators
    Union,       // \union / \cup
    Intersect,   // \intersect / \cap
    SetMinus,    // \ (backslash)
    // Sequence concat
    Concat,      // \o / \circ
}

/// Quantifier kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Forall,
    Exists,
}

/// A single `x \in S` (or `x, y \in S`) quantifier bound.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantBound {
    pub vars: Vec<String>,
    /// The domain expression (parsed as a sub-expression, usually an Atom).
    pub domain: Box<Expr>,
}

/// A `LET` binding: `id == value`, `f[x] == value`, or `Op(a, b) == value`.
#[derive(Debug, Clone, PartialEq)]
pub struct LetDef {
    pub name: String,
    /// Operator/function parameters, if any (`Op(a,b)` -> `["a","b"]`).
    pub params: Vec<String>,
    /// For `f[x] == e`, the function-argument binders (`["x"]`). Empty otherwise.
    pub func_args: Vec<String>,
    /// Raw text of the binding's value (lowered via the caller-provided leaf
    /// compiler, OR — if it is itself a structural form — as an Expr).
    pub value: Box<Expr>,
}

/// The structural AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A bulleted junction list: `op` bullets all aligned at `col`, with the
    /// given items in source order.
    Junction {
        op: JunctionOp,
        col: u32,
        items: Vec<Expr>,
        span: Span,
    },
    /// A binary operator application.
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// Logical negation `~e` / `\lnot e`.
    Not { operand: Box<Expr>, span: Span },
    /// A quantifier `\A b1, b2, ... : body` / `\E ... : body`.
    Quant {
        kind: QuantKind,
        bounds: Vec<QuantBound>,
        body: Box<Expr>,
        span: Span,
    },
    /// `LET def1 def2 ... IN body`.
    Let {
        defs: Vec<LetDef>,
        body: Box<Expr>,
        span: Span,
    },
    /// `IF cond THEN then_ ELSE else_`.
    If {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
        span: Span,
    },
    /// A parenthesized expression (kept explicit so lowering can be exact).
    Paren { inner: Box<Expr>, span: Span },
    /// An opaque leaf: text that v2 does not parse structurally. Lowered via the
    /// existing `compile_expr` so leaf semantics are unchanged.
    Atom { text: String, span: Span },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Junction { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Not { span, .. }
            | Expr::Quant { span, .. }
            | Expr::Let { span, .. }
            | Expr::If { span, .. }
            | Expr::Paren { span, .. }
            | Expr::Atom { span, .. } => *span,
        }
    }

    /// A compact, deterministic debug rendering used by golden tests and the
    /// shadow-compare harness. Deliberately structural (ignores spans).
    pub fn shape(&self) -> String {
        match self {
            Expr::Junction { op, items, .. } => {
                let tag = match op {
                    JunctionOp::And => "AND",
                    JunctionOp::Or => "OR",
                };
                let inner: Vec<String> = items.iter().map(|i| i.shape()).collect();
                format!("{}[{}]", tag, inner.join(", "))
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                format!("{:?}({}, {})", op, lhs.shape(), rhs.shape())
            }
            Expr::Not { operand, .. } => format!("Not({})", operand.shape()),
            Expr::Quant { kind, bounds, body, .. } => {
                let bs: Vec<String> = bounds
                    .iter()
                    .map(|b| format!("{}\\in {}", b.vars.join(","), b.domain.shape()))
                    .collect();
                format!("{:?}({} : {})", kind, bs.join("; "), body.shape())
            }
            Expr::Let { defs, body, .. } => {
                let ds: Vec<String> = defs
                    .iter()
                    .map(|d| format!("{}=={}", d.name, d.value.shape()))
                    .collect();
                format!("Let([{}] IN {})", ds.join(", "), body.shape())
            }
            Expr::If { cond, then_, else_, .. } => {
                format!("If({}, {}, {})", cond.shape(), then_.shape(), else_.shape())
            }
            Expr::Paren { inner, .. } => format!("({})", inner.shape()),
            Expr::Atom { text, .. } => format!("Atom({:?})", text.trim()),
        }
    }
}
