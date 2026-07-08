//! `expr_v2` — a layout-aware TLA+ expression parser (Phase 0).
//!
//! Pipeline: `source → lexer (byte span + line/col + comment stripping) →
//! hand-written recursive-descent + Pratt parser owning a layout-fence stack →
//! ExprAst (with spans + explicit Junction nodes) → lower to CompiledExpr`.
//!
//! As of Phase 2 the v2 parser is the DEFAULT parsing path. It runs unless
//! `TLAPLUS_EXPR_PARSER=v1` (or `=off`) is explicitly set, and it always falls
//! back to the old parser on any parse error, so it can never make an
//! expression fail to compile that used to. The env var is the rollback lever.
//!
//! What v2 owns STRUCTURALLY: `/\`/`\/` bulleted junctions, `=>`/`<=>`,
//! `\A`/`\E`, `LET .. IN ..`, `IF .. THEN .. ELSE ..`, and the Pratt binary /
//! unary operators. Everything else is an opaque `Atom` leaf lowered via the
//! existing `compile_expr`.

pub mod ast;
pub mod lexer;
pub mod lower;
pub mod parser;

pub use ast::Expr;

use crate::tla::compiled_expr::CompiledExpr;

/// Parse an expression with v2 and lower it to a `CompiledExpr`. Returns `Err`
/// on any parse failure (the caller falls back to the old parser).
pub fn parse_and_lower(expr: &str) -> Result<CompiledExpr, String> {
    let ast = parser::parse(expr)?;
    Ok(lower::lower(&ast))
}

/// Parse to the structural AST (used by golden tests / shadow-compare).
pub fn parse_ast(expr: &str) -> Result<Expr, String> {
    parser::parse(expr)
}

/// True iff the layout-aware v2 parser should run. As of Phase 2 this is the
/// DEFAULT (returns `true`); v2 is disabled only when `TLAPLUS_EXPR_PARSER` is
/// explicitly set to `v1` or `off` (case-insensitive), which is the rollback
/// escape hatch. Any other value (including unset or `v2`) enables v2. Read
/// once per call; cheap enough for the compile path (compilation is not on the
/// hot eval loop).
pub fn v2_enabled() -> bool {
    match std::env::var("TLAPLUS_EXPR_PARSER") {
        Ok(v) if v.eq_ignore_ascii_case("v1") || v.eq_ignore_ascii_case("off") => false,
        _ => true,
    }
}

#[cfg(test)]
mod golden_tests;
