//! `expr_v2` — a layout-aware TLA+ expression parser (Phase 0).
//!
//! Pipeline: `source → lexer (byte span + line/col + comment stripping) →
//! hand-written recursive-descent + Pratt parser owning a layout-fence stack →
//! ExprAst (with spans + explicit Junction nodes) → lower to CompiledExpr`.
//!
//! Phase 0 is ADDITIVE and FLAG-GATED. The new parser only runs when
//! `TLAPLUS_EXPR_PARSER=v2` is set (falling back to the old parser on any parse
//! error), or in tests / the shadow-compare harness. It changes NO default
//! behavior.
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

/// True iff the `TLAPLUS_EXPR_PARSER=v2` env flag is set. Read once per call;
/// cheap enough for the compile path (compilation is not on the hot eval loop).
pub fn v2_enabled() -> bool {
    std::env::var("TLAPLUS_EXPR_PARSER")
        .map(|v| v.eq_ignore_ascii_case("v2"))
        .unwrap_or(false)
}

#[cfg(test)]
mod golden_tests;
