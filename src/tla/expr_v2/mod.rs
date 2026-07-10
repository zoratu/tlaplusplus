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
    let ast = parse_ast(expr)?;
    Ok(lower::lower(&ast))
}

/// Parse to the structural AST (used by golden tests / shadow-compare).
///
/// This is the SINGLE choke point all v2 consumers hit — `parse_and_lower`
/// (the compiled path), `classify_boolean_v2` (the interpreted path), and the
/// `action_ir` helpers all funnel through here. After a successful parse it
/// runs the shared mis-fence detector [`root_junction_mis_fences`]; when the
/// root junction op DISAGREES with the leading bullet (a `\/`-led body that
/// mis-parsed to a root `And`, or vice versa) it returns `Err` so EVERY
/// consumer falls back to the alignment-tolerant v1 parser rather than
/// compiling/evaluating a flipped junction. See `root_junction_mis_fences`.
pub fn parse_ast(expr: &str) -> Result<Expr, String> {
    let ast = parser::parse(expr)?;
    if root_junction_mis_fences(&ast, expr) {
        return Err(
            "expr_v2 mis-fence: root junction op disagrees with leading bullet (→ v1 fallback)"
                .to_string(),
        );
    }
    Ok(ast)
}

/// Shared mis-fence detector (the soundness choke point).
///
/// The class of bug this closes: a body whose FIRST logical token is a `\/`
/// (resp. `/\`) bullet is semantically that junction, so its top operator MUST
/// be that same junction op. But v2's strict column fence mis-parses a junction
/// whose leading bullet sits at a SHALLOWER column than its continuation
/// bullets — the leading bullet collapses to a single item and a body-extending
/// form (`\A`/`\E`/`LET`/`IF`/`=>`) swallows the deeper sibling bullet, flipping
/// the root junction op. For example (leading `\/` at col 0, continuation `\/`
/// at col 4):
/// ```text
///     \/ a /\ \A c : P
///         \/ b /\ \A c : Q
/// ```
/// parses with a ROOT `Junction{And}` even though the body is an OR. Lowering
/// that blindly (`lower.rs` maps `And→CompiledExpr::And`) yields a wrong `And` —
/// an unguarded compiled-path false-safe / false-violation.
///
/// This detector fires ONLY when the ROOT node is a `Junction` of the OPPOSITE
/// op from the leading bullet. False-positive safety (critical): a body like
/// `/\ A /\ B => C` parses with root `Implies` (`=>` is looser than `/\`), NOT a
/// `Junction`, so the detector does NOT fire — a leading bullet whose true top
/// operator is a looser `=>`/`<=>` is never wrongly rejected. This mirrors the
/// corpus-proven guard in `classify_boolean_v2` (interpreted path) and the
/// `action_ir` disjunction/conjunction-body guards, consolidated to one place.
///
/// The leading-bullet probe skips leading whitespace, `\*` line comments, and
/// `(* *)` block comments (which nest), so a body led by a comment is judged on
/// its first *logical* token — matching how the parser lays out fences.
fn root_junction_mis_fences(ast: &Expr, expr: &str) -> bool {
    use ast::JunctionOp;
    let Expr::Junction { op, .. } = ast else {
        return false;
    };
    let lead = crate::tla::text_util::first_logical_line_skipping_comments(expr);
    let leads_with_or = lead.starts_with("\\/");
    let leads_with_and = lead.starts_with("/\\");
    match op {
        // Root is a conjunction but the body is led by a `\/` bullet → mis-fence.
        JunctionOp::And => leads_with_or,
        // Root is a disjunction but the body is led by a `/\` bullet → mis-fence.
        JunctionOp::Or => leads_with_and,
    }
}

/// True iff the layout-aware v2 parser should run. As of Phase 2 this is the
/// DEFAULT (returns `true`); v2 is disabled only when `TLAPLUS_EXPR_PARSER` is
/// explicitly set to `v1` or `off` (case-insensitive), which is the rollback
/// escape hatch. Any other value (including unset or `v2`) enables v2. Read
/// once per call; cheap enough for the compile path (compilation is not on the
/// hot eval loop).
pub fn v2_enabled() -> bool {
    v2_enabled_for(std::env::var("TLAPLUS_EXPR_PARSER").ok().as_deref())
}

/// Pure config decision underneath the env layer: maps the raw
/// `TLAPLUS_EXPR_PARSER` value (`None` when unset) to the enabled bool. v2 is
/// disabled only for an explicit `v1`/`off` (case-insensitive); every other
/// value — including `None` and `v2` — enables it. Factored out so the decision
/// can be unit-tested without mutating the process-global env (which is
/// `unsafe` and parallel-racy under the default test harness).
pub fn v2_enabled_for(value: Option<&str>) -> bool {
    match value {
        Some(v) if v.eq_ignore_ascii_case("v1") || v.eq_ignore_ascii_case("off") => false,
        _ => true,
    }
}

#[cfg(test)]
mod config_tests {
    use super::v2_enabled_for;

    #[test]
    fn v1_and_off_disable_case_insensitively() {
        for v in ["v1", "V1", "off", "OFF", "Off"] {
            assert!(!v2_enabled_for(Some(v)), "{v} should disable v2");
        }
    }

    #[test]
    fn unset_and_other_values_enable() {
        assert!(v2_enabled_for(None));
        assert!(v2_enabled_for(Some("v2")));
        assert!(v2_enabled_for(Some("")));
        assert!(v2_enabled_for(Some("anything")));
    }
}

#[cfg(test)]
mod mis_fence_tests {
    use super::{parse_and_lower, parse_ast};
    use crate::tla::compiled_expr::CompiledExpr;

    // The `\/`-led mis-fence shape (SingleLaneBridge `HaveSameDirection`-like):
    // leading `\/` at col 0, continuation `\/` at col 4, with a body-extending
    // `\A` that swallows the deeper sibling. v2 alone would parse this with a
    // ROOT `Junction{And}`; the choke-point detector must REJECT it so the
    // compiled path falls back to v1 and produces the CORRECT `Or`.
    const MIS_FENCE_OR: &str = "\\/ a /\\ \\A c \\in S : P\n    \\/ b /\\ \\A c \\in S : Q";

    #[test]
    fn parse_ast_rejects_or_led_and_root_mis_fence() {
        let r = parse_ast(MIS_FENCE_OR);
        assert!(
            r.is_err(),
            "mis-fenced \\/-led body must reject (→ v1 fallback), got: {:?}",
            r.map(|e| e.shape())
        );
    }

    // THE POINT OF THE FIX: the compiled path must NOT emit a flipped `And`.
    // With the choke-point reject, `parse_and_lower` returns `Err`, so
    // `parse_and_lower` refuses the mis-fence (the flipped `And` is never
    // produced), so the full `compile_expr` defers to the v1 fallback instead.
    #[test]
    fn compiled_path_does_not_produce_flipped_and() {
        // The detector fires, so v2 refuses this mis-fenced parse and can NEVER
        // hand `lower` a flipped root `And` on the compiled path.
        let r = parse_and_lower(MIS_FENCE_OR);
        assert!(
            r.is_err(),
            "compiled path must fall back on mis-fence, not emit a junction; got: {:?}",
            r
        );
        // Consequently the full compiled path falls back to v1 verbatim rather
        // than v2's flipped `And`. (This synthetic fixture is misaligned beyond
        // v1's tolerance too, so v1 does not itself yield `Or` here — the point
        // of the fix is that the compiled path defers to the v1 oracle. For a
        // real-world misaligned shape like SingleLaneBridge `HaveSameDirection`,
        // v1 is alignment-tolerant and agrees with TLC; end-to-end count
        // correctness is covered by the corpus gate, not this unit test.)
        let full = crate::tla::compiled_expr::compile_expr(MIS_FENCE_OR);
        let v1 = crate::tla::compiled_expr::compile_expr_v1(MIS_FENCE_OR);
        assert_eq!(
            format!("{full:?}"),
            format!("{v1:?}"),
            "compiled path must equal the v1 fallback, not v2's flipped And"
        );
    }

    // False-positive safety #1: `/\ A /\ B => C` has a leading `/\` bullet but
    // its TRUE top operator is the looser `=>`, so the root is `Implies`, NOT a
    // `Junction`. The detector must NOT fire (no false rejection).
    #[test]
    fn implies_looser_top_not_rejected() {
        let s = parse_ast("/\\ A /\\ B => C").expect("=>-topped body must NOT be rejected");
        assert!(
            s.shape().starts_with("Implies("),
            "expected Implies root, got: {}",
            s.shape()
        );
    }

    // False-positive safety #2: a well-aligned `\/ A \/ B` parses with a correct
    // root `Or` — leading bullet AGREES with the op → not rejected.
    #[test]
    fn aligned_or_not_rejected() {
        let s = parse_ast("\\/ A\n\\/ B").expect("aligned \\/ must parse");
        assert_eq!(s.shape(), "OR[Atom(\"A\"), Atom(\"B\")]");
    }

    // Aligned `/\ A /\ B` likewise stays a correct `And`.
    #[test]
    fn aligned_and_not_rejected() {
        let s = parse_ast("/\\ A\n/\\ B").expect("aligned /\\ must parse");
        assert_eq!(s.shape(), "AND[Atom(\"A\"), Atom(\"B\")]");
    }

    // The leading-bullet probe skips comments: a `\*`/`(* *)`-prefixed mis-fence
    // is still detected on its first LOGICAL token.
    #[test]
    fn mis_fence_detected_through_leading_comment() {
        let src = format!("(* lead comment *)\n{MIS_FENCE_OR}");
        assert!(
            parse_ast(&src).is_err(),
            "comment-prefixed mis-fence must still reject"
        );
    }
}

#[cfg(test)]
mod golden_tests;
