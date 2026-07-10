//! Dual-evaluator consistency check (Phase 0 of the eval unification).
//!
//! The model hot path evaluates Init/Next/guard predicates through the
//! *interpreted* evaluator (`eval_expr`), while `check_invariants` goes
//! through the *compiled* evaluator (`eval_compiled(compile_expr(..))`). The
//! two are meant to agree on every predicate; historically they have drifted
//! (MCBakery, PRs #157–#159). This module provides an opt-in debug assertion
//! that, when a hot-path predicate is evaluated, also runs it through the
//! compiled path and reports any divergence.
//!
//! It is entirely gated behind the `eval-consistency-check` cargo feature.
//! In a default build every function here compiles to an empty no-op, so
//! there is **zero cost on the production path**. When the feature is on, the
//! check additionally requires the `TLAPP_EVAL_CONSISTENCY` env var to be set
//! (to `1`/`true`) before it does any work, so even a feature-enabled binary
//! pays nothing unless explicitly asked.
//!
//! By design this only *logs* divergences (to stderr) rather than panicking,
//! so a corpus run can enumerate every real-spec divergence in one pass
//! instead of aborting on the first. Set `TLAPP_EVAL_CONSISTENCY=panic` to
//! turn a divergence into a panic when running a focused repro.

#[cfg(feature = "eval-consistency-check")]
use crate::tla::EvalContext;

/// Compare the interpreted result for `expr` against the compiled result and
/// report any divergence. `interp` is the value the caller already obtained
/// from `eval_expr(expr, ctx)` — passing it in avoids re-evaluating the
/// interpreted path.
///
/// No-op unless built with `--features eval-consistency-check` AND
/// `TLAPP_EVAL_CONSISTENCY` is set.
#[cfg(feature = "eval-consistency-check")]
pub fn check_predicate_consistency(
    expr: &str,
    ctx: &EvalContext<'_>,
    interp: &anyhow::Result<crate::tla::TlaValue>,
) {
    use std::sync::OnceLock;
    static MODE: OnceLock<Mode> = OnceLock::new();
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Off,
        Log,
        Panic,
    }
    let mode = *MODE.get_or_init(|| match std::env::var("TLAPP_EVAL_CONSISTENCY") {
        Ok(v) if v == "panic" => Mode::Panic,
        Ok(v) if v == "1" || v == "true" || v == "log" => Mode::Log,
        _ => Mode::Off,
    });
    if mode == Mode::Off {
        return;
    }

    let compiled = crate::tla::compile_expr(expr);
    let compi = crate::tla::eval_compiled(&compiled, ctx);

    let divergent = match (interp, &compi) {
        (Ok(a), Ok(b)) => a != b,
        (Err(_), Err(_)) => false, // both-error is agreement
        _ => true,                 // one Ok, one Err
    };
    if !divergent {
        return;
    }

    let msg = format!(
        "EVAL-CONSISTENCY DIVERGENCE on predicate `{expr}`:\n  interpreted -> {:?}\n  compiled    -> {:?}",
        interp.as_ref().map_err(|e| e.to_string()),
        compi.as_ref().map_err(|e| e.to_string()),
    );
    if mode == Mode::Panic {
        panic!("{msg}");
    } else {
        eprintln!("{msg}");
    }
}

/// No-op stub for default builds (feature off). Monomorphised away entirely.
#[cfg(not(feature = "eval-consistency-check"))]
#[inline(always)]
pub fn check_predicate_consistency(
    _expr: &str,
    _ctx: &crate::tla::EvalContext<'_>,
    _interp: &anyhow::Result<crate::tla::TlaValue>,
) {
}
