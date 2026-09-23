#![no_main]

//! Fuzz `split_action_body_disjuncts` — the top-level action-body disjunct
//! splitter that PR #187 crashed on (it mis-split
//! `\E x \in S: g /\ (\/ A \/ B)` and mishandled `UNCHANGED <<..>>`). Random
//! UTF-8 almost never forms a valid TLA action body, so this target GENERATES
//! TLA-action-shaped strings from a small grammar (nested `/\`, `\/`, `\E`/`\A`
//! quantifiers, parens, `ENABLED`, `UNCHANGED`, primes) and also appends an
//! adversarial raw-byte suffix, giving both deep structural coverage of the
//! splitter and the pathological-input coverage of a byte fuzzer.
//!
//! ## What the splitter is (per the TLC reference) and what this target checks
//!
//! TLC decomposes `Next` in `Tool.getActions`, before checking begins, into "as
//! many simple subactions as possible" — materializing even a top-level
//! `\E x \in S` into one subaction PER element of `S` ("Model Checking TLA+
//! Specifications", Yu/Manolios/Lamport 1999). That decomposition is maximal
//! and one-shot; a maximal decomposition is idempotent by construction.
//!
//! `split_action_body_disjuncts` is our string-level FIRST stage of that
//! decomposition, and it is deliberately NOT maximal: it separates top-level
//! `\/` but keeps `\E`-scoped and guard-shared `\/` grouped, leaving the
//! `\E`/guard distribution to eval time (`eval_exists_action_multi`). That is
//! the T1.5/#187 soundness design — distributing `\E i: (g /\ (A \/ B))` at the
//! string level drops the shared `g` from all but one branch. So this stage is
//! a PARTIAL, non-idempotent decomposition whose COMPOSITION with eval-time
//! expansion reproduces TLC's successor set. Idempotence is therefore neither a
//! contract nor a useful oracle here: a "fixpoint convergence" check would stay
//! green even if the splitter regressed into distributing
//! `\E i: (g /\ (A \/ B))` and dropping `g` — the exact #187 conjunct-drop —
//! because that still converges. Successor-set faithfulness to TLC is not
//! checkable without a TLC oracle; it lives in the diff-gate
//! (`corpus/diff_test/list.tsv`), not here.
//!
//! The reference-grounded contracts this target CAN check, and does:
//!   1. Totality: the splitter terminates and returns at least one subaction
//!      for a non-empty body (TLC always has at least the whole as one
//!      subaction). A hang or empty result is a bug.
//!   2. No panic / no abort — the #187 failure mode.
//! Both are asserted on the fully-structured input; the raw-suffixed and
//! raw-suffix-alone inputs are panic-checked only, since a truncated/garbled
//! tail can legitimately defeat top-level-delimiter scanning.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tlaplusplus::tla::action_ir::split_action_body_disjuncts;

/// A generated TLA+ action-body expression.
#[derive(Arbitrary, Debug)]
enum ActExpr {
    /// A plain state var: `x`.
    Var,
    /// A primed var: `x'`.
    Primed,
    /// A simple guard: `x > 0`.
    Guard,
    /// A simple assignment: `x' = x + 1`.
    Assign,
    /// `UNCHANGED <<x, y>>` — the shape that tripped prime staging in #187.
    Unchanged,
    /// `ENABLED ( body )`.
    Enabled(Box<ActExpr>),
    /// `( body )`.
    Paren(Box<ActExpr>),
    /// `\E i \in S: body` — leading quantifier over a conjunction/disjunction.
    Exists(Box<ActExpr>),
    /// `\A i \in S: body`.
    Forall(Box<ActExpr>),
    /// Conjunction: `a /\ b /\ ...`.
    And(Vec<ActExpr>),
    /// Disjunction: `a \/ b \/ ...` — the thing being split at top level.
    Or(Vec<ActExpr>),
}

impl ActExpr {
    /// Render to a TLA+ source string, bounding recursion so a deeply nested
    /// Arbitrary value degrades to a leaf rather than overflowing the RENDERER
    /// (a generator-side stack overflow would be a false positive; a splitter
    /// stack overflow, were one reachable, still surfaces from the split call).
    fn render(&self, depth: u32, out: &mut String) {
        if depth == 0 {
            out.push('x');
            return;
        }
        let d = depth - 1;
        match self {
            ActExpr::Var => out.push('x'),
            ActExpr::Primed => out.push_str("x'"),
            ActExpr::Guard => out.push_str("x > 0"),
            ActExpr::Assign => out.push_str("x' = x + 1"),
            ActExpr::Unchanged => out.push_str("UNCHANGED <<x, y>>"),
            ActExpr::Enabled(b) => {
                out.push_str("ENABLED (");
                b.render(d, out);
                out.push(')');
            }
            ActExpr::Paren(b) => {
                out.push('(');
                b.render(d, out);
                out.push(')');
            }
            ActExpr::Exists(b) => {
                out.push_str("\\E i \\in S: ");
                b.render(d, out);
            }
            ActExpr::Forall(b) => {
                out.push_str("\\A i \\in S: ");
                b.render(d, out);
            }
            ActExpr::And(items) => render_join(items, " /\\ ", d, out),
            ActExpr::Or(items) => render_join(items, " \\/ ", d, out),
        }
    }
}

fn render_join(items: &[ActExpr], sep: &str, depth: u32, out: &mut String) {
    if items.is_empty() {
        out.push('x');
        return;
    }
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        it.render(depth, out);
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    expr: ActExpr,
    /// Adversarial tail appended after the structured expression, and also
    /// split on its own. Exercises unbalanced delimiters / stray bytes.
    raw_suffix: String,
}

const MAX_DEPTH: u32 = 12;

fuzz_target!(|input: FuzzInput| {
    let mut structured = String::new();
    input.expr.render(MAX_DEPTH, &mut structured);

    // 1. Structured input: totality (terminates + at least one subaction) and
    // no panic. A non-empty action body always decomposes to >= 1 subaction —
    // TLC keeps at least the whole body as one. We do NOT re-split the output:
    // this stage is a deliberately partial (non-maximal, non-idempotent)
    // decomposition finished at eval time, so re-feeding its pieces exercises a
    // path no caller takes, and successor-set faithfulness is the diff-gate's
    // job against TLC, not something checkable here (see module docs).
    let pieces = split_action_body_disjuncts(&structured);
    assert!(
        !pieces.is_empty(),
        "splitter returned no subactions for non-empty body {structured:?}"
    );

    // 2. Structured input with an adversarial tail: panic check only.
    let mut with_suffix = structured.clone();
    with_suffix.push_str(&input.raw_suffix);
    let _ = split_action_body_disjuncts(&with_suffix);

    // 3. Raw suffix alone: panic check only.
    let _ = split_action_body_disjuncts(&input.raw_suffix);
});
