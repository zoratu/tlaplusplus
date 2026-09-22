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
//! Primary oracle: no panic / no abort (the #187 failure mode).
//!
//! Secondary oracle (fixpoint convergence): the splitter separates top-level
//! `\/` disjuncts. It is NOT strictly idempotent — an existential whose body is
//! a bare disjunction, `\E i \in S: A \/ B`, is left intact when it appears as
//! one disjunct of a larger expression but is distributed into `\E i: A`,
//! `\E i: B` when that piece is re-split on its own. That distribution is a
//! sound rewrite (`\E i: (A \/ B) <=> (\E i: A) \/ (\E i: B)`; the body has no
//! shared conjunct to drop, so it is not the #187 conjunct-dropping shape), so
//! demanding one-step stability is too strong (a fuzz run falsified it). The
//! invariant that DOES hold, and is worth gating, is that repeated splitting
//! CONVERGES to a fixpoint: it never oscillates, never grows without bound, and
//! never collapses to nothing. This still catches "didn't split", "not stable
//! on its own output past a bounded rewrite", and "dropped every piece"
//! regressions. It is only asserted for the fully-structured input (no raw
//! suffix), where the grammar guarantees balanced delimiters; raw-suffixed
//! inputs are panic-checked only, since a truncated/garbled tail can
//! legitimately defeat top-level-delimiter scanning.

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

    // 1. Structured input: panic check + fixpoint-convergence check.
    // Re-split every piece until the set stops changing. A sound one-shot
    // rewrite (e.g. distributing `\E i: A \/ B`) is allowed; oscillation,
    // unbounded growth, and total collapse are not.
    let mut frontier = split_action_body_disjuncts(&structured);
    assert!(
        !frontier.is_empty(),
        "splitter returned no pieces for {structured:?}"
    );
    // MAX_DEPTH bounds disjunction nesting, so a handful of passes suffices;
    // the cap only fires on a genuine non-converging (oscillating/growing) bug.
    const FIXPOINT_CAP: u32 = 24;
    let mut converged = false;
    for _ in 0..FIXPOINT_CAP {
        let mut next = Vec::new();
        for piece in &frontier {
            next.extend(split_action_body_disjuncts(piece));
        }
        assert!(
            !next.is_empty(),
            "splitter collapsed a non-empty piece set to nothing; \
             frontier {frontier:?} from {structured:?}"
        );
        if next == frontier {
            converged = true;
            break;
        }
        frontier = next;
    }
    assert!(
        converged,
        "splitter did not reach a fixpoint within {FIXPOINT_CAP} passes \
         (oscillation or unbounded growth) for {structured:?}; \
         last frontier {frontier:?}"
    );

    // 2. Structured input with an adversarial tail: panic check only.
    let mut with_suffix = structured.clone();
    with_suffix.push_str(&input.raw_suffix);
    let _ = split_action_body_disjuncts(&with_suffix);

    // 3. Raw suffix alone: panic check only.
    let _ = split_action_body_disjuncts(&input.raw_suffix);
});
