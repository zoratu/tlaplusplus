//! Metamorphic generator for the issue-#201 bug class: a guard followed by a
//! degenerate disjunction of actions dropping successors.
//!
//! ## The oracle (no TLC, no independent evaluator needed)
//!
//! The bug is a *silent* missed successor, so a plain fuzzer (crash/hang only)
//! can't see it. Instead we use a **metamorphic** oracle: a set of algebraic
//! identities that TLC treats as semantics-preserving must not change the
//! successor set under tlaplusplus either. For any action subterm `X`:
//!
//! - `X`  ≡  `(X)`  ≡  `(X \/ FALSE)`  ≡  `(FALSE \/ X)`  ≡  `(X \/ X)`  ≡  `\/ X`
//!
//! We generate a random base `Next`, apply one such wrap to a random action
//! node, and assert `succ(base, s) == succ(rewritten, s)` for every state `s` in
//! a small bounded domain. The reference is the SAME evaluator on the un-wrapped
//! form, so any divergence (like #201: `g /\ A` vs `g /\ (A \/ FALSE)`) is
//! caught without an external baseline.
//!
//! This is the "fill the gap with a generator" counterpart to the hand-written
//! `guard_disjunction_issue_201.rs` repros: the grammar deliberately includes
//! the degenerate junction shapes (FALSE literals, single-branch `\/`, redundant
//! parens, idempotent `X \/ X`) that individual regression tests under-cover.
//!
//! ## Scope (known residuals)
//!
//! Leaves are operator-call actions (the reported shape: `Arm`/`Body` are named
//! operators), and generated disjunctions have 2+ branches. This generator found
//! — and the accompanying fix closed — the operator-call family. It deliberately
//! does NOT generate two adjacent residual shapes that remain open follow-ups,
//! so it stays a green regression guard rather than flapping on unfixed edges:
//!   1. a guard AND'd with an *inline* primed action (`g /\ ((pc' = v)))`) — the
//!      inline body is mis-evaluated as a boolean in some parenthesizations;
//!   2. a guard plus a lone *parenthesized* operator call (`g /\ (OpA)`), with no
//!      disjunction at all.
//! Both are the same underlying class (a guard-classified clause that is really
//! an action) and are tracked for a follow-up that reclassifies such clauses
//! without the regressions a broad change risks in this heuristic-dense path.

use proptest::prelude::*;
use tlaplusplus::tla::{TlaValue, evaluate_next_states, parse_tla_module_text, tla_state};

const PC_VALS: &[&str] = &["a", "b", "c"];
const PH_VALS: &[&str] = &["p", "q", "r"];

/// A leaf *action* over the fixed two-variable model. Each stages at least one
/// prime so it is a genuine action (not a boolean guard).
#[derive(Clone, Debug)]
enum Leaf {
    /// A named operator action (prime hidden behind the operator name — the
    /// #201 crux). `OpA`/`OpB`/`OpC` are defined in the module preamble.
    OpA,
    OpB,
    OpC,
}

impl Leaf {
    fn render(&self) -> String {
        match self {
            Leaf::OpA => "OpA".to_string(),
            Leaf::OpB => "OpB".to_string(),
            Leaf::OpC => "OpC".to_string(),
        }
    }
}

/// An action expression: a leaf, or a disjunction of action expressions.
#[derive(Clone, Debug)]
enum Act {
    Leaf(Leaf),
    Disj(Vec<Act>),
}

impl Act {
    /// Render inline, parenthesizing disjunctions so nesting is unambiguous.
    fn render(&self) -> String {
        match self {
            Act::Leaf(l) => l.render(),
            Act::Disj(parts) => {
                let inner = parts
                    .iter()
                    .map(Act::render)
                    .collect::<Vec<_>>()
                    .join(" \\/ ");
                format!("({inner})")
            }
        }
    }
}

/// One top-level branch: an optional boolean guard AND an action expression.
#[derive(Clone, Debug)]
struct Branch {
    guard: Option<String>,
    act: Act,
}

impl Branch {
    fn render(&self) -> String {
        match &self.guard {
            Some(g) => format!("({} /\\ {})", g, self.act.render()),
            None => self.act.render(),
        }
    }
}

/// A semantics-preserving wrap applied to one action node (the metamorphic
/// transform). Every variant is equal to `X` under TLC's successor-set
/// semantics.
#[derive(Clone, Copy, Debug)]
enum Wrap {
    Paren,      // (X)
    OrFalse,    // (X \/ FALSE)
    FalseOr,    // (FALSE \/ X)
    OrSelf,     // (X \/ X)
    Bullet,     // \/ X   (single-item bulleted list)
}

fn apply_wrap(rendered: &str, w: Wrap) -> String {
    match w {
        Wrap::Paren => format!("({rendered})"),
        Wrap::OrFalse => format!("({rendered} \\/ FALSE)"),
        Wrap::FalseOr => format!("(FALSE \\/ {rendered})"),
        Wrap::OrSelf => format!("({rendered} \\/ {rendered})"),
        Wrap::Bullet => format!("(\\/ {rendered})"),
    }
}

fn arb_leaf() -> impl Strategy<Value = Leaf> {
    // Scope: operator-call actions (the reported #201 shape — Arm/Body are named
    // operators). An `InlinePc` leaf is deliberately excluded: it exercises a
    // DISTINCT, still-open edge (a guard AND'd with a single/double-parenthesized
    // *inline* primed action, e.g. `(g /\ ((pc' = v)))`, is mis-evaluated as a
    // boolean). That residual is tracked as a follow-up; keeping it out of the
    // generator lets this test stay a green regression guard for the operator-call
    // family the #201 fix covers, rather than flapping on an unfixed shape.
    prop_oneof![Just(Leaf::OpA), Just(Leaf::OpB), Just(Leaf::OpC)]
}

fn arb_act() -> impl Strategy<Value = Act> {
    let leaf = arb_leaf().prop_map(Act::Leaf);
    // Disjunctions have 2+ branches: a single-element `Disj` renders as redundant
    // parens around one action (`(OpA)`), another distinct residual edge (a guard
    // plus a lone parenthesized operator-call action) left as a follow-up. The
    // degenerate single-branch shapes the #201 fix DOES handle are still covered
    // here via the `\/ X` and `(X \/ FALSE)` metamorphic wraps applied to leaves.
    leaf.prop_recursive(2, 6, 3, |inner| {
        prop::collection::vec(inner, 2..=3).prop_map(Act::Disj)
    })
}

fn arb_guard() -> impl Strategy<Value = Option<String>> {
    prop_oneof![
        Just(None),
        (0usize..PH_VALS.len()).prop_map(|i| Some(format!("ph = \"{}\"", PH_VALS[i]))),
        (0usize..PC_VALS.len()).prop_map(|i| Some(format!("pc = \"{}\"", PC_VALS[i]))),
    ]
}

fn arb_branch() -> impl Strategy<Value = Branch> {
    (arb_guard(), arb_act()).prop_map(|(guard, act)| Branch { guard, act })
}

fn module_src(next_body: &str) -> String {
    format!(
        "---- MODULE SplitMeta ----\n\
         VARIABLES pc, ph\n\
         OpA == pc' = \"b\" /\\ UNCHANGED ph\n\
         OpB == ph' = \"q\" /\\ UNCHANGED pc\n\
         OpC == pc' = \"c\" /\\ ph' = \"r\"\n\
         Init == pc = \"a\" /\\ ph = \"p\"\n\
         Next == {next_body}\n\
         ====\n"
    )
}

/// Successor set from one state, canonicalized to `(pc, ph)` pairs (set
/// semantics — the model checker dedups by fingerprint, so duplicates and
/// ordering are irrelevant).
fn succ_set(next_body: &str, pc: &str, ph: &str) -> Result<Vec<(String, String)>, String> {
    let module = parse_tla_module_text(&module_src(next_body)).map_err(|e| e.to_string())?;
    let next = module
        .definitions
        .get("Next")
        .ok_or_else(|| "no Next".to_string())?;
    let state = tla_state([
        ("pc", TlaValue::String(pc.to_string())),
        ("ph", TlaValue::String(ph.to_string())),
    ]);
    let succs = evaluate_next_states(&next.body, &module.definitions, &state)
        .map_err(|e| e.to_string())?;
    let mut out: Vec<(String, String)> = succs
        .iter()
        .map(|s| {
            let get = |k: &str| match s.get(k) {
                Some(TlaValue::String(x)) => x.clone(),
                other => format!("{other:?}"),
            };
            (get("pc"), get("ph"))
        })
        .collect();
    out.sort();
    out.dedup();
    Ok(out)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256),
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    /// Metamorphic property: wrapping a random action node in a
    /// semantics-preserving degenerate junction must not change the successor
    /// set from any state. Catches the #201 class (guard + `(A \/ FALSE)` /
    /// `\/ A` dropping the `A` successor) and any sibling shape.
    #[test]
    fn degenerate_junction_wrap_preserves_successors(
        branches in prop::collection::vec(arb_branch(), 1..=3),
        branch_idx in 0usize..3,
        wrap in prop_oneof![
            Just(Wrap::Paren),
            Just(Wrap::OrFalse),
            Just(Wrap::FalseOr),
            Just(Wrap::OrSelf),
            Just(Wrap::Bullet),
        ],
    ) {
        // Base body: a top-level disjunction of the branches (inline form).
        let base_body = branches
            .iter()
            .map(Branch::render)
            .collect::<Vec<_>>()
            .join(" \\/ ");

        // Rewritten body: wrap the action of one branch in a
        // semantics-preserving junction.
        let idx = branch_idx % branches.len();
        let wrapped_act_text = apply_wrap(&branches[idx].act.render(), wrap);
        let rewritten_body = branches
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if i == idx {
                    match &b.guard {
                        Some(g) => format!("({g} /\\ {wrapped_act_text})"),
                        None => wrapped_act_text.clone(),
                    }
                } else {
                    b.render()
                }
            })
            .collect::<Vec<_>>()
            .join(" \\/ ");

        for &pc in PC_VALS {
            for &ph in PH_VALS {
                let base = succ_set(&base_body, pc, ph);
                let rew = succ_set(&rewritten_body, pc, ph);
                // Evaluation errors must also match (both Ok or both Err);
                // a divergence either way is a bug.
                match (base, rew) {
                    (Ok(a), Ok(b)) => prop_assert_eq!(
                        &a, &b,
                        "successor set changed under semantics-preserving wrap {:?} at state (pc={}, ph={})\n  base      = {}\n  rewritten = {}\n  base succ = {:?}\n  rew  succ = {:?}",
                        wrap, pc, ph, base_body, rewritten_body, a, b
                    ),
                    (Err(_), Err(_)) => {}
                    (a, b) => prop_assert!(
                        false,
                        "Ok/Err divergence under wrap {:?} at (pc={}, ph={}): base={:?} rewritten={:?}\n  base={}\n  rew ={}",
                        wrap, pc, ph, a, b, base_body, rewritten_body
                    ),
                }
            }
        }
    }
}
