//! Golden tests for `expr_v2`: verify the parser produces the CORRECT
//! structure on the exact shapes the old string-based parser gets wrong, plus
//! regression tests that genuine siblings still split and `=>` is right-assoc.

use super::parse_ast;

fn shape(src: &str) -> String {
    match parse_ast(src) {
        Ok(e) => e.shape(),
        Err(err) => panic!("parse failed for {src:?}: {err}"),
    }
}

// 1. `P => LET X == v IN /\ Q /\ R` — the consequent is the WHOLE LET (with its
//    junction), NOT `And([Implies(P, LET..Q), R])`.
#[test]
fn implies_let_junction_consequent() {
    // `/\ Q` bullet is at column 19 (after "P => LET X == v IN "); the
    // continuation `/\ R` must align to the SAME column (TLA+ layout rule).
    let s = shape("P => LET X == v IN /\\ Q\n                   /\\ R");
    // Implies(P, Let([X==v] IN AND[Q, R]))
    assert!(
        s.starts_with("Implies(Atom(\"P\"), Let("),
        "got: {s}"
    );
    assert!(s.contains("AND[Atom(\"Q\"), Atom(\"R\")]"), "got: {s}");
    // Must NOT be a top-level AND (that would be the OLD-parser bug).
    assert!(!s.starts_with("AND["), "consequent wrongly split: {s}");
}

// 2. `\A x \in S : /\ A /\ B` — body is the junction.
#[test]
fn forall_body_is_junction() {
    let s = shape("\\A x \\in S : /\\ A\n             /\\ B");
    assert_eq!(s, "Forall(x\\in Atom(\"S\") : AND[Atom(\"A\"), Atom(\"B\")])");
}

// 3. `/\ P => LET X==v IN /\ Q /\ R  /\ S` — item1 = P => (LET.. /\ Q /\ R),
//    item2 = S.
#[test]
fn junction_item_implies_let_then_sibling() {
    // Layout: two top-level conjuncts at col 0. Item 1's consequent LET junction
    // is indented deeper so it stays inside item 1.
    let src = "\
/\\ P => LET X == v IN /\\ Q
                      /\\ R
/\\ S";
    let s = shape(src);
    // AND[ Implies(P, Let([X==v] IN AND[Q, R])), S ]
    assert!(s.starts_with("AND[Implies(Atom(\"P\"), Let("), "got: {s}");
    assert!(s.ends_with(", Atom(\"S\")]"), "sibling S not separate: {s}");
    assert!(s.contains("AND[Atom(\"Q\"), Atom(\"R\")]"), "got: {s}");
}

// 4a. Nested Paxos-Phase2a-like: `/\ \A i,j : \/ A \/ B  /\ C`.
#[test]
fn paxos_like_nested() {
    // `\/ A` bullet is at column 25 (after "/\ \A i \in I, j \in J : "); the
    // continuation `\/ B` aligns to column 25; the sibling `/\ C` is at col 0.
    let src = "\
/\\ \\A i \\in I, j \\in J : \\/ A
                         \\/ B
/\\ C";
    let s = shape(src);
    // AST: AND[ Forall(i\in I; j\in J : OR[A, B]), C ]
    // (multi-bound is a single Quant node in the AST; lowering nests it.)
    assert!(
        s.starts_with(
            "AND[Forall(i\\in Atom(\"I\"); j\\in Atom(\"J\") : OR[Atom(\"A\"), Atom(\"B\")])"
        ),
        "got: {s}"
    );
    assert!(s.ends_with(", Atom(\"C\")]"), "sibling C not separate: {s}");
}

// Lowering the multi-bound quantifier must nest single-var Foralls.
#[test]
fn multibound_quant_lowers_nested() {
    use super::parse_and_lower;
    use crate::tla::compiled_expr::CompiledExpr;
    let c = parse_and_lower("\\A i \\in I, j \\in J : P").unwrap();
    match c {
        CompiledExpr::Forall { var, body, .. } => {
            assert_eq!(var, "i");
            assert!(matches!(*body, CompiledExpr::Forall { .. }), "not nested");
        }
        other => panic!("expected Forall, got {other:?}"),
    }
}

// 4b. 3-level checkpoint-coordination-like: `P => /\ (R => /\ X /\ Y)`.
#[test]
fn checkpoint_three_level() {
    let src = "\
P => /\\ Base
     /\\ (R => /\\ X
              /\\ Y)";
    let s = shape(src);
    // Implies(P, AND[Base, (Implies(R, AND[X, Y]))])
    assert!(s.starts_with("Implies(Atom(\"P\"), AND[Atom(\"Base\"), "), "got: {s}");
    assert!(s.contains("Implies(Atom(\"R\"), AND[Atom(\"X\"), Atom(\"Y\")])"), "got: {s}");
}

// 5. NanoBlockchain shape:
//    `\A h : LET sb == f[h] IN /\ sb # NoBlock => LET pk == g(h) IN /\ Validate(sb.sig, pk, h)`
#[test]
fn nano_blockchain_shape() {
    let src = "\
\\A h \\in H : LET sb == f[h] IN /\\ sb # NoBlock => LET pk == g(h) IN /\\ Validate(sb.sig, pk, h)";
    let s = shape(src);
    // Forall(h : Let([sb==f[h]] IN <single-item junction collapses to the =>>))
    // The inner `/\` has a single item so it collapses; the => consequent is the
    // inner LET.
    assert!(s.starts_with("Forall(h\\in Atom(\"H\") : Let("), "got: {s}");
    assert!(s.contains("Implies("), "missing implication: {s}");
    assert!(s.contains("Let("), "missing inner let: {s}");
    // The inner `=> LET pk == g(h) IN ...` must attach the inner LET as the
    // consequent (structural body extension), not split it off.
    // Leaf comparison `sb # NoBlock` stays inside a single Atom (v1 lowers it);
    // v2 owns only the `=>` structure, attaching the inner LET as the consequent.
    assert!(
        s.contains("Implies(Atom(\"sb # NoBlock\"), Let("),
        "consequent not the inner LET: {s}"
    );
}

// --- Regression: genuine siblings still split ---
#[test]
fn genuine_siblings_split() {
    let s = shape("/\\ A\n/\\ B");
    assert_eq!(s, "AND[Atom(\"A\"), Atom(\"B\")]");
}

#[test]
fn three_siblings_split() {
    let s = shape("/\\ A\n/\\ B\n/\\ C");
    assert_eq!(s, "AND[Atom(\"A\"), Atom(\"B\"), Atom(\"C\")]");
}

// --- Regression: => is right-associative ---
#[test]
fn implies_right_assoc() {
    let s = shape("A => B => C");
    assert_eq!(s, "Implies(Atom(\"A\"), Implies(Atom(\"B\"), Atom(\"C\")))");
}

// A mid-line prefix bullet: `Foo == /\ A /\ B` (the body of a definition, no
// requirement that the bullet is first-on-line).
#[test]
fn mid_line_prefix_bullet() {
    // `Foo == /\ A /\ B`: the definition body starts with a prefix bullet mid-
    // line (not first-token-on-line). Inline continuation bullets form an infix
    // conjunction list.
    let s = shape("/\\ A /\\ B");
    assert_eq!(s, "AND[Atom(\"A\"), Atom(\"B\")]");
}

// Simple sanity: parenthesized junction.
#[test]
fn paren_junction() {
    let s = shape("(/\\ A\n /\\ B)");
    assert_eq!(s, "(AND[Atom(\"A\"), Atom(\"B\")])");
}

// ===================== Fix #1: entry-stop =====================

use super::parse_ast;

// `/\ A =>\n/\ B\n/\ C`: the `=>` consequent must STOP at the second `/\`
// (same fence column). Item 1 (`A =>`) then has an EMPTY consequent → v2
// rejects (→ v1 fallback), it does NOT swallow B and C into the consequent.
#[test]
fn entry_stop_empty_consequent_rejects() {
    let src = "/\\ A =>\n/\\ B\n/\\ C";
    let r = parse_ast(src);
    assert!(
        r.is_err(),
        "empty `=>` consequent at the fence must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// An empty junction item `/\ \n /\ B` must NOT collapse to B — it rejects.
#[test]
fn entry_stop_empty_item_rejects() {
    let r = parse_ast("/\\\n/\\ B");
    assert!(
        r.is_err(),
        "empty junction item must reject, got: {:?}",
        r.map(|e| e.shape())
    );
}

// `/\ P => /\ Q\n         /\ R` where the `=>` consequent `/\ Q` is DEEPER
// (col > 0) attaches to the `=>`; a `/\ R` sibling at the OUTER column would
// NOT. Here both Q and R are deep, so both attach to the consequent junction.
#[test]
fn entry_stop_deeper_consequent_attaches() {
    // P => (deep /\ Q /\ R), all inside item 1's `=>`.
    let src = "\
/\\ P => /\\ Q
         /\\ R";
    let s = shape(src);
    // Single top-level item `P => AND[Q,R]` collapses (1-item junction).
    assert_eq!(s, "Implies(Atom(\"P\"), AND[Atom(\"Q\"), Atom(\"R\")])");
}

// The sibling-vs-consequent split: `/\ P => /\ Q  (deep)` then `/\ R` at the
// OUTER column 0 is a SIBLING, not part of the `=>` consequent.
#[test]
fn entry_stop_outer_sibling_not_swallowed() {
    let src = "\
/\\ P => /\\ Q
/\\ R";
    let s = shape(src);
    // AND[ Implies(P, Q) , R ]  — R is a sibling at col 0.
    assert_eq!(
        s,
        "AND[Implies(Atom(\"P\"), Atom(\"Q\")), Atom(\"R\")]",
        "outer sibling R wrongly attached: {s}"
    );
}

// ===================== Fix #2: IF keeps caller fence =====================

// An IF inside a junction: a sibling bullet after the ELSE branch must NOT be
// swallowed into the else-branch.
#[test]
fn if_in_junction_sibling_not_swallowed() {
    let src = "\
/\\ IF c THEN t ELSE e
/\\ Sib";
    let s = shape(src);
    assert_eq!(
        s,
        "AND[If(Atom(\"c\"), Atom(\"t\"), Atom(\"e\")), Atom(\"Sib\")]",
        "sibling after IF/ELSE wrongly swallowed: {s}"
    );
}

// ===================== Fix #3: parameterized/function LET =====================

// A parameterized operator LET must reject (→ v1 fallback), never lower to a
// binding that drops the params.
#[test]
fn param_let_rejects() {
    let r = parse_ast("LET Op(x) == x + 1 IN Op(2)");
    assert!(
        r.is_err(),
        "parameterized LET must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// A function-def LET must reject too.
#[test]
fn func_let_rejects() {
    let r = parse_ast("LET f[x] == x + 1 IN f[2]");
    assert!(
        r.is_err(),
        "function-def LET must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// A plain (non-parameterized) LET still parses structurally.
#[test]
fn plain_let_still_parses() {
    let s = shape("LET x == 1 IN x + 1");
    assert_eq!(s, "Let([x==Atom(\"1\")] IN Atom(\"x + 1\"))");
}

// ===================== Fix #4: comment stripping in atoms =====================

// A comment inside an atom run must NOT reach v1 in the Atom text.
#[test]
fn atom_strips_inline_block_comment() {
    let s = shape("x (* c *) + y");
    assert_eq!(s, "Atom(\"x  + y\")", "comment leaked into atom text: {s}");
}

#[test]
fn atom_strips_line_comment() {
    // `a + b \* trailing` — the line comment is dropped from the atom text.
    let s = shape("a + b \\* trailing");
    assert_eq!(s, "Atom(\"a + b\")", "line comment leaked: {s}");
}

// ===================== Fix #5: unterminated block comment =====================

#[test]
fn unterminated_block_comment_rejects() {
    let r = parse_ast("A (* unterminated");
    assert!(
        r.is_err(),
        "unterminated block comment must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}
