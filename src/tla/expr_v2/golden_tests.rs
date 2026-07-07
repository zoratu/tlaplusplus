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
    let s = shape("P => LET X == v IN /\\ Q\n              /\\ R");
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
    let src = "\
/\\ \\A i \\in I, j \\in J : \\/ A
                             \\/ B
/\\ C";
    let s = shape(src);
    // AND[ Forall(i : Forall(j : OR[A, B])), C ]
    assert!(s.starts_with("AND[Forall(i\\in Atom(\"I\") : Forall(j\\in Atom(\"J\") : OR[Atom(\"A\"), Atom(\"B\")])"), "got: {s}");
    assert!(s.ends_with(", Atom(\"C\")]"), "sibling C not separate: {s}");
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
    assert!(
        s.contains("Implies(Neq(Atom(\"sb\"), Atom(\"NoBlock\")), Let("),
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
