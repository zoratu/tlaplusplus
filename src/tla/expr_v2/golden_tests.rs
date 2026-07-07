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

/// Assert that v2 lowers `src` to EXACTLY the same `CompiledExpr` as v1's
/// `compile_expr_v1` — the shadow-compare fidelity bar for a construct we lower
/// structurally. (v2 must both PARSE it and lower it identically to v1.)
fn assert_lower_matches_v1(src: &str) {
    use crate::tla::compiled_expr::compile_expr_v1;
    let v1 = compile_expr_v1(src);
    let v2 = super::parse_and_lower(src)
        .unwrap_or_else(|e| panic!("v2 failed to parse {src:?}: {e}"));
    assert_eq!(
        format!("{v1:?}"),
        format!("{v2:?}"),
        "v2 lowering diverged from v1 for {src:?}"
    );
}

/// Assert v2 REJECTS `src` (→ v1 fallback). Used for shapes v1 lowers to
/// `Unparsed`/temporal forms that v2 deliberately does not model.
fn assert_v2_rejects(src: &str) {
    let r = parse_ast(src);
    assert!(
        r.is_err(),
        "expected v2 to reject {src:?} (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
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

// --- Phase 0.6 Fix 1: entry-stop bypass via `~` and infix-bullet RHS ---

// `/\ A => ~\n/\ B\n/\ C`: after the `~`, the operand parse reaches a STOP
// bullet (`/\ B` on a new line at the fence column). The Phase-0.5 fix only
// guarded `parse_expr`; the `~` operand path bypassed it and `~` swallowed
// `/\ B` as its operand. The single-point `parse_prefix` guard now rejects the
// stop bullet on ALL paths → v1 fallback (NOT `~` swallowing B/C).
#[test]
fn entry_stop_not_operand_swallow() {
    let src = "/\\ A => ~\n/\\ B\n/\\ C";
    let r = parse_ast(src);
    assert!(
        r.is_err(),
        "`~` must NOT swallow the sibling `/\\ B` at the fence (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// `/\ A /\\n/\ B\n/\ C`: the trailing INFIX `/\` (same line, after A) has its
// RHS parsed via `parse_bin`, which reaches `parse_prefix` on the next token —
// `/\ B` on a new line at the fence column, a STOP bullet. The infix RHS must
// NOT consume that sibling bullet as a fresh junction-start. Rejects → v1
// fallback.
#[test]
fn entry_stop_not_infix_rhs_swallow() {
    let src = "/\\ A /\\\n/\\ B\n/\\ C";
    let r = parse_ast(src);
    assert!(
        r.is_err(),
        "infix `/\\` RHS must NOT swallow the sibling `/\\ B` at the fence \
         (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// Companion: an inline-`/\` whose bullets are all on ONE line (a genuine infix
// conjunction, no following STOP sibling) still parses — the guard rejects only
// a stop bullet on a NEW line at/shallower-than the fence, not legitimate infix
// continuations. This confirms the Fix-1 guard is not over-rejecting.
#[test]
fn entry_stop_inline_infix_still_parses() {
    let s = shape("/\\ A /\\ B /\\ C");
    assert_eq!(s, "AND[Atom(\"A\"), Atom(\"B\"), Atom(\"C\")]");
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

// --- Phase 0.6 Fix 2: EMPTY param / func-arg lists must still reject ---

// `LET Op() == 1 IN Op()`: an EMPTY param list `()` collects zero names but is
// still a syntactically parameterized def. Keying the reject off collected-name
// counts would leak the binder past the reject into `lower_let` (a release-mode
// only `debug_assert`). The `saw_param_list` flag catches it → v1 fallback.
#[test]
fn empty_param_let_rejects() {
    let r = parse_ast("LET Op() == 1 IN Op()");
    assert!(
        r.is_err(),
        "empty-param-list LET `Op()` must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// `LET f[] == 1 IN f[0]`: an EMPTY func-arg list `[]` — same leak risk, caught
// by `saw_func_arg_list`.
#[test]
fn empty_func_arg_let_rejects() {
    let r = parse_ast("LET f[] == 1 IN f[0]");
    assert!(
        r.is_err(),
        "empty-func-arg-list LET `f[]` must reject (→ v1 fallback), got: {:?}",
        r.map(|e| e.shape())
    );
}

// ===================== Fix #4: comment stripping in atoms =====================

// A comment inside an atom run must NOT reach v1 in the Atom text.
#[test]
fn atom_strips_inline_block_comment() {
    let s = shape("x (* c *) + y");
    // The comment body must be gone; leaf tokens `x`, `+`, `y` remain (exact
    // interior spacing is irrelevant — v1 re-tokenizes).
    assert!(!s.contains("(*") && !s.contains("*)") && !s.contains('c'),
        "comment leaked into atom text: {s}");
    assert!(s.starts_with("Atom(\"x") && s.contains("+ y"), "atom malformed: {s}");
}

// Phase 0.6 Fix 3: a block comment that spans NEWLINES must preserve the
// newline count in the stripped atom text (so line/column structure handed to
// v1 mirrors the source the lexer saw), not collapse to a single space.
#[test]
fn atom_preserves_newlines_across_block_comment() {
    let s = shape("a (* multi\nline\ncomment *) + b");
    // Comment body gone.
    assert!(
        !s.contains("multi") && !s.contains("comment") && !s.contains("(*"),
        "block comment leaked: {s}"
    );
    // The two interior newlines are preserved (rendered as `\n` in the AST
    // shape's Atom debug). Leaf tokens `a`, `+`, `b` remain.
    assert!(s.contains("a") && s.contains("+ b"), "atom malformed: {s}");
    assert!(
        s.matches("\\n").count() >= 2,
        "block-comment newlines not preserved (expected >=2 `\\n`): {s}"
    );
}

#[test]
fn atom_strips_line_comment() {
    // `a + b \* trailing` — the line comment is dropped from the atom text.
    let s = shape("a + b \\* trailing");
    assert!(!s.contains("trailing") && !s.contains("\\*"),
        "line comment leaked: {s}");
    assert!(s.contains("a + b"), "atom malformed: {s}");
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

// ===================== Phase 1: comment COLUMN preservation ===================

// A single-line block comment must be replaced by whitespace of the SAME width
// so a token AFTER it on the same line keeps its column (Codex carry-over).
#[test]
fn atom_block_comment_preserves_trailing_column() {
    use super::parser;
    let s = shape("A (* cc *) => B");
    assert!(s.starts_with("Implies(Atom(\"A\")"), "got: {s}");
    // A single-line comment is replaced by whitespace of EXACTLY its own width,
    // so the following token keeps its column. `(* comment *)` is 13 chars.
    let comment = "(* comment *)";
    let stripped = parser::strip_comments_for_test(&format!("xx{comment}yy"));
    let expected = format!("xx{}yy", " ".repeat(comment.chars().count()));
    assert_eq!(stripped, expected, "single-line width not preserved");
    // The next token's column is unchanged from the original.
    let orig = format!("xx{comment}yy");
    assert_eq!(
        stripped.find("yy").unwrap(),
        orig.find("yy").unwrap(),
        "column of trailing token shifted"
    );
    // Multi-line: interior newlines preserved AND the final line padded to the
    // width after the last newline so a token after `*)` keeps its column.
    let ml = parser::strip_comments_for_test("a (* c\nd *) + b");
    assert_eq!(ml.matches('\n').count(), 1, "interior newline not preserved");
    let last_line = ml.rsplit('\n').next().unwrap();
    // Original final comment-line is "d *)" (4 cols) then " + b"; the `+` must
    // sit at the same column as in the source line "d *) + b".
    assert_eq!(
        last_line.find('+'),
        "d *) + b".find('+'),
        "final-line column not preserved: {last_line:?}"
    );
}

// ===================== Phase 1: SETS =====================

#[test]
fn set_enum_basic() {
    assert_eq!(shape("{1, 2, 3}"), "SetEnum{Atom(\"1\"), Atom(\"2\"), Atom(\"3\")}");
    assert_lower_matches_v1("{1, 2, 3}");
}

#[test]
fn set_enum_empty() {
    assert_eq!(shape("{}"), "SetEnum{}");
    assert_lower_matches_v1("{}");
}

#[test]
fn set_filter_form() {
    assert_eq!(shape("{x \\in S : x > 0}"), "SetFilter(x\\in Atom(\"S\") : Atom(\"x > 0\"))");
    assert_lower_matches_v1("{x \\in S : x > 0}");
}

#[test]
fn set_map_form() {
    assert_eq!(shape("{x + 1 : x \\in S}"), "SetMap(Atom(\"x + 1\") : x\\in Atom(\"S\"))");
    assert_lower_matches_v1("{x + 1 : x \\in S}");
}

// A junction INSIDE a set element must be fenced by v2 (not flattened).
#[test]
fn set_element_junction_fenced() {
    let s = shape("{IF a THEN b ELSE c, d}");
    assert!(s.starts_with("SetEnum{If("), "got: {s}");
    assert!(s.ends_with(", Atom(\"d\")}"), "got: {s}");
}

// A set that is an OPERAND of a leaf operator must NOT be parsed structurally —
// it stays a single Atom for v1 (so v2 doesn't strand the `\union`).
#[test]
fn set_operand_of_leaf_stays_atom() {
    let s = shape("{1, 2} \\union {3}");
    assert!(s.starts_with("Atom("), "set operand of \\union should be atom: {s}");
    assert_lower_matches_v1("{1, 2} \\union {3}");
}

// ===================== Phase 1: TUPLES =====================

#[test]
fn tuple_basic() {
    assert_eq!(shape("<<a, b, c>>"), "Tuple<<Atom(\"a\"), Atom(\"b\"), Atom(\"c\")>>");
    assert_lower_matches_v1("<<a, b, c>>");
}

#[test]
fn tuple_empty() {
    assert_eq!(shape("<<>>"), "Tuple<<>>");
    assert_lower_matches_v1("<<>>");
}

#[test]
fn tuple_nested() {
    assert_lower_matches_v1("<<1, <<2, 3>>, 4>>");
}

// A tuple as a leaf operand stays an atom (with its commas intact).
#[test]
fn tuple_operand_stays_atom() {
    // `<<a, b>> = <<c>>` — the whole thing is one atom for v1.
    let s = shape("<<a, b>> = <<c>>");
    assert!(s.starts_with("Atom("), "got: {s}");
    assert!(s.contains(","), "tuple commas must survive in the atom: {s}");
    assert_lower_matches_v1("<<a, b>> = <<c>>");
}

// ===================== Phase 1: [...] family =====================

#[test]
fn record_literal_basic() {
    assert_eq!(
        shape("[a |-> 1, b |-> 2]"),
        "Rec[a|->Atom(\"1\"), b|->Atom(\"2\")]"
    );
    assert_lower_matches_v1("[a |-> 1, b |-> 2]");
}

#[test]
fn record_set_basic() {
    assert_eq!(shape("[a : S, b : T]"), "RecSet[a:Atom(\"S\"), b:Atom(\"T\")]");
    assert_lower_matches_v1("[a : S, b : T]");
}

#[test]
fn function_set_basic() {
    assert_eq!(shape("[S -> T]"), "FnSet[Atom(\"S\") -> Atom(\"T\")]");
    assert_lower_matches_v1("[S -> T]");
}

#[test]
fn function_construct_basic() {
    assert_eq!(
        shape("[x \\in S |-> x + 1]"),
        "FnCon[x\\in Atom(\"S\") |-> Atom(\"x + 1\")]"
    );
    assert_lower_matches_v1("[x \\in S |-> x + 1]");
}

// A junction inside a record VALUE is fenced by v2.
#[test]
fn record_value_junction_fenced() {
    let s = shape("[ok |-> /\\ A\n            /\\ B, n |-> 0]");
    assert!(s.starts_with("Rec[ok|->AND["), "got: {s}");
}

// EXCEPT is left to v1 (Err → fallback), never mis-lowered.
#[test]
fn except_rejects_to_v1() {
    assert_v2_rejects("[f EXCEPT ![k] = v]");
    assert_v2_rejects("[f EXCEPT !.field = w]");
}

// Action / stuttering box `[A]_v` is not modeled STRUCTURALLY by v2: the `]_`
// subscript makes the bracket non-standalone, so v2 absorbs the WHOLE thing as
// one Atom that v1 lowers (never a structural record/function-set mis-parse).
#[test]
fn action_box_stays_atom_for_v1() {
    // Whole-atom absorption → lowering matches v1 exactly (v1 owns temporal).
    assert_eq!(shape("[Next]_vars"), "Atom(\"[Next]_vars\")");
    assert_eq!(shape("[A]_v"), "Atom(\"[A]_v\")");
    assert_lower_matches_v1("[Next]_vars");
    assert_lower_matches_v1("[A]_v");
}

// A bracket that is an operand of a leaf op stays an atom.
#[test]
fn bracket_operand_stays_atom() {
    let s = shape("[a |-> 1].a");
    assert!(s.starts_with("Atom("), "record access should stay atom: {s}");
    assert_lower_matches_v1("[a |-> 1].a");
}

// ===================== Phase 1: CASE =====================

#[test]
fn case_basic() {
    assert_lower_matches_v1("CASE p -> e [] q -> f");
}

#[test]
fn case_with_other() {
    assert_lower_matches_v1("CASE p -> e [] q -> f [] OTHER -> g");
}

// A junction inside a CASE arm result is fenced by v2.
#[test]
fn case_arm_junction_fenced() {
    let s = shape("CASE p -> /\\ A\n          /\\ B [] OTHER -> c");
    assert!(s.contains("-> AND["), "case arm junction not fenced: {s}");
}

// ===================== Phase 1.5: CASE wrong-parse fixes =====================

// Bug #1: a CASE arm's RESULT is a full expression; a depth-0 `=>` inside the
// result belongs to the ARM, not the CASE boundary. `CASE p -> a => b [] OTHER
// -> c` parses as `Case[p -> (a => b) [] OTHER -> c]`, NOT `(CASE p -> a) => ...`.
#[test]
fn case_arm_result_implication_stays_in_arm() {
    let s = shape("CASE p -> a => b [] OTHER -> c");
    // Top-level is a Case (not an Implies), and the arm result is `a => b`.
    assert!(s.starts_with("Case["), "CASE sliced at =>: {s}");
    assert!(
        s.contains("Implies(Atom(\"a\"), Atom(\"b\"))"),
        "arm result implication lost: {s}"
    );
    // And it lowers EXACTLY as v1 does (v1 matches TLC).
    assert_lower_matches_v1("CASE p -> a => b [] OTHER -> c");
}

// Bug #1 (guard variant): a depth-0 `=>` inside a GUARD belongs to the arm.
#[test]
fn case_arm_guard_implication_stays_in_arm() {
    let s = shape("CASE p => q -> a [] OTHER -> c");
    assert!(s.starts_with("Case["), "CASE sliced at => in guard: {s}");
    assert!(
        s.contains("Implies(Atom(\"p\"), Atom(\"q\"))"),
        "arm guard implication lost: {s}"
    );
    assert_lower_matches_v1("CASE p => q -> a [] OTHER -> c");
}

// Bug #1 (<=> variant): `<=>` inside an arm result stays in the arm.
#[test]
fn case_arm_result_iff_stays_in_arm() {
    let s = shape("CASE p -> a <=> b [] OTHER -> c");
    assert!(s.starts_with("Case["), "CASE sliced at <=>: {s}");
    assert_lower_matches_v1("CASE p -> a <=> b [] OTHER -> c");
}

// Bug #2: a CASE that is a LEAF OPERAND of a leaf `=` must NOT be sliced at the
// `=>` inside its arm. `x = CASE p -> a => b [] OTHER -> c` falls back to v1
// (which parses the whole `x = CASE ...` correctly) rather than emitting a
// partial Atom + outer Implies.
#[test]
fn case_operand_sliced_by_implies_falls_back() {
    assert_v2_rejects("x = CASE p -> a => b [] OTHER -> c");
}

// Bug #2 (<=> variant).
#[test]
fn case_operand_sliced_by_iff_falls_back() {
    assert_v2_rejects("x = CASE p -> a <=> b [] OTHER -> c");
}

// Bug #3a: an UNPARENTHESIZED nested CASE (whose inner `[]` would be stolen by
// the outer CASE's separator scan) falls back to v1.
#[test]
fn case_nested_unparenthesized_falls_back() {
    assert_v2_rejects("CASE p -> CASE q -> a [] r -> b [] OTHER -> c");
}

// Bug #3a (control): a PARENTHESIZED nested CASE is at depth>0 (absorbed as an
// atom sub-parse) and stays fine — it lowers identically to v1.
#[test]
fn case_nested_parenthesized_ok() {
    assert_lower_matches_v1("CASE p -> (CASE q -> a [] OTHER -> b) [] OTHER -> c");
}

// Bug #3b: a CASE with a DUPLICATE OTHER falls back (TLC: OTHER unique).
#[test]
fn case_duplicate_other_falls_back() {
    assert_v2_rejects("CASE p -> a [] OTHER -> b [] OTHER -> c");
}

// Bug #3b: a CASE with a NON-FINAL OTHER falls back (TLC: OTHER must be last).
#[test]
fn case_non_final_other_falls_back() {
    assert_v2_rejects("CASE OTHER -> a [] p -> b");
}

// Bug #4: a bracket interior containing a top-level CASE must NOT be classified
// as a function set `[D -> R]` (the `->` are arm arrows). Falls back to v1.
#[test]
fn bracket_with_case_falls_back() {
    assert_v2_rejects("[CASE p -> a]");
}

// ===== TLC-verified goldens: =>/<=> grouping + CASE-arm implication =====
// These pin v2's lowering against v1 (which matches TLC v2.19) for the exact
// operator-precedence shapes flagged in review.

// `=>` binds TIGHTER than `<=>`: `a => b <=> c` == `(a => b) <=> c`.
#[test]
fn implies_tighter_than_iff_left() {
    assert_lower_matches_v1("a => b <=> c");
}

// `a <=> b => c` == `a <=> (b => c)`.
#[test]
fn iff_looser_than_implies_right() {
    assert_lower_matches_v1("a <=> b => c");
}

// `=>` is right-associative: `a => b => c` == `a => (b => c)`. (Lowering-match
// counterpart to `implies_right_assoc`, which checks the shape.)
#[test]
fn implies_right_assoc_lowers_like_v1() {
    assert_lower_matches_v1("a => b => c");
}

// Full CASE with an arm-result implication + OTHER, lowered exactly as v1.
#[test]
fn case_arm_implication_full_matches_v1() {
    assert_lower_matches_v1("CASE p -> a => b [] q -> c [] OTHER -> d");
}

// ===================== Phase 1: CHOOSE =====================

#[test]
fn choose_bounded() {
    assert_eq!(shape("CHOOSE x \\in S : x > 0"), "Choose(x\\in Atom(\"S\") : Atom(\"x > 0\"))");
    assert_lower_matches_v1("CHOOSE x \\in S : x > 0");
}

// Unbounded CHOOSE is Unparsed in v1 → v2 falls back rather than mis-lower.
#[test]
fn choose_unbounded_rejects() {
    assert_v2_rejects("CHOOSE x : P(x)");
}

// Tuple-binder CHOOSE is Unparsed in v1 → v2 falls back.
#[test]
fn choose_tuple_binder_rejects() {
    assert_v2_rejects("CHOOSE <<a, b>> \\in S : a > b");
}
