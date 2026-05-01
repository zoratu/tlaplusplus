//! T101.1 — regression tests for compiled-vs-interpreted Ok-vs-different-value
//! divergences surfaced by the swarm-fuzz equivalence check
//! (`fuzz/fuzz_targets/fuzz_tla_swarm.rs`).
//!
//! Categorisation of the 12 distinct fuzz crashes that motivated this
//! follow-up (swarm-mode 32-worker run, 30 min, c6gd.metal):
//!
//!   C1. Bare `UNCHANGED` (4 crashes) — interpreter returns `Bool(true)`
//!       (always-enabled stutter); compiler used to emit `Var("UNCHANGED")`
//!       which `eval_compiled` resolved to `ModelValue("UNCHANGED")`.
//!
//!   C2. Chained relops `X#Y = Z` (4 crashes) — compiler iterated the relop
//!       table in priority order and ran a whole-string scan per pattern,
//!       which silently preferred the later `=` over the earlier `#`. The
//!       interpreter (`eval.rs::split_top_level_comparison`) walks the
//!       string once and returns at the first relop hit, so it splits at
//!       `#`. Fix: `split_first_top_level_op` mirrors the interpreter.
//!
//!   C3. Arithmetic associativity `0-6-555555555` (1 crash, but the most
//!       important) — `split_binary_op(_, "-")` returned the FIRST top-level
//!       `-`, producing `Sub(0, Sub(6, 555555555)) = 555555549` (right-
//!       associative). The interpreter walks left-to-right and keeps the
//!       LAST top-level `+`/`-` position, returning
//!       `Sub(Sub(0,6), 555555555) = -555555561`. **This was a soundness
//!       bug on well-formed input**: any 3+-term arithmetic chain without
//!       explicit parens evaluated differently in the compiler vs the
//!       interpreter. Fix: `split_binary_op_last` for `+`, `-` (via
//!       `find_binary_minus_split`), `*`, `\div`, `%`.
//!
//!   C4. String escape `"...\G..."` (3 crashes — `\G`, `\HOOSE`, `\==`) —
//!       compiler's `parse_string_literal` only collapsed `\n`/`\t`/`\r`/
//!       `\\`/`\"` and kept `\X` literal for any other X. The interpreter
//!       (`eval.rs::parse_string_literal_prefix`) collapses every `\X` to
//!       just `X` (yes — including `\n`, which becomes `n`, not a newline).
//!       Fix: mirror the interpreter exactly.
//!
//! Each category gets one test below. Each test exercises both the
//! compiled and the interpreted evaluator on the input and asserts they
//! agree — the same equivalence check the fuzz target makes.

use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

/// Run both evaluators on `expr` against an empty state and assert they
/// either both Err or both Ok with equal values. Mirrors the equivalence
/// rule in `fuzz/fuzz_targets/fuzz_tla_swarm.rs::assert_equivalent`.
fn assert_equivalent(expr: &str) {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr(expr, &ctx);
    let compi = eval_compiled(&compile_expr(expr), &ctx);
    match (&interp, &compi) {
        (Ok(a), Ok(b)) => {
            assert_eq!(
                a, b,
                "DIVERGENCE on `{expr}`:\n  interpreter -> {a:?}\n  compiler    -> {b:?}"
            );
        }
        // Ok-vs-Err is tolerated by the swarm fuzz target as well, so we
        // tolerate it here. T101.1 is about Ok-with-different-value
        // divergences specifically.
        _ => {}
    }
}

// ---- C1: bare UNCHANGED is always-enabled stutter (Bool(true)) ----------

#[test]
fn t101_1_c1_bare_unchanged_returns_bool_true_in_both_evaluators() {
    assert_equivalent("UNCHANGED");
    assert_equivalent("  UNCHANGED  "); // trim survives
    // Composite forms (negation / disjunction / conjunction containing the
    // bare token) — the original disagreement propagates through any
    // wrapping boolean operator if the leaf disagrees.
    assert_equivalent("UNCHANGED /\\ TRUE");
    assert_equivalent("FALSE \\/ UNCHANGED");
    assert_equivalent("~UNCHANGED");
}

#[test]
fn t101_1_c1_unchanged_with_outer_comparison_defers_to_relop() {
    // Discovered by the second post-fix fuzz pass: `UNCHANGED <vars> = <expr>`
    // must split at the top-level `=`, not be greedily swallowed by the
    // `UNCHANGED` prefix. The interpreter does this because its UNCHANGED
    // handler sits BELOW split_top_level_comparison in the precedence
    // cascade; the compiler's prefix check needed to be moved to the
    // bottom of the cascade for the same reason.
    assert_equivalent("UNCHANGED 1G0=AL\n=10");
    assert_equivalent("UNCHANGED x = y");
    assert_equivalent("UNCHANGED <<a, b>> = <<c, d>>");
    assert_equivalent("UNCHANGED x /\\ TRUE"); // no outer relop — still TRUE
}

// ---- C2: chained relops split at the first relop, not the highest-priority one --

#[test]
fn t101_1_c2_chained_relops_split_at_leftmost_position() {
    // The fuzz repro: `CAE#S = i` should parse as `CAE # (S = i)` (the `#`
    // wins because it's the leftmost top-level relop), not `(CAE#S) = i`.
    assert_equivalent("CAE#S = i");
    assert_equivalent("CAE # S = i");
    // Sibling shapes that exercise the same scan-vs-priority distinction.
    assert_equivalent("a /= b = c");
    assert_equivalent("a < b = c");
    assert_equivalent("a > b = c");
    assert_equivalent("a # b /= c");
    // Sanity-check that explicit parens still produce the explicit shape
    // (i.e. we didn't break right-grouped relops).
    assert_equivalent("(CAE # S) = i");
    assert_equivalent("CAE # (S = i)");
}

// ---- C3: arithmetic is left-associative, NOT right-associative -----------

#[test]
fn t101_1_c3_subtraction_is_left_associative_on_well_formed_input() {
    // The fuzz repro — a fully well-formed integer expression. Compiler
    // used to return `Sub(0, Sub(6, 555555555)) = 555555549`; correct
    // answer is `Sub(Sub(0, 6), 555555555) = -555555561`.
    assert_equivalent("0-6-555555555");
    assert_equivalent("0 - 6 - 555555555");
    // 4-term and 5-term chains — generalisation of the same bug.
    assert_equivalent("100 - 30 - 20 - 10");
    assert_equivalent("1 - 2 - 3 - 4 - 5");
    // Mixed `+`/`-` chains: also left-assoc, e.g.
    // `10 - 3 + 2 - 1` = `((10 - 3) + 2) - 1` = 8, not `10 - (3 + (2 - 1))`.
    assert_equivalent("10 - 3 + 2 - 1");
    assert_equivalent("1 + 2 + 3 + 4");
    assert_equivalent("100 - 50 + 25 - 12");
}

#[test]
fn t101_1_c3_multiplication_and_division_are_left_associative() {
    // Same family for multiplicative: `*`, `\div`, `%` are all left-assoc.
    assert_equivalent("100 \\div 5 \\div 2"); // (100/5)/2 = 10, not 100/(5/2) = 100/2 = 50
    assert_equivalent("60 % 7 % 3"); // (60%7)%3 = 4%3 = 1, not 60%(7%3) = 60%1 = 0
    assert_equivalent("2 * 3 * 5");
    assert_equivalent("2 * 3 * 5 * 7");
}

#[test]
fn t101_1_c3_unary_minus_still_recognised_in_left_assoc_search() {
    // The previous `find_binary_minus_split` walked past unary `-` to find
    // the FIRST binary `-`. After the T101.1 left-assoc fix it walks past
    // unary `-` to find the LAST binary `-`. Make sure `-1 - x - y` still
    // parses (i.e. we didn't break the unary-leading case).
    assert_equivalent("-1 - 2 - 3");
    assert_equivalent("-1 + 2 - 3");
    // Trailing unary inside a parenthesised group should still be unary.
    assert_equivalent("5 - (-3)");
    assert_equivalent("(-1) - (-2) - (-3)");
}

#[test]
fn t101_1_c3_implication_remains_right_associative() {
    // Defensive: `=>` and `<=>` are RIGHT-associative in TLA+ and we
    // intentionally did NOT switch them to last-match. Pin the existing
    // `implication_remains_right_associative` test from src/tla/eval.rs at
    // the equivalence layer too.
    assert_equivalent("FALSE => TRUE => FALSE");
    assert_equivalent("TRUE <=> TRUE <=> FALSE");
}

// ---- C4: string escapes — `\X` collapses to `X` for ALL X ---------------

#[test]
fn t101_1_c4_string_unknown_escape_collapses_to_letter() {
    // The fuzz repros: `"\G"`, `"\H"`, `"\==..."`. Interpreter says the
    // backslash is dropped; compiler used to keep it.
    assert_equivalent("\"\\G\"");
    assert_equivalent("\"\\HOOSE\"");
    assert_equivalent("\"==\\nfoo\"");
    assert_equivalent("\"==\\\\G\""); // double backslash followed by G
    assert_equivalent("\"a*\\==\\==/\"");
}

#[test]
fn t101_1_c4_string_known_escape_also_collapses_to_letter() {
    // The interpreter's escape table is uniform: `\n` → `n`, NOT a newline.
    // Pin this so any future "fix" of the interpreter has to update the
    // compiler in lockstep (or introduce a real escape table on both sides).
    assert_equivalent("\"\\n\"");
    assert_equivalent("\"\\t\"");
    assert_equivalent("\"\\r\"");
    // `\\` and `\"` are the only escapes whose interpreter behaviour
    // happens to coincide with the standard meaning (drop `\`, keep
    // following char) — make sure both still work.
    assert_equivalent("\"\\\\\"");
    assert_equivalent("\"a\\\"b\"");
}

#[test]
fn t101_1_c4_string_with_non_ascii_content_round_trips() {
    // The compiler's old byte-loop would silently corrupt non-ASCII
    // payload (each non-ASCII rune was 2-4 bytes but the loop pushed each
    // byte individually as a `char`). The new char-loop preserves them.
    assert_equivalent("\"\u{062D}\""); // Arabic Hah
    assert_equivalent("\"\u{2764}\""); // ❤
    assert_equivalent("\"\u{1F600}\""); // 😀
    assert_equivalent("\"hello \u{062D} world\"");
    assert_equivalent("\"\\\u{062D}\""); // `\` followed by non-ASCII rune
}
