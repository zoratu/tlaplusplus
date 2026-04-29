//! Regression tests for panic-resistance bugs surfaced by the T101 fuzz pass
//! (see `RELEASE_1.0.1_PLAN.md`).
//!
//! Each test pins one fuzzer-discovered crashing input to assert that it now
//! returns gracefully (Ok or Err) instead of panicking. The original fuzz
//! corpus is preserved in `fuzz/corpus/` on the spot host; the inputs here
//! are minimal reproducers extracted from those crashes.

use tlaplusplus::tla::cfg::parse_tla_config;
use tlaplusplus::tla::module::parse_tla_module_text;
use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

/// T101: `parse_recursive_declarations` indexed `text[start..i]` using char
/// indices rather than byte offsets. Any non-ASCII byte inside a RECURSIVE
/// declaration (or in surrounding text whose char count diverged from byte
/// count) triggered a panic:
///   `start byte index 52 is not a char boundary;
///    it is inside '\u{89}' (bytes 51..53 of string)`
///
/// The fix is to walk `char_indices()` so slice boundaries are always on
/// UTF-8 code-point boundaries. This test pins one minimised reproducer.
#[test]
fn t101_recursive_decl_with_non_ascii_does_not_panic() {
    // Arabic letter Hah (U+062D) — 2 bytes in UTF-8. Position chosen so a
    // naive char-index-vs-byte-index slice mismatch panics.
    let module = "\
---- MODULE NonAsciiRecursive ----
EXTENDS Naturals
RECURSIVE Op(_), Op\u{062D}2(_, _)

Op(n) == n
====
";
    // The point: this must not panic, irrespective of whether the parse
    // ultimately succeeds or returns Err.
    let _ = parse_tla_module_text(module);
}

/// T101: same family as above, but inside an INSTANCE WITH clause where
/// `parse_with_substitutions` and `instance_with_clause_is_incomplete` had
/// the same `chars().collect()` + byte-slice pattern. Both byte indices
/// `start` and `i` were really char indices, so a non-ASCII rune embedded
/// in the substitution payload would trip the same panic.
#[test]
fn t101_instance_with_non_ascii_substitution_does_not_panic() {
    let module = "\
---- MODULE NonAsciiInstance ----
EXTENDS Naturals
INSTANCE Helper WITH x <- \"\u{062D}\", y <- 1
====
";
    let _ = parse_tla_module_text(module);
}

/// T101: the .cfg parser (`parse_tla_config`) shares family with the module
/// parser. Ensure it tolerates non-ASCII bytes in arbitrary positions.
#[test]
fn t101_cfg_with_non_ascii_does_not_panic() {
    let cfg = "CONSTANTS\n  N = \u{062D}\nINIT Init\nNEXT Next\n";
    let _ = parse_tla_config(cfg);
}

/// T101: `strip_label_prefix` in the compiled expression compiler computed
/// `id_end` as `1 + sum(rest chars' len_utf8)`, hard-coding the leading
/// char's byte length to 1. With a non-ASCII leading char (`len_utf8() > 1`)
/// the resulting `s[id_end..]` slice landed inside a UTF-8 code point and
/// panicked:
///   `start byte index 3 is not a char boundary;
///    it is inside '¹' (bytes 2..4 of string)`
#[test]
fn t101_compiled_expr_non_ascii_label_does_not_panic() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    // Leading char is U+00B9 (SUPERSCRIPT ONE) — 2 bytes in UTF-8.
    for expr in &["\u{00B9}", "\u{00B9}::TRUE", "\u{062D}"] {
        let _ = compile_expr(expr);
        let _ = eval_compiled(&compile_expr(expr), &ctx);
        let _ = eval_expr(expr, &ctx);
    }
}

/// T101: defence-in-depth — an empty-ish module with stray non-ASCII bytes
/// in places where the parser is normally relaxed (between declarations,
/// inside comments, after `====`, etc.) must never panic.
#[test]
fn t101_stray_non_ascii_in_module_does_not_panic() {
    for sample in &[
        "\u{062D}",
        "MODULE \u{062D}",
        "---- MODULE \u{062D} ----\n====",
        "---- MODULE M ----\n\u{062D} == 1\n====",
        "---- MODULE M ----\nVARIABLE x\n\\* \u{062D}\nInit == TRUE\n====",
    ] {
        let _ = parse_tla_module_text(sample);
    }
}
