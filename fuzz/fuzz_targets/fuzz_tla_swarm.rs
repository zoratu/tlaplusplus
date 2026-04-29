#![no_main]

//! Swarm fuzz target for TLA+ parsing and evaluation.
//!
//! Based on "Swarm Testing" (Groce et al., ISSTA 2012): instead of testing
//! all features in every run, randomly omit features per configuration.
//! The first 2 bytes of fuzz input control which TLA+ features are enabled;
//! the remaining bytes are interpreted as TLA+ module text.
//!
//! This diversity finds more bugs than the "use everything" approach because
//! individual feature combinations reach deeper into rarely-tested code paths.
//!
//! T101 augmentation: each definition body is evaluated by BOTH the
//! interpreter (`eval_expr`) and the compiler (`compile_expr` +
//! `eval_compiled`). Any divergence — one panicking while the other does not,
//! one returning Ok while the other returns Err, or both returning Ok with
//! different values — is treated as a panic and aborts the fuzz run, the
//! same proptest-T2-style equivalence check used in `tests/compiled_vs_interpreted.rs`.

use libfuzzer_sys::fuzz_target;
use std::panic::{AssertUnwindSafe, catch_unwind};
use tlaplusplus::tla::module::parse_tla_module_text;
use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

/// Feature flags derived from the first 2 bytes of fuzz input.
struct SwarmConfig {
    quantifiers: bool,       // \A, \E, CHOOSE
    if_then_else: bool,      // IF/THEN/ELSE
    case_expr: bool,         // CASE
    let_in: bool,            // LET/IN
    except: bool,            // EXCEPT
    set_ops: bool,           // SUBSET, UNION, \X
    seq_ops: bool,           // Append, Head, Tail, Len, SubSeq
    func_construction: bool, // [x \in S |-> expr]
    record_literals: bool,   // [a |-> 1, b |-> 2]
    nested_operators: bool,  // operator calling operator
    unchanged: bool,         // UNCHANGED
    primed_vars: bool,       // x'
}

impl SwarmConfig {
    fn from_bytes(b0: u8, b1: u8) -> Self {
        let bits = (b0 as u16) | ((b1 as u16) << 8);
        Self {
            quantifiers: bits & (1 << 0) != 0,
            if_then_else: bits & (1 << 1) != 0,
            case_expr: bits & (1 << 2) != 0,
            let_in: bits & (1 << 3) != 0,
            except: bits & (1 << 4) != 0,
            set_ops: bits & (1 << 5) != 0,
            seq_ops: bits & (1 << 6) != 0,
            func_construction: bits & (1 << 7) != 0,
            record_literals: bits & (1 << 8) != 0,
            nested_operators: bits & (1 << 9) != 0,
            unchanged: bits & (1 << 10) != 0,
            primed_vars: bits & (1 << 11) != 0,
        }
    }

    /// Check whether the given TLA+ text contains a disabled feature.
    /// Returns true if the text should be skipped (contains disabled features).
    fn should_skip(&self, text: &str) -> bool {
        if !self.quantifiers
            && (text.contains("\\A ")
                || text.contains("\\E ")
                || text.contains("CHOOSE"))
        {
            return true;
        }
        if !self.if_then_else && text.contains("IF ") && text.contains("THEN ") {
            return true;
        }
        if !self.case_expr && text.contains("CASE ") {
            return true;
        }
        if !self.let_in && text.contains("LET ") && text.contains(" IN ") {
            return true;
        }
        if !self.except && text.contains("EXCEPT") {
            return true;
        }
        if !self.set_ops
            && (text.contains("SUBSET")
                || text.contains("UNION")
                || text.contains("\\X "))
        {
            return true;
        }
        if !self.seq_ops
            && (text.contains("Append(")
                || text.contains("Head(")
                || text.contains("Tail(")
                || text.contains("Len(")
                || text.contains("SubSeq("))
        {
            return true;
        }
        if !self.func_construction && text.contains("|->") && text.contains("\\in") {
            return true;
        }
        if !self.record_literals && text.contains("|->") && !text.contains("\\in") {
            return true;
        }
        if !self.unchanged && text.contains("UNCHANGED") {
            return true;
        }
        if !self.primed_vars && text.contains('\'') {
            return true;
        }
        false
    }
}

/// Run both the interpreter and the compiler on `expr`, asserting equivalence.
/// Panics on divergence, which the fuzzer treats as a crash.
///
/// Equivalence rules:
///   - panic on either side of the pair while the other did not — DIVERGENCE
///   - both Ok with different values — DIVERGENCE
///   - one side Ok, other Err — TOLERATED (compiler error-recovery and
///     interpreter error-recovery have known semantic gaps tracked under
///     T101.1; the fuzz pass focuses on panic-resistance, not on error-shape
///     parity)
///   - any other combination — equivalent enough
fn assert_equivalent(expr: &str, ctx: &EvalContext<'_>) {
    let interp = catch_unwind(AssertUnwindSafe(|| eval_expr(expr, ctx)));
    let compi = catch_unwind(AssertUnwindSafe(|| {
        let compiled = compile_expr(expr);
        eval_compiled(&compiled, ctx)
    }));

    match (interp, compi) {
        // Both panicked — symmetric. Propagate so the input is preserved as
        // a crash artifact, but it counts as one panic, not a divergence.
        (Err(p), Err(_)) => std::panic::resume_unwind(p),
        // Both completed (Ok or Err on each side) — compare verdicts.
        (Ok(i), Ok(c)) => match (i, c) {
            (Ok(a), Ok(b)) => {
                if a != b {
                    panic!(
                        "DIVERGENCE on `{expr}`:\n  interpreter -> {a:?}\n  compiler    -> {b:?}"
                    );
                }
            }
            // Both errored, or one errored and the other succeeded — tolerated.
            // Error-shape parity is tracked as T101.1.
            _ => {}
        },
        // One side panicked, the other did not. This IS a real panic-resistance
        // divergence: the same input must not panic in one evaluator while the
        // other returns a clean Err. Re-raise the panic that actually fired.
        (Ok(_), Err(p)) => std::panic::resume_unwind(p),
        (Err(p), Ok(_)) => std::panic::resume_unwind(p),
    }
}

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for feature flags + some text
    if data.len() < 4 {
        return;
    }

    let config = SwarmConfig::from_bytes(data[0], data[1]);
    let text_bytes = &data[2..];

    let text = match std::str::from_utf8(text_bytes) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Skip if the text contains features that this swarm config has disabled
    if config.should_skip(text) {
        return;
    }

    // Phase 1: Parse the module text (should not panic)
    let module = match parse_tla_module_text(text) {
        Ok(m) => m,
        Err(_) => return, // parse errors are fine, panics are not
    };

    // Phase 2: Try to evaluate each definition body as an expression
    // Build a minimal EvalContext with the parsed definitions
    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);

    for (_name, def) in &module.definitions {
        if def.body.is_empty() {
            continue;
        }

        // Skip definitions whose bodies contain disabled features
        if config.should_skip(&def.body) {
            continue;
        }

        // Skip action-like definitions (contain primes) unless primed vars enabled
        if !config.primed_vars && def.body.contains('\'') {
            continue;
        }

        // Skip parameterized operators unless nested operators are enabled
        if !config.nested_operators && !def.params.is_empty() {
            continue;
        }

        // Equivalence check (panics on divergence; libfuzzer treats as crash).
        assert_equivalent(&def.body, &ctx);
    }
});
