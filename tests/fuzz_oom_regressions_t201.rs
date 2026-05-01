//! T201: regression tests pinning the `fuzz_tla_swarm` OOM artifact and the
//! unbudgeted set/range/function-construction code paths it surfaced.
//!
//! Background: the swarm fuzz harness (`fuzz/fuzz_targets/fuzz_tla_swarm.rs`)
//! evaluated arbitrary TLA+ expressions against an `EvalContext` with **no**
//! `eval_budget` installed. Inputs that synthesised a large-range `1..N`,
//! function constructor `[i \in 1..N |-> ...]`, or set comprehension
//! `{ <expr> : x \in <huge> }` walked the iteration loops without any
//! per-element accounting, allocated gigabytes of `BTreeMap`/`BTreeSet`
//! nodes, and OOM-aborted the fuzz process.
//!
//! Fix shape (this commit):
//!   1. `eval.rs::eval_expr_inner` (range `..`), `compiled_eval.rs::SetRange`,
//!      `eval.rs::collect_binder_map_set` / `_filter_set` /
//!      `_function_mapping`, and `compiled_eval.rs::SetComprehension` /
//!      `FuncConstruct` now charge against the eval budget BEFORE the
//!      allocating step.
//!   2. The swarm fuzz harness installs `set_active_eval_budget(100_000)`
//!      around each evaluation (matching the probe path in `src/main.rs`).
//!
//! These tests assert:
//!   - The exact 57-byte fuzzer artifact terminates within 1 second.
//!   - Synthetic OOM seeds (range, set comprehension, function constructor)
//!     return `Err("evaluation budget exceeded ...")` quickly under the
//!     budget that the harness now installs.
//!
//! Why both shapes? The 57-byte artifact alone has structural recursion
//! that returns Err quickly even unbudgeted; it is preserved as a
//! "the headline input must not regress" pin. The synthetic seeds are the
//! load-bearing tests — they would OOM the test process WITHOUT the budget
//! checks added in this commit.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use tlaplusplus::tla::module::parse_tla_module_text;
use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{
    EvalContext, compile_expr, eval_compiled, eval_expr, restore_eval_budget,
    set_active_eval_budget,
};

/// The exact 57-byte fuzz artifact from the T201 OOM repro
/// (sha256 3d4b2ad3dbcd569f5928560be8e3becb6e9cd89fccfe2fdb7a2d82f8d56926b3).
/// First two bytes are swarm feature flags (0x7e 0x5b -> if_then_else,
/// case_expr, let_in, except, set_ops, seq_ops, record_literals,
/// nested_operators, primed_vars). Remaining 55 bytes are the module text.
const T201_OOM_BYTES: [u8; 57] = [
    126, 91, 4, 61, 61, 73, 9, 73, 9, 9, 4, 123, 4, 123, 91, 4, 61, 61, 61, 73, 9, 73, 9, 9, 4,
    123, 4, 123, 91, 4, 61, 61, 73, 8, 1, 0, 0, 0, 0, 43, 0, 61, 73, 8, 1, 0, 0, 0, 0, 43, 0, 0,
    0, 61, 61, 195, 185,
];

/// Evaluate every parsed-out definition body with both interpreter and
/// compiler under a fresh thread-local 100K-element eval budget — the same
/// shape the swarm fuzz harness uses post-fix. Asserts that the whole batch
/// completes within `wall_cap`.
fn eval_under_budget(text: &str, wall_cap: Duration) {
    let module = match parse_tla_module_text(text) {
        Ok(m) => m,
        Err(_) => return, // parse-only inputs are fine
    };
    let prev = set_active_eval_budget(100_000);
    let state = TlaState::new();
    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    let start = Instant::now();
    for (_name, def) in &module.definitions {
        if def.body.is_empty() {
            continue;
        }
        // Don't care whether they return Ok or Err — just that they return
        // promptly without panicking.
        let _ = eval_expr(&def.body, &ctx);
        let compiled = compile_expr(&def.body);
        let _ = eval_compiled(&compiled, &ctx);
        if start.elapsed() > wall_cap {
            restore_eval_budget(prev);
            panic!(
                "T201: eval batch exceeded wall cap {:?} (elapsed {:?}); \
                 budget enforcement regressed",
                wall_cap,
                start.elapsed()
            );
        }
    }
    restore_eval_budget(prev);
}

/// Headline pin: the exact fuzzer-emitted 57-byte input must terminate
/// within 1s under the harness's 100K budget. Without budget checks AND
/// the harness-side budget install, this would OOM the fuzzer.
#[test]
fn t201_oom_artifact_terminates_quickly() {
    // The bytes after the first 2 swarm flags are the module text.
    let text = std::str::from_utf8(&T201_OOM_BYTES[2..])
        .expect("artifact text bytes are valid UTF-8");
    eval_under_budget(text, Duration::from_secs(1));
}

/// Synthetic seed for the set-comprehension OOM class:
/// `{ [i \in 1..100 |-> i] : i \in 1..1_000_000 }` would allocate a 100-key
/// function for each of 1M iterations (~1.1 MB each) without a per-iteration
/// budget check. Under the harness's 100K budget the comprehension errors
/// out within milliseconds.
#[test]
fn t201_unbounded_set_comprehension_errors_under_budget() {
    let module = "\
---- MODULE T201Comprehension ----
EXTENDS Naturals
M == { [i \\in 1..100 |-> i] : i \\in 1..1000000 }
====
";
    let parsed = parse_tla_module_text(module).expect("parse");
    let body = parsed
        .definitions
        .get("M")
        .expect("M definition")
        .body
        .clone();

    // Install the same 100K budget the fuzz harness installs.
    let budget = Rc::new(Cell::new(100_000usize));
    let state = TlaState::new();
    let mut ctx = EvalContext::with_definitions(&state, &parsed.definitions);
    ctx.eval_budget = Some(Rc::clone(&budget));

    let start = Instant::now();
    let interp = eval_expr(&body, &ctx);
    let interp_elapsed = start.elapsed();
    let interp_err = interp.as_ref().err().map(ToString::to_string).unwrap_or_default();
    assert!(
        interp.is_err() && interp_err.contains("evaluation budget"),
        "interpreter must error under budget; got {:?} after {:?}",
        interp,
        interp_elapsed
    );
    assert!(
        interp_elapsed < Duration::from_secs(1),
        "interpreter must error promptly; took {:?}",
        interp_elapsed
    );

    // Reset and verify compiled side too.
    budget.set(100_000);
    let start = Instant::now();
    let compiled = compile_expr(&body);
    let comp = eval_compiled(&compiled, &ctx);
    let comp_elapsed = start.elapsed();
    // Compiled may take a different shape (parse error vs budget); accept
    // either Err(_) provided it is prompt and not OOM.
    assert!(
        comp.is_err(),
        "compiled side must Err on unbudgeted comprehension; got {:?} after {:?}",
        comp,
        comp_elapsed
    );
    assert!(
        comp_elapsed < Duration::from_secs(1),
        "compiled must error promptly; took {:?}",
        comp_elapsed
    );
}

/// Synthetic seed for the giant-range OOM class: `1..1_000_000_000` would
/// allocate 8 GB of `TlaValue::Int(_)` nodes without a budget check.
#[test]
fn t201_huge_range_errors_under_budget() {
    let module = "\
---- MODULE T201Range ----
EXTENDS Naturals
M == 1..1000000000
====
";
    let parsed = parse_tla_module_text(module).expect("parse");
    let body = parsed.definitions.get("M").expect("M").body.clone();

    let budget = Rc::new(Cell::new(100_000usize));
    let state = TlaState::new();
    let mut ctx = EvalContext::with_definitions(&state, &parsed.definitions);
    ctx.eval_budget = Some(Rc::clone(&budget));

    let start = Instant::now();
    let r = eval_expr(&body, &ctx);
    let elapsed = start.elapsed();
    let err_msg = r.as_ref().err().map(ToString::to_string).unwrap_or_default();
    assert!(
        r.is_err() && err_msg.contains("evaluation budget"),
        "interp range must error under budget; got {:?} after {:?}",
        r,
        elapsed
    );
    assert!(
        elapsed < Duration::from_millis(100),
        "range budget check must be O(1); took {:?}",
        elapsed
    );

    // Compiled side.
    budget.set(100_000);
    let start = Instant::now();
    let compiled = compile_expr(&body);
    let r = eval_compiled(&compiled, &ctx);
    let elapsed = start.elapsed();
    let err_msg = r.as_ref().err().map(ToString::to_string).unwrap_or_default();
    assert!(
        r.is_err() && err_msg.contains("evaluation budget"),
        "compiled range must error under budget; got {:?} after {:?}",
        r,
        elapsed
    );
    assert!(
        elapsed < Duration::from_millis(100),
        "compiled range budget check must be O(1); took {:?}",
        elapsed
    );
}

/// Synthetic seed for the function-constructor OOM class:
/// `[i \in 1..1_000_000 |-> i]` would allocate 1M-key BTreeMap without a
/// per-iteration budget check.
#[test]
fn t201_huge_function_constructor_errors_under_budget() {
    let module = "\
---- MODULE T201FuncCtor ----
EXTENDS Naturals
M == [i \\in 1..1000000 |-> i]
====
";
    let parsed = parse_tla_module_text(module).expect("parse");
    let body = parsed.definitions.get("M").expect("M").body.clone();

    let budget = Rc::new(Cell::new(100_000usize));
    let state = TlaState::new();
    let mut ctx = EvalContext::with_definitions(&state, &parsed.definitions);
    ctx.eval_budget = Some(Rc::clone(&budget));

    let start = Instant::now();
    let r = eval_expr(&body, &ctx);
    let elapsed = start.elapsed();
    let err_msg = r.as_ref().err().map(ToString::to_string).unwrap_or_default();
    assert!(
        r.is_err() && err_msg.contains("evaluation budget"),
        "interp func ctor must error under budget; got {:?} after {:?}",
        r,
        elapsed
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "func ctor budget check must be prompt; took {:?}",
        elapsed
    );
}
