//! T203: regression test for the LET-binding OOM surfaced by post-T201
//! full-arsenal fuzz validation. The unbounded-allocator path was
//! `EvalContext::with_local_definitions` cloning the BTreeMap once per
//! nested LET. Fix charges the clone cost against the eval budget at the
//! `eval_let_expression` and `eval_let_action_multi_branch` callers.

use std::time::{Duration, Instant};
use tlaplusplus::tla::value::TlaState;
use tlaplusplus::tla::{EvalContext, eval_expr, restore_eval_budget, set_active_eval_budget};

const T203_OOM_BYTES: &[u8] = &[
    234, 36, 5, 126, 69, 84, 0, 61, 61, 197, 146, 61, 66, 61, 61, 76, 69, 84, 10, 32, 10, 32, 10,
    32, 5, 4, 73, 78, 73, 15, 0, 61, 66, 61, 61, 61, 76, 69, 84, 61, 76, 10, 32, 10, 38, 10, 32,
    66, 61, 61, 76, 69, 84, 10, 32, 10, 32, 10, 32, 5, 4, 73, 78, 73, 15, 0, 61, 66, 61, 61, 61,
    76, 69, 84, 61, 76, 10, 32, 10, 38, 10, 32, 66, 61, 61, 76, 69, 84, 10, 32, 10, 32, 10, 32, 5,
    4, 73, 78, 73, 15, 0, 61, 66, 61, 61, 61, 76, 69, 84, 61, 76, 10, 32, 10, 32, 10, 32, 5, 4,
    73, 78, 73, 15, 0, 61, 66, 61, 61, 61, 76, 69, 84, 61, 76, 10, 32, 10, 32, 10, 32,
];

#[test]
fn t203_oom_artifact_terminates_quickly_under_budget() {
    let prev = set_active_eval_budget(100_000);
    let expr = String::from_utf8_lossy(T203_OOM_BYTES).into_owned();
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let start = Instant::now();
    let _ = eval_expr(&expr, &ctx);
    let elapsed = start.elapsed();
    restore_eval_budget(prev);
    assert!(
        elapsed < Duration::from_secs(1),
        "T203 artifact took {elapsed:?} (budget should have stopped it well under 1s)"
    );
}

#[test]
fn t203_deeply_nested_let_errors_under_budget() {
    let prev = set_active_eval_budget(1_000);
    let mut expr = String::from("a");
    for _ in 0..2_000 {
        expr = format!("LET a == 1 IN {expr}");
    }
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let start = Instant::now();
    let res = eval_expr(&expr, &ctx);
    let elapsed = start.elapsed();
    restore_eval_budget(prev);
    // Either budget-exceeded or recursion-depth-exceeded is acceptable —
    // both are bounded failure modes. The point is no OOM and quick exit.
    assert!(
        res.is_err(),
        "deeply-nested LET under 1000-budget should Err, got {:?}",
        res
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "deeply-nested LET took {elapsed:?} (should fail fast)"
    );
}
