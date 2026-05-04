//! T205: regression — bare `IOEnv` in compiled eval must dispatch to the
//! built-in operator (returning the env Record), not fall through to
//! `ModelValue("IOEnv")`. Surfaced by fuzz_tla_swarm equivalence check.

use tlaplusplus::tla::value::{TlaState, TlaValue};
use tlaplusplus::tla::{EvalContext, compile_expr, eval_compiled, eval_expr};

#[test]
fn t205_bare_ioenv_dispatches_to_builtin_in_compiler() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr("IOEnv", &ctx).expect("interp ok");
    let compi = eval_compiled(&compile_expr("IOEnv"), &ctx).expect("compi ok");
    // Both should return Record(env vars), not ModelValue("IOEnv").
    match (&interp, &compi) {
        (TlaValue::Record(_), TlaValue::Record(_)) => {}
        _ => panic!("T205: expected both to be Record(env), got interp={interp:?} compi={compi:?}"),
    }
    assert_eq!(interp, compi, "T205: interp/compiler must agree on IOEnv");
}

#[test]
fn t205_bare_emptybag_dispatches_to_builtin_in_compiler() {
    let state = TlaState::new();
    let ctx = EvalContext::new(&state);
    let interp = eval_expr("EmptyBag", &ctx).expect("interp ok");
    let compi = eval_compiled(&compile_expr("EmptyBag"), &ctx).expect("compi ok");
    assert_eq!(interp, compi, "T205: interp/compiler must agree on EmptyBag");
}
