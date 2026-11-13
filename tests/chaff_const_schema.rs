//! Chaff gate for `constants_in_fingerprint` (chaffs.toml).
//!
//! The `const_state_omit` diff_tlc spec guards the #193 "constants must not be
//! carried in the fingerprinted state" fix by distinct-state COUNT. But carrying
//! a constant adds the *same* key->value to every state, so the distinct count
//! and the invariant verdict are unchanged — a count-based differential oracle
//! structurally cannot observe the mutation (this is exactly what the first chaff
//! matrix run surfaced). This test closes that oracle gap by asserting on the
//! fingerprint SCHEMA (the set of state keys) directly.
//!
//! Without the chaff (normal build, or `--features chaff` with the chaff
//! inactive) the initial states must contain only VARIABLE keys. With
//! `TLAPP_CHAFF=constants_in_fingerprint` active, `TlaModel::from_files` skips the
//! variable-only filter, so the constants `MaxN` and `NULL` leak into the state
//! map and this test fails — i.e. the gate CATCHES the chaff.
//!
//! `scripts/chaff_matrix.sh` runs this crate with that env var set for the
//! `constants_in_fingerprint` chaff (gate = "const_schema").

use std::path::PathBuf;

use tlaplusplus::Model;
use tlaplusplus::models::tla_native::TlaModel;

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn constants_are_not_carried_in_state_schema() {
    let module = repo_path("corpus/examples/ConstStateOmit.tla");
    let cfg = repo_path("corpus/examples/ConstStateOmit.cfg");
    let model = TlaModel::from_files(&module, Some(&cfg), None, None).expect("model loads");

    let initial = model.initial_states();
    assert!(
        !initial.is_empty(),
        "expected at least one initial state for ConstStateOmit"
    );

    // Constants declared in the .cfg must never appear as state keys. Under the
    // chaff this assertion fails (the gate catches it); under normal operation it
    // holds. We check every initial state, not just the first.
    // The `[CHAFF_CATCH_OK]` sentinel marks the INTENDED assertion. The chaff
    // matrix greps the cargo log for it so that only a failure HERE (a constant
    // leaking into the fingerprinted state) is credited as a catch — an unrelated
    // panic, a compile error, or a `from_files` crash does not carry the sentinel
    // and is reported as an unknown-mode HOLE instead of a false "caught".
    for (idx, state) in initial.iter().enumerate() {
        for constant in ["MaxN", "NULL"] {
            assert!(
                !state.contains_key(constant),
                "[CHAFF_CATCH_OK] constant `{constant}` leaked into the fingerprinted state \
                 (initial state #{idx}): constants belong to the fixed model context, not the \
                 per-state schema (see #193)"
            );
        }
        // Positive check: the real state variable `x` must be present.
        assert!(
            state.contains_key("x"),
            "expected state variable `x` in initial state #{idx}"
        );
    }
}
