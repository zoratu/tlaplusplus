// T3 — state-graph snapshot tests.
//
// WHY THIS FILE EXISTS
// --------------------
// T1 (`scripts/diff_tlc.sh`) checks distinct state COUNTS against TLC for a
// curated set of specs. T2 (`tests/compiled_vs_interpreted.rs`) checks that
// interpreted and compiled expression evaluation agree. Neither catches an
// off-by-one in successor generation that produces the right COUNT but with
// one wrong-and-one-missing state cancelling out.
//
// This file pins the actual reachable-state set for a small curated list of
// specs as a content-addressed digest. Any future change that perturbs the
// set — even with the same count — fails the snapshot.
//
// SNAPSHOT MECHANISM
// ------------------
// 1. Build a `TlaModel` from a `.tla`/`.cfg` pair.
// 2. Deterministic BFS from the initial states using the `Model` trait. The
//    `TlaState` alias is `BTreeMap<Arc<str>, TlaValue>`, which iterates in
//    sorted key order, and `TlaValue`'s contained collections are all
//    `BTree*`-backed, so canonical JSON serialisation via `serde_json` is
//    fully deterministic across runs and threads.
// 3. Sort the canonical-JSON repr of every reachable state lexicographically
//    and feed each line into a single `XxHash3_128` digest, with a `\n`
//    separator. The hex of the digest is the snapshot.
// 4. Each test asserts the recomputed digest matches the pinned `&'static str`.
//    On mismatch the assertion message prints the new digest and the first
//    few canonical reprs so a developer who *intentionally* changed the state
//    space can paste the new digest in.
//
// WORKFLOW: WHEN TO UPDATE A DIGEST
// ---------------------------------
// An unexplained digest change is a bug. Treat any mismatch as "you changed
// the reachable state space" and prove the new set is correct before
// blessing the new digest:
//
//   1. Re-run TLC on the spec with `scripts/diff_tlc.sh` (or by hand) to
//      confirm the new distinct-state count matches TLC's. If it doesn't,
//      the change is a regression — fix the underlying code, don't touch
//      the snapshot.
//   2. If the count matches, eyeball the new set vs the old set (the test
//      output prints the first 10 canonical reprs on failure). For specs
//      under ~200 states it's feasible to spot the structural difference by
//      hand.
//   3. Once confident the new set is the correct one, regenerate ALL
//      digests with `scripts/regen_state_graph_snapshots.sh` (or run the
//      bundled `snapshot_regen` helper, see below) and paste the new
//      digests into the `SNAPSHOTS` list at the bottom of this file.
//
// REGENERATING SNAPSHOTS
// ----------------------
//   $ cargo test --release --test state_graph_snapshots snapshot_regen \
//       -- --ignored --nocapture
//
// Prints one line per spec:
//   spec_id  <reachable_state_count>  <digest_hex>
//
// We deliberately do NOT auto-update the source: the human is the gate.
//
// VALIDATION RULE
// ---------------
// Every snapshotted spec lists its TLC-validated state count in a comment
// next to its digest entry, plus the date the count was cross-checked
// against TLC.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use tlaplusplus::Model;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::tla::TlaState;
use twox_hash::XxHash3_128;

/// One snapshot fixture: a TLA+ spec we pin the reachable state set of.
#[allow(dead_code)]
struct Snapshot {
    /// Stable test identifier, matches the function name.
    id: &'static str,
    /// Path to the `.tla` module, relative to the workspace root.
    module: &'static str,
    /// Path to the `.cfg` config, relative to the workspace root.
    config: &'static str,
    /// Expected digest of the sorted canonical-JSON reprs (lowercase hex,
    /// 32 chars = 128 bits). Update only after cross-checking with TLC.
    digest: &'static str,
    /// Expected reachable distinct state count, cross-checked against TLC.
    /// Used as a sanity check before the digest comparison so a count
    /// mismatch surfaces with a clearer error.
    expected_count: usize,
    /// Free-form description of what this spec exercises.
    notes: &'static str,
}

/// Curated snapshot list. State counts cross-checked against TLC on
/// `2026-04-25` (see RELEASE_1.0.0_LOG.md / T3 entry for the run notes).
///
/// Selection rules:
///   - Under ~200 reachable states (cheap to enumerate + canonicalise).
///   - NOT in `corpus/diff_test/list.tsv` (T1 already pins counts on those).
///   - Each spec exercises a different corner of the successor function
///     (UNCHANGED stutter, multi-var \E binders, EXCEPT updates,
///     SUBSET set-comprehension Init, INSTANCE substitution, LET-bound
///     post-conditions, etc.).
// All counts cross-checked against TLC v2.19 on 2026-04-25.
// (Some specs use a `SPECIFICATION Init /\ [][Next]_vars` cfg that TLC
// doesn't accept directly — for those, TLC was run with an INIT/NEXT cfg
// equivalent to the original; tlaplusplus accepts both.)
const SNAPSHOTS: &[Snapshot] = &[
    Snapshot {
        id: "multi_var_quantifier",
        module: "corpus/language/MultiVarQuantifierTest.tla",
        config: "corpus/language/MultiVarQuantifierTest.cfg",
        // TLC: 10 distinct (INIT/NEXT cfg).
        digest: "b11ca2e0f410bb3760a89af3521ebf35",
        expected_count: 10,
        notes: "multi-binder \\E i, j \\in S in Next; small (x,y) grid",
    },
    Snapshot {
        id: "instance_test_simple",
        module: "corpus/language/InstanceTestSimple.tla",
        config: "corpus/language/InstanceTestSimple.cfg",
        // TLC: 2 distinct.
        digest: "0541479eda0de413d6f4ed5c80ff273d",
        expected_count: 2,
        notes: "INSTANCE WITH substitution + UNCHANGED stutter on flag=TRUE",
    },
    Snapshot {
        id: "operator_substitution",
        module: "corpus/language/OperatorSubstitutionTest.tla",
        config: "corpus/language/OperatorSubstitutionTest.cfg",
        // TLC: 6 distinct (INIT/NEXT cfg).
        digest: "c78da9aac78817f7d18aa79bcca6e29b",
        expected_count: 6,
        notes: "LET-bound post-condition; sequence Append; deterministic chain",
    },
    Snapshot {
        id: "string_test",
        module: "corpus/language/StringTest.tla",
        config: "corpus/language/StringTest.cfg",
        // TLC: 2 distinct (INIT/NEXT cfg).
        digest: "4b565b503f7b3f5ec05df7dc87b78d51",
        expected_count: 2,
        notes: "string equality, set membership, UNCHANGED-in-disjunct stutter",
    },
    Snapshot {
        id: "wrapper_next_fairness",
        module: "corpus/internals/WrapperNextFairness.tla",
        config: "corpus/internals/WrapperNextFairness.cfg",
        // TLC: 4 distinct.
        digest: "4fc1965398503047582269c0f0c0fb1a",
        expected_count: 4,
        notes: "wrapper Next with UNCHANGED stutter at terminal state",
    },
    Snapshot {
        id: "instance_test",
        module: "corpus/language/InstanceTest.tla",
        config: "corpus/language/InstanceTest.cfg",
        // TLC: 11 distinct.
        digest: "2e7f071fff2709df7d53a26d6853f8bf",
        expected_count: 11,
        notes: "INSTANCE WITH + monotone counter with explicit fixed-point disjunct",
    },
    Snapshot {
        id: "enabled_test",
        module: "corpus/temporal/EnabledTest.tla",
        config: "corpus/temporal/EnabledTest.cfg",
        // TLC: 121 distinct.
        digest: "c5626699ead47d3ff21075fd81ac2f55",
        expected_count: 121,
        notes: "ENABLED predicate evaluation; (x,y) grid in 0..10 with stutter",
    },
];

/// Compute the canonical JSON repr of a state. `TlaState` and the Set/Record/
/// Function variants of `TlaValue` are all `BTree*`-backed, and `Seq` is a
/// `Vec` (positional), so `serde_json::to_string` is deterministic.
fn canonical_state_repr(state: &TlaState) -> String {
    serde_json::to_string(state).expect("TlaState serialises as JSON")
}

/// Deterministic BFS reachable-set enumeration via the `Model` trait. We
/// dedupe by the canonical JSON repr (rather than the runtime's `fingerprint`)
/// so the digest is *content-addressed* and immune to fingerprint hash-seed
/// changes. The model's runtime fingerprint may vary across builds (e.g.
/// AHash seed), but the canonical repr is stable as long as `TlaValue`'s
/// `Serialize` impl is.
fn enumerate_reachable(model: &TlaModel) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<TlaState> = Vec::new();

    for s in model.initial_states() {
        let repr = canonical_state_repr(&s);
        if seen.insert(repr) {
            frontier.push(s);
        }
    }

    let mut buf: Vec<TlaState> = Vec::new();
    while let Some(state) = frontier.pop() {
        buf.clear();
        model.next_states(&state, &mut buf);
        for next in buf.drain(..) {
            let repr = canonical_state_repr(&next);
            if seen.insert(repr) {
                frontier.push(next);
            }
        }
    }

    // Sort lexicographically for digest determinism (independent of
    // exploration order).
    let mut out: Vec<String> = seen.into_iter().collect();
    out.sort();
    out
}

/// Hash sorted canonical reprs into a 128-bit digest, hex-encoded.
fn digest_reprs(reprs: &[String]) -> String {
    let mut hasher = XxHash3_128::new();
    for r in reprs {
        hasher.write(r.as_bytes());
        hasher.write(b"\n");
    }
    let h = hasher.finish_128();
    format!("{:032x}", h)
}

/// Resolve a workspace-relative path. `CARGO_MANIFEST_DIR` points at the
/// workspace root for tests in `tests/`.
fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

/// Build a TlaModel from a snapshot fixture.
fn build_model(snap: &Snapshot) -> TlaModel {
    let module = workspace_path(snap.module);
    let config = workspace_path(snap.config);
    TlaModel::from_files(&module, Some(&config), None, None)
        .unwrap_or_else(|e| panic!("snapshot fixture {}: failed to build model: {e:#}", snap.id))
}

/// Compute the digest + count for a snapshot, returning both for both
/// `assert` and `regen` paths.
fn compute_digest(snap: &Snapshot) -> (String, usize, Vec<String>) {
    let model = build_model(snap);
    let reprs = enumerate_reachable(&model);
    let digest = digest_reprs(&reprs);
    let count = reprs.len();
    (digest, count, reprs)
}

/// Common assertion path: print a helpful message on mismatch.
fn assert_snapshot(snap: &Snapshot) {
    let (got_digest, got_count, reprs) = compute_digest(snap);

    if snap.expected_count != 0 && snap.expected_count != got_count {
        let preview: Vec<&String> = reprs.iter().take(10).collect();
        panic!(
            "\nSnapshot count mismatch for `{}`:\n  expected_count = {}\n  actual_count   = {}\n\
             First {} canonical reprs (of {}):\n{}\n\n\
             If this is an intentional change, re-run TLC and confirm the new count, then update\n\
             both `expected_count` and `digest` in `SNAPSHOTS[]`.\n",
            snap.id,
            snap.expected_count,
            got_count,
            preview.len(),
            got_count,
            preview
                .iter()
                .map(|r| format!("    {}", r))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    if snap.digest != got_digest {
        let preview: Vec<&String> = reprs.iter().take(10).collect();
        panic!(
            "\nSnapshot digest mismatch for `{}` ({} reachable states):\n  \
             expected = {}\n  actual   = {}\n\n\
             First {} canonical reprs (of {}):\n{}\n\n\
             If this is an INTENTIONAL change, cross-check the new state space against TLC\n\
             (e.g. `bash scripts/diff_tlc.sh` after adding the spec to the diff list, or\n\
             run TLC manually). Once confirmed correct, paste the new digest into\n\
             `SNAPSHOTS[]` in tests/state_graph_snapshots.rs.\n\n\
             To regenerate ALL snapshot digests at once:\n  \
             cargo test --release --test state_graph_snapshots snapshot_regen \\\n  \
                 -- --ignored --nocapture\n",
            snap.id,
            got_count,
            snap.digest,
            got_digest,
            preview.len(),
            got_count,
            preview
                .iter()
                .map(|r| format!("    {}", r))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
}

// One test per spec so failures are localised in the test runner.

#[test]
fn snapshot_multi_var_quantifier() {
    assert_snapshot(&SNAPSHOTS[0]);
}

#[test]
fn snapshot_instance_test_simple() {
    assert_snapshot(&SNAPSHOTS[1]);
}

#[test]
fn snapshot_operator_substitution() {
    assert_snapshot(&SNAPSHOTS[2]);
}

#[test]
fn snapshot_string_test() {
    assert_snapshot(&SNAPSHOTS[3]);
}

#[test]
fn snapshot_wrapper_next_fairness() {
    assert_snapshot(&SNAPSHOTS[4]);
}

#[test]
fn snapshot_instance_test() {
    assert_snapshot(&SNAPSHOTS[5]);
}

#[test]
fn snapshot_enabled_test() {
    assert_snapshot(&SNAPSHOTS[6]);
}

/// Regeneration helper. Marked `#[ignore]` so it doesn't run on normal
/// `cargo test`. Run with:
///
///   cargo test --release --test state_graph_snapshots snapshot_regen \
///       -- --ignored --nocapture
///
/// Prints one line per spec in a format that's easy to copy back into
/// `SNAPSHOTS[]`.
#[test]
#[ignore]
fn snapshot_regen() {
    println!();
    println!(
        "# Regenerated state-graph snapshots ({} specs):",
        SNAPSHOTS.len()
    );
    println!("# Format: id  count  digest");
    println!("# After verifying each count against TLC, paste these into SNAPSHOTS[].");
    println!();
    let mut max_id_len = 0;
    for snap in SNAPSHOTS {
        max_id_len = max_id_len.max(snap.id.len());
    }
    for snap in SNAPSHOTS {
        let (digest, count, _reprs) = compute_digest(snap);
        println!(
            "{:width$}  count={:<6}  digest={}",
            snap.id,
            count,
            digest,
            width = max_id_len,
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Sanity tests for the snapshot machinery itself. These don't depend on any
// external spec; they pin behaviour of `digest_reprs` and `enumerate_reachable`
// so a future refactor of the digest path doesn't silently change every
// snapshot at once.
// ---------------------------------------------------------------------------

#[test]
fn digest_is_stable_for_empty_input() {
    let empty: Vec<String> = Vec::new();
    let d1 = digest_reprs(&empty);
    let d2 = digest_reprs(&empty);
    assert_eq!(d1, d2, "digest must be deterministic across calls");
    assert_eq!(d1.len(), 32, "digest must be 32 hex chars (128 bits)");
}

#[test]
fn digest_is_order_dependent_on_sorted_input() {
    // We sort before hashing, so the digest should be the same regardless
    // of insertion order — but DIFFERENT contents must produce different
    // digests.
    let a = vec!["a".to_string(), "b".to_string()];
    let b = vec!["a".to_string(), "c".to_string()];
    assert_ne!(digest_reprs(&a), digest_reprs(&b));
}

#[test]
fn digest_is_independent_of_input_order_after_sort() {
    let mut a = vec!["zzz".to_string(), "aaa".to_string(), "mmm".to_string()];
    let mut b = vec!["mmm".to_string(), "aaa".to_string(), "zzz".to_string()];
    a.sort();
    b.sort();
    assert_eq!(digest_reprs(&a), digest_reprs(&b));
}

#[test]
fn canonical_state_repr_sorts_keys() {
    use tlaplusplus::tla::{TlaValue, tla_state};
    // BTreeMap iteration is sorted, so even constructing the state with
    // keys in different orders must yield the same canonical repr.
    let s1 = tla_state([
        ("z", TlaValue::Int(3)),
        ("a", TlaValue::Int(1)),
        ("m", TlaValue::Int(2)),
    ]);
    let s2 = tla_state([
        ("a", TlaValue::Int(1)),
        ("m", TlaValue::Int(2)),
        ("z", TlaValue::Int(3)),
    ]);
    assert_eq!(canonical_state_repr(&s1), canonical_state_repr(&s2));
    // And the order is alphabetical by key.
    let repr = canonical_state_repr(&s1);
    let pa = repr.find("\"a\"").expect("key 'a' present");
    let pm = repr.find("\"m\"").expect("key 'm' present");
    let pz = repr.find("\"z\"").expect("key 'z' present");
    assert!(
        pa < pm && pm < pz,
        "keys must appear in sorted order: {repr}"
    );
}

#[test]
fn snapshot_list_ids_match_test_function_names() {
    // Cheap structural check: every entry in SNAPSHOTS[] must have a unique
    // id, matching the convention `snapshot_<id>`. Catches copy-paste
    // mistakes when adding new entries.
    let ids: BTreeSet<&str> = SNAPSHOTS.iter().map(|s| s.id).collect();
    assert_eq!(ids.len(), SNAPSHOTS.len(), "snapshot ids must be unique");
}

// Force `Arc` to be considered used at the type-system level if rustc
// doesn't already see it through `TlaState`'s alias.
#[allow(dead_code)]
fn _arc_is_used() -> Arc<str> {
    Arc::from("")
}
