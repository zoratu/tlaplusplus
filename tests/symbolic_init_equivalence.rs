//! Property-based equivalence between symbolic Init enumeration and
//! brute-force enumeration for filtered record-set Init expressions.
//!
//! Only runs when the `symbolic-init` feature is enabled — without it
//! the symbolic path returns `None` for every input.
//!
//! For each randomly generated triple
//! `(field_specs, predicate, expected_set)`, we:
//! 1. Brute-force enumerate the predicate against the record domain to
//!    compute the expected truth set.
//! 2. Call `try_symbolic_record_set_enumerate` to get the symbolic set.
//! 3. Assert the two sets agree (when symbolic returns Some).

#![cfg(feature = "symbolic-init")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use proptest::prelude::*;
use tlaplusplus::tla::eval::EvalContext;
use tlaplusplus::tla::module::TlaModuleInstance;
use tlaplusplus::tla::symbolic_init::{
    try_symbolic_function_set_enumerate, try_symbolic_record_set_enumerate,
};
use tlaplusplus::tla::value::TlaValue;

fn arb_int_domain() -> impl Strategy<Value = Vec<TlaValue>> {
    (0i64..6, 1i64..6).prop_map(|(lo, len)| (lo..lo + len).map(TlaValue::Int).collect::<Vec<_>>())
}

fn arb_two_field_int_spec() -> impl Strategy<Value = (Vec<(String, Vec<TlaValue>)>, &'static str)> {
    arb_int_domain()
        .prop_flat_map(|d1| arb_int_domain().prop_map(move |d2| (d1.clone(), d2)))
        .prop_map(|(d1, d2)| {
            let specs = vec![("a".to_string(), d1), ("b".to_string(), d2)];
            (specs, "tup")
        })
}

fn arb_predicate(_field_count: usize) -> impl Strategy<Value = String> {
    prop_oneof![
        Just("TRUE".to_string()),
        Just("tup.a = tup.b".to_string()),
        Just("tup.a # tup.b".to_string()),
        Just("tup.a < tup.b".to_string()),
        Just("tup.a <= tup.b".to_string()),
        Just("tup.a + tup.b = 5".to_string()),
        Just("tup.a + tup.b <= 5".to_string()),
        Just("tup.a * 2 = tup.b".to_string()),
        Just("tup.a + 1 = tup.b".to_string()),
        Just("tup.a = 0 \\/ tup.b = 0".to_string()),
        Just("tup.a < 3 /\\ tup.b > 1".to_string()),
        Just("~(tup.a = tup.b)".to_string()),
        Just("tup.a \\in {0, 1, 2}".to_string()),
        Just("tup.a \\in 0..3 /\\ tup.b \\in 2..4".to_string()),
        Just("tup.a # tup.b /\\ tup.a + tup.b = 4".to_string()),
        Just("(tup.a + tup.b = 3) \\/ (tup.a + tup.b = 5)".to_string()),
    ]
}

fn make_ctx<'a>(
    state: &'a tlaplusplus::tla::value::TlaState,
    defs: &'a BTreeMap<String, tlaplusplus::tla::module::TlaDefinition>,
    instances: &'a BTreeMap<String, TlaModuleInstance>,
) -> EvalContext<'a> {
    EvalContext::with_definitions_and_instances(state, defs, instances)
}

/// Brute-force evaluator: directly encode the predicate's semantics in
/// Rust for the small subset we test, so this acts as the ground truth.
/// (Avoids depending on the in-tree TLA+ predicate evaluator for the
/// equivalence test, eliminating a circular dependency.)
fn brute_force(field_specs: &[(String, Vec<TlaValue>)], pred: &str) -> BTreeSet<TlaValue> {
    let mut out = BTreeSet::new();
    let dom_a = &field_specs[0].1;
    let dom_b = &field_specs[1].1;
    for av in dom_a {
        for bv in dom_b {
            let a = av.as_int().unwrap();
            let b = bv.as_int().unwrap();
            let ok = match pred {
                "TRUE" => true,
                "tup.a = tup.b" => a == b,
                "tup.a # tup.b" => a != b,
                "tup.a < tup.b" => a < b,
                "tup.a <= tup.b" => a <= b,
                "tup.a + tup.b = 5" => a + b == 5,
                "tup.a + tup.b <= 5" => a + b <= 5,
                "tup.a * 2 = tup.b" => a * 2 == b,
                "tup.a + 1 = tup.b" => a + 1 == b,
                "tup.a = 0 \\/ tup.b = 0" => a == 0 || b == 0,
                "tup.a < 3 /\\ tup.b > 1" => a < 3 && b > 1,
                "~(tup.a = tup.b)" => a != b,
                "tup.a \\in {0, 1, 2}" => (0..=2).contains(&a),
                "tup.a \\in 0..3 /\\ tup.b \\in 2..4" => {
                    (0..=3).contains(&a) && (2..=4).contains(&b)
                }
                "tup.a # tup.b /\\ tup.a + tup.b = 4" => a != b && a + b == 4,
                "(tup.a + tup.b = 3) \\/ (tup.a + tup.b = 5)" => a + b == 3 || a + b == 5,
                _ => unreachable!("unhandled predicate: {pred}"),
            };
            if ok {
                let mut rec = BTreeMap::new();
                rec.insert("a".to_string(), TlaValue::Int(a));
                rec.insert("b".to_string(), TlaValue::Int(b));
                out.insert(TlaValue::Record(Arc::new(rec)));
            }
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn symbolic_matches_brute_force_two_int_fields(
        spec in arb_two_field_int_spec(),
        pred in arb_predicate(2),
    ) {
        let (field_specs, var_name) = spec;
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);

        let symbolic = try_symbolic_record_set_enumerate(&pred, var_name, &field_specs, &ctx);
        let bf = brute_force(&field_specs, &pred);

        // Symbolic may legitimately return None for shapes outside the
        // supported subset. When it returns Some, it must agree exactly.
        if let Some(sym) = symbolic {
            let sym_set: BTreeSet<TlaValue> = sym.into_iter().collect();
            prop_assert_eq!(
                &sym_set, &bf,
                "symbolic vs brute_force mismatch on predicate `{}`",
                pred
            );
        }
    }
}

// ============================================================================
// T5.1 — sequence-set / function-set equivalence (symbolic vs brute force)
// ============================================================================

fn arb_seq_predicate(_seq_len: usize) -> impl Strategy<Value = String> {
    prop_oneof![
        Just("TRUE".to_string()),
        Just("p[1] = 1".to_string()),
        Just("p[1] # p[2]".to_string()),
        Just("p[1] + p[2] = 4".to_string()),
        Just("p[1] < p[2]".to_string()),
        Just("p[2] \\in {1,2}".to_string()),
        Just("p[1] # p[2] /\\ p[2] # p[3]".to_string()),
        Just("p[1] # p[2] /\\ p[1] # p[3] /\\ p[2] # p[3]".to_string()),
        Just("p[3] \\in {1,2,3} \\ {p[1]}".to_string()),
        Just("p[3] \\in {1,2,3} \\ {p[1], p[2]}".to_string()),
        Just("p[1] = 1 \\/ p[2] = 1".to_string()),
        Just("(p[1] + p[3] = 4) /\\ (p[2] = 2)".to_string()),
    ]
}

/// Brute-force predicate evaluator over int sequences of length n in 1..=k.
fn brute_force_seqs(n: usize, range_max: i64, pred: &str) -> std::collections::BTreeSet<TlaValue> {
    let mut out = std::collections::BTreeSet::new();
    let mut idx = vec![1i64; n];
    loop {
        let satisfies = match pred {
            "TRUE" => true,
            "p[1] = 1" => idx[0] == 1,
            "p[1] # p[2]" => n >= 2 && idx[0] != idx[1],
            "p[1] + p[2] = 4" => n >= 2 && idx[0] + idx[1] == 4,
            "p[1] < p[2]" => n >= 2 && idx[0] < idx[1],
            "p[2] \\in {1,2}" => n >= 2 && (idx[1] == 1 || idx[1] == 2),
            "p[1] # p[2] /\\ p[2] # p[3]" => n >= 3 && idx[0] != idx[1] && idx[1] != idx[2],
            "p[1] # p[2] /\\ p[1] # p[3] /\\ p[2] # p[3]" => {
                n >= 3 && idx[0] != idx[1] && idx[0] != idx[2] && idx[1] != idx[2]
            }
            "p[3] \\in {1,2,3} \\ {p[1]}" => {
                n >= 3 && idx[2] != idx[0] && (1..=3).contains(&idx[2])
            }
            "p[3] \\in {1,2,3} \\ {p[1], p[2]}" => {
                n >= 3
                    && idx[2] != idx[0]
                    && idx[2] != idx[1]
                    && (1..=3).contains(&idx[2])
            }
            "p[1] = 1 \\/ p[2] = 1" => n >= 2 && (idx[0] == 1 || idx[1] == 1),
            "(p[1] + p[3] = 4) /\\ (p[2] = 2)" => {
                n >= 3 && idx[0] + idx[2] == 4 && idx[1] == 2
            }
            _ => unreachable!("unhandled seq predicate {pred}"),
        };
        if satisfies {
            let s: Vec<TlaValue> = idx.iter().map(|&v| TlaValue::Int(v)).collect();
            out.insert(TlaValue::Seq(Arc::new(s)));
        }
        // Increment odometer (1..=range_max).
        let mut carry = true;
        for i in (0..n).rev() {
            if carry {
                idx[i] += 1;
                if idx[i] <= range_max {
                    carry = false;
                } else {
                    idx[i] = 1;
                }
            }
        }
        if carry {
            break;
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn symbolic_sequence_matches_brute_force_int_range(
        n in 2usize..=4,
        range_max in 2i64..=4,
        pred in arb_seq_predicate(3),
    ) {
        let state = BTreeMap::new();
        let defs = BTreeMap::new();
        let instances = BTreeMap::new();
        let ctx = make_ctx(&state, &defs, &instances);
        let range: Vec<TlaValue> = (1..=range_max).map(TlaValue::Int).collect();

        let symbolic = try_symbolic_function_set_enumerate(&pred, "p", n, &range, &ctx);
        let bf = brute_force_seqs(n, range_max, &pred);

        if let Some(sym) = symbolic {
            let sym_set: BTreeSet<TlaValue> = sym.into_iter().collect();
            prop_assert_eq!(
                &sym_set, &bf,
                "symbolic vs brute_force mismatch (n={}, range={}, pred={})",
                n, range_max, pred
            );
        }
    }
}

#[test]
fn empty_intersection_predicate_returns_empty_set() {
    let state = BTreeMap::new();
    let defs = BTreeMap::new();
    let instances = BTreeMap::new();
    let ctx = make_ctx(&state, &defs, &instances);
    let fields = vec![
        (
            "a".to_string(),
            (0..3).map(TlaValue::Int).collect::<Vec<_>>(),
        ),
        (
            "b".to_string(),
            (0..3).map(TlaValue::Int).collect::<Vec<_>>(),
        ),
    ];
    let pred = "FALSE";
    let result = try_symbolic_record_set_enumerate(pred, "tup", &fields, &ctx)
        .expect("symbolic should succeed on FALSE");
    assert!(result.is_empty());
}
