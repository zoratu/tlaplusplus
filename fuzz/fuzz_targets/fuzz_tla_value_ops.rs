#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeSet;
use std::sync::Arc;
use tlaplusplus::tla::value::TlaValue;

/// Arbitrary TlaValue for fuzzing
#[derive(Debug, Clone, Arbitrary)]
enum FuzzValue {
    Bool(bool),
    Int(i64),
    String(String),
    ModelValue(String),
}

impl From<FuzzValue> for TlaValue {
    fn from(fv: FuzzValue) -> Self {
        match fv {
            FuzzValue::Bool(b) => TlaValue::Bool(b),
            FuzzValue::Int(i) => TlaValue::Int(i),
            FuzzValue::String(s) => TlaValue::String(s),
            FuzzValue::ModelValue(s) => TlaValue::ModelValue(s),
        }
    }
}

/// Arbitrary set operations
#[derive(Debug, Arbitrary)]
enum SetOp {
    Union,
    Intersection,
    Minus,
}

/// Fuzz input: two sets and an operation
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    set_a: Vec<FuzzValue>,
    set_b: Vec<FuzzValue>,
    op: SetOp,
}

fuzz_target!(|input: FuzzInput| {
    // Build TlaValue sets from fuzzer input
    let set_a: BTreeSet<TlaValue> = input.set_a.into_iter().map(|v| v.into()).collect();
    let set_b: BTreeSet<TlaValue> = input.set_b.into_iter().map(|v| v.into()).collect();

    let a = TlaValue::Set(Arc::new(set_a));
    let b = TlaValue::Set(Arc::new(set_b));

    // Perform the operation - should never panic
    let result = match input.op {
        SetOp::Union => a.set_union(&b),
        SetOp::Intersection => a.set_intersection(&b),
        SetOp::Minus => a.set_minus(&b),
    };

    // Verify the result is a valid set
    if let Ok(r) = result {
        let _ = r.len();
        let _ = r.is_empty();
    }
});
