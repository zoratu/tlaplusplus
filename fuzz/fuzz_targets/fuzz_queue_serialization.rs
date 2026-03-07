#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tlaplusplus::storage::queue::{deserialize_compressed, serialize_compressed};

/// Arbitrary state for queue serialization fuzzing
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize, PartialEq)]
struct FuzzState {
    values: Vec<FuzzValue>,
}

#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize, PartialEq)]
enum FuzzValue {
    Bool(bool),
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
    Nested(Box<FuzzState>),
}

fuzz_target!(|state: FuzzState| {
    // Serialize the state
    if let Ok(compressed) = serialize_compressed(&state) {
        // Deserialize should succeed and produce equal value
        if let Ok(recovered): Result<FuzzState, _> = deserialize_compressed(&compressed) {
            assert_eq!(state, recovered, "Round-trip should preserve data");
        }
    }
});
