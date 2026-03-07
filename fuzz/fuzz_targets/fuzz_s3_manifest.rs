#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to parse as JSON manifest
    if let Ok(input) = std::str::from_utf8(data) {
        // Try parsing as S3 checkpoint manifest
        let _ = serde_json::from_str::<serde_json::Value>(input);
    }
});
