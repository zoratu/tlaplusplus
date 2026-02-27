#![no_main]

use libfuzzer_sys::fuzz_target;
use tlaplusplus::tla::cfg::parse_tla_config;

fuzz_target!(|data: &[u8]| {
    // Try to parse the input as a TLA+ config file
    // We don't care about the result - just that it doesn't panic
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = parse_tla_config(input);
    }
});
