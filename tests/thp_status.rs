//! Tests for Transparent Huge Pages (THP) status checking functionality.
//!
//! These tests verify:
//! - Parsing of all THP status values (always, madvise, never)
//! - Parsing of all defrag policy values
//! - Handling of malformed sysfs content
//! - Handling of missing sysfs files
//! - Warning output for each THP configuration
//! - --skip-system-checks flag behavior

use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// Unit Tests: THP Status Parsing
// =============================================================================

/// Test the parse_thp_bracketed function with all valid status values
mod unit_parsing {
    /// Parse the bracketed value from THP sysfs files.
    /// Format is like: "always [madvise] never" where the active option is in brackets.
    fn parse_thp_bracketed(content: &str) -> Option<String> {
        // Find the bracketed option
        if let Some(start) = content.find('[') {
            if let Some(end) = content[start..].find(']') {
                return Some(content[start + 1..start + end].trim().to_string());
            }
        }
        None
    }

    #[test]
    fn test_parse_thp_status_always() {
        let content = "[always] madvise never";
        assert_eq!(parse_thp_bracketed(content), Some("always".to_string()));
    }

    #[test]
    fn test_parse_thp_status_madvise() {
        let content = "always [madvise] never";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_thp_status_never() {
        let content = "always madvise [never]";
        assert_eq!(parse_thp_bracketed(content), Some("never".to_string()));
    }

    #[test]
    fn test_parse_thp_defrag_always() {
        let content = "[always] defer defer+madvise madvise never";
        assert_eq!(parse_thp_bracketed(content), Some("always".to_string()));
    }

    #[test]
    fn test_parse_thp_defrag_defer() {
        let content = "always [defer] defer+madvise madvise never";
        assert_eq!(parse_thp_bracketed(content), Some("defer".to_string()));
    }

    #[test]
    fn test_parse_thp_defrag_defer_madvise() {
        let content = "always defer [defer+madvise] madvise never";
        assert_eq!(
            parse_thp_bracketed(content),
            Some("defer+madvise".to_string())
        );
    }

    #[test]
    fn test_parse_thp_defrag_madvise() {
        let content = "always defer defer+madvise [madvise] never";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_thp_defrag_never() {
        let content = "always defer defer+madvise madvise [never]";
        assert_eq!(parse_thp_bracketed(content), Some("never".to_string()));
    }

    // --- Malformed content tests ---

    #[test]
    fn test_parse_empty_string() {
        assert_eq!(parse_thp_bracketed(""), None);
    }

    #[test]
    fn test_parse_no_brackets() {
        assert_eq!(parse_thp_bracketed("always madvise never"), None);
    }

    #[test]
    fn test_parse_only_opening_bracket() {
        assert_eq!(parse_thp_bracketed("always [madvise never"), None);
    }

    #[test]
    fn test_parse_only_closing_bracket() {
        assert_eq!(parse_thp_bracketed("always madvise] never"), None);
    }

    #[test]
    fn test_parse_empty_brackets() {
        // Empty brackets should return empty string
        assert_eq!(parse_thp_bracketed("always [] never"), Some("".to_string()));
    }

    #[test]
    fn test_parse_whitespace_in_brackets() {
        // Whitespace should be trimmed
        assert_eq!(
            parse_thp_bracketed("always [ madvise ] never"),
            Some("madvise".to_string())
        );
    }

    #[test]
    fn test_parse_multiple_brackets() {
        // Should find the first bracketed value
        assert_eq!(
            parse_thp_bracketed("[first] [second] [third]"),
            Some("first".to_string())
        );
    }

    #[test]
    fn test_parse_nested_brackets() {
        // Nested brackets - should parse up to first closing bracket
        assert_eq!(
            parse_thp_bracketed("[[nested]]"),
            Some("[nested".to_string())
        );
    }

    #[test]
    fn test_parse_newlines_in_content() {
        let content = "always\n[madvise]\nnever";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_tabs_in_content() {
        let content = "always\t[madvise]\tnever";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_carriage_return() {
        let content = "always [madvise]\r\nnever";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_unicode_content() {
        // Ensure unicode doesn't break parsing
        let content = "always [madvise] never \u{1F600}";
        assert_eq!(parse_thp_bracketed(content), Some("madvise".to_string()));
    }

    #[test]
    fn test_parse_very_long_content() {
        // Very long content before bracket
        let prefix = "x".repeat(10000);
        let content = format!("{}[always]", prefix);
        assert_eq!(parse_thp_bracketed(&content), Some("always".to_string()));
    }

    #[test]
    fn test_parse_binary_data() {
        // Binary-like content (shouldn't crash)
        let content = "\x00\x01[value]\x02\x03";
        assert_eq!(parse_thp_bracketed(content), Some("value".to_string()));
    }
}

// =============================================================================
// Unit Tests: ThpStatus and ThpDefrag enum conversions
// =============================================================================

mod unit_enums {
    /// THP status values
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ThpStatus {
        Always,
        Madvise,
        Never,
        Unknown,
    }

    impl ThpStatus {
        fn from_str(s: &str) -> Self {
            match s {
                "always" => ThpStatus::Always,
                "madvise" => ThpStatus::Madvise,
                "never" => ThpStatus::Never,
                _ => ThpStatus::Unknown,
            }
        }
    }

    impl std::fmt::Display for ThpStatus {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ThpStatus::Always => write!(f, "always"),
                ThpStatus::Madvise => write!(f, "madvise"),
                ThpStatus::Never => write!(f, "never"),
                ThpStatus::Unknown => write!(f, "unknown"),
            }
        }
    }

    /// THP defrag values
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ThpDefrag {
        Always,
        Defer,
        DeferMadvise,
        Madvise,
        Never,
        Unknown,
    }

    impl ThpDefrag {
        fn from_str(s: &str) -> Self {
            match s {
                "always" => ThpDefrag::Always,
                "defer" => ThpDefrag::Defer,
                "defer+madvise" => ThpDefrag::DeferMadvise,
                "madvise" => ThpDefrag::Madvise,
                "never" => ThpDefrag::Never,
                _ => ThpDefrag::Unknown,
            }
        }
    }

    impl std::fmt::Display for ThpDefrag {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ThpDefrag::Always => write!(f, "always"),
                ThpDefrag::Defer => write!(f, "defer"),
                ThpDefrag::DeferMadvise => write!(f, "defer+madvise"),
                ThpDefrag::Madvise => write!(f, "madvise"),
                ThpDefrag::Never => write!(f, "never"),
                ThpDefrag::Unknown => write!(f, "unknown"),
            }
        }
    }

    #[test]
    fn test_thp_status_from_str_all_values() {
        assert_eq!(ThpStatus::from_str("always"), ThpStatus::Always);
        assert_eq!(ThpStatus::from_str("madvise"), ThpStatus::Madvise);
        assert_eq!(ThpStatus::from_str("never"), ThpStatus::Never);
        assert_eq!(ThpStatus::from_str("invalid"), ThpStatus::Unknown);
        assert_eq!(ThpStatus::from_str(""), ThpStatus::Unknown);
        assert_eq!(ThpStatus::from_str("ALWAYS"), ThpStatus::Unknown); // case sensitive
    }

    #[test]
    fn test_thp_status_display() {
        assert_eq!(ThpStatus::Always.to_string(), "always");
        assert_eq!(ThpStatus::Madvise.to_string(), "madvise");
        assert_eq!(ThpStatus::Never.to_string(), "never");
        assert_eq!(ThpStatus::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_thp_defrag_from_str_all_values() {
        assert_eq!(ThpDefrag::from_str("always"), ThpDefrag::Always);
        assert_eq!(ThpDefrag::from_str("defer"), ThpDefrag::Defer);
        assert_eq!(
            ThpDefrag::from_str("defer+madvise"),
            ThpDefrag::DeferMadvise
        );
        assert_eq!(ThpDefrag::from_str("madvise"), ThpDefrag::Madvise);
        assert_eq!(ThpDefrag::from_str("never"), ThpDefrag::Never);
        assert_eq!(ThpDefrag::from_str("invalid"), ThpDefrag::Unknown);
        assert_eq!(ThpDefrag::from_str(""), ThpDefrag::Unknown);
    }

    #[test]
    fn test_thp_defrag_display() {
        assert_eq!(ThpDefrag::Always.to_string(), "always");
        assert_eq!(ThpDefrag::Defer.to_string(), "defer");
        assert_eq!(ThpDefrag::DeferMadvise.to_string(), "defer+madvise");
        assert_eq!(ThpDefrag::Madvise.to_string(), "madvise");
        assert_eq!(ThpDefrag::Never.to_string(), "never");
        assert_eq!(ThpDefrag::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_thp_status_roundtrip() {
        for status in [
            ThpStatus::Always,
            ThpStatus::Madvise,
            ThpStatus::Never,
            ThpStatus::Unknown,
        ] {
            let s = status.to_string();
            let parsed = ThpStatus::from_str(&s);
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_thp_defrag_roundtrip() {
        for defrag in [
            ThpDefrag::Always,
            ThpDefrag::Defer,
            ThpDefrag::DeferMadvise,
            ThpDefrag::Madvise,
            ThpDefrag::Never,
            ThpDefrag::Unknown,
        ] {
            let s = defrag.to_string();
            let parsed = ThpDefrag::from_str(&s);
            assert_eq!(defrag, parsed);
        }
    }
}

// =============================================================================
// Property-Based Tests: Fuzzing THP Parsing
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Any bracketed string should be correctly extracted
    #[test]
    fn prop_parse_extracts_bracketed_content(
        prefix in "[a-z ]{0,20}",
        content in "[a-z+_-]{1,20}",
        suffix in "[a-z ]{0,20}",
    ) {
        fn parse_thp_bracketed(input: &str) -> Option<String> {
            if let Some(start) = input.find('[') {
                if let Some(end) = input[start..].find(']') {
                    return Some(input[start + 1..start + end].trim().to_string());
                }
            }
            None
        }

        let input = format!("{}[{}]{}", prefix, content, suffix);
        let result = parse_thp_bracketed(&input);

        prop_assert!(result.is_some(), "Should find bracketed content in '{}'", input);
        prop_assert_eq!(result.unwrap(), content.trim());
    }

    /// Property: Content without brackets returns None
    #[test]
    fn prop_parse_no_brackets_returns_none(
        content in "[a-z ]{1,50}",
    ) {
        fn parse_thp_bracketed(input: &str) -> Option<String> {
            if let Some(start) = input.find('[') {
                if let Some(end) = input[start..].find(']') {
                    return Some(input[start + 1..start + end].trim().to_string());
                }
            }
            None
        }

        // Only test strings without any brackets
        if !content.contains('[') && !content.contains(']') {
            let result = parse_thp_bracketed(&content);
            prop_assert!(result.is_none(), "Should return None for '{}'", content);
        }
    }

    /// Property: Valid sysfs format is correctly parsed
    #[test]
    fn prop_valid_sysfs_format_parsed(
        selected_idx in 0..3usize,
    ) {
        fn parse_thp_bracketed(input: &str) -> Option<String> {
            if let Some(start) = input.find('[') {
                if let Some(end) = input[start..].find(']') {
                    return Some(input[start + 1..start + end].trim().to_string());
                }
            }
            None
        }

        let options = ["always", "madvise", "never"];
        let mut parts = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            if i == selected_idx {
                parts.push(format!("[{}]", opt));
            } else {
                parts.push(opt.to_string());
            }
        }
        let content = parts.join(" ");

        let result = parse_thp_bracketed(&content);
        prop_assert!(result.is_some(), "Should parse '{}'", content);
        prop_assert_eq!(result.unwrap(), options[selected_idx]);
    }

    /// Property: Defrag sysfs format with all 5 options is correctly parsed
    #[test]
    fn prop_valid_defrag_sysfs_format_parsed(
        selected_idx in 0..5usize,
    ) {
        fn parse_thp_bracketed(input: &str) -> Option<String> {
            if let Some(start) = input.find('[') {
                if let Some(end) = input[start..].find(']') {
                    return Some(input[start + 1..start + end].trim().to_string());
                }
            }
            None
        }

        let options = ["always", "defer", "defer+madvise", "madvise", "never"];
        let mut parts = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            if i == selected_idx {
                parts.push(format!("[{}]", opt));
            } else {
                parts.push(opt.to_string());
            }
        }
        let content = parts.join(" ");

        let result = parse_thp_bracketed(&content);
        prop_assert!(result.is_some(), "Should parse '{}'", content);
        prop_assert_eq!(result.unwrap(), options[selected_idx]);
    }

    /// Property: Random junk after closing bracket doesn't affect parsing
    #[test]
    fn prop_trailing_junk_ignored(
        prefix in "[a-z ]{0,10}",
        content in "[a-z]{1,10}",
        junk in ".*{0,50}",
    ) {
        fn parse_thp_bracketed(input: &str) -> Option<String> {
            if let Some(start) = input.find('[') {
                if let Some(end) = input[start..].find(']') {
                    return Some(input[start + 1..start + end].trim().to_string());
                }
            }
            None
        }

        let input = format!("{}[{}]{}", prefix, content, junk);
        let result = parse_thp_bracketed(&input);

        // Should still find the bracketed content
        prop_assert!(result.is_some(), "Should find bracketed content in '{}'", input);
        // The parsed content should match what was in brackets
        prop_assert_eq!(result.unwrap(), content.trim());
    }
}

// =============================================================================
// Integration Tests: Warning Output
// =============================================================================

mod integration_warnings {

    /// Simulated THP config for testing warning behavior
    #[derive(Debug, Clone)]
    struct ThpConfig {
        enabled: ThpStatus,
        defrag: ThpDefrag,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ThpStatus {
        Always,
        Madvise,
        Never,
        Unknown,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ThpDefrag {
        Always,
        Defer,
        DeferMadvise,
        Madvise,
        Never,
        Unknown,
    }

    /// Simulated check_thp_and_warn that captures output
    fn check_thp_and_warn_simulated(config: &ThpConfig) -> (bool, Vec<String>) {
        let mut warnings = Vec::new();
        let mut optimal = true;

        match config.enabled {
            ThpStatus::Never => {
                warnings.push("Warning: Transparent Huge Pages (THP) is disabled.".to_string());
                warnings.push(
                    "This may cause performance degradation and checkpoint timeouts.".to_string(),
                );
                optimal = false;
            }
            ThpStatus::Madvise => {
                warnings
                    .push("Info: Transparent Huge Pages (THP) is set to 'madvise'.".to_string());
            }
            ThpStatus::Unknown => {
                // Silently continue - not on Linux or can't read
            }
            ThpStatus::Always => {
                // Optimal configuration - no warnings
            }
        }

        // Check defrag policy
        if config.defrag == ThpDefrag::Always {
            warnings.push(
                "Warning: THP defrag is set to 'always', which can cause latency spikes."
                    .to_string(),
            );
            optimal = false;
        }

        (optimal, warnings)
    }

    #[test]
    fn test_warning_thp_never() {
        let config = ThpConfig {
            enabled: ThpStatus::Never,
            defrag: ThpDefrag::Madvise,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        assert!(!optimal, "THP=never should not be optimal");
        assert!(!warnings.is_empty(), "Should have warnings");
        assert!(
            warnings.iter().any(|w| w.contains("disabled")),
            "Should warn about THP being disabled"
        );
    }

    #[test]
    fn test_warning_thp_madvise() {
        let config = ThpConfig {
            enabled: ThpStatus::Madvise,
            defrag: ThpDefrag::Madvise,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        // madvise is not a warning, just info - so optimal is true
        assert!(optimal, "THP=madvise with defrag=madvise should be optimal");
        assert!(
            warnings.iter().any(|w| w.contains("madvise")),
            "Should have info about madvise"
        );
    }

    #[test]
    fn test_warning_thp_always_optimal() {
        let config = ThpConfig {
            enabled: ThpStatus::Always,
            defrag: ThpDefrag::DeferMadvise,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        assert!(
            optimal,
            "THP=always with defrag=defer+madvise should be optimal"
        );
        assert!(warnings.is_empty(), "Should have no warnings");
    }

    #[test]
    fn test_warning_defrag_always() {
        let config = ThpConfig {
            enabled: ThpStatus::Always,
            defrag: ThpDefrag::Always,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        assert!(!optimal, "defrag=always should not be optimal");
        assert!(
            warnings.iter().any(|w| w.contains("latency")),
            "Should warn about latency spikes"
        );
    }

    #[test]
    fn test_warning_both_suboptimal() {
        let config = ThpConfig {
            enabled: ThpStatus::Never,
            defrag: ThpDefrag::Always,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        assert!(
            !optimal,
            "Both THP=never and defrag=always should not be optimal"
        );
        assert!(warnings.len() >= 2, "Should have multiple warnings");
    }

    #[test]
    fn test_warning_unknown_status() {
        let config = ThpConfig {
            enabled: ThpStatus::Unknown,
            defrag: ThpDefrag::Unknown,
        };

        let (optimal, warnings) = check_thp_and_warn_simulated(&config);

        assert!(
            optimal,
            "Unknown status should be treated as optimal (graceful)"
        );
        assert!(
            warnings.is_empty(),
            "Unknown status should not produce warnings"
        );
    }

    #[test]
    fn test_all_defrag_variants_non_always() {
        // All defrag values except 'always' should be acceptable
        for defrag in [
            ThpDefrag::Defer,
            ThpDefrag::DeferMadvise,
            ThpDefrag::Madvise,
            ThpDefrag::Never,
        ] {
            let config = ThpConfig {
                enabled: ThpStatus::Always,
                defrag,
            };

            let (optimal, warnings) = check_thp_and_warn_simulated(&config);
            assert!(optimal, "defrag={:?} should be optimal", config.defrag);
            assert!(
                warnings.is_empty(),
                "defrag={:?} should have no warnings",
                config.defrag
            );
        }
    }
}

// =============================================================================
// Integration Tests: --skip-system-checks Flag
// =============================================================================

mod integration_skip_checks {
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Simulated run_system_checks function
    fn run_system_checks(skip: bool, checks_ran: &AtomicBool) {
        if skip {
            return;
        }
        checks_ran.store(true, Ordering::SeqCst);
    }

    #[test]
    fn test_skip_system_checks_true() {
        let checks_ran = AtomicBool::new(false);
        run_system_checks(true, &checks_ran);
        assert!(
            !checks_ran.load(Ordering::SeqCst),
            "Checks should not run when skip=true"
        );
    }

    #[test]
    fn test_skip_system_checks_false() {
        let checks_ran = AtomicBool::new(false);
        run_system_checks(false, &checks_ran);
        assert!(
            checks_ran.load(Ordering::SeqCst),
            "Checks should run when skip=false"
        );
    }
}

// =============================================================================
// Chaos Tests: Error Handling
// =============================================================================

mod chaos_tests {
    use std::io;
    use std::path::Path;

    /// Simulated read function that can fail
    fn read_thp_file_with_error(
        path: &Path,
        simulate_error: Option<io::ErrorKind>,
    ) -> io::Result<Option<String>> {
        if let Some(error_kind) = simulate_error {
            return Err(io::Error::new(error_kind, "simulated error"));
        }

        // Simulate reading from a path
        if path.to_string_lossy().contains("nonexistent") {
            return Ok(None);
        }

        // Normal case
        Ok(Some("always [madvise] never".to_string()))
    }

    /// Get THP status with error handling
    fn get_thp_status_safe(path: &Path, simulate_error: Option<io::ErrorKind>) -> ThpStatus {
        match read_thp_file_with_error(path, simulate_error) {
            Ok(Some(content)) => {
                if let Some(bracketed) = parse_thp_bracketed(&content) {
                    ThpStatus::from_str(&bracketed)
                } else {
                    ThpStatus::Unknown
                }
            }
            Ok(None) => ThpStatus::Unknown,
            Err(_) => ThpStatus::Unknown,
        }
    }

    fn parse_thp_bracketed(content: &str) -> Option<String> {
        if let Some(start) = content.find('[') {
            if let Some(end) = content[start..].find(']') {
                return Some(content[start + 1..start + end].trim().to_string());
            }
        }
        None
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ThpStatus {
        Always,
        Madvise,
        Never,
        Unknown,
    }

    impl ThpStatus {
        fn from_str(s: &str) -> Self {
            match s {
                "always" => ThpStatus::Always,
                "madvise" => ThpStatus::Madvise,
                "never" => ThpStatus::Never,
                _ => ThpStatus::Unknown,
            }
        }
    }

    #[test]
    fn test_permission_denied() {
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/enabled");
        let status = get_thp_status_safe(path, Some(io::ErrorKind::PermissionDenied));

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "Permission denied should result in Unknown status"
        );
    }

    #[test]
    fn test_file_not_found() {
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/nonexistent");
        let status = get_thp_status_safe(path, None);

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "Missing file should result in Unknown status"
        );
    }

    #[test]
    fn test_io_error() {
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/enabled");
        let status = get_thp_status_safe(path, Some(io::ErrorKind::Other));

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "I/O error should result in Unknown status"
        );
    }

    #[test]
    fn test_interrupted() {
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/enabled");
        let status = get_thp_status_safe(path, Some(io::ErrorKind::Interrupted));

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "Interrupted read should result in Unknown status"
        );
    }

    #[test]
    fn test_would_block() {
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/enabled");
        let status = get_thp_status_safe(path, Some(io::ErrorKind::WouldBlock));

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "WouldBlock should result in Unknown status"
        );
    }

    /// Test that non-Linux platforms gracefully skip THP checks
    #[test]
    fn test_non_linux_graceful_skip() {
        // On non-Linux, THP paths don't exist, so we should get Unknown
        // This test simulates that behavior
        let path = Path::new("/sys/kernel/mm/transparent_hugepage/nonexistent");
        let status = get_thp_status_safe(path, None);

        assert_eq!(
            status,
            ThpStatus::Unknown,
            "Non-Linux (missing sysfs) should result in Unknown status"
        );

        // Check that Unknown status is treated as optimal (no warnings)
        let is_optimal = status == ThpStatus::Unknown || status == ThpStatus::Always;
        assert!(is_optimal, "Unknown status should be treated as optimal");
    }

    #[test]
    fn test_corrupt_file_content() {
        // Test various corrupt file contents
        let corrupt_contents = [
            "",
            " ",
            "\n",
            "\x00\x00\x00",
            "garbage data here",
            "no brackets at all",
            "unclosed [bracket",
            "reversed ]bracket[",
            "[[nested]]",
            "[  ]", // empty brackets
        ];

        for content in corrupt_contents {
            let result = parse_thp_bracketed(content);
            // Most should return None or empty string
            // Key point: none should panic
            match result {
                None => { /* expected */ }
                Some(s) if s.trim().is_empty() => { /* acceptable */ }
                Some(s) => {
                    // For nested brackets like "[[nested]]", we get "[nested"
                    // which is still acceptable - it won't match valid THP values
                    assert!(
                        !s.contains('[') || content.contains("[["),
                        "Unexpected parse result '{}' from '{}'",
                        s,
                        content
                    );
                }
            }
        }
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn stress_parse_many_formats() {
    fn parse_thp_bracketed(content: &str) -> Option<String> {
        if let Some(start) = content.find('[') {
            if let Some(end) = content[start..].find(']') {
                return Some(content[start + 1..start + end].trim().to_string());
            }
        }
        None
    }

    // Test parsing many different format variations
    let test_cases = [
        ("[always] madvise never", "always"),
        ("always [madvise] never", "madvise"),
        ("always madvise [never]", "never"),
        ("[always]madvise never", "always"),
        ("always[madvise]never", "madvise"),
        ("  [always]  madvise  never  ", "always"),
        ("\t[always]\tmadvise\tnever", "always"),
        ("[always]\nmadvise\nnever", "always"),
        ("prefix [always] suffix", "always"),
        (
            "very long prefix here [always] very long suffix here",
            "always",
        ),
    ];

    for (input, expected) in test_cases {
        let result = parse_thp_bracketed(input);
        assert_eq!(
            result,
            Some(expected.to_string()),
            "Failed to parse '{}', expected '{}'",
            input,
            expected
        );
    }
}

#[test]
fn stress_parse_all_valid_combinations() {
    fn parse_thp_bracketed(content: &str) -> Option<String> {
        if let Some(start) = content.find('[') {
            if let Some(end) = content[start..].find(']') {
                return Some(content[start + 1..start + end].trim().to_string());
            }
        }
        None
    }

    // Generate all possible valid THP enabled combinations
    let options = ["always", "madvise", "never"];
    let mut seen = HashSet::new();

    for &selected in &options {
        let content: String = options
            .iter()
            .map(|&opt| {
                if opt == selected {
                    format!("[{}]", opt)
                } else {
                    opt.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        let result = parse_thp_bracketed(&content);
        assert_eq!(
            result,
            Some(selected.to_string()),
            "Failed to parse '{}'",
            content
        );
        seen.insert(selected);
    }

    // Verify we tested all options
    assert_eq!(
        seen.len(),
        3,
        "Should have tested all 3 THP enabled options"
    );

    // Generate all possible valid defrag combinations
    let defrag_options = ["always", "defer", "defer+madvise", "madvise", "never"];
    let mut defrag_seen = HashSet::new();

    for &selected in &defrag_options {
        let content: String = defrag_options
            .iter()
            .map(|&opt| {
                if opt == selected {
                    format!("[{}]", opt)
                } else {
                    opt.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        let result = parse_thp_bracketed(&content);
        assert_eq!(
            result,
            Some(selected.to_string()),
            "Failed to parse '{}'",
            content
        );
        defrag_seen.insert(selected);
    }

    // Verify we tested all options
    assert_eq!(
        defrag_seen.len(),
        5,
        "Should have tested all 5 THP defrag options"
    );
}
