use proptest::prelude::*;
use tlaplusplus::system::{
    ThpConfig, ThpDefrag, ThpStatus, evaluate_thp_config, parse_thp_defrag_content,
    parse_thp_status_content,
};

#[test]
fn parses_known_enabled_variants() {
    assert_eq!(
        parse_thp_status_content("[always] madvise never"),
        ThpStatus::Always
    );
    assert_eq!(
        parse_thp_status_content("always [madvise] never"),
        ThpStatus::Madvise
    );
    assert_eq!(
        parse_thp_status_content("always madvise [never]"),
        ThpStatus::Never
    );
}

#[test]
fn parses_known_defrag_variants() {
    assert_eq!(
        parse_thp_defrag_content("[always] defer defer+madvise madvise never"),
        ThpDefrag::Always
    );
    assert_eq!(
        parse_thp_defrag_content("always defer [defer+madvise] madvise never"),
        ThpDefrag::DeferMadvise
    );
    assert_eq!(
        parse_thp_defrag_content("always defer defer+madvise [madvise] never"),
        ThpDefrag::Madvise
    );
}

#[test]
fn warns_for_disabled_thp() {
    let report = evaluate_thp_config(&ThpConfig {
        enabled: ThpStatus::Never,
        defrag: ThpDefrag::Never,
    });
    assert!(!report.optimal);
    assert!(
        report
            .messages
            .iter()
            .any(|line| line.contains("THP) is disabled"))
    );
}

#[test]
fn warns_for_always_defrag() {
    let report = evaluate_thp_config(&ThpConfig {
        enabled: ThpStatus::Always,
        defrag: ThpDefrag::Always,
    });
    assert!(!report.optimal);
    assert!(
        report
            .messages
            .iter()
            .any(|line| line.contains("latency spikes"))
    );
}

proptest! {
    #[test]
    fn status_parser_uses_active_bracketed_value(
        prefix in "[A-Za-z ]{0,8}",
        suffix in "[A-Za-z ]{0,8}",
        token in prop::sample::select(vec!["always", "madvise", "never"]),
    ) {
        let content = format!("{prefix}[{token}]{suffix}");
        let expected = match token {
            "always" => ThpStatus::Always,
            "madvise" => ThpStatus::Madvise,
            "never" => ThpStatus::Never,
            _ => unreachable!(),
        };
        prop_assert_eq!(parse_thp_status_content(&content), expected);
    }

    #[test]
    fn status_parser_without_brackets_is_unknown(content in "[A-Za-z ]{0,16}") {
        prop_assert_eq!(parse_thp_status_content(&content), ThpStatus::Unknown);
    }
}
