use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClauseKind {
    PrimedAssignment {
        var: String,
        expr: String,
    },
    PrimedMembership {
        var: String,
        set_expr: String,
    },
    UnprimedEquality {
        var: String,
        expr: String,
    },
    /// Membership assignment: var \in set (used in Init to assign var from set)
    UnprimedMembership {
        var: String,
        set_expr: String,
    },
    Unchanged {
        vars: Vec<String>,
    },
    Other,
}

pub fn split_top_level(expr: &str, delimiter: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = expr.chars().collect();
    let delim_chars: Vec<char> = delimiter.chars().collect();
    let mut i = 0usize;

    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize; // Track LET...IN nesting
    let mut if_depth = 0usize; // Track IF...ELSE nesting
    let mut case_depth = 0usize; // Track CASE expression nesting
    let mut else_branch_uses_delimiter = false; // True if ELSE branch starts with the delimiter
    let mut in_quantifier_body = false;
    let mut quantifier_body_uses_delimiter = false; // True if quantifier body starts with the delimiter
    while i < chars.len() {
        let c = chars[i];
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };

        if c == '<' && next == Some('<') {
            angle += 1;
            current.push(c);
            current.push('<');
            i += 2;
            continue;
        }
        if c == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
            current.push(c);
            current.push('>');
            i += 2;
            continue;
        }

        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            _ => {}
        }

        // Check for keywords at word boundary when at bracket top level
        let at_bracket_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;

        if at_bracket_top {
            // Track LET...IN nesting
            // NOTE: We increment let_depth on LET but DO NOT decrement on IN.
            // This is because in TLA+ actions, the pattern:
            //   /\ LET x == ... IN /\ expr1 /\ expr2
            // should treat "expr1 /\ expr2" as the body of the LET, not as
            // top-level conjuncts. The LET expression extends to the end of the
            // current grouping, and we only want to split at conjunctions that
            // are truly at the top level (outside any LET).
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                current.push_str("LET");
                i += 3;
                continue;
            }
            // Just push IN without decrementing - the LET body extends to the end
            if let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
                current.push_str("IN");
                i += 2;
                continue;
            }

            // Track IF...ELSE nesting - don't split inside IF-THEN-ELSE
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                current.push_str("IF");
                i += 2;
                continue;
            }
            if if_depth > 0 && matches_keyword_at(&chars, i, "ELSE") {
                // Don't decrement if_depth yet - check if ELSE branch starts with delimiter
                current.push_str("ELSE");
                i += 4;
                // Look ahead to see if ELSE is followed by the delimiter (ignoring whitespace)
                let remaining: String = chars[i..].iter().collect();
                let remaining_trimmed = remaining.trim_start();
                if remaining_trimmed.starts_with(delimiter) {
                    // ELSE branch starts with delimiter - treat the entire IF-THEN-ELSE
                    // including this ELSE branch as atomic (don't decrement if_depth)
                    else_branch_uses_delimiter = true;
                } else {
                    // ELSE branch doesn't start with delimiter - we can decrement
                    if_depth = if_depth.saturating_sub(1);
                }
                continue;
            }

            // Track CASE expressions - don't split inside CASE
            if matches_keyword_at(&chars, i, "CASE") {
                case_depth += 1;
                current.push_str("CASE");
                i += 4;
                continue;
            }

            if c == '\\' && i + 2 < chars.len() {
                let quant = chars[i + 1];
                let after = chars[i + 2];
                if (quant == 'A' || quant == 'E') && (after.is_whitespace() || after == '(') {
                    let mut j = i + 2;
                    let mut inner_depth = 0usize;
                    while j < chars.len() {
                        match chars[j] {
                            '(' | '[' | '{' => inner_depth += 1,
                            ')' | ']' | '}' => inner_depth = inner_depth.saturating_sub(1),
                            '<' if j + 1 < chars.len() && chars[j + 1] == '<' => {
                                inner_depth += 1;
                                j += 1;
                            }
                            '>' if j + 1 < chars.len() && chars[j + 1] == '>' => {
                                inner_depth = inner_depth.saturating_sub(1);
                                j += 1;
                            }
                            ':' if inner_depth == 0 => {
                                in_quantifier_body = true;
                                // Check if the body starts with the delimiter we're splitting by
                                // This determines if subsequent delimiters are part of the body
                                let body_start: String = chars[j + 1..].iter().collect();
                                let body_start = body_start.trim_start();
                                quantifier_body_uses_delimiter = body_start.starts_with(delimiter);
                                for ch in &chars[i..=j] {
                                    current.push(*ch);
                                }
                                i = j + 1;
                                break;
                            }
                            _ => {}
                        }
                        j += 1;
                    }
                    if j < chars.len() {
                        continue;
                    }
                }
            }
        }

        let at_top = at_bracket_top
            && let_depth == 0
            && if_depth == 0
            && case_depth == 0
            && !else_branch_uses_delimiter;
        if at_top && matches_at(&chars, i, &delim_chars) {
            // Don't split inside quantifier bodies when the body started with the same delimiter
            // For example: \E op \in S : \/ A \/ B \/ C
            // The body starts with \/, so all subsequent \/ are part of the body.
            // Similarly: \A t \in S : /\ \A k \in Key : f[t][k] /\ written[t] = {}
            // The body starts with /\, so all /\ (including those before inner quantifiers)
            // are part of the body.
            if in_quantifier_body && quantifier_body_uses_delimiter {
                current.push(c);
                i += 1;
                continue;
            }
            // Also keep simple conjunctions inside quantifier bodies when splitting on /\
            // e.g.: ~ \E m \in msgs : m.type = "2a" /\ m.bal = b
            // BUT: if the delimiter introduces a new top-level quantifier, we MUST
            // split here so the new quantifier is treated as a separate conjunct.
            // e.g.: \A t \in tx : ... /\ \A t \in TxId \ tx : ...
            if in_quantifier_body && delimiter == "/\\" && !quantifier_body_uses_delimiter {
                let after_delim: String = chars[i + delim_chars.len()..].iter().collect();
                let after_delim = after_delim.trim_start();
                let starts_new_quantifier = after_delim.starts_with("\\A ")
                    || after_delim.starts_with("\\A(")
                    || after_delim.starts_with("\\E ")
                    || after_delim.starts_with("\\E(");
                if !starts_new_quantifier {
                    current.push(c);
                    i += 1;
                    continue;
                }
                // Falls through to split - in_quantifier_body will be reset below
            }
            let piece = current.trim();
            if !piece.is_empty() {
                out.push(piece.to_string());
            }
            current.clear();
            // Reset case depth when we split (CASE expressions don't span conjuncts)
            case_depth = 0;
            // Reset quantifier tracking when we split
            in_quantifier_body = false;
            quantifier_body_uses_delimiter = false;
            i += delim_chars.len();
            continue;
        }

        current.push(c);
        i += 1;
    }

    let piece = current.trim();
    if !piece.is_empty() {
        out.push(piece.to_string());
    }

    out
}

/// Check if a keyword appears at position i with word boundaries
fn matches_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();

    // Check if keyword matches
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (j, kc) in kw_chars.iter().enumerate() {
        if chars[i + j] != *kc {
            return false;
        }
    }

    // Check word boundary before (must be start or non-alphanumeric)
    if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        return false;
    }

    // Check word boundary after (must be end or non-alphanumeric)
    let after = i + kw_chars.len();
    if after < chars.len() && (chars[after].is_alphanumeric() || chars[after] == '_') {
        return false;
    }

    true
}

pub fn classify_clause(clause: &str) -> ClauseKind {
    let mut trimmed = strip_leading_action_labels(clause.trim());
    // Strip outer parentheses: (h_turn = 1) → h_turn = 1
    while trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Only strip if the parens are balanced (not part of a nested expression)
        let mut depth = 0i32;
        let mut balanced = true;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if balanced && depth == 0 {
            trimmed = inner.trim();
        } else {
            break;
        }
    }

    if let Some(rest) = trimmed.strip_prefix("UNCHANGED") {
        let vars = parse_unchanged_list(rest);
        return ClauseKind::Unchanged { vars };
    }

    if let Some((lhs, rhs)) = trimmed.split_once('=') {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        if let Some(var) = lhs.strip_suffix('\'') {
            let var = var.trim();
            if !is_simple_name(var) || rhs.is_empty() {
                return ClauseKind::Other;
            }
            return ClauseKind::PrimedAssignment {
                var: var.to_string(),
                expr: rhs.to_string(),
            };
        }
        if is_simple_name(lhs) && !rhs.is_empty() {
            return ClauseKind::UnprimedEquality {
                var: lhs.to_string(),
                expr: rhs.to_string(),
            };
        }
    }

    // Check for membership: var \in set (used in Init to pick from set)
    if let Some((var, set_expr)) = split_membership(trimmed) {
        if let Some(var) = var.strip_suffix('\'') {
            let var = var.trim();
            if is_simple_name(var) && !set_expr.is_empty() {
                return ClauseKind::PrimedMembership {
                    var: var.to_string(),
                    set_expr,
                };
            }
        }
        if is_simple_name(&var) && !set_expr.is_empty() {
            return ClauseKind::UnprimedMembership { var, set_expr };
        }
    }

    ClauseKind::Other
}

/// Split "var \in set_expr" into (var, set_expr)
fn split_membership(s: &str) -> Option<(String, String)> {
    // Look for " \in " at top level (not inside brackets)
    let in_patterns = [" \\in ", " ∈ "];

    for pattern in in_patterns {
        if let Some(pos) = find_top_level_pattern(s, pattern) {
            let var = s[..pos].trim().to_string();
            let set_expr = s[pos + pattern.len()..].trim().to_string();
            return Some((var, set_expr));
        }
    }
    None
}

/// Find pattern at top level (not inside brackets/parens)
fn find_top_level_pattern(s: &str, pattern: &str) -> Option<usize> {
    let mut depth: usize = 0;
    let mut i = 0;
    let bytes = s.as_bytes();
    let pattern_bytes = pattern.as_bytes();

    while i + pattern_bytes.len() <= bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => {
                // Check if pattern matches at this position
                if &bytes[i..i + pattern_bytes.len()] == pattern_bytes {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn is_simple_name(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return false;
    }

    let mut saw_identifier_marker = first.is_ascii_alphabetic() || first == '_';
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
        saw_identifier_marker |= c.is_ascii_alphabetic() || c == '_';
    }

    saw_identifier_marker
}

pub fn parse_stuttering_action_expr(expr: &str) -> Option<(String, Vec<String>)> {
    let trimmed = strip_leading_action_labels(expr.trim());
    let (action, tail) = take_top_level_group(trimmed, '[', ']')?;
    let tail = tail.trim_start();
    let subscript = tail.strip_prefix('_')?.trim();
    if subscript.is_empty() {
        return None;
    }
    if !tail[1..].trim().eq(subscript) {
        return None;
    }

    let vars = parse_unchanged_list(subscript);
    if vars.is_empty() || !vars.iter().all(|var| is_simple_name(var)) {
        return None;
    }

    let action = action.trim();
    if action.is_empty() {
        return None;
    }

    Some((action.to_string(), vars))
}

pub fn expand_stuttering_action_expr(expr: &str) -> Option<String> {
    let (action, vars) = parse_stuttering_action_expr(expr)?;
    Some(format!("{action} \\/ {}", format_unchanged_clause(&vars)))
}

pub fn format_unchanged_clause(vars: &[String]) -> String {
    match vars {
        [] => "UNCHANGED <<>>".to_string(),
        [single] => format!("UNCHANGED {single}"),
        _ => format!("UNCHANGED <<{}>>", vars.join(", ")),
    }
}

fn parse_unchanged_list(rest: &str) -> Vec<String> {
    let trimmed = rest.trim();
    if let Some(inner) = trimmed
        .strip_prefix("<<")
        .and_then(|s| s.strip_suffix(">>"))
    {
        return inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();
    }

    if trimmed.is_empty() {
        Vec::new()
    } else {
        vec![trimmed.to_string()]
    }
}

fn strip_leading_action_labels(mut expr: &str) -> &str {
    loop {
        let trimmed = expr.trim_start();
        let Some((label, rest)) = trimmed.split_once("::") else {
            return trimmed;
        };
        if !is_simple_name(label.trim()) {
            return trimmed;
        }
        expr = rest;
    }
}

fn take_top_level_group<'a>(expr: &'a str, open: char, close: char) -> Option<(&'a str, &'a str)> {
    let mut chars = expr.char_indices();
    let (_, first) = chars.next()?;
    if first != open {
        return None;
    }

    let mut depth = 1usize;
    for (idx, ch) in chars {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some((&expr[1..idx], &expr[idx + ch.len_utf8()..]));
            }
        }
    }

    None
}

fn matches_at(chars: &[char], idx: usize, needle: &[char]) -> bool {
    if idx + needle.len() > chars.len() {
        return false;
    }
    for j in 0..needle.len() {
        if chars[idx + j] != needle[j] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_top_level_disjunction() {
        let parts = split_top_level("\\/ A /\\ B \\/ \\E x \\in S: (C \\/ D) \\/ E", "\\/");
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn does_not_split_quantified_action_body() {
        let parts = split_top_level(
            "/\\ Cardinality(staleSlots) >= needSlots /\\ \\E targets \\in SUBSET staleSlots : /\\ Cardinality(targets) = needSlots /\\ pendingWrite' = bestGen /\\ writeTargets' = targets",
            "/\\",
        );
        assert_eq!(parts.len(), 2);
        assert!(parts[1].starts_with("\\E targets \\in SUBSET staleSlots :"));
        assert!(parts[1].contains("writeTargets' = targets"));
    }

    #[test]
    fn does_not_split_simple_conjunction_inside_quantifier_body() {
        let parts = split_top_level(r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#, "/\\");
        assert_eq!(
            parts,
            vec![r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#]
        );
    }

    /// Test that nested disjunctions inside a quantifier body are kept together
    /// when the body starts with \/ (ClusterLeaseFailover pattern)
    #[test]
    fn does_not_split_nested_disjunction_in_quantifier_body() {
        // Pattern from ClusterLeaseFailover.tla:
        // \E op \in pendingOps : \/ JournalOperation(op) \/ CommitOperation(op) \/ FailOperation(op)
        let parts = split_top_level(
            r"\/ \E c \in Clients : AcquireLease(c) \/ \E op \in pendingOps : \/ JournalOperation(op) \/ CommitOperation(op) \/ FailOperation(op)",
            "\\/",
        );
        // Should split into 2 parts:
        // 1. \E c \in Clients : AcquireLease(c)
        // 2. \E op \in pendingOps : \/ JournalOperation(op) \/ CommitOperation(op) \/ FailOperation(op)
        assert_eq!(parts.len(), 2);
        assert!(parts[0].starts_with("\\E c \\in Clients"));
        assert!(parts[1].starts_with("\\E op \\in pendingOps :"));
        assert!(parts[1].contains("FailOperation(op)"));
    }

    /// When the quantifier body does NOT start with \/, subsequent \/ at the top
    /// level should still split normally.
    #[test]
    fn splits_disjunction_after_simple_quantifier_body() {
        let parts = split_top_level(r"\/ A \/ \E x \in S: (C \/ D) \/ E", "\\/");
        // Body starts with '(', not '\/', so '\/ E' is a separate top-level disjunct
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].trim(), "A");
        assert!(parts[1].starts_with("\\E x \\in S:"));
        assert_eq!(parts[2].trim(), "E");
    }

    #[test]
    fn classifies_unchanged_and_assignment() {
        assert_eq!(
            classify_clause("UNCHANGED <<x, y>>"),
            ClauseKind::Unchanged {
                vars: vec!["x".to_string(), "y".to_string()]
            }
        );
        assert_eq!(
            classify_clause("x' = x + 1"),
            ClauseKind::PrimedAssignment {
                var: "x".to_string(),
                expr: "x + 1".to_string()
            }
        );
        assert_eq!(
            classify_clause("x' \\in 1..3"),
            ClauseKind::PrimedMembership {
                var: "x".to_string(),
                set_expr: "1..3".to_string()
            }
        );
    }

    #[test]
    fn expands_box_action_formulas_into_disjunctions() {
        assert_eq!(
            expand_stuttering_action_expr("[NowNext]_now"),
            Some("NowNext \\/ UNCHANGED now".to_string())
        );
        assert_eq!(
            expand_stuttering_action_expr("[HCnxt]_<<hr, now>>"),
            Some("HCnxt \\/ UNCHANGED <<hr, now>>".to_string())
        );
        assert_eq!(expand_stuttering_action_expr("[][Next]_vars"), None);
        assert_eq!(expand_stuttering_action_expr("[A]_[x + 1]"), None);
    }

    #[test]
    fn classifies_numeric_prefixed_identifiers_in_membership_and_assignment() {
        assert_eq!(
            classify_clause("2avSent \\in [Acceptor -> SUBSET Msg]"),
            ClauseKind::UnprimedMembership {
                var: "2avSent".to_string(),
                set_expr: "[Acceptor -> SUBSET Msg]".to_string()
            }
        );
        assert_eq!(
            classify_clause("1bCount = 0"),
            ClauseKind::UnprimedEquality {
                var: "1bCount".to_string(),
                expr: "0".to_string()
            }
        );
    }

    #[test]
    fn classifies_pluscal_labeled_assignments() {
        assert_eq!(
            classify_clause("UP:: upEdge' = [upEdge EXCEPT ![p] = NotAnEdge]"),
            ClauseKind::PrimedAssignment {
                var: "upEdge".to_string(),
                expr: "[upEdge EXCEPT ![p] = NotAnEdge]".to_string()
            }
        );
    }

    #[test]
    fn parses_pluscal_labeled_stuttering_actions() {
        assert_eq!(
            parse_stuttering_action_expr("N:: [HCnxt]_<<hr, now>>"),
            Some((
                "HCnxt".to_string(),
                vec!["hr".to_string(), "now".to_string()]
            ))
        );
    }

    #[test]
    fn does_not_split_else_starting_with_conjunction() {
        // Test the specific pattern that was causing empty ELSE branches:
        // ELSE /\ content should not split at the /\ after ELSE
        let expr = "IF cond THEN a ELSE /\\ b /\\ c";
        let parts = split_top_level(expr, "/\\");

        // Should be one part - the entire IF-THEN-ELSE including the ELSE branch
        assert_eq!(
            parts.len(),
            1,
            "Should not split inside IF-THEN-ELSE when ELSE starts with /\\: {:?}",
            parts
        );
        assert!(
            parts[0].contains("b"),
            "Should contain ELSE branch content 'b': {}",
            parts[0]
        );
        assert!(
            parts[0].contains("c"),
            "Should contain ELSE branch content 'c': {}",
            parts[0]
        );
    }

    #[test]
    fn splits_after_else_without_leading_delimiter() {
        // When ELSE is NOT followed by /\, subsequent /\ should split
        let expr = "/\\ IF cond THEN a ELSE b /\\ c";
        let parts = split_top_level(expr, "/\\");

        // Should split into 2 parts:
        // 1. IF cond THEN a ELSE b
        // 2. c
        assert_eq!(parts.len(), 2, "Should split: {:?}", parts);
        assert!(
            parts[0].contains("ELSE b"),
            "First part should contain complete IF-THEN-ELSE: {}",
            parts[0]
        );
        assert_eq!(
            parts[1].trim(),
            "c",
            "Second part should be 'c': {}",
            parts[1]
        );
    }

    #[test]
    fn handles_nested_if_then_else_with_else_starting_with_delimiter() {
        // Test nested IF-THEN-ELSE where both levels have ELSE starting with /\
        let expr = "IF outer THEN a ELSE /\\ IF inner THEN b ELSE /\\ c /\\ d";
        let parts = split_top_level(expr, "/\\");

        // Should be one part - the entire nested structure
        assert_eq!(
            parts.len(),
            1,
            "Should not split nested IF-THEN-ELSE: {:?}",
            parts
        );
        assert!(parts[0].contains("outer"), "Should contain outer condition");
        assert!(parts[0].contains("inner"), "Should contain inner condition");
        assert!(
            parts[0].contains("c"),
            "Should contain nested ELSE content 'c'"
        );
        assert!(
            parts[0].contains("d"),
            "Should contain nested ELSE content 'd'"
        );
    }

    #[test]
    fn handles_deeply_nested_if_then_else() {
        // Test deeply nested IF-THEN-ELSE to check for stack issues
        let mut expr = String::new();
        for _ in 0..100 {
            expr.push_str("IF cond THEN ");
        }
        expr.push_str("base");
        for _ in 0..100 {
            expr.push_str(" ELSE fallback");
        }

        // Should not panic or stack overflow
        let parts = split_top_level(&expr, "/\\");

        // Should be one part - the entire deeply nested expression
        assert_eq!(
            parts.len(),
            1,
            "Should not split deeply nested IF-THEN-ELSE"
        );
        assert!(parts[0].contains("base"), "Should contain inner value");
    }

    #[test]
    fn test_split_conjunction_after_quantifier_body_with_new_quantifier() {
        // Regression test: when a quantifier body does NOT start with /\,
        // a subsequent /\ that introduces a new quantifier must still split.
        // This matches the TxLifecycle invariant pattern from KeyValueStore.
        let expr = r"\A t \in tx : \A k \in Key : f[t][k] /\ \A t \in TxId \ tx : g[t]";
        let parts = split_top_level(expr, "/\\");
        assert_eq!(
            parts.len(),
            2,
            "Should split into two conjuncts when /\\ precedes a new \\A quantifier, got: {:?}",
            parts
        );
        assert!(
            parts[0].starts_with("\\A t \\in tx"),
            "First part should be the first forall: {:?}",
            parts[0]
        );
        assert!(
            parts[1].starts_with("\\A t \\in TxId"),
            "Second part should be the second forall: {:?}",
            parts[1]
        );
    }

    #[test]
    fn test_split_multi_line_conjunction_with_leading_slashes() {
        // Exact pattern from TxLifecycle invariant body (multi-line with leading /\)
        let expr = r"/\ \A t \in tx :
        \A k \in Key : (store[k] /= snapshotStore[t][k] /\ k \notin written[t]) => k \in missed[t]
    /\ \A t \in TxId \ tx :
        /\ \A k \in Key : snapshotStore[t][k] = NoVal
        /\ written[t] = {}
        /\ missed[t] = {}";
        let parts = split_top_level(expr, "/\\");
        assert_eq!(
            parts.len(),
            2,
            "Should split into two conjuncts (both leading /\\), got {} parts: {:?}",
            parts.len(),
            parts
        );
        assert!(
            parts[0].contains("\\A t \\in tx"),
            "First part should contain first forall: {:?}",
            parts[0]
        );
        assert!(
            parts[1].contains("\\A t \\in TxId"),
            "Second part should contain second forall: {:?}",
            parts[1]
        );
    }

    #[test]
    fn test_simple_conjunction_inside_quantifier_body_not_split() {
        // Verify that simple conjunctions inside a quantifier body are NOT split.
        // e.g.: ~ \E m \in msgs : m.type = "2a" /\ m.bal = b
        let expr = r#"~ \E m \in msgs : m.type = "2a" /\ m.bal = b"#;
        let parts = split_top_level(expr, "/\\");
        assert_eq!(
            parts.len(),
            1,
            "Should not split conjunctions inside quantifier body: {:?}",
            parts
        );
    }
}
