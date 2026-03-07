use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClauseKind {
    PrimedAssignment {
        var: String,
        expr: String,
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
    let mut in_quantifier_body = false;

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
                if_depth = if_depth.saturating_sub(1);
                current.push_str("ELSE");
                i += 4;
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

        let at_top = at_bracket_top && let_depth == 0 && if_depth == 0 && case_depth == 0;
        if at_top && matches_at(&chars, i, &delim_chars) {
            if in_quantifier_body && delimiter == "/\\" {
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
                in_quantifier_body = false;
            }
            let piece = current.trim();
            if !piece.is_empty() {
                out.push(piece.to_string());
            }
            current.clear();
            // Reset case depth when we split (CASE expressions don't span conjuncts)
            case_depth = 0;
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
    let trimmed = clause.trim();

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
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
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
    }
}
