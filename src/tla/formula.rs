use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClauseKind {
    PrimedAssignment { var: String, expr: String },
    UnprimedEquality { var: String, expr: String },
    Unchanged { vars: Vec<String> },
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

        // Check for LET keyword (at word boundary)
        let at_bracket_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_bracket_top && matches_keyword_at(&chars, i, "LET") {
            let_depth += 1;
            // Push "LET" and advance
            current.push_str("LET");
            i += 3;
            continue;
        }

        // Check for IN keyword (at word boundary) - only decrements if we're inside a LET
        if at_bracket_top && let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
            let_depth = let_depth.saturating_sub(1);
            // Push "IN" and advance
            current.push_str("IN");
            i += 2;
            continue;
        }

        let at_top = at_bracket_top && let_depth == 0;
        if at_top && matches_at(&chars, i, &delim_chars) {
            let piece = current.trim();
            if !piece.is_empty() {
                out.push(piece.to_string());
            }
            current.clear();
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

    ClauseKind::Other
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
