use crate::tla::{ClauseKind, TlaDefinition, classify_clause, split_top_level};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionClause {
    PrimedAssignment {
        var: String,
        expr: String,
    },
    Unchanged {
        vars: Vec<String>,
    },
    Guard {
        expr: String,
    },
    Exists {
        binders: String,
        body: String,
    },
    /// LET expression that contains primed assignments in its IN body
    LetWithPrimes {
        expr: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionIr {
    pub name: String,
    pub params: Vec<String>,
    pub clauses: Vec<ActionClause>,
}

pub(crate) fn split_action_body_clauses(expr: &str) -> Vec<String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.starts_with("\\E") || trimmed.starts_with("\\A") {
        return vec![trimmed.to_string()];
    }
    split_top_level(trimmed, "/\\")
}

pub(crate) fn parse_action_if(expr: &str) -> Option<(&str, &str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("IF")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();

    let then_idx = find_action_keyword(rest, "THEN")?;
    let condition = rest[..then_idx].trim();
    let after_then = rest[then_idx + "THEN".len()..].trim();

    let else_idx = find_action_keyword(after_then, "ELSE")?;
    let then_branch = after_then[..else_idx].trim();
    let else_branch = after_then[else_idx + "ELSE".len()..].trim();

    Some((condition, then_branch, else_branch))
}

pub(crate) fn parse_action_exists(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let rest = trimmed.strip_prefix("\\E")?;
    if !rest.starts_with(char::is_whitespace) && !rest.starts_with('(') {
        return None;
    }
    let rest = rest.trim_start();
    let colon_idx = find_action_char(rest, ':')?;
    let binders = rest[..colon_idx].trim();
    let body = rest[colon_idx + 1..].trim();
    if binders.is_empty() || body.is_empty() {
        return None;
    }
    Some((binders, body))
}

pub fn compile_action_ir(def: &TlaDefinition) -> ActionIr {
    let trimmed = def.body.trim();
    let conjuncts = if trimmed.starts_with("\\E") || trimmed.starts_with("\\A") {
        vec![trimmed.to_string()]
    } else {
        let raw = split_top_level(&def.body, "/\\");
        let mut merged = Vec::with_capacity(raw.len().max(1));
        let mut idx = 0usize;
        while idx < raw.len() {
            let part = raw[idx].trim().to_string();
            let starts_quant = part.starts_with("\\E") || part.starts_with("\\A");
            let open_quant_or_let = (starts_quant && (part.ends_with(':') || part.ends_with("IN")))
                || (part.starts_with("LET") && part.ends_with("IN"));
            if open_quant_or_let {
                let mut combined = part;
                for rest in raw.iter().skip(idx + 1) {
                    combined.push_str(" /\\ ");
                    combined.push_str(rest.trim());
                }
                merged.push(combined);
                break;
            }

            merged.push(part);
            idx += 1;
        }
        merged
    };
    let mut clauses = Vec::with_capacity(conjuncts.len().max(1));

    if conjuncts.is_empty() {
        clauses.push(ActionClause::Guard {
            expr: def.body.trim().to_string(),
        });
    } else {
        for part in conjuncts {
            if let Some((binders, body)) = parse_action_exists(&part) {
                clauses.push(ActionClause::Exists {
                    binders: binders.to_string(),
                    body: body.to_string(),
                });
                continue;
            }
            match classify_clause(&part) {
                ClauseKind::PrimedAssignment { var, expr } => {
                    clauses.push(ActionClause::PrimedAssignment { var, expr });
                }
                ClauseKind::Unchanged { vars } => {
                    clauses.push(ActionClause::Unchanged { vars });
                }
                _ => {
                    // Check if this is a LET expression with primed assignments
                    let trimmed = part.trim();
                    if trimmed.starts_with("LET") && trimmed.contains('\'') {
                        clauses.push(ActionClause::LetWithPrimes {
                            expr: trimmed.to_string(),
                        });
                    } else {
                        clauses.push(ActionClause::Guard {
                            expr: trimmed.to_string(),
                        });
                    }
                }
            }
        }
    }

    ActionIr {
        name: def.name.clone(),
        params: def.params.clone(),
        clauses,
    }
}

pub fn looks_like_action(def: &TlaDefinition) -> bool {
    def.body.contains('\'') || def.body.contains("UNCHANGED")
}

fn find_action_keyword(expr: &str, keyword: &str) -> Option<usize> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize;
    let mut if_depth = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();

        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }

        let at_top = paren == 0 && bracket == 0 && brace == 0 && angle == 0;
        if at_top {
            if matches_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                i += 3;
                continue;
            }
            if let_depth > 0 && matches_keyword_at(&chars, i, "IN") {
                let_depth = let_depth.saturating_sub(1);
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "IF") {
                if_depth += 1;
                i += 2;
                continue;
            }
            if matches_keyword_at(&chars, i, "ELSE") {
                if if_depth == 0 {
                    if keyword == "ELSE" {
                        let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                        return Some(byte_offset);
                    }
                } else {
                    if_depth = if_depth.saturating_sub(1);
                }
                i += 4;
                continue;
            }
        }

        if at_top && let_depth == 0 && if_depth == 0 && matches_keyword_at(&chars, i, keyword) {
            let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            return Some(byte_offset);
        }

        i += 1;
    }

    None
}

fn find_action_char(expr: &str, target: char) -> Option<usize> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();

        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' if next == Some('<') => {
                angle += 1;
                i += 2;
                continue;
            }
            '>' if next == Some('>') => {
                angle = angle.saturating_sub(1);
                i += 2;
                continue;
            }
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && c == target {
            if target == ':' && next == Some('>') {
                i += 1;
                continue;
            }
            let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
            return Some(byte_offset);
        }

        i += 1;
    }

    None
}

fn matches_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (j, kc) in kw_chars.iter().enumerate() {
        if chars[i + j] != *kc {
            return false;
        }
    }
    if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        return false;
    }
    let after = i + kw_chars.len();
    if after < chars.len() && (chars[after].is_alphanumeric() || chars[after] == '_') {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_action_clauses() {
        let def = TlaDefinition {
            name: "Tick".to_string(),
            params: vec![],
            body: "/\\ x' = x + 1 /\\ UNCHANGED <<y>> /\\ x < 10".to_string(),
        };

        let ir = compile_action_ir(&def);
        assert_eq!(ir.clauses.len(), 3);
        assert!(matches!(
            &ir.clauses[0],
            ActionClause::PrimedAssignment { var, .. } if var == "x"
        ));
        assert!(matches!(ir.clauses[1], ActionClause::Unchanged { .. }));
        assert!(matches!(ir.clauses[2], ActionClause::Guard { .. }));
    }

    #[test]
    fn parses_exists_action_clause() {
        let clause = parse_action_exists(
            "\\E targets \\in SUBSET staleSlots : /\\ Cardinality(targets) = needSlots /\\ writeTargets' = targets",
        )
        .expect("expected existential action clause");
        assert_eq!(clause.0, "targets \\in SUBSET staleSlots");
        assert!(clause.1.contains("writeTargets' = targets"));
    }
}
