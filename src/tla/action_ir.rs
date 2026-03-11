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
    let normalized = normalize_multiline_action_indentation(expr);
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.starts_with("\\E") || trimmed.starts_with("\\A") {
        return vec![trimmed.to_string()];
    }

    let raw =
        split_indented_action_conjuncts(trimmed).unwrap_or_else(|| split_top_level(trimmed, "/\\"));
    let mut merged = Vec::with_capacity(raw.len().max(1));
    let mut idx = 0usize;
    while idx < raw.len() {
        let part = raw[idx].trim().to_string();
        if part.is_empty() {
            idx += 1;
            continue;
        }
        let starts_quant = part.starts_with("\\E") || part.starts_with("\\A");
        let open_quant_or_let = (starts_quant && (part.ends_with(':') || part.ends_with("IN")))
            || (part.starts_with("LET") && part.ends_with("IN"));
        if open_quant_or_let {
            let mut combined = part;
            for rest in raw.iter().skip(idx + 1) {
                if rest.trim().is_empty() {
                    continue;
                }
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
}

pub fn split_action_body_disjuncts(expr: &str) -> Vec<String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if let Some(rest) = trimmed.strip_prefix("/\\").map(str::trim_start)
        && rest.starts_with("\\/")
    {
        return split_action_body_disjuncts(rest);
    }

    if let Some(disjuncts) = split_indented_action_disjuncts(trimmed) {
        return disjuncts;
    }

    split_top_level(trimmed, "\\/")
        .into_iter()
        .filter_map(|part| {
            let part = part.trim().to_string();
            if part.is_empty() { None } else { Some(part) }
        })
        .collect()
}

fn split_indented_action_conjuncts(expr: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_action_indentation(expr);
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut base_indent = None;
    let mut saw_top_level = false;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed.starts_with("/\\") {
            let top_level_indent = *base_indent.get_or_insert(indent);
            if indent == top_level_indent {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(trimmed.trim_start_matches("/\\").trim_start());
                saw_top_level = true;
                continue;
            }
        }

        if !saw_top_level {
            return None;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() > 1 {
        Some(clauses)
    } else {
        None
    }
}

fn split_indented_action_disjuncts(expr: &str) -> Option<Vec<String>> {
    if !expr.contains('\n') {
        return None;
    }

    let normalized = normalize_multiline_action_indentation(expr);
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut base_indent = None;
    let mut saw_top_level = false;

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }

        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed.starts_with("\\/") {
            let top_level_indent = *base_indent.get_or_insert(indent);
            if indent == top_level_indent {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(trimmed.trim_start_matches("\\/").trim_start());
                saw_top_level = true;
                continue;
            }
        }

        if !saw_top_level {
            return None;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }

    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }

    if clauses.len() > 1 {
        Some(clauses)
    } else {
        None
    }
}

fn normalize_multiline_action_indentation(expr: &str) -> String {
    let mut lines = expr.lines();
    let Some(first) = lines.next() else {
        return String::new();
    };

    let rest: Vec<&str> = lines.collect();
    if rest.is_empty() {
        return expr.to_string();
    }

    let dedent = rest
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                None
            } else {
                Some(line.len().saturating_sub(trimmed.len()))
            }
        })
        .min()
        .unwrap_or(0);

    if dedent == 0 {
        return expr.to_string();
    }

    let mut normalized = String::with_capacity(expr.len());
    normalized.push_str(first);
    for line in rest {
        normalized.push('\n');
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let indent = line.len().saturating_sub(trimmed.len());
        let keep = indent.saturating_sub(dedent);
        normalized.push_str(&" ".repeat(keep));
        normalized.push_str(trimmed);
    }
    normalized
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
    let conjuncts = split_action_body_clauses(&def.body);
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

/// Compile an action into one or more IR branches for expression probing.
///
/// `analyze-tla` evaluates action clauses independently. For actions written as
/// a top-level disjunction of conjunctive branches, we need to probe each branch
/// separately instead of feeding raw `\/` separators into `compile_action_ir`.
pub fn compile_action_ir_branches(def: &TlaDefinition) -> Vec<ActionIr> {
    let trimmed = def.body.trim();
    let clauses = split_action_body_clauses(trimmed);
    let clause_options: Vec<Vec<String>> = if clauses.is_empty() {
        vec![vec![trimmed.to_string()]]
    } else {
        clauses
            .iter()
            .map(|clause| {
                let disjuncts = split_action_body_disjuncts(clause);
                if disjuncts.len() > 1 || clause.trim().starts_with("\\/") {
                    disjuncts
                        .into_iter()
                        .map(|disjunct| normalize_branch_clause(&disjunct))
                        .collect()
                } else {
                    vec![clause.trim().to_string()]
                }
            })
            .collect()
    };

    let mut branch_bodies = vec![Vec::<String>::new()];
    for options in clause_options {
        let mut next = Vec::new();
        for existing in &branch_bodies {
            for option in &options {
                let mut branch = existing.clone();
                branch.push(option.clone());
                next.push(branch);
            }
        }
        branch_bodies = next;
    }

    branch_bodies
        .into_iter()
        .map(|clauses| {
            if clauses.len() == 1 {
                clauses[0].trim().to_string()
            } else {
                clauses
                    .into_iter()
                    .filter(|clause| !clause.trim().is_empty())
                    .map(|clause| format!("/\\ {}", clause.trim()))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        })
        .filter_map(|body| {
            let body = body.trim().to_string();
            if body.is_empty() {
                return None;
            }

            Some(compile_action_ir(&TlaDefinition {
                name: def.name.clone(),
                params: def.params.clone(),
                body,
                is_recursive: def.is_recursive,
            }))
        })
        .collect()
}

fn normalize_branch_clause(clause: &str) -> String {
    clause
        .trim()
        .strip_prefix("/\\")
        .map(str::trim_start)
        .unwrap_or_else(|| clause.trim())
        .to_string()
}

pub fn looks_like_action(def: &TlaDefinition) -> bool {
    // Filter out THEOREM/LEMMA/proof definitions that contain primes
    // in proof steps but are not actual action definitions
    let name_upper = def.name.to_uppercase();
    if name_upper.starts_with("THEOREM")
        || name_upper.starts_with("LEMMA")
        || name_upper.starts_with("COROLLARY")
        || name_upper.starts_with("PROPOSITION")
    {
        return false;
    }
    // Proof bodies contain step labels like <1>, BY DEF, QED, SUFFICES
    if def.body.contains("BY DEF ")
        || def.body.contains(" QED")
        || def.body.contains("SUFFICES")
        || def.body.contains("<1>")
    {
        return false;
    }
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
            is_recursive: false,
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

    #[test]
    fn compiles_top_level_disjunctive_actions_as_probe_branches() {
        let def = TlaDefinition {
            name: "Receive".to_string(),
            params: vec!["i".to_string()],
            body: r#"
  \/ /\ pc[i] = "SENT"
     /\ x' = x + 1
     /\ UNCHANGED <<y>>
  \/ /\ pc[i] = "SENT"
     /\ y' = y + 1
     /\ UNCHANGED <<x>>
"#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 2);

        for branch in branches {
            assert!(!branch.clauses.is_empty());
            assert!(!branch.clauses.iter().any(
                |clause| matches!(clause, ActionClause::Guard { expr } if expr.trim() == "\\/")
            ));
        }
    }

    #[test]
    fn split_action_body_clauses_keeps_quantified_action_bodies_together() {
        let clauses = split_action_body_clauses(
            r#"
                /\ ready
                /\ \A call \in ActiveCalls :
                    /\ Service(call) =>
                        /\ \E worker \in Workers :
                            /\ worker /= self
                            /\ ServiceBy(worker, call)
                /\ nextFloor \in Floor
                /\ state' = nextState
            "#,
        );

        assert_eq!(clauses.len(), 4);
        assert!(clauses[1].contains(r"\A call \in ActiveCalls"));
        assert!(clauses[1].contains("ServiceBy(worker, call)"));
        assert!(!clauses[1].contains("nextFloor \\in Floor"));
        assert_eq!(clauses[2], "nextFloor \\in Floor");
        assert_eq!(clauses[3], "state' = nextState");
    }

    #[test]
    fn split_action_body_clauses_keeps_exists_bodies_with_nested_conjuncts_together() {
        let clauses = split_action_body_clauses(
            r#"
                /\ \E i \in unchecked[self]:
                    /\ unchecked' = [unchecked EXCEPT ![self] = unchecked[self] \ {i}]
                    /\ IF num[i] > max[self]
                          THEN /\ max' = [max EXCEPT ![self] = num[i]]
                          ELSE /\ TRUE
                               /\ max' = max
                /\ pc' = [pc EXCEPT ![self] = "e2"]
            "#,
        );

        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].starts_with(r"\E i \in unchecked[self]:"));
        assert!(
            clauses[0]
                .contains(r#"unchecked' = [unchecked EXCEPT ![self] = unchecked[self] \ {i}]"#)
        );
        assert!(clauses[0].contains(r#"max' = [max EXCEPT ![self] = num[i]]"#));
        assert!(clauses[0].contains(r#"max' = max"#));
        assert!(clauses[0].contains(r#"pc' = [pc EXCEPT ![self] = "e2"]"#));
    }

    #[test]
    fn split_action_body_clauses_ignores_leading_empty_conjuncts() {
        let clauses = split_action_body_clauses(
            r#"
                /\ \/ /\ rmState[self] = "working"
                      /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                   \/ /\ tmState = "commit"
                      /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                /\ pc' = [pc EXCEPT ![self] = "RS"]
            "#,
        );

        assert_eq!(clauses.len(), 2);
        assert!(!clauses.iter().any(|clause| clause.trim().is_empty()));
        assert!(clauses[0].starts_with(r#"\/ /\ rmState[self] = "working""#));
        assert_eq!(clauses[1], r#"pc' = [pc EXCEPT ![self] = "RS"]"#);
    }

    #[test]
    fn split_action_body_clauses_reindents_trimmed_if_then_branches() {
        let clauses = split_action_body_clauses(
            r#"/\ \/ /\ rmState[self] = "working"
                             /\ rmState' = [rmState EXCEPT ![self] = "prepared"]
                          \/ /\ \/ /\ tmState="commit"
                                   /\ rmState' = [rmState EXCEPT ![self] = "committed"]
                                \/ /\ rmState[self]="working" \/ tmState="abort"
                                   /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
                          \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                                   THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                                   ELSE /\ TRUE
                                        /\ UNCHANGED rmState
                       /\ pc' = [pc EXCEPT ![self] = "RS"]"#,
        );

        assert_eq!(clauses.len(), 2);
        assert!(clauses[0].starts_with(r#"\/ /\ rmState[self] = "working""#));
        assert!(clauses[0].contains(r#"IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed""#));
        assert_eq!(clauses[1], r#"pc' = [pc EXCEPT ![self] = "RS"]"#);
        assert!(!clauses.iter().any(|clause| clause.trim() == r#"\/"#));
    }

    #[test]
    fn split_action_body_clauses_separates_let_assignment_from_unchanged() {
        let clauses = split_action_body_clauses(
            r#"
                /\ c \in ActiveElevatorCalls
                /\ ElevatorState' =
                    LET closest == CHOOSE e \in stationary \cup approaching :
                        /\ \A e2 \in stationary \cup approaching :
                            /\ GetDistance[ElevatorState[e].floor, c.floor] <= GetDistance[ElevatorState[e2].floor, c.floor]
                    IN
                    IF closest \in stationary
                    THEN [ElevatorState EXCEPT ![closest] = [@ EXCEPT !.floor = c.floor, !.direction = c.direction]]
                    ELSE ElevatorState
                /\ UNCHANGED <<PersonState, ActiveElevatorCalls>>
            "#,
        );

        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], r#"c \in ActiveElevatorCalls"#);
        assert!(clauses[1].starts_with("ElevatorState' ="));
        assert!(clauses[1].contains("LET closest =="));
        assert_eq!(
            clauses[2],
            r#"UNCHANGED <<PersonState, ActiveElevatorCalls>>"#
        );
    }

    #[test]
    fn split_action_body_disjuncts_preserves_boolean_or_inside_branch_guards() {
        let disjuncts = split_action_body_disjuncts(
            r#"/\ \/ /\ tmState="commit"
                 /\ rmState' = [rmState EXCEPT ![self] = "committed"]
              \/ /\ rmState[self]="working" \/ tmState="abort"
                 /\ rmState' = [rmState EXCEPT ![self] = "aborted"]
              \/ /\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed"
                       THEN /\ rmState' = [rmState EXCEPT ![self] = "failed"]
                       ELSE /\ TRUE
                            /\ UNCHANGED rmState"#,
        );

        assert_eq!(disjuncts.len(), 3);
        assert!(disjuncts[0].starts_with(r#"/\ tmState="commit""#));
        assert!(disjuncts[0].contains(r#"rmState' = [rmState EXCEPT ![self] = "committed"]"#));
        assert!(disjuncts[1].contains(r#"rmState[self]="working" \/ tmState="abort""#));
        assert!(disjuncts[1].contains(r#"rmState' = [rmState EXCEPT ![self] = "aborted"]"#));
        assert!(
            disjuncts[2].starts_with(r#"/\ IF RMMAYFAIL /\ ~\E rm \in RM:rmState[rm]="failed""#)
        );
    }

    #[test]
    fn compile_action_ir_branches_expands_nested_disjunctions_with_shared_clauses() {
        let def = TlaDefinition {
            name: "e1".to_string(),
            params: vec!["self".to_string()],
            body: r#"
                /\ pc[self] = "e1"
                /\ \/ /\ flag' = [flag EXCEPT ![self] = ~ flag[self]]
                      /\ pc' = [pc EXCEPT ![self] = "e1"]
                   \/ /\ flag' = [flag EXCEPT ![self] = TRUE]
                      /\ unchecked' = [unchecked EXCEPT ![self] = Procs \ {self}]
                      /\ pc' = [pc EXCEPT ![self] = "e2"]
                /\ UNCHANGED num
            "#
            .to_string(),
            is_recursive: false,
        };

        let branches = compile_action_ir_branches(&def);
        assert_eq!(branches.len(), 2);
        for branch in branches {
            let texts: Vec<String> = branch
                .clauses
                .into_iter()
                .map(|clause| match clause {
                    ActionClause::Guard { expr }
                    | ActionClause::PrimedAssignment { expr, .. }
                    | ActionClause::LetWithPrimes { expr } => expr,
                    ActionClause::Exists { body, .. } => body,
                    ActionClause::Unchanged { vars } => vars.join(","),
                })
                .collect();
            assert!(texts.iter().any(|expr| expr.contains(r#"pc[self] = "e1""#)));
            assert!(
                texts
                    .iter()
                    .any(|expr| expr.contains(r#"pc EXCEPT ![self]"#))
            );
        }
    }
}
