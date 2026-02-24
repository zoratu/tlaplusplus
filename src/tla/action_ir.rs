use crate::tla::{ClauseKind, TlaDefinition, classify_clause, split_top_level};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionClause {
    PrimedAssignment { var: String, expr: String },
    Unchanged { vars: Vec<String> },
    Guard { expr: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionIr {
    pub name: String,
    pub params: Vec<String>,
    pub clauses: Vec<ActionClause>,
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
            match classify_clause(&part) {
                ClauseKind::PrimedAssignment { var, expr } => {
                    clauses.push(ActionClause::PrimedAssignment { var, expr });
                }
                ClauseKind::Unchanged { vars } => {
                    clauses.push(ActionClause::Unchanged { vars });
                }
                _ => clauses.push(ActionClause::Guard {
                    expr: part.trim().to_string(),
                }),
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
}
