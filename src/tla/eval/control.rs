//! Control-flow expression evaluators: `IF/THEN/ELSE`, `CASE`,
//! and `LET ... IN`.
//!
//! Also hosts the LET-block parser (`split_outer_let`,
//! `parse_let_definitions`, plus head-/parameter-list helpers) that the
//! action evaluator and the LET expression evaluator both call.
//!
//! All three expression evaluators delegate back into
//! `super::eval_expr_inner` for sub-expression evaluation; that fn is
//! the precedence-ladder dispatcher and lives in the parent module.

use anyhow::{Result, anyhow};
use std::collections::BTreeMap;

use crate::tla::{TlaDefinition, TlaValue};

use super::{
    EvalContext, eval_expr_inner, find_outer_else, find_outer_then,
    find_top_level_definition_eqs, find_top_level_keyword_index, line_start_before,
    next_word, skip_leading_ws, split_once_top_level, split_top_level_symbol,
    take_keyword_prefix,
};

pub(super) fn eval_if_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (_, after_if) =
        take_keyword_prefix(expr.trim(), "IF").ok_or_else(|| anyhow!("expected IF expression"))?;

    let then_pos =
        find_outer_then(after_if).ok_or_else(|| anyhow!("IF expression missing THEN: {expr}"))?;
    let cond_text = after_if[..then_pos].trim();

    let after_then = after_if[then_pos + "THEN".len()..].trim_start();
    let else_pos =
        find_outer_else(after_then).ok_or_else(|| anyhow!("IF expression missing ELSE: {expr}"))?;

    let then_text = after_then[..else_pos].trim();
    let else_text = after_then[else_pos + "ELSE".len()..].trim();

    if eval_expr_inner(cond_text, ctx, depth + 1)?.as_bool()? {
        eval_expr_inner(then_text, ctx, depth + 1)
    } else {
        eval_expr_inner(else_text, ctx, depth + 1)
    }
}

pub(super) fn eval_case_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (_, after_case) = take_keyword_prefix(expr.trim(), "CASE")
        .ok_or_else(|| anyhow!("expected CASE expression"))?;
    let arms = split_top_level_symbol(after_case, "[]");

    let mut default_arm: Option<String> = None;
    for arm in arms {
        let arm = arm.trim();
        if arm.is_empty() {
            continue;
        }

        let (cond, rhs) =
            split_once_top_level(arm, "->").ok_or_else(|| anyhow!("invalid CASE arm: {arm}"))?;
        if cond.trim() == "OTHER" {
            default_arm = Some(rhs.trim().to_string());
            continue;
        }

        if eval_expr_inner(cond.trim(), ctx, depth + 1)?.as_bool()? {
            return eval_expr_inner(rhs.trim(), ctx, depth + 1);
        }
    }

    if let Some(default_expr) = default_arm {
        eval_expr_inner(&default_expr, ctx, depth + 1)
    } else {
        Err(anyhow!(
            "CASE expression has no matching arm and no OTHER arm"
        ))
    }
}

pub(super) fn eval_let_expression(expr: &str, ctx: &EvalContext<'_>, depth: usize) -> Result<TlaValue> {
    let (defs_text, body_text) =
        split_outer_let(expr).ok_or_else(|| anyhow!("invalid LET expression: {expr}"))?;
    let defs = parse_let_definitions(defs_text)?;
    // T203: charge budget for the clone-on-write of local_definitions.
    // Pathological nested LET inputs allocate O(N x M) without this.
    ctx.check_budget(ctx.local_definitions.len().saturating_add(defs.len()))?;
    let child = ctx.with_local_definitions(defs);
    eval_expr_inner(body_text, &child, depth + 1)
}

pub(crate) fn split_outer_let(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let (let_start, after_let) = take_keyword_prefix(trimmed, "LET")?;
    debug_assert!(let_start.is_empty());

    let mut depth = 1usize;
    let mut i = 0usize;
    while i < after_let.len() {
        if let Some((word, start, end)) = next_word(after_let, i) {
            match word {
                "LET" => depth += 1,
                "IN" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        let defs = after_let[..start].trim();
                        let body = after_let[end..].trim_end();
                        return Some((defs, body));
                    }
                }
                _ => {}
            }
            i = end;
        } else {
            break;
        }
    }

    None
}

pub(super) fn parse_let_definitions(defs_text: &str) -> Result<BTreeMap<String, TlaDefinition>> {
    let eq_positions = find_top_level_definition_eqs(defs_text);

    if eq_positions.is_empty() {
        return Err(anyhow!("LET block has no definitions"));
    }

    let mut defs = BTreeMap::new();
    let mut cursor = skip_leading_ws(defs_text, 0);

    for (idx, eq_pos) in eq_positions.iter().enumerate() {
        let next_head_start = if idx + 1 < eq_positions.len() {
            line_start_before(defs_text, eq_positions[idx + 1])
        } else {
            defs_text.len()
        };

        // Defensive: malformed input may pack two `==` markers adjacently,
        // making `next_head_start < eq_pos + 2`. Skip rather than panic.
        let body_start = *eq_pos + 2;
        if body_start > next_head_start {
            cursor = next_head_start;
            continue;
        }
        let head = trim_let_edge_comments(&defs_text[cursor..*eq_pos]);
        let body = trim_let_edge_comments(&defs_text[body_start..next_head_start]);
        let (name, params) = parse_local_def_head(head);

        if !name.is_empty() {
            defs.insert(
                name.clone(),
                TlaDefinition {
                    name,
                    params,
                    body: body.to_string(),
                    is_recursive: false,
                },
            );
        }

        cursor = skip_leading_ws(defs_text, next_head_start);
    }

    Ok(defs)
}

fn trim_let_edge_comments(text: &str) -> &str {
    let mut start = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("\\*") {
            start += line.len();
            continue;
        }
        break;
    }

    let trimmed = &text[start..];
    let mut end = trimmed.len();
    let lines: Vec<&str> = trimmed.split_inclusive('\n').collect();
    for line in lines.into_iter().rev() {
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() || line_trimmed.starts_with("\\*") {
            end = end.saturating_sub(line.len());
            continue;
        }
        break;
    }

    trimmed[..end].trim()
}

fn parse_local_def_head(head: &str) -> (String, Vec<String>) {
    if let Some((open, close)) = first_local_def_param_delims(head)
        && let Some((name, params)) = parse_local_def_head_with_delims(head, open, close)
    {
        return (name, params);
    }

    let name = head
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn first_local_def_param_delims(head: &str) -> Option<(char, char)> {
    match (head.find('('), head.find('[')) {
        (Some(paren), Some(bracket)) if bracket < paren => Some(('[', ']')),
        (Some(_), Some(_)) => Some(('(', ')')),
        (Some(_), None) => Some(('(', ')')),
        (None, Some(_)) => Some(('[', ']')),
        (None, None) => None,
    }
}

fn parse_local_def_head_with_delims(
    head: &str,
    open_delim: char,
    close_delim: char,
) -> Option<(String, Vec<String>)> {
    let open = head.find(open_delim)?;
    let close = head.rfind(close_delim)?;
    if close <= open {
        return None;
    }

    let name = head[..open].trim().to_string();
    let inside = &head[open + 1..close];
    let params = if open_delim == '(' {
        split_top_level_symbol(inside, ",")
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
    } else {
        let mut params = Vec::new();
        for part in split_top_level_symbol(inside, ",") {
            let part = part.trim();
            let lhs = if let Some(pos) = find_top_level_keyword_index(part, "\\in") {
                &part[..pos]
            } else {
                part
            };
            for candidate in split_top_level_symbol(lhs, ",") {
                let candidate = candidate.trim();
                if candidate.is_empty() {
                    continue;
                }
                params.push(candidate.to_string());
            }
        }
        params
    };
    Some((name, params))
}
