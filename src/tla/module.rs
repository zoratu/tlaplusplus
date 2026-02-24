use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlaDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub body: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlaModule {
    pub name: String,
    pub path: String,
    pub extends: Vec<String>,
    pub constants: Vec<String>,
    pub variables: Vec<String>,
    pub definitions: BTreeMap<String, TlaDefinition>,
}

pub fn parse_tla_module_file(path: &Path) -> Result<TlaModule> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let mut module = parse_tla_module_text(&raw)?;
    module.path = path.display().to_string();
    Ok(module)
}

pub fn parse_tla_module_text(input: &str) -> Result<TlaModule> {
    let cleaned = strip_comments(input);
    let mut module = TlaModule::default();

    let mut current_def: Option<TlaDefinition> = None;
    let mut current_def_indent = 0usize;
    let mut mode = NameListMode::None;

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if let Some(def) = current_def.as_mut() {
                def.body.push('\n');
            }
            continue;
        }

        if module.name.is_empty()
            && trimmed.starts_with("----")
            && trimmed.contains("MODULE")
            && trimmed.ends_with("----")
        {
            if let Some(name) = extract_module_name(trimmed) {
                module.name = name;
            }
            continue;
        }

        if trimmed.starts_with("====") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("EXTENDS") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            for m in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                module.extends.push(m.to_string());
            }
            continue;
        }

        if trimmed == "CONSTANT" || trimmed == "CONSTANTS" {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Constants;
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("CONSTANT ") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Constants;
            push_names(rest, &mut module.constants);
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("CONSTANTS ") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Constants;
            push_names(rest, &mut module.constants);
            continue;
        }

        if trimmed == "VARIABLE" || trimmed == "VARIABLES" {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Variables;
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("VARIABLE ") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Variables;
            push_names(rest, &mut module.variables);
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("VARIABLES ") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::Variables;
            push_names(rest, &mut module.variables);
            continue;
        }

        let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
        let can_start_definition = current_def.is_none() || line_indent <= current_def_indent;
        if can_start_definition && let Some((lhs, rhs)) = split_definition_line(trimmed) {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = line_indent;
            mode = NameListMode::None;
            let (name, params) = parse_def_head(lhs);
            current_def = Some(TlaDefinition {
                name,
                params,
                body: rhs.trim().to_string(),
            });
            continue;
        }

        if let Some(def) = current_def.as_mut() {
            if is_section_separator(trimmed) {
                continue;
            }
            if !def.body.is_empty() {
                def.body.push('\n');
            }
            def.body.push_str(trimmed);
            continue;
        }

        match mode {
            NameListMode::Constants if is_pure_name_list(trimmed) => {
                push_names(trimmed, &mut module.constants);
            }
            NameListMode::Variables if is_pure_name_list(trimmed) => {
                push_names(trimmed, &mut module.variables);
            }
            _ => {
                mode = NameListMode::None;
            }
        }
    }

    flush_definition(&mut module, &mut current_def);

    module.constants.sort();
    module.constants.dedup();
    module.variables.sort();
    module.variables.dedup();

    Ok(module)
}

fn flush_definition(module: &mut TlaModule, current: &mut Option<TlaDefinition>) {
    if let Some(def) = current.take() {
        module.definitions.insert(def.name.clone(), def);
    }
}

fn split_definition_line(line: &str) -> Option<(&str, &str)> {
    let idx = line.find("==")?;
    let lhs = line[..idx].trim();
    let rhs = line[idx + 2..].trim();
    if lhs.is_empty() {
        return None;
    }

    if lhs.starts_with("INVARIANT")
        || lhs.starts_with("PROPERTY")
        || lhs.starts_with("CONSTRAINT")
        || lhs.starts_with("ACTION_CONSTRAINT")
        || lhs.starts_with("SPECIFICATION")
        || lhs.starts_with("INIT")
        || lhs.starts_with("NEXT")
    {
        return None;
    }

    if lhs.starts_with("/\\") || lhs.starts_with("\\/") {
        return None;
    }

    Some((lhs, rhs))
}

fn parse_def_head(lhs: &str) -> (String, Vec<String>) {
    if let Some(open) = lhs.find('(')
        && let Some(close) = lhs.rfind(')')
    {
        let name = lhs[..open].trim().to_string();
        let params = lhs[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        return (name, params);
    }

    let name = lhs
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn push_names(text: &str, out: &mut Vec<String>) {
    for token in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let mut name = String::new();
        for c in token.chars() {
            if c.is_alphanumeric() || c == '_' {
                name.push(c);
            } else {
                break;
            }
        }
        if !name.is_empty() {
            out.push(name);
        }
    }
}

fn is_pure_name_list(line: &str) -> bool {
    line.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .all(|name| {
            let mut chars = name.chars();
            match chars.next() {
                Some(c) if c.is_alphabetic() || c == '_' => {}
                _ => return false,
            }
            chars.all(|c| c.is_alphanumeric() || c == '_')
        })
}

fn is_section_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 5 && trimmed.chars().all(|c| c == '-')
}

fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut block_depth = 0usize;
    let mut in_line_comment = false;

    while i < bytes.len() {
        let c = bytes[i] as char;
        let next = if i + 1 < bytes.len() {
            Some(bytes[i + 1] as char)
        } else {
            None
        };

        if in_line_comment {
            if c == '\n' {
                in_line_comment = false;
                out.push('\n');
            }
            i += 1;
            continue;
        }

        if block_depth > 0 {
            if c == '(' && next == Some('*') {
                block_depth += 1;
                i += 2;
                continue;
            }
            if c == '*' && next == Some(')') {
                block_depth -= 1;
                i += 2;
                continue;
            }
            if c == '\n' {
                out.push('\n');
            }
            i += 1;
            continue;
        }

        if c == '\\' && next == Some('*') {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if c == '(' && next == Some('*') {
            block_depth = 1;
            i += 2;
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

fn extract_module_name(line: &str) -> Option<String> {
    let marker = "MODULE";
    let idx = line.find(marker)?;
    let after = line[idx + marker.len()..].trim();
    let mut name = String::new();
    for c in after.chars() {
        if c.is_alphanumeric() || c == '_' {
            name.push(c);
        } else {
            break;
        }
    }
    if name.is_empty() { None } else { Some(name) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameListMode {
    None,
    Constants,
    Variables,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_module_bits() {
        let src = r#"
        ---- MODULE Demo ----
        EXTENDS Naturals, Sequences

        CONSTANTS A, B
        VARIABLES x, y

        Init == x = 0 /\\ y = 1
        Next(a) == a' = a + 1
        ==== 
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(m.name, "Demo");
        assert!(m.extends.contains(&"Naturals".to_string()));
        assert!(m.constants.contains(&"A".to_string()));
        assert!(m.variables.contains(&"x".to_string()));
        assert!(m.definitions.contains_key("Init"));
        assert!(m.definitions.contains_key("Next"));
        assert_eq!(m.definitions["Next"].params, vec!["a"]);
    }
}
