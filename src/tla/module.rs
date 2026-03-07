use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlaDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub body: String,
    /// Whether this operator was declared with RECURSIVE
    #[serde(default)]
    pub is_recursive: bool,
}

/// Represents a module instance (INSTANCE declaration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlaModuleInstance {
    /// Alias name for this instance (e.g., "Helpers")
    pub alias: String,
    /// Module name being instantiated (e.g., "CoverageHelper")
    pub module_name: String,
    /// Parameter substitutions: maps parameter name to substitution expression
    /// e.g., "Node" -> "Node" for "WITH Node <- Node"
    pub substitutions: BTreeMap<String, String>,
    /// Whether this instance is LOCAL
    pub is_local: bool,
    /// The loaded module (populated after parsing)
    pub module: Option<Box<TlaModule>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlaModule {
    pub name: String,
    pub path: String,
    pub extends: Vec<String>,
    pub constants: Vec<String>,
    pub variables: Vec<String>,
    pub definitions: BTreeMap<String, TlaDefinition>,
    pub instances: BTreeMap<String, TlaModuleInstance>,
    /// True if module contains PlusCal code (detected via --algorithm marker)
    pub is_pluscal: bool,
    /// Set of operator names declared with RECURSIVE
    #[serde(default)]
    pub recursive_declarations: BTreeSet<String>,
}

pub fn parse_tla_module_file(path: &Path) -> Result<TlaModule> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let mut module = parse_tla_module_text(&raw)?;
    module.path = path.display().to_string();

    // Detect PlusCal by looking for --algorithm or --fair algorithm marker in the raw text
    // PlusCal algorithms are typically in block comments: (* --algorithm Name ... *)
    module.is_pluscal = detect_pluscal(&raw);

    // Load module instances
    load_module_instances(&mut module, path)?;

    Ok(module)
}

/// Detect if the module contains PlusCal code by looking for the algorithm marker.
/// PlusCal specs contain `--algorithm` or `--fair algorithm` inside a block comment.
fn detect_pluscal(raw: &str) -> bool {
    // Look for --algorithm or --fair algorithm in the raw text
    // These markers indicate PlusCal code which generates Init_ and Next_
    raw.contains("--algorithm") || raw.contains("--fair algorithm")
}

/// Load modules referenced by INSTANCE declarations
fn load_module_instances(module: &mut TlaModule, base_path: &Path) -> Result<()> {
    let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));

    for (alias, instance) in module.instances.iter_mut() {
        // Try to find the module file
        let instance_path = module_dir.join(format!("{}.tla", instance.module_name));

        if instance_path.exists() {
            // Load the module
            let instance_module = parse_tla_module_file(&instance_path).with_context(|| {
                format!(
                    "failed to load instance module '{}' for alias '{}'",
                    instance.module_name, alias
                )
            })?;

            instance.module = Some(Box::new(instance_module));
        } else {
            // Check if it's a built-in module that we can skip
            let builtin_modules = ["Naturals", "Integers", "Sequences", "FiniteSets", "TLC"];
            if !builtin_modules.contains(&instance.module_name.as_str()) {
                eprintln!(
                    "Warning: Instance module '{}' not found at {}",
                    instance.module_name,
                    instance_path.display()
                );
            }
        }
    }

    Ok(())
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

        // Parse RECURSIVE declarations
        // Format: RECURSIVE Op1(_), Op2(_, _), ...
        if let Some(rest) = trimmed.strip_prefix("RECURSIVE ") {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            parse_recursive_declarations(rest, &mut module.recursive_declarations);
            continue;
        }

        // Parse INSTANCE declarations
        // Formats:
        //   Alias == INSTANCE ModuleName
        //   Alias == INSTANCE ModuleName WITH Param1 <- Value1, Param2 <- Value2
        //   LOCAL Alias == INSTANCE ModuleName
        if let Some(instance) = parse_instance_declaration(trimmed) {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            module.instances.insert(instance.alias.clone(), instance);
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
                is_recursive: false, // Will be set by flush_definition
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
            // Preserve indentation relative to the definition start
            // This is important for TLA+ parsing where indentation indicates scope
            let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            let indent_diff = line_indent.saturating_sub(current_def_indent);
            for _ in 0..indent_diff {
                def.body.push(' ');
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
    if let Some(mut def) = current.take() {
        // Mark definition as recursive if it was declared with RECURSIVE
        def.is_recursive = module.recursive_declarations.contains(&def.name);
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
    // Strip LOCAL prefix if present
    let lhs = if let Some(rest) = lhs.strip_prefix("LOCAL ") {
        rest.trim()
    } else {
        lhs
    };

    if let Some(open) = lhs.find('(')
        && let Some(close) = lhs.rfind(')')
        && close > open
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

/// Parse an INSTANCE declaration
/// Formats:
///   Alias == INSTANCE ModuleName
///   Alias == INSTANCE ModuleName WITH Param1 <- Value1, Param2 <- Value2
///   LOCAL Alias == INSTANCE ModuleName WITH...
fn parse_instance_declaration(line: &str) -> Option<TlaModuleInstance> {
    let line = line.trim();

    // Check if line contains INSTANCE
    if !line.contains("INSTANCE") {
        return None;
    }

    // Check for LOCAL prefix
    let (is_local, line) = if let Some(rest) = line.strip_prefix("LOCAL ") {
        (true, rest.trim())
    } else {
        (false, line)
    };

    // Split on ==
    let (lhs, rhs) = line.split_once("==")?;
    let alias = lhs.trim();

    // Check if this is a valid alias (simple identifier)
    if !alias.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    // Parse RHS: INSTANCE ModuleName [WITH ...]
    let rhs = rhs.trim();
    if !rhs.starts_with("INSTANCE") {
        return None;
    }

    let after_instance = rhs["INSTANCE".len()..].trim();

    // Check for WITH clause
    let (module_name, with_clause) = if let Some(idx) = after_instance.find(" WITH ") {
        (
            after_instance[..idx].trim(),
            Some(after_instance[idx + " WITH ".len()..].trim()),
        )
    } else {
        (after_instance, None)
    };

    // Parse substitutions from WITH clause
    let mut substitutions = BTreeMap::new();
    if let Some(with_str) = with_clause {
        // Split on <- at the top level (not inside brackets)
        let mut depth = 0;
        let mut start = 0;
        let chars: Vec<char> = with_str.chars().collect();

        for i in 0..chars.len() {
            match chars[i] {
                '{' | '[' | '(' | '<' => depth += 1,
                '}' | ']' | ')' | '>' => depth -= 1,
                ',' if depth == 0 => {
                    // Found a top-level comma - this separates substitutions
                    let subst = &with_str[start..i];
                    if let Some((param, value)) = subst.split_once("<-") {
                        substitutions.insert(param.trim().to_string(), value.trim().to_string());
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }

        // Handle the last substitution
        if start < with_str.len() {
            let subst = &with_str[start..];
            if let Some((param, value)) = subst.split_once("<-") {
                substitutions.insert(param.trim().to_string(), value.trim().to_string());
            }
        }
    }

    Some(TlaModuleInstance {
        alias: alias.to_string(),
        module_name: module_name.to_string(),
        substitutions,
        is_local,
        module: None,
    })
}

/// Parse RECURSIVE declarations like "Op1(_), Op2(_, _)" and extract operator names.
/// The underscore placeholders represent the arity but we just need the names.
fn parse_recursive_declarations(text: &str, recursive_names: &mut BTreeSet<String>) {
    // Split on commas at the top level (not inside parentheses)
    let mut depth: usize = 0;
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let part = text[start..i].trim();
                if let Some(name) = extract_recursive_op_name(part) {
                    recursive_names.insert(name);
                }
                start = i + 1;
            }
            _ => {}
        }
    }

    // Handle the last part
    if start < text.len() {
        let part = text[start..].trim();
        if let Some(name) = extract_recursive_op_name(part) {
            recursive_names.insert(name);
        }
    }
}

/// Extract the operator name from a RECURSIVE declaration like "Op(_)" or "Op(_, _)".
fn extract_recursive_op_name(decl: &str) -> Option<String> {
    let decl = decl.trim();
    if decl.is_empty() {
        return None;
    }

    // Find the opening parenthesis
    if let Some(open) = decl.find('(') {
        let name = decl[..open].trim();
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(name.to_string());
        }
    } else {
        // No parenthesis - just a name (zero-arity recursive operator, though rare)
        if decl.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(decl.to_string());
        }
    }

    None
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

    #[test]
    fn detect_pluscal_detects_algorithm_marker() {
        // Test --algorithm marker
        let pluscal_src = r#"
        ---- MODULE PlusCal ----
        (* --algorithm Counter
        begin
          skip;
        end algorithm; *)
        ====
        "#;
        assert!(detect_pluscal(pluscal_src));

        // Test --fair algorithm marker
        let fair_pluscal_src = r#"
        ---- MODULE FairPlusCal ----
        (* --fair algorithm Counter
        begin
          skip;
        end algorithm; *)
        ====
        "#;
        assert!(detect_pluscal(fair_pluscal_src));

        // Test non-PlusCal module
        let normal_src = r#"
        ---- MODULE Normal ----
        VARIABLES x
        Init == x = 0
        Next == x' = x + 1
        ====
        "#;
        assert!(!detect_pluscal(normal_src));
    }

    #[test]
    fn malformed_input_does_not_panic() {
        // Regression test for fuzzer-found bug: ')' before '(' in def head
        // The parser should handle malformed input gracefully without panicking
        let malformed_inputs = [
            ")]..(==y",
            "Op)==x",
            "Foo ) ( Bar",
            "((()))",
            "",
            "   ",
            "-----MODULE Test-----",
        ];

        for input in &malformed_inputs {
            // parse_tla_module_text may return Ok or Err, but must not panic
            let _ = parse_tla_module_text(input);
        }
    }

    #[test]
    fn parses_recursive_declarations() {
        let src = r#"
        ---- MODULE RecursiveDemo ----
        EXTENDS Naturals

        RECURSIVE Factorial(_)
        RECURSIVE SumSeq(_), SeqLen(_)

        Factorial(n) ==
            IF n <= 1 THEN 1
            ELSE n * Factorial(n - 1)

        SumSeq(s) ==
            IF s = <<>> THEN 0
            ELSE Head(s) + SumSeq(Tail(s))

        SeqLen(s) ==
            IF s = <<>> THEN 0
            ELSE 1 + SeqLen(Tail(s))

        NonRecursive(x) == x + 1
        ====
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");

        // Check recursive declarations are tracked
        assert!(m.recursive_declarations.contains("Factorial"));
        assert!(m.recursive_declarations.contains("SumSeq"));
        assert!(m.recursive_declarations.contains("SeqLen"));
        assert!(!m.recursive_declarations.contains("NonRecursive"));

        // Check definitions are marked as recursive
        assert!(m.definitions["Factorial"].is_recursive);
        assert!(m.definitions["SumSeq"].is_recursive);
        assert!(m.definitions["SeqLen"].is_recursive);
        assert!(!m.definitions["NonRecursive"].is_recursive);

        // Check the definitions have the expected bodies
        assert!(m.definitions["Factorial"].body.contains("Factorial(n - 1)"));
        assert!(m.definitions["SumSeq"].body.contains("SumSeq(Tail(s))"));
    }

    #[test]
    fn extracts_recursive_op_names() {
        // Test the helper function directly
        assert_eq!(
            extract_recursive_op_name("Factorial(_)"),
            Some("Factorial".to_string())
        );
        assert_eq!(
            extract_recursive_op_name("Sum(_, _)"),
            Some("Sum".to_string())
        );
        assert_eq!(
            extract_recursive_op_name("Op(_,_,_)"),
            Some("Op".to_string())
        );
        assert_eq!(
            extract_recursive_op_name("NoParams"),
            Some("NoParams".to_string())
        );
        assert_eq!(
            extract_recursive_op_name("  Spaced(x)  "),
            Some("Spaced".to_string())
        );
        assert_eq!(extract_recursive_op_name(""), None);
        assert_eq!(extract_recursive_op_name("   "), None);
    }
}
