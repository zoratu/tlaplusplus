use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
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
    /// Unnamed instances (INSTANCE M without alias) - definitions are merged directly
    #[serde(default)]
    pub unnamed_instances: Vec<TlaModuleInstance>,
    /// True if module contains PlusCal code (detected via --algorithm marker)
    pub is_pluscal: bool,
    /// Set of operator names declared with RECURSIVE
    #[serde(default)]
    pub recursive_declarations: BTreeSet<String>,
}

/// Standard library modules that are built-in and don't need to be loaded from disk.
/// These modules' operators are implemented directly in the evaluator.
const BUILTIN_MODULES: &[&str] = &[
    "Naturals",
    "Integers",
    "Reals",
    "Sequences",
    "FiniteSets",
    "TLC",
    "Bags",
    "Randomization",
    "Json",
    "TLCExt",
];

/// Check if a module name is a built-in standard library module.
pub fn is_builtin_module(name: &str) -> bool {
    BUILTIN_MODULES.contains(&name)
}

pub fn parse_tla_module_file(path: &Path) -> Result<TlaModule> {
    parse_tla_module_file_with_visited(path, &mut HashSet::new())
}

/// Internal function that tracks visited modules to detect circular EXTENDS.
fn parse_tla_module_file_with_visited(
    path: &Path,
    visited: &mut HashSet<String>,
) -> Result<TlaModule> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let mut module = parse_tla_module_text(&raw)?;
    module.path = path.display().to_string();

    // Detect PlusCal by looking for --algorithm or --fair algorithm marker in the raw text
    // PlusCal algorithms are typically in block comments: (* --algorithm Name ... *)
    module.is_pluscal = detect_pluscal(&raw);

    // Check for circular EXTENDS
    let canonical_path = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string();
    if visited.contains(&canonical_path) {
        return Err(anyhow!(
            "Circular EXTENDS detected: module '{}' has already been loaded",
            module.name
        ));
    }
    visited.insert(canonical_path);

    // Load extended modules and merge their definitions
    load_extended_modules(&mut module, path, visited)?;

    // Load module instances
    load_module_instances(&mut module, path)?;

    Ok(module)
}

/// Load modules referenced by EXTENDS declarations and merge their definitions
/// into the current module.
///
/// Definitions from extended modules are available in the extending module but can
/// be overridden. Variables and constants are also inherited.
fn load_extended_modules(
    module: &mut TlaModule,
    base_path: &Path,
    visited: &mut HashSet<String>,
) -> Result<()> {
    let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));

    for extended_name in module.extends.clone() {
        // Skip built-in modules - their operators are implemented in the evaluator
        if is_builtin_module(&extended_name) {
            continue;
        }

        // Try to find the extended module file
        let extended_path = module_dir.join(format!("{}.tla", extended_name));

        if extended_path.exists() {
            // Load the extended module recursively
            let extended_module = parse_tla_module_file_with_visited(&extended_path, visited)
                .with_context(|| {
                    format!(
                        "failed to load extended module '{}' from '{}'",
                        extended_name,
                        extended_path.display()
                    )
                })?;

            // Merge definitions from extended module into current module
            // Extended module definitions come first, current module definitions override them
            for (name, def) in extended_module.definitions {
                // Only insert if not already defined in current module (current overrides extended)
                if !module.definitions.contains_key(&name) {
                    module.definitions.insert(name, def);
                }
            }

            // Merge constants from extended module
            for constant in extended_module.constants {
                if !module.constants.contains(&constant) {
                    module.constants.push(constant);
                }
            }

            // Merge variables from extended module
            for variable in extended_module.variables {
                if !module.variables.contains(&variable) {
                    module.variables.push(variable);
                }
            }

            // Merge recursive declarations
            for rec_decl in extended_module.recursive_declarations {
                module.recursive_declarations.insert(rec_decl);
            }

            // Merge instances from extended module
            for (alias, instance) in extended_module.instances {
                if !module.instances.contains_key(&alias) {
                    module.instances.insert(alias, instance);
                }
            }

            // If extended module is PlusCal, current module is also considered PlusCal
            if extended_module.is_pluscal {
                module.is_pluscal = true;
            }
        } else {
            // Extended module not found - this might be an error or a missing dependency
            eprintln!(
                "Warning: Extended module '{}' not found at {}",
                extended_name,
                extended_path.display()
            );
        }
    }

    Ok(())
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

    // Load named instances (Alias == INSTANCE M)
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
            if !is_builtin_module(&instance.module_name) {
                eprintln!(
                    "Warning: Instance module '{}' not found at {}",
                    instance.module_name,
                    instance_path.display()
                );
            }
        }
    }

    // Load unnamed instances (INSTANCE M) and merge their definitions
    // Collect modules to merge first to avoid borrow checker issues
    let mut modules_to_merge: Vec<(usize, TlaModule, BTreeMap<String, String>)> = Vec::new();

    for (idx, instance) in module.unnamed_instances.iter().enumerate() {
        // Skip built-in modules - their operators are implemented in the evaluator
        if is_builtin_module(&instance.module_name) {
            continue;
        }

        let instance_path = module_dir.join(format!("{}.tla", instance.module_name));

        if instance_path.exists() {
            let instance_module = parse_tla_module_file(&instance_path).with_context(|| {
                format!(
                    "failed to load unnamed instance module '{}'",
                    instance.module_name
                )
            })?;

            modules_to_merge.push((idx, instance_module, instance.substitutions.clone()));
        } else {
            eprintln!(
                "Warning: Unnamed instance module '{}' not found at {}",
                instance.module_name,
                instance_path.display()
            );
        }
    }

    // Now merge the collected modules
    for (idx, instance_module, substitutions) in modules_to_merge {
        // For unnamed instances, merge definitions directly into the current module
        // Apply substitutions if present
        merge_instance_definitions(module, &instance_module, &substitutions);

        // Store the loaded module
        module.unnamed_instances[idx].module = Some(Box::new(instance_module));
    }

    Ok(())
}

/// Merge definitions from an instanced module into the current module.
/// Applies parameter substitutions from the WITH clause.
fn merge_instance_definitions(
    target: &mut TlaModule,
    source: &TlaModule,
    substitutions: &BTreeMap<String, String>,
) {
    // Merge definitions, applying substitutions
    for (name, def) in &source.definitions {
        if !target.definitions.contains_key(name) {
            let mut new_def = def.clone();
            // Apply substitutions to the definition body
            if !substitutions.is_empty() {
                new_def.body = apply_substitutions(&new_def.body, substitutions);
            }
            target.definitions.insert(name.clone(), new_def);
        }
    }

    // Merge variables (don't duplicate)
    for var in &source.variables {
        if !target.variables.contains(var) {
            target.variables.push(var.clone());
        }
    }

    // Merge constants (don't duplicate)
    for constant in &source.constants {
        if !target.constants.contains(constant) {
            target.constants.push(constant.clone());
        }
    }

    // Merge recursive declarations
    for rec_decl in &source.recursive_declarations {
        target.recursive_declarations.insert(rec_decl.clone());
    }

    // Merge named instances from source module
    for (alias, inst) in &source.instances {
        if !target.instances.contains_key(alias) {
            target.instances.insert(alias.clone(), inst.clone());
        }
    }
}

/// Apply substitutions to an expression string.
/// This does simple text replacement for identifier names.
fn apply_substitutions(expr: &str, substitutions: &BTreeMap<String, String>) -> String {
    let mut result = expr.to_string();
    for (from, to) in substitutions {
        // Replace whole-word occurrences only
        result = replace_identifier(&result, from, to);
    }
    result
}

/// Replace identifier occurrences in an expression, preserving word boundaries.
fn replace_identifier(expr: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(expr.len());
    let mut chars = expr.chars().peekable();
    let mut current_word = String::new();

    while let Some(c) = chars.next() {
        if c.is_alphanumeric() || c == '_' {
            current_word.push(c);
        } else {
            if !current_word.is_empty() {
                if current_word == from {
                    result.push_str(to);
                } else {
                    result.push_str(&current_word);
                }
                current_word.clear();
            }
            result.push(c);
        }
    }

    // Handle trailing word
    if !current_word.is_empty() {
        if current_word == from {
            result.push_str(to);
        } else {
            result.push_str(&current_word);
        }
    }

    result
}

pub fn parse_tla_module_text(input: &str) -> Result<TlaModule> {
    let cleaned = strip_comments(input);
    let mut pending_lines: VecDeque<String> = cleaned.lines().map(|line| line.to_string()).collect();
    let mut module = TlaModule::default();

    let mut current_def: Option<TlaDefinition> = None;
    let mut current_def_indent = 0usize;
    let mut mode = NameListMode::None;

    while let Some(line) = pending_lines.pop_front() {
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

        if trimmed.starts_with("ASSUME") || trimmed.starts_with("AXIOM") {
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
        //   INSTANCE ModuleName (unnamed - definitions merge into current module)
        //   INSTANCE ModuleName WITH Param1 <- Value1
        //   LOCAL INSTANCE ModuleName
        if let Some(instance) = parse_instance_declaration(trimmed) {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            module.instances.insert(instance.alias.clone(), instance);
            continue;
        }
        if let Some(instance) = parse_unnamed_instance_declaration(trimmed) {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            module.unnamed_instances.push(instance);
            continue;
        }

        let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
        let can_start_definition = current_def.is_none() || line_indent <= current_def_indent;
        if can_start_definition
            && let Some((lhs, rhs, remainder)) = split_definition_line_with_remainder(trimmed)
        {
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
            if let Some(remainder) = remainder {
                let indent_width = line.len() - line.trim_start().len();
                let indent = &line[..indent_width];
                pending_lines.push_front(format!("{indent}{remainder}"));
            }
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

fn split_definition_line_with_remainder(line: &str) -> Option<(&str, &str, Option<&str>)> {
    let (lhs, rhs) = split_definition_line(line)?;
    let remainder_start = find_inline_definition_start(rhs);
    let (body, remainder) = match remainder_start {
        Some(idx) => (rhs[..idx].trim_end(), Some(rhs[idx..].trim_start())),
        None => (rhs, None),
    };
    Some((lhs, body, remainder.filter(|rest| !rest.is_empty())))
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

fn find_inline_definition_start(rhs: &str) -> Option<usize> {
    let trimmed = rhs.trim_start();
    if trimmed.starts_with("LET ")
        || trimmed.starts_with("LET\n")
        || trimmed.starts_with("IF ")
        || trimmed.starts_with("CASE ")
    {
        return None;
    }

    let chars: Vec<(usize, char)> = rhs.char_indices().collect();
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;

    let mut i = 0usize;
    while i < chars.len() {
        let (byte_idx, ch) = chars[i];
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '<' {
                    angle += 1;
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '>' {
                    angle = angle.saturating_sub(1);
                    i += 1;
                }
            }
            _ => {}
        }

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 && ch.is_whitespace() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].1.is_whitespace() {
                j += 1;
            }
            if j < chars.len() {
                let candidate = &rhs[chars[j].0..];
                if let Some((lhs, _)) = split_definition_line(candidate)
                    && is_plausible_inline_definition_head(lhs)
                {
                    return Some(chars[j].0);
                }
            }
        }

        let _ = byte_idx;
        i += 1;
    }

    None
}

fn is_plausible_inline_definition_head(lhs: &str) -> bool {
    let lhs = lhs.trim();
    let lhs = lhs.strip_prefix("LOCAL ").unwrap_or(lhs).trim();
    matches!(lhs.chars().next(), Some(c) if c.is_alphabetic() || c == '_')
}

fn parse_def_head(lhs: &str) -> (String, Vec<String>) {
    // Strip LOCAL prefix if present
    let lhs = if let Some(rest) = lhs.strip_prefix("LOCAL ") {
        rest.trim()
    } else {
        lhs
    };

    if let Some((open_delim, close_delim)) = first_param_delims(lhs)
        && let Some((name, params)) = parse_def_head_with_delims(lhs, open_delim, close_delim)
    {
        return (name, params);
    }

    let name = lhs
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn parse_def_head_with_delims(
    lhs: &str,
    open_delim: char,
    close_delim: char,
) -> Option<(String, Vec<String>)> {
    let open = lhs.find(open_delim)?;
    let close = lhs.rfind(close_delim)?;
    if close <= open {
        return None;
    }

    let name = lhs[..open].trim().to_string();
    let params = split_operator_params(&lhs[open + 1..close]);
    Some((name, params))
}

fn first_param_delims(lhs: &str) -> Option<(char, char)> {
    match (lhs.find('('), lhs.find('[')) {
        (Some(paren), Some(bracket)) if bracket < paren => Some(('[', ']')),
        (Some(_), Some(_)) => Some(('(', ')')),
        (Some(_), None) => Some(('(', ')')),
        (None, Some(_)) => Some(('[', ']')),
        (None, None) => None,
    }
}

fn split_operator_params(params_text: &str) -> Vec<String> {
    split_top_level_commas(params_text)
        .into_iter()
        .flat_map(|part| {
            let lhs = top_level_in_pos(part)
                .map(|pos| &part[..pos])
                .unwrap_or(part);
            split_top_level_commas(lhs)
                .into_iter()
                .map(normalize_operator_param)
                .filter(|param| !param.is_empty())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn normalize_operator_param(param: &str) -> String {
    let param = param.trim();
    if let Some(paren_pos) = param.find('(') {
        return param[..paren_pos].trim().to_string();
    }
    param.to_string()
}

fn split_top_level_commas(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();

    let mut i = 0usize;
    while i < chars.len() {
        let (byte_idx, ch) = chars[i];
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '<' {
                    angle += 1;
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '>' {
                    angle = angle.saturating_sub(1);
                    i += 1;
                }
            }
            ',' if paren == 0 && bracket == 0 && brace == 0 && angle == 0 => {
                parts.push(text[start..byte_idx].trim());
                start = byte_idx + ch.len_utf8();
            }
            _ => {}
        }
        i += 1;
    }

    parts.push(text[start..].trim());
    parts
}

fn top_level_in_pos(text: &str) -> Option<usize> {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();

    let mut i = 0usize;
    while i < chars.len() {
        let (byte_idx, ch) = chars[i];
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '<' {
                    angle += 1;
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '>' {
                    angle = angle.saturating_sub(1);
                    i += 1;
                }
            }
            '\\' if paren == 0
                && bracket == 0
                && brace == 0
                && angle == 0
                && text[byte_idx..].starts_with("\\in") =>
            {
                return Some(byte_idx);
            }
            _ => {}
        }
        i += 1;
    }

    None
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
    trimmed.len() >= 4 && trimmed.chars().all(|c| c == '-')
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
    let substitutions = parse_with_substitutions(with_clause);

    Some(TlaModuleInstance {
        alias: alias.to_string(),
        module_name: module_name.to_string(),
        substitutions,
        is_local,
        module: None,
    })
}

/// Parse WITH substitutions from a WITH clause string.
/// Format: "Param1 <- Value1, Param2 <- Value2"
/// Handles nested brackets correctly (but NOT `<` from `<-`).
fn parse_with_substitutions(with_clause: Option<&str>) -> BTreeMap<String, String> {
    let mut substitutions = BTreeMap::new();
    let Some(with_str) = with_clause else {
        return substitutions;
    };

    // Split on commas at the top level (not inside brackets)
    // But be careful: `<` followed by `-` is the substitution operator, not a bracket
    let mut depth: usize = 0;
    let mut start = 0;
    let chars: Vec<char> = with_str.chars().collect();
    let n = chars.len();

    let mut i = 0;
    while i < n {
        let c = chars[i];
        match c {
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth = depth.saturating_sub(1),
            '<' => {
                // Check if this is `<-` (substitution) or `<<` (tuple) or just `<`
                if i + 1 < n && chars[i + 1] == '<' {
                    // `<<` - tuple opening, increase depth
                    depth += 1;
                    i += 1; // skip the second `<`
                } else if i + 1 < n && chars[i + 1] == '-' {
                    // `<-` - substitution operator, NOT a bracket
                    // Just skip the `-` next iteration
                } else {
                    // Single `<` - could be comparison, don't treat as bracket
                }
            }
            '>' => {
                // Check if this is `>>` (tuple closing)
                if i + 1 < n && chars[i + 1] == '>' {
                    depth = depth.saturating_sub(1);
                    i += 1; // skip the second `>`
                }
                // Single `>` could be comparison, don't treat as bracket
            }
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
        i += 1;
    }

    // Handle the last substitution
    if start < with_str.len() {
        let subst = &with_str[start..];
        if let Some((param, value)) = subst.split_once("<-") {
            substitutions.insert(param.trim().to_string(), value.trim().to_string());
        }
    }

    substitutions
}

/// Parse an unnamed INSTANCE declaration (without alias)
/// Formats:
///   INSTANCE ModuleName
///   INSTANCE ModuleName WITH Param1 <- Value1, Param2 <- Value2
///   LOCAL INSTANCE ModuleName
fn parse_unnamed_instance_declaration(line: &str) -> Option<TlaModuleInstance> {
    let line = line.trim();

    // Must start with INSTANCE or LOCAL INSTANCE (no "==" before it)
    // Reject if it contains "==" before INSTANCE (that would be a named instance)
    if let Some(eq_pos) = line.find("==") {
        if let Some(inst_pos) = line.find("INSTANCE") {
            if eq_pos < inst_pos {
                // This is a named instance (Alias == INSTANCE ...), not unnamed
                return None;
            }
        }
    }

    // Check for LOCAL prefix
    let (is_local, rest) = if let Some(rest) = line.strip_prefix("LOCAL ") {
        (true, rest.trim())
    } else {
        (false, line)
    };

    // Must start with INSTANCE
    let after_instance = rest.strip_prefix("INSTANCE")?.trim();
    if after_instance.is_empty() {
        return None;
    }

    // Check for WITH clause
    let (module_name, with_clause) = if let Some(idx) = after_instance.find(" WITH ") {
        (
            after_instance[..idx].trim(),
            Some(after_instance[idx + " WITH ".len()..].trim()),
        )
    } else {
        (after_instance, None)
    };

    // Validate module name is a simple identifier
    if !module_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    // Parse substitutions from WITH clause
    let substitutions = parse_with_substitutions(with_clause);

    // For unnamed instances, use the module name as alias (for internal tracking)
    // but definitions will be merged directly
    Some(TlaModuleInstance {
        alias: format!("__unnamed__{}", module_name),
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
    fn parses_bracketed_operator_parameters() {
        let src = r#"
        ---- MODULE Demo ----
        EXTENDS Naturals

        HaveQuorumFrom[leader \in Node] == leader
        GetDirection[current, destination \in Floor] == <<current, destination>>
        HigherOrder[Op(_), value \in Values] == Op(value)
        ====
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(m.definitions["HaveQuorumFrom"].params, vec!["leader"]);
        assert_eq!(
            m.definitions["GetDirection"].params,
            vec!["current", "destination"]
        );
        assert_eq!(m.definitions["HigherOrder"].params, vec!["Op", "value"]);
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

    #[test]
    fn extends_inherits_definitions_from_base_module() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create base module with Init, Next, Spec definitions
        let base_module = tmp.join("DieHarder.tla");
        fs::write(
            &base_module,
            r#"
---- MODULE DieHarder ----
EXTENDS Naturals

VARIABLES small, big

Init ==
    /\ small = 0
    /\ big = 0

Next ==
    \/ /\ small' = 3
       /\ big' = big
    \/ /\ big' = 5
       /\ small' = small
    \/ /\ small' = 0
       /\ big' = big
    \/ /\ big' = 0
       /\ small' = small

Spec == Init /\ [][Next]_<<small, big>>

Goal == small + big <= 10
====
"#,
        )
        .expect("base module should be written");

        // Create extending module
        let extending_module = tmp.join("MCDieHarder.tla");
        fs::write(
            &extending_module,
            r#"
---- MODULE MCDieHarder ----
EXTENDS DieHarder

CONSTANTS MaxVal

TypeOK == small \in 0..MaxVal /\ big \in 0..MaxVal
====
"#,
        )
        .expect("extending module should be written");

        // Parse the extending module
        let module =
            parse_tla_module_file(&extending_module).expect("extending module should parse");

        // Check that inherited definitions are present
        assert!(
            module.definitions.contains_key("Init"),
            "Init should be inherited from DieHarder"
        );
        assert!(
            module.definitions.contains_key("Next"),
            "Next should be inherited from DieHarder"
        );
        assert!(
            module.definitions.contains_key("Spec"),
            "Spec should be inherited from DieHarder"
        );
        assert!(
            module.definitions.contains_key("Goal"),
            "Goal should be inherited from DieHarder"
        );

        // Check that local definitions are present
        assert!(
            module.definitions.contains_key("TypeOK"),
            "TypeOK should be defined in MCDieHarder"
        );

        // Check inherited variables
        assert!(
            module.variables.contains(&"small".to_string()),
            "small should be inherited"
        );
        assert!(
            module.variables.contains(&"big".to_string()),
            "big should be inherited"
        );

        // Check local constants
        assert!(
            module.constants.contains(&"MaxVal".to_string()),
            "MaxVal should be local constant"
        );

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_local_definitions_override_inherited() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-override-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create base module with an Init definition
        let base_module = tmp.join("Base.tla");
        fs::write(
            &base_module,
            r#"
---- MODULE Base ----
VARIABLES x
Init == x = 0
SharedOp == x + 1
====
"#,
        )
        .expect("base module should be written");

        // Create extending module that overrides Init
        let extending_module = tmp.join("Derived.tla");
        fs::write(
            &extending_module,
            r#"
---- MODULE Derived ----
EXTENDS Base

\* Override Init
Init == x = 100

\* New operator
NewOp == x * 2
====
"#,
        )
        .expect("extending module should be written");

        let module =
            parse_tla_module_file(&extending_module).expect("extending module should parse");

        // Check that local Init overrides inherited Init
        let init_def = module.definitions.get("Init").expect("Init should exist");
        assert!(
            init_def.body.contains("100"),
            "Init should be overridden to use 100, got: {}",
            init_def.body
        );

        // Check that inherited SharedOp is present
        assert!(
            module.definitions.contains_key("SharedOp"),
            "SharedOp should be inherited"
        );

        // Check that NewOp is present
        assert!(
            module.definitions.contains_key("NewOp"),
            "NewOp should be defined"
        );

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_chain_inherits_transitively() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-chain-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create base module
        let base_module = tmp.join("Level0.tla");
        fs::write(
            &base_module,
            r#"
---- MODULE Level0 ----
VARIABLES x
BaseOp == x + 1
====
"#,
        )
        .expect("base module should be written");

        // Create intermediate module
        let mid_module = tmp.join("Level1.tla");
        fs::write(
            &mid_module,
            r#"
---- MODULE Level1 ----
EXTENDS Level0
MidOp == x + 2
====
"#,
        )
        .expect("intermediate module should be written");

        // Create final extending module
        let top_module = tmp.join("Level2.tla");
        fs::write(
            &top_module,
            r#"
---- MODULE Level2 ----
EXTENDS Level1
TopOp == x + 3
====
"#,
        )
        .expect("top module should be written");

        let module = parse_tla_module_file(&top_module).expect("top module should parse");

        // Check that all operators are transitively inherited
        assert!(
            module.definitions.contains_key("BaseOp"),
            "BaseOp should be inherited from Level0"
        );
        assert!(
            module.definitions.contains_key("MidOp"),
            "MidOp should be inherited from Level1"
        );
        assert!(
            module.definitions.contains_key("TopOp"),
            "TopOp should be defined in Level2"
        );

        // Check that variable is inherited through the chain
        assert!(
            module.variables.contains(&"x".to_string()),
            "x should be inherited"
        );

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_builtin_modules_are_skipped() {
        // Test that built-in modules don't cause errors
        let src = r#"
---- MODULE TestBuiltin ----
EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC
VARIABLES x
Init == x = 0
====
"#;
        let module = parse_tla_module_text(src).expect("parse should work");

        // Should have the EXTENDS list but no definitions from builtins
        // (they are implemented in the evaluator)
        assert!(module.extends.contains(&"Naturals".to_string()));
        assert!(module.extends.contains(&"Integers".to_string()));
        assert!(module.extends.contains(&"Sequences".to_string()));
        assert!(module.extends.contains(&"FiniteSets".to_string()));
        assert!(module.extends.contains(&"TLC".to_string()));

        // Local definitions should be present
        assert!(module.definitions.contains_key("Init"));
    }

    #[test]
    fn is_builtin_module_returns_correct_results() {
        assert!(is_builtin_module("Naturals"));
        assert!(is_builtin_module("Integers"));
        assert!(is_builtin_module("Sequences"));
        assert!(is_builtin_module("FiniteSets"));
        assert!(is_builtin_module("TLC"));
        assert!(is_builtin_module("Bags"));
        assert!(is_builtin_module("Reals"));
        assert!(is_builtin_module("Randomization"));
        assert!(is_builtin_module("Json"));
        assert!(is_builtin_module("TLCExt"));

        // Non-builtin modules
        assert!(!is_builtin_module("MyModule"));
        assert!(!is_builtin_module("DieHarder"));
        assert!(!is_builtin_module(""));
    }

    #[test]
    fn parses_multiline_if_then_else_with_else_on_separate_line() {
        // This test reproduces the bug where ELSE on a separate line causes
        // the ELSE branch to be empty or missing
        let src = r#"
---- MODULE MultiLineIfThenElse ----
EXTENDS Integers

Loop(self) ==
    IF condition
    THEN something
    ELSE other_thing

SimpleOp == 42
====
"#;

        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that Loop definition exists and has complete body
        assert!(m.definitions.contains_key("Loop"), "Loop should be defined");
        let loop_def = &m.definitions["Loop"];

        // The body should contain all parts of the IF-THEN-ELSE
        assert!(
            loop_def.body.contains("IF"),
            "Body should contain IF: {}",
            loop_def.body
        );
        assert!(
            loop_def.body.contains("THEN"),
            "Body should contain THEN: {}",
            loop_def.body
        );
        assert!(
            loop_def.body.contains("ELSE"),
            "Body should contain ELSE: {}",
            loop_def.body
        );
        assert!(
            loop_def.body.contains("other_thing"),
            "Body should contain ELSE branch content: {}",
            loop_def.body
        );

        // SimpleOp should also be defined
        assert!(
            m.definitions.contains_key("SimpleOp"),
            "SimpleOp should be defined"
        );
    }

    #[test]
    fn parses_nested_multiline_if_then_else() {
        // Test nested IF-THEN-ELSE spanning multiple lines similar to DiningPhilosophers
        let src = r#"
---- MODULE NestedIfThenElse ----
EXTENDS Integers

Loop(self) ==
    IF outer_condition
    THEN outer_then
    ELSE /\ IF /\ forks[RightFork(self)].holder = self
              /\ ~forks[RightFork(self)].clean
         THEN /\ forks' = [forks EXCEPT ![RightFork(self)] = something]
         ELSE /\ TRUE
              /\ UNCHANGED forks

Next == TRUE
====
"#;

        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that Loop definition exists and has complete body with nested IF
        assert!(m.definitions.contains_key("Loop"), "Loop should be defined");
        let loop_def = &m.definitions["Loop"];

        // Count IF and ELSE occurrences - there should be 2 of each (outer and nested)
        let if_count = loop_def.body.matches("IF").count();
        let else_count = loop_def.body.matches("ELSE").count();

        assert_eq!(
            if_count, 2,
            "Body should have 2 IFs (outer and nested): {}",
            loop_def.body
        );
        assert_eq!(
            else_count, 2,
            "Body should have 2 ELSEs (outer and nested): {}",
            loop_def.body
        );

        // The nested ELSE should contain UNCHANGED forks
        assert!(
            loop_def.body.contains("UNCHANGED forks"),
            "Body should contain UNCHANGED forks from nested ELSE: {}",
            loop_def.body
        );

        // Next should also be defined
        assert!(m.definitions.contains_key("Next"), "Next should be defined");
    }

    #[test]
    fn parses_named_instance_declaration() {
        let src = r#"
---- MODULE TestInstance ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH Node <- {1, 2, 3}

VARIABLES x
Init == x = 0
Next == x' = x + 1
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that the named instance is parsed
        assert!(
            m.instances.contains_key("Helper"),
            "Helper instance should be parsed"
        );

        let helper = m.instances.get("Helper").unwrap();
        assert_eq!(helper.module_name, "CoverageHelper");
        assert_eq!(helper.alias, "Helper");
        assert!(
            helper.substitutions.contains_key("Node"),
            "Should have Node substitution"
        );
        assert_eq!(helper.substitutions.get("Node").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn parses_unnamed_instance_declaration() {
        let src = r#"
---- MODULE TestUnnamedInstance ----
EXTENDS Naturals

INSTANCE Sailfish

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that the unnamed instance is parsed
        assert_eq!(
            m.unnamed_instances.len(),
            1,
            "Should have one unnamed instance"
        );

        let instance = &m.unnamed_instances[0];
        assert_eq!(instance.module_name, "Sailfish");
        assert!(
            instance.alias.starts_with("__unnamed__"),
            "Alias should be auto-generated"
        );
    }

    #[test]
    fn parses_unnamed_instance_with_substitution() {
        let src = r#"
---- MODULE TestUnnamedInstanceWithSubst ----
EXTENDS Naturals

INSTANCE Sailfish WITH Node <- Servers, F <- Faulty

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that the unnamed instance with substitutions is parsed
        assert_eq!(
            m.unnamed_instances.len(),
            1,
            "Should have one unnamed instance"
        );

        let instance = &m.unnamed_instances[0];
        assert_eq!(instance.module_name, "Sailfish");
        assert!(
            instance.substitutions.contains_key("Node"),
            "Should have Node substitution"
        );
        assert_eq!(instance.substitutions.get("Node").unwrap(), "Servers");
        assert!(
            instance.substitutions.contains_key("F"),
            "Should have F substitution"
        );
        assert_eq!(instance.substitutions.get("F").unwrap(), "Faulty");
    }

    #[test]
    fn parses_local_instance() {
        let src = r#"
---- MODULE TestLocalInstance ----
EXTENDS Naturals

LOCAL INSTANCE Helpers

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");

        // Check that the LOCAL unnamed instance is parsed
        assert_eq!(
            m.unnamed_instances.len(),
            1,
            "Should have one unnamed instance"
        );

        let instance = &m.unnamed_instances[0];
        assert_eq!(instance.module_name, "Helpers");
        assert!(instance.is_local, "Instance should be marked as LOCAL");
    }

    #[test]
    fn unnamed_instance_merges_definitions() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-unnamed-instance-merge-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create the instanced module
        let instanced_module = tmp.join("BaseModule.tla");
        fs::write(
            &instanced_module,
            r#"
---- MODULE BaseModule ----
EXTENDS Naturals

CONSTANT Param

BaseInit == TRUE
BaseNext == TRUE
BaseHelper(x) == x + 1
====
"#,
        )
        .expect("instanced module should be written");

        // Create the main module that instances BaseModule
        let main_module = tmp.join("MainModule.tla");
        fs::write(
            &main_module,
            r#"
---- MODULE MainModule ----
EXTENDS Naturals

INSTANCE BaseModule WITH Param <- 42

VARIABLES x
Init == /\ x = 0 /\ BaseInit
Next == x' = BaseHelper(x)
====
"#,
        )
        .expect("main module should be written");

        let module = parse_tla_module_file(&main_module).expect("main module should parse");

        // Check that definitions from BaseModule are merged
        assert!(
            module.definitions.contains_key("BaseInit"),
            "BaseInit should be merged from BaseModule"
        );
        assert!(
            module.definitions.contains_key("BaseNext"),
            "BaseNext should be merged from BaseModule"
        );
        assert!(
            module.definitions.contains_key("BaseHelper"),
            "BaseHelper should be merged from BaseModule"
        );

        // Check that local definitions are present
        assert!(
            module.definitions.contains_key("Init"),
            "Init should be defined in MainModule"
        );
        assert!(
            module.definitions.contains_key("Next"),
            "Next should be defined in MainModule"
        );

        // Check that constant from BaseModule is available (though not assigned a value)
        // The substitution should have been applied
        let base_helper = module.definitions.get("BaseHelper").unwrap();
        // The body should have Param replaced with 42
        // (Note: substitution is text-based, so this should work)

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parses_multiple_simple_definitions_on_one_line() {
        let src = r#"
---- MODULE SlidingLike ----
EXTENDS Integers

W == 4 H == 5
Pos == 0 .. W + H
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(module.definitions.get("W").map(|def| def.body.as_str()), Some("4"));
        assert_eq!(module.definitions.get("H").map(|def| def.body.as_str()), Some("5"));
        assert_eq!(
            module.definitions.get("Pos").map(|def| def.body.as_str()),
            Some("0 .. W + H")
        );
    }

    #[test]
    fn does_not_split_let_bodies_into_top_level_definitions() {
        let src = r#"
---- MODULE SlidingLike ----
EXTENDS Integers

VARIABLES board
Next == LET empty == Pos \ UNION board
        IN  \E e \in empty : board' \in update(e, empty)
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert!(module.definitions.contains_key("Next"));
        assert!(!module.definitions.contains_key("empty"));
        assert!(
            module
                .definitions
                .get("Next")
                .unwrap()
                .body
                .starts_with("LET empty ==")
        );
    }

    #[test]
    fn strips_tlc_generated_definition_separators() {
        let src = r#"
---- MODULE GeneratedModel ----
CONSTANTS a1, a2

const_vals ==
{a1, a2}
----

def_ov ==
0..2
----
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(
            module
                .definitions
                .get("const_vals")
                .map(|def| def.body.trim_end()),
            Some("{a1, a2}")
        );
        assert_eq!(
            module
                .definitions
                .get("def_ov")
                .map(|def| def.body.trim_end()),
            Some("0..2")
        );
    }

    #[test]
    fn top_level_assume_does_not_extend_previous_definition() {
        let src = r#"
---- MODULE SpanTreeRandomLike ----
Edges ==
  UNION {{1}, {2}}

ASSUME TRUE
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        let edges = module
            .definitions
            .get("Edges")
            .expect("Edges should be defined");
        assert_eq!(edges.body.trim(), "UNION {{1}, {2}}");
    }
}
