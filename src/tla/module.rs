use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::path::{Path, PathBuf};

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
    "DyadicRationals",
    "SequencesExt",
    "Functions",
    "FiniteSetsExt",
    "UndirectedGraphs",
    "Folds",
    "Graphs",
    // Proof modules (no operators, just need to be skipped during loading)
    "FiniteSetTheorems",
    "NaturalsInduction",
    "SequenceTheorems",
    "SequencesExtTheorems",
    "FoldsTheorems",
    "FunctionTheorems",
    "FiniteSetsExtTheorems",
    // Other community modules
    "Relation",
    "Combinatorics",
    "BagsExt",
    "Bitwise",
    "Statistics",
    "VectorClocks",
];

/// Check if a module name is a built-in standard library module.
pub fn is_builtin_module(name: &str) -> bool {
    BUILTIN_MODULES.contains(&name)
}

pub fn parse_tla_module_file(path: &Path) -> Result<TlaModule> {
    parse_tla_module_file_with_visited(path, &mut HashSet::new())
}

fn is_module_terminator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 4 && trimmed.chars().all(|c| c == '=')
}

fn extract_named_module_text(input: &str, module_name: &str) -> Option<String> {
    let mut in_target = false;
    let mut buffer = String::new();

    for line in input.lines() {
        let trimmed = line.trim();
        let header_name = trimmed
            .find("MODULE ")
            .and_then(|idx| trimmed[idx + "MODULE ".len()..].split_whitespace().next());

        if let Some(name) = header_name {
            if name == module_name {
                in_target = true;
            } else if in_target {
                break;
            }
        }

        if !in_target {
            continue;
        }

        buffer.push_str(line);
        buffer.push('\n');

        if is_module_terminator(trimmed) {
            return Some(buffer);
        }
    }

    None
}

fn extract_first_module_text(input: &str) -> Option<String> {
    let module_name = input
        .lines()
        .find_map(|line| extract_module_name(line.trim()))?;
    extract_named_module_text(input, &module_name)
}

fn library_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(paths) = std::env::var("TLA_LIBRARY_PATH") {
        for path in std::env::split_paths(&paths) {
            roots.push(path);
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        roots.push(home.join("src/CommunityModules/modules"));
        roots.push(home.join("src/communitymodules/modules"));
        roots.push(home.join("src/tlaplus-community-modules/modules"));
        roots.push(home.join("src/tlaplus/CommunityModules/modules"));
    }

    roots
}

fn resolve_module_path(base_path: &Path, module_name: &str) -> Option<PathBuf> {
    resolve_module_path_with_roots(base_path, module_name, &library_search_roots())
}

fn resolve_module_path_with_roots(
    base_path: &Path,
    module_name: &str,
    library_roots: &[PathBuf],
) -> Option<PathBuf> {
    let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));
    let mut candidates = Vec::with_capacity(1 + library_roots.len());
    candidates.push(module_dir.join(format!("{module_name}.tla")));
    candidates.extend(
        library_roots
            .iter()
            .map(|root| root.join(format!("{module_name}.tla"))),
    );

    candidates
        .into_iter()
        .find(|candidate| candidate.exists())
        .or_else(|| resolve_unique_module_path_in_ancestor_tree(base_path, module_name))
}

fn resolve_unique_module_path_in_ancestor_tree(
    base_path: &Path,
    module_name: &str,
) -> Option<PathBuf> {
    let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));
    let repo_root = module_dir
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists());
    let specs_root = module_dir.ancestors().find(|ancestor| {
        ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| matches!(name, "specifications" | "examples" | "models"))
    });

    let mut roots = Vec::new();
    if let Some(root) = repo_root {
        roots.push(root.to_path_buf());
    }
    if let Some(root) = specs_root
        && !roots.iter().any(|existing| existing == root)
    {
        roots.push(root.to_path_buf());
    }

    for root in roots {
        if let Some(path) = find_unique_module_in_tree(&root, module_name) {
            return Some(path);
        }
    }

    None
}

fn find_unique_module_in_tree(root: &Path, module_name: &str) -> Option<PathBuf> {
    let target = format!("{module_name}.tla");
    let mut queue = VecDeque::from([root.to_path_buf()]);
    let mut matches = Vec::new();

    while let Some(dir) = queue.pop_front() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skip = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with('.') || matches!(name, "target" | "node_modules")
                    });
                if !skip {
                    queue.push_back(path);
                }
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == target)
            {
                matches.push(path);
                if matches.len() > 1 {
                    return None;
                }
            }
        }
    }

    matches.into_iter().next()
}

/// Internal function that tracks visited modules to detect circular EXTENDS.
fn parse_tla_module_file_with_visited(
    path: &Path,
    visited: &mut HashSet<String>,
) -> Result<TlaModule> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let module_text = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| extract_named_module_text(&raw, stem))
        .or_else(|| extract_first_module_text(&raw))
        .unwrap_or(raw);
    parse_tla_module_raw_with_visited(path, &module_text, visited)
}

fn parse_tla_module_raw_with_visited(
    path: &Path,
    raw: &str,
    visited: &mut HashSet<String>,
) -> Result<TlaModule> {
    let mut module = parse_tla_module_text(raw)?;
    module.path = path.display().to_string();

    // Detect PlusCal by looking for --algorithm or --fair algorithm marker in the raw text
    // PlusCal algorithms are typically in block comments: (* --algorithm Name ... *)
    module.is_pluscal = detect_pluscal(&raw);

    // Check for circular EXTENDS
    let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let visited_key = format!("{}::{}", canonical_path.display(), module.name);
    if visited.contains(&visited_key) {
        return Err(anyhow!(
            "Circular EXTENDS detected: module '{}' has already been loaded",
            module.name
        ));
    }
    visited.insert(visited_key.clone());

    let result = (|| {
        // Load extended modules and merge their definitions
        load_extended_modules(&mut module, path, visited)?;

        // Load module instances
        load_module_instances(&mut module, path)?;

        Ok(module)
    })();

    visited.remove(&visited_key);
    result
}

fn parse_embedded_module_from_file_with_visited(
    path: &Path,
    module_name: &str,
    visited: &mut HashSet<String>,
) -> Result<Option<TlaModule>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let Some(module_text) = extract_named_module_text(&raw, module_name) else {
        return Ok(None);
    };
    Ok(Some(parse_tla_module_raw_with_visited(
        path,
        &module_text,
        visited,
    )?))
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
    for extended_name in module.extends.clone() {
        // Skip built-in modules - their operators are implemented in the evaluator
        if is_builtin_module(&extended_name) {
            // Inject synthetic definitions for community modules that define constants
            if extended_name == "DyadicRationals" {
                module
                    .definitions
                    .entry("Zero".to_string())
                    .or_insert(TlaDefinition {
                        name: "Zero".to_string(),
                        params: vec![],
                        body: "[num |-> 0, den |-> 1]".to_string(),
                        is_recursive: false,
                    });
                module
                    .definitions
                    .entry("One".to_string())
                    .or_insert(TlaDefinition {
                        name: "One".to_string(),
                        params: vec![],
                        body: "[num |-> 1, den |-> 1]".to_string(),
                        is_recursive: false,
                    });
            }
            // Inject theorem definitions as TRUE for proof modules.
            // TLC treats these as axioms (trusted without proof).
            if extended_name == "FiniteSetTheorems" {
                for thm in &[
                    "FS_CardinalityType",
                    "FS_EmptySet",
                    "FS_Singleton",
                    "FS_Subset",
                    "FS_Union",
                    "FS_Intersection",
                    "FS_Difference",
                    "FS_UNION",
                    "FS_SUBSET",
                    "FS_Product",
                    "FS_Interval",
                    "FS_AddElement",
                    "FS_RemoveElement",
                    "FS_Surjection",
                    "FS_Injection",
                    "FS_Bijection",
                    "FS_Image",
                    "FS_PigeonHole",
                    "FS_Induction",
                    "FS_WFInduction",
                    "FS_SameCardinalityBij",
                    "FS_SurjCardinalityBound",
                    "FS_SurjSameCardinalityImpliesInj",
                    "FS_BoundedSetOfNaturals",
                    "FS_StrictSubsetOrderingWellFounded",
                    "FS_MajoritiesIntersect",
                    "FS_NatSurjection",
                    "FS_NatBijection",
                    "FS_CountingElements",
                    "FS_FiniteSubsetsOfFinite",
                ] {
                    module
                        .definitions
                        .entry(thm.to_string())
                        .or_insert(TlaDefinition {
                            name: thm.to_string(),
                            params: vec![],
                            body: "TRUE".to_string(),
                            is_recursive: false,
                        });
                }
            }
            if extended_name == "NaturalsInduction" {
                for thm in &[
                    "NatInduction",
                    "DownwardNatInduction",
                    "GeneralNatInduction",
                    "SmallestNatural",
                    "RecursiveFcnOfNat",
                    "NatInductiveDef",
                    "RecursiveFcnOfNatType",
                    "NatInductiveDefType",
                    "RecursiveFcnOfNatUnique",
                    "NatInductiveUnique",
                    "FiniteNatInductiveDef",
                    "FiniteNatInductiveDefType",
                    "FiniteNatInductiveUnique",
                ] {
                    module
                        .definitions
                        .entry(thm.to_string())
                        .or_insert(TlaDefinition {
                            name: thm.to_string(),
                            params: vec![],
                            body: "TRUE".to_string(),
                            is_recursive: false,
                        });
                }
            }
            if extended_name == "Graphs" {
                module
                    .definitions
                    .entry("EmptyGraph".to_string())
                    .or_insert(TlaDefinition {
                        name: "EmptyGraph".to_string(),
                        params: vec![],
                        body: "[node |-> {}, edge |-> {}]".to_string(),
                        is_recursive: false,
                    });
            }
            continue;
        }

        if let Some(extended_module) =
            parse_embedded_module_from_file_with_visited(base_path, &extended_name, visited)?
        {
            // Merge definitions from extended module into current module
            // Extended module definitions come first, current module definitions override them
            for (name, def) in extended_module.definitions {
                if !module.definitions.contains_key(&name) {
                    module.definitions.insert(name, def);
                }
            }

            for constant in extended_module.constants {
                if !module.constants.contains(&constant) {
                    module.constants.push(constant);
                }
            }

            for variable in extended_module.variables {
                if !module.variables.contains(&variable) {
                    module.variables.push(variable);
                }
            }

            for rec_decl in extended_module.recursive_declarations {
                module.recursive_declarations.insert(rec_decl);
            }

            for (alias, instance) in extended_module.instances {
                if !module.instances.contains_key(&alias) {
                    module.instances.insert(alias, instance);
                }
            }

            if extended_module.is_pluscal {
                module.is_pluscal = true;
            }
        } else if let Some(extended_path) = resolve_module_path(base_path, &extended_name) {
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
            let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));
            let extended_path = module_dir.join(format!("{}.tla", extended_name));
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
    // Load named instances (Alias == INSTANCE M)
    for (alias, instance) in module.instances.iter_mut() {
        if let Some(instance_module) = parse_embedded_module_from_file_with_visited(
            base_path,
            &instance.module_name,
            &mut HashSet::new(),
        )? {
            instance.module = Some(Box::new(instance_module));
        } else if let Some(instance_path) = resolve_module_path(base_path, &instance.module_name) {
            // Load the module
            let instance_module = parse_tla_module_file(&instance_path).with_context(|| {
                format!(
                    "failed to load instance module '{}' for alias '{}'",
                    instance.module_name, alias
                )
            })?;

            instance.module = Some(Box::new(instance_module));
        } else {
            let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));
            let instance_path = module_dir.join(format!("{}.tla", instance.module_name));
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

        if let Some(instance_module) = parse_embedded_module_from_file_with_visited(
            base_path,
            &instance.module_name,
            &mut HashSet::new(),
        )? {
            modules_to_merge.push((idx, instance_module, instance.substitutions.clone()));
        } else if let Some(instance_path) = resolve_module_path(base_path, &instance.module_name) {
            let instance_module = parse_tla_module_file(&instance_path).with_context(|| {
                format!(
                    "failed to load unnamed instance module '{}'",
                    instance.module_name
                )
            })?;

            modules_to_merge.push((idx, instance_module, instance.substitutions.clone()));
        } else {
            let module_dir = base_path.parent().unwrap_or_else(|| Path::new("."));
            let instance_path = module_dir.join(format!("{}.tla", instance.module_name));
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
    let mut pending_lines: VecDeque<String> =
        cleaned.lines().map(|line| line.to_string()).collect();
    let mut module = TlaModule::default();

    let mut current_def: Option<TlaDefinition> = None;
    let mut current_def_indent = 0usize;
    let mut mode = NameListMode::None;

    while let Some(line) = pending_lines.pop_front() {
        let mut definition_line = line.clone();
        let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
        let raw_trimmed = line.trim();
        let can_start_definition = current_def.is_none()
            || line_indent <= current_def_indent
            || current_def
                .as_ref()
                .is_some_and(|def| can_start_indented_definition_after_gap(def, raw_trimmed));
        if can_start_definition && definition_head_needs_continuation(raw_trimmed) {
            while let Some(next_line) = pending_lines.front() {
                definition_line.push('\n');
                definition_line.push_str(next_line.trim_end());
                pending_lines.pop_front();
                if split_definition_line_with_remainder(definition_line.trim()).is_some() {
                    break;
                }
            }
        }
        if can_start_definition && instance_declaration_needs_continuation(definition_line.trim()) {
            while instance_declaration_needs_continuation(definition_line.trim()) {
                let Some(next_line) = pending_lines.front() else {
                    break;
                };
                if !is_instance_substitution_continuation(next_line, line_indent) {
                    break;
                }
                let next_line = pending_lines.pop_front().expect("front line should exist");
                let next_trimmed = next_line.trim();
                if next_trimmed.is_empty() {
                    continue;
                }
                definition_line.push(' ');
                definition_line.push_str(next_trimmed);
            }
        }

        let trimmed = definition_line.trim();
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

        if is_module_terminator(trimmed) {
            flush_definition(&mut module, &mut current_def);
            current_def_indent = 0;
            mode = NameListMode::None;
            continue;
        }

        if trimmed.starts_with("ASSUME")
            || trimmed.starts_with("AXIOM")
            || is_top_level_proof_header(trimmed)
        {
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
                let indent_width = definition_line.len() - definition_line.trim_start().len();
                let indent = &definition_line[..indent_width];
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
            NameListMode::Constants if is_name_list_continuation(trimmed, true) => {
                push_names(trimmed, &mut module.constants);
            }
            NameListMode::Variables if is_name_list_continuation(trimmed, false) => {
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

fn definition_head_needs_continuation(line: &str) -> bool {
    if split_definition_line(line).is_some() {
        return false;
    }

    let head = line
        .trim_start()
        .strip_prefix("LOCAL ")
        .unwrap_or(line.trim_start());
    let Some(first) = head.chars().next() else {
        return false;
    };
    if !(first.is_alphabetic() || first == '_') {
        return false;
    }

    let mut paren = 0usize;
    let mut bracket = 0usize;
    let chars: Vec<char> = head.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '<' if i + 1 < chars.len() && chars[i + 1] == '<' => i += 1,
            '>' if i + 1 < chars.len() && chars[i + 1] == '>' => i += 1,
            _ => {}
        }
        i += 1;
    }

    paren > 0 || bracket > 0
}

fn instance_declaration_needs_continuation(line: &str) -> bool {
    let Some(with_clause) = extract_instance_with_clause(line) else {
        return false;
    };
    instance_with_clause_is_incomplete(with_clause)
}

fn is_instance_substitution_continuation(line: &str, base_indent: usize) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    let indent = line.chars().take_while(|c| c.is_whitespace()).count();
    indent > base_indent
}

fn extract_instance_with_clause(line: &str) -> Option<&str> {
    let line = line.trim();
    let line = line.strip_prefix("LOCAL ").unwrap_or(line).trim();
    let rhs = line
        .split_once("==")
        .map(|(_, rhs)| rhs.trim())
        .unwrap_or(line);
    let after_instance = rhs.strip_prefix("INSTANCE")?.trim();
    if let Some(idx) = after_instance.find(" WITH ") {
        return Some(after_instance[idx + " WITH ".len()..].trim());
    }
    after_instance.strip_suffix(" WITH").map(str::trim)
}

fn instance_with_clause_is_incomplete(with_clause: &str) -> bool {
    let with_clause = with_clause.trim();
    if with_clause.is_empty() {
        return true;
    }

    let mut depth: usize = 0;
    let mut start = 0usize;
    let chars: Vec<char> = with_clause.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    while i < n {
        match chars[i] {
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth = depth.saturating_sub(1),
            '<' => {
                if i + 1 < n && chars[i + 1] == '<' {
                    depth += 1;
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && chars[i + 1] == '>' {
                    depth = depth.saturating_sub(1);
                    i += 1;
                }
            }
            ',' if depth == 0 => {
                let segment = with_clause[start..i].trim();
                if instance_substitution_segment_incomplete(segment) {
                    return true;
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    if depth > 0 {
        return true;
    }

    let tail = with_clause[start..].trim();
    if tail.is_empty() {
        return true;
    }

    instance_substitution_segment_incomplete(tail)
}

fn instance_substitution_segment_incomplete(segment: &str) -> bool {
    let Some((param, value)) = segment.split_once("<-") else {
        return true;
    };
    param.trim().is_empty() || value.trim().is_empty()
}

fn can_start_indented_definition_after_gap(current_def: &TlaDefinition, line: &str) -> bool {
    if !current_def.body.ends_with("\n\n") {
        return false;
    }
    if definition_body_has_open_let_scope(&current_def.body) {
        return false;
    }

    let last_non_empty_line = current_def
        .body
        .trim_end_matches('\n')
        .rsplit('\n')
        .find(|segment| !segment.trim().is_empty())
        .unwrap_or("")
        .trim();
    if definition_body_requires_continuation(last_non_empty_line) {
        return false;
    }

    split_definition_line(line)
        .map(|(lhs, _)| is_plausible_inline_definition_head(lhs))
        .unwrap_or(false)
}

fn definition_body_requires_continuation(line: &str) -> bool {
    let line = line.trim_end();
    line.ends_with("LET")
        || line.ends_with("IN")
        || line.ends_with("THEN")
        || line.ends_with("ELSE")
        || line.ends_with(':')
        || line.ends_with("/\\")
        || line.ends_with("\\/")
}

fn definition_body_has_open_let_scope(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut let_depth = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();

        if c == '<' && next == Some('<') {
            angle += 1;
            i += 2;
            continue;
        }
        if c == '>' && next == Some('>') {
            angle = angle.saturating_sub(1);
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

        if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
            if matches_module_keyword_at(&chars, i, "LET") {
                let_depth += 1;
                i += 3;
                continue;
            }
            if let_depth > 0 && matches_module_keyword_at(&chars, i, "IN") {
                let_depth = let_depth.saturating_sub(1);
                i += 2;
                continue;
            }
        }

        i += 1;
    }

    let_depth > 0
}

fn matches_module_keyword_at(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw_chars: Vec<char> = keyword.chars().collect();
    if i + kw_chars.len() > chars.len() {
        return false;
    }
    for (j, expected) in kw_chars.iter().enumerate() {
        if chars[i + j] != *expected {
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

fn flush_definition(module: &mut TlaModule, current: &mut Option<TlaDefinition>) {
    if let Some(mut def) = current.take() {
        // Mark definition as recursive if it was declared with RECURSIVE
        def.is_recursive = module.recursive_declarations.contains(&def.name);
        module.definitions.insert(def.name.clone(), def);
    }
}

fn is_top_level_proof_header(line: &str) -> bool {
    let line = line.trim_start();
    let line = line.strip_prefix("LOCAL ").unwrap_or(line);
    line.starts_with("THEOREM")
        || line.starts_with("LEMMA")
        || line.starts_with("COROLLARY")
        || line.starts_with("PROPOSITION")
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

    if let Some((name, params)) = parse_infix_def_head(lhs) {
        return (name, params);
    }

    let name = lhs
        .split_whitespace()
        .next()
        .map(ToString::to_string)
        .unwrap_or_default();
    (name, Vec::new())
}

fn parse_infix_def_head(lhs: &str) -> Option<(String, Vec<String>)> {
    if lhs.contains('(') || lhs.contains('[') || lhs.contains(',') {
        return None;
    }

    let parts: Vec<&str> = lhs.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }

    let left = normalize_operator_param(parts[0]);
    let op = parts[1].trim();
    let right = normalize_operator_param(parts[2]);
    if left.is_empty() || right.is_empty() || !is_symbolic_operator_name(op) {
        return None;
    }

    Some((op.to_string(), vec![left, right]))
}

fn is_symbolic_operator_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_'))
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
    for token in split_top_level_commas(text)
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some((name, _)) = split_declared_name(token) {
            out.push(name);
        }
    }
}

fn is_name_list_continuation(line: &str, allow_operator_params: bool) -> bool {
    let mut saw_entry = false;
    for entry in split_top_level_commas(line)
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        saw_entry = true;
        if !is_name_list_entry(entry, allow_operator_params) {
            return false;
        }
    }
    saw_entry
}

fn is_name_list_entry(entry: &str, allow_operator_params: bool) -> bool {
    let Some((_, suffix)) = split_declared_name(entry) else {
        return false;
    };
    if suffix.is_empty() {
        return true;
    }

    if suffix.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return true;
    }

    if !allow_operator_params {
        return false;
    }

    let Some(rest) = suffix.strip_prefix('(') else {
        return false;
    };
    let Some(args) = rest.strip_suffix(')') else {
        return false;
    };
    if args.trim().is_empty() {
        return true;
    }
    args.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .all(|arg| arg == "_")
}

fn split_declared_name(entry: &str) -> Option<(String, &str)> {
    let mut chars = entry.char_indices();
    let Some((_, first)) = chars.next() else {
        return None;
    };
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return None;
    }

    let mut end = first.len_utf8();
    let mut saw_identifier_marker = first.is_ascii_alphabetic() || first == '_';
    for (idx, c) in entry.char_indices().skip(1) {
        if c.is_alphanumeric() || c == '_' {
            end = idx + c.len_utf8();
            saw_identifier_marker |= c.is_ascii_alphabetic() || c == '_';
        } else {
            end = idx;
            return saw_identifier_marker.then(|| (entry[..end].to_string(), entry[end..].trim()));
        }
    }
    saw_identifier_marker.then(|| (entry[..end].to_string(), entry[end..].trim()))
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
    use std::sync::{Mutex, OnceLock};

    fn tla_library_path_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

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
    fn parses_multiline_bracketed_operator_definition_heads() {
        let src = r#"
        ---- MODULE Demo ----
        sc[<<x, y>> \in (0 .. N + 1) \X
                        (0 .. N + 1)] == CASE \/ x = 0
                                               \/ y = 0
                                            [] OTHER -> 1
        ====
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");
        assert!(m.definitions.contains_key("sc"));
        assert_eq!(m.definitions["sc"].params, vec!["<<x, y>>"]);
        assert!(m.definitions["sc"].body.contains("CASE"));
        assert!(m.definitions["sc"].body.contains("OTHER -> 1"));
    }

    #[test]
    fn parses_parenthesized_operator_parameters_with_internal_spaces() {
        let src = r#"
        ---- MODULE Demo ----
        Add(t, k, v) == <<t, k, v>>
        Update(tx, key, value) == <<tx, key, value>>
        ====
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");
        assert!(m.definitions.contains_key("Add"));
        assert!(m.definitions.contains_key("Update"));
        assert_eq!(m.definitions["Add"].params, vec!["t", "k", "v"]);
        assert_eq!(m.definitions["Update"].params, vec!["tx", "key", "value"]);
    }

    #[test]
    fn parses_infix_symbolic_operator_definitions() {
        let src = r#"
        ---- MODULE Demo ----
        EXTENDS Naturals

        a \prec b == a < b
        left ^^ right == left + right
        ====
        "#;

        let m = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(m.definitions[r"\prec"].params, vec!["a", "b"]);
        assert_eq!(m.definitions["^^"].params, vec!["left", "right"]);
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
    fn extends_can_load_modules_from_tla_library_path() {
        use std::fs;

        let _guard = tla_library_path_lock().lock().expect("env lock");
        let original = std::env::var_os("TLA_LIBRARY_PATH");

        let tmp = std::env::temp_dir().join("tlapp-extends-library-path-test");
        let _ = fs::remove_dir_all(&tmp);
        let spec_dir = tmp.join("spec");
        let lib_dir = tmp.join("lib");
        fs::create_dir_all(&spec_dir).expect("spec dir should exist");
        fs::create_dir_all(&lib_dir).expect("lib dir should exist");

        fs::write(
            lib_dir.join("ExtraMath.tla"),
            r#"
---- MODULE ExtraMath ----
One == 1
====
"#,
        )
        .expect("library module should be written");

        let entry = spec_dir.join("MC.tla");
        fs::write(
            &entry,
            r#"
---- MODULE MC ----
EXTENDS ExtraMath

Init == x = One
====
"#,
        )
        .expect("entry module should be written");

        unsafe {
            std::env::set_var("TLA_LIBRARY_PATH", &lib_dir);
        }
        let parsed = parse_tla_module_file(&entry).expect("parse should load library module");
        assert!(parsed.definitions.contains_key("One"));
        assert_eq!(parsed.definitions.get("One").unwrap().body.trim(), "1");

        if let Some(value) = original {
            unsafe {
                std::env::set_var("TLA_LIBRARY_PATH", value);
            }
        } else {
            unsafe {
                std::env::remove_var("TLA_LIBRARY_PATH");
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_can_load_unique_modules_from_repo_tree() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-ancestor-search-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join(".git")).expect("git dir should exist");
        fs::create_dir_all(tmp.join("specifications/spec_a")).expect("spec dir should exist");
        fs::create_dir_all(tmp.join("specifications/shared")).expect("shared dir should exist");

        let entry = tmp.join("specifications/spec_a/Entry.tla");
        fs::write(
            &entry,
            r#"
---- MODULE Entry ----
EXTENDS Shared

Init == SharedDef
====
"#,
        )
        .expect("entry module should be written");
        fs::write(
            tmp.join("specifications/shared/Shared.tla"),
            r#"
---- MODULE Shared ----
SharedDef == TRUE
====
"#,
        )
        .expect("shared module should be written");

        let parsed = parse_tla_module_file(&entry).expect("parse should load unique shared module");
        assert!(parsed.definitions.contains_key("SharedDef"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_does_not_guess_between_multiple_repo_tree_matches() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-ancestor-ambiguous-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join(".git")).expect("git dir should exist");
        fs::create_dir_all(tmp.join("specifications/spec_a")).expect("spec dir should exist");
        fs::create_dir_all(tmp.join("specifications/one"))
            .expect("first candidate dir should exist");
        fs::create_dir_all(tmp.join("specifications/two"))
            .expect("second candidate dir should exist");

        let entry = tmp.join("specifications/spec_a/Entry.tla");
        fs::write(
            &entry,
            r#"
---- MODULE Entry ----
EXTENDS Shared

Init == TRUE
====
"#,
        )
        .expect("entry module should be written");
        fs::write(
            tmp.join("specifications/one/Shared.tla"),
            r#"
---- MODULE Shared ----
SharedDef == 1
====
"#,
        )
        .expect("first shared module should be written");
        fs::write(
            tmp.join("specifications/two/Shared.tla"),
            r#"
---- MODULE Shared ----
SharedDef == 2
====
"#,
        )
        .expect("second shared module should be written");

        let parsed = parse_tla_module_file(&entry).expect("entry module should still parse");
        assert!(!parsed.definitions.contains_key("SharedDef"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn instances_can_load_modules_from_tla_library_path() {
        use std::fs;

        let _guard = tla_library_path_lock().lock().expect("env lock");
        let original = std::env::var_os("TLA_LIBRARY_PATH");

        let tmp = std::env::temp_dir().join("tlapp-instance-library-path-test");
        let _ = fs::remove_dir_all(&tmp);
        let spec_dir = tmp.join("spec");
        let lib_dir = tmp.join("lib");
        fs::create_dir_all(&spec_dir).expect("spec dir should exist");
        fs::create_dir_all(&lib_dir).expect("lib dir should exist");

        fs::write(
            lib_dir.join("Helper.tla"),
            r#"
---- MODULE Helper ----
Value == 42
====
"#,
        )
        .expect("library module should be written");

        let entry = spec_dir.join("MC.tla");
        fs::write(
            &entry,
            r#"
---- MODULE MC ----
Alias == INSTANCE Helper
====
"#,
        )
        .expect("entry module should be written");

        unsafe {
            std::env::set_var("TLA_LIBRARY_PATH", &lib_dir);
        }
        let parsed = parse_tla_module_file(&entry).expect("parse should load instance module");
        let alias = parsed
            .instances
            .get("Alias")
            .expect("instance should be present");
        assert_eq!(alias.module_name, "Helper");
        assert!(alias.module.is_some(), "instance module should be loaded");

        if let Some(value) = original {
            unsafe {
                std::env::set_var("TLA_LIBRARY_PATH", value);
            }
        } else {
            unsafe {
                std::env::remove_var("TLA_LIBRARY_PATH");
            }
        }
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
    fn shared_extends_dependencies_are_not_treated_as_circular() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-extends-diamond-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        fs::write(
            tmp.join("Shared.tla"),
            r#"
---- MODULE Shared ----
SharedOp == 1
====
"#,
        )
        .expect("shared module should be written");

        fs::write(
            tmp.join("Left.tla"),
            r#"
---- MODULE Left ----
EXTENDS Shared
LeftOp == SharedOp + 1
====
"#,
        )
        .expect("left module should be written");

        fs::write(
            tmp.join("Right.tla"),
            r#"
---- MODULE Right ----
EXTENDS Shared
RightOp == SharedOp + 2
====
"#,
        )
        .expect("right module should be written");

        let root = tmp.join("Root.tla");
        fs::write(
            &root,
            r#"
---- MODULE Root ----
EXTENDS Left, Right
RootOp == LeftOp + RightOp
====
"#,
        )
        .expect("root module should be written");

        let module = parse_tla_module_file(&root).expect("diamond extends should parse");
        assert!(module.definitions.contains_key("SharedOp"));
        assert!(module.definitions.contains_key("LeftOp"));
        assert!(module.definitions.contains_key("RightOp"));
        assert!(module.definitions.contains_key("RootOp"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_can_load_embedded_module_from_same_file() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-embedded-extends-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let root = tmp.join("Embedded.tla");
        fs::write(
            &root,
            r#"
---- MODULE Root ----
EXTENDS Helper
Init == x = Value
===============================================================================

---- MODULE Helper ----
Value == 42
===============================================================================
"#,
        )
        .expect("embedded module file should be written");

        let module = parse_tla_module_file(&root).expect("embedded extends should parse");
        assert_eq!(module.definitions.get("Value").unwrap().body.trim(), "42");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parse_tla_module_file_keeps_entry_module_definitions_when_siblings_overlap() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-embedded-entry-module-precedence");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let root = tmp.join("BufferedRandomAccessFile.tla");
        fs::write(
            &root,
            r#"
---- MODULE BufferedRandomAccessFile ----
EXTENDS Common

Init ==
    /\ dirty = FALSE
    /\ length = 0

TypeOK ==
    /\ dirty \in BOOLEAN
    /\ length \in Offset
===============================================================================

---- MODULE RandomAccessFile ----
EXTENDS Common

Init ==
    /\ file_content = EmptyArray
    /\ file_pointer = 0

TypeOK ==
    /\ file_content \in ArrayOfAnyLength(SymbolOrArbitrary)
    /\ file_pointer \in Offset
===============================================================================

---- MODULE Common ----
CONSTANTS MaxOffset, Symbols, ArbitrarySymbol
Offset == 0..MaxOffset
SymbolOrArbitrary == Symbols \union {ArbitrarySymbol}
ArrayOfAnyLength(T) == [elems: Seq(T)]
EmptyArray == [elems |-> <<>>]
===============================================================================
"#,
        )
        .expect("embedded module file should be written");

        let module = parse_tla_module_file(&root).expect("entry module should parse");
        assert_eq!(
            module.definitions.get("Init").unwrap().body.trim(),
            "/\\ dirty = FALSE\n    /\\ length = 0"
        );
        assert_eq!(
            module.definitions.get("TypeOK").unwrap().body.trim(),
            "/\\ dirty \\in BOOLEAN\n    /\\ length \\in Offset"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn instances_can_load_embedded_module_from_same_file() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-embedded-instance-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let root = tmp.join("Embedded.tla");
        fs::write(
            &root,
            r#"
---- MODULE Root ----
Alias == INSTANCE Helper
===============================================================================

---- MODULE Helper ----
Value == 42
===============================================================================
"#,
        )
        .expect("embedded module file should be written");

        let module = parse_tla_module_file(&root).expect("embedded instance should parse");
        let alias = module
            .instances
            .get("Alias")
            .expect("instance should exist");
        assert_eq!(alias.module_name, "Helper");
        assert!(
            alias.module.is_some(),
            "embedded instance module should load"
        );
        assert_eq!(
            alias
                .module
                .as_ref()
                .unwrap()
                .definitions
                .get("Value")
                .unwrap()
                .body
                .trim(),
            "42"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn embedded_instances_can_chain_embedded_extends_from_same_file() {
        use std::fs;

        let tmp = std::env::temp_dir().join("tlapp-embedded-instance-extends-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let root = tmp.join("Embedded.tla");
        fs::write(
            &root,
            r#"
---- MODULE Root ----
Alias == INSTANCE Helper
===============================================================================

---- MODULE Helper ----
EXTENDS Common
Value == SharedValue
===============================================================================

---- MODULE Common ----
SharedValue == 42
===============================================================================
"#,
        )
        .expect("embedded module file should be written");

        let module = parse_tla_module_file(&root).expect("embedded modules should parse");
        let alias = module
            .instances
            .get("Alias")
            .expect("instance should exist");
        let helper = alias
            .module
            .as_ref()
            .expect("embedded instance module should load");
        assert_eq!(
            helper
                .definitions
                .get("Value")
                .expect("helper value should be present")
                .body
                .trim(),
            "SharedValue"
        );
        assert_eq!(
            helper
                .definitions
                .get("SharedValue")
                .expect("extended embedded definition should be merged")
                .body
                .trim(),
            "42"
        );

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
    fn parses_named_instance_with_multiline_substitutions() {
        let src = r#"
---- MODULE TestMultilineNamedInstance ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH
    Node <- {1, 2, 3},
    Mode <- "safe"

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");
        let helper = m
            .instances
            .get("Helper")
            .expect("Helper instance should exist");
        assert_eq!(helper.module_name, "CoverageHelper");
        assert_eq!(
            helper.substitutions.get("Node"),
            Some(&"{1, 2, 3}".to_string())
        );
        assert_eq!(
            helper.substitutions.get("Mode"),
            Some(&"\"safe\"".to_string())
        );
    }

    #[test]
    fn parses_named_instance_with_first_substitution_inline_and_more_on_following_lines() {
        let src = r#"
---- MODULE TestInlineThenMultilineNamedInstance ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH Node <- {1, 2, 3},
                               Mode <- "safe",
                               Limit <- 4

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");
        let helper = m
            .instances
            .get("Helper")
            .expect("Helper instance should exist");
        assert_eq!(helper.module_name, "CoverageHelper");
        assert_eq!(
            helper.substitutions.get("Node"),
            Some(&"{1, 2, 3}".to_string())
        );
        assert_eq!(
            helper.substitutions.get("Mode"),
            Some(&"\"safe\"".to_string())
        );
        assert_eq!(helper.substitutions.get("Limit"), Some(&"4".to_string()));
    }

    #[test]
    fn parses_named_instance_with_multiline_substitution_values_and_comments() {
        let src = r#"
---- MODULE TestNestedInstanceValue ----
EXTENDS Naturals

Helper == INSTANCE CoverageHelper WITH Node <-
    [n \in Nodes |-> 0],
    \* stripped comments should not terminate the WITH clause
    Mode <- [kind |-> "safe",
             enabled |-> TRUE]

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");
        let helper = m
            .instances
            .get("Helper")
            .expect("Helper instance should exist");
        assert_eq!(helper.module_name, "CoverageHelper");
        assert_eq!(
            helper.substitutions.get("Node"),
            Some(&"[n \\in Nodes |-> 0]".to_string())
        );
        assert_eq!(
            helper.substitutions.get("Mode"),
            Some(&"[kind |-> \"safe\", enabled |-> TRUE]".to_string())
        );
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
    fn parses_unnamed_instance_with_multiline_substitutions() {
        let src = r#"
---- MODULE TestMultilineUnnamedInstance ----
EXTENDS Naturals

INSTANCE Sailfish WITH
    Node <- Servers,
    F <- Faulty

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(m.unnamed_instances.len(), 1);
        let instance = &m.unnamed_instances[0];
        assert_eq!(instance.module_name, "Sailfish");
        assert_eq!(
            instance.substitutions.get("Node"),
            Some(&"Servers".to_string())
        );
        assert_eq!(instance.substitutions.get("F"), Some(&"Faulty".to_string()));
    }

    #[test]
    fn parses_unnamed_instance_with_inline_then_multiline_substitutions() {
        let src = r#"
---- MODULE TestInlineThenMultilineUnnamedInstance ----
EXTENDS Naturals

INSTANCE Sailfish WITH Node <- Servers,
                      F <- Faulty,
                      Mode <- [kind |-> "safe"]

VARIABLES x
Init == x = 0
====
"#;
        let m = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(m.unnamed_instances.len(), 1);
        let instance = &m.unnamed_instances[0];
        assert_eq!(instance.module_name, "Sailfish");
        assert_eq!(
            instance.substitutions.get("Node"),
            Some(&"Servers".to_string())
        );
        assert_eq!(instance.substitutions.get("F"), Some(&"Faulty".to_string()));
        assert_eq!(
            instance.substitutions.get("Mode"),
            Some(&"[kind |-> \"safe\"]".to_string())
        );
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

        // Check that BaseHelper from BaseModule is available
        // (WITH substitution not yet implemented, so body may still contain Param)
        assert!(
            module.definitions.contains_key("BaseHelper"),
            "BaseHelper should be inherited from BaseModule"
        );

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
        assert_eq!(
            module.definitions.get("W").map(|def| def.body.as_str()),
            Some("4")
        );
        assert_eq!(
            module.definitions.get("H").map(|def| def.body.as_str()),
            Some("5")
        );
        assert_eq!(
            module.definitions.get("Pos").map(|def| def.body.as_str()),
            Some("0 .. W + H")
        );
    }

    #[test]
    fn parses_numeric_prefixed_variable_and_constant_names() {
        let src = r#"
---- MODULE NumericNames ----
EXTENDS Integers

CONSTANTS 1bMessage, Value
VARIABLES maxBal, 2avSent, knowsSent

TypeOK == /\ 2avSent \in [Value -> Int]
          /\ knowsSent \subseteq 1bMessage
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert!(module.constants.contains(&"1bMessage".to_string()));
        assert!(module.constants.contains(&"Value".to_string()));
        assert!(module.variables.contains(&"2avSent".to_string()));
        assert!(module.variables.contains(&"knowsSent".to_string()));
        let type_ok_body = module
            .definitions
            .get("TypeOK")
            .map(|def| def.body.trim().to_string());
        assert!(type_ok_body.is_some());
        let body = type_ok_body.unwrap();
        assert!(body.contains("2avSent \\in [Value -> Int]"));
        assert!(body.contains("knowsSent \\subseteq 1bMessage"));
    }

    #[test]
    fn parses_indented_top_level_definitions_after_comment_gap() {
        let src = r#"
---- MODULE IndentedDefs ----
Base == 1

\* A comment block between top-level definitions.
   omem == vmem
   octl == ctl
   obuf == buf
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(
            module
                .definitions
                .get("Base")
                .map(|def| def.body.trim())
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            module.definitions.get("omem").map(|def| def.body.as_str()),
            Some("vmem")
        );
        assert_eq!(
            module.definitions.get("octl").map(|def| def.body.as_str()),
            Some("ctl")
        );
        assert_eq!(
            module.definitions.get("obuf").map(|def| def.body.as_str()),
            Some("buf")
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
    fn does_not_split_comment_separated_let_bindings_into_top_level_definitions() {
        let src = r#"
---- MODULE MarkToMarketLike ----
EXTENDS Integers, FiniteSets

VARIABLES futures, referencePrice, balances

MarkToMarket(futureId) ==
    \E f \in futures :
        /\ f.id = futureId
        /\ LET
               priceDiff == referencePrice[f.asset] - f.price
               \* For each participant, calculate their P&L
               settlementPayments == {
                   <<p, priceDiff>> : p \in Participants
               }
               \* Check that all participants can cover their losses
               canSettle == \A <<p, pnl>> \in settlementPayments :
                   pnl >= 0 \/ balances[p] >= -pnl
           IN
           /\ canSettle
           /\ balances' = [p \in Participants |->
                  LET pnl == priceDiff
                  IN balances[p] + pnl]
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert!(module.definitions.contains_key("MarkToMarket"));
        assert!(!module.definitions.contains_key("priceDiff"));
        assert!(!module.definitions.contains_key("settlementPayments"));
        assert!(!module.definitions.contains_key("canSettle"));
        let body = &module.definitions["MarkToMarket"].body;
        assert!(body.contains("priceDiff == referencePrice[f.asset] - f.price"));
        assert!(body.contains("settlementPayments == {"));
        assert!(body.contains("canSettle == \\A <<p, pnl>> \\in settlementPayments"));
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

    #[test]
    fn top_level_theorem_does_not_extend_previous_definition() {
        let src = r#"
---- MODULE DistributedReplicatedLogLike ----
AllExtending ==
  \A s \in Servers: []<><<IsStrictPrefix(cLogs[s], cLogs'[s])>>_cLogs

THEOREM Spec => AllExtending
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        let all_extending = module
            .definitions
            .get("AllExtending")
            .expect("AllExtending should be defined");
        assert_eq!(
            all_extending.body.trim(),
            r"\A s \in Servers: []<><<IsStrictPrefix(cLogs[s], cLogs'[s])>>_cLogs"
        );
    }

    #[test]
    fn parses_multiline_constants_with_operator_entries() {
        let src = r#"
---- MODULE NanoLike ----
CONSTANTS
    Hash,
    CalculateHash(_,_,_),
    PrivateKey,
    PublicKey
VARIABLES
    lastHash,
    received
====
"#;

        let module = parse_tla_module_text(src).expect("parse should work");
        assert_eq!(
            module.constants,
            vec![
                "CalculateHash".to_string(),
                "Hash".to_string(),
                "PrivateKey".to_string(),
                "PublicKey".to_string()
            ]
        );
        assert_eq!(
            module.variables,
            vec!["lastHash".to_string(), "received".to_string()]
        );
    }
}
