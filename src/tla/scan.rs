use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct ModuleScan {
    pub module_name: String,
    pub path: PathBuf,
    pub extends: Vec<String>,
    pub instances: Vec<String>,
    pub operators: Vec<String>,
    pub features: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default)]
pub struct ScanAggregate {
    pub modules: Vec<ModuleScan>,
    pub combined_features: BTreeMap<String, u64>,
    pub operator_names: BTreeSet<String>,
}

const FEATURE_PATTERNS: &[(&str, &str)] = &[
    ("quant_forall", "\\A"),
    ("quant_exists", "\\E"),
    ("temporal_forall", "\\AA"),
    ("temporal_exists", "\\EE"),
    ("choose", "CHOOSE"),
    ("let_in", "LET"),
    ("if_then_else", "IF"),
    ("case", "CASE"),
    ("other", "OTHER"),
    ("except", "EXCEPT"),
    ("unchanged", "UNCHANGED"),
    ("enabled", "ENABLED"),
    ("prime", "'"),
    ("set_membership", "\\in"),
    ("subseteq", "\\subseteq"),
    ("union", "\\union"),
    ("intersect", "\\intersect"),
    ("subset", "SUBSET"),
    ("cartesian", "\\X"),
    ("record_map", "|->"),
    ("record_dot", "."),
    ("seq_concat", "\\o"),
    ("wf", "WF_"),
    ("sf", "SF_"),
    ("always", "[]"),
    ("eventually", "<>"),
    ("leads_to", "~>"),
    ("instance", "INSTANCE"),
    ("local", "LOCAL"),
    ("assume", "ASSUME"),
    ("theorem", "THEOREM"),
    ("lambda", "LAMBDA"),
];

const BUILTIN_EXTENDS: &[&str] = &[
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

pub fn scan_module_closure(entry_module: &Path) -> Result<ScanAggregate> {
    let entry_module = entry_module
        .canonicalize()
        .with_context(|| format!("cannot resolve entry module {}", entry_module.display()))?;

    let mut queue = VecDeque::new();
    let mut visited = BTreeSet::new();
    let mut scans = Vec::new();

    queue.push_back(entry_module);

    while let Some(path) = queue.pop_front() {
        if !visited.insert(path.clone()) {
            continue;
        }
        let scan = scan_single_module(&path)?;

        let module_dir = path.parent().unwrap_or_else(|| Path::new("."));

        for name in scan.extends.iter().chain(scan.instances.iter()) {
            if BUILTIN_EXTENDS.iter().any(|m| m == name) {
                continue;
            }
            let local_path = module_dir.join(format!("{name}.tla"));
            if local_path.exists() {
                queue.push_back(local_path.canonicalize().with_context(|| {
                    format!("cannot resolve local module {}", local_path.display())
                })?);
            }
        }

        scans.push(scan);
    }

    scans.sort_by(|a, b| a.path.cmp(&b.path));

    let mut combined = BTreeMap::new();
    let mut operators = BTreeSet::new();
    for scan in &scans {
        for (k, v) in &scan.features {
            *combined.entry(k.clone()).or_insert(0) += *v;
        }
        for op in &scan.operators {
            operators.insert(op.clone());
        }
    }

    Ok(ScanAggregate {
        modules: scans,
        combined_features: combined,
        operator_names: operators,
    })
}

fn scan_single_module(path: &Path) -> Result<ModuleScan> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed reading module {}", path.display()))?;
    let cleaned = strip_comments(&raw);

    let mut module_name = String::new();
    let mut extends = Vec::new();
    let mut instances = Vec::new();
    let mut operators = Vec::new();

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if module_name.is_empty()
            && trimmed.starts_with("----")
            && trimmed.contains("MODULE")
            && trimmed.ends_with("----")
            && let Some(name) = extract_module_name(trimmed)
        {
            module_name = name;
        }

        if let Some(rest) = trimmed.strip_prefix("EXTENDS") {
            for m in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                extends.push(m.to_string());
            }
        }

        if let Some(name) = extract_instance_name(trimmed) {
            instances.push(name);
        }

        if let Some(name) = extract_operator_name(trimmed) {
            operators.push(name);
        }
    }

    let mut features = BTreeMap::new();
    for (name, pattern) in FEATURE_PATTERNS {
        let count = cleaned.match_indices(pattern).count() as u64;
        if count > 0 {
            features.insert((*name).to_string(), count);
        }
    }
    let set_minus_count = count_set_minus(&cleaned);
    if set_minus_count > 0 {
        features.insert("set_minus".to_string(), set_minus_count);
    }

    Ok(ModuleScan {
        module_name,
        path: path.to_path_buf(),
        extends,
        instances,
        operators,
        features,
    })
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

fn extract_instance_name(line: &str) -> Option<String> {
    let idx = line.find("INSTANCE")?;
    let after = line[idx + "INSTANCE".len()..].trim();
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

fn extract_operator_name(line: &str) -> Option<String> {
    let (lhs, _rhs) = line.split_once("==")?;
    let lhs = lhs.trim();
    if lhs.is_empty() {
        return None;
    }

    if lhs.starts_with("SPECIFICATION")
        || lhs.starts_with("INIT")
        || lhs.starts_with("NEXT")
        || lhs.starts_with("INVARIANT")
        || lhs.starts_with("PROPERTY")
        || lhs.starts_with("CONSTRAINT")
    {
        return None;
    }

    let mut name = String::new();
    for c in lhs.chars() {
        if c.is_alphanumeric() || c == '_' {
            name.push(c);
        } else {
            break;
        }
    }

    if name.is_empty() { None } else { Some(name) }
}

fn count_set_minus(input: &str) -> u64 {
    let bytes = input.as_bytes();
    let mut count = 0u64;
    for i in 0..bytes.len() {
        if bytes[i] != b'\\' {
            continue;
        }
        let prev = if i > 0 {
            Some(bytes[i - 1] as char)
        } else {
            None
        };
        let next = if i + 1 < bytes.len() {
            Some(bytes[i + 1] as char)
        } else {
            None
        };
        let prev_ws = prev.map(|c| c.is_whitespace()).unwrap_or(false);
        let next_ws = next.map(|c| c.is_whitespace()).unwrap_or(false);
        if prev_ws && next_ws {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_comments_handles_line_and_block_comments() {
        let src = r#"
        A == 1 \* inline
        (* block
           comment *)
        B == 2
        "#;
        let stripped = strip_comments(src);
        assert!(stripped.contains("A == 1"));
        assert!(stripped.contains("B == 2"));
        assert!(!stripped.contains("inline"));
        assert!(!stripped.contains("comment"));
    }

    #[test]
    fn operator_extraction_works() {
        assert_eq!(
            extract_operator_name("Init == x = 0"),
            Some("Init".to_string())
        );
        assert_eq!(
            extract_operator_name("Next(a, b) == a' = b"),
            Some("Next".to_string())
        );
    }

    #[test]
    fn set_minus_count_is_not_confused_with_escaped_operators() {
        let src = r#"
        A == {1, 2} \ {2}
        B == x \in S
        C == y \union T
        "#;
        assert_eq!(count_set_minus(src), 1);
    }
}
