use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ConfigValue {
    Int(i64),
    Bool(bool),
    String(String),
    ModelValue(String),
    Set(Vec<ConfigValue>),
    Tuple(Vec<ConfigValue>),
    OperatorRef(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlaConfig {
    pub specification: Option<String>,
    pub init: Option<String>,
    pub next: Option<String>,
    pub symmetry: Option<String>,
    pub view: Option<String>,
    pub check_deadlock: Option<bool>,
    pub constants: BTreeMap<String, ConfigValue>,
    pub invariants: Vec<String>,
    pub properties: Vec<String>,
    pub constraints: Vec<String>,
    pub action_constraints: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Constants,
    Invariants,
    Properties,
    Constraints,
    ActionConstraints,
    // Single-value sections (value on next line)
    Specification,
    Init,
    Next,
    Symmetry,
    View,
    CheckDeadlock,
}

pub fn parse_tla_config(input: &str) -> Result<TlaConfig> {
    let mut cfg = TlaConfig::default();
    let mut section: Option<Section> = None;

    // First strip block comments (* ... *) which can span multiple lines
    let input_no_block_comments = strip_block_comments(input);

    for raw_line in input_no_block_comments.lines() {
        // Then strip line comments \* ...
        let line = strip_line_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        // Handle SPECIFICATION, INIT, NEXT, SYMMETRY, VIEW
        // These can be either "KEYWORD value" on one line, or "KEYWORD" then value on next line
        if line == "SPECIFICATION" {
            section = Some(Section::Specification);
            continue;
        }
        if let Some(value) = line.strip_prefix("SPECIFICATION ") {
            cfg.specification = Some(value.trim().to_string());
            section = None;
            continue;
        }
        if line == "INIT" {
            section = Some(Section::Init);
            continue;
        }
        if let Some(value) = line.strip_prefix("INIT ") {
            cfg.init = Some(value.trim().to_string());
            section = None;
            continue;
        }
        if line == "NEXT" {
            section = Some(Section::Next);
            continue;
        }
        if let Some(value) = line.strip_prefix("NEXT ") {
            cfg.next = Some(value.trim().to_string());
            section = None;
            continue;
        }
        if line == "SYMMETRY" {
            section = Some(Section::Symmetry);
            continue;
        }
        if let Some(value) = line.strip_prefix("SYMMETRY ") {
            cfg.symmetry = Some(value.trim().to_string());
            section = None;
            continue;
        }
        if line == "VIEW" {
            section = Some(Section::View);
            continue;
        }
        if let Some(value) = line.strip_prefix("VIEW ") {
            cfg.view = Some(value.trim().to_string());
            section = None;
            continue;
        }
        // CHECK_DEADLOCK can be on same line or next line
        if line == "CHECK_DEADLOCK" {
            section = Some(Section::CheckDeadlock);
            continue;
        }
        if let Some(value) = line.strip_prefix("CHECK_DEADLOCK ") {
            let v = value.trim();
            cfg.check_deadlock = match v {
                "TRUE" => Some(true),
                "FALSE" => Some(false),
                _ => return Err(anyhow!("invalid CHECK_DEADLOCK value: {v}")),
            };
            section = None;
            continue;
        }

        if line == "CONSTANT" || line == "CONSTANTS" {
            section = Some(Section::Constants);
            continue;
        }
        // Handle inline constant assignment: "CONSTANT Name = Value" or "CONSTANT Name <- Op"
        if let Some(rest) = line
            .strip_prefix("CONSTANT ")
            .or(line.strip_prefix("CONSTANTS "))
        {
            // Check if this is an assignment (contains = or <-)
            if rest.contains('=') || rest.contains("<-") {
                parse_constant_line(rest, &mut cfg)?;
                continue;
            }
            // Otherwise it's just "CONSTANT" with items listed after, fall through to section handling
            section = Some(Section::Constants);
            continue;
        }
        if line == "INVARIANT" || line == "INVARIANTS" {
            section = Some(Section::Invariants);
            continue;
        }
        if line == "PROPERTY" || line == "PROPERTIES" {
            section = Some(Section::Properties);
            continue;
        }
        if line == "CONSTRAINT" || line == "CONSTRAINTS" {
            section = Some(Section::Constraints);
            continue;
        }
        if line == "ACTION_CONSTRAINT" || line == "ACTION_CONSTRAINTS" {
            section = Some(Section::ActionConstraints);
            continue;
        }

        // Handle "INVARIANT foo" or "INVARIANTS foo bar baz" (multiple items on same line)
        if let Some(items) = line
            .strip_prefix("INVARIANT ")
            .or(line.strip_prefix("INVARIANTS "))
        {
            for item in items.split_whitespace() {
                cfg.invariants.push(item.to_string());
            }
            section = Some(Section::Invariants);
            continue;
        }
        if let Some(items) = line
            .strip_prefix("PROPERTY ")
            .or(line.strip_prefix("PROPERTIES "))
        {
            for item in items.split_whitespace() {
                cfg.properties.push(item.to_string());
            }
            section = Some(Section::Properties);
            continue;
        }
        if let Some(items) = line
            .strip_prefix("CONSTRAINT ")
            .or(line.strip_prefix("CONSTRAINTS "))
        {
            for item in items.split_whitespace() {
                cfg.constraints.push(item.to_string());
            }
            section = Some(Section::Constraints);
            continue;
        }
        if let Some(items) = line
            .strip_prefix("ACTION_CONSTRAINT ")
            .or(line.strip_prefix("ACTION_CONSTRAINTS "))
        {
            for item in items.split_whitespace() {
                cfg.action_constraints.push(item.to_string());
            }
            section = Some(Section::ActionConstraints);
            continue;
        }

        match section {
            Some(Section::Constants) => parse_constant_line(line, &mut cfg)?,
            Some(Section::Invariants) => cfg.invariants.push(line.to_string()),
            Some(Section::Properties) => cfg.properties.push(line.to_string()),
            Some(Section::Constraints) => cfg.constraints.push(line.to_string()),
            Some(Section::ActionConstraints) => cfg.action_constraints.push(line.to_string()),
            Some(Section::Specification) => {
                cfg.specification = Some(line.to_string());
                section = None;
            }
            Some(Section::Init) => {
                cfg.init = Some(line.to_string());
                section = None;
            }
            Some(Section::Next) => {
                cfg.next = Some(line.to_string());
                section = None;
            }
            Some(Section::Symmetry) => {
                cfg.symmetry = Some(line.to_string());
                section = None;
            }
            Some(Section::View) => {
                cfg.view = Some(line.to_string());
                section = None;
            }
            Some(Section::CheckDeadlock) => {
                cfg.check_deadlock = match line {
                    "TRUE" => Some(true),
                    "FALSE" => Some(false),
                    _ => return Err(anyhow!("invalid CHECK_DEADLOCK value: {}", line)),
                };
                section = None;
            }
            None => {
                return Err(anyhow!(
                    "unrecognized config line outside section: {}",
                    line
                ));
            }
        }
    }

    Ok(cfg)
}

fn strip_line_comment(line: &str) -> &str {
    match line.find("\\*") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Strip TLA+ block comments (* ... *) from input, supporting nested comments.
/// Returns the input with all block comments replaced by spaces (to preserve spacing).
fn strip_block_comments(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut depth = 0;

    while i < chars.len() {
        // Check for opening comment (*
        if i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
            depth += 1;
            // Replace with spaces to preserve line structure
            result.push(' ');
            result.push(' ');
            i += 2;
            continue;
        }

        // Check for closing comment *)
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
            if depth > 0 {
                depth -= 1;
                result.push(' ');
                result.push(' ');
                i += 2;
                continue;
            }
            // If depth is 0, this is an unmatched *) - just pass through
        }

        if depth > 0 {
            // Inside a comment - preserve newlines for line structure, replace others with space
            if chars[i] == '\n' {
                result.push('\n');
            } else {
                result.push(' ');
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
}

/// Parse one or more constant assignments from a single line.
/// Supports multiple space-separated assignments like: `a1=a1  a2=a2  a3=a3`
fn parse_constant_line(line: &str, cfg: &mut TlaConfig) -> Result<()> {
    let mut offset = 0;
    let line = line.trim();

    while offset < line.len() {
        // Skip leading whitespace
        let remaining = &line[offset..];
        let trimmed = remaining.trim_start();
        offset += remaining.len() - trimmed.len();

        if offset >= line.len() {
            break;
        }

        let remaining = &line[offset..];

        // Try operator reference first (Name <- Op)
        if let Some(arrow_pos) = remaining.find("<-") {
            // Check if there's a `=` before the `<-` (indicating a different assignment type comes first)
            let eq_pos = remaining.find('=');
            if eq_pos.is_none() || arrow_pos < eq_pos.unwrap() {
                let name = remaining[..arrow_pos].trim();
                let after_arrow = &remaining[arrow_pos + 2..];
                let after_arrow_trimmed = after_arrow.trim_start();
                let ws_len = after_arrow.len() - after_arrow_trimmed.len();
                // Read the operator name (identifier)
                let op_name = read_identifier(after_arrow_trimmed);
                if op_name.is_empty() {
                    return Err(anyhow!("expected operator name after '<-' in: {}", line));
                }
                cfg.constants
                    .insert(name.to_string(), ConfigValue::OperatorRef(op_name.clone()));
                // Move offset past: Name <- whitespace OpName
                offset += arrow_pos + 2 + ws_len + op_name.len();
                continue;
            }
        }

        // Try value assignment (Name = Value)
        if let Some(eq_pos) = remaining.find('=') {
            let name = remaining[..eq_pos].trim();
            if name.is_empty() {
                return Err(anyhow!("empty constant name in: {}", line));
            }
            let after_eq = &remaining[eq_pos + 1..];
            let after_eq_trimmed = after_eq.trim_start();
            let ws_len = after_eq.len() - after_eq_trimmed.len();
            let mut p = ValueParser::new(after_eq_trimmed);
            let value = p.parse_value()?;
            cfg.constants.insert(name.to_string(), value);
            // Move offset past: Name = whitespace Value
            let consumed = after_eq_trimmed.len() - p.rest().len();
            offset += eq_pos + 1 + ws_len + consumed;
            continue;
        }

        // No more assignments found
        let remaining = &line[offset..].trim();
        if !remaining.is_empty() {
            return Err(anyhow!("invalid constant assignment: {}", remaining));
        }
        break;
    }

    Ok(())
}

/// Read an identifier (sequence of alphanumeric chars and underscores, starting with letter or _)
fn read_identifier(s: &str) -> String {
    let mut chars = s.chars().peekable();
    let mut ident = String::new();

    // First char must be letter or underscore
    if let Some(&c) = chars.peek() {
        if c.is_alphabetic() || c == '_' {
            ident.push(c);
            chars.next();
        } else {
            return ident;
        }
    }

    // Rest can be alphanumeric or underscore
    while let Some(&c) = chars.peek() {
        if c.is_alphanumeric() || c == '_' {
            ident.push(c);
            chars.next();
        } else {
            break;
        }
    }

    ident
}

struct ValueParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> ValueParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn rest(&self) -> &str {
        &self.src[self.pos..]
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn eat_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn starts_with(&self, token: &str) -> bool {
        self.rest().starts_with(token)
    }

    fn eat_token(&mut self, token: &str) -> bool {
        if self.starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn parse_value(&mut self) -> Result<ConfigValue> {
        self.skip_ws();
        if self.eat_token("{") {
            return self.parse_set();
        }
        if self.eat_token("<<") {
            return self.parse_tuple();
        }
        if self.peek_char() == Some('"') {
            return self.parse_string();
        }
        self.parse_atom()
    }

    fn parse_set(&mut self) -> Result<ConfigValue> {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat_token("}") {
                break;
            }
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            if self.eat_token("}") {
                break;
            }
            if !self.eat_token(",") {
                return Err(anyhow!("expected ',' or '}}' in set: {}", self.rest()));
            }
        }
        Ok(ConfigValue::Set(items))
    }

    fn parse_tuple(&mut self) -> Result<ConfigValue> {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat_token(">>") {
                break;
            }
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            if self.eat_token(">>") {
                break;
            }
            if !self.eat_token(",") {
                return Err(anyhow!("expected ',' or '>>' in tuple: {}", self.rest()));
            }
        }
        Ok(ConfigValue::Tuple(items))
    }

    fn parse_string(&mut self) -> Result<ConfigValue> {
        let quote = self.eat_char();
        if quote != Some('"') {
            return Err(anyhow!("internal parser error: expected string quote"));
        }

        let mut out = String::new();
        loop {
            let c = self
                .eat_char()
                .ok_or_else(|| anyhow!("unterminated string literal"))?;
            if c == '"' {
                break;
            }
            if c == '\\' {
                let esc = self
                    .eat_char()
                    .ok_or_else(|| anyhow!("unterminated string escape"))?;
                out.push(esc);
            } else {
                out.push(c);
            }
        }

        Ok(ConfigValue::String(out))
    }

    fn parse_atom(&mut self) -> Result<ConfigValue> {
        let atom = self.read_atom();
        if atom.is_empty() {
            return Err(anyhow!("expected value atom at: {}", self.rest()));
        }

        if atom == "TRUE" {
            return Ok(ConfigValue::Bool(true));
        }
        if atom == "FALSE" {
            return Ok(ConfigValue::Bool(false));
        }
        if let Ok(v) = atom.parse::<i64>() {
            return Ok(ConfigValue::Int(v));
        }
        Ok(ConfigValue::ModelValue(atom))
    }

    fn read_atom(&mut self) -> String {
        self.skip_ws();
        let mut atom = String::new();
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() || c == ',' || c == '}' {
                break;
            }
            if c == '<' && self.rest().starts_with("<<") {
                break;
            }
            if c == '>' && self.rest().starts_with(">>") {
                break;
            }
            atom.push(c);
            self.pos += c.len_utf8();
        }
        atom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_combined_cfg_constants() {
        let cfg = parse_tla_config(
            r#"
            SPECIFICATION Spec
            CONSTANTS
                Bots = {bot1, bot2}
                MaxTime = 2
                CollateralRate = 2
            INVARIANT Inv1
            INVARIANT Inv2
            CHECK_DEADLOCK FALSE
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.check_deadlock, Some(false));
        assert_eq!(cfg.invariants.len(), 2);
        assert_eq!(cfg.constants.get("MaxTime"), Some(&ConfigValue::Int(2)));
    }

    #[test]
    fn parses_multiline_section_format() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
                MaxBeanCount = 100

            SPECIFICATION
                Spec

            PROPERTY
                EventuallyTerminates
                MonotonicDecrease

            INVARIANTS
                TypeInvariant
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(
            cfg.properties,
            vec!["EventuallyTerminates", "MonotonicDecrease"]
        );
        assert_eq!(cfg.invariants, vec!["TypeInvariant"]);
        assert_eq!(
            cfg.constants.get("MaxBeanCount"),
            Some(&ConfigValue::Int(100))
        );
    }

    #[test]
    fn parses_multiple_invariants_on_same_line() {
        let cfg = parse_tla_config(
            r#"
            SPECIFICATION Spec
            INVARIANTS TypeOK NotSolved Safety
            PROPERTIES Liveness Progress
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.invariants, vec!["TypeOK", "NotSolved", "Safety"]);
        assert_eq!(cfg.properties, vec!["Liveness", "Progress"]);
    }

    #[test]
    fn parses_operator_ref_and_tuple() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
                Seq <- BoundedSeq
                Pair = <<a, b>>
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(
            cfg.constants.get("Seq"),
            Some(&ConfigValue::OperatorRef("BoundedSeq".to_string()))
        );
        match cfg.constants.get("Pair") {
            Some(ConfigValue::Tuple(v)) => assert_eq!(v.len(), 2),
            other => panic!("unexpected tuple parse result: {other:?}"),
        }
    }

    #[test]
    fn parses_inline_constant_assignment() {
        let cfg = parse_tla_config(
            r#"
            SPECIFICATION Spec
            CONSTANT N = 3
            CONSTANT Flag = TRUE
            CONSTANT Server = s1
            CONSTANT Procs = {p1, p2, p3}
            CONSTANT SeqOp <- BoundedSeq
            INVARIANT TypeOK
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
        assert_eq!(cfg.constants.get("Flag"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            cfg.constants.get("Server"),
            Some(&ConfigValue::ModelValue("s1".to_string()))
        );
        match cfg.constants.get("Procs") {
            Some(ConfigValue::Set(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], ConfigValue::ModelValue("p1".to_string()));
                assert_eq!(items[1], ConfigValue::ModelValue("p2".to_string()));
                assert_eq!(items[2], ConfigValue::ModelValue("p3".to_string()));
            }
            other => panic!("unexpected Procs parse result: {other:?}"),
        }
        assert_eq!(
            cfg.constants.get("SeqOp"),
            Some(&ConfigValue::OperatorRef("BoundedSeq".to_string()))
        );
        assert_eq!(cfg.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn parses_mixed_constant_formats() {
        // Test mixing inline CONSTANT with block CONSTANTS
        let cfg = parse_tla_config(
            r#"
            CONSTANT A = 1
            CONSTANTS
                B = 2
                C = 3
            CONSTANT D = 4
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.constants.get("A"), Some(&ConfigValue::Int(1)));
        assert_eq!(cfg.constants.get("B"), Some(&ConfigValue::Int(2)));
        assert_eq!(cfg.constants.get("C"), Some(&ConfigValue::Int(3)));
        assert_eq!(cfg.constants.get("D"), Some(&ConfigValue::Int(4)));
    }

    #[test]
    fn parses_single_line_block_comment() {
        let cfg = parse_tla_config(
            r#"
            (* This is a comment *)
            SPECIFICATION Spec
            CONSTANT N = 3
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
    }

    #[test]
    fn parses_inline_block_comment() {
        let cfg = parse_tla_config(
            r#"
            SPECIFICATION Spec
            CONSTANT N = 3  (* inline comment *)
            INVARIANT TypeOK
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
        assert_eq!(cfg.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn parses_multi_line_block_comment() {
        let cfg = parse_tla_config(
            r#"
            (* Multi-line
               comment here *)
            SPECIFICATION Spec
            INVARIANT TypeOK
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn parses_nested_block_comments() {
        let cfg = parse_tla_config(
            r#"
            (* outer (* inner *) outer *)
            SPECIFICATION Spec
            CONSTANT N = 5
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(5)));
    }

    #[test]
    fn parses_mixed_comment_styles() {
        let cfg = parse_tla_config(
            r#"
            (* block comment *)
            SPECIFICATION Spec
            CONSTANT N = 3  \* line comment
            (* another block *)
            INVARIANT TypeOK
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
        assert_eq!(cfg.invariants, vec!["TypeOK"]);
    }

    #[test]
    fn strip_block_comments_basic() {
        // Check that comment content is removed and replaced with spaces
        let result = strip_block_comments("hello (* world *) test");
        assert!(result.contains("hello"));
        assert!(result.contains("test"));
        assert!(!result.contains("world"));
        assert!(!result.contains("(*"));
        assert!(!result.contains("*)"));

        // Empty after stripping
        let result = strip_block_comments("(* comment *)");
        assert!(result.trim().is_empty());

        // No comments - unchanged
        assert_eq!(strip_block_comments("no comments here"), "no comments here");
    }

    #[test]
    fn strip_block_comments_nested() {
        let result = strip_block_comments("a (* outer (* inner *) outer *) b");
        assert!(result.contains("a"));
        assert!(result.contains("b"));
        assert!(!result.contains("outer"));
        assert!(!result.contains("inner"));
        assert!(!result.contains("(*"));
        assert!(!result.contains("*)"));
    }

    #[test]
    fn strip_block_comments_multiline() {
        let input = "line1\n(* comment\nspanning\nlines *)\nline2";
        let result = strip_block_comments(input);
        // Should preserve line structure
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[4], "line2");
        // Middle lines should be empty/whitespace after stripping
        assert!(lines[1].trim().is_empty());
        assert!(lines[2].trim().is_empty());
        assert!(lines[3].trim().is_empty());
    }

    #[test]
    fn parses_multiple_constants_on_same_line() {
        // This is the Paxos-style format that was failing
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
              a1=a1  a2=a2  a3=a3  v1=v1  v2=v2
            SPECIFICATION Spec
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(
            cfg.constants.get("a1"),
            Some(&ConfigValue::ModelValue("a1".to_string()))
        );
        assert_eq!(
            cfg.constants.get("a2"),
            Some(&ConfigValue::ModelValue("a2".to_string()))
        );
        assert_eq!(
            cfg.constants.get("a3"),
            Some(&ConfigValue::ModelValue("a3".to_string()))
        );
        assert_eq!(
            cfg.constants.get("v1"),
            Some(&ConfigValue::ModelValue("v1".to_string()))
        );
        assert_eq!(
            cfg.constants.get("v2"),
            Some(&ConfigValue::ModelValue("v2".to_string()))
        );
        assert_eq!(cfg.specification.as_deref(), Some("Spec"));
    }

    #[test]
    fn parses_multiple_constants_with_mixed_types_on_same_line() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
              N=3  M=5  Flag=TRUE  Server=s1
            SPECIFICATION Spec
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
        assert_eq!(cfg.constants.get("M"), Some(&ConfigValue::Int(5)));
        assert_eq!(cfg.constants.get("Flag"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            cfg.constants.get("Server"),
            Some(&ConfigValue::ModelValue("s1".to_string()))
        );
    }

    #[test]
    fn parses_multiple_constants_with_sets_on_same_line() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
              Acceptors={a1, a2}  Values={v1, v2, v3}
            SPECIFICATION Spec
            "#,
        )
        .expect("cfg should parse");

        match cfg.constants.get("Acceptors") {
            Some(ConfigValue::Set(items)) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], ConfigValue::ModelValue("a1".to_string()));
                assert_eq!(items[1], ConfigValue::ModelValue("a2".to_string()));
            }
            other => panic!("unexpected Acceptors parse result: {other:?}"),
        }

        match cfg.constants.get("Values") {
            Some(ConfigValue::Set(items)) => {
                assert_eq!(items.len(), 3);
            }
            other => panic!("unexpected Values parse result: {other:?}"),
        }
    }

    #[test]
    fn parses_multiple_operator_refs_on_same_line() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
              Seq1<-BoundedSeq  Seq2<-AnotherSeq
            SPECIFICATION Spec
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(
            cfg.constants.get("Seq1"),
            Some(&ConfigValue::OperatorRef("BoundedSeq".to_string()))
        );
        assert_eq!(
            cfg.constants.get("Seq2"),
            Some(&ConfigValue::OperatorRef("AnotherSeq".to_string()))
        );
    }

    #[test]
    fn parses_mixed_assignments_and_operator_refs_on_same_line() {
        let cfg = parse_tla_config(
            r#"
            CONSTANTS
              N=3  Seq<-BoundedSeq  M=5
            SPECIFICATION Spec
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.constants.get("N"), Some(&ConfigValue::Int(3)));
        assert_eq!(
            cfg.constants.get("Seq"),
            Some(&ConfigValue::OperatorRef("BoundedSeq".to_string()))
        );
        assert_eq!(cfg.constants.get("M"), Some(&ConfigValue::Int(5)));
    }
}
