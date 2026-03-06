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

    for raw_line in input.lines() {
        let line = strip_comment(raw_line).trim();
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

fn strip_comment(line: &str) -> &str {
    match line.find("\\*") {
        Some(i) => &line[..i],
        None => line,
    }
}

fn parse_constant_line(line: &str, cfg: &mut TlaConfig) -> Result<()> {
    if let Some((name, rhs)) = line.split_once("<-") {
        cfg.constants.insert(
            name.trim().to_string(),
            ConfigValue::OperatorRef(rhs.trim().to_string()),
        );
        return Ok(());
    }
    if let Some((name, rhs)) = line.split_once('=') {
        let value = parse_value(rhs.trim())?;
        cfg.constants.insert(name.trim().to_string(), value);
        return Ok(());
    }

    Err(anyhow!("invalid constant assignment: {}", line))
}

fn parse_value(input: &str) -> Result<ConfigValue> {
    let mut p = ValueParser::new(input);
    let value = p.parse_value()?;
    p.skip_ws();
    if !p.is_eof() {
        return Err(anyhow!("unexpected trailing value content: {}", p.rest()));
    }
    Ok(value)
}

struct ValueParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> ValueParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
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
}
