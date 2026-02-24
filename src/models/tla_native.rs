use crate::model::Model;
use crate::tla::{
    ClauseKind, ConfigValue, EvalContext, TlaConfig, TlaModule, TlaState, TlaValue,
    classify_clause, eval_expr, evaluate_next_states, parse_tla_config, parse_tla_module_file,
    split_top_level,
};
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct TlaModel {
    pub module: TlaModule,
    pub config: TlaConfig,
    pub init_name: String,
    pub next_name: String,
    pub invariant_exprs: Vec<(String, String)>,
    pub initial_state: TlaState,
}

impl TlaModel {
    pub fn from_files(
        module_path: &Path,
        cfg_path: Option<&Path>,
        init_override: Option<&str>,
        next_override: Option<&str>,
    ) -> Result<Self> {
        let module = parse_tla_module_file(module_path)?;
        let config = if let Some(path) = cfg_path {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed reading cfg {}", path.display()))?;
            parse_tla_config(&raw)?
        } else {
            TlaConfig::default()
        };

        let (init_name, next_name) = resolve_init_next_names(
            &module,
            &config,
            init_override.map(ToString::to_string),
            next_override.map(ToString::to_string),
        )?;

        let initial_state = evaluate_init_state(&module, &config, &init_name)?;
        let invariant_exprs = resolve_invariant_exprs(&module, &config);

        Ok(Self {
            module,
            config,
            init_name,
            next_name,
            invariant_exprs,
            initial_state,
        })
    }
}

impl Model for TlaModel {
    type State = TlaState;

    fn name(&self) -> &'static str {
        "tla-native"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        vec![self.initial_state.clone()]
    }

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        let next_def = self
            .module
            .definitions
            .get(&self.next_name)
            .unwrap_or_else(|| panic!("missing Next definition '{}'", self.next_name));

        let states = evaluate_next_states(&next_def.body, &self.module.definitions, state)
            .unwrap_or_else(|err| panic!("native next-state evaluation failed: {err}"));
        out.extend(states);
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        if self.invariant_exprs.is_empty() {
            return Ok(());
        }

        let ctx = EvalContext::with_definitions(state, &self.module.definitions);
        for (name, expr) in &self.invariant_exprs {
            match eval_expr(expr, &ctx) {
                Ok(TlaValue::Bool(true)) => {}
                Ok(TlaValue::Bool(false)) => {
                    return Err(format!("invariant '{name}' violated"));
                }
                Ok(other) => {
                    return Err(format!(
                        "invariant '{name}' did not evaluate to BOOLEAN: {other:?}"
                    ));
                }
                Err(err) => {
                    return Err(format!("failed evaluating invariant '{name}': {err}"));
                }
            }
        }

        Ok(())
    }
}

fn resolve_invariant_exprs(module: &TlaModule, cfg: &TlaConfig) -> Vec<(String, String)> {
    cfg.invariants
        .iter()
        .map(|inv| {
            if let Some(def) = module.definitions.get(inv)
                && def.params.is_empty()
            {
                return (inv.clone(), def.body.clone());
            }
            (inv.clone(), inv.clone())
        })
        .collect()
}

fn resolve_init_next_names(
    module: &TlaModule,
    cfg: &TlaConfig,
    init_override: Option<String>,
    next_override: Option<String>,
) -> Result<(String, String)> {
    let mut init = init_override.or_else(|| cfg.init.clone());
    let mut next = next_override.or_else(|| cfg.next.clone());

    if (init.is_none() || next.is_none())
        && let Some(spec_name) = cfg.specification.as_ref()
        && let Some(spec_def) = module.definitions.get(spec_name)
        && let Some((spec_init, spec_next)) = extract_init_next_from_spec(&spec_def.body)
    {
        if init.is_none() {
            init = Some(spec_init);
        }
        if next.is_none() {
            next = Some(spec_next);
        }
    }

    let init = init.unwrap_or_else(|| "Init".to_string());
    let next = next.unwrap_or_else(|| "Next".to_string());

    if !module.definitions.contains_key(&init) {
        return Err(anyhow!("Init definition '{init}' not found in module"));
    }
    if !module.definitions.contains_key(&next) {
        return Err(anyhow!("Next definition '{next}' not found in module"));
    }

    Ok((init, next))
}

fn extract_init_next_from_spec(spec_body: &str) -> Option<(String, String)> {
    let parts = split_top_level(spec_body, "/\\");
    let mut init = None;
    let mut next = None;

    for part in parts {
        let part = part.trim();

        if init.is_none()
            && let Some(name) = parse_simple_identifier(part)
        {
            init = Some(name);
            continue;
        }

        if next.is_none()
            && let Some(rest) = part.strip_prefix("[][")
            && let Some(idx) = rest.find("]_")
        {
            let candidate = rest[..idx].trim();
            if let Some(name) = parse_simple_identifier(candidate) {
                next = Some(name);
            }
        }
    }

    match (init, next) {
        (Some(i), Some(n)) => Some((i, n)),
        _ => None,
    }
}

fn parse_simple_identifier(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut chars = trimmed.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return None,
    }

    if chars.all(|c| c.is_alphanumeric() || c == '_') {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn evaluate_init_state(module: &TlaModule, cfg: &TlaConfig, init_name: &str) -> Result<TlaState> {
    let init_def = module
        .definitions
        .get(init_name)
        .ok_or_else(|| anyhow!("missing Init definition '{init_name}'"))?;

    let mut state = BTreeMap::new();
    for (k, v) in &cfg.constants {
        if let Some(tv) = config_value_to_tla(v) {
            state.insert(k.clone(), tv);
        }
    }

    let mut assignments: Vec<(String, String)> = Vec::new();
    let mut guards: Vec<String> = Vec::new();

    for clause in split_top_level(&init_def.body, "/\\") {
        match classify_clause(&clause) {
            ClauseKind::UnprimedEquality { var, expr } if module.variables.contains(&var) => {
                assignments.push((var, expr));
            }
            _ => guards.push(clause),
        }
    }

    let mut pending = assignments;
    for _ in 0..pending.len().saturating_add(1) {
        if pending.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();
        for (var, expr) in pending {
            let ctx = EvalContext::with_definitions(&state, &module.definitions);
            match eval_expr(&expr, &ctx) {
                Ok(value) => {
                    state.insert(var, value);
                    progress = true;
                }
                Err(_) => next_pending.push((var, expr)),
            }
        }

        if !progress {
            let names = next_pending
                .iter()
                .map(|(var, _)| var.as_str())
                .collect::<Vec<_>>()
                .join(",");
            return Err(anyhow!("failed to resolve Init assignments: {names}"));
        }

        pending = next_pending;
    }

    let ctx = EvalContext::with_definitions(&state, &module.definitions);
    for guard in guards {
        if guard.trim().is_empty() {
            continue;
        }
        let ok = eval_expr(&guard, &ctx)
            .with_context(|| format!("failed evaluating Init guard: {guard}"))?
            .as_bool()?;
        if !ok {
            return Err(anyhow!("Init guard evaluated to FALSE: {guard}"));
        }
    }

    for var in &module.variables {
        if !state.contains_key(var) {
            return Err(anyhow!("Init does not assign variable '{var}'"));
        }
    }

    Ok(state)
}

fn config_value_to_tla(value: &ConfigValue) -> Option<TlaValue> {
    match value {
        ConfigValue::Int(v) => Some(TlaValue::Int(*v)),
        ConfigValue::Bool(v) => Some(TlaValue::Bool(*v)),
        ConfigValue::String(v) => Some(TlaValue::String(v.clone())),
        ConfigValue::ModelValue(v) => Some(TlaValue::ModelValue(v.clone())),
        ConfigValue::OperatorRef(_) => None,
        ConfigValue::Tuple(values) => Some(TlaValue::Seq(
            values.iter().filter_map(config_value_to_tla).collect(),
        )),
        ConfigValue::Set(values) => Some(TlaValue::Set(
            values.iter().filter_map(config_value_to_tla).collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    #[ignore] // TODO: Support inline conjunctive expressions in Next branches
    fn builds_and_steps_simple_tla_model() {
        let tmp = std::env::temp_dir().join("tlapp-native-model-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("Simple.tla");
        fs::write(
            &module,
            r#"
---- MODULE Simple ----
EXTENDS Naturals
VARIABLES x
Init == /\ x = 0
Next == \/ x < 2 /\ x' = x + 1
        \/ x = 2 /\ UNCHANGED <<x>>
Inv == x >= 0
Spec == Init /\ [][Next]_x
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("Simple.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
INVARIANT Inv
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty());
        assert!(model.check_invariants(&init[0]).is_ok());
    }
}
