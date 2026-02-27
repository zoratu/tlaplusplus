use crate::fairness::{ActionLabel, FairnessConstraint, LabeledTransition};
use crate::model::Model;
use crate::symmetry::{SymmetrySpec, canonicalize_state};
use crate::tla::{
    ClauseKind, CompiledActionIr, CompiledExpr, ConfigValue, EvalContext, TemporalFormula,
    TlaConfig, TlaDefinition, TlaModule, TlaState, TlaValue, classify_clause, compile_action_ir,
    compile_expr, eval_action_constraint, eval_compiled, eval_expr, evaluate_next_states,
    evaluate_next_states_labeled, insert_compiled_action, looks_like_action, parse_tla_config,
    parse_tla_module_file, split_top_level,
};
use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TlaModel {
    pub module: TlaModule,
    pub config: TlaConfig,
    pub init_name: String,
    pub next_name: String,
    pub invariant_exprs: Vec<(String, String)>,
    pub temporal_properties: Vec<(String, TemporalFormula)>,
    pub fairness_constraints: Vec<FairnessConstraint>,
    pub state_constraints: Vec<(String, String)>,
    pub action_constraints: Vec<(String, String)>,
    pub symmetry: Option<SymmetrySpec>,
    pub view: Option<String>,
    pub initial_state: TlaState,
    /// Pre-compiled action definitions for fast execution
    pub compiled_actions: BTreeMap<String, Arc<CompiledActionIr>>,
    /// Pre-compiled invariant expressions (name, compiled expr)
    pub compiled_invariants: Vec<(String, Arc<CompiledExpr>)>,
    /// Pre-compiled state constraint expressions (name, compiled expr)
    pub compiled_state_constraints: Vec<(String, Arc<CompiledExpr>)>,
}

impl TlaModel {
    pub fn from_files(
        module_path: &Path,
        cfg_path: Option<&Path>,
        init_override: Option<&str>,
        next_override: Option<&str>,
    ) -> Result<Self> {
        let mut module = parse_tla_module_file(module_path)?;
        let config = if let Some(path) = cfg_path {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed reading cfg {}", path.display()))?;
            parse_tla_config(&raw)?
        } else {
            TlaConfig::default()
        };

        // Inject constants from config into module definitions
        // This makes constants available during action evaluation
        inject_constants_into_definitions(&mut module, &config);

        let (init_name, next_name) = resolve_init_next_names(
            &module,
            &config,
            init_override.map(ToString::to_string),
            next_override.map(ToString::to_string),
        )?;

        let initial_state = evaluate_init_state(&module, &config, &init_name)?;
        let invariant_exprs = resolve_invariant_exprs(&module, &config);
        let temporal_properties = resolve_temporal_properties(&module, &config)?;
        let mut fairness_constraints = extract_fairness_constraints(&temporal_properties);

        // Also extract fairness from SPECIFICATION if present
        if let Some(spec_name) = config.specification.as_ref() {
            if std::env::var("TLAPP_VERBOSE").is_ok() {
                eprintln!(
                    "Checking SPECIFICATION '{}' for fairness constraints",
                    spec_name
                );
            }
            if let Some(spec_def) = module.definitions.get(spec_name) {
                if std::env::var("TLAPP_VERBOSE").is_ok() {
                    eprintln!("  Spec body: {}", spec_def.body);
                }
                match TemporalFormula::parse(&spec_def.body) {
                    Ok(spec_formula) => {
                        extract_fairness_from_formula(&spec_formula, &mut fairness_constraints);
                    }
                    Err(e) => {
                        eprintln!("  Failed to parse Spec as temporal formula: {}", e);
                    }
                }
            } else {
                eprintln!("  SPECIFICATION '{}' not found in definitions", spec_name);
            }
        }

        if !fairness_constraints.is_empty() {
            eprintln!(
                "Extracted {} fairness constraints from specification",
                fairness_constraints.len()
            );
            for constraint in &fairness_constraints {
                match constraint {
                    FairnessConstraint::Weak { vars, action } => {
                        eprintln!("  WF_<<{}>>({}) - Weak fairness", vars.join(", "), action);
                    }
                    FairnessConstraint::Strong { vars, action } => {
                        eprintln!("  SF_<<{}>>({}) - Strong fairness", vars.join(", "), action);
                    }
                }
            }
        }

        let state_constraints = resolve_constraint_exprs(&module, &config);
        let action_constraints = resolve_action_constraint_exprs(&module, &config);
        let symmetry = resolve_symmetry(&module, &config);
        let view = resolve_view(&module, &config);

        // Pre-compile all action definitions
        let compiled_actions = precompile_actions(&module.definitions);

        // Pre-compile invariant expressions
        let compiled_invariants = precompile_expressions(&invariant_exprs);

        // Pre-compile state constraint expressions
        let compiled_state_constraints = precompile_expressions(&state_constraints);

        // Warm up the global action cache with our pre-compiled actions
        warm_up_action_cache(&compiled_actions, &module.definitions);

        Ok(Self {
            module,
            config,
            init_name,
            next_name,
            invariant_exprs,
            temporal_properties,
            fairness_constraints,
            state_constraints,
            action_constraints,
            symmetry,
            view,
            initial_state,
            compiled_actions,
            compiled_invariants,
            compiled_state_constraints,
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
        if self.compiled_invariants.is_empty() {
            return Ok(());
        }

        let ctx = EvalContext::with_definitions_and_instances(
            state,
            &self.module.definitions,
            &self.module.instances,
        );
        for (name, compiled_expr) in &self.compiled_invariants {
            match eval_compiled(compiled_expr, &ctx) {
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

    fn check_state_constraints(&self, state: &Self::State) -> Result<(), String> {
        if self.compiled_state_constraints.is_empty() {
            return Ok(());
        }

        let ctx = EvalContext::with_definitions_and_instances(
            state,
            &self.module.definitions,
            &self.module.instances,
        );
        for (name, compiled_expr) in &self.compiled_state_constraints {
            match eval_compiled(compiled_expr, &ctx) {
                Ok(TlaValue::Bool(true)) => {}
                Ok(TlaValue::Bool(false)) => {
                    return Err(format!("state constraint '{name}' violated (state pruned)"));
                }
                Ok(other) => {
                    return Err(format!(
                        "state constraint '{name}' did not evaluate to BOOLEAN: {other:?}"
                    ));
                }
                Err(err) => {
                    return Err(format!(
                        "failed evaluating state constraint '{name}': {err}"
                    ));
                }
            }
        }

        Ok(())
    }

    fn check_action_constraints(
        &self,
        current: &Self::State,
        next: &Self::State,
    ) -> Result<(), String> {
        if self.action_constraints.is_empty() {
            return Ok(());
        }

        for (name, expr) in &self.action_constraints {
            match eval_action_constraint(expr, current, next, Some(&self.module.definitions)) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(format!("action constraint '{name}' violated"));
                }
                Err(err) => {
                    return Err(format!(
                        "failed evaluating action constraint '{name}': {err}"
                    ));
                }
            }
        }

        Ok(())
    }

    fn fingerprint(&self, state: &Self::State) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        // Apply symmetry reduction if specified
        let canonical_state = if let Some(ref symmetry) = self.symmetry {
            canonicalize_state(state, symmetry)
        } else {
            state.clone()
        };

        // If view function is defined, fingerprint only the view
        if self.view.is_some() {
            match self.evaluate_view(&canonical_state) {
                Ok(view_value) => {
                    // Hash the view value
                    if let Ok(bytes) = bincode::serialize(&view_value) {
                        let mut hasher = DefaultHasher::new();
                        hasher.write(&bytes);
                        return hasher.finish();
                    }
                }
                Err(_) => {
                    // View evaluation failed - fall back to full state fingerprint
                }
            }
        }

        // No view or view failed - hash the full state using serialization
        let mut hasher = DefaultHasher::new();
        if let Ok(bytes) = bincode::serialize(&canonical_state) {
            hasher.write(&bytes);
        }
        hasher.finish()
    }

    fn next_states_labeled(
        &self,
        state: &Self::State,
    ) -> Option<Vec<crate::model::LabeledTransition<Self::State>>> {
        if !self.has_fairness_constraints() {
            return None;
        }

        let next_def = self.module.definitions.get(&self.next_name)?;

        match evaluate_next_states_labeled(
            &next_def.body,
            &self.next_name,
            &self.module.definitions,
            state,
        ) {
            Ok(transitions) => {
                // Convert from fairness::LabeledTransition to model::LabeledTransition
                let converted: Vec<crate::model::LabeledTransition<Self::State>> = transitions
                    .into_iter()
                    .map(|t| crate::model::LabeledTransition {
                        from: t.from,
                        to: t.to,
                        action: crate::model::ActionLabel {
                            name: t.action.name,
                            disjunct_index: t.action.disjunct_index,
                        },
                    })
                    .collect();
                Some(converted)
            }
            Err(_) => None,
        }
    }

    fn has_fairness_constraints(&self) -> bool {
        !self.fairness_constraints.is_empty()
    }
}

impl TlaModel {
    /// Evaluate the view function on a state to get the projected state
    ///
    /// View functions allow state space reduction by fingerprinting only a
    /// projection of the state. If no view is defined, returns the full state.
    pub fn evaluate_view(&self, state: &TlaState) -> Result<TlaValue, String> {
        if let Some(view_expr) = &self.view {
            let ctx = EvalContext::with_definitions_and_instances(
                state,
                &self.module.definitions,
                &self.module.instances,
            );
            eval_expr(view_expr, &ctx).map_err(|e| format!("view evaluation failed: {}", e))
        } else {
            // No view defined - return full state as a value
            // Convert state to TlaValue::Record
            let record: BTreeMap<String, TlaValue> = state.clone();
            Ok(TlaValue::Record(Arc::new(record)))
        }
    }

    /// Generate next states with action labels for fairness checking
    ///
    /// This is used when fairness constraints are present and we need to track
    /// which action generated each transition for fairness verification.
    pub fn next_states_labeled(
        &self,
        state: &TlaState,
    ) -> Result<Vec<LabeledTransition<TlaState>>, String> {
        let next_def = self
            .module
            .definitions
            .get(&self.next_name)
            .ok_or_else(|| format!("missing Next definition '{}'", self.next_name))?;

        evaluate_next_states_labeled(
            &next_def.body,
            &self.next_name,
            &self.module.definitions,
            state,
        )
        .map_err(|e| format!("labeled next-state evaluation failed: {}", e))
    }

    /// Check if this model has fairness constraints
    pub fn has_fairness_constraints(&self) -> bool {
        !self.fairness_constraints.is_empty()
    }

    /// Check if this model has liveness properties
    pub fn has_liveness_properties(&self) -> bool {
        self.temporal_properties
            .iter()
            .any(|(_, formula)| formula.is_liveness_property())
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

fn resolve_temporal_properties(
    module: &TlaModule,
    cfg: &TlaConfig,
) -> Result<Vec<(String, TemporalFormula)>> {
    let mut properties = Vec::new();

    for prop_name in &cfg.properties {
        // Look up definition
        let expr = if let Some(def) = module.definitions.get(prop_name) {
            if !def.params.is_empty() {
                return Err(anyhow!(
                    "temporal property '{}' must not have parameters",
                    prop_name
                ));
            }
            &def.body
        } else {
            // Treat as inline expression
            prop_name
        };

        // Parse temporal formula
        let formula = TemporalFormula::parse(expr)
            .with_context(|| format!("failed parsing temporal property '{}'", prop_name))?;

        properties.push((prop_name.clone(), formula));
    }

    Ok(properties)
}

fn resolve_constraint_exprs(module: &TlaModule, cfg: &TlaConfig) -> Vec<(String, String)> {
    cfg.constraints
        .iter()
        .map(|constraint| {
            if let Some(def) = module.definitions.get(constraint)
                && def.params.is_empty()
            {
                return (constraint.clone(), def.body.clone());
            }
            (constraint.clone(), constraint.clone())
        })
        .collect()
}

fn resolve_action_constraint_exprs(module: &TlaModule, cfg: &TlaConfig) -> Vec<(String, String)> {
    cfg.action_constraints
        .iter()
        .map(|constraint| {
            if let Some(def) = module.definitions.get(constraint)
                && def.params.is_empty()
            {
                return (constraint.clone(), def.body.clone());
            }
            (constraint.clone(), constraint.clone())
        })
        .collect()
}

fn resolve_symmetry(module: &TlaModule, cfg: &TlaConfig) -> Option<SymmetrySpec> {
    cfg.symmetry.as_ref().map(|sym_name| {
        let mut spec = SymmetrySpec::new(sym_name.clone());

        // Try to extract symmetric values from constants
        // If the constant is a set of model values, those are the symmetric values
        if let Some(ConfigValue::Set(values)) = cfg.constants.get(sym_name) {
            let mut symmetric_values = HashSet::new();
            for val in values {
                if let ConfigValue::ModelValue(model_val) = val {
                    symmetric_values.insert(model_val.clone());
                }
            }
            if !symmetric_values.is_empty() {
                spec.initialize_from_config(symmetric_values);
            }
        }

        spec
    })
}

fn resolve_view(module: &TlaModule, cfg: &TlaConfig) -> Option<String> {
    cfg.view.as_ref().map(|view_name| {
        // Look up the view definition
        if let Some(def) = module.definitions.get(view_name) {
            if !def.params.is_empty() {
                // View function with parameters - not supported yet
                // Return the name as-is for now
                return view_name.clone();
            }
            // Return the view expression body
            def.body.clone()
        } else {
            // Treat as inline expression
            view_name.clone()
        }
    })
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

    // If the init body is just a reference to another definition, resolve it recursively
    let trimmed_body = init_def.body.trim();
    if let Some(ref_name) = parse_simple_identifier(trimmed_body) {
        // Check if this is a reference to another definition (not just a bare identifier)
        if module.definitions.contains_key(&ref_name) && ref_name != init_name {
            // Recursively evaluate the referenced init
            return evaluate_init_state(module, cfg, &ref_name);
        }
    }

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
            let ctx = EvalContext::with_definitions_and_instances(
                &state,
                &module.definitions,
                &module.instances,
            );
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

    let ctx =
        EvalContext::with_definitions_and_instances(&state, &module.definitions, &module.instances);
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
        ConfigValue::Tuple(values) => Some(TlaValue::Seq(Arc::new(
            values.iter().filter_map(config_value_to_tla).collect(),
        ))),
        ConfigValue::Set(values) => Some(TlaValue::Set(Arc::new(
            values.iter().filter_map(config_value_to_tla).collect(),
        ))),
    }
}

/// Extract fairness constraints from temporal properties
///
/// Fairness constraints (WF and SF) are stored in temporal_properties but need
/// to be extracted separately for fairness checking on the state graph.
fn extract_fairness_constraints(
    temporal_properties: &[(String, TemporalFormula)],
) -> Vec<FairnessConstraint> {
    let mut constraints = Vec::new();

    for (_name, formula) in temporal_properties {
        extract_fairness_from_formula(formula, &mut constraints);
    }

    constraints
}

/// Recursively extract fairness constraints from a temporal formula
fn extract_fairness_from_formula(
    formula: &TemporalFormula,
    constraints: &mut Vec<FairnessConstraint>,
) {
    match formula {
        TemporalFormula::WeakFairness { vars, action } => {
            constraints.push(FairnessConstraint::Weak {
                vars: vars.clone(),
                action: action.clone(),
            });
        }
        TemporalFormula::StrongFairness { vars, action } => {
            constraints.push(FairnessConstraint::Strong {
                vars: vars.clone(),
                action: action.clone(),
            });
        }
        TemporalFormula::Always(inner)
        | TemporalFormula::Eventually(inner)
        | TemporalFormula::InfinitelyOften(inner)
        | TemporalFormula::EventuallyAlways(inner)
        | TemporalFormula::Not(inner) => {
            extract_fairness_from_formula(inner, constraints);
        }
        TemporalFormula::TemporalForAll { formula: inner, .. }
        | TemporalFormula::TemporalExists { formula: inner, .. } => {
            extract_fairness_from_formula(inner, constraints);
        }
        TemporalFormula::And(left, right)
        | TemporalFormula::Or(left, right)
        | TemporalFormula::LeadsTo(left, right) => {
            extract_fairness_from_formula(left, constraints);
            extract_fairness_from_formula(right, constraints);
        }
        TemporalFormula::StatePredicate(_) => {
            // No fairness constraints in state predicates
        }
    }
}

/// Inject constants from config into module definitions
///
/// This makes constants available during action evaluation by creating
/// zero-parameter operator definitions for each constant.
fn inject_constants_into_definitions(module: &mut TlaModule, config: &TlaConfig) {
    for (name, value) in &config.constants {
        // Convert ConfigValue to a TLA+ expression string
        let body = config_value_to_expr(value);

        // Add as a zero-parameter definition
        module.definitions.insert(
            name.clone(),
            TlaDefinition {
                name: name.clone(),
                params: vec![],
                body,
            },
        );
    }
}

/// Convert a ConfigValue to a TLA+ expression string
fn config_value_to_expr(value: &ConfigValue) -> String {
    match value {
        ConfigValue::Int(n) => n.to_string(),
        ConfigValue::String(s) => format!("\"{}\"", s),
        ConfigValue::ModelValue(s) => s.clone(),
        ConfigValue::Bool(b) => {
            if *b {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        ConfigValue::Set(values) => {
            let items: Vec<String> = values.iter().map(config_value_to_expr).collect();
            format!("{{{}}}", items.join(", "))
        }
        ConfigValue::Tuple(values) => {
            let items: Vec<String> = values.iter().map(config_value_to_expr).collect();
            format!("<<{}>>", items.join(", "))
        }
        ConfigValue::OperatorRef(name) => name.clone(),
    }
}

/// Pre-compile all action definitions in the module
///
/// This compiles actions at model load time instead of lazily during execution,
/// improving runtime performance by avoiding repeated compilation.
fn precompile_actions(
    definitions: &BTreeMap<String, TlaDefinition>,
) -> BTreeMap<String, Arc<CompiledActionIr>> {
    let mut compiled = BTreeMap::new();

    for (name, def) in definitions {
        // Only compile definitions that look like actions (contain primed variables or UNCHANGED)
        if looks_like_action(def) {
            let ir = compile_action_ir(def);
            let compiled_ir = Arc::new(CompiledActionIr::from_ir(&ir));
            compiled.insert(name.clone(), compiled_ir);
        }
    }

    compiled
}

/// Pre-compile a list of (name, expression) pairs into compiled expressions
fn precompile_expressions(exprs: &[(String, String)]) -> Vec<(String, Arc<CompiledExpr>)> {
    exprs
        .iter()
        .map(|(name, expr)| {
            let compiled = Arc::new(compile_expr(expr));
            (name.clone(), compiled)
        })
        .collect()
}

/// Warm up the global COMPILED_ACTION_CACHE with pre-compiled actions
///
/// This ensures that when actions are looked up during execution, they're
/// already in the cache and don't need to be compiled.
fn warm_up_action_cache(
    compiled_actions: &BTreeMap<String, Arc<CompiledActionIr>>,
    definitions: &BTreeMap<String, TlaDefinition>,
) {
    for (name, compiled_ir) in compiled_actions {
        if let Some(def) = definitions.get(name) {
            // Use the same cache key format as get_or_compile_action
            let cache_key = format!("{}:{}", def.name, def.body);
            insert_compiled_action(cache_key, Arc::clone(compiled_ir));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
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
