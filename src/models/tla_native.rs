use crate::fairness::{FairnessConstraint, LabeledTransition};
use crate::model::Model;
use crate::symmetry::{SymmetrySpec, canonicalize_tla_state};
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
    pub initial_states_vec: Vec<TlaState>,
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

        let initial_states_vec = evaluate_init_states(&module, &config, &init_name)?;
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
            initial_states_vec,
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
        self.initial_states_vec.clone()
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
            let canonical = canonicalize_tla_state(state, symmetry);
            // Debug: track canonicalization statistics (only in debug builds)
            #[cfg(debug_assertions)]
            {
                static FP_TOTAL: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                static FP_CHANGED: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);

                let total = FP_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if state != &canonical {
                    FP_CHANGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                // Print first state and periodic summary
                if total == 0 {
                    eprintln!(
                        "DEBUG fingerprint: symmetry.symmetric_values = {:?}",
                        symmetry.symmetric_values
                    );
                    if state != &canonical {
                        eprintln!("DEBUG fingerprint: state WAS canonicalized (states differ)");
                    } else {
                        eprintln!("DEBUG fingerprint: state unchanged by canonicalization");
                    }
                }
                if total > 0 && total % 500_000 == 0 {
                    let changed = FP_CHANGED.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!(
                        "DEBUG symmetry: {} total fingerprints, {} ({:.1}%) had canonicalization changes",
                        total,
                        changed,
                        (changed as f64 / total as f64) * 100.0
                    );
                }
            }
            canonical
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
                if std::env::var("TLAPP_TRACE_INVARIANT").is_ok() {
                    eprintln!(
                        "=== Invariant '{}' body ===\n{}\n=== End ===",
                        inv, def.body
                    );
                }
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
        let mut symmetry_groups: Vec<HashSet<String>> = Vec::new();

        // Try to extract symmetric values from constants first (single group)
        if let Some(ConfigValue::Set(values)) = cfg.constants.get(sym_name) {
            let mut group = HashSet::new();
            for val in values {
                if let ConfigValue::ModelValue(model_val) = val {
                    group.insert(model_val.clone());
                }
            }
            if !group.is_empty() {
                symmetry_groups.push(group);
            }
        }

        // If not found in constants, look up in module definitions
        // This handles cases like: Symmetry == Permutations(Bots) \union Permutations(Sellers)
        // IMPORTANT: Each Permutations(SetName) creates a SEPARATE symmetry group!
        if symmetry_groups.is_empty() {
            if let Some(def) = module.definitions.get(sym_name) {
                // Extract Permutations(SetName) calls from the definition
                let set_names = extract_permutation_sets(&def.body, &module.definitions);
                for set_name in &set_names {
                    // Each set becomes its own symmetry group
                    if let Some(ConfigValue::Set(values)) = cfg.constants.get(set_name) {
                        let mut group = HashSet::new();
                        for val in values {
                            if let ConfigValue::ModelValue(model_val) = val {
                                group.insert(model_val.clone());
                            }
                        }
                        if !group.is_empty() {
                            symmetry_groups.push(group);
                        }
                    }
                }
                if !symmetry_groups.is_empty() {
                    let total_values: usize = symmetry_groups.iter().map(|g| g.len()).sum();
                    eprintln!("Symmetry '{}' initialized from module definition: {} groups, {} total values",
                        sym_name, symmetry_groups.len(), total_values);
                    for (i, group) in symmetry_groups.iter().enumerate() {
                        eprintln!("  Group {}: {} values ({:?})", i, group.len(),
                            group.iter().take(5).cloned().collect::<Vec<_>>());
                    }
                }
            }
        }

        if !symmetry_groups.is_empty() {
            spec.initialize_with_groups(symmetry_groups);
        } else {
            eprintln!("Warning: Symmetry '{}' specified but no symmetric values found", sym_name);
        }

        spec
    })
}

/// Extract set names from Permutations(...) calls in an expression
/// Handles expressions like: Permutations(Bots) \union Permutations(Sellers)
/// Also follows definition references like: BotSymmetry \union SellerSymmetry
fn extract_permutation_sets(
    expr: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
) -> Vec<String> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    extract_permutation_sets_recursive(expr, definitions, &mut result, &mut visited);
    result
}

fn extract_permutation_sets_recursive(
    expr: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    result: &mut Vec<String>,
    visited: &mut HashSet<String>,
) {
    // Find Permutations(SetName) calls using regex-like matching
    let mut pos = 0;
    while let Some(perm_start) = expr[pos..].find("Permutations(") {
        let abs_start = pos + perm_start + "Permutations(".len();
        // Find the matching closing paren
        let rest = &expr[abs_start..];
        if let Some(paren_end) = rest.find(')') {
            let set_name = rest[..paren_end].trim().to_string();
            if !set_name.is_empty() && !result.contains(&set_name) {
                result.push(set_name);
            }
        }
        pos = abs_start;
    }

    // Also look for references to other definitions that might contain Permutations
    // Handle expressions like: BotSymmetry \union SellerSymmetry
    for (def_name, def) in definitions {
        // Check if this definition name appears in the expression
        if expr.contains(def_name) && !visited.contains(def_name) {
            visited.insert(def_name.clone());
            // Recursively extract from this definition
            extract_permutation_sets_recursive(&def.body, definitions, result, visited);
        }
    }
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

    let mut init = init.unwrap_or_else(|| "Init".to_string());
    let mut next = next.unwrap_or_else(|| "Next".to_string());

    // PlusCal fallback: If the requested Init/Next don't exist but this is a PlusCal spec,
    // try Init_/Next_ which are the names PlusCal generates.
    // Also try fallback if the config specifies Init/Next but they don't exist in the module.
    if !module.definitions.contains_key(&init) {
        let pluscal_init = format!("{}_", init);
        if module.definitions.contains_key(&pluscal_init) {
            if module.is_pluscal {
                // PlusCal detected, silently use the fallback
                init = pluscal_init;
            } else {
                // Not detected as PlusCal but Init_ exists - still use it with a note
                eprintln!(
                    "Note: '{}' not found but '{}' exists, using PlusCal-generated name",
                    init, pluscal_init
                );
                init = pluscal_init;
            }
        } else {
            return Err(anyhow!("Init definition '{init}' not found in module"));
        }
    }

    if !module.definitions.contains_key(&next) {
        let pluscal_next = format!("{}_", next);
        if module.definitions.contains_key(&pluscal_next) {
            if module.is_pluscal {
                // PlusCal detected, silently use the fallback
                next = pluscal_next;
            } else {
                // Not detected as PlusCal but Next_ exists - still use it with a note
                eprintln!(
                    "Note: '{}' not found but '{}' exists, using PlusCal-generated name",
                    next, pluscal_next
                );
                next = pluscal_next;
            }
        } else {
            return Err(anyhow!("Next definition '{next}' not found in module"));
        }
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

/// Evaluate Init predicate and return all possible initial states.
/// Handles:
/// - Equality assignments: var = expr (deterministic)
/// - Membership assignments: var \in Set (nondeterministic - enumerate all values)
/// - Guards: other predicates that must be TRUE
fn evaluate_init_states(
    module: &TlaModule,
    cfg: &TlaConfig,
    init_name: &str,
) -> Result<Vec<TlaState>> {
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
            return evaluate_init_states(module, cfg, &ref_name);
        }
    }

    // Start with constants from config
    let mut base_state = BTreeMap::new();
    for (k, v) in &cfg.constants {
        if let Some(tv) = config_value_to_tla(v) {
            base_state.insert(k.clone(), tv);
        }
    }

    // Classify all clauses
    let mut equality_assignments: Vec<(String, String)> = Vec::new();
    let mut membership_assignments: Vec<(String, String)> = Vec::new();
    let mut guards: Vec<String> = Vec::new();

    for clause in split_top_level(&init_def.body, "/\\") {
        match classify_clause(&clause) {
            ClauseKind::UnprimedEquality { var, expr } if module.variables.contains(&var) => {
                equality_assignments.push((var, expr));
            }
            ClauseKind::UnprimedMembership { var, set_expr } if module.variables.contains(&var) => {
                membership_assignments.push((var, set_expr));
            }
            _ => guards.push(clause),
        }
    }

    // First, resolve equality assignments (deterministic)
    let mut pending = equality_assignments;
    for _ in 0..pending.len().saturating_add(1) {
        if pending.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();
        for (var, expr) in pending {
            let ctx = EvalContext::with_definitions_and_instances(
                &base_state,
                &module.definitions,
                &module.instances,
            );
            match eval_expr(&expr, &ctx) {
                Ok(value) => {
                    base_state.insert(var, value);
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

    // Now handle membership assignments (nondeterministic)
    // Evaluate each set expression and collect possible values
    let mut membership_choices: Vec<(String, Vec<TlaValue>)> = Vec::new();

    for (var, set_expr) in membership_assignments {
        let ctx = EvalContext::with_definitions_and_instances(
            &base_state,
            &module.definitions,
            &module.instances,
        );
        let set_val = eval_expr(&set_expr, &ctx)
            .with_context(|| format!("failed evaluating membership set for {var}: {set_expr}"))?;
        let set = set_val
            .as_set()
            .with_context(|| format!("membership expression for {var} is not a set: {set_expr}"))?;

        if set.is_empty() {
            return Err(anyhow!(
                "membership set for {var} is empty, no initial states possible"
            ));
        }

        let values: Vec<TlaValue> = set.iter().cloned().collect();
        membership_choices.push((var, values));
    }

    // Generate all combinations of membership choices (cross product)
    let had_membership_choices = !membership_choices.is_empty();
    let base_state_for_error = base_state.clone();
    let mut states = vec![base_state];

    for (var, values) in membership_choices {
        let mut new_states = Vec::new();
        for state in states {
            for value in &values {
                let mut new_state = state.clone();
                new_state.insert(var.clone(), value.clone());
                new_states.push(new_state);
            }
        }
        states = new_states;

        // Limit total number of initial states
        const MAX_INIT_STATES: usize = 1_000_000;
        if states.len() > MAX_INIT_STATES {
            return Err(anyhow!(
                "too many initial states ({} > {}). Consider constraining Init.",
                states.len(),
                MAX_INIT_STATES
            ));
        }
    }

    // Filter states by guards
    let mut valid_states = Vec::new();
    for state in states {
        let ctx = EvalContext::with_definitions_and_instances(
            &state,
            &module.definitions,
            &module.instances,
        );

        let mut all_guards_pass = true;
        for guard in &guards {
            if guard.trim().is_empty() {
                continue;
            }
            match eval_expr(guard, &ctx) {
                Ok(val) => {
                    if !val.as_bool().unwrap_or(false) {
                        all_guards_pass = false;
                        break;
                    }
                }
                Err(_) => {
                    all_guards_pass = false;
                    break;
                }
            }
        }

        if all_guards_pass {
            // Verify all variables are assigned
            let all_assigned = module.variables.iter().all(|v| state.contains_key(v));
            if all_assigned {
                valid_states.push(state);
            }
        }
    }

    if valid_states.is_empty() {
        // Provide helpful error message
        if had_membership_choices {
            return Err(anyhow!(
                "no valid initial states found (membership assignments didn't produce valid states)"
            ));
        }
        // Check which variables are missing
        let missing: Vec<_> = module
            .variables
            .iter()
            .filter(|v| !base_state_for_error.contains_key(*v))
            .collect();
        if !missing.is_empty() {
            return Err(anyhow!(
                "Init does not assign variables: {}",
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        return Err(anyhow!("no valid initial states found"));
    }

    Ok(valid_states)
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
                is_recursive: false,
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

    #[test]
    fn function_set_membership_invariant_does_not_trigger_empty_expression() {
        let tmp = std::env::temp_dir().join("tlapp-native-function-set-invariant");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("FunctionSetInvariant.tla");
        fs::write(
            &module,
            r#"
---- MODULE FunctionSetInvariant ----
EXTENDS Naturals
CONSTANTS S
VARIABLES f
Init == /\ f = [x \in S |-> 0]
Next == /\ UNCHANGED <<f>>
TypeOK == f \in [S -> 0..1]
Spec == Init /\ [][Next]_<<f>>
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("FunctionSetInvariant.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
CONSTANTS
    S = {a, b}
INVARIANT TypeOK
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        let invariant_result = model.check_invariants(&init[0]);
        assert!(
            invariant_result.is_ok(),
            "unexpected invariant result: {invariant_result:?}"
        );
    }

    #[test]
    fn pluscal_naming_fallback_uses_init_underscore_and_next_underscore() {
        // Test that PlusCal specs with Init_ and Next_ work when config specifies Init/Next
        let tmp = std::env::temp_dir().join("tlapp-pluscal-naming-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create a spec that mimics PlusCal-generated code:
        // - Contains --algorithm marker in a comment
        // - Has Init_ and Next_ instead of Init and Next
        let module = tmp.join("PlusCal.tla");
        fs::write(
            &module,
            r#"
---- MODULE PlusCal ----
EXTENDS Naturals

(* --algorithm Counter
variables x = 0;
begin
  while x < 3 do
    x := x + 1;
  end while;
end algorithm; *)

\* BEGIN TRANSLATION (this is what PlusCal translator generates)
VARIABLES x, pc

vars == << x, pc >>

Init_ == /\ x = 0
         /\ pc = "start"

Next_ == \/ /\ pc = "start"
            /\ x < 3
            /\ x' = x + 1
            /\ pc' = pc
         \/ /\ pc = "start"
            /\ x >= 3
            /\ pc' = "Done"
            /\ x' = x
         \/ /\ pc = "Done"
            /\ UNCHANGED <<x, pc>>

Spec == Init_ /\ [][Next_]_vars

TypeOK == x >= 0 /\ x <= 3
\* END TRANSLATION
====
"#,
        )
        .expect("module should be written");

        // Config file specifies INIT and NEXT without underscore (standard TLC style)
        let cfg = tmp.join("PlusCal.cfg");
        fs::write(
            &cfg,
            r#"
INIT Init
NEXT Next
INVARIANT TypeOK
"#,
        )
        .expect("cfg should be written");

        // This should work because the code detects PlusCal and falls back to Init_/Next_
        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        // Verify the correct names were resolved
        assert_eq!(model.init_name, "Init_");
        assert_eq!(model.next_name, "Next_");

        // Verify model works correctly
        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));
        assert_eq!(
            init[0].get("pc"),
            Some(&TlaValue::String("start".to_string()))
        );

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty());
        assert!(model.check_invariants(&init[0]).is_ok());
    }

    #[test]
    fn pluscal_fallback_works_without_algorithm_marker() {
        // Test that fallback also works when Init_ exists but --algorithm is not present
        // (e.g., hand-written spec that happens to use Init_/Next_ naming)
        let tmp = std::env::temp_dir().join("tlapp-pluscal-fallback-no-marker");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("ManualUnderscore.tla");
        fs::write(
            &module,
            r#"
---- MODULE ManualUnderscore ----
EXTENDS Naturals
VARIABLES x

Init_ == x = 0
Next_ == x' = x + 1 \/ UNCHANGED x
TypeOK == x >= 0
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("ManualUnderscore.cfg");
        fs::write(
            &cfg,
            r#"
INIT Init
NEXT Next
INVARIANT TypeOK
"#,
        )
        .expect("cfg should be written");

        // Should still work - fallback to Init_/Next_ even without PlusCal marker
        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        assert_eq!(model.init_name, "Init_");
        assert_eq!(model.next_name, "Next_");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert!(model.check_invariants(&init[0]).is_ok());
    }
}
