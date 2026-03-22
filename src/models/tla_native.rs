use crate::fairness::{FairnessConstraint, LabeledTransition};
use crate::model::Model;
use crate::symmetry::{SymmetrySpec, canonicalize_tla_state};
use crate::tla::module::TlaModuleInstance;
#[cfg(test)]
use crate::tla::tla_state;
use crate::tla::{
    ClauseKind, CompiledActionIr, CompiledExpr, ConfigValue, EvalContext, TemporalFormula,
    TlaConfig, TlaDefinition, TlaModule, TlaState, TlaValue, classify_clause, compile_action_ir,
    compile_expr, count_next_disjuncts, eval_action_constraint, eval_compiled, eval_expr,
    evaluate_next_states_labeled_with_instances, evaluate_next_states_swarm,
    evaluate_next_states_with_instances, insert_compiled_action, looks_like_action,
    normalize_operator_ref_name, parse_tla_config, parse_tla_module_file, split_top_level,
};
use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, BTreeSet, HashSet};
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
    /// Whether to allow deadlocked states (no successors) without error.
    /// Equivalent to TLC's -deadlock flag.
    pub allow_deadlock: bool,
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

        // TLC-style directory search: if config references definitions (invariants,
        // properties, init, next) not found in the module, search sibling .tla files
        // in the same directory. This matches TLC's behavior of implicitly searching
        // the working directory for unresolved definitions.
        resolve_sibling_definitions(module_path, &mut module, &config);

        let (init_name, next_name) = resolve_init_next_names(
            &mut module,
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

        // CHECK_DEADLOCK FALSE in cfg means allow deadlocked states
        let allow_deadlock = config.check_deadlock == Some(false);

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
            allow_deadlock,
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

        let instances = if self.module.instances.is_empty() {
            None
        } else {
            Some(&self.module.instances)
        };

        match evaluate_next_states_with_instances(
            &next_def.body,
            &self.module.definitions,
            instances,
            state,
        ) {
            Ok(states) => {
                out.extend(states);
                // Empty states = deadlocked state. If deadlock checking is enabled,
                // this will be detected by the runtime when out remains empty.
            }
            Err(err) => {
                if self.allow_deadlock {
                    // Treat evaluation errors as deadlocked states (no successors).
                    // This handles cases like: guards that fail type evaluation
                    // (e.g., record access on ModelValue) when all action branches
                    // are disabled for this state.
                    return;
                }
                panic!("native next-state evaluation failed: {err}");
            }
        }
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
                Ok(TlaValue::ModelValue(mv)) if mv == *name => {
                    // The invariant name resolved to a ModelValue of itself,
                    // meaning the definition was not found.  This typically
                    // happens when the invariant is defined in an EXTENDS'd
                    // module that could not be loaded.
                    return Err(format!(
                        "invariant '{name}' is not defined (not found in module \
                         definitions or EXTENDS chain)"
                    ));
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

    fn canonicalize(&self, state: Self::State) -> Self::State {
        if let Some(ref symmetry) = self.symmetry {
            canonicalize_tla_state(&state, symmetry)
        } else {
            state
        }
    }

    fn fingerprint(&self, state: &Self::State) -> u64 {
        use ahash::AHasher;
        use std::hash::Hasher;

        // Symmetry canonicalization is handled by canonicalize() in the
        // runtime before fingerprinting, so we fingerprint the state directly.

        // If view function is defined, fingerprint only the view
        if self.view.is_some() {
            match self.evaluate_view(state) {
                Ok(view_value) => {
                    if let Ok(bytes) = bincode::serialize(&view_value) {
                        let mut hasher = AHasher::default();
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
        let mut hasher = AHasher::default();
        if let Ok(bytes) = bincode::serialize(state) {
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

        let instances = if self.module.instances.is_empty() {
            None
        } else {
            Some(&self.module.instances)
        };

        match evaluate_next_states_labeled_with_instances(
            &next_def.body,
            &self.next_name,
            &self.module.definitions,
            instances,
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

    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        self.fairness_constraints.clone()
    }

    fn num_next_disjuncts(&self) -> usize {
        let next_def = match self.module.definitions.get(&self.next_name) {
            Some(d) => d,
            None => return 0,
        };
        count_next_disjuncts(&next_def.body)
    }

    fn next_states_swarm(
        &self,
        state: &Self::State,
        enabled_mask: &[usize],
        out: &mut Vec<Self::State>,
    ) {
        let next_def = self
            .module
            .definitions
            .get(&self.next_name)
            .unwrap_or_else(|| panic!("missing Next definition '{}'", self.next_name));

        let instances = if self.module.instances.is_empty() {
            None
        } else {
            Some(&self.module.instances)
        };

        match evaluate_next_states_swarm(
            &next_def.body,
            &self.module.definitions,
            instances,
            state,
            enabled_mask,
        ) {
            Ok(states) => {
                out.extend(states);
            }
            Err(err) => {
                if self.allow_deadlock {
                    return;
                }
                panic!("swarm next-state evaluation failed: {err}");
            }
        }
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
            let record: BTreeMap<String, TlaValue> = state
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
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

        let instances = if self.module.instances.is_empty() {
            None
        } else {
            Some(&self.module.instances)
        };

        evaluate_next_states_labeled_with_instances(
            &next_def.body,
            &self.next_name,
            &self.module.definitions,
            instances,
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
            if let Some(def) = module.definitions.get(inv) {
                if def.params.is_empty() {
                    if std::env::var("TLAPP_TRACE_INVARIANT").is_ok() {
                        eprintln!(
                            "=== Invariant '{}' body ===\n{}\n=== End ===",
                            inv, def.body
                        );
                    }
                    return (inv.clone(), def.body.clone());
                }
                // Parameterized definition used as invariant — treat the name
                // as an identifier so the runtime evaluator can resolve it via
                // the definition context (which turns it into a zero-arg call
                // if the user really intended a constant).
            } else {
                // Definition not found in module — this can happen when the
                // invariant is defined in an EXTENDS'd module that could not
                // be loaded.  Fall through and use the name as the expression;
                // the runtime evaluator will attempt to resolve it via
                // ctx.definition().
                if std::env::var("TLAPP_TRACE_INVARIANT").is_ok() {
                    eprintln!(
                        "Warning: invariant '{}' not found in module definitions; \
                         will attempt runtime resolution",
                        inv
                    );
                }
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

/// Result of extracting Init or Next from a SPECIFICATION formula.
/// Can be either a reference to an existing definition or an inline expression.
#[derive(Debug, Clone)]
enum SpecComponent {
    /// Reference to an existing definition (e.g., "Init", "Next")
    Reference(String),
    /// Inline expression that needs to be wrapped in a synthetic definition
    Inline(String),
}

fn resolve_init_next_names(
    module: &mut TlaModule,
    cfg: &TlaConfig,
    init_override: Option<String>,
    next_override: Option<String>,
) -> Result<(String, String)> {
    let mut init = init_override.or_else(|| cfg.init.clone());
    let mut next = next_override.or_else(|| cfg.next.clone());

    // Try to extract Init/Next from SPECIFICATION if not explicitly provided
    if init.is_none() || next.is_none() {
        if let Some(spec_name) = cfg.specification.as_ref() {
            if let Some(spec_def) = module.definitions.get(spec_name).cloned() {
                let (spec_init, spec_next) = extract_init_next_from_spec(&spec_def.body, module);

                if init.is_none() {
                    if let Some(component) = spec_init {
                        match component {
                            SpecComponent::Reference(name) => {
                                init = Some(name);
                            }
                            SpecComponent::Inline(expr) => {
                                // Create a synthetic definition for the inline Init
                                let synth_name = "__SyntheticInit__".to_string();
                                module.definitions.insert(
                                    synth_name.clone(),
                                    TlaDefinition {
                                        name: synth_name.clone(),
                                        params: vec![],
                                        body: expr,
                                        is_recursive: false,
                                    },
                                );
                                init = Some(synth_name);
                            }
                        }
                    }
                }

                if next.is_none() {
                    if let Some(component) = spec_next {
                        match component {
                            SpecComponent::Reference(name) => {
                                next = Some(name);
                            }
                            SpecComponent::Inline(expr) => {
                                // Create a synthetic definition for the inline Next
                                let synth_name = "__SyntheticNext__".to_string();
                                module.definitions.insert(
                                    synth_name.clone(),
                                    TlaDefinition {
                                        name: synth_name.clone(),
                                        params: vec![],
                                        body: expr,
                                        is_recursive: false,
                                    },
                                );
                                next = Some(synth_name);
                            }
                        }
                    }
                }
            }
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

/// Extract Init and Next components from a SPECIFICATION formula.
///
/// Handles patterns like:
/// - `Spec == Init /\ [][Next]_vars`
/// - `Spec == /\ Init /\ [][Next]_vars`
/// - `Spec == Init /\ [][Next]_<<x, y, z>>`
/// - `Spec == x = 0 /\ [][x' = x + 1]_vars` (inline expressions)
///
/// Returns (Option<SpecComponent>, Option<SpecComponent>) for Init and Next respectively.
fn extract_init_next_from_spec(
    spec_body: &str,
    module: &TlaModule,
) -> (Option<SpecComponent>, Option<SpecComponent>) {
    extract_init_next_from_spec_inner(spec_body, module, 0)
}

fn extract_init_next_from_spec_inner(
    spec_body: &str,
    module: &TlaModule,
    depth: usize,
) -> (Option<SpecComponent>, Option<SpecComponent>) {
    if depth > 5 {
        return (None, None);
    }

    let parts = split_top_level(spec_body, "/\\");

    // If there's a single part that's a definition reference, chase it
    if parts.len() == 1 {
        let part = parts[0].trim();
        if let Some(name) = parse_simple_identifier(part) {
            if let Some(def) = module.definitions.get(&name) {
                let result = extract_init_next_from_spec_inner(&def.body, module, depth + 1);
                if result.0.is_some() || result.1.is_some() {
                    return result;
                }
            }
        }
    }

    let mut init: Option<SpecComponent> = None;
    let mut next: Option<SpecComponent> = None;

    for part in &parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Check for [][...]_vars pattern (Next)
        if next.is_none() {
            if let Some(rest) = part.strip_prefix("[][") {
                if let Some(idx) = rest.find("]_") {
                    let action_expr = rest[..idx].trim();
                    if let Some(name) = parse_simple_identifier(action_expr) {
                        // It's a reference to a definition
                        if module.definitions.contains_key(&name) {
                            next = Some(SpecComponent::Reference(name));
                        } else {
                            // Treat as inline expression (might be a variable name used as action)
                            next = Some(SpecComponent::Inline(action_expr.to_string()));
                        }
                    } else {
                        // It's an inline expression
                        next = Some(SpecComponent::Inline(action_expr.to_string()));
                    }
                    continue;
                }
            }
        }

        // Check for WF_/SF_ fairness constraints - skip these for Init/Next extraction
        if part.starts_with("WF_") || part.starts_with("SF_") {
            continue;
        }

        // If part is a definition reference whose body contains [][, chase it
        // This handles patterns like `Spec == TypeOK /\ LiveSpec` where LiveSpec
        // contains the [][Next]_vars pattern
        if next.is_none() {
            if let Some(name) = parse_simple_identifier(part) {
                if let Some(def) = module.definitions.get(&name) {
                    if def.body.contains("[][") {
                        let (sub_init, sub_next) =
                            extract_init_next_from_spec_inner(&def.body, module, depth + 1);
                        if sub_next.is_some() {
                            next = sub_next;
                            if init.is_none() {
                                init = sub_init;
                            }
                            continue;
                        }
                    }
                }
            }
        }

        // Otherwise, it might be Init
        if init.is_none() {
            if let Some(name) = parse_simple_identifier(part) {
                // It's a simple identifier - check if it's a known definition
                if module.definitions.contains_key(&name) {
                    init = Some(SpecComponent::Reference(name));
                } else {
                    // Might be a variable being constrained - treat as inline
                    init = Some(SpecComponent::Inline(part.to_string()));
                }
            } else {
                // It's an inline expression (e.g., "x = 0" or "x \in 1..10")
                init = Some(SpecComponent::Inline(part.to_string()));
            }
        }
    }

    (init, next)
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

fn parse_zero_arg_state_operator_ref(text: &str) -> Option<(Option<String>, String)> {
    let trimmed = text.trim();
    if let Some(name) = parse_simple_identifier(trimmed) {
        return Some((None, name));
    }

    let (alias, operator) = trimmed.split_once('!')?;
    let alias = alias.trim();
    let operator = operator.trim();
    if parse_simple_identifier(alias).is_some() && parse_simple_identifier(operator).is_some() {
        Some((Some(alias.to_string()), operator.to_string()))
    } else {
        None
    }
}

fn resolve_zero_arg_state_definition<'a>(
    expr: &str,
    definitions: &'a BTreeMap<String, TlaDefinition>,
    instances: &'a BTreeMap<String, TlaModuleInstance>,
) -> Option<(
    String,
    &'a TlaDefinition,
    &'a BTreeMap<String, TlaDefinition>,
    &'a BTreeMap<String, TlaModuleInstance>,
    Option<&'a TlaModuleInstance>,
)> {
    let (alias, name) = parse_zero_arg_state_operator_ref(expr)?;
    match alias {
        Some(alias) => {
            let instance = instances.get(alias.as_str())?;
            let module = instance.module.as_ref()?;
            let def = module.definitions.get(name.as_str())?;
            if !def.params.is_empty() || def.body.trim() == expr.trim() {
                return None;
            }
            Some((
                format!("{alias}!{name}"),
                def,
                &module.definitions,
                &module.instances,
                Some(instance),
            ))
        }
        None => {
            let def = definitions.get(name.as_str())?;
            if !def.params.is_empty() || def.body.trim() == expr.trim() {
                return None;
            }
            Some((name.to_string(), def, definitions, instances, None))
        }
    }
}

fn expand_state_predicate_clauses(
    body: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut visiting = BTreeSet::new();
    append_expanded_state_predicate_clause(
        body,
        definitions,
        instances,
        None,
        &mut visiting,
        &mut out,
    );
    out
}

fn append_expanded_state_predicate_clause(
    clause: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    instances: &BTreeMap<String, TlaModuleInstance>,
    active_instance: Option<&TlaModuleInstance>,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<String>,
) {
    let trimmed = clause.trim();
    if trimmed.is_empty() {
        return;
    }

    let parts = split_top_level(trimmed, "/\\");
    if parts.len() > 1 || trimmed.starts_with("/\\") {
        for part in parts {
            append_expanded_state_predicate_clause(
                &part,
                definitions,
                instances,
                active_instance,
                visiting,
                out,
            );
        }
        return;
    }

    if let Some((key, def, child_definitions, child_instances, instance_override)) =
        resolve_zero_arg_state_definition(trimmed, definitions, instances)
        && visiting.insert(key.clone())
    {
        let next_instance = instance_override.or(active_instance);
        append_expanded_state_predicate_clause(
            &def.body,
            child_definitions,
            child_instances,
            next_instance,
            visiting,
            out,
        );
        visiting.remove(&key);
        return;
    }

    if let Some(instance) = active_instance {
        out.push(apply_instance_substitutions_to_text(trimmed, instance));
    } else {
        out.push(trimmed.to_string());
    }
}

fn apply_instance_substitutions_to_text(expr: &str, instance: &TlaModuleInstance) -> String {
    if instance.substitutions.is_empty() {
        return expr.to_string();
    }

    let chars: Vec<char> = expr.chars().collect();
    let mut substitutions: Vec<(&str, &str)> = instance
        .substitutions
        .iter()
        .map(|(param, value_expr)| (param.as_str(), value_expr.as_str()))
        .collect();
    substitutions.sort_by_key(|(param, _)| std::cmp::Reverse(param.len()));

    let mut out = String::with_capacity(expr.len());
    let mut i = 0usize;
    let mut in_string = false;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            in_string = !in_string;
            out.push(ch);
            i += 1;
            continue;
        }
        if in_string {
            out.push(ch);
            i += 1;
            continue;
        }

        let is_ident_start = ch.is_alphabetic() || ch == '_';
        if !is_ident_start {
            out.push(ch);
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        let ident: String = chars[start..i].iter().collect();
        let mut replaced = false;
        for (param, value_expr) in &substitutions {
            if ident == *param {
                out.push_str(value_expr);
                replaced = true;
                break;
            }
        }
        if !replaced {
            out.push_str(&ident);
        }
    }

    out
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
    let definition_scope = merged_definition_scope(module);
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
    let mut deferred_operator_refs = Vec::new();
    for (k, v) in &cfg.constants {
        match v {
            ConfigValue::OperatorRef(name) => {
                deferred_operator_refs
                    .push((k.clone(), normalize_operator_ref_name(name).to_string()));
            }
            _ => {
                if let Some(tv) = config_value_to_tla(v) {
                    base_state.insert(Arc::from(k.as_str()), tv);
                }
            }
        }
    }
    for _ in 0..deferred_operator_refs.len().saturating_add(2) {
        if deferred_operator_refs.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_deferred = Vec::new();
        for (name, ref_name) in deferred_operator_refs {
            let ctx = EvalContext::with_definitions_and_instances(
                &base_state,
                &definition_scope,
                &module.instances,
            );
            match eval_expr(&ref_name, &ctx) {
                Ok(value) => {
                    base_state.insert(Arc::from(name.as_str()), value);
                    progress = true;
                }
                Err(_) => {
                    // Try resolving as a zero-arg definition from definition_scope
                    if let Some(def) = definition_scope.get(&ref_name) {
                        if def.params.is_empty() {
                            if let Ok(value) = eval_expr(&def.body, &ctx) {
                                base_state.insert(Arc::from(name.as_str()), value);
                                progress = true;
                                continue;
                            }
                        }
                    }
                    next_deferred.push((name, ref_name));
                }
            }
        }

        if !progress {
            break;
        }
        deferred_operator_refs = next_deferred;
    }

    // Classify all clauses
    let mut equality_assignments: Vec<(String, String)> = Vec::new();
    let mut membership_assignments: Vec<(String, String)> = Vec::new();
    let mut guards: Vec<String> = Vec::new();

    for clause in
        expand_state_predicate_clauses(&init_def.body, &module.definitions, &module.instances)
    {
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
    for _ in 0..pending.len().saturating_add(2) {
        if pending.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();
        for (var, expr) in pending {
            let ctx = EvalContext::with_definitions_and_instances(
                &base_state,
                &definition_scope,
                &module.instances,
            );
            match eval_expr(&expr, &ctx) {
                Ok(value) => {
                    base_state.insert(Arc::from(var.as_str()), value);
                    progress = true;
                }
                Err(_) => next_pending.push((var, expr)),
            }
        }

        if !progress {
            // Recovery: try evaluating with definitions injected as state values.
            // Some Init expressions reference operators/constants that aren't in state
            // but are in definition_scope (e.g., PlusCal ProcSet, Initiator).
            let mut augmented_state = base_state.clone();
            for (name, def) in &definition_scope {
                if !augmented_state.contains_key(name.as_str()) && def.params.is_empty() {
                    let ctx = EvalContext::with_definitions_and_instances(
                        &augmented_state,
                        &definition_scope,
                        &module.instances,
                    );
                    if let Ok(val) = eval_expr(&def.body, &ctx) {
                        augmented_state.insert(Arc::from(name.as_str()), val);
                    }
                }
            }

            // Retry with augmented state
            let mut retry_pending = Vec::new();
            for (var, expr) in next_pending {
                let ctx = EvalContext::with_definitions_and_instances(
                    &augmented_state,
                    &definition_scope,
                    &module.instances,
                );
                match eval_expr(&expr, &ctx) {
                    Ok(value) => {
                        base_state.insert(Arc::from(var.as_str()), value);
                        progress = true;
                    }
                    Err(_) => retry_pending.push((var, expr)),
                }
            }
            next_pending = retry_pending;

            if !progress {
                // Demote remaining to membership or guard — some variables may
                // get assigned during guard evaluation or can be skipped
                for (var, expr) in &next_pending {
                    eprintln!(
                        "Warning: could not resolve Init assignment for '{}', treating as guard",
                        var
                    );
                    guards.push(format!("{} = {}", var, expr));
                }
                next_pending.clear();
                break;
            }
        }

        pending = next_pending;
    }

    // Now handle membership assignments (nondeterministic)
    // Evaluate each set expression and collect possible values
    let mut membership_choices: Vec<(String, Vec<TlaValue>)> = Vec::new();

    for (var, set_expr) in membership_assignments {
        let ctx = EvalContext::with_definitions_and_instances(
            &base_state,
            &definition_scope,
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
                new_state.insert(Arc::from(var.as_str()), value.clone());
                new_states.push(new_state);
            }
        }
        states = new_states;

        // Limit total number of initial states
        const MAX_INIT_STATES: usize = 10_000_000;
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
            &definition_scope,
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
            let all_assigned = module
                .variables
                .iter()
                .all(|v| state.contains_key(v.as_str()));
            if all_assigned {
                valid_states.push(state);
            }
        }
    }

    // If no valid states but some variables are unassigned, try recovery:
    // re-scan guards for patterns like `var = expr` or `var \in set` that
    // classify_clause missed (common in PlusCal-generated Init)
    if valid_states.is_empty() && !had_membership_choices {
        let missing: Vec<String> = module
            .variables
            .iter()
            .filter(|v| !base_state_for_error.contains_key(v.as_str()))
            .cloned()
            .collect();

        if !missing.is_empty() {
            let mut recovered = false;
            let mut recovery_state = base_state_for_error.clone();
            for guard in &guards {
                for var in &missing {
                    if recovery_state.contains_key(var.as_str()) {
                        continue;
                    }
                    // Try to find `var = expr` pattern using bracket-aware splitting
                    let pattern = format!("{} = ", var);
                    if let Some(idx) = guard.find(&pattern) {
                        let rhs = guard[idx + pattern.len()..].trim();
                        let ctx = EvalContext::with_definitions_and_instances(
                            &recovery_state,
                            &definition_scope,
                            &module.instances,
                        );
                        if let Ok(value) = eval_expr(rhs, &ctx) {
                            recovery_state.insert(Arc::from(var.as_str()), value);
                            recovered = true;
                        }
                    }
                    // Try `var \in set` pattern
                    let mem_pattern = format!("{} \\in ", var);
                    if let Some(idx) = guard.find(&mem_pattern) {
                        let rhs = guard[idx + mem_pattern.len()..].trim();
                        let ctx = EvalContext::with_definitions_and_instances(
                            &recovery_state,
                            &definition_scope,
                            &module.instances,
                        );
                        if let Ok(set_val) = eval_expr(rhs, &ctx) {
                            if let Ok(set) = set_val.as_set() {
                                if let Some(first) = set.iter().next() {
                                    recovery_state.insert(Arc::from(var.as_str()), first.clone());
                                    recovered = true;
                                }
                            }
                        }
                    }
                }
            }

            if recovered {
                // Check if all variables now assigned
                let all_assigned = module
                    .variables
                    .iter()
                    .all(|v| recovery_state.contains_key(v.as_str()));
                if all_assigned {
                    valid_states.push(recovery_state);
                }
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
            .filter(|v| !base_state_for_error.contains_key(v.as_str()))
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

fn merged_definition_scope(module: &TlaModule) -> BTreeMap<String, TlaDefinition> {
    let mut definitions = module.definitions.clone();
    add_instance_definition_fallbacks(&module.instances, &mut definitions);
    definitions
}

fn add_instance_definition_fallbacks(
    instances: &BTreeMap<String, TlaModuleInstance>,
    definitions: &mut BTreeMap<String, TlaDefinition>,
) {
    for instance in instances.values() {
        let Some(module) = instance.module.as_ref() else {
            continue;
        };
        for (name, def) in &module.definitions {
            definitions
                .entry(name.clone())
                .or_insert_with(|| def.clone());
        }
        add_instance_definition_fallbacks(&module.instances, definitions);
    }
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

/// TLC-style directory search: find definitions referenced by config (invariants,
/// properties) that aren't in the module's EXTENDS chain, by parsing sibling .tla
/// files in the same directory.
fn resolve_sibling_definitions(module_path: &Path, module: &mut TlaModule, config: &TlaConfig) {
    // Collect names referenced by config that aren't in module definitions
    let mut missing: Vec<String> = Vec::new();
    for inv in &config.invariants {
        if !module.definitions.contains_key(inv) {
            missing.push(inv.clone());
        }
    }
    for prop in &config.properties {
        if !module.definitions.contains_key(prop) {
            missing.push(prop.clone());
        }
    }

    if missing.is_empty() {
        return;
    }

    // Search sibling .tla files in the same directory
    let dir = module_path.parent().unwrap_or(Path::new("."));
    let module_filename = module_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");

    let tla_files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_str().unwrap_or("");
            name.ends_with(".tla") && name != module_filename
        })
        .collect();

    for entry in &tla_files {
        let path = entry.path();
        let sibling = match parse_tla_module_file(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let mut found_any = false;
        for name in &missing {
            if let Some(def) = sibling.definitions.get(name) {
                eprintln!(
                    "Resolved '{}' from sibling module {} (TLC-style directory search)",
                    name,
                    path.display()
                );
                module.definitions.insert(name.clone(), def.clone());
                found_any = true;
            }
        }

        if found_any {
            // Merge definitions (operators) from sibling — needed to evaluate
            // the resolved invariant/property expressions. Do NOT merge variables
            // — those belong to the sibling's state space, not ours.
            for (k, v) in &sibling.definitions {
                module
                    .definitions
                    .entry(k.clone())
                    .or_insert_with(|| v.clone());
            }
        }
    }

    // Report any still-missing definitions
    for name in &missing {
        if !module.definitions.contains_key(name) {
            eprintln!(
                "Warning: '{}' not found in module definitions or sibling .tla files",
                name
            );
        }
    }
}

/// Inject constants from config into module definitions
///
/// This makes constants available during action evaluation by creating
/// zero-parameter operator definitions for each constant.
fn inject_constants_into_definitions(module: &mut TlaModule, config: &TlaConfig) {
    inject_constants_into_module_tree(module, config, true);
}

fn inject_constants_into_module_tree(
    module: &mut TlaModule,
    config: &TlaConfig,
    include_all_constants: bool,
) {
    for (name, value) in &config.constants {
        if !include_all_constants
            && !module.constants.iter().any(|constant| constant == name)
            && !module.definitions.contains_key(name)
        {
            continue;
        }

        let (params, body) = match value {
            ConfigValue::OperatorRef(target_name) => {
                let target_name = normalize_operator_ref_name(target_name);
                if let Some(target_def) = module.definitions.get(target_name) {
                    let params = target_def.params.clone();
                    let body = if params.is_empty() {
                        target_name.to_string()
                    } else {
                        format!("{target_name}({})", params.join(", "))
                    };
                    (params, body)
                } else {
                    (Vec::new(), target_name.to_string())
                }
            }
            _ => (Vec::new(), config_value_to_expr(value)),
        };

        module.definitions.insert(
            name.clone(),
            TlaDefinition {
                name: name.clone(),
                params,
                body,
                is_recursive: false,
            },
        );
    }

    for instance in module.instances.values_mut() {
        if let Some(instance_module) = instance.module.as_mut() {
            inject_constants_into_module_tree(instance_module, config, false);
        }
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
        ConfigValue::OperatorRef(name) => normalize_operator_ref_name(name).to_string(),
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
    fn deadlocked_state_does_not_panic_with_allow_deadlock() {
        let tmp = std::env::temp_dir().join("tlapp-native-deadlock-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("Deadlock.tla");
        fs::write(
            &module,
            r#"
---- MODULE Deadlock ----
EXTENDS Naturals
VARIABLES x
Init == x = 0
Next == x < 3 /\ x' = x + 1
Spec == Init /\ [][Next]_x
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("Deadlock.cfg");
        fs::write(&cfg, "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\n")
            .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        assert!(model.allow_deadlock);

        // State x=3 should be deadlocked (x < 3 is false)
        let deadlock_state = tla_state([("x", TlaValue::Int(3))]);
        let mut next = Vec::new();
        model.next_states(&deadlock_state, &mut next);
        // With allow_deadlock, should return empty vec (no successors) without panic
        assert!(next.is_empty());
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

    #[test]
    fn specification_with_inline_init_and_next() {
        // Test that inline Init/Next expressions in SPECIFICATION work
        let tmp = std::env::temp_dir().join("tlapp-spec-inline-init-next");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("InlineSpec.tla");
        fs::write(
            &module,
            r#"
---- MODULE InlineSpec ----
EXTENDS Naturals
VARIABLES x
vars == <<x>>
Spec == x = 0 /\ [][x' = x + 1 \/ UNCHANGED x]_vars
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("InlineSpec.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        // Synthetic definitions should be created
        assert_eq!(model.init_name, "__SyntheticInit__");
        assert_eq!(model.next_name, "__SyntheticNext__");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty());
    }

    #[test]
    fn specification_with_named_init_and_next() {
        // Test that named Init/Next references in SPECIFICATION work
        let tmp = std::env::temp_dir().join("tlapp-spec-named-init-next");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("NamedSpec.tla");
        fs::write(
            &module,
            r#"
---- MODULE NamedSpec ----
EXTENDS Naturals
VARIABLES x
MyInit == x = 0
MyNext == x' = x + 1 \/ UNCHANGED x
vars == <<x>>
Spec == MyInit /\ [][MyNext]_vars
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("NamedSpec.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        // Named definitions should be referenced directly
        assert_eq!(model.init_name, "MyInit");
        assert_eq!(model.next_name, "MyNext");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty());
    }

    #[test]
    fn specification_with_leading_conjunction() {
        // Test Spec == /\ Init /\ [][Next]_vars pattern
        let tmp = std::env::temp_dir().join("tlapp-spec-leading-conj");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("LeadingConj.tla");
        fs::write(
            &module,
            r#"
---- MODULE LeadingConj ----
EXTENDS Naturals
VARIABLES x
Init == x = 0
Next == x' = x + 1 \/ UNCHANGED x
vars == <<x>>
Spec == /\ Init /\ [][Next]_vars
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("LeadingConj.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        assert_eq!(model.init_name, "Init");
        assert_eq!(model.next_name, "Next");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
    }

    #[test]
    fn specification_with_tuple_vars() {
        // Test Spec == Init /\ [][Next]_<<x, y>> pattern
        let tmp = std::env::temp_dir().join("tlapp-spec-tuple-vars");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("TupleVars.tla");
        fs::write(
            &module,
            r#"
---- MODULE TupleVars ----
EXTENDS Naturals
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y \/ UNCHANGED <<x, y>>
Spec == Init /\ [][Next]_<<x, y>>
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("TupleVars.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        assert_eq!(model.init_name, "Init");
        assert_eq!(model.next_name, "Next");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));
        assert_eq!(init[0].get("y"), Some(&TlaValue::Int(0)));
    }

    #[test]
    fn specification_with_mixed_inline_and_named() {
        // Test inline Init with named Next
        let tmp = std::env::temp_dir().join("tlapp-spec-mixed-inline-named");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("MixedSpec.tla");
        fs::write(
            &module,
            r#"
---- MODULE MixedSpec ----
EXTENDS Naturals
VARIABLES x
MyNext == x' = x + 1 \/ UNCHANGED x
vars == <<x>>
Spec == x = 0 /\ [][MyNext]_vars
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("MixedSpec.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        // Init should be synthetic, Next should be named
        assert_eq!(model.init_name, "__SyntheticInit__");
        assert_eq!(model.next_name, "MyNext");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));
    }

    #[test]
    fn explicit_init_next_overrides_specification() {
        // Test that explicit INIT/NEXT in config override SPECIFICATION extraction
        let tmp = std::env::temp_dir().join("tlapp-spec-explicit-override");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("OverrideSpec.tla");
        fs::write(
            &module,
            r#"
---- MODULE OverrideSpec ----
EXTENDS Naturals
VARIABLES x
Init == x = 0
AltInit == x = 10
Next == x' = x + 1 \/ UNCHANGED x
AltNext == x' = x + 2 \/ UNCHANGED x
vars == <<x>>
Spec == Init /\ [][Next]_vars
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("OverrideSpec.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
INIT AltInit
NEXT AltNext
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");

        // Explicit INIT/NEXT should override SPECIFICATION extraction
        assert_eq!(model.init_name, "AltInit");
        assert_eq!(model.next_name, "AltNext");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(10)));
    }

    #[test]
    fn extends_inherits_init_next_from_base_module() {
        // Test case: MCDieHarder EXTENDS DieHarder where Init, Next, Spec are in DieHarder
        let tmp = std::env::temp_dir().join("tlapp-extends-full-model-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Create base module with Init, Next, Spec
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
    \/ /\ UNCHANGED <<small, big>>

Goal == small + big <= 10

Spec == Init /\ [][Next]_<<small, big>>
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

\* Additional invariant for model checking
TypeOK ==
    /\ small \in 0..5
    /\ big \in 0..5
====
"#,
        )
        .expect("extending module should be written");

        // Create config file referencing Init/Next from extended module
        let cfg = tmp.join("MCDieHarder.cfg");
        fs::write(
            &cfg,
            r#"
INIT Init
NEXT Next
INVARIANT TypeOK
INVARIANT Goal
"#,
        )
        .expect("cfg should be written");

        // Build the model - this should work because Init/Next are inherited from DieHarder
        let model = TlaModel::from_files(&extending_module, Some(&cfg), None, None)
            .expect("model should build with inherited Init/Next");

        // Verify that init_name and next_name are resolved
        assert_eq!(model.init_name, "Init");
        assert_eq!(model.next_name, "Next");

        // Check initial states
        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("small"), Some(&TlaValue::Int(0)));
        assert_eq!(init[0].get("big"), Some(&TlaValue::Int(0)));

        // Check that next states work
        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty(), "Next states should be computed");

        // Verify invariants work
        let invariant_result = model.check_invariants(&init[0]);
        assert!(
            invariant_result.is_ok(),
            "Initial state should satisfy invariants"
        );

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn extends_chain_model_works() {
        // Test a chain of EXTENDS: C extends B extends A
        let tmp = std::env::temp_dir().join("tlapp-extends-chain-model-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        // Module A: Base definitions
        let module_a = tmp.join("BaseSpec.tla");
        fs::write(
            &module_a,
            r#"
---- MODULE BaseSpec ----
EXTENDS Naturals

VARIABLES x

Init == x = 0
====
"#,
        )
        .expect("module A should be written");

        // Module B: Extends A, adds Next
        let module_b = tmp.join("MidSpec.tla");
        fs::write(
            &module_b,
            r#"
---- MODULE MidSpec ----
EXTENDS BaseSpec

Next == x' = x + 1 \/ UNCHANGED x
====
"#,
        )
        .expect("module B should be written");

        // Module C: Extends B, adds invariants
        let module_c = tmp.join("TopSpec.tla");
        fs::write(
            &module_c,
            r#"
---- MODULE TopSpec ----
EXTENDS MidSpec

TypeOK == x \in 0..10
====
"#,
        )
        .expect("module C should be written");

        let cfg = tmp.join("TopSpec.cfg");
        fs::write(
            &cfg,
            r#"
INIT Init
NEXT Next
INVARIANT TypeOK
"#,
        )
        .expect("cfg should be written");

        // Build model - Init from BaseSpec, Next from MidSpec, TypeOK from TopSpec
        let model = TlaModel::from_files(&module_c, Some(&cfg), None, None)
            .expect("model should build with chain of EXTENDS");

        assert_eq!(model.init_name, "Init");
        assert_eq!(model.next_name, "Next");

        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(0)));

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert!(!next.is_empty());

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_init_membership_constraint() {
        // Test that Init with membership constraints like `f \in [Proc -> Values]`
        // correctly enumerates all possible initial states
        let tmp = std::env::temp_dir().join("tlaplusplus_init_membership_test");
        let _ = fs::create_dir_all(&tmp);

        let module = tmp.join("MembershipInit.tla");
        fs::write(
            &module,
            r#"
---- MODULE MembershipInit ----
EXTENDS Naturals

CONSTANT Proc, Values

VARIABLE f

Init == f \in [Proc -> Values]

Next == UNCHANGED f

TypeOK == f \in [Proc -> Values]
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("MembershipInit.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANT Proc = {p1, p2}
CONSTANT Values = {v1, v2}
INIT Init
NEXT Next
INVARIANT TypeOK
"#,
        )
        .expect("cfg should be written");

        // Build model - should properly enumerate all functions in [Proc -> Values]
        let model = TlaModel::from_files(&module, Some(&cfg), None, None)
            .expect("model should build with Init membership constraint");

        let init = model.initial_states();
        // [Proc -> Values] with |Proc| = 2, |Values| = 2 should give 2^2 = 4 functions
        assert_eq!(
            init.len(),
            4,
            "Expected 4 initial states for [{{p1, p2}} -> {{v1, v2}}]"
        );

        // Each initial state should have a function f that is valid for the constraint
        for state in &init {
            let f = state.get("f").expect("f should be defined");
            // f should be a Function
            assert!(
                matches!(f, TlaValue::Function(_)),
                "f should be a function, got: {:?}",
                f
            );
            // The function should have 2 keys (one for each element in Proc)
            if let TlaValue::Function(func) = f {
                assert_eq!(func.len(), 2, "function should map 2 domain elements");
            }
        }

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_init_mixed_equality_and_membership() {
        // Test Init with both equality and membership constraints
        let tmp = std::env::temp_dir().join("tlaplusplus_init_mixed_test");
        let _ = fs::create_dir_all(&tmp);

        let module = tmp.join("MixedInit.tla");
        fs::write(
            &module,
            r#"
---- MODULE MixedInit ----
EXTENDS Naturals

CONSTANT S

VARIABLES x, y

Init ==
    /\ x = 0
    /\ y \in S

Next == UNCHANGED <<x, y>>
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("MixedInit.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANT S = {a, b, c}
INIT Init
NEXT Next
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&module, Some(&cfg), None, None)
            .expect("model should build with mixed Init constraints");

        let init = model.initial_states();
        // x = 0 is deterministic, y \in {a, b, c} gives 3 choices
        assert_eq!(
            init.len(),
            3,
            "Expected 3 initial states for x=0 and y in {{a, b, c}}"
        );

        // Each state should have x = 0
        for state in &init {
            assert_eq!(state.get("x"), Some(&TlaValue::Int(0)));
            // y should be one of the model values
            let y = state.get("y").expect("y should be defined");
            assert!(
                matches!(y, TlaValue::ModelValue(_)),
                "y should be a model value, got: {:?}",
                y
            );
        }

        // Clean up
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn generated_model_separators_do_not_break_operator_ref_init_resolution() {
        let tmp = std::env::temp_dir().join("tlapp-generated-model-separators");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let consensus = tmp.join("Consensus.tla");
        fs::write(
            &consensus,
            r#"
---- MODULE Consensus ----
VARIABLE chosen
Init == chosen = {}
Next == UNCHANGED chosen
Spec == Init /\ [][Next]_<<chosen>>
====
"#,
        )
        .expect("consensus module should be written");

        let voting = tmp.join("Voting.tla");
        fs::write(
            &voting,
            r#"
---- MODULE Voting ----
EXTENDS Integers
CONSTANTS Value, Acceptor, Quorum
Ballot == Nat
VARIABLES votes, maxBal
Init == /\ votes  = [a \in Acceptor |-> {}]
        /\ maxBal = [a \in Acceptor |-> -1]
IncreaseMaxBal(a, b) ==
    /\ b > maxBal[a]
    /\ maxBal' = [maxBal EXCEPT ![a] = b]
    /\ UNCHANGED votes
VoteFor(a, b, v) ==
    /\ maxBal[a] =< b
    /\ votes' = [votes EXCEPT ![a] = votes[a] \cup {<<b, v>>}]
    /\ maxBal' = [maxBal EXCEPT ![a] = b]
Next == \E a \in Acceptor, b \in Ballot :
            \/ IncreaseMaxBal(a, b)
            \/ \E v \in Value : VoteFor(a, b, v)
chosen == {}
C == INSTANCE Consensus WITH chosen <- chosen
Spec == Init /\ [][Next]_<<votes, maxBal>>
====
"#,
        )
        .expect("voting module should be written");

        let model = tmp.join("GeneratedVoting.tla");
        fs::write(
            &model,
            r#"
---- MODULE GeneratedVoting ----
EXTENDS Voting, TLC
CONSTANTS a1, a2, v1, v2

const_acceptor ==
{a1, a2}
----

const_value ==
{v1, v2}
----

const_quorum ==
{{a1, a2}}
----

def_ballot ==
0..2
----

====
"#,
        )
        .expect("generated model should be written");

        let cfg = tmp.join("GeneratedVoting.cfg");
        fs::write(
            &cfg,
            r#"
CONSTANT Acceptor <- const_acceptor
CONSTANT Value <- const_value
CONSTANT Quorum <- const_quorum
CONSTANT Ballot <- def_ballot
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&model, Some(&cfg), None, None)
            .expect("generated model should build");
        let init = model.initial_states();

        assert_eq!(init.len(), 1);
        let votes = init[0].get("votes").expect("votes should be defined");
        let max_bal = init[0].get("maxBal").expect("maxBal should be defined");
        assert!(matches!(votes, TlaValue::Function(_)));
        assert!(matches!(max_bal, TlaValue::Function(_)));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn initial_states_expand_instance_init_references() {
        let tmp = std::env::temp_dir().join("tlapp-instance-init-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let nano = tmp.join("Nano.tla");
        fs::write(
            &nano,
            r#"
---- MODULE Nano ----
VARIABLES x, y
Init == /\ x = 1
        /\ y = 2
Next == /\ UNCHANGED <<x, y>>
====
"#,
        )
        .expect("nano module should be written");

        let mc = tmp.join("MC.tla");
        fs::write(
            &mc,
            r#"
---- MODULE MC ----
VARIABLES x, y, z
N == INSTANCE Nano
Init == /\ z = 0
        /\ N!Init
Next == /\ UNCHANGED <<x, y, z>>
Spec == Init /\ [][Next]_<<x, y, z>>
====
"#,
        )
        .expect("mc module should be written");

        let cfg = tmp.join("MC.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model = TlaModel::from_files(&mc, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();

        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("x"), Some(&TlaValue::Int(1)));
        assert_eq!(init[0].get("y"), Some(&TlaValue::Int(2)));
        assert_eq!(init[0].get("z"), Some(&TlaValue::Int(0)));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn initial_states_apply_instance_substitutions_from_multiline_with_clauses() {
        let tmp = std::env::temp_dir().join("tlapp-instance-init-substitution-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let child = tmp.join("Child.tla");
        fs::write(
            &child,
            r#"
---- MODULE Child ----
CONSTANT Jug
VARIABLE contents
Init == /\ contents = [j \in Jug |-> 0]
Next == /\ UNCHANGED contents
====
"#,
        )
        .expect("child module should be written");

        let model = tmp.join("MC.tla");
        fs::write(
            &model,
            r#"
---- MODULE MC ----
VARIABLES c1, z
D == INSTANCE Child WITH contents <- c1,
                           Jug <- {"j1", "j2"}
Init == /\ z = 0
        /\ D!Init
Next == /\ UNCHANGED <<c1, z>>
Spec == Init /\ [][Next]_<<c1, z>>
====
"#,
        )
        .expect("mc module should be written");

        let cfg = tmp.join("MC.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&model, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();

        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("z"), Some(&TlaValue::Int(0)));
        let c1 = init[0].get("c1").expect("c1 should be defined");
        let TlaValue::Function(entries) = c1 else {
            panic!("c1 should be a function");
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries.get(&TlaValue::String("j1".to_string())),
            Some(&TlaValue::Int(0))
        );
        assert_eq!(
            entries.get(&TlaValue::String("j2".to_string())),
            Some(&TlaValue::Int(0))
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn initial_states_resolve_helper_defs_from_instance_inits() {
        let tmp = std::env::temp_dir().join("tlapp-instance-init-helper-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let child = tmp.join("Child.tla");
        fs::write(
            &child,
            r#"
---- MODULE Child ----
VARIABLE x
LastX == 2
Init == /\ x = [i \in 0..LastX |-> i]
Next == /\ UNCHANGED x
====
"#,
        )
        .expect("child module should be written");

        let model = tmp.join("MC.tla");
        fs::write(
            &model,
            r#"
---- MODULE MC ----
VARIABLES x, z
C == INSTANCE Child
Init == /\ z = 0
        /\ C!Init
Next == /\ UNCHANGED <<x, z>>
Spec == Init /\ [][Next]_<<x, z>>
====
"#,
        )
        .expect("mc module should be written");

        let cfg = tmp.join("MC.cfg");
        fs::write(
            &cfg,
            r#"
SPECIFICATION Spec
"#,
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&model, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();

        assert_eq!(init.len(), 1);
        assert_eq!(init[0].get("z"), Some(&TlaValue::Int(0)));
        let x = init[0].get("x").expect("x should be defined");
        let TlaValue::Function(entries) = x else {
            panic!("x should be a function");
        };
        assert_eq!(entries.len(), 3);
        assert_eq!(entries.get(&TlaValue::Int(0)), Some(&TlaValue::Int(0)));
        assert_eq!(entries.get(&TlaValue::Int(2)), Some(&TlaValue::Int(2)));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exclusive_lease_guard_blocks_second_acquisition() {
        // Reproduces Bug 4: IF/THEN with \A guard must block exclusive lease
        // when another client already holds one.
        let tmp = std::env::temp_dir().join("tlapp-exclusive-guard-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir should be created");

        let module = tmp.join("ExclusiveGuard.tla");
        fs::write(
            &module,
            r#"
---- MODULE ExclusiveGuard ----
EXTENDS Naturals, FiniteSets
CONSTANTS Clients
VARIABLES leases

InitLeaseState == [held |-> FALSE, access |-> "none"]

Init == leases = [c \in Clients |-> InitLeaseState]

AcquireLease(client, accessType) ==
    /\ ~leases[client].held
    /\ accessType \in {"read", "write", "exclusive"}
    /\ IF accessType = "exclusive"
       THEN \A c \in Clients : ~leases[c].held \/ c = client
       ELSE TRUE
    /\ leases' = [leases EXCEPT ![client] = [held |-> TRUE, access |-> accessType]]

Next == \E c \in Clients, a \in {"read", "write", "exclusive"} : AcquireLease(c, a)

ExclusiveUnique ==
    Cardinality({c \in Clients : leases[c].held /\ leases[c].access = "exclusive"}) <= 1

Spec == Init /\ [][Next]_leases
====
"#,
        )
        .expect("module should be written");

        let cfg = tmp.join("ExclusiveGuard.cfg");
        fs::write(
            &cfg,
            "SPECIFICATION Spec\nCONSTANTS Clients = {c1, c2}\nINVARIANT ExclusiveUnique\n",
        )
        .expect("cfg should be written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1);

        // Get all successors from initial state
        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        // Should be able to acquire leases (3 access types x 2 clients = up to 6)
        assert!(!next.is_empty());

        // Now from a state where c1 holds an exclusive lease, try to get successors
        // c2 should NOT be able to acquire an exclusive lease
        let c1_exclusive = next
            .iter()
            .find(|s| {
                if let Some(TlaValue::Function(f)) = s.get("leases") {
                    f.values().any(|v| {
                        if let TlaValue::Record(r) = v {
                            r.get("held") == Some(&TlaValue::Bool(true))
                                && r.get("access")
                                    == Some(&TlaValue::String("exclusive".to_string()))
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
            })
            .expect("should have a state where someone holds exclusive lease");

        let mut next2 = Vec::new();
        model.next_states(c1_exclusive, &mut next2);

        // Check invariant: no state should have 2 clients with exclusive leases
        for state in &next2 {
            let result = model.check_invariants(state);
            assert!(
                result.is_ok(),
                "ExclusiveUnique invariant violated! State: {state:?}"
            );
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Test: primed variables from one disjunct must not leak into another.
    /// If branch isolation is broken, x' from the first branch could appear
    /// in states generated by the second branch.
    #[test]
    fn disjunct_branches_do_not_leak_primed_variables() {
        let tmp = std::env::temp_dir().join("tlapp-branch-isolation-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        let module = tmp.join("BranchIsolation.tla");
        fs::write(
            &module,
            r#"
---- MODULE BranchIsolation ----
EXTENDS Naturals
VARIABLES x, y

Init == x = 0 /\ y = 0

Next ==
    \/ /\ x' = x + 1    \* Branch 1: only changes x
       /\ UNCHANGED y
    \/ /\ y' = y + 10   \* Branch 2: only changes y
       /\ UNCHANGED x

\* If branches leak, we might see x=1,y=10 in one step
NoLeakInvariant ==
    \/ (x = 0 /\ y = 0)               \* Initial
    \/ (x = 1 /\ y = 0)               \* After branch 1
    \/ (x = 0 /\ y = 10)              \* After branch 2
    \/ (x >= 1 /\ y >= 0)             \* Multi-step combinations
    \/ (x >= 0 /\ y >= 10)

\* Single-step invariant: from init, can't reach x=1,y=10
SingleStepNoLeak ==
    ~(x = 1 /\ y = 10)

Spec == Init /\ [][Next]_<<x, y>>
====
"#,
        )
        .expect("write module");

        let cfg = tmp.join("BranchIsolation.cfg");
        fs::write(&cfg, "SPECIFICATION Spec\nINVARIANT SingleStepNoLeak\n").expect("write cfg");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1);

        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);

        // Should produce exactly 2 successors from init
        assert_eq!(next.len(), 2, "expected 2 successors, got {next:?}");

        // Verify no state has both x=1 and y=10 (would indicate leaked primes)
        for state in &next {
            let x = state.get("x").and_then(|v| v.as_int().ok()).unwrap_or(-1);
            let y = state.get("y").and_then(|v| v.as_int().ok()).unwrap_or(-1);
            assert!(
                !(x == 1 && y == 10),
                "Branch isolation violated: x={x}, y={y} in single step from init"
            );
            // Also verify invariant
            assert!(
                model.check_invariants(state).is_ok(),
                "Invariant violated: {state:?}"
            );
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Test: UNCHANGED must produce exact copies, not defaults.
    #[test]
    fn unchanged_preserves_non_default_values() {
        let tmp = std::env::temp_dir().join("tlapp-unchanged-preserves-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        let module = tmp.join("UnchangedPreserves.tla");
        fs::write(
            &module,
            r#"
---- MODULE UnchangedPreserves ----
EXTENDS Naturals
VARIABLES counter, label

Init == counter = 42 /\ label = "hello"

Tick == /\ counter' = counter + 1
       /\ UNCHANGED label

\* label must never lose its value
LabelPreserved == label = "hello"

Spec == Init /\ [][Tick]_<<counter, label>>
====
"#,
        )
        .expect("write module");

        let cfg = tmp.join("UnchangedPreserves.cfg");
        fs::write(&cfg, "SPECIFICATION Spec\nINVARIANT LabelPreserved\n").expect("write cfg");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init[0].get("counter"), Some(&TlaValue::Int(42)));
        assert_eq!(
            init[0].get("label"),
            Some(&TlaValue::String("hello".to_string()))
        );

        // Step multiple times and verify label stays "hello"
        let mut current = init[0].clone();
        for step in 0..5 {
            let mut next = Vec::new();
            model.next_states(&current, &mut next);
            assert!(!next.is_empty(), "step {step} produced no successors");
            let state = &next[0];
            assert_eq!(
                state.get("label"),
                Some(&TlaValue::String("hello".to_string())),
                "UNCHANGED label failed at step {step}: {state:?}"
            );
            assert!(
                model.check_invariants(state).is_ok(),
                "Invariant violated at step {step}: {state:?}"
            );
            current = state.clone();
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Test: implication in action guards (P => Q).
    #[test]
    fn implication_guard_blocks_correctly() {
        let tmp = std::env::temp_dir().join("tlapp-implication-guard-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        let module = tmp.join("ImplicationGuard.tla");
        fs::write(
            &module,
            r#"
---- MODULE ImplicationGuard ----
EXTENDS Naturals
CONSTANTS Procs
VARIABLES pc, val

Init == pc = [p \in Procs |-> "idle"] /\ val = [p \in Procs |-> 0]

\* Can only write if pc = "ready" (implication guard pattern)
Write(p) ==
    /\ pc[p] = "ready" => val[p] > 0   \* If ready, must have positive val
    /\ pc[p] = "ready"
    /\ pc' = [pc EXCEPT ![p] = "done"]
    /\ UNCHANGED val

Ready(p) ==
    /\ pc[p] = "idle"
    /\ val' = [val EXCEPT ![p] = 1]
    /\ pc' = [pc EXCEPT ![p] = "ready"]

Next == \E p \in Procs : Ready(p) \/ Write(p)

\* A proc in "done" state must have had val > 0
DoneImpliesPositiveVal ==
    \A p \in Procs : pc[p] = "done" => val[p] > 0

Spec == Init /\ [][Next]_<<pc, val>>
====
"#,
        )
        .expect("write module");

        let cfg = tmp.join("ImplicationGuard.cfg");
        fs::write(
            &cfg,
            "SPECIFICATION Spec\nCONSTANTS Procs = {p1, p2}\nINVARIANT DoneImpliesPositiveVal\n",
        )
        .expect("write cfg");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();

        // Explore 3 levels deep and check invariant at every state
        let mut frontier = init.clone();
        let mut all_states = init.clone();
        for depth in 0..3 {
            let mut next_frontier = Vec::new();
            for state in &frontier {
                let mut next = Vec::new();
                model.next_states(state, &mut next);
                for s in &next {
                    assert!(
                        model.check_invariants(s).is_ok(),
                        "Invariant violated at depth {depth}: {s:?}"
                    );
                }
                next_frontier.extend(next);
            }
            all_states.extend(next_frontier.clone());
            frontier = next_frontier;
        }
        assert!(
            all_states.len() > 3,
            "should explore multiple states, got {}",
            all_states.len()
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Test: nested function access in guards (f[x][y].field).
    #[test]
    fn nested_function_field_access_guard() {
        let tmp = std::env::temp_dir().join("tlapp-nested-access-guard-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        let module = tmp.join("NestedAccess.tla");
        fs::write(
            &module,
            r#"
---- MODULE NestedAccess ----
EXTENDS Naturals
CONSTANTS Nodes
VARIABLES state

Init == state = [n \in Nodes |-> [status |-> "idle", count |-> 0]]

Activate(n) ==
    /\ state[n].status = "idle"
    /\ state[n].count < 3
    /\ state' = [state EXCEPT ![n] = [status |-> "active", count |-> @ .count + 1]]

Deactivate(n) ==
    /\ state[n].status = "active"
    /\ state' = [state EXCEPT ![n] = [@ EXCEPT !.status = "idle"]]

Next == \E n \in Nodes : Activate(n) \/ Deactivate(n)

\* Count never exceeds 3
CountBounded == \A n \in Nodes : state[n].count <= 3

\* Active nodes must have count > 0
ActiveImpliesPositiveCount ==
    \A n \in Nodes : state[n].status = "active" => state[n].count > 0

Spec == Init /\ [][Next]_state
====
"#,
        )
        .expect("write module");

        let cfg = tmp.join("NestedAccess.cfg");
        fs::write(
            &cfg,
            "SPECIFICATION Spec\nCONSTANTS Nodes = {a, b}\nINVARIANT CountBounded\nINVARIANT ActiveImpliesPositiveCount\n",
        )
        .expect("write cfg");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1);

        // Explore 4 levels and verify invariants
        let mut frontier = init.clone();
        for depth in 0..4 {
            let mut next_frontier = Vec::new();
            for state in &frontier {
                let mut next = Vec::new();
                model.next_states(state, &mut next);
                for s in &next {
                    assert!(
                        model.check_invariants(s).is_ok(),
                        "Invariant violated at depth {depth}: {s:?}"
                    );
                }
                next_frontier.extend(next);
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Test: negated existential as guard (~\E x \in S : P(x)).
    #[test]
    fn negated_existential_guard() {
        let tmp = std::env::temp_dir().join("tlapp-negated-exists-guard-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        let module = tmp.join("NegatedExists.tla");
        fs::write(
            &module,
            r#"
---- MODULE NegatedExists ----
EXTENDS Naturals, FiniteSets
CONSTANTS Items
VARIABLES taken

Init == taken = {}

Take(item) ==
    /\ item \notin taken
    /\ ~\E i \in taken : i = item   \* Redundant but tests ~\E guard
    /\ taken' = taken \union {item}

Next == \E i \in Items : Take(i)

\* No duplicates (enforced by set semantics + guard)
NoDuplicates == Cardinality(taken) <= Cardinality(Items)

Spec == Init /\ [][Next]_taken
====
"#,
        )
        .expect("write module");

        let cfg = tmp.join("NegatedExists.cfg");
        fs::write(
            &cfg,
            "SPECIFICATION Spec\nCONSTANTS Items = {x, y, z}\nINVARIANT NoDuplicates\n",
        )
        .expect("write cfg");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();

        // From empty set, should be able to take 3 items
        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert_eq!(next.len(), 3, "should have 3 successors from empty taken");

        // From {x}, should be able to take y and z but NOT x again
        let one_taken = next
            .iter()
            .find(|s| {
                if let Some(TlaValue::Set(set)) = s.get("taken") {
                    set.len() == 1
                } else {
                    false
                }
            })
            .expect("should have a state with 1 item taken");

        let mut next2 = Vec::new();
        model.next_states(one_taken, &mut next2);
        assert_eq!(
            next2.len(),
            2,
            "should have 2 successors (can't retake same item): {next2:?}"
        );

        for s in &next2 {
            assert!(
                model.check_invariants(s).is_ok(),
                "Invariant violated: {s:?}"
            );
        }

        let _ = fs::remove_dir_all(&tmp);
    }
}
