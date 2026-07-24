use crate::fairness::{FairnessConstraint, LabeledTransition};
use crate::tla::hashed_arc::HashedArc;
use crate::tla::compiled_expr::{resolve_state_vars, resolve_state_vars_in_action_ir};
use crate::model::Model;
use crate::symmetry::{SymmetrySpec, canonicalize_tla_state};
use crate::tla::module::TlaModuleInstance;
#[cfg(test)]
use crate::tla::tla_state;
use crate::tla::{
    ClauseKind, CompiledActionIr, CompiledExpr, ConfigValue, EvalContext, PorAnalysis,
    TemporalFormula, TlaConfig, TlaDefinition, TlaModule, TlaState, TlaValue, classify_clause,
    compile_action_ir, compile_expr, count_next_disjuncts, eval_action_body_multi,
    eval_action_constraint, eval_compiled, eval_predicate,
    evaluate_next_states_labeled_with_instances,
    evaluate_next_states_per_disjunct, evaluate_next_states_swarm,
    evaluate_next_states_with_instances, insert_compiled_action,
    looks_like_action, normalize_operator_ref_name, parse_tla_config, parse_tla_module_file,
    split_top_level,
};
use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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
    /// True if the behaviour `SPECIFICATION` conjoins a *non-fairness* temporal
    /// assumption (`[]<>R`, `<>R`, `~>`, `<>[]`, …) beyond `Init /\ [][Next]_vars`
    /// and `WF_`/`SF_` fairness. Such an assumption can exclude the
    /// stutter-forever suffix, so the no-fairness stutter-reachability path in
    /// `graph_liveness_violation` would be UNSOUND (could false-violate). When
    /// set, that path is skipped (conservative: miss rather than false-flag).
    /// INIT/NEXT configs have no SPECIFICATION formula → always `false` → safe.
    /// (Guardrail from a temporal-logic correctness review — see `(d)` in the
    /// RealTime/LTL design notes.)
    pub spec_has_non_fairness_liveness: bool,
    pub state_constraints: Vec<(String, String)>,
    pub action_constraints: Vec<(String, String)>,
    pub symmetry: Option<SymmetrySpec>,
    pub view: Option<String>,
    pub initial_states_vec: Vec<TlaState>,
    /// If true, [`Model::initial_states_streaming`] re-runs Init enumeration in
    /// a background thread instead of returning `initial_states_vec` directly.
    ///
    /// This is the T5.4 streaming path: the runtime spawns its own producer
    /// thread that pulls from the iterator and pushes to the global queue, so
    /// workers can begin invariant evaluation on partially-enumerated Init.
    /// For Init predicates whose enumeration dominates wall time (Einstein-class
    /// puzzles), this cuts time-to-first-invariant-check from "Init-eval-time"
    /// to "first-state-arrival-time".
    ///
    /// Defaults to `false` for back-compat. Toggle via
    /// [`TlaModel::with_streaming_init`].
    pub streaming_init: bool,
    /// Pre-compiled action definitions for fast execution
    pub compiled_actions: BTreeMap<String, Arc<CompiledActionIr>>,
    /// Pre-compiled invariant expressions (name, compiled expr)
    pub compiled_invariants: Vec<(String, Arc<CompiledExpr>)>,
    /// Pre-compiled state constraint expressions (name, compiled expr)
    pub compiled_state_constraints: Vec<(String, Arc<CompiledExpr>)>,
    /// Whether to allow deadlocked states (no successors) without error.
    /// Equivalent to TLC's -deadlock flag.
    pub allow_deadlock: bool,
    /// True if Next is trivially UNCHANGED vars (no transitions).
    /// When set, next_states() returns empty immediately.
    trivial_next: bool,
    /// Partial-order reduction (POR) analysis, populated only when POR is
    /// enabled.  When `Some`, `next_states()` computes a stubborn-set subset
    /// of disjuncts to fire instead of all enabled disjuncts.
    ///
    /// POR in this implementation preserves only safety properties; it is
    /// disabled automatically when fairness or liveness checking is active.
    pub por_analysis: Option<Arc<PorAnalysis>>,
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

        // Resolve invariants and trivial-Next FIRST so the T5.5 joint
        // Init+invariant solver can use them. (Both are pure analyses over
        // already-parsed module structure; cheap to do up-front.)
        let mut invariant_exprs = resolve_invariant_exprs(&module, &config);
        let trivial_next = module
            .definitions
            .get(&next_name)
            .map(|def| {
                let body = def.body.trim();
                body.starts_with("UNCHANGED") || body == "FALSE" || body == "TRUE /\\ FALSE"
            })
            .unwrap_or(false);

        // T5.5 — joint Init+invariant symbolic encoding. Only attempt when
        // Next is trivial UNCHANGED (so a no-violation result is a complete
        // proof). For Einstein-class specs this returns a single witness
        // initial state in <1s instead of enumerating ~199M cross-product
        // states. Falls back to brute-force if the shape doesn't fit.
        let joint_solved: Option<Vec<TlaState>> = if trivial_next && !invariant_exprs.is_empty() {
            try_joint_init_invariant_solve(&module, &config, &init_name, &invariant_exprs)
        } else {
            None
        };

        let mut initial_states_vec = match joint_solved {
            Some(states) => states,
            None => evaluate_init_states(&module, &config, &init_name)?,
        };
        // Constants belong to the fixed model context, not the mutable state:
        // in TLA+/TLC a state is a valuation of the declared VARIABLES only.
        // Init evaluation seeds constants into each state so Init predicates can
        // read them (e.g. `x \in Node`), but we must NOT carry them in the
        // fingerprinted state. They are identical across every reachable state
        // (so removing them cannot change any distinct-state count) yet they
        // bloat every fingerprint (bincode-serialized + hashed per state) and
        // every stored state. Strip them here — centrally, on the already-
        // materialized initial states — so this one filter covers the
        // brute-force, joint-symbolic (T5.5) and streaming Init paths at once
        // (evaluate_init_states itself has several early-return paths that each
        // seed constants). Constants stay resolvable during action/invariant
        // eval via their injected definitions (non-model-value constants) or via
        // the `ModelValue(name)` identifier fallback (model values, whose self-
        // referential def is intentionally NOT injected — see
        // inject_constants_into_module_tree). Because the active schema below is
        // derived from these (now variable-only) states, constant references
        // compile to name-based `Var` rather than per-state `StateVar { slot }`.
        {
            let var_set: std::collections::HashSet<&str> =
                module.variables.iter().map(|s| s.as_str()).collect();
            initial_states_vec = initial_states_vec
                .into_iter()
                .map(|state| {
                    state
                        .into_iter()
                        .filter(|(k, _)| var_set.contains(k.as_ref()))
                        .collect()
                })
                .collect();
        }
        if let Some(first_state) = initial_states_vec.first() {
            let names: Vec<Arc<str>> = first_state.keys().cloned().collect();
            crate::tla::value::set_active_schema(names);
            initial_states_vec = initial_states_vec
                .into_iter()
                .map(|state| state.into_iter().collect())
                .collect();
        } else {
            crate::tla::value::clear_active_schema();
        }
        let temporal_properties = resolve_temporal_properties(&module, &config)?;

        // Lower box-safety temporal properties (`[] P` where P is a pure
        // state predicate) into per-state invariants.  Without this, a
        // property like `AC1 == [] SomeStatePred` declared under PROPERTIES
        // parses and classifies as safety but is NEVER evaluated per state,
        // so we would report SAFE on specs TLC flags as VIOLATED (a
        // missed-violation soundness bug).  See extract_box_safety_invariant
        // for the conservative guardrail on what is safe to lower.
        for (prop_name, formula) in &temporal_properties {
            if let Some(pred_text) = extract_box_safety_invariant(formula) {
                if std::env::var("TLAPP_TRACE_INVARIANT").is_ok() {
                    eprintln!(
                        "Lowering box-safety property '{}' into a per-state invariant: {}",
                        prop_name, pred_text
                    );
                }
                invariant_exprs.push((prop_name.clone(), pred_text));
            }
        }

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

        // (d) guardrail for the no-fairness graph-liveness check: detect a
        // non-fairness temporal assumption conjoined into the behaviour
        // SPECIFICATION. Stutter-reachability for `[](P => <>[]Q)` is only
        // sound when the spec is exactly `Init /\ [][Next]_vars` (stuttering is
        // always a legal extension). A `SPECIFICATION` that conjoins e.g.
        // `[]<>R` excludes the stutter suffix; INIT/NEXT configs (no Spec
        // formula) never do. WF_/SF_ conjuncts are already extracted as
        // fairness_constraints and route to the fairness path, so they don't
        // trip this flag on their own.
        let spec_has_non_fairness_liveness = config
            .specification
            .as_ref()
            .and_then(|name| module.definitions.get(name))
            .and_then(|def| TemporalFormula::parse(&def.body).ok())
            .map(|f| formula_has_non_fairness_liveness(&f))
            .unwrap_or(false);

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

        // (trivial_next was computed earlier so the T5.5 joint solver could
        // use it.)
        if trivial_next {
            eprintln!("Note: Next is UNCHANGED — skipping state exploration");
        }

        Ok(Self {
            module,
            config,
            init_name,
            next_name,
            invariant_exprs,
            temporal_properties,
            fairness_constraints,
            spec_has_non_fairness_liveness,
            state_constraints,
            action_constraints,
            symmetry,
            view,
            initial_states_vec,
            compiled_actions,
            compiled_invariants,
            compiled_state_constraints,
            allow_deadlock,
            trivial_next,
            por_analysis: None,
            streaming_init: false,
        })
    }

    /// Enable the T5.4 streaming Init path. When set, `initial_states_streaming`
    /// re-runs Init enumeration in a producer thread instead of returning the
    /// pre-computed `initial_states_vec`. The runtime then begins invariant
    /// checking on initial states as they arrive, instead of waiting for Init
    /// enumeration to complete.
    ///
    /// Note: with the current implementation, Init evaluation itself is still
    /// synchronous within the producer thread — true incremental yielding from
    /// the symbolic SMT loop is tracked as T5.5. The producer-thread shape is
    /// the architectural prerequisite.
    pub fn with_streaming_init(mut self, on: bool) -> Self {
        self.streaming_init = on;
        self
    }

    /// Enable partial-order reduction.
    ///
    /// Pure-safety mode: when no fairness/liveness constraints are present,
    /// `next_states()` switches to firing only a stubborn-set subset of
    /// enabled `Next` disjuncts.
    ///
    /// **Liveness mode (T7.3)**: when fairness or temporal properties are
    /// present, POR is enabled with the Peled (1994) visible-action
    /// proviso — every enabled visible action is always included in the
    /// stubborn set.  Visible actions are: (a) actions named in WF/SF
    /// clauses, and (b) actions whose footprint touches any variable used
    /// by a temporal state predicate.  This preserves stutter-equivalent
    /// LTL\X over visible actions, including WF/SF on those actions.
    ///
    /// **Caveat**: the BFS cycle-proviso (Bošnački & Holzmann 2007) is
    /// not yet implemented, which means the reduced graph could in
    /// principle contain a cycle of invisible actions that starves a fair
    /// action.  In practice, the visibility proviso is sufficient when
    /// every fair action is also a top-level Next disjunct (the common
    /// case).  See `RELEASE_1.0.0_LOG.md` for the full proviso roadmap.
    pub fn enable_por(&mut self) -> anyhow::Result<()> {
        let next_def = self
            .module
            .definitions
            .get(&self.next_name)
            .ok_or_else(|| anyhow!("Next definition '{}' not found", self.next_name))?;

        // Collect visible information for the Peled proviso.
        let mut visible_action_names: Vec<String> = Vec::new();
        let mut visible_vars: BTreeSet<String> = BTreeSet::new();
        for fc in &self.fairness_constraints {
            let action = match fc {
                FairnessConstraint::Weak { action, .. } => action,
                FairnessConstraint::Strong { action, .. } => action,
            };
            visible_action_names.push(action.clone());
            // Also mark every variable touched by the action.  If the
            // action name resolves to a definition, walk its body and mark
            // all `var` and `var'` references.
            if let Some(def) = self.module.definitions.get(action) {
                collect_referenced_vars(&def.body, &self.module, &mut visible_vars);
            }
        }
        // Add variables mentioned in any temporal state predicate (the
        // arguments of <>, [], leads-to, and Eventually/Always sub-trees).
        for (_name, formula) in &self.temporal_properties {
            collect_temporal_vars(formula, &self.module, &mut visible_vars);
        }

        let analysis = PorAnalysis::from_next_with_visibility(
            &next_def.body,
            &self.module,
            &visible_action_names,
            &visible_vars,
        );

        let liveness_mode =
            !self.fairness_constraints.is_empty() || self.has_liveness_properties();

        if std::env::var("TLAPP_POR_VERBOSE").is_ok() {
            eprintln!(
                "POR analysis: {} actions, {} variables, {} visible{}\n{}",
                analysis.num_actions(),
                analysis.variables.len(),
                analysis.visible.len(),
                if liveness_mode {
                    " [liveness mode]"
                } else {
                    " [safety mode]"
                },
                analysis.describe()
            );
        }

        // Sanity check: in liveness mode the visibility set must not be
        // empty (else fairness would be unprovably ignored).  Conversely,
        // in safety mode visibility may be empty (and that's the T7
        // baseline behaviour).
        if liveness_mode && analysis.visible.is_empty() {
            return Err(anyhow!(
                "POR with liveness/fairness requested but no visible actions \
                 could be identified — refusing to risk silent fairness loss. \
                 Either name fair actions as top-level Next disjuncts or \
                 disable POR for this spec."
            ));
        }

        self.por_analysis = Some(Arc::new(analysis));
        Ok(())
    }
}

/// Walk an expression body, collect every state-variable identifier (with
/// or without prime) it references.  Used by the visibility heuristic to
/// determine which variables a fairness action touches.
fn collect_referenced_vars(
    body: &str,
    module: &TlaModule,
    out: &mut BTreeSet<String>,
) {
    let bytes = body.as_bytes();
    let mut i = 0usize;
    let var_set: BTreeSet<&str> = module.variables.iter().map(|s| s.as_str()).collect();
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let name = &body[start..i];
            if var_set.contains(name) {
                out.insert(name.to_string());
            }
        } else {
            i += 1;
        }
    }
}

/// Walk a temporal formula and collect every state variable referenced by
/// any embedded state predicate (the argument of <>, [], ~>, etc.).
fn collect_temporal_vars(
    formula: &TemporalFormula,
    module: &TlaModule,
    out: &mut BTreeSet<String>,
) {
    match formula {
        TemporalFormula::StatePredicate(expr) => {
            collect_referenced_vars(expr, module, out);
        }
        TemporalFormula::Always(inner)
        | TemporalFormula::Eventually(inner)
        | TemporalFormula::InfinitelyOften(inner)
        | TemporalFormula::EventuallyAlways(inner)
        | TemporalFormula::Not(inner) => collect_temporal_vars(inner, module, out),
        TemporalFormula::And(a, b)
        | TemporalFormula::Or(a, b)
        | TemporalFormula::Implies(a, b)
        | TemporalFormula::LeadsTo(a, b) => {
            collect_temporal_vars(a, module, out);
            collect_temporal_vars(b, module, out);
        }
        TemporalFormula::TemporalForAll { formula: inner, .. }
        | TemporalFormula::TemporalExists { formula: inner, .. } => {
            collect_temporal_vars(inner, module, out)
        }
        TemporalFormula::WeakFairness { vars, .. }
        | TemporalFormula::StrongFairness { vars, .. } => {
            for v in vars {
                if module.variables.iter().any(|m| m == v) {
                    out.insert(v.clone());
                }
            }
        }
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
        // Mark committed next-state generation: a reached `Assert(FALSE)` in the
        // action is a safety violation and is recorded on the reached-assertion
        // side channel (drained by the worker), rather than being masked as a
        // disabled branch. See crate::model.
        let _committed = crate::model::enter_committed_next_state();

        // Optimization: if Next is trivially UNCHANGED vars, skip evaluation.
        // Every initial state is a fixed point — no transitions to explore.
        if self.trivial_next {
            return;
        }

        // Partial-order reduction path: compute per-disjunct successors,
        // then fire only the stubborn-set subset.
        if let Some(ref analysis) = self.por_analysis {
            self.next_states_por(state, analysis.as_ref(), out);
            return;
        }

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
                // A reached Assert(FALSE) recorded a pending violation on the
                // side channel (see crate::model). Return with no successors so
                // the worker drains and reports it as a safety violation — do NOT
                // panic, and do so regardless of `allow_deadlock`.
                if crate::model::has_pending_assertion_violation() {
                    return;
                }
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
        use crate::model::fingerprint_hasher;
        use std::hash::Hasher;

        // Symmetry canonicalization is handled by canonicalize() in the
        // runtime before fingerprinting, so we fingerprint the state directly.

        // If view function is defined, fingerprint only the view
        if self.view.is_some() {
            match self.evaluate_view(state) {
                Ok(view_value) => {
                    if let Ok(bytes) = bincode::serialize(&view_value) {
                        let mut hasher = fingerprint_hasher();
                        hasher.write(&bytes);
                        return hasher.finish();
                    }
                }
                Err(_) => {
                    // View evaluation failed - fall back to full state fingerprint
                }
            }
        }

        // No view or view failed - hash the full state using serialization.
        //
        // Normalize 1..n-domain functions to their Seq form FOR THE HASH ONLY
        // (a TLA+ sequence IS a function over 1..Len). A value built as
        // `[i \in 1..n |-> e]` and the same value built via `Append`/`<<...>>`
        // are distinct `TlaValue` variants and otherwise serialize differently,
        // so logically-equal states would fail to dedup — inflating the explored
        // state space (MCCheckpointCoordination explored ~8x more states than
        // TLC because logs (`Log == Seq(...)`) are built both ways). We hash the
        // normalized form but the runtime keeps exploring the original state, so
        // invariant evaluation is untouched (normalizing the *stored* state can
        // expose ops that treat Seq vs Function inconsistently and yield false
        // violations — this avoids that entirely). Only allocates when a
        // 1..n-domain function is actually present.
        let mut hasher = fingerprint_hasher();
        let bytes = match crate::tla::value::normalize_state_if_changed(state) {
            Some(normalized) => bincode::serialize(&normalized),
            None => bincode::serialize(state),
        };
        if let Ok(bytes) = bytes {
            hasher.write(&bytes);
        }
        hasher.finish()
    }

    fn next_states_labeled(
        &self,
        state: &Self::State,
    ) -> Option<Vec<crate::model::LabeledTransition<Self::State>>> {
        // Produce labeled transitions when the runtime needs the transition
        // graph for EITHER the fairness SCC pass OR the graph-level
        // `[](P => <>[]Q)` liveness check (which can hold with no fairness
        // constraints). Action labels are still emitted in the graph-liveness
        // case (harmless; the graph-liveness check only uses the adjacency).
        if !self.has_fairness_constraints() && !self.needs_graph_liveness_check() {
            return None;
        }

        // Committed next-state generation (see next_states): a reached
        // Assert(FALSE) here is recorded on the reached-assertion side channel.
        let _committed = crate::model::enter_committed_next_state();

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

    fn should_check_fairness(&self) -> bool {
        // Fairness constraints are only *checkable* against a declared
        // liveness property. With fairness but no liveness property (the
        // common `SPECIFICATION Spec` + `INVARIANTS ...` shape, e.g.
        // transaction_commit/2PCwithBTM), TLC checks only the invariants
        // and never reports a fairness/liveness violation — so neither
        // should we. Requiring a liveness property here also avoids
        // collecting the labeled-transition graph for safety-only runs.
        !self.fairness_constraints.is_empty() && self.has_liveness_properties()
    }

    fn fairness_constraints(&self) -> Vec<FairnessConstraint> {
        self.fairness_constraints.clone()
    }

    fn next_action_name(&self) -> Option<&str> {
        Some(&self.next_name)
    }

    fn scc_violates_liveness_property(&self, scc_states: &[Self::State]) -> Option<bool> {
        if self.temporal_properties.is_empty() || scc_states.is_empty() {
            return None;
        }

        let mut any_evaluable = false;
        for (_name, formula) in &self.temporal_properties {
            match self.scc_violates_single_property(formula, scc_states) {
                Some(true) => return Some(true), // a property definitively fails
                Some(false) => any_evaluable = true,
                None => {} // unsupported / unevaluable shape — skip it
            }
        }

        if any_evaluable {
            // At least one property could be evaluated and NONE of the
            // evaluable ones is violated by this SCC → suppress the
            // fairness-only "violation".
            Some(false)
        } else {
            // No property was evaluable (e.g. only a refinement `TD!Spec`
            // property). Fall back to the fairness-only verdict.
            None
        }
    }

    fn needs_graph_liveness_check(&self) -> bool {
        self.temporal_properties
            .iter()
            .any(|(_, f)| graph_liveness_shape(f).is_some())
    }

    fn graph_liveness_violation(
        &self,
        adjacency_fp: &HashMap<u64, Vec<u64>>,
        state_by_fp: &HashMap<u64, Self::State>,
        non_trivial_sccs: &[Vec<u64>],
        fair_scc: &[bool],
    ) -> Option<Self::State> {
        // For each `[](P => <>[]Q)` property, look for a genuine violation:
        // a P-state that can REACH a bad cycle (a reachable FAIR non-trivial
        // cycle containing a `¬Q` state). Because every state in the graph is
        // reachable from Init, a P-state that reaches such a cycle witnesses a
        // behaviour that satisfies P but for which `<>[]Q` fails.
        let has_fairness = !self.fairness_constraints.is_empty();

        for (_name, formula) in &self.temporal_properties {
            let Some((p_text, q_text)) = graph_liveness_shape(formula) else {
                continue;
            };

            // 1. Identify "bad cycle" fingerprints — states from which a
            //    behaviour can loop forever while `Q` is always false (so
            //    `<>[]Q` fails).
            //
            //    * NO fairness: stuttering (`[Next]_vars` allows `vars'=vars`)
            //      is always a legal infinite behaviour, so ANY reachable
            //      `¬Q` state is a bad cycle — you can stutter on it forever.
            //      Stutter self-loops are NOT materialised as graph edges, so
            //      we must NOT restrict to SCC membership here (RealTime: the
            //      bad `now=4` loop is a stutter self-loop, not a recorded
            //      graph cycle). This is the RealTime case.
            //
            //    * WITH fairness: a stutter-forever behaviour is unfair and
            //      TLC excludes it, so a bad cycle must be a genuine FAIR
            //      non-trivial graph cycle (SCC) containing a `¬Q` state.
            //      Restricting to fair SCCs is the guardrail that prevents
            //      false violations on fair liveness specs.
            //
            //    In both cases we require `Q` to be *confidently* false
            //    (Some(false)); unevaluable states never establish ¬Q, so an
            //    uncheckable predicate can only miss a violation, never invent
            //    one.
            let mut bad_cycle_fps: HashSet<u64> = HashSet::new();
            if has_fairness {
                for (idx, scc) in non_trivial_sccs.iter().enumerate() {
                    if !fair_scc.get(idx).copied().unwrap_or(true) {
                        continue; // unfair cycle — TLC excludes it
                    }
                    let has_not_q = scc.iter().any(|fp| {
                        state_by_fp
                            .get(fp)
                            .map(|st| self.eval_state_pred_on_state(q_text, st) == Some(false))
                            .unwrap_or(false)
                    });
                    if has_not_q {
                        for fp in scc {
                            bad_cycle_fps.insert(*fp);
                        }
                    }
                }
            } else {
                // No fairness: every reachable ¬Q state can be stuttered on
                // forever → each is a bad cycle. BUT this stutter-reachability
                // is only sound when the spec is exactly `Init /\ [][Next]_vars`.
                // If the SPECIFICATION conjoins a non-fairness temporal
                // assumption (`[]<>R`, `<>R`, `~>`, …) it excludes the
                // stutter-forever suffix, so firing here could false-violate —
                // leave the property unchecked instead ((d) guardrail).
                if self.spec_has_non_fairness_liveness {
                    continue;
                }
                for (&fp, st) in state_by_fp.iter() {
                    if self.eval_state_pred_on_state(q_text, st) == Some(false) {
                        bad_cycle_fps.insert(fp);
                    }
                }
            }

            if bad_cycle_fps.is_empty() {
                continue;
            }

            // 2. Backward-reachability: set of fingerprints that can REACH any
            //    bad-cycle fingerprint. Build the reverse adjacency once and
            //    BFS from the bad-cycle set.
            let mut reverse: HashMap<u64, Vec<u64>> =
                HashMap::with_capacity(adjacency_fp.len());
            for (&from, succs) in adjacency_fp.iter() {
                for &to in succs {
                    reverse.entry(to).or_default().push(from);
                }
            }
            let mut can_reach_bad: HashSet<u64> = HashSet::new();
            let mut stack: Vec<u64> = bad_cycle_fps.iter().copied().collect();
            for &fp in &stack {
                can_reach_bad.insert(fp);
            }
            while let Some(fp) = stack.pop() {
                if let Some(preds) = reverse.get(&fp) {
                    for &pred in preds {
                        if can_reach_bad.insert(pred) {
                            stack.push(pred);
                        }
                    }
                }
            }

            // 3. A violation exists iff some reachable P-state can reach a bad
            //    cycle. Return the P-state as the counterexample witness.
            for (&fp, st) in state_by_fp.iter() {
                if can_reach_bad.contains(&fp)
                    && self.eval_state_pred_on_state(p_text, st) == Some(true)
                {
                    return Some(st.clone());
                }
            }
        }
        None
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
            eval_predicate(view_expr, &ctx).map_err(|e| format!("view evaluation failed: {}", e))
        } else {
            // No view defined - return full state as a value
            // Convert state to TlaValue::Record
            let record: BTreeMap<String, TlaValue> = state
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            Ok(TlaValue::Record(HashedArc::new(record)))
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

    /// Evaluate a state-predicate text on a single state, returning
    /// `Some(bool)` on a clean boolean result and `None` if the predicate
    /// cannot be evaluated to a boolean (unknown operator, type error, ...).
    fn eval_state_pred_on_state(&self, pred_text: &str, state: &TlaState) -> Option<bool> {
        let ctx = EvalContext::with_definitions_and_instances(
            state,
            &self.module.definitions,
            &self.module.instances,
        );
        match eval_predicate(pred_text, &ctx) {
            Ok(TlaValue::Bool(b)) => Some(b),
            _ => None,
        }
    }

    /// Decide whether a single temporal property is violated by the states of
    /// a fairness-unfair SCC. This is what turns "the SCC is unfair" into "the
    /// SCC is (or is not) an actual counterexample to a declared property".
    ///
    /// An SCC is a strongly-connected cycle revisited infinitely often, so a
    /// behaviour can loop through exactly its states forever. We use that to
    /// evaluate the recurrent-cycle semantics of the common property shapes:
    ///
    ///   * `<>P` (Eventually) / `[]<>P` (InfinitelyOften): the cycle violates
    ///     it iff `P` holds at NO state of the SCC — an infinite loop through
    ///     the SCC never makes `P` true.
    ///   * `P ~> Q` (LeadsTo): violated iff some SCC state satisfies `P` but NO
    ///     SCC state satisfies `Q` — `P` is triggered yet `Q` is never reached
    ///     within the recurrent cycle. (If `P` never holds in the SCC, the
    ///     leads-to is vacuously satisfied — this is exactly the EWD840 case:
    ///     `terminated` never holds in the environment-only cycle.)
    ///   * `<>[]P` (EventuallyAlways): violated iff some SCC state has `¬P` —
    ///     the cycle keeps toggling `P` off, so `P` can never become always-true.
    ///
    /// Returns `Some(true)`/`Some(false)` when the shape is evaluable, and
    /// `None` for shapes we do not check here (refinement `TD!Spec`, nested
    /// temporal, temporal quantifiers, fairness terms), so the caller can fall
    /// back conservatively.
    fn scc_violates_single_property(
        &self,
        formula: &TemporalFormula,
        scc_states: &[TlaState],
    ) -> Option<bool> {
        // A state-predicate text that holds at *some* / *no* SCC state.
        let holds_somewhere = |pred: &str| -> Option<bool> {
            let mut saw_unevaluable = false;
            let mut any_true = false;
            for st in scc_states {
                match self.eval_state_pred_on_state(pred, st) {
                    Some(true) => any_true = true,
                    Some(false) => {}
                    None => saw_unevaluable = true,
                }
            }
            if any_true {
                Some(true)
            } else if saw_unevaluable {
                // Could not confidently establish "P holds nowhere".
                None
            } else {
                Some(false)
            }
        };

        match formula {
            TemporalFormula::Eventually(inner) | TemporalFormula::InfinitelyOften(inner) => {
                let TemporalFormula::StatePredicate(p) = inner.as_ref() else {
                    return None;
                };
                // Violated iff P holds nowhere in the SCC.
                holds_somewhere(p).map(|somewhere| !somewhere)
            }
            TemporalFormula::LeadsTo(p, q) => {
                let (TemporalFormula::StatePredicate(p), TemporalFormula::StatePredicate(q)) =
                    (p.as_ref(), q.as_ref())
                else {
                    return None;
                };
                let p_somewhere = holds_somewhere(p)?;
                if !p_somewhere {
                    // P never triggers in the cycle → vacuously satisfied.
                    return Some(false);
                }
                let q_somewhere = holds_somewhere(q)?;
                // Violated iff P triggers but Q is never reached in the cycle.
                Some(!q_somewhere)
            }
            TemporalFormula::EventuallyAlways(inner) => {
                let TemporalFormula::StatePredicate(p) = inner.as_ref() else {
                    return None;
                };
                // Violated iff some SCC state has ¬P (P can't stay always-true).
                let mut saw_false = false;
                let mut saw_unevaluable = false;
                for st in scc_states {
                    match self.eval_state_pred_on_state(p, st) {
                        Some(true) => {}
                        Some(false) => saw_false = true,
                        None => saw_unevaluable = true,
                    }
                }
                if saw_false {
                    Some(true)
                } else if saw_unevaluable {
                    None
                } else {
                    Some(false)
                }
            }
            // Temporal implication `P => Q` (e.g. `[]P => <>Q`, or the
            // `[](P => <>[]Q)` inner). The per-SCC fairness check CANNOT judge
            // these — the real check is the graph-level `graph_liveness_violation`
            // pass (for the `<>[]` consequent shape) or nothing at all (for
            // other shapes, which we deliberately leave unchecked). Crucially,
            // we must return `Some(false)` here rather than `None`: a declared
            // `Implies` liveness property makes `has_liveness_properties()` true,
            // which enables the fairness SCC pass; if this returned `None` the
            // caller would fall through to a fairness-ONLY verdict and flag an
            // unfair SCC as a violation TLC does not report (e.g. nbacg_guer01's
            // `[]P => <>Q` termination property with `WF_vars`). Returning
            // `Some(false)` says "this implication is not violated per-SCC", so
            // the spurious fairness-only violation is suppressed. The genuine
            // `[](P => <>[]Q)` violation is still caught by the graph pass.
            TemporalFormula::Implies(_, _) => Some(false),
            // Unsupported shapes: let the caller fall back.
            _ => None,
        }
    }

    /// Compute next states under partial-order reduction.
    ///
    /// Strategy:
    ///   1. Evaluate every disjunct of `Next` independently (gives both
    ///      successor sets and per-disjunct enabledness).
    ///   2. Ask the POR analysis for a stubborn-set subset of enabled
    ///      disjuncts.
    ///   3. Concatenate successors from only the stubborn-set disjuncts
    ///      into `out`.
    ///
    /// When the stubborn set equals the full enabled set we get no
    /// reduction at all — that's correct, just no win.
    pub(crate) fn next_states_por(
        &self,
        state: &TlaState,
        analysis: &PorAnalysis,
        out: &mut Vec<TlaState>,
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

        let n = analysis.num_actions();
        if n == 0 {
            return;
        }

        // T7.1: batched per-disjunct evaluation — split `next_body` once
        // and evaluate every disjunct in a single pass.  Replaces N calls
        // to `evaluate_next_states_swarm(.., &[idx])` (each of which
        // re-split the body).  On shared-state specs (Pipeline) this lifts
        // POR from net-slower to net-faster.
        let mut per_disjunct = evaluate_next_states_per_disjunct(
            &next_def.body,
            &self.module.definitions,
            instances,
            state,
        );

        // Defensive: if the splitter returned a different count than the
        // POR analysis expected, fall back to firing every successor it
        // produced (always correct, no reduction).
        if per_disjunct.len() != n {
            for succs in per_disjunct {
                out.extend(succs);
            }
            return;
        }

        let enabled: Vec<bool> = per_disjunct.iter().map(|s| !s.is_empty()).collect();
        let stubborn = analysis.stubborn_set(&enabled);
        for idx in stubborn {
            let succs = std::mem::take(&mut per_disjunct[idx]);
            out.extend(succs);
        }
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

/// Returns true if `text` is a pure, non-temporal, prime-free state predicate
/// that is safe to evaluate as a per-state invariant.
///
/// CONSERVATIVE GUARDRAIL (soundness-critical): we must never lower an action
/// formula (`[A]_v`, contains primes) or a nested temporal formula as a
/// per-state invariant — evaluating primes as a state predicate would either
/// crash or produce false violations, and re-checking a temporal operator per
/// state is meaningless. When in doubt, reject: leaving a property unchecked
/// is a missed-violation (bad), but wrongly lowering an action/temporal
/// formula is a crash/false-violation (worse). We only ever return true for
/// text we are confident is a plain state predicate.
fn is_pure_state_predicate_text(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    // Reject `[A]_v` action forms and any leading bracket construct.
    if t.starts_with('[') {
        return false;
    }
    // Reject primed variables (action formulas). A lone `'` anywhere in the
    // predicate text means it references a next-state value.
    if t.contains('\'') {
        return false;
    }
    // Reject nested temporal operators.
    for pat in ["[]", "<>", "~>", "WF_", "SF_", "\\AA", "\\EE"] {
        if t.contains(pat) {
            return false;
        }
    }
    true
}

/// Given a temporal property formula, return `Some(state_predicate_text)` if it
/// is a box-safety property `[] P` whose body reduces to a pure state
/// predicate, suitable for lowering into a per-state invariant. Returns `None`
/// for anything else (including action/temporal bodies, per the guardrail in
/// [`is_pure_state_predicate_text`]).
fn extract_box_safety_invariant(formula: &TemporalFormula) -> Option<String> {
    match formula {
        TemporalFormula::Always(inner) => box_body_predicate_text(inner),
        _ => None,
    }
}

/// True if `formula` (a behaviour SPECIFICATION) contains a *non-fairness*
/// temporal-liveness conjunct — `<>P`, `[]<>P`, `<>[]P`, `P ~> Q`, or a temporal
/// `=>` — as opposed to `Init` (state predicate), `[][Next]_vars` / `[]Inv`
/// (`Always` of a state predicate — stutter-preserving safety), or `WF_`/`SF_`
/// fairness (which is extracted separately and routed to the fairness path).
///
/// Used to gate the no-fairness stutter-reachability path in
/// `graph_liveness_violation`: such an assumption can exclude the
/// stutter-forever suffix, making that path unsound. `WF_`/`SF_` return `false`
/// here on purpose (they are not "extra" temporal assumptions for this check).
fn formula_has_non_fairness_liveness(formula: &TemporalFormula) -> bool {
    match formula {
        // Fairness is handled by the fairness path — not an "extra" assumption.
        TemporalFormula::WeakFairness { .. } | TemporalFormula::StrongFairness { .. } => false,
        // Genuine non-fairness liveness operators.
        TemporalFormula::Eventually(_)
        | TemporalFormula::InfinitelyOften(_)
        | TemporalFormula::EventuallyAlways(_)
        | TemporalFormula::LeadsTo(_, _) => true,
        // Recurse structurally.
        TemporalFormula::And(a, b) | TemporalFormula::Or(a, b) => {
            formula_has_non_fairness_liveness(a) || formula_has_non_fairness_liveness(b)
        }
        TemporalFormula::Implies(a, b) => {
            formula_has_non_fairness_liveness(a) || formula_has_non_fairness_liveness(b)
        }
        TemporalFormula::Not(inner)
        | TemporalFormula::Always(inner)
        | TemporalFormula::TemporalForAll { formula: inner, .. }
        | TemporalFormula::TemporalExists { formula: inner, .. } => {
            // `[][Next]_vars` / `[]Inv` → Always(StatePredicate) → false (safe);
            // `[](P => <>[]Q)` as a spec conjunct → Always(Implies(.., temporal))
            // → recurse catches the temporal consequent.
            formula_has_non_fairness_liveness(inner)
        }
        TemporalFormula::StatePredicate(_) => false,
    }
}

/// Recognise the graph-structured liveness shape `[](P => <>[]Q)` where `P`
/// and `Q` are pure state predicates. Returns `Some((P_text, Q_text))` for
/// the exactly-supported shape, `None` otherwise.
///
/// This is the ONLY graph-level shape the post-processing check handles. It is
/// deliberately narrow: both operands must be pure, prime-free, non-temporal
/// state predicates so they can be evaluated per state. Anything else (nested
/// temporal, action formulas, `<>P` consequents without the `[]`) returns
/// `None` and is left unchecked (a missed violation is safer than a false one).
fn graph_liveness_shape(formula: &TemporalFormula) -> Option<(&str, &str)> {
    let TemporalFormula::Always(inner) = formula else {
        return None;
    };
    let TemporalFormula::Implies(ante, cons) = inner.as_ref() else {
        return None;
    };
    let TemporalFormula::StatePredicate(p) = ante.as_ref() else {
        return None;
    };
    let TemporalFormula::EventuallyAlways(q_inner) = cons.as_ref() else {
        return None;
    };
    let TemporalFormula::StatePredicate(q) = q_inner.as_ref() else {
        return None;
    };
    if is_pure_state_predicate_text(p) && is_pure_state_predicate_text(q) {
        Some((p.trim(), q.trim()))
    } else {
        None
    }
}

/// Reduce the body of an `Always(..)` to a single pure state-predicate text, if
/// possible. Handles the common shapes:
///   - `[] P`                       -> StatePredicate("P")
///   - `[] (P /\ Q)`                -> And of two state predicates
///   - `[] \AA i \in D : P`         -> TemporalForAll over a state predicate
fn box_body_predicate_text(inner: &TemporalFormula) -> Option<String> {
    match inner {
        TemporalFormula::StatePredicate(text) => {
            if is_pure_state_predicate_text(text) {
                Some(text.trim().to_string())
            } else {
                None
            }
        }
        TemporalFormula::And(left, right) => {
            let l = box_body_predicate_text(left)?;
            let r = box_body_predicate_text(right)?;
            Some(format!("({}) /\\ ({})", l, r))
        }
        TemporalFormula::TemporalForAll {
            var,
            domain,
            formula,
        } => {
            // `[] \AA i \in D : P` where P is a state predicate is equivalent to
            // the per-state invariant `\A i \in D : P`.
            let body = box_body_predicate_text(formula)?;
            Some(format!("\\A {} \\in {} : {}", var, domain, body))
        }
        _ => None,
    }
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
            // Check if this is an evaluation-only module (no state machine)
            // by looking for cfg with no INIT/NEXT/SPECIFICATION and module with no Init/Next
            if cfg.init.is_none()
                && cfg.next.is_none()
                && cfg.specification.is_none()
                && init == "Init"
            {
                // For eval-only modules, return empty init/next names.
                // The model will have 0 initial states and complete immediately.
                eprintln!("Note: evaluation-only module (no Init/Next/SPECIFICATION)");
                return Ok(("__EVAL_ONLY__".to_string(), "__EVAL_ONLY__".to_string()));
            }
            return Err(anyhow!("Init definition '{init}' not found in module"));
        }
    }

    if next != "__EVAL_ONLY__" && !module.definitions.contains_key(&next) {
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
                            // Prefer the chased Init over an inline expression
                            // that was likely just a side-effect (e.g., PrintT(R))
                            if sub_init.is_some() {
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

/// Find the position of a top-level `:` (not inside brackets/parens/braces).
fn find_top_level_colon(expr: &str) -> Option<usize> {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    for (i, c) in expr.char_indices() {
        match c {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            ':' if paren == 0 && bracket == 0 && brace == 0 => return Some(i),
            _ => {}
        }
    }
    None
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

/// Returns `true` iff the state-predicate expander must NOT split `clause` on
/// top-level `/\` — because expr_v2 CONFIDENTLY reports that `clause` is rooted
/// in a top-level `=>`/`<=>` whose `/\` tokens live inside the ANTECEDENT,
/// not a genuine top-level conjunction.
///
/// The naive `split_top_level(_, "/\\")` ignores operator precedence. The one
/// shape this guard fixes:
///   - `/\ A /\ B => C` — root `=>`/`<=>` (LOOSER than `/\`), so the `/\` bullets
///     are inside the implication ANTECEDENT `(A /\ B)`. Splitting turns the
///     leading `/\ A` into a spurious REQUIRED conjunct, so an inductive-
///     invariant-as-Init predicate rejects nearly all valid initial states.
///
/// Rule: keep the clause WHOLE only when v2 confidently parses it AND its ROOT
/// is `Binary{Implies|Iff}` AND the antecedent has no top-level membership
/// (`x \in S`) the enumerator would otherwise harvest. EVERY other root —
/// `Junction{And}` (genuine top-level conjunction), `Junction{Or}`, `Quant`
/// (quantifier, e.g. `\A i \in S : /\ A /\ B`), `Let`, `If`, a bare leaf, a
/// container — falls through to the CURRENT split behavior UNCHANGED. This
/// function returns `true` in exactly the keep-whole implication case above.
///
/// NOTE: this guard does NOT fix quantifier-rooted (`\A .../ \E ...`) predicates
/// — those explicitly fall through to the existing split. In particular it does
/// NOT resolve the MCBakery `IInv` under-exploration, whose root cause is a
/// separate, deeper eval-level bug in implication-under-`\A`.
///
/// Deliberately conservative on UNCERTAINTY — returns `false` (→ current split
/// behavior, unchanged) whenever v2 is disabled (`TLAPLUS_EXPR_PARSER=v1`/`off`)
/// or the parse errors / mis-fences. So a genuine top-level `/\` conjunction
/// still splits exactly as before, and no spec regresses on a v2 parse failure.
/// Mirrors the root-check + fallback pattern in `split_action_conjuncts_v2`
/// (`action_ir`) and `classify_boolean_v2` (`dispatch`).
fn state_predicate_root_is_looser_than_and(clause: &str) -> bool {
    use crate::tla::expr_v2::{self, ast};

    if !expr_v2::v2_enabled() {
        return false;
    }
    // Uniform-dedent so the parser's column coordinate system matches the source
    // layout (a state-predicate clause pulled from a definition body may carry
    // continuation indentation). `parse_ast` errors on trailing tokens, so `Ok`
    // == full consumption; any error → fallback (current split behavior).
    let dedented = crate::tla::action_ir::uniform_dedent(clause);
    let src = dedented.trim();
    if src.is_empty() {
        return false;
    }
    match expr_v2::parse_ast(src) {
        // A top-level implication/iff whose antecedent is a `/\` block — the
        // `/\` bullets are the ANTECEDENT, not top-level conjuncts. Splitting on
        // `/\` would shred the leading antecedent bullet into a spurious REQUIRED
        // conjunct (`/\ A /\ B => C` -> `[A, "B => C"]`). Keep the clause whole
        // ONLY when the antecedent has no top-level state-predicate membership
        // (`x \in S`) that the enumerator would otherwise harvest — otherwise
        // keeping it whole would strand a needed variable assignment inside the
        // antecedent (an "Init does not assign" failure). See
        // `implies_antecedent_has_top_level_membership`.
        Ok(ast::Expr::Binary {
            op: ast::BinOp::Implies | ast::BinOp::Iff,
            lhs,
            ..
        }) => !implies_antecedent_has_top_level_membership(&lhs),
        // Everything else — a genuine `Junction{And}`, a `Junction{Or}` (whose
        // downstream `\/`-branch merge logic depends on the current split), a
        // quantifier, a bare leaf, a container, `Let`/`If`, … — keeps the CURRENT
        // behavior. Parse/lex errors and mis-fences likewise fall back. This is
        // the conservative choice: only the proven-shredded implication shape
        // above changes, and only when it strands no membership assignment.
        _ => false,
    }
}

/// True iff `ante` (the antecedent of a top-level `=>`/`<=>`) is a `/\` junction
/// whose items include a top-level `x \in S` membership. Such a membership is a
/// candidate variable assignment the Init enumerator harvests during the `/\`
/// split; keeping the whole implication as one clause would strand it (the
/// enumerator would never bind that variable → "Init does not assign"). When the
/// antecedent has no such membership (the shredded-guard case: `/\ A /\ B => C`
/// where `A`,`B` are pure predicates), it is safe — and correct — to keep the
/// implication whole.
fn implies_antecedent_has_top_level_membership(ante: &crate::tla::expr_v2::ast::Expr) -> bool {
    use crate::tla::expr_v2::ast;
    match ante {
        ast::Expr::Junction { op: ast::JunctionOp::And, items, .. } => items.iter().any(|it| {
            matches!(it, ast::Expr::Binary { op: ast::BinOp::In, .. })
                || matches!(it, ast::Expr::Atom { text, .. } if text.contains("\\in"))
        }),
        // A bare `x \in S` antecedent (no `/\`) is also a membership.
        ast::Expr::Binary { op: ast::BinOp::In, .. } => true,
        ast::Expr::Atom { text, .. } => text.contains("\\in"),
        _ => false,
    }
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

    // SOUNDNESS GUARD (expr_v2-root-aware — the Phase-4 pattern). The naive
    // `split_top_level(_, "/\\")` below ignores `=>`/`<=>` precedence: for a
    // state predicate like `/\ A /\ B => C` it yields `["A", "B => C"]`, turning
    // the leading `/\ A` into a SEPARATE REQUIRED conjunct. But `=>`/`<=>` are
    // LOOSER than `/\`, so the true parse is `((A /\ B) => C)` — one implication,
    // NOT two conjuncts. Shredding the antecedent makes an inductive-invariant-
    // as-Init predicate reject nearly all valid initial states.
    //
    // Fix: consult expr_v2 BEFORE splitting. When v2 is enabled, `parse_ast`
    // fully consumes `trimmed`, and its ROOT is `Binary{Implies|Iff}` (a looser
    // top operator) with no top-level membership in the antecedent, the `/\`
    // lives INSIDE the antecedent — do NOT split; append `trimmed` as ONE clause
    // (evaluated whole via `eval_expr` → v2, which groups it correctly). This is
    // the ONLY shape kept whole: a genuine top-level `/\` (root `Junction{And}`),
    // a `Junction{Or}`, a quantifier root (`\A`/`\E`), `Let`/`If`, a bare leaf, a
    // parse failure, or v2 being disabled ALL fall through to the CURRENT split
    // behavior unchanged. This narrowly targets the shredded-antecedent bug and
    // cannot alter a genuine conjunction's (or quantifier's) enumeration.
    if !state_predicate_root_is_looser_than_and(trimmed) {
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
    }

    // Handle TLC sub-expression references: Name!N refers to the N-th conjunct
    // of the definition body. E.g., Init!1 = first conjunct of Init.
    if let Some(bang_pos) = trimmed.find('!') {
        let name_part = trimmed[..bang_pos].trim();
        let index_part = trimmed[bang_pos + 1..].trim();
        if let Ok(index) = index_part.parse::<usize>() {
            if let Some(def) = definitions.get(name_part) {
                // Filter out empty conjuncts (from comment-stripped lines)
                let conjuncts: Vec<String> = split_top_level(&def.body, "/\\")
                    .into_iter()
                    .filter(|c| !c.trim().is_empty())
                    .collect();
                if index >= 1 && index <= conjuncts.len() {
                    let conjunct = conjuncts[index - 1].trim();
                    append_expanded_state_predicate_clause(
                        conjunct,
                        definitions,
                        instances,
                        active_instance,
                        visiting,
                        out,
                    );
                    return;
                }
            }
        }
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

/// T5.5 — try to solve Init+invariant jointly via Z3.
///
/// Returns `Some(states)` only when:
///   1. Every Init clause is `module_var \in <sequence-set-comprehension>` for
///      a *distinct* module variable (no duplicate-variable scopes), AND
///   2. The set-comprehension destructures into a known function-set shape
///      (`Permutation(S)`, `[1..n -> S]`, or a filtered version of either),
///      AND
///   3. Every invariant body translates into the supported Z3 subset
///      (boolean ops, `\E i \in lit-set`, `var[i] = const`, `var[i] # var[j]`,
///      etc. — same surface as T5.1/T5.2).
///
/// On a witness Violation result, the returned vector contains exactly that
/// one violating initial state — the runtime then immediately reports the
/// invariant as violated. On `NoViolation`, the returned vector is empty,
/// which combined with `trivial_next` means the runtime explores zero states
/// and reports "0 states checked, no invariant violations" — a complete
/// proof since SMT covered every initial state.
///
/// Returns `None` (caller falls back to brute-force Init enumeration) if any
/// step above is unsupported or the SMT solver returns Unknown. Soundness:
/// since the symbolic translator only emits an answer when it has fully
/// translated every constraint, every `Some` return is sound.
fn try_joint_init_invariant_solve(
    module: &TlaModule,
    cfg: &TlaConfig,
    init_name: &str,
    invariant_exprs: &[(String, String)],
) -> Option<Vec<TlaState>> {
    #[cfg(not(feature = "symbolic-init"))]
    {
        let _ = (module, cfg, init_name, invariant_exprs);
        None
    }

    #[cfg(feature = "symbolic-init")]
    {
        let dbg = std::env::var("TLAPLUSPLUS_DEBUG_SYMBOLIC_INIT").is_ok();
        if invariant_exprs.is_empty() {
            return None;
        }
        if init_name == "__EVAL_ONLY__" {
            return None;
        }
        let init_def = module.definitions.get(init_name)?;
        let body = init_def.body.trim();

        // Walk the Init body, splitting at top-level /\ and harvesting
        // `var \in <set>` clauses. Anything else (top-level disjunctions,
        // \E quantifiers, equality-only Init, ...) is unsupported here and
        // we bail to fall back.
        let raw_clauses = expand_state_predicate_clauses(body, &module.definitions, &module.instances);
        let mut membership_pairs: Vec<(String, String)> = Vec::new();
        for c in &raw_clauses {
            let c = c.trim();
            if c.is_empty() {
                continue;
            }
            match classify_clause(c) {
                ClauseKind::UnprimedMembership { var, set_expr }
                    if module.variables.contains(&var) =>
                {
                    membership_pairs.push((var, set_expr));
                }
                ClauseKind::UnprimedEquality { .. } => {
                    if dbg {
                        eprintln!("T5.5 bail: UnprimedEquality in clause: {:?}", c);
                    }
                    return None;
                }
                other => {
                    if dbg {
                        eprintln!("T5.5 bail: unsupported clause kind {:?} in {:?}", other, c);
                    }
                    return None;
                }
            }
        }
        if membership_pairs.is_empty() {
            return None;
        }
        // Disjoint scopes: every module variable must appear in at most one
        // clause. (Required for the cross-product-as-Cartesian-product
        // assumption.)
        let mut seen = HashSet::new();
        for (v, _) in &membership_pairs {
            if !seen.insert(v.clone()) {
                return None;
            }
        }

        // Build an EvalContext seeded with config constants.
        let definition_scope = merged_definition_scope(module);
        let mut base_state = TlaState::new();
        for (k, v) in &cfg.constants {
            if let Some(tv) = config_value_to_tla(v) {
                base_state.insert(Arc::from(k.as_str()), tv);
            }
        }
        let ctx = EvalContext::with_definitions_and_instances(
            &base_state,
            &definition_scope,
            &module.instances,
        );

        // Resolve each membership clause to a sequence-set spec.
        let mut var_specs: Vec<crate::tla::symbolic_init::JointVarSpec> = Vec::with_capacity(
            membership_pairs.len(),
        );
        for (var_name, set_expr) in &membership_pairs {
            match resolve_joint_var_spec(var_name, set_expr, &ctx) {
                Some(spec) => {
                    if dbg {
                        eprintln!(
                            "T5.5 resolved {} -> seq_len={} range_size={}",
                            spec.name,
                            spec.seq_len,
                            spec.range.len(),
                        );
                    }
                    var_specs.push(spec);
                }
                None => {
                    if dbg {
                        eprintln!(
                            "T5.5 bail: failed to resolve var spec for {} <- {:?}",
                            var_name, set_expr
                        );
                    }
                    return None;
                }
            }
        }

        // Hand off to the joint Z3 solver.
        let started = std::time::Instant::now();
        let outcome = crate::tla::symbolic_init::try_symbolic_init_with_invariants(
            &var_specs,
            invariant_exprs,
            &ctx,
        )?;
        let elapsed = started.elapsed();

        match outcome {
            crate::tla::symbolic_init::JointInitOutcome::NoViolation => {
                eprintln!(
                    "T5.5 joint Init+invariant solver: PROVED no violation \
                     ({} variables, {} invariants) in {:.3}s",
                    var_specs.len(),
                    invariant_exprs.len(),
                    elapsed.as_secs_f64()
                );
                Some(Vec::new())
            }
            crate::tla::symbolic_init::JointInitOutcome::Violation { state } => {
                eprintln!(
                    "T5.5 joint Init+invariant solver: VIOLATION witness \
                     found in {:.3}s",
                    elapsed.as_secs_f64()
                );
                let mut tla_state = base_state.clone();
                for (name, value) in state {
                    tla_state.insert(Arc::from(name.as_str()), value);
                }
                Some(vec![tla_state])
            }
        }
    }
}

/// Resolve a single `var \in <set-expr>` Init clause into a JointVarSpec.
/// Recognises the same shapes as `try_symbolic_function_set_enumerate` /
/// `try_funasseq_wrapper_symbolic`:
///   - `var \in [1..n -> R]`              → JointVarSpec { pred: "TRUE" }
///   - `var \in { p \in [1..n -> R] : Q(p) }` (with rewrites)
///   - `var \in Permutation(R)`           → unwraps via `try_resolve_funasseq_permutation_set`
///   - `var \in { p \in Permutation(R) : Q(p) }`
///
/// The returned `init_pred` references positions as `var_name[i]` (the
/// outer Init variable name, not the inner binder).
#[cfg(feature = "symbolic-init")]
fn resolve_joint_var_spec(
    var_name: &str,
    set_expr: &str,
    ctx: &EvalContext<'_>,
) -> Option<crate::tla::symbolic_init::JointVarSpec> {
    use crate::tla::eval::{
        try_destructure_function_set_comprehension, try_resolve_funasseq_permutation_set,
        try_resolve_sequence_domain,
    };
    let set_expr = set_expr.trim();

    // Shape A: `var \in { x \in <inner> : outer_pred(x) }` where <inner>
    // resolves through Permutation/FunAsSeq to a function-set comprehension.
    if let Some(stripped) = set_expr.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        if let Some(colon) = crate::tla::eval::find_top_level_char(stripped, ':') {
            let lhs = stripped[..colon].trim();
            let outer_pred = stripped[colon + 1..].trim();
            if let Some(in_idx) = crate::tla::eval::find_top_level_keyword_index(lhs, "\\in") {
                let outer_var = lhs[..in_idx].trim();
                let inner_set_expr = lhs[in_idx + 3..].trim();
                if crate::tla::eval::is_valid_identifier(outer_var) {
                    if let Some((p_name, dom_text, range_text, inner_pred)) =
                        try_resolve_funasseq_permutation_set(inner_set_expr, ctx)
                    {
                        if let Some((seq_len, range_vals)) =
                            try_resolve_sequence_domain(&dom_text, &range_text, ctx, 0)
                        {
                            // Substitute outer_var -> p_name in outer_pred,
                            // then p_name -> var_name in the combined pred.
                            let outer_pred_sub = substitute_ident(outer_pred, outer_var, &p_name);
                            let combined = format!("({}) /\\ ({})", inner_pred, outer_pred_sub);
                            let final_pred = substitute_ident(&combined, &p_name, var_name);
                            return Some(crate::tla::symbolic_init::JointVarSpec {
                                name: var_name.to_string(),
                                seq_len,
                                range: range_vals,
                                init_pred: final_pred,
                            });
                        }
                    }
                    // Or: inner is a literal `[Dom -> Range]` with no inner pred.
                    if inner_set_expr.starts_with('[') && inner_set_expr.ends_with(']') {
                        let inner_bracket = &inner_set_expr[1..inner_set_expr.len() - 1];
                        if let Some((dom_text, range_text)) =
                            crate::tla::eval::split_once_top_level(inner_bracket, "->")
                        {
                            if !dom_text.contains('|') {
                                if let Some((seq_len, range_vals)) =
                                    try_resolve_sequence_domain(dom_text.trim(), range_text.trim(), ctx, 0)
                                {
                                    let pred = substitute_ident(outer_pred, outer_var, var_name);
                                    return Some(crate::tla::symbolic_init::JointVarSpec {
                                        name: var_name.to_string(),
                                        seq_len,
                                        range: range_vals,
                                        init_pred: pred,
                                    });
                                }
                            }
                        }
                    }
                    // Or: `{p \in [Dom -> Range] : pred}` directly
                    if let Some((p_name, dom_text, range_text, inner_pred)) =
                        try_destructure_function_set_comprehension(inner_set_expr, ctx)
                    {
                        if let Some((seq_len, range_vals)) =
                            try_resolve_sequence_domain(&dom_text, &range_text, ctx, 0)
                        {
                            let outer_pred_sub = substitute_ident(outer_pred, outer_var, &p_name);
                            let combined = format!("({}) /\\ ({})", inner_pred, outer_pred_sub);
                            let final_pred = substitute_ident(&combined, &p_name, var_name);
                            return Some(crate::tla::symbolic_init::JointVarSpec {
                                name: var_name.to_string(),
                                seq_len,
                                range: range_vals,
                                init_pred: final_pred,
                            });
                        }
                    }
                }
            }
        }
    }

    // Shape B: `var \in Permutation(S)` (no outer filter).
    if let Some((p_name, dom_text, range_text, inner_pred)) =
        try_resolve_funasseq_permutation_set(set_expr, ctx)
    {
        if let Some((seq_len, range_vals)) =
            try_resolve_sequence_domain(&dom_text, &range_text, ctx, 0)
        {
            let pred = substitute_ident(&inner_pred, &p_name, var_name);
            return Some(crate::tla::symbolic_init::JointVarSpec {
                name: var_name.to_string(),
                seq_len,
                range: range_vals,
                init_pred: pred,
            });
        }
    }

    // Shape C: `var \in [1..n -> Range]` (no filter, no permutation).
    if set_expr.starts_with('[') && set_expr.ends_with(']') {
        let inner = &set_expr[1..set_expr.len() - 1];
        if let Some((dom_text, range_text)) = crate::tla::eval::split_once_top_level(inner, "->") {
            if !dom_text.contains('|') {
                if let Some((seq_len, range_vals)) =
                    try_resolve_sequence_domain(dom_text.trim(), range_text.trim(), ctx, 0)
                {
                    return Some(crate::tla::symbolic_init::JointVarSpec {
                        name: var_name.to_string(),
                        seq_len,
                        range: range_vals,
                        init_pred: "TRUE".to_string(),
                    });
                }
            }
        }
    }

    None
}

/// Replace standalone identifier `from` with `to` in `text`.
#[cfg(feature = "symbolic-init")]
fn substitute_ident(text: &str, from: &str, to: &str) -> String {
    let bytes = text.as_bytes();
    let needle = from.as_bytes();
    if needle.is_empty() || from == to {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + needle.len() <= bytes.len() && &bytes[i..i + needle.len()] == needle {
            let prev_ok =
                i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let next_idx = i + needle.len();
            let next_ok = next_idx == bytes.len()
                || !(bytes[next_idx].is_ascii_alphanumeric() || bytes[next_idx] == b'_');
            if prev_ok && next_ok {
                out.push_str(to);
                i += needle.len();
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// True if `expr` textually references any declared module variable other
/// than `assign_var` itself, at an identifier boundary.
///
/// Used by Init enumeration to decide whether an `assign_var = expr`
/// equality can be resolved eagerly (RHS is constant/self-contained) or must
/// be deferred until after the membership cross-product binds the referenced
/// variable(s). Without deferral, an early `eval_expr` on the RHS resolves the
/// unbound variable to a spurious `ModelValue`, corrupting the state.
fn expr_references_declared_variable(
    expr: &str,
    assign_var: &str,
    variables: &[String],
) -> bool {
    let bytes = expr.as_bytes();
    for var in variables {
        if var == assign_var {
            continue;
        }
        let needle = var.as_bytes();
        if needle.is_empty() || bytes.len() < needle.len() {
            continue;
        }
        let mut i = 0;
        while i + needle.len() <= bytes.len() {
            if &bytes[i..i + needle.len()] == needle {
                let prev_ok =
                    i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                let next_idx = i + needle.len();
                // A trailing `'` marks a primed reference (not this unprimed
                // variable); a trailing word char means it's a longer ident.
                let next_ok = next_idx == bytes.len()
                    || !(bytes[next_idx].is_ascii_alphanumeric()
                        || bytes[next_idx] == b'_'
                        || bytes[next_idx] == b'\'');
                if prev_ok && next_ok {
                    return true;
                }
            }
            i += 1;
        }
    }
    false
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
    // Eval-only modules have no Init — return empty state list
    if init_name == "__EVAL_ONLY__" {
        return Ok(Vec::new());
    }

    let mut definition_scope = merged_definition_scope(module);
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

    // If the entire Init body is wrapped in \E quantifier(s), we need to
    // handle it specially. Pattern: `\E x \in S : body` where body contains
    // the actual variable assignments. We synthesize states by evaluating
    // each binding of x and then processing the body as Init.
    if trimmed_body.starts_with("\\E ") || trimmed_body.starts_with("\\exists ") {
        // Find the colon that separates binder from body
        if let Some(colon_pos) = find_top_level_colon(trimmed_body) {
            let binder_part = trimmed_body[..colon_pos].trim();
            let body_part = trimmed_body[colon_pos + 1..].trim();

            // Parse the binder: \E x \in S
            let binder_text = binder_part
                .strip_prefix("\\E ")
                .or_else(|| binder_part.strip_prefix("\\exists "))
                .unwrap_or(binder_part);

            // Find \in to split variable name and domain
            if let Some(in_pos) = binder_text.find("\\in") {
                let var_name = binder_text[..in_pos].trim();
                let domain_expr = binder_text[in_pos + 3..].trim();

                // Create a synthetic Init definition with the body
                let synth_name = format!("__SyntheticInitBody_{init_name}__");
                let mut module_clone = module.clone();
                module_clone.definitions.insert(
                    synth_name.clone(),
                    TlaDefinition {
                        name: synth_name.clone(),
                        params: vec![],
                        body: body_part.to_string(),
                        is_recursive: false,
                    },
                );

                // Evaluate domain
                let temp_state = TlaState::new();
                let ctx = EvalContext::with_definitions_and_instances(
                    &temp_state,
                    &definition_scope,
                    &module.instances,
                );

                // Inject constants into temp state for domain eval
                let mut eval_state = TlaState::new();
                for (k, v) in &cfg.constants {
                    if let Some(tv) = config_value_to_tla(v) {
                        eval_state.insert(Arc::from(k.as_str()), tv);
                    }
                }
                let ctx = EvalContext::with_definitions_and_instances(
                    &eval_state,
                    &definition_scope,
                    &module.instances,
                );

                if let Ok(domain_val) = eval_predicate(domain_expr, &ctx) {
                    if let Ok(domain_set) = domain_val.as_set() {
                        let mut all_states = Vec::new();
                        for binding_val in domain_set.iter() {
                            // Inject the binding as a constant
                            let mut cfg_clone = cfg.clone();
                            // Use a sentinel value that will be converted to TlaValue
                            cfg_clone.constants.insert(
                                var_name.to_string(),
                                match binding_val {
                                    TlaValue::Int(n) => ConfigValue::Int(*n),
                                    TlaValue::Bool(b) => ConfigValue::Bool(*b),
                                    TlaValue::String(s) => ConfigValue::String(s.clone()),
                                    TlaValue::ModelValue(s) => ConfigValue::ModelValue(s.clone()),
                                    _ => ConfigValue::ModelValue(format!("{:?}", binding_val)),
                                },
                            );
                            // Try evaluating Init with this binding
                            if let Ok(states) =
                                evaluate_init_states(&module_clone, &cfg_clone, &synth_name)
                            {
                                all_states.extend(states);
                            }
                        }
                        if !all_states.is_empty() {
                            return Ok(all_states);
                        }
                    }
                }
            }
        }
    }

    // If the Init body is a top-level disjunction (\/ branch1 \/ branch2),
    // process each branch as a separate Init and merge results.
    // This handles SmokeInit-style specs with multiple Init alternatives.
    if trimmed_body.starts_with("\\/") {
        let branches = split_top_level(trimmed_body, "\\/");
        let mut all_states = Vec::new();
        for branch in &branches {
            let branch = branch.trim();
            if branch.is_empty() {
                continue;
            }
            let synth_name = format!("__SyntheticInitBranch_{init_name}__");
            let mut module_clone = module.clone();
            module_clone.definitions.insert(
                synth_name.clone(),
                TlaDefinition {
                    name: synth_name.clone(),
                    params: vec![],
                    body: branch.to_string(),
                    is_recursive: false,
                },
            );
            if let Ok(states) = evaluate_init_states(&module_clone, cfg, &synth_name) {
                all_states.extend(states);
            }
        }
        if !all_states.is_empty() {
            return Ok(all_states);
        }
    }

    // Start with constants from config
    let mut base_state = TlaState::new();
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
            match eval_predicate(&ref_name, &ctx) {
                Ok(value) => {
                    base_state.insert(Arc::from(name.as_str()), value);
                    progress = true;
                }
                Err(_) => {
                    // Try resolving as a zero-arg definition from definition_scope
                    if let Some(def) = definition_scope.get(&ref_name) {
                        if def.params.is_empty() {
                            if let Ok(value) = eval_predicate(&def.body, &ctx) {
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
            // Log unresolved operator refs for debugging
            for (name, ref_name) in &next_deferred {
                eprintln!(
                    "Warning: could not resolve constant operator '{}' <- '{}', \
                     trying as definition injection",
                    name, ref_name
                );
                // Last resort: if the ref is a definition, inject its body as a
                // synthetic definition for the constant name
                if let Some(def) = definition_scope.get(ref_name).cloned() {
                    let mut injected = def.clone();
                    injected.name = name.clone();
                    definition_scope.insert(name.clone(), injected);
                    progress = true;
                }
            }
            if progress {
                // Clear next_deferred since we injected as definitions
                next_deferred.clear();
            }
            if !progress {
                break;
            }
        }
        deferred_operator_refs = next_deferred;
    }

    // Classify all clauses
    let mut equality_assignments: Vec<(String, String)> = Vec::new();
    let mut membership_assignments: Vec<(String, String)> = Vec::new();
    let mut guards: Vec<String> = Vec::new();

    // Expand and merge clauses — rejoin disjunctive branches that split_top_level
    // may have broken apart. E.g., Init with:
    //   /\ \/ Guard1 /\ var = expr1
    //      \/ Guard2 /\ var = expr2
    // produces clauses ["\\/ Guard1", "var = expr1 \\/ Guard2", "var = expr2"]
    // We need to rejoin these into a single disjunctive clause.
    let raw_clauses =
        expand_state_predicate_clauses(&init_def.body, &module.definitions, &module.instances);
    let mut merged_clauses = Vec::new();
    let mut i = 0;
    while i < raw_clauses.len() {
        let c = raw_clauses[i].trim().to_string();

        // If this clause starts with \E (existential quantifier), the body
        // may have been split across subsequent clauses. Rejoin them.
        // Pattern: \E x \in S : body1 /\ body2 gets split into
        // ["\E x \in S : body1", "body2"] — we need to rejoin.
        if c.starts_with("\\E ") || c.starts_with("\\exists ") {
            let mut merged = c.clone();
            // The \E clause should contain a colon. If it does, check if
            // there are subsequent clauses that are part of the body.
            // We greedily grab following clauses that look like variable
            // assignments inside the quantifier.
            while i + 1 < raw_clauses.len() {
                let next = raw_clauses[i + 1].trim();
                // Stop if next clause is another \E, \/ or clearly independent
                if next.starts_with("\\E ")
                    || next.starts_with("\\exists ")
                    || next.starts_with("\\/")
                {
                    break;
                }
                // Check if next clause references variables that might be
                // bound by the quantifier, or contains assignments that
                // depend on it. Heuristic: if the clause contains an identifier
                // from the \E binder, it belongs to the body.
                merged = format!("{} /\\ {}", merged, next);
                i += 1;
            }
            merged_clauses.push(merged);
            i += 1;
            continue;
        }

        // If this clause starts with \/ or contains \/ midway, it's part of
        // a disjunction that got split. Merge it back.
        if c.starts_with("\\/") {
            // This clause and potentially following ones form a disjunction
            let mut merged = c.clone();
            // Look ahead for continuations that contain \/ or follow the pattern
            while i + 1 < raw_clauses.len() {
                let next = raw_clauses[i + 1].trim();
                if next.contains("\\/") || next.starts_with("\\/") {
                    merged = format!("{} /\\ {}", merged, next);
                    i += 1;
                } else {
                    break;
                }
            }
            merged_clauses.push(merged);
        } else if c.contains("\\/") {
            // The \/ appeared in the middle of a clause (e.g., "expr1 \\/ Guard")
            // This shouldn't normally happen from a clean split, but handle it
            merged_clauses.push(c);
        } else {
            merged_clauses.push(c);
        }
        i += 1;
    }

    // A conjunctive Init with a disjunctive conjunct — `/\ A /\ (\/ B1 \/ B2)
    // /\ C` — must be DISTRIBUTED into `(A /\ B1 /\ C) \/ (A /\ B2 /\ C)` and
    // each alternative recursed as its own Init, so the disjunct branches'
    // membership/equality alternatives are enumerated. Without this the loop
    // below evaluates the `\/` clause as a boolean guard; with unbound state
    // variables that reads FALSE (`light \in {"off","on"}` with `light` unbound
    // is `ModelValue("light") \in {..}` = false), so the branch is silently
    // dropped and the initial-state count collapses — Prisoner{Solo,}LightUnknown
    // lost the `Light_Unknown /\ light \in {"off","on"}` alternative → 1 init
    // state instead of 2. We distribute over `merged_clauses` (NOT a raw
    // `split_top_level(_, "/\\")`, which shreds a disjunct's inner `/\` guards);
    // `merged_clauses` has already rejoined each disjunction into one clause. A
    // disjunct with a false constant guard (`~Light_Unknown` when `Light_Unknown
    // = TRUE`) yields zero states via the guard check, so the union is exact.
    // Recursion terminates (each level removes one disjunctive conjunct).
    if merged_clauses.len() > 1 {
        let disj_pos = merged_clauses.iter().position(|c| {
            let t = c.trim();
            t.starts_with("\\/") && split_top_level(t, "\\/").len() > 1
        });
        if let Some(pos) = disj_pos {
            let disjuncts = split_top_level(merged_clauses[pos].trim(), "\\/");
            if disjuncts.len() > 1 {
                let others: Vec<&String> = merged_clauses
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != pos)
                    .map(|(_, c)| c)
                    .collect();
                let mut all_states = Vec::new();
                for d in &disjuncts {
                    let d = d.trim();
                    if d.is_empty() {
                        continue;
                    }
                    let mut new_body = format!("/\\ {d}");
                    for o in &others {
                        new_body.push_str("\n/\\ ");
                        new_body.push_str(o);
                    }
                    let synth_name = format!("__SyntheticInitDistrib_{init_name}__");
                    let mut module_clone = module.clone();
                    module_clone.definitions.insert(
                        synth_name.clone(),
                        TlaDefinition {
                            name: synth_name.clone(),
                            params: vec![],
                            body: new_body,
                            is_recursive: false,
                        },
                    );
                    if let Ok(states) = evaluate_init_states(&module_clone, cfg, &synth_name) {
                        all_states.extend(states);
                    }
                }
                if !all_states.is_empty() {
                    return Ok(all_states);
                }
            }
        }
    }

    for clause in merged_clauses {
        // For disjunctive clauses (\/ branches), try to resolve them
        // using already-known constants. This handles patterns like:
        //   \/ Light_Unknown /\ light \in {"off","on"}
        //   \/ ~Light_Unknown /\ light = "off"
        // When Light_Unknown is a known constant (FALSE), this simplifies
        // to `light = "off"`.
        let clause_trimmed = clause.trim();
        if clause_trimmed.contains("\\/") {
            let ctx = EvalContext::with_definitions_and_instances(
                &base_state,
                &definition_scope,
                &module.instances,
            );
            // Try evaluating the whole disjunction — if it produces a
            // definite value, we can skip it (TRUE guard) or reject (FALSE)
            if let Ok(TlaValue::Bool(_)) = eval_predicate(clause_trimmed, &ctx) {
                guards.push(clause);
                continue;
            }
            // Otherwise, try splitting on \/ and finding the branch that
            // has satisfiable guards + variable assignments
            let branches = split_top_level(clause_trimmed, "\\/");
            if branches.len() > 1 {
                let mut handled = false;
                for branch in &branches {
                    let branch = branch.trim();
                    if branch.is_empty() {
                        continue;
                    }
                    // Check if this branch's guard evaluates to TRUE
                    let sub_clauses = split_top_level(branch, "/\\");
                    let mut branch_ok = true;
                    let mut branch_assignments = Vec::new();
                    let mut branch_memberships = Vec::new();
                    for sc in &sub_clauses {
                        let sc = sc.trim();
                        match classify_clause(sc) {
                            ClauseKind::UnprimedEquality { ref var, .. }
                                if module.variables.contains(var) =>
                            {
                                branch_assignments.push(sc.to_string());
                            }
                            ClauseKind::UnprimedMembership { ref var, .. }
                                if module.variables.contains(var) =>
                            {
                                branch_memberships.push(sc.to_string());
                            }
                            _ => {
                                // Guard — check if it's satisfied
                                if let Ok(TlaValue::Bool(false)) = eval_predicate(sc, &ctx) {
                                    branch_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                    if branch_ok
                        && (!branch_assignments.is_empty() || !branch_memberships.is_empty())
                    {
                        for sc in &branch_assignments {
                            if let ClauseKind::UnprimedEquality { var, expr } = classify_clause(sc)
                            {
                                equality_assignments.push((var, expr));
                            }
                        }
                        for sc in &branch_memberships {
                            if let ClauseKind::UnprimedMembership { var, set_expr } =
                                classify_clause(sc)
                            {
                                membership_assignments.push((var, set_expr));
                            }
                        }
                        handled = true;
                        break;
                    }
                }
                if handled {
                    continue;
                }
            }
        }

        match classify_clause(&clause) {
            ClauseKind::UnprimedEquality { var, expr } if module.variables.contains(&var) => {
                // If the RHS references another declared variable (e.g.
                // `ack = rdy` where `rdy \in {0, 1}` is a membership
                // assignment), the RHS variable is not yet bound during the
                // eager equality phase. Evaluating it early would resolve the
                // bare identifier to a bogus `ModelValue("rdy")`, corrupting
                // the state and both undercounting exploration AND tripping a
                // false invariant violation. Route it through the guards list
                // so it lands in `late_equalities`, which re-evaluates the RHS
                // per-state AFTER the cross-product has bound every variable.
                if expr_references_declared_variable(&expr, &var, &module.variables) {
                    guards.push(format!("{} = {}", var, expr));
                } else {
                    equality_assignments.push((var, expr));
                }
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
            match eval_predicate(&expr, &ctx) {
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
                    if let Ok(val) = eval_predicate(&def.body, &ctx) {
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
                match eval_predicate(&expr, &ctx) {
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
    // Use a fixed-point loop: some membership sets depend on other variables
    // being assigned first (e.g., `terminationDetected \in {FALSE, terminated}`
    // where `terminated` depends on `active` which is also a membership assignment).
    let mut membership_choices: Vec<(String, Vec<TlaValue>)> = Vec::new();
    let mut pending_memberships = membership_assignments;

    for _ in 0..pending_memberships.len().saturating_add(1) {
        if pending_memberships.is_empty() {
            break;
        }

        let mut progress = false;
        let mut next_pending = Vec::new();

        for (var, set_expr) in pending_memberships {
            let ctx = EvalContext::with_definitions_and_instances(
                &base_state,
                &definition_scope,
                &module.instances,
            );
            match eval_predicate(&set_expr, &ctx) {
                Ok(set_val) => {
                    // Accept both Set and Seq as membership domains.
                    // Constraint-propagated record sets return Seq for O(n)
                    // construction instead of BTreeSet O(n log n).
                    let values: Option<Vec<TlaValue>> = if let Ok(set) = set_val.as_set() {
                        Some(set.iter().cloned().collect())
                    } else if let Ok(seq) = set_val.as_seq() {
                        Some(seq.clone())
                    } else {
                        None
                    };

                    if let Some(values) = values {
                        if values.is_empty() {
                            return Err(anyhow!(
                                "membership set for {var} is empty, no initial states possible"
                            ));
                        }
                        if values.len() == 1 {
                            base_state.insert(Arc::from(var.as_str()), values[0].clone());
                        }
                        membership_choices.push((var.clone(), values));
                        progress = true;
                    } else {
                        next_pending.push((var, set_expr));
                    }
                }
                Err(_) => next_pending.push((var, set_expr)),
            }
        }

        if !progress {
            // Final attempt: try with all definitions injected as state values
            let mut augmented = base_state.clone();
            for (name, def) in &definition_scope {
                if !augmented.contains_key(name.as_str()) && def.params.is_empty() {
                    let ctx = EvalContext::with_definitions_and_instances(
                        &augmented,
                        &definition_scope,
                        &module.instances,
                    );
                    if let Ok(val) = eval_predicate(&def.body, &ctx) {
                        augmented.insert(Arc::from(name.as_str()), val);
                    }
                }
            }

            let mut final_pending = Vec::new();
            for (var, set_expr) in next_pending {
                let ctx = EvalContext::with_definitions_and_instances(
                    &augmented,
                    &definition_scope,
                    &module.instances,
                );
                match eval_predicate(&set_expr, &ctx) {
                    Ok(set_val) => {
                        if let Ok(set) = set_val.as_set() {
                            if !set.is_empty() {
                                let values: Vec<TlaValue> = set.iter().cloned().collect();
                                membership_choices.push((var, values));
                                progress = true;
                                continue;
                            }
                        }
                        final_pending.push((var, set_expr));
                    }
                    Err(_) => final_pending.push((var, set_expr)),
                }
            }
            next_pending = final_pending;

            if !progress && !next_pending.is_empty() {
                // Defer these memberships to the cross-product phase where
                // dependent variables will already be bound in each state.
                // This handles patterns like:
                //   active \in [Node -> BOOLEAN]
                //   terminationDetected \in {FALSE, terminated}
                // where `terminated` depends on `active`.
                for (var, set_expr) in next_pending {
                    membership_choices.push((var, vec![TlaValue::String("__DEFERRED__".into())]));
                    guards.push(format!(
                        "__DEFERRED_MEMBERSHIP__:{}:{}",
                        membership_choices.len() - 1,
                        set_expr
                    ));
                }
                next_pending = Vec::new();
                break;
            }
        }

        pending_memberships = next_pending;
    }

    // Extract deferred membership expressions from guards
    let mut deferred_memberships: Vec<(usize, String, String)> = Vec::new();
    let mut real_guards: Vec<String> = Vec::new();
    for guard in &guards {
        if let Some(rest) = guard.strip_prefix("__DEFERRED_MEMBERSHIP__:") {
            if let Some(colon_idx) = rest.find(':') {
                let idx: usize = rest[..colon_idx].parse().unwrap_or(0);
                let set_expr = rest[colon_idx + 1..].to_string();
                // Find the variable name from membership_choices
                if idx < membership_choices.len() {
                    let var = membership_choices[idx].0.clone();
                    deferred_memberships.push((idx, var, set_expr));
                }
            }
        } else {
            real_guards.push(guard.clone());
        }
    }
    let guards = real_guards;

    // Generate all combinations of membership choices (cross product)
    let had_membership_choices = !membership_choices.is_empty();
    let base_state_for_error = base_state.clone();

    // Calculate cross-product size to decide eager vs lazy
    let mut cross_product_size: u64 = 1;
    let mut active_choices: Vec<(String, Vec<TlaValue>)> = Vec::new();
    for (var, values) in &membership_choices {
        if values.len() == 1 && values[0] == TlaValue::String("__DEFERRED__".into()) {
            continue;
        }
        cross_product_size = cross_product_size.saturating_mul(values.len() as u64);
        active_choices.push((var.clone(), values.clone()));
    }

    const MAX_EAGER_INIT_STATES: u64 = 10_000_000;

    let mut states = vec![base_state];

    if cross_product_size > MAX_EAGER_INIT_STATES {
        // Lazy enumeration: use an odometer-style iterator
        eprintln!(
            "Init cross-product has {} states, using lazy enumeration",
            cross_product_size
        );

        let base = states.into_iter().next().unwrap();
        let mut indices = vec![0usize; active_choices.len()];
        let mut generated = 0u64;
        let mut lazy_states = Vec::new();

        loop {
            // Build state from current indices
            let mut state = base.clone();
            for (i, (var, values)) in active_choices.iter().enumerate() {
                state.insert(Arc::from(var.as_str()), values[indices[i]].clone());
            }
            lazy_states.push(state);
            generated += 1;

            // Progress reporting every 1M states
            if generated % 1_000_000 == 0 {
                eprintln!(
                    "  Generated {} / {} initial states...",
                    generated, cross_product_size
                );
            }

            // Advance indices (odometer pattern, rightmost increments first)
            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] < active_choices[i].1.len() {
                        carry = false;
                    } else {
                        indices[i] = 0;
                    }
                }
            }
            if carry {
                break; // All combinations exhausted
            }

            // Memory safety: cap at 100M states
            if generated >= 100_000_000 {
                eprintln!(
                    "Warning: capped lazy enumeration at {} states (of {} total)",
                    generated, cross_product_size
                );
                break;
            }
        }

        states = lazy_states;
    } else {
        // Eager cross-product (existing path)
        for (var, values) in &active_choices {
            let mut new_states = Vec::new();
            for state in &states {
                for value in values {
                    let mut new_state = state.clone();
                    new_state.insert(Arc::from(var.as_str()), value.clone());
                    new_states.push(new_state);
                }
            }
            states = new_states;
        }
    }

    // Now expand deferred memberships — these depend on variables that
    // are already bound in each state from the cross-product above
    for (_idx, var, set_expr) in &deferred_memberships {
        let mut new_states = Vec::new();
        for state in &states {
            let ctx = EvalContext::with_definitions_and_instances(
                state,
                &definition_scope,
                &module.instances,
            );
            if let Ok(set_val) = eval_predicate(set_expr, &ctx) {
                if let Ok(set) = set_val.as_set() {
                    for value in set.iter() {
                        let mut new_state = state.clone();
                        new_state.insert(Arc::from(var.as_str()), value.clone());
                        new_states.push(new_state);
                    }
                }
            }
        }
        if !new_states.is_empty() {
            states = new_states;
        }
    }

    // Before filtering by guards, try to resolve guards that are actually
    // late-binding equality assignments (e.g., `state = [self \in Node |-> ...]`
    // where the RHS depends on membership variables now bound in each state).
    // Also handle \E quantifier guards and membership guards.
    let mut pure_guards = Vec::new();
    let mut late_equalities: Vec<(String, String)> = Vec::new();
    let mut late_memberships: Vec<(String, String)> = Vec::new();
    let mut existential_guards: Vec<String> = Vec::new();
    for guard in &guards {
        let g = guard.trim();
        if g.is_empty() {
            continue;
        }
        if g.starts_with("\\E ") || g.starts_with("\\exists ") {
            existential_guards.push(g.to_string());
            continue;
        }
        match classify_clause(g) {
            ClauseKind::UnprimedEquality { ref var, .. } if module.variables.contains(var) => {
                if let ClauseKind::UnprimedEquality { var, expr } = classify_clause(g) {
                    late_equalities.push((var, expr));
                }
            }
            ClauseKind::UnprimedMembership { ref var, .. } if module.variables.contains(var) => {
                if let ClauseKind::UnprimedMembership { var, set_expr } = classify_clause(g) {
                    late_memberships.push((var, set_expr));
                }
            }
            _ => {
                // Check if this is a parameterized operator call like XInit(x)
                // where x is a module variable — expand inline and re-classify
                let mut handled_as_op = false;
                if let Some(paren_pos) = g.find('(') {
                    let op_name = g[..paren_pos].trim();
                    if let Some(close) = g.rfind(')') {
                        let arg = g[paren_pos + 1..close].trim();
                        if module.variables.contains(&arg.to_string()) {
                            if let Some(def) = definition_scope.get(op_name) {
                                if def.params.len() == 1 {
                                    // Substitute: replace param with arg in body
                                    let expanded = def.body.replace(&def.params[0], arg);
                                    match classify_clause(&expanded) {
                                        ClauseKind::UnprimedEquality { var, expr }
                                            if module.variables.contains(&var) =>
                                        {
                                            late_equalities.push((var, expr));
                                            handled_as_op = true;
                                        }
                                        ClauseKind::UnprimedMembership { var, set_expr }
                                            if module.variables.contains(&var) =>
                                        {
                                            late_memberships.push((var, set_expr));
                                            handled_as_op = true;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                if !handled_as_op {
                    pure_guards.push(g.to_string());
                }
            }
        }
    }

    // Apply late equality assignments to each state
    if !late_equalities.is_empty() {
        for state in &mut states {
            for (var, expr) in &late_equalities {
                if !state.contains_key(var.as_str()) {
                    let ctx = EvalContext::with_definitions_and_instances(
                        state,
                        &definition_scope,
                        &module.instances,
                    );
                    if let Ok(value) = eval_predicate(expr, &ctx) {
                        state.insert(Arc::from(var.as_str()), value);
                    }
                }
            }
        }
    }

    // Apply late membership assignments (expand states)
    for (var, set_expr) in &late_memberships {
        let mut new_states = Vec::new();
        for state in &states {
            let ctx = EvalContext::with_definitions_and_instances(
                state,
                &definition_scope,
                &module.instances,
            );
            if let Ok(set_val) = eval_predicate(set_expr, &ctx) {
                if let Ok(set) = set_val.as_set() {
                    for value in set.iter() {
                        let mut new_state = state.clone();
                        new_state.insert(Arc::from(var.as_str()), value.clone());
                        new_states.push(new_state);
                    }
                }
            }
        }
        if !new_states.is_empty() {
            states = new_states;
        }
    }

    // Handle \E quantifier guards — expand states for each satisfying assignment.
    // For Init, the \E body contains unprimed assignments (not primed like Next).
    // We parse the binder, enumerate domain values, and for each value evaluate
    // the body conjuncts to extract variable assignments.
    for guard in &existential_guards {
        if let Some(colon_pos) = find_top_level_colon(guard) {
            let binder_part = guard[..colon_pos].trim();
            let body_part = guard[colon_pos + 1..].trim();

            let binder_text = binder_part
                .strip_prefix("\\E ")
                .or_else(|| binder_part.strip_prefix("\\exists "))
                .unwrap_or(binder_part);

            if let Some(in_pos) = binder_text.find("\\in") {
                let bind_var = binder_text[..in_pos].trim();
                let domain_expr = binder_text[in_pos + 3..].trim();

                let mut new_states = Vec::new();
                for state in &states {
                    let ctx = EvalContext::with_definitions_and_instances(
                        state,
                        &definition_scope,
                        &module.instances,
                    );

                    // Evaluate domain
                    let domain_set = match eval_predicate(domain_expr, &ctx) {
                        Ok(v) => match v.as_set() {
                            Ok(s) => s.clone(),
                            Err(_) => continue,
                        },
                        Err(_) => continue,
                    };

                    // For each binding value, evaluate body conjuncts
                    for bind_val in domain_set.iter() {
                        let mut trial_state = state.clone();
                        trial_state.insert(Arc::from(bind_var), bind_val.clone());

                        // Split body on /\ and classify each conjunct
                        let body_clauses = split_top_level(body_part, "/\\");
                        let mut all_ok = true;
                        for bc in &body_clauses {
                            let bc = bc.trim();
                            if bc.is_empty() {
                                continue;
                            }
                            match classify_clause(bc) {
                                ClauseKind::UnprimedEquality { ref var, .. }
                                    if module.variables.contains(var) =>
                                {
                                    if let ClauseKind::UnprimedEquality { var, expr } =
                                        classify_clause(bc)
                                    {
                                        let ctx2 = EvalContext::with_definitions_and_instances(
                                            &trial_state,
                                            &definition_scope,
                                            &module.instances,
                                        );
                                        if let Ok(val) = eval_predicate(&expr, &ctx2) {
                                            trial_state.insert(Arc::from(var.as_str()), val);
                                        } else {
                                            all_ok = false;
                                        }
                                    }
                                }
                                _ => {
                                    // Guard conjunct — evaluate as boolean
                                    let ctx2 = EvalContext::with_definitions_and_instances(
                                        &trial_state,
                                        &definition_scope,
                                        &module.instances,
                                    );
                                    match eval_predicate(bc, &ctx2) {
                                        Ok(TlaValue::Bool(false)) => {
                                            all_ok = false;
                                        }
                                        Err(_) => {
                                            all_ok = false;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        if all_ok {
                            // Remove the binding variable (it's not a module variable)
                            if !module.variables.contains(&bind_var.to_string()) {
                                trial_state.remove(bind_var);
                            }
                            new_states.push(trial_state);
                        }
                    }
                }
                if !new_states.is_empty() {
                    states = new_states;
                }
                continue;
            }
        }

        // Fallback: try eval_action_body_multi
        let mut new_states = Vec::new();
        for state in &states {
            let ctx = EvalContext::with_definitions_and_instances(
                state,
                &definition_scope,
                &module.instances,
            );
            let staged = BTreeMap::<String, TlaValue>::new();
            if let Ok(result) = eval_action_body_multi(guard, &ctx, &staged) {
                for (bindings, _) in result {
                    let mut new_state = state.clone();
                    for (k, v) in bindings {
                        if module.variables.contains(&k) {
                            new_state.insert(Arc::from(k.as_str()), v);
                        }
                    }
                    new_states.push(new_state);
                }
            }
            if new_states.is_empty() {
                new_states.push(state.clone());
            }
        }
        states = new_states;
    }

    // Filter states by pure guards
    let mut valid_states = Vec::new();
    for state in states {
        let ctx = EvalContext::with_definitions_and_instances(
            &state,
            &definition_scope,
            &module.instances,
        );

        let mut all_guards_pass = true;
        for guard in &pure_guards {
            // Phase-1: hot-path guard eval routes through the compiled path.
            let guard_result = eval_predicate(guard, &ctx);
            // Opt-in dual-evaluator consistency check (no-op unless built with
            // --features eval-consistency-check + env flag). Computes both
            // interpreted and compiled results internally and compares.
            crate::tla::eval_consistency::check_predicate_consistency(guard, &ctx);
            match guard_result {
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
            // Try to fill missing variables from other Init definitions
            // This handles Init <- MCInit where MCInit doesn't assign all vars
            let mut state = state;
            let missing_vars: Vec<String> = module
                .variables
                .iter()
                .filter(|v| !state.contains_key(v.as_str()))
                .cloned()
                .collect();
            if !missing_vars.is_empty() {
                // Search all definitions (including instance modules) for
                // assignments to missing variables. This handles Init <- MCInit
                // where MCInit doesn't assign all vars but the original Init does.
                for var in &missing_vars {
                    if state.contains_key(var.as_str()) {
                        continue;
                    }
                    // Search top-level definitions
                    for def in definition_scope.values() {
                        if state.contains_key(var.as_str()) {
                            break;
                        }
                        if !def.params.is_empty() || !def.body.contains(var.as_str()) {
                            continue;
                        }
                        for clause in expand_state_predicate_clauses(
                            &def.body,
                            &module.definitions,
                            &module.instances,
                        ) {
                            if let ClauseKind::UnprimedEquality {
                                var: ref v,
                                ref expr,
                            } = classify_clause(&clause)
                            {
                                if v == var {
                                    if let Ok(val) = eval_predicate(
                                        expr,
                                        &EvalContext::with_definitions_and_instances(
                                            &state,
                                            &definition_scope,
                                            &module.instances,
                                        ),
                                    ) {
                                        state.insert(Arc::from(var.as_str()), val);
                                    }
                                }
                            }
                        }
                    }
                    // Search instance module definitions
                    if !state.contains_key(var.as_str()) {
                        for instance in module.instances.values() {
                            if let Some(ref m) = instance.module {
                                for def in m.definitions.values() {
                                    if def.params.is_empty() && def.body.contains(var.as_str()) {
                                        for clause in expand_state_predicate_clauses(
                                            &def.body,
                                            &m.definitions,
                                            &m.instances,
                                        ) {
                                            if let ClauseKind::UnprimedEquality {
                                                var: ref v,
                                                ref expr,
                                            } = classify_clause(&clause)
                                            {
                                                if v == var {
                                                    if let Ok(val) =
                                                        eval_predicate(
                                                            expr, &EvalContext::with_definitions_and_instances(&state, &definition_scope, &module.instances),
                                                        )
                                                    {
                                                        state.insert(
                                                            Arc::from(
                                                                var.as_str(),
                                                            ),
                                                            val,
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

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
    if valid_states.is_empty() {
        let missing: Vec<String> = module
            .variables
            .iter()
            .filter(|v| !base_state_for_error.contains_key(v.as_str()))
            .cloned()
            .collect();

        if !missing.is_empty() {
            let mut recovered = false;
            let mut recovery_state = base_state_for_error.clone();

            // First try: evaluate the entire Init body as an expression.
            // This handles \E quantifiers, LET/IN blocks, and complex patterns
            // that clause-by-clause classification misses.
            {
                let ctx = EvalContext::with_definitions_and_instances(
                    &recovery_state,
                    &definition_scope,
                    &module.instances,
                );
                if let Ok(TlaValue::Bool(true)) = eval_predicate(&init_def.body, &ctx) {
                    // Init evaluated to TRUE with current state — check if
                    // variables got bound through side effects in the context
                }
            }

            for guard in &guards {
                // Handle \E quantifiers: `\E x \in S : body`
                // Extract variable assignments from inside the quantifier body
                let guard_trimmed = guard.trim();
                if guard_trimmed.starts_with("\\E ") || guard_trimmed.starts_with("\\exists ") {
                    let ctx = EvalContext::with_definitions_and_instances(
                        &recovery_state,
                        &definition_scope,
                        &module.instances,
                    );
                    // Use eval_action_body_multi to evaluate the existential
                    // and extract variable assignments
                    let staged = BTreeMap::<String, TlaValue>::new();
                    if let Ok(result) = eval_action_body_multi(guard_trimmed, &ctx, &staged) {
                        let result: Vec<(BTreeMap<String, TlaValue>, Vec<String>)> = result;
                        if let Some((bindings, _)) = result.into_iter().next() {
                            for (var_name, val) in bindings {
                                if missing.contains(&var_name)
                                    && !recovery_state.contains_key(var_name.as_str())
                                {
                                    recovery_state.insert(Arc::from(var_name.as_str()), val);
                                    recovered = true;
                                }
                            }
                        }
                    }
                    continue;
                }

                for var in &missing {
                    if recovery_state.contains_key(var.as_str()) {
                        continue;
                    }
                    // Try to find `var = expr` pattern
                    let pattern = format!("{} = ", var);
                    if let Some(idx) = guard_trimmed.find(&pattern) {
                        let rhs = guard_trimmed[idx + pattern.len()..].trim();
                        let ctx = EvalContext::with_definitions_and_instances(
                            &recovery_state,
                            &definition_scope,
                            &module.instances,
                        );
                        if let Ok(value) = eval_predicate(rhs, &ctx) {
                            recovery_state.insert(Arc::from(var.as_str()), value);
                            recovered = true;
                        }
                    }
                    // Try `var \in set` pattern
                    let mem_pattern = format!("{} \\in ", var);
                    if let Some(idx) = guard_trimmed.find(&mem_pattern) {
                        let rhs = guard_trimmed[idx + mem_pattern.len()..].trim();
                        let ctx = EvalContext::with_definitions_and_instances(
                            &recovery_state,
                            &definition_scope,
                            &module.instances,
                        );
                        if let Ok(set_val) = eval_predicate(rhs, &ctx) {
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
        ConfigValue::Tuple(values) => Some(TlaValue::Seq(HashedArc::new(
            values.iter().filter_map(config_value_to_tla).collect(),
        ))),
        ConfigValue::Set(values) => Some(TlaValue::Set(HashedArc::new(
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
        | TemporalFormula::Implies(left, right)
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

        // A SELF-REFERENTIAL model-value constant (`NULL = NULL`, where the
        // constant name equals the model value) must NOT be injected as the
        // definition `NULL == NULL`. Now that constants are no longer carried in
        // the state (see `TlaModel::from_files`), resolving the name through that
        // definition would recurse to MAX_DEPTH; the `ModelValue(name)` identifier
        // fallback (present in both evaluators — compiled `eval_var_by_name` and
        // interpreted `resolve_identifier`) resolves it correctly instead.
        // Skipping only the self-referential case is deliberately narrow: an
        // ALIASED model value (`NoBlock = NoBlockVal`, name != value) injects the
        // NON-self-referential def `NoBlock == NoBlockVal`, which must be KEPT —
        // otherwise `NoBlock` would fall back to `ModelValue("NoBlock")` instead
        // of resolving to the intended `NoBlockVal`. (Model-value SETS such as
        // `Node = {n1, n2}` are `ConfigValue::Set`, not `ModelValue`, and inject
        // normally; their elements resolve via the same fallback.) This is also
        // safer than a body-equals-name guard in the evaluators, which would
        // silently reinterpret a user-written recursive `X == X` as a model value
        // instead of surfacing the recursion.
        if let ConfigValue::ModelValue(mv) = value {
            if mv == name {
                continue;
            }
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

        // Save original definition under a backup name before overriding.
        // This allows Init <- MCInit to still find assignments from the
        // original Init (e.g., passes = IF terminated THEN 0 ELSE -1).
        if let Some(original) = module.definitions.get(name) {
            let backup_name = format!("__Original_{}__", name);
            if !module.definitions.contains_key(&backup_name) {
                let mut backup = original.clone();
                backup.name = backup_name.clone();
                module.definitions.insert(backup_name, backup);
            }
        }

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
    let resolution_schema = crate::tla::value::get_resolution_schema();

    for (name, def) in definitions {
        // Only compile definitions that look like actions (contain primed variables or UNCHANGED)
        if looks_like_action(def) {
            let ir = compile_action_ir(def);
            let mut compiled_ir = CompiledActionIr::from_ir(&ir);
            if let Some(schema) = resolution_schema.as_ref() {
                resolve_state_vars_in_action_ir(&mut compiled_ir, schema.as_ref());
            }
            let compiled_ir = Arc::new(compiled_ir);
            compiled.insert(name.clone(), compiled_ir);
        }
    }

    compiled
}

/// Pre-compile a list of (name, expression) pairs into compiled expressions
fn precompile_expressions(exprs: &[(String, String)]) -> Vec<(String, Arc<CompiledExpr>)> {
    let resolution_schema = crate::tla::value::get_resolution_schema();
    exprs
        .iter()
        .map(|(name, expr)| {
            let mut compiled_expr = compile_expr(expr);
            if let Some(schema) = resolution_schema.as_ref() {
                resolve_state_vars(&mut compiled_expr, schema.as_ref());
            }
            let compiled = Arc::new(compiled_expr);
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
            // Use the same parameter-sensitive cache key as
            // `get_or_compile_action`.
            insert_compiled_action(def, Arc::clone(compiled_ir));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // Regression: a state predicate whose top operator is a `=>` (looser than
    // `/\`) whose ANTECEDENT is a `/\` block must be kept as ONE clause, NOT
    // shredded into a leading required conjunct + a `B => C` conjunct. This is
    // the MCBakery inductive-invariant-as-Init under-exploration (~110x).
    #[test]
    fn implies_rooted_state_predicate_is_not_shredded_on_and() {
        let defs: BTreeMap<String, TlaDefinition> = BTreeMap::new();
        let instances: BTreeMap<String, TlaModuleInstance> = BTreeMap::new();

        // `(A /\ B) => C` written with a leading `/\` in the antecedent.
        let clause = "/\\ (x = 1) /\\ (y = 2) => (z = 3)";
        let out = expand_state_predicate_clauses(clause, &defs, &instances);
        assert_eq!(
            out.len(),
            1,
            "implies-rooted predicate must stay one clause, got {out:?}"
        );
        assert!(
            out[0].contains("=>"),
            "the whole implication must be preserved, got {out:?}"
        );

        // Iff (`<=>`) root is likewise looser than `/\`.
        let iff = "/\\ (x = 1) /\\ (y = 2) <=> (z = 3)";
        let out_iff = expand_state_predicate_clauses(iff, &defs, &instances);
        assert_eq!(
            out_iff.len(),
            1,
            "iff-rooted predicate must stay one clause, got {out_iff:?}"
        );
    }

    // A genuine top-level conjunction must STILL split into its conjuncts —
    // the guard only fires for implication/iff roots.
    #[test]
    fn genuine_top_level_and_still_splits() {
        let defs: BTreeMap<String, TlaDefinition> = BTreeMap::new();
        let instances: BTreeMap<String, TlaModuleInstance> = BTreeMap::new();

        let clause = "/\\ (x = 1) /\\ (y = 2)";
        let out = expand_state_predicate_clauses(clause, &defs, &instances);
        assert_eq!(
            out.len(),
            2,
            "genuine top-level /\\ must split into 2 conjuncts, got {out:?}"
        );
    }

    // Safety: an implication whose ANTECEDENT contains a top-level membership
    // (`x \in S`) must NOT be kept whole — the enumerator harvests that `\in` as
    // a variable assignment, and stranding it inside the antecedent would leave
    // the variable unassigned ("Init does not assign"). Such a clause must still
    // reach the split path (the guard returns false).
    #[test]
    fn implies_with_membership_in_antecedent_is_not_kept_whole() {
        assert!(
            !state_predicate_root_is_looser_than_and(
                "/\\ x \\in 0..2 /\\ y \\in 0..2 => (x + y = 2)"
            ),
            "an implication whose antecedent has a top-level membership must not be kept whole"
        );
        // A bare `x \in S => C` (single-membership antecedent) is likewise unsafe.
        assert!(
            !state_predicate_root_is_looser_than_and("x \\in S => P"),
            "a bare-membership antecedent must not be kept whole"
        );
        // But a pure-predicate antecedent (no membership) IS kept whole.
        assert!(
            state_predicate_root_is_looser_than_and("/\\ (a = 1) /\\ (b = 2) => (c = 3)"),
            "a pure-predicate implication antecedent must be kept whole"
        );
    }

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
    fn box_safety_property_lowers_to_state_predicate() {
        // `[] P` with P a plain state predicate lowers to P.
        let f = TemporalFormula::parse("[] x > 0").unwrap();
        assert_eq!(extract_box_safety_invariant(&f).as_deref(), Some("x > 0"));

        // `[] \A i, j : P` — single `\A` falls through to StatePredicate.
        let f = TemporalFormula::parse("[] \\A i, j : consensus[i] = consensus[j]").unwrap();
        assert_eq!(
            extract_box_safety_invariant(&f).as_deref(),
            Some("\\A i, j : consensus[i] = consensus[j]")
        );

        // `[] (P /\ Q)` — conjunction of two state predicates.
        let f = TemporalFormula::parse("[] (a > 0 /\\ b > 0)").unwrap();
        assert!(matches!(f, TemporalFormula::Always(_)));
        let lowered = extract_box_safety_invariant(&f).unwrap();
        assert!(lowered.contains("a > 0") && lowered.contains("b > 0"));
    }

    #[test]
    fn box_safety_lowering_rejects_action_and_temporal_bodies() {
        // Action formula `[][Next]_vars` — must NOT be lowered.
        let f = TemporalFormula::parse("[][Next]_vars").unwrap();
        assert_eq!(extract_box_safety_invariant(&f), None);

        // Primed state predicate — must NOT be lowered.
        let f = TemporalFormula::Always(Box::new(TemporalFormula::StatePredicate(
            "x' = x + 1".to_string(),
        )));
        assert_eq!(extract_box_safety_invariant(&f), None);

        // Nested temporal `[] <> P` (InfinitelyOften) — not a box-safety pred.
        let f = TemporalFormula::parse("[]<> P").unwrap();
        assert_eq!(extract_box_safety_invariant(&f), None);

        // Liveness `<> P` — not Always at all.
        let f = TemporalFormula::parse("<> done").unwrap();
        assert_eq!(extract_box_safety_invariant(&f), None);

        // Body that embeds a temporal operator — reject conservatively.
        let f = TemporalFormula::Always(Box::new(TemporalFormula::StatePredicate(
            "P /\\ <>Q".to_string(),
        )));
        assert_eq!(extract_box_safety_invariant(&f), None);
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

    /// Regression (Disruptor family): a named `INSTANCE` whose constant is
    /// substituted to an infinite built-in set (`Values <- Int`), a record-set
    /// `TypeOk` invariant using `UNION { [D -> Values \union {NULL}] }`, and a
    /// `Next` action that routes a primed variable through an instance-action
    /// call inside a LET (`Buffer!Write(index, w, next)`). Before the fix, the
    /// checker halted at 1 distinct state reporting a false `TypeOk` violation
    /// (the codomain membership test errored on `ModelValue("Int")` and the
    /// record set was misparsed as a function set), and even with the invariant
    /// fixed the compiled action IR silently dropped every successor of the
    /// instance-action-in-LET action.
    #[test]
    fn instance_infinite_codomain_typeok_and_let_instance_action_explore() {
        let tmp = std::env::temp_dir().join("tlapp-disruptor-regression-test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("tmp dir");

        // Minimal RingBuffer-shaped instanced module.
        fs::write(
            tmp.join("MiniRing.tla"),
            r#"
---- MODULE MiniRing ----
LOCAL INSTANCE Naturals
CONSTANTS Size, Values, NULL
VARIABLE slot
Init == slot = [ i \in 0 .. (Size - 1) |-> NULL ]
IndexOf(seq) == seq % Size
Write(index, value) == slot' = [ slot EXCEPT ![index] = value ]
TypeOk == slot \in UNION { [0 .. (Size - 1) -> Values \union { NULL }] }
====
"#,
        )
        .expect("ring module written");

        let module = tmp.join("MiniTop.tla");
        fs::write(
            &module,
            r#"
---- MODULE MiniTop ----
EXTENDS Integers
CONSTANTS Size, NULL
VARIABLES slot, published

Buffer == INSTANCE MiniRing WITH Values <- Int

BeginWrite ==
  LET next  == published + 1
      index == Buffer!IndexOf(next)
  IN /\ published < 3
     /\ Buffer!Write(index, next)
     /\ published' = next

Init == Buffer!Init /\ published = -1
Next == BeginWrite
TypeOk == Buffer!TypeOk /\ published \in Int
====
"#,
        )
        .expect("top module written");

        let cfg = tmp.join("MiniTop.cfg");
        fs::write(
            &cfg,
            "INIT Init\nNEXT Next\n\
             CONSTANTS Size = 4 NULL = NULL\nINVARIANT TypeOk\n",
        )
        .expect("cfg written");

        let model =
            TlaModel::from_files(&module, Some(&cfg), None, None).expect("model should build");
        let init = model.initial_states();
        assert_eq!(init.len(), 1, "one initial state");

        // TypeOk must hold on the initial state (was a false violation).
        model
            .check_invariants(&init[0])
            .expect("TypeOk must hold on init");

        // The instance-action-in-LET action must produce a successor (was
        // silently dropped by the compiled action IR).
        let mut next = Vec::new();
        model.next_states(&init[0], &mut next);
        assert_eq!(next.len(), 1, "BeginWrite should yield exactly one successor");
        assert_eq!(next[0].get("published"), Some(&TlaValue::Int(0)));
        model
            .check_invariants(&next[0])
            .expect("TypeOk must hold on the successor");

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
