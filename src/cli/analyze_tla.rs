//! `analyze-tla` subcommand handler — runs static probes against a TLA+ spec.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::models::tla_native::TlaModel;
use crate::tla::action_exec::probe_next_disjuncts_with_instances;
use crate::tla::module::TlaModuleInstance;
use crate::tla::{
    ActionClause, ClauseKind, ConfigValue, EvalContext, TlaConfig, TlaDefinition, TlaModule,
    TlaState, TlaValue, classify_clause, compile_action_ir, compile_action_ir_branches,
    eval_action_body_multi, eval_expr, eval_let_action_multi, looks_like_action,
    normalize_operator_ref_name, normalize_param_name, parse_action_exists,
    parse_stuttering_action_expr, parse_tla_config, parse_tla_module_file, restore_eval_budget,
    scan_module_closure, set_active_eval_budget, split_action_body_disjuncts, split_top_level,
};

use super::probe::*;
use super::shared::{
    config_value_to_tla, format_num, inject_constants_into_definitions,
};

pub(crate) fn handle(
    module: std::path::PathBuf,
    config: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let mut parsed_module = parse_tla_module_file(&module)?;
    let scan = scan_module_closure(&module)?;
    let parsed_cfg = match config.as_ref() {
        Some(cfg_path) => {
            let raw = std::fs::read_to_string(cfg_path)?;
            Some(parse_tla_config(&raw)?)
        }
        None => None,
    };
    // Inject constants from config into module definitions (handles OperatorRef)
    if let Some(cfg) = parsed_cfg.as_ref() {
        inject_constants_into_definitions(&mut parsed_module, cfg);
    }
    let analyzed_model = TlaModel::from_files(&module, config.as_deref(), None, None).ok();
    if let Some(model) = analyzed_model.as_ref() {
        parsed_module = model.module.clone();
    }
    let resolved_init_name = analyzed_model
        .as_ref()
        .map(|model| model.init_name.clone())
        .unwrap_or_else(|| "Init".to_string());
    let resolved_next_name = analyzed_model
        .as_ref()
        .map(|model| model.next_name.clone())
        .unwrap_or_else(|| "Next".to_string());
    println!("entry_module={}", module.display());
    println!(
        "parsed_module_name={} parsed_constants={} parsed_variables={} parsed_definitions={}",
        parsed_module.name,
        parsed_module.constants.len(),
        parsed_module.variables.len(),
        parsed_module.definitions.len()
    );
    println!(
        "parsed_has_init={} parsed_has_next={} parsed_has_spec={}",
        parsed_module.definitions.contains_key("Init"),
        parsed_module.definitions.contains_key("Next"),
        parsed_module.definitions.contains_key("Spec")
    );
    println!("resolved_init={resolved_init_name}");
    println!("resolved_next={resolved_next_name}");
    println!(
        "resolved_initial_states={}",
        analyzed_model
            .as_ref()
            .map(|model| model.initial_states_vec.len())
            .unwrap_or(0)
    );
    if let Some(next_def) = parsed_module.definitions.get(&resolved_next_name) {
        let disjuncts = split_top_level(&next_def.body, "\\/");
        println!("next_top_level_disjuncts={}", disjuncts.len());

        let mut max_conjuncts = 0usize;
        let mut total_primed_assignments = 0usize;
        let mut total_unchanged_clauses = 0usize;
        for branch in &disjuncts {
            let conjuncts = split_top_level(branch, "/\\");
            max_conjuncts = max_conjuncts.max(conjuncts.len());
            for clause in conjuncts {
                match classify_clause(&clause) {
                    ClauseKind::PrimedAssignment { .. }
                    | ClauseKind::PrimedMembership { .. } => {
                        total_primed_assignments += 1;
                    }
                    ClauseKind::Unchanged { .. } => {
                        total_unchanged_clauses += 1;
                    }
                    _ => {}
                }
            }
        }
        println!("next_max_top_level_conjuncts={}", max_conjuncts);
        println!(
            "next_detected_primed_assignments={}",
            total_primed_assignments
        );
        println!(
            "next_detected_unchanged_clauses={}",
            total_unchanged_clauses
        );
    }

    let mut action_defs = 0usize;
    let mut action_conjunct_max = 0usize;
    let mut action_primed_assignments = 0usize;
    let mut action_unchanged_clauses = 0usize;
    let mut action_guard_clauses = 0usize;
    for def in parsed_module.definitions.values() {
        if !definition_is_contextually_probeable_action(
            def,
            &parsed_module.definitions,
            &parsed_module.instances,
            &mut BTreeSet::new(),
        ) {
            continue;
        }
        action_defs += 1;
        let ir = compile_action_ir(def);
        action_conjunct_max = action_conjunct_max.max(ir.clauses.len());
        for clause in ir.clauses {
            match clause {
                ActionClause::PrimedAssignment { .. }
                | ActionClause::PrimedMembership { .. } => action_primed_assignments += 1,
                ActionClause::Unchanged { .. } => action_unchanged_clauses += 1,
                ActionClause::Guard { .. } => action_guard_clauses += 1,
                ActionClause::Exists { .. } => action_guard_clauses += 1,
                ActionClause::LetWithPrimes { .. } => action_guard_clauses += 1,
            }
        }
    }
    println!("action_like_definitions={}", action_defs);
    println!("action_conjunct_max={}", action_conjunct_max);
    println!(
        "action_detected_primed_assignments={}",
        action_primed_assignments
    );
    println!(
        "action_detected_unchanged_clauses={}",
        action_unchanged_clauses
    );
    println!("action_detected_guard_clauses={}", action_guard_clauses);

    let mut probe_state: TlaState = analyzed_model
        .as_ref()
        .and_then(|model| model.initial_states_vec.first().cloned())
        .unwrap_or_default();
    if let Some(cfg) = parsed_cfg.as_ref() {
        let mut deferred_operator_refs = Vec::new();
        for (k, v) in &cfg.constants {
            match v {
                ConfigValue::OperatorRef(name) => {
                    deferred_operator_refs
                        .push((k.clone(), normalize_operator_ref_name(name).to_string()));
                }
                _ => {
                    if let Some(tv) = config_value_to_tla(v) {
                        probe_state.insert(Arc::from(k.as_str()), tv);
                    }
                }
            }
        }
        for _ in 0..deferred_operator_refs.len().saturating_add(1) {
            if deferred_operator_refs.is_empty() {
                break;
            }
            let mut progress = false;
            let mut next_deferred = Vec::new();
            for (name, ref_name) in deferred_operator_refs {
                let ctx = build_probe_eval_context(
                    &probe_state,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                );
                if let Ok(value) = eval_expr(&ref_name, &ctx) {
                    probe_state.insert(Arc::from(name.as_str()), value);
                    progress = true;
                } else if let Some(value) =
                    representative_value_from_set_expr(&ref_name, &ctx)
                {
                    probe_state.insert(Arc::from(name.as_str()), value);
                    progress = true;
                } else {
                    next_deferred.push((name, ref_name));
                }
            }
            if !progress {
                break;
            }
            deferred_operator_refs = next_deferred;
        }
    }
    // Set evaluation budget for all probing (Init seeding, branch probing, expr probing).
    // This prevents exponential blowup from Seq, SUBSET, [D -> R], etc.
    let prev_budget = set_active_eval_budget(100_000);
    let mut probe_init_seeded = 0usize;
    let mut probe_init_unresolved = 0usize;
    let mut probe_init_unresolved_vars: Vec<String> = Vec::new();
    if let Some(init_def) = parsed_module.definitions.get(&resolved_init_name) {
        // Collect both equality and membership assignments
        let mut pending_eq: Vec<(String, String)> = Vec::new();
        let mut pending_mem: Vec<(String, String)> = Vec::new();
        for clause in expand_state_predicate_clauses(
            &init_def.body,
            &parsed_module.definitions,
            &parsed_module.instances,
        ) {
            match classify_clause(&clause) {
                ClauseKind::UnprimedEquality { var, expr } => {
                    pending_eq.push((var, expr));
                }
                ClauseKind::UnprimedMembership { var, set_expr } => {
                    pending_mem.push((var, set_expr));
                }
                _ => {}
            }
        }

        // Process both types together with fixed-point iteration
        let total_pending = pending_eq.len() + pending_mem.len();
        for _ in 0..total_pending.saturating_add(1) {
            if pending_eq.is_empty() && pending_mem.is_empty() {
                break;
            }
            let mut progress = false;

            // Process equality assignments
            let mut next_pending_eq = Vec::new();
            for (var, expr) in pending_eq {
                let ctx = build_probe_eval_context(
                    &probe_state,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                );
                match eval_expr(&expr, &ctx) {
                    Ok(value) => {
                        probe_state.insert(Arc::from(var.as_str()), value);
                        probe_init_seeded += 1;
                        progress = true;
                    }
                    Err(_) => next_pending_eq.push((var, expr)),
                }
            }

            // Process membership assignments - pick representative value
            let mut next_pending_mem = Vec::new();
            for (var, set_expr) in pending_mem {
                let ctx = build_probe_eval_context(
                    &probe_state,
                    &parsed_module.definitions,
                    &parsed_module.instances,
                );
                match eval_expr(&set_expr, &ctx) {
                    Ok(set_val) => {
                        // Pick a representative from the set
                        if let Some(repr) = pick_representative_from_set(&set_val) {
                            probe_state.insert(Arc::from(var.as_str()), repr);
                            probe_init_seeded += 1;
                            progress = true;
                        } else {
                            // Set was empty or not a set, keep pending
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                    Err(_) => {
                        if let Some(repr) =
                            representative_value_from_set_expr(&set_expr, &ctx)
                        {
                            probe_state.insert(Arc::from(var.as_str()), repr);
                            probe_init_seeded += 1;
                            progress = true;
                        } else {
                            next_pending_mem.push((var, set_expr));
                        }
                    }
                }
            }

            if !progress {
                probe_init_unresolved = next_pending_eq.len() + next_pending_mem.len();
                probe_init_unresolved_vars = next_pending_eq
                    .iter()
                    .chain(next_pending_mem.iter())
                    .map(|(var, _)| var.clone())
                    .collect();
                break;
            }
            pending_eq = next_pending_eq;
            pending_mem = next_pending_mem;
        }
    }
    probe_init_seeded += seed_probe_state_from_type_invariants(
        &mut probe_state,
        &parsed_module,
        parsed_cfg.as_ref(),
    );
    for var in &parsed_module.variables {
        probe_state
            .entry(Arc::from(var.as_str()))
            .or_insert(TlaValue::Int(0));
    }
    println!("probe_init_seeded={probe_init_seeded}");
    println!("probe_init_unresolved={probe_init_unresolved}");
    if !probe_init_unresolved_vars.is_empty() {
        println!(
            "probe_init_unresolved_vars={}",
            probe_init_unresolved_vars.join(",")
        );
    }
    println!("modules_scanned={}", scan.modules.len());
    println!("operators_total={}", scan.operator_names.len());
    println!("features_total={}", scan.combined_features.len());
    println!("native_frontend.module_parse=true");
    println!("native_frontend.cfg_parse={}", config.is_some());
    println!("native_frontend.value_domain=true");
    let mut action_eval_ready = false;
    if let Some(next_def) = parsed_module.definitions.get(&resolved_next_name) {
        let next_probe = probe_next_disjuncts_with_instances(
            &next_def.body,
            &parsed_module.definitions,
            if parsed_module.instances.is_empty() {
                None
            } else {
                Some(&parsed_module.instances)
            },
            &probe_state,
        );
        println!("next_branch_probe_total={}", next_probe.total_disjuncts);
        println!(
            "next_branch_probe_supported={}",
            next_probe.supported_disjuncts
        );
        println!(
            "next_branch_probe_generated_successors={}",
            next_probe.generated_successors
        );
        println!(
            "next_branch_probe_unsupported={}",
            next_probe
                .total_disjuncts
                .saturating_sub(next_probe.supported_disjuncts)
        );
        action_eval_ready = next_probe.total_disjuncts > 0
            && next_probe.supported_disjuncts == next_probe.total_disjuncts;
        if !next_probe.failures.is_empty() {
            println!("--- next_branch_probe_errors ---");
            for (idx, (err, count)) in next_probe.failures.iter().enumerate() {
                if idx >= 10 {
                    break;
                }
                println!("error_count={count} error={err}");
            }
        }
    }
    let mut expr_total = 0usize;
    let mut expr_ok = 0usize;
    let mut expr_errors: BTreeMap<String, u64> = BTreeMap::new();
    let mut expr_error_examples: BTreeMap<String, String> = BTreeMap::new();
    let action_param_samples = infer_action_param_samples_from_module_contexts(
        &resolved_next_name,
        &parsed_module.definitions,
        &parsed_module.instances,
        &probe_state,
    );
    for def in parsed_module.definitions.values() {
        if !definition_is_contextually_probeable_action(
            def,
            &parsed_module.definitions,
            &parsed_module.instances,
            &mut BTreeSet::new(),
        ) {
            continue;
        }
        for ir in compile_action_ir_branches(def) {
            let mut ctx = build_action_expr_probe_context(
                &probe_state,
                &parsed_module.definitions,
                &parsed_module.instances,
                &ir.params,
                &ir.clauses,
                action_param_samples.get(&def.name),
            );
            for clause in &ir.clauses {
                let Some(result) = probe_action_clause_expr(clause, &mut ctx) else {
                    continue;
                };
                expr_total += 1;
                match result {
                    Ok(()) => expr_ok += 1,
                    Err(err) if is_probe_sampling_limitation_error(&err) => {
                        // Treat probe-sampling-limitation errors as OK.
                        // These errors arise because the probe state uses
                        // default/ModelValue placeholders that don't match
                        // the actual runtime domains.
                        expr_ok += 1;
                    }
                    Err(err) => {
                        let key = err.to_string();
                        *expr_errors.entry(key).or_insert(0) += 1;
                        expr_error_examples
                            .entry(err.to_string())
                            .or_insert_with(|| {
                                let expr = match clause {
                                    ActionClause::Guard { expr }
                                    | ActionClause::PrimedAssignment { expr, .. }
                                    | ActionClause::PrimedMembership {
                                        set_expr: expr,
                                        ..
                                    }
                                    | ActionClause::LetWithPrimes { expr } => expr,
                                    ActionClause::Exists { binders, body } => {
                                        return format!(
                                            "def={} expr=\\\\E {}: {}",
                                            def.name,
                                            binders.replace('\n', " "),
                                            body.replace('\n', " ")
                                        );
                                    }
                                    ActionClause::Unchanged { .. } => unreachable!(),
                                };
                                format!("def={} expr={}", def.name, expr.replace('\n', " "))
                            });
                    }
                }
            }
        }
    }
    restore_eval_budget(prev_budget);
    let expr_eval_ready = expr_probe_is_ready(expr_total, expr_ok);
    println!("native_frontend.expr_eval={expr_eval_ready}");
    // If the spec has no Next definition, action_eval is not applicable
    let has_next = parsed_module.definitions.contains_key(&resolved_next_name);
    if has_next {
        println!("native_frontend.action_eval={action_eval_ready}");
    } else {
        println!("native_frontend.action_eval=na");
    }
    println!(
        "native_frontend.unsupported_feature_count={}",
        scan.combined_features.len()
    );
    println!("expr_probe_total={expr_total}");
    println!("expr_probe_ok={expr_ok}");
    println!("expr_probe_failed={}", expr_total.saturating_sub(expr_ok));
    if !expr_errors.is_empty() {
        println!("--- expr_probe_errors ---");
        for (idx, (err, count)) in expr_errors.iter().enumerate() {
            if idx >= 20 {
                break;
            }
            println!("error_count={count} error={err}");
            if let Some(example) = expr_error_examples.get(err) {
                println!("error_example={example}");
            }
        }
    }
    println!("--- modules ---");
    for m in &scan.modules {
        println!(
            "module={} path={} operators={} features={}",
            if m.module_name.is_empty() {
                "unknown"
            } else {
                m.module_name.as_str()
            },
            m.path.display(),
            m.operators.len(),
            m.features.len()
        );
    }
    println!("--- feature_counts ---");
    for (feature, count) in &scan.combined_features {
        println!("{feature}={count}");
    }
    println!("--- operators ---");
    for op in &scan.operator_names {
        println!("{op}");
    }

    if let Some(cfg) = parsed_cfg.as_ref() {
        println!("--- config ---");
        println!(
            "specification={}",
            cfg.specification.as_deref().unwrap_or("none")
        );
        println!("init={}", cfg.init.as_deref().unwrap_or("none"));
        println!("next={}", cfg.next.as_deref().unwrap_or("none"));
        println!("symmetry={}", cfg.symmetry.as_deref().unwrap_or("none"));
        println!(
            "check_deadlock={}",
            cfg.check_deadlock
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
        println!("constants_count={}", cfg.constants.len());
        println!("invariants_count={}", cfg.invariants.len());
        println!("properties_count={}", cfg.properties.len());
        println!("constraints_count={}", cfg.constraints.len());
        println!("action_constraints_count={}", cfg.action_constraints.len());
        for (k, v) in &cfg.constants {
            println!("constant.{k}={v:?}");
        }
    }
    Ok(())
}
