# `src/tla/eval.rs` Refactor Plan

`src/tla/eval.rs` is **11,528 lines** containing **279 functions** and **240
tests** in a single trailing `#[cfg(test)] mod tests`. The file is the TLA+
interpreter — the *reference* for the compiler — so any drift introduced
during a split becomes a soundness bug (we keep finding these: T1.1, T1.4,
T1.5, T2.1, T2.2, T2.3, T2.4, T5.6, T101.1, T201).

This document specifies how to split the file into a `src/tla/eval/`
submodule tree without changing observable behavior. It is written as a
**Path-B deliverable**: the code split itself is deferred because the
quality bar requires running the full validation matrix
(~790 release / ~812 failpoints / ~813 symbolic-init / 13/13 diff-TLC /
PROPTEST_CASES=2048 × 3 seeds / chaos smoke) on a c8g.metal-24xl spot
between every move. Done correctly the refactor is a multi-day exercise;
done badly it is a permanent semantics regression. The doc captures the
exact target tree, the function-to-module mapping, the dependency graph,
and the specific coupling that has to be untangled before the moves can
land cleanly.

## Goals

1. Split the file into single-concern modules whose names map to TLA+
   surface concepts (arithmetic, sets, sequences, ...).
2. Keep behavior **byte-for-byte identical** — every test in the existing
   240-test suite continues to pass, every gate stays green.
3. Keep the public/`pub(crate)` surface **identical**. No widening "to
   make the move easier."
4. No semantic changes. Bugs spotted during the move become follow-up
   issues, never inline fixes.

## Non-goals

- Renaming or re-signing functions.
- Changing recursion shape (every `eval_*_inner` call site stays in place).
- Changing the budget plumbing or any thread-locals.
- Touching `compiled_eval.rs`, `compiled_expr.rs`, `symbolic_init.rs`,
  `action_exec.rs`, `models/tla_native.rs`, or `tla/mod.rs` re-exports
  beyond the minimum needed to compile against the new module path.

## External API surface (must remain stable)

`src/tla/mod.rs` re-exports the following from `eval`:

```rust
pub use eval::{
    EvalContext, TransitionContext, apply_action_ir, apply_action_ir_with_context,
    eval_action_body_multi, eval_action_constraint, eval_expr, eval_guard,
    eval_let_action_multi, normalize_param_name, restore_eval_budget,
    set_active_eval_budget,
};
```

External crate-internal callers reach further into `eval`:

- `compiled_eval.rs` imports `EvalContext`, `apply_value`,
  `eval_expr`, `eval_operator_call`, `normalize_param_name`,
  `eval_let_action_multi`, `parse_action_binder_specs`,
  `eval_action_body_multi`, `parse_action_call_expr`.
- `compiled_expr.rs` (test-only) imports `EvalContext`, `eval_expr`.
- `symbolic_init.rs` imports `EvalContext`,
  `find_top_level_char`, `find_top_level_keyword_index`,
  `is_valid_identifier`, `split_top_level_keyword`,
  `split_top_level_symbol`, `eval_expr`.
- `action_exec.rs` imports `apply_action_ir_with_context_multi`,
  `set_active_eval_budget`, `restore_eval_budget`.
- `models/tla_native.rs` imports
  `try_destructure_function_set_comprehension`,
  `try_resolve_funasseq_permutation_set`,
  `try_resolve_sequence_domain`, `find_top_level_char`,
  `find_top_level_keyword_index`, `is_valid_identifier`,
  `split_once_top_level`.
- `tests/symbolic_init_equivalence.rs` imports `EvalContext`.

Every name above must remain reachable at `crate::tla::eval::<name>` with
the same visibility after the split. The plan uses `pub(crate) use`
re-exports inside `eval/mod.rs` to preserve the path while letting the
implementation live in submodules.

## Target module tree

```
src/tla/eval/
├── mod.rs           (~ 450 lines)  EvalContext, TransitionContext, public entry
│                                   points (eval_expr, eval_guard,
│                                   eval_action_constraint), pub(crate) re-exports.
├── budget.rs        (~  40 lines)  ACTIVE_EVAL_BUDGET thread-local;
│                                   set_active_eval_budget, restore_eval_budget,
│                                   get_active_eval_budget, MAX_EVAL_DEPTH,
│                                   normalize_param_name.
├── transition.rs    (~ 130 lines)  TransitionContext impl + eval_transition_expr,
│                                   parse_simple_comparison, apply_comparison,
│                                   is_identifier (private to this module).
├── action.rs        (~1100 lines)  eval_let_action, eval_action_body_multi,
│                                   eval_let_action_multi*, apply_action_ir*,
│                                   eval_action_clauses_multi,
│                                   eval_action_clause_to_branch,
│                                   eval_action_body_text_multi,
│                                   eval_disjunctive_action_body_multi,
│                                   eval_stuttering_action_multi,
│                                   parse_action_call_expr,
│                                   seed_implicit_instance_constant_bindings,
│                                   bind_instance_substitutions,
│                                   resolved_instance_primed_substitution_value,
│                                   seed_parent_definition_fallbacks,
│                                   expand_action_call_multi,
│                                   eval_action_clause_text_multi,
│                                   eval_exists_action_multi,
│                                   expand_action_exists_branches,
│                                   ctx_with_staged_primes,
│                                   parse_action_binder_specs,
│                                   matches_membership_expr,
│                                   ActionEvalBranch (struct).
├── expr.rs          (~ 800 lines)  eval_expr_inner — the giant top-of-grammar
│                                   dispatcher. Stays in one piece because every
│                                   precedence rung is interleaved with budget
│                                   checks and recursion that all funnel back
│                                   through the same function.
├── control.rs       (~ 110 lines)  eval_if_expression, eval_case_expression,
│                                   eval_let_expression, split_outer_let,
│                                   parse_let_definitions, trim_let_edge_comments,
│                                   parse_local_def_head*, first_local_def_param_delims.
├── quantifier.rs    (~ 700 lines)  eval_quantifier_expression,
│                                   eval_choose_expression,
│                                   choose_candidates_without_domain,
│                                   stable_choose_model_value, eval_lambda_expression,
│                                   eval_implies_parts, BinderSpec,
│                                   parse_binder_pattern_vars, parse_binders,
│                                   assign_binder_value, remove_binder_assignments,
│                                   bind_param_value, evaluate_exists,
│                                   evaluate_forall, collect_function_mapping,
│                                   collect_binder_filter_set,
│                                   collect_binder_map_set, binder_key.
├── postfix.rs       (~ 270 lines)  parse_atom_with_postfix, eval_atom_with_postfix,
│                                   eval_primed_postfix_expr, parse_base.
├── set.rs           (~ 600 lines)  eval_set_expression, try_eval_record_set,
│                                   try_resolve_sequence_domain,
│                                   try_funasseq_wrapper_symbolic,
│                                   try_destructure_function_set_comprehension,
│                                   try_resolve_funasseq_permutation_set,
│                                   parse_funasseq_comprehension,
│                                   substitute_identifier_owned,
│                                   powerset, generate_permutations, permute,
│                                   extract_sum_range_constraint.
├── bracket.rs       (~ 250 lines)  eval_bracket_expression, eval_except_expression,
│                                   PathSegment, parse_except_path,
│                                   get_path_value, set_path_value,
│                                   parse_argument_list, eval_bracket_index_key,
│                                   apply_value.
├── instance.rs      (~ 100 lines)  eval_module_instance_call,
│                                   effective_instance_scope,
│                                   eval_module_instance_ref, eval_enabled.
├── operator.rs      (~1300 lines)  eval_operator_call (the 1257-line giant
│                                   match), eval_builtin_extremum,
│                                   eval_builtin_bounded_seq, eval_builtin_tlc_get,
│                                   is_user_defined_infix_operator,
│                                   tla_to_string, json_to_tla_value,
│                                   tla_value_to_json, gcd,
│                                   sequence_like_values, seq_or_string_concat,
│                                   parse_record_key, record_key_from_value.
├── splitter.rs      (~1500 lines)  All split_top_level_* helpers, find_top_level_*
│                                   helpers, take_bracket_group, take_angle_group,
│                                   parse_identifier_prefix,
│                                   parse_string_literal_prefix, parse_int_prefix,
│                                   has_word_boundaries, has_keyword_boundaries,
│                                   is_word_char, is_binary_operator, next_word,
│                                   find_outer_then, find_outer_else,
│                                   find_top_level_definition_eqs,
│                                   line_start_before, skip_leading_ws,
│                                   strip_outer_parens, is_wrapped_by,
│                                   starts_with_keyword, take_keyword_prefix,
│                                   split_indented_top_level_boolean,
│                                   normalize_multiline_boolean_indentation,
│                                   split_top_level_set_minus,
│                                   starts_with_tla_backslash_operator,
│                                   contains_top_level_keyword,
│                                   find_top_level_char_from,
│                                   is_valid_identifier.
└── tests/           (~5300 lines)  See "Test split" below. The 240-test block
                                    moves out of `eval.rs` into per-submodule
                                    `#[cfg(test)] mod tests` blocks colocated
                                    with the items they exercise.
```

Total ~11.5K lines redistributed; no line is dropped. The `tests/` count is
high because the existing test block is ~5,300 lines of the file (lines
7649–11528).

## Dependency graph (compile order, leaf → root)

```
splitter      (leaf — only reads &str)
budget        (leaf — Cell + Rc, no eval refs)
                    ↓
transition    (uses splitter, value, action_ir)
action.rs     (uses every other module — sits at the call site of
               eval_expr_inner, eval_operator_call, parse_binders, etc.)
                    ↓
operator      ←─┐
quantifier    ←─┤
control       ←─┤   all five mutually call eval_expr_inner via expr::
postfix       ←─┤   and call back into operator::eval_operator_call.
set           ←─┤
bracket       ←─┤
instance      ←─┘
                    ↓
expr          (top-of-grammar dispatcher; calls into ALL of the above)
                    ↓
mod.rs        (re-exports + EvalContext + public eval_expr/eval_guard
               which delegate to expr::eval_expr_inner)
```

The graph is **not a DAG**: `expr::eval_expr_inner` calls
`operator::eval_operator_call` (~6 sites) which calls back into
`expr::eval_expr_inner` (via `eval_expr`) for arguments that need lazy
re-evaluation. `quantifier::evaluate_forall`, `set::eval_set_expression`,
`bracket::eval_bracket_expression`, `control::eval_let_expression`,
`postfix::eval_atom_with_postfix`, and `instance::eval_module_instance_call`
all sit on the same mutual-recursion ring. Rust handles intra-crate cyclic
calls fine — it's the visibility plumbing that costs lines:

- Every cross-submodule call needs `pub(super)` or `pub(crate)`.
- Helpers that were `fn foo(...)` (private) become `pub(super) fn foo(...)`
  the moment a sibling submodule needs them. This is a controlled
  widening: the symbol stays invisible outside `eval/`.
- `eval_expr_inner` and `eval_operator_call` themselves become
  `pub(super)` so siblings can call them; `eval_expr` and
  `eval_operator_call` stay `pub(crate)` for `compiled_eval.rs` /
  `models/tla_native.rs`.

## Function-to-module mapping (full table)

For every numbered function in `eval.rs` the table records target file and
the new visibility. (Visibility today → visibility after.) Symbol names
with `pub(super)` are widened only across the new submodule boundary;
crate visibility is unchanged.

Listed in source order; line numbers are the *original* `eval.rs` lines.

| Line | Function                                                | Target           | Vis change            |
| ---- | ------------------------------------------------------- | ---------------- | --------------------- |
| 25   | `set_active_eval_budget`                                | `budget.rs`      | `pub` → `pub`         |
| 32   | `restore_eval_budget`                                   | `budget.rs`      | `pub` → `pub`         |
| 37   | `get_active_eval_budget`                                | `budget.rs`      | `fn` → `pub(super)`   |
| 51   | `normalize_param_name`                                  | `budget.rs`      | `pub` → `pub`         |
| 68   | `EvalContext`                                           | `mod.rs`         | `pub` → `pub`         |
| 82   | `TransitionContext`                                     | `mod.rs`         | `pub` → `pub`         |
| 90   | `impl EvalContext`                                      | `mod.rs`         | unchanged             |
| 281  | `impl TransitionContext`                                | `mod.rs`         | unchanged             |
| 330  | `eval_expr`                                             | `mod.rs`         | `pub` → `pub`         |
| 334  | `definition_as_lambda`                                  | `mod.rs`         | `fn` → `pub(super)`   |
| 345  | `eval_guard`                                            | `mod.rs`         | `pub` → `pub`         |
| 353  | `eval_action_constraint`                                | `transition.rs`  | `pub` → `pub`         |
| 369  | `eval_transition_expr`                                  | `transition.rs`  | unchanged (private)   |
| 418  | `is_identifier`                                         | `transition.rs`  | unchanged             |
| 444  | `parse_simple_comparison`                               | `transition.rs`  | unchanged             |
| 455  | `apply_comparison`                                      | `transition.rs`  | unchanged             |
| 481  | `ActionEvalBranch`                                      | `action.rs`      | unchanged (struct)    |
| 490  | `eval_let_action`                                       | `action.rs`      | `pub(crate)` → same   |
| 508  | `eval_action_body_multi`                                | `action.rs`      | `pub` → `pub`         |
| 526  | `eval_let_action_multi`                                 | `action.rs`      | `pub` → `pub`         |
| 542  | `eval_let_action_multi_branch`                          | `action.rs`      | unchanged             |
| 554  | `apply_action_ir`                                       | `action.rs`      | `pub` → `pub`         |
| 561  | `apply_action_ir_with_context_multi`                    | `action.rs`      | `pub(crate)` → same   |
| 584  | `apply_action_ir_with_context`                          | `action.rs`      | `pub` → `pub`         |
| 594  | `eval_action_clauses_multi`                             | `action.rs`      | unchanged             |
| 613  | `eval_action_clause_to_branch`                          | `action.rs`      | unchanged             |
| 672  | `eval_action_body_text_multi`                           | `action.rs`      | unchanged             |
| 698  | `eval_disjunctive_action_body_multi`                    | `action.rs`      | unchanged             |
| 740  | `eval_stuttering_action_multi`                          | `action.rs`      | unchanged             |
| 765  | `parse_action_call_expr`                                | `action.rs`      | `pub(crate)` → same   |
| 809  | `seed_implicit_instance_constant_bindings`              | `action.rs`      | unchanged             |
| 833  | `bind_instance_substitutions`                           | `action.rs`      | unchanged             |
| 852  | `resolved_instance_primed_substitution_value`           | `action.rs`      | unchanged             |
| 864  | `seed_parent_definition_fallbacks`                      | `action.rs`      | unchanged             |
| 883  | `expand_action_call_multi`                              | `action.rs`      | unchanged             |
| 962  | `eval_action_clause_text_multi`                         | `action.rs`      | unchanged             |
| 1048 | `eval_exists_action_multi`                              | `action.rs`      | unchanged             |
| 1058 | `expand_action_exists_branches`                         | `action.rs`      | unchanged             |
| 1087 | `ctx_with_staged_primes`                                | `action.rs`      | unchanged             |
| 1098 | `parse_action_binder_specs`                             | `action.rs`      | `pub(crate)` → same   |
| 1142 | `matches_membership_expr`                               | `action.rs`      | unchanged             |
| 1257 | `eval_expr_inner`                                       | `expr.rs`        | `fn` → `pub(super)`   |
| 1751 | `eval_implies_parts`                                    | `quantifier.rs`  | unchanged             |
| 1767 | `eval_if_expression`                                    | `control.rs`     | unchanged             |
| 1789 | `eval_case_expression`                                  | `control.rs`     | unchanged             |
| 1822 | `eval_quantifier_expression`                            | `quantifier.rs`  | unchanged             |
| 1863 | `eval_choose_expression`                                | `quantifier.rs`  | unchanged             |
| 1908 | `choose_candidates_without_domain`                      | `quantifier.rs`  | unchanged             |
| 1955 | `stable_choose_model_value`                             | `quantifier.rs`  | unchanged             |
| 1969 | `eval_lambda_expression`                                | `quantifier.rs`  | unchanged             |
| 2000 | `eval_let_expression`                                   | `control.rs`     | unchanged             |
| 2008 | `parse_atom_with_postfix`                               | `postfix.rs`     | `fn` → `pub(super)`   |
| 2053 | `eval_atom_with_postfix`                                | `postfix.rs`     | `fn` → `pub(super)`   |
| 2079 | `eval_primed_postfix_expr`                              | `postfix.rs`     | unchanged             |
| 2133 | `parse_base`                                            | `postfix.rs`     | unchanged             |
| 2295 | `eval_set_expression`                                   | `set.rs`         | `fn` → `pub(super)`   |
| 2670 | `try_resolve_sequence_domain`                           | `set.rs`         | `pub(crate)` → same   |
| 2713 | `try_funasseq_wrapper_symbolic`                         | `set.rs`         | unchanged (cfg)       |
| 2780 | `try_destructure_function_set_comprehension`            | `set.rs`         | `pub(crate)` → same   |
| 2869 | `try_resolve_funasseq_permutation_set`                  | `set.rs`         | `pub(crate)` → same   |
| 2923 | `parse_funasseq_comprehension`                          | `set.rs`         | unchanged             |
| 2970 | `substitute_identifier_owned`                           | `set.rs`         | unchanged             |
| 2994 | `eval_bracket_expression`                               | `bracket.rs`     | `fn` → `pub(super)`   |
| 3118 | `try_eval_record_set`                                   | `set.rs`         | unchanged             |
| 3230 | `is_valid_identifier`                                   | `splitter.rs`    | `pub(crate)` → same   |
| 3242 | `eval_except_expression`                                | `bracket.rs`     | unchanged             |
| 3274 | `apply_value`                                           | `bracket.rs`     | `pub(crate)` → same   |
| 3334 | `eval_module_instance_call`                             | `instance.rs`    | unchanged             |
| 3407 | `effective_instance_scope`                              | `instance.rs`    | unchanged             |
| 3419 | `eval_module_instance_ref`                              | `instance.rs`    | unchanged             |
| 3436 | `eval_enabled`                                          | `instance.rs`    | unchanged             |
| 3492 | `eval_operator_call`                                    | `operator.rs`    | `pub(crate)` → same   |
| 4748 | `eval_builtin_extremum`                                 | `operator.rs`    | unchanged             |
| 4779 | `eval_builtin_bounded_seq`                              | `operator.rs`    | unchanged             |
| 4825 | `eval_builtin_tlc_get`                                  | `operator.rs`    | unchanged             |
| 4840 | `split_outer_let`                                       | `control.rs`     | unchanged             |
| 4870 | `parse_let_definitions`                                 | `control.rs`     | unchanged             |
| 4916 | `trim_let_edge_comments`                                | `control.rs`     | unchanged             |
| 4942 | `parse_local_def_head`                                  | `control.rs`     | unchanged             |
| 4957 | `first_local_def_param_delims`                          | `control.rs`     | unchanged             |
| 4967 | `parse_local_def_head_with_delims`                      | `control.rs`     | unchanged             |
| 5009 | `BinderSpec`                                            | `quantifier.rs`  | unchanged (struct)    |
| 5014 | `parse_binder_pattern_vars`                             | `quantifier.rs`  | unchanged             |
| 5036 | `parse_binders`                                         | `quantifier.rs`  | `fn` → `pub(super)`   |
| 5082 | `assign_binder_value`                                   | `quantifier.rs`  | unchanged             |
| 5116 | `remove_binder_assignments`                             | `quantifier.rs`  | unchanged             |
| 5122 | `bind_param_value`                                      | `quantifier.rs`  | unchanged             |
| 5140 | `evaluate_exists`                                       | `quantifier.rs`  | unchanged             |
| 5172 | `evaluate_forall`                                       | `quantifier.rs`  | unchanged             |
| 5204 | `collect_function_mapping`                              | `quantifier.rs`  | unchanged             |
| 5241 | `collect_binder_filter_set`                             | `quantifier.rs`  | unchanged             |
| 5288 | `collect_binder_map_set`                                | `quantifier.rs`  | unchanged             |
| 5333 | `binder_key`                                            | `quantifier.rs`  | unchanged             |
| 5368 | `PathSegment`                                           | `bracket.rs`     | unchanged (enum)      |
| 5373 | `parse_except_path`                                     | `bracket.rs`     | unchanged             |
| 5404 | `get_path_value`                                        | `bracket.rs`     | unchanged             |
| 5444 | `set_path_value`                                        | `bracket.rs`     | unchanged             |
| 5496 | `parse_argument_list`                                   | `bracket.rs`     | `fn` → `pub(super)`   |
| 5513 | `eval_bracket_index_key`                                | `bracket.rs`     | unchanged             |
| 5523 | `gcd`                                                   | `operator.rs`    | unchanged             |
| 5527 | `sequence_like_values`                                  | `operator.rs`    | unchanged             |
| 5545 | `seq_or_string_concat`                                  | `operator.rs`    | `fn` → `pub(super)`   |
| 5568 | `parse_record_key`                                      | `bracket.rs`     | unchanged             |
| 5592 | `record_key_from_value`                                 | `operator.rs`    | unchanged             |
| 5601 | `powerset`                                              | `set.rs`         | unchanged             |
| 5640 | `generate_permutations`                                 | `set.rs`         | unchanged             |
| 5651 | `permute`                                               | `set.rs`         | unchanged             |
| 5664 | `tla_to_string`                                         | `operator.rs`    | unchanged             |
| 5710 | `strip_outer_parens`                                    | `splitter.rs`    | `fn` → `pub(super)`   |
| 5718 | `is_wrapped_by`                                         | `splitter.rs`    | unchanged             |
| 5763 | `starts_with_keyword`                                   | `splitter.rs`    | `fn` → `pub(super)`   |
| 5774 | `take_keyword_prefix`                                   | `splitter.rs`    | `fn` → `pub(super)`   |
| 5785 | `split_top_level_symbol`                                | `splitter.rs`    | `pub(crate)` → same   |
| 5789 | `split_top_level_keyword`                               | `splitter.rs`    | `pub(crate)` → same   |
| 5793 | `split_top_level`                                       | `splitter.rs`    | `fn` → `pub(super)`   |
| 6038 | `split_indented_top_level_boolean`                      | `splitter.rs`    | unchanged             |
| 6122 | `normalize_multiline_boolean_indentation`               | `splitter.rs`    | unchanged             |
| 6168 | `split_top_level_set_minus`                             | `splitter.rs`    | `fn` → `pub(super)`   |
| 6260 | `starts_with_tla_backslash_operator`                    | `splitter.rs`    | unchanged             |
| 6286 | `split_top_level_comparison`                            | `splitter.rs`    | `fn` → `pub(super)`   |
| 6485 | `split_top_level_range`                                 | `splitter.rs`    | `fn` → `pub(super)`   |
| 6567 | `split_top_level_additive`                              | `splitter.rs`    | `fn` → `pub(super)`   |
| 6645 | `split_top_level_multiplicative`                        | `splitter.rs`    | `fn` → `pub(super)`   |
| 6750 | `split_once_top_level`                                  | `splitter.rs`    | `pub(crate)` → same   |
| 6819 | `split_top_level_defined_infix`                         | `splitter.rs`    | `fn` → `pub(super)`   |
| 6919 | `is_user_defined_infix_operator`                        | `operator.rs`    | unchanged             |
| 6966 | `find_top_level_keyword_index`                          | `splitter.rs`    | `pub(crate)` → same   |
| 7041 | `contains_top_level_keyword`                            | `splitter.rs`    | `fn` → `pub(super)`   |
| 7045 | `find_top_level_char`                                   | `splitter.rs`    | `pub(crate)` → same   |
| 7049 | `find_top_level_char_from`                              | `splitter.rs`    | unchanged             |
| 7123 | `take_bracket_group`                                    | `splitter.rs`    | `fn` → `pub(super)`   |
| 7178 | `take_angle_group`                                      | `splitter.rs`    | `fn` → `pub(super)`   |
| 7239 | `parse_identifier_prefix`                               | `splitter.rs`    | `fn` → `pub(super)`   |
| 7267 | `parse_string_literal_prefix`                           | `splitter.rs`    | `fn` → `pub(super)`   |
| 7304 | `parse_int_prefix`                                      | `splitter.rs`    | `fn` → `pub(super)`   |
| 7331 | `has_word_boundaries`                                   | `splitter.rs`    | unchanged             |
| 7341 | `has_keyword_boundaries`                                | `splitter.rs`    | unchanged             |
| 7349 | `is_word_char`                                          | `splitter.rs`    | unchanged             |
| 7353 | `is_binary_operator`                                    | `splitter.rs`    | unchanged             |
| 7373 | `next_word`                                             | `splitter.rs`    | unchanged             |
| 7401 | `find_outer_then`                                       | `splitter.rs`    | `fn` → `pub(super)`   |
| 7416 | `find_outer_else`                                       | `splitter.rs`    | `fn` → `pub(super)`   |
| 7431 | `find_top_level_definition_eqs`                         | `splitter.rs`    | `fn` → `pub(super)`   |
| 7522 | `line_start_before`                                     | `splitter.rs`    | unchanged             |
| 7529 | `skip_leading_ws`                                       | `splitter.rs`    | unchanged             |
| 7544 | `extract_sum_range_constraint`                          | `set.rs`         | unchanged             |
| 7589 | `json_to_tla_value`                                     | `operator.rs`    | unchanged             |
| 7619 | `tla_value_to_json`                                     | `operator.rs`    | unchanged             |
| 7649 | `mod tests`                                             | per-submodule    | see Test split        |

**Summary of vis widening**: 32 functions go from `fn` to `pub(super)`.
None gain `pub(crate)` or `pub` they did not already have.

## The coupling that blocks a clean Path-A landing

Three categories of coupling make the move riskier than line counts
suggest. None are unsolvable; they are why the move has to be staged with
a green build between every chunk and why Path B is the safe call when
the validation matrix can only run remotely.

### 1. `eval_expr_inner` is a ~492-line precedence ladder

It is a single function (`expr.rs:1257-1748`) that hand-rolls TLA+'s
operator precedence by sequentially trying every splitter
(`split_top_level_symbol`, `split_top_level_keyword`,
`split_indented_top_level_boolean`, `split_top_level_set_minus`,
`split_top_level_comparison`, `split_top_level_range`,
`split_top_level_additive`, `split_top_level_multiplicative`,
`split_top_level_defined_infix`, `split_once_top_level`) in a precise
order. **Reordering any pair of branches changes parse semantics.** The
T1.6 regression in the file (lines 1346–1349 comment) is exactly this:
splitting `<=>` before `=>` matters because `<=>` contains `=>`. The
function therefore moves to `expr.rs` as a single contiguous block; it
is not internally splittable without introducing an explicit precedence
table, which is a separate refactor.

### 2. `eval_operator_call` is a 1,257-line `match name`

It dispatches every TLA+ standard-library operator (`Cardinality`,
`Max`, `Min`, `Append`, `Head`, `Tail`, `Len`, `DOMAIN`,
`SubSeq`, `SetToSeq`, `Permutations`, `FunAsSeq`, `Print`, `PrintT`,
`Assert`, `IsFiniteSet`, `Bag*`, `IO*`, `Bitwise*`, `Combinations`,
`VectorClock*`, `Random*`, etc., plus user-defined operator inlining at
the bottom). Many arms fall through to user-defined-operator inlining
that re-enters `expr::eval_expr_inner` on the operator body. Splitting
the match by category (sequences vs sets vs bags vs IO) would multiply
the dispatch cost (string compare in N small files instead of one big
file) and force a crate of new helper functions to centralize the
fallthrough. The function therefore moves intact to `operator.rs`.
(Future work: replace string match with an `OperatorId` enum table —
out of scope for this refactor.)

### 3. The mutual-recursion ring

`eval_expr_inner ↔ eval_operator_call ↔ eval_set_expression ↔
eval_bracket_expression ↔ eval_let_expression ↔ eval_quantifier_expression
↔ evaluate_forall ↔ collect_binder_filter_set` (and back) all reach each
other. Splitting them into seven submodules is fine — Rust resolves
intra-crate cycles — but each call site needs `super::sibling::name` and
each callee needs `pub(super)`. There is no order in which the new
modules compile cleanly one at a time; they have to land as one
atomic change with the test suite green at the boundary.

## Test split

The 240 tests in the trailing `mod tests` (lines 7649–11528, ~5,300
lines) split by what they exercise:

- `splitter::tests` — anything calling `split_top_level_*`,
  `find_top_level_*`, `take_bracket_group`, `parse_int_prefix`,
  `is_valid_identifier`. Estimated 70-80 tests.
- `expr::tests` — `evals_arithmetic_and_boolean`,
  `evals_cup_and_cap_aliases`, precedence-ladder tests, range tests,
  comparison tests. Estimated 40-50 tests.
- `set::tests` — set comprehensions, record-set fast path,
  Permutation/FunAsSeq destructuring. Estimated 25-30 tests.
- `quantifier::tests` — `\A`, `\E`, CHOOSE, LAMBDA, binder parsing.
  Estimated 30-35 tests.
- `bracket::tests` — `[...]`, EXCEPT, function/record literals,
  function application. Estimated 20-25 tests.
- `operator::tests` — every builtin operator unit test. Estimated 30-35 tests.
- `action.rs::tests` — multi-action evaluation, primed propagation,
  `\cdot` composition. Estimated 15-20 tests.
- `transition.rs::tests` — `eval_action_constraint` cases. Estimated 5-10 tests.
- `mod.rs::tests` — `EvalContext` constructor / budget interaction
  tests, `normalizes_binder_and_higher_order_param_names`.
  Estimated 10-15 tests.

Each test today calls `super::*`. After the move the tests use
`super::*` referring to their *new* module — but several reach across
boundaries (`evals_arithmetic_and_boolean` calls `eval_expr` from
`mod.rs` and triggers `expr::eval_expr_inner` indirectly through the
public entry; that works as-is). Tests that touch only-private siblings
of a different submodule have to either stay in their original module
or have their target widened to `pub(super)` only — the table above
already accounts for the helpers that get widened.

## Landing protocol (when validation hardware is available)

To deliver Path A safely, run the following on a c8g.metal-24xl spot
between every chunk; nothing local. **A red gate at any step rolls back
that chunk and re-evaluates.**

```
gate-1  cargo test --release
gate-2  cargo test --release --features failpoints
gate-3  cargo test --release --features symbolic-init
gate-4  scripts/diff_tlc.sh                                # 13/13
gate-5  PROPTEST_CASES=2048 cargo test --release \
            --test compiled_vs_interpreted                 # 3 seeds
gate-6  cargo test --release --test state_graph_snapshots
gate-7  scripts/chaos_smoke.sh                             # 5-min, gates >= 6 fp fire
```

Move chunks (each ends with all 7 gates green; commit only when green):

1. **Chunk A — leaf modules**: create `eval/` directory; move
   `splitter.rs` and `budget.rs` first. These have zero inbound deps
   from other new submodules; only `eval.rs` (now `expr.rs` shell)
   imports them. Keep `eval.rs` as the shell that re-imports from the
   two new files via `mod splitter; mod budget; use splitter::*; use
   budget::*;`. Run gates 1-7. Commit.

2. **Chunk B — transition + control**: move `transition.rs`,
   `control.rs`. Both have low coupling outward (control needs splitter
   only; transition needs splitter + value). Gates. Commit.

3. **Chunk C — postfix + bracket + instance**: these three lift
   together because `parse_atom_with_postfix` (in postfix) calls
   `eval_bracket_expression` (in bracket) which calls
   `eval_module_instance_call` (in instance). Each calls
   `expr::eval_expr_inner` which still lives in the shell. Widen
   the necessary fns to `pub(super)`. Gates. Commit.

4. **Chunk D — set + quantifier**: these two lift together because
   `eval_set_expression` calls `parse_binders` (quantifier) and
   `evaluate_forall` (quantifier) calls back into
   `eval_expr_inner`. Gates. Commit.

5. **Chunk E — operator**: lifts on its own; only inbound dep is
   `expr::eval_expr_inner`. Gates. Commit.

6. **Chunk F — action**: largest single chunk, mostly self-contained
   action evaluation. Gates. Commit.

7. **Chunk G — expr.rs**: the remaining shell becomes
   `expr.rs` containing only `eval_expr_inner`. Promote
   `EvalContext`/`TransitionContext`/`eval_expr`/`eval_guard` to
   `mod.rs`. Add the `pub(crate) use` re-exports listed under
   "External API surface" so external paths
   (`crate::tla::eval::find_top_level_char`, etc.) resolve unchanged.
   Gates. Commit.

8. **Chunk H — test split**: move tests into per-submodule blocks.
   This is the lowest-risk chunk (compilation + test harness catches
   any path break) but the largest line move. Gates. Commit.

Total: 8 gated commits, each green across all 7 gates. Estimated
remote-spot wall time per gate run ~25-40 minutes — gates 1-3 dominate.
A clean run is ~6 hours of build/test on the spot; with the inevitable
yellow-then-green retries closer to 12.

## Why Path B today

The 12-hour budget covers **either** the full doc **or** the chunked
landing — not both, given the constraint that cargo only runs on the
remote spot. Without a green test matrix between every chunk this is
exactly the "half-shipped refactor that swallows errors or changes
evaluator semantics" the quality bar forbids. The doc is the safer
deliverable: it pre-pays the design cost so the next session can land
the chunks mechanically with the gate matrix as the safety net.
