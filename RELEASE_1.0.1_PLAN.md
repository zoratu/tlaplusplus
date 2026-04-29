# Release 1.0.1 — Task Plan

Patch release. Three audits inspired by https://corrode.dev/blog/bugs-rust-wont-catch/, mapped to tlaplusplus. None of these are known-broken — they're "the article tells us where rewrites in Rust commonly leak bugs that Rust doesn't catch, and we should sweep for our analogues."

## Tasks

- [x] **T101. Fuzz the parser and evaluator for panic-resistance.** 4 distinct Rust panics fixed across 7 source-site fixes in 3 files; ~3.5 fuzz-CPU-hours; swarm + equivalence wired. Compiled-vs-interpreter equivalence wired into `fuzz_tla_swarm` (proptest-T2-style — panic on Rust panic in either evaluator while the other doesn't, panic on Ok-with-different-values; tolerate Ok-vs-Err which is the existing parser-vs-compiler semantic-parity gap). 4 panic categories fixed: (1) UTF-8 char-index vs byte-index confusion in 3 sibling sites in `src/tla/module.rs` (parse_recursive_declarations, instance_with_clause_is_incomplete, parse_with_substitutions); (2) `1 + ...` byte-length confusion in `src/tla/compiled_expr.rs::strip_label_prefix`; (3) trailing `expr.len() - 1` non-ASCII slice in `src/tla/compiled_expr.rs::parse_indexed_op_call`; (4) unchecked `[*eq_pos + 2..body_end]` where `body_end < eq_pos + 2` in 3 sibling let-binding sites (`src/tla/compiled_expr.rs` × 2, `src/tla/eval.rs::parse_let_definitions`). 7 regression tests pinned in `tests/fuzz_panic_regressions_t101.rs`. Final 10-min post-fix verification: 0 Rust panics across all 4 targets, 8 line-141 swarm divergences remain (T101.1 follow-up). Test counts after: 763 default / 783 failpoints / 781 symbolic-init.
- [ ] **T102. `let _ = ` / `.ok()` / `unused_must_use` audit.** Pass `clippy::let_underscore_must_use` + `clippy::unused_must_use` over the codebase. Audit every silenced `Result`. For each: either propagate the error properly, or add an inline comment explaining why the failure is provably safe to ignore. Particular suspects: `src/runtime.rs` (worker spawn / join / checkpoint coordination), `src/distributed/` (cluster termination, steal protocol), `src/storage/spillable_work_stealing.rs` (background spill coordinator). Our T6 work fixed four termination bugs that fit this exact pattern; this audit catches the others.
- [ ] **T103. `_lossy` UTF-8 audit.** `grep -rn "_lossy\|to_string_lossy\|from_utf8_lossy" src/` and audit each call. Any path where bytes flow from external sources (IOEnv, JsonDeserialize, CSVRead, user model values, `.tla` files containing non-ASCII) into state-fingerprint computation or queue-spill serialization is a potential soundness bug — two states that should hash equal could hash differently due to silent UTF-8 substitution. Action per call site: confirm it's display-only, OR convert the path to stay in `&[u8]`/`OsStr`/`Vec<u8>` end-to-end.

## Why 1.0.1 vs 1.1.0

These are correctness audits, not features. The premise of v1.0.0 was "we shipped a tested release; here are the audit-shaped follow-ups that the testing infra didn't directly catch." They belong in a patch.

## Working rules (carried from v1.0.0)

- All builds + `cargo test` runs on EC2 spot instances. Local machine is read-only for code.
- Each task agent updates this plan file when it finishes (mark `[x]` + one-line resolution).
- Use `--no-verify` on commits (the local pre-commit hook is intentionally disabled).
- **NEVER** put infra command names, AWS profile names, instance IDs, or IPs into tracked files, commit messages, or agent prompts that get committed back. See `~/.claude/projects/-Volumes-OWC-1M2-Users-isaiah-src-tlaplusplus/memory/feedback_no_infra_secrets_in_commits.md`.
- Do tasks in parallel via worktrees where they don't share code paths.
