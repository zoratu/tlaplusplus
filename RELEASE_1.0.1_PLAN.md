# Release 1.0.1 — Task Plan

Patch release. Three audits inspired by https://corrode.dev/blog/bugs-rust-wont-catch/, mapped to tlaplusplus. None of these are known-broken — they're "the article tells us where rewrites in Rust commonly leak bugs that Rust doesn't catch, and we should sweep for our analogues."

## Tasks

- [ ] **T101. Fuzz the parser and evaluator for panic-resistance.** Extend the existing `fuzz_tla_module` target (which only covers `.tla` parsing) with two more: one feeds random byte sequences as `.cfg` files into `parse_tla_config`, the other drives `eval_expr` / `compile_expr` + `eval_compiled` on AST fragments derived from the proptest swarm generator (T16a). Acceptance: fuzz each target for 4+ hours on a spot, no panics / aborts / OOMs. Any panic on user-supplied bytes is a denial-of-service vector for `analyze-tla` / `run-tla` and must be converted to `Result::Err`. Likely surfaces dozens of `unwrap()` / `expect()` / unchecked indexing in `src/tla/eval.rs`, `src/tla/compiled_expr.rs`, `src/tla/cfg.rs`, `src/tla/module.rs` that the curated test corpus never exercised.
- [x] **T102. `let _ = ` / `.ok()` / `unused_must_use` audit.** Ran `clippy::let_underscore_must_use` + `clippy::let_underscore_future` and `clippy::unused_must_use` over the workspace. 457 total clippy warnings (291 from `let_underscore_must_use`, 166 unrelated baseline lints; no real `unused_must_use` callsites). 140 unique `let _ = ` callsites audited. Resolutions: 9 propagated (or upgraded to log+continue: thread joins in `runtime.rs`, `spillable_work_stealing.rs::shutdown`, `storage/queue.rs::shutdown`, `autotune.rs`, `main.rs`; corrupt-checkpoint cleanup in `runtime.rs`; final `fp_store.flush` at run shutdown; S3 `abort_multipart_upload` in `s3_persistence.rs`; cross-node `stolen_tx.try_send` count drift in `distributed/handler.rs`); 36 documented (best-effort NUMA hints, persist channel try_send, violation channel overflow with explicit max-violations cap, donate channel try_send, signal-handler stdio flushes, idempotent notify_tx wakes, infallible-by-construction unbounded sends); 95 left as test-cleanup `fs::remove_dir_all` (`#[cfg(test)]` blocks; recognized Rust idiom — verified pattern by spot-check, no production-code analogues). One genuinely-broken path uncovered: `error_tx` was `bounded(1)` so a second worker erroring concurrently with the first would block forever in `Sender::send` (mirrors the T6 termination-detection bug class) — switched to `unbounded` and now drain all worker errors at run end. Tests after: 756 (release) and 776 (failpoints), 0 failed.
- [ ] **T103. `_lossy` UTF-8 audit.** `grep -rn "_lossy\|to_string_lossy\|from_utf8_lossy" src/` and audit each call. Any path where bytes flow from external sources (IOEnv, JsonDeserialize, CSVRead, user model values, `.tla` files containing non-ASCII) into state-fingerprint computation or queue-spill serialization is a potential soundness bug — two states that should hash equal could hash differently due to silent UTF-8 substitution. Action per call site: confirm it's display-only, OR convert the path to stay in `&[u8]`/`OsStr`/`Vec<u8>` end-to-end.

## Why 1.0.1 vs 1.1.0

These are correctness audits, not features. The premise of v1.0.0 was "we shipped a tested release; here are the audit-shaped follow-ups that the testing infra didn't directly catch." They belong in a patch.

## Working rules (carried from v1.0.0)

- All builds + `cargo test` runs on EC2 spot instances. Local machine is read-only for code.
- Each task agent updates this plan file when it finishes (mark `[x]` + one-line resolution).
- Use `--no-verify` on commits (the local pre-commit hook is intentionally disabled).
- **NEVER** put infra command names, AWS profile names, instance IDs, or IPs into tracked files, commit messages, or agent prompts that get committed back. See `~/.claude/projects/-Volumes-OWC-1M2-Users-isaiah-src-tlaplusplus/memory/feedback_no_infra_secrets_in_commits.md`.
- Do tasks in parallel via worktrees where they don't share code paths.
