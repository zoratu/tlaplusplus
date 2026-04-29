# Release 1.0.1 — Task Plan

Patch release. Three audits inspired by https://corrode.dev/blog/bugs-rust-wont-catch/, mapped to tlaplusplus. None of these are known-broken — they're "the article tells us where rewrites in Rust commonly leak bugs that Rust doesn't catch, and we should sweep for our analogues."

## Tasks

- [ ] **T101. Fuzz the parser and evaluator for panic-resistance.** Extend the existing `fuzz_tla_module` target (which only covers `.tla` parsing) with two more: one feeds random byte sequences as `.cfg` files into `parse_tla_config`, the other drives `eval_expr` / `compile_expr` + `eval_compiled` on AST fragments derived from the proptest swarm generator (T16a). Acceptance: fuzz each target for 4+ hours on a spot, no panics / aborts / OOMs. Any panic on user-supplied bytes is a denial-of-service vector for `analyze-tla` / `run-tla` and must be converted to `Result::Err`. Likely surfaces dozens of `unwrap()` / `expect()` / unchecked indexing in `src/tla/eval.rs`, `src/tla/compiled_expr.rs`, `src/tla/cfg.rs`, `src/tla/module.rs` that the curated test corpus never exercised.
- [ ] **T102. `let _ = ` / `.ok()` / `unused_must_use` audit.** Pass `clippy::let_underscore_must_use` + `clippy::unused_must_use` over the codebase. Audit every silenced `Result`. For each: either propagate the error properly, or add an inline comment explaining why the failure is provably safe to ignore. Particular suspects: `src/runtime.rs` (worker spawn / join / checkpoint coordination), `src/distributed/` (cluster termination, steal protocol), `src/storage/spillable_work_stealing.rs` (background spill coordinator). Our T6 work fixed four termination bugs that fit this exact pattern; this audit catches the others.
- [x] **T103. `_lossy` UTF-8 audit.** 8 `to_string_lossy` call sites in `src/` (no `from_utf8_lossy`, no `from_utf8_unchecked`). 6 are display-only or kernel-controlled `/sys` path scans (annotated SAFE inline). 0 are on the state-fingerprint, queue-spill, or canonical state-serialization path — TLA+ string ingress (`IOEnv`/`JsonDeserialize`/`CSVRead`) already uses UTF-8-checked `read_to_string` / `serde_json::from_str`, so no soundness bug exists. 2 fixed defensively: `system.rs::get_disk_stats` now passes `OsStr::as_bytes()` (Unix) to `CString::new` instead of round-tripping through `to_string_lossy` so non-UTF-8 paths reach `statvfs(2)` byte-identical, and `s3_persistence.rs::collect_pending_uploads` now skips non-UTF-8 local paths with a warning rather than uploading them under a U+FFFD-corrupted S3 key. Regression test `collect_pending_uploads_skips_non_utf8_paths` constructs a 0xff-named file via `OsStr::from_bytes` and asserts it is skipped while the ASCII sibling is enumerated.

## Why 1.0.1 vs 1.1.0

These are correctness audits, not features. The premise of v1.0.0 was "we shipped a tested release; here are the audit-shaped follow-ups that the testing infra didn't directly catch." They belong in a patch.

## Working rules (carried from v1.0.0)

- All builds + `cargo test` runs on EC2 spot instances. Local machine is read-only for code.
- Each task agent updates this plan file when it finishes (mark `[x]` + one-line resolution).
- Use `--no-verify` on commits (the local pre-commit hook is intentionally disabled).
- **NEVER** put infra command names, AWS profile names, instance IDs, or IPs into tracked files, commit messages, or agent prompts that get committed back. See `~/.claude/projects/-Volumes-OWC-1M2-Users-isaiah-src-tlaplusplus/memory/feedback_no_infra_secrets_in_commits.md`.
- Do tasks in parallel via worktrees where they don't share code paths.
