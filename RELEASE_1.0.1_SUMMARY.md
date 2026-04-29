# tlaplusplus v1.0.1

Patch release covering the **"Bugs Rust Won't Catch"** audit — three
sub-audits (T101 / T102 / T103) that target the defect classes the Rust
type system cannot catch on its own: parser/compiler panics on adversarial
input, silently-discarded `Result` values, and lossy UTF-8 conversions.

## Highlights

### T101 — Parser/evaluator panic-resistance (fuzz audit)

- Stood up `cargo-fuzz` targets across the TLA+ parser and compiled-IR
  evaluator and drove crash counts to zero.
- Fixed **4 panic classes** in the parser/compiler, all from `&str[..]`
  slicing on non-character boundaries with non-ASCII input. Affected
  sites: indexed-op-call parser, recursive-decl parser, INSTANCE
  substitution, CFG comment stripping, and two LET-binding range
  computations in `compiled_expr` / `eval`.
- Added **7 regression tests** in `tests/fuzz_panic_regressions_t101.rs`.
- Wired the swarm-equivalence fuzz target so symmetry-reduced vs
  un-reduced runs are diff-checked on every fuzz iteration.

### T102 — `Result`-discard audit (silent-error audit)

- **Headline fix:** the runtime's per-worker `error_tx` could deadlock
  under concurrent send when the receiver had drained. Converted to a
  non-blocking try-send with an explicit drop-on-full path.
- Fixed **9 additional propagation sites** where I/O errors,
  checkpoint-write failures, and parser warnings were silently swallowed.
- Audited every `let _ = ...`, `.ok()`, and `#[must_use]`-bypass and
  classified each remaining case as intentional.

### T103 — Lossy UTF-8 conversion audit

- **Two fixes,** both on defensive observability paths:
  - S3 checkpoint key construction was using `_lossy` on borrowed
    `OsStr`, which could silently mangle the key on non-UTF-8 paths.
    Now strict UTF-8.
  - Disk-stats logging path used `_lossy` on a logged path; now strict.
- **No state-path soundness issue.** The hot path (state serialization,
  fingerprinting, action equality) is confirmed lossy-free.

## Validation gates

All gates green on a fresh clone of `5154f4b`:

| Gate | Result |
|---|---|
| `cargo test --release` | 765 pass / 0 fail |
| `cargo test --release --features failpoints` | 785 pass / 0 fail |
| `cargo test --release --features symbolic-init` | 783 pass / 0 fail |
| `scripts/diff_tlc.sh` (vs TLC v1.7.4) | 13 / 13 |
| `cargo test --release --test fuzz_panic_regressions_t101` | 7 / 7 |

## Compatibility

Drop-in for v1.0.0. No public-API or CLI changes. Fingerprint format,
checkpoint format, and state-graph dump format are all unchanged.
