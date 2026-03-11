# Public TLA+ Corpus Plan

Goal: build a broad comparison corpus of public TLA+ models for TLC vs `tlaplusplus` parity and performance testing.

## Current local seed corpus

- `corpus/language_coverage/LanguageFeatureMatrix.tla`
- `corpus/language_coverage/LanguageFeatureMatrix.cfg`
- `corpus/language_coverage/LanguageFeatureMatrixFair.cfg`
- `corpus/language_coverage/InitNextTemporalQuant.tla`
- `corpus/language_coverage/InitNextTemporalQuant.cfg`
- `corpus/index.tsv` (corpus manifest with id/module/cfg/tags)

Execution harness:

- `scripts/tlc_corpus.sh` runs all entries from `corpus/index.tsv`.
- Writes parsed summary to `.tlc-out/corpus/summary.tsv`.
- Writes raw TLC logs per entry under `.tlc-out/corpus/<id>/tlc.log`.
- `scripts/analyze_tla_corpus.sh` runs the native frontend over the in-repo corpus.
- Writes parsed summary to `.analyze-tla/corpus/summary.tsv`.
- Supports sharding with `SHARD_INDEX` / `SHARD_COUNT`.

Public corpus seed (internet/GitHub sourced):

- `corpus/public/public_corpus.tsv`: curated repo/module/cfg list (source-of-truth).
- `scripts/refresh_public_corpus_shas.sh`: resolves `HEAD` to pinned commit SHAs.
- `corpus/public/public_corpus.lock.tsv`: generated pinned lockfile.
- `scripts/tlc_public_corpus.sh`: clones/fetches pinned entries and runs TLC.
- Summary output: `.tlc-out/public-corpus/summary.tsv`.

Current pinned public entries (lockfile):

- `examples_barrier` from `tlaplus/Examples`
- `examples_majority` from `tlaplus/Examples`
- `examples_keyvalue_small` from `tlaplus/Examples`
- `fpaxos_twoacc` from `fpaxos/fpaxos-tlaplus`
- `radix_validation` from `mitchellh/tlaplus-radix-tree`

These are the initial in-repo language coverage models used to bootstrap parser/evaluator parity tests while public-corpus harvesting is in progress.

## Current Linux verification baseline

As of 2026-03-11, the first repo-wide Linux coverage baseline for the merged runtime/frontend test suite is:

- command: `cargo llvm-cov --release --lib --summary-only`
- total region coverage: `65.17%`
- total line coverage: `64.61%`
- `src/runtime.rs`: `46.90%` regions
- `src/storage/work_stealing_queues.rs`: `80.36%` regions

Use this as a moving baseline, not an acceptance threshold. The gap list should be driven by the surface matrix in `notes/tla-feature-reference.md`, and every new direct regression should ideally move one of the low-coverage or unsupported rows out of `partial` / `backlog`.

## TODO

1. Scour public sources for models:
   - GitHub code search (`language:TLA`, `*.tla`, `*.cfg`).
   - TLA+ community example repos and conference/tutorial material.
   - Public corpora linked from blogs/papers/issues.
2. Curate a reproducible corpus index:
   - repo URL, commit SHA, module path, cfg path, license.
   - tags: feature usage (sets/sequences/functions/liveness/symmetry/etc).
3. Add automated fetch script and lockfile:
   - pin exact SHAs to avoid drift.
   - mirror metadata into this repo.
4. Add comparison harness:
   - run TLC and `tlaplusplus` on each corpus entry.
   - record reachability counts, violations, and runtime metrics.
5. Add regression gates:
   - block merges on semantic mismatches for supported features.
6. Tie corpus tags and failures back to `notes/tla-feature-reference.md`:
   - every matrix row should point at at least one in-repo corpus anchor
   - every public-example failure should map to a tracked matrix gap
