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
