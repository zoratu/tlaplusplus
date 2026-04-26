# Release 1.0.0 — Work Log

Chronological log of work toward 1.0.0. Each entry: date, task ID, agent, outcome, commit (if any), follow-ups.

---

## 2026-04-25

### Setup
- Wrote `RELEASE_1.0.0_PLAN.md` with 15 tasks across 5 phases.
- Working rule confirmed: all builds run on EC2 spot instances; local machine is code-edit-only.
- Starting with Phase 1 (test infrastructure) so later perf work has a regression gate.

### T1 — Differential testing against TLC as CI gate (Phase 1)

**Status:** harness landed, CI workflow wired, end-to-end validated. **4 real divergences uncovered** (parked for follow-up bugs, not patched).

**Artifacts:**
- `scripts/diff_tlc.sh` — runs each curated spec under both checkers, parses
  `<N> states generated, <M> distinct states found` lines plus violation/deadlock
  markers, exits non-zero on divergence. Treats expected violations correctly
  (skips state-count compare when either side stopped early). Has an allowlist
  escape hatch via `corpus/diff_test/allowlist.tsv`.
- `corpus/diff_test/list.tsv` — 9 curated specs, all <10s under TLC, covering:
  set ops, record sets, signed arithmetic, nested EXCEPT, RECURSIVE operators,
  EXTENDS chains, expected-violation case, plus a moderately large internal
  model (CheckpointDrain, ~26K distinct states, ~3s) so we exercise the parallel
  runtime, not just trivial state spaces.
- `corpus/diff_test/allowlist.tsv` — empty placeholder with usage comment.
- `.github/workflows/diff-tlc.yml` — runs on push and PR to main, builds debug
  binary, caches cargo, uploads `.diff-tlc-out/` logs on failure. Uses
  `actions/setup-java@v4` (temurin 17) and `dtolnay/rust-toolchain@stable`.

**Commit:** `c91de88` — `test: differential testing harness vs TLC as CI gate (T1)`

**End-to-end validation:**
- Provisioned `c8g.xlarge` spot instance via `REDACTED l 72` (4 vCPU, 8GB,
  us-west-2). Tailscale hostname `REDACTED-INSTANCE`. Installed Rust stable
  (1.95) + JDK 17. Synced repo, ran `cargo build --bin tlaplusplus` (~5 min),
  ran `scripts/diff_tlc.sh`. Result: **9/9 passing, exit 0**.
- An earlier 13-spec list also ran cleanly through the harness and surfaced the
  divergences below before they were trimmed.

**State-count parity (current curated list, all 9 specs):**
- TLC and tlaplusplus agree on distinct state counts for every passing spec
  (tested values range from 3 to 26,344). For the one expected-violation spec
  (SimpleCounterViolation), violation status agrees and we deliberately skip
  state-count compare since both checkers stop early.

**Real divergences uncovered (NOT patched per agent rules; documented in
`corpus/diff_test/list.tsv` and listed below for follow-up):**

1. `corpus/language/ViewTest.tla` — VIEW projection appears to not be applied to
   fingerprinting. tlaplusplus reports 121 distinct, TLC 106 (the spec uses
   `VIEW <<x, y>>` to elide a monotonically-increasing `timestamp` field; TLC
   correctly collapses states that differ only in `timestamp`, tlaplusplus does
   not). Look at `src/models/tla_native.rs::resolve_view` and how the runtime
   feeds the projection into the fingerprint store.

2. `corpus/internals/QueueSegmentSync.tla` (both `_Fixed.cfg` and `_Buggy.cfg`)
   — INIT/NEXT cfg style produces only 5 distinct states under tlaplusplus vs
   1531 under TLC. Each generated state has exactly 1 successor (debug log:
   `1 successors generated`), so the `Next == \/ A \/ B \/ ...` disjunction with
   `\E segId \in localSegments : ConsumeSegment(segId)` etc. is not enumerating
   the disjuncts properly. The Buggy cfg, which TLC reports a violation on, is
   silently passing under tlaplusplus — i.e. **soundness bug, not just a count
   mismatch**. Suspect `src/tla/action_exec.rs` or `src/tla/action_ir.rs`.
   Reproduces with both `INIT/NEXT` and `SPECIFICATION Spec` cfg forms.

3. `corpus/internals/WorkQueue.tla` — distinct state counts agree (15003), but
   tlaplusplus's fairness checker reports an SCC fairness violation that TLC
   accepts. Message: "Fairness violation in SCC 0 (1 states): Fairness
   constraint may be violated: action 'Next' does not occur in SCC". TLC says
   "No error has been found." Investigate `src/fairness.rs` SCC handling when
   the SCC has only the implicit Next action.

These 4 divergences are tracked inline in `corpus/diff_test/list.tsv` (commented
KNOWN DIVERGENT SPECS section). The CI gate is intentionally narrowed to
specs that already agree, so any *future* regression on the working surface
fails CI immediately. Re-add divergent specs to the active list when fixed.

**Caveats / follow-ups:**
- Fairness divergence (#3) suggests the broader corpus may have similar latent
  divergences; T2 (proptest equivalence) and T3 (snapshot tests) should help
  surface them. Worth a deliberate sweep of the full corpus comparing
  tlaplusplus against TLC once we tackle T2/T3.
- Spec coverage skew toward language features; we should add ViewTest /
  QueueSegmentSync back as soon as the underlying bugs are fixed, since they
  exercise VIEW and INIT/NEXT cfg respectively — both production paths.
- Per-spec timeout set to 60s by default, 120s in CI to absorb cold-cache
  variance on Actions runners.

