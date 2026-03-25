# Changelog

## v0.3.0 (2026-03-25)

### TLA+ Language Compatibility

**External corpus: 0 errors, 90% pass at 60s, 94% at 15 min** (up from 63% at start of release cycle). Every spec that defines a state machine (Init + Next) runs correctly.

- **SPECIFICATION definition chasing**: Follow definition reference chains to extract Init/Next from temporal formulas like `Spec == LiveSpec` where `LiveSpec == Init /\ [][Next]_vars /\ WF(Next)`
- **Disjunctive Init branches**: Handle `\/ Guard1 /\ var = expr1 \/ Guard2 /\ var = expr2` in Init by evaluating branch guards with known constants
- **Existential quantifier Init**: Expand `\E x \in S : body` in Init bodies by iterating domain values and extracting variable assignments
- **Init!N sub-expression references**: Support TLC-style `Init!1`, `Init!2` to reference specific conjuncts of a definition
- **Disjunctive Init bodies**: Handle Init definitions that are top-level disjunctions (`\/ branch1 \/ branch2`)
- **Parameterized operator calls in Init**: Inline-expand `XInit(x)` where `XInit(v) == v = 0` to extract variable assignments
- **Late-binding equality/membership**: After cross-product expansion, re-classify guards as equality assignments when dependencies are now satisfied
- **Deferred membership evaluation**: Membership sets that depend on other membership variables are deferred to cross-product phase
- **Outer parenthesis stripping**: `(h_turn = 1)` in Init correctly classified as equality assignment
- **INSTANCE variable shadowing**: Skip instance module variables that shadow definitions (e.g., Stuttering's `vars` vs Lock's `vars == <<pc, lock>>`)
- **Definition override preservation**: Save original definitions before `Init <- MCInit` override for missing variable recovery
- **Evaluation-only modules**: Specs without Init/Next/SPECIFICATION complete instantly with 0 states instead of erroring
- **Local .tla file loading for built-in modules**: When a spec ships its own `Functions.tla` alongside a built-in module, load definitions from the local file as fallback

### Community Modules

16 new built-in operators across 6 modules:

- **Bags**: EmptyBag, SetToBag, BagToSet, IsABag, BagIn, BagOfAll, BagUnion, CopiesIn
- **BagsExt**: BagAdd, BagRemove
- **IOUtils**: IOEnv, ndJsonDeserialize, JsonDeserialize, JsonSerialize, ToString
- **Bitwise**: BitsAnd, BitsOr, BitsXor, BitNot, LeftShift, RightShift, IsABitVector, IsANatural
- **Combinatorics**: Factorial, nCk, nPk
- **CSV**: CSVRead, CSVWrite
- **VectorClocks**: VCLess, VCLessOrEqual, VCMerge
- **Randomization**: RandomSubset

### Performance Optimizations

- **Trivial Next detection**: When `Next == UNCHANGED vars`, skip BFS exploration entirely — just enumerate Init states and check invariants
- **Constraint propagation**: For `{c \in [f1: 0..N, f2: 0..N] : c.f1 + c.f2 \in lo..hi}`, compute valid ranges directly instead of iterating all N² pairs (~3000x speedup for CoffeeCan)
- **Compiled predicate evaluation**: Use `compile_expr` + `eval_compiled` for record set filtering instead of re-parsing expression text per record
- **Range membership fast path**: `x \in a..b` evaluates as `a <= x && x <= b` instead of constructing the full integer set
- **Inline record set generation**: Generate records inline with predicate filtering instead of materializing the full record set
- **Vec-based constraint output**: Return constraint-propagated record sets as Seq (O(n)) instead of BTreeSet (O(n log n))
- **Lazy Init enumeration**: For cross-products exceeding 10M states, use odometer-style enumeration instead of materialization
- **Early exit for 0 initial states**: Evaluation-only modules complete instantly without spawning worker threads
- **FunAsSeq fix**: Corrected key indexing from `n..n+m-1` to `1..m` matching TLC semantics

### Distributed Model Checking

- **`--fetch-module` / `--fetch-config`**: Fetch spec files from S3 URIs for distributed runs where nodes don't share a filesystem
- **FLURM integration**: Plugin support for job scheduling on spot instances with automatic S3 file distribution
- **S3 checkpoint resume validated**: Round-trip checkpoint → clear → resume verified on spot instances

### Bug Fixes

- Fixed `FunAsSeq(f, n, m)` key indexing (was `n..n+m-1`, now `1..m`)
- Fixed empty expression handling in `Init!N` references with comment-stripped lines
- Resolve `IOEnv` and `EmptyBag` as zero-arg built-in operators (bare identifier usage)
- TLCGet ASSUME failures downgraded to warnings (TLC-specific runtime introspection)
- MAX_INIT_STATES raised from 1M to 10M
- Record set size limit raised to 10M

### Scaling Benchmark

Corpus run on c6gd.2xlarge (8 vCPU) instances, 60s timeout per spec:

| Nodes | vCPUs | Wall Time | Pass Rate | Speedup |
|-------|-------|-----------|-----------|---------|
| 1 | 8 | 29 min | 162/182 (89%) | 1.0x |
| 2 | 16 | 18 min | 161/182 (88%) | 1.6x |
| 4 | 32 | 8.6 min | 166/182 (91%)* | 3.4x |
| 8 | 64 | 7.5 min | 170/182 (93%)* | 3.9x |

*Adjusted for connectivity issues on some nodes

With longer timeouts on larger machines (192 cores):
- 300s: 169/182 (93%)
- 900s: 171/182 (94%)

### Test Suite

- 600 unit tests, all passing
- 620 failpoint tests (with `--features failpoints`), all passing
- 25/32 internal corpus specs passing
- 163/182 external corpus specs passing at 60s (0 errors, 19 timeouts)
- 161/182 analysis probes: FULL_PASS (88%), 0 FAIL

## v0.2.0

Initial release with parallel runtime, NUMA-aware work-stealing, lock-free fingerprint store, and native TLA+ frontend.
