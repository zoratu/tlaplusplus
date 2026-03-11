# TLA+ / TLC Surface Matrix

This file is the implementation target for the native frontend and runtime.

Use it as the source of truth for:
- parser surface area
- evaluator/runtime surface area
- TLC config compatibility
- corpus and public-example coverage planning

Goal:
- support the standard TLA+ language surface end to end
- support as much TLC-specific config and standard-module behavior as practical
- drive the remaining work from explicit requirements rather than example-by-example fixes

Status labels:
- `covered`: exercised in `corpus/` and currently expected to work in the native frontend
- `partial`: implemented enough for many specs, but public examples still expose gaps
- `backlog`: not yet represented well enough in code or corpus

## Canonical sources

Primary sources to mirror:
- Leslie Lamport, [A Summary of TLA+ Syntax](https://lamport.azurewebsites.net/tla/summary-standalone.pdf)
- Leslie Lamport, [A TLA+2 Guide](https://lamport.azurewebsites.net/tla/tla2-guide.pdf)
- Leslie Lamport, [The Module Structure of TLA+](https://lamport.azurewebsites.net/tla/newmodule.html)
- `tlaplus/tlaplus` TLC model config parser:
  [`ModelConfig.java`](https://github.com/tlaplus/tlaplus/blob/master/tlatools/org.lamport.tlatools/src/tlc2/tool/impl/ModelConfig.java)
- Vendored standard modules in this repo:
  [`tools/tla2sany/StandardModules`](../tools/tla2sany/StandardModules)

## Acceptance bar

Work should converge toward:
- every standard TLA+ primitive, binder, operator class, and module-system form represented in an internal corpus anchor
- every TLC config keyword represented in at least one internal corpus config
- every supported standard-module operator represented in internal coverage
- every public example either:
  - passes end to end, or
  - fails for an explicit tracked reason such as a missing community module or unsupported TLC extension

## Validation discipline

Each nontrivial feature should normally end up with:
- a direct unit regression that exercises the exact production path that broke
- property coverage where the semantics are algebraic or combinatorial
- chaos or fault-injection coverage for runtime coordination, checkpointing, or storage failure paths
- Linux-only sanitizer and coverage runs before claims about completeness
- a TLA+ model when the logic is itself a stateful coordination protocol

## Coverage anchors

Current in-repo anchors:
- `corpus/language_coverage/LanguageFeatureMatrix.tla`
- `corpus/language_coverage/LanguageFeatureMatrix.cfg`
- `corpus/language_coverage/LanguageFeatureMatrixFair.cfg`
- `corpus/language_coverage/InitNextTemporalQuant.tla`
- `corpus/language_coverage/InitNextTemporalQuant.cfg`
- `corpus/temporal/*.tla`
- `corpus/language/*.tla`
- `corpus/internals/*.tla`

Current internal protocol models:
- `corpus/internals/BloomAutoSwitch.tla`
- `corpus/internals/CheckpointDrain.tla`
- `corpus/internals/FingerprintResize.tla`
- `corpus/internals/FingerprintStoreResize.tla`
- `corpus/internals/QueueSegmentSync.tla`
- `corpus/internals/S3SegmentPrune.tla`
- `corpus/internals/WorkQueue.tla`
- `corpus/internals/WorkStealingTermination.tla`

Current public-sweep validation entrypoint:
- `scripts/analyze_tla_examples.sh`

Current repo-corpus validation entrypoint:
- `scripts/analyze_tla_corpus.sh`

## Core TLA+ language

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| Module headers and declarations | `MODULE`, `EXTENDS`, `CONSTANT(S)`, `VARIABLE(S)` | `language/MultipleExtendsTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Operator definitions | nullary, prefix, indexed, higher-order params | `language/OperatorSubstitutionTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Infix operator definitions and use | `a \prec b`, symbolic operator calls | public examples `Bakery-Boulangerie/*` | `partial` |
| `LOCAL` definitions | `LOCAL Foo == ...` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| `LET ... IN` | local helper defs, nested `LET` in actions | `language/OperatorSubstitutionTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| `IF / THEN / ELSE` | expression and action bodies | `language_coverage/LanguageFeatureMatrix.tla`, `temporal/EnabledTest.tla` | `covered` |
| `CASE ... [] ... [] OTHER` | expression branches | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Quantifiers | `\A`, `\E`, tuple binders, multi-binders | `language/MultiVarQuantifierTest.tla`, `language_coverage/InitNextTemporalQuant.tla` | `covered` |
| Choice | `CHOOSE x \in S : P` | `language_coverage/LanguageFeatureMatrix.tla` | `partial` |
| Recursive declarations | `RECURSIVE Op(_)` | `language/RecursiveTest.tla` | `covered` |
| Assumptions and theorems | `ASSUME`, `THEOREM` parse support | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |

## Data model and operators

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| Booleans and integers | `BOOLEAN`, `TRUE`, `FALSE`, arithmetic | `language/NegativeIntTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Strings and string literals | quoted strings, equality, sequence-style string operators | `language/StringTest.tla` | `covered` |
| Intervals and ranges | `a..b` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Sets | enumeration, comprehension, filter, `\in`, `\notin`, `\union`, `\intersect`, `SUBSET` | `language/SetOperatorsTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Set difference | `S \ T`, compact forms like `Nodes\{n}` | `language/SetOperatorsTest.tla` | `covered` |
| Higher-level set constructors | `UNION S`, Cartesian product `S \X T`, set-of-functions `[S -> T]`, record sets `[a: S, b: T]` | `language/RecordSetTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `partial` |
| Tuples and sequences | `<<...>>`, `Append`, `Head`, `Tail`, `Len`, `SubSeq`, `SelectSeq` | `language/StringTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Functions | `[x \in S |-> e]`, application, `DOMAIN`, `Range`, `@@`, `:>` | `language/NestedExceptTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Records | constructors, field access, nested updates | `language/RecordSetTest.tla`, `language/NestedExceptTest.tla` | `covered` |
| `EXCEPT` | function and record updates | `language/NestedExceptTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Higher-order operator values | passing `read` / `write` into `SelectSeq`, lambda-like operator parameters | public examples `ReadersWriters/*` | `partial` |
| Non-core symbolic operators | `^^`, other community-module operators | public examples `SDP_Verification/*` | `partial` |

## State and action language

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| Primed assignments | `x' = e` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| `UNCHANGED` | `UNCHANGED x`, tuples of vars | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Conjunctive actions | `/\` clauses with shared staging | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Disjunctive actions | `\/` top-level branches | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Nested disjunctive action clauses | PlusCal `either/or` or translated branch clauses inside a larger conjunction | public examples `Bakery-Boulangerie/*` | `partial` |
| Existential action binders | `\E x \in S : Action` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Action `IF` bodies | `IF cond THEN /\ ... ELSE /\ ...` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| `ENABLED` | `ENABLED A`, `ENABLED A(arg)` | `temporal/EnabledTest.tla`, public examples `MultiCarElevator/*` | `covered` |
| Spec skeletons | `Init /\ [][Next]_vars` | `language_coverage/InitNextTemporalQuant.tla` | `covered` |
| Action wrappers | `[A]_vars`, `<<A>>_vars` / non-stuttering action forms | `language_coverage/InitNextTemporalQuant.tla`, temporal corpus | `partial` |
| Probe seeding for disabled branches | guard-false branches should not poison later clauses | public examples `Prisoners_Single_Switch/*` | `partial` |

## Temporal and liveness surface

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| Box and diamond | `[]P`, `<>P` | `temporal/LivenessTest.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Leads-to | `P ~> Q` | `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Fairness | `WF`, `SF` | `temporal/FairnessTest.tla`, `language_coverage/LanguageFeatureMatrixFair.cfg` | `covered` |
| Temporal quantification | `\AA`, `\EE` | `language_coverage/InitNextTemporalQuant.tla` | `partial` |

Notes:
- TLC itself still does not model-check `\AA` / `\EE` properties as ordinary `PROPERTY` formulas.
- `analyze-tla` is still a frontend/probe path, not the checkpointable search engine used by `run-tla`.

## Module system

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| `INSTANCE ... WITH ...` | named and unnamed instances | `language/InstanceTestSimple.tla`, `language_coverage/LanguageFeatureMatrix.tla` | `covered` |
| Module-instance refs | `Alias!Const`, `Alias!Op(args)` | corpus language tests and recent frontend regressions | `covered` |
| Nested instance helper resolution | outer helper refs from inner instances | targeted unit tests in `src/tla/eval.rs` and `src/tla/action_exec.rs` | `covered` |
| Community / proof modules | `TLAPS`, `Bitwise`, `Functions`, etc. | public examples | `partial` |

## TLC config surface

Mirror the keyword set in upstream `ModelConfig.java`.

| Surface | Representative forms | Corpus anchor | Status |
|---|---|---|---|
| Constant assignment | `CONSTANT`, `CONSTANTS`, model values, sets, records | `corpus/language/*.cfg` | `covered` |
| Operator overrides | `Foo <- Bar`, bracket-qualified refs like `[ZSequences]ZSeqNat` | `language/OperatorSubstitutionTest.cfg`, public examples | `partial` |
| Entrypoint overrides | `INIT`, `NEXT`, `SPECIFICATION` | `language_coverage/InitNextTemporalQuant.cfg`, `language_coverage/LanguageFeatureMatrix.cfg` | `covered` |
| Safety/liveness checks | `INVARIANT(S)`, `PROPERTY(IES)` | `corpus/language_coverage/*.cfg`, `corpus/temporal/*.cfg` | `covered` |
| Constraints | `CONSTRAINT`, `ACTION_CONSTRAINT` | `language_coverage/LanguageFeatureMatrix.cfg` | `covered` |
| Plural TLC keywords | `CONSTRAINTS`, `ACTION_CONSTRAINTS`, `PROPERTIES` | upstream `ModelConfig.java` plus corpus configs | `partial` |
| Symmetry and deadlock | `SYMMETRY`, `CHECK_DEADLOCK` | `language_coverage/LanguageFeatureMatrix.cfg` | `covered` |
| State projection | `VIEW` | upstream `ModelConfig.java` | `backlog` |
| TLC extras | `ALIAS`, `POSTCONDITION` | parser support plus public examples | `partial` |
| Toolbox / advanced config hooks | `_PERIODIC`, `_RL_REWARD` | upstream `ModelConfig.java` | `backlog` |

## Standard and extended modules

Vendored standard modules visible in this checkout:
- `Naturals`
- `Integers`
- `Sequences`
- `FiniteSets`
- `Bags`
- `Reals`
- `RealTime`
- `Randomization`
- `TLC`
- `Toolbox`

Operator inventory for the vendored set:
- `notes/standard-module-operators.md`

Implementation priority:
1. Keep core TLA+ and TLC config surface green across `corpus/` and public examples.
2. Match operators from vendored standard modules before adding ad hoc community-module behavior.
3. Track community-module compatibility explicitly instead of silently treating missing modules as frontend bugs.

Current non-core modules seen in public examples and still worth tracking explicitly:
- `Bitwise`
- `Functions`
- `TLAPS`

## Public-example backlog (current)

These are the example-driven gaps to keep burning down after the current fixes:
- representative synthesis for `CHOOSE`-heavy specs like `YoYo`
- action-guard typing issues in `aba-asyn-byz`
- enum / identifier-set parsing cases still exposed by `ACP_NB_WRONG_TLC`
- sequence-emptiness and `CHOOSE` probe issues in `SDP_Verification/*`
- community-module compatibility beyond the vendored standard library

## Validation commands

Native frontend:

```bash
scripts/analyze_tla_corpus.sh
scripts/analyze_tla_examples.sh ~/src/tlaplus-examples/specifications
```

TLC parity:

```bash
scripts/tlc_check.sh
scripts/tlc_corpus.sh
```

Coverage:

```bash
cargo llvm-cov --lib --bin tlaplusplus --summary-only
```
