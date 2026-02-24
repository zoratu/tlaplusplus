# TLA+ Feature Reference (Implementation Checklist)

This checklist is the implementation target for native `tlaplusplus` frontend work.

## 1) Core language forms

| Feature | Status |
|---|---|
| Module headers, `EXTENDS`, constants, variables | planned |
| Operator definitions (`Op == Expr`) | planned |
| `LET ... IN` | planned |
| `IF ... THEN ... ELSE` | planned |
| `CASE ... [] ... [] OTHER` | planned |
| Quantifiers `\A`, `\E` | planned |
| `CHOOSE` | planned |
| Function updates via `EXCEPT` | planned |

## 2) State/action forms

| Feature | Status |
|---|---|
| Primed vars (`x'`) | planned |
| `UNCHANGED` | planned |
| `Init /\ [][Next]_vars` style specs | planned |
| Action disjunction/conjunction | planned |
| `ENABLED` | planned |
| Action constraints / model constraints | planned |

## 3) Data model and operators

| Feature | Status |
|---|---|
| Booleans, integers, intervals (`a..b`) | planned |
| Sets (`\in`, `\union`, `\intersect`, set comprehension) | planned |
| Sequences (`<<...>>`, `Append`, `Head`, `Tail`, `Len`, `SubSeq`, `SelectSeq`) | planned |
| Functions (`[x \in S |-> e]`, function application, `DOMAIN`) | planned |
| Records and record updates | planned |
| Tuples | planned |

## 4) Temporal/liveness

| Feature | Status |
|---|---|
| Safety invariants | in progress |
| Temporal operators `[]`, `<>`, `~>` | planned |
| Fairness `WF`, `SF` | planned |
| Temporal quantification (`\AA`, `\EE`) | planned |

## 5) Module system

| Feature | Status |
|---|---|
| `INSTANCE ... WITH ...` | planned |
| `LOCAL` definitions | planned |
| Assumptions/theorems parsing support | planned |

## 6) TLC model config compatibility

| Feature | Status |
|---|---|
| `CONSTANT(S)` assignments | planned |
| `INIT`, `NEXT`, `SPECIFICATION` override | planned |
| `INVARIANT(S)`, `PROPERTY(IES)` | planned |
| `SYMMETRY` | planned |
| `CHECK_DEADLOCK` | planned |

## Corpus linkage

Seed language-coverage model:

- `corpus/language_coverage/LanguageFeatureMatrix.tla`
- `corpus/language_coverage/LanguageFeatureMatrix.cfg`
- `corpus/language_coverage/LanguageFeatureMatrixFair.cfg`
- `corpus/language_coverage/InitNextTemporalQuant.tla`
- `corpus/language_coverage/InitNextTemporalQuant.cfg`
- `corpus/language_coverage/CoverageHelper.tla`

It is intended as a living corpus entry used for both parser/evaluator regressions and TLC parity checks.

### Seed coverage status (current)

`LanguageFeatureMatrix` now exercises, in one bounded model:

- Core forms: module/constants/variables, operator defs, `LET`, `IF`, `CASE`, `\A`, `\E`, `CHOOSE`, `EXCEPT`.
- State/action forms: primed vars, `UNCHANGED`, disjunctive `Next`, `Init /\ [][Next]_Vars`, `ENABLED`, state/action constraints.
- Data/operators: sets (`\union`, `\intersect`, comprehensions), sequences (`Append`, `Head`, `Tail`, `Len`, `SubSeq`, `SelectSeq`), functions/`DOMAIN`, records, tuples, intervals.
- Temporal/liveness: `[]`, `<>`, `~>`, plus `WF`/`SF` in `FairSpec`.
- Module system: `LOCAL` definitions and `INSTANCE ... WITH ...`.
- Assertions: `ASSUME` and `THEOREM`.
- TLC config features: `CONSTANTS`, `SPECIFICATION`, `INVARIANT`, `PROPERTY`, `CONSTRAINT`, `ACTION_CONSTRAINT`, `SYMMETRY`, `CHECK_DEADLOCK`.
- Additional config coverage via `InitNextTemporalQuant.cfg`: explicit `INIT` and `NEXT` override.
- Temporal quantifier syntax artifact via `InitNextTemporalQuant.tla`: `\AA` and `\EE` temporal formulas are defined.

Known TLC limitation:

- TLC (v1.7.4 / TLC2 2.19) parses `\AA`/`\EE` temporal formulas in definitions but does not model-check them as `PROPERTY` formulas.

### TLC validation command

```bash
scripts/tlc_check.sh
scripts/tlc_check.sh corpus/language_coverage/LanguageFeatureMatrix.tla \
  corpus/language_coverage/LanguageFeatureMatrixFair.cfg \
  .tlc-out/language_coverage_fair
scripts/tlc_check.sh corpus/language_coverage/InitNextTemporalQuant.tla \
  corpus/language_coverage/InitNextTemporalQuant.cfg \
  .tlc-out/init_next_temporal_quant
scripts/tlc_corpus.sh
scripts/tlc_public_corpus.sh
```

## Primary references

- [The Module Structure of TLA+ (Lamport)](https://lamport.azurewebsites.net/tla/newmodule.html)
- [TLA+ Wiki: Standard Library](https://docs.tlapl.us/using%3Astandard_lib)
- [The TLA+ Hyperbook](https://lamport.azurewebsites.net/tla/hyperbook.html)
- [Learn TLA+ (language-oriented guide)](https://www.learntla.com/core/tla.html)
