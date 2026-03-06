# TLA+ Test Corpus

Test specifications for validating tlaplusplus against TLC.

## Quick Start

```bash
# Run a single spec
./target/release/tlaplusplus run-tla \
  --module corpus/examples/SimpleCounter.tla \
  --config corpus/examples/SimpleCounter.cfg

# Run full corpus validation against TLC
scripts/tlc_corpus.sh
```

## Directory Structure

### [`examples/`](examples/)
Basic models for getting started.

| Spec | Description |
|------|-------------|
| [SimpleCounter](examples/SimpleCounter.tla) | Minimal counter model (~100 states) |
| [SimpleCounterViolation](examples/SimpleCounterViolation.tla) | Counter with intentional invariant violation |
| [SimpleCounterNoConfig](examples/SimpleCounterNoConfig.tla) | Counter without separate config file |

### [`language/`](language/)
Individual TLA+ language feature tests.

| Spec | Features Tested |
|------|-----------------|
| [SetOperatorsTest](language/SetOperatorsTest.tla) | Set operations: union, intersection, subset |
| [StringTest](language/StringTest.tla) | String values and operations |
| [NegativeIntTest](language/NegativeIntTest.tla) | Negative integer handling |
| [NestedExceptTest](language/NestedExceptTest.tla) | Nested EXCEPT expressions |
| [MultiVarQuantifierTest](language/MultiVarQuantifierTest.tla) | Multi-variable quantifiers |
| [RecursiveTest](language/RecursiveTest.tla) | RECURSIVE operator definitions |
| [ViewTest](language/ViewTest.tla) | VIEW for state symmetry |
| [OperatorSubstitutionTest](language/OperatorSubstitutionTest.tla) | Operator argument substitution |
| [InstanceTest](language/InstanceTest.tla) | INSTANCE with substitution |
| [InstanceTestSimple](language/InstanceTestSimple.tla) | Basic INSTANCE usage |
| [MultipleExtendsTest](language/MultipleExtendsTest.tla) | Multiple EXTENDS clauses |

### [`temporal/`](temporal/)
Temporal property and fairness tests.

| Spec | Features Tested |
|------|-----------------|
| [LivenessTest](temporal/LivenessTest.tla) | Eventually, Always, leads-to |
| [FairnessTest](temporal/FairnessTest.tla) | Weak/strong fairness |
| [TemporalQuantTest](temporal/TemporalQuantTest.tla) | Temporal quantifiers (\E x: <>P) |
| [EnabledTest](temporal/EnabledTest.tla) | ENABLED operator |

### [`integration/`](integration/)
Complex multi-feature models.

| Spec | Description |
|------|-------------|
| [Combined](integration/Combined.tla) | Multi-module production-style spec |
| [CombinedSimple](integration/CombinedSimple.tla) | Simplified version of Combined |

### [`internals/`](internals/)
TLA+ specifications modeling tlaplusplus's own internal algorithms.

| Spec | Description |
|------|-------------|
| [WorkQueue](internals/WorkQueue.tla) | Work-stealing queue with checkpoint/drain protocol |
| [FingerprintResize](internals/FingerprintResize.tla) | Lock-free fingerprint store resize using seqlock |

### [`language_coverage/`](language_coverage/)
Systematic language feature coverage suite.

| Spec | Config | Description |
|------|--------|-------------|
| [LanguageFeatureMatrix](language_coverage/LanguageFeatureMatrix.tla) | [default](language_coverage/LanguageFeatureMatrix.cfg) | Comprehensive feature coverage |
| | [fairness](language_coverage/LanguageFeatureMatrixFair.cfg) | With fairness properties |
| | [no-enabled](language_coverage/LanguageFeatureMatrix_NoEnabled.cfg) | Without ENABLED |
| [InitNextTemporalQuant](language_coverage/InitNextTemporalQuant.tla) | [cfg](language_coverage/InitNextTemporalQuant.cfg) | Init/Next override with temporal quantifiers |

## Running Validation

```bash
# Run all corpus specs through TLC
scripts/tlc_corpus.sh

# Run a specific spec through TLC
scripts/tlc_check.sh corpus/examples/SimpleCounter.tla corpus/examples/SimpleCounter.cfg
```
