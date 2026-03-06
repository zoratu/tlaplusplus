# TLA+ Test Corpus

Test specifications for validating tlaplusplus against TLC.

## Quick Start

```bash
# Run a single spec
./target/release/tlaplusplus run-tla \
  --module corpus/SimpleCounter.tla \
  --config corpus/SimpleCounter.cfg

# Run full corpus validation
scripts/tlc_check.sh
```

## Specifications

### Basic Models

| Spec | Config | Description |
|------|--------|-------------|
| [SimpleCounter.tla](SimpleCounter.tla) | [.cfg](SimpleCounter.cfg) | Minimal counter model (~100 states) |
| [SimpleCounterViolation.tla](SimpleCounterViolation.tla) | [.cfg](SimpleCounterViolation.cfg) | Counter with intentional invariant violation |
| [SimpleCounterNoConfig.tla](SimpleCounterNoConfig.tla) | - | Counter without separate config file |

### Language Features

| Spec | Config | Features Tested |
|------|--------|-----------------|
| [SetOperatorsTest.tla](SetOperatorsTest.tla) | [.cfg](SetOperatorsTest.cfg) | Set operations: union, intersection, subset |
| [StringTest.tla](StringTest.tla) | [.cfg](StringTest.cfg) | String values and operations |
| [NegativeIntTest.tla](NegativeIntTest.tla) | [.cfg](NegativeIntTest.cfg) | Negative integer handling |
| [NestedExceptTest.tla](NestedExceptTest.tla) | [.cfg](NestedExceptTest.cfg) | Nested EXCEPT expressions |
| [MultiVarQuantifierTest.tla](MultiVarQuantifierTest.tla) | [.cfg](MultiVarQuantifierTest.cfg) | Multi-variable quantifiers |
| [RecursiveTest.tla](RecursiveTest.tla) | [.cfg](RecursiveTest.cfg) | RECURSIVE operator definitions |
| [ViewTest.tla](ViewTest.tla) | [.cfg](ViewTest.cfg) | VIEW for state symmetry |
| [OperatorSubstitutionTest.tla](OperatorSubstitutionTest.tla) | [.cfg](OperatorSubstitutionTest.cfg) | Operator argument substitution |

### Module System

| Spec | Config | Features Tested |
|------|--------|-----------------|
| [InstanceTest.tla](InstanceTest.tla) | [.cfg](InstanceTest.cfg) | INSTANCE with substitution |
| [InstanceTestSimple.tla](InstanceTestSimple.tla) | [.cfg](InstanceTestSimple.cfg) | Basic INSTANCE usage |
| [MultipleExtendsTest.tla](MultipleExtendsTest.tla) | [.cfg](MultipleExtendsTest.cfg) | Multiple EXTENDS clauses |
| [CoverageHelper.tla](CoverageHelper.tla) | - | Helper module for other tests |

### Temporal Properties

| Spec | Config | Features Tested |
|------|--------|-----------------|
| [LivenessTest.tla](LivenessTest.tla) | [.cfg](LivenessTest.cfg) | Eventually, Always, leads-to |
| [FairnessTest.tla](FairnessTest.tla) | [.cfg](FairnessTest.cfg) | Weak/strong fairness |
| [TemporalQuantTest.tla](TemporalQuantTest.tla) | [.cfg](TemporalQuantTest.cfg) | Temporal quantifiers (\E x: <>P) |
| [EnabledTest.tla](EnabledTest.tla) | [.cfg](EnabledTest.cfg) | ENABLED operator |

### Complex Models

| Spec | Config | Description |
|------|--------|-------------|
| [Combined.tla](Combined.tla) | [.cfg](Combined.cfg) | Multi-module production-style spec |
| [CombinedSimple.tla](CombinedSimple.tla) | [.cfg](CombinedSimple.cfg) | Simplified version of Combined |

### tlaplusplus Internal Models

TLA+ specifications modeling tlaplusplus's own internal algorithms:

| Spec | Config | Description |
|------|--------|-------------|
| [WorkQueue.tla](WorkQueue.tla) | [.cfg](WorkQueue.cfg) | Work-stealing queue with checkpoint/drain protocol |
| [FingerprintResize.tla](FingerprintResize.tla) | [.cfg](FingerprintResize.cfg) | Lock-free fingerprint store resize using seqlock |

## Language Coverage

The `language_coverage/` directory contains systematic tests for TLA+ language features:

| Spec | Config | Description |
|------|--------|-------------|
| [LanguageFeatureMatrix.tla](language_coverage/LanguageFeatureMatrix.tla) | [.cfg](language_coverage/LanguageFeatureMatrix.cfg) | Comprehensive language feature coverage |
| [LanguageFeatureMatrix.tla](language_coverage/LanguageFeatureMatrix.tla) | [Fair.cfg](language_coverage/LanguageFeatureMatrixFair.cfg) | Same spec with fairness properties |
| [LanguageFeatureMatrix.tla](language_coverage/LanguageFeatureMatrix.tla) | [NoEnabled.cfg](language_coverage/LanguageFeatureMatrix_NoEnabled.cfg) | Same spec without ENABLED |
| [InitNextTemporalQuant.tla](language_coverage/InitNextTemporalQuant.tla) | [.cfg](language_coverage/InitNextTemporalQuant.cfg) | Init/Next override with temporal quantifiers |

## Running Validation

```bash
# Compare tlaplusplus output against TLC for all specs
scripts/tlc_check.sh

# Run just the language coverage tests
scripts/tlc_corpus.sh
```
