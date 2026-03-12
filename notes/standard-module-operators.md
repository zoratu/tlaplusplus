# Vendored Standard Module Operator Inventory

This note is a practical inventory of the operators defined in
`tools/tla2sany/StandardModules/*.tla`.

Purpose:
- drive feature-completeness work against the actual vendored standard library
- make it obvious which operators still need parser/evaluator/runtime coverage
- complement `notes/tla-feature-reference.md`, which tracks surface areas more broadly

This is not meant to replace parser-driven extraction forever. It is a maintained
operator checklist for the modules vendored in this repo today.

## Bags

- `IsABag`
- `BagToSet`
- `SetToBag`
- `BagIn`
- `EmptyBag`
- `(+)`
- `(-)`
- `BagUnion`
- `\sqsubseteq`
- `SubBag`
- `BagOfAll`
- `BagCardinality`
- `CopiesIn`

## FiniteSets

- `IsFiniteSet`
- `Cardinality`

## Integers

- `Int`

## Naturals

- `Nat`
- `+`
- `-`
- `*`
- `^`
- `<`
- `>`
- `\leq`
- `\geq`
- `%`
- `\div`
- `..`

## Randomization

- `RandomSubset`
- `RandomSetOfSubsets`
- `TestRandomSetOfSubsets`

## RealTime

- `RTBound`
- `RTnow`

## Reals

- `Real`
- `/`
- `Infinity`

## Sequences

- `Seq`
- `Len`
- `\o`
- `Append`
- `Head`
- `Tail`
- `SubSeq`
- `SelectSeq`

## TLC

- `Print`
- `PrintT`
- `Assert`
- `JavaTime`
- `TLCGet`
- `TLCSet`
- `:>`
- `@@`
- `Permutations`
- `SortSeq`
- `RandomElement`
- `Any`
- `ToString`
- `TLCEval`

## Toolbox

- `_TETraceLength`

## Next step

Each operator above should eventually map to:
- parser support expectations, if syntactically special
- evaluator support expectations
- at least one corpus or public-example anchor
- direct regression tests for any previously broken operator
