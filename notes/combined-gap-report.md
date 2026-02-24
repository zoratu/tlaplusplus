# Combined.tla Native Support Gap Report

Generated from:

```bash
cargo run -- analyze-tla \
  --module /Volumes/OWC\ 1M2/Users/isaiah/src/openport/specs/Combined.tla \
  --config /Volumes/OWC\ 1M2/Users/isaiah/src/openport/specs/Combined.cfg
```

## Model structure

- Module: `Combined`
- Constants parsed: `18`
- Variables parsed: `25`
- Definitions parsed: `90`
- Key defs present: `Init`, `Next`, `Spec`
- `Next` top-level disjuncts: `31`
- Action-like definitions (contain primed vars or `UNCHANGED`): `32`
- Detected primed assignment clauses: `21`
- Detected `UNCHANGED` clauses: `17`
- Detected guard clauses: `81`
- `Init` probe seeding: `24` assignments evaluated, `1` unresolved (`clock`)

## Required feature surface (current model)

Feature counts are lexical (occurrence counts) and used to prioritize implementation effort:

- `record_dot`: 231
- `set_membership`: 164
- `prime`: 75
- `record_map`: 69
- `quant_exists`: 56
- `except`: 55
- `let_in`: 38
- `unchanged`: 32
- `quant_forall`: 24
- `union`: 22
- `if_then_else`: 20
- `set_minus`: 17
- `choose`: 6
- `always`: 5
- `subset`: 4
- `seq_concat`: 1
- `case`: 1
- `other`: 1

## Native frontend status

- `module_parse`: done
- `cfg_parse`: done
- `value_domain`: done
- `expr_eval`: done for the current action-clause probe surface
- `action_eval`: done for current `Next` branch probe surface

Expression probe against action clauses:

- Clause expressions probed: `102`
- Probe succeeded: `102`
- Probe failed: `0`

`Next` branch probe:

- Top-level disjuncts probed: `31`
- Supported disjuncts: `31`
- Unsupported disjuncts: `0`
- Successors generated from probe state: `18`

## Immediate implementation order for Combined

1. Runtime integration
   - plug native `Init`/`Next` execution into the main checker loop (fingerprint/queue engine) as a generic TLA model backend
2. Native `Init` completion
   - close the remaining unresolved `Init` assignment (`clock`) and remove fallback placeholders
3. Invariant/property checking path
   - run `INVARIANT` clauses from cfg against explored states in native runtime
4. Temporal surface
   - safety first (stable), then liveness (`[]`, `<>`, fairness) as follow-on

## Notes

- This is native Rust frontend work with no per-model adapters.
- The many-core host currently runs TLC Java for `Combined.tla`; native end-to-end model execution is still pending despite full clause-expression probe coverage.
