---- MODULE ConstStateOmit ----
EXTENDS Integers
\* Regression for "constants are not carried in the fingerprinted state"
\* (they belong to the fixed model context, not the state). Exercises a scalar
\* constant `MaxN` (a guard) and a model value `NULL` (initial value + equality
\* in the invariant), so if model-value resolution regressed after removing the
\* self-referential `NULL == NULL` def, this spec's count or verdict would move.
CONSTANTS MaxN, NULL
VARIABLES x, last
Init == x = 0 /\ last = NULL
Next == /\ x < MaxN
        /\ x' = x + 1
        /\ last' = x
Inv == (x = 0) => (last = NULL)
====
