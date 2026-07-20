---- MODULE LetHiddenPrime ----
EXTENDS Integers
VARIABLE msg
\* `Send` hides the prime: the LET body `Send(x)` has no literal `'`, but Send
\* assigns `msg'`. The compiled LET path must still stage it (compiled_expr.rs
\* gated LET-with-primes on a literal `'`, dropping helper-hidden primes ->
\* PaxosAccept produced 0 successors; worked around in #175, fixed directly here).
Send(m) == msg' = m
Init == msg = 0
Act == \E i \in {1, 2, 3} : LET x == i + 10 IN Send(x)
Next == Act
====
