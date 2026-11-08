---- MODULE LetHiddenPrimeTop ----
EXTENDS Integers
VARIABLE msg
\* Like LetHiddenPrime, but the LET is a TOP-LEVEL action clause (not nested
\* under `\E`), so it reaches the compiled `compile_action_clause_text` LET arm
\* directly — the exact site the #175 fix guards. `Send` hides the prime: the LET
\* body `Send(x)` has no literal `'`, but Send assigns `msg'`. The compiled LET
\* path must still stage it (the pre-#175 gate required a literal `'` in the LET
\* text and dropped helper-hidden primes, compiling the clause as a boolean Guard
\* -> 0 successors). The chaff `let_helper_prime_as_guard` reintroduces exactly
\* that gate; this spec is its catcher.
Send(m) == msg' = m
Init == msg = 0
Act == LET x == 7 IN Send(x)
Next == Act
====
