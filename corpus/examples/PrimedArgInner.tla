---- MODULE PrimedArgInner ----
EXTENDS Integers
\* Instantiated callee for PrimedArgMC. Its action `Receive` has NO leading
\* guard-disjunction (a straight conjunction), so it isolates the FIRST #188 site
\* — the primed-argument staged-context read in expand_action_call_multi — without
\* also exercising the leading-guard-disjunction split (the SECOND #188 site).
VARIABLES clock, seen
Receive(i, incoming) ==
  /\ clock' = [ clock EXCEPT ![i] = (clock[i] + 1) % 3 ]
  /\ seen'  = [ seen  EXCEPT ![i] = incoming ]
====
