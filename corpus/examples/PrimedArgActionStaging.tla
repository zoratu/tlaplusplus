---------------------------- MODULE PrimedArgActionStaging ----------------------------
(***************************************************************************)
(* Regression for the "operator called with a primed argument" staging     *)
(* bug (found while investigating the NanoBlockchain MCNanoSmall            *)
(* under-exploration; the canonical shape is Nano's                         *)
(* `CalculateHash(block, lastHash, lastHash')`).                            *)
(*                                                                          *)
(* `Bump(old, new)` is called `Bump(x, x')`: the second *argument* is the   *)
(* primed variable `x'`, so the parameter `new` it binds aliases the        *)
(* next-state variable `x'`, and the body `new = old + 1` is really the     *)
(* primed assignment `x' = x + 1`. A later conjunct reads the staged prime: *)
(* `seen' = seen \cup {x'}`.                                                *)
(*                                                                          *)
(* The bug: the interpreted `execute_branch` (and the compiled             *)
(* `try_eval_compiled_guard_as_action`) value-bound `new` to the (unstaged) *)
(* value of `x'` instead of substituting the argument text, so `new = old+1`*)
(* became a boolean guard, `x'` was never staged, and `Step` never fired    *)
(* -- exploration froze at the initial state and `Inv` held, masking the    *)
(* reachable violation. The fix substitutes the primed argument textually    *)
(* (matching the interpreted `eval_action_body_multi` path), so `x' = x + 1`*)
(* surfaces as a real primed assignment. Deliberately CHOOSE-free so the     *)
(* reachable state graph is a single deterministic path and the state count *)
(* agrees with TLC exactly.                                                  *)
(***************************************************************************)
EXTENDS Naturals

VARIABLES x, seen

Bump(old, new) == new = old + 1

Init == /\ x = 0
        /\ seen = {}

Step == /\ x < 3
        /\ Bump(x, x')
        /\ seen' = seen \cup {x'}

Next == Step

Spec == Init /\ [][Next]_<<x, seen>>

\* Reachable once `Step` fires three times (x: 0->1->2->3); the bug froze x at 0
\* so this held vacuously.
Inv == 3 \notin seen
=======================================================================================
