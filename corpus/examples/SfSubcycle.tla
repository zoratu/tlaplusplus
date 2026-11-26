---- MODULE SfSubcycle ----
EXTENDS Naturals
\* Strong-fairness subcycle discriminator (Codex tableau review #1, the key SF
\* case). Maximal SCC {0,1,2}. SF_vars(Drain) is enabled only at x=2 and, on a
\* behavior that stays in {0,1,2}, is never taken (Drain leaves to the x=3 sink).
\* The old SINGLE-PASS check tested SF over the whole maximal SCC {0,1,2}: Drain
\* enabled-some + never-taken-internally -> marks {0,1,2} UNFAIR -> excluded ->
\* MISSED the violation. But the fair recurrent set is the proper sub-SCC {0,1}
\* (Flip forever, x=2 never visited, Drain never enabled -> SF vacuous), which
\* never reaches x=3. So `<>(x=3)` is VIOLATED. The tableau removes the SF-enabled
\* state (x=2), re-decomposes, and finds {0,1}. TLC agrees.
VARIABLE x
vars == <<x>>
Init == x = 0
Flip == \/ (x = 0 /\ x' = 1)
        \/ (x = 1 /\ x' = 0)
Up == x = 1 /\ x' = 2
Down == x = 2 /\ x' = 0
Drain == x = 2 /\ x' = 3
Next == Flip \/ Up \/ Down \/ Drain
Spec == Init /\ [][Next]_vars /\ SF_vars(Drain)
EvDrain == <>(x = 3)
====
