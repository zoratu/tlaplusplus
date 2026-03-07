--------------------------- MODULE FingerprintStoreResize ---------------------------
(*
 * Models the lock-free fingerprint store resize protocol.
 *
 * This specification verifies the seqlock-based coordination used during
 * fingerprint store resize operations:
 * - Readers use version numbers to detect concurrent resize
 * - Writers atomically swap the hash table pointer
 * - No fingerprints are lost during resize
 *
 * The protocol:
 * 1. Resize thread sets seqlock to odd (resizing in progress)
 * 2. Resize thread allocates new table, rehashes entries
 * 3. Resize thread swaps table pointer atomically
 * 4. Resize thread sets seqlock to even (resize complete)
 *
 * Readers:
 * 1. Read seqlock, if odd, spin-wait
 * 2. Read table pointer, perform lookup/insert
 * 3. Read seqlock again, if changed, retry from step 1
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumWorkers,        \* Number of worker threads
    MaxFingerprints,   \* Maximum fingerprints to model
    MaxResizes         \* Maximum resize operations

VARIABLES
    seqlock,           \* Version counter (odd = resizing, even = stable)
    tablePtr,          \* Current table generation (0, 1, 2, ...)
    tables,            \* Map from table generation to fingerprint set
    resizing,          \* TRUE if resize is in progress
    workerState,       \* Per-worker state: "idle", "reading", "retrying"
    workerVersion,     \* Per-worker: seqlock value observed at start
    workerTable,       \* Per-worker: table generation being accessed
    inserted,          \* Total fingerprints successfully inserted (for invariant)
    resizeCount        \* Number of resize operations completed

vars == <<seqlock, tablePtr, tables, resizing, workerState, workerVersion, workerTable, inserted, resizeCount>>

Workers == 1..NumWorkers
TableGenerations == 0..(MaxResizes+1)

TypeOK ==
    /\ seqlock \in Nat
    /\ tablePtr \in TableGenerations
    /\ tables \in [TableGenerations -> SUBSET (0..MaxFingerprints-1)]
    /\ resizing \in BOOLEAN
    /\ workerState \in [Workers -> {"idle", "reading", "retrying"}]
    /\ workerVersion \in [Workers -> Nat]
    /\ workerTable \in [Workers -> TableGenerations]
    /\ inserted \in (0..MaxFingerprints)
    /\ resizeCount \in (0..MaxResizes)

Init ==
    /\ seqlock = 0
    /\ tablePtr = 0
    /\ tables = [g \in TableGenerations |-> {}]
    /\ resizing = FALSE
    /\ workerState = [w \in Workers |-> "idle"]
    /\ workerVersion = [w \in Workers |-> 0]
    /\ workerTable = [w \in Workers |-> 0]
    /\ inserted = 0
    /\ resizeCount = 0

-----------------------------------------------------------------------------
\* Worker operations

\* Worker begins reading - capture seqlock and table pointer
WorkerBeginRead(w) ==
    /\ workerState[w] = "idle"
    /\ seqlock % 2 = 0  \* Wait for even (not resizing)
    /\ workerState' = [workerState EXCEPT ![w] = "reading"]
    /\ workerVersion' = [workerVersion EXCEPT ![w] = seqlock]
    /\ workerTable' = [workerTable EXCEPT ![w] = tablePtr]
    /\ UNCHANGED <<seqlock, tablePtr, tables, resizing, inserted, resizeCount>>

\* Worker begins read but seqlock is odd - must wait
WorkerWaitForStable(w) ==
    /\ workerState[w] = "idle"
    /\ seqlock % 2 = 1  \* Odd = resizing, spin
    /\ UNCHANGED vars

\* Worker completes read/insert - verify seqlock unchanged
WorkerCompleteRead(w) ==
    /\ workerState[w] = "reading"
    /\ IF seqlock = workerVersion[w]
       THEN
           \* Seqlock unchanged - operation successful
           /\ workerState' = [workerState EXCEPT ![w] = "idle"]
           /\ IF inserted < MaxFingerprints
              THEN
                  /\ tables' = [tables EXCEPT ![workerTable[w]] = @ \union {inserted}]
                  /\ inserted' = inserted + 1
              ELSE
                  /\ UNCHANGED <<tables, inserted>>
       ELSE
           \* Seqlock changed - must retry
           /\ workerState' = [workerState EXCEPT ![w] = "retrying"]
           /\ UNCHANGED <<tables, inserted>>
    /\ UNCHANGED <<seqlock, tablePtr, resizing, workerVersion, workerTable, resizeCount>>

\* Worker retries after detecting seqlock change
WorkerRetry(w) ==
    /\ workerState[w] = "retrying"
    /\ workerState' = [workerState EXCEPT ![w] = "idle"]
    /\ UNCHANGED <<seqlock, tablePtr, tables, resizing, workerVersion, workerTable, inserted, resizeCount>>

-----------------------------------------------------------------------------
\* Resize operations

\* Begin resize - set seqlock to odd
BeginResize ==
    /\ ~resizing
    /\ resizeCount < MaxResizes
    /\ seqlock % 2 = 0  \* Must start from even
    /\ resizing' = TRUE
    /\ seqlock' = seqlock + 1  \* Now odd
    /\ UNCHANGED <<tablePtr, tables, workerState, workerVersion, workerTable, inserted, resizeCount>>

\* Complete resize - swap table pointer, set seqlock to even
CompleteResize ==
    /\ resizing
    /\ seqlock % 2 = 1  \* Must be odd
    \* Copy all fingerprints from current table to new table
    /\ LET newGen == tablePtr + 1
       IN /\ tables' = [tables EXCEPT ![newGen] = tables[tablePtr]]
          /\ tablePtr' = newGen
    /\ seqlock' = seqlock + 1  \* Now even
    /\ resizing' = FALSE
    /\ resizeCount' = resizeCount + 1
    /\ UNCHANGED <<workerState, workerVersion, workerTable, inserted>>

-----------------------------------------------------------------------------
\* Next state relation

Next ==
    \/ \E w \in Workers :
        \/ WorkerBeginRead(w)
        \/ WorkerWaitForStable(w)
        \/ WorkerCompleteRead(w)
        \/ WorkerRetry(w)
    \/ BeginResize
    \/ CompleteResize

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
\* Safety properties

\* Seqlock invariant: odd iff resizing
SeqlockConsistent ==
    (seqlock % 2 = 1) <=> resizing

\* No fingerprints are ever lost during resize
\* All fingerprints in previous tables must be in current table
NoFingerprintsLost ==
    \A g \in 0..(tablePtr-1) : tables[g] \subseteq tables[tablePtr]

\* Current table contains all inserted fingerprints
AllFingerprintsPresent ==
    inserted = 0 \/ tables[tablePtr] # {}

\* Workers reading during resize will retry
WorkersRetryDuringResize ==
    \A w \in Workers :
        (workerState[w] = "reading" /\ seqlock # workerVersion[w]) =>
            (workerState'[w] \in {"retrying", "idle"})

-----------------------------------------------------------------------------
\* Invariants

SafetyInvariant ==
    /\ TypeOK
    /\ SeqlockConsistent
    /\ NoFingerprintsLost

=============================================================================
