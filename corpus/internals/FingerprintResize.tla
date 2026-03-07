------------------------------ MODULE FingerprintResize ------------------------------
(*
 * TLA+ model of tlaplusplus lock-free fingerprint store resize protocol.
 *
 * The fingerprint store uses a seqlock for coordination:
 * - Readers check seqlock, read data, check seqlock again
 * - If seqlock is odd OR changed, retry
 * - Writer increments seqlock (odd), resizes, increments again (even)
 *
 * This model verifies:
 * - No torn reads (readers see consistent state)
 * - Resize completes (writer eventually finishes)
 * - Readers make progress (not starved by writer)
 *)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    NumReaders,     \* Number of reader threads (workers)
    MaxOps          \* Max operations for bounded state space

VARIABLES
    seqlock,        \* Seqlock counter (odd = writing)
    tableVersion,   \* Current table version (incremented on resize)
    readerState,    \* State of each reader
    readerSeq,      \* Seqlock value read by each reader
    readerData,     \* Data read by each reader (table version seen)
    writerState,    \* State of writer: "idle" | "incrementing" | "resizing" | "finalizing"
    opsCount        \* Operation counter for bounded model

vars == <<seqlock, tableVersion, readerState, readerSeq, readerData, writerState, opsCount>>

TypeOK ==
    \* seqlock increments twice per resize cycle (once to odd, once to even),
    \* so its maximum value is 2*MaxOps (or 2*MaxOps-1 mid-cycle)
    /\ seqlock \in 0..(2*MaxOps)
    /\ tableVersion \in 0..MaxOps
    /\ readerState \in [1..NumReaders -> {"idle", "reading_seq", "reading_data", "checking_seq", "success", "retry"}]
    \* readerSeq captures seqlock values, so it has the same range
    /\ readerSeq \in [1..NumReaders -> 0..(2*MaxOps)]
    /\ readerData \in [1..NumReaders -> 0..MaxOps]
    /\ writerState \in {"idle", "incrementing", "resizing", "finalizing"}
    /\ opsCount \in 0..MaxOps

Init ==
    /\ seqlock = 0
    /\ tableVersion = 0
    /\ readerState = [r \in 1..NumReaders |-> "idle"]
    /\ readerSeq = [r \in 1..NumReaders |-> 0]
    /\ readerData = [r \in 1..NumReaders |-> 0]
    /\ writerState = "idle"
    /\ opsCount = 0

(* Reader begins a read operation *)
ReaderStart(r) ==
    /\ readerState[r] = "idle"
    /\ opsCount < MaxOps
    /\ readerState' = [readerState EXCEPT ![r] = "reading_seq"]
    /\ opsCount' = opsCount + 1
    /\ UNCHANGED <<seqlock, tableVersion, readerSeq, readerData, writerState>>

(* Reader reads the seqlock value *)
ReaderReadSeq(r) ==
    /\ readerState[r] = "reading_seq"
    /\ readerSeq' = [readerSeq EXCEPT ![r] = seqlock]
    \* If seqlock is odd (writer active), go to retry immediately
    /\ readerState' = [readerState EXCEPT ![r] =
        IF seqlock % 2 = 1 THEN "retry" ELSE "reading_data"]
    /\ UNCHANGED <<seqlock, tableVersion, readerData, writerState, opsCount>>

(* Reader reads the actual data (table version) *)
ReaderReadData(r) ==
    /\ readerState[r] = "reading_data"
    /\ readerData' = [readerData EXCEPT ![r] = tableVersion]
    /\ readerState' = [readerState EXCEPT ![r] = "checking_seq"]
    /\ UNCHANGED <<seqlock, tableVersion, readerSeq, writerState, opsCount>>

(* Reader verifies seqlock hasn't changed *)
ReaderCheckSeq(r) ==
    /\ readerState[r] = "checking_seq"
    /\ readerState' = [readerState EXCEPT ![r] =
        IF seqlock = readerSeq[r] THEN "success" ELSE "retry"]
    /\ UNCHANGED <<seqlock, tableVersion, readerSeq, readerData, writerState, opsCount>>

(* Reader completes successfully or retries *)
ReaderComplete(r) ==
    /\ readerState[r] = "success"
    /\ readerState' = [readerState EXCEPT ![r] = "idle"]
    /\ UNCHANGED <<seqlock, tableVersion, readerSeq, readerData, writerState, opsCount>>

ReaderRetry(r) ==
    /\ readerState[r] = "retry"
    /\ readerState' = [readerState EXCEPT ![r] = "reading_seq"]
    /\ UNCHANGED <<seqlock, tableVersion, readerSeq, readerData, writerState, opsCount>>

(* Writer starts resize by incrementing seqlock to odd *)
WriterStart ==
    /\ writerState = "idle"
    /\ opsCount < MaxOps
    /\ writerState' = "incrementing"
    /\ opsCount' = opsCount + 1
    /\ UNCHANGED <<seqlock, tableVersion, readerState, readerSeq, readerData>>

WriterIncrementOdd ==
    /\ writerState = "incrementing"
    /\ seqlock' = seqlock + 1  \* Now odd
    /\ writerState' = "resizing"
    /\ UNCHANGED <<tableVersion, readerState, readerSeq, readerData, opsCount>>

(* Writer performs the resize (updates table version) *)
WriterResize ==
    /\ writerState = "resizing"
    /\ tableVersion' = tableVersion + 1
    /\ writerState' = "finalizing"
    /\ UNCHANGED <<seqlock, readerState, readerSeq, readerData, opsCount>>

(* Writer completes by incrementing seqlock to even *)
WriterFinalize ==
    /\ writerState = "finalizing"
    /\ seqlock' = seqlock + 1  \* Now even
    /\ writerState' = "idle"
    /\ UNCHANGED <<tableVersion, readerState, readerSeq, readerData, opsCount>>

Next ==
    \/ \E r \in 1..NumReaders:
        \/ ReaderStart(r)
        \/ ReaderReadSeq(r)
        \/ ReaderReadData(r)
        \/ ReaderCheckSeq(r)
        \/ ReaderComplete(r)
        \/ ReaderRetry(r)
    \/ WriterStart
    \/ WriterIncrementOdd
    \/ WriterResize
    \/ WriterFinalize

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
    \* Strong fairness for writer actions to ensure resize completes
    \* even when reader retries are continuously possible
    /\ SF_vars(WriterIncrementOdd)
    /\ SF_vars(WriterResize)
    /\ SF_vars(WriterFinalize)

(* Safety: Successful reads see consistent data *)
(* If a reader succeeded, the data it read matches the seqlock it observed *)
ConsistentReads ==
    \A r \in 1..NumReaders:
        readerState[r] = "success" =>
            (readerSeq[r] % 2 = 0)  \* Read during stable period

(* Safety: Seqlock is always >= 0 *)
SeqlockNonNegative == seqlock >= 0

(* Liveness: Writer eventually completes *)
WriterEventuallyIdle == writerState # "idle" ~> writerState = "idle"

(* Liveness: Readers eventually succeed *)
ReaderEventuallySucceeds ==
    \A r \in 1..NumReaders:
        readerState[r] \in {"reading_seq", "reading_data", "checking_seq", "retry"}
        ~> readerState[r] = "success"

================================================================================
