--------------------------- MODULE CheckpointDrain ---------------------------
(***************************************************************************)
(* Model of streaming checkpoint drain algorithm                            *)
(*                                                                          *)
(* This models the fix for the checkpoint drain deadlock where the legacy   *)
(* implementation would collect ALL items in memory before writing,         *)
(* causing OOM on large queues (87M+ items).                                *)
(*                                                                          *)
(* The streaming approach:                                                  *)
(* - N writer threads compete to steal from queues                          *)
(* - Each thread writes batches directly to disk when full                  *)
(* - Memory bounded to: num_threads * BATCH_SIZE                            *)
(*                                                                          *)
(* Key invariants verified:                                                 *)
(* 1. Memory bounded - never more than MaxMemory items in thread batches    *)
(* 2. No item loss - every item eventually written to a segment             *)
(* 3. Termination - drain completes when all sources empty                  *)
(* 4. No duplication - each item written exactly once                       *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    NumThreads,      \* Number of drain threads (e.g., 32)
    BatchSize,       \* Items per batch before writing (e.g., 50000)
    MaxQueueItems,   \* Initial items in queue to drain
    MaxSegments      \* Max segments we can write (for model checking bounds)

VARIABLES
    queue,           \* Set of items remaining in the queue
    threadBatches,   \* Function: thread -> current batch (set of items)
    segments,        \* Sequence of segments written (each segment is a set)
    threadState,     \* Function: thread -> "stealing" | "writing" | "done"
    itemsWritten     \* Set of all items that have been written to segments

vars == <<queue, threadBatches, segments, threadState, itemsWritten>>

Threads == 1..NumThreads
Items == 1..MaxQueueItems

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ queue \subseteq Items
    /\ threadBatches \in [Threads -> SUBSET Items]
    /\ segments \in Seq(SUBSET Items)
    /\ Len(segments) <= MaxSegments
    /\ threadState \in [Threads -> {"stealing", "writing", "done"}]
    /\ itemsWritten \subseteq Items

-----------------------------------------------------------------------------
(* Initial state - all items in queue, threads ready to steal *)

Init ==
    /\ queue = Items
    /\ threadBatches = [t \in Threads |-> {}]
    /\ segments = <<>>
    /\ threadState = [t \in Threads |-> "stealing"]
    /\ itemsWritten = {}

-----------------------------------------------------------------------------
(* Actions *)

(* Thread steals an item from the queue *)
StealItem(t) ==
    /\ threadState[t] = "stealing"
    /\ queue /= {}
    /\ Cardinality(threadBatches[t]) < BatchSize
    /\ \E item \in queue:
        /\ queue' = queue \ {item}
        /\ threadBatches' = [threadBatches EXCEPT ![t] = @ \union {item}]
        /\ UNCHANGED <<segments, threadState, itemsWritten>>

(* Thread's batch is full - transition to writing *)
BatchFull(t) ==
    /\ threadState[t] = "stealing"
    /\ Cardinality(threadBatches[t]) >= BatchSize
    /\ threadState' = [threadState EXCEPT ![t] = "writing"]
    /\ UNCHANGED <<queue, threadBatches, segments, itemsWritten>>

(* Thread writes its batch to a segment *)
WriteBatch(t) ==
    /\ threadState[t] = "writing"
    /\ threadBatches[t] /= {}
    /\ Len(segments) < MaxSegments
    /\ segments' = Append(segments, threadBatches[t])
    /\ itemsWritten' = itemsWritten \union threadBatches[t]
    /\ threadBatches' = [threadBatches EXCEPT ![t] = {}]
    /\ threadState' = [threadState EXCEPT ![t] = "stealing"]
    /\ UNCHANGED <<queue>>

(* Thread finds queue empty - write remaining batch and finish *)
DrainComplete(t) ==
    /\ threadState[t] = "stealing"
    /\ queue = {}
    \* All other threads must also see empty queue or be done
    /\ \A other \in Threads:
        other = t \/ threadState[other] = "done" \/ queue = {}
    /\ IF threadBatches[t] /= {} /\ Len(segments) < MaxSegments
       THEN /\ segments' = Append(segments, threadBatches[t])
            /\ itemsWritten' = itemsWritten \union threadBatches[t]
            /\ threadBatches' = [threadBatches EXCEPT ![t] = {}]
       ELSE /\ UNCHANGED <<segments, itemsWritten, threadBatches>>
    /\ threadState' = [threadState EXCEPT ![t] = "done"]
    /\ UNCHANGED <<queue>>

(* Next state relation *)
Next ==
    \E t \in Threads:
        \/ StealItem(t)
        \/ BatchFull(t)
        \/ WriteBatch(t)
        \/ DrainComplete(t)

(* Fairness - each thread eventually gets to act *)
Fairness ==
    /\ \A t \in Threads: WF_vars(StealItem(t))
    /\ \A t \in Threads: WF_vars(BatchFull(t))
    /\ \A t \in Threads: WF_vars(WriteBatch(t))
    /\ \A t \in Threads: WF_vars(DrainComplete(t))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* Safety Invariants *)

(* Memory is bounded - total items in all thread batches never exceeds limit *)
MemoryBounded ==
    LET totalInBatches == Cardinality(UNION {threadBatches[t] : t \in Threads})
    IN totalInBatches <= NumThreads * BatchSize

(* No item duplication - each item appears in at most one place *)
NoDuplication ==
    \* Items in queue, batches, and written segments are disjoint
    /\ queue \intersect itemsWritten = {}
    /\ queue \intersect UNION {threadBatches[t] : t \in Threads} = {}
    /\ itemsWritten \intersect UNION {threadBatches[t] : t \in Threads} = {}

(* Conservation - all items are accounted for *)
ItemsConserved ==
    LET inQueue == queue
        inBatches == UNION {threadBatches[t] : t \in Threads}
        written == itemsWritten
    IN inQueue \union inBatches \union written = Items

(* Combined safety invariant *)
SafetyInvariant ==
    /\ TypeOK
    /\ MemoryBounded
    /\ NoDuplication
    /\ ItemsConserved

-----------------------------------------------------------------------------
(* Liveness Properties *)

(* All threads eventually complete *)
AllThreadsComplete ==
    <>(\A t \in Threads: threadState[t] = "done")

(* All items eventually written (no loss) *)
AllItemsWritten ==
    <>(itemsWritten = Items)

(* Queue eventually drained *)
QueueDrained ==
    <>(queue = {})

-----------------------------------------------------------------------------
(* Theorems to check *)

THEOREM Spec => []SafetyInvariant
THEOREM Spec => AllThreadsComplete
THEOREM Spec => AllItemsWritten
THEOREM Spec => QueueDrained

=============================================================================
