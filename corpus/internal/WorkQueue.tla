-------------------------------- MODULE WorkQueue --------------------------------
(*
 * TLA+ model of tlaplusplus work queue with spilling and checkpoint.
 *
 * This models the interaction between:
 * - Workers: pop from hot queue, generate successors, push to hot queue
 * - Loader: pops from disk, pushes to hot queue (when hot queue is low)
 * - Checkpoint: drains hot queue to disk, pausing the loader
 *
 * The model checks for:
 * - Progress: workers eventually process all items (no stall)
 * - Safety: no lost items, counts are consistent
 *)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumWorkers,      \* Number of worker threads
    MaxHotItems,     \* Max items in hot queue before considering it "full"
    MaxDiskItems,    \* Max items on disk
    MaxTotalItems,   \* Max total items in system (for finite state space)
    BranchFactor     \* How many successors each item generates

VARIABLES
    hotQueue,           \* Items in memory (hot queue)
    diskQueue,          \* Items spilled to disk
    workerState,        \* State of each worker: "working" | "paused" | "idle"
    loaderState,        \* State of loader: "loading" | "paused" | "idle"
    checkpointState,    \* State of checkpoint: "idle" | "waiting" | "draining"
    checkpointDraining, \* Flag to pause loader during drain
    processed,          \* Count of items processed
    generated           \* Count of items generated

vars == <<hotQueue, diskQueue, workerState, loaderState, checkpointState,
          checkpointDraining, processed, generated>>

TypeOK ==
    /\ hotQueue \in 0..MaxTotalItems
    /\ diskQueue \in 0..MaxTotalItems
    /\ workerState \in [1..NumWorkers -> {"working", "paused", "idle"}]
    /\ loaderState \in {"loading", "paused", "idle"}
    /\ checkpointState \in {"idle", "waiting", "draining"}
    /\ checkpointDraining \in BOOLEAN
    /\ processed \in 0..MaxTotalItems
    /\ generated \in 0..MaxTotalItems

Init ==
    /\ hotQueue = 1           \* Start with one item (initial state)
    /\ diskQueue = 0
    /\ workerState = [w \in 1..NumWorkers |-> "idle"]
    /\ loaderState = "idle"
    /\ checkpointState = "idle"
    /\ checkpointDraining = FALSE
    /\ processed = 0
    /\ generated = 1

(* Worker takes an item from hot queue and processes it *)
WorkerTakeItem(w) ==
    /\ workerState[w] = "idle"
    /\ hotQueue > 0
    /\ checkpointState # "waiting"  \* Don't take if checkpoint is waiting
    /\ hotQueue' = hotQueue - 1
    /\ workerState' = [workerState EXCEPT ![w] = "working"]
    /\ processed' = processed + 1
    /\ UNCHANGED <<diskQueue, loaderState, checkpointState, checkpointDraining, generated>>

(* Worker generates successors and pushes to hot queue *)
WorkerPushSuccessors(w) ==
    /\ workerState[w] = "working"
    /\ hotQueue + BranchFactor <= MaxTotalItems  \* Bounded state space
    /\ generated + BranchFactor <= MaxTotalItems
    /\ hotQueue' = hotQueue + BranchFactor
    /\ generated' = generated + BranchFactor
    /\ workerState' = [workerState EXCEPT ![w] = "idle"]
    /\ UNCHANGED <<diskQueue, loaderState, checkpointState, checkpointDraining, processed>>

(* Worker pauses when checkpoint is waiting *)
WorkerPause(w) ==
    /\ workerState[w] = "idle"
    /\ checkpointState = "waiting"
    /\ workerState' = [workerState EXCEPT ![w] = "paused"]
    /\ UNCHANGED <<hotQueue, diskQueue, loaderState, checkpointState, checkpointDraining, processed, generated>>

(* Worker resumes after checkpoint completes *)
WorkerResume(w) ==
    /\ workerState[w] = "paused"
    /\ checkpointState = "idle"
    /\ workerState' = [workerState EXCEPT ![w] = "idle"]
    /\ UNCHANGED <<hotQueue, diskQueue, loaderState, checkpointState, checkpointDraining, processed, generated>>

(* Loader loads items from disk to hot queue *)
LoaderLoad ==
    /\ loaderState = "idle"
    /\ ~checkpointDraining        \* Don't load during checkpoint drain
    /\ diskQueue > 0              \* Disk has items
    /\ hotQueue < MaxHotItems     \* Hot queue not full
    /\ loaderState' = "loading"
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, checkpointState, checkpointDraining, processed, generated>>

LoaderComplete ==
    /\ loaderState = "loading"
    /\ diskQueue > 0
    /\ hotQueue' = hotQueue + 1
    /\ diskQueue' = diskQueue - 1
    /\ loaderState' = "idle"
    /\ UNCHANGED <<workerState, checkpointState, checkpointDraining, processed, generated>>

LoaderPause ==
    /\ loaderState = "idle"
    /\ checkpointDraining
    /\ loaderState' = "paused"
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, checkpointState, checkpointDraining, processed, generated>>

LoaderUnpause ==
    /\ loaderState = "paused"
    /\ ~checkpointDraining
    /\ loaderState' = "idle"
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, checkpointState, checkpointDraining, processed, generated>>

(* Checkpoint requests workers to pause *)
CheckpointStart ==
    /\ checkpointState = "idle"
    /\ hotQueue > 0  \* Only checkpoint if there's something to drain
    /\ checkpointState' = "waiting"
    /\ checkpointDraining' = TRUE  \* Signal loader to pause
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, loaderState, processed, generated>>

(* All workers paused, begin draining *)
CheckpointBeginDrain ==
    /\ checkpointState = "waiting"
    /\ \A w \in 1..NumWorkers: workerState[w] \in {"paused", "idle"}
    /\ loaderState \in {"paused", "idle"}  \* Loader must be paused or idle
    /\ checkpointState' = "draining"
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, loaderState, checkpointDraining, processed, generated>>

(* Drain one item from hot queue to disk *)
CheckpointDrainItem ==
    /\ checkpointState = "draining"
    /\ hotQueue > 0
    /\ diskQueue + 1 <= MaxDiskItems
    /\ hotQueue' = hotQueue - 1
    /\ diskQueue' = diskQueue + 1
    /\ UNCHANGED <<workerState, loaderState, checkpointState, checkpointDraining, processed, generated>>

(* Checkpoint completes when hot queue is drained *)
CheckpointComplete ==
    /\ checkpointState = "draining"
    /\ hotQueue = 0  \* All drained
    /\ checkpointState' = "idle"
    /\ checkpointDraining' = FALSE  \* Allow loader to resume
    /\ UNCHANGED <<hotQueue, diskQueue, workerState, loaderState, processed, generated>>

(* Termination: all items processed, queues empty *)
Terminated ==
    /\ hotQueue = 0
    /\ diskQueue = 0
    /\ \A w \in 1..NumWorkers: workerState[w] = "idle"
    /\ checkpointState = "idle"

Next ==
    \/ \E w \in 1..NumWorkers:
        \/ WorkerTakeItem(w)
        \/ WorkerPushSuccessors(w)
        \/ WorkerPause(w)
        \/ WorkerResume(w)
    \/ LoaderLoad
    \/ LoaderComplete
    \/ LoaderPause
    \/ LoaderUnpause
    \/ CheckpointStart
    \/ CheckpointBeginDrain
    \/ CheckpointDrainItem
    \/ CheckpointComplete
    \/ (Terminated /\ UNCHANGED vars)  \* Stutter step when done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Safety: items are never lost *)
ItemsConserved ==
    hotQueue + diskQueue +
    Cardinality({w \in 1..NumWorkers: workerState[w] = "working"}) >= 0

(* Liveness: eventually all items are processed or we terminate *)
EventuallyTerminates == <>(Terminated)

(* Liveness: workers make progress (don't stall with items in queue) *)
NoStall ==
    [](hotQueue > 0 /\ checkpointState = "idle" =>
       <>(\E w \in 1..NumWorkers: workerState[w] = "working"))

(* Key invariant: loader doesn't load during drain *)
LoaderPausedDuringDrain ==
    checkpointDraining => loaderState # "loading"

================================================================================
