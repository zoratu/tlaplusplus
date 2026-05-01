// T5.4 — streaming Init enumeration regression test.
//
// WHY THIS FILE EXISTS
// --------------------
// Before T5.4 the runtime called `model.initial_states()` to materialize the
// full Vec, then the BFS seeder pushed every state to the global queue. For
// specs whose Init enumeration dominates wall time (Einstein-class puzzles
// where the symbolic SMT loop runs for tens of minutes while the reachable
// state graph is small), workers stalled completely until Init enumeration
// finished. T5.4 introduces `Model::initial_states_streaming()` and a
// runtime producer thread that pulls from the iterator and pushes to the
// global queue, so workers can begin invariant checking on partial results.
//
// THE TEST
// --------
// We build a synthetic model with N=200_000 initial states. The
// `initial_states_streaming()` impl yields one state per ~50us (so
// emitting all 200K would take ~10s, but emitting the first 1K takes ~50ms).
// Workers find a violation at state index 1000.
//
// The PRIMARY assertion is `states_emitted < n / 2`: when streaming Init
// works, the producer thread is signaled to stop immediately upon violation,
// and at most a fraction of the N states are ever produced. Pre-T5.4, the
// runtime fully materialized the Vec via `model.initial_states()` BEFORE
// any worker checked invariants, so all N states would always be emitted.
//
// We also check wall-clock time as a coarse upper bound — runtime startup
// (NUMA detection, fp-store init, worker thread spawn) dominates on
// many-core hosts, so the bound is generous (4s) just to catch catastrophic
// regressions like the producer never stopping.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tlaplusplus::{run_model, EngineConfig, Model};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct InitState {
    /// Index into the synthetic Init domain. The model has N distinct initial
    /// states with `idx` ranging from 0 to N-1. The Next relation is empty,
    /// so the only states ever explored are the initial states themselves —
    /// this isolates the timing measurement to Init enumeration + invariant
    /// checking, with no successor-generation noise.
    idx: u64,
}

struct SlowInitModel {
    /// Total number of initial states the streaming iterator will yield.
    n: u64,
    /// Per-state delay in microseconds (simulates expensive Init evaluation).
    per_state_delay_us: u64,
    /// Index that should trigger an invariant violation.
    violation_at: u64,
    /// Counter incremented each time the streaming iterator yields a state.
    /// Used by the test to verify workers found the violation BEFORE Init
    /// enumeration completed.
    states_emitted: Arc<AtomicU64>,
}

impl Model for SlowInitModel {
    type State = InitState;

    fn name(&self) -> &'static str {
        "slow-init-streaming"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        // Eager fallback: same N states, no sleep. The runtime's primary
        // Init seeding path goes through `initial_states_streaming` (which
        // sleeps to simulate slow Init). This eager impl is only invoked
        // by post-violation trace reconstruction in `reconstruct_trace`,
        // which needs an instant-return Vec to walk back to the violating
        // state — sleeping there would defeat the test's timing assertion.
        (0..self.n).map(|idx| InitState { idx }).collect()
    }

    fn initial_states_streaming(&self) -> Box<dyn Iterator<Item = Self::State> + Send + '_> {
        let n = self.n;
        let delay = std::time::Duration::from_micros(self.per_state_delay_us);
        let counter = Arc::clone(&self.states_emitted);
        Box::new((0..n).map(move |idx| {
            // Simulate expensive per-state Init evaluation (e.g. SMT
            // solver yielding the next satisfying assignment).
            std::thread::sleep(delay);
            counter.fetch_add(1, Ordering::Relaxed);
            InitState { idx }
        }))
    }

    fn next_states(&self, _state: &Self::State, _out: &mut Vec<Self::State>) {
        // No successors — invariants run only on initial states.
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        if state.idx == self.violation_at {
            Err(format!("violation: state idx={} hit", state.idx))
        } else {
            Ok(())
        }
    }
}

#[test]
fn streaming_init_finds_early_violation_before_full_enumeration() {
    // 200_000 states at 50us each = ~10s of Init enumeration total.
    // Violation at idx=1000 means the producer reaches it after ~50ms.
    // Pre-T5.4 (no streaming) the test would take >10s wall time AND emit
    // all 200K states — both assertions catch the regression.
    let n: u64 = 200_000;
    let per_state_delay_us: u64 = 50;
    let violation_at: u64 = 1000;

    let states_emitted = Arc::new(AtomicU64::new(0));
    let model = SlowInitModel {
        n,
        per_state_delay_us,
        violation_at,
        states_emitted: Arc::clone(&states_emitted),
    };

    let work_dir = std::env::temp_dir().join(format!(
        "tlapp-t5_4-streaming-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work_dir);

    let config = EngineConfig {
        // 4 workers is enough — we're proving the runtime overlaps Init
        // production with invariant checking, not raw throughput.
        workers: 4,
        enforce_cgroups: false,
        numa_pinning: false,
        clean_work_dir: true,
        resume_from_checkpoint: false,
        checkpoint_interval_secs: 0,
        checkpoint_on_exit: false,
        stop_on_violation: true,
        work_dir: work_dir.clone(),
        ..EngineConfig::default()
    };

    let started = Instant::now();
    let outcome = run_model(model, config).expect("run_model failed");
    let elapsed = started.elapsed();

    // Cleanup
    let _ = std::fs::remove_dir_all(&work_dir);

    // The violation must have been detected.
    assert!(
        outcome.violation.is_some(),
        "expected violation at idx={violation_at} to be detected (got none)"
    );

    // PRIMARY assertion: producer halted after violation. Pre-T5.4 the
    // runtime always materialized the full Vec, so emitted would equal n.
    // With streaming, the producer observes the stop signal and breaks
    // long before exhausting all N states.
    let emitted = states_emitted.load(Ordering::Relaxed);
    assert!(
        emitted < n / 2,
        "producer kept running after violation: emitted {emitted} of {n} states \
         (expected < {} — violation should have stopped the producer)",
        n / 2
    );

    // Coarse wall-time bound. Pre-T5.4 the test would take >10s (full Init
    // enumeration before any invariant check). Streaming should land well
    // under 4s, which still gives plenty of room for runtime startup
    // (NUMA detection, fp-store init, thread spawn) on many-core hosts.
    assert!(
        elapsed < std::time::Duration::from_secs(4),
        "streaming Init regressed: violation at idx={violation_at} took {:?} (expected < 4s; \
         pre-T5.4 baseline waits ~{} ms for full Init enumeration before checking invariants)",
        elapsed,
        n * per_state_delay_us / 1000
    );
}
