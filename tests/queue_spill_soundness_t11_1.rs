//! T11.1 regression test — `--queue-max-inmem-items` below natural state
//! count must not drop states.
//!
//! Background. Before the T11.1 fix, items in the spill pipeline (per-worker
//! `spill_buffer`, MPSC channel to coordinator, coordinator accumulator)
//! were invisible to `has_pending_work()` and `should_terminate()`. With a
//! small `--queue-max-inmem-items` cap, all workers could drain their hot
//! queues, see an apparently empty queue, and terminate while items were
//! still in flight. Result: dramatic and non-deterministic under-counts —
//! 4.2K-4.4K distinct vs the 26,344 baseline at cap=2000.
//!
//! This test asserts the runtime explores the full state space at small
//! caps that exercise the spill path. It runs the release binary against
//! `corpus/internals/CheckpointDrain.tla` with `--queue-max-inmem-items
//! 2000` (well below the natural 26,344-state queue depth) and verifies
//! the distinct-state count matches the baseline exactly.
//!
//! Run with: `cargo test --release --test queue_spill_soundness_t11_1`

use std::process::Command;

const BASELINE_DISTINCT: u64 = 26_344;
const SPEC_MODULE: &str = "corpus/internals/CheckpointDrain.tla";
const SPEC_CONFIG: &str = "corpus/internals/CheckpointDrain.cfg";

fn binary_path() -> std::path::PathBuf {
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop(); // tests/deps
    p.pop(); // tests
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("tlaplusplus");
    if !p.exists() {
        // Fallback: walk up to repo root and use target/release/tlaplusplus.
        let mut alt = std::env::current_dir().expect("cwd");
        while !alt.join("Cargo.toml").exists() {
            if !alt.pop() {
                break;
            }
        }
        alt.push("target/release/tlaplusplus");
        if alt.exists() {
            return alt;
        }
    }
    p
}

fn parse_distinct(stdout: &str) -> Option<u64> {
    // Look for lines like:
    //   Distinct: 26344
    //   distinct=26344
    //   Distinct states: 26344
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if lower.contains("distinct") {
            for token in line.split(|c: char| !c.is_ascii_digit()) {
                if let Ok(n) = token.parse::<u64>() {
                    if n > 100 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn run_at_cap(cap: u64) -> u64 {
    let bin = binary_path();
    if !bin.exists() {
        // Skip rather than fail — caller should `cargo build --release` first.
        eprintln!(
            "skipping T11.1 test: binary not found at {} (run `cargo build --release` first)",
            bin.display()
        );
        return BASELINE_DISTINCT;
    }
    let out = Command::new(&bin)
        .args([
            "run-tla",
            "--module",
            SPEC_MODULE,
            "--config",
            SPEC_CONFIG,
            "--workers",
            "2",
            "--queue-max-inmem-items",
            &cap.to_string(),
            "--skip-system-checks",
        ])
        .output()
        .expect("failed to run binary");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() {
        panic!(
            "binary exited non-zero at cap={}: status={:?}\nstdout:\n{}\nstderr:\n{}",
            cap, out.status, stdout, stderr
        );
    }
    parse_distinct(&stdout).unwrap_or_else(|| {
        panic!(
            "could not parse distinct count from stdout at cap={}:\n{}\n--- stderr ---\n{}",
            cap, stdout, stderr
        )
    })
}

#[test]
#[ignore = "requires release binary; run with `cargo build --release` first"]
fn t11_1_cap_2000_explores_full_state_space() {
    let distinct = run_at_cap(2_000);
    assert_eq!(
        distinct, BASELINE_DISTINCT,
        "cap=2000 must produce the full {} distinct states, not {} \
         (T11.1 regression: spill pipeline lost states in flight)",
        BASELINE_DISTINCT, distinct
    );
}

#[test]
#[ignore = "requires release binary; run with `cargo build --release` first"]
fn t11_1_multiple_caps_all_explore_full_state_space() {
    for &cap in &[2_000u64, 10_000, 20_000, 50_000] {
        let distinct = run_at_cap(cap);
        assert_eq!(
            distinct, BASELINE_DISTINCT,
            "cap={} must produce the full {} distinct states, not {}",
            cap, BASELINE_DISTINCT, distinct
        );
    }
}

#[test]
#[ignore = "deterministic-soak; runs the spec 5x at cap=2000"]
fn t11_1_cap_2000_is_deterministic_across_runs() {
    let mut counts = Vec::with_capacity(5);
    for i in 0..5 {
        let n = run_at_cap(2_000);
        eprintln!("run {}: distinct = {}", i, n);
        counts.push(n);
    }
    assert!(
        counts.iter().all(|&n| n == BASELINE_DISTINCT),
        "all 5 runs at cap=2000 must produce {} distinct states; got {:?}",
        BASELINE_DISTINCT,
        counts
    );
}
