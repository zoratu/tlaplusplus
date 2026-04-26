// T8 — Queue compression benchmark.
//
// Quantifies the in-memory queue compression layer in isolation. We push a
// large workload of TLA+-shaped state objects through the spillable queue
// with compression on and off, then measure:
//   - Wall-clock time for push + drain.
//   - Peak resident set size (process RSS) during the run.
//   - Compression ratio achieved when on.
//
// Run with: cargo test --release --test queue_compression_benchmark -- --nocapture
//
// This is a focused micro-benchmark of the queue layer because forcing a
// real spec to spill requires very specific timing/scale; the value of T8
// is purely in the queue tier and is best measured there directly.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tlaplusplus::storage::spillable_work_stealing::{SpillableConfig, SpillableWorkStealingQueues};
use tlaplusplus::system::get_memory_stats;

/// State shape mimics a record-heavy TLA+ state: a small fixed integer
/// header plus a string-keyed map serialised as Vec<(String, i64)>. This
/// shape compresses well with zstd because the string keys repeat across
/// every state, which is also true of real TLA+ records.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct BenchState {
    pc: i64,
    epoch: i64,
    counter: i64,
    inbox: Vec<(String, i64)>,
}

fn make_state(seed: i64) -> BenchState {
    BenchState {
        pc: seed % 8,
        epoch: seed / 1024,
        counter: seed,
        inbox: vec![
            ("alice".to_string(), seed.wrapping_mul(7) & 0xFFFF),
            ("bob".to_string(), seed.wrapping_mul(11) & 0xFFFF),
            ("carol".to_string(), seed.wrapping_mul(13) & 0xFFFF),
            ("dave".to_string(), seed.wrapping_mul(17) & 0xFFFF),
        ],
    }
}

fn run_workload(
    n_items: u64,
    chunk: u64,
    compression_enabled: bool,
    workers: usize,
) -> (std::time::Duration, u64, Option<(u64, u64, f64)>) {
    let dir = std::env::temp_dir().join(format!(
        "t8-bench-{}-{}",
        compression_enabled,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let config = SpillableConfig {
        // Tight memory budget so the spill path engages.
        max_inmem_items: 5_000,
        spill_dir: dir.clone(),
        spill_batch: 4_096,
        load_existing: false,
        worker_spill_buffer_size: 1_024,
        worker_channel_bound: 32,
        defer_segment_deletion: false,
        compression_enabled,
        compression_max_bytes: 256 * 1024 * 1024,
        compression_level: 1,
    };

    let (queues, mut worker_states) =
        SpillableWorkStealingQueues::<BenchState>::new(workers, vec![0; workers], config)
            .expect("create queues");

    let start = Instant::now();
    let baseline_rss = get_memory_stats().rss_bytes;
    let mut peak_rss = baseline_rss;

    // Push phase
    let mut next: u64 = 0;
    while next < n_items {
        let end = (next + chunk).min(n_items);
        let batch: Vec<BenchState> = (next..end).map(|i| make_state(i as i64)).collect();
        queues.push_local_batch(&mut worker_states[0], batch.into_iter());
        next = end;
        if next % (chunk * 32) == 0 {
            let rss = get_memory_stats().rss_bytes;
            if rss > peak_rss {
                peak_rss = rss;
            }
        }
    }
    queues.flush_worker_counters(&mut worker_states[0]);

    // Allow coordinator + ring + disk writers to settle.
    let settle_deadline = Instant::now() + std::time::Duration::from_secs(60);
    while Instant::now() < settle_deadline {
        let pending = queues.total_pending_count();
        let rss = get_memory_stats().rss_bytes;
        if rss > peak_rss {
            peak_rss = rss;
        }
        if pending >= n_items {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Capture peak after the push has fully landed.
    let rss = get_memory_stats().rss_bytes;
    if rss > peak_rss {
        peak_rss = rss;
    }
    let resident_above_baseline = peak_rss.saturating_sub(baseline_rss);

    // Drain phase — pop everything back through the queue.
    let mut popped: u64 = 0;
    let drain_deadline = Instant::now() + std::time::Duration::from_secs(120);
    while popped < n_items && Instant::now() < drain_deadline {
        let mut got_one = false;
        for w in 0..workers {
            if let Some(_state) = queues.pop_for_worker(&mut worker_states[w]) {
                popped += 1;
                got_one = true;
            }
        }
        if !got_one {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    assert_eq!(popped, n_items, "should round-trip every item");

    let elapsed = start.elapsed();

    // Capture compression ratio if enabled.
    let comp_info = queues
        .compression_stats()
        .map(|snap| (snap.bytes_uncompressed, snap.bytes_compressed, snap.ratio()));

    let _ = std::fs::remove_dir_all(&dir);
    (elapsed, resident_above_baseline, comp_info)
}

#[test]
#[ignore = "benchmark — run with --ignored --nocapture"]
fn bench_queue_compression() {
    bench_at_scale(250_000, 1_000, 4);
    bench_at_scale(1_000_000, 4_000, 8);
}

fn bench_at_scale(n_items: u64, chunk: u64, workers: usize) {
    eprintln!(
        "\n========== T8 queue-compression benchmark: {} items, chunk={}, workers={} ==========",
        n_items, chunk, workers
    );

    eprintln!();
    eprintln!("--- Compression OFF ---");
    let (off_time, off_rss, _) = run_workload(n_items, chunk, false, workers);
    eprintln!(
        "wall_time={:?} peak_rss_above_baseline={:.1} MiB",
        off_time,
        off_rss as f64 / (1024.0 * 1024.0)
    );

    eprintln!();
    eprintln!("--- Compression ON ---");
    let (on_time, on_rss, on_comp) = run_workload(n_items, chunk, true, workers);
    eprintln!(
        "wall_time={:?} peak_rss_above_baseline={:.1} MiB",
        on_time,
        on_rss as f64 / (1024.0 * 1024.0)
    );
    if let Some((unc, cmp, ratio)) = on_comp {
        eprintln!(
            "compression_ratio={:.2}x bytes_uncompressed={} bytes_compressed={}",
            ratio, unc, cmp
        );
    }

    eprintln!();
    eprintln!(
        "Time delta: ON / OFF = {:.2}x ({:+.1}%)",
        on_time.as_secs_f64() / off_time.as_secs_f64(),
        100.0 * (on_time.as_secs_f64() / off_time.as_secs_f64() - 1.0),
    );
    if off_rss > 0 {
        eprintln!(
            "Memory delta (peak above baseline): ON / OFF = {:.2}x ({:+.1}%)",
            on_rss as f64 / off_rss as f64,
            100.0 * (on_rss as f64 / off_rss as f64 - 1.0),
        );
    }
}
