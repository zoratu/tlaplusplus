//! `run-adaptive-branching` subcommand handler (with memory-monitor thread).
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use crate::run_model;

use super::shared::{build_engine_config, print_stats, run_system_checks};

pub(crate) fn handle(
    max_depth: u32,
    min_branching: u32,
    max_branching: u32,
    memory_threshold_pct: u8,
    adjustment_interval_secs: u64,
    runtime: crate::cli::args::RuntimeArgs,
    storage: crate::cli::args::StorageArgs,
    _s3: crate::cli::args::S3Args,
) -> anyhow::Result<()> {
    run_system_checks(runtime.skip_system_checks);
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;
    use crate::models::adaptive_branching::AdaptiveBranchingModel;

    let model = AdaptiveBranchingModel::new(max_depth, min_branching, max_branching);
    let model_clone = model.clone();

    // Get memory limit from config
    let memory_max = runtime
        .memory_max_bytes
        .or_else(|| crate::system::cgroup_memory_max_bytes())
        .unwrap_or(16 * 1024 * 1024 * 1024); // 16GB default

    let threshold_bytes =
        (memory_max as f64 * (memory_threshold_pct as f64 / 100.0)) as u64;

    // Spawn monitoring thread
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();

    eprintln!("🚀 Starting adaptive branching test:");
    eprintln!("   Depth: {}", max_depth);
    eprintln!(
        "   Branching: {} → {} (adaptive)",
        min_branching, max_branching
    );
    eprintln!(
        "   Memory limit: {:.1} GB, threshold: {}% ({:.1} GB)",
        memory_max as f64 / (1024.0 * 1024.0 * 1024.0),
        memory_threshold_pct,
        threshold_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!("   Adjustment interval: {}s", adjustment_interval_secs);
    eprintln!();

    let monitor_thread = thread::spawn(move || {
        let mut cycles_since_ramp = 0;
        let ramp_up_delay = 3; // Wait 3 cycles before ramping up

        while !done_clone.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(adjustment_interval_secs));

            // Check memory usage (RSS of current process)
            #[cfg(target_os = "linux")]
            let mem_info_opt = procfs::process::Process::myself()
                .and_then(|p| p.stat())
                .map(|stat| stat.rss * 4096) // rss is in pages
                .ok();

            #[cfg(not(target_os = "linux"))]
            let mem_info_opt: Option<u64> = None; // Not available on non-Linux

            if let Some(mem_info) = mem_info_opt {
                let memory_pct = (mem_info as f64 / memory_max as f64) * 100.0;

                eprintln!(
                    "📊 Memory: {:.1} GB ({:.1}%), Branching factor: {}",
                    mem_info as f64 / (1024.0 * 1024.0 * 1024.0),
                    memory_pct,
                    model_clone.current_branching()
                );

                if mem_info > threshold_bytes {
                    // Memory pressure - back off immediately
                    eprintln!("⚠️  Memory threshold exceeded!");
                    model_clone.back_off();
                    cycles_since_ramp = 0;
                } else if memory_pct < 60.0 {
                    // Memory usage low and stable - consider ramping up
                    cycles_since_ramp += 1;
                    if cycles_since_ramp >= ramp_up_delay {
                        model_clone.ramp_up();
                        cycles_since_ramp = 0;
                    }
                }
            } else {
                // Memory monitoring not available - just ramp up conservatively
                eprintln!(
                    "📊 Memory monitoring N/A, Branching factor: {}",
                    model_clone.current_branching()
                );
                cycles_since_ramp += 1;
                if cycles_since_ramp >= ramp_up_delay {
                    model_clone.ramp_up();
                    cycles_since_ramp = 0;
                }
            }
        }
    });

    let config = build_engine_config(&runtime, &storage, false)?;
    let outcome = run_model(model, config)?;

    // Signal monitor thread to stop. The monitor is read-only stats
    // collection; a panic here doesn't compromise the run result we're
    // about to print, but log it so it's not invisible.
    done.store(true, Ordering::Relaxed);
    if let Err(panic) = monitor_thread.join() {
        eprintln!("warning: adaptive-branching monitor thread panicked: {panic:?}");
    }

    print_stats("adaptive-branching", &outcome.stats);
    if let Some(violation) = outcome.violation {
        println!("violation=true");
        println!("violation_message={}", violation.message);
        println!("violation_state={:?}", violation.state);
    } else {
        println!("violation=false");
    }
    Ok(())
}
