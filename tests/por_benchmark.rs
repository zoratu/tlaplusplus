// T7 — POR benchmark.
//
// Quantifies state-space reduction and wall-clock time for full vs
// POR-reduced enumeration on three multi-process synthetic specs.
//
// Run with `cargo test --release --test por_benchmark -- --nocapture` for
// the readable summary.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use tlaplusplus::Model;
use tlaplusplus::models::tla_native::TlaModel;
use tlaplusplus::tla::TlaState;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

fn enumerate_count_and_time(model: &TlaModel) -> (usize, std::time::Duration) {
    let started = Instant::now();
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<TlaState> = Vec::new();

    for s in model.initial_states() {
        let repr = serde_json::to_string(&s).expect("serialise");
        if seen.insert(repr) {
            frontier.push(s);
        }
    }

    let mut buf: Vec<TlaState> = Vec::new();
    while let Some(state) = frontier.pop() {
        buf.clear();
        model.next_states(&state, &mut buf);
        for next in buf.drain(..) {
            let repr = serde_json::to_string(&next).expect("serialise");
            if seen.insert(repr) {
                frontier.push(next);
            }
        }
    }

    (seen.len(), started.elapsed())
}

fn build(module_rel: &str, config_rel: &str) -> TlaModel {
    let module = workspace_path(module_rel);
    let config = workspace_path(config_rel);
    TlaModel::from_files(&module, Some(&config), None, None)
        .unwrap_or_else(|e| panic!("failed to build {}: {e:#}", module_rel))
}

fn bench_one(name: &str, module_rel: &str, config_rel: &str) -> (usize, usize) {
    let model_full = build(module_rel, config_rel);
    let (full_count, full_dur) = enumerate_count_and_time(&model_full);

    let mut model_por = build(module_rel, config_rel);
    model_por.enable_por().expect("POR enable");
    let (por_count, por_dur) = enumerate_count_and_time(&model_por);

    let reduction = if por_count > 0 {
        full_count as f64 / por_count as f64
    } else {
        f64::INFINITY
    };
    let speedup = if por_dur.as_secs_f64() > 0.0 {
        full_dur.as_secs_f64() / por_dur.as_secs_f64()
    } else {
        f64::INFINITY
    };

    println!(
        "{:40} full={:>8} ({:>8.3}ms)  por={:>8} ({:>8.3}ms)  reduction={:>5.1}x  speedup={:>5.1}x",
        name,
        full_count,
        full_dur.as_secs_f64() * 1000.0,
        por_count,
        por_dur.as_secs_f64() * 1000.0,
        reduction,
        speedup,
    );

    (full_count, por_count)
}

#[test]
fn por_benchmark_summary() {
    println!();
    println!("=== T7 POR Benchmark ===");
    println!(
        "{:40} {:>13}  {:>13}  {:>11}  {:>11}",
        "spec", "full(states/ms)", "por(states/ms)", "reduction", "speedup"
    );

    let (f1, p1) = bench_one(
        "PorBenchProcessGrid (4 procs, MAX=4)",
        "corpus/internals/PorBenchProcessGrid.tla",
        "corpus/internals/PorBenchProcessGrid.cfg",
    );
    assert!(p1 < f1, "expected POR reduction on PorBenchProcessGrid");
    assert!(
        (f1 as f64 / p1 as f64) >= 2.0,
        "expected >=2x state reduction on PorBenchProcessGrid (got {} vs {})",
        f1,
        p1
    );

    let (_f2, _p2) = bench_one(
        "PorBenchPipeline (3 procs, MAX=5)",
        "corpus/internals/PorBenchPipeline.tla",
        "corpus/internals/PorBenchPipeline.cfg",
    );

    let (f3, p3) = bench_one(
        "PorTwoCounters (2 procs, MAX=3)",
        "corpus/internals/PorTwoCounters.tla",
        "corpus/internals/PorTwoCounters.cfg",
    );
    assert!(p3 < f3, "expected POR reduction on PorTwoCounters");
}
