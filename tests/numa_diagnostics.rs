//! Comprehensive tests for NUMA diagnostics functionality
//!
//! These tests verify:
//! - NUMA node detection with mock data
//! - Diagnostic output formatting
//! - Edge cases (single NUMA node, no NUMA support)
//! - Worker NUMA assignments tracking
//! - Full diagnostic flow from worker spawn to timeout reporting
//! - Behavior when NUMA detection syscalls fail
//!
//! Run with: `cargo test --test numa_diagnostics`

use parking_lot::RwLock;
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Mock NumaTopology for Testing
// =============================================================================

/// Mock NUMA topology for testing without requiring actual NUMA hardware
#[derive(Debug, Clone)]
struct MockNumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Map from CPU ID to NUMA node
    pub cpu_to_node: HashMap<usize, usize>,
    /// NUMA distance matrix: distances[from_node][to_node]
    pub distances: Vec<Vec<u32>>,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
}

impl MockNumaTopology {
    /// Create a single NUMA node topology (common on desktops/laptops)
    fn single_node(num_cpus: usize) -> Self {
        let cpu_to_node: HashMap<usize, usize> = (0..num_cpus).map(|cpu| (cpu, 0)).collect();
        Self {
            node_count: 1,
            cpu_to_node,
            distances: vec![vec![10]], // Local distance = 10
            cpus_per_node: vec![(0..num_cpus).collect()],
        }
    }

    /// Create a dual-socket topology (typical server with 2 NUMA nodes)
    fn dual_socket(cpus_per_socket: usize) -> Self {
        let num_cpus = cpus_per_socket * 2;
        let mut cpu_to_node = HashMap::new();
        let mut cpus_per_node = vec![Vec::new(), Vec::new()];

        for cpu in 0..num_cpus {
            let node = if cpu < cpus_per_socket { 0 } else { 1 };
            cpu_to_node.insert(cpu, node);
            cpus_per_node[node].push(cpu);
        }

        Self {
            node_count: 2,
            cpu_to_node,
            // Typical dual-socket distances: local=10, remote=21
            distances: vec![vec![10, 21], vec![21, 10]],
            cpus_per_node,
        }
    }

    /// Create a 4-socket topology (high-end server)
    fn quad_socket(cpus_per_socket: usize) -> Self {
        let num_cpus = cpus_per_socket * 4;
        let mut cpu_to_node = HashMap::new();
        let mut cpus_per_node = vec![Vec::new(); 4];

        for cpu in 0..num_cpus {
            let node = cpu / cpus_per_socket;
            cpu_to_node.insert(cpu, node);
            cpus_per_node[node].push(cpu);
        }

        // Typical 4-socket distances
        // Diagonal = 10 (local), adjacent = 21, non-adjacent = 31
        let distances = vec![
            vec![10, 21, 21, 31],
            vec![21, 10, 31, 21],
            vec![21, 31, 10, 21],
            vec![31, 21, 21, 10],
        ];

        Self {
            node_count: 4,
            cpu_to_node,
            distances,
            cpus_per_node,
        }
    }

    /// Create a 6-NUMA node topology (AWS c7i.metal-24xl equivalent)
    fn six_node(cpus_per_node_count: usize) -> Self {
        let num_cpus = cpus_per_node_count * 6;
        let mut cpu_to_node = HashMap::new();
        let mut cpus_per_node = vec![Vec::new(); 6];

        for cpu in 0..num_cpus {
            let node = cpu / cpus_per_node_count;
            cpu_to_node.insert(cpu, node);
            cpus_per_node[node].push(cpu);
        }

        // 6-node distances (realistic for AMD EPYC or similar)
        // Local=10, adjacent=21, 2-hop=31, 3-hop=41
        let distances = vec![
            vec![10, 21, 21, 31, 31, 41],
            vec![21, 10, 31, 21, 41, 31],
            vec![21, 31, 10, 21, 31, 21],
            vec![31, 21, 21, 10, 21, 31],
            vec![31, 41, 31, 21, 10, 21],
            vec![41, 31, 21, 31, 21, 10],
        ];

        Self {
            node_count: 6,
            cpu_to_node,
            distances,
            cpus_per_node,
        }
    }

    /// Create empty/invalid NUMA topology (no NUMA support)
    fn no_numa_support() -> Self {
        Self {
            node_count: 1,
            cpu_to_node: HashMap::new(),
            distances: vec![vec![10]],
            cpus_per_node: vec![],
        }
    }

    /// Get NUMA node for a CPU
    fn cpu_to_node(&self, cpu_id: usize) -> usize {
        self.cpu_to_node.get(&cpu_id).copied().unwrap_or(0)
    }

    /// Get max distance from node 0
    fn max_distance_from_node0(&self) -> u32 {
        if self.distances.is_empty() {
            return 10;
        }
        self.distances[0].iter().copied().max().unwrap_or(10)
    }

    /// Find close nodes (within threshold distance)
    fn close_nodes(&self, reference_node: usize, threshold: u32) -> Vec<usize> {
        if reference_node >= self.distances.len() {
            return vec![0];
        }
        self.distances[reference_node]
            .iter()
            .enumerate()
            .filter(|(_, dist)| **dist <= threshold)
            .map(|(node, _)| node)
            .collect()
    }
}

/// Mock NUMA diagnostics that mirrors the real NumaDiagnostics structure
#[derive(Debug, Clone)]
struct MockNumaDiagnostics {
    pub node_count: usize,
    pub workers_by_node: Vec<Vec<usize>>,
    pub fingerprint_store_node: Option<usize>,
    pub distances: Vec<Vec<u32>>,
}

impl MockNumaDiagnostics {
    fn new(
        topology: &MockNumaTopology,
        worker_numa_nodes: &[usize],
        fingerprint_store_node: Option<usize>,
    ) -> Self {
        let mut workers_by_node: Vec<Vec<usize>> = vec![Vec::new(); topology.node_count.max(1)];
        for (worker_id, &node) in worker_numa_nodes.iter().enumerate() {
            if node < workers_by_node.len() {
                workers_by_node[node].push(worker_id);
            }
        }

        MockNumaDiagnostics {
            node_count: topology.node_count,
            workers_by_node,
            fingerprint_store_node,
            distances: topology.distances.clone(),
        }
    }

    /// Format startup diagnostics info as string (for testing output)
    fn format_startup_info(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("NUMA nodes detected: {}\n", self.node_count));

        if self.node_count > 1 {
            output.push_str("NUMA distance matrix:\n");
            for (from, row) in self.distances.iter().enumerate() {
                output.push_str(&format!("  Node {} -> {:?}\n", from, row));
            }
        }

        output.push_str("Worker distribution across NUMA nodes:\n");
        for (node, workers) in self.workers_by_node.iter().enumerate() {
            if !workers.is_empty() {
                output.push_str(&format!(
                    "  Node {}: {} workers (IDs: {:?})\n",
                    node,
                    workers.len(),
                    if workers.len() <= 10 {
                        workers.clone()
                    } else {
                        let mut preview = workers[..5].to_vec();
                        preview.push(usize::MAX);
                        preview.extend_from_slice(&workers[workers.len() - 3..]);
                        preview
                    }
                ));
            }
        }

        if let Some(fp_node) = self.fingerprint_store_node {
            output.push_str(&format!(
                "Fingerprint store memory located on NUMA node: {}\n",
                fp_node
            ));
        }

        output
    }

    /// Analyze stuck workers and return diagnostic info
    fn analyze_stuck_workers(
        &self,
        stuck_worker_ids: &[usize],
        worker_numa_nodes: &[usize],
    ) -> StuckWorkerAnalysis {
        let mut stuck_by_node: Vec<Vec<usize>> = vec![Vec::new(); self.node_count.max(1)];
        for &worker_id in stuck_worker_ids {
            if let Some(&node) = worker_numa_nodes.get(worker_id) {
                if node < stuck_by_node.len() {
                    stuck_by_node[node].push(worker_id);
                }
            }
        }

        let (local_stuck, remote_stuck) = if let Some(fp_node) = self.fingerprint_store_node {
            let mut local = 0usize;
            let mut remote = 0usize;

            for (node, workers) in stuck_by_node.iter().enumerate() {
                if workers.is_empty() {
                    continue;
                }
                let distance = self
                    .distances
                    .get(fp_node)
                    .and_then(|row| row.get(node))
                    .copied()
                    .unwrap_or(10);

                if distance <= 20 {
                    local += workers.len();
                } else {
                    remote += workers.len();
                }
            }
            (local, remote)
        } else {
            (stuck_worker_ids.len(), 0)
        };

        let potential_numa_issue = local_stuck + remote_stuck > 0
            && remote_stuck * 100 / (local_stuck + remote_stuck) > 60;

        StuckWorkerAnalysis {
            stuck_by_node,
            local_stuck,
            remote_stuck,
            potential_numa_issue,
        }
    }
}

#[derive(Debug)]
struct StuckWorkerAnalysis {
    stuck_by_node: Vec<Vec<usize>>,
    local_stuck: usize,
    remote_stuck: usize,
    potential_numa_issue: bool,
}

// =============================================================================
// Unit Tests: NUMA Node Detection
// =============================================================================

#[test]
fn test_single_node_topology() {
    let topo = MockNumaTopology::single_node(8);

    assert_eq!(topo.node_count, 1);
    assert_eq!(topo.cpu_to_node.len(), 8);
    assert_eq!(topo.distances, vec![vec![10]]);

    // All CPUs should map to node 0
    for cpu in 0..8 {
        assert_eq!(topo.cpu_to_node(cpu), 0);
    }
}

#[test]
fn test_dual_socket_topology() {
    let topo = MockNumaTopology::dual_socket(64);

    assert_eq!(topo.node_count, 2);
    assert_eq!(topo.cpu_to_node.len(), 128);

    // First 64 CPUs on node 0, rest on node 1
    for cpu in 0..64 {
        assert_eq!(topo.cpu_to_node(cpu), 0);
    }
    for cpu in 64..128 {
        assert_eq!(topo.cpu_to_node(cpu), 1);
    }

    // Check distances
    assert_eq!(topo.distances[0][0], 10); // Local
    assert_eq!(topo.distances[0][1], 21); // Remote
    assert_eq!(topo.distances[1][0], 21); // Remote
    assert_eq!(topo.distances[1][1], 10); // Local
}

#[test]
fn test_quad_socket_topology() {
    let topo = MockNumaTopology::quad_socket(32);

    assert_eq!(topo.node_count, 4);
    assert_eq!(topo.cpu_to_node.len(), 128);

    // Check node assignment
    assert_eq!(topo.cpu_to_node(0), 0);
    assert_eq!(topo.cpu_to_node(31), 0);
    assert_eq!(topo.cpu_to_node(32), 1);
    assert_eq!(topo.cpu_to_node(64), 2);
    assert_eq!(topo.cpu_to_node(96), 3);

    // Verify non-adjacent distance is higher
    assert_eq!(topo.distances[0][3], 31); // 0 to 3 is non-adjacent
}

#[test]
fn test_six_node_topology() {
    let topo = MockNumaTopology::six_node(64);

    assert_eq!(topo.node_count, 6);
    assert_eq!(topo.cpu_to_node.len(), 384);

    // Max distance from node 0
    assert_eq!(topo.max_distance_from_node0(), 41);
}

#[test]
fn test_no_numa_support_topology() {
    let topo = MockNumaTopology::no_numa_support();

    assert_eq!(topo.node_count, 1);
    assert!(topo.cpu_to_node.is_empty());
    assert!(topo.cpus_per_node.is_empty());

    // Should return default node 0 for any CPU
    assert_eq!(topo.cpu_to_node(0), 0);
    assert_eq!(topo.cpu_to_node(999), 0);
}

#[test]
fn test_close_nodes_calculation() {
    let topo = MockNumaTopology::quad_socket(32);

    // Nodes close to node 0 (threshold 20 = local only)
    let close = topo.close_nodes(0, 20);
    assert_eq!(close, vec![0]); // Only local

    // Nodes close to node 0 (threshold 21 = local + adjacent)
    let close = topo.close_nodes(0, 21);
    assert!(close.contains(&0));
    assert!(close.contains(&1));
    assert!(close.contains(&2));
    assert!(!close.contains(&3)); // Non-adjacent

    // All nodes (high threshold)
    let close = topo.close_nodes(0, 100);
    assert_eq!(close.len(), 4);
}

#[test]
fn test_unknown_cpu_defaults_to_node0() {
    let topo = MockNumaTopology::dual_socket(64);

    // CPU IDs beyond the topology should default to node 0
    assert_eq!(topo.cpu_to_node(1000), 0);
    assert_eq!(topo.cpu_to_node(usize::MAX), 0);
}

// =============================================================================
// Unit Tests: Diagnostic Output Formatting
// =============================================================================

#[test]
fn test_startup_info_single_node() {
    let topo = MockNumaTopology::single_node(4);
    let worker_numa_nodes = vec![0, 0, 0, 0];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    let output = diag.format_startup_info();

    assert!(output.contains("NUMA nodes detected: 1"));
    assert!(output.contains("Node 0: 4 workers"));
    assert!(output.contains("Fingerprint store memory located on NUMA node: 0"));
    // Single node should NOT show distance matrix
    assert!(!output.contains("distance matrix"));
}

#[test]
fn test_startup_info_dual_socket() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    let output = diag.format_startup_info();

    assert!(output.contains("NUMA nodes detected: 2"));
    assert!(output.contains("NUMA distance matrix"));
    assert!(output.contains("Node 0: 2 workers"));
    assert!(output.contains("Node 1: 2 workers"));
}

#[test]
fn test_startup_info_many_workers_truncation() {
    let topo = MockNumaTopology::single_node(100);
    let worker_numa_nodes = vec![0; 100];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, None);

    let output = diag.format_startup_info();

    // Should show 100 workers
    assert!(output.contains("100 workers"));
    // Should truncate worker IDs (showing first 5 + marker + last 3)
    assert!(output.contains(&format!("{}", usize::MAX))); // Marker for "..."
}

#[test]
fn test_startup_info_no_fingerprint_store_node() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, None);

    let output = diag.format_startup_info();

    // Should NOT contain fingerprint store node info
    assert!(!output.contains("Fingerprint store memory located"));
}

// =============================================================================
// Unit Tests: Stuck Worker Analysis
// =============================================================================

#[test]
fn test_stuck_worker_analysis_all_local() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 0, 1, 1, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Only workers on node 0 are stuck (local to FP store)
    let stuck_workers = vec![0, 1, 2];
    let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

    assert_eq!(analysis.local_stuck, 3);
    assert_eq!(analysis.remote_stuck, 0);
    assert!(!analysis.potential_numa_issue);
}

#[test]
fn test_stuck_worker_analysis_all_remote() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 0, 1, 1, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Only workers on node 1 are stuck (remote from FP store)
    let stuck_workers = vec![3, 4, 5];
    let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

    assert_eq!(analysis.local_stuck, 0);
    assert_eq!(analysis.remote_stuck, 3);
    assert!(analysis.potential_numa_issue); // >60% remote
}

#[test]
fn test_stuck_worker_analysis_mixed() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Mixed stuck workers: 2 local, 2 remote (50/50)
    let stuck_workers = vec![0, 1, 2, 3];
    let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

    assert_eq!(analysis.local_stuck, 2);
    assert_eq!(analysis.remote_stuck, 2);
    assert!(!analysis.potential_numa_issue); // 50% is not >60%
}

#[test]
fn test_stuck_worker_analysis_no_fingerprint_node() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, None);

    let stuck_workers = vec![0, 1, 2, 3];
    let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

    // Without FP node info, all are considered "local"
    assert_eq!(analysis.local_stuck, 4);
    assert_eq!(analysis.remote_stuck, 0);
}

#[test]
fn test_stuck_worker_by_node_grouping() {
    let topo = MockNumaTopology::quad_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Workers 0, 2, 4, 6 are stuck (one from each node)
    let stuck_workers = vec![0, 2, 4, 6];
    let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

    // Verify grouping by node
    assert_eq!(analysis.stuck_by_node[0], vec![0]);
    assert_eq!(analysis.stuck_by_node[1], vec![2]);
    assert_eq!(analysis.stuck_by_node[2], vec![4]);
    assert_eq!(analysis.stuck_by_node[3], vec![6]);
}

// =============================================================================
// Unit Tests: Edge Cases
// =============================================================================

#[test]
fn test_empty_worker_list() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes: Vec<usize> = vec![];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    assert!(diag.workers_by_node[0].is_empty());
    assert!(diag.workers_by_node[1].is_empty());
}

#[test]
fn test_invalid_node_in_worker_assignment() {
    let topo = MockNumaTopology::dual_socket(4);
    // Worker 2 assigned to non-existent node 99
    let worker_numa_nodes = vec![0, 1, 99];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Workers 0 and 1 should be assigned correctly
    assert_eq!(diag.workers_by_node[0], vec![0]);
    assert_eq!(diag.workers_by_node[1], vec![1]);
    // Worker 2 with invalid node should be ignored
}

#[test]
fn test_out_of_bounds_reference_node() {
    let topo = MockNumaTopology::dual_socket(4);

    // Reference node beyond topology
    let close = topo.close_nodes(99, 20);
    assert_eq!(close, vec![0]); // Should default to [0]
}

// =============================================================================
// Property-Based Tests: Worker NUMA Assignments
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_worker_assignment_tracked_correctly(
        num_workers in 1..100usize,
        node_count_input in 1..8usize,
    ) {
        // Clamp node_count to supported topologies: 1, 2, 4, or 6
        let node_count = match node_count_input {
            1 => 1,
            2 => 2,
            3 | 4 => 4,
            _ => 6,
        };

        // Generate random worker-to-node assignments (must use actual node_count)
        let worker_numa_nodes: Vec<usize> = (0..num_workers)
            .map(|i| i % node_count)
            .collect();

        let topo = match node_count {
            1 => MockNumaTopology::single_node(num_workers),
            2 => MockNumaTopology::dual_socket(num_workers / 2 + 1),
            4 => MockNumaTopology::quad_socket(num_workers / 4 + 1),
            _ => MockNumaTopology::six_node(num_workers / 6 + 1),
        };

        let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, None);

        // Verify all workers are tracked
        let total_tracked: usize = diag.workers_by_node.iter().map(|v| v.len()).sum();
        prop_assert_eq!(total_tracked, num_workers, "All workers should be tracked");

        // Verify each worker appears exactly once
        let mut all_workers: Vec<usize> = diag.workers_by_node.iter().flat_map(|v| v.iter().copied()).collect();
        all_workers.sort();
        let expected: Vec<usize> = (0..num_workers).collect();
        prop_assert_eq!(all_workers, expected, "Each worker should appear exactly once");
    }

    #[test]
    fn prop_stuck_worker_analysis_consistent(
        num_workers in 4..50usize,
        stuck_ratio in 0.1..0.9f64,
    ) {
        let topo = MockNumaTopology::dual_socket(num_workers / 2 + 1);
        let worker_numa_nodes: Vec<usize> = (0..num_workers).map(|i| i % 2).collect();
        let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

        // Select some workers to be stuck
        let num_stuck = ((num_workers as f64) * stuck_ratio) as usize;
        let stuck_workers: Vec<usize> = (0..num_stuck).collect();

        let analysis = diag.analyze_stuck_workers(&stuck_workers, &worker_numa_nodes);

        // Total stuck should match input
        let total_stuck = analysis.local_stuck + analysis.remote_stuck;
        prop_assert_eq!(total_stuck, num_stuck, "Total stuck should match input");

        // stuck_by_node should contain exactly the stuck workers
        let stuck_in_groups: Vec<usize> = analysis.stuck_by_node.iter().flat_map(|v| v.iter().copied()).collect();
        prop_assert_eq!(stuck_in_groups.len(), num_stuck, "Grouped stuck should match");
    }

    #[test]
    fn prop_close_nodes_includes_self(
        node_count in 2..8usize,
        reference_node in 0..8usize,
        threshold in 10..50u32,
    ) {
        let topo = match node_count {
            2 => MockNumaTopology::dual_socket(16),
            4 => MockNumaTopology::quad_socket(16),
            _ => MockNumaTopology::six_node(16),
        };

        let ref_node = reference_node % topo.node_count;
        let close = topo.close_nodes(ref_node, threshold);

        // Self should always be included (local distance = 10)
        prop_assert!(close.contains(&ref_node), "Reference node should be in close set");
    }

    #[test]
    fn prop_distances_are_symmetric(
        node_count in 2..6usize,
    ) {
        let topo = match node_count {
            2 => MockNumaTopology::dual_socket(16),
            4 => MockNumaTopology::quad_socket(16),
            _ => MockNumaTopology::six_node(16),
        };

        for i in 0..topo.node_count {
            for j in 0..topo.node_count {
                prop_assert_eq!(
                    topo.distances[i][j],
                    topo.distances[j][i],
                    "NUMA distances should be symmetric: d({},{}) != d({},{})",
                    i, j, j, i
                );
            }
        }
    }

    #[test]
    fn prop_local_distance_is_minimum(
        node_count in 1..6usize,
    ) {
        let topo = match node_count {
            1 => MockNumaTopology::single_node(16),
            2 => MockNumaTopology::dual_socket(16),
            4 => MockNumaTopology::quad_socket(16),
            _ => MockNumaTopology::six_node(16),
        };

        // Local distance (diagonal) should be minimum
        for i in 0..topo.node_count {
            let local_dist = topo.distances[i][i];
            prop_assert_eq!(local_dist, 10, "Local distance should be 10");

            for j in 0..topo.node_count {
                if i != j {
                    prop_assert!(
                        topo.distances[i][j] > local_dist,
                        "Remote distance should be > local: {} to {} is {}",
                        i, j, topo.distances[i][j]
                    );
                }
            }
        }
    }
}

// =============================================================================
// Integration Tests: Full Diagnostic Flow
// =============================================================================

/// Test pause controller with NUMA tracking
#[allow(dead_code)]
struct TestNUMAPauseController {
    pause_requested: AtomicBool,
    paused_workers: AtomicUsize,
    /// Worker NUMA node assignments (kept for potential future use)
    worker_numa_nodes: RwLock<Vec<usize>>,
    worker_paused_flags: RwLock<Vec<AtomicBool>>,
}

impl TestNUMAPauseController {
    fn new(num_workers: usize, worker_numa_nodes: Vec<usize>) -> Self {
        let paused_flags: Vec<AtomicBool> =
            (0..num_workers).map(|_| AtomicBool::new(false)).collect();

        Self {
            pause_requested: AtomicBool::new(false),
            paused_workers: AtomicUsize::new(0),
            worker_numa_nodes: RwLock::new(worker_numa_nodes),
            worker_paused_flags: RwLock::new(paused_flags),
        }
    }

    fn request_pause(&self) {
        self.pause_requested.store(true, Ordering::Release);
    }

    fn resume(&self) {
        self.pause_requested.store(false, Ordering::Release);
    }

    fn is_pause_requested(&self) -> bool {
        self.pause_requested.load(Ordering::Acquire)
    }

    fn worker_enter_pause(&self, worker_id: usize) {
        let flags = self.worker_paused_flags.read();
        if worker_id < flags.len() {
            flags[worker_id].store(true, Ordering::Release);
        }
        self.paused_workers.fetch_add(1, Ordering::AcqRel);
    }

    fn worker_exit_pause(&self, worker_id: usize) {
        let flags = self.worker_paused_flags.read();
        if worker_id < flags.len() {
            flags[worker_id].store(false, Ordering::Release);
        }
        self.paused_workers.fetch_sub(1, Ordering::AcqRel);
    }

    fn paused_count(&self) -> usize {
        self.paused_workers.load(Ordering::Acquire)
    }

    fn get_unpaused_workers(&self) -> Vec<usize> {
        let flags = self.worker_paused_flags.read();
        flags
            .iter()
            .enumerate()
            .filter(|(_, paused)| !paused.load(Ordering::Acquire))
            .map(|(id, _)| id)
            .collect()
    }

    #[allow(dead_code)]
    fn get_worker_numa_nodes(&self) -> Vec<usize> {
        self.worker_numa_nodes.read().clone()
    }
}

#[test]
fn test_full_diagnostic_flow_success() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let num_workers = 4;

    let pause = Arc::new(TestNUMAPauseController::new(
        num_workers,
        worker_numa_nodes.clone(),
    ));
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));
    let stop = Arc::new(AtomicBool::new(false));

    // Spawn workers
    let handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let pause = Arc::clone(&pause);
            let stop = Arc::clone(&stop);

            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if pause.is_pause_requested() {
                        pause.worker_enter_pause(worker_id);

                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(5));
                        }

                        pause.worker_exit_pause(worker_id);
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            })
        })
        .collect();

    // Let workers start
    thread::sleep(Duration::from_millis(50));

    // Request pause
    pause.request_pause();

    // Wait for quiescence
    let start = Instant::now();
    let timeout = Duration::from_secs(2);
    while pause.paused_count() < num_workers && start.elapsed() < timeout {
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(
        pause.paused_count(),
        num_workers,
        "All workers should be paused"
    );

    // Resume and stop
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    // Verify diagnostic output format
    let output = diag.format_startup_info();
    assert!(output.contains("NUMA nodes detected: 2"));
}

#[test]
fn test_diagnostic_flow_with_stuck_workers() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let num_workers = 4;

    let pause = Arc::new(TestNUMAPauseController::new(
        num_workers,
        worker_numa_nodes.clone(),
    ));
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));
    let stop = Arc::new(AtomicBool::new(false));

    // Spawn workers - worker 3 will be "stuck" (on remote NUMA node)
    let handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let pause = Arc::clone(&pause);
            let stop = Arc::clone(&stop);

            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if pause.is_pause_requested() {
                        // Worker 3 is stuck - never enters pause
                        if worker_id == 3 {
                            while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                                thread::sleep(Duration::from_millis(10));
                            }
                            continue;
                        }

                        pause.worker_enter_pause(worker_id);

                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(5));
                        }

                        pause.worker_exit_pause(worker_id);
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            })
        })
        .collect();

    // Let workers start
    thread::sleep(Duration::from_millis(50));

    // Request pause
    pause.request_pause();

    // Wait for partial quiescence (with short timeout)
    let start = Instant::now();
    let timeout = Duration::from_millis(500);
    while pause.paused_count() < num_workers && start.elapsed() < timeout {
        thread::sleep(Duration::from_millis(10));
    }

    // Should have 3 paused, 1 stuck
    assert_eq!(pause.paused_count(), 3);

    // Get stuck workers and analyze
    let unpaused = pause.get_unpaused_workers();
    assert_eq!(unpaused, vec![3]);

    let analysis = diag.analyze_stuck_workers(&unpaused, &worker_numa_nodes);

    // Worker 3 is on node 1, which is remote from FP store on node 0
    assert_eq!(analysis.remote_stuck, 1);
    assert_eq!(analysis.local_stuck, 0);

    // Resume and stop
    pause.resume();
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_diagnostics_dont_affect_normal_operation() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];
    let num_workers = 4;

    let pause = Arc::new(TestNUMAPauseController::new(
        num_workers,
        worker_numa_nodes.clone(),
    ));
    let _diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));
    let stop = Arc::new(AtomicBool::new(false));
    let work_counter = Arc::new(AtomicUsize::new(0));

    // Spawn workers doing actual work
    let handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let pause = Arc::clone(&pause);
            let stop = Arc::clone(&stop);
            let counter = Arc::clone(&work_counter);

            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if pause.is_pause_requested() {
                        pause.worker_enter_pause(worker_id);

                        while pause.is_pause_requested() && !stop.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(5));
                        }

                        pause.worker_exit_pause(worker_id);
                    } else {
                        // Do "work"
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Let workers run for a bit
    thread::sleep(Duration::from_millis(100));

    let work_before_pause = work_counter.load(Ordering::Relaxed);
    assert!(work_before_pause > 0, "Workers should have done work");

    // Pause
    pause.request_pause();
    thread::sleep(Duration::from_millis(100));

    // Work should have stopped
    let work_during_pause = work_counter.load(Ordering::Relaxed);

    // Resume
    pause.resume();
    thread::sleep(Duration::from_millis(100));

    let work_after_resume = work_counter.load(Ordering::Relaxed);
    assert!(
        work_after_resume > work_during_pause,
        "Work should resume after unpause"
    );

    // Stop
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }
}

// =============================================================================
// Chaos Tests: NUMA Detection Failures
// =============================================================================

/// Simulates NUMA detection syscall failure scenarios
mod numa_failure_simulation {
    use super::*;

    /// Mock topology that simulates detection failure
    #[derive(Debug)]
    struct FailingNumaTopology {
        should_fail: bool,
    }

    impl FailingNumaTopology {
        fn detect(&self) -> Result<MockNumaTopology, &'static str> {
            if self.should_fail {
                Err("NUMA detection failed: /sys/devices/system/node not accessible")
            } else {
                Ok(MockNumaTopology::single_node(4))
            }
        }
    }

    #[test]
    fn test_numa_detection_failure_fallback() {
        let failing = FailingNumaTopology { should_fail: true };

        // Should fail detection
        let result = failing.detect();
        assert!(result.is_err());

        // Fallback to single node
        let fallback_topo = MockNumaTopology::no_numa_support();
        assert_eq!(fallback_topo.node_count, 1);
        assert_eq!(fallback_topo.max_distance_from_node0(), 10);
    }

    #[test]
    fn test_numa_detection_success() {
        let working = FailingNumaTopology { should_fail: false };

        let result = working.detect();
        assert!(result.is_ok());
        let topo = result.unwrap();
        assert!(topo.node_count >= 1);
    }
}

/// Tests behavior when NUMA syscalls return partial/invalid data
mod numa_partial_data {
    use super::*;

    #[test]
    fn test_partial_cpu_to_node_mapping() {
        // Simulate scenario where only some CPUs have NUMA info
        let mut cpu_to_node = HashMap::new();
        cpu_to_node.insert(0, 0);
        cpu_to_node.insert(1, 0);
        // CPUs 2, 3 have no mapping

        let topo = MockNumaTopology {
            node_count: 2,
            cpu_to_node,
            distances: vec![vec![10, 21], vec![21, 10]],
            cpus_per_node: vec![vec![0, 1], vec![]],
        };

        // Unmapped CPUs should return default node 0
        assert_eq!(topo.cpu_to_node(0), 0);
        assert_eq!(topo.cpu_to_node(1), 0);
        assert_eq!(topo.cpu_to_node(2), 0); // Fallback
        assert_eq!(topo.cpu_to_node(3), 0); // Fallback
    }

    #[test]
    fn test_empty_distance_matrix() {
        let topo = MockNumaTopology {
            node_count: 2,
            cpu_to_node: HashMap::new(),
            distances: vec![], // Empty distances
            cpus_per_node: vec![],
        };

        // Should handle gracefully
        assert_eq!(topo.max_distance_from_node0(), 10); // Default
        assert_eq!(topo.close_nodes(0, 20), vec![0]); // Default fallback
    }

    #[test]
    fn test_mismatched_node_count_and_distances() {
        // node_count says 4, but distances only has 2 entries
        let topo = MockNumaTopology {
            node_count: 4,
            cpu_to_node: HashMap::new(),
            distances: vec![vec![10, 21], vec![21, 10]],
            cpus_per_node: vec![vec![], vec![], vec![], vec![]],
        };

        // close_nodes for node 3 should handle missing distance data
        let close = topo.close_nodes(3, 20);
        // Should fallback since node 3 is beyond distances matrix
        assert_eq!(close, vec![0]);
    }
}

/// Tests behavior on non-NUMA systems
#[test]
fn test_non_numa_system_behavior() {
    let topo = MockNumaTopology::no_numa_support();
    let worker_numa_nodes = vec![0, 0, 0, 0]; // All on node 0
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // Should work correctly
    assert_eq!(diag.node_count, 1);
    assert_eq!(diag.workers_by_node.len(), 1);
    assert_eq!(diag.workers_by_node[0], vec![0, 1, 2, 3]);

    // Stuck worker analysis should not report NUMA issues
    let stuck = vec![0, 1];
    let analysis = diag.analyze_stuck_workers(&stuck, &worker_numa_nodes);

    assert_eq!(analysis.local_stuck, 2);
    assert_eq!(analysis.remote_stuck, 0);
    assert!(!analysis.potential_numa_issue);
}

/// Tests proper handling when querying memory NUMA node fails
#[test]
fn test_memory_numa_node_query_failure() {
    let topo = MockNumaTopology::dual_socket(4);
    let worker_numa_nodes = vec![0, 0, 1, 1];

    // Fingerprint store node is None (query failed)
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, None);

    let stuck = vec![0, 1, 2, 3];
    let analysis = diag.analyze_stuck_workers(&stuck, &worker_numa_nodes);

    // Without FP node info, all should be "local" and no NUMA issue
    assert_eq!(analysis.local_stuck, 4);
    assert_eq!(analysis.remote_stuck, 0);
    assert!(!analysis.potential_numa_issue);
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn stress_rapid_topology_queries() {
    let topo = MockNumaTopology::six_node(64);

    // Rapidly query topology
    for _ in 0..1000 {
        for cpu in 0..384 {
            let _ = topo.cpu_to_node(cpu);
        }
        for node in 0..6 {
            let _ = topo.close_nodes(node, 20);
        }
    }

    assert_eq!(topo.max_distance_from_node0(), 41);
}

#[test]
fn stress_concurrent_diagnostic_updates() {
    let topo = Arc::new(MockNumaTopology::dual_socket(64));
    let worker_numa_nodes: Vec<usize> = (0..128).map(|i| i % 2).collect();
    let diag = Arc::new(MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0)));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let diag = Arc::clone(&diag);
            let nodes = worker_numa_nodes.clone();

            thread::spawn(move || {
                for _ in 0..100 {
                    // Simulate concurrent diagnostic queries
                    let stuck: Vec<usize> = (0..32).collect();
                    let _ = diag.analyze_stuck_workers(&stuck, &nodes);
                    let _ = diag.format_startup_info();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_many_workers_single_node() {
    let topo = MockNumaTopology::single_node(1024);
    let worker_numa_nodes: Vec<usize> = vec![0; 1024];
    let diag = MockNumaDiagnostics::new(&topo, &worker_numa_nodes, Some(0));

    // All 1024 workers on node 0
    assert_eq!(diag.workers_by_node[0].len(), 1024);

    // Analyze many stuck workers
    let stuck: Vec<usize> = (0..512).collect();
    let analysis = diag.analyze_stuck_workers(&stuck, &worker_numa_nodes);

    assert_eq!(analysis.local_stuck, 512);
    assert_eq!(analysis.remote_stuck, 0);
}
