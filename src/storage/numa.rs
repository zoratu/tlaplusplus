// NUMA-aware memory allocation for optimal performance on multi-socket systems
//
// On 128-core AWS EC2 (dual-socket), NUMA placement matters:
// - Local memory access: ~60ns
// - Remote memory access: ~140ns (2.3x slower!)
//
// We map each fingerprint shard to the NUMA node of its primary workers.
//
// For many-core systems (384+ cores, 6 NUMA nodes):
// - Cross-NUMA memory access causes 38%+ kernel time
// - Optimal worker count is limited by NUMA distances
// - States should be allocated on the local NUMA node

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// NUMA distance threshold for "local" access
/// Nodes with distance > this will be avoided for cross-node work
pub const NUMA_LOCAL_DISTANCE_THRESHOLD: u32 = 20;

/// Detect NUMA topology of the system
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Map from CPU ID to NUMA node
    pub cpu_to_node: HashMap<usize, usize>,
    /// NUMA distance matrix: distances[from_node][to_node]
    /// Distance 10 = local, higher = more latency
    pub distances: Vec<Vec<u32>>,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
}

impl NumaTopology {
    /// Detect NUMA topology from /sys/devices/system/node
    pub fn detect() -> Result<Self> {
        let node_dir = Path::new("/sys/devices/system/node");

        if !node_dir.exists() {
            // NUMA not available - assume single node
            if std::env::var("TLAPP_VERBOSE").is_ok() {
                eprintln!("NUMA topology not available, assuming single node");
            }
            return Ok(Self {
                node_count: 1,
                cpu_to_node: HashMap::new(),
                distances: vec![vec![10]], // Local distance
                cpus_per_node: vec![],
            });
        }

        let mut cpu_to_node = HashMap::new();
        let mut cpus_per_node: Vec<Vec<usize>> = Vec::new();
        let mut max_node = 0;

        // Scan for node directories
        if let Ok(entries) = fs::read_dir(node_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                // Look for node0, node1, etc.
                if name_str.starts_with("node") {
                    if let Ok(node_id) = name_str[4..].parse::<usize>() {
                        max_node = max_node.max(node_id);

                        // Ensure cpus_per_node is large enough
                        while cpus_per_node.len() <= node_id {
                            cpus_per_node.push(Vec::new());
                        }

                        // Read CPUs for this node
                        let cpulist_path = entry.path().join("cpulist");
                        if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                            for cpu_id in parse_cpulist(&cpulist) {
                                cpu_to_node.insert(cpu_id, node_id);
                                cpus_per_node[node_id].push(cpu_id);
                            }
                        }
                    }
                }
            }
        }

        let node_count = max_node + 1;

        // Read NUMA distances
        let distances = Self::read_numa_distances(node_count);

        eprintln!(
            "Detected NUMA topology: {} nodes, {} CPU mappings",
            node_count,
            cpu_to_node.len()
        );

        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!("NUMA distances:");
            for (from, row) in distances.iter().enumerate() {
                eprintln!("  Node {}: {:?}", from, row);
            }
        }

        Ok(Self {
            node_count,
            cpu_to_node,
            distances,
            cpus_per_node,
        })
    }

    /// Read NUMA distance matrix from /sys/devices/system/node/nodeX/distance
    fn read_numa_distances(node_count: usize) -> Vec<Vec<u32>> {
        let mut distances = vec![vec![10u32; node_count]; node_count];

        for from_node in 0..node_count {
            let distance_path = format!("/sys/devices/system/node/node{}/distance", from_node);
            if let Ok(content) = fs::read_to_string(&distance_path) {
                let parts: Vec<u32> = content
                    .trim()
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                for (to_node, &dist) in parts.iter().enumerate() {
                    if to_node < node_count {
                        distances[from_node][to_node] = dist;
                    }
                }
            }
        }

        distances
    }

    /// Get the maximum NUMA distance from node 0 to any other node
    pub fn max_distance_from_node0(&self) -> u32 {
        if self.distances.is_empty() {
            return 10;
        }
        self.distances[0].iter().copied().max().unwrap_or(10)
    }

    /// Find NUMA nodes that are "close" (distance <= threshold) to a reference node
    pub fn close_nodes(&self, reference_node: usize, threshold: u32) -> Vec<usize> {
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

    /// Calculate optimal number of workers based on NUMA topology
    /// Returns (optimal_worker_count, recommended_numa_nodes)
    pub fn optimal_worker_count(&self, available_cpus: &[usize]) -> (usize, Vec<usize>) {
        if self.node_count <= 1 || self.distances.is_empty() {
            // Single NUMA node or no topology info - use all CPUs
            return (available_cpus.len(), vec![0]);
        }

        // Find which NUMA nodes have available CPUs
        let mut nodes_with_cpus: Vec<(usize, usize)> = Vec::new(); // (node_id, cpu_count)
        for (node_id, node_cpus) in self.cpus_per_node.iter().enumerate() {
            let count = node_cpus
                .iter()
                .filter(|cpu| available_cpus.contains(cpu))
                .count();
            if count > 0 {
                nodes_with_cpus.push((node_id, count));
            }
        }

        if nodes_with_cpus.is_empty() {
            return (available_cpus.len(), vec![0]);
        }

        // Strategy: Use nodes that are "close" to each other (distance <= threshold)
        // Start with the node that has the most CPUs
        nodes_with_cpus.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        let primary_node = nodes_with_cpus[0].0;
        let close_nodes = self.close_nodes(primary_node, NUMA_LOCAL_DISTANCE_THRESHOLD);

        // Count CPUs on close nodes
        let mut total_cpus = 0;
        let mut selected_nodes = Vec::new();
        for &(node_id, count) in &nodes_with_cpus {
            if close_nodes.contains(&node_id) {
                total_cpus += count;
                selected_nodes.push(node_id);
            }
        }

        // If we have fewer than 64 CPUs on close nodes, try to include more nodes
        // up to 2x the threshold distance
        if total_cpus < 64 {
            let extended_close = self.close_nodes(primary_node, NUMA_LOCAL_DISTANCE_THRESHOLD * 2);
            for &(node_id, count) in &nodes_with_cpus {
                if !selected_nodes.contains(&node_id) && extended_close.contains(&node_id) {
                    total_cpus += count;
                    selected_nodes.push(node_id);
                }
            }
        }

        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "NUMA worker optimization: primary_node={}, selected_nodes={:?}, cpus={}",
                primary_node, selected_nodes, total_cpus
            );
        }

        (total_cpus, selected_nodes)
    }

    /// Get NUMA node for a CPU
    pub fn cpu_to_node(&self, cpu_id: usize) -> usize {
        self.cpu_to_node.get(&cpu_id).copied().unwrap_or(0)
    }

    /// Map shard IDs to NUMA nodes based on worker assignments
    ///
    /// Strategy: Assign each shard to the NUMA node of the worker
    /// that will access it most frequently.
    pub fn shard_to_numa_mapping(
        &self,
        shard_count: usize,
        worker_cpus: &[Option<usize>],
    ) -> Vec<usize> {
        let mut shard_to_numa = Vec::with_capacity(shard_count);

        for shard_id in 0..shard_count {
            // Find which worker will access this shard most
            // Simple heuristic: worker_id = shard_id % worker_count
            let worker_id = shard_id % worker_cpus.len();

            // Get the CPU and NUMA node for that worker
            let numa_node = worker_cpus
                .get(worker_id)
                .and_then(|&cpu_opt| cpu_opt)
                .map(|cpu| self.cpu_to_node(cpu))
                .unwrap_or(0);

            shard_to_numa.push(numa_node);
        }

        shard_to_numa
    }
}

/// Parse Linux cpulist format (e.g., "0-3,8,10-15")
fn parse_cpulist(cpulist: &str) -> Vec<usize> {
    let mut cpus = Vec::new();

    for part in cpulist.trim().split(',') {
        if part.contains('-') {
            // Range like "0-3"
            let mut range = part.split('-');
            if let (Some(start), Some(end)) = (range.next(), range.next()) {
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    for cpu in start..=end {
                        cpus.push(cpu);
                    }
                }
            }
        } else {
            // Single CPU
            if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }
    }

    cpus
}

/// Memory policy modes for set_mempolicy
#[cfg(target_os = "linux")]
mod mempolicy {
    pub const MPOL_DEFAULT: i32 = 0;
    pub const MPOL_PREFERRED: i32 = 1;
    pub const MPOL_BIND: i32 = 2;
    pub const MPOL_INTERLEAVE: i32 = 3;
    pub const MPOL_LOCAL: i32 = 4;
}

/// Set preferred NUMA node for current thread's allocations
/// This affects all subsequent memory allocations on this thread
#[cfg(target_os = "linux")]
pub fn set_preferred_node(node_id: usize) -> Result<()> {
    use std::io::Error;

    // Create nodemask with just this node set
    // nodemask is an array of unsigned long, with bit N representing node N
    let mut nodemask: [libc::c_ulong; 16] = [0; 16]; // Support up to 1024 nodes
    let word_idx = node_id / (std::mem::size_of::<libc::c_ulong>() * 8);
    let bit_idx = node_id % (std::mem::size_of::<libc::c_ulong>() * 8);

    if word_idx < nodemask.len() {
        nodemask[word_idx] = 1 << bit_idx;
    }

    // SAFETY: set_mempolicy is a system call that sets memory policy for the calling thread
    let result = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            mempolicy::MPOL_PREFERRED,
            nodemask.as_ptr(),
            (nodemask.len() * std::mem::size_of::<libc::c_ulong>() * 8) as libc::c_ulong,
        )
    };

    if result != 0 {
        let err = Error::last_os_error();
        if std::env::var("TLAPP_VERBOSE").is_ok() {
            eprintln!(
                "Warning: set_mempolicy for node {} failed: {}",
                node_id, err
            );
        }
        // Don't fail - just continue with default policy
    }

    Ok(())
}

/// Reset memory policy to default (local allocation)
#[cfg(target_os = "linux")]
pub fn reset_mempolicy() -> Result<()> {
    // SAFETY: set_mempolicy with MPOL_DEFAULT and NULL nodemask resets to default
    let result = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            mempolicy::MPOL_DEFAULT,
            std::ptr::null::<libc::c_ulong>(),
            0 as libc::c_ulong,
        )
    };

    if result != 0 && std::env::var("TLAPP_VERBOSE").is_ok() {
        eprintln!(
            "Warning: reset_mempolicy failed: {}",
            std::io::Error::last_os_error()
        );
    }

    Ok(())
}

/// Set preferred NUMA node for current thread's allocations (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn set_preferred_node(_node_id: usize) -> Result<()> {
    Ok(())
}

/// Reset memory policy to default (no-op on non-Linux)
#[cfg(not(target_os = "linux"))]
pub fn reset_mempolicy() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpulist_parsing() {
        assert_eq!(parse_cpulist("0"), vec![0]);
        assert_eq!(parse_cpulist("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0,2,4"), vec![0, 2, 4]);
        assert_eq!(
            parse_cpulist("0-3,8,10-12"),
            vec![0, 1, 2, 3, 8, 10, 11, 12]
        );
    }

    #[test]
    fn test_numa_detection() {
        let topology = NumaTopology::detect().unwrap();
        println!("Detected NUMA topology:");
        println!("  Nodes: {}", topology.node_count);
        println!(
            "  CPU->Node mappings: {} entries",
            topology.cpu_to_node.len()
        );

        // Should have at least 1 node
        assert!(topology.node_count >= 1);
    }

    #[test]
    fn test_shard_mapping() {
        let topology = NumaTopology::detect().unwrap();

        // Simulate 128 shards, 120 workers on CPUs 2-121
        let worker_cpus: Vec<Option<usize>> = (2..122).map(Some).collect();

        let mapping = topology.shard_to_numa_mapping(128, &worker_cpus);

        assert_eq!(mapping.len(), 128);

        // All mappings should be valid node IDs
        for &node in &mapping {
            assert!(node < topology.node_count);
        }

        println!(
            "Shard->NUMA mapping (first 16): {:?}",
            &mapping[..16.min(mapping.len())]
        );
    }
}
