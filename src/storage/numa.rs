// NUMA-aware memory allocation for optimal performance on multi-socket systems
//
// On 128-core AWS EC2 (dual-socket), NUMA placement matters:
// - Local memory access: ~60ns
// - Remote memory access: ~140ns (2.3x slower!)
//
// We map each fingerprint shard to the NUMA node of its primary workers.

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Detect NUMA topology of the system
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Map from CPU ID to NUMA node
    pub cpu_to_node: HashMap<usize, usize>,
}

impl NumaTopology {
    /// Detect NUMA topology from /sys/devices/system/node
    pub fn detect() -> Result<Self> {
        let node_dir = Path::new("/sys/devices/system/node");

        if !node_dir.exists() {
            // NUMA not available - assume single node
            eprintln!("NUMA topology not available, assuming single node");
            return Ok(Self {
                node_count: 1,
                cpu_to_node: HashMap::new(),
            });
        }

        let mut cpu_to_node = HashMap::new();
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

                        // Read CPUs for this node
                        let cpulist_path = entry.path().join("cpulist");
                        if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                            for cpu_id in parse_cpulist(&cpulist) {
                                cpu_to_node.insert(cpu_id, node_id);
                            }
                        }
                    }
                }
            }
        }

        let node_count = max_node + 1;

        eprintln!(
            "Detected NUMA topology: {} nodes, {} CPU mappings",
            node_count,
            cpu_to_node.len()
        );

        Ok(Self {
            node_count,
            cpu_to_node,
        })
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

/// Set preferred NUMA node for current thread's allocations (no-op in this implementation)
pub fn set_preferred_node(_node_id: usize) -> Result<()> {
    // Would use libc::set_mempolicy here, but it's complex
    // For now, mmap with huge pages is sufficient
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
