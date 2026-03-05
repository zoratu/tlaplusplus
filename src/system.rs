use crate::storage::numa::NumaTopology;
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct WorkerPlan {
    pub worker_count: usize,
    pub assigned_cpus: Vec<Option<usize>>,
    /// NUMA node assignment for each worker (index = worker_id, value = numa_node)
    pub worker_numa_nodes: Vec<usize>,
    pub allowed_cpus: Vec<usize>,
    pub cgroup_cpuset_cores: Option<usize>,
    pub cgroup_quota_cores: Option<usize>,
    pub numa_nodes_used: usize,
    /// Number of workers per NUMA node
    pub workers_per_numa: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct WorkerPlanRequest {
    pub requested_workers: usize,
    pub enforce_cgroups: bool,
    pub enable_numa_pinning: bool,
    pub requested_core_ids: Option<Vec<usize>>,
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn read_first(paths: &[PathBuf]) -> Option<String> {
    for path in paths {
        if let Some(value) = read_trimmed(path) {
            return Some(value);
        }
    }
    None
}

fn cgroup_v2_relative_path() -> Option<String> {
    let content = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in content.lines() {
        let mut parts = line.splitn(3, ':');
        let hierarchy = parts.next()?;
        let controllers = parts.next()?;
        let path = parts.next()?;
        if hierarchy == "0" && controllers.is_empty() {
            return Some(path.to_string());
        }
    }
    None
}

fn cgroup_root_candidates(file: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(relative) = cgroup_v2_relative_path() {
        let rel = relative.trim_start_matches('/');
        out.push(PathBuf::from("/sys/fs/cgroup").join(rel).join(file));
    }
    out.push(PathBuf::from("/sys/fs/cgroup").join(file));
    out
}

pub fn parse_cpu_list(spec: &str) -> Result<Vec<usize>> {
    let mut cpus = BTreeSet::new();
    for raw_segment in spec.split(',') {
        let segment = raw_segment.trim();
        if segment.is_empty() {
            continue;
        }
        let segment = segment.split(':').next().unwrap_or(segment).trim();
        if let Some((start_raw, end_raw)) = segment.split_once('-') {
            let start = start_raw
                .trim()
                .parse::<usize>()
                .with_context(|| format!("invalid cpu id '{}'", start_raw.trim()))?;
            let end = end_raw
                .trim()
                .parse::<usize>()
                .with_context(|| format!("invalid cpu id '{}'", end_raw.trim()))?;
            if end < start {
                anyhow::bail!("invalid cpu range '{}': end < start", segment);
            }
            for cpu in start..=end {
                cpus.insert(cpu);
            }
        } else {
            let cpu = segment
                .parse::<usize>()
                .with_context(|| format!("invalid cpu id '{}'", segment))?;
            cpus.insert(cpu);
        }
    }
    if cpus.is_empty() {
        anyhow::bail!("cpu list is empty");
    }
    Ok(cpus.into_iter().collect())
}

fn online_cpu_list() -> Option<Vec<usize>> {
    let raw = read_first(&[
        PathBuf::from("/sys/devices/system/cpu/online"),
        PathBuf::from("/sys/devices/system/cpu/present"),
    ])?;
    parse_cpu_list(&raw).ok()
}

fn cgroup_cpuset() -> Option<Vec<usize>> {
    let mut candidates = cgroup_root_candidates("cpuset.cpus.effective");
    candidates.extend(cgroup_root_candidates("cpuset.cpus"));
    candidates.push(PathBuf::from("/sys/fs/cgroup/cpuset/cpuset.cpus"));
    let raw = read_first(&candidates)?;
    parse_cpu_list(&raw).ok()
}

fn cgroup_quota_cores() -> Option<usize> {
    let cpu_max = read_first(&cgroup_root_candidates("cpu.max"));
    if let Some(raw) = cpu_max {
        let mut parts = raw.split_whitespace();
        let quota_raw = parts.next()?;
        let period_raw = parts.next()?;
        if quota_raw == "max" {
            return None;
        }
        let quota = quota_raw.parse::<u64>().ok()?;
        let period = period_raw.parse::<u64>().ok()?;
        if period == 0 {
            return None;
        }
        let floor = (quota / period) as usize;
        return Some(floor.max(1));
    }

    let quota = read_first(&[PathBuf::from("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")])?
        .parse::<i64>()
        .ok()?;
    let period = read_first(&[PathBuf::from("/sys/fs/cgroup/cpu/cpu.cfs_period_us")])?
        .parse::<i64>()
        .ok()?;
    if quota <= 0 || period <= 0 {
        return None;
    }
    let floor = (quota / period) as usize;
    Some(floor.max(1))
}

pub fn cgroup_memory_max_bytes() -> Option<u64> {
    let mut candidates = cgroup_root_candidates("memory.max");
    candidates.push(PathBuf::from("/sys/fs/cgroup/memory/memory.limit_in_bytes"));
    let raw = read_first(&candidates)?;
    if raw == "max" {
        return None;
    }
    let value = raw.parse::<u64>().ok()?;
    if value == 0 { None } else { Some(value) }
}

fn discover_numa_nodes() -> Vec<Vec<usize>> {
    let node_root = Path::new("/sys/devices/system/node");
    let Ok(entries) = std::fs::read_dir(node_root) else {
        return Vec::new();
    };
    let mut nodes_with_id: Vec<(usize, Vec<usize>)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("node") {
            continue;
        }
        let Ok(node_id) = name.trim_start_matches("node").parse::<usize>() else {
            continue;
        };
        let cpulist_path = entry.path().join("cpulist");
        let Some(raw) = read_trimmed(&cpulist_path) else {
            continue;
        };
        let Ok(cpus) = parse_cpu_list(&raw) else {
            continue;
        };
        if !cpus.is_empty() {
            nodes_with_id.push((node_id, cpus));
        }
    }
    nodes_with_id.sort_by_key(|(id, _)| *id);
    nodes_with_id.into_iter().map(|(_, cpus)| cpus).collect()
}

fn intersect_sorted(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

pub fn build_worker_plan(req: WorkerPlanRequest) -> WorkerPlan {
    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| num_cpus::get())
        .max(1);

    let mut allowed_cpus = if let Some(mut user) = req.requested_core_ids {
        user.sort_unstable();
        user.dedup();
        user
    } else {
        cgroup_cpuset()
            .or_else(online_cpu_list)
            .unwrap_or_else(|| (0..host_cpus).collect())
    };
    allowed_cpus.sort_unstable();
    allowed_cpus.dedup();

    let cpuset_limit = cgroup_cpuset().map(|cpus| cpus.len()).filter(|n| *n > 0);
    let quota_limit = cgroup_quota_cores();

    if req.enforce_cgroups
        && let Some(cpuset) = cgroup_cpuset()
    {
        let mut cpuset_sorted = cpuset;
        cpuset_sorted.sort_unstable();
        cpuset_sorted.dedup();
        if !cpuset_sorted.is_empty() {
            allowed_cpus = intersect_sorted(&allowed_cpus, &cpuset_sorted);
        }
    }

    if allowed_cpus.is_empty() {
        allowed_cpus = (0..host_cpus).collect();
    }

    if req.enforce_cgroups
        && let Some(quota) = quota_limit
        && quota < allowed_cpus.len()
    {
        allowed_cpus.truncate(quota.max(1));
    }

    // Calculate optimal worker count based on NUMA topology when auto mode (workers = 0)
    let (optimal_workers, recommended_nodes) =
        if req.requested_workers == 0 && req.enable_numa_pinning {
            // Use NUMA-aware optimization: limit to nodes with low inter-node latency
            match NumaTopology::detect() {
                Ok(topology) => {
                    let (optimal, nodes) = topology.optimal_worker_count(&allowed_cpus);
                    eprintln!(
                        "NUMA-optimized worker count: {} workers across {} nodes (of {} available)",
                        optimal,
                        nodes.len(),
                        topology.node_count
                    );
                    (optimal, Some(nodes))
                }
                Err(_) => (allowed_cpus.len(), None),
            }
        } else {
            (allowed_cpus.len(), None)
        };

    let requested_workers = if req.requested_workers == 0 {
        optimal_workers.max(1)
    } else {
        req.requested_workers
    };

    // Filter allowed_cpus to recommended NUMA nodes if auto mode selected them
    let effective_allowed_cpus = if let Some(ref nodes) = recommended_nodes {
        // Only use CPUs from recommended NUMA nodes
        let topology = NumaTopology::detect().unwrap_or_else(|_| NumaTopology {
            node_count: 1,
            cpu_to_node: std::collections::HashMap::new(),
            distances: vec![vec![10]],
            cpus_per_node: vec![],
        });
        allowed_cpus
            .iter()
            .copied()
            .filter(|cpu| {
                let node = topology.cpu_to_node.get(cpu).copied().unwrap_or(0);
                nodes.contains(&node)
            })
            .collect::<Vec<_>>()
    } else {
        allowed_cpus.clone()
    };

    let worker_count = requested_workers
        .min(effective_allowed_cpus.len().max(1))
        .max(1);

    // Build NUMA-aware worker assignments using effective CPUs (filtered by NUMA optimization)
    let raw_nodes = discover_numa_nodes();
    let mut numa_nodes: Vec<Vec<usize>> = Vec::new();
    let mut effective_sorted = effective_allowed_cpus.clone();
    effective_sorted.sort_unstable();
    for node_cpus in &raw_nodes {
        let mut node_allowed = intersect_sorted(&effective_sorted, node_cpus);
        if !node_allowed.is_empty() {
            node_allowed.sort_unstable();
            node_allowed.dedup();
            numa_nodes.push(node_allowed);
        }
    }

    let (assigned_cpus, worker_numa_nodes) = if req.enable_numa_pinning && !numa_nodes.is_empty() {
        let mut cpus = Vec::with_capacity(worker_count);
        let mut numa_assignments = Vec::with_capacity(worker_count);

        for idx in 0..worker_count {
            let node_idx = idx % numa_nodes.len();
            let node_round = idx / numa_nodes.len();
            let node = &numa_nodes[node_idx];
            let cpu = node[node_round % node.len()];
            cpus.push(Some(cpu));
            numa_assignments.push(node_idx);
        }
        (cpus, numa_assignments)
    } else {
        // No NUMA pinning - all workers on "node 0"
        (vec![None; worker_count], vec![0; worker_count])
    };

    // Count workers per NUMA node
    let num_numa_nodes = numa_nodes.len().max(1);
    let mut workers_per_numa = vec![0usize; num_numa_nodes];
    for &node in &worker_numa_nodes {
        if node < workers_per_numa.len() {
            workers_per_numa[node] += 1;
        }
    }

    let numa_nodes_used = if req.enable_numa_pinning {
        let mut set = BTreeSet::new();
        for cpu in assigned_cpus.iter().flatten() {
            set.insert(*cpu);
        }
        if set.is_empty() {
            0
        } else {
            raw_nodes
                .into_iter()
                .filter(|node| node.iter().any(|cpu| set.contains(cpu)))
                .count()
        }
    } else {
        0
    };

    WorkerPlan {
        worker_count,
        assigned_cpus,
        worker_numa_nodes,
        allowed_cpus,
        cgroup_cpuset_cores: cpuset_limit,
        cgroup_quota_cores: quota_limit,
        numa_nodes_used,
        workers_per_numa,
    }
}

#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> Result<()> {
    // SAFETY: cpu_set_t is initialized before use and passed with its exact size.
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(cpu, &mut cpuset);
        let rc = libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &cpuset as *const libc::cpu_set_t,
        );
        if rc != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to set cpu affinity to core {}", cpu));
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread_to_cpu(_cpu: usize) -> Result<()> {
    Ok(())
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Resident set size in bytes
    pub rss_bytes: u64,
    /// Virtual memory size in bytes
    pub vms_bytes: u64,
    /// Available system memory in bytes
    pub available_bytes: u64,
    /// Total system memory in bytes
    pub total_bytes: u64,
}

impl MemoryStats {
    /// Memory pressure as percentage (0.0 - 1.0)
    pub fn pressure(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.rss_bytes as f64 / self.total_bytes as f64
    }

    /// Check if under memory pressure (>80% of total)
    pub fn is_under_pressure(&self) -> bool {
        self.pressure() > 0.80
    }

    /// Check if critical memory (>90% of total)
    pub fn is_critical(&self) -> bool {
        self.pressure() > 0.90
    }
}

/// Get current process memory usage
pub fn get_memory_stats() -> MemoryStats {
    let mut stats = MemoryStats::default();

    // Read /proc/self/status for process memory
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb) = parse_proc_kb(line) {
                    stats.rss_bytes = kb * 1024;
                }
            } else if line.starts_with("VmSize:") {
                if let Some(kb) = parse_proc_kb(line) {
                    stats.vms_bytes = kb * 1024;
                }
            }
        }
    }

    // Read /proc/meminfo for system memory
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb) = parse_proc_kb(line) {
                    stats.total_bytes = kb * 1024;
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(kb) = parse_proc_kb(line) {
                    stats.available_bytes = kb * 1024;
                }
            }
        }
    }

    // Also check cgroup memory limit
    if let Some(limit) = cgroup_memory_limit() {
        // Use cgroup limit as effective total if lower than system total
        if limit < stats.total_bytes && limit > 0 {
            stats.total_bytes = limit;
        }
    }

    stats
}

fn parse_proc_kb(line: &str) -> Option<u64> {
    // Format: "VmRSS:     12345 kB"
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].parse().ok()
    } else {
        None
    }
}

/// Get cgroup memory limit (if any)
pub fn cgroup_memory_limit() -> Option<u64> {
    // Try cgroup v2 first
    let v2_paths = [
        PathBuf::from("/sys/fs/cgroup/memory.max"),
        // Also try with the relative path
    ];

    for path in &v2_paths {
        if let Some(content) = read_trimmed(path) {
            if content != "max" {
                if let Ok(limit) = content.parse::<u64>() {
                    return Some(limit);
                }
            }
        }
    }

    // Try cgroup v1
    let v1_path = PathBuf::from("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    if let Some(content) = read_trimmed(&v1_path) {
        if let Ok(limit) = content.parse::<u64>() {
            // Ignore very large limits (essentially unlimited)
            if limit < u64::MAX / 2 {
                return Some(limit);
            }
        }
    }

    None
}

/// Memory monitor that can be polled periodically
pub struct MemoryMonitor {
    /// Memory limit in bytes (process will apply backpressure above this)
    pub limit_bytes: u64,
    /// Warning threshold (start applying backpressure)
    pub warn_threshold: f64,
    /// Critical threshold (emergency measures)
    pub critical_threshold: f64,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        // Get system/cgroup memory and set limit to 85%
        let stats = get_memory_stats();
        let effective_total = if stats.total_bytes > 0 {
            stats.total_bytes
        } else {
            // Fallback: assume 64GB
            64 * 1024 * 1024 * 1024
        };

        Self {
            limit_bytes: (effective_total as f64 * 0.85) as u64,
            warn_threshold: 0.75,
            critical_threshold: 0.90,
        }
    }
}

impl MemoryMonitor {
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes,
            warn_threshold: 0.75,
            critical_threshold: 0.90,
        }
    }

    /// Check current memory status
    pub fn check(&self) -> MemoryStatus {
        let stats = get_memory_stats();
        let usage_ratio = if self.limit_bytes > 0 {
            stats.rss_bytes as f64 / self.limit_bytes as f64
        } else {
            stats.pressure()
        };

        if usage_ratio >= self.critical_threshold {
            MemoryStatus::Critical {
                rss_bytes: stats.rss_bytes,
                limit_bytes: self.limit_bytes,
                ratio: usage_ratio,
            }
        } else if usage_ratio >= self.warn_threshold {
            MemoryStatus::Warning {
                rss_bytes: stats.rss_bytes,
                limit_bytes: self.limit_bytes,
                ratio: usage_ratio,
            }
        } else {
            MemoryStatus::Ok {
                rss_bytes: stats.rss_bytes,
                limit_bytes: self.limit_bytes,
                ratio: usage_ratio,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryStatus {
    Ok {
        rss_bytes: u64,
        limit_bytes: u64,
        ratio: f64,
    },
    Warning {
        rss_bytes: u64,
        limit_bytes: u64,
        ratio: f64,
    },
    Critical {
        rss_bytes: u64,
        limit_bytes: u64,
        ratio: f64,
    },
}

impl MemoryStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, MemoryStatus::Ok { .. })
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, MemoryStatus::Warning { .. })
    }

    pub fn is_critical(&self) -> bool {
        matches!(self, MemoryStatus::Critical { .. })
    }

    pub fn ratio(&self) -> f64 {
        match self {
            MemoryStatus::Ok { ratio, .. } => *ratio,
            MemoryStatus::Warning { ratio, .. } => *ratio,
            MemoryStatus::Critical { ratio, .. } => *ratio,
        }
    }
}

/// Disk space statistics for a path
#[derive(Debug, Clone, Default)]
pub struct DiskStats {
    /// Total disk space in bytes
    pub total_bytes: u64,
    /// Available disk space in bytes
    pub available_bytes: u64,
    /// Used disk space in bytes
    pub used_bytes: u64,
}

impl DiskStats {
    /// Check if disk is under pressure (>80% used)
    pub fn is_under_pressure(&self) -> bool {
        if self.total_bytes == 0 {
            return false;
        }
        self.used_bytes as f64 / self.total_bytes as f64 > 0.80
    }

    /// Check if disk is critical (>95% used or <1GB available)
    pub fn is_critical(&self) -> bool {
        if self.total_bytes == 0 {
            return false;
        }
        let used_ratio = self.used_bytes as f64 / self.total_bytes as f64;
        used_ratio > 0.95 || self.available_bytes < 1024 * 1024 * 1024 // <1GB
    }

    /// Get available space in human-readable format
    pub fn available_human(&self) -> String {
        let gb = self.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        format!("{:.1} GB", gb)
    }
}

/// Get disk space statistics for a given path
pub fn get_disk_stats(path: &Path) -> DiskStats {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem::MaybeUninit;

        let path_cstr = match CString::new(path.to_string_lossy().as_bytes()) {
            Ok(s) => s,
            Err(_) => return DiskStats::default(),
        };

        let mut statvfs = MaybeUninit::<libc::statvfs>::uninit();
        let result = unsafe { libc::statvfs(path_cstr.as_ptr(), statvfs.as_mut_ptr()) };

        if result == 0 {
            let statvfs = unsafe { statvfs.assume_init() };
            let block_size = statvfs.f_frsize as u64;
            DiskStats {
                total_bytes: statvfs.f_blocks as u64 * block_size,
                available_bytes: statvfs.f_bavail as u64 * block_size,
                used_bytes: (statvfs.f_blocks - statvfs.f_bfree) as u64 * block_size,
            }
        } else {
            DiskStats::default()
        }
    }

    #[cfg(not(unix))]
    {
        let _ = path;
        DiskStats::default()
    }
}

/// Check disk space and return an error if critical
pub fn check_disk_space(path: &Path) -> Result<DiskStats> {
    let stats = get_disk_stats(path);
    if stats.is_critical() {
        anyhow::bail!(
            "CRITICAL: Disk space critically low at {}: {} available ({}% used). \
             Free up disk space or reduce checkpoint/spill data.",
            path.display(),
            stats.available_human(),
            (stats.used_bytes as f64 / stats.total_bytes as f64 * 100.0) as u32
        );
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_list;

    #[test]
    fn parses_cpu_ranges() {
        let cpus = parse_cpu_list("0-3,8,10-11").expect("cpu list should parse");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 10, 11]);
    }

    #[test]
    fn parses_cpu_ranges_with_stride_suffix() {
        let cpus = parse_cpu_list("0-10:2,12").expect("cpu list should parse");
        assert_eq!(cpus, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]);
    }
}
