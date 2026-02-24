use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct WorkerPlan {
    pub worker_count: usize,
    pub assigned_cpus: Vec<Option<usize>>,
    pub allowed_cpus: Vec<usize>,
    pub cgroup_cpuset_cores: Option<usize>,
    pub cgroup_quota_cores: Option<usize>,
    pub numa_nodes_used: usize,
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

    let requested_workers = if req.requested_workers == 0 {
        allowed_cpus.len().max(1)
    } else {
        req.requested_workers
    };
    let worker_count = requested_workers.min(allowed_cpus.len().max(1)).max(1);

    let assigned_cpus = if req.enable_numa_pinning && !allowed_cpus.is_empty() {
        let raw_nodes = discover_numa_nodes();
        let mut nodes = Vec::new();
        for node_cpus in raw_nodes {
            let mut node_allowed = intersect_sorted(&allowed_cpus, &node_cpus);
            if !node_allowed.is_empty() {
                node_allowed.sort_unstable();
                node_allowed.dedup();
                nodes.push(node_allowed);
            }
        }
        if nodes.is_empty() {
            allowed_cpus
                .iter()
                .cycle()
                .take(worker_count)
                .map(|cpu| Some(*cpu))
                .collect()
        } else {
            let mut out = Vec::with_capacity(worker_count);
            for idx in 0..worker_count {
                let node_idx = idx % nodes.len();
                let node_round = idx / nodes.len();
                let node = &nodes[node_idx];
                let cpu = node[node_round % node.len()];
                out.push(Some(cpu));
            }
            out
        }
    } else {
        vec![None; worker_count]
    };

    let numa_nodes_used = if req.enable_numa_pinning {
        let mut set = BTreeSet::new();
        for cpu in assigned_cpus.iter().flatten() {
            set.insert(*cpu);
        }
        if set.is_empty() {
            0
        } else {
            discover_numa_nodes()
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
        allowed_cpus,
        cgroup_cpuset_cores: cpuset_limit,
        cgroup_quota_cores: quota_limit,
        numa_nodes_used,
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
