use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar};
use std::thread::JoinHandle;

#[derive(Clone, Debug)]
pub struct DiskQueueConfig {
    pub spill_dir: PathBuf,
    pub inmem_limit: usize,
    pub spill_batch: usize,
    pub spill_channel_bound: usize,
    pub load_existing_segments: bool,
}

impl Default for DiskQueueConfig {
    fn default() -> Self {
        Self {
            spill_dir: PathBuf::from("./.tlapp/queue"),
            inmem_limit: 5_000_000,
            spill_batch: 50_000,
            spill_channel_bound: 128,
            load_existing_segments: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct QueueStats {
    pub pushed: u64,
    pub popped: u64,
    pub spilled_items: u64,
    pub spill_batches: u64,
    pub loaded_segments: u64,
    pub loaded_items: u64,
    pub max_inmem_len: u64,
}

#[derive(Default)]
struct QueueStatsAtomic {
    pushed: AtomicU64,
    popped: AtomicU64,
    spilled_items: AtomicU64,
    spill_batches: AtomicU64,
    loaded_segments: AtomicU64,
    loaded_items: AtomicU64,
    max_inmem_len: AtomicU64,
}

pub struct DiskBackedQueue<T> {
    inmem: SegQueue<T>,
    inmem_len: AtomicUsize,
    inmem_limit: usize,
    spill_batch: usize,
    spill_tx: Mutex<Option<Sender<Vec<T>>>>,
    segments: Arc<Mutex<VecDeque<PathBuf>>>,
    spill_inflight: Arc<AtomicUsize>,
    load_lock: Mutex<()>,
    writer_handle: Mutex<Option<JoinHandle<()>>>,
    error: Arc<Mutex<Option<String>>>,
    stats: QueueStatsAtomic,
    // Blocking dequeue support - efficient crossbeam channel notifications
    notify_tx: Sender<()>,
    notify_rx: Receiver<()>,
    finished: AtomicBool,
    pub num_waiting: AtomicUsize,
    num_workers: AtomicUsize,
}

impl<T> DiskBackedQueue<T>
where
    T: Serialize + DeserializeOwned + Send + 'static,
{
    pub fn new(config: DiskQueueConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.spill_dir).with_context(|| {
            format!(
                "failed to create queue spill dir {}",
                config.spill_dir.display()
            )
        })?;

        let mut existing_segments: Vec<(u64, PathBuf)> = Vec::new();
        if config.load_existing_segments {
            for entry in std::fs::read_dir(&config.spill_dir)
                .with_context(|| {
                    format!(
                        "failed reading queue spill dir {}",
                        config.spill_dir.display()
                    )
                })?
                .flatten()
            {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !(file_name.starts_with("segment-") && file_name.ends_with(".bin")) {
                    continue;
                }
                let raw = file_name
                    .trim_start_matches("segment-")
                    .trim_end_matches(".bin");
                let Ok(id) = raw.parse::<u64>() else {
                    continue;
                };
                existing_segments.push((id, path));
            }
            existing_segments.sort_by_key(|(id, _)| *id);
        }

        let (spill_tx, spill_rx) = crossbeam_channel::bounded::<Vec<T>>(config.spill_channel_bound);
        let segments = Arc::new(Mutex::new(VecDeque::new()));
        {
            let mut segment_queue = segments.lock();
            for (_, path) in &existing_segments {
                segment_queue.push_back(path.clone());
            }
        }
        let spill_inflight = Arc::new(AtomicUsize::new(0));
        let writer_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let writer_segments = Arc::clone(&segments);
        let writer_inflight = Arc::clone(&spill_inflight);
        let writer_error_ref = Arc::clone(&writer_error);
        let writer_dir = config.spill_dir.clone();
        let mut next_segment_id = 0u64;
        if let Some((id, _)) = existing_segments.last() {
            next_segment_id = id.saturating_add(1);
        }
        let writer_handle = std::thread::Builder::new()
            .name("tlapp-queue-spill".to_string())
            .spawn(move || {
                let mut segment_id = next_segment_id;
                while let Ok(batch) = spill_rx.recv() {
                    let segment_path = writer_dir.join(format!("segment-{segment_id:016}.bin"));
                    segment_id += 1;

                    let serialized = bincode::serialize(&batch);
                    match serialized {
                        Ok(bytes) => {
                            if let Err(err) = std::fs::write(&segment_path, &bytes) {
                                let mut guard = writer_error_ref.lock();
                                if guard.is_none() {
                                    *guard = Some(format!(
                                        "failed writing queue segment {}: {err}",
                                        segment_path.display()
                                    ));
                                }
                            } else {
                                writer_segments.lock().push_back(segment_path);
                            }
                        }
                        Err(err) => {
                            let mut guard = writer_error_ref.lock();
                            if guard.is_none() {
                                *guard =
                                    Some(format!("failed serializing queue spill batch: {err}"));
                            }
                        }
                    }

                    writer_inflight.fetch_sub(1, Ordering::Release);
                }
            })
            .context("failed to spawn queue spill writer")?;

        let (notify_tx, notify_rx) = unbounded();

        Ok(Self {
            inmem: SegQueue::new(),
            inmem_len: AtomicUsize::new(0),
            inmem_limit: config.inmem_limit.max(100),
            spill_batch: config.spill_batch.max(16),
            spill_tx: Mutex::new(Some(spill_tx)),
            segments,
            spill_inflight,
            load_lock: Mutex::new(()),
            writer_handle: Mutex::new(Some(writer_handle)),
            error: writer_error,
            stats: QueueStatsAtomic::default(),
            notify_tx,
            notify_rx,
            finished: AtomicBool::new(false),
            num_waiting: AtomicUsize::new(0),
            num_workers: AtomicUsize::new(0),
        })
    }

    fn set_error(&self, msg: String) {
        let mut guard = self.error.lock();
        if guard.is_none() {
            *guard = Some(msg);
        }
    }

    fn check_error(&self) -> Result<()> {
        let guard = self.error.lock();
        if let Some(msg) = guard.as_ref() {
            return Err(anyhow!(msg.clone()));
        }
        Ok(())
    }

    #[inline]
    fn update_max_inmem(&self) {
        let current = self.inmem_len.load(Ordering::Relaxed) as u64;
        let max = &self.stats.max_inmem_len;
        let mut seen = max.load(Ordering::Relaxed);
        while current > seen {
            match max.compare_exchange(seen, current, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => seen = next,
            }
        }
    }

    #[inline]
    fn try_reserve_inmem_slot(&self) -> bool {
        let mut current = self.inmem_len.load(Ordering::Acquire);
        loop {
            if current >= self.inmem_limit {
                return false;
            }
            match self.inmem_len.compare_exchange(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
    }

    #[inline]
    fn push_inmem_reserved(&self, item: T) {
        self.inmem.push(item);
        self.update_max_inmem();
    }

    #[inline]
    fn force_push_inmem(&self, item: T) {
        self.inmem.push(item);
        self.inmem_len.fetch_add(1, Ordering::Release);
        self.update_max_inmem();
    }

    fn spill_batch_now(&self, batch: Vec<T>) -> Result<()> {
        let sender = self
            .spill_tx
            .lock()
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("queue spill writer already shut down"))?;
        self.spill_inflight.fetch_add(1, Ordering::Release);
        self.stats
            .spilled_items
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        self.stats.spill_batches.fetch_add(1, Ordering::Relaxed);

        if let Err(err) = sender.send(batch) {
            self.spill_inflight.fetch_sub(1, Ordering::Release);
            return Err(anyhow!("failed sending spill batch to writer: {err}"));
        }
        Ok(())
    }

    pub fn push(&self, item: T) -> Result<()> {
        self.check_error()?;
        self.stats.pushed.fetch_add(1, Ordering::Relaxed);

        if self.try_reserve_inmem_slot() {
            self.push_inmem_reserved(item);
            // Wake up one waiting worker - efficient channel notification
            let _ = self.notify_tx.try_send(());
            return Ok(());
        }

        let mut spill = Vec::with_capacity(self.spill_batch);
        spill.push(item);
        while spill.len() < self.spill_batch {
            match self.inmem.pop() {
                Some(old) => {
                    self.inmem_len.fetch_sub(1, Ordering::Release);
                    spill.push(old);
                }
                None => break,
            }
        }

        if let Err(err) = self.spill_batch_now(spill) {
            self.set_error(err.to_string());
            return self.check_error();
        }

        // Wake up one worker after spilling
        let _ = self.notify_tx.try_send(());
        Ok(())
    }

    pub fn pop(&self) -> Result<Option<T>> {
        self.check_error()?;

        if let Some(item) = self.inmem.pop() {
            self.inmem_len.fetch_sub(1, Ordering::Release);
            self.stats.popped.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(item));
        }

        self.load_one_segment()?;

        if let Some(item) = self.inmem.pop() {
            self.inmem_len.fetch_sub(1, Ordering::Release);
            self.stats.popped.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(item));
        }
        Ok(None)
    }

    /// Set the expected number of workers for auto-finish detection
    pub fn set_worker_count(&self, count: usize) {
        self.num_workers.store(count, Ordering::Release);
    }

    /// Blocking pop - waits until an item is available or queue is finished
    /// Returns None only when queue is finished and empty
    /// Uses efficient crossbeam channel for worker notifications
    pub fn pop_blocking(&self) -> Result<Option<T>> {
        loop {
            self.check_error()?;

            // Fast path: try to pop without blocking
            if let Some(item) = self.inmem.pop() {
                self.inmem_len.fetch_sub(1, Ordering::Release);
                self.stats.popped.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(item));
            }

            // Try loading a segment
            self.load_one_segment()?;

            // Try again after loading
            if let Some(item) = self.inmem.pop() {
                self.inmem_len.fetch_sub(1, Ordering::Release);
                self.stats.popped.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(item));
            }

            // Queue is empty - check if finished
            if self.finished.load(Ordering::Acquire) {
                // Final check after seeing finished flag
                if let Some(item) = self.inmem.pop() {
                    self.inmem_len.fetch_sub(1, Ordering::Release);
                    self.stats.popped.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(item));
                }
                return Ok(None);
            }

            // Block efficiently on channel - much faster than Condvar
            self.num_waiting.fetch_add(1, Ordering::Release);
            let _ = self.notify_rx.recv();
            self.num_waiting.fetch_sub(1, Ordering::Release);

            // Loop back to try popping again after wake-up
        }
    }

    /// Pop up to `max_items` from the queue in bulk for better cache locality
    pub fn pop_bulk(&self, max_items: usize) -> Result<Vec<T>> {
        self.check_error()?;

        let mut result = Vec::with_capacity(max_items.min(256));

        // Pop from in-memory queue
        while result.len() < max_items {
            if let Some(item) = self.inmem.pop() {
                self.inmem_len.fetch_sub(1, Ordering::Release);
                result.push(item);
            } else {
                break;
            }
        }

        // If we got some items, return them
        if !result.is_empty() {
            self.stats
                .popped
                .fetch_add(result.len() as u64, Ordering::Relaxed);
            return Ok(result);
        }

        // Try loading a segment
        self.load_one_segment()?;

        // Try popping again after load
        while result.len() < max_items {
            if let Some(item) = self.inmem.pop() {
                self.inmem_len.fetch_sub(1, Ordering::Release);
                result.push(item);
            } else {
                break;
            }
        }

        if !result.is_empty() {
            self.stats
                .popped
                .fetch_add(result.len() as u64, Ordering::Relaxed);
        }

        Ok(result)
    }

    fn load_one_segment(&self) -> Result<()> {
        let Some(_guard) = self.load_lock.try_lock() else {
            return Ok(());
        };

        let next_segment = self.segments.lock().pop_front();
        let Some(path) = next_segment else {
            return Ok(());
        };

        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed reading queue segment {}", path.display()))?;
        let batch: Vec<T> = bincode::deserialize(&bytes)
            .with_context(|| format!("failed deserializing queue segment {}", path.display()))?;
        let loaded_len = batch.len() as u64;
        for item in batch {
            self.force_push_inmem(item);
        }

        if let Err(err) = std::fs::remove_file(&path) {
            self.set_error(format!(
                "failed removing queue segment {}: {err}",
                path.display()
            ));
            return self.check_error();
        }

        self.stats.loaded_segments.fetch_add(1, Ordering::Relaxed);
        self.stats
            .loaded_items
            .fetch_add(loaded_len, Ordering::Relaxed);
        Ok(())
    }

    pub fn is_drained(&self) -> bool {
        if self.inmem_len.load(Ordering::Acquire) != 0 {
            return false;
        }
        if self.spill_inflight.load(Ordering::Acquire) != 0 {
            return false;
        }
        self.segments.lock().is_empty()
    }

    pub fn has_pending_work(&self) -> bool {
        !self.is_drained()
    }

    pub fn checkpoint_flush(&self) -> Result<()> {
        self.check_error()?;

        let mut spill = Vec::with_capacity(self.spill_batch);
        while let Some(item) = self.inmem.pop() {
            self.inmem_len.fetch_sub(1, Ordering::Release);
            spill.push(item);
            if spill.len() >= self.spill_batch {
                let batch = std::mem::take(&mut spill);
                self.spill_batch_now(batch)?;
            }
        }
        if !spill.is_empty() {
            self.spill_batch_now(spill)?;
        }

        while self.spill_inflight.load(Ordering::Acquire) != 0 {
            self.check_error()?;
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.check_error()
    }

    pub fn stats(&self) -> QueueStats {
        QueueStats {
            pushed: self.stats.pushed.load(Ordering::Relaxed),
            popped: self.stats.popped.load(Ordering::Relaxed),
            spilled_items: self.stats.spilled_items.load(Ordering::Relaxed),
            spill_batches: self.stats.spill_batches.load(Ordering::Relaxed),
            loaded_segments: self.stats.loaded_segments.load(Ordering::Relaxed),
            loaded_items: self.stats.loaded_items.load(Ordering::Relaxed),
            max_inmem_len: self.stats.max_inmem_len.load(Ordering::Relaxed),
        }
    }

    /// Check if queue is empty (both in-memory and on-disk)
    pub fn is_empty(&self) -> bool {
        self.inmem_len.load(Ordering::Acquire) == 0
            && self.segments.lock().is_empty()
            && self.spill_inflight.load(Ordering::Acquire) == 0
    }

    /// Mark queue as finished - wakes up all blocked workers
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        // Send many notifications to wake all waiting workers
        // Unbounded channel means this won't block
        for _ in 0..1000 {
            let _ = self.notify_tx.try_send(());
        }
    }

    pub fn shutdown(&self) -> Result<()> {
        self.finish(); // Wake up any blocked workers before shutdown
        self.spill_tx.lock().take();
        if let Some(handle) = self.writer_handle.lock().take() {
            let _ = handle.join();
        }
        self.check_error()
    }

    pub fn spill_dir(&self) -> Option<PathBuf> {
        let first = self.segments.lock().front().cloned();
        match first {
            Some(path) => path.parent().map(Path::to_path_buf),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DiskBackedQueue, DiskQueueConfig};
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("tlapp-{prefix}-{nanos}-{}", std::process::id()))
    }

    #[test]
    fn queue_push_pop_basic() -> Result<()> {
        let dir = temp_path("queue-basic");
        let queue = DiskBackedQueue::<u64>::new(DiskQueueConfig {
            spill_dir: dir.clone(),
            inmem_limit: 128,
            spill_batch: 16,
            spill_channel_bound: 8,
            load_existing_segments: false,
        })?;

        for i in 0..100u64 {
            queue.push(i)?;
        }
        let mut out = Vec::new();
        while let Some(v) = queue.pop()? {
            out.push(v);
        }
        out.sort_unstable();
        assert_eq!(out.len(), 100);
        assert_eq!(out[0], 0);
        assert_eq!(out[99], 99);

        queue.shutdown()?;
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn queue_spills_and_loads() -> Result<()> {
        let dir = temp_path("queue-spill");
        let queue = DiskBackedQueue::<u64>::new(DiskQueueConfig {
            spill_dir: dir.clone(),
            inmem_limit: 32,
            spill_batch: 8,
            spill_channel_bound: 8,
            load_existing_segments: false,
        })?;

        for i in 0..1_000u64 {
            queue.push(i)?;
        }

        let mut out = Vec::with_capacity(1_000);
        let mut idle_spins = 0usize;
        while out.len() < 1_000 && idle_spins < 10_000 {
            match queue.pop()? {
                Some(v) => {
                    out.push(v);
                    idle_spins = 0;
                }
                None => {
                    if queue.is_drained() {
                        break;
                    }
                    idle_spins += 1;
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }

        out.sort_unstable();
        assert_eq!(out.len(), 1_000);
        assert_eq!(out[0], 0);
        assert_eq!(out[999], 999);

        let stats = queue.stats();
        assert!(stats.spill_batches > 0);
        assert!(stats.loaded_segments > 0);

        queue.shutdown()?;
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn queue_checkpoint_flush_and_resume_segments() -> Result<()> {
        let dir = temp_path("queue-resume");
        let queue = DiskBackedQueue::<u64>::new(DiskQueueConfig {
            spill_dir: dir.clone(),
            inmem_limit: 16,
            spill_batch: 8,
            spill_channel_bound: 8,
            load_existing_segments: false,
        })?;

        for i in 0..200u64 {
            queue.push(i)?;
        }
        queue.checkpoint_flush()?;
        queue.shutdown()?;

        let resumed = DiskBackedQueue::<u64>::new(DiskQueueConfig {
            spill_dir: dir.clone(),
            inmem_limit: 16,
            spill_batch: 8,
            spill_channel_bound: 8,
            load_existing_segments: true,
        })?;

        let mut out = Vec::new();
        while let Some(v) = resumed.pop()? {
            out.push(v);
        }
        out.sort_unstable();
        assert_eq!(out.len(), 200);
        assert_eq!(out[0], 0);
        assert_eq!(out[199], 199);

        resumed.shutdown()?;
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
