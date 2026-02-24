// Simple blocking queue using Mutex + Condvar - TLC-style
// This is SIMPLER and FASTER at high core counts than lock-free approaches
// because it avoids atomic contention and CAS retry loops

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub struct SimpleBlockingQueue<T> {
    inner: Mutex<VecDeque<T>>,
    not_empty: Condvar,
    finished: AtomicBool,
    pushed: AtomicU64,
    popped: AtomicU64,
}

impl<T> SimpleBlockingQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(10_000)),
            not_empty: Condvar::new(),
            finished: AtomicBool::new(false),
            pushed: AtomicU64::new(0),
            popped: AtomicU64::new(0),
        }
    }

    /// Push an item - wakes one waiting worker
    pub fn push(&self, item: T) {
        let mut queue = self.inner.lock().unwrap();
        queue.push_back(item);
        self.pushed.fetch_add(1, Ordering::Relaxed);
        // Wake one waiting worker
        self.not_empty.notify_one();
    }

    /// Blocking pop - efficient sleep when empty
    pub fn pop_blocking(&self) -> Option<T> {
        let mut queue = self.inner.lock().unwrap();

        loop {
            // Try to pop
            if let Some(item) = queue.pop_front() {
                self.popped.fetch_add(1, Ordering::Relaxed);
                return Some(item);
            }

            // Check if finished
            if self.finished.load(Ordering::Acquire) {
                return None;
            }

            // Wait for notification - releases lock and sleeps
            queue = self.not_empty.wait(queue).unwrap();
        }
    }

    /// Mark as finished and wake all workers
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        self.not_empty.notify_all();
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    pub fn stats(&self) -> (u64, u64) {
        (
            self.pushed.load(Ordering::Relaxed),
            self.popped.load(Ordering::Relaxed),
        )
    }

    pub fn has_pending_work(&self) -> bool {
        !self.is_empty()
    }
}

impl<T> Default for SimpleBlockingQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
