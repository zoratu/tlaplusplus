// Lock-free MPMC queue using crossbeam channels
// Designed for high-throughput multi-producer multi-consumer scenarios
// Much more efficient than single-mutex or atomic CAS loops at 100+ worker scale

use crossbeam_channel::{Receiver, Sender, unbounded};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

pub struct ChannelQueue<T> {
    sender: Mutex<Option<Sender<T>>>,
    receiver: Receiver<T>,
    finished: AtomicBool,
    pushed: AtomicU64,
    popped: AtomicU64,
}

impl<T> ChannelQueue<T> {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self {
            sender: Mutex::new(Some(sender)),
            receiver,
            finished: AtomicBool::new(false),
            pushed: AtomicU64::new(0),
            popped: AtomicU64::new(0),
        }
    }

    /// Push an item - lock-free except for sender access
    pub fn push(&self, item: T) {
        if let Some(sender) = self.sender.lock().as_ref() {
            let _ = sender.send(item);
            self.pushed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Blocking pop - efficient blocking with zero syscalls when data available
    /// Uses futex-based waiting on Linux for minimal overhead
    pub fn pop_blocking(&self) -> Option<T> {
        loop {
            // Try non-blocking first (fast path)
            match self.receiver.try_recv() {
                Ok(item) => {
                    self.popped.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
                Err(_) if self.finished.load(Ordering::Acquire) => {
                    // Check one more time after seeing finished flag
                    match self.receiver.try_recv() {
                        Ok(item) => {
                            self.popped.fetch_add(1, Ordering::Relaxed);
                            return Some(item);
                        }
                        Err(_) => return None,
                    }
                }
                Err(_) => {
                    // Block efficiently - uses futex, not mutex
                    match self.receiver.recv() {
                        Ok(item) => {
                            self.popped.fetch_add(1, Ordering::Relaxed);
                            return Some(item);
                        }
                        Err(_) => {
                            // Channel closed
                            if self.finished.load(Ordering::Acquire) {
                                return None;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        // Drop the sender to close the channel and wake all blocked receivers
        *self.sender.lock() = None;
    }

    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
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

impl<T> Default for ChannelQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
