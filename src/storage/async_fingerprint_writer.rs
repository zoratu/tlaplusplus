// Async fingerprint persistence - workers never block on I/O
//
// Architecture:
// - Workers send fingerprints via non-blocking try_send()
// - Separate async task per shard batches and writes to disk
// - If channel full, fingerprint stays in memory (safe, just slower recovery)

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs::{OpenOptions, create_dir_all};
use tokio::io::AsyncWriteExt;

const BATCH_SIZE: usize = 10_000;
const BATCH_TIMEOUT_MS: u64 = 10;

/// Message sent from workers to async writer
#[derive(Debug, Clone, Copy)]
pub struct FingerprintPersistMsg {
    pub fp: u64,
}

/// Create channels for fingerprint persistence
pub fn create_persist_channels(
    shard_count: usize,
    channel_capacity: usize,
) -> (
    Vec<Sender<FingerprintPersistMsg>>,
    Vec<Receiver<FingerprintPersistMsg>>,
) {
    let mut senders = Vec::with_capacity(shard_count);
    let mut receivers = Vec::with_capacity(shard_count);

    for _ in 0..shard_count {
        let (tx, rx) = crossbeam_channel::bounded(channel_capacity);
        senders.push(tx);
        receivers.push(rx);
    }

    (senders, receivers)
}

/// Async task that writes fingerprints to disk
pub async fn fingerprint_writer_task(
    shard_id: usize,
    rx: Receiver<FingerprintPersistMsg>,
    work_dir: PathBuf,
) -> Result<()> {
    let shard_dir = work_dir
        .join("fingerprints")
        .join(format!("shard-{:03}", shard_id));
    create_dir_all(&shard_dir)
        .await
        .context("Failed to create shard directory")?;

    let file_path = shard_dir.join("segment.bin");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .await
        .context("Failed to open fingerprint file")?;

    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut total_written = 0u64;

    loop {
        batch.clear();

        // Try to collect a batch with timeout
        match tokio::time::timeout(Duration::from_millis(BATCH_TIMEOUT_MS), async {
            while batch.len() < BATCH_SIZE {
                match rx.try_recv() {
                    Ok(msg) => batch.push(msg.fp),
                    Err(TryRecvError::Empty) => {
                        // Wait a bit before trying again
                        tokio::time::sleep(Duration::from_micros(100)).await;
                        if batch.is_empty() {
                            continue;
                        } else {
                            break; // Have some items, write them
                        }
                    }
                    Err(TryRecvError::Disconnected) => {
                        return Err(anyhow::anyhow!("Channel disconnected"));
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                // Channel disconnected - write final batch and exit
                if !batch.is_empty() {
                    write_batch(&mut file, &batch).await?;
                    total_written += batch.len() as u64;
                }
                if std::env::var("TLAPP_VERBOSE").is_ok() {
                    eprintln!(
                        "Shard {} writer exiting: {} fingerprints written",
                        shard_id, total_written
                    );
                }
                return Err(e);
            }
            Err(_) => {
                // Timeout - if we have items, write them
                if batch.is_empty() {
                    continue;
                }
            }
        }

        // Drain any additional items (non-blocking)
        while batch.len() < BATCH_SIZE {
            match rx.try_recv() {
                Ok(msg) => batch.push(msg.fp),
                Err(_) => break,
            }
        }

        if batch.is_empty() {
            continue;
        }

        // Write batch to disk
        write_batch(&mut file, &batch).await?;
        total_written += batch.len() as u64;

        // Periodic fsync (every ~100K fingerprints)
        if total_written % 100_000 < BATCH_SIZE as u64 {
            file.sync_data().await?;
        }
    }
}

async fn write_batch(file: &mut tokio::fs::File, batch: &[u64]) -> Result<()> {
    // Convert u64 array to bytes
    let bytes: Vec<u8> = batch.iter().flat_map(|fp| fp.to_le_bytes()).collect();

    file.write_all(&bytes)
        .await
        .context("Failed to write fingerprint batch")?;

    Ok(())
}

/// Load fingerprints from disk into memory
pub async fn load_fingerprints_from_disk(
    work_dir: &PathBuf,
    shard_count: usize,
) -> Result<Vec<Vec<u64>>> {
    let mut all_fingerprints = Vec::with_capacity(shard_count);

    // Load all shards in parallel
    let mut handles = Vec::with_capacity(shard_count);

    for shard_id in 0..shard_count {
        let work_dir = work_dir.clone();
        let handle = tokio::spawn(async move {
            let file_path = work_dir
                .join("fingerprints")
                .join(format!("shard-{:03}", shard_id))
                .join("segment.bin");

            // If file doesn't exist, return empty vec
            if !file_path.exists() {
                return Ok::<Vec<u64>, anyhow::Error>(Vec::new());
            }

            let bytes = tokio::fs::read(&file_path)
                .await
                .context("Failed to read fingerprint file")?;

            // Parse u64 fingerprints
            let fps: Vec<u64> = bytes
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            Ok(fps)
        });
        handles.push(handle);
    }

    // Wait for all loads
    for handle in handles {
        let fps = handle.await.context("Load task panicked")??;
        all_fingerprints.push(fps);
    }

    Ok(all_fingerprints)
}
