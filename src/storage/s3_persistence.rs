//! S3 Persistence for checkpoint/resume across spot instances
//!
//! This module provides continuous background upload of exploration state to S3,
//! enabling resumption after spot instance termination or upgrades.
//!
//! Design:
//! - Segments are append-only with monotonic IDs
//! - Background thread continuously uploads completed segments
//! - Tracks uploaded byte offsets (no MD5 needed)
//! - On SIGTERM: flush buffers, upload tails, write manifest (~30 seconds)
//!
//! Usage:
//! ```ignore
//! let s3 = S3Persistence::new("my-bucket", "runs/run-123", &local_dir).await?;
//! s3.download_state().await?;  // Resume from S3 if exists
//! s3.start_background_upload();
//! // ... run exploration ...
//! s3.emergency_flush().await?;  // On SIGTERM
//! ```

use anyhow::{Context, Result, anyhow};
use aws_config::BehaviorVersion;
use aws_config::Region;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// Manifest file tracking what's been uploaded to S3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Manifest {
    /// Timestamp of last update
    pub updated_at: u64,
    /// Run identifier
    pub run_id: String,
    /// Files and their uploaded byte offsets
    pub files: HashMap<String, FileState>,
    /// Checkpoint state (if any)
    pub checkpoint: Option<CheckpointState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    /// Bytes uploaded to S3
    pub uploaded_bytes: u64,
    /// S3 key for this file
    pub s3_key: String,
    /// Whether this file is complete (no more appends expected)
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Checkpoint ID
    pub id: u64,
    /// States generated at checkpoint
    pub states_generated: u64,
    /// Distinct states at checkpoint
    pub states_distinct: u64,
    /// Queue pending at checkpoint
    pub queue_pending: u64,
}

/// S3 persistence manager
pub struct S3Persistence {
    client: Client,
    bucket: String,
    prefix: String,
    local_dir: PathBuf,
    /// Track uploaded byte offsets per file (relative path -> offset)
    uploaded_offsets: Arc<DashMap<String, u64>>,
    /// Background upload task handle
    upload_handle: Option<JoinHandle<()>>,
    /// Stop signal for background task
    stop: Arc<AtomicBool>,
    /// Notify for immediate flush
    flush_notify: Arc<Notify>,
    /// Stats: bytes uploaded
    bytes_uploaded: Arc<AtomicU64>,
    /// Stats: files uploaded
    files_uploaded: Arc<AtomicU64>,
    /// Upload interval in seconds
    upload_interval_secs: u64,
}

impl S3Persistence {
    /// Create a new S3 persistence manager
    /// If region is None, uses the default region from environment/instance metadata
    pub async fn new(
        bucket: &str,
        prefix: &str,
        local_dir: &Path,
        region: Option<&str>,
    ) -> Result<Self> {
        // Load AWS config from environment/credentials file
        let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

        // Override region if specified
        if let Some(r) = region {
            config_loader = config_loader.region(Region::new(r.to_string()));
        }

        let config = config_loader.load().await;

        let client = Client::new(&config);

        // Try a lightweight check - list with max 1 key to verify access
        // head_bucket requires additional permissions that may not be granted
        match client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(prefix)
            .max_keys(1)
            .send()
            .await
        {
            Ok(_) => {}
            Err(e) => {
                // Log the full error for debugging
                eprintln!("S3: Bucket access check failed: {:?}", e);
                return Err(anyhow::anyhow!(
                    "Cannot access S3 bucket: {} - {}",
                    bucket,
                    e
                ));
            }
        }

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.trim_end_matches('/').to_string(),
            local_dir: local_dir.to_path_buf(),
            uploaded_offsets: Arc::new(DashMap::new()),
            upload_handle: None,
            stop: Arc::new(AtomicBool::new(false)),
            flush_notify: Arc::new(Notify::new()),
            bytes_uploaded: Arc::new(AtomicU64::new(0)),
            files_uploaded: Arc::new(AtomicU64::new(0)),
            upload_interval_secs: 10,
        })
    }

    /// Set upload interval (default: 10 seconds)
    pub fn with_upload_interval(mut self, secs: u64) -> Self {
        self.upload_interval_secs = secs;
        self
    }

    /// Download existing state from S3 for resume
    pub async fn download_state(&self) -> Result<DownloadResult> {
        let manifest_key = format!("{}/manifest.json", self.prefix);

        // Try to get manifest
        let manifest_result = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&manifest_key)
            .send()
            .await;

        let manifest = match manifest_result {
            Ok(response) => {
                let bytes = response.body.collect().await?.into_bytes();
                let manifest: S3Manifest =
                    serde_json::from_slice(&bytes).context("Failed to parse S3 manifest")?;
                manifest
            }
            Err(e) => {
                // Check if it's a "not found" error
                if e.to_string().contains("NoSuchKey") || e.to_string().contains("404") {
                    return Ok(DownloadResult::NoExistingState);
                }
                return Err(anyhow!("Failed to get manifest from S3: {}", e));
            }
        };

        eprintln!("S3: Found existing state from run {}", manifest.run_id);
        eprintln!("S3: Downloading {} files...", manifest.files.len());

        // Create local directory
        tokio::fs::create_dir_all(&self.local_dir).await?;

        // Download all files
        let mut downloaded_bytes = 0u64;
        let mut downloaded_files = 0u64;

        for (rel_path, file_state) in &manifest.files {
            let local_path = self.local_dir.join(rel_path);

            // Create parent directories
            if let Some(parent) = local_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            // Download file
            let response = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(&file_state.s3_key)
                .send()
                .await
                .context(format!("Failed to download {}", file_state.s3_key))?;

            let bytes = response.body.collect().await?.into_bytes();
            tokio::fs::write(&local_path, &bytes).await?;

            downloaded_bytes += bytes.len() as u64;
            downloaded_files += 1;

            // Track as already uploaded
            self.uploaded_offsets
                .insert(rel_path.clone(), file_state.uploaded_bytes);
        }

        eprintln!(
            "S3: Downloaded {} files ({:.2} MB)",
            downloaded_files,
            downloaded_bytes as f64 / 1_048_576.0
        );

        Ok(DownloadResult::Resumed {
            manifest,
            downloaded_bytes,
            downloaded_files,
        })
    }

    /// Start background upload thread
    pub fn start_background_upload(&mut self, runtime: &tokio::runtime::Handle) {
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let prefix = self.prefix.clone();
        let local_dir = self.local_dir.clone();
        let uploaded_offsets = Arc::clone(&self.uploaded_offsets);
        let stop = Arc::clone(&self.stop);
        let flush_notify = Arc::clone(&self.flush_notify);
        let bytes_uploaded = Arc::clone(&self.bytes_uploaded);
        let files_uploaded = Arc::clone(&self.files_uploaded);
        let interval_secs = self.upload_interval_secs;

        let handle = runtime.spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

            loop {
                tokio::select! {
                    _ = interval.tick() => {}
                    _ = flush_notify.notified() => {
                        // Immediate flush requested
                    }
                }

                if stop.load(Ordering::Acquire) {
                    break;
                }

                // Scan and upload
                if let Err(e) = upload_changed_files(
                    &client,
                    &bucket,
                    &prefix,
                    &local_dir,
                    &uploaded_offsets,
                    &bytes_uploaded,
                    &files_uploaded,
                )
                .await
                {
                    eprintln!("S3: Background upload error: {}", e);
                }
            }
        });

        self.upload_handle = Some(handle);
        eprintln!(
            "S3: Started background upload (interval: {}s)",
            interval_secs
        );
    }

    /// Emergency flush - upload all pending data immediately
    /// Called on SIGTERM or explicit checkpoint
    pub async fn emergency_flush(&self) -> Result<FlushResult> {
        let start = Instant::now();
        eprintln!("S3: Emergency flush starting...");

        // Trigger immediate upload
        self.flush_notify.notify_one();

        // Give background thread a moment to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Do our own full scan to ensure everything is uploaded
        let result = upload_changed_files(
            &self.client,
            &self.bucket,
            &self.prefix,
            &self.local_dir,
            &self.uploaded_offsets,
            &self.bytes_uploaded,
            &self.files_uploaded,
        )
        .await?;

        // Upload manifest
        self.upload_manifest(None).await?;

        let duration = start.elapsed();
        eprintln!(
            "S3: Emergency flush complete in {:.2}s ({} files, {:.2} MB)",
            duration.as_secs_f64(),
            result.files_uploaded,
            result.bytes_uploaded as f64 / 1_048_576.0
        );

        Ok(FlushResult {
            duration,
            files_uploaded: result.files_uploaded,
            bytes_uploaded: result.bytes_uploaded,
        })
    }

    /// Upload manifest with current state
    pub async fn upload_manifest(&self, checkpoint: Option<CheckpointState>) -> Result<()> {
        let mut files = HashMap::new();

        for entry in self.uploaded_offsets.iter() {
            let rel_path = entry.key().clone();
            let uploaded_bytes = *entry.value();
            let s3_key = format!("{}/{}", self.prefix, rel_path);

            files.insert(
                rel_path,
                FileState {
                    uploaded_bytes,
                    s3_key,
                    complete: false, // We don't track completion status currently
                },
            );
        }

        let manifest = S3Manifest {
            updated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            run_id: self
                .prefix
                .split('/')
                .last()
                .unwrap_or("unknown")
                .to_string(),
            files,
            checkpoint,
        };

        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        let manifest_key = format!("{}/manifest.json", self.prefix);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&manifest_key)
            .body(ByteStream::from(manifest_json.into_bytes()))
            .content_type("application/json")
            .send()
            .await
            .context("Failed to upload manifest")?;

        Ok(())
    }

    /// Stop background upload and wait for completion
    pub async fn stop(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        self.flush_notify.notify_one();

        if let Some(handle) = self.upload_handle.take() {
            handle.await?;
        }

        Ok(())
    }

    /// Get upload statistics
    pub fn stats(&self) -> S3Stats {
        S3Stats {
            bytes_uploaded: self.bytes_uploaded.load(Ordering::Relaxed),
            files_uploaded: self.files_uploaded.load(Ordering::Relaxed),
            files_tracked: self.uploaded_offsets.len() as u64,
        }
    }

    /// Check if S3 persistence is active
    pub fn is_running(&self) -> bool {
        self.upload_handle.is_some() && !self.stop.load(Ordering::Acquire)
    }
}

/// Result of downloading state from S3
#[derive(Debug)]
pub enum DownloadResult {
    /// No existing state found in S3
    NoExistingState,
    /// Successfully resumed from S3
    Resumed {
        manifest: S3Manifest,
        downloaded_bytes: u64,
        downloaded_files: u64,
    },
}

/// Result of emergency flush
#[derive(Debug)]
pub struct FlushResult {
    pub duration: Duration,
    pub files_uploaded: u64,
    pub bytes_uploaded: u64,
}

/// S3 upload statistics
#[derive(Debug, Clone)]
pub struct S3Stats {
    pub bytes_uploaded: u64,
    pub files_uploaded: u64,
    pub files_tracked: u64,
}

/// Internal result of upload scan
struct UploadScanResult {
    files_uploaded: u64,
    bytes_uploaded: u64,
}

/// Scan local directory and upload changed files
async fn upload_changed_files(
    client: &Client,
    bucket: &str,
    prefix: &str,
    local_dir: &Path,
    uploaded_offsets: &DashMap<String, u64>,
    total_bytes: &AtomicU64,
    total_files: &AtomicU64,
) -> Result<UploadScanResult> {
    let mut files_uploaded = 0u64;
    let mut bytes_uploaded = 0u64;

    // Recursively scan local directory
    let entries = collect_files(local_dir).await?;

    for entry_path in entries {
        let rel_path = entry_path
            .strip_prefix(local_dir)
            .unwrap_or(&entry_path)
            .to_string_lossy()
            .to_string();

        // Skip manifest (we manage it separately)
        if rel_path == "manifest.json" {
            continue;
        }

        // Get current file size
        let metadata = match tokio::fs::metadata(&entry_path).await {
            Ok(m) => m,
            Err(_) => continue, // File may have been deleted
        };
        let current_size = metadata.len();

        // Get previously uploaded offset
        let uploaded_offset = uploaded_offsets.get(&rel_path).map(|v| *v).unwrap_or(0);

        // Skip if nothing new
        if current_size <= uploaded_offset {
            continue;
        }

        // Calculate bytes to upload
        let bytes_to_upload = current_size - uploaded_offset;

        // Read the new bytes
        let mut file = File::open(&entry_path).await?;

        let s3_key = format!("{}/{}", prefix, rel_path);

        if uploaded_offset == 0 {
            // Fresh upload - upload entire file
            let mut contents = Vec::with_capacity(current_size as usize);
            file.read_to_end(&mut contents).await?;

            client
                .put_object()
                .bucket(bucket)
                .key(&s3_key)
                .body(ByteStream::from(contents))
                .send()
                .await
                .context(format!("Failed to upload {}", s3_key))?;
        } else {
            // Incremental upload - we need to re-upload the whole file
            // S3 doesn't support appending, so we replace the object
            // This is still efficient because we only upload when there are changes
            let mut contents = Vec::with_capacity(current_size as usize);
            file.read_to_end(&mut contents).await?;

            client
                .put_object()
                .bucket(bucket)
                .key(&s3_key)
                .body(ByteStream::from(contents))
                .send()
                .await
                .context(format!("Failed to upload {}", s3_key))?;
        }

        // Update tracking
        uploaded_offsets.insert(rel_path, current_size);
        bytes_uploaded += bytes_to_upload;
        files_uploaded += 1;
        total_bytes.fetch_add(bytes_to_upload, Ordering::Relaxed);
        total_files.fetch_add(1, Ordering::Relaxed);
    }

    Ok(UploadScanResult {
        files_uploaded,
        bytes_uploaded,
    })
}

/// Recursively collect all files in a directory
async fn collect_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];

    while let Some(current) = stack.pop() {
        let mut entries = match tokio::fs::read_dir(&current).await {
            Ok(e) => e,
            Err(_) => continue,
        };

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let metadata = entry.metadata().await?;

            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                files.push(path);
            }
        }
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_serialization() {
        let manifest = S3Manifest {
            updated_at: 1234567890,
            run_id: "test-run".to_string(),
            files: HashMap::new(),
            checkpoint: Some(CheckpointState {
                id: 1,
                states_generated: 1000,
                states_distinct: 900,
                queue_pending: 500,
            }),
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: S3Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.run_id, "test-run");
        assert!(parsed.checkpoint.is_some());
    }
}
