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
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::{Notify, Semaphore};
use tokio::task::{JoinHandle, JoinSet};

const DEFAULT_S3_DOWNLOAD_CONCURRENCY: usize = 16;
const DEFAULT_S3_UPLOAD_CONCURRENCY: usize = 8;
const DEFAULT_S3_MULTIPART_THRESHOLD_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_S3_MULTIPART_PART_SIZE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_S3_MULTIPART_UPLOAD_CONCURRENCY: usize = 4;
const S3_MIN_MULTIPART_PART_SIZE_BYTES: usize = 5 * 1024 * 1024;
const S3_MAX_MULTIPART_PARTS: usize = 10_000;

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
    /// Minimum segment ID needed by this checkpoint.
    /// All segments with ID >= min_segment_id must be retained for resume to work.
    /// Segments with ID < min_segment_id can be safely pruned.
    #[serde(default)]
    pub min_segment_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct S3TransferTuning {
    download_concurrency: usize,
    upload_concurrency: usize,
    multipart_threshold_bytes: u64,
    multipart_part_size_bytes: usize,
    multipart_upload_concurrency: usize,
}

impl Default for S3TransferTuning {
    fn default() -> Self {
        Self {
            download_concurrency: DEFAULT_S3_DOWNLOAD_CONCURRENCY,
            upload_concurrency: DEFAULT_S3_UPLOAD_CONCURRENCY,
            multipart_threshold_bytes: DEFAULT_S3_MULTIPART_THRESHOLD_BYTES,
            multipart_part_size_bytes: DEFAULT_S3_MULTIPART_PART_SIZE_BYTES,
            multipart_upload_concurrency: DEFAULT_S3_MULTIPART_UPLOAD_CONCURRENCY,
        }
    }
}

impl S3TransferTuning {
    fn from_env() -> Self {
        let default = Self::default();
        Self {
            download_concurrency: parse_env_usize(
                "TLAPP_S3_DOWNLOAD_CONCURRENCY",
                default.download_concurrency,
                1,
                256,
            ),
            upload_concurrency: parse_env_usize(
                "TLAPP_S3_UPLOAD_CONCURRENCY",
                default.upload_concurrency,
                1,
                256,
            ),
            multipart_threshold_bytes: parse_env_u64(
                "TLAPP_S3_MULTIPART_THRESHOLD_MB",
                default.multipart_threshold_bytes / 1_048_576,
                8,
                16 * 1024,
            ) * 1_048_576,
            multipart_part_size_bytes: parse_env_usize(
                "TLAPP_S3_MULTIPART_PART_SIZE_MB",
                default.multipart_part_size_bytes / 1_048_576,
                S3_MIN_MULTIPART_PART_SIZE_BYTES / 1_048_576,
                512,
            ) * 1_048_576,
            multipart_upload_concurrency: parse_env_usize(
                "TLAPP_S3_MULTIPART_UPLOAD_CONCURRENCY",
                default.multipart_upload_concurrency,
                1,
                64,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadMode {
    SinglePut,
    Multipart {
        part_size_bytes: usize,
        part_count: usize,
    },
}

#[derive(Debug, Clone)]
struct PendingUpload {
    entry_path: PathBuf,
    rel_path: String,
    s3_key: String,
    current_size: u64,
    bytes_to_upload: u64,
    is_queue_spill: bool,
}

#[derive(Debug)]
struct CompletedUpload {
    rel_path: String,
    current_size: u64,
    bytes_to_upload: u64,
    is_queue_spill: bool,
}

fn parse_env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn parse_env_u64(name: &str, default: u64, min: u64, max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn compute_multipart_part_size(file_size: u64, preferred_part_size: usize) -> usize {
    let preferred = preferred_part_size.max(S3_MIN_MULTIPART_PART_SIZE_BYTES);
    let min_needed = ((file_size.saturating_add(S3_MAX_MULTIPART_PARTS as u64 - 1))
        / S3_MAX_MULTIPART_PARTS as u64) as usize;
    preferred.max(min_needed)
}

fn plan_multipart_ranges(file_size: u64, preferred_part_size: usize) -> Vec<(i32, u64, usize)> {
    if file_size == 0 {
        return Vec::new();
    }

    let part_size = compute_multipart_part_size(file_size, preferred_part_size);
    let mut ranges = Vec::new();
    let mut offset = 0u64;
    let mut part_number = 1i32;

    while offset < file_size {
        let remaining = file_size - offset;
        let len = remaining.min(part_size as u64) as usize;
        ranges.push((part_number, offset, len));
        offset += len as u64;
        part_number += 1;
    }

    ranges
}

fn choose_upload_mode(file_size: u64, tuning: S3TransferTuning) -> UploadMode {
    if file_size < tuning.multipart_threshold_bytes {
        return UploadMode::SinglePut;
    }

    let ranges = plan_multipart_ranges(file_size, tuning.multipart_part_size_bytes);
    UploadMode::Multipart {
        part_size_bytes: compute_multipart_part_size(file_size, tuning.multipart_part_size_bytes),
        part_count: ranges.len(),
    }
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
        let tuning = S3TransferTuning::from_env();
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
                let err_str = e.to_string();
                // Check if it's a "not found" error - the manifest may not exist yet
                if err_str.contains("NoSuchKey")
                    || err_str.contains("404")
                    || err_str.contains("Not Found")
                    || err_str.contains("AccessDenied")
                {
                    if err_str.contains("AccessDenied") {
                        eprintln!(
                            "S3: Warning: AccessDenied reading manifest at s3://{}/{} \
                             (check IAM permissions for s3:GetObject)",
                            self.bucket, manifest_key
                        );
                    }
                    return Ok(DownloadResult::NoExistingState);
                }
                // For service errors (wrong region, network, etc.), warn and start fresh
                // rather than failing hard - the checkpoint will be created on first write
                eprintln!(
                    "S3: Warning: could not read manifest from s3://{}/{}: {}",
                    self.bucket, manifest_key, err_str
                );
                return Ok(DownloadResult::NoExistingState);
            }
        };

        eprintln!("S3: Found existing state from run {}", manifest.run_id);
        let total_files = manifest.files.len();
        eprintln!(
            "S3: [DOWNLOAD] Starting download of {} files with concurrency {}...",
            total_files, tuning.download_concurrency
        );

        // Create local directory
        tokio::fs::create_dir_all(&self.local_dir).await?;

        // Download files concurrently to better use available network bandwidth.
        let mut downloaded_bytes = 0u64;
        let mut downloaded_files = 0u64;
        let start_time = std::time::Instant::now();
        let mut last_progress_time = start_time;

        let semaphore = Arc::new(Semaphore::new(
            tuning.download_concurrency.max(1).min(total_files.max(1)),
        ));
        let mut join_set = JoinSet::new();
        let download_jobs: Vec<(String, FileState)> = manifest
            .files
            .iter()
            .map(|(rel_path, file_state)| (rel_path.clone(), file_state.clone()))
            .collect();

        for (rel_path, file_state) in download_jobs {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .context("S3 download semaphore closed unexpectedly")?;
            let client = self.client.clone();
            let bucket = self.bucket.clone();
            let local_dir = self.local_dir.clone();
            let uploaded_offsets = Arc::clone(&self.uploaded_offsets);

            join_set.spawn(async move {
                let _permit = permit;
                let local_path = local_dir.join(&rel_path);
                if let Some(parent) = local_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }

                let response = client
                    .get_object()
                    .bucket(&bucket)
                    .key(&file_state.s3_key)
                    .send()
                    .await
                    .context(format!("Failed to download {}", file_state.s3_key))?;

                let bytes = response.body.collect().await?.into_bytes();
                tokio::fs::write(&local_path, &bytes).await?;
                uploaded_offsets.insert(rel_path, file_state.uploaded_bytes);
                Ok::<u64, anyhow::Error>(bytes.len() as u64)
            });
        }

        while let Some(result) = join_set.join_next().await {
            let bytes = result.context("S3 download task panicked")??;
            downloaded_bytes += bytes;
            downloaded_files += 1;

            let now = std::time::Instant::now();
            if now.duration_since(last_progress_time).as_secs() >= 2 || downloaded_files % 100 == 0
            {
                let pct = (downloaded_files as f64 / total_files.max(1) as f64) * 100.0;
                let elapsed = now.duration_since(start_time).as_secs_f64();
                let rate_mbps = if elapsed > 0.0 {
                    (downloaded_bytes as f64 / 1_048_576.0) / elapsed
                } else {
                    0.0
                };
                eprintln!(
                    "S3: [DOWNLOAD] {}/{} files ({:.1}%) - {:.1} MB @ {:.1} MB/s",
                    downloaded_files,
                    total_files,
                    pct,
                    downloaded_bytes as f64 / 1_048_576.0,
                    rate_mbps
                );
                last_progress_time = now;
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        eprintln!(
            "S3: [DOWNLOAD] Complete: {} files ({:.1} MB) in {:.1}s",
            downloaded_files,
            downloaded_bytes as f64 / 1_048_576.0,
            elapsed
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
            let mut iteration_count = 0u64;
            // Prune every 6 iterations (~60s with 10s interval)
            const PRUNE_INTERVAL: u64 = 6;
            // Keep last 5 checkpoints
            const KEEP_CHECKPOINTS: usize = 5;

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

                // Periodically prune S3 files that no longer exist locally
                iteration_count += 1;

                // Log cumulative upload stats periodically
                if iteration_count % PRUNE_INTERVAL == 0 {
                    let total_bytes = bytes_uploaded.load(Ordering::Relaxed);
                    let total_files = files_uploaded.load(Ordering::Relaxed);
                    if total_files > 0 {
                        eprintln!(
                            "S3: [UPLOAD] Cumulative: {} files ({:.1} MB) synced to s3://{}/{}",
                            total_files,
                            total_bytes as f64 / 1_048_576.0,
                            bucket,
                            prefix
                        );
                    }
                }

                if iteration_count % PRUNE_INTERVAL == 0 {
                    if let Err(e) = prune_deleted_files(
                        &client,
                        &bucket,
                        &prefix,
                        &local_dir,
                        &uploaded_offsets,
                        KEEP_CHECKPOINTS,
                    )
                    .await
                    {
                        eprintln!("S3: Background prune error: {}", e);
                    }
                }
            }
        });

        self.upload_handle = Some(handle);
        eprintln!(
            "S3: [UPLOAD] Background sync started (interval: {}s)",
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
            "S3: [UPLOAD] Emergency flush complete in {:.2}s ({} files, {:.1} MB)",
            duration.as_secs_f64(),
            result.files_uploaded,
            result.bytes_uploaded as f64 / 1_048_576.0
        );
        if result.queue_files_uploaded > 0 {
            eprintln!(
                "S3: [UPLOAD]   Including {} queue-spill files ({:.1} MB)",
                result.queue_files_uploaded,
                result.queue_bytes_uploaded as f64 / 1_048_576.0
            );
        }

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

    /// Prune old checkpoints from S3, keeping only the most recent `keep_count`.
    /// Also removes queue-spill segments that are no longer needed.
    ///
    /// CRITICAL: Segments are only pruned if their ID is below the min_segment_id
    /// stored in the manifest. This ensures that segments needed for resume are
    /// never deleted, even if they've been consumed locally.
    pub async fn prune_old_checkpoints(&self, keep_count: usize) -> Result<PruneResult> {
        let mut checkpoints_deleted = 0u64;
        let mut segments_deleted = 0u64;

        // First, fetch the manifest to get the min_segment_id
        let manifest_key = format!("{}/manifest.json", self.prefix);
        let min_segment_id: Option<u64> = match self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&manifest_key)
            .send()
            .await
        {
            Ok(response) => match response.body.collect().await {
                Ok(data) => {
                    let bytes = data.into_bytes();
                    match serde_json::from_slice::<S3Manifest>(&bytes) {
                        Ok(manifest) => manifest
                            .checkpoint
                            .as_ref()
                            .and_then(|cp| cp.min_segment_id),
                        Err(_) => None,
                    }
                }
                Err(_) => None,
            },
            Err(_) => None,
        };

        // List all checkpoint files on S3
        let checkpoint_prefix = format!("{}/checkpoints/checkpoint-", self.prefix);
        let mut checkpoint_keys: Vec<String> = Vec::new();

        let mut continuation_token: Option<String> = None;
        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&checkpoint_prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;

            for obj in response.contents() {
                if let Some(key) = obj.key() {
                    // Only match checkpoint-*.json, not latest.json
                    if key.ends_with(".json") && key.contains("checkpoint-") {
                        checkpoint_keys.push(key.to_string());
                    }
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        // Sort by key (contains timestamp, so lexicographic = chronological)
        checkpoint_keys.sort();

        // Delete oldest checkpoints if we have more than keep_count
        if checkpoint_keys.len() > keep_count {
            let to_delete = checkpoint_keys.len() - keep_count;
            for key in checkpoint_keys.into_iter().take(to_delete) {
                match self
                    .client
                    .delete_object()
                    .bucket(&self.bucket)
                    .key(&key)
                    .send()
                    .await
                {
                    Ok(_) => {
                        checkpoints_deleted += 1;
                        eprintln!("S3: Pruned old checkpoint {}", key);
                    }
                    Err(e) => {
                        eprintln!("S3: Warning: failed to delete {}: {}", key, e);
                    }
                }
            }
        }

        // Prune queue-spill segments ONLY if we have a min_segment_id
        let Some(min_id) = min_segment_id else {
            // No min_segment_id - skip segment pruning to be safe
            if checkpoints_deleted > 0 {
                eprintln!(
                    "S3: Pruned {} checkpoints (skipped segment pruning - no min_segment_id)",
                    checkpoints_deleted
                );
            }
            return Ok(PruneResult {
                checkpoints_deleted,
                segments_deleted: 0,
            });
        };

        let queue_prefix = format!("{}/queue-spill/", self.prefix);
        let mut s3_segments: Vec<String> = Vec::new();
        continuation_token = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&queue_prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;

            for obj in response.contents() {
                if let Some(key) = obj.key() {
                    s3_segments.push(key.to_string());
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        // Only prune segments with ID < min_segment_id
        for key in s3_segments {
            let filename = key.split('/').last().unwrap_or(&key);

            // Extract segment ID from filename
            let segment_id = match extract_segment_id(filename) {
                Some(id) => id,
                None => continue, // Skip non-segment files
            };

            // CRITICAL: Only prune if segment ID is BELOW the minimum needed
            if segment_id < min_id {
                match self
                    .client
                    .delete_object()
                    .bucket(&self.bucket)
                    .key(&key)
                    .send()
                    .await
                {
                    Ok(_) => {
                        segments_deleted += 1;
                        // Remove from tracked offsets
                        let rel_path = format!("queue-spill/{}", filename);
                        self.uploaded_offsets.remove(&rel_path);
                    }
                    Err(e) => {
                        eprintln!("S3: Warning: failed to delete segment {}: {}", key, e);
                    }
                }
            }
        }

        if checkpoints_deleted > 0 || segments_deleted > 0 {
            eprintln!(
                "S3: Pruned {} checkpoints, {} segments (min_segment_id={})",
                checkpoints_deleted, segments_deleted, min_id
            );
        }

        Ok(PruneResult {
            checkpoints_deleted,
            segments_deleted,
        })
    }

    /// Get the S3 client for external use (e.g., runtime pruning)
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get the bucket name
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get the prefix
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Result of S3 pruning operation
#[derive(Debug)]
pub struct PruneResult {
    pub checkpoints_deleted: u64,
    pub segments_deleted: u64,
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
    queue_files_uploaded: u64,
    queue_bytes_uploaded: u64,
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
    let tuning = S3TransferTuning::from_env();
    let mut files_uploaded = 0u64;
    let mut bytes_uploaded = 0u64;
    let mut queue_files_uploaded = 0u64;
    let mut queue_bytes_uploaded = 0u64;

    let pending_uploads = collect_pending_uploads(local_dir, prefix, uploaded_offsets).await?;
    let semaphore = Arc::new(Semaphore::new(
        tuning
            .upload_concurrency
            .max(1)
            .min(pending_uploads.len().max(1)),
    ));
    let mut join_set = JoinSet::new();

    for upload in pending_uploads {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("S3 upload semaphore closed unexpectedly")?;
        let client = client.clone();
        let bucket = bucket.to_string();

        join_set.spawn(async move {
            let _permit = permit;
            upload_file(client, bucket, upload, tuning).await
        });
    }

    while let Some(result) = join_set.join_next().await {
        let completed = result.context("S3 upload task panicked")??;
        if completed.is_queue_spill {
            queue_files_uploaded += 1;
            queue_bytes_uploaded += completed.bytes_to_upload;
        }

        uploaded_offsets.insert(completed.rel_path, completed.current_size);
        bytes_uploaded += completed.bytes_to_upload;
        files_uploaded += 1;
        total_bytes.fetch_add(completed.bytes_to_upload, Ordering::Relaxed);
        total_files.fetch_add(1, Ordering::Relaxed);
    }

    Ok(UploadScanResult {
        files_uploaded,
        bytes_uploaded,
        queue_files_uploaded,
        queue_bytes_uploaded,
    })
}

async fn collect_pending_uploads(
    local_dir: &Path,
    prefix: &str,
    uploaded_offsets: &DashMap<String, u64>,
) -> Result<Vec<PendingUpload>> {
    let mut pending = Vec::new();
    let entries = collect_files(local_dir).await?;

    for entry_path in entries {
        let rel_path = entry_path
            .strip_prefix(local_dir)
            .unwrap_or(&entry_path)
            .to_string_lossy()
            .to_string();

        if rel_path == "manifest.json" {
            continue;
        }

        let metadata = match tokio::fs::metadata(&entry_path).await {
            Ok(m) => m,
            Err(_) => continue,
        };
        let current_size = metadata.len();
        let uploaded_offset = uploaded_offsets.get(&rel_path).map(|v| *v).unwrap_or(0);
        if current_size <= uploaded_offset {
            continue;
        }

        pending.push(PendingUpload {
            entry_path,
            rel_path: rel_path.clone(),
            s3_key: format!("{}/{}", prefix, rel_path),
            current_size,
            bytes_to_upload: current_size - uploaded_offset,
            is_queue_spill: rel_path.contains("queue-spill") || rel_path.contains("queue/"),
        });
    }

    Ok(pending)
}

async fn upload_file(
    client: Client,
    bucket: String,
    upload: PendingUpload,
    tuning: S3TransferTuning,
) -> Result<CompletedUpload> {
    match choose_upload_mode(upload.current_size, tuning) {
        UploadMode::SinglePut => {
            let contents = tokio::fs::read(&upload.entry_path).await?;
            client
                .put_object()
                .bucket(&bucket)
                .key(&upload.s3_key)
                .body(ByteStream::from(contents))
                .send()
                .await
                .context(format!("Failed to upload {}", upload.s3_key))?;
        }
        UploadMode::Multipart {
            part_size_bytes, ..
        } => {
            multipart_upload_file(
                &client,
                &bucket,
                &upload.s3_key,
                &upload.entry_path,
                upload.current_size,
                part_size_bytes,
                tuning.multipart_upload_concurrency,
            )
            .await
            .context(format!("Failed multipart upload for {}", upload.s3_key))?;
        }
    }

    Ok(CompletedUpload {
        rel_path: upload.rel_path,
        current_size: upload.current_size,
        bytes_to_upload: upload.bytes_to_upload,
        is_queue_spill: upload.is_queue_spill,
    })
}

async fn multipart_upload_file(
    client: &Client,
    bucket: &str,
    s3_key: &str,
    entry_path: &Path,
    file_size: u64,
    part_size_bytes: usize,
    multipart_upload_concurrency: usize,
) -> Result<()> {
    let create = client
        .create_multipart_upload()
        .bucket(bucket)
        .key(s3_key)
        .send()
        .await
        .context(format!("Failed to start multipart upload for {}", s3_key))?;
    let upload_id = create
        .upload_id()
        .context(format!(
            "Multipart upload for {} did not return upload_id",
            s3_key
        ))?
        .to_string();

    let ranges = plan_multipart_ranges(file_size, part_size_bytes);
    let semaphore = Arc::new(Semaphore::new(
        multipart_upload_concurrency.max(1).min(ranges.len().max(1)),
    ));
    let mut join_set = JoinSet::new();
    let mut file = File::open(entry_path).await?;

    for (part_number, _offset, len) in ranges {
        let mut chunk = vec![0u8; len];
        file.read_exact(&mut chunk).await?;
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("S3 multipart semaphore closed unexpectedly")?;
        let client = client.clone();
        let bucket = bucket.to_string();
        let s3_key = s3_key.to_string();
        let upload_id = upload_id.clone();

        join_set.spawn(async move {
            let _permit = permit;
            let response = client
                .upload_part()
                .bucket(bucket)
                .key(s3_key)
                .upload_id(upload_id)
                .part_number(part_number)
                .content_length(len as i64)
                .body(ByteStream::from(chunk))
                .send()
                .await?;
            Ok::<CompletedPart, anyhow::Error>(
                CompletedPart::builder()
                    .part_number(part_number)
                    .set_e_tag(response.e_tag().map(|etag| etag.to_string()))
                    .build(),
            )
        });
    }

    let mut completed_parts = Vec::new();
    let mut upload_error = None;
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(part)) => completed_parts.push(part),
            Ok(Err(err)) => {
                upload_error = Some(err);
                break;
            }
            Err(join_err) => {
                upload_error = Some(anyhow!("S3 multipart task panicked: {}", join_err));
                break;
            }
        }
    }

    if let Some(err) = upload_error {
        join_set.abort_all();
        while join_set.join_next().await.is_some() {}
        // Best-effort multipart cleanup: we're already returning the original
        // error. If the abort itself fails (network flake, expired creds),
        // the orphaned upload is left for the bucket lifecycle policy to GC.
        // Log so operators can see leaked-multipart-cost surprises.
        if let Err(abort_err) = client
            .abort_multipart_upload()
            .bucket(bucket)
            .key(s3_key)
            .upload_id(&upload_id)
            .send()
            .await
        {
            eprintln!(
                "warning: failed to abort orphaned S3 multipart upload {} (bucket {}): {}",
                upload_id, bucket, abort_err
            );
        }
        return Err(err);
    }

    completed_parts.sort_by_key(|part| part.part_number.unwrap_or_default());
    client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(s3_key)
        .upload_id(&upload_id)
        .multipart_upload(
            CompletedMultipartUpload::builder()
                .set_parts(Some(completed_parts))
                .build(),
        )
        .send()
        .await
        .context(format!(
            "Failed to complete multipart upload for {}",
            s3_key
        ))?;

    Ok(())
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

/// Extract segment ID from a segment filename like "segment-0000000000000042.bin"
fn extract_segment_id(filename: &str) -> Option<u64> {
    if !filename.starts_with("segment-") || !filename.ends_with(".bin") {
        return None;
    }
    let id_str = filename
        .trim_start_matches("segment-")
        .trim_end_matches(".bin");
    id_str.parse::<u64>().ok()
}

/// Prune S3 files that are no longer needed.
///
/// CRITICAL FIX: This function now respects segment dependencies.
/// Previously, it deleted segments that didn't exist locally, but this caused
/// resume to fail because:
/// 1. Segments are deleted locally after being consumed (loaded into memory)
/// 2. S3 pruning would then delete them from S3 too
/// 3. On resume (after rm -rf ~/.tlapp), segments were missing
///
/// The fix: Only prune segments with ID < min_segment_id from the manifest.
/// The min_segment_id represents the minimum segment ID needed for resume.
/// Any segment with ID >= min_segment_id must be retained.
async fn prune_deleted_files(
    client: &Client,
    bucket: &str,
    prefix: &str,
    local_dir: &Path,
    uploaded_offsets: &DashMap<String, u64>,
    keep_checkpoints: usize,
) -> Result<()> {
    let mut checkpoints_deleted = 0u64;
    let mut segments_deleted = 0u64;
    let mut local_segments_deleted = 0u64;

    // First, fetch the manifest to get the min_segment_id
    let manifest_key = format!("{}/manifest.json", prefix);
    let min_segment_id: Option<u64> = match client
        .get_object()
        .bucket(bucket)
        .key(&manifest_key)
        .send()
        .await
    {
        Ok(response) => match response.body.collect().await {
            Ok(data) => {
                let bytes = data.into_bytes();
                match serde_json::from_slice::<S3Manifest>(&bytes) {
                    Ok(manifest) => manifest
                        .checkpoint
                        .as_ref()
                        .and_then(|cp| cp.min_segment_id),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        },
        Err(_) => None,
    };

    // 1. Prune old checkpoint files (keep last N)
    let checkpoint_prefix = format!("{}/checkpoints/checkpoint-", prefix);
    let mut checkpoint_keys: Vec<String> = Vec::new();

    let mut continuation_token: Option<String> = None;
    loop {
        let mut request = client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(&checkpoint_prefix);

        if let Some(token) = continuation_token {
            request = request.continuation_token(token);
        }

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("S3: Warning: failed to list checkpoints for pruning: {}", e);
                return Ok(());
            }
        };

        for obj in response.contents() {
            if let Some(key) = obj.key() {
                if key.ends_with(".json") && key.contains("checkpoint-") {
                    checkpoint_keys.push(key.to_string());
                }
            }
        }

        if response.is_truncated() == Some(true) {
            continuation_token = response.next_continuation_token().map(|s| s.to_string());
        } else {
            break;
        }
    }

    // Sort by key (timestamp-based, oldest first)
    checkpoint_keys.sort();

    // Delete oldest checkpoints if we have more than keep_checkpoints
    if checkpoint_keys.len() > keep_checkpoints {
        let to_delete = checkpoint_keys.len() - keep_checkpoints;
        for key in checkpoint_keys.into_iter().take(to_delete) {
            match client.delete_object().bucket(bucket).key(&key).send().await {
                Ok(_) => {
                    checkpoints_deleted += 1;
                }
                Err(e) => {
                    eprintln!("S3: Warning: failed to delete checkpoint {}: {}", key, e);
                }
            }
        }
    }

    // 2. Prune queue-spill segments ONLY if we have a min_segment_id
    // Without min_segment_id, we cannot safely determine which segments are pruneable
    let Some(min_id) = min_segment_id else {
        // No min_segment_id in manifest - skip segment pruning to be safe
        // This happens on first checkpoint or with old manifests
        if checkpoints_deleted > 0 {
            eprintln!(
                "S3: Pruned {} checkpoints (skipped segment pruning - no min_segment_id)",
                checkpoints_deleted
            );
        }
        return Ok(());
    };

    let queue_prefix = format!("{}/queue-spill/", prefix);
    let mut s3_segments: Vec<String> = Vec::new();
    continuation_token = None;

    loop {
        let mut request = client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(&queue_prefix);

        if let Some(token) = continuation_token {
            request = request.continuation_token(token);
        }

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "S3: Warning: failed to list queue segments for pruning: {}",
                    e
                );
                return Ok(());
            }
        };

        for obj in response.contents() {
            if let Some(key) = obj.key() {
                s3_segments.push(key.to_string());
            }
        }

        if response.is_truncated() == Some(true) {
            continuation_token = response.next_continuation_token().map(|s| s.to_string());
        } else {
            break;
        }
    }

    // Only prune segments with ID < min_segment_id
    // Segments with ID >= min_segment_id are needed for resume
    for key in s3_segments {
        let filename = key.split('/').last().unwrap_or(&key);

        // Extract segment ID from filename
        let segment_id = match extract_segment_id(filename) {
            Some(id) => id,
            None => continue, // Skip non-segment files
        };

        // CRITICAL: Only prune if segment ID is BELOW the minimum needed
        if segment_id < min_id {
            match client.delete_object().bucket(bucket).key(&key).send().await {
                Ok(_) => {
                    segments_deleted += 1;
                    // Remove from tracked offsets
                    let rel_path = format!("queue-spill/{}", filename);
                    uploaded_offsets.remove(&rel_path);
                }
                Err(e) => {
                    eprintln!("S3: Warning: failed to delete segment {}: {}", key, e);
                }
            }
        }
    }

    // 3. Also prune LOCAL segment files with ID < min_segment_id
    // This frees local disk space after segments have been uploaded to S3
    let local_spill_dir = local_dir.join("queue-spill");
    if local_spill_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&local_spill_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "bin").unwrap_or(false) {
                    if let Some(segment_id) =
                        extract_segment_id(path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
                    {
                        // Only delete if segment ID is below the minimum needed
                        if segment_id < min_id {
                            if std::fs::remove_file(&path).is_ok() {
                                local_segments_deleted += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if checkpoints_deleted > 0 || segments_deleted > 0 || local_segments_deleted > 0 {
        eprintln!(
            "S3: Pruned {} checkpoints, {} S3 segments, {} local segments (min_segment_id={})",
            checkpoints_deleted, segments_deleted, local_segments_deleted, min_id
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
                min_segment_id: Some(42),
            }),
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: S3Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.run_id, "test-run");
        assert!(parsed.checkpoint.is_some());
        assert_eq!(parsed.checkpoint.as_ref().unwrap().min_segment_id, Some(42));
    }

    #[test]
    fn test_extract_segment_id() {
        // Valid segment filenames
        assert_eq!(extract_segment_id("segment-0000000000000042.bin"), Some(42));
        assert_eq!(extract_segment_id("segment-0000000000000000.bin"), Some(0));
        assert_eq!(
            extract_segment_id("segment-9999999999999999.bin"),
            Some(9999999999999999)
        );

        // Invalid filenames
        assert_eq!(extract_segment_id("checkpoint-123.json"), None);
        assert_eq!(extract_segment_id("segment-abc.bin"), None);
        assert_eq!(extract_segment_id("segment-42"), None);
        assert_eq!(extract_segment_id("other.bin"), None);
        assert_eq!(extract_segment_id(""), None);
    }

    #[test]
    fn test_manifest_backwards_compatibility() {
        // Test that old manifests without min_segment_id can still be parsed
        let old_manifest_json = r#"{
            "updated_at": 1234567890,
            "run_id": "old-run",
            "files": {},
            "checkpoint": {
                "id": 1,
                "states_generated": 1000,
                "states_distinct": 900,
                "queue_pending": 500
            }
        }"#;

        let parsed: S3Manifest = serde_json::from_str(old_manifest_json).unwrap();
        assert_eq!(parsed.run_id, "old-run");
        assert!(parsed.checkpoint.is_some());
        // min_segment_id should default to None for old manifests
        assert_eq!(parsed.checkpoint.as_ref().unwrap().min_segment_id, None);
    }

    #[test]
    fn test_checkpoint_state_with_min_segment_id() {
        // Test serialization and deserialization with min_segment_id
        let checkpoint = CheckpointState {
            id: 123,
            states_generated: 5000,
            states_distinct: 4500,
            queue_pending: 100,
            min_segment_id: Some(42),
        };

        let json = serde_json::to_string(&checkpoint).unwrap();
        let parsed: CheckpointState = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, 123);
        assert_eq!(parsed.min_segment_id, Some(42));

        // Test with None
        let checkpoint_none = CheckpointState {
            id: 456,
            states_generated: 1000,
            states_distinct: 900,
            queue_pending: 0,
            min_segment_id: None,
        };

        let json = serde_json::to_string(&checkpoint_none).unwrap();
        let parsed: CheckpointState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.min_segment_id, None);
    }

    #[test]
    fn chooses_single_put_below_threshold_and_multipart_above_it() {
        let tuning = S3TransferTuning {
            multipart_threshold_bytes: 64 * 1024 * 1024,
            ..S3TransferTuning::default()
        };

        assert_eq!(
            choose_upload_mode(63 * 1024 * 1024, tuning),
            UploadMode::SinglePut
        );

        match choose_upload_mode(96 * 1024 * 1024, tuning) {
            UploadMode::Multipart {
                part_size_bytes,
                part_count,
            } => {
                assert!(part_size_bytes >= S3_MIN_MULTIPART_PART_SIZE_BYTES);
                assert!(part_count >= 2);
            }
            UploadMode::SinglePut => panic!("expected multipart upload for large object"),
        }
    }

    #[test]
    fn multipart_ranges_expand_part_size_to_stay_under_s3_limit() {
        let file_size =
            (S3_MAX_MULTIPART_PARTS as u64 * S3_MIN_MULTIPART_PART_SIZE_BYTES as u64) + 1;
        let part_size = compute_multipart_part_size(file_size, S3_MIN_MULTIPART_PART_SIZE_BYTES);
        let ranges = plan_multipart_ranges(file_size, S3_MIN_MULTIPART_PART_SIZE_BYTES);

        assert!(part_size > S3_MIN_MULTIPART_PART_SIZE_BYTES);
        assert!(ranges.len() <= S3_MAX_MULTIPART_PARTS);
        assert_eq!(ranges.first().unwrap().1, 0);
        let total_len: u64 = ranges.iter().map(|(_, _, len)| *len as u64).sum();
        assert_eq!(total_len, file_size);
    }

    proptest! {
        #[test]
        fn multipart_ranges_cover_the_object_without_gaps(
            file_size in 1u64..(512 * 1024 * 1024u64),
            preferred_mb in 1usize..64usize,
        ) {
            let preferred_part_size = preferred_mb * 1024 * 1024;
            let ranges = plan_multipart_ranges(file_size, preferred_part_size);
            prop_assert!(!ranges.is_empty());
            prop_assert!(ranges.len() <= S3_MAX_MULTIPART_PARTS);

            let computed_part_size = compute_multipart_part_size(file_size, preferred_part_size);
            let mut expected_offset = 0u64;
            for (idx, (part_number, offset, len)) in ranges.iter().enumerate() {
                prop_assert_eq!(*part_number as usize, idx + 1);
                prop_assert_eq!(*offset, expected_offset);
                prop_assert!(*len > 0);
                if idx + 1 < ranges.len() {
                    prop_assert_eq!(*len, computed_part_size);
                } else {
                    prop_assert!(*len <= computed_part_size);
                }
                expected_offset += *len as u64;
            }
            prop_assert_eq!(expected_offset, file_size);
        }
    }
}
