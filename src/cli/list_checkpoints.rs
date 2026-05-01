//! `list-checkpoints` subcommand handler.
//!
//! Extracted from `src/main.rs` as part of the cli/ refactor.

use super::shared::format_num;

pub(crate) fn list_checkpoints(
    work_dir: Option<std::path::PathBuf>,
    s3_bucket: Option<String>,
    s3_prefix: String,
    s3_region: Option<String>,
    validate: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let rt = tokio::runtime::Runtime::new()?;

    // Check S3 if bucket is provided
    if let Some(bucket) = s3_bucket {
        println!("Checking S3: s3://{}/{}", bucket, s3_prefix);
        println!();

        rt.block_on(async {
            list_s3_checkpoints(&bucket, &s3_prefix, s3_region.as_deref(), validate).await
        })?;
    }

    // Check local disk if work_dir is provided
    if let Some(dir) = work_dir {
        println!("Checking local: {}", dir.display());
        println!();

        let checkpoints_dir = dir.join("checkpoints");
        if checkpoints_dir.exists() {
            let mut checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoints_dir)
                .context("failed reading checkpoints directory")?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    // SAFE: lossy fine here — we only match the ASCII prefix of
                    // checkpoint files we generate ourselves. A non-UTF-8 sibling
                    // file would get U+FFFD-converted and fail the prefix match,
                    // so it is correctly skipped (not enumerated as a checkpoint).
                    e.path()
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with("checkpoint-"))
                        .unwrap_or(false)
                })
                .collect();

            checkpoint_files.sort_by_key(|e| e.path());

            if checkpoint_files.is_empty() {
                println!("  No checkpoint files found");
            } else {
                for entry in checkpoint_files {
                    let path = entry.path();
                    if let Ok(contents) = std::fs::read_to_string(&path) {
                        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&contents) {
                            let states_generated = manifest
                                .get("states_generated")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let states_distinct = manifest
                                .get("states_distinct")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let created = manifest
                                .get("created_unix_secs")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);

                            let time_str = if created > 0 {
                                chrono::DateTime::from_timestamp(created as i64, 0)
                                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                                    .unwrap_or_else(|| "unknown".to_string())
                            } else {
                                "unknown".to_string()
                            };

                            println!(
                                "  {} - {} generated, {} distinct ({})",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                format_num(states_generated),
                                format_num(states_distinct),
                                time_str
                            );
                        }
                    }
                }
            }
        } else {
            println!(
                "  Checkpoints directory not found: {}",
                checkpoints_dir.display()
            );
        }

        // Check queue-spill segments
        let queue_spill_dir = dir.join("queue-spill");
        if queue_spill_dir.exists() {
            let segment_count = std::fs::read_dir(&queue_spill_dir)
                .map(|r| r.filter_map(|e| e.ok()).count())
                .unwrap_or(0);
            println!("  Queue segments on disk: {}", segment_count);
        }
    }

    Ok(())
}


pub(crate) async fn list_s3_checkpoints(
    bucket: &str,
    prefix: &str,
    region: Option<&str>,
    validate: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;
    use aws_config::BehaviorVersion;
    use aws_sdk_s3::Client;
    use aws_sdk_s3::config::Region;

    // Load AWS config
    let mut config_loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(r) = region {
        config_loader = config_loader.region(Region::new(r.to_string()));
    }
    let config = config_loader.load().await;
    let client = Client::new(&config);

    // Fetch manifest.json
    let manifest_key = if prefix.is_empty() {
        "manifest.json".to_string()
    } else {
        format!("{}/manifest.json", prefix)
    };

    let manifest_result = client
        .get_object()
        .bucket(bucket)
        .key(&manifest_key)
        .send()
        .await;

    match manifest_result {
        Ok(resp) => {
            let body = resp
                .body
                .collect()
                .await
                .context("failed reading S3 manifest body")?;
            let manifest: serde_json::Value = serde_json::from_slice(&body.into_bytes())
                .context("failed parsing manifest JSON")?;

            println!(
                "  Run ID: {}",
                manifest
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
            println!(
                "  Last updated: {}",
                manifest
                    .get("updated_at")
                    .and_then(|v| v.as_u64())
                    .map(|ts| {
                        chrono::DateTime::from_timestamp(ts as i64, 0)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                            .unwrap_or_else(|| "unknown".to_string())
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            );

            // List checkpoint info
            if let Some(checkpoint) = manifest.get("checkpoint") {
                let id = checkpoint.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                let states_generated = checkpoint
                    .get("states_generated")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let states_distinct = checkpoint
                    .get("states_distinct")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let queue_pending = checkpoint
                    .get("queue_pending")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let min_segment_id = checkpoint.get("min_segment_id").and_then(|v| v.as_u64());

                println!();
                println!("  Checkpoint #{}", id);
                println!("    States generated: {}", format_num(states_generated));
                println!("    States distinct:  {}", format_num(states_distinct));
                println!("    Queue pending:    {}", format_num(queue_pending));
                if let Some(min_seg) = min_segment_id {
                    println!("    Min segment ID:   {}", min_seg);
                }

                // Validate segments if requested
                if validate {
                    println!();
                    println!("  Validating queue segments...");

                    // List all segment files in S3
                    let segment_prefix = if prefix.is_empty() {
                        "queue-spill/".to_string()
                    } else {
                        format!("{}/queue-spill/", prefix)
                    };

                    let mut segments: Vec<(u64, String)> = Vec::new();
                    let mut continuation_token: Option<String> = None;

                    loop {
                        let mut req = client
                            .list_objects_v2()
                            .bucket(bucket)
                            .prefix(&segment_prefix);

                        if let Some(token) = continuation_token.take() {
                            req = req.continuation_token(token);
                        }

                        let resp = req.send().await.context("failed listing S3 segments")?;

                        for obj in resp.contents() {
                            if let Some(key) = obj.key() {
                                // Extract segment ID from filename like "segment-0000000000012345.bin"
                                if let Some(filename) = key.rsplit('/').next() {
                                    if filename.starts_with("segment-")
                                        && filename.ends_with(".bin")
                                    {
                                        let id_str = filename
                                            .trim_start_matches("segment-")
                                            .trim_end_matches(".bin");
                                        if let Ok(seg_id) = id_str.parse::<u64>() {
                                            segments.push((seg_id, key.to_string()));
                                        }
                                    }
                                }
                            }
                        }

                        if resp.is_truncated() == Some(true) {
                            continuation_token =
                                resp.next_continuation_token().map(|s| s.to_string());
                        } else {
                            break;
                        }
                    }

                    segments.sort_by_key(|(id, _)| *id);

                    let total_segments = segments.len();
                    println!("    Total segments in S3: {}", total_segments);

                    if let Some(min_seg) = min_segment_id {
                        let required_segments: Vec<_> =
                            segments.iter().filter(|(id, _)| *id >= min_seg).collect();

                        let required_count = required_segments.len();
                        println!(
                            "    Segments >= min_segment_id ({}): {}",
                            min_seg, required_count
                        );

                        // Check for gaps
                        if !required_segments.is_empty() {
                            let first_id = required_segments.first().unwrap().0;
                            let last_id = required_segments.last().unwrap().0;
                            let expected_count = (last_id - first_id + 1) as usize;

                            if required_count < expected_count {
                                let mut missing = Vec::new();
                                let segment_ids: std::collections::HashSet<u64> =
                                    required_segments.iter().map(|(id, _)| *id).collect();

                                for id in first_id..=last_id {
                                    if !segment_ids.contains(&id) {
                                        missing.push(id);
                                        if missing.len() >= 10 {
                                            break;
                                        }
                                    }
                                }

                                println!("    ⚠️  MISSING SEGMENTS detected!");
                                println!(
                                    "       Expected {} segments, found {}",
                                    expected_count, required_count
                                );
                                println!("       Missing (first 10): {:?}", missing);
                                println!();
                                println!("    This checkpoint may not be resumable.");
                            } else {
                                println!("    ✓ All required segments present (no gaps)");
                            }
                        }
                    } else {
                        println!(
                            "    ⚠️  No min_segment_id in checkpoint - cannot validate completeness"
                        );
                    }
                }
            } else {
                println!("  No checkpoint state in manifest");
            }

            // Count files by type
            if let Some(files) = manifest.get("files").and_then(|v| v.as_object()) {
                let mut checkpoint_files = 0;
                let mut fingerprint_files = 0;
                let mut queue_files = 0;
                let mut other_files = 0;

                for key in files.keys() {
                    if key.contains("checkpoint") {
                        checkpoint_files += 1;
                    } else if key.contains("fingerprint") {
                        fingerprint_files += 1;
                    } else if key.contains("queue") || key.contains("segment") {
                        queue_files += 1;
                    } else {
                        other_files += 1;
                    }
                }

                println!();
                println!("  Files in manifest:");
                println!("    Checkpoint files: {}", checkpoint_files);
                println!("    Fingerprint files: {}", fingerprint_files);
                println!("    Queue/segment files: {}", queue_files);
                if other_files > 0 {
                    println!("    Other files: {}", other_files);
                }
            }
        }
        Err(e) => {
            println!("  No manifest found at s3://{}/{}", bucket, manifest_key);
            println!("  Error: {}", e);
        }
    }

    Ok(())
}


pub(crate) fn handle(
    work_dir: Option<std::path::PathBuf>,
    s3_bucket: Option<String>,
    s3_prefix: String,
    s3_region: Option<String>,
    validate: bool,
) -> anyhow::Result<()> {
    list_checkpoints(work_dir, s3_bucket, s3_prefix, s3_region, validate)
}
