# S3 Persistence Feature - Handoff Notes

## What Was Implemented

S3 persistence for checkpoint/resume across spot instance terminations. Continuous background upload of exploration state to S3, with emergency flush on SIGTERM (spot 2-minute warning).

## Key Files

- `src/storage/s3_persistence.rs` - S3 client, background upload, emergency flush
- `src/main.rs` - CLI integration (`S3Args`, `run_model_with_s3()`)
- `Cargo.toml` - Optional `s3` feature with `aws-sdk-s3`, `aws-config`

## Build

```bash
cargo build --release --features s3
```

## CLI Usage

```bash
# Start new run with S3 persistence
./target/release/tlaplusplus run-high-branching \
  --s3-bucket my-bucket \
  --s3-prefix runs/my-run-123 \
  --max-depth 10 \
  --branching-factor 200 \
  --workers 120 \
  --use-bloom-fingerprints

# Resume after spot termination
./target/release/tlaplusplus run-high-branching \
  --s3-bucket my-bucket \
  --s3-prefix runs/my-run-123 \
  --resume \
  ... (same args)
```

## CLI Options

- `--s3-bucket <BUCKET>` - S3 bucket name (required to enable S3)
- `--s3-prefix <PREFIX>` - Path prefix in bucket (default: auto-generated timestamp)
- `--s3-upload-interval-secs <N>` - Background upload interval (default: 10)

## How It Works

1. **Background upload** - Every N seconds, scans work_dir and uploads changed files to S3
2. **Byte offset tracking** - No MD5 checksums; tracks uploaded bytes per file for efficient deltas
3. **SIGTERM handler** - On signal, performs emergency flush (uploads all pending + manifest)
4. **Resume** - Downloads manifest + all files from S3 before starting exploration

## AWS Credentials

Uses AWS SDK auto-discovery:
- EC2 instance profile (IAM role) - recommended for spot instances
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- `~/.aws/credentials` file

## S3 Layout

```
s3://bucket/prefix/
├── manifest.json          # Tracks uploaded files, checkpoint state
├── fingerprints/
│   └── shard-*/           # Fingerprint segment files
├── queue/
│   └── worker-*/          # Queue spill files
└── checkpoints/
    └── *.json             # Checkpoint manifests
```

## Tests

All 120 tests pass. No S3-specific unit tests yet (would require localstack or mocking).

## Notes

- S3 feature is optional - builds without it by default
- Emergency flush targets ~30 seconds (spot gives 2 minutes warning)
- Instance needs IAM policy with s3:GetObject, s3:PutObject, s3:ListBucket, s3:HeadBucket
