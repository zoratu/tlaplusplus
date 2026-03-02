# Cloud-Init Setup for tlaplusplus Spot Instances

Add this to your EC2 launch template user data to pre-install dependencies:

```yaml
#cloud-config
package_update: true
packages:
  - build-essential
  - cmake
  - pkg-config
  - libssl-dev
  - git

runcmd:
  # Install Rust for ubuntu user
  - su - ubuntu -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

  # Pre-clone the repo (optional)
  - su - ubuntu -c 'git clone https://github.com/YOUR_ORG/tlaplusplus.git ~/tlaplusplus || true'

  # Pre-build release binary with S3 feature (optional, takes ~2 min)
  - su - ubuntu -c 'source ~/.cargo/env && cd ~/tlaplusplus && cargo build --release --features s3 || true'
```

## Required IAM Policy for S3 Persistence

Attach this policy to the instance profile:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:HeadBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET_NAME",
        "arn:aws:s3:::YOUR_BUCKET_NAME/*"
      ]
    }
  ]
}
```

## Usage

```bash
./target/release/tlaplusplus run-high-branching \
  --s3-bucket YOUR_BUCKET_NAME \
  --s3-prefix runs/$(date +%Y%m%d-%H%M%S) \
  --max-depth 10 \
  --branching-factor 200 \
  --workers 120 \
  --use-bloom-fingerprints
```

## On Spot Termination

The SIGTERM handler automatically flushes to S3. Resume with:

```bash
./target/release/tlaplusplus run-high-branching \
  --s3-bucket YOUR_BUCKET_NAME \
  --s3-prefix runs/PREVIOUS_RUN_PREFIX \
  --resume \
  ... (same args)
```
