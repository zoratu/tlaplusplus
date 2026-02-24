#!/usr/bin/env bash
# Run performance benchmarks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🏃 Running performance benchmarks..."
echo ""

# Build release
echo "▶ Building release..."
cargo build --release
echo ""

# Run counter-grid benchmark
echo "▶ Benchmark: Counter Grid (small)"
./target/release/tlaplusplus run-counter-grid \
    --max-x 100 \
    --max-y 100 \
    --max-sum 200 \
    --workers 4 \
    --work-dir ./.tlapp-bench \
    --clean-work-dir=true \
    --checkpoint-interval-secs 0 \
    --checkpoint-on-exit=false

echo ""
echo "✅ Benchmarks complete"
