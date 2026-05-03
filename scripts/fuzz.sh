#!/usr/bin/env bash
# Fuzzing script with AddressSanitizer (ASAN) support
#
# Usage:
#   ./scripts/fuzz.sh <target> [duration_secs]
#
# Examples:
#   ./scripts/fuzz.sh fuzz_tla_module           # Run indefinitely
#   ./scripts/fuzz.sh fuzz_fingerprint_store 300  # Run for 5 minutes
#
# Available targets:
#   fuzz_tla_module, fuzz_tla_config, fuzz_tla_value_ops
#   fuzz_queue_serialization, fuzz_fingerprint_store
#   fuzz_bloom_filter, fuzz_s3_manifest

set -e

TARGET=${1:-""}
DURATION=${2:-0}

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target> [duration_secs]"
    echo ""
    echo "Available targets:"
    cargo +nightly fuzz list 2>/dev/null || echo "  (run 'cargo +nightly fuzz list' to see targets)"
    exit 1
fi

echo "🔍 Running fuzzer: $TARGET"
echo "   Sanitizer: AddressSanitizer (ASAN)"
if [ "$DURATION" -gt 0 ]; then
    echo "   Duration: ${DURATION}s"
else
    echo "   Duration: unlimited (Ctrl+C to stop)"
fi
echo ""

# Build and run with AddressSanitizer
# ASAN catches:
#   - Buffer overflows (stack/heap/global)
#   - Use-after-free
#   - Double-free
#   - Memory leaks
#   - Invalid pointer operations

# RSS cap: libFuzzer defaults to 2 GB and exits with status 71 on overflow.
# Eval-path fuzz targets (notably fuzz_tla_swarm) can construct adversarial
# TLA+ expressions that legally allocate gigabytes — the eval budget
# infrastructure (`set_active_eval_budget` + `EvalContext::check_budget`)
# bounds memory when set, but the fuzz harness uses a 100K-element budget
# that doesn't translate to a hard byte cap. Raising the libFuzzer RSS
# limit to 8 GB lets adversarial inputs run to completion so the fuzzer
# focuses on what matters: panics, divergences, sanitizer trips. OOM noise
# is suppressed; real soundness bugs still fail the run. Override with
# `RSS_LIMIT_MB=N ./scripts/fuzz.sh ...`.
RSS_LIMIT_MB=${RSS_LIMIT_MB:-8192}

FUZZ_ARGS="-- -rss_limit_mb=$RSS_LIMIT_MB"
if [ "$DURATION" -gt 0 ]; then
    FUZZ_ARGS="$FUZZ_ARGS -max_total_time=$DURATION"
fi

cargo +nightly fuzz run "$TARGET" \
    --sanitizer=address \
    $FUZZ_ARGS
