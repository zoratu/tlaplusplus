#!/usr/bin/env bash
#
# regen_state_graph_snapshots.sh — print fresh digests for tests/state_graph_snapshots.rs
#
# Convenience wrapper around the `snapshot_regen` ignored test. Prints one
# line per spec showing the current state count and content-addressed
# digest. Does NOT auto-update the source — paste new values into the
# `SNAPSHOTS[]` array in tests/state_graph_snapshots.rs after cross-checking
# the count against TLC.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cargo test --release --test state_graph_snapshots snapshot_regen \
    -- --ignored --nocapture
