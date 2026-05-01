#!/usr/bin/env bash
#
# T11.3 — CI-gate smoke variant of the 1-hour chaos soak.
#
# The full release-ritual soak (`scripts/chaos_soak.sh --duration 3600 \
# --swarm-mode auto`) takes an hour and is run pre-release. That cadence
# does not catch chaos-bug regressions on per-PR commits. This wrapper
# runs the same harness with a 5-minute budget and tighter per-iter
# timeouts so it fits inside a per-PR CI workflow (~10 min total when
# you include the build).
#
# Defaults are tuned to:
#   - exercise >= 6 of the 12 failpoints in the catalog at least once
#     (statistically ~certain at 30+ iterations with --swarm-mode auto)
#   - finish in 5 minutes wall-clock on a 2-core CI runner
#   - share the EXACT same harness as the long-form soak so any
#     divergence/hang surfaces here too — only the duration differs
#
# Usage:
#   scripts/chaos_smoke.sh [--duration SECS] [--workdir PATH]
#                          [--per-iter-timeout SECS]
#                          [--binary PATH] [--summary-out PATH]
#
# Exit codes:
#   0 — clean (no divergences, no hangs)
#   1 — divergence or hang detected
#   2 — usage error / missing inputs
#
# CI usage example: see .github/workflows/chaos-smoke.yml

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- defaults --------------------------------------------------------------
# 5 minutes total wall-clock — enough for ~30-60 iterations of CheckpointDrain
# at 2 workers. With swarm-mode auto picking 1-4 failpoints/iter, that
# expectation-covers all 12 failpoints with overwhelming probability.
DURATION_SECS=300
# Tighter per-iter timeout than the long-form soak: smoke iterations
# should be 1-5s on the small spec, so 30s gives plenty of headroom for
# slow CI runners while still flagging hangs quickly.
PER_ITER_TIMEOUT=30
# CI runners typically have 2 vCPUs.
WORKERS=2
WORKDIR_BASE=""
BINARY_PATH=""
SUMMARY_OUT=""
# Minimum failpoints we expect to fire over the smoke. The acceptance
# criterion is >= 6 of the 12 in the catalog.
MIN_FAILPOINTS_EXERCISED=6

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION_SECS="$2"; shift 2 ;;
    --per-iter-timeout) PER_ITER_TIMEOUT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --workdir) WORKDIR_BASE="$2"; shift 2 ;;
    --binary) BINARY_PATH="$2"; shift 2 ;;
    --summary-out) SUMMARY_OUT="$2"; shift 2 ;;
    --min-failpoints) MIN_FAILPOINTS_EXERCISED="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${WORKDIR_BASE}" ]]; then
  WORKDIR_BASE="${REPO_ROOT}/.chaos-smoke"
fi
if [[ -z "${BINARY_PATH}" ]]; then
  BINARY_PATH="${REPO_ROOT}/target/release/tlaplusplus"
fi
if [[ -z "${SUMMARY_OUT}" ]]; then
  SUMMARY_OUT="${WORKDIR_BASE}/smoke-summary.txt"
fi

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "ERROR: binary not found or not executable: ${BINARY_PATH}" >&2
  echo "       build with: cargo build --release --features failpoints" >&2
  exit 2
fi

mkdir -p "${WORKDIR_BASE}"

echo "============================================================"
echo "T11.3 chaos smoke (CI gate)"
echo "  duration:    ${DURATION_SECS}s"
echo "  per-iter:    ${PER_ITER_TIMEOUT}s"
echo "  workers:     ${WORKERS}"
echo "  workdir:     ${WORKDIR_BASE}"
echo "  binary:      ${BINARY_PATH}"
echo "  min-fps:     ${MIN_FAILPOINTS_EXERCISED} of 12 expected to fire"
echo "============================================================"

# Delegate to the long-form soak with smoke parameters. Sharing the
# harness means any harness fix or new failpoint flows into the smoke
# automatically — there is exactly one chaos test path, just two
# durations.
SOAK_EXIT=0
"${SCRIPT_DIR}/chaos_soak.sh" \
  --duration "${DURATION_SECS}" \
  --per-iter-timeout "${PER_ITER_TIMEOUT}" \
  --workers "${WORKERS}" \
  --workdir "${WORKDIR_BASE}" \
  --binary "${BINARY_PATH}" \
  --summary-out "${SUMMARY_OUT}" \
  --swarm-mode auto \
  --swarm-max 4 \
  || SOAK_EXIT=$?

echo
echo "============================================================"
echo "SMOKE COVERAGE GATE"
echo "============================================================"

# Count distinct failpoints that actually fired (>0 fires) by parsing
# the iterations TSV. Column layout (chaos_soak.sh sets this):
#   1=iter 2=wall_ms 3=swarm_n 4=failpoint 5=action 6=exit
#   7=distinct 8=verdict 9=divergent 10=notes
# Column 4 may contain comma-separated names in swarm mode.
ITER_TSV="${WORKDIR_BASE}/iterations.tsv"
if [[ ! -f "${ITER_TSV}" ]]; then
  echo "ERROR: ${ITER_TSV} not produced by chaos_soak.sh — cannot gate" >&2
  exit 1
fi

# Skip header (NR>1) and the CONTROL row (col4 == "CONTROL"). Split
# col4 on commas to get individual failpoint names from swarm iters.
EXERCISED=$(awk -F'\t' 'NR>1 && $4 != "CONTROL" {
  n = split($4, names, ",");
  for (i = 1; i <= n; i++) {
    if (names[i] != "") seen[names[i]] = 1;
  }
}
END {
  count = 0;
  for (k in seen) {
    print k;
    count++;
  }
}' "${ITER_TSV}" | sort)

EXERCISED_COUNT=$(echo "${EXERCISED}" | grep -c -v '^$' || true)
echo "Failpoints exercised: ${EXERCISED_COUNT} of 12 (min required: ${MIN_FAILPOINTS_EXERCISED})"
echo "${EXERCISED}" | sed 's/^/  - /'

GATE_EXIT=0
if [[ "${EXERCISED_COUNT}" -lt "${MIN_FAILPOINTS_EXERCISED}" ]]; then
  echo "FAIL: only ${EXERCISED_COUNT} of 12 failpoints exercised — minimum is ${MIN_FAILPOINTS_EXERCISED}" >&2
  GATE_EXIT=1
fi
if [[ "${SOAK_EXIT}" -ne 0 ]]; then
  echo "FAIL: chaos_soak.sh reported divergence or hang (exit ${SOAK_EXIT})" >&2
  GATE_EXIT=1
fi

if [[ "${GATE_EXIT}" -eq 0 ]]; then
  echo "PASS: chaos smoke clean — ${EXERCISED_COUNT}/12 failpoints exercised, 0 divergences, 0 hangs"
fi

exit "${GATE_EXIT}"
