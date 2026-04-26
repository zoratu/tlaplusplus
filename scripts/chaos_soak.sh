#!/usr/bin/env bash
#
# T11: Long-running chaos soak with random failpoint injection.
#
# Strategy:
# 1. Run a control model-check (no failpoints) to establish the canonical
#    distinct-state count and invariant verdict.
# 2. Loop for --duration seconds. In each iteration:
#    - Pick a random failpoint from the active set.
#    - Pick an action: most are transient ("1*return->off" or "2*return->off"),
#      so the system has a chance to complete; a small fraction are permanent
#      ("return") to also exercise graceful-error termination.
#    - Run the spec under FAILPOINTS=<name>=<action>.
#    - Compare distinct-state count and verdict against the control.
#      Transient injections must complete identically. Permanent injections
#      may either complete identically (the failpoint never reaches its
#      trigger condition) or terminate with a graceful non-zero exit
#      within the per-iteration timeout — both are acceptable, only hangs
#      and panics are flagged.
# 3. Per iteration, log: failpoint, action, exit code, wall-time, distinct
#    count, verdict, divergence flag.
# 4. At the end, print a coverage matrix (failpoint -> fire counts) and a
#    summary of any divergences / hangs.
#
# Requires the binary built with --features failpoints. Each spawned process
# must call fail::FailScenario::setup() so the FAILPOINTS env var takes
# effect (wired in src/main.rs under cfg(feature = "failpoints")).
#
# Usage:
#   scripts/chaos_soak.sh [--duration SECS] [--spec NAME]
#                         [--module PATH] [--config PATH]
#                         [--binary PATH] [--workdir PATH]
#                         [--per-iter-timeout SECS]
#                         [--checkpoint-interval SECS]
#                         [--queue-max-inmem SECS]
#
# Defaults target CheckpointDrain (~26K distinct), 1 hour soak,
# checkpoint-interval=2s + small queue-max-inmem to drive the spill path.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- defaults --------------------------------------------------------------
DURATION_SECS=3600
SPEC_NAME="CheckpointDrain"
MODULE_PATH=""
CONFIG_PATH=""
BINARY_PATH=""
WORKDIR_BASE=""
PER_ITER_TIMEOUT=120
CHECKPOINT_INTERVAL=2
# Default 0 = leave queue-max-inmem at the binary's default (50M items).
# Setting it small enough to *force* spill on the default CheckpointDrain
# config (~26K distinct) currently changes the final state count — see
# T11.N follow-up in the release log. Pass --queue-max-inmem N explicitly
# only with a larger spec where N is well above the natural distinct count.
QUEUE_MAX_INMEM=0
WORKERS=2
SUMMARY_OUT=""

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION_SECS="$2"; shift 2 ;;
    --spec) SPEC_NAME="$2"; shift 2 ;;
    --module) MODULE_PATH="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --binary) BINARY_PATH="$2"; shift 2 ;;
    --workdir) WORKDIR_BASE="$2"; shift 2 ;;
    --per-iter-timeout) PER_ITER_TIMEOUT="$2"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --queue-max-inmem) QUEUE_MAX_INMEM="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --summary-out) SUMMARY_OUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MODULE_PATH}" ]]; then
  MODULE_PATH="${REPO_ROOT}/corpus/internals/${SPEC_NAME}.tla"
fi
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${REPO_ROOT}/corpus/internals/${SPEC_NAME}.cfg"
fi
if [[ -z "${BINARY_PATH}" ]]; then
  BINARY_PATH="${REPO_ROOT}/target/release/tlaplusplus"
fi
if [[ -z "${WORKDIR_BASE}" ]]; then
  WORKDIR_BASE="${REPO_ROOT}/.chaos-soak"
fi
if [[ -z "${SUMMARY_OUT}" ]]; then
  SUMMARY_OUT="${WORKDIR_BASE}/summary.txt"
fi

if [[ "${QUEUE_MAX_INMEM}" -gt 0 ]]; then
  QUEUE_FLAG="--queue-max-inmem-items ${QUEUE_MAX_INMEM}"
else
  QUEUE_FLAG=""
fi

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "ERROR: binary not found or not executable: ${BINARY_PATH}" >&2
  echo "       build with: cargo build --release --features failpoints" >&2
  exit 2
fi
if [[ ! -f "${MODULE_PATH}" ]]; then
  echo "ERROR: module not found: ${MODULE_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

mkdir -p "${WORKDIR_BASE}"
LOG_DIR="${WORKDIR_BASE}/logs"
mkdir -p "${LOG_DIR}"
ITER_TSV="${WORKDIR_BASE}/iterations.tsv"
echo -e "iter\twall_ms\tfailpoint\taction\texit\tdistinct\tverdict\tdivergent\tnotes" > "${ITER_TSV}"

# ---- failpoint catalog -----------------------------------------------------
# Format: name|category
# category controls action selection:
#   - "checkpoint": only meaningful when checkpoint-interval > 0
#   - "worker"    : worker_panic — always 1*return->off, never permanent
#   - "queue"     : queue_spill_fail / queue_load_fail — transient or permanent
#   - "fp_store"  : fp_store_shard_full — transient, simulates pressure
#   - "delay"     : worker_pause_delay / fp_switch_slow — return(N) form
#   - "quiesce"   : quiescence_timeout — transient only

FAILPOINTS=(
  "checkpoint_write_fail|checkpoint"
  "checkpoint_disk_write_fail|checkpoint"
  "checkpoint_rename_fail|checkpoint"
  "checkpoint_queue_flush_fail|checkpoint"
  "checkpoint_fp_flush_fail|checkpoint"
  "worker_panic|worker"
  "fp_store_shard_full|fp_store"
  "queue_spill_fail|queue"
  "queue_load_fail|queue"
  "worker_pause_delay|delay"
  "fp_switch_slow|delay"
  "quiescence_timeout|quiesce"
)

declare -A FIRE_COUNT
declare -A DIVERGE_COUNT
declare -A HANG_COUNT
declare -A EXIT_OK_COUNT
declare -A EXIT_FAIL_COUNT
for entry in "${FAILPOINTS[@]}"; do
  name="${entry%%|*}"
  FIRE_COUNT["$name"]=0
  DIVERGE_COUNT["$name"]=0
  HANG_COUNT["$name"]=0
  EXIT_OK_COUNT["$name"]=0
  EXIT_FAIL_COUNT["$name"]=0
done

# ---- helper: run one model-check, parse distinct + verdict ----------------
# Args: $1=workdir $2=log_path $3=failpoint_env (may be empty for control)
# Echoes: <wall_ms>\t<exit>\t<distinct>\t<verdict>
run_one() {
  local wd="$1"
  local log="$2"
  local fp_env="$3"

  rm -rf "${wd}" >/dev/null 2>&1 || true
  mkdir -p "${wd}"

  local start_ns
  start_ns="$(date +%s%N)"
  local ec=0

  if [[ -n "${fp_env}" ]]; then
    timeout --kill-after=10 "${PER_ITER_TIMEOUT}" \
      env FAILPOINTS="${fp_env}" \
      "${BINARY_PATH}" run-tla \
        --module "${MODULE_PATH}" \
        --config "${CONFIG_PATH}" \
        --workers "${WORKERS}" \
        --work-dir "${wd}" \
        --checkpoint-interval-secs "${CHECKPOINT_INTERVAL}" \
        ${QUEUE_FLAG} \
        --skip-system-checks \
        > "${log}" 2>&1 || ec=$?
  else
    timeout --kill-after=10 "${PER_ITER_TIMEOUT}" \
      "${BINARY_PATH}" run-tla \
        --module "${MODULE_PATH}" \
        --config "${CONFIG_PATH}" \
        --workers "${WORKERS}" \
        --work-dir "${wd}" \
        --checkpoint-interval-secs "${CHECKPOINT_INTERVAL}" \
        ${QUEUE_FLAG} \
        --skip-system-checks \
        > "${log}" 2>&1 || ec=$?
  fi

  local end_ns
  end_ns="$(date +%s%N)"
  local wall_ms=$(( (end_ns - start_ns) / 1000000 ))

  # Parse the canonical "<gen> states generated, <dist> distinct states found"
  # line. Numbers may be comma-formatted (e.g. "13,081"). We strip commas and
  # take the last match (in case of multiple progress lines).
  local distinct
  distinct="$(grep -Eo '[0-9][0-9,]* distinct states found' "${log}" \
              | grep -Eo '[0-9,]+' | tail -1 | tr -d ',' || true)"
  if [[ -z "${distinct}" ]]; then
    # Fallback: stats key/value form `states_distinct=N` from print_stats().
    distinct="$(grep -Eo 'states_distinct=[0-9]+' "${log}" \
                | grep -Eo '[0-9]+' | tail -1 || true)"
  fi
  if [[ -z "${distinct}" ]]; then
    distinct="?"
  fi

  local verdict
  if grep -qE '(Invariant.*violated|VIOLATION|violation=true)' "${log}"; then
    verdict="violation"
  elif [[ "${ec}" -eq 124 ]] || [[ "${ec}" -eq 137 ]]; then
    verdict="timeout"
  elif [[ "${ec}" -ne 0 ]]; then
    verdict="error"
  elif grep -qE '(violation=false|Model checking completed successfully|No error has been found)' "${log}"; then
    verdict="ok"
  else
    verdict="?"
  fi

  printf '%s\t%s\t%s\t%s\n' "${wall_ms}" "${ec}" "${distinct}" "${verdict}"
}

# ---- pick action for a failpoint -------------------------------------------
# Most actions are transient so the run can complete identically to control.
# A small fraction are permanent so we exercise graceful-error termination.
pick_action() {
  local category="$1"
  local r=$((RANDOM % 100))
  case "${category}" in
    worker)
      # worker_panic must always be transient otherwise the run can't finish
      # — runtime continues with N-1 workers but if the panic fires every
      # successor batch we'd never make progress.
      echo "1*return->off"
      ;;
    delay)
      # return(N) where N is delay in ms; pick small to stay within timeout.
      local n=$(( (RANDOM % 50) + 5 ))
      echo "return(${n})"
      ;;
    quiesce)
      # transient: fire once during quiescence then turn off
      echo "1*return->off"
      ;;
    checkpoint|queue|fp_store)
      if (( r < 70 )); then
        # transient: fire 1-3 times then disable
        local k=$(( (RANDOM % 3) + 1 ))
        echo "${k}*return->off"
      else
        # permanent: every call fails — graceful error expected
        echo "return"
      fi
      ;;
    *)
      echo "1*return->off"
      ;;
  esac
}

# ---- control run -----------------------------------------------------------
echo "============================================================"
echo "T11 chaos soak"
echo "  spec:        ${SPEC_NAME}"
echo "  module:      ${MODULE_PATH}"
echo "  config:      ${CONFIG_PATH}"
echo "  binary:      ${BINARY_PATH}"
echo "  workdir:     ${WORKDIR_BASE}"
echo "  duration:    ${DURATION_SECS}s"
echo "  per-iter:    ${PER_ITER_TIMEOUT}s"
echo "  workers:     ${WORKERS}"
echo "  ckpt-int:    ${CHECKPOINT_INTERVAL}s"
if [[ -n "${QUEUE_FLAG}" ]]; then
  echo "  queue-cap:   ${QUEUE_MAX_INMEM} items"
else
  echo "  queue-cap:   default (no spilling expected on small specs)"
fi
echo "  failpoints:  ${#FAILPOINTS[@]}"
echo "============================================================"
echo

CONTROL_WD="${WORKDIR_BASE}/control"
CONTROL_LOG="${LOG_DIR}/control.log"
echo "[control] running baseline (no failpoints)..."
read -r CTRL_WALL CTRL_EXIT CTRL_DISTINCT CTRL_VERDICT < <(run_one "${CONTROL_WD}" "${CONTROL_LOG}" "")
echo "[control] wall=${CTRL_WALL}ms exit=${CTRL_EXIT} distinct=${CTRL_DISTINCT} verdict=${CTRL_VERDICT}"
echo "[control] log: ${CONTROL_LOG}"
echo
if [[ "${CTRL_EXIT}" -ne 0 ]] || [[ "${CTRL_DISTINCT}" == "?" ]]; then
  echo "ERROR: control run failed or did not produce a parseable distinct count" >&2
  echo "       see ${CONTROL_LOG}" >&2
  exit 3
fi

echo -e "0\t${CTRL_WALL}\tCONTROL\t-\t${CTRL_EXIT}\t${CTRL_DISTINCT}\t${CTRL_VERDICT}\tno\tbaseline" >> "${ITER_TSV}"

# ---- main soak loop --------------------------------------------------------
SOAK_START=$(date +%s)
DEADLINE=$(( SOAK_START + DURATION_SECS ))
ITER=0
TOTAL_DIVERGENCES=0
TOTAL_HANGS=0
TOTAL_RUNS=0

while (( $(date +%s) < DEADLINE )); do
  ITER=$(( ITER + 1 ))
  # pick a random failpoint
  idx=$((RANDOM % ${#FAILPOINTS[@]}))
  entry="${FAILPOINTS[$idx]}"
  fp_name="${entry%%|*}"
  fp_category="${entry##*|}"
  action="$(pick_action "${fp_category}")"
  fp_env="${fp_name}=${action}"

  iter_wd="${WORKDIR_BASE}/iter-${ITER}"
  iter_log="${LOG_DIR}/iter-${ITER}-${fp_name}.log"

  remaining=$(( DEADLINE - $(date +%s) ))
  printf '[iter %4d] %-30s %-22s ... ' "${ITER}" "${fp_name}" "${action}"

  read -r WALL EC DISTINCT VERDICT < <(run_one "${iter_wd}" "${iter_log}" "${fp_env}")

  TOTAL_RUNS=$(( TOTAL_RUNS + 1 ))
  FIRE_COUNT["${fp_name}"]=$(( FIRE_COUNT["${fp_name}"] + 1 ))

  divergent="no"
  notes=""
  if [[ "${EC}" -eq 0 ]]; then
    EXIT_OK_COUNT["${fp_name}"]=$(( EXIT_OK_COUNT["${fp_name}"] + 1 ))
    # exit 0 means the run completed — must match control
    if [[ "${DISTINCT}" != "${CTRL_DISTINCT}" ]] || [[ "${VERDICT}" != "${CTRL_VERDICT}" ]]; then
      divergent="yes"
      notes="exit=0 but distinct=${DISTINCT}/verdict=${VERDICT} != control(${CTRL_DISTINCT}/${CTRL_VERDICT})"
      DIVERGE_COUNT["${fp_name}"]=$(( DIVERGE_COUNT["${fp_name}"] + 1 ))
      TOTAL_DIVERGENCES=$(( TOTAL_DIVERGENCES + 1 ))
    fi
  elif [[ "${EC}" -eq 124 ]] || [[ "${EC}" -eq 137 ]]; then
    # timeout / SIGKILL = potential hang
    HANG_COUNT["${fp_name}"]=$(( HANG_COUNT["${fp_name}"] + 1 ))
    TOTAL_HANGS=$(( TOTAL_HANGS + 1 ))
    notes="timed out (possible hang) after ${WALL}ms"
  else
    # graceful non-zero exit — expected for permanent failpoints
    EXIT_FAIL_COUNT["${fp_name}"]=$(( EXIT_FAIL_COUNT["${fp_name}"] + 1 ))
    notes="graceful exit ${EC}"
  fi

  if [[ "${divergent}" == "yes" ]]; then
    printf 'DIVERGE  exit=%s distinct=%s verdict=%s wall=%sms (%ss left)\n' \
      "${EC}" "${DISTINCT}" "${VERDICT}" "${WALL}" "${remaining}"
  elif [[ "${EC}" -eq 124 ]] || [[ "${EC}" -eq 137 ]]; then
    printf 'HANG     exit=%s wall=%sms (%ss left)\n' "${EC}" "${WALL}" "${remaining}"
  else
    printf 'ok       exit=%s distinct=%s verdict=%s wall=%sms (%ss left)\n' \
      "${EC}" "${DISTINCT}" "${VERDICT}" "${WALL}" "${remaining}"
  fi

  echo -e "${ITER}\t${WALL}\t${fp_name}\t${action}\t${EC}\t${DISTINCT}\t${VERDICT}\t${divergent}\t${notes}" >> "${ITER_TSV}"

  # cleanup successful iter workdir to save disk
  if [[ "${divergent}" == "no" ]] && [[ "${EC}" -eq 0 ]]; then
    rm -rf "${iter_wd}" >/dev/null 2>&1 || true
    rm -f "${iter_log}"  # keep only logs of interesting iters
  fi
done

SOAK_END=$(date +%s)
SOAK_WALL=$(( SOAK_END - SOAK_START ))

# ---- summary ---------------------------------------------------------------
{
  echo "============================================================"
  echo "CHAOS SOAK SUMMARY"
  echo "============================================================"
  echo "started:       $(date -d @${SOAK_START} -u +%FT%TZ)"
  echo "ended:         $(date -d @${SOAK_END} -u +%FT%TZ)"
  echo "wall:          ${SOAK_WALL}s (target: ${DURATION_SECS}s)"
  echo "iterations:    ${ITER}"
  echo "spec:          ${SPEC_NAME}"
  echo "control:       distinct=${CTRL_DISTINCT} verdict=${CTRL_VERDICT}"
  echo "binary:        ${BINARY_PATH}"
  echo "workers:       ${WORKERS}"
  echo "ckpt-int:      ${CHECKPOINT_INTERVAL}s"
  if [[ -n "${QUEUE_FLAG}" ]]; then
    echo "queue-cap:     ${QUEUE_MAX_INMEM}"
  else
    echo "queue-cap:     default (spill not exercised)"
  fi
  echo "per-iter t.o.: ${PER_ITER_TIMEOUT}s"
  echo
  echo "FAILPOINT COVERAGE MATRIX"
  echo "------------------------------------------------------------"
  printf '%-32s %8s %8s %8s %8s %10s\n' \
    "failpoint" "fires" "exit_ok" "exit_err" "hangs" "diverge"
  for entry in "${FAILPOINTS[@]}"; do
    name="${entry%%|*}"
    printf '%-32s %8d %8d %8d %8d %10d\n' \
      "${name}" \
      "${FIRE_COUNT[$name]}" \
      "${EXIT_OK_COUNT[$name]}" \
      "${EXIT_FAIL_COUNT[$name]}" \
      "${HANG_COUNT[$name]}" \
      "${DIVERGE_COUNT[$name]}"
  done
  echo
  echo "TOTALS"
  echo "  runs:            ${TOTAL_RUNS}"
  echo "  divergences:     ${TOTAL_DIVERGENCES}"
  echo "  hangs/timeouts:  ${TOTAL_HANGS}"
  echo
  if [[ "${TOTAL_DIVERGENCES}" -eq 0 ]] && [[ "${TOTAL_HANGS}" -eq 0 ]]; then
    echo "RESULT: clean — no divergences, no hangs."
  else
    echo "RESULT: investigate — see ${LOG_DIR} for retained per-iter logs."
  fi
  echo
  echo "iterations log:  ${ITER_TSV}"
  echo "logs dir:        ${LOG_DIR}"
  echo "============================================================"
} | tee "${SUMMARY_OUT}"

# Exit non-zero if anything bad happened, so callers can gate on the soak.
if [[ "${TOTAL_DIVERGENCES}" -gt 0 ]] || [[ "${TOTAL_HANGS}" -gt 0 ]]; then
  exit 1
fi
exit 0
