#!/usr/bin/env bash
#
# T11: Long-running chaos soak with random failpoint injection.
# T16b: Swarm-mode chaos soak — multiple concurrent failpoints per iteration.
#
# Strategy:
# 1. Run a control model-check (no failpoints) to establish the canonical
#    distinct-state count and invariant verdict.
# 2. Loop for --duration seconds. In each iteration:
#    - Pick N distinct failpoints from the active set (N = --swarm-mode;
#      default 1 = T11 backward-compatible behaviour, "auto" = random 1-4
#      per iter, integer = exact N).
#    - For each, pick an action: most are transient ("1*return->off" or
#      "2*return->off"), so the system has a chance to complete; a small
#      fraction are permanent ("return") to also exercise graceful-error
#      termination.
#    - Run the spec under FAILPOINTS=name1=action1;name2=action2;... — the
#      `fail` crate parses `;` as a config delimiter (verified in
#      .cargo/registry/src/.../fail-0.5.1/src/lib.rs setup() at line ~568).
#    - Compare distinct-state count and verdict against the control.
#      Transient injections must complete identically. Permanent injections
#      may either complete identically (the failpoint never reaches its
#      trigger condition) or terminate with a graceful non-zero exit
#      within the per-iteration timeout — both are acceptable, only hangs
#      and panics are flagged.
# 3. Per iteration, log: failpoints (joined by ","), actions, exit code,
#    wall-time, distinct count, verdict, divergence flag.
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
#                         [--swarm-mode N|auto]
#
# Defaults target CheckpointDrain (~26K distinct), 1 hour soak,
# checkpoint-interval=2s + small queue-max-inmem to drive the spill path.
# Default --swarm-mode 1 preserves the T11 single-failpoint behaviour;
# --swarm-mode auto enables T16b swarm injection (random 1-4 per iter).

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
# T16b: --swarm-mode controls how many concurrent failpoints fire per
# iteration. "1" (default) reproduces T11's single-failpoint baseline;
# any positive integer N pins concurrent count = N (clamped to catalog
# size); "auto" picks a random N in [1, SWARM_MAX] each iter.
SWARM_MODE=1
SWARM_MAX=4

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
    --swarm-mode) SWARM_MODE="$2"; shift 2 ;;
    --swarm-max) SWARM_MAX="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- swarm-mode validation -------------------------------------------------
if [[ "${SWARM_MODE}" != "auto" ]] && ! [[ "${SWARM_MODE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --swarm-mode must be a positive integer or 'auto', got: ${SWARM_MODE}" >&2
  exit 2
fi
if [[ "${SWARM_MODE}" =~ ^[0-9]+$ ]] && [[ "${SWARM_MODE}" -lt 1 ]]; then
  echo "ERROR: --swarm-mode integer must be >= 1, got: ${SWARM_MODE}" >&2
  exit 2
fi
if ! [[ "${SWARM_MAX}" =~ ^[0-9]+$ ]] || [[ "${SWARM_MAX}" -lt 1 ]]; then
  echo "ERROR: --swarm-max must be a positive integer, got: ${SWARM_MAX}" >&2
  exit 2
fi

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
# T16b: `failpoint` and `action` columns now contain comma-joined lists when
# --swarm-mode > 1 fires multiple failpoints concurrently. `swarm_n` records
# how many concurrent failpoints fired this iteration.
echo -e "iter\twall_ms\tswarm_n\tfailpoint\taction\texit\tdistinct\tverdict\tdivergent\tnotes" > "${ITER_TSV}"

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
# T16b: per-(N) swarm-size histogram + per-pair coverage matrix to show
# which combinations of failpoints have fired together.
declare -A SWARM_N_COUNT
# Pair key = "<name1>+<name2>" (lex-sorted) so {a,b} and {b,a} collapse.
declare -A PAIR_FIRE_COUNT
for entry in "${FAILPOINTS[@]}"; do
  name="${entry%%|*}"
  FIRE_COUNT["$name"]=0
  DIVERGE_COUNT["$name"]=0
  HANG_COUNT["$name"]=0
  EXIT_OK_COUNT["$name"]=0
  EXIT_FAIL_COUNT["$name"]=0
done

# ---- helper: pick K distinct failpoints from FAILPOINTS -------------------
# Echoes K entries (one per line, "name|category" form). K is clamped to
# the catalog size.
pick_k_failpoints() {
  local k="$1"
  local total="${#FAILPOINTS[@]}"
  if [[ "${k}" -gt "${total}" ]]; then
    k="${total}"
  fi
  # Build a shuffled list of indices using awk's Fisher-Yates; portable
  # and avoids relying on `shuf`.
  local indices
  indices=$(awk -v n="${total}" -v seed="${RANDOM}${RANDOM}" '
    BEGIN {
      srand(seed);
      for (i = 0; i < n; i++) idx[i] = i;
      for (i = n - 1; i > 0; i--) {
        j = int(rand() * (i + 1));
        t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }
      for (i = 0; i < n; i++) printf "%d\n", idx[i];
    }')
  local count=0
  while IFS= read -r idx; do
    echo "${FAILPOINTS[$idx]}"
    count=$((count + 1))
    if [[ "${count}" -ge "${k}" ]]; then
      break
    fi
  done <<< "${indices}"
}

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
if [[ "${SWARM_MODE}" == "auto" ]]; then
  echo "  swarm-mode:  auto (random 1..${SWARM_MAX} concurrent failpoints/iter)"
elif [[ "${SWARM_MODE}" -gt 1 ]]; then
  echo "  swarm-mode:  ${SWARM_MODE} concurrent failpoints/iter (T16b)"
else
  echo "  swarm-mode:  1 (T11 baseline — single failpoint per iter)"
fi
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

echo -e "0\t${CTRL_WALL}\t0\tCONTROL\t-\t${CTRL_EXIT}\t${CTRL_DISTINCT}\t${CTRL_VERDICT}\tno\tbaseline" >> "${ITER_TSV}"

# ---- main soak loop --------------------------------------------------------
SOAK_START=$(date +%s)
DEADLINE=$(( SOAK_START + DURATION_SECS ))
ITER=0
TOTAL_DIVERGENCES=0
TOTAL_HANGS=0
TOTAL_RUNS=0

while (( $(date +%s) < DEADLINE )); do
  ITER=$(( ITER + 1 ))

  # T16b: decide how many concurrent failpoints fire this iteration.
  if [[ "${SWARM_MODE}" == "auto" ]]; then
    # Random N in [1, SWARM_MAX] (default 1-4). Uniform — small N is the
    # baseline behaviour and most likely to also exercise the well-tested
    # single-failpoint paths; larger N hunts for fault-cascade bugs.
    swarm_n=$(( (RANDOM % SWARM_MAX) + 1 ))
  else
    swarm_n="${SWARM_MODE}"
  fi
  if [[ "${swarm_n}" -gt "${#FAILPOINTS[@]}" ]]; then
    swarm_n="${#FAILPOINTS[@]}"
  fi
  SWARM_N_COUNT["${swarm_n}"]=$(( ${SWARM_N_COUNT["${swarm_n}"]:-0} + 1 ))

  # Pick `swarm_n` distinct failpoints; build the joined FAILPOINTS env value
  # plus comma-separated name / action lists for the TSV log.
  fp_names=()
  fp_actions=()
  fp_env_parts=()
  while IFS= read -r entry; do
    [[ -z "${entry}" ]] && continue
    name="${entry%%|*}"
    category="${entry##*|}"
    act="$(pick_action "${category}")"
    fp_names+=("${name}")
    fp_actions+=("${act}")
    fp_env_parts+=("${name}=${act}")
  done < <(pick_k_failpoints "${swarm_n}")

  # Join with `;` for FAILPOINTS (the `fail` crate's config delimiter)
  # and `,` for the TSV columns.
  fp_env=""
  for part in "${fp_env_parts[@]}"; do
    if [[ -z "${fp_env}" ]]; then
      fp_env="${part}"
    else
      fp_env="${fp_env};${part}"
    fi
  done
  fp_names_csv=""
  for n in "${fp_names[@]}"; do
    if [[ -z "${fp_names_csv}" ]]; then
      fp_names_csv="${n}"
    else
      fp_names_csv="${fp_names_csv},${n}"
    fi
  done
  fp_actions_csv=""
  for a in "${fp_actions[@]}"; do
    if [[ -z "${fp_actions_csv}" ]]; then
      fp_actions_csv="${a}"
    else
      fp_actions_csv="${fp_actions_csv},${a}"
    fi
  done

  # Per-iter log filename uses the first failpoint name (truncated for
  # filesystems with short component limits) so swarm iters stay locatable.
  primary="${fp_names[0]}"
  iter_wd="${WORKDIR_BASE}/iter-${ITER}"
  iter_log="${LOG_DIR}/iter-${ITER}-n${swarm_n}-${primary}.log"

  remaining=$(( DEADLINE - $(date +%s) ))
  if [[ "${swarm_n}" -eq 1 ]]; then
    printf '[iter %4d] n=%d %-30s %-22s ... ' "${ITER}" "${swarm_n}" \
      "${fp_names_csv}" "${fp_actions_csv}"
  else
    # Truncate the joined names so the line stays readable; the full list
    # is in the TSV.
    short_names="${fp_names_csv}"
    if [[ "${#short_names}" -gt 50 ]]; then
      short_names="${short_names:0:47}..."
    fi
    printf '[iter %4d] n=%d %-50s ... ' "${ITER}" "${swarm_n}" "${short_names}"
  fi

  read -r WALL EC DISTINCT VERDICT < <(run_one "${iter_wd}" "${iter_log}" "${fp_env}")

  TOTAL_RUNS=$(( TOTAL_RUNS + 1 ))
  # Fire counts are incremented per-name (each name in the swarm counts
  # once per iter where it appears) — matches T11 semantics for swarm_n=1.
  for n in "${fp_names[@]}"; do
    FIRE_COUNT["${n}"]=$(( FIRE_COUNT["${n}"] + 1 ))
  done
  # Pair coverage: every unordered pair of names in this swarm.
  if [[ "${#fp_names[@]}" -ge 2 ]]; then
    for ((pi=0; pi<${#fp_names[@]}; pi++)); do
      for ((pj=pi+1; pj<${#fp_names[@]}; pj++)); do
        a="${fp_names[$pi]}"; b="${fp_names[$pj]}"
        if [[ "$a" < "$b" ]]; then key="${a}+${b}"; else key="${b}+${a}"; fi
        PAIR_FIRE_COUNT["${key}"]=$(( ${PAIR_FIRE_COUNT["${key}"]:-0} + 1 ))
      done
    done
  fi

  divergent="no"
  notes=""
  if [[ "${EC}" -eq 0 ]]; then
    for n in "${fp_names[@]}"; do
      EXIT_OK_COUNT["${n}"]=$(( EXIT_OK_COUNT["${n}"] + 1 ))
    done
    # exit 0 means the run completed — must match control
    if [[ "${DISTINCT}" != "${CTRL_DISTINCT}" ]] || [[ "${VERDICT}" != "${CTRL_VERDICT}" ]]; then
      divergent="yes"
      notes="exit=0 but distinct=${DISTINCT}/verdict=${VERDICT} != control(${CTRL_DISTINCT}/${CTRL_VERDICT}) [n=${swarm_n} fps=${fp_names_csv}]"
      for n in "${fp_names[@]}"; do
        DIVERGE_COUNT["${n}"]=$(( DIVERGE_COUNT["${n}"] + 1 ))
      done
      TOTAL_DIVERGENCES=$(( TOTAL_DIVERGENCES + 1 ))
    fi
  elif [[ "${EC}" -eq 124 ]] || [[ "${EC}" -eq 137 ]]; then
    # timeout / SIGKILL = potential hang
    for n in "${fp_names[@]}"; do
      HANG_COUNT["${n}"]=$(( HANG_COUNT["${n}"] + 1 ))
    done
    TOTAL_HANGS=$(( TOTAL_HANGS + 1 ))
    notes="timed out (possible hang) after ${WALL}ms [n=${swarm_n} fps=${fp_names_csv}]"
  else
    # graceful non-zero exit — expected for permanent failpoints
    for n in "${fp_names[@]}"; do
      EXIT_FAIL_COUNT["${n}"]=$(( EXIT_FAIL_COUNT["${n}"] + 1 ))
    done
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

  echo -e "${ITER}\t${WALL}\t${swarm_n}\t${fp_names_csv}\t${fp_actions_csv}\t${EC}\t${DISTINCT}\t${VERDICT}\t${divergent}\t${notes}" >> "${ITER_TSV}"

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
  # T16b: swarm-N histogram + concurrent-pair coverage matrix.
  echo "SWARM SIZE HISTOGRAM (concurrent failpoints per iter)"
  echo "------------------------------------------------------------"
  printf '%-8s %10s\n' "n" "iters"
  # Print N from 1 upward in numeric order.
  swarm_keys=$(printf '%s\n' "${!SWARM_N_COUNT[@]}" | sort -n)
  for n in ${swarm_keys}; do
    printf '%-8d %10d\n' "${n}" "${SWARM_N_COUNT[$n]}"
  done
  echo
  # T16b pair matrix: only meaningful when swarm_n > 1 was sampled.
  # Under `set -u` an empty associative array's `${!PAIR_FIRE_COUNT[@]}` is
  # treated as unbound, so guard with `${PAIR_FIRE_COUNT[@]+...}` first.
  pair_total=0
  if [[ -n "${PAIR_FIRE_COUNT[*]+set}" ]]; then
    pair_total="${#PAIR_FIRE_COUNT[@]}"
  fi
  if [[ "${pair_total}" -gt 0 ]]; then
    echo "TOP CONCURRENT PAIRS (multi-failpoint coverage)"
    echo "------------------------------------------------------------"
    printf '%-60s %8s\n' "pair" "fires"
    # Sort by fire count descending; show top 20 to keep summary bounded.
    for k in "${!PAIR_FIRE_COUNT[@]}"; do
      printf '%d\t%s\n' "${PAIR_FIRE_COUNT[$k]}" "${k}"
    done | sort -rn | head -20 | while IFS=$'\t' read -r cnt pair; do
      printf '%-60s %8d\n' "${pair}" "${cnt}"
    done
    echo
    echo "  distinct concurrent pairs observed: ${pair_total}"
    echo
  fi
  echo "TOTALS"
  echo "  runs:            ${TOTAL_RUNS}"
  echo "  divergences:     ${TOTAL_DIVERGENCES}"
  echo "  hangs/timeouts:  ${TOTAL_HANGS}"
  echo "  swarm-mode:      ${SWARM_MODE}"
  if [[ "${SWARM_MODE}" == "auto" ]]; then
    echo "  swarm-max:       ${SWARM_MAX}"
  fi
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
