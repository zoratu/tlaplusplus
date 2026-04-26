#!/usr/bin/env bash
#
# diff_tlc.sh — Differential testing of tlaplusplus against TLC.
#
# Reads a TSV list of (spec_id, module, config, expect_violation) entries,
# runs each spec under both checkers, and compares:
#
#   1. Violation status (must match)
#   2. Distinct state count (compared only when neither side reported a violation,
#      because both checkers stop early on first violation and that is non-deterministic)
#   3. Deadlock detection (we always pass `-deadlock` to TLC and `--allow-deadlock` to
#      tlaplusplus so the two agree on the absence-of-deadlock contract)
#
# Exit codes:
#   0 — every spec ran on both checkers and all comparisons matched
#   1 — at least one divergence (also non-zero if a checker failed to run a spec)
#   2 — usage / configuration error
#
# Allowlist: lines in `corpus/diff_test/allowlist.tsv` of the form
#   spec_id<TAB>reason
# are reported but not counted as failures. Use sparingly, document why.
#
# Environment variables:
#   TLAPLUSPLUS_BIN     — path to tlaplusplus binary (default: ./target/release/tlaplusplus
#                         or ./target/debug/tlaplusplus, whichever exists)
#   TLA2TOOLS_JAR       — path to tla2tools.jar (default: tools/tla2tools.jar)
#   JAVA_BIN            — path to java executable (default: `java` from PATH)
#   DIFF_LIST           — path to the TSV list (default: corpus/diff_test/list.tsv)
#   DIFF_ALLOWLIST      — path to allowlist (default: corpus/diff_test/allowlist.tsv)
#   DIFF_TIMEOUT_SECS   — per-spec per-checker timeout (default: 60)
#   DIFF_OUTPUT_DIR     — where to drop per-spec logs (default: .diff-tlc-out)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TLA2TOOLS_JAR="${TLA2TOOLS_JAR:-${ROOT_DIR}/tools/tla2tools.jar}"
JAVA_BIN="${JAVA_BIN:-$(command -v java || true)}"
DIFF_LIST="${DIFF_LIST:-${ROOT_DIR}/corpus/diff_test/list.tsv}"
DIFF_ALLOWLIST="${DIFF_ALLOWLIST:-${ROOT_DIR}/corpus/diff_test/allowlist.tsv}"
DIFF_TIMEOUT_SECS="${DIFF_TIMEOUT_SECS:-60}"
DIFF_OUTPUT_DIR="${DIFF_OUTPUT_DIR:-${ROOT_DIR}/.diff-tlc-out}"

if [[ -z "${TLAPLUSPLUS_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/target/release/tlaplusplus" ]]; then
    TLAPLUSPLUS_BIN="${ROOT_DIR}/target/release/tlaplusplus"
  elif [[ -x "${ROOT_DIR}/target/debug/tlaplusplus" ]]; then
    TLAPLUSPLUS_BIN="${ROOT_DIR}/target/debug/tlaplusplus"
  else
    echo "ERROR: tlaplusplus binary not found; build with 'cargo build [--release]' or set TLAPLUSPLUS_BIN" >&2
    exit 2
  fi
fi

if [[ ! -x "${TLAPLUSPLUS_BIN}" ]]; then
  echo "ERROR: tlaplusplus binary not executable: ${TLAPLUSPLUS_BIN}" >&2
  exit 2
fi

if [[ ! -f "${TLA2TOOLS_JAR}" ]]; then
  echo "ERROR: tla2tools.jar not found at ${TLA2TOOLS_JAR}" >&2
  exit 2
fi

if [[ -z "${JAVA_BIN}" || ! -x "${JAVA_BIN}" ]]; then
  echo "ERROR: java binary not found; install JDK or set JAVA_BIN" >&2
  exit 2
fi

if [[ ! -f "${DIFF_LIST}" ]]; then
  echo "ERROR: diff list not found: ${DIFF_LIST}" >&2
  exit 2
fi

mkdir -p "${DIFF_OUTPUT_DIR}"

# Decide on a `timeout` command (BSD `gtimeout` on macOS via coreutils)
TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout"
fi

# Load allowlist as a newline-separated "id<TAB>reason" string. We avoid
# associative arrays so this works on macOS' bash 3.2 as well as Linux.
ALLOWLIST_DATA=""
if [[ -f "${DIFF_ALLOWLIST}" ]]; then
  while IFS=$'\t' read -r aid areason; do
    [[ -z "${aid:-}" ]] && continue
    [[ "${aid:0:1}" == "#" ]] && continue
    ALLOWLIST_DATA+="${aid}"$'\t'"${areason:-no reason given}"$'\n'
  done < "${DIFF_ALLOWLIST}"
fi

allowlist_reason() {
  # Echoes the reason for a given spec_id, or empty if not allowlisted.
  local sid=$1
  printf '%s' "${ALLOWLIST_DATA}" | awk -F'\t' -v id="${sid}" '$1 == id { print $2; exit }'
}

run_with_timeout() {
  if [[ -n "${TIMEOUT_CMD}" ]]; then
    "${TIMEOUT_CMD}" --foreground "${DIFF_TIMEOUT_SECS}s" "$@"
  else
    "$@"
  fi
}

# Parse the TLC output. TLC prints, on success:
#   "<N> states generated, <M> distinct states found, <K> states left on queue."
# and on invariant violation also "Error: Invariant <name> is violated."
parse_tlc_output() {
  local file=$1
  local out_var_states=$2
  local out_var_distinct=$3
  local out_var_violation=$4
  local out_var_deadlock=$5

  local line states distinct
  line=$(grep -E "^[0-9,]+ states generated" "${file}" | tail -1 || true)
  if [[ -n "${line}" ]]; then
    # Strip commas
    states=$(echo "${line}" | sed -E 's/^([0-9,]+) states generated.*/\1/' | tr -d ',')
    distinct=$(echo "${line}" | sed -E 's/.* ([0-9,]+) distinct states found.*/\1/' | tr -d ',')
  else
    states=""
    distinct=""
  fi

  local violation="no" deadlock="no"
  if grep -qE "Error: (Invariant|Action property|.*violated)" "${file}"; then
    violation="yes"
  fi
  if grep -qE "Error: Deadlock reached" "${file}"; then
    deadlock="yes"
  fi
  printf -v "${out_var_states}" '%s' "${states}"
  printf -v "${out_var_distinct}" '%s' "${distinct}"
  printf -v "${out_var_violation}" '%s' "${violation}"
  printf -v "${out_var_deadlock}" '%s' "${deadlock}"
}

# Parse tlaplusplus run-tla output. The relevant lines:
#   "<N> states generated, <M> distinct states found, <K> states left on queue."
#   "violation=true (...)" / "violation=false"
parse_tlaplusplus_output() {
  local file=$1
  local out_var_states=$2
  local out_var_distinct=$3
  local out_var_violation=$4
  local out_var_deadlock=$5

  local line states distinct
  line=$(grep -E "^[0-9,]+ states generated" "${file}" | tail -1 || true)
  if [[ -n "${line}" ]]; then
    states=$(echo "${line}" | sed -E 's/^([0-9,]+) states generated.*/\1/' | tr -d ',')
    distinct=$(echo "${line}" | sed -E 's/.* ([0-9,]+) distinct states found.*/\1/' | tr -d ',')
  else
    states=""
    distinct=""
  fi

  local violation="no" deadlock="no"
  if grep -qE "^violation=true" "${file}"; then
    violation="yes"
  fi
  # tlaplusplus reports deadlock as a violation when --allow-deadlock is not set;
  # we pass --allow-deadlock so deadlock-only stops shouldn't be reported.
  # We still scan for the literal "deadlock" mention as a defense.
  if grep -qiE "deadlock detected|deadlocked" "${file}"; then
    deadlock="yes"
  fi
  printf -v "${out_var_states}" '%s' "${states}"
  printf -v "${out_var_distinct}" '%s' "${distinct}"
  printf -v "${out_var_violation}" '%s' "${violation}"
  printf -v "${out_var_deadlock}" '%s' "${deadlock}"
}

# Run TLC against a spec.
# Args: module_path config_path output_file
run_tlc() {
  local module_path=$1
  local config_path=$2
  local out_file=$3

  local module_dir module_file config_file
  module_dir="$(cd "$(dirname "${module_path}")" && pwd)"
  module_file="$(basename "${module_path}" .tla)"
  config_file="$(realpath "${config_path}")"

  # TLC needs to run from the directory containing the module so it can find
  # sibling modules (e.g. CoverageHelper.tla for InstanceTestSimple). Use a
  # tmpdir so multiple specs can run cleanly without colliding metadirs.
  local meta_dir
  meta_dir="$(mktemp -d)"
  (
    cd "${module_dir}"
    run_with_timeout "${JAVA_BIN}" -XX:+UseParallelGC -jar "${TLA2TOOLS_JAR}" \
      -cleanup \
      -workers auto \
      -deadlock \
      -metadir "${meta_dir}" \
      -config "${config_file}" \
      "${module_file}"
  ) >"${out_file}" 2>&1
  local rc=$?
  rm -rf "${meta_dir}"
  return ${rc}
}

# Run tlaplusplus against a spec.
# Args: module_path config_path output_file
run_tlaplusplus() {
  local module_path=$1
  local config_path=$2
  local out_file=$3

  run_with_timeout "${TLAPLUSPLUS_BIN}" run-tla \
    --module "${module_path}" \
    --config "${config_path}" \
    --allow-deadlock \
    --skip-system-checks \
    >"${out_file}" 2>&1
  return $?
}

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

echo "============================================================"
echo "diff_tlc.sh — tlaplusplus vs TLC differential testing"
echo "  tlaplusplus: ${TLAPLUSPLUS_BIN}"
echo "  tla2tools:   ${TLA2TOOLS_JAR}"
echo "  java:        ${JAVA_BIN}"
echo "  list:        ${DIFF_LIST}"
echo "  output:      ${DIFF_OUTPUT_DIR}"
echo "  timeout:     ${DIFF_TIMEOUT_SECS}s per checker"
echo "============================================================"
echo

TOTAL=0
PASSED=0
FAILED=0
ALLOWLISTED=0
DIVERGENCES=()

while IFS=$'\t' read -r spec_id module_path config_path expect_violation notes; do
  # Skip blank/comment lines
  [[ -z "${spec_id:-}" ]] && continue
  [[ "${spec_id:0:1}" == "#" ]] && continue

  TOTAL=$((TOTAL + 1))
  echo "------------------------------------------------------------"
  echo "[${TOTAL}] ${spec_id}"
  echo "    module: ${module_path}"
  echo "    config: ${config_path}"
  echo "    expect_violation: ${expect_violation}"
  [[ -n "${notes:-}" ]] && echo "    notes: ${notes}"

  if [[ ! -f "${ROOT_DIR}/${module_path}" ]]; then
    echo "  !! MISSING MODULE: ${ROOT_DIR}/${module_path}"
    DIVERGENCES+=("${spec_id}: missing module")
    FAILED=$((FAILED + 1))
    continue
  fi
  if [[ ! -f "${ROOT_DIR}/${config_path}" ]]; then
    echo "  !! MISSING CONFIG: ${ROOT_DIR}/${config_path}"
    DIVERGENCES+=("${spec_id}: missing config")
    FAILED=$((FAILED + 1))
    continue
  fi

  tlc_out="${DIFF_OUTPUT_DIR}/${spec_id}.tlc.log"
  tlapp_out="${DIFF_OUTPUT_DIR}/${spec_id}.tlapp.log"

  # Run TLC
  tlc_start=$(date +%s%N)
  run_tlc "${ROOT_DIR}/${module_path}" "${ROOT_DIR}/${config_path}" "${tlc_out}"
  tlc_rc=$?
  tlc_ms=$(( ($(date +%s%N) - tlc_start) / 1000000 ))

  # Run tlaplusplus
  tlapp_start=$(date +%s%N)
  run_tlaplusplus "${ROOT_DIR}/${module_path}" "${ROOT_DIR}/${config_path}" "${tlapp_out}"
  tlapp_rc=$?
  tlapp_ms=$(( ($(date +%s%N) - tlapp_start) / 1000000 ))

  # tlaplusplus exits 1 when a violation is found — that's expected for
  # expect_violation=yes, so we don't treat that exit code as a hard failure.
  # We rely on parsed output to compare.

  parse_tlc_output "${tlc_out}" tlc_states tlc_distinct tlc_violation tlc_deadlock
  parse_tlaplusplus_output "${tlapp_out}" tlapp_states tlapp_distinct tlapp_violation tlapp_deadlock

  echo "    TLC       : rc=${tlc_rc}  ${tlc_ms}ms  states=${tlc_states:-?}  distinct=${tlc_distinct:-?}  violation=${tlc_violation}  deadlock=${tlc_deadlock}"
  echo "    tlaplusplus: rc=${tlapp_rc}  ${tlapp_ms}ms  states=${tlapp_states:-?}  distinct=${tlapp_distinct:-?}  violation=${tlapp_violation}  deadlock=${tlapp_deadlock}"

  spec_failures=()

  # Did either side fail to produce parseable output?
  if [[ -z "${tlc_distinct}" && "${tlc_violation}" != "yes" ]]; then
    spec_failures+=("TLC produced no state count (rc=${tlc_rc}); see ${tlc_out}")
  fi
  if [[ -z "${tlapp_distinct}" && "${tlapp_violation}" != "yes" ]]; then
    spec_failures+=("tlaplusplus produced no state count (rc=${tlapp_rc}); see ${tlapp_out}")
  fi

  # Compare violation status
  if [[ "${tlc_violation}" != "${tlapp_violation}" ]]; then
    spec_failures+=("violation mismatch: TLC=${tlc_violation} tlaplusplus=${tlapp_violation}")
  fi

  # Compare expectation
  if [[ "${expect_violation}" == "yes" && "${tlc_violation}" != "yes" ]]; then
    spec_failures+=("expected violation but TLC reported none")
  fi
  if [[ "${expect_violation}" == "no" && "${tlc_violation}" == "yes" ]]; then
    spec_failures+=("did not expect violation but TLC reported one")
  fi

  # Compare distinct state counts only when neither side violated (both stop on first violation).
  if [[ "${tlc_violation}" == "no" && "${tlapp_violation}" == "no" ]]; then
    if [[ -n "${tlc_distinct}" && -n "${tlapp_distinct}" && "${tlc_distinct}" != "${tlapp_distinct}" ]]; then
      spec_failures+=("distinct state count mismatch: TLC=${tlc_distinct} tlaplusplus=${tlapp_distinct}")
    fi
  fi

  # Deadlock comparison (we both run with deadlock detection disabled, so this
  # should always be "no". Any "yes" indicates one checker found something the
  # other missed, which is a divergence.)
  if [[ "${tlc_deadlock}" != "${tlapp_deadlock}" ]]; then
    spec_failures+=("deadlock detection mismatch: TLC=${tlc_deadlock} tlaplusplus=${tlapp_deadlock}")
  fi

  if [[ ${#spec_failures[@]} -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    echo "    => OK"
  else
    allow_reason=$(allowlist_reason "${spec_id}")
    if [[ -n "${allow_reason}" ]]; then
      ALLOWLISTED=$((ALLOWLISTED + 1))
      echo "    => ALLOWLISTED (${allow_reason})"
      for f in "${spec_failures[@]}"; do echo "       - ${f}"; done
    else
      FAILED=$((FAILED + 1))
      echo "    => DIVERGENCE"
      for f in "${spec_failures[@]}"; do
        echo "       - ${f}"
        DIVERGENCES+=("${spec_id}: ${f}")
      done
    fi
  fi
  echo
done < "${DIFF_LIST}"

echo "============================================================"
echo "diff_tlc.sh summary"
echo "  total:        ${TOTAL}"
echo "  passed:       ${PASSED}"
echo "  failed:       ${FAILED}"
echo "  allowlisted:  ${ALLOWLISTED}"
echo "============================================================"

if [[ ${FAILED} -gt 0 ]]; then
  echo
  echo "Divergences:"
  for d in "${DIVERGENCES[@]}"; do
    echo "  - ${d}"
  done
  exit 1
fi

exit 0
