#!/usr/bin/env bash
#
# chaff_matrix.sh — measure the *sensitivity* of our test suite (Examplar-style).
#
# A "chaff" is an intentional, feature-gated semantic regression (see
# `src/chaff.rs` and `chaffs.toml`) sourced from a bug we have actually shipped.
# For each chaff this script:
#
#   1. activates it via TLAPP_CHAFF=<id> against a `--features chaff` build, and
#   2. runs the gate that is *supposed* to catch it, and
#   3. asserts that gate FAILS (exits non-zero).
#
# A chaff the gate does NOT catch is a proven hole in the suite — this script
# exits non-zero and names it. Conversely a "control" pass (no chaff active)
# must PASS, confirming the gate is not simply always-failing.
#
# This is the other half of differential testing: diff_tlc tells us "this build
# agrees with TLC"; the chaff matrix tells us "a gate would fail if bug class X
# came back." See docs/chaff-testing.md.
#
# Exit codes:
#   0 — every wired chaff was caught by its gate, and the control passed
#   1 — at least one wired chaff slipped through its gate (a hole), or control failed
#   2 — usage / configuration error (missing binary, missing build, etc.)
#
# Environment variables:
#   TLAPLUSPLUS_BIN  — chaff-enabled binary (default: ./target/release/tlaplusplus).
#                      MUST be built with `--features chaff`, or every chaff is a
#                      no-op and the matrix (correctly) reports them all uncaught.
#   CHAFF_LEDGER     — path to the ledger (default: chaffs.toml)
#   DIFF_LIST        — diff_tlc spec list (default: corpus/diff_test/list.tsv)
#   CHAFF_ONLY       — run only this chaff id (default: all)
#   CHAFF_INCLUDE_TODO — if "1", attempt to run chaffs marked status=todo too
#                        (expected to be reported uncaught; useful while wiring)
#   TLA2TOOLS_JAR / JAVA_BIN — forwarded to diff_tlc.sh for diff_tlc-gated chaffs
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHAFF_LEDGER="${CHAFF_LEDGER:-${ROOT_DIR}/chaffs.toml}"
DIFF_LIST="${DIFF_LIST:-${ROOT_DIR}/corpus/diff_test/list.tsv}"
CHAFF_ONLY="${CHAFF_ONLY:-}"
CHAFF_INCLUDE_TODO="${CHAFF_INCLUDE_TODO:-0}"

# Resolve the binary.
if [[ -n "${TLAPLUSPLUS_BIN:-}" ]]; then
  BIN="${TLAPLUSPLUS_BIN}"
elif [[ -x "${ROOT_DIR}/target/release/tlaplusplus" ]]; then
  BIN="${ROOT_DIR}/target/release/tlaplusplus"
elif [[ -x "${ROOT_DIR}/target/debug/tlaplusplus" ]]; then
  BIN="${ROOT_DIR}/target/debug/tlaplusplus"
else
  echo "error: no tlaplusplus binary found; build with:" >&2
  echo "   cargo build --release --features chaff" >&2
  exit 2
fi

if [[ ! -f "${CHAFF_LEDGER}" ]]; then
  echo "error: chaff ledger not found: ${CHAFF_LEDGER}" >&2
  exit 2
fi

echo "chaff_matrix: using binary ${BIN}"
echo "chaff_matrix: ledger       ${CHAFF_LEDGER}"
echo "chaff_matrix: NOTE — the binary MUST be built with --features chaff or all"
echo "              chaffs will (correctly) report as uncaught no-ops."
echo

# ---- minimal TOML array-of-tables parser (awk) ------------------------------
# Emits one TSV line per [[chaff]]: id<TAB>spec<TAB>gate<TAB>status
parse_ledger() {
  awk '
    /^\[\[chaff\]\]/ { if (id != "") print id "\t" spec "\t" gate "\t" status;
                       id=""; spec=""; gate="diff_tlc"; status="todo"; next }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*id[[:space:]]*=/      { status_line($0, "id");     id=val }
    /^[[:space:]]*spec[[:space:]]*=/    { status_line($0, "spec");   spec=val }
    /^[[:space:]]*gate[[:space:]]*=/    { status_line($0, "gate");   gate=val }
    /^[[:space:]]*status[[:space:]]*=/  { status_line($0, "status"); status=val }
    END { if (id != "") print id "\t" spec "\t" gate "\t" status }
    function status_line(line, key,   s) {
      s = line
      sub(/^[^=]*=[[:space:]]*/, "", s)   # strip up to and including "="
      gsub(/^"/, "", s); gsub(/".*$/, "", s)  # strip surrounding quotes
      val = s
    }
  ' "${CHAFF_LEDGER}"
}

# ---- gate runners -----------------------------------------------------------
# Each returns 0 if the gate PASSED (no divergence) and non-zero if it FAILED
# (caught a divergence). The matrix interprets a chaff as "caught" when its gate
# returns non-zero.

run_gate_diff_tlc() {
  local spec="$1"
  local row tmp_list
  row="$(grep -P "^${spec}\t" "${DIFF_LIST}" || true)"
  if [[ -z "${row}" ]]; then
    echo "    gate=diff_tlc: spec '${spec}' not found in ${DIFF_LIST}" >&2
    return 125  # configuration error, distinct from pass/fail
  fi
  tmp_list="$(mktemp)"
  printf '%s\n' "${row}" > "${tmp_list}"
  DIFF_LIST="${tmp_list}" TLAPLUSPLUS_BIN="${BIN}" \
    bash "${ROOT_DIR}/scripts/diff_tlc.sh" >/dev/null 2>&1
  local rc=$?
  rm -f "${tmp_list}"
  return ${rc}
}

run_gate_cargo_test() {
  # Runs an integration test crate under --features chaff with TLAPP_CHAFF set.
  local test_name="$1"
  cargo test --release --features chaff --test "${test_name}" >/dev/null 2>&1
  return $?
}

dispatch_gate() {
  local gate="$1" spec="$2"
  case "${gate}" in
    diff_tlc)          run_gate_diff_tlc "${spec}" ;;
    snapshot)          run_gate_cargo_test state_graph_snapshots ;;
    proptest)          run_gate_cargo_test compiled_vs_interpreted ;;
    reached_assertion) run_gate_cargo_test reached_assertion_violation ;;
    *)                 echo "    unknown gate '${gate}'" >&2; return 125 ;;
  esac
}

# ---- control: no chaff must pass -------------------------------------------
echo "== control (no chaff active) =="
control_fail=0
while IFS=$'\t' read -r id spec gate status; do
  [[ "${status}" == "wired" ]] || continue
  [[ "${gate}" == "diff_tlc" ]] || continue  # control only meaningful for diff_tlc specs
  if TLAPP_CHAFF="" dispatch_gate "${gate}" "${spec}"; then
    :
  else
    echo "  CONTROL FAIL: gate '${gate}' on spec '${spec}' failed with NO chaff active" >&2
    control_fail=1
  fi
done < <(parse_ledger)
if [[ "${control_fail}" == "1" ]]; then
  echo "control failed — gate is failing without any chaff; aborting." >&2
  exit 1
fi
echo "  control OK"
echo

# ---- chaff sweep ------------------------------------------------------------
echo "== chaff sweep =="
total=0; caught=0; uncaught=0; skipped=0
declare -a HOLES=()
while IFS=$'\t' read -r id spec gate status; do
  [[ -n "${id}" ]] || continue
  if [[ -n "${CHAFF_ONLY}" && "${id}" != "${CHAFF_ONLY}" ]]; then continue; fi
  if [[ "${status}" != "wired" && "${CHAFF_INCLUDE_TODO}" != "1" ]]; then
    printf '  [skip] %-34s (status=%s)\n' "${id}" "${status}"
    skipped=$((skipped+1))
    continue
  fi
  total=$((total+1))
  TLAPP_CHAFF="${id}" dispatch_gate "${gate}" "${spec}"
  rc=$?
  if [[ ${rc} -eq 125 ]]; then
    printf '  [ERR ] %-34s gate=%-10s (configuration error)\n' "${id}" "${gate}"
    HOLES+=("${id} (config error)")
    uncaught=$((uncaught+1))
  elif [[ ${rc} -ne 0 ]]; then
    printf '  [OK  ] %-34s gate=%-10s CAUGHT (gate failed as expected)\n' "${id}" "${gate}"
    caught=$((caught+1))
  else
    printf '  [HOLE] %-34s gate=%-10s UNCAUGHT — gate passed with chaff active!\n' "${id}" "${gate}"
    HOLES+=("${id} via ${gate} on ${spec}")
    uncaught=$((uncaught+1))
  fi
done < <(parse_ledger)

echo
echo "== summary =="
echo "  wired chaffs run : ${total}"
echo "  caught           : ${caught}"
echo "  UNCAUGHT (holes) : ${uncaught}"
echo "  skipped (todo)   : ${skipped}"
if [[ ${uncaught} -gt 0 ]]; then
  echo
  echo "SUITE HOLES — these chaffs slipped through their gate:" >&2
  for h in "${HOLES[@]}"; do echo "  - ${h}" >&2; done
  exit 1
fi
if [[ ${total} -eq 0 ]]; then
  echo "  (no wired chaffs ran — nothing asserted)"
fi
exit 0
