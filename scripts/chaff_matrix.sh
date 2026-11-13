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
cd "${ROOT_DIR}" || { echo "error: cannot cd to repo root ${ROOT_DIR}" >&2; exit 2; }

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
# Emits one TSV line per [[chaff]]: id<TAB>spec<TAB>gate<TAB>status<TAB>catch
# `catch` is the OPTIONAL expected catch-mode (divergence|infra|test). Empty when
# the entry omits it, in which case the observed mode is reported but not asserted.
parse_ledger() {
  awk '
    /^\[\[chaff\]\]/ { if (id != "") print id "\t" spec "\t" gate "\t" status "\t" catch;
                       id=""; spec=""; gate="diff_tlc"; status="todo"; catch=""; next }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*id[[:space:]]*=/      { status_line($0); id=val }
    /^[[:space:]]*spec[[:space:]]*=/    { status_line($0); spec=val }
    /^[[:space:]]*gate[[:space:]]*=/    { status_line($0); gate=val }
    /^[[:space:]]*status[[:space:]]*=/  { status_line($0); status=val }
    /^[[:space:]]*catch[[:space:]]*=/   { status_line($0); catch=val }
    END { if (id != "") print id "\t" spec "\t" gate "\t" status "\t" catch }
    function status_line(line,   s) {
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

# Where per-gate logs are retained so CI artifacts show WHY a gate failed (so we
# can tell an expected semantic divergence from an infra/timeout failure).
CHAFF_OUTPUT_DIR="${CHAFF_OUTPUT_DIR:-${ROOT_DIR}/.chaff-matrix-out}"
mkdir -p "${CHAFF_OUTPUT_DIR}"

# GATE_MODE: HOW the gate failed when a chaff is caught, so an infra failure
# (timeout / crash / no output) is never silently miscounted as a semantic catch.
# Set by the gate runners; one of: divergence | infra | test | none | unknown.
#   - divergence : diff_tlc saw a clean state-count/verdict/deadlock MISMATCH
#                  (our binary ran to a verdict that differs from TLC)
#   - infra      : diff_tlc saw "produced no state count" (our run timed out,
#                  panicked, or errored out without a verdict) OR a missing file
#   - test       : a cargo integration-test gate failed (always semantic)
#   - none       : the gate passed (no catch)
#   - unknown    : non-zero but no recognized marker (treat as a real failure)
GATE_MODE="none"

# Classify a diff_tlc run's stdout log into a GATE_MODE. The marker strings are
# emitted by scripts/diff_tlc.sh (see its spec_failures reasons).
classify_diff_tlc_log() {
  local log="$1"
  # ONLY our-side, our-vs-TLC failures count as a catch mode. Anything else
  # (a TLC-baseline failure, a missing file, a ledger-vs-TLC expectation
  # mismatch) is NOT our checker diverging, so it falls through to "unknown" and
  # is treated as a HOLE — never silently credited as a catch.
  if grep -qiE "tlaplusplus produced no state count" "${log}" 2>/dev/null; then
    # Our run reached no parseable verdict (timeout / panic / eval error).
    echo "infra"
  elif grep -qiE "state count mismatch|violation mismatch|COUNT-DROP false-safe|deadlock detection mismatch" "${log}" 2>/dev/null; then
    # Our binary ran to a verdict that cleanly differs from TLC's. (These four
    # diff_tlc markers all name both sides, e.g. "TLC=.. tlaplusplus=..".) The
    # TLC-vs-ledger markers ("expected violation but TLC", "produced no state
    # count" on TLC's side) and "MISSING MODULE/CONFIG" are deliberately NOT here:
    # they are not our checker diverging, so they must not satisfy a catch.
    echo "divergence"
  else
    echo "unknown"
  fi
}

run_gate_diff_tlc() {
  local spec="$1"
  local tmp_list nrows rc log
  log="${CHAFF_OUTPUT_DIR}/${TLAPP_CHAFF:-control}.diff_tlc.${spec}.log"
  tmp_list="$(mktemp)"
  # Literal match on the first TSV field (not a regex), and require EXACTLY one
  # row — a regex match could pull in unintended rows or break on metacharacters
  # in a future spec id, and a duplicate id would silently run several specs.
  awk -F'\t' -v s="${spec}" '$1 == s { print }' "${DIFF_LIST}" > "${tmp_list}"
  nrows="$(wc -l < "${tmp_list}" | tr -d ' ')"
  if [[ "${nrows}" -eq 0 ]]; then
    echo "    gate=diff_tlc: spec '${spec}' not found in ${DIFF_LIST}" >&2
    rm -f "${tmp_list}"
    GATE_MODE="none"
    return 125  # configuration error, distinct from pass/fail
  fi
  if [[ "${nrows}" -ne 1 ]]; then
    echo "    gate=diff_tlc: spec '${spec}' matched ${nrows} rows in ${DIFF_LIST} (expected 1)" >&2
    rm -f "${tmp_list}"
    GATE_MODE="none"
    return 125
  fi
  DIFF_LIST="${tmp_list}" TLAPLUSPLUS_BIN="${BIN}" \
    bash "${ROOT_DIR}/scripts/diff_tlc.sh" >"${log}" 2>&1
  rc=$?
  rm -f "${tmp_list}"
  if [[ ${rc} -eq 0 ]]; then
    GATE_MODE="none"
  else
    GATE_MODE="$(classify_diff_tlc_log "${log}")"
  fi
  return ${rc}
}

run_gate_cargo_test() {
  # Runs an integration test crate under --features chaff with TLAPP_CHAFF already
  # set by the caller. Debug profile (no --release): it reuses the debug lib the
  # matrix build produced, so only the test crate itself compiles.
  local test_name="$1"
  local log="${CHAFF_OUTPUT_DIR}/${TLAPP_CHAFF:-control}.cargo.${test_name}.log"
  cargo test --features chaff --test "${test_name}" >"${log}" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    GATE_MODE="none"
  elif grep -q "CHAFF_CATCH_OK" "${log}" 2>/dev/null; then
    # Failed AT the intended assertion (it carries the [CHAFF_CATCH_OK] sentinel),
    # so this is a genuine semantic catch.
    GATE_MODE="test"
  else
    # Non-zero but the intended assertion was never reached — a compile/link error,
    # a signal, a harness failure, or an unrelated panic (e.g. run_model/from_files
    # crashing before the assertion). Do NOT credit this as a catch.
    GATE_MODE="unknown"
  fi
  return ${rc}
}

dispatch_gate() {
  local gate="$1" spec="$2"
  GATE_MODE="none"
  case "${gate}" in
    diff_tlc)          run_gate_diff_tlc "${spec}" ;;
    snapshot)          run_gate_cargo_test state_graph_snapshots ;;
    proptest)          run_gate_cargo_test compiled_vs_interpreted ;;
    reached_assertion) run_gate_cargo_test reached_assertion_violation ;;
    const_schema)      run_gate_cargo_test chaff_const_schema ;;
    liveness_check)    run_gate_cargo_test chaff_liveness_check ;;
    *)                 echo "    unknown gate '${gate}'" >&2; GATE_MODE="none"; return 125 ;;
  esac
}

# ---- parse the ledger ONCE and validate ------------------------------------
# Parsing once (not per-loop via process substitution) lets us detect a parser
# failure and an empty/garbled ledger up front, instead of silently iterating
# over nothing and exiting 0 ("false green").
LEDGER_ROWS="$(parse_ledger)"
if [[ $? -ne 0 || -z "${LEDGER_ROWS}" ]]; then
  echo "error: failed to parse any chaff entries from ${CHAFF_LEDGER}" >&2
  exit 2
fi

# A typo'd CHAFF_ONLY must be a hard error, never a silent no-op that exits 0.
if [[ -n "${CHAFF_ONLY}" ]]; then
  only_matches="$(awk -F'\t' -v only="${CHAFF_ONLY}" '$1 == only { n++ } END { print n+0 }' <<<"${LEDGER_ROWS}")"
  if [[ "${only_matches}" -ne 1 ]]; then
    echo "error: CHAFF_ONLY='${CHAFF_ONLY}' matched ${only_matches} ledger entries (expected exactly 1)" >&2
    exit 2
  fi
fi

# How many wired chaffs the sweep is expected to assert on, under the active
# filter. The sweep must run exactly this many, and (for a full run) at least one.
wired_expected="$(awk -F'\t' -v only="${CHAFF_ONLY}" \
  '$4 == "wired" && (only == "" || $1 == only) { n++ } END { print n+0 }' <<<"${LEDGER_ROWS}")"

# ---- control: no chaff must pass -------------------------------------------
echo "== control (no chaff active) =="
control_fail=0
declare -A control_seen=()
while IFS=$'\t' read -r id spec gate status catch; do
  [[ "${status}" == "wired" ]] || continue
  # One control run per (gate, spec) pair — with no chaff active every gate must
  # PASS, confirming it is not simply always-failing.
  key="${gate}::${spec}"
  [[ -n "${control_seen[$key]:-}" ]] && continue
  control_seen[$key]=1
  TLAPP_CHAFF="" dispatch_gate "${gate}" "${spec}"
  crc=$?
  if [[ ${crc} -eq 0 ]]; then
    printf '  control OK: %s\n' "${key}"
  elif [[ ${crc} -eq 125 ]]; then
    echo "  CONTROL CONFIG ERROR: gate '${gate}' on spec '${spec}' is misconfigured" >&2
    control_fail=2
  else
    echo "  CONTROL FAIL: gate '${gate}' on spec '${spec}' failed with NO chaff active" >&2
    [[ "${control_fail}" == "0" ]] && control_fail=1
  fi
done <<<"${LEDGER_ROWS}"
if [[ "${control_fail}" == "2" ]]; then
  echo "control aborted — a gate is misconfigured (see .chaff-matrix-out logs)." >&2
  exit 2
elif [[ "${control_fail}" == "1" ]]; then
  echo "control failed — a gate is failing without any chaff; aborting." >&2
  exit 1
fi
echo "  control: all gates pass clean"
echo

# ---- chaff sweep ------------------------------------------------------------
echo "== chaff sweep =="
total=0; caught=0; uncaught=0; skipped=0; config_err=0; mode_mismatch=0
declare -a HOLES=()
declare -a CONFIG_ERRORS=()
declare -a MODE_MISMATCHES=()
while IFS=$'\t' read -r id spec gate status catch; do
  [[ -n "${id}" ]] || continue
  if [[ -n "${CHAFF_ONLY}" && "${id}" != "${CHAFF_ONLY}" ]]; then continue; fi
  # Only "wired" chaffs are asserted on. "open" (a tracked finding) and "todo"
  # (not yet wired) are skipped so CI stays green while the gap is documented in
  # chaffs.toml. CHAFF_INCLUDE_TODO=1 forces them to run (they are expected to
  # report UNCAUGHT — useful while investigating a re-site).
  if [[ "${status}" != "wired" && "${CHAFF_INCLUDE_TODO}" != "1" ]]; then
    printf '  [skip] %-34s (status=%s — see chaffs.toml note)\n' "${id}" "${status}"
    skipped=$((skipped+1))
    continue
  fi
  total=$((total+1))
  TLAPP_CHAFF="${id}" dispatch_gate "${gate}" "${spec}"
  rc=$?
  mode="${GATE_MODE}"
  if [[ ${rc} -eq 125 ]]; then
    # Misconfiguration (spec missing / ambiguous / unknown gate). This is NOT a
    # verdict about the chaff — it is a setup error, reported separately (exit 2).
    printf '  [ERR ] %-34s gate=%-10s (configuration error — see logs)\n' "${id}" "${gate}"
    CONFIG_ERRORS+=("${id} via ${gate} on ${spec}")
    config_err=$((config_err+1))
  elif [[ ${rc} -ne 0 ]]; then
    # CAUGHT. Report HOW it was caught (the catch mode) so an infra failure
    # (timeout/crash/no-output) is never silently credited as a semantic catch.
    # When the ledger PINS an expected `catch`, enforce it; otherwise report only.
    if [[ "${mode}" == "unknown" ]]; then
      # Non-zero but no recognized marker — do not credit this as a clean catch.
      printf '  [HOLE] %-34s gate=%-10s FAILED but mode=unknown (not a recognized catch — see logs)\n' "${id}" "${gate}"
      HOLES+=("${id} via ${gate} on ${spec} (mode=unknown)")
      uncaught=$((uncaught+1))
    elif [[ -n "${catch}" && "${catch}" != "${mode}" ]]; then
      printf '  [MODE] %-34s gate=%-10s CAUGHT but mode=%s, expected catch=%s\n' "${id}" "${gate}" "${mode}" "${catch}"
      MODE_MISMATCHES+=("${id}: observed '${mode}' but ledger pins catch='${catch}'")
      mode_mismatch=$((mode_mismatch+1))
    else
      local_pin="${catch:+ (pinned)}"
      printf '  [OK  ] %-34s gate=%-10s CAUGHT via %s%s (rc=%d)\n' "${id}" "${gate}" "${mode}" "${local_pin}" "${rc}"
      caught=$((caught+1))
    fi
  else
    printf '  [HOLE] %-34s gate=%-10s UNCAUGHT — gate passed with chaff active!\n' "${id}" "${gate}"
    HOLES+=("${id} via ${gate} on ${spec}")
    uncaught=$((uncaught+1))
  fi
done <<<"${LEDGER_ROWS}"

echo
echo "== summary =="
echo "  wired chaffs run   : ${total} (expected ${wired_expected})"
echo "  caught             : ${caught}"
echo "  UNCAUGHT (holes)   : ${uncaught}"
echo "  mode mismatches    : ${mode_mismatch}"
echo "  config errors      : ${config_err}"
echo "  skipped (open/todo): ${skipped}"
echo "  logs               : ${CHAFF_OUTPUT_DIR}"

# Config errors are a setup problem, not a verdict — surface them with exit 2
# (matches the documented 'usage/configuration error' code) before any pass.
if [[ ${config_err} -gt 0 ]]; then
  echo
  echo "CONFIGURATION ERRORS — these chaffs could not be evaluated:" >&2
  for c in "${CONFIG_ERRORS[@]}"; do echo "  - ${c}" >&2; done
  exit 2
fi

if [[ ${uncaught} -gt 0 ]]; then
  echo
  echo "SUITE HOLES — these chaffs slipped through their gate:" >&2
  for h in "${HOLES[@]}"; do echo "  - ${h}" >&2; done
  exit 1
fi

# A pinned catch-mode that does not match the observed mode is a real failure:
# either the chaff's failure character changed (e.g. a divergence became a
# timeout) or the pin is stale. Either way, don't report success.
if [[ ${mode_mismatch} -gt 0 ]]; then
  echo
  echo "CATCH-MODE MISMATCHES — observed mode differs from the pinned 'catch':" >&2
  for m in "${MODE_MISMATCHES[@]}"; do echo "  - ${m}" >&2; done
  exit 1
fi

# Guard against "false green": in a normal run the sweep must have run exactly
# the wired chaffs the filter selected, and a full run must assert on at least
# one. (CHAFF_INCLUDE_TODO=1 is an investigation mode that intentionally also
# runs non-wired entries, so the strict count guard does not apply there.)
if [[ "${CHAFF_INCLUDE_TODO}" != "1" ]]; then
  if [[ ${total} -ne ${wired_expected} ]]; then
    echo "ERROR: ran ${total} wired chaffs but expected ${wired_expected} — refusing to report success." >&2
    exit 1
  fi
  if [[ -z "${CHAFF_ONLY}" && ${total} -eq 0 ]]; then
    echo "ERROR: no wired chaffs ran in a full sweep — refusing to report success (nothing was asserted)." >&2
    exit 1
  fi
  if [[ ${total} -eq 0 ]]; then
    echo "  (CHAFF_ONLY=${CHAFF_ONLY} selected a non-wired entry — nothing asserted, by request)"
  fi
fi
exit 0
