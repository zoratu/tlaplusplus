#!/usr/bin/env bash
#
# scan_underexploration.sh — under-exploration / false-safe detector for the
# expr-v2 layout-aware parser (and any future default-parser change).
#
# For every `<spec>.tla` + `<spec>.cfg` pair under the given corpus root(s), run
# tlaplusplus model checking TWICE:
#   * once with the DEFAULT parser (v2), and
#   * once with `TLAPLUS_EXPR_PARSER=v1` (the trusted string parser).
# Then FLAG every spec where the v2 distinct-state count is materially BELOW the
# v1 count (v2 < RATIO * v1, default 0.90) while BOTH runs completed (neither
# timed out or errored). That gap — fewer states explored under the new default,
# same or "safe" verdict — is exactly the false-safe under-exploration class the
# verdict-only corpus gates miss (see the EWD998PCal `node` regression, where the
# v2 default under-explored 321,370 -> ~11k distinct with an unchanged
# "noviolation" verdict).
#
# This needs NO TLC: v1 is the ground-truth oracle for "did the new default lose
# states". Exit non-zero if any spec is flagged, so it can gate CI.
#
# Usage:
#   scripts/scan_underexploration.sh [corpus_dir ...]
#
# Env:
#   TLAPLUSPLUS_BIN   path to the binary (default: target/release/tlaplusplus)
#   SCAN_RATIO        under-exploration threshold (default 0.90)
#   SCAN_TIMEOUT      per-run timeout seconds (default 150)
#   SCAN_WORKERS      workers per run (default 3)
#   SCAN_PAR          parallel specs (default: nproc/4, min 1)
#   SCAN_OUTDIR       per-spec result dir (default: .scan-underexplore)
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN="${TLAPLUSPLUS_BIN:-${ROOT_DIR}/target/release/tlaplusplus}"
RATIO="${SCAN_RATIO:-0.90}"
TMO="${SCAN_TIMEOUT:-150}"
WK="${SCAN_WORKERS:-3}"
PAR="${SCAN_PAR:-$(( $(nproc 2>/dev/null || echo 4) / 4 ))}"; [ "${PAR}" -lt 1 ] && PAR=1
OUTDIR="${SCAN_OUTDIR:-${ROOT_DIR}/.scan-underexplore}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: tlaplusplus binary not found/executable: ${BIN}" >&2
  exit 2
fi

ROOTS=("$@")
[ ${#ROOTS[@]} -eq 0 ] && ROOTS=("${ROOT_DIR}/corpus" "${ROOT_DIR}/specifications")

rm -rf "${OUTDIR}" /tmp/scan-underexplore-work.$$ 2>/dev/null
mkdir -p "${OUTDIR}"
WORK="/tmp/scan-underexplore-work.$$"
mkdir -p "${WORK}"

CFGS="${WORK}/cfgs.txt"
: > "${CFGS}"
for r in "${ROOTS[@]}"; do
  [ -d "${r}" ] && find "${r}" -name '*.cfg' 2>/dev/null >> "${CFGS}"
done
sort -u -o "${CFGS}" "${CFGS}"
echo "scan_underexploration: $(wc -l < "${CFGS}") cfg(s), ratio=${RATIO}, timeout=${TMO}s, par=${PAR}"

scan_one() {
  local cfg="$1"
  local dir base tla id of wd
  dir="$(dirname "${cfg}")"; base="$(basename "${cfg}" .cfg)"; tla="${dir}/${base}.tla"
  [ -f "${tla}" ] || return 0
  id="$(basename "${dir}")__${base}"
  of="${SU_OUT}/${id}.res"
  [ -f "${of}" ] && return 0
  wd="${SU_WORK}/${id}.$$"

  _run() {
    local parser="$1" log rc cnt verd
    local -a pre=()
    [ "${parser}" = "v1" ] && pre=(env TLAPLUS_EXPR_PARSER=v1)
    log="$(cd "${dir}" && "${pre[@]}" timeout "${SU_TMO}" "${SU_BIN}" run-tla \
        --module "${tla}" --config "${cfg}" --workers "${SU_WK}" \
        --numa-pinning false --auto-tune false --work-dir "${wd}/${parser}" \
        --allow-deadlock 2>&1)"
    rc=$?
    cnt="$(printf '%s\n' "${log}" | grep -oE '[0-9,]+ distinct states found' | tr -d , | grep -oE '^[0-9]+' | tail -1)"
    [ -z "${cnt}" ] && cnt=-1
    if [ ${rc} -eq 124 ]; then verd=timeout
    elif printf '%s\n' "${log}" | grep -q 'violation=true'; then verd=viol
    elif printf '%s\n' "${log}" | grep -q 'violation=false'; then verd=noviol
    else verd=err; fi
    echo "${cnt} ${verd}"
  }

  local c2 v2 c1 v1 flag
  read -r c2 v2 <<<"$(_run v2)"
  read -r c1 v1 <<<"$(_run v1)"
  flag=ok
  # MISSED-VIOLATION: v1 (trusted oracle) found a violation the v2 default did not.
  # That is an unambiguous soundness regression, independent of state counts, so
  # flag it whenever both runs actually completed a verdict.
  if [ "${v1}" = viol ] && [ "${v2}" = noviol ]; then
    flag=MISSED_VIOLATION
  # COUNT-DROP false-safe: v2 explored materially fewer states than v1 while still
  # reporting NO violation. A low count paired with a violation is NOT a false
  # positive — violation runs legitimately stop early — so only flag when v2 is
  # a completed `noviol`. v1 must likewise have completed a verdict (not
  # timeout/err) to be a trustworthy oracle.
  elif [ "${v2}" = noviol ] \
     && [ "${v1}" != timeout ] && [ "${v1}" != err ] \
     && [ "${c1}" -gt 0 ] 2>/dev/null && [ "${c2}" -ge 0 ] 2>/dev/null; then
    awk "BEGIN{exit !(${c2} < ${SU_RATIO} * ${c1})}" && flag=UNDEREXPLORE
  fi
  printf '%s\tv2=%s\tv1=%s\tv2v=%s\tv1v=%s\t%s\n' "${id}" "${c2}" "${c1}" "${v2}" "${v1}" "${flag}" > "${of}"
  rm -rf "${wd}"
}
export -f scan_one
export SU_OUT="${OUTDIR}" SU_WORK="${WORK}" SU_BIN="${BIN}" SU_TMO="${TMO}" SU_WK="${WK}" SU_RATIO="${RATIO}"

xargs -P "${PAR}" -I{} bash -c 'scan_one "$1"' _ {} < "${CFGS}"

ALL="${OUTDIR}/all.tsv"
cat "${OUTDIR}"/*.res 2>/dev/null | sort > "${ALL}"
UNDER=$(grep -c 'UNDEREXPLORE' "${ALL}" 2>/dev/null); UNDER=${UNDER:-0}
MISSED=$(grep -c 'MISSED_VIOLATION' "${ALL}" 2>/dev/null); MISSED=${MISSED:-0}
FLAGGED=$(( UNDER + MISSED ))
echo "============================================================"
echo "scan_underexploration summary: $(wc -l < "${ALL}") spec(s) scanned"
echo "  under-exploration flagged: ${UNDER}"
echo "  missed-violation flagged:  ${MISSED}"
echo "============================================================"
if [ "${FLAGGED}" -gt 0 ]; then
  if [ "${MISSED}" -gt 0 ]; then
    echo "MISSED VIOLATION (v1 found a violation the v2 default did not):"
    grep 'MISSED_VIOLATION' "${ALL}"
  fi
  if [ "${UNDER}" -gt 0 ]; then
    echo "UNDER-EXPLORATION (v2 default explored < ${RATIO}x the v1 distinct count, verdict noviolation):"
    grep 'UNDEREXPLORE' "${ALL}"
  fi
  rm -rf "${WORK}"
  exit 1
fi
rm -rf "${WORK}"
exit 0
