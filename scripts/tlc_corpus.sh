#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_PATH="${1:-${ROOT_DIR}/corpus/index.tsv}"
OUT_ROOT="${2:-${ROOT_DIR}/.tlc-out/corpus}"
SUMMARY_PATH="${3:-${OUT_ROOT}/summary.tsv}"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "missing corpus index: ${INDEX_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

printf "id\tstatus\texit_code\tduration_sec\tstates_generated\tstates_distinct\tdepth\tmodule\tcfg\ttags\tlog\n" > "${SUMMARY_PATH}"

total=0
failures=0

while IFS=$'\t' read -r id module cfg tags _; do
  [[ -z "${id}" ]] && continue
  [[ "${id}" == \#* ]] && continue

  total=$((total + 1))

  module_path="${ROOT_DIR}/${module}"
  cfg_path="${ROOT_DIR}/${cfg}"
  run_dir="${OUT_ROOT}/${id}"
  log_path="${run_dir}/tlc.log"

  mkdir -p "${run_dir}"

  start_epoch="$(date +%s)"
  set +e
  "${ROOT_DIR}/scripts/tlc_check.sh" "${module_path}" "${cfg_path}" "${run_dir}" >"${log_path}" 2>&1
  exit_code=$?
  set -e
  end_epoch="$(date +%s)"

  duration_sec=$((end_epoch - start_epoch))
  status="ok"
  if [[ ${exit_code} -ne 0 ]]; then
    status="fail"
    failures=$((failures + 1))
  fi

  states_generated="$(sed -n 's/^\([0-9][0-9]*\) states generated, .*$/\1/p' "${log_path}" | tail -n 1)"
  states_distinct="$(sed -n 's/^[0-9][0-9]* states generated, \([0-9][0-9]*\) distinct states found, .*$/\1/p' "${log_path}" | tail -n 1)"
  depth="$(sed -n 's/^The depth of the complete state graph search is \([0-9][0-9]*\)\.$/\1/p' "${log_path}" | tail -n 1)"

  [[ -z "${states_generated}" ]] && states_generated="na"
  [[ -z "${states_distinct}" ]] && states_distinct="na"
  [[ -z "${depth}" ]] && depth="na"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${status}" "${exit_code}" "${duration_sec}" \
    "${states_generated}" "${states_distinct}" "${depth}" \
    "${module}" "${cfg}" "${tags}" "${log_path}" >> "${SUMMARY_PATH}"

  printf "[%s] status=%s generated=%s distinct=%s depth=%s log=%s\n" \
    "${id}" "${status}" "${states_generated}" "${states_distinct}" "${depth}" "${log_path}"
done < "${INDEX_PATH}"

printf "\nCorpus TLC run complete: total=%d failures=%d summary=%s\n" "${total}" "${failures}" "${SUMMARY_PATH}"

if [[ ${failures} -ne 0 ]]; then
  exit 1
fi
