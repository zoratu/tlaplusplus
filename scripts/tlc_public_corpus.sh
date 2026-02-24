#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_LIST="${1:-${ROOT_DIR}/corpus/public/public_corpus.tsv}"
LOCK_PATH="${2:-${ROOT_DIR}/corpus/public/public_corpus.lock.tsv}"
CACHE_ROOT="${3:-${ROOT_DIR}/.corpus-cache/public}"
OUT_ROOT="${4:-${ROOT_DIR}/.tlc-out/public-corpus}"
SUMMARY_PATH="${OUT_ROOT}/summary.tsv"

if [[ ! -f "${PUBLIC_LIST}" ]]; then
  echo "missing public corpus list: ${PUBLIC_LIST}" >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}" "${OUT_ROOT}"

"${ROOT_DIR}/scripts/refresh_public_corpus_shas.sh" "${PUBLIC_LIST}" "${LOCK_PATH}" >/dev/null

printf "id\tstatus\texit_code\tduration_sec\tstates_generated\tstates_distinct\tdepth\trepo_url\tcommit_sha\tmodule_path\tcfg_path\ttags\tlog\n" > "${SUMMARY_PATH}"

total=0
failures=0

while IFS=$'\t' read -r id repo_url commit_sha module_path cfg_path license tags _; do
  [[ -z "${id}" ]] && continue
  [[ "${id}" == \#* ]] && continue

  total=$((total + 1))

  repo_dir="${CACHE_ROOT}/${id}/repo"
  run_dir="${OUT_ROOT}/${id}"
  log_path="${run_dir}/tlc.log"

  mkdir -p "${run_dir}" "$(dirname "${repo_dir}")"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${repo_url}" "${repo_dir}" >/dev/null 2>&1
  fi

  git -C "${repo_dir}" fetch --depth 1 origin "${commit_sha}" >/dev/null 2>&1
  git -C "${repo_dir}" checkout --detach "${commit_sha}" >/dev/null 2>&1

  module_abs="${repo_dir}/${module_path}"
  cfg_abs="${repo_dir}/${cfg_path}"

  start_epoch="$(date +%s)"
  set +e
  "${ROOT_DIR}/scripts/tlc_check.sh" "${module_abs}" "${cfg_abs}" "${run_dir}" >"${log_path}" 2>&1
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

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${status}" "${exit_code}" "${duration_sec}" \
    "${states_generated}" "${states_distinct}" "${depth}" \
    "${repo_url}" "${commit_sha}" "${module_path}" "${cfg_path}" "${tags}" "${log_path}" \
    >> "${SUMMARY_PATH}"

  printf "[%s] status=%s generated=%s distinct=%s depth=%s\n" \
    "${id}" "${status}" "${states_generated}" "${states_distinct}" "${depth}"
done < "${LOCK_PATH}"

printf "\nPublic corpus run complete: total=%d failures=%d summary=%s\n" "${total}" "${failures}" "${SUMMARY_PATH}"

if [[ ${failures} -ne 0 ]]; then
  exit 1
fi
