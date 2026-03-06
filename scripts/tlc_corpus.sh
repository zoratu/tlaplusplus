#!/usr/bin/env bash
set -euo pipefail

# Run TLC on all corpus specs (auto-discovered from directory structure)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${ROOT_DIR}/corpus"
OUT_ROOT="${1:-${ROOT_DIR}/.tlc-out/corpus}"
SUMMARY_PATH="${OUT_ROOT}/summary.tsv"

mkdir -p "${OUT_ROOT}"

printf "id\tstatus\texit_code\tduration_sec\tstates_generated\tstates_distinct\tdepth\tmodule\tcfg\tlog\n" > "${SUMMARY_PATH}"

total=0
failures=0

# Find all .cfg files and run the corresponding .tla
find "${CORPUS_DIR}" -name "*.cfg" -type f | sort | while read -r cfg_path; do
  cfg_dir="$(dirname "${cfg_path}")"
  cfg_name="$(basename "${cfg_path}" .cfg)"

  # Find matching .tla file
  module_path=""
  if [[ -f "${cfg_dir}/${cfg_name}.tla" ]]; then
    module_path="${cfg_dir}/${cfg_name}.tla"
  else
    # Try progressively shorter prefixes to find matching .tla
    # e.g., LanguageFeatureMatrixFair.cfg -> LanguageFeatureMatrix.tla
    # e.g., LanguageFeatureMatrix_NoEnabled.cfg -> LanguageFeatureMatrix.tla
    test_name="${cfg_name}"
    while [[ -z "${module_path}" && -n "${test_name}" ]]; do
      # Try stripping _suffix first
      if [[ "${test_name}" == *_* ]]; then
        test_name="${test_name%_*}"
        [[ -f "${cfg_dir}/${test_name}.tla" ]] && module_path="${cfg_dir}/${test_name}.tla"
      # Then try stripping CamelCase suffix (remove last uppercase word)
      elif [[ "${test_name}" =~ ^(.+[a-z])([A-Z][a-z]+)$ ]]; then
        test_name="${BASH_REMATCH[1]}"
        [[ -f "${cfg_dir}/${test_name}.tla" ]] && module_path="${cfg_dir}/${test_name}.tla"
      else
        break
      fi
    done
  fi

  if [[ -z "${module_path}" ]]; then
    echo "SKIP: No .tla found for ${cfg_path}" >&2
    continue
  fi

  # Create ID from relative path
  rel_cfg="${cfg_path#${CORPUS_DIR}/}"
  id="${rel_cfg%.cfg}"
  id="${id//\//_}"  # Replace / with _

  rel_module="${module_path#${CORPUS_DIR}/}"

  run_dir="${OUT_ROOT}/${id}"
  log_path="${run_dir}/tlc.log"

  mkdir -p "${run_dir}"

  echo "Running: ${rel_module} with ${rel_cfg}..."

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
    echo "  FAILED (exit ${exit_code})"
  else
    echo "  OK (${duration_sec}s)"
  fi

  states_generated="$(sed -n 's/^\([0-9][0-9]*\) states generated, .*$/\1/p' "${log_path}" | tail -n 1)"
  states_distinct="$(sed -n 's/^[0-9][0-9]* states generated, \([0-9][0-9]*\) distinct states found, .*$/\1/p' "${log_path}" | tail -n 1)"
  depth="$(sed -n 's/^The depth of the complete state graph search is \([0-9][0-9]*\)\.$/\1/p' "${log_path}" | tail -n 1)"

  [[ -z "${states_generated}" ]] && states_generated="na"
  [[ -z "${states_distinct}" ]] && states_distinct="na"
  [[ -z "${depth}" ]] && depth="na"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${status}" "${exit_code}" "${duration_sec}" \
    "${states_generated}" "${states_distinct}" "${depth}" \
    "${rel_module}" "${rel_cfg}" "${log_path}" >> "${SUMMARY_PATH}"
done

# Count results from summary
total=$(tail -n +2 "${SUMMARY_PATH}" | wc -l | tr -d ' ')
failures=$(grep -c $'\tfail\t' "${SUMMARY_PATH}" || true)

printf "\nCorpus TLC run complete: total=%d failures=%d summary=%s\n" "${total}" "${failures}" "${SUMMARY_PATH}"

if [[ ${failures} -ne 0 ]]; then
  echo "Failed specs:"
  grep $'\tfail\t' "${SUMMARY_PATH}" | cut -f1
  exit 1
fi
