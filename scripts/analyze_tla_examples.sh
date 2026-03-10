#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="${1:-${HOME}/src/tlaplus-examples/specifications}"
OUT_ROOT="${2:-${ROOT_DIR}/.analyze-tla/examples}"
BINARY="${BINARY:-${ROOT_DIR}/target/release/tlaplusplus}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SUMMARY_PATH="${OUT_ROOT}/summary.tsv"

mkdir -p "${OUT_ROOT}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found or not executable: ${BINARY}" >&2
  exit 1
fi

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
  echo "Examples directory not found: ${EXAMPLES_DIR}" >&2
  exit 1
fi

resolve_module_path() {
  local cfg_path="$1"
  local cfg_dir cfg_name candidate test_name tla_count
  local best_match="" best_len=0 tla_path tla_base
  cfg_dir="$(dirname "${cfg_path}")"
  cfg_name="$(basename "${cfg_path}" .cfg)"

  if [[ -f "${cfg_dir}/${cfg_name}.tla" ]]; then
    printf '%s\n' "${cfg_dir}/${cfg_name}.tla"
    return 0
  fi

  while IFS= read -r tla_path; do
    tla_base="$(basename "${tla_path}" .tla)"
    if [[ "${cfg_name}" == "${tla_base}"* ]]; then
      if (( ${#tla_base} > best_len )); then
        best_match="${tla_path}"
        best_len=${#tla_base}
      fi
    fi
  done < <(find "${cfg_dir}" -maxdepth 1 -name '*.tla' | sort)

  if [[ -n "${best_match}" ]]; then
    printf '%s\n' "${best_match}"
    return 0
  fi

  test_name="${cfg_name}"
  while [[ -n "${test_name}" ]]; do
    if [[ "${test_name}" == *_* ]]; then
      test_name="${test_name%_*}"
    elif [[ "${test_name}" =~ ^(.+?)[0-9].*$ ]]; then
      test_name="${BASH_REMATCH[1]}"
    elif [[ "${test_name}" =~ ^(.+[a-z])([A-Z][A-Za-z0-9]+)$ ]]; then
      test_name="${BASH_REMATCH[1]}"
    else
      break
    fi

    candidate="${cfg_dir}/${test_name}.tla"
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  tla_count="$(find "${cfg_dir}" -maxdepth 1 -name '*.tla' | wc -l | tr -d ' ')"
  if [[ "${tla_count}" == "1" ]]; then
    find "${cfg_dir}" -maxdepth 1 -name '*.tla' | head -n 1
    return 0
  fi

  return 1
}

extract_field() {
  local key="$1"
  local log_path="$2"
  sed -n "s/^${key}=//p" "${log_path}" | tail -n 1
}

extract_first_error() {
  local log_path="$1"
  sed -n 's/^error_count=[0-9][0-9]* error=//p' "${log_path}" | head -n 1
}

extract_first_example() {
  local log_path="$1"
  sed -n 's/^error_example=//p' "${log_path}" | head -n 1
}

printf "id\tstatus\texit_code\tduration_sec\tmodule\tcfg\taction_eval\texpr_eval\texpr_total\texpr_ok\texpr_failed\tunsupported_feature_count\tfirst_error\tfirst_error_example\tlog\n" > "${SUMMARY_PATH}"

CFGS=()
while IFS= read -r cfg_path; do
  CFGS+=("${cfg_path}")
done < <(find "${EXAMPLES_DIR}" -name '*.cfg' -type f | sort)

processed=0
failures=0
for idx in "${!CFGS[@]}"; do
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi

  cfg_path="${CFGS[idx]}"
  rel_cfg="${cfg_path#${EXAMPLES_DIR}/}"
  id="${rel_cfg%.cfg}"
  id="${id//\//_}"
  run_dir="${OUT_ROOT}/${id}"
  log_path="${run_dir}/analyze.log"
  mkdir -p "${run_dir}"

  module_path=""
  if ! module_path="$(resolve_module_path "${cfg_path}")"; then
    printf "%s\tskip\tna\t0\tna\t%s\tna\tna\tna\tna\tna\tna\tno module\tna\t%s\n" \
      "${id}" "${rel_cfg}" "${log_path}" >> "${SUMMARY_PATH}"
    continue
  fi

  rel_module="${module_path#${EXAMPLES_DIR}/}"
  echo "[${SHARD_INDEX}/${SHARD_COUNT}] analyzing ${rel_cfg}" >&2

  start_epoch="$(date +%s)"
  set +e
  "${BINARY}" analyze-tla --module "${module_path}" --config "${cfg_path}" >"${log_path}" 2>&1
  exit_code=$?
  set -e
  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  action_eval="$(extract_field "native_frontend.action_eval" "${log_path}")"
  expr_eval="$(extract_field "native_frontend.expr_eval" "${log_path}")"
  expr_total="$(extract_field "expr_probe_total" "${log_path}")"
  expr_ok="$(extract_field "expr_probe_ok" "${log_path}")"
  expr_failed="$(extract_field "expr_probe_failed" "${log_path}")"
  unsupported_count="$(extract_field "native_frontend.unsupported_feature_count" "${log_path}")"
  first_error="$(extract_first_error "${log_path}")"
  first_example="$(extract_first_example "${log_path}")"

  [[ -z "${action_eval}" ]] && action_eval="na"
  [[ -z "${expr_eval}" ]] && expr_eval="na"
  [[ -z "${expr_total}" ]] && expr_total="na"
  [[ -z "${expr_ok}" ]] && expr_ok="na"
  [[ -z "${expr_failed}" ]] && expr_failed="na"
  [[ -z "${unsupported_count}" ]] && unsupported_count="na"
  [[ -z "${first_error}" ]] && first_error="na"
  [[ -z "${first_example}" ]] && first_example="na"

  status="fail"
  if [[ ${exit_code} -eq 0 ]]; then
    if [[ "${action_eval}" == "true" && "${expr_eval}" == "true" ]]; then
      status="full_pass"
    elif [[ "${action_eval}" == "true" ]]; then
      status="action_only"
    elif [[ "${expr_eval}" == "true" ]]; then
      status="expr_only"
    else
      status="parse_only"
    fi
  else
    failures=$((failures + 1))
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${status}" "${exit_code}" "${duration_sec}" \
    "${rel_module}" "${rel_cfg}" "${action_eval}" "${expr_eval}" \
    "${expr_total}" "${expr_ok}" "${expr_failed}" "${unsupported_count}" \
    "${first_error}" "${first_example}" "${log_path}" >> "${SUMMARY_PATH}"

  processed=$((processed + 1))
done

echo "processed=${processed} failures=${failures} summary=${SUMMARY_PATH}" >&2
