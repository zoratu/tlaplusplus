#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MODULE="${ROOT_DIR}/corpus/language_coverage/LanguageFeatureMatrix.tla"
DEFAULT_CFG="${ROOT_DIR}/corpus/language_coverage/LanguageFeatureMatrix.cfg"
MODULE_PATH="${1:-${DEFAULT_MODULE}}"
CFG_PATH="${2:-${DEFAULT_CFG}}"
OUT_DIR="${3:-${ROOT_DIR}/.tlc-out/language_coverage}"
JAR_PATH="${TLA2TOOLS_JAR:-${ROOT_DIR}/tools/tla2tools.jar}"
JAVA_PATH="${JAVA_BIN:-/opt/homebrew/opt/openjdk/bin/java}"

abs_file_path() {
  local input="$1"
  local dir base
  dir="$(cd "$(dirname "${input}")" && pwd)"
  base="$(basename "${input}")"
  printf "%s/%s" "${dir}" "${base}"
}

if [[ ! -x "${JAVA_PATH}" ]]; then
  JAVA_PATH="$(command -v java)"
fi

if [[ ! -f "${JAR_PATH}" ]]; then
  echo "missing tla2tools jar: ${JAR_PATH}" >&2
  echo "set TLA2TOOLS_JAR=/absolute/path/to/tla2tools.jar" >&2
  exit 1
fi

if [[ ! -f "${MODULE_PATH}" ]]; then
  echo "missing module: ${MODULE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "missing cfg: ${CFG_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
MODULE_PATH="$(abs_file_path "${MODULE_PATH}")"
CFG_PATH="$(abs_file_path "${CFG_PATH}")"
MODULE_DIR="$(dirname "${MODULE_PATH}")"
CFG_DIR="$(dirname "${CFG_PATH}")"

COMMON_DIR="$(python3 - "${MODULE_DIR}" "${CFG_DIR}" <<'PY'
import os
import sys
print(os.path.commonpath([sys.argv[1], sys.argv[2]]))
PY
)"

MODULE_FILE="$(python3 - "${MODULE_PATH}" "${COMMON_DIR}" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
)"

CFG_FILE="$(python3 - "${CFG_PATH}" "${COMMON_DIR}" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
)"

LIB_DIRS=()
if [[ "${MODULE_DIR}" != "${COMMON_DIR}" ]]; then
  LIB_DIRS+=("${MODULE_DIR}")
fi
if [[ "${CFG_DIR}" != "${COMMON_DIR}" && "${CFG_DIR}" != "${MODULE_DIR}" ]]; then
  LIB_DIRS+=("${CFG_DIR}")
fi

if git_root="$(git -C "${MODULE_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  if [[ "${git_root}" != "${COMMON_DIR}" && "${git_root}" != "${MODULE_DIR}" ]]; then
    LIB_DIRS+=("${git_root}")
  fi
fi

if git_root_cfg="$(git -C "${CFG_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  if [[ "${git_root_cfg}" != "${COMMON_DIR}" && "${git_root_cfg}" != "${MODULE_DIR}" && "${git_root_cfg}" != "${CFG_DIR}" ]]; then
    LIB_DIRS+=("${git_root_cfg}")
  fi
fi

LIB_ARG=""
if [[ "${#LIB_DIRS[@]}" -gt 0 ]]; then
  LIB_ARG="${LIB_DIRS[0]}"
  for ((i = 1; i < ${#LIB_DIRS[@]}; i++)); do
    LIB_ARG="${LIB_ARG}:${LIB_DIRS[i]}"
  done
fi

pushd "${COMMON_DIR}" >/dev/null
if [[ -n "${LIB_ARG}" ]]; then
  TLA_LIBRARY_VALUE="${LIB_ARG}"
  if [[ -n "${TLA_LIBRARY:-}" ]]; then
    TLA_LIBRARY_VALUE="${TLA_LIBRARY_VALUE}:${TLA_LIBRARY}"
  fi
  TLA_LIBRARY="${TLA_LIBRARY_VALUE}" \
  "${JAVA_PATH}" -jar "${JAR_PATH}" \
    -cleanup \
    -workers auto \
    -metadir "${OUT_DIR}/meta" \
    -config "${CFG_FILE}" \
    "${MODULE_FILE}"
else
  "${JAVA_PATH}" -jar "${JAR_PATH}" \
    -cleanup \
    -workers auto \
    -metadir "${OUT_DIR}/meta" \
    -config "${CFG_FILE}" \
    "${MODULE_FILE}"
fi
popd >/dev/null
