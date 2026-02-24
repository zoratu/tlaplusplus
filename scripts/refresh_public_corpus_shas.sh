#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-${ROOT_DIR}/corpus/public/public_corpus.tsv}"
OUTPUT_PATH="${2:-${ROOT_DIR}/corpus/public/public_corpus.lock.tsv}"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "missing input corpus list: ${INPUT_PATH}" >&2
  exit 1
fi

printf "# id\trepo_url\tcommit_sha\tmodule_path\tcfg_path\tlicense\ttags\n" > "${OUTPUT_PATH}"

while IFS=$'\t' read -r id repo_url commit_sha module_path cfg_path license tags _; do
  [[ -z "${id}" ]] && continue
  [[ "${id}" == \#* ]] && continue

  if [[ "${commit_sha}" == "HEAD" || -z "${commit_sha}" ]]; then
    resolved_sha="$(git ls-remote "${repo_url}" HEAD | awk '{print $1}')"
  else
    resolved_sha="${commit_sha}"
  fi

  if [[ -z "${resolved_sha}" ]]; then
    echo "failed to resolve sha for ${id} (${repo_url})" >&2
    exit 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${repo_url}" "${resolved_sha}" "${module_path}" "${cfg_path}" "${license}" "${tags}" \
    >> "${OUTPUT_PATH}"
done < "${INPUT_PATH}"

echo "wrote ${OUTPUT_PATH}"
