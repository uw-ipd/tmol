#!/usr/bin/env bash
# GPU benchmark lane (mirrors foundry-dev gpu job pattern).
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

if [[ -n "${GPU_ALLOC_SENTINEL:-}" ]]; then
  touch "${GPU_ALLOC_SENTINEL}"
fi

source .venv/bin/activate

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/usr/local/cuda/compat' | paste -sd:)

BENCHMARK_DIR="benchmark/${GITHUB_REPOSITORY}/${GITHUB_REF_NAME}"
BENCHMARK_RESULT="${BENCHMARK_DIR}/${GITHUB_RUN_NUMBER}.json"
mkdir -p "$BENCHMARK_DIR"

pytest -p no:rerunfailures --benchmark-enable --benchmark-only \
  --benchmark-name=short --benchmark-sort=fullname \
  --benchmark-columns=ops,mean,iqr \
  --benchmark-json="${BENCHMARK_RESULT}" \
  --benchmark-max-time=.1

# First run on a branch has no history to compare against.
json_files=$(find benchmark -name '*.json' -print -quit)
if [[ -n "$json_files" ]]; then
  pytest-benchmark compare --name=short --sort=fullname \
    --columns=ops,mean,iqr $(find benchmark -name '*.json')
fi
