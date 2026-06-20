#!/usr/bin/env bash
# GPU pytest lane with allocation sentinel (mirrors foundry-dev gpu job pattern).
#
# Writes GPU_ALLOC_SENTINEL once the srun allocation starts so the workflow can
# distinguish TaskProlog/dirty-GPU failures (no sentinel) from real test failures.
set -euo pipefail

: "${GPU_ALLOC_SENTINEL:?GPU_ALLOC_SENTINEL must be set}"
: "${GITHUB_WORKSPACE:?}"

touch "${GPU_ALLOC_SENTINEL}"

source .venv/bin/activate

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/usr/local/cuda/compat' | paste -sd:)

pytest -p no:rerunfailures -s -v --durations=25 \
  --cov="${GITHUB_WORKSPACE}/tmol" --cov-report=xml \
  --junitxml="${GITHUB_WORKSPACE}/testing.cuda.junit.xml" -k "cuda"
