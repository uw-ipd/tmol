#!/usr/bin/env bash
# GPU benchmark lane.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

source .venv/bin/activate

BENCHMARK_DIR="benchmark/${GITHUB_REPOSITORY}/${GITHUB_REF_NAME}"
BENCHMARK_RESULT="${BENCHMARK_DIR}/${GITHUB_RUN_NUMBER}.json"
mkdir -p "$BENCHMARK_DIR"

pytest -p no:rerunfailures --benchmark-enable --benchmark-only \
  --benchmark-name=short --benchmark-sort=fullname \
  --benchmark-columns=ops,mean,iqr \
  --benchmark-json="${BENCHMARK_RESULT}" \
  --benchmark-max-time=.1

# First run on a branch has no history to compare against.
mapfile -t json_files < <(find benchmark -name '*.json' -print 2>/dev/null || true)
if ((${#json_files[@]} > 0)); then
  pytest-benchmark compare --name=short --sort=fullname \
    --columns=ops,mean,iqr "${json_files[@]}"
fi
