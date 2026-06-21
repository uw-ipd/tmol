#!/usr/bin/env bash
# GPU benchmark lane.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-local/tmol}"
GITHUB_REF_NAME="${GITHUB_REF_NAME:-local}"
GITHUB_RUN_NUMBER="${GITHUB_RUN_NUMBER:-0}"

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

source .venv/bin/activate
assert_torch_cuda

BENCHMARK_DIR="benchmark/${GITHUB_REPOSITORY}/${GITHUB_REF_NAME}"
BENCHMARK_RESULT="${BENCHMARK_DIR}/${GITHUB_RUN_NUMBER}.json"
mkdir -p "$BENCHMARK_DIR"

pytest -p no:rerunfailures --benchmark-enable --benchmark-only \
  --benchmark-name=short --benchmark-sort=fullname \
  --benchmark-columns=ops,mean,iqr \
  --benchmark-json="${BENCHMARK_RESULT}" \
  --benchmark-max-time=.1

# Compare only when a prior run exists (skip first run on a branch).
mapfile -t json_files < <(find benchmark -name '*.json' -print 2>/dev/null || true)
if ((${#json_files[@]} > 1)); then
  pytest-benchmark compare --name=short --sort=fullname \
    --columns=ops,mean,iqr "${json_files[@]}"
fi
