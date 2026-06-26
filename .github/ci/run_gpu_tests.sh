#!/usr/bin/env bash
# CUDA pytest lane. Writes GPU_ALLOC_SENTINEL once the allocation starts so
# srun_gpu_retry.sh can distinguish TaskProlog failures from real test failures.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

source .venv/bin/activate
assert_torch_cuda

pytest -p no:rerunfailures -s -v --durations=25 \
  --cov="${GITHUB_WORKSPACE}/tmol" --cov-report=xml --cov-append \
  --junitxml="${GITHUB_WORKSPACE}/testing.cuda.junit.xml" -k "cuda"
