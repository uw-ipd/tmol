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

# Most of this lane is ligand preparation and pose construction on the host,
# so one process leaves the allocation's CPUs idle and the GPU mostly unused.
# Workers share the single GPU; keep the count well under the CPU allocation so
# the memory-heavy packing tests still have room.
GPU_PYTEST_WORKERS="${GPU_PYTEST_WORKERS:-4}"

COVERAGE_FILE="${GITHUB_WORKSPACE}/.coverage.cuda" \
  pytest -p no:rerunfailures -ra --durations=25 \
  -n "${GPU_PYTEST_WORKERS}" --dist=worksteal \
  --cov="${GITHUB_WORKSPACE}/tmol" \
  --cov-report="xml:${GITHUB_WORKSPACE}/coverage.cuda.xml" \
  --junitxml="${GITHUB_WORKSPACE}/testing.cuda.junit.xml" -k "cuda"
