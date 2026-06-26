#!/usr/bin/env bash
# All GPU-allocation work in one Slurm job: build, CPU tests, CUDA tests, benchmarks.
#
# Invoked inside apptainer on a gpu-train node (see ci.yml). CPU tests run on the
# same node with CUDA_VISIBLE_DEVICES="" — no separate cpu-partition srun.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GPU_ALLOC_SENTINEL:?}"

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

source .venv/bin/activate

echo "=== build ==="
.github/ci/build_package.sh

echo "=== tests (CPU) ==="
.github/ci/run_cpu_tests.sh

echo "=== tests (CUDA) ==="
.github/ci/run_gpu_tests.sh

echo "=== benchmarks ==="
.github/ci/run_benchmarks.sh
