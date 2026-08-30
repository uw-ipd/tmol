#!/usr/bin/env bash
# All GPU-allocation work in one Slurm job: build, CUDA tests, benchmarks.
#
# Invoked inside apptainer on a gpu-train node (see ci.yml). CPU tests run in a
# separate GitHub-hosted job and do not occupy the GPU allocation.
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

echo "=== tests (CUDA) ==="
.github/ci/run_gpu_tests.sh

echo "=== execute GPU tutorial smokes ==="
python .github/scripts/smoke_tutorial_notebooks.py \
  docs/tutorial/02_gpu_batching.ipynb \
  docs/tutorial/06_fast_relax.ipynb \
  --execution-device cuda

echo "=== benchmarks ==="
.github/ci/run_benchmarks.sh
