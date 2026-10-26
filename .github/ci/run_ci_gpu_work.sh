#!/usr/bin/env bash
# Build and test inside one GPU allocation.
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

echo "=== representative CUDA JIT compile ==="
TMOL_USE_JIT=1 python - <<'PY'
import torch
from tmol.kinematics.compiled import forward_kin_op

assert torch.cuda.is_available()
print("loaded", forward_kin_op, "with torch", torch.__version__)
PY

# The tutorial smokes run one kernel at a time and mostly wait on the host, so
# they overlap the CUDA tests instead of adding their runtime to the job. They
# finish before the benchmarks start, which are timing-sensitive.
echo "=== execute GPU tutorial smokes (in the background, overlapping the tests) ==="
smoke_log=$(mktemp)
python .github/scripts/smoke_tutorial_notebooks.py \
  --gpu-smoke \
  --execution-device cuda >"${smoke_log}" 2>&1 &
smoke_pid=$!

echo "=== tests (CUDA) ==="
tests_rc=0
.github/ci/run_gpu_tests.sh || tests_rc=$?

echo "=== tutorial smoke results ==="
smoke_rc=0
wait "${smoke_pid}" || smoke_rc=$?
cat "${smoke_log}"
if ((tests_rc != 0)); then
  exit "${tests_rc}"
fi

echo "=== benchmarks (CUDA) ==="
.github/ci/run_benchmarks.sh

exit "${smoke_rc}"
