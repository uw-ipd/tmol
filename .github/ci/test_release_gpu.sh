#!/usr/bin/env bash
# Run inside the GPU allocation/container, using the downloaded release wheel.
set -euo pipefail
# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bootstrap=$(mktemp -d)
trap 'rm -rf "$bootstrap"' EXIT
python3.12 -m venv "$bootstrap/env"
"$bootstrap/env/bin/python" -m pip install packaging
"$bootstrap/env/bin/python" scripts/test_package_install.py \
  --wheel wheels/*.whl --torch-version 2.8.0 \
  --torch-index https://download.pytorch.org/whl/cu129 --device cuda
