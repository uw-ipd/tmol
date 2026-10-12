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
for backend in cu12 cu13; do
  case "$backend" in
    cu12) torch_cuda=cu128 ;;
    cu13) torch_cuda=cu130 ;;
  esac
  "$bootstrap/env/bin/python" scripts/test_package_install.py \
    --wheel wheels/tmol_ops_${backend}-*.whl --base-wheel base-wheel/*.whl \
    --torch-version 2.11.0 \
    --torch-index "https://download.pytorch.org/whl/$torch_cuda" --device cuda
done
