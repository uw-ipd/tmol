#!/usr/bin/env bash
# Compile and install tmol inside an existing GPU srun allocation.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

source .venv/bin/activate
# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
strip_cuda_compat_from_ld_path

uv pip compile pyproject.toml --all-extras --output-file requirements.txt
grep -vE "^(torch(|vision|audio)|numpy|nvidia-.*|triton|tensorrt|pynvml|pandas|scipy)==" \
  requirements.txt > to_install.txt
uv pip install -r to_install.txt
uv pip install torch

RUN_GPU=$(python -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo "n/a")
CUDA_ARCHS="${TMOL_CI_CUDA_ARCHITECTURES:-80;86;89;90;100}"
NVCC_VER=$(nvcc --version 2>&1 | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
case "${NVCC_VER}" in
  11.*|12.*) CUDA_ARCHS="80;86;89;90" ;;
esac
unset CMAKE_CUDA_ARCHITECTURES
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
TORCH_ARCH_LIST=""
IFS=';' read -ra _CUDA_ARCH_ARR <<< "${CUDA_ARCHS}"
for _A in "${_CUDA_ARCH_ARR[@]}"; do
  if [ "${#_A}" -eq 3 ]; then
    TORCH_ARCH_LIST+=" ${_A:0:2}.${_A:2:1}"
  elif [ "${#_A}" -eq 2 ]; then
    TORCH_ARCH_LIST+=" ${_A:0:1}.${_A:1:1}"
  fi
done
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST# }"
echo "=== Runner GPU sm_${RUN_GPU} | nvcc ${NVCC_VER} | CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} ==="
MAX_JOBS=12 pip install -v --no-deps \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON \
  -Ccmake.define.TMOL_NVCC_THREADS=2 \
  -e .
