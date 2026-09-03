#!/usr/bin/env bash
# Compile and install tmol inside an existing GPU srun allocation.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

source .venv/bin/activate
# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
strip_cuda_compat_from_ld_path

# The GPU compute nodes have flaky outbound access to PyPI / download.pytorch.org
# (intermittent "Connection reset by peer"). uv's own retries all fire within a
# few seconds, so wrap network installs in a backoff loop to ride out short
# outages instead of failing the whole CI run.
retry() {
  local -i attempt=1 max="${RETRY_MAX_ATTEMPTS:-5}" delay="${RETRY_BASE_DELAY:-15}"
  while true; do
    if "$@"; then
      return 0
    fi
    if ((attempt >= max)); then
      echo "retry: command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "retry: attempt ${attempt}/${max} failed; sleeping ${delay}s: $*" >&2
    sleep "${delay}"
    attempt+=1
    delay=$((delay * 2))
  done
}

retry uv pip compile pyproject.toml --all-extras --output-file requirements.txt
grep -vE "^(torch(|vision|audio)|numpy|nvidia-.*|triton|tensorrt|pynvml|pandas|scipy)==" \
  requirements.txt > to_install.txt
retry uv pip install -r to_install.txt
TORCH_CUDA_INDEX="${TMOL_CI_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
retry uv pip install torch --index-url "${TORCH_CUDA_INDEX}"
retry uv pip install 'cmake>=3.24,<4' 'scikit-build-core>=0.10' ninja \
  'packaging>=24.2' 'pybind11>=2.12'
assert_torch_cuda

RUN_GPU=$(python -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo "n/a")
# Test jobs only execute on this runner, so compiling every wheel architecture
# wastes most of the job. Release wheels retain the all-supported-SM default.
CUDA_ARCHS="${TMOL_CI_CUDA_ARCHITECTURES:-native}"
NVCC_VER=$(nvcc --version 2>&1 | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
if [[ "${CUDA_ARCHS}" == "native" ]]; then
  # Exercise CMake's native path in CI. JIT extensions still need PyTorch's
  # numeric spelling for the same runner GPU.
  TORCH_CUDA_ARCHS="${RUN_GPU/./}"
else
  case "${NVCC_VER}" in
    11.*|12.*) CUDA_ARCHS="80;86;89;90" ;;
  esac
  TORCH_CUDA_ARCHS="${CUDA_ARCHS}"
fi
unset CMAKE_CUDA_ARCHITECTURES
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
TORCH_ARCH_LIST=""
IFS=';' read -ra _CUDA_ARCH_ARR <<< "${TORCH_CUDA_ARCHS}"
for _A in "${_CUDA_ARCH_ARR[@]}"; do
  if [ "${#_A}" -eq 3 ]; then
    TORCH_ARCH_LIST+=" ${_A:0:2}.${_A:2:1}"
  elif [ "${#_A}" -eq 2 ]; then
    TORCH_ARCH_LIST+=" ${_A:0:1}.${_A:1:1}"
  fi
done
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST# }"
echo "=== Runner GPU sm_${RUN_GPU} | nvcc ${NVCC_VER} | CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} ==="
# Compile against the same PyTorch installation used by the test process.
# Build isolation may resolve a newer PyTorch with an incompatible C++ ABI.
MAX_JOBS=12 pip install -v --no-build-isolation --no-deps \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON \
  -Ccmake.define.TMOL_NVCC_THREADS=2 \
  -e .
