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

BUILD_DEPS_TMP=$(mktemp -d)
trap 'rm -rf "${BUILD_DEPS_TMP}"' EXIT
retry uv pip compile --quiet pyproject.toml --all-extras \
  --output-file "${BUILD_DEPS_TMP}/requirements.txt"
grep -vE "^(torch(|vision|audio)|numpy|cuda-.*|nvidia-.*|triton|tensorrt|pynvml|pandas|scipy)==" \
  "${BUILD_DEPS_TMP}/requirements.txt" > "${BUILD_DEPS_TMP}/to_install.txt"
retry uv pip install -r "${BUILD_DEPS_TMP}/to_install.txt"
NVCC_VER=$(nvcc --version 2>&1 | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
case "${NVCC_VER}" in
  13.*) DEFAULT_TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130" ;;
  *) DEFAULT_TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128" ;;
esac
TORCH_CUDA_INDEX="${TMOL_CI_TORCH_CUDA_INDEX:-${DEFAULT_TORCH_CUDA_INDEX}}"
retry uv pip install --upgrade torch --index-url "${TORCH_CUDA_INDEX}"
# The extension must be configured with the same Torch/CUDA installation that
# will load it at runtime.  PEP 517 build isolation otherwise resolves the
# unbounded ``torch>=2.5`` build requirement independently; when a newer CUDA
# major is current on PyPI this can silently produce (for example) a CUDA 13
# extension in a CUDA 12.8 runtime environment.
retry uv pip install "scikit-build-core>=0.10" "pybind11>=2.12" ninja packaging
assert_torch_cuda
TORCH_CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
NVCC_MAJOR=${NVCC_VER%%.*}
if [[ "${TORCH_CUDA_MAJOR}" != "${NVCC_MAJOR}" ]]; then
  echo "PyTorch CUDA ${TORCH_CUDA_MAJOR} does not match nvcc ${NVCC_VER}" >&2
  exit 1
fi

RUN_GPU=$(python -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo "n/a")
# Test jobs only execute on this runner, so compiling every wheel architecture
# wastes most of the job. Release wheels retain the all-supported-SM default.
CUDA_ARCHS="${TMOL_CI_CUDA_ARCHITECTURES:-native}"
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
MAX_JOBS=12 pip install -v --no-deps --no-build-isolation \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON \
  -Ccmake.define.TMOL_NVCC_THREADS=2 \
  -e .
