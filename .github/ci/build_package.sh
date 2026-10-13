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
# CUDA extras are separate runtime distributions; build this checkout directly.
retry uv pip compile --quiet pyproject.toml --extra dev --extra docs \
  --output-file "${BUILD_DEPS_TMP}/requirements.txt"
grep -vE "^(torch(|vision|audio)|numpy|cuda-.*|nvidia-.*|triton|tensorrt|pynvml|pandas|scipy)==" \
  "${BUILD_DEPS_TMP}/requirements.txt" > "${BUILD_DEPS_TMP}/to_install.txt"
retry uv pip install -r "${BUILD_DEPS_TMP}/to_install.txt"
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
NVCC_MAJOR=${NVCC_VER%%.*}
case "${NVCC_MAJOR}" in
  12) TORCH_VERSION=2.11.0; TORCH_INDEX=cu128 ;;
  13) TORCH_VERSION=2.14.1; TORCH_INDEX=cu132 ;;
  *) echo "CUDA 12 or 13 is required" >&2; exit 1 ;;
esac
retry uv pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
retry uv pip install 'cmake>=3.24,<4' 'scikit-build-core>=0.10' ninja 'packaging>=24.2' 'pybind11>=2.12'
assert_torch_cuda

# Production uses current stable headers and the runtime's C ABI. Test-only
# extensions use the runtime's own C++ headers/libraries, including on CUDA 12.
STABLE_HEADERS=$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "include")')
if [[ "${NVCC_MAJOR}" == 12 ]]; then
  retry python -m pip download --no-deps --only-binary=:all: --dest "${BUILD_DEPS_TMP}" \
    torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  python - "${BUILD_DEPS_TMP}" <<'HEADERS'
from pathlib import Path
import sys, zipfile
with zipfile.ZipFile(next(Path(sys.argv[1]).glob('torch-*.whl'))) as archive:
    for member in archive.namelist():
        if member.startswith('torch/include/'):
            archive.extract(member, 'build/stable-headers')
HEADERS
  STABLE_HEADERS="$PWD/build/stable-headers/torch/include"
fi
CUDA_ARCHS="${TMOL_CI_CUDA_ARCHITECTURES:-native}"
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; c=torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")')
CMAKE_BUILD_PARALLEL_LEVEL=4 pip install -v --no-build-isolation --no-deps \
  -Ccmake.define.TMOL_ENABLE_CUDA=ON \
  -Ccmake.define.TMOL_STABLE_HEADERS="${STABLE_HEADERS}" \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -Ccmake.define.TMOL_BUILD_TESTS=ON \
  -Ccmake.define.TMOL_NVCC_THREADS=2 -e .
