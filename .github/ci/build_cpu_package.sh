#!/usr/bin/env bash
# Build an editable CPU-only tmol package on a GitHub-hosted runner.
set -euo pipefail

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip uv
if [ "$(uname -s)" = Darwin ]; then
  # PyPI supplies the native Apple Silicon build; the PyTorch CPU index is
  # specific to Linux and Windows.
  uv pip install torch
else
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
if [ "$(uname -m)" = aarch64 ]; then
  # biotraj has no Linux aarch64 wheel, and its sdist omits both build
  # requirements and VCS version metadata. Build the matching tag explicitly.
  uv pip install numpy Cython hatchling hatch-vcs
  uv pip install --no-build-isolation \
    'biotraj @ git+https://github.com/biotite-dev/biotraj.git@v1.2.2'
fi
uv pip install 'cmake>=3.18,<4' 'scikit-build-core>=0.10' ninja \
  'packaging>=24.2' 'pybind11>=2.12'

export CMAKE_PREFIX_PATH
CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake_args=(
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
  -Ccmake.define.TMOL_BUILD_TESTS=ON
)
if command -v ccache >/dev/null; then
  ccache --max-size="${CCACHE_MAXSIZE:-750M}"
  ccache --zero-stats
  cmake_args+=(
    -Ccmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi
MAX_JOBS="${MAX_JOBS:-4}" uv pip install --no-build-isolation -e '.[dev]' \
  "${cmake_args[@]}"
