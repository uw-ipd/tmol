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
uv pip install 'cmake>=3.18,<4' 'scikit-build-core>=0.10' ninja \
  'packaging>=24.2' 'pybind11>=2.12'

export CMAKE_PREFIX_PATH
CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
MAX_JOBS="${MAX_JOBS:-4}" uv pip install --no-build-isolation -e '.[dev]' \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF \
  -Ccmake.define.TMOL_BUILD_TESTS=ON
