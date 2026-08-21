#!/usr/bin/env bash
# Build tmol, execute notebook outputs, and render docs in one GPU allocation.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GPU_ALLOC_SENTINEL:?}"

# shellcheck source=/dev/null
source .github/ci/gpu_env.sh
touch_gpu_sentinel
strip_cuda_compat_from_ld_path

source .venv/bin/activate

echo "=== build CUDA-enabled tmol ==="
.github/ci/build_package.sh

echo "=== verify tutorial thumbnails ==="
python .github/scripts/render_tutorial_thumbnails.py
git diff --exit-code -- docs/_static/tutorials

echo "=== execute CPU tutorial outputs ==="
python .github/scripts/smoke_tutorial_notebooks.py --write

echo "=== execute GPU tutorial outputs ==="
python .github/scripts/smoke_tutorial_notebooks.py \
  docs/tutorial/02_gpu_batching.ipynb \
  docs/tutorial/06_fast_relax.ipynb \
  --execution-device cuda \
  --write

echo "=== build Sphinx documentation ==="
make -C docs html

echo "=== verify rendered public API coverage ==="
python .github/scripts/check_api_docs.py

echo "=== verify rendered workflow navigation ==="
python .github/scripts/check_docs_navigation.py
