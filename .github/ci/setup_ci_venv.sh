#!/usr/bin/env bash
# Bootstrap the CI virtualenv inside the tmol apptainer image (runner-side).
set -euo pipefail

SIF="${SIF:-/home/bench/git_ci_apptainer/tmol.sif}"

apptainer exec --nv --fakeroot --containall \
  --bind "${GITHUB_WORKSPACE}:${GITHUB_WORKSPACE}" \
  --pwd "${GITHUB_WORKSPACE}" \
  "${SIF}" bash <<'SETUP'
set -ex

PYTHON_BIN=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"; then
      PYTHON_BIN=$(command -v "$cand")
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: No Python >=3.11 found in CI container."
  exit 1
fi

if [ ! -d .venv ]; then
  # The GPU image already provides a CUDA-matched NVIDIA PyTorch build. Make it
  # visible to the venv so clean CI jobs do not redownload several GB of CUDA
  # wheels; build_package.sh still replaces it when the CUDA majors differ.
  "$PYTHON_BIN" -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install pip --upgrade
pip install uv
SETUP
