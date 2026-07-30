#!/usr/bin/env bash
# CPU pytest lane (runs on the self-hosted runner, no Slurm).
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

source .venv/bin/activate

CUDA_VISIBLE_DEVICES="" pytest -p no:rerunfailures -s -v --durations=25 \
  --cov="${GITHUB_WORKSPACE}/tmol" --cov-report=xml \
  --junitxml="${GITHUB_WORKSPACE}/testing.cpu.junit.xml" -k "not cuda"
