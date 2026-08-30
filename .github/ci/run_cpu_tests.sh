#!/usr/bin/env bash
# CPU pytest lane (runs on a GitHub-hosted runner).
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"

source .venv/bin/activate

CPU_PYTEST_WORKERS="${CPU_PYTEST_WORKERS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CUDA_VISIBLE_DEVICES="" COVERAGE_FILE="${GITHUB_WORKSPACE}/.coverage.cpu" \
  pytest -p no:rerunfailures -ra --durations=25 \
  -n "${CPU_PYTEST_WORKERS}" --dist=worksteal \
  --cov="${GITHUB_WORKSPACE}/tmol" \
  --cov-report="xml:${GITHUB_WORKSPACE}/coverage.cpu.xml" \
  --junitxml="${GITHUB_WORKSPACE}/testing.cpu.junit.xml" -k "not cuda"
