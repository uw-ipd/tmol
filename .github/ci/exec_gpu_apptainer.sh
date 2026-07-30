#!/usr/bin/env bash
# Run a script inside the CI GPU apptainer, preserving Slurm CUDA env vars that
# apptainer --containall would otherwise drop.
#
# Usage: exec_gpu_apptainer.sh path/to/script.sh
set -euo pipefail

SIF="${SIF:-/home/bench/git_ci_apptainer/tmol.sif}"
: "${GITHUB_WORKSPACE:?}"

script="${1:?usage: exec_gpu_apptainer.sh script.sh}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "${script_dir}/gpu_env.sh"
resolve_cuda_visible_devices

env_args=(
  --env "GITHUB_WORKSPACE=${GITHUB_WORKSPACE}"
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
)
if [[ -n "${GPU_ALLOC_SENTINEL:-}" ]]; then
  env_args+=(--env "GPU_ALLOC_SENTINEL=${GPU_ALLOC_SENTINEL}")
fi
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  env_args+=(--env "GITHUB_REPOSITORY=${GITHUB_REPOSITORY}")
fi
if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
  env_args+=(--env "GITHUB_REF_NAME=${GITHUB_REF_NAME}")
fi
if [[ -n "${GITHUB_RUN_NUMBER:-}" ]]; then
  env_args+=(--env "GITHUB_RUN_NUMBER=${GITHUB_RUN_NUMBER}")
fi

apptainer exec --nv --fakeroot --containall \
  --bind "${GITHUB_WORKSPACE}:${GITHUB_WORKSPACE}" \
  --pwd "${GITHUB_WORKSPACE}" \
  "${env_args[@]}" \
  "$SIF" bash "$script"
