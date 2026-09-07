#!/usr/bin/env bash
#SBATCH --job-name=tmol-bench-provenance
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/provenance-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/provenance-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
harness_git_root=$(git -C "${harness}" rev-parse --show-toplevel)

apptainer exec \
    --bind "${bench}:/bench,${harness}:/harness,${harness_git_root}:/harness-worktree" \
    --pwd /harness \
    "${image}" \
    env TMOL_BENCH_ROOT=/bench TMOL_BENCH_IMAGE="${image}" \
    TMOL_BENCH_HARNESS_GIT_ROOT=/harness-worktree \
    python3 /harness/src/capture_provenance.py
