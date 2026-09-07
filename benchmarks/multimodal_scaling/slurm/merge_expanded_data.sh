#!/usr/bin/env bash
#SBATCH --job-name=tmol-data-merge
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/merge-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/merge-%j.err

set -euo pipefail

bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
env TMOL_BENCH_ROOT="${bench}" python3 "${harness}/src/merge_expanded_manifests.py"
env TMOL_BENCH_ROOT="${bench}" python3 "${harness}/src/build_task_matrix.py"
