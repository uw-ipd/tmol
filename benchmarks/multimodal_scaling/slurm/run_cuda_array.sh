#!/usr/bin/env bash
#SBATCH --job-name=tmol-matrix-cuda
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=06:15:00
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/cuda-%A_%a.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/cuda-%A_%a.err

set -euo pipefail
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
exec "${harness}/slurm/run_task_common.sh"
