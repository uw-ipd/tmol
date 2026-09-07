#!/usr/bin/env bash
#SBATCH --job-name=tmol-matrix-cpu
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:15:00
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/cpu-%A_%a.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/cpu-%A_%a.err

set -euo pipefail
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
exec "${harness}/slurm/run_task_common.sh"
