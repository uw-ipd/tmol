#!/usr/bin/env bash
#SBATCH --job-name=tmol-candidate-cpu
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-cpu-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-cpu-%j.err

set -euo pipefail

harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
TMOL_AB_DEVICE_FILTER=cpu bash "${harness}/slurm/run_candidate_ab.sh"
