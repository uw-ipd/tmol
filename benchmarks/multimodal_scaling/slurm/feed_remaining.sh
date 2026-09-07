#!/usr/bin/env bash
#SBATCH --job-name=tmol-matrix-feeder
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/feeder-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/feeder-%j.err

set -euo pipefail

bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}

exec env TMOL_BENCH_ROOT="${bench}" TMOL_BENCH_HARNESS="${harness}" \
    python3 "${harness}/src/feed_scheduler.py" "$@"
