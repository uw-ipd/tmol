#!/usr/bin/env bash
#SBATCH --job-name=tmol-multimodal-plots
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/plots-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/plots-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/home/kdidi/tmol-paper-benchmark}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
envs=${TMOL_BENCH_ENVS:-/mnt/home/kdidi/tmol-paper-benchmark/envs}
python=${envs}/tmol-0.1.55/bin/python

mkdir -p "${bench}/logs" "${bench}/results/summary" "${bench}/figures"

for script in collect_results.py summarize_results.py plot_results.py; do
    apptainer exec \
        --bind "${bench}:/bench,${harness}:/harness" \
        --pwd /harness \
        "${image}" \
        env TMOL_BENCH_ROOT=/bench \
        "${python}" "/harness/src/${script}"
done
