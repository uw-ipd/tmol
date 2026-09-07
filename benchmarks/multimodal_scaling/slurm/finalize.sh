#!/usr/bin/env bash
#SBATCH --job-name=tmol-paper-finalize
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/finalize-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/finalize-%j.err

set -euo pipefail

image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
bench=/mnt/home/kdidi/tmol-paper-benchmark
python=${bench}/envs/tmol-0.1.55/bin/python

for script in collect_results.py summarize_results.py plot_results.py; do
    apptainer exec \
        --bind "${bench}:/bench" \
        --pwd /bench \
        "${image}" \
        "${python}" "/bench/src/${script}"
done
