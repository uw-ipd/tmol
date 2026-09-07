#!/usr/bin/env bash
#SBATCH --job-name=tmol-na-preflight
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/na-preflight-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/na-preflight-%j.err

set -uo pipefail

image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
bench=/mnt/home/kdidi/tmol-paper-benchmark

status=0
for version in 0.1.55 0.1.47; do
    apptainer exec \
        --bind "/mnt/home/kdidi/tmol-paper-sources/v${version}:/work,${bench}:/bench" \
        --pwd /work \
        "${image}" \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${bench}/envs/tmol-${version}/bin/python" \
        /bench/src/preflight_tmol_inputs.py --tmol-version "${version}" || status=1
done
exit "${status}"
