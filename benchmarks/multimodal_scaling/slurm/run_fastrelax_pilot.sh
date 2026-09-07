#!/usr/bin/env bash
#SBATCH --job-name=tmol-relax-pilot
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/relax-pilot-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/relax-pilot-%j.err

set -u -o pipefail
image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
bench=/mnt/home/kdidi/tmol-paper-benchmark
source=/mnt/home/kdidi/tmol-paper-sources/v0.1.55

for batch in 1 10; do
    timeout 45m apptainer exec --nv \
        --bind "${source}:/work,${bench}:/bench" --pwd /work "${image}" \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${bench}/envs/tmol-0.1.55/bin/python" /bench/src/benchmark_tmol.py \
        --dataset 5uoi --device cuda --batch-size "${batch}" --protocol fastrelax \
        --tmol-version 0.1.55 --tmol-commit 39f757f8837f853b85d6ce367938934e832c1033 \
        --output "/bench/results/raw/pilot-relax-tmol-0.1.55-cuda-b${batch}-5uoi.json" || true
done

timeout 45m env PYTHONPATH=/mnt/home/kdidi/projects/pyrosetta-2024.39 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /mnt/home/kdidi/.local/bin/python3.12 "${bench}/src/benchmark_pyrosetta.py" \
    --dataset 5uoi --protocol fastrelax \
    --output "${bench}/results/raw/pilot-relax-pyrosetta-5uoi.json" || true
