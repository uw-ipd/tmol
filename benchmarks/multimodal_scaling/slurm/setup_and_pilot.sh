#!/usr/bin/env bash
#SBATCH --job-name=tmol-paper-setup
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/setup-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/setup-%j.err

set -euo pipefail

image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
bench=/mnt/home/kdidi/tmol-paper-benchmark
seed_env=/mnt/home/kdidi/tmol-v0154-followup-env

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
lscpu

build_version() {
    version=$1
    source=/mnt/home/kdidi/tmol-paper-sources/v${version}
    environment=${bench}/envs/tmol-${version}
    if [[ ! -e ${environment}/bin/activate ]]; then
        cp -a "${seed_env}" "${environment}"
    fi
    apptainer exec --nv \
        --bind "${source}:/work,${bench}:/bench" \
        --pwd /work \
        "${image}" \
        env CMAKE_BUILD_PARALLEL_LEVEL=16 CMAKE_CUDA_ARCHITECTURES=90 \
        CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake \
        pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11 \
        "${environment}/bin/python" -m pip install -e /work --no-deps --no-build-isolation
}

build_version 0.1.46
build_version 0.1.47
build_version 0.1.55

apptainer exec --nv \
    --bind "/mnt/home/kdidi/tmol-paper-sources/v0.1.55:/work,${bench}:/bench" \
    --pwd /work \
    "${image}" \
    "${bench}/envs/tmol-0.1.55/bin/python" /bench/src/export_ligand_params.py

for device in cpu cuda; do
    apptainer exec --nv \
        --bind "/mnt/home/kdidi/tmol-paper-sources/v0.1.55:/work,${bench}:/bench" \
        --pwd /work \
        "${image}" \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${bench}/envs/tmol-0.1.55/bin/python" /bench/src/benchmark_tmol.py \
        --dataset 5uoi --device "${device}" --batch-size 1 --protocol score_gradient \
        --tmol-version 0.1.55 --tmol-commit 39f757f8837f853b85d6ce367938934e832c1033 \
        --output "/bench/results/raw/pilot-tmol-0.1.55-${device}-5uoi-gradient.json"
done
