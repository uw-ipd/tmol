#!/usr/bin/env bash
#SBATCH --job-name=tmol-candidate-build
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-release-ab/logs/candidate-build-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-release-ab/logs/candidate-build-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
source_dir=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
environment=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}

apptainer exec --bind "${source_dir}:/work" --pwd /work "${image}" \
    env CMAKE_BUILD_PARALLEL_LEVEL=16 CMAKE_CUDA_ARCHITECTURES=90 \
    CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake \
    pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11 \
    "${environment}/bin/python" -m pip install -e /work \
    --no-deps --no-build-isolation -Ccmake.define.TMOL_BUILD_TESTS=OFF

grep -Fx 'CMAKE_BUILD_TYPE:STRING=Release' "${source_dir}/CMakeCache.txt"
grep -Fx 'TMOL_BUILD_TESTS:BOOL=OFF' "${source_dir}/CMakeCache.txt"
apptainer exec --bind "${source_dir}:/work" --pwd /work "${image}" \
    env TMOL_USE_JIT=0 "${environment}/bin/python" -c \
    'import tmol; print(tmol.__version__, tmol.__file__)'
