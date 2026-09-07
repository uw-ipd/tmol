#!/usr/bin/env bash
#SBATCH --job-name=tmol-candidate-cpu-test
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-cpu-test-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-cpu-test-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
source_dir=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
environment=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}

apptainer exec --bind "${source_dir}:/work" --pwd /work "${image}" \
    env TMOL_USE_JIT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    "${environment}/bin/python" -m pytest \
    tmol/tests/score/test_score_function.py \
    tmol/tests/score/cartbonded \
    tmol/tests/score/ljlk \
    tmol/tests/score/elec \
    tmol/tests/score/hbond \
    tmol/tests/score/lk_ball \
    tmol/tests/optimization/test_armijo_compiled.py \
    tmol/tests/optimization/test_lbfgs_armijo.py \
    tmol/tests/optimization/test_minimizers.py \
    tmol/tests/relax/test_fast_relax.py \
    -q
