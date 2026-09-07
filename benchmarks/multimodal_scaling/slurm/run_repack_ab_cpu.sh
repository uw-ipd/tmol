#!/usr/bin/env bash
#SBATCH --job-name=tmol-repack-ab
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-ab/logs/repack-ab-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-ab/logs/repack-ab-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
output=${TMOL_CANDIDATE_AB_ROOT:-/mnt/data/kdidi/tmol-candidate-ab}
baseline_source=${TMOL_BASELINE_SOURCE:-/mnt/home/kdidi/tmol-paper-sources/v0.1.55}
baseline_env=${TMOL_BASELINE_ENV:-/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-0.1.55}
candidate_source=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
candidate_env=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}

mkdir -p "${output}/repack"

run_one() {
    local label=$1
    local replicate=$2
    local source=$3
    local environment=$4
    apptainer exec \
        --bind "${source}:/work,${candidate_source}:/candidate,${output}:/results" \
        --pwd /candidate "${image}" \
        env TMOL_USE_JIT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /candidate/dev/benchmarks/rosetta_comparison.py \
        --engine tmol --workflow repack --device cpu --threads 1 --batch 1 \
        --warmup 1 --repeats 3 \
        --pdb /work/tmol/tests/data/pdb/bysize_150_res_5yzf.pdb \
        --output "/results/repack/${label}-r${replicate}.json"
}

run_one baseline 1 "${baseline_source}" "${baseline_env}"
run_one candidate 1 "${candidate_source}" "${candidate_env}"
run_one candidate 2 "${candidate_source}" "${candidate_env}"
run_one baseline 2 "${baseline_source}" "${baseline_env}"
