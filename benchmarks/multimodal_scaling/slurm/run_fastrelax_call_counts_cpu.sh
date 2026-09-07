#!/usr/bin/env bash
#SBATCH --job-name=tmol-relax-counts
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-ab/logs/relax-counts-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-ab/logs/relax-counts-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
output=${TMOL_CANDIDATE_AB_ROOT:-/mnt/data/kdidi/tmol-candidate-ab}
baseline_source=${TMOL_BASELINE_SOURCE:-/mnt/home/kdidi/tmol-paper-sources/v0.1.55}
baseline_env=${TMOL_BASELINE_ENV:-/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-0.1.55}
candidate_source=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
candidate_env=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}

mkdir -p "${output}/call-counts"

run_one() {
    local label=$1
    local source=$2
    local environment=$3
    local commit=$4
    apptainer exec \
        --bind "${bench}:/bench,${harness}:/harness,${source}:/work,${output}:/results" \
        --pwd /harness "${image}" \
        env TMOL_BENCH_ROOT=/bench TMOL_USE_JIT=0 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /harness/src/benchmark_fastrelax_calls.py \
        --dataset 1acd --modality protein --device cpu --batch-size 1 \
        --label "${label}" --commit "${commit}" \
        --output "/results/call-counts/protein-1acd-${label}.json"
}

run_one baseline "${baseline_source}" "${baseline_env}" \
    "$(git -C "${baseline_source}" rev-parse HEAD)"
run_one candidate "${candidate_source}" "${candidate_env}" \
    "$(git -C "${candidate_source}" rev-parse HEAD)"
