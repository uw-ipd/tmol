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
dataset=${TMOL_FASTRELAX_PROFILE_DATASET:-1acd}
modality=${TMOL_FASTRELAX_PROFILE_MODALITY:-protein}

mkdir -p "${output}/call-counts"
harness_git_root=$(git -C "${harness}" rev-parse --show-toplevel)
provenance="/results/metadata/fastrelax_modes_provenance-${SLURM_JOB_ID:-manual}.json"

apptainer exec --bind "${harness}:/harness,${output}:/results" "${image}" \
    python3 /harness/src/capture_candidate_provenance.py \
    --baseline-source "${baseline_source}" --baseline-env "${baseline_env}" \
    --candidate-source "${candidate_source}" --candidate-env "${candidate_env}" \
    --harness-root "${harness_git_root}" --image "${image}" \
    --output "${provenance}"

run_one() {
    local label=$1
    local replicate=$2
    local source=$3
    local environment=$4
    local commit=$5
    local shared_neighbors=$6
    local fused_ljlk_elec=$7
    apptainer exec \
        --bind "${bench}:/bench,${harness}:/harness,${source}:/work,${output}:/results" \
        --pwd /harness "${image}" \
        env TMOL_BENCH_ROOT=/bench TMOL_USE_JIT=0 \
        TMOL_SHARED_BLOCK_NEIGHBORS="${shared_neighbors}" \
        TMOL_FUSED_LJLK_ELEC="${fused_ljlk_elec}" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /harness/src/benchmark_fastrelax_calls.py \
        --dataset "${dataset}" --modality "${modality}" \
        --device cpu --batch-size 1 \
        --label "${label}" --commit "${commit}" \
        --output "/results/call-counts/${modality}-${dataset}-${label}-r${replicate}.json"
}

baseline_commit=$(git -C "${baseline_source}" rev-parse HEAD)
candidate_commit=$(git -C "${candidate_source}" rev-parse HEAD)

run_one baseline 1 "${baseline_source}" "${baseline_env}" \
    "${baseline_commit}" 0 0
run_one optimizer-only 1 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" 0 0
run_one shared-ordered 1 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" compact 0
run_one candidate 1 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" compact auto
run_one candidate 2 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" compact auto
run_one shared-ordered 2 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" compact 0
run_one optimizer-only 2 "${candidate_source}" "${candidate_env}" \
    "${candidate_commit}" 0 0
run_one baseline 2 "${baseline_source}" "${baseline_env}" \
    "${baseline_commit}" 0 0

python3 "${harness}/src/summarize_fastrelax_calls.py" \
    --input "${output}/call-counts" --output "${output}/summary"
