#!/usr/bin/env bash
#SBATCH --job-name=tmol-candidate-ab
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-ab-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-candidate-ab/logs/candidate-ab-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
output=${TMOL_CANDIDATE_AB_ROOT:-/mnt/data/kdidi/tmol-candidate-ab}
baseline_source=${TMOL_BASELINE_SOURCE:-/mnt/home/kdidi/tmol-paper-sources/v0.1.55}
baseline_env=${TMOL_BASELINE_ENV:-/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-0.1.55}
candidate_source=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
candidate_env=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}
device_filter=${TMOL_AB_DEVICE_FILTER:-all}

mkdir -p "${output}/logs" "${output}/raw"
baseline_commit=$(git -C "${baseline_source}" rev-parse HEAD)
candidate_commit=$(git -C "${candidate_source}" rev-parse HEAD)
harness_git_root=$(git -C "${harness}" rev-parse --show-toplevel)
provenance="/results/metadata/candidate_ab_provenance-${SLURM_JOB_ID:-manual}.json"

apptainer exec --bind "${harness}:/harness,${output}:/results" "${image}" \
    python3 /harness/src/capture_candidate_provenance.py \
    --baseline-source "${baseline_source}" --baseline-env "${baseline_env}" \
    --candidate-source "${candidate_source}" --candidate-env "${candidate_env}" \
    --harness-root "${harness_git_root}" --image "${image}" \
    --output "${provenance}"

run_one() {
    local label=$1
    local source=$2
    local environment=$3
    local commit=$4
    local replicate=$5
    local protocol=$6
    local modality=$7
    local dataset=$8
    local device=$9
    local batch=${10}
    local filename="${protocol}-${modality}-${dataset}-${device}-b${batch}-${label}-r${replicate}.json"
    local host_destination="${output}/raw/${filename}"
    local container_destination="/results/raw/${filename}"
    local nv=()
    if [[ ${device} == cuda ]]; then
        nv=(--nv)
    fi
    if [[ -s ${host_destination} ]]; then
        return
    fi
    if ! apptainer exec "${nv[@]}" \
        --bind "${bench}:/bench,${harness}:/harness,${source}:/work,${output}:/results" \
        --pwd /harness "${image}" \
        env TMOL_BENCH_ROOT=/bench TMOL_USE_JIT=0 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /harness/src/benchmark_tmol.py \
        --dataset "${dataset}" --modality "${modality}" \
        --device "${device}" --batch-size "${batch}" \
        --protocol "${protocol}" --cuda-execution eager \
        --tmol-version "${label}" --tmol-commit "${commit}" \
        --output "${container_destination}"; then
        [[ -s ${host_destination} ]] || return 1
    fi
}

configurations=(
    "score_gradient protein 1aie cpu 1"
    "score_gradient protein 1ecp cpu 1"
    "score_gradient protein 1aie cuda 1"
    "score_gradient protein 1ecp cuda 1"
    "score_gradient protein 1aie cuda 100"
    "score_gradient protein 1ecp cuda 100"
    "score_gradient protein_ligand 1aie cpu 1"
    "score_gradient protein_ligand 1ecp cpu 1"
    "score_gradient protein_ligand 1aie cuda 1"
    "score_gradient protein_ligand 1ecp cuda 1"
    "score_gradient protein_ligand 1aie cuda 100"
    "score_gradient protein_ligand 1ecp cuda 100"
    "score_gradient protein_nucleic 9j8g cpu 1"
    "score_gradient protein_nucleic 7ox9 cpu 1"
    "score_gradient protein_nucleic 9j8g cuda 1"
    "score_gradient protein_nucleic 7ox9 cuda 1"
    "score_gradient protein_nucleic 9j8g cuda 100"
    "score_gradient protein_nucleic 7ox9 cuda 100"
    "fastrelax protein 1acd cpu 1"
    "fastrelax protein 1acd cuda 1"
    "fastrelax protein 1acd cuda 10"
    "fastrelax protein_ligand 1acd cpu 1"
    "fastrelax protein_ligand 1acd cuda 1"
    "fastrelax protein_ligand 1acd cuda 10"
    "fastrelax protein_nucleic 9j8g cpu 1"
    "fastrelax protein_nucleic 9j8g cuda 1"
    "fastrelax protein_nucleic 9j8g cuda 10"
)

for configuration in "${configurations[@]}"; do
    read -r protocol modality dataset device batch <<< "${configuration}"
    if [[ ${device_filter} != all && ${device} != "${device_filter}" ]]; then
        continue
    fi
    baseline_replicate=0
    candidate_replicate=0
    for label in baseline candidate candidate baseline; do
        if [[ ${label} == baseline ]]; then
            baseline_replicate=$((baseline_replicate + 1))
            run_one baseline "${baseline_source}" "${baseline_env}" \
                "${baseline_commit}" "${baseline_replicate}" \
                "${protocol}" "${modality}" "${dataset}" "${device}" "${batch}"
        else
            candidate_replicate=$((candidate_replicate + 1))
            run_one candidate "${candidate_source}" "${candidate_env}" \
                "${candidate_commit}" "${candidate_replicate}" \
                "${protocol}" "${modality}" "${dataset}" "${device}" "${batch}"
        fi
    done
done

apptainer exec --bind "${output}:/results,${harness}:/harness" \
    --pwd /harness "${image}" \
    python3 /harness/src/summarize_candidate_ab.py \
    --input /results/raw --output /results/summary

apptainer exec --bind "${output}:/results,${harness}:/harness,${bench}:/bench" \
    --pwd /harness "${image}" \
    python3 /harness/src/compare_candidate_to_pyrosetta.py \
    --candidate-summary /results/summary/candidate_ab_summary.csv \
    --reference-summary /bench/results/summary/timing_summary.csv \
    --output /results/summary
