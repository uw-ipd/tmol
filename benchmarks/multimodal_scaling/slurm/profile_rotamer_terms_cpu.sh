#!/usr/bin/env bash
#SBATCH --job-name=tmol-rotamer-terms
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/kdidi/tmol-final-candidate-ab/logs/rotamer-terms-%j.out
#SBATCH --error=/mnt/data/kdidi/tmol-final-candidate-ab/logs/rotamer-terms-%j.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
output=${TMOL_CANDIDATE_AB_ROOT:-/mnt/data/kdidi/tmol-final-candidate-ab}
baseline_source=${TMOL_BASELINE_SOURCE:-/mnt/home/kdidi/tmol-paper-sources/v0.1.55}
baseline_env=${TMOL_BASELINE_ENV:-/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-0.1.55}
candidate_source=${TMOL_CANDIDATE_SOURCE:-/mnt/home/kdidi/projects/tmol-pr468-shared-integration}
candidate_env=${TMOL_CANDIDATE_ENV:-/mnt/home/kdidi/tmol-shared-neighbor-bench/env}
# These are included protein records near 31, 150, and 547 residues in the
# frozen expanded manifest. Override with a whitespace-separated list when
# profiling a different manifest.
profile_datasets=${TMOL_ROTAMER_PROFILE_DATASETS:-"1aie 1f3g 1v0b"}

mkdir -p "${output}/rotamer-term-profile"
baseline_commit=$(git -C "${baseline_source}" rev-parse HEAD)
candidate_commit=$(git -C "${candidate_source}" rev-parse HEAD)
harness_git_root=$(git -C "${harness}" rev-parse --show-toplevel)
provenance="/results/metadata/rotamer_terms_provenance-${SLURM_JOB_ID:-manual}.json"

apptainer exec --bind "${harness}:/harness,${output}:/results" "${image}" \
    python3 /harness/src/capture_candidate_provenance.py \
    --baseline-source "${baseline_source}" --baseline-env "${baseline_env}" \
    --candidate-source "${candidate_source}" --candidate-env "${candidate_env}" \
    --harness-root "${harness_git_root}" --image "${image}" \
    --output "${provenance}"

run_one() {
    local label=$1 source=$2 environment=$3 commit=$4 dataset=$5 replicate=$6
    apptainer exec \
        --bind "${bench}:/bench,${harness}:/harness,${source}:/work,${output}:/results" \
        --pwd /harness "${image}" \
        env TMOL_BENCH_ROOT=/bench TMOL_USE_JIT=0 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "${environment}/bin/python" /harness/src/profile_rotamer_terms.py \
        --dataset "${dataset}" --modality protein --device cpu --batch-size 1 \
        --label "${label}" --commit "${commit}" \
        --output "/results/rotamer-term-profile/${dataset}-${label}-r${replicate}.json"
}

read -r -a datasets <<< "${profile_datasets}"
for dataset in "${datasets[@]}"; do
    run_one baseline "${baseline_source}" "${baseline_env}" "${baseline_commit}" "${dataset}" 1
    run_one candidate "${candidate_source}" "${candidate_env}" "${candidate_commit}" "${dataset}" 1
    run_one candidate "${candidate_source}" "${candidate_env}" "${candidate_commit}" "${dataset}" 2
    run_one baseline "${baseline_source}" "${baseline_env}" "${baseline_commit}" "${dataset}" 2
done

python3 "${harness}/src/summarize_rotamer_terms.py" \
    --input "${output}/rotamer-term-profile" \
    --output "${output}/summary"
