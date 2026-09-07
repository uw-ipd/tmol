#!/usr/bin/env bash
#SBATCH --job-name=tmol-candidate-cpu
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:15:00

set -u -o pipefail

image=${TMOL_BENCH_IMAGE:?set TMOL_BENCH_IMAGE}
bench=${TMOL_BENCH_ROOT:?set TMOL_BENCH_ROOT}
harness=${TMOL_BENCH_HARNESS:?set TMOL_BENCH_HARNESS}
output=${TMOL_BROAD_CANDIDATE_ROOT:?set TMOL_BROAD_CANDIDATE_ROOT}
candidate_source=${TMOL_CANDIDATE_SOURCE:?set TMOL_CANDIDATE_SOURCE}
candidate_env=${TMOL_CANDIDATE_ENV:?set TMOL_CANDIDATE_ENV}
expected_commit=${TMOL_CANDIDATE_COMMIT:?set TMOL_CANDIDATE_COMMIT}
protocol=${TMOL_BENCH_PROTOCOL:?set TMOL_BENCH_PROTOCOL}
offset=${TMOL_TASK_OFFSET:-0}

actual_commit=$(git -C "${candidate_source}" rev-parse HEAD)
if [[ ${actual_commit} != "${expected_commit}" ]]; then
    echo "Candidate moved: expected ${expected_commit}, found ${actual_commit}" >&2
    exit 2
fi
source_status=$(git -C "${candidate_source}" status --porcelain | \
    sed '/^?? \.ninja_lock$/d')
if [[ -n ${source_status} ]]; then
    echo "Candidate source is dirty:" >&2
    echo "${source_status}" >&2
    exit 2
fi

task_index=$((offset + SLURM_ARRAY_TASK_ID))
task_file=${output}/metadata/tasks-broad-candidate-${protocol}-cpu.tsv
line=$(sed -n "$((task_index + 2))p" "${task_file}")
line=${line%$'\r'}
[[ -n ${line} ]] || { echo "No task ${task_index} in ${task_file}" >&2; exit 2; }
IFS=$'\t' read -r task_protocol modality dataset batch execution <<< "${line}"
[[ ${task_protocol} == "${protocol}" ]] || exit 2

short_commit=${expected_commit:0:12}
name=tmol-candidate-${short_commit}-${protocol}-cpu-b${batch}-${execution}-${modality}-${dataset}.json
output_host=${output}/raw/${name}
output_container=/results/raw/${name}
mkdir -p "${output}/raw"
[[ -s ${output_host} ]] && { echo "already complete: ${name}"; exit 0; }

if [[ ${protocol} == fastrelax ]]; then
    limit=6h
else
    limit=30m
fi

status=0
timeout "${limit}" apptainer exec \
    --bind "${candidate_source}:/work,${bench}:/bench,${harness}:/harness,${output}:/results" \
    --pwd /work "${image}" \
    env TMOL_BENCH_ROOT=/bench TMOL_USE_JIT=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "${candidate_env}/bin/python" /harness/src/benchmark_tmol.py \
    --dataset "${dataset}" --modality "${modality}" \
    --device cpu --batch-size "${batch}" --protocol "${protocol}" \
    --tmol-version "candidate-${short_commit}" --tmol-commit "${expected_commit}" \
    --cuda-execution "${execution}" --output "${output_container}" || status=$?

if [[ ! -s ${output_host} ]]; then
    env TMOL_BENCH_ROOT="${bench}" python3 "${harness}/src/record_failure.py" \
        --output "${output_host}" --engine tmol \
        --engine-version "candidate-${short_commit}" \
        --engine-commit "${expected_commit}" --protocol "${protocol}" \
        --device cpu --batch-size "${batch}" --dataset "${dataset}" \
        --modality "${modality}" --cuda-execution "${execution}" \
        --error "process exited with status ${status}"
fi
exit "${status}"
