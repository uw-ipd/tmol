#!/usr/bin/env bash

set -u -o pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
protocol=${TMOL_BENCH_PROTOCOL:?set TMOL_BENCH_PROTOCOL}
device=${TMOL_BENCH_DEVICE:?set TMOL_BENCH_DEVICE}
offset=${TMOL_TASK_OFFSET:-0}
task_index=$((offset + SLURM_ARRAY_TASK_ID))
task_file=${bench}/metadata/tasks-${protocol}-${device}.tsv
line=$(sed -n "$((task_index + 2))p" "${task_file}")
line=${line%$'\r'}
[[ -n ${line} ]] || { echo "No task ${task_index} in ${task_file}" >&2; exit 2; }
IFS=$'\t' read -r engine version commit task_protocol task_device batch modality dataset execution <<< "${line}"
[[ ${task_protocol} == "${protocol}" && ${task_device} == "${device}" ]] || exit 2

mkdir -p "${bench}/results/raw"
execution_label=${execution:-none}
name=${engine}-${version}-${protocol}-${device}-b${batch}-${execution_label}-${modality}-${dataset}.json
output_host=${bench}/results/raw/${name}
output_container=/bench/results/raw/${name}
[[ -s ${output_host} ]] && { echo "already complete: ${name}"; exit 0; }

if [[ ${protocol} == fastrelax ]]; then
    limit=6h
else
    limit=30m
fi

status=0
if [[ ${engine} == pyrosetta ]]; then
    timeout "${limit}" env \
        TMOL_BENCH_ROOT="${bench}" \
        PYTHONPATH=/mnt/home/kdidi/projects/pyrosetta-2024.39 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        /mnt/home/kdidi/.local/bin/python3.12 "${harness}/src/benchmark_pyrosetta.py" \
        --dataset "${dataset}" --modality "${modality}" \
        --protocol "${protocol}" --output "${output_host}" || status=$?
else
    source=/mnt/home/kdidi/tmol-paper-sources/v${version}
    python=/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-${version}/bin/python
    nv=()
    [[ ${device} == cuda ]] && nv=(--nv)
    timeout "${limit}" apptainer exec "${nv[@]}" \
        --bind "${source}:/work,${bench}:/bench,${harness}:/harness" \
        --pwd /work "${image}" \
        env TMOL_BENCH_ROOT=/bench OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "${python}" /harness/src/benchmark_tmol.py \
        --dataset "${dataset}" --modality "${modality}" \
        --device "${device}" --batch-size "${batch}" --protocol "${protocol}" \
        --tmol-version "${version}" --tmol-commit "${commit}" \
        --cuda-execution "${execution}" --output "${output_container}" || status=$?
fi

if [[ ! -s ${output_host} ]]; then
    env TMOL_BENCH_ROOT="${bench}" python3 "${harness}/src/record_failure.py" \
        --output "${output_host}" --engine "${engine}" --engine-version "${version}" \
        --engine-commit "${commit}" --protocol "${protocol}" --device "${device}" \
        --batch-size "${batch}" --dataset "${dataset}" --modality "${modality}" \
        --cuda-execution "${execution}" --error "process exited with status ${status}"
fi
exit "${status}"
