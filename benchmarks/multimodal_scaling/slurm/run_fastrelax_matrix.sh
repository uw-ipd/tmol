#!/usr/bin/env bash
#SBATCH --job-name=tmol-paper-relax
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/logs/relax-%j.out
#SBATCH --error=/mnt/home/kdidi/tmol-paper-benchmark/logs/relax-%j.err

set -u -o pipefail

image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
bench=/mnt/home/kdidi/tmol-paper-benchmark

historical_version() {
    if [[ $modality == protein_nucleic ]]; then
        echo 0.1.47
    else
        echo 0.1.46
    fi
}

commit_for_version() {
    case "$1" in
        0.1.46) echo 4ac54af52f9dc02cfc6a439f151eea63561e6cf0 ;;
        0.1.47) echo 2e73c9fb030e64e175161e9a385bb3229262a99b ;;
        0.1.55) echo 39f757f8837f853b85d6ce367938934e832c1033 ;;
        *) return 2 ;;
    esac
}

run_tmol() {
    version=$1
    dataset=$2
    device=$3
    batch=$4
    commit=$(commit_for_version "${version}")
    source=/mnt/home/kdidi/tmol-paper-sources/v${version}
    output_host=${bench}/results/raw/tmol-${version}-fastrelax-${device}-b${batch}-${modality}-${dataset}.json
    output=/bench/results/raw/$(basename "${output_host}")
    [[ -s ${output_host} ]] && return
    timeout 90m apptainer exec --nv \
        --bind "${source}:/work,${bench}:/bench" --pwd /work "${image}" \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "${bench}/envs/tmol-${version}/bin/python" /bench/src/benchmark_tmol.py \
        --dataset "${dataset}" --device "${device}" --batch-size "${batch}" \
        --protocol fastrelax --tmol-version "${version}" --tmol-commit "${commit}" \
        --output "${output}"
    status=$?
    if [[ ! -s ${output_host} ]]; then
        python3 "${bench}/src/record_failure.py" --output "${output_host}" \
            --engine tmol --engine-version "${version}" --engine-commit "${commit}" \
            --protocol fastrelax --device "${device}" --batch-size "${batch}" \
            --dataset "${dataset}" --error "process exited with status ${status}"
    fi
}

run_pyrosetta() {
    dataset=$1
    output=${bench}/results/raw/pyrosetta-2024.39-fastrelax-cpu-b1-${modality}-${dataset}.json
    [[ -s ${output} ]] && return
    timeout 90m env \
        PYTHONPATH=/mnt/home/kdidi/projects/pyrosetta-2024.39 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        /mnt/home/kdidi/.local/bin/python3.12 "${bench}/src/benchmark_pyrosetta.py" \
        --dataset "${dataset}" --protocol fastrelax --output "${output}"
    status=$?
    if [[ ! -s ${output} ]]; then
        python3 "${bench}/src/record_failure.py" --output "${output}" \
            --engine pyrosetta --engine-version 2024.39 \
            --engine-commit 59628fbc5bc09f1221e1642f1f8d157ce49b1410 \
            --protocol fastrelax --device cpu --batch-size 1 \
            --dataset "${dataset}" --error "process exited with status ${status}"
    fi
}

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
read -r -a modalities <<< "${BENCH_MODALITIES:-protein protein_ligand protein_nucleic}"
for modality in "${modalities[@]}"; do
    case "${modality}" in
        protein) datasets=(5uoi 5yzf 5m4a 1o7j) ;;
        protein_ligand) datasets=(hsp90 p38 src ace) ;;
        protein_nucleic) datasets=(5exh 1ysa 3ndh 6q1h) ;;
    esac
    historical=$(historical_version)
    for dataset in "${datasets[@]}"; do
        run_pyrosetta "${dataset}"
        for version in 0.1.55 "${historical}"; do
            run_tmol "${version}" "${dataset}" cpu 1
            for batch in 1 10 100 1000; do
                run_tmol "${version}" "${dataset}" cuda "${batch}"
            done
        done
    done
done

apptainer exec \
    --bind "${bench}:/bench" \
    --pwd /bench \
    "${image}" \
    "${bench}/envs/tmol-0.1.55/bin/python" /bench/src/collect_results.py
