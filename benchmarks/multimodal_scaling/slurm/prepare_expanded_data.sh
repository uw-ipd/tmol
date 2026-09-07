#!/usr/bin/env bash
#SBATCH --job-name=tmol-data-prep
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/data/kdidi/tmol-multimodal-scale/logs/prepare-%A_%a.out
#SBATCH --error=/mnt/data/kdidi/tmol-multimodal-scale/logs/prepare-%A_%a.err

set -euo pipefail

image=${TMOL_BENCH_IMAGE:-/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif}
bench=${TMOL_BENCH_ROOT:-/mnt/data/kdidi/tmol-multimodal-scale}
harness=${TMOL_BENCH_HARNESS:-/mnt/home/kdidi/projects/tmol-multimodal-benchmark/benchmarks/multimodal_scaling}
target=${TMOL_PREP_TARGET:-128}
case "${SLURM_ARRAY_TASK_ID}" in
    0) modality=protein ;;
    1) modality=protein_ligand ;;
    2) modality=protein_nucleic ;;
    *) exit 2 ;;
esac

mkdir -p "${bench}/logs" "${bench}/metadata"

env \
    TMOL_BENCH_ROOT="${bench}" \
    PYTHONPATH=/mnt/home/kdidi/projects/pyrosetta-2024.39 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /mnt/home/kdidi/.local/bin/python3.12 \
    "${harness}/src/prepare_expanded_datasets.py" \
    --modality "${modality}" --target "${target}"
