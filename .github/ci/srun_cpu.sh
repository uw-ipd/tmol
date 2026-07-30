#!/usr/bin/env bash
# Run a CPU-only srun step (no GPU gres). Used for pytest -k "not cuda".
#
# Usage:
#   .github/ci/srun_cpu.sh <<'SCRIPT'
#   apptainer exec ... bash <<'INNER'
#   ...
#   INNER
#   SCRIPT

set -euo pipefail

SLURM_CPU_PARTITION="${SLURM_CPU_PARTITION:-cpu}"
SLURM_TIME="${SLURM_CPU_TIME:-06:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
SLURM_MEM="${SLURM_CPU_MEM:-${SLURM_MEM:-64G}}"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
inner_script="${tmpdir}/inner.sh"
cat >"$inner_script"

# Workflow env sets SLURM_GRES=gpu:1 for GPU lanes. Slurm applies that to every
# srun when we --export=ALL, which makes cpu-partition jobs fail with
# "Requested node configuration is not available".
unset SLURM_GRES

srun \
  -p "$SLURM_CPU_PARTITION" \
  -t "$SLURM_TIME" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --mem="$SLURM_MEM" \
  --gres=none \
  --export=ALL \
  bash -s <"$inner_script"
