#!/usr/bin/env bash
# Retry GPU srun allocations when Slurm TaskProlog or GPU health checks fail.
#
# Typical failure (bad node):
#   Failed to get device handle for GPU 0: Unknown Error
#   TaskProlog failed status=1
#   srun: error: g2507: task 0: Exited with exit code 1
#
# Usage (stdin is the shell script to run inside the allocation):
#   .github/ci/srun_gpu_retry.sh <<'SCRIPT'
#   apptainer exec --nv ... bash <<'INNER'
#   set -ex
#   ...
#   INNER
#   SCRIPT
#
# Environment (set by CI workflow):
#   SLURM_PARTITION, SLURM_GRES, SLURM_TIME, SLURM_CPUS_PER_TASK, SLURM_MEM
#   SRUN_GPU_MAX_ATTEMPTS (default 5)
#   SRUN_GPU_RETRY_SLEEP (default 10 seconds between attempts)

set -euo pipefail

MAX_ATTEMPTS="${SRUN_GPU_MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${SRUN_GPU_RETRY_SLEEP:-10}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu-train}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_TIME="${SLURM_TIME:-06:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
SLURM_MEM="${SLURM_MEM:-128G}"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
inner_script="${tmpdir}/inner.sh"
cat >"$inner_script"

exclude=""
attempt=1

_gpu_retry_pattern='TaskProlog failed|Failed to get device handle|Unable to determine the device handle|GPU problem:|nvidia-smi failed'

while (( attempt <= MAX_ATTEMPTS )); do
  echo "=== srun GPU attempt ${attempt}/${MAX_ATTEMPTS} (exclude: ${exclude:-none}) ==="

  srun_args=(
    -p "$SLURM_PARTITION"
    --gres="$SLURM_GRES"
    -t "$SLURM_TIME"
    --cpus-per-task="$SLURM_CPUS_PER_TASK"
    --mem="$SLURM_MEM"
    --export=ALL
  )
  if [[ -n "$exclude" ]]; then
    srun_args+=(--exclude="$exclude")
  fi

  log="${tmpdir}/srun_${attempt}.log"
  set +e
  {
    cat <<'GPUCHECK'
set -euo pipefail
echo "=== GPU health check ==="
if ! nvidia-smi --query-gpu=index,name --format=csv,noheader; then
  echo "nvidia-smi failed"
  exit 42
fi
GPUCHECK
    cat "$inner_script"
  } | srun "${srun_args[@]}" bash -s 2>&1 | tee "$log"
  rc=${PIPESTATUS[1]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    exit 0
  fi

  # If the inner job created the allocation sentinel, tests started — do not
  # retry (real test failure, not a dirty GPU / TaskProlog rejection).
  if [[ -n "${GPU_ALLOC_SENTINEL:-}" && -f "${GPU_ALLOC_SENTINEL}" ]]; then
    echo "GPU_ALLOC_SENTINEL present — tests started; not retrying (rc=${rc})."
    exit "$rc"
  fi

  if [[ "$rc" -eq 42 ]] || grep -qE "$_gpu_retry_pattern" "$log"; then
    bad_node=$(grep -oE 'srun: error: [a-zA-Z0-9][a-zA-Z0-9_-]*' "$log" | head -1 | awk '{print $3}')
    if [[ -n "$bad_node" ]]; then
      if [[ -n "$exclude" ]]; then
        exclude="${exclude},${bad_node}"
      else
        exclude="$bad_node"
      fi
      echo "GPU node failure on ${bad_node}; excluding and retrying in ${RETRY_SLEEP}s"
    else
      echo "GPU allocation failure (node unknown); retrying in ${RETRY_SLEEP}s"
    fi
    sleep "$RETRY_SLEEP"
    ((attempt++))
    continue
  fi

  echo "srun failed with exit code ${rc} (not a retriable GPU allocation error)"
  exit "$rc"
done

echo "srun GPU failed after ${MAX_ATTEMPTS} attempts"
exit 1
