#!/usr/bin/env bash
# GPU container env tweaks shared by CUDA test and benchmark lanes.
#
# Drop the container's CUDA forward-compat libcuda so torch uses the host
# driver (avoids CUDA init error 803 on newer-driver nodes).
strip_cuda_compat_from_ld_path() {
  local stripped
  stripped=$(
    echo "${LD_LIBRARY_PATH:-}" \
      | tr ':' '\n' \
      | grep -v '/usr/local/cuda/compat' \
      | paste -sd: \
      || true
  )
  export LD_LIBRARY_PATH="${stripped}"
}

# Apptainer --containall drops Slurm's GPU env; re-resolve before apptainer exec.
resolve_cuda_visible_devices() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    return 0
  fi
  if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
}

touch_gpu_sentinel() {
  if [[ -n "${GPU_ALLOC_SENTINEL:-}" ]]; then
    touch "${GPU_ALLOC_SENTINEL}"
  fi
}

assert_torch_cuda() {
  python - <<'PY'
import sys

import torch

print(
    f"torch {torch.__version__} built_cuda={torch.version.cuda} "
    f"is_available={torch.cuda.is_available()} "
    f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
)
if torch.version.cuda is None:
    sys.exit(
        "PyTorch has no CUDA support (CPU-only wheel). "
        "Install from https://download.pytorch.org/whl/cu128"
    )
if not torch.cuda.is_available():
    sys.exit(
        "torch.cuda.is_available() is False inside the GPU container "
        "(check CUDA_VISIBLE_DEVICES and apptainer --nv)."
    )
print(f"cuda device: {torch.cuda.get_device_name(0)}")
PY
}
