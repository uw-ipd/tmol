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

touch_gpu_sentinel() {
  if [[ -n "${GPU_ALLOC_SENTINEL:-}" ]]; then
    touch "${GPU_ALLOC_SENTINEL}"
  fi
}
