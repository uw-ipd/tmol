#!/usr/bin/env bash
set -euo pipefail

wheel_dir="${1:-dist}"
python_bin="${PYTHON:-python}"
plat="${TMOL_AUDITWHEEL_PLAT:-}"

if [ ! -d "$wheel_dir" ]; then
  echo "ERROR: wheel directory does not exist: $wheel_dir" >&2
  exit 1
fi

if [ -z "$plat" ]; then
  case "$(uname -m)" in
    x86_64) plat="manylinux_2_28_x86_64" ;;
    aarch64) plat="manylinux_2_28_aarch64" ;;
    *)
      echo "ERROR: cannot infer manylinux platform for $(uname -m)" >&2
      exit 1
      ;;
  esac
fi

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one wheel in $wheel_dir, found ${#wheels[@]}" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

exclude_libs=(
  libc10.so
  libc10_cuda.so
  libshm.so
  libtorch.so
  libtorch_cpu.so
  libtorch_cuda.so
  libtorch_cuda_linalg.so
  libtorch_global_deps.so
  libtorch_python.so
  libcuda.so.1
  libcudart.so.12
  libcudart.so.13
  libcublas.so.12
  libcublas.so.13
  libcublasLt.so.12
  libcublasLt.so.13
  libcufft.so.11
  libcufft.so.12
  libcurand.so.10
  libcurand.so.11
  libcusolver.so.11
  libcusolver.so.12
  libcusparse.so.12
  libcusparse.so.13
  libcusparseLt.so.0
  libnccl.so.2
  libnvJitLink.so.12
  libnvJitLink.so.13
  libnvToolsExt.so.1
  libnvrtc.so.12
  libnvrtc.so.13
  libnvrtc-builtins.so.12.8
  libnvrtc-builtins.so.12.9
  libnvrtc-builtins.so.13.0
  libnvrtc-builtins.so.13.2
)

exclude_args=()
for lib in "${exclude_libs[@]}"; do
  exclude_args+=(--exclude "$lib")
done

repair_dir="$(mktemp -d)"
trap 'rm -rf "$repair_dir"' EXIT

"$python_bin" -m auditwheel repair \
  --plat "$plat" \
  "${exclude_args[@]}" \
  -w "$repair_dir" \
  "${wheels[0]}"

repaired=("$repair_dir"/*.whl)
if [ "${#repaired[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one repaired wheel, found ${#repaired[@]}" >&2
  printf '  %s\n' "${repaired[@]}" >&2
  exit 1
fi

repaired_name="$(basename "${repaired[0]}")"
case "$repaired_name" in
  *-linux_*.whl)
    echo "ERROR: repair produced a native linux tag: $repaired_name" >&2
    exit 1
    ;;
  *-"$plat".whl) ;;
  *)
    echo "ERROR: repair did not produce expected $plat tag: $repaired_name" >&2
    exit 1
    ;;
esac

rm -f "$wheel_dir"/*.whl
mv "${repaired[0]}" "$wheel_dir/$repaired_name"
echo "Repaired wheel: $wheel_dir/$repaired_name"
