#!/bin/bash
# Pre-expand a tmol header as MSL. Runtime shader compilation cannot resolve
# filesystem includes, so the TU must be flattened first.
# usage: expand.sh <in.metal.in> <out.metal>
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
clang++ -E -x c++ -std=c++17 -D__METAL_VERSION__=320 \
  -I"$root/dev/mps_scratch/mslstub" -I"$root" -I"$root/tmol/extern" "$1" \
  | grep -v '^#' > "$2".body
printf '#include <metal_stdlib>\nusing namespace metal;\n' | cat - "$2".body > "$2"
rm -f "$2".body
