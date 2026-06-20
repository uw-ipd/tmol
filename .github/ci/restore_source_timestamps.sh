#!/usr/bin/env bash
# Restore C++/CUDA source timestamps from git so incremental builds behave.
set -euo pipefail

git ls-files -- '*.cpp' '*.cu' '*.cuh' '*.h' '*.hh' '*.cc' | \
  xargs -P8 -I{} bash -c 'ts=$(git log -1 --format="%ct" -- "$1") && touch -d "@$ts" "$1"' _ {}
