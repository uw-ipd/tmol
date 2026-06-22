#!/usr/bin/env bash
# Repair Linux wheels with auditwheel for manylinux_2_28 glibc compatibility.
set -euo pipefail

DIST_DIR="${1:-dist}"
if [ ! -d "$DIST_DIR" ]; then
  echo "ERROR: dist directory not found: $DIST_DIR" >&2
  exit 1
fi

shopt -s nullglob
wheels=("$DIST_DIR"/*.whl)
if ((${#wheels[@]} == 0)); then
  echo "ERROR: no wheels found in $DIST_DIR" >&2
  exit 1
fi

REPAIRED_DIR="$DIST_DIR/repaired"
mkdir -p "$REPAIRED_DIR"

for whl in "${wheels[@]}"; do
  echo "=== auditwheel repair $(basename "$whl") ==="
  auditwheel repair "$whl" -w "$REPAIRED_DIR" --plat manylinux_2_28_x86_64
  rm -f "$whl"
done

mv "$REPAIRED_DIR"/*.whl "$DIST_DIR"/
rmdir "$REPAIRED_DIR"
ls -lh "$DIST_DIR"/*.whl
