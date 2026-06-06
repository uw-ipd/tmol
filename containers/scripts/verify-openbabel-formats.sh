#!/usr/bin/env bash
# Fail fast when openbabel-wheel format plugins did not load (missing libXrender, etc.).
set -euo pipefail

python - <<'PY'
from openbabel import pybel

n = len(pybel.informats)
if n < 100:
    raise SystemExit(
        f"openbabel-wheel loaded only {n} input formats (expected >100).\n"
        "Format plugins likely failed to load; check for libXrender.so.1 errors.\n"
        "Fix: rebuild the dev/CI image so containers/scripts/install-openbabel-runtime-deps.sh ran,\n"
        "  e.g. containers/apptainer/build-tmol-sif.sh --deploy-ci"
    )
print(f"Open Babel OK ({n} input formats)")
PY
