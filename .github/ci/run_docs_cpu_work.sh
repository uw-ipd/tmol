#!/usr/bin/env bash
# Execute CPU tutorial cells and render documentation on a hosted runner.
set -euo pipefail

source .venv/bin/activate

echo "=== execute CPU tutorial outputs ==="
python .github/scripts/smoke_tutorial_notebooks.py --write

echo "=== build Sphinx documentation ==="
make -C docs html
