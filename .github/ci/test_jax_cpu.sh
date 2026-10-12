#!/usr/bin/env bash
# Install the pure wheel without Torch and run JAX tests outside the checkout.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
JAX_DIST=${JAX_DIST:-jax-dist}

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
python -m venv "${work}/env"
"${work}/env/bin/python" -m pip install --upgrade pip
"${work}/env/bin/python" -m pip install pytest "${JAX_SPEC:-jax>=0.11.2}"
"${work}/env/bin/python" -m pip install "${JAX_DIST}"/tmol_jax-*.whl

mkdir -p "${work}/source/tests" "${work}/source/tmol/tests/data/pdb"
cp -R tests/jax "${work}/source/tests/jax"
cp tmol/tests/data/pdb/1ubq.pdb "${work}/source/tmol/tests/data/pdb/1ubq.pdb"
mkdir -p "${work}/source/tmol/database/default/scoring"
cp tmol/database/default/scoring/hbond.yaml \
  "${work}/source/tmol/database/default/scoring/hbond.yaml"

version=$("${work}/env/bin/python" -I -c \
  'from importlib.metadata import version; print(version("tmol-jax"))')
(
  cd "${work}/source"
  JAX_PLATFORMS=cpu "${work}/env/bin/python" -I -m pytest -ra tests/jax
  JAX_PLATFORMS=cpu "${work}/env/bin/python" -I \
    "${GITHUB_WORKSPACE}/scripts/smoke_installed_jax.py" \
    --pdb tmol/tests/data/pdb/1ubq.pdb --version "${version}" \
    --device cpu --expect-no-torch
)
