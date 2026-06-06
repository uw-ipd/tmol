#!/usr/bin/env bash
# Runtime libraries for openbabel-wheel format plugins (mol2, smiles, MMFF94, …).
#
# openbabel-wheel ships .so plugins that link against libXrender (and libX11/libXext).
# Headless NGC PyTorch images do not include these; without them most format plugins
# fail to load and pybel.informats is nearly empty.
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    libxrender1
ldconfig
rm -rf /var/lib/apt/lists/*

for lib in libXrender.so.1 libX11.so.6 libXext.so.6; do
    if ! ldconfig -p | grep -q "${lib}"; then
        echo "ERROR: ${lib} not visible to the dynamic linker after apt install." >&2
        exit 1
    fi
done

echo "Open Babel X11 runtime libraries installed."
