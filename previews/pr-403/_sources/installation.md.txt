# Installation

tmol supports Python 3.11 and newer. It depends on PyTorch and ships custom
C++/CUDA extensions for scoring, packing, kinematics, and minimization kernels.

## Pre-built Wheels

Pre-built wheels include ahead-of-time compiled extensions, so installing a
wheel does not require `nvcc`.

tmol uses two distribution channels:

- PyPI provides source distributions for `pip install tmol`.
- GitHub Releases provide pre-built CPU and GPU wheels.

The most deterministic install path is an explicit wheel URL from the GitHub
Releases page:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu130torch2.13-cp313-cp313-manylinux_2_28_x86_64.whl"
```

Install the matching PyTorch build first:

```bash
pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu132
```

Wheel tags select Python, PyTorch, and CUDA compatibility. For example,
`cp313` selects Python 3.13 and `+cu130torch2.13` selects the CUDA/PyTorch lane.
They do not replace the host C++ runtime.

## PyPI Source Distribution

The simple install is:

```bash
pip install tmol
```

During a PyPI source-distribution build, tmol tries to fetch a matching
pre-built wheel from GitHub Releases. If no compatible wheel exists, it builds
locally.

Useful environment variables:

- `TMOL_DISABLE_WHEEL_FETCH=1`: skip the pre-built lookup and build locally.
- `TMOL_FORCE_BUILD=1`: force the local build path.
- `TMOL_ENABLE_LOCAL_FETCH=1`: allow wheel fetch from a git checkout install.
- `TMOL_WHEEL_LOCAL_TAG=cu132torch2.12`: pin the wheel lane.
- `TMOL_WHEEL_RELEASE_TAG=vX.Y.Z`: override the GitHub release tag.
- `TMOL_WHEEL_RELEASE_BASE_URL=...`: use a release mirror.
- `TMOL_WHEEL_FETCH_RETRIES=2`: set HTTP retry attempts.
- `TMOL_WHEEL_FETCH_TIMEOUT_S=20`: set per-request timeout.
- `TMOL_WHEEL_FETCH_BACKOFF_S=1.5`: set retry backoff.

## From Source

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
pip install -e ".[dev]"
```

This builds C++/CUDA extensions through CMake. If no CUDA toolkit is available,
tmol can build CPU-only extensions:

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

The same CPU-only path is the normal macOS source install:

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

## Linux Runtime Notes

Release wheels use `manylinux_2_28` platform tags on `x86_64` and `aarch64`.
They require glibc 2.28 or newer. PyTorch supplies the matching CUDA shared
libraries; tmol wheels do not bundle the PyTorch or NVIDIA runtime libraries.

If `import tmol` fails with a `GLIBCXX_* not found` error, the host
`libstdc++` is too old for the wheel. Use one of these paths:

```bash
# Build against system libraries
TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .

# Or allow just-in-time extension compilation if nvcc is available
export TMOL_JIT_FALLBACK=1
```

Other fixes include loading a newer GCC module, installing
`conda-forge::libstdcxx-ng` and setting `LD_LIBRARY_PATH`, or running in a
recent container image.

Check the active Python, PyTorch, and CUDA environment with:

```bash
python -c "import sys, torch; print(f'Python {sys.version_info.major}.{sys.version_info.minor}, Torch {torch.__version__}, CUDA {torch.version.cuda}')"
```

## Google Colab

For Colab runtimes on a Turing T4 GPU, use the wheel lane that includes `sm_75`
support:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu128torch2.8-cp312-cp312-manylinux_2_28_x86_64.whl"
```

Confirm the current Colab Python and PyTorch versions before installing a
versioned wheel URL.
