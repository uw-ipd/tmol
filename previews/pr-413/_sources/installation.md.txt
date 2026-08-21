# Installation

TMol supports Python 3.11 and newer. It depends on PyTorch and ships custom
C++/CUDA extensions for scoring, packing, kinematics, and minimization kernels.

> - **Choose a path:** Use a release wheel for the shortest supported install,
>   or build from source when developing TMol or targeting an unavailable
>   platform combination.
> - **Next step:** Run the {doc}`Quickstart </quickstart>`, then choose a
>   {doc}`learning path </learning_paths>`.
> - **Development setup:** See {doc}`Development </user_guide/development>`.

## Pre-built Wheels

Pre-built wheels include ahead-of-time compiled extensions, so installing a
wheel does not require `nvcc`.

tmol uses two distribution channels:

- PyPI provides source distributions for `pip install tmol`.
- GitHub Releases provide pre-built CPU and GPU wheels.

The most deterministic install path is an explicit wheel URL from the
[GitHub Releases page](https://github.com/uw-ipd/tmol/releases):

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu132torch2.12-cp313-cp313-manylinux_2_28_x86_64.whl"
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

During a PyPI source-distribution build, TMol tries to fetch a matching
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
TMol can build CPU-only extensions:

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
libraries; TMol wheels do not bundle the PyTorch or NVIDIA runtime libraries.

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

The current Colab GPU runtime uses Python 3.12, PyTorch 2.11.0 with CUDA 12.8,
and commonly a Turing T4 (`sm_75`). TMol v0.1.47 provides a matching wheel
compiled for T4 (`sm_75`), A100 (`sm_80`), and L4 (`sm_89`) GPUs:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/v0.1.47/tmol-0.1.47+cu128torch2.11-cp312-cp312-manylinux_2_28_x86_64.whl"
```

The tutorial bootstrap installs this wheel directly and constrains pip to keep
Colab's active PyTorch. It stops with a clear compatibility error instead of
attempting a long source build when Python, PyTorch, or CUDA do not match.
Always confirm the active versions before installing an ABI-specific wheel URL.
