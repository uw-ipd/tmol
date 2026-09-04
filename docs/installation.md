# Installation

TMol supports Python 3.11 and newer. It depends on PyTorch and ships custom
C++/CUDA extensions for scoring, packing, kinematics, and minimization kernels.

> - **Choose a path:** Use a release wheel for the shortest supported install,
>   or build from source when developing TMol or targeting an unavailable
>   platform combination.
> - **Next step:** Run the {doc}`Quickstart </quickstart>`, then choose an
>   {doc}`interactive example </examples_index>` or a concise
>   {doc}`workflow </workflows/index>`.
> - **Development setup:** See {doc}`Development </user_guide/development>`.

## Pre-built Wheels

Pre-built wheels include ahead-of-time compiled extensions, so installing a
wheel does not require `nvcc`.

TMol uses two distribution channels:

- PyPI provides source distributions for `pip install tmol`.
- GitHub Releases provide pre-built CPU and GPU wheels.

The most deterministic install path is an explicit wheel URL from the
[GitHub Releases page](https://github.com/uw-ipd/tmol/releases):

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu132torch2.14-cp313-cp313-manylinux_2_28_x86_64.whl"
```

Install the matching PyTorch build first:

```bash
pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu132
```

Wheel tags select Python, PyTorch, and CUDA compatibility. For example,
`cp313` selects Python 3.13 and `+cu132torch2.14` selects the CUDA/PyTorch lane.
CPU wheels are also PyTorch-minor-specific: `+cputorch2.14` selects the CPU
extension built against PyTorch 2.14. PyTorch 2.13 wheels remain available as
the corresponding `torch2.13` lanes. TMol wheels do not replace the host C++
runtime.

Release builds provide CPU wheels for Linux x86-64, Linux aarch64, and Apple
Silicon. For example, after installing PyTorch 2.14, an Apple Silicon wheel is:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cputorch2.14-cp313-cp313-macosx_14_0_arm64.whl"
```

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
- `TMOL_WHEEL_LOCAL_TAG=cu132torch2.14`: pin the wheel lane.
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

This builds C++/CUDA extensions through CMake. To request a CPU-only build
(the normal source build on Apple Silicon), use:

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

CMake also falls back to CPU-only when it cannot find a CUDA compiler. This
path needs CMake and a compatible C++ compiler, but no `nvcc`. Alternatively,
CPU kernels can be compiled on first use:

```bash
TMOL_USE_JIT=1 python -c "import tmol; print(tmol.__version__)"
```

CPU-only JIT needs a C++ compiler and `ninja`; `nvcc` is required only for
CUDA kernels.

CPU source builds and release wheels are tested on Linux x86-64, Linux
aarch64, and Apple Silicon. Native Windows is not currently supported; use a
Linux environment such as WSL2.

## Linux Runtime Notes

Linux release wheels use `manylinux_2_28` platform tags on `x86_64` and
`aarch64`. They require glibc 2.28 or newer. Apple Silicon wheels use
`macosx_14_0_arm64`, matching the PyTorch 2.13 and 2.14 deployment target. PyTorch
supplies the matching shared libraries; TMol wheels do not bundle the PyTorch
or NVIDIA runtime libraries.

If `import tmol` fails with a `GLIBCXX_* not found` error, the host
`libstdc++` is too old for the wheel. Use one of these paths:

```bash
# Build against system libraries
TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .

# Or allow just-in-time extension compilation (CPU-only needs no nvcc)
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

Colab GPU runtimes currently use PyTorch 2.11.0 with CUDA 12.8 and may provide
Python 3.12 or 3.13. TMol v0.1.54 provides separate Python-ABI wheels compiled
for T4 (`sm_75`), A100 (`sm_80`), and L4 (`sm_89`) GPUs. For Python 3.13:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/v0.1.54/tmol-0.1.54+cu128torch2.11-cp313-cp313-manylinux_2_28_x86_64.whl"
```

The tutorial bootstrap selects the wheel matching the runtime's Python ABI and
constrains pip to keep Colab's active PyTorch. It stops with a clear
compatibility error instead of attempting a long source build when Python,
PyTorch, or CUDA do not match. Always confirm the active versions before
installing an ABI-specific wheel URL.
