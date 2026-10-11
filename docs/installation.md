# Installation

TMol requires Python 3.11+ and PyTorch 2.11+. The upcoming 0.1.64 release
separates the Python API, data, and CPU kernels from optional CUDA kernels.
These instructions describe that release; see the versioned documentation for
0.1.63 and earlier packages.

## CPU

```bash
python -m pip install "tmol==0.1.64" --only-binary=:all:
```

On Linux, install CPU-only PyTorch first to avoid downloading its CUDA libraries:

```bash
python -m pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "tmol==0.1.64" --only-binary=:all:
```

## GPU

Install CUDA-enabled PyTorch, then choose the matching CUDA **major**:

```bash
# CUDA 13 example
python -m pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu132
python -m pip install "tmol[cu13]==0.1.64" --only-binary=:all:

# CUDA 12 example
python -m pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "tmol[cu12]==0.1.64" --only-binary=:all:
```

The extras install `tmol-ops-cu13` or `tmol-ops-cu12` at the same version as
`tmol`. They contain only native operators and coexist with the CPU package.
Pip does not detect the GPU or replace an already installed CPU build of
PyTorch. A CUDA-enabled PyTorch build and a compatible NVIDIA driver are needed;
`nvcc` is unnecessary for wheel installs.

TMol selects the operator package matching `torch.version.cuda`. Without an
operator package it uses its CPU backend. Check both compiled capabilities and
hardware availability before selecting a device:

```python
import tmol
import torch

can_use_gpu = tmol.has_cuda() and torch.cuda.is_available()
device = torch.device("cuda" if can_use_gpu else "cpu")
```

For a downstream package, declare `tmol>=0.1.64` for CPU support or
`tmol[cu13]>=0.1.64` for a CUDA 13 installation. Applications supporting both
CUDA families can expose their own `cu12` and `cu13` extras. Keep runtime
capability checks even when an extra is requested.

## Binary compatibility

The native bindings use Python autograd and PyTorch's stable C ABI targeted
at 2.11. Tensor exchange uses the stable Python interop API when available
(PyTorch 2.14+), and a stable dispatcher fallback on older runtimes. Both preserve tensor storage
and autograd ownership. A TMol wheel no longer pins a PyTorch minor version. Python, platform, and CUDA-major wheel variants remain; these
are not Python `abi3` wheels. Changing CUDA major requires the corresponding
operator package. Changing PyTorch minor does not require rebuilding TMol.

### CUDA on Linux ARM64

Affected PyTorch builds pin cuSPARSELt 0.7.1 (CUDA 12) or 0.8.0/0.8.1
(CUDA 13). NVIDIA's ARM64 wheels for
these versions contain the correct ARM64 library but label it internally as
`manylinux2014_sbsa`. Pip accepts the `aarch64` wheel filename at installation,
then `python -m pip check` reports that cuSPARSELt is unsupported.

After installing PyTorch and TMol, run the supplied metadata repair in the
same virtual environment:

```bash
curl -fLO https://github.com/uw-ipd/tmol/releases/download/v0.1.64/repair_cuda13_arm64_metadata.py
python repair_cuda13_arm64_metadata.py
python -m pip check
```

The script checks the original NVIDIA library's SHA-256 and ARM64 ELF header,
then corrects only the platform tag and its checksum in `RECORD`. It preserves
the native code, package version and dependency requirements, and refuses
unrecognized files. Other platforms and already-corrected metadata are left
alone. Upgrading cuSPARSELt independently conflicts with PyTorch's exact pins.

The release installation tests use this same repair before checking all
dependencies, loading TMol's native extensions, scoring a protein and checking
its gradients. Repeat the repair if reinstalling the affected NVIDIA package.

<span id="from-source"></span>

## Build from source

Build headers require PyTorch 2.14+, a C++17 compiler (C++20 for test/JIT extensions with recent PyTorch), CMake, and Ninja. The
resulting production binary supports PyTorch 2.11+. CPU builds are the default:

```bash
python -m pip install "torch>=2.14" "scikit-build-core>=0.10" "cmake>=3.24,<4" "pybind11>=2.12" ninja packaging
python -m pip install --no-build-isolation -e ".[dev]"
```

From a repository checkout, build the optional operators with a CUDA toolkit
of the requested major and CUDA-enabled build-time PyTorch:

```bash
python -m pip install --no-build-isolation ./packages/cu13
# Or, with a CUDA 12 toolkit:
python -m pip install --no-build-isolation ./packages/cu12
```

The build-time PyTorch CUDA minor need not match `nvcc`: TMol compiles its own
kernels and uses the stable C interface. Test-only and JIT extensions still
compile against the local PyTorch C++ API; rebuild those when changing PyTorch.
See {doc}`Development <user_guide/development>` for test extensions and compiler
flags. Native Windows is unsupported; use Linux or WSL2.

<span id="linux-runtime-notes"></span>

## Runtime compatibility

Linux release wheels use `manylinux_2_28` platform tags on `x86_64` and
`aarch64`. They require glibc 2.28 or newer. Apple Silicon wheels use
`macosx_14_0_arm64`, matching the PyTorch 2.14 deployment target. PyTorch
supplies the matching shared libraries; TMol wheels do not bundle the PyTorch
or NVIDIA runtime libraries.

If `import tmol` fails with a `GLIBCXX_* not found` error, the host
`libstdc++` is too old for the wheel. Use one of these paths:

```bash
# Build against system libraries
python -m pip install --no-build-isolation -e .

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

The tutorial bootstrap supports PyTorch 2.11.0, CUDA 12.8, and Python 3.12 or
3.13. TMol v0.1.59 provides separate Python-ABI wheels compiled
for T4 (`sm_75`), A100 (`sm_80`), and L4 (`sm_89`) GPUs. For Python 3.13:

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/v0.1.59/tmol-0.1.59+cu128torch2.11-cp313-cp313-manylinux_2_28_x86_64.whl"
```

The tutorial bootstrap selects the wheel matching the runtime's Python ABI and
constrains pip to keep Colab's active PyTorch. It stops with a clear
compatibility error instead of attempting a long source build when Python,
PyTorch, or CUDA do not match. Always confirm the active versions before
installing an ABI-specific wheel URL.

After installation, run the {doc}`Quickstart <quickstart>`. For an editable
development environment, see {doc}`Development <user_guide/development>`.
