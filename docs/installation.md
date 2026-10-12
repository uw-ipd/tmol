# Installation

TMol 0.1.64 is published on PyPI for Python 3.11–3.14 and PyTorch 2.11+.
`tmol` contains the Python API, data, and CPU kernels; optional `tmol-ops-cu12`
and `tmol-ops-cu13` packages add CUDA kernels. All three share the same version.

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

In a fresh environment, install CUDA-enabled PyTorch, then choose the matching
CUDA **major**. Use one of these tested combinations:

```bash
# CUDA 13 example
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu132
python -m pip install "tmol[cu13]==0.1.64" --only-binary=:all:

# CUDA 12 example
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "tmol[cu12]==0.1.64" --only-binary=:all:
```

The extras install `tmol-ops-cu13` or `tmol-ops-cu12` at the same version as
`tmol`. They contain only native operators and coexist with the CPU package.
Pip does not detect the GPU or replace an already installed CPU build of
PyTorch. A CUDA-enabled PyTorch build and a compatible NVIDIA driver are needed;
`nvcc` is unnecessary for wheel installs. These Torch pins are tested examples,
not a requirement to match the Torch version used to build TMol. An existing
Torch 2.11+ installation can be kept if its CUDA major matches the extra.

The 0.1.64 CUDA wheels use compressed device code that requires a driver from
CUDA 12.4 or later, **and** a driver compatible with the chosen PyTorch CUDA
runtime. NVIDIA documents this [compression restriction](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cmdoption-compress-mode).
Public-wheel checks passed on driver 595.71.05; an A100 with driver 535.129.03
failed with `device kernel image is invalid`. Older-driver compatibility-library
recovery has not been verified. `tmol.has_cuda()` reports compiled capability,
not driver compatibility.

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

Select a GPU runtime, then run the tutorial's setup cell. The shared bootstrap
installs `tmol[cu12]==0.1.64` or `tmol[cu13]==0.1.64` from PyPI according to
`torch.version.cuda`, keeps Colab's installed PyTorch, and requires binary wheels.
It no longer constructs wheel URLs for individual Python or PyTorch versions.
The Python, PyTorch, and driver requirements above still apply.

After installation, run the {doc}`Quickstart <quickstart>`. For an editable
development environment, see {doc}`Development <user_guide/development>`.
