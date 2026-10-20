# Installation

Install TMol from PyPI in a virtual environment. Choose CPU support or a CUDA
backend matching your PyTorch installation:

| Use | Install |
| --- | --- |
| CPU, including Apple Silicon | `python -m pip install tmol` |
| GPU with CUDA 12-enabled PyTorch | `python -m pip install "tmol[cu12]"` |
| GPU with CUDA 13-enabled PyTorch | `python -m pip install "tmol[cu13]"` |

`tmol` contains the Python API, data, and CPU kernels. The GPU extras add
`tmol-ops-cu12` or `tmol-ops-cu13` at the **same version** as the base package.
GPU installations also support CPU execution. You do not need a separate TMol
wheel for each PyTorch minor version or CUDA minor version.

## Requirements

- Python 3.11–3.14 and PyTorch 2.11 or newer.
- Linux `x86_64` or `aarch64` with glibc 2.28 or newer, or Apple Silicon with
  macOS 14 or newer. Native Windows and Intel Mac wheels are not provided;
  Windows users can use Linux through WSL2.
- For GPU execution: Linux, an NVIDIA GPU with compute capability 7.5 or newer,
  CUDA-enabled PyTorch, and a compatible NVIDIA driver. See
  [driver compatibility](#cuda-driver-errors).

Pip selects the wheel for your Python version and platform. Wheel installation
needs no compiler or local CUDA toolkit (`nvcc`). Add `--only-binary=:all:` to
an install command if you want it to fail rather than build any dependency
from source when a compatible wheel is unavailable.

## CPU-only PyTorch

On Linux, install CPU-only PyTorch first to avoid downloading its CUDA libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install tmol
```

## GPU installation

First install CUDA-enabled PyTorch using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
Choose a build supported by your NVIDIA driver, then inspect its CUDA runtime:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Choose the TMol extra from `torch.version.cuda`: `12.x` uses `cu12`, and `13.x`
uses `cu13`. For example, with CUDA 13-enabled PyTorch:

```bash
python -m pip install "tmol[cu13]"
```

Use `"tmol[cu12]"` for CUDA 12. If `torch.version.cuda` is `None`, install a
CUDA-enabled PyTorch build first. Pip does not detect your GPU or replace an
installed CPU build of PyTorch just because a TMol GPU extra is requested.
The CUDA version shown by `nvidia-smi` describes driver capability; it does not
select the TMol backend. TMol selects operators using PyTorch's CUDA major.

## Verify the installation

```python
import torch
import tmol

print("TMol:", tmol.__version__)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("TMol CUDA operators:", tmol.has_cuda())
print("GPU available:", torch.cuda.is_available())
```

For CPU execution, use `torch.device("cpu")`. On Apple silicon this is also
the device to use: poses stay on the CPU, and the most expensive operations --
the pairwise score terms, the packer's interaction graph, and simulated
annealing -- run on the GPU automatically, with their inputs copied over as
needed. Set `TMOL_MPS_OFFLOAD=0` to keep everything on the CPU. Do not place
poses on `torch.device("mps")`. For GPU execution with CUDA, both CUDA
checks must be true. Applications that require a GPU should fail explicitly:

```python
if not (tmol.has_cuda() and torch.cuda.is_available()):
    raise RuntimeError("Install matching TMol/PyTorch CUDA packages and check GPU access")
device = torch.device("cuda")
```

`tmol.has_cuda()` reports compiled capability, not a successful kernel launch or
driver compatibility. Run the {doc}`Quickstart <quickstart>` on your chosen
device to check structure loading, scoring, gradients, and refinement.

## Upgrade or change backends

Upgrade using the same extra so pip updates the base and operators together:

```bash
python -m pip install --upgrade "tmol[cu13]"  # or tmol[cu12], or tmol for CPU
python -m pip check
```

When changing PyTorch's CUDA major, remove the old operator package and install
the matching extra. To return to CPU-only TMol, uninstall both optional
operator packages; keep `tmol` installed. Changing PyTorch minor does not require
rebuilding TMol, provided its runtime meets the requirements above.

For reproducible environments, pin your chosen versions in a requirements or
lock file. The ordinary install commands above select the latest compatible
release.

## Use TMol in another package

Declare `tmol` as a dependency for CPU support. Applications can expose `cu12`
and `cu13` extras depending on `tmol[cu12]` and `tmol[cu13]`, respectively.
Set a minimum TMol version when your application requires a particular API.
Users still choose a compatible PyTorch build; retain runtime capability checks
when GPU execution is required.

## Google Colab and notebooks

Select a GPU runtime in Colab, then run the tutorial's setup cell. The shared
bootstrap installs a tested TMol release with the matching `cu12` or `cu13`
extra and preserves Colab's installed PyTorch. The requirements above still
apply. No Python- or PyTorch-specific wheel URL is needed.

For local tutorials, use an editable {doc}`development installation
<user_guide/development>` with the `docs` extra (`-e ".[docs]"`) for notebook
dependencies and viewers. This also makes the checkout’s fixtures available;
release wheels omit test data. Colab setup downloads the fixtures automatically.

<span id="from-source"></span>

## Build from source

Source builds require PyTorch 2.14+ **build headers**, a C++17 compiler, CMake,
and Ninja. The resulting production binary supports PyTorch 2.11+ at runtime.
From a repository checkout, build the CPU package with:

```bash
python -m pip install "torch>=2.14" "scikit-build-core>=0.10" "cmake>=3.24,<4" "pybind11>=2.12" ninja packaging
python -m pip install --no-build-isolation -e .
```

Build optional GPU operators with a CUDA toolkit of the requested major and
CUDA-enabled build-time PyTorch:

```bash
python -m pip install --no-build-isolation ./packages/cu13
# Or, with a CUDA 12 toolkit:
python -m pip install --no-build-isolation ./packages/cu12
```

The toolkit minor need not match PyTorch's CUDA minor. See
{doc}`Development <user_guide/development>` for compiler requirements,
test extensions, JIT compilation, and build flags.

<span id="linux-runtime-notes"></span>

## Troubleshooting

### CUDA driver errors

The CUDA wheels contain compressed device code, which requires a driver from
CUDA 12.4 or later, as well as a driver compatible with your PyTorch CUDA
runtime. See NVIDIA's [compression restriction](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cmdoption-compress-mode).
An older driver can expose the GPU to PyTorch but fail when TMol launches a
kernel, for example with `device kernel image is invalid`. Update the driver
or use a compatible system. Compatibility-library recovery on older drivers
has not been verified.

### Linux ARM64 dependency metadata

Some PyTorch builds pin NVIDIA cuSPARSELt wheels whose internal platform tag
is `manylinux2014_sbsa` instead of `aarch64`. Installation succeeds, but
`python -m pip check` reports that cuSPARSELt is unsupported. If you encounter
this error, use the metadata repair supplied with the
[TMol release](https://github.com/uw-ipd/tmol/releases):

```bash
curl -fLO https://github.com/uw-ipd/tmol/releases/download/v0.1.64/repair_cuda13_arm64_metadata.py
python repair_cuda13_arm64_metadata.py
python -m pip check
```

This deliberately versioned repair verifies known library hashes and ARM64
headers before correcting metadata; it leaves native code and versions intact
and refuses unknown files. Repeat it after reinstalling an affected NVIDIA
package. Do not independently upgrade cuSPARSELt against PyTorch's exact pins.

### C++ runtime errors

If importing TMol reports `GLIBCXX_* not found`, use a newer system C++ runtime
or a recent container image. A source build against your system libraries is
another option; follow the build instructions above. JIT development also needs
a source checkout and build tools; setting an environment variable alone does
not repair an installed wheel.
