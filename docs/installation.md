# Installation

TMol requires Python 3.11+ and PyTorch. AtomWorks 3, RDKit, and OpenBabel are
included in the standard dependencies.

## GPU

Install the PyTorch build for your CUDA version, then select the matching TMol
wheel index. For PyTorch 2.14 and CUDA 13.2 on Linux:

```bash
python -m pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu132
python -m pip install "tmol==0.1.60+cu132torch2.14" --only-binary=tmol \
  --find-links https://uw-ipd.github.io/tmol/wheels/v0.1.60/cu132torch2.14/
```

Pre-built wheels include C++/CUDA extensions; installing them does not require
`nvcc`. A compatible NVIDIA driver is still required.

Start with a fresh virtual environment. When replacing an existing CPU or CUDA
PyTorch variant, add `--force-reinstall` to the PyTorch install command.

Each wheel page contains one CUDA/PyTorch combination. Pip selects the Python
and platform tags; it does **not** detect your GPU or choose a CUDA version.
The exact version qualifier and `--only-binary=tmol` prevent a CPU or source
fallback when the requested GPU wheel is unavailable.

Other combinations are listed in [GitHub Releases](https://github.com/uw-ipd/tmol/releases).
Use the corresponding `cuNNNtorchX.Y` page and version qualifier, or install a
wheel's download URL directly. A GitHub release's ordinary web page is not a
pip wheel index.

### CUDA 13 on Linux ARM64

PyTorch currently pins cuSPARSELt 0.8.0 or 0.8.1. NVIDIA's ARM64 wheels for
these versions contain the correct ARM64 library but label it internally as
`manylinux2014_sbsa`. Pip accepts the `aarch64` wheel filename at installation,
then `python -m pip check` reports that cuSPARSELt is unsupported.

After installing PyTorch and TMol, run the supplied metadata repair in the
same virtual environment:

```bash
curl -fLO https://github.com/uw-ipd/tmol/releases/download/v0.1.60/repair_cuda13_arm64_metadata.py
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

## CPU

From 0.1.60, PyPI carries CPU wheels for PyTorch 2.14 and Python 3.11–3.14 on
Linux x86-64, Linux aarch64, and Apple Silicon:

```bash
python -m pip install tmol
```

On Linux, install CPU-only PyTorch first to avoid downloading PyTorch's CUDA
libraries:

```bash
python -m pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cpu
python -m pip install tmol --only-binary=tmol
```

CPU wheels constrain the PyTorch minor version they were compiled against.
Plain pip installs a **CPU-only TMol build**, even if CUDA-enabled PyTorch is
already present. Use the GPU instructions above for CUDA scoring.

## Distribution channels

| Channel | Contents |
| --- | --- |
| PyPI | Standard CPU wheels and the source distribution. |
| GitHub Releases | CPU and CUDA wheels with explicit PyTorch/CUDA version qualifiers, plus source. |
| Versioned wheel pages | Links to GitHub wheels for one variant, with SHA-256 hashes. |

Use a virtual environment for each PyTorch/CUDA combination. Reinstall the
matching TMol wheel when changing PyTorch's minor version or CUDA variant.
TMol 0.1.59's PyPI source installer has a version mismatch; use 0.1.60 or a
0.1.59 wheel download URL.

<span id="from-source"></span>

## Build from source

Install the desired PyTorch build first. Source builds need a C++ compiler;
CUDA builds also need a matching CUDA toolkit with `nvcc`. Disable build
isolation so compilation uses the PyTorch that will load the extension:

```bash
python -m pip install "scikit-build-core>=0.10" "cmake>=3.24,<4" "pybind11>=2.12" ninja packaging
python -m pip install tmol --no-binary=tmol --no-build-isolation
```

Source installs compile locally. They do not download a substitute wheel.
Rebuild after changing PyTorch. For a CPU-only build:

```bash
python -m pip install tmol --no-binary=tmol --no-build-isolation \
  -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

For editable development:

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
python -m pip install --no-build-isolation -e ".[dev]"
```

See {doc}`Development <user_guide/development>` for compiler flags and JIT
compilation. Native Windows is not supported; use Linux or WSL2.

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
