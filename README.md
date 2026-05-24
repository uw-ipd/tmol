# Tmol

`tmol` (TensorMol) is a GPU-accelerated reimplementation of the Rosetta molecular modeling energy function (`beta_nov2016_cart`) in PyTorch with custom C++/CUDA kernels. It computes energies and derivatives for protein structures and supports gradient-based minimization, enabling ML models to incorporate biophysical scoring during training or to refine predicted structures with Rosetta's experimentally validated energy function.

Full documentation: [tmol Wiki](https://github.com/uw-ipd/tmol/wiki/DevHome)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Integrations](#integrations)
- [Citation](#citation)
- [Development](#development)

## Installation

### Pre-built wheels (recommended)

Pre-built wheels ship with **ahead-of-time (AOT) compiled** C++/CUDA extensions, so install does not require `nvcc`.

tmol uses two channels:

- **PyPI**: source distribution (`sdist`) for `pip install tmol`
- **GitHub Releases**: prebuilt CPU/GPU wheels

Use the mode that fits your needs:

- **Deterministic binary install (canonical):** direct wheel URL or local `--find-links`.
- **Convenience install:** `pip install tmol` (best-effort wheel auto-fetch, source-build fallback).
- **Forced source build:** disable fetch and compile locally.

CI currently uploads these wheel variants to [GitHub Releases](https://github.com/uw-ipd/tmol/releases):

- GPU wheels (Linux `x86_64` and `aarch64`) for:
  - Python `cp312`, `cp313`, `cp314`
  - Torch/CUDA tags:
    - `+cu129torch2.8`
    - `+cu130torch2.9`
    - `+cu131torch2.10`
    - `+cu131torch2.11`
    - `+cu132torch2.12`
  - plus Colab override wheel on `x86_64`: `+cu128torch2.10`
- CPU wheels (Linux `x86_64`) for:
  - Python `cp312`, `cp313`, `cp314`
  - local version tag `+cpu`

Wheel filename format:

```text
tmol-{VERSION}+{LOCAL_TAG}-cp{PYTAG}-cp{PYTAG}-manylinux_2_28_{ARCH}.whl
```

Examples:

- `tmol-0.1.22+cu132torch2.12-cp313-cp313-manylinux_2_28_x86_64.whl`
- `tmol-0.1.22+cpu-cp314-cp314-manylinux_2_28_x86_64.whl`

> [!TIP]
> CUDA wheels are forward-compatible within a major family (e.g. `cu132` wheels run on appropriate CUDA 13.x driver stacks).

### System requirements (Linux wheels)

Pre-built Linux wheels are built for **manylinux_2_28** (glibc ≥ 2.28, typical minimum: **Ubuntu 20.04**, RHEL/CentOS 8+, or equivalent).

Wheel tags such as `cp312` and `+cu130torch2.9` select **Python**, **PyTorch**, and **CUDA** — they do not override your system's C++ runtime (`libstdc++`). If `import tmol` fails with `GLIBCXX_3.4.xx not found`, your **libstdc++ is older than the wheel was built for** (not a wrong CUDA wheel tag).

**On older HPC clusters or minimal Linux images:**

```bash
# Build against your system libraries (recommended)
TMOL_DISABLE_WHEEL_FETCH=1 pip install -e .

# Or allow JIT compile at import if nvcc is available
export TMOL_JIT_FALLBACK=1
```

Other workarounds: load a newer GCC module, `conda install -c conda-forge libstdcxx-ng` and set `LD_LIBRARY_PATH`, or use a recent container image.

Check your environment:

```bash
python -c "import sys, torch; print(f'Python {sys.version_info.major}.{sys.version_info.minor}, Torch {torch.__version__}, CUDA {torch.version.cuda}')"
```

Install torch first so it matches your chosen wheel tag:

```bash
pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu132
# or e.g. cu131/cu130/cu129/cu128 depending on the wheel you pick
```

#### Install by direct wheel URL (recommended)

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu132torch2.12-cp313-cp313-manylinux_2_28_x86_64.whl"
```

#### Auto-fetch matching wheel, fallback to source build

tmol supports a FlashAttention-style bootstrap when installing from PyPI `sdist`:

1. During wheel build, tmol tries to download a matching prebuilt wheel from GitHub Releases.
2. If no match is found, tmol falls back to local source build.

In pip's default PEP517 isolated build environment, tmol performs **best-effort auto-detection** of CUDA/Torch lane. For deterministic behavior, pin the lane explicitly.

Simplest command (safe default):

```bash
pip install tmol
```

For deterministic wheel auto-fetch in isolated builds, pin the lane:

```bash
TMOL_WHEEL_LOCAL_TAG=cu132torch2.12 pip install "tmol==X.Y.Z"
```

If you want detection based on the currently active runtime environment instead, you can disable build isolation:

```bash
pip install --no-build-isolation "tmol==X.Y.Z"
```

Install a specific release version:

```bash
pip install "tmol==X.Y.Z"
```

If auto-detection picks the wrong wheel variant, pin the exact local tag:

```bash
TMOL_WHEEL_LOCAL_TAG=cu132torch2.12 \
pip install "tmol==X.Y.Z"
```

Useful toggles:

- `TMOL_DISABLE_WHEEL_FETCH=1`: skip prebuilt lookup and always build locally.
- `TMOL_FORCE_BUILD=1`: same as above (explicit force-local-build path).
- `TMOL_ENABLE_LOCAL_FETCH=1`: allow fetch even from a git checkout (`pip install .`).
- `TMOL_WHEEL_RELEASE_TAG=vX.Y.Z`: override GitHub release tag.
- `TMOL_WHEEL_RELEASE_BASE_URL=...`: override release base URL (mirrors/internal hosting).
- `TMOL_WHEEL_FETCH_RETRIES=2`: number of retry attempts after the first failed request.
- `TMOL_WHEEL_FETCH_TIMEOUT_S=20`: HTTP timeout in seconds per request.
- `TMOL_WHEEL_FETCH_BACKOFF_S=1.5`: linear backoff multiplier between retries.

#### Install from a local wheel cache (`--find-links`)

```bash
# 1) Download wheel files for your environment into ./wheels
mkdir -p wheels
# e.g. use browser/curl/wget from the release page

# 2) Install from local directory only
pip install --no-index --find-links ./wheels "tmol==X.Y.Z+cu132torch2.12"
```

#### CPU-only install

```bash
pip install "tmol @ https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
```

The CPU wheel works with CPU-only or CUDA torch installs; CUDA ops in tmol are unavailable.

### From PyPI sdist (source-build baseline)

By default, `pip install tmol` installs from PyPI `sdist`. tmol applies the auto-fetch safety policy described above and otherwise builds locally.

To force local source build explicitly:

```bash
TMOL_DISABLE_WHEEL_FETCH=1 pip install tmol
```

For dev extras:

```bash
TMOL_DISABLE_WHEEL_FETCH=1 pip install "tmol[dev]"
```

> [!NOTE]
> Current CI publishes `sdist` to PyPI and prebuilt wheels to GitHub Releases.
> If you need deterministic binary selection, use direct wheel URL or local `--find-links`.

### From source

```bash
git clone https://github.com/uw-ipd/tmol.git && cd tmol
pip install -e ".[dev]"   # builds extensions via CMake (CUDA auto-detected)
```

If you don't have a CUDA toolkit, the build automatically falls back to CPU-only extensions. You can also force a CPU-only build explicitly:

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

For macOS, install from source (CPU-only build):

```bash
pip install -e . -Ccmake.define.TMOL_ENABLE_CUDA=OFF
```

## Usage

### Quick start

```python
import tmol

# Load a structure
pose_stack = tmol.pose_stack_from_pdb("1ubq.pdb")

# Score it
sfxn = tmol.beta2016_score_function(pose_stack.device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
print(scorer(pose_stack.coords))
```

### Minimization

```python
cart_sfxn_network = tmol.cart_sfxn_network(sfxn, pose_stack)
optimizer = tmol.lbfgs_armijo(cart_sfxn_network.parameters())

def closure():
    optimizer.zero_grad()
    E = cart_sfxn_network().sum()
    E.backward()
    return E

optimizer.step(closure)
```

### Save output

```python
tmol.write_pose_stack_pdb(pose_stack, "output.pdb")
```

### Verify installation

```python
import tmol
print(f"tmol {tmol.__version__} loaded successfully")
```

## Integrations

### RosettaFold2

Install tmol into your RF2 environment:

```bash
cd <tmol repo root>
pip install -e .
```

```python
# RF2 -> tmol
seq, xyz, chainlens = rosettafold2_model.infer(sequence)
pose_stack = tmol.pose_stack_from_rosettafold2(seq[0], xyz[0], chainlens[0])

# tmol -> RF2
xyz = tmol.pose_stack_to_rosettafold2(...)
```

> [!NOTE]
> Tested on Ubuntu 20.04. Other platforms should work but are not yet verified.

> [!WARNING]
> Call `torch.set_grad_enabled(True)` before using the tmol minimizer, since RF2 disables gradients during inference by default.

### OpenFold

```python
output = openfold_model.infer(sequences)
pose_stack = tmol.pose_stack_from_openfold(output)
```

## Citation

If you use tmol in your work, please cite:

> Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg, David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, *tmol: a GPU-accelerated, PyTorch implementation of Rosetta's relax protocol*, (manuscript in preparation)

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for building from source, running tests, extension loading (AOT vs JIT), CI, containers, and contributing guidelines.
