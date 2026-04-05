# Tmol

`tmol` (TensorMol) is a GPU-accelerated reimplementation of the Rosetta molecular modeling energy function (`beta_nov2016_cart`) in PyTorch with custom C++/CUDA/Metal kernels. It computes energies and derivatives for protein structures and supports gradient-based minimization, enabling ML models to incorporate biophysical scoring during training or to refine predicted structures with Rosetta's experimentally validated energy function.

tmol runs on **NVIDIA GPUs** (CUDA), **Apple Silicon Macs** (MPS / Metal), and **CPU**.

Full documentation: [tmol Wiki](https://github.com/uw-ipd/tmol/wiki/DevHome)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Integrations](#integrations)
- [Citation](#citation)
- [Development](#development)

## Installation

### Apple Silicon / MPS (macOS)

tmol runs natively on Apple Silicon (M1/M2/M3/M4) via PyTorch's Metal Performance Shaders (MPS) backend. No CUDA toolkit or NVIDIA GPU is needed.

> [!IMPORTANT]
> MPS support is maintained in the **[fnachon/tmol](https://github.com/fnachon/tmol)** fork.
> The upstream [uw-ipd/tmol](https://github.com/uw-ipd/tmol) repository targets NVIDIA GPUs (CUDA/Linux).
> Use `https://github.com/fnachon/tmol` for Apple Silicon.

**Requirements:**
- macOS 13.0 (Ventura) or later
- Apple Silicon Mac (M-series)
- PyTorch ≥ 2.0 with MPS support (`torch.backends.mps.is_available()` returns `True`)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.10+

**Install from source (MPS):**

```bash
# Install PyTorch with MPS support (ships in the standard macOS wheel)
pip install torch

# Clone the MPS-enabled fork
git clone https://github.com/fnachon/tmol.git && cd tmol
pip install -e ".[dev,mps]"
```

**Verify MPS is working:**

```python
import torch
print(torch.backends.mps.is_available())  # must be True

import tmol
pose_stack = tmol.pose_stack_from_pdb("1ubq.pdb", device=torch.device("mps"))
sfxn = tmol.beta2016_score_function(torch.device("mps"))
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
print(scorer(pose_stack.coords))
```

> [!NOTE]
> The MPS backend uses Apple's unified memory architecture — CPU and GPU share the same physical RAM — so there is no host↔device copy overhead. All energy terms, gradients, and minimization work identically to CUDA.

> [!TIP]
> Run the MPS smoke tests to confirm everything is wired up:
> ```bash
> pytest tmol/tests/test_mps.py -v
> ```

---

### Pre-built wheels (Linux / NVIDIA GPU only)

Pre-built wheels ship with **ahead-of-time (AOT) compiled** C++/CUDA extensions — no `nvcc` or CUDA toolkit needed at install time.
MPS users should install [from source](#from-source) using the [fnachon/tmol](https://github.com/fnachon/tmol) fork.

Wheels are available for Linux x86_64. Pick the one matching your **PyTorch version** and **CXX11 ABI**:

<details>
<summary><b>Which ABI do I have?</b></summary>

```bash
python -c "import torch; print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
```

| Result  | Typical source                           | Wheel suffix        |
|---------|------------------------------------------|---------------------|
| `True`  | NGC container, conda, source-built torch | `cxx11abiTRUE`      |
| `False` | `pip install torch` on bare metal        | `cxx11abiFALSE`     |

The ABI must match because C++ extensions are linked against PyTorch's C++ standard library. A mismatch causes segfaults or missing-symbol errors. See [flash-attention#457](https://github.com/Dao-AILab/flash-attention/issues/457) for more background.

</details>

**x86_64 (Linux):**

| PyTorch | Python | CUDA | ABI   | Wheel tag                              |
|---------|--------|------|-------|----------------------------------------|
| 2.8     | 3.12   | 12.6 | TRUE  | `+cu126torch2.8cxx11abiTRUE`          |
| 2.8     | 3.12   | 12.6 | FALSE | `+cu126torch2.8cxx11abiFALSE`         |
| 2.9     | 3.12   | 13.0 | TRUE  | `+cu130torch2.9cxx11abiTRUE`          |
| 2.9     | 3.12   | 12.6 | FALSE | `+cu126torch2.9cxx11abiFALSE`         |
| 2.10    | 3.12   | 13.1 | TRUE  | `+cu131torch2.10cxx11abiTRUE`         |
| 2.10    | 3.12   | 12.6 | FALSE | `+cu126torch2.10cxx11abiFALSE`        |
| 2.8     | 3.10   | 12.6 | TRUE  | `+cu126torch2.8cxx11abiTRUE`          |
| 2.8     | 3.10   | 12.6 | FALSE | `+cu126torch2.8cxx11abiFALSE`         |
| 2.9     | 3.10   | 12.6 | TRUE  | `+cu126torch2.9cxx11abiTRUE`          |
| 2.9     | 3.10   | 12.6 | FALSE | `+cu126torch2.9cxx11abiFALSE`         |
| 2.10    | 3.10   | 12.6 | TRUE  | `+cu126torch2.10cxx11abiTRUE`         |
| 2.10    | 3.10   | 12.6 | FALSE | `+cu126torch2.10cxx11abiFALSE`        |

**ARM64 / aarch64 (Linux, e.g., Grace Hopper, Jetson):**

| PyTorch | Python | CUDA | ABI   | Wheel tag                              |
|---------|--------|------|-------|----------------------------------------|
| 2.8     | 3.12   | 12.6 | TRUE  | `+cu126torch2.8cxx11abiTRUE`          |
| 2.9     | 3.12   | 13.0 | TRUE  | `+cu130torch2.9cxx11abiTRUE`          |
| 2.10    | 3.12   | 13.1 | TRUE  | `+cu131torch2.10cxx11abiTRUE`         |
| 2.8     | 3.10   | 12.6 | TRUE  | `+cu126torch2.8cxx11abiTRUE`          |
| 2.9     | 3.10   | 12.6 | TRUE  | `+cu126torch2.9cxx11abiTRUE`          |
| 2.10    | 3.10   | 12.6 | TRUE  | `+cu126torch2.10cxx11abiTRUE`         |

> [!NOTE]
> Python 3.10 wheels use `cp310` in the filename; Python 3.12 wheels use `cp312`. pip automatically selects the correct one for your Python version. Python 3.11 users can install from the source distribution (requires nvcc).

> [!TIP]
> CUDA wheels are **forward-compatible** within a major version: a `cu124` wheel works on any CUDA 12.x driver >= 12.4. You do not need an exact CUDA version match.

Check your environment:

```bash
python -c "import sys; import torch; print(f'Python: {sys.version_info.major}.{sys.version_info.minor}, PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

Install from [GitHub Releases](https://github.com/uw-ipd/tmol/releases):

```bash
# Direct URL (replace RELEASE_TAG and WHEEL_FILENAME):
pip install https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/WHEEL_FILENAME.whl

# Or use --find-links to let pip resolve by version:
pip install tmol --find-links https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/
```

### From PyPI (source distribution, NVIDIA GPU)

The source distribution on PyPI compiles C++ extensions during installation. This targets NVIDIA GPUs; for Apple Silicon use the [fnachon/tmol](https://github.com/fnachon/tmol) fork directly.

```bash
# NVIDIA GPU (requires nvcc / CUDA toolkit)
pip install tmol              # requires nvcc for CUDA kernel compilation
pip install tmol[dev]         # includes development tools (black, flake8, pytest, etc.)
pip install tmol[cuda]        # also installs pip-distributed nvcc + CCCL headers
```

### From source

```bash
# NVIDIA GPU (upstream repository)
git clone https://github.com/uw-ipd/tmol.git && cd tmol
pip install -e ".[dev]"       # builds C++/CUDA extensions via CMake

# Apple Silicon — use the MPS fork
git clone https://github.com/fnachon/tmol.git && cd tmol
pip install -e ".[dev,mps]"   # builds C++/Metal extensions via CMake
```

## Usage

### Quick start

```python
import torch
import tmol

# Pick your device: "cpu", "cuda", or "mps" (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# Load a structure
pose_stack = tmol.pose_stack_from_pdb("1ubq.pdb", device=device)

# Score it
sfxn = tmol.beta2016_score_function(device)
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
> Tested on Ubuntu 20.04 (CUDA) and macOS 14+ (MPS). Other platforms should work but are not yet verified.

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
