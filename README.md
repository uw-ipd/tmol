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

Pre-built wheels ship with **ahead-of-time (AOT) compiled** C++/CUDA extensions -- no `nvcc` or CUDA toolkit needed at install time.

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

### From PyPI (source distribution)

The source distribution on PyPI compiles C++/CUDA extensions during installation.
This requires `nvcc` (CUDA toolkit) and a C++17-capable compiler.

```bash
pip install tmol              # requires nvcc for kernel compilation
pip install tmol[dev]         # includes development tools (black, flake8, pytest, etc.)
```

### From source

```bash
git clone https://github.com/uw-ipd/tmol.git && cd tmol
pip install -e ".[dev]"
python setup.py build_ext --inplace   # build C++/CUDA extensions
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
