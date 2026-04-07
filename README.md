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

Pre-built wheels ship with **ahead-of-time (AOT) compiled** C++/CUDA extensions -- no `nvcc` or CUDA toolkit needed at install time. Pick the wheel matching your **PyTorch version** and **platform**:

**x86_64 GPU (Linux):**

| PyTorch | CUDA | Wheel tag              | Note |
|---------|------|------------------------|------|
| 2.8     | 12.6 | `+cu126torch2.8`       | NGC native |
| 2.9     | 13.0 | `+cu130torch2.9`       | NGC native |
| 2.10    | 13.1 | `+cu131torch2.10`      | NGC native |
| 2.10    | 12.8 | `+cu128torch2.10`      | Google Colab compatible |

**ARM64 / aarch64 GPU (Linux, e.g., Grace Hopper, Jetson):**

| PyTorch | CUDA | Wheel tag              |
|---------|------|------------------------|
| 2.8     | 12.6 | `+cu126torch2.8`       |
| 2.9     | 13.0 | `+cu130torch2.9`       |
| 2.10    | 13.1 | `+cu131torch2.10`      |

**CPU-only (any platform):**

| PyTorch | Wheel tag |
|---------|-----------|
| 2.10    | `+cpu`    |

All wheels are Python 3.12 (`cp312`).

> [!TIP]
> CUDA wheels are **forward-compatible** within a major version: a `cu126` wheel works on any CUDA 12.x driver >= 12.6. You do not need an exact CUDA version match.

Check your environment:

```bash
python -c "import sys, torch; print(f'Python {sys.version_info.major}.{sys.version_info.minor}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

Install from [GitHub Releases](https://github.com/uw-ipd/tmol/releases):

```bash
# Direct URL (replace RELEASE_TAG and WHEEL_FILENAME):
pip install https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/WHEEL_FILENAME.whl

# Or use --find-links to let pip resolve by version:
pip install tmol --find-links https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/
```

<details>
<summary><b>Google Colab</b></summary>

Colab ships PyTorch 2.10 with CUDA 12.8. Install the Colab-specific wheel:

```bash
pip install https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cu128torch2.10-cp312-cp312-linux_x86_64.whl
```

Replace `vX.Y.Z` and `X.Y.Z` with the desired release version.

</details>

<details>
<summary><b>CPU-only (no GPU)</b></summary>

For machines without a GPU (laptops, CI servers, data preprocessing):

```bash
pip install https://github.com/uw-ipd/tmol/releases/download/vX.Y.Z/tmol-X.Y.Z+cpu-cp312-cp312-linux_x86_64.whl
```

The CPU wheel works with any PyTorch installation (CPU or CUDA). CUDA operations will raise a runtime error; all CPU operations work normally.

</details>

### From PyPI (source distribution)

The source distribution compiles C++/CUDA extensions during installation. If `nvcc` is available, both CPU and CUDA extensions are built. Without `nvcc`, only CPU extensions are built.

```bash
pip install tmol              # builds extensions (CUDA if nvcc available, CPU otherwise)
pip install tmol[dev]         # includes development tools (black, flake8, pytest, etc.)
```

### From source

```bash
git clone https://github.com/uw-ipd/tmol.git && cd tmol
pip install -e ".[dev]"   # builds extensions via CMake (CUDA auto-detected)
```

If you don't have a CUDA toolkit, the build automatically falls back to CPU-only extensions. You can also force a CPU-only build explicitly:

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
