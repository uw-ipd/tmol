# Tmol

`tmol`, short for TensorMol, is a faithful reimplementation of the Rosetta molecular modeling energy function ("beta_nov2016_cart") in PyTorch with custom kernels written in C++ and CUDA. Given the coordinates of one or more proteins, `tmol` can compute both energies and derivatives. `tmol` can also perform gradient-based minimization on those structures. Thus, ML models that produce cartesian coordinates for proteins can include biophysical features in their loss during training or refine their output structures using Rosetta's experimentally validated energy function. You can read the full wiki [here](https://github.com/uw-ipd/tmol/wiki/DevHome).

## Table of Contents

- Installation
- JIT Development Mode
- Containers
- Verification
- Usage
- RosettaFold2
- OpenFold
- CI Strategy
- Development Workflow
- Citation

## Installation

### Pre-built wheels (recommended)

Pre-built wheels are available for Linux x86_64 with CUDA. Pick the one matching your **PyTorch version** and **ABI**:

**Which ABI do I have?**
```bash
python -c "import torch; print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
```
- `True` → NGC container, conda, or source-built PyTorch → use **cxx11abiTRUE** wheels
- `False` → `pip install torch` on bare metal → use **cxx11abiFALSE** wheels

If you want more background on the CXX11 ABI split, see this discussion:
https://github.com/Dao-AILab/flash-attention/issues/457

| PyTorch | CUDA | ABI | Wheel tag |
|---------|------|-----|-----------|
| 2.5     | 12.6 | TRUE  | `+cu126torch2.5cxx11abiTRUE` |
| 2.5     | 12.4 | FALSE | `+cu124torch2.5cxx11abiFALSE` |
| 2.8     | 12.9 | TRUE  | `+cu129torch2.8cxx11abiTRUE` |
| 2.8     | 12.6 | FALSE | `+cu126torch2.8cxx11abiFALSE` |
| 2.10    | 13.1 | TRUE  | `+cu131torch2.10cxx11abiTRUE` |
| 2.10    | 12.6 | FALSE | `+cu126torch2.10cxx11abiFALSE` |

> [!TIP]
> CUDA wheels are **forward-compatible** within a major version: a `cu124` wheel works on any CUDA 12.x driver ≥ 12.4. You don't need an exact CUDA version match.

Check your full environment:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

Install from [GitHub Releases](https://github.com/uw-ipd/tmol/releases):
```bash
# Direct URL (replace RELEASE_TAG and WHEEL_FILENAME):
pip install https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/WHEEL_FILENAME.whl

# Or use --find-links to let pip resolve by version:
pip install tmol --find-links https://github.com/uw-ipd/tmol/releases/download/RELEASE_TAG/
```

### From PyPI (source build)

The source distribution on PyPI compiles C++/CUDA extensions during installation.
This requires `nvcc` (CUDA toolkit) to be available.

```bash
pip install tmol              # base (requires nvcc for kernel compilation)
pip install tmol[dev]         # with development tools (ruff, pytest, etc.)
```

### From source (development)

```bash
git clone https://github.com/uw-ipd/tmol.git && cd tmol
pip install -e ".[dev]"

# Build C++/CUDA extensions in-place
python setup.py build_ext --inplace
```

## JIT Development Mode

By default, tmol loads the precompiled `_C` extension if present. For kernel
development you can opt into JIT compilation:

```bash
export TMOL_USE_JIT=1
```

Optional fallback when the precompiled extension is missing:

```bash
export TMOL_JIT_FALLBACK=1
```

JIT mode requires `nvcc` and the CUDA headers/libraries to be discoverable:

1. **Provide your own CUDA toolkit** (e.g., CUDA-enabled container or system CUDA with `CUDA_HOME` set)
2. **Install the pip CUDA extra** and let tmol auto-configure the environment

```bash
pip install .[cuda]
```

When using the pip CUDA extra, tmol auto-detects `nvcc` and sets `CUDA_HOME`,
`PATH`, and `LD_LIBRARY_PATH`, plus creates compatibility symlinks needed by
PyTorch (e.g., `nvidia/cu12 -> nvidia/cu13`, `libcudart.so -> libcudart.so.N`).

## Containers

The container definitions install all dependencies into an NVIDIA NGC PyTorch base image that already provides `torch`, `numpy`, `nvcc`, and CUDA libraries. Bind-mount your tmol checkout at runtime.

**Docker:**
```bash
docker build -t tmol-dev -f containers/docker/tmol-dev.Dockerfile .
docker run --gpus all -it -v $(pwd):/tmol_host -w /tmol_host tmol-dev bash
pip install -e .  # inside container
```

**Apptainer:**
```bash
apptainer build tmol-dev.sif containers/apptainer/tmol-dev.def
apptainer run --nv --bind $(pwd):/tmol_host tmol-dev.sif
```

## Verification

```python
import tmol
# If extensions are loaded correctly, this will print without error:
print("tmol loaded successfully")
```

## Usage

`tmol` can be used as a standalone library, or integrated with [RosettaFold2](https://github.com/uw-ipd/tmol/wiki/DevHome#RosettaFold2) or [OpenFold](https://github.com/uw-ipd/tmol/wiki/DevHome#OpenFold).

Each platform has functions for constructing a [PoseStack](https://github.com/uw-ipd/tmol/wiki/PoseStack), performing operations on that PoseStack, and retrieving the structure back from `tmol`.

#### Create a PoseStack from a PDB file
```python
import tmol
tmol.pose_stack_from_pdb('1ubq.pdb')
```

#### Create a ScoreFunction and score a PoseStack
```python
sfxn = tmol.beta2016_score_function(pose_stack.device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
print(scorer(pose_stack.coords))
```

#### Create a Minimizer and run it on a PoseStack with our ScoreFunction
```python
start_coords = pose_stack.coords.clone()
pose_stack.coords[:] = start_coords

cart_sfxn_network = tmol.cart_sfxn_network(sfxn, pose_stack)
optimizer = tmol.lbfgs_armijo(cart_sfxn_network.parameters())

cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)


def closure():
    optimizer.zero_grad()
    E = cart_sfxn_network().sum()
    E.backward()
    return E

optimizer.step(closure)

cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)
```

#### Save a PoseStack to a PDB
```python
tmol.write_pose_stack_pdb(pose_stack, 'output.pdb')
```

## RosettaFold2

To use `tmol` within RosettaFold2, install `tmol` into your RF2 environment:

```bash
cd <your local tmol repository root directory>
pip install -e .
```

>[!NOTE]
>This has been tested on Ubuntu 20.04 - other platforms should work but are currently untested.

Example usage from within RosettaFold2:

#### Create a PoseStack from RF2 coordinates
```python
seq, xyz, chainlens = rosettafold2_model.infer(sequence)
pose_stack = tmol.pose_stack_from_rosettafold2(seq[0], xyz[0], chainlens[0])
```

#### Load a PoseStack into RF2 coordinates
```python
xyz = tmol.pose_stack_to_rosettafold2( ... )
```

>[!NOTE]
>Hydrogens and OXT coordinates from the terminal residues in RosettaFold are not preserved across the RF2<->tmol interface.

>[!WARNING]
>You must call `torch.set_grad_enabled(True)` if you wish to use the `tmol` minimizer, as by default RF2 has grad disabled during inference.

## OpenFold

Full OpenFold documentation coming soon.

#### Create a PoseStack from an OpenFold dictionary
```python
output = openfold_model.infer(sequences)
pose_stack = tmol.pose_stack_from_openfold(output)
```

## CI Strategy

We use a hybrid CI setup:

- **Buildkite (GPU runners)** runs CUDA-heavy tests and benchmarks.
- **GitHub Actions** builds the wheel matrix, builds sdist, and publishes to TestPyPI/GitHub Releases.

To enable Buildkite on a branch, the Buildkite pipeline must be connected to this repo
in the Buildkite UI and the GitHub webhook must be configured to trigger builds for
the desired branches/PRs. The pipeline definition lives in `.buildkite/pipeline.yml`.

## Development Workflow

`tmol` uses Test-Driven Development. If you are writing `tmol` code, [you should start by writing test cases for your code](https://github.com/uw-ipd/tmol/wiki/Testing#writing-tests).

### Building extensions locally

```bash
# Build all extensions (production + test)
python setup.py build_ext --inplace

# Skip test extensions (faster)
TMOL_SKIP_TEST_EXTS=TRUE python setup.py build_ext --inplace

# Specify GPU architectures (default: "8.0 8.6 8.9 9.0 10.0+PTX")
TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" python setup.py build_ext --inplace
```

### Committing

tmol uses pre-commit hooks to ensure uniform code formatting. These pre-commit hooks run `clang-format` and `ruff`. If your code needed reformatting, the initial commit will fail, and clang-format/ruff will reformat your code. You can see these changes via `git diff`, and you can `git add` the files to accept the new formatting before committing again.

To install the pre-commit hooks:
```bash
pip install -e ".[dev]"
pre-commit install
```

### Submitting changes to master

All changes to master should be performed via pull request flow, with a PR serving as a core point of development, discussion, testing and review. We close pull requests via squash or rebase, so that master contains a tidy, linear project history.

A pull request should land as an "atomic" unit of work, representing a single set of related changes. A larger feature may span multiple pull requests, however each pull request should stand alone. If a request appears to be growing "too large" to review, utilize a stacked pull to partition the work.

### Automated Testing

We maintain an automated test suite. The test suite must always be passing for master, and is available for any open branch via pull request.

```bash
pytest tmol/tests/ -v
```

## Citation

If you use `tmol` in your work, please cite:

Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg, David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, _tmol: a GPU-accelerated, PyTorch implementation of Rosetta's relax protocol_, (manuscript in preparation)
