<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/uw-ipd/tmol/master/docs/_static/brand/tmol-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/uw-ipd/tmol/master/docs/_static/brand/tmol-logo-light.svg" alt="TMol" width="440">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/v/tmol.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/pyversions/tmol.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/dm/tmol.svg" alt="PyPI downloads"></a>
  <a href="https://github.com/uw-ipd/tmol/actions/workflows/ci.yml"><img src="https://github.com/uw-ipd/tmol/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://uw-ipd.github.io/tmol/"><img src="https://github.com/uw-ipd/tmol/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/uw-ipd/tmol"><img src="https://codecov.io/gh/uw-ipd/tmol/graph/badge.svg" alt="Code coverage"></a>
  <a href="https://github.com/uw-ipd/tmol/blob/master/LICENSE"><img src="https://img.shields.io/github/license/uw-ipd/tmol.svg" alt="License"></a>
</p>

TMol provides all-atom scoring, packing, minimization, and
FastRelax in PyTorch. It batches proteins, nucleic acids, ligands, and complexes
on CPU or CUDA, with gradients through coordinate scoring.

[Documentation](https://uw-ipd.github.io/tmol/) ·
[Tutorials](https://uw-ipd.github.io/tmol/latest/examples_index.html) ·
[API reference](https://uw-ipd.github.io/tmol/latest/api_reference.html)

## Install

For GPU scoring on Linux with PyTorch 2.14 and CUDA 13.2:

```bash
python -m pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu132
python -m pip install "tmol==0.1.60+cu132torch2.14" --only-binary=tmol \
  --find-links https://uw-ipd.github.io/tmol/wheels/v0.1.60/cu132torch2.14/
```

For CPU scoring:

```bash
python -m pip install tmol
```

PyPI provides CPU wheels; GitHub hosts CUDA variants. Both include AtomWorks 3
and ligand preparation with RDKit and OpenBabel. See the
[installation guide](https://uw-ipd.github.io/tmol/latest/installation.html)
for other CUDA/PyTorch combinations, platform support, and source builds.

Verify the installation:

```bash
python -c "import tmol; print(tmol.__version__)"
```

## Quick start

Score a structure on CPU. With a CUDA TMol wheel, use `torch.device("cuda")`:

```python
import torch
import tmol

device = torch.device("cpu")
pose = tmol.pose_stack_from_pdb("input.pdb", device)

score_function = tmol.beta2016_score_function(device)
score = score_function.render_whole_pose_scoring_module(pose)
print(score(pose.coords))
```

See the [quickstart](https://uw-ipd.github.io/tmol/latest/quickstart.html) for
minimization and ligand preparation. The
[guides](https://uw-ipd.github.io/tmol/latest/workflows/index.html) cover batching,
packing, score analysis, and model inputs; the
[task index](https://uw-ipd.github.io/tmol/latest/tutorial/recipe_index.html)
links individual operations to examples and APIs.

## Development

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
python -m pip install "scikit-build-core>=0.10" "cmake>=3.24,<4" "pybind11>=2.12" ninja packaging
python -m pip install --no-build-isolation -e ".[dev]"
```

The [development guide](https://uw-ipd.github.io/tmol/latest/user_guide/development.html)
covers builds, tests, benchmarks, and releases. See the
[contributor guide](https://uw-ipd.github.io/tmol/latest/contributor_guide.html)
for code and documentation conventions, or [agent skills](https://github.com/uw-ipd/tmol/blob/master/skills/README.md)
for reusable coding-agent instructions.

## Citation

If you use TMol in your work, please cite:

> Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg,
> David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, *tmol: a
> GPU-accelerated, PyTorch implementation of Rosetta's relax protocol*
> (manuscript in preparation).

TMol is available under the terms in [LICENSE](https://github.com/uw-ipd/tmol/blob/master/LICENSE).
