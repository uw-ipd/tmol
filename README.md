<h1 align="center">TMol</h1>

<p align="center"><em>Rosetta molecular modeling at PyTorch speed.</em></p>

<p align="center">
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/v/tmol.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/pyversions/tmol.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/tmol/"><img src="https://img.shields.io/pypi/dm/tmol.svg" alt="PyPI downloads"></a>
  <a href="https://github.com/uw-ipd/tmol/actions/workflows/ci.yml"><img src="https://github.com/uw-ipd/tmol/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://uw-ipd.github.io/tmol/"><img src="https://github.com/uw-ipd/tmol/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/uw-ipd/tmol"><img src="https://codecov.io/gh/uw-ipd/tmol/graph/badge.svg" alt="Code coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/uw-ipd/tmol.svg" alt="License"></a>
</p>

TMol scores, packs, minimizes, and relaxes all-atom molecular structures as
batched PyTorch tensors—on CPU or GPU, with gradients. It provides fast
C++/CUDA kernels and modeling primitives for proteins, nucleic acids, ligands,
and their complexes.

Explore the **[TMol documentation](https://uw-ipd.github.io/tmol/)** for
complete installation guidance, executable tutorials, workflows, and the API
reference.

Three ways in:

- 🚀 **Start scoring** → [quick start](#quick-start), then the full **[Quickstart](https://uw-ipd.github.io/tmol/latest/quickstart.html)**.
- 🧬 **Build a workflow** → **[scoring, packing, minimization, FastRelax, and ligand recipes](https://uw-ipd.github.io/tmol/latest/workflows/index.html)**.
- 🛠️ **Develop TMol** → **[contributor guide](https://uw-ipd.github.io/tmol/latest/contributor_guide.html)**.

## Install

The shortest path is:

```bash
pip install tmol
```

TMol first looks for a matching prebuilt wheel and otherwise builds locally.
For a deterministic CPU/GPU binary install, supported Python/PyTorch/CUDA
combinations, Colab, macOS, and HPC troubleshooting, see the
**[installation guide](https://uw-ipd.github.io/tmol/latest/installation.html)**
and **[GitHub Releases](https://github.com/uw-ipd/tmol/releases)**.

Verify the installation:

```bash
python -c "import tmol; print(tmol.__version__)"
```

---

## Quick start

Score a structure on GPU when CUDA is available, otherwise on CPU:

```python
import torch
import tmol

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose = tmol.pose_stack_from_pdb("input.pdb", device)

score_function = tmol.beta2016_score_function(device)
score = score_function.render_whole_pose_scoring_module(pose)
print(score(pose.coords))
```

From here, use the **[interactive examples](https://uw-ipd.github.io/tmol/latest/examples_index.html)**
to pack side chains, analyze score terms, minimize coordinates, run FastRelax,
prepare ligands, or model nucleic acids.

---

## What TMol provides

- Batched, differentiable all-atom structures backed by PyTorch tensors.
- Rosetta-inspired score terms with CPU and CUDA implementations.
- Side-chain packing and design, Cartesian and kinematic minimization, and FastRelax.
- Protein, ligand, RNA, and DNA structure preparation and analysis.
- RoseTTAFold2, OpenFold, Biotite, PDB, and canonical tensor integrations.
- Ahead-of-time compiled wheels plus source and just-in-time build paths.

See the **[task index](https://uw-ipd.github.io/tmol/latest/tutorial/recipe_index.html)**
to jump from a modeling task to its maintained tutorial, workflow, and API.

---

## Development

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
TMOL_DISABLE_WHEEL_FETCH=1 pip install -e ".[dev]"
```

The **[development guide](https://uw-ipd.github.io/tmol/latest/user_guide/development.html)**
covers CMake/CUDA builds, tests, benchmarks, containers, CI, and releases.

## Citation

If you use TMol in your work, please cite:

> Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg,
> David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, *tmol: a
> GPU-accelerated, PyTorch implementation of Rosetta's relax protocol*
> (manuscript in preparation).

TMol is available under the terms in [LICENSE](LICENSE).
