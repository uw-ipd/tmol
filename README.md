# tmol

`tmol` is a GPU-accelerated PyTorch implementation of Rosetta's molecular
modeling energy function. It scores protein structures, supports differentiable
minimization, packs side chains, and prepares small-molecule ligands for
protein-ligand scoring workflows.

Documentation: <https://uw-ipd.github.io/tmol/latest/>

## Quickstart

Install tmol:

```bash
pip install tmol
```

For source development:

```bash
git clone https://github.com/uw-ipd/tmol.git
cd tmol
pip install -e ".[dev]"
```

Score a PDB structure:

```python
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_stack = pose_stack_from_pdb("1ubq.pdb", device)

sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
score = scorer(pose_stack.coords)

print(score)
```

Minimize and write the result:

```python
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb
from tmol.optimization.minimizers import run_cart_min

minimized = run_cart_min(pose_stack, sfxn)
write_pose_stack_pdb(minimized, "minimized.pdb")
```

For installation details, protein-ligand preparation, examples, API reference,
and contributor guidance, see the full documentation.

## Citation

Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg,
David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio, *tmol: a
GPU-accelerated, PyTorch implementation of Rosetta's relax protocol*,
manuscript in preparation.
