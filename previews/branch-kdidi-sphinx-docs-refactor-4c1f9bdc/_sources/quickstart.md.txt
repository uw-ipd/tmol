# Quickstart

TMol provides Rosetta-inspired all-atom score terms and batched molecular
representations in PyTorch.

## Install TMol

```bash
pip install tmol
```

## Load and score one structure

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

`score` contains one value for each pose in the `PoseStack`. These are
beta2016-weighted TMol score units, not kcal/mol, binding free energies, or
values guaranteed to match Rosetta score units numerically.

## Refine and write the structure

```python
from tmol.optimization import run_cart_min

minimized = run_cart_min(pose_stack, sfxn)
rescored = scorer(minimized.coords)
print(rescored)
```

To write the result:

```python
from tmol.io import write_pose_stack_pdb

write_pose_stack_pdb(minimized, "minimized.pdb")
```

## Protein-ligand input

For a protein-ligand complex, load mmCIF through Biotite so TMol can use its
bond table during ligand preparation:

Helpers whose historical names include `ddg` report a chosen one-complex
interaction-score convention; they do not calculate thermodynamic binding free
energies. See the [scoring guide](user_guide/scoring.md).

```python
import biotite.structure as struc
import biotite.structure.io

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite

structure = biotite.structure.io.load_structure(
    "complex.cif",
    model=1,
    include_bonds=True,
)
if isinstance(structure, struc.AtomArrayStack):
    structure = structure[0]

pose_stack, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    param_db=ParameterDatabase.get_default(),
    return_context=True,
)

sfxn = beta2016_score_function(device, param_db=context.parameter_database)
```

Use the ligand-extended `context.parameter_database` when scoring a pose that
contains freshly prepared ligands.

## Next steps

- [Working with TMol](tutorial/01_working_with_tmol.ipynb)
- [Scoring and analysis](tutorial/03_scoring_and_analysis.ipynb)
- [Ligands and parameter files](tutorial/07_ligand_and_params.ipynb)
