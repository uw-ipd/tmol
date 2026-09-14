# Quickstart

TMol provides batched molecular representations and Rosetta-inspired all-atom
modeling primitives in PyTorch.

This page is a short first-score recipe. For explanations, visualization, and
exercises, continue with the numbered Tutorials.

> - **Prerequisite:** {doc}`Install TMol </installation>`.
> - **Deep tutorial:** {doc}`01 — Working with TMol
>   </tutorial/01_working_with_tmol>`.
> - **Related workflows:** {doc}`Workflow recipes </workflows/index>`.
> - **API reference:** {doc}`Input and Output </api/io>` and
>   {doc}`Scoring </api/score>`.
> - **Rosetta mapping:** {doc}`Rosetta-to-TMol crosswalk
>   </tutorial/rosetta_crosswalk>`.

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

Helpers whose historical names include `ddg` report a chosen one-complex
interaction-score convention; they do not calculate thermodynamic binding free
energies. See the {doc}`scoring and analysis guide </user_guide/scoring>`.

## Choose the next path

- Work through the ten {doc}`interactive examples </examples_index>` for
  complete, executable tutorials with molecular viewers and exercises.
- Use the {doc}`workflow hub </workflows/index>` for short, reusable recipes.
- Search the {doc}`task index </tutorial/recipe_index>` when you already know
  the operation you need.

If you want to continue directly, start with
{doc}`Tutorial 01 — Working with TMol </tutorial/01_working_with_tmol>`.

```{toctree}
:hidden:

Installation <installation>
```
