# Quickstart

Load, score, minimize, and write a protein structure on CPU or CUDA.
The example expects a local file named `1ubq.pdb`.

## Install TMol

```bash
python -m pip install tmol --only-binary=tmol
```

This installs the CPU build. For CUDA, follow {doc}`Installation <installation>`
and change the device below to `"cuda"`.

## Load and score

```python
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cpu")
pose_stack = pose_stack_from_pdb("1ubq.pdb", device)

sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
score = scorer(pose_stack.coords)

print(score)
```

`score` contains one weighted total per pose.

## Minimize and write

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

Load mmCIF with bond information for ligand preparation:

```python
import biotite.structure as struc
import biotite.structure.io

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_file, pose_stack_from_biotite

structure = atom_array_from_file("complex.cif")

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

Continue with the {doc}`structure tutorial </tutorial/01_working_with_tmol>`
or the {doc}`scoring guide </user_guide/scoring>`.
