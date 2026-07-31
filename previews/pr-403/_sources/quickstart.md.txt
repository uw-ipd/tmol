# Quickstart

`tmol` scores molecular structures with Rosetta-like all-atom energy terms in
PyTorch. A minimal workflow is:

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

To refine the structure with Cartesian minimization:

```python
from tmol.optimization.minimizers import run_cart_min

minimized = run_cart_min(pose_stack, sfxn)
rescored = scorer(minimized.coords)
print(rescored)
```

To write the result:

```python
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

write_pose_stack_pdb(minimized, "minimized.pdb")
```

For protein-ligand complexes, load through Biotite so tmol can use bond-table
information for ligand preparation:

```python
import biotite.structure as struc
import biotite.structure.io

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

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
