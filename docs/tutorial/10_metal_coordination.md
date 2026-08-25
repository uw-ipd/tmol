# Score and minimize an explicitly bonded metal site

Read a PDB or mmCIF with its explicit bond table, then build the pose normally:

```python
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure

from tmol import ScoreType, beta2016_score_function, run_cart_min
from tmol.io import pose_stack_from_biotite

device = torch.device("cuda")
structure = get_structure(
    CIFFile.read("metal_site.cif"),
    model=1,
    include_bonds=True,
)
pose, context = pose_stack_from_biotite(
    structure,
    device,
    no_optH=True,
    return_context=True,
)
```

Mg, Ca, and Zn single-ion components are generated from their element. Only
explicitly bonded coordinating waters are retained. The pose contains the
metal-donor connections and Rosetta-style deposited-geometry constraints.
Distance constraints cover every donor. A donor-parent angle constraint is
also generated when the donor component supplies a distinct local parent;
single-atom deposited waters therefore have no angle constraint.

Enable those constraints when scoring or minimizing, just as Rosetta requires
the `metalbinding_constraint` term to be active:

```python
score_function = beta2016_score_function(
    device,
    param_db=context.parameter_database,
)
score_function.set_weight(ScoreType.constraint, 1.0)

score = score_function.render_whole_pose_scoring_module(pose)(pose.coords).sum()
minimized = run_cart_min(
    pose.clone(),
    score_function,
    optimizer_kwargs={"max_iter": 200},
)
```

Pass `metal_distance_constraint_multiplier=0.0` or
`metal_angle_constraint_multiplier=0.0` to `pose_stack_from_biotite()` to omit
one constraint family. Do not disable both for unconstrained minimization: the
ion and donor geometry can distort even though their connection topology is
still present.
