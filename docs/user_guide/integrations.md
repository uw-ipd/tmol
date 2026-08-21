# Model and structure integrations

TMol is designed to sit inside PyTorch-based structural-biology workflows. It can
score structures loaded from standard files and convert outputs from structure
prediction systems into `PoseStack` objects.

> - **Prerequisites:** {doc}`Quickstart </quickstart>` and the output schema for
>   the source model or structure library.
> - **Deep tutorial:** {doc}`01 — Working with TMol
>   </tutorial/01_working_with_tmol>`.
> - **Related workflows:** {doc}`GPU batching </workflows/gpu_batching>` and
>   {doc}`Ligand preparation </user_guide/ligands>`.
> - **API reference:** {doc}`Input and Output </api/io>` and
>   {doc}`Pose </api/pose>`.
> - **Rosetta mapping:** {doc}`I/O, selections, and options
>   </tutorial/rosetta_crosswalk>`.

## RoseTTAFold2

Install TMol into the RoseTTAFold2 environment:

```bash
cd <tmol repo root>
pip install -e .
```

Convert one prediction by passing one-dimensional residue-type indices,
three-dimensional atom coordinates, and chain lengths:

```python
from tmol.io import pose_stack_from_rosettafold2

pose_stack = pose_stack_from_rosettafold2(
    seq=sequence_indices,
    xyz=atom_coordinates,
    chainlens=chain_lengths,
)
```

The adapter supports canonical amino acids and canonical termini. It returns a
single-pose stack on the same device as the input tensors. RoseTTAFold2
inference code often disables gradients globally; enable them around any
differentiable TMol scoring or minimization:

```python
import torch

from tmol.score import beta2016_score_function

sfxn = beta2016_score_function(pose_stack.device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

with torch.enable_grad():
    coords = pose_stack.coords.detach().clone().requires_grad_(True)
    score = scorer(coords).sum()
    score.backward()
```

## OpenFold

The OpenFold adapter consumes a result dictionary containing `aatype`,
`positions`, and `chain_index` tensors:

```python
from tmol.io import pose_stack_from_openfold

pose_stack = pose_stack_from_openfold(openfold_output)
```

It supports batched canonical-protein predictions, uses the final entry in
`positions`, and preserves the input device. Additional keys are ignored by the
adapter.

## Biotite and AtomArray

The preferred path for rich structure IO is Biotite `AtomArray`:

```python
import biotite.structure as struc
from biotite.structure.io import load_structure
import torch

from tmol.io import pose_stack_from_biotite
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = load_structure(
    "complex.cif",
    model=1,
    include_bonds=True,
)
if isinstance(structure, struc.AtomArrayStack):
    structure = structure[0]

pose_stack = pose_stack_from_biotite(structure, device)
sfxn = beta2016_score_function(device)
```

The score function and `PoseStack` must use the same device. Batching unlike
structures requires compatible chemistry and
`PoseStackBuilder.from_poses()`; TMol does not schedule multi-GPU work
internally.

Biotite is especially important for ligands because TMol uses explicit bond
tables from the `AtomArray` during ligand preparation. Follow
{doc}`07 — Ligands and Parameter Files </tutorial/07_ligand_and_params>` when
the structure contains non-standard residues; the resulting score function
must use the ligand-extended parameter database.
