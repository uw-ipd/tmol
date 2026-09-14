# Structure I/O and visualization

**Prerequisites:** {doc}`Installation </installation>` and a basic understanding
of {doc}`PoseStack terminology </terminology>`.

**Deep tutorial:** {doc}`Tutorial 01 — Working with TMol
</tutorial/01_working_with_tmol>`.

**API:** {doc}`I/O </api/io>`, {doc}`pose objects </api/pose>`, and
{doc}`ligand preparation </api/ligand>`.

**Related workflows:** {doc}`integrations </user_guide/integrations>`,
{doc}`GPU batching <gpu_batching>`, and {doc}`ligands
</user_guide/ligands>`.

## Choose an input path

- Read **mmCIF and PDB through AtomWorks** with `atom_array_from_file()` to
  preserve chemical identities, declared bonds, and unresolved atoms.
- Pass an existing **Biotite AtomArray** to `pose_stack_from_biotite()`.
- Use the **Atom37 tensor paths** for differentiable model coordinates, or map
  another named layout such as Atom14 to canonical tensors as shown in the
  {doc}`model input tutorial </model_inputs>`.

## Build a PoseStack from mmCIF

```python
import torch

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_file, pose_stack_from_biotite

device = torch.device("cuda")
structure = atom_array_from_file("input.cif")

pose_stack, context = pose_stack_from_biotite(
    structure,
    device,
    param_db=ParameterDatabase.get_default(),
    no_optH=False,
    prepare_ligands=True,
    return_context=True,
)
```

The returned context records the parameter database, canonical ordering, and
packed block types selected during preparation. Reuse it for compatible
structures to keep chemical interpretation stable and avoid rebuilding setup
data.

Set `prepare_ligands=False` when all required nonstandard chemistry has already
been registered. See {doc}`Terminology and modeling choices </terminology>` for
the deposited-versus-built atom distinction and the `no_optH` decision.

## Inspect deposited and built structures

Convert the pose back to Biotite for selections, metadata inspection, or file
output:

```python
import numpy

from tmol.io import biotite_from_pose_stack, selection_gallery

built = biotite_from_pose_stack(pose_stack)
selection_gallery(
    built,
    {
        "whole structure": numpy.ones(built.array_length(), dtype=bool),
        "chain A": built.chain_id == "A",
    },
)
```

The interactive gallery is intended for notebook output and rendered TMol
Tutorials. For automated checks, inspect the `AtomArray`, build context, block
types, and connectivity directly.

## Export

Use Biotite writers when preserving mmCIF-level metadata matters. TMol also
provides `write_pose_stack_pdb()` and `pose_stack_to_pdb_string()` for PDB
compatibility. A PDB round trip is not lossless for every ligand bond,
noncanonical residue, or preparation decision.

## Common checks

Before scoring or refinement, verify:

1. the intended model and chains were selected;
2. internal gaps remain disconnected rather than becoming false termini;
3. nonstandard residues have authoritative chemistry and bonds;
4. histidine, disulfide, terminus, and missing-atom choices are expected;
5. the pose, score function, and packed types use the same device; and
6. a scorer is rerendered after any change to atom or block layout.
