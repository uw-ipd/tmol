# Structure I/O and visualization

Read a structure, build a `PoseStack`, and write its coordinates.

## The six functions

Every file path goes through a Biotite `AtomArray`: a file is read into one and
a pose is built from it, and a pose is written by way of one.

| Function | Converts |
| --- | --- |
| `atom_array_from_file()` | PDB, CIF, MOL2 or SDF -> `AtomArray` |
| `file_from_atom_array()` | `AtomArray` -> PDB or CIF |
| `pose_stack_from_file()` | PDB, CIF, MOL2 or SDF -> `PoseStack` |
| `file_from_pose_stack()` | `PoseStack` -> PDB or CIF |
| `pose_stack_from_atom_array()` | `AtomArray` -> `PoseStack` |
| `atom_array_from_pose_stack()` | `PoseStack` -> `AtomArray` |

The format is inferred from the file extension, or else from the contents. A
source may be a path, the file's text, or a stream; pass `format=` to override
the inference. For differentiable model coordinates, use the Atom37 tensor
paths described in the {doc}`model input tutorial </model_inputs>`.

## Read a structure

```python
import torch

from tmol import pose_stack_from_file

device = torch.device("cuda")
pose_stack = pose_stack_from_file("input.cif", device)
```

`pose_stack_from_file()` reads with `atom_array_from_file()` and builds with
`pose_stack_from_atom_array()`. It takes the reader options `model` and
`assembly_id`, a half-open residue range `residue_start`/`residue_end`, and
passes any other keyword arguments to `pose_stack_from_atom_array()`.

Read the `AtomArray` yourself when you want to inspect or edit it first:

```python
from tmol import ParameterDatabase, atom_array_from_file, pose_stack_from_atom_array

structure = atom_array_from_file("input.cif")

pose_stack, context = pose_stack_from_atom_array(
    structure,
    device,
    param_db=ParameterDatabase.get_default(),
    prepare_ligands=True,
    return_context=True,
)
```

Reading preserves supplied names, coordinates, hydrogens and covalent bonds;
declared unresolved heavy atoms carry NaN coordinates. The context contains the
parameter database, canonical ordering, and packed block types. Reuse it for
structures with compatible chemistry.

Set `prepare_ligands=False` when all required nonstandard chemistry has already
been registered. By default, pose construction packs hydrogens with OptH; pass
`no_optH=True` to keep finite input hydrogen coordinates instead. See
{doc}`Terminology and modeling choices </terminology>` for how preparation
changes atoms.

## Inspect the built structure

Convert the pose back to an `AtomArray` for selections, metadata inspection, or
file output:

```python
import numpy

from tmol import atom_array_from_pose_stack, selection_gallery

built = atom_array_from_pose_stack(pose_stack)
selection_gallery(
    built,
    {
        "whole structure": numpy.ones(built.array_length(), dtype=bool),
        "chain A": built.chain_id == "A",
    },
)
```

`selection_gallery()` displays the built structure in a notebook. For automated
checks, inspect its `AtomArray`, context, block types, and connectivity.

## Write

```python
from tmol import file_from_atom_array, file_from_pose_stack

file_from_pose_stack(pose_stack, "output.cif")
pdb_text = file_from_pose_stack(pose_stack, format="pdb")
file_from_atom_array(built, "built.pdb")
```

The format is inferred from the path's extension. Without a path, the text is
returned and `format=` is required. A CIF carries its own component chemistry,
including that of generated ligands. A PDB round trip is not lossless for every
ligand bond, noncanonical residue, or preparation decision.

## Before scoring

Before scoring or refinement, verify:

1. the intended model and chains were selected;
2. internal gaps remain disconnected rather than becoming false termini;
3. nonstandard residues have authoritative chemistry and bonds;
4. histidine, disulfide, terminus, and missing-atom choices are expected;
5. the pose, score function, and packed types use the same device; and
6. a scorer is rerendered after any change to atom or block layout.

## Examples and reference

{doc}`Structure I/O tutorial </tutorial/01_tmol_how_to_guide>` · {doc}`I/O API </api/io>` · {doc}`Ligand preparation </user_guide/ligands>`
