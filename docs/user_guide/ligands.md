# Ligand Preparation

tmol can turn a non-standard, non-polymer residue into a parameterized residue
type with protonated 3D coordinates, MMFF94 partial charges,
generic-potential-style atom types used by tmol, and cartbonded parameters. The
prepared ligand is injected into a new `ParameterDatabase` and can then be
scored and minimized like a normal residue. These atom-type names do not by
themselves make a ligand parameterization usable by Rosetta.

## Entry Points

There are three single-ligand entry points:

```python
from tmol.ligand import (
    prepare_ligand_from_cif,
    prepare_ligand_from_mol2,
    prepare_ligand_from_smiles,
)

param_db, co = prepare_ligand_from_mol2("ligand.mol2")
param_db, co = prepare_ligand_from_cif("ligand.cif")
param_db, co = prepare_ligand_from_smiles("c1ccccc1C(=O)O", res_name="BEN")
```

Each returns a new `(ParameterDatabase, CanonicalOrdering)`. The input database
is not mutated.

MOL2 is the richest input. With authoritative MMFF94 charges, tmol reads atom
names, coordinates, bond orders, and partial charges directly. CIF input uses
the bond table to derive chemistry. SMILES input has no input geometry, so tmol
generates protonation, conformer coordinates, and MMFF94 charges.

## Loading Complexes

For full protein-ligand structures, load with Biotite and let
`pose_stack_from_biotite()` prepare every non-standard residue:

```python
import biotite.structure as struc
import biotite.structure.io
import torch

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
```

PDB files are fine for protein structures, but PDB does not carry reliable
ligand bond orders. Use CIF, MOL2, SMILES, or a prepared `.tmol` file for ligand
chemistry.

## Reuse Prepared Context

When scoring many structures that contain the same ligand definitions, build
the structure-independent context once:

```python
from tmol.io.pose_stack_from_biotite import (
    build_context_from_biotite,
    pose_stack_from_biotite,
)

context = build_context_from_biotite(struct0, device, prepare_ligands=True)
for structure in structures:
    pose_stack = pose_stack_from_biotite(structure, device, context=context)
```

This skips rebuilding the parameter database, canonical ordering, residue type
set, and packed block types for every structure.

## Persist Prepared Ligands

For manual edits or cold reuse, write `.tmol` params and load them later:

```python
from tmol.database import ParameterDatabase
from tmol.ligand import prepare_ligands

param_db, co = prepare_ligands(
    atom_array,
    ph=7.4,
    params_output="my_ligands.tmol",
)

param_db, co = prepare_ligands(
    atom_array,
    param_db=ParameterDatabase.get_default(),
    params_files=["my_ligands.tmol"],
)
```

The same prepared files can be passed through IO:

```python
pose_stack, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    ligand_params_files=["my_ligands.tmol"],
    return_context=True,
)
```

## SMILES to Params CLI

The ligand-prep script writes Rosetta `.params` and tmol `.tmol` files:

```bash
python scripts/ligand_prep/smiles_to_params.py "<SMILES>" <out_prefix> \
    --res-name LG1 --ph 7.4
```

Useful flags include `--no-protonate`, `--sample-proton-chi`, and
`--no-conformer-search`.

The emitted Rosetta-syntax `.params` file is an experimental interchange
artifact, not a Rosetta-validated parameterization. It carries tmol's atom-type
strings into the Rosetta atom-type field, uses MM type `X`, and writes a
placeholder `NBR_RADIUS 999.0`. Use a Rosetta-native preparation and validation
workflow before running the ligand in Rosetta.

## Interaction Scores

Use the ligand-aware score function with an explicit ligand block mask:

```python
from tmol.score.score_utils import calculate_block_pair_ddg

interaction = calculate_block_pair_ddg(
    pose_stack,
    ligand_mask,
    sfxn=sfxn,
    minimize=False,
    pack=False,
    database=context.parameter_database,
)
```

With both flags disabled, the helper returns a fixed-coordinate, weighted
cross-mask block-pair interaction score from one complex. It performs no
separated-state subtraction and is not a binding free energy despite its
historical name. `minimize` defaults to `True`; `pack=True` additionally invokes
local repacking. Set those options only when the resulting refined structure is
part of the intended scoring convention.

## Troubleshooting

`prepare_ligands=True` is strict by default. An unpreparable ligand raises
`LigandPreparationError` rather than silently disappearing. Pass
`strict_ligands=False` only when dropping unprepared ligands is acceptable.

If pose construction says `Unrecognized 3lc <NAME>`, the residue code was not in
the active `CanonicalOrdering`. Usually this means the ligand was not prepared
or was skipped under lenient preparation.

If a ligand appears to score as zero, make sure the score function was built
from the ligand-extended database:

```python
sfxn = beta2016_score_function(device, param_db=context.parameter_database)
```

## Module Inventory

The ligand package is organized around a small set of responsibilities:

- `preparation.py`: user-facing preparation APIs.
- `detect.py`: non-standard residue detection and MOL2/SMILES readers.
- `structure_to_smiles.py`: SMILES from AtomArray bond tables.
- `dimorphite_dl.py`: pH-aware protonation-state enumeration.
- `conformer_generation.py`: RDKit distance-geometry coordinates.
- `openbabel_compat.py`: OpenBabel compatibility for MMFF94 mol2 generation.
- `mol3d.py`: MMFF94 charges by atom index.
- `rdkit_mol.py`: AtomArray to RDKit `Mol`.
- `atom_typing.py`: tmol's port of Rosetta generic-potential atom
  classification.
- `chi_topology.py`: rotatable-bond and `PROTON_CHI` topology.
- `residue_builder.py`: `RawResidueType` construction.
- `registry.py`: database injection and cartbonded params.
- `params_file.py`: `.tmol` YAML load/injection.
- `params_io.py`: `.params` and `.tmol` writing.

## Related examples

- {doc}`Ligands and Parameter Files </tutorial/07_ligand_and_params>` traces
  authoritative chemistry, `.tmol` persistence, the experimental Rosetta
  writer, pocket selection, and local interaction scoring.
- {doc}`Working with DNA and RNA </tutorial/08_nucleic_acids>` prepares a ligand
  in an RNA aptamer and repacks nearby RNA while holding the ligand fixed.
