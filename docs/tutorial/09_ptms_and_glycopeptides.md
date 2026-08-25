# PTMs and covalently linked glycans

This workflow imports a modified structure, prepares every unknown chemical
component programmatically, scores it, repacks protein conformations, minimizes
the complete covalent system, and exports the result.

## Import

```python
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure
from tmol.io import pose_stack_from_biotite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = get_structure(
    CIFFile.read("modified_peptide.cif"),
    model=1,
    include_bonds=True,
)
pose, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    no_optH=True,
    return_context=True,
)
```

`include_bonds=True` is essential: glycosidic and other noncanonical bonds must
be present in the Biotite bond table. Peptide and nucleotide monomers are
recognized from CCD type plus backbone topology. Sugars are recognized from
CCD saccharide type, while their lower/upper/branch roles come from the actual
inter-component graph.

Inspect the chosen paths:

```python
block_types = [
    pose.packed_block_types.active_block_types[int(index)]
    for index in pose.block_type_ind64[0]
    if index >= 0
]
for block_type in block_types:
    print(
        block_type.name3,
        block_type.base_name,
        block_type.properties.polymer.polymer_type,
        tuple(connection.name for connection in block_type.connections),
    )
```

A generated phosphoserine reports `name3="SEP"`, `base_name="SER"`, and normal
peptide `down`/`up` connections. An attached NAG reports carbohydrate polymer
semantics and graph-derived connections.

## Score and differentiate

```python
from tmol import beta2016_score_function

score_function = beta2016_score_function(
    device, param_db=context.parameter_database
)
scorer = score_function.render_whole_pose_scoring_module(pose)
coords = pose.coords.detach().clone().requires_grad_(True)
score = scorer(coords).sum()
score.backward()
assert torch.isfinite(score)
assert torch.isfinite(coords.grad).all()
```

## Repack protein conformations

Modified amino acids inherit their canonical parent's Dunbrack identity.
Generated carbohydrates do not use a sugar rotamer library, so retain their
deposited conformation during the discrete packing stage:

```python
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

task = PackerTask(pose, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(IncludeCurrentSampler())
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(context.parameter_database, device)
)
task.add_conformer_sampler(FixedAAChiSampler())

freeze_carbohydrates = torch.tensor(
    [[
        block_type.properties.polymer.polymer_type == "carbohydrate"
        for block_type in block_types
    ]],
    device=device,
)
task.disable_packing_by_block_mask(freeze_carbohydrates)
packed = pack_rotamers(pose, score_function, task, verbose=False)
```

## Minimize and round-trip

```python
from tmol import run_cart_min
from tmol.io import biotite_from_pose_stack

minimized = run_cart_min(
    packed.clone(),
    score_function,
    optimizer_kwargs={"max_iter": 200},
)
output = biotite_from_pose_stack(
    minimized,
    co=context.canonical_ordering,
)
```

The output uses deposited component names and preserves explicit covalent
bonds. Reuse a build `context` only for structures with the same component
chemistry and attachment pattern.

## Current modeling boundary

Generated sugars receive general bonded and nonbonded parameters and can be
Cartesian-minimized. They do not receive Rosetta `sugar_bb`, linkage-conformer,
or ring-pucker statistics. Comparing absolute Rosetta and TMol totals is
therefore not meaningful; compare shared terms or controlled energy changes.
Generated PTMs follow the same rule: a generated phosphate uses general-ligand
`PG3`/`OG2`, not the curated `Phos`/`OOC` PTM-patch parameterization.
