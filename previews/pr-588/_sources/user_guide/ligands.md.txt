# Ligand Preparation

Prepare ligand parameters, reuse them across structures, and calculate interaction scores.

Preparation produces protonated coordinates, MMFF94 charges, TMol atom types,
and Cartesian bonded parameters in a new `ParameterDatabase`. The resulting
types support scoring and minimization.

<span id="chemistry-classes"></span>

## Supported chemistry

`prepare_ligands=True` also handles nonstandard polymers and covalent groups.
Backbone type is inferred from connectivity. The following fixtures use this
same preparation path:

| Class | Example fixture |
| --- | --- |
| Modified alpha-amino acid (PTM) | `ncaa_fixtures/collagen_hyp_1bkv.cif` (4-hydroxyproline), `ncaa_fixtures/phosphopeptide_5ema.cif` (phosphoserine) |
| N-substituted backbone | `ncaa_fixtures/nmethyl_peptide_6mvz.cif` |
| Beta and gamma backbones | `ncaa_fixtures/beta_peptide_3c3g.cif`, `ncaa_fixtures/gamma_peptide_1gac.cif` |
| D-amino acid | `ncaa_fixtures/6dmz_mod_d.cif` (with its L mirror, `6dmz_mod_l.cif`) |
| Terminal caps | `ncaa_fixtures/capped_peptide_ace_nme.cif`, `capped_peptide_ace_nh2.cif` |
| Chromophore / macrocycle | `ncaa_fixtures/chromophore_nrq_3svu.cif` |
| Modified DNA | `ncaa_fixtures/na_dna_5mc_1d17.cif` (5-methylcytosine), `na_dna_8og_183d.cif` (8-oxoguanine), `na_dna_ttd_1ttd.cif` (thymine dimer) |
| Modified RNA | `ncaa_fixtures/na_rna_psu_1bzt.cif` (pseudouridine), `na_rna_2ome_310d.cif` (2'-O-methyl) |
| N- and O-glycans | `covalent_fixtures/nglycan_tree_1ax2.cif`, `covalent_fixtures/oglycan_sia_1g1s.cif` |
| Covalent conjugate | `covalent_fixtures/lys_biotin_1bdo.cif` (biotinylated lysine) |
| Covalent inhibitor | `covalent_fixtures/peptide_inhibitor_7zv5.cif` |

For example:

```python
import torch
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.tests.data import data_path

array = atom_array_from_cif(data_path("ncaa_fixtures", "collagen_hyp_1bkv.cif"))
pose, context = pose_stack_from_biotite(
    array, torch.device("cuda"), prepare_ligands=True, return_context=True
)
```

`context.parameter_database` carries the generated chemistry, and
{func}`tmol.ligand.write_params_file` persists it as `.tmol` so a later run reads
it back instead of regenerating. Reuse the context across structures that share
the same components; see [Reuse a prepared context](#reuse-a-prepared-context).

A prepared polymer residue records which backbone it was recognized as in
`properties.polymer.backbone_type`: `alpha_aa`, `nonstandard_aa`, `dna`, `rna`,
or `nonstandard_na`. That value selects the terminus patches and the torsion
terms the residue is scored with, so it is the first thing to check when a
residue is not treated the way you expect.

<span id="entry-points"></span>

## Preparation functions

There are three single-ligand entry points:

```python
from tmol.ligand import (
    prepare_ligand_from_cif,
    prepare_ligand_from_mol2,
    prepare_ligand_from_smiles,
)

param_db, co = prepare_ligand_from_mol2("ligand.mol2")
param_db, co = prepare_ligand_from_cif("ligand.cif")
param_db, co = prepare_ligand_from_smiles("c1ccccc1C(=O)O", res_name="L_1")
```

Each returns a new `(ParameterDatabase, CanonicalOrdering)`. The input database
is not mutated.

MOL2 and CIF input use their bond tables to derive chemistry; source heavy-atom
names are retained. CIF and SMILES preparation apply pH-dependent protonation
(default pH 7.4), generate conformers and calculate MMFF94 charges.

MOL2 preparation and `write_params_from_mol2()` share three modes:

- `mode="auto"` (default): preserve complete inputs with supported, finite,
  charge-conserving partial charges; otherwise run the preparation pipeline.
- `mode="keep"`: require prepared input and preserve its protonation and charges.
- `mode="regenerate"`: always run pH-dependent preparation, including for
  neutralized phosphate inputs with explicit hydrogens and valid partial charges.

A MOL2 charge-model label alone does not establish the intended protonation pH.

<span id="loading-complexes"></span>

## Load a complex

For full protein-ligand structures, load with Biotite and let
`pose_stack_from_biotite()` prepare every non-standard residue:

```python
import biotite.structure as struc
import biotite.structure.io
import torch

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_file, pose_stack_from_biotite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure = atom_array_from_file("complex.cif")

pose_stack, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    param_db=ParameterDatabase.get_default(),
    return_context=True,
)
```

Known residues need only residue names, atom names, and coordinates. For example,
prepare a ligand from MOL2 once, then load coordinate-only PDB or CIF complexes:

```python
from tmol.io import pose_stack_from_file
from tmol.ligand import write_params_from_mol2

write_params_from_mol2(
    "ligand.mol2", "ligand.tmol", res_name="L_1", mode="auto"
)
pose_stack, context = pose_stack_from_file(
    "complex.pdb", device,
    ligand_params_files=["ligand.tmol"],
    return_context=True,
)
```

Use the complex's ligand residue name and matching atom names. Alternatively,
pass the database returned by `prepare_ligand_from_mol2()` as `param_db=`.
Neither route regenerates known ligand parameters.

For mixtures of known and unknown components, add `prepare_ligands=True`: saved
parameters are reused, and only unknown chemistry enters preparation. Unknown
residues need chemical bond orders from the input or CCD; otherwise the error
names the residue that needs more information. PDB `CONECT` records alone do not
provide those orders.

Name a custom ligand with a code containing `_` (TMol generates `L_1`, `L_2`, ...):
no CCD entry has one. A PDB or CIF residue whose atoms or bonds contradict the CCD
entry of its name (PDBbind names ligands `MOL` or `ACT`) is read as its own component.

<span id="reuse-prepared-context"></span>

## Reuse a prepared context

When scoring many structures that contain the same ligand definitions, build
the structure-independent context once:

```python
from tmol.io import (
    build_context_from_biotite,
    pose_stack_from_biotite,
)

context = build_context_from_biotite(struct0, device, prepare_ligands=True)
for structure in structures:
    pose_stack = pose_stack_from_biotite(structure, device, context=context)
```

This skips rebuilding the parameter database, canonical ordering, residue type
set, and packed block types for every structure.

<span id="persist-prepared-ligands"></span>

## Save prepared ligands

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

<span id="smiles-to-params-cli"></span>

## Prepare from SMILES

The ligand-prep script writes a TMol `.tmol` parameter bundle:

```bash
python scripts/ligand_prep/smiles_to_params.py "<SMILES>" <out_prefix> \
    --res-name L_1 --ph 7.4
```

Useful flags include `--no-protonate`, `--heavy-chi-samples`, and
`--seed` for a reproducible conformer.

`.tmol` is TMol's parameter format.

## Interaction scores

Use the ligand-aware score function with an explicit ligand block mask:

```python
from tmol.ops import calculate_block_pair_ddg

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
cross-mask block-pair interaction score. `minimize` defaults to `True`;
`pack=True` additionally invokes
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

## Public API

Import supported ligand-preparation functions from `tmol.ligand`. Files whose
names begin with an underscore are implementation details and may change
without a compatibility alias. The {doc}`ligand API reference </api/ligand>`
lists the currently supported exports.

## Examples and reference

{doc}`Ligand tutorial </tutorial/07_ligand_and_params>` · {doc}`Ligand API </api/ligand>` · {doc}`Scoring </user_guide/scoring>`
