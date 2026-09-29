# Ligand Preparation

This guide is a focused reference for preparing, reusing, and scoring ligand
chemistry. The linked tutorial provides a complete protein–ligand walkthrough.

> - **Prerequisites:** {doc}`Integrations </user_guide/integrations>` for
>   Biotite input and {doc}`Scoring </user_guide/scoring>`.
> - **Deep tutorial:** {doc}`07 — Ligands and Parameter Files
>   </tutorial/07_ligand_and_params>`.
> - **Related workflows:** {doc}`Packing </workflows/packing>` and
>   {doc}`Nucleic acids </workflows/nucleic_acids>`.
> - **API reference:** {doc}`Ligands </api/ligand>`,
>   {doc}`Input and Output </api/io>`, and {doc}`Scoring </api/score>`.
> - **Rosetta mapping:** {doc}`Ligands and residue-parameter files
>   </tutorial/rosetta_crosswalk>`.

TMol can turn a non-standard, non-polymer residue into a parameterized residue
type with protonated 3D coordinates, MMFF94 partial charges,
generic-potential-style atom types used by TMol, and cartbonded parameters. The
prepared ligand is injected into a new `ParameterDatabase` and can then be
scored and minimized like a normal residue. These atom-type names do not by
themselves make a ligand parameterization usable by Rosetta.

## Chemistry Classes

`prepare_ligands`, and `prepare_ligands=True` on the IO entry points, is not
limited to free small molecules. The backbone is inferred from a residue's own
connectivity, so every class below enters by the same path and needs no
hand-written parameter file. Each row names a shipped fixture you can run
directly.

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

The call is the same whichever row you pick:

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
the same components; see [Reuse Prepared Context](#reuse-prepared-context).

A prepared polymer residue records which backbone it was recognized as in
`properties.polymer.backbone_type`: `alpha_aa`, `nonstandard_aa`, `dna`, `rna`,
or `nonstandard_na`. That value selects the terminus patches and the torsion
terms the residue is scored with, so it is the first thing to check when a
residue is not treated the way you expect.

For how this maps onto Rosetta's `molfile_to_params_polymer.py` / `MakeRotLib`
workflow, and which of its steps have no counterpart here, see the
{doc}`Rosetta crosswalk <../tutorial/rosetta_crosswalk>`.

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

## Loading Complexes

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
    "ligand.mol2", "ligand.tmol", res_name="LIG", format="tmol", mode="auto"
)
pose_stack, context = pose_stack_from_file(
    "complex.pdb", device,
    ligand_params_files=["ligand.tmol"],
    use_ccd=False,
    return_context=True,
)
```

Use the complex's ligand residue name and matching atom names. Alternatively,
pass the database returned by `prepare_ligand_from_mol2()` as `param_db=`.
Neither route regenerates known ligand parameters. `use_ccd=False` reads only
supplied information; the default `True` permits CCD completion.

For mixtures of known and unknown components, add `prepare_ligands=True`: saved
parameters are reused, and only unknown chemistry enters preparation. Unknown
residues need chemical bond orders from the input or CCD; otherwise the error
names the residue that needs more information. PDB `CONECT` records alone do not
provide those orders.

## Reuse Prepared Context

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

The ligand-prep script writes a TMol `.tmol` parameter bundle:

```bash
python scripts/ligand_prep/smiles_to_params.py "<SMILES>" <out_prefix> \
    --res-name LG1 --ph 7.4
```

Useful flags include `--no-protonate`, `--heavy-chi-samples`, and
`--seed` for a reproducible conformer.

`.tmol` is TMol's only parameter format. TMol does not read or write Rosetta
`.params`; use a Rosetta-native preparation workflow to parameterize a ligand
for Rosetta.

## Interaction Scores

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

## Public API

Import supported ligand-preparation functions from `tmol.ligand`. Files whose
names begin with an underscore are implementation details and may change
without a compatibility alias. The {doc}`ligand API reference </api/ligand>`
lists the currently supported exports.
