# Generalized chemical preparation

tmol targets scoring, packing and relaxation of arbitrary molecules through one
chemical representation. Metal support is planned; the implementation details
below describe the current PR #503 snapshot.

PR #503 extends the ligand machinery to supported noncanonical polymer residues,
modified nucleic acids, covalent ligands and glycan attachments. Use the
[generalized chemistry guide](../../docs/noncanonical_chemistry.rst),
[scoring/packing/relaxation recipes](../../docs/chemistry_workflows.rst), and
[Rosetta comparison](../../docs/rosetta_comparison.rst). Those guides identify
the feature revision and distinguish implementation from measured accuracy.

## Prepare a deposited structure

```python
import torch
from tmol.io import pose_stack_from_cif
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose, context = pose_stack_from_cif(
    "complex.cif", device,
    prepare_ligands=True, strict_ligands=True,
    ligand_seed=20250828, return_context=True,
)
sfxn = beta2016_score_function(device, param_db=context.parameter_database)
scorer = sfxn.render_whole_pose_scoring_module(pose)
print(scorer(pose.coords))
```

The keyword is `prepare_ligands=True`. Use the prepared database for both
scoring and library samplers. Prefer mmCIF with reliable chemical definitions
and bonds; the default PDB-text reader is not an equivalent automatic
noncanonical-preparation route. A bonded AtomArray, including one with NaN
coordinates for missing atoms, can use `pose_stack_from_biotite`.

Install `tmol[ligand]` for preparation paths that invoke OpenBabel, including
derived-SMILES preparation from CIF. The intended charge model is MMFF94, but
warned fallback charge models and stereochemical compatibility transformations
are possible. Retain preparation warnings and inspect the result.

## Free-ligand entry points

`prepare_ligand_from_mol2`, `prepare_ligand_from_cif` and
`prepare_ligand_from_smiles` return an extended
`(ParameterDatabase, CanonicalOrdering)` pair. SMILES and MOL2 entry points
prepare free ligands; they do not infer polymer context. MOL2 with authoritative
charges can preserve its input names, coordinates, bonds and charges instead of
generating a new conformer.

## Persistence and reuse

For compatible chemistry, reuse `PoseBuildContext` through
`pose_stack_from_biotite(structure, device, context=context)`. This avoids
repeating parameter preparation and packed-type construction. Render a new
scorer for a different pose layout or topology.

For persistence, `prepare_ligands(..., params_output="prepared.tmol")` writes
prepared records. Reuse them through `params_files` at the preparation layer or
`ligand_params_files` at the pose-construction layer. `inject_params_file`
extends a database directly. Preserve the `.tmol` file with input chemistry,
seed, pH, environment and source commit; edit deliberately when correcting
parameters.

## Fragmentation is separate from group packing

An integer `tmol_fragment_id` annotation partitions one ligand into connected
scoring blocks. Every atom in that residue must have a positive fragment ID;
zero is reserved for atoms outside fragmented residues. Repeated instances of
one residue name must have the same layout. Generated fragment names must not
collide with other chemistry.

Current restrictions require connected fragments with at least three heavy
atoms and at most four connections. An atom cannot participate in multiple cut
bonds, a four-atom bonded path cannot cross multiple cuts, and cuts cannot
break supported hbond/LK-ball acceptor-frame geometry. These restrictions keep
terms within supported one-/two-block scoring paths.

`calculate_fragment_interactions` is exported from `tmol.score`. The pose's
`fragmented_ligand_mapping` records the original component and fragment blocks.
Select partners explicitly, rather than assuming the ligand is the last block.
`write_pose_stack_pdb` and `biotite_from_pose_stack` restore the original residue
identity by default; `merge_fragments=False` exports separate blocks.

Group packing instead couples an anchor and attached blocks to one common
conformer choice. It requires the explicit group sampler described in the
workflow guide. It does not automatically sample free ligands, and merely
carrying a free ligand through the fallback sampler is not conformer search.

## Failure interpretation

With `strict_ligands=True`, detected unpreparable components raise
`LigandPreparationError`. `strict_ligands=False` can warn and drop them. Covalent
attachments are now supported by the preparation machinery; they are no longer
a blanket rejection category. Automatic preparation still rejects
metal-containing unknown components and unsupported elements, and requires
identifiable polymer profiles and adequate chemical definitions.

Strict preparation does not verify every force-field parameter: unmatched
generic torsions can be omitted, and warned preparation fallbacks remain
possible. Validate atom retention, connectivity, parameter coverage, actual
sampling, finite derivatives and resulting geometry independently. The
[workflow guide](../../docs/chemistry_workflows.rst) documents sampler selection,
budget limitations and the FastRelax task configuration.
