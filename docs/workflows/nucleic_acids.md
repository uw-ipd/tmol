# Nucleic acids

Use this compact recipe to score canonical DNA or RNA and repack selected
nucleic-acid blocks. The linked tutorial covers score interpretation and
worked DNA and RNA examples.

> - **Prerequisites:** {doc}`Scoring </user_guide/scoring>` and
>   {doc}`Packing </workflows/packing>`.
> - **Deep tutorial:** {doc}`08 — Working with DNA and RNA
>   </tutorial/08_nucleic_acids>`.
> - **Related workflows:** {doc}`Optimization </user_guide/optimization>` and
>   {doc}`Ligand preparation </user_guide/ligands>`.
> - **API reference:** {doc}`Scoring </api/score>` and
>   {doc}`Packing </api/pack>`.
> - **Rosetta mapping:** {doc}`DNA and RNA </tutorial/rosetta_crosswalk>`.

Canonical DNA and RNA use the same `PoseStack` and score-function interfaces as
proteins. Load a Biotite structure with the default parameter database, then
build the score function from that same database:

```python
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.score import beta2016_score_function

param_db = ParameterDatabase.get_default()
pose_stack = pose_stack_from_biotite(structure, device, param_db=param_db)
sfxn = beta2016_score_function(device, param_db=param_db)
scores = sfxn.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)
```

The beta2016-style preset includes the combined nucleic-acid torsion model,
ordinary all-atom nonbonded terms, and nucleic-acid cartbonded parameters.
Interpret weighted outputs as TMol score units, not physical free energies.

## Repack selected bases

Use `NaChiRotamerSampler` with an explicit block mask:

```python
from tmol.pack import pack_rotamers
from tmol.pack import PackerPalette, PackerTask
from tmol.pack.rotamer import IncludeCurrentSampler
from tmol.pack.rotamer import NaChiRotamerSampler

task = PackerTask(pose_stack, PackerPalette())
task.restrict_to_repacking()
task.disable_packing_by_block_mask(~selected_na_blocks)
task.add_conformer_sampler(
    NaChiRotamerSampler.from_database(
        param_db, device, chi_sample_level=1, sample_syn=True
    )
)
task.add_conformer_sampler(IncludeCurrentSampler())
packed = pack_rotamers(pose_stack, sfxn, task)
```

This sampler changes glycosidic chi and configured hydroxyl proton chis. It reads
sugar pucker from the input but does not sample pucker. Because this task is
restricted to repacking, it also does not change base identity.

For protein–DNA or RNA–ligand systems, keep block masks explicit. If ligand
preparation extends the parameter database, build both the pose and score
function from the returned context. Generic Cartesian or kinematic minimization
can follow packing, but its movable atoms must be selected separately; TMol does
not provide a complete RosettaDNA specificity, RNA fragment-assembly, docking,
or ligand-pose protocol.
