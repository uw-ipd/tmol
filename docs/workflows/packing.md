# Packing

Use this compact recipe for fixed-sequence repacking. The linked tutorial
develops local repacking and explicitly scoped mutation or design experiments.

> - **Prerequisites:** A prepared `PoseStack` and a score function built from the
>   same parameter database.
> - **Deep tutorial:** {doc}`04 — Packing and Mutation Scan
>   </tutorial/04_packing_and_mutation_scan>`.
> - **Advanced extension tutorial:** {doc}`13 — Extending the Packer and
>   Inspecting Rotamers </tutorial/13_extending_the_packer>`.
> - **Related workflows:** {doc}`Optimization </user_guide/optimization>` and
>   {doc}`Nucleic acids </workflows/nucleic_acids>`; for CPU execution, see
>   {doc}`CPU threading </user_guide/cpu_threading>`.
> - **API reference:** {doc}`Packing </api/pack>` and
>   {doc}`Relax </api/relax>`.
> - **Rosetta mapping:** {doc}`Packing, design, and mutation scans
>   </tutorial/rosetta_crosswalk>`.

Fixed-sequence repacking searches side-chain or nucleic-acid chi conformers
without changing block identity. The usual workflow creates a `PackerTask`,
restricts it to repacking, attaches rotamer samplers, and calls
`pack_rotamers()`. Mutation or sequence design requires explicit identity
masks; TMol does not provide Rosetta resfiles or a built-in mutation-scan
protocol.

```python
from tmol.pack import pack_rotamers
from tmol.pack import PackerPalette, PackerTask
from tmol.pack.rotamer.dunbrack import (
    create_dunbrack_sampler_from_database,
)
from tmol.pack.rotamer import FixedAAChiSampler
from tmol.pack.rotamer import IncludeCurrentSampler

task = PackerTask(pose_stack, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(context.parameter_database, device)
)
task.add_conformer_sampler(FixedAAChiSampler())
task.add_conformer_sampler(IncludeCurrentSampler())

packed_pose_stack = pack_rotamers(pose_stack, sfxn, task)
```

To keep a subset of residues fixed, build a boolean block mask and disable
packing for those blocks:

```python
task.disable_packing_by_block_mask(fixed_block_mask)
```

The protein-ligand refinement example uses this pattern to repack protein side
chains while holding the ligand block fixed.

`restrict_to_repacking()` intersects the task with each block's original
identity. Mutation or design therefore needs an explicitly constructed identity
task instead of this fixed-sequence recipe. {doc}`FastRelax
</tutorial/06_fast_relax>` composes packing with minimization, while
{doc}`08 — Working with DNA and RNA </tutorial/08_nucleic_acids>` uses an
NA-specific chi sampler and explicit masks.

## Protonation alternatives

Packing keeps each residue's protonation state by default. To let free titratable side chains take other states supported by AtomWorks
and the chemical database, record the alternatives when
building the pose and turn them on in the palette:

```python
from tmol.pack.protonation_alternatives import chosen_protonation_variants

pose_stack = pose_stack_from_biotite(
    structure, device, protonation_alternatives=True
)
task = PackerTask(pose_stack, PackerPalette(protonation_alternatives=True))
# ... restrict_to_repacking and samplers as above ...
packed_pose_stack = pack_rotamers(pose_stack, sfxn, task)
for choice in chosen_protonation_variants(packed_pose_stack):
    print(choice.chain, choice.res_label, choice.label, choice.charge)
```

Only sites without input hydrogens or external bonds record alternatives; metal
coordination and disulfides remain fixed. Candidate states come from calling
AtomWorks on either side of its Dimorphite-DL titration boundaries within one pH
unit of `ligand_ph`. Candidates must match a database hydrogen-count pattern;
free histidine tautomers are also offered with equal offsets. This uses the
current AtomWorks model, including its aromatic-nitrogen pKa of about 4.35,
thiol pKa of 9.12 and amine pKa of 8.16, rather than separate protein estimates.

Each proton gained contributes `1.364 * (pH - pKa)` kcal/mol relative to the
assigned state. Packing adds these offsets to one-body energies. FastRelax
uses the same offsets when accepting poses if the palette enables alternatives;
within a fixed state they are constant and do not affect coordinate gradients.
`protonation_state_energy(pose_stack)` exposes the offsets separately from raw
score-function totals. The score function is not calibrated for proton transfer,
so chosen states remain hypotheses. Terminal states are held fixed during this
side-chain search, including neutral amino termini supported by the database.
