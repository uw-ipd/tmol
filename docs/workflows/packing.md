# Packing

Start with a prepared `PoseStack` and a score function from the same parameter database.

Fixed-sequence repacking changes conformations while preserving block identities.
Create a `PackerTask`, restrict it to repacking, and attach samplers:

```python
from tmol import pack_rotamers, PackerPalette, PackerTask
from tmol.pack import (
    create_dunbrack_sampler_from_database,
    FixedAAChiSampler,
    IncludeCurrentSampler,
)

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

For mutation or design, construct a task with the allowed identities instead
of calling `restrict_to_repacking()`. Construct the masks explicitly.
See {ref}`FastRelax </tutorial/01_tmol_tutorial.ipynb#2.-scoring-and-optimization>`
to combine packing with minimization, or {doc}`DNA/RNA <nucleic_acids>` for
nucleic-acid samplers.

## Examples and reference

{ref}`Scoring and optimization tutorial </tutorial/01_tmol_tutorial.ipynb#2.-scoring-and-optimization>` · {ref}`Packer extensions </tutorial/01_tmol_tutorial.ipynb#8.-advanced-topics>` · {doc}`Packing API </api/pack>`
