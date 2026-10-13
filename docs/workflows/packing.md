# Packing

Start with a prepared `PoseStack` and a score function from the same parameter database.

Fixed-sequence repacking changes conformations while preserving block identities.
Create a `PackerTask`, restrict it to repacking, and attach samplers:

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

For mutation or design, construct a task with the allowed identities instead
of calling `restrict_to_repacking()`. Construct the masks explicitly.
See {doc}`FastRelax </tutorial/06_fast_relax>`
to combine packing with minimization, or {doc}`DNA/RNA <nucleic_acids>` for
nucleic-acid samplers.

## Examples and reference

{doc}`Packing tutorial </tutorial/04_packing_and_mutation_scan>` · {doc}`Packer extensions </tutorial/13_extending_the_packer>` · {doc}`Packing API </api/pack>`
