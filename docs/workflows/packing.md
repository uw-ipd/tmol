# Packing

Packing searches side-chain conformers while holding the protein sequence fixed.
The usual workflow creates a `PackerTask`, restricts it to repacking, attaches
rotamer samplers, and calls `pack_rotamers()`.

```python
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.pack.packer_task import PackerPalette, PackerTask
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import (
    create_dunbrack_sampler_from_database,
)
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler

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
