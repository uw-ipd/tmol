# Packing

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
task instead of this fixed-sequence recipe. TMol does not provide a built-in
mutation-scan or Rosetta resfile protocol.

## Related examples

- The {doc}`packing example </tutorial/04_packing_and_mutation_scan>` expands
  this recipe into local repacking and carefully scoped mutation/design
  experiments.
- {doc}`FastRelax </tutorial/06_fast_relax>` composes packing with minimization.
- {doc}`Working with DNA and RNA </tutorial/08_nucleic_acids>` uses an
  NA-specific chi sampler and explicit masks.
