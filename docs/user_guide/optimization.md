# Optimization

tmol exposes the same primitives used by Rosetta-style refinement workflows:
hydrogen placement, missing-side-chain rebuild, fixed-sequence repacking,
Cartesian minimization, kinematic minimization, and relax.

## Cartesian Minimization

Use `run_cart_min()` to optimize coordinates directly:

```python
from tmol.optimization.minimizers import run_cart_min

minimized_pose_stack = run_cart_min(pose_stack, sfxn)
```

Pass a boolean coordinate mask to restrict which atoms move:

```python
coord_mask = torch.zeros(pose_stack.coords.shape[:-1], dtype=torch.bool, device=device)
coord_mask[:, ligand_atom_indices] = True
minimized_pose_stack = run_cart_min(pose_stack, sfxn, coord_mask=coord_mask)
```

## Repacking

Fixed-sequence side-chain optimization uses a `PackerTask` and one or more
conformer samplers:

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

Disable packing outside a selected residue mask with
`task.disable_packing_by_block_mask()`.

## Missing Side Chains and Hydrogens

`pose_stack_from_biotite()` automatically routes blocks with missing heavy atoms
through `build_missing_sidechains()`. By default it also places and optimizes
hydrogens for complete residues. Pass `no_optH=True` to skip that hydrogen
optimization path.

```python
pose_stack = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    no_optH=False,
)
```

Ligand heavy atoms must be present in the input. tmol can prepare and protonate
ligands, but the side-chain-rebuild sampler only handles polymer residues.

## Kinematic Minimization

Kinematic minimization optimizes internal degrees of freedom over a fold forest:

```python
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MoveMap
from tmol.optimization.minimizers import run_kin_min

fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
move_map = MoveMap.from_pose_stack(pose_stack)
move_map.move_all_named_torsions = True
kin_minimized = run_kin_min(pose_stack, sfxn, fold_forest, move_map)
```

`CartesianMoveMap` and `MoveMap` control different spaces. A
`CartesianMoveMap` is a lightweight wrapper around a boolean atom-coordinate
mask; it is used by Cartesian FastRelax and does not describe torsions or
jumps. A `MoveMap` controls internal main-chain, side-chain, named-torsion, and
rigid-body jump DOFs for kinematic minimization. Constructing a `MoveMap` does
not enable those DOFs: set the relevant flags or per-residue masks explicitly,
as above.

Use Cartesian minimization when an atom coordinate mask is the natural control
surface. Use kinematic minimization when torsion and rigid-body DOFs should be
the optimization variables.

## Relax

`tmol.relax.fast_relax.fast_relax()` combines repacking and minimization over a
schedule of score-function weights. It is the highest-level refinement
primitive in the package:

```python
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import CartesianMoveMap
from tmol.pack.packer_task import PackerPalette
from tmol.relax.fast_relax import fast_relax

palette = PackerPalette()
move_map = CartesianMoveMap()  # coord_mask=None allows all atom coordinates
fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

relaxed_pose_stack = fast_relax(
    pose_stack,
    sfxn,
    palette,
    move_map,
    fold_forest,
)
```

The default FastRelax minimizer is Cartesian, so it reads
`CartesianMoveMap.coord_mask` and ignores the fold forest. The fold forest is
still a required argument because the same protocol can run a custom
kinematic `min_fn`; in that case pass a full `MoveMap` and have `min_fn` call
`run_kin_min()` with the supplied fold forest and move map.
