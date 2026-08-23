# Minimization and FastRelax

This guide collects concise recipes for preparation, constraints, minimization,
and relax. The tutorials explain coordinate choices, masks, and schedules in
depth.

> - **Prerequisites:** {doc}`Scoring </user_guide/scoring>` and, for relax,
>   {doc}`Packing </workflows/packing>`.
> - **Deep tutorials:** {doc}`05 — Minimization, Constraints, and Kinematics
>   </tutorial/05_minimization_constraints_kinematics>` and
>   {doc}`06 — FastRelax </tutorial/06_fast_relax>`.
> - **Related workflows:** {doc}`Ligand preparation </user_guide/ligands>` and
>   {doc}`Nucleic acids </workflows/nucleic_acids>`.
> - **API reference:** {doc}`Optimization </api/optimization>`,
>   {doc}`Kinematics </api/kinematics>`, and {doc}`Relax </api/relax>`.
> - **Rosetta mapping:** {doc}`Minimization, constraints, kinematics, and
>   FastRelax </tutorial/rosetta_crosswalk>`.

TMol exposes hydrogen placement, missing-side-chain rebuild, Cartesian and
kinematic minimization, constraints, and relax. Fixed-sequence repacking is
covered separately in the {doc}`Packing workflow </workflows/packing>`.

## Cartesian Minimization

Use `run_cart_min()` to optimize coordinates directly:

```python
from tmol.optimization import run_cart_min

minimized_pose_stack = run_cart_min(pose_stack, sfxn)
```

Pass a boolean coordinate mask to restrict which atoms move:

```python
coord_mask = torch.zeros(pose_stack.coords.shape[:-1], dtype=torch.bool, device=device)
coord_mask[:, ligand_atom_indices] = True
minimized_pose_stack = run_cart_min(pose_stack, sfxn, coord_mask=coord_mask)
```

## Constraints

Constraints affect optimization only when they are attached to the pose and the
score function gives the constraint term a nonzero weight. This helper returns a
new pose with harmonic coordinate restraints targeting a copy of each residue
type's declared main-chain atom coordinates:

```python
from tmol.score.constraint import create_mainchain_coordinate_constraints
from tmol.score import ScoreType

constrained_pose = create_mainchain_coordinate_constraints(pose_stack)
sfxn.set_weight(ScoreType.constraint, 1.0)
minimized_pose_stack = run_cart_min(constrained_pose, sfxn)
```

The helper uses a 0.5 Å harmonic standard deviation. For the standard amino-acid
types, the declared main-chain atoms are N, CA, and C, not O. The lower-level
`ConstraintSet` and `ConstraintEnergyTerm` interfaces support harmonic and
bounded atom-pair distances, harmonic coordinates, and circular-harmonic
four-atom torsions. TMol does not currently provide Rosetta's constraint-file
parser, ambiguous-constraint layer, or a dedicated three-atom angle constraint.

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

Ligand heavy atoms must be present in the input. TMol can prepare and protonate
ligands, but the side-chain-rebuild sampler only handles polymer residues.

## Kinematic Minimization

Kinematic minimization optimizes internal degrees of freedom over a fold forest:

```python
from tmol.kinematics import FoldForest
from tmol.kinematics import MoveMap
from tmol.optimization import run_kin_min

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

`fast_relax()` combines repacking and minimization over a schedule of
score-function weights:

```python
from tmol.kinematics import CartesianMoveMap, FoldForest
from tmol.pack import PackerPalette
from tmol.relax import fast_relax

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

The default minimizer is Cartesian and reads
`CartesianMoveMap.coord_mask`; the fold forest is accepted by the common
protocol but is not used by that minimizer. To minimize kinematic degrees of
freedom, pass a configured `MoveMap`, a `FoldForest`, and a compatible
kinematic `min_fn`.
