# Minimization and FastRelax

Minimize a prepared `pose_stack` with a matching score function `sfxn`.
Cartesian minimization moves atoms; kinematic minimization changes torsions and
rigid-body degrees of freedom. FastRelax alternates packing and minimization.

## Cartesian minimization

Use `run_cart_min()` to optimize coordinates directly:

```python
from tmol import run_cart_min

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
from tmol.score import create_mainchain_coordinate_constraints
from tmol import ScoreType

constrained_pose = create_mainchain_coordinate_constraints(pose_stack)
sfxn.set_weight(ScoreType.constraint, 1.0)
minimized_pose_stack = run_cart_min(constrained_pose, sfxn)
```

The helper uses a 0.5 Å harmonic standard deviation. For the standard amino-acid
types, the declared main-chain atoms are N, CA, and C, not O. The lower-level
`ConstraintSet` and `ConstraintEnergyTerm` interfaces support harmonic and
bounded atom-pair distances, harmonic coordinates, and circular-harmonic
four-atom torsions.

## Missing side chains and hydrogens

`pose_stack_from_atom_array()` automatically rebuilds the missing heavy atoms
of incomplete side chains by packing them. By default it also optimizes
supported hydrogen positions and NHQ flips. Pass `no_optH=True` to preserve
finite supplied protein hydrogens and build only missing atoms.

For standard residues and the default parameter database, TMol automatically
reuses the structure-independent construction and packing setup. If many
structures share prepared ligand definitions, explicitly reuse a
`PoseBuildContext` as described in the ligand guide.

```python
pose_stack = pose_stack_from_atom_array(
    structure,
    device,
    prepare_ligands=True,
)
```

Missing non-polymer atoms can be reconstructed from prepared conformers when
sufficient coordinate anchors are available. Underdetermined geometry raises
an error. These starting conformers are not recovered experimental coordinates.

## Kinematic minimization

Kinematic minimization optimizes internal degrees of freedom over a fold forest:

```python
from tmol import FoldForest, MoveMap, run_kin_min

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

Use Cartesian masks to select atoms; use a `MoveMap` to select torsions and jumps.

## Relax

`fast_relax()` combines repacking and minimization over a schedule of
score-function weights:

```python
from tmol import CartesianMoveMap, FoldForest, PackerPalette, fast_relax

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

Default repacking uses the score function's parameter database, including
nucleic-acid chi sampling and joint conformers for covalently attached groups.
Free ligands retain their current conformation during packing and can move in
Cartesian minimization. Custom `task_operations` replace this sampler setup.
Supplemental Dunbrack chi products obey the sampling budget before enumeration;
required library states are retained. Frozen chi keep their input angles for
repacking, or generated ideal angles for a new chemical identity.

The default minimizer is Cartesian and reads
`CartesianMoveMap.coord_mask`; the fold forest is accepted by the common
protocol but is not used by that minimizer. To minimize kinematic degrees of
freedom, pass a configured `MoveMap`, a `FoldForest`, and a compatible
kinematic `min_fn`.

On CUDA, `fast_relax()` automatically uses graph replay for poses containing
DNA or RNA, where repeated kernel-launch overhead is significant. Protein-only
and protein–ligand poses remain eager by default because graph capture does
not consistently recover its setup cost in a single relaxation. For repeated
protein workloads, pass `cuda_graph=True` to favor steady-state throughput;
pass `cuda_graph=False` to force eager execution.

Graph replay reduces launch overhead and retains working tensors; measure both
runtime and memory. Custom minimizers cannot use `cuda_graph=True`. CPU
execution remains eager.

## Examples and reference

{ref}`Scoring and optimization tutorial </tutorial/01_tmol_tutorial.ipynb#2.-scoring-and-optimization>` · {ref}`Advanced kinematics </tutorial/01_tmol_tutorial.ipynb#7.-advanced-kinematics>` · {doc}`Optimization API </api/optimization>` · {doc}`Kinematics API </api/kinematics>`
