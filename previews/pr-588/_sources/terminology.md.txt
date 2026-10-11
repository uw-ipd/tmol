# Terminology and modeling choices

<span id="posestack-pose-block-and-atom"></span>

## Poses, blocks, and atoms

A {class}`tmol.pose.PoseStack` stores molecular systems as padded tensors on one
PyTorch device. `n_poses` is the batch dimension. Use `real_atoms` and block-index
tensors to distinguish molecular entries from padding.

A **block** is a chemical unit described by one `RefinedResidueType`: an amino
acid, nucleotide, ligand fragment, ion, or other residue type. A single system
still uses a `PoseStack`.

<span id="parameterdatabase-and-packedblocktypes"></span>

## Chemical definitions

{class}`tmol.database.ParameterDatabase` holds chemical definitions and scoring
parameters. Extending it returns a new database.

{class}`tmol.pose.PackedBlockTypes` holds block types and device-resident setup
data. Reuse it for compatible structures on the same device. It does not cache
conformation energies.

<span id="deposited-atoms-and-built-atoms"></span>

## Deposited and built atoms

TMol assigns chemical types to input atoms and may build missing atoms. Check
histidine state, termini, disulfides, and noncanonical chemistry in the I/O build
context. Prefer mmCIF through Biotite when metadata or explicit ligand bonds
matter; PDB cannot preserve every preparation decision.

<span id="the-no-opth-choice"></span>

## Polar hydrogens: `no_optH`

`no_optH=False` optimizes movable polar hydrogens during preparation and requires
a score function. Use `no_optH=True` to skip this step when retaining supplied
proton geometry or deferring optimization. This choice affects coordinates and
scores; record it with your results.

<span id="rendered-scorers-and-changing-coordinates"></span>

## Reusing a scorer

A {class}`tmol.score.ScoreFunction` renders a PyTorch module for a specific
`PoseStack` layout. Reuse it when only coordinates change. Render a new scorer
after changing block types, atom counts, connectivity, or batch layout.

## Cartesian and kinematic movement

Cartesian minimization changes selected atom coordinates. Kinematic minimization
changes internal and rigid-body degrees of freedom selected by a
{class}`tmol.kinematics.MoveMap` over a {class}`tmol.kinematics.FoldForest`.
Compare the two only with matched masks, weights, stopping rules, and iteration
budgets. See {doc}`optimization <user_guide/optimization>` for examples.
