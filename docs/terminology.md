# Terminology and modeling choices

This page collects distinctions that recur across TMol workflows. It is a
concept guide, not a replacement for the {doc}`API reference <api_reference>`.

## PoseStack, pose, block, and atom

A {class}`tmol.pose.PoseStack` stores one or more molecular systems as padded
tensors on one PyTorch device. `n_poses` is the batch dimension. Padding lets
systems with different numbers of atoms and blocks share a tensor layout;
`real_atoms` and block-index tensors distinguish molecular entries from
padding.

TMol uses **block** where Rosetta users often expect **residue**. A block can be
an amino acid, nucleotide, ligand fragment, ion, or another chemical unit
described by one `RefinedResidueType`. A one-pose `PoseStack` is still a
`PoseStack`; there is no separate Rosetta-compatible `Pose` class.

## ParameterDatabase and PackedBlockTypes

{class}`tmol.database.ParameterDatabase` is the immutable source of chemical
definitions and scoring parameters. Extending it for a ligand or custom
residue returns a new database.

{class}`tmol.pose.PackedBlockTypes` contains the block types and device-resident
setup data used by a `PoseStack`. Reuse it when constructing compatible
structures on the same device. It is not a cache of conformation energies.

## Deposited atoms and built atoms

PDB or mmCIF atom records describe deposited coordinates. TMol selects chemical
types and may build supported missing atoms from its database. Histidine state,
termini, disulfides, missing atoms, and noncanonical chemistry therefore need
to be checked in the I/O build context rather than assumed to round-trip
unchanged.

Prefer mmCIF through Biotite when metadata, explicit ligand bonds, or
noncanonical chemistry matter. PDB remains useful as a compatibility format,
but it cannot represent every input decision losslessly.

## The `no_optH` choice

Hydroxyl and other movable polar hydrogens can be optimized during preparation.
Use `no_optH=False` when you want the standard optimization step and the score
function needed to choose those conformations. Use `no_optH=True` when the
incoming proton geometry is authoritative, when you are intentionally
deferring that choice, or when a lightweight preprocessing path is more
important than optimizing those hydrogens.

The choice changes coordinates and can change scores. Record it as part of a
reproducible workflow rather than treating it as an implementation detail.

## Score units and score differences

TMol reports weighted **score units**. They are not calibrated kcal/mol and are
not guaranteed to equal scores from Rosetta, even when names or weight sets are
similar.

A block-pair score, interface sum, mutation-score difference, or
`calculate_block_pair_ddg()` result describes the exact computational
experiment used to produce it. Such values are not automatically physical
binding free energies or experimentally calibrated ΔΔG values.

## Rendered scorers and changing coordinates

A {class}`tmol.score.ScoreFunction` renders a PyTorch module for a particular
`PoseStack` layout. You can generally reuse that module while changing only the
coordinate tensor. If block types, atom counts, connectivity, or batch layout
change, render a new scorer for the new stack.

## Cartesian and kinematic movement

Cartesian minimization directly changes selected atom coordinates. Kinematic
minimization changes internal and rigid-body degrees of freedom selected by a
{class}`tmol.kinematics.MoveMap` over a
{class}`tmol.kinematics.FoldForest`. These are different coordinate models, so
their trajectories and convergence behavior should not be compared without
matching masks, weights, stopping rules, and iteration budgets.

See the {doc}`optimization workflow <user_guide/optimization>` and
{doc}`Tutorial 05 <tutorial/05_minimization_constraints_kinematics>` for
executable examples.
