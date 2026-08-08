# Rosetta-to-TMol crosswalk

This page maps familiar Rosetta and PyRosetta concepts onto the TMol
tutorial suite. It is a capability crosswalk, not a promise of API or protocol
parity: Rosetta is a broad modeling suite, while TMol is a PyTorch library built
around batched, differentiable molecular operations.

TMol statements below are grounded in the current repository sources, including
the nucleic-acid functionality merged through
[TMol PR #404](https://github.com/uw-ipd/tmol/pull/404).

## Molecular objects and chemical types

Rosetta's central object is a single `Pose`, an ordered collection of
`Residue` objects plus conformation, scoring, and metadata. Start with
[Working with Rosetta](https://docs.rosettacommons.org/demos/latest/tutorials/Working_With_Rosetta/working_with_rosetta)
and the [core-concepts tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/Core_Concepts/Core_Concepts).

TMol's corresponding object is a
[`PoseStack`](https://github.com/uw-ipd/tmol/blob/master/tmol/pose/pose_stack.py):
one or more molecular systems represented by padded and indexed tensors on one
PyTorch device. A Rosetta residue corresponds most closely to a TMol **block**.
Each block selects a refined block type and contributes its atoms to the
stack's contiguous coordinate tensor.

- A
  [`PackedBlockTypes`](https://github.com/uw-ipd/tmol/blob/master/tmol/pose/packed_block_types.py)
  contains the ordered block types available to a stack and caches
  device-resident, type-dependent tensors prepared by score terms. Those are
  setup caches, not cached conformation energies.
- The immutable
  [`ParameterDatabase`](https://github.com/uw-ipd/tmol/blob/master/tmol/database/__init__.py)
  owns the chemical definitions and scoring parameters from which block types,
  score functions, and I/O contexts are built. Extending it returns a new
  database.
- Chemistry and connectivity are fixed for a particular `PoseStack`; coordinate
  tensors may change. Packing or design therefore returns a new stack when a
  block's type or atom count changes.

The batch dimension is fundamental. A one-structure `PoseStack` is valid, but
it is not a separate single-pose class analogous to Rosetta's `Pose`.

## I/O, selections, and options

Rosetta applications combine command-line flags, a Rosetta database, and
protocol-specific objects. They read PDB/mmCIF and sequence inputs and often
write PDB or Rosetta silent archives. See the official
[input/output tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/input_and_output/input_and_output),
[common options](https://docs.rosettacommons.org/demos/latest/tutorials/commonly_used_options/commonly_used_options),
and [silent-file format](https://docs.rosettacommons.org/docs/latest/rosetta_basics/file_types/silent-file).

TMol has no Rosetta-style process-global flags system or silent-file archive.
Callers construct explicit Python objects and pass function arguments:

- Input normally enters through a Biotite `AtomArray` or `AtomArrayStack`.
  [`pose_stack_from_biotite()`](https://github.com/uw-ipd/tmol/blob/master/tmol/io/pose_stack_from_biotite.py)
  builds the `ParameterDatabase`/`PackedBlockTypes` context and the
  `PoseStack`; the context can be reused for structures with the same chemistry.
- PDB and mmCIF parsing and writing can be delegated to Biotite. TMol also
  exports a stack back to a Biotite structure and has a direct PDB writer.
- Selections are explicit NumPy or Torch boolean masks over atoms, blocks, or
  poses. Biotite masks can select and slice an `AtomArray` before conversion;
  block masks drive packing and analysis after conversion. TMol does not
  provide Rosetta's `ResidueSelector` framework.
- Device, database, score function, packing task, move map, fold forest, and
  constraints remain visible objects. This makes provenance explicit but means
  Rosetta command lines do not translate one-for-one.

## Scoring and analysis

Rosetta commonly uses the `ref2015`/REF15 weights or the experimental
`beta_nov16` family and can switch between centroid and full-atom residue-type
sets. The official
[scoring tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/scoring/scoring),
[analysis tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/analysis/Analysis),
and [full-atom versus centroid tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/full_atom_vs_centroid/fullatom_centroid)
explain those choices. The implementation entry point is
[`ScoreFunction.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/scoring/ScoreFunction.cc).
Primary descriptions are Park et al.'s
[beta2016 paper](https://doi.org/10.1021/acs.jctc.6b00819) and Alford et al.'s
[REF15 paper](https://doi.org/10.1021/acs.jctc.7b00125).

TMol currently ships
[`beta2016_score_function()`](https://github.com/uw-ipd/tmol/blob/master/tmol/score/__init__.py),
its implementation of Rosetta's beta-November-2016 all-atom score function.
It does **not** implement `ref2015`, centroid residue types, or centroid/full-atom
mode switching.

Rendered
[`ScoreFunction` modules](https://github.com/uw-ipd/tmol/blob/master/tmol/score/score_function.py)
return tensors rather than updating a pose-owned energy graph:

- whole-pose scoring returns one value per pose, or weighted/unweighted values
  separated by score term;
- block-pair scoring returns a dense block-by-block matrix for each pose, again
  either summed or separated by term;
- coordinates participate in PyTorch autograd, and minimization calls
  `backward()` through these scoring modules.

TMol has no Rosetta-like per-residue `Energies` cache attached to each pose.
Block-pair tensors can be reduced to residue/block views when needed, but they
are results of an explicit scoring call. They are also energy decompositions,
not automatically physical binding free energies.

## Packing, design, and mutation scans

Both projects use a packer-task model. Rosetta's
[packer tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/Optimizing_Sidechains_The_Packer/Optimizing_Sidechains_The_Packer)
and
[`PackerTask.hh`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/pack/task/PackerTask.hh)
describe the Rosetta side.

TMol provides
[`PackerTask` and `PackerPalette`](https://github.com/uw-ipd/tmol/blob/master/tmol/pack/packer_task.py),
pluggable conformer samplers, per-block disabling, fixed-sequence repacking,
and sequence design over compatible block types. These operations can pack a
`PoseStack` batch on one device.

TMol does not provide a native point-mutation-scan protocol equivalent to the
[PyRosetta point-mutation scan](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb).
A tutorial can explicitly construct variants or packing tasks, batch them,
repack/minimize, and compare scores. A change in selected block-pair
interactions is an energy proxy for that defined calculation; it must not be
labeled a binding free energy or a built-in TMol mutation-scanning method.

## Minimization, constraints, relax, and kinematics

Rosetta's relevant guides are
[minimization](https://docs.rosettacommons.org/demos/latest/tutorials/minimization/minimization),
[constraints](https://docs.rosettacommons.org/demos/latest/tutorials/Constraints_Tutorial/Constraints),
[Relax](https://docs.rosettacommons.org/demos/latest/tutorials/Relax_Tutorial/Relax),
and [FoldTree](https://docs.rosettacommons.org/demos/latest/tutorials/fold_tree/fold_tree).
Implementation anchors include
[`ConstraintSet.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/scoring/constraints/ConstraintSet.cc),
[`FastRelax.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/protocols/relax/FastRelax.cc),
and
[`FoldTree.hh`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/kinematics/FoldTree.hh).

TMol exposes two differentiable minimization paths:

- Cartesian minimization optimizes selected coordinates directly through a
  boolean coordinate mask.
- Kinematic minimization optimizes internal and rigid-body degrees of freedom
  selected by a `MoveMap` over a `FoldForest`.

The current
[`ConstraintEnergyTerm`](https://github.com/uw-ipd/tmol/blob/master/tmol/score/constraint/constraint_energy_term.py)
supports harmonic or bounded atom-pair distances, harmonic coordinate
restraints, and circular-harmonic four-atom torsions. It does not reproduce the
full Rosetta constraint-function catalog; in particular, there is no dedicated
three-atom angle-constraint function in the current implementation.

TMol's
[`fast_relax()`](https://github.com/uw-ipd/tmol/blob/master/tmol/relax/fast_relax.py)
runs repeated repack/minimize steps with `fa_rep` and optional constraint-weight
ramps. Its current default minimizer is Cartesian; callers can supply a
kinematic minimizer. This is a partial Rosetta-style relax analogue built from
TMol primitives, not an exact port of every Rosetta FastRelax option or
acceptance detail.

A Rosetta `FoldTree` roots a pose at a selected residue and represents polymer
edges, jumps, and cutpoints. A TMol
[`FoldForest`](https://github.com/uw-ipd/tmol/blob/master/tmol/kinematics/fold_forest.py)
holds one kinematic tree per pose and **always has a virtual root at the
origin**; root jumps are distinct from ordinary jumps. Its convenience builder
uses backbone polymer connectivity, roots separate chains, turns intra-chain
gaps into jumps, ignores non-polymer connections such as disulfides, and breaks
a cyclic polymer by dropping one bond. Use explicit edges when that automatic
forest is not the intended kinematics.

## Ligands and residue-parameter files

Rosetta defines a noncanonical residue in a line-oriented
[residue `.params` file](https://docs.rosettacommons.org/docs/latest/rosetta_basics/file_types/Residue-Params-file).
The [ligand-preparation tutorial](https://rosettacommons.org/demos/latest/tutorials/prepare_ligand/prepare_ligand_tutorial)
and
[`molfile_to_params.py`](https://github.com/RosettaCommons/rosetta/blob/main/source/scripts/python/public/molfile_to_params.py)
show the conventional workflow.

TMol separates canonical database concerns:

- `chemical/chemical.yaml` defines atom and residue chemistry;
- `scoring/elec.yaml` supplies partial charges;
- `scoring/cartbonded.yaml` supplies residue-specific bonded parameters.

For a portable prepared ligand, TMol's versioned `.tmol` YAML bundles the same
three sections as `chemical`, `elec`, and `cartbonded`. The current schema
version is checked on load. A
[`LigandPreparation`](https://github.com/uw-ipd/tmol/blob/master/tmol/ligand/registry.py)
contains the residue definition, partial charges, and cartbonded parameters,
and the supported
[`write_params_file()`](https://github.com/uw-ipd/tmol/blob/master/tmol/ligand/params_io.py)
can write that one preparation as either Rosetta `.params` or TMol `.tmol`.
Writing both outputs from the same preparation is the supported cross-format
workflow.

The reverse direction is intentionally partial. TMol's Rosetta reader handles
the records it recognizes (`NAME`, `ATOM`, `BOND`/`BOND_TYPE`, `CHI`,
`PROTON_CHI`, `NBR_ATOM`, and `ICOOR_INTERNAL`) and ignores other records. It
does not reconstruct a complete TMol scoring database from an arbitrary
`.params`: notably, the reader does not recover the `.tmol` `elec` and
`cartbonded` sections. Rosetta `.params` to TMol `.tmol` is therefore not a
general or lossless conversion. Use a prepared `.tmol` file, or rerun ligand
preparation and inspect the resulting chemistry and parameters.

These facilities support ligand scoring, repacking, and refinement. They are
not a native ligand-docking protocol.

## GPU batching and external orchestration

PyRosetta Chapter 16 demonstrates independent-job parallelism:

- [Chapter 16 landing page](https://rosettacommons.github.io/PyRosetta.notebooks/)
- [distributed ddG/PSSM with multiprocessing and PyData tools](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.01-PyData-ddG-pssm.ipynb)
- [distributed miniprotein design](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.02-PyData-miniprotein-design.ipynb)
- [GNU Parallel through Slurm](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.03-GNU-Parallel-Via-Slurm.ipynb)
- [Dask through Slurm](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.04-dask.delayed-Via-Slurm.ipynb)
- [PyRosettaCluster](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.06-PyRosettaCluster-Simple-protocol.ipynb)

Those examples distribute separate jobs or trajectories across processes,
nodes, or workers. TMol's native parallelism is at a different level: a
`PoseStack` batches structures inside scoring, packing, or refinement
operations as tensors on **one device**. TMol does not currently include a
distributed runner or multi-GPU scheduler.

The two levels are complementary. An external Python job system, Dask, GNU
Parallel, Slurm, or another scheduler can assign one TMol process and one
`PoseStack` shard to each GPU. That orchestration remains application code; it
should not be presented as a built-in TMol feature.

## DNA and RNA

[PR #404](https://github.com/uw-ipd/tmol/pull/404) added one unified
`na_torsion` functional form for both polymers,
based on Rosetta's DNA-torsion form but using separate fitted parameter sets for
DNA and RNA. The associated terms are `na_torsion` and `na_torsion_well`.
Nucleic-acid chi packing is generated from those wells:
anti-only sampling for DNA and anti/syn sampling for RNA.

That scope is much narrower than Rosetta's RNA and DNA toolset. Even after the
addition, TMol does not have:

- low-resolution or centroid RNA scoring;
- RNA motif, base-pair, or suite classifiers;
- RNA fragment insertion, FARNA, FARFAR, or FARFAR2 structure generation;
- a mature coupled nucleobase-sequence design protocol;
- Rosetta's A-form helix assembly, RNA threading, or mature RNA protocol
  ecosystem.

Use the
[PyRosetta RNA tutorial](https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/14.00-RNA-Basics.ipynb)
for the Rosetta workflow and
[FARFAR2 documentation](https://rosettacommons.org/docs/latest/FARFAR2) for the
full Rosetta protocol. Relevant Rosetta implementation sources are
[`DNA_DihedralPotential.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/scoring/dna/DNA_DihedralPotential.cc),
[`RNA_TorsionPotential.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/scoring/rna/RNA_TorsionPotential.cc),
and
[`RNA_SuiteEnergy.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/energy_methods/RNA_SuiteEnergy.cc).

Primary background:

- Pérez et al.,
  [parmbsc0 DNA torsions](https://doi.org/10.1529/biophysj.106.097782)
- Havranek, Duarte, and Baker,
  [protein-DNA scoring and design](https://doi.org/10.1016/j.jmb.2004.09.029)
- Das and Baker,
  [FARNA](https://doi.org/10.1073/pnas.0703836104)
- Das, Karanicolas, and Baker,
  [all-atom RNA/FARFAR](https://doi.org/10.1038/nmeth.1433)
- Richardson et al.,
  [RNA backbone suites](https://doi.org/10.1261/rna.657708)
- Watkins, Rangan, and Das,
  [FARFAR2](https://doi.org/10.1016/j.str.2020.05.011)

PR #404 attributes its DNA/RNA nonbonded parameters to an exact OptE fit
provided by the author. That provenance is **author-provided and was not
confirmed by a publication found for that exact fit**. Cite the
[general OptE documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/utilities/opt-e-parallel-doc)
and the PR, not a nonexistent dedicated publication.

## Parametric helical bundles

Rosetta natively generates parametric backbones with
[`MakeBundle`](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/MakeBundleMover)
and explores parameter grids with
[`BundleGridSampler`](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/BundleGridSamplerMover).
The
[PyRosetta parametric-backbone tutorial](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.06-Introduction-to-Parametric-backbone-design.ipynb)
is a Rosetta-native workflow, not a TMol tutorial that can be directly ported.

TMol has no corresponding parametric backbone generator or bundle-grid
sampler. It can score, pack, constrain, minimize, or relax structures generated
by Rosetta or another external tool after they have been converted into a
supported `PoseStack`; it cannot create the bundle parameterization itself.
