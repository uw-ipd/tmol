# Rosetta-to-TMol crosswalk

This page maps concepts used in the eight TMol tutorials to their nearest
Rosetta and PyRosetta counterparts. It is not an API, numerical, or protocol
parity claim: Rosetta is a broad modeling suite, whereas TMol provides batched,
differentiable molecular primitives in PyTorch.

## Molecular objects and chemical types

Rosetta's central object is a `Pose`: an ordered collection of `Residue`
objects plus conformation and metadata. See
[Working with Rosetta](https://docs.rosettacommons.org/demos/latest/tutorials/Working_With_Rosetta/working_with_rosetta)
and the [core-concepts tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/Core_Concepts/Core_Concepts).

The nearest TMol object is a
[`PoseStack`](https://github.com/uw-ipd/tmol/blob/master/tmol/pose/pose_stack.py):
one or more molecular systems represented by padded, indexed tensors on one
PyTorch device. A Rosetta residue corresponds most closely to a TMol **block**,
but a one-pose `PoseStack` is not a separate `Pose`-compatible class.

- A
  [`PackedBlockTypes`](https://github.com/uw-ipd/tmol/blob/master/tmol/pose/packed_block_types.py)
  contains the block types available to a stack and device-resident setup data;
  it does not cache conformation energies.
- The immutable
  [`ParameterDatabase`](https://github.com/uw-ipd/tmol/blob/master/tmol/database/__init__.py)
  owns chemical definitions and scoring parameters. Extending it returns a new
  database.
- Chemistry and connectivity are fixed for a particular `PoseStack`; coordinate
  tensors may change. Packing or design returns a new stack when block types or
  atom counts change.

## I/O, selections, and options

Rosetta applications combine command-line flags, a database, and
protocol-specific objects. See the official
[input/output tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/input_and_output/input_and_output),
[common options](https://docs.rosettacommons.org/demos/latest/tutorials/commonly_used_options/commonly_used_options),
and [silent-file format](https://docs.rosettacommons.org/docs/latest/rosetta_basics/file_types/silent-file).

TMol has no Rosetta-style process-global flags system or silent-file archive.
Callers construct explicit Python objects:

- Input normally enters through a Biotite `AtomArray` or `AtomArrayStack`.
  [`pose_stack_from_biotite()`](https://github.com/uw-ipd/tmol/blob/master/tmol/io/pose_stack_from_biotite.py)
  builds a pose and an I/O context; reusing that context preserves the same
  chemical typing and canonical ordering across compatible structures.
- [`biotite_from_pose_stack()`](https://github.com/uw-ipd/tmol/blob/master/tmol/io/pose_stack_from_biotite.py)
  returns a Biotite structure for scientific round trips. Biotite handles
  PDB/mmCIF parsing and writing, and TMol also has a direct PDB writer.
- Selections are explicit NumPy or Torch boolean masks over atoms, blocks, or
  poses. TMol does not provide Rosetta's `ResidueSelector` framework.
- Deposited atom records and the atoms TMol builds from its chemical database
  are distinct representations. Histidine state, termini, disulfides, missing
  atoms, and noncanonical chemistry must be checked in the build context rather
  than assumed to round-trip identically.

## Scoring and analysis

Rosetta commonly uses REF15 or beta-November-2016-family weights and can switch
between centroid and full-atom residue-type sets. The official
[scoring tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/scoring/scoring),
[analysis tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/analysis/Analysis),
and [full-atom versus centroid tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/full_atom_vs_centroid/fullatom_centroid)
explain those choices; see also the
[beta2016](https://doi.org/10.1021/acs.jctc.6b00819) and
[REF15](https://doi.org/10.1021/acs.jctc.7b00125) papers.

TMol ships
[`beta2016_score_function()`](https://github.com/uw-ipd/tmol/blob/master/tmol/score/__init__.py),
a beta-November-2016-style all-atom preset. It does **not** implement
`ref2015`, centroid residue types, centroid/full-atom switching, or exact
Rosetta numerical parity. Report its weighted outputs as score units, not
calibrated kcal/mol.

Rendered
[`ScoreFunction` modules](https://github.com/uw-ipd/tmol/blob/master/tmol/score/score_function.py)
are prepared for one `PoseStack` layout and return tensors:

- whole-pose scoring returns totals or weighted/unweighted values by score term;
- block-pair scoring returns dense block-by-block tensors; term implementations
  may store a pair contribution in one directed/upper-triangle entry, so an
  undirected block interaction is `matrix[i, j] + matrix[j, i]`;
- coordinate changes can reuse a rendered scorer, but a changed pose layout
  requires rerendering; and
- coordinates participate in PyTorch autograd.

TMol has no Rosetta-like per-residue `Energies` cache attached to each pose.
Block-pair tensors are explicit score decompositions, not physical binding free
energies.

## Packing, design, and mutation scans

Both projects separate the packing task from the search algorithm. Rosetta's
[packer tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/Optimizing_Sidechains_The_Packer/Optimizing_Sidechains_The_Packer)
and
[`PackerTask.hh`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/pack/task/PackerTask.hh)
describe its mature TaskOperation, resfile, and mover layers.

TMol provides
[`PackerTask` and `PackerPalette`](https://github.com/uw-ipd/tmol/blob/master/tmol/pack/packer_task.py),
pluggable conformer samplers, per-block disabling, fixed-sequence repacking,
and sequence design over compatible block types. The compiled annealer has CPU
and CUDA paths. CUDA randomness uses PyTorch's CUDA generator; the current CPU
path uses C `rand()`, so `torch.manual_seed()` alone does not make CPU packing
deterministic.

TMol does not provide a native point-mutation-scan protocol equivalent to the
[PyRosetta point-mutation scan](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb).
A caller can explicitly construct variants, define a local packing shell,
repack/minimize each variant, and compare scores. Such deltas describe that
specific computational experiment; they are not a built-in scan or physical
delta-delta G.

## Minimization, constraints, and kinematics

Rosetta's relevant guides are
[minimization](https://docs.rosettacommons.org/demos/latest/tutorials/minimization/minimization),
[constraints](https://docs.rosettacommons.org/demos/latest/tutorials/Constraints_Tutorial/Constraints),
and [FoldTree](https://docs.rosettacommons.org/demos/latest/tutorials/fold_tree/fold_tree).

TMol exposes two differentiable paths:

- Cartesian minimization optimizes selected coordinates directly through a
  `CartesianMoveMap`/boolean coordinate mask. Unrestrained Cartesian
  minimization includes global rigid-body null modes.
- Kinematic minimization optimizes internal and rigid-body degrees of freedom
  selected by a `MoveMap` over a `FoldForest`; named main-chain and side-chain
  torsions and jumps are controlled separately.

The current
[`ConstraintEnergyTerm`](https://github.com/uw-ipd/tmol/blob/master/tmol/score/constraint/constraint_energy_term.py)
supports harmonic or bounded atom-pair distances, harmonic coordinate
restraints, and circular-harmonic four-atom torsions. It does not reproduce the
full Rosetta constraint-function catalog, text constraint parser, or ambiguous
constraint layer; in particular, it has no dedicated three-atom angle
constraint.

A Rosetta `FoldTree` roots a pose at a selected residue and represents polymer
edges, jumps, and cutpoints. A TMol
[`FoldForest`](https://github.com/uw-ipd/tmol/blob/master/tmol/kinematics/fold_forest.py)
holds one kinematic tree per pose and **always has a virtual root at the
origin**; root jumps are distinct from ordinary jumps. Its convenience builder
uses backbone polymer connectivity, roots separate chains, turns intra-chain
gaps into jumps, ignores non-polymer connections such as disulfides, and breaks
a cyclic polymer by dropping one bond. Use explicit edges when that automatic
forest is not the intended kinematics.

## FastRelax

Rosetta's [Relax tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/Relax_Tutorial/Relax),
[FastRelax mover documentation](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover),
and
[`FastRelax.cc`](https://github.com/RosettaCommons/rosetta/blob/main/source/src/protocols/relax/FastRelax.cc)
describe a mature, commonly torsional protocol with extensive script, MoveMap,
symmetry, membrane, and acceptance options.

TMol's
[`fast_relax()`](https://github.com/uw-ipd/tmol/blob/master/tmol/relax/fast_relax.py)
is a smaller Rosetta-inspired refinement routine. It repeats packing and
minimization while ramping `fa_rep` and optionally constraint weights, then
selects the best-scoring result across repeats. Its default minimizer is
Cartesian; callers may supply a kinematic minimizer. The schedule, score
function, acceptance details, and available integrations are not equivalent to
Rosetta FastRelax, and matching trajectories or outputs should not be expected.

## Ligands and residue-parameter files

Rosetta defines a noncanonical residue in a line-oriented
[residue `.params` file](https://docs.rosettacommons.org/docs/latest/rosetta_basics/file_types/Residue-Params-file).
The [ligand-preparation tutorial](https://docs.rosettacommons.org/demos/latest/tutorials/prepare_ligand/prepare_ligand_tutorial)
and
[`molfile_to_params.py`](https://github.com/RosettaCommons/rosetta/blob/main/source/scripts/python/public/molfile_to_params.py)
show the conventional workflow.

TMol separates canonical database concerns:

- `chemical/chemical.yaml` defines atom and residue chemistry;
- `scoring/elec.yaml` supplies partial charges;
- `scoring/cartbonded.yaml` supplies residue-specific bonded parameters.

TMol's versioned `.tmol` YAML bundles ligand additions to those three domains. A
[`LigandPreparation`](https://github.com/uw-ipd/tmol/blob/master/tmol/ligand/registry.py)
contains the residue definition, partial charges, and cartbonded parameters,
and the supported
[`write_params_file()`](https://github.com/uw-ipd/tmol/blob/master/tmol/ligand/params_io.py)
can emit either format from one preparation. The Rosetta writer emits
`BOND_TYPE` records and writes partial charges in `ATOM` records, but the two
outputs are not equivalent parameterizations.

The Rosetta reader is intentionally partial: it recognizes `NAME`, `ATOM`,
`BOND`/`BOND_TYPE`, `CHI`, `PROTON_CHI`, `NBR_ATOM`, and `ICOOR_INTERNAL`, but
ignores other records and drops the charge values on `ATOM` lines. It cannot
reconstruct the `.tmol` electrostatic and cartbonded sections, so arbitrary
`.params` to `.tmol` conversion is not general or lossless.

These facilities support preparation, registration, scoring, local pocket
repacking, and Cartesian refinement. They do not provide native ligand docking,
global pose search, GALigandDock, or a binding-affinity calculation.

## GPU batching and external orchestration

PyRosetta [Chapter 16](https://rosettacommons.github.io/PyRosetta.notebooks/)
and its [distributed ddG/PSSM example](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/16.01-PyData-ddG-pssm.ipynb)
distribute independent jobs or trajectories across processes and workers.
TMol parallelism is at a different level: one `PoseStack` batches structures
inside scoring, packing, or refinement on **one device**. Padding means batch
memory and latency depend on the largest member and operation; absolute scores
of unrelated proteins are not made comparable by batching.

TMol has no built-in distributed runner or multi-GPU scheduler. Slurm, Dask, or
another external system may assign one process and `PoseStack` shard per GPU,
but that orchestration remains application code.

## DNA and RNA

TMol has canonical DNA/RNA chemistry and a unified `na_torsion` functional form
with separate fitted parameter sets for the two polymers. The beta2016-style
preset includes `na_torsion`, `na_torsion_well`, ordinary all-atom nonbonded
terms (including `lk_ball` and `lk_ball_iso`), and cartbonded terms. Some
protein-only terms are expected to be zero on polymer-only NA structures.

[`NaChiRotamerSampler`](https://github.com/uw-ipd/tmol/blob/master/tmol/pack/rotamer/na_chi_sampler.py)
samples anti glycosidic chi for DNA and anti plus eligible syn wells for RNA,
along with configured hydroxyl proton chis. It reads sugar pucker from the input
and does not sample pucker. Generic task masks can construct a local base
substitution experiment, but TMol has no native RosettaDNA specificity or
nucleobase-design protocol.

Rosetta's scope is much broader. See the official
[PyRosetta RNA Basics notebook](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/14.00-RNA-Basics.ipynb),
[FARFAR2 documentation](https://docs.rosettacommons.org/docs/latest/FARFAR2),
and [RosettaDNA documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/design/rosetta-dna).
TMol does not provide low-resolution RNA scoring, motif/base-pair/suite
classification, fragment insertion, FARNA/FARFAR/FARFAR2 generation, stepwise
modeling, RNA threading, docking, or those mature design protocol layers.

The exact DNA/RNA nonbonded OptE fit in TMol was provided by its author, but no
dedicated publication for that exact fit was identified in the audit. Cite the
[general OptE documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/utilities/opt-e-parallel-doc)
and [TMol PR #404](https://github.com/uw-ipd/tmol/pull/404), not a nonexistent
fit-specific paper.
