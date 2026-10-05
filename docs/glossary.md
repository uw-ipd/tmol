# Glossary

The objects, moves, and scores used in TMol. Select parts of the diagrams to
see how they connect, or try the {doc}`protein score playground <playground>`.

(glossary-score-function)=
## Score function and score terms

A score function combines weighted contributions from atomic interactions and
molecular geometry. Lower totals are favored during minimization and packing.
Inspect terms separately to understand *why* a conformation scores differently.

```{raw} html
:file: _includes/score-circuit.html
```

`sum_terms=False` returns weighted contributions in `sfxn.all_score_types()`
order. `apply_weights=False` returns unweighted values instead.
See {doc}`score terms <api/score_terms>` for individual definitions.

(glossary-posestack)=
## Pose, block, and PoseStack

A **pose** is one molecular system. A **block** is a typed chemical unit, such
as an amino acid, nucleotide, or ligand fragment. A **PoseStack** batches poses
as padded tensors on a single PyTorch device.

```text
PoseStack
  pose 0  [ALA]—[PHE]—[GLY]—[SER]
  pose 1  [LEU]—[TYR]  ···   ···   ← padding, not atoms
```

Use atom and block masks to distinguish real entries from padding. Group
similarly sized poses to limit padding; see {doc}`tensor layouts <datatypes>`.

(glossary-repacking)=
## Torsions, rotamers, and repacking

A **torsion** is a rotation around a bond, described by four atoms. Protein
backbone angles include φ and ψ; side-chain angles are called χ1, χ2, and so on.
A **rotamer** is a candidate side-chain conformation.

**Repacking** chooses conformations for a fixed sequence. It considers how
neighboring choices interact. **Design** also permits residue identities to
change, according to the packer task.

Try rotating two χ angles in the {doc}`playground`. Its finite angle grid makes
the tradeoffs visible; TMol's packer uses residue-specific conformer samplers.
The {doc}`packing notebook <tutorial/04_packing_and_mutation_scan>` runs a full
repacking calculation.

## Gradients and minimization

A coordinate gradient tells you how the score changes as an atom moves.
Minimization uses these derivatives to improve a conformation continuously.
Repacking searches discrete candidates; minimization can refine their geometry.

**Cartesian minimization** changes atom coordinates. **Kinematic minimization**
changes selected internal coordinates and rigid-body jumps. The allowed moves
come from a coordinate mask or a `MoveMap`, respectively.

(glossary-foldforest)=
## Fold tree, fold forest, and jumps

A **fold tree** is a rooted, acyclic description of how coordinates are built
for one pose. A **fold forest** stores these trees across a batch. A **jump**
defines a rigid-body connection in that coordinate model; it is not a chemical
bond. Polymer and chemical edges describe other kinematic connections.

Select a move below. Orange nodes are downstream of that move; another pose
in the batch stays independent. This is a schematic, not a molecular geometry.

```{raw} html
:file: _includes/fold-forest.html
```

Chemical connectivity and the kinematic tree are distinct. A cyclic molecule
needs a cut in its kinematic representation even though its chemical bond
remains. Choose the tree and allowed moves together; see the
{doc}`fold-forest notebook <tutorial/12_explicit_foldforests_and_torsions>`.

## MoveMap and constraints

A **MoveMap** selects torsions and jumps for kinematic movement. Creating one
does not enable movement: set its flags or masks explicitly. A
**CartesianMoveMap** selects atom coordinates for Cartesian FastRelax.

A **constraint** adds a score penalty, such as a restraint to a target position.
It affects optimization only when attached to the pose and given a nonzero
score-function weight. A frozen degree of freedom cannot move; a restrained
one may move at a scoring cost.

(glossary-fastrelax)=
## FastRelax

FastRelax alternates repacking with minimization. Early stages soften steric
repulsion so side chains can rearrange; later stages restore its full weight.
The protocol retains the best conformation for each pose under the acceptance
score. Lower repulsion during a stage does not itself mean the structure improved.

Select a stage to see the default repulsion-weight fractions:

```{raw} html
:file: _includes/relax-schedule.html
```

The default protocol uses Cartesian minimization and repeats the four-stage
schedule twice. See the {doc}`FastRelax notebook <tutorial/06_fast_relax>` to run
it and compare structures.

## Parameters and prepared chemistry

**ParameterDatabase** holds chemical definitions and scoring parameters.
**PackedBlockTypes** holds the block types and their device-resident setup data;
it does not cache conformation energies.

Preparing a ligand may extend the parameter database. Use the returned
`context.parameter_database` to score that pose. Check termini, protonation,
disulfides, and built atoms when preparing inputs. `no_optH=True` skips polar
hydrogen optimization; it does not freeze those atoms in later optimization.

## Rendered scorer

Rendering builds a scoring module for a particular pose layout. Reuse it while
coordinates change. Render a new scorer after changing block types, atom counts,
connectivity, or batch layout, including after packing.

```{toctree}
:hidden:

playground
```
