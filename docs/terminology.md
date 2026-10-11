# Glossary

Terms used in the {doc}`quickstart` and API. The diagrams show how scores
and coordinate changes relate; the {doc}`playground` gives a working example.

(glossary-score-function)=
## Score function and score terms

A score function combines weighted contributions from atomic interactions and
molecular geometry. Lower totals are favored during minimization and packing.
Individual terms show which interactions contribute to a change.

```{raw} html

<section class="concept-explorer" id="score-circuit" aria-label="Explore the scoring calculation">
  <div class="score-flow" role="group" aria-label="Parts of the scoring calculation">
    <button type="button" data-part="coordinates" aria-pressed="false">Coordinates<br><small>atom positions</small></button>
    <span class="flow-arrow" aria-hidden="true"></span>
    <div class="score-branches">
      <button type="button" data-part="contacts" aria-pressed="true">Atomic contacts</button>
      <button type="button" data-part="hydrogen" aria-pressed="false">H bonds &amp; solvation</button>
      <button type="button" data-part="geometry" aria-pressed="false">Geometry &amp; torsions</button>
    </div>
    <span class="flow-arrow" aria-hidden="true"></span>
    <button type="button" data-part="total" aria-pressed="false">Weighted sum<br><small>one total per pose</small></button>
  </div>
  <div class="gradient-flow"><span aria-hidden="true">↳</span><button type="button" data-part="gradient" aria-pressed="false">Coordinate gradients → minimization</button></div>
  <p class="explorer-explanation" aria-live="polite">Attraction, short-range repulsion, and electrostatics depend on the relative positions of atoms.</p>
</section>
```

`sum_terms=False` returns weighted contributions in `sfxn.all_score_types()`
order. `apply_weights=False` returns unweighted values instead.
See {doc}`score terms <api/score_terms>` for individual definitions.

<span id="posestack-pose-block-and-atom"></span>

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

The {doc}`playground` isolates two side-chain angles. The
{doc}`packing notebook <tutorial/04_packing_and_mutation_scan>` runs a full
repacking calculation with residue-specific conformer samplers.

<span id="cartesian-and-kinematic-movement"></span>

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

In this schematic, orange nodes move downstream of the selected edge.
The two poses remain independent.

```{raw} html

<section class="concept-explorer" id="fold-forest" aria-label="Explore movement through a fold forest">
  <div class="explorer-buttons" role="group" aria-label="Choose a kinematic move">
    <button type="button" data-move="backbone" aria-pressed="true">Backbone torsion</button>
    <button type="button" data-move="jump" aria-pressed="false">Rigid-body jump</button>
    <button type="button" data-move="other" aria-pressed="false">Second pose</button>
  </div>
  <svg viewBox="0 0 660 250" role="img" aria-labelledby="forest-title forest-desc">
    <title id="forest-title">Two independent pose trees</title>
    <desc id="forest-desc">Pose 0 has chain A with nodes A1, A2, A3, and chain B with nodes B1 and B2 connected by a jump from A1. Pose 1 has C1 and C2. Highlighted nodes move downstream of the selected edge.</desc>
    <defs><marker id="forest-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs>
    <rect x="10" y="10" width="406" height="230" rx="12" class="forest-panel"/>
    <rect x="432" y="10" width="218" height="230" rx="12" class="forest-panel"/>
    <path d="M424 10V240" class="forest-divider"/>
    <text x="30" y="40" class="forest-heading">Pose 0</text><text x="452" y="40" class="forest-heading">Pose 1</text>
    <path d="M96 91H203" class="forest-edge active" data-edge="backbone" marker-end="url(#forest-arrow)"/>
    <path d="M255 91H349" class="forest-edge" marker-end="url(#forest-arrow)"/>
    <path d="M76 116V185H203" class="forest-edge jump" data-edge="jump" marker-end="url(#forest-arrow)"/>
    <path d="M255 185H349" class="forest-edge" marker-end="url(#forest-arrow)"/>
    <path d="M494 116V159" class="forest-edge" data-edge="other" marker-end="url(#forest-arrow)"/>
    <g class="forest-node" data-node="A1"><circle cx="76" cy="91" r="24"/><text x="76" y="97">A1</text></g>
    <g class="forest-node active" data-node="A2"><circle cx="230" cy="91" r="24"/><text x="230" y="97">A2</text></g>
    <g class="forest-node active" data-node="A3"><circle cx="376" cy="91" r="24"/><text x="376" y="97">A3</text></g>
    <g class="forest-node" data-node="B1"><circle cx="230" cy="185" r="24"/><text x="230" y="191">B1</text></g>
    <g class="forest-node" data-node="B2"><circle cx="376" cy="185" r="24"/><text x="376" y="191">B2</text></g>
    <g class="forest-node" data-node="C1"><circle cx="494" cy="91" r="24"/><text x="494" y="97">C1</text></g>
    <g class="forest-node" data-node="C2"><circle cx="494" cy="185" r="24"/><text x="494" y="191">C2</text></g>
    <text x="108" y="171" class="forest-label">jump</text>
    <text x="111" y="75" class="forest-label">polymer</text>
    <text x="530" y="138" class="forest-label">polymer</text>
  </svg>
  <p class="explorer-explanation" aria-live="polite">A backbone torsion along A1 → A2 moves downstream A2 and A3. Chain B is attached through A1, so this move leaves it fixed.</p>
</section>
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

<section class="concept-explorer" id="relax-schedule" aria-label="Explore the default FastRelax schedule">
  <div class="relax-stages" role="group" aria-label="FastRelax stage">
    <button type="button" data-stage="0" aria-pressed="true"><span class="stage-bar" style="--stage-height: 5%"></span>Stage 1</button>
    <button type="button" data-stage="1" aria-pressed="false"><span class="stage-bar" style="--stage-height: 28%"></span>Stage 2</button>
    <button type="button" data-stage="2" aria-pressed="false"><span class="stage-bar" style="--stage-height: 58%"></span>Stage 3</button>
    <button type="button" data-stage="3" aria-pressed="false"><span class="stage-bar" style="--stage-height: 100%"></span>Stage 4</button>
  </div>
  <div class="relax-operations"><span>Repack <strong id="relax-pack">4.0%</strong></span><span aria-hidden="true">→</span><span>Minimize <strong id="relax-min">5.1%</strong></span><span aria-hidden="true">→</span><span>Next stage</span></div>
  <p class="explorer-explanation" aria-live="polite">Stage 1: repack, then minimize at the indicated fractions of the full repulsion weight.</p>
</section>
```

The default protocol uses Cartesian minimization and repeats the four-stage
schedule twice. See the {doc}`FastRelax notebook <tutorial/06_fast_relax>` to run
it and compare structures.

<span id="parameterdatabase-and-packedblocktypes"></span>

## Parameters and prepared chemistry

**ParameterDatabase** holds chemical definitions and scoring parameters.
**PackedBlockTypes** holds the block types and their device-resident setup data;
it does not cache conformation energies.

<span id="deposited-atoms-and-built-atoms"></span>

<span id="the-no-opth-choice"></span>

Preparing a ligand may extend the parameter database. Use the returned
`context.parameter_database` to score that pose. Check termini, protonation,
disulfides, and built atoms when preparing inputs. `no_optH=True` skips polar
hydrogen optimization; it does not freeze those atoms in later optimization.

<span id="rendered-scorers-and-changing-coordinates"></span>

## Rendered scorer

Rendering builds a scoring module for a particular pose layout. Reuse it while
coordinates change. Render a new scorer after changing block types, atom counts,
connectivity, or batch layout, including after packing.

```{toctree}
:hidden:

playground
```
