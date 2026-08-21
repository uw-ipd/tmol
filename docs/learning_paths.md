# Learning paths

TMol tutorials build a shared vocabulary in a deliberate order. Start with
structure input, batching, and scoring. Then choose packing, minimization, or
both before combining them in FastRelax. Ligand and nucleic-acid tutorials are
specialized paths that reuse the same core objects.

```{raw} html
<nav class="workflow-map" aria-label="TMol tutorial sequence">
  <a class="workflow-node" href="tutorial/01_working_with_tmol.html">
    <span class="workflow-node-number">01</span>
    <span>Structures and PoseStack</span>
  </a>
  <span class="workflow-arrow" aria-hidden="true">→</span>
  <a class="workflow-node" href="tutorial/02_gpu_batching.html">
    <span class="workflow-node-number">02</span>
    <span>GPU batching</span>
  </a>
  <span class="workflow-arrow" aria-hidden="true">→</span>
  <a class="workflow-node" href="tutorial/03_scoring_and_analysis.html">
    <span class="workflow-node-number">03</span>
    <span>Scoring and analysis</span>
  </a>
  <span class="workflow-arrow" aria-hidden="true">→</span>
  <span class="workflow-branch">
    <a class="workflow-node" href="tutorial/04_packing_and_mutation_scan.html">
      <span class="workflow-node-number">04</span>
      <span>Packing and design</span>
    </a>
    <a class="workflow-node" href="tutorial/05_minimization_constraints_kinematics.html">
      <span class="workflow-node-number">05</span>
      <span>Minimization and kinematics</span>
    </a>
  </span>
  <span class="workflow-arrow" aria-hidden="true">→</span>
  <a class="workflow-node" href="tutorial/06_fast_relax.html">
    <span class="workflow-node-number">06</span>
    <span>FastRelax</span>
  </a>
  <span class="workflow-arrow" aria-hidden="true">→</span>
  <span class="workflow-branch">
    <a class="workflow-node" href="tutorial/07_ligand_and_params.html">
      <span class="workflow-node-number">07</span>
      <span>Ligands</span>
    </a>
    <a class="workflow-node" href="tutorial/08_nucleic_acids.html">
      <span class="workflow-node-number">08</span>
      <span>DNA and RNA</span>
    </a>
  </span>
</nav>
```

## New to TMol

Follow Tutorials 01–03 in order:

1. {doc}`Working with TMol <tutorial/01_working_with_tmol>` introduces
   `ParameterDatabase`, `PoseStack`, structure preparation, export, and
   visualization.
2. {doc}`GPU batching <tutorial/02_gpu_batching>` explains heterogeneous
   batches, padding, reliable timing, and application-level chunking.
3. {doc}`Scoring and analysis <tutorial/03_scoring_and_analysis>` covers score
   terms, block-pair decompositions, and coordinate gradients.

Continue with {doc}`packing <tutorial/04_packing_and_mutation_scan>`,
{doc}`minimization <tutorial/05_minimization_constraints_kinematics>`, or both.
Tutorial 06 assumes both branches.

## Coming from Rosetta or PyRosetta

Read the {doc}`Rosetta-to-TMol crosswalk <tutorial/rosetta_crosswalk>` beside
Tutorials 01 and 03. It explains where `Pose`, `Residue`, `ScoreFunction`,
`PackerTask`, `FoldTree`, constraints, and FastRelax have close TMol concepts
and where protocol or numerical parity does not exist.

Then choose:

- {doc}`Tutorial 04 <tutorial/04_packing_and_mutation_scan>` for repacking,
  local design, and mutation-score experiments;
- {doc}`Tutorial 05 <tutorial/05_minimization_constraints_kinematics>` for
  Cartesian or kinematic minimization, constraints, `MoveMap`, and
  `FoldForest`; and
- {doc}`Tutorial 06 <tutorial/06_fast_relax>` for a combined pack/minimize
  schedule.

## GPU and machine-learning applications

After Tutorial 01, focus on {doc}`Tutorial 02
<tutorial/02_gpu_batching>` and the {doc}`GPU batching workflow
<workflows/gpu_batching>`. Tutorial 03 then shows differentiable scoring and
explicit tensor decompositions. The {doc}`architecture guide <architecture>`
describes when rendered scoring modules can be reused.

## Specialized chemistry

- {doc}`Tutorial 07 <tutorial/07_ligand_and_params>` covers authoritative
  ligand chemistry, parameter injection, pocket selections, and local
  interaction analysis.
- {doc}`Tutorial 08 <tutorial/08_nucleic_acids>` covers DNA/RNA scoring,
  glycosidic-chi sampling, base substitutions, and fixed-ligand RNA examples.

For a direct lookup rather than a curriculum, use the
{doc}`task index <tutorial/recipe_index>`.
