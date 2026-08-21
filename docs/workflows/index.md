# Workflows

Use these concise recipes for recurring TMol tasks. The
{doc}`Tutorials </examples_index>` are executable, notebook-length
demonstrations; these pages focus on reusable steps and link to the
corresponding Tutorials for deeper analysis.

```{raw} html
<nav class="docs-card-grid" aria-label="TMol workflow groups">
  <a class="docs-card" href="structure_io.html">
    <span class="docs-card-kicker">Prepare</span>
    <span class="docs-card-title">Structure I/O and visualization</span>
    <span class="docs-card-description">Choose CIF, Biotite, PDB, or model-tensor input; preserve chemistry; inspect and export structures.</span>
  </a>
  <a class="docs-card" href="../user_guide/integrations.html">
    <span class="docs-card-kicker">Connect</span>
    <span class="docs-card-title">Model and structure integrations</span>
    <span class="docs-card-description">Convert RoseTTAFold2, OpenFold, Biotite, and AtomWorks representations into compatible PoseStacks.</span>
  </a>
  <a class="docs-card" href="gpu_batching.html">
    <span class="docs-card-kicker">Scale</span>
    <span class="docs-card-title">GPU batching</span>
    <span class="docs-card-description">Batch compatible structures, measure CUDA work correctly, and chunk larger application workloads.</span>
  </a>
  <a class="docs-card" href="../user_guide/scoring.html">
    <span class="docs-card-kicker">Evaluate</span>
    <span class="docs-card-title">Scoring and analysis</span>
    <span class="docs-card-description">Render score modules, decompose block interactions, and differentiate with respect to coordinates.</span>
  </a>
  <a class="docs-card" href="packing.html">
    <span class="docs-card-kicker">Design</span>
    <span class="docs-card-title">Packing and local design</span>
    <span class="docs-card-description">Configure conformer samplers, repack selected blocks, and run explicit mutation-score experiments.</span>
  </a>
  <a class="docs-card" href="../user_guide/optimization.html">
    <span class="docs-card-kicker">Refine</span>
    <span class="docs-card-title">Minimization and FastRelax</span>
    <span class="docs-card-description">Choose Cartesian or kinematic movement, add constraints, and combine packing with minimization.</span>
  </a>
  <a class="docs-card" href="../user_guide/ligands.html">
    <span class="docs-card-kicker">Extend</span>
    <span class="docs-card-title">Ligand preparation</span>
    <span class="docs-card-description">Prepare authoritative ligand chemistry, inject parameters, and analyze a local pocket.</span>
  </a>
  <a class="docs-card" href="nucleic_acids.html">
    <span class="docs-card-kicker">Specialize</span>
    <span class="docs-card-title">DNA and RNA</span>
    <span class="docs-card-description">Score nucleic acids, sample glycosidic chi, and construct local base-substitution tasks.</span>
  </a>
</nav>
```

```{toctree}
:maxdepth: 2
:caption: Prepare structures

structure_io
../user_guide/integrations
gpu_batching
```

```{toctree}
:maxdepth: 2
:caption: Score, design, and refine

../user_guide/scoring
packing
../user_guide/optimization
```

```{toctree}
:maxdepth: 2
:caption: Specialized chemistry

../user_guide/ligands
nucleic_acids
```

```{toctree}
:maxdepth: 2
:caption: Develop and benchmark

../user_guide/benchmarking
../user_guide/development
```

## How the workflows interact

Structure preparation determines chemical types, atom layout, and topology.
Those choices define the `PoseStack` consumed by scoring, packing, and
minimization. Packing can change block identities and atom counts, so render a
new scorer for its returned stack. Coordinate-only minimization can reuse a
scorer built for the same layout. FastRelax alternates packing and minimization
and handles that transition for its supported schedule.

Ligand and nucleic-acid workflows reuse the same I/O, scoring, packing, and
movement primitives with additional chemistry and samplers. GPU batching is an
execution strategy that can wrap many of these workflows; it does not make
unrelated scores directly comparable.

Use the {doc}`task index </tutorial/recipe_index>` for a direct operation lookup
or the {doc}`learning paths </learning_paths>` for the numbered curriculum.
