TMol documentation
==================

TMol is a PyTorch molecular-modeling library for batched, differentiable
all-atom calculations on CPU and GPU. It provides Rosetta-inspired scoring,
side-chain packing and design, Cartesian and kinematic minimization, FastRelax,
ligand preparation, and nucleic-acid modeling primitives.

TMol is not a PyRosetta compatibility layer. Weighted outputs are TMol score
units—not kcal/mol or calibrated Rosetta score units—and matching Rosetta
protocol trajectories or numerical results should not be expected.

Choose a path
-------------

Start with :doc:`quickstart` for a first score calculation. Continue with the
interactive :doc:`examples <examples_index>`, use a concise
:doc:`workflow <workflows/index>`, or look up a specific operation in the
:doc:`task index <tutorial/recipe_index>`.

.. raw:: html

   <nav class="docs-card-grid docs-card-grid--three" aria-label="Choose a TMol documentation path">
     <a class="docs-card" href="examples_index.html">
       <span class="docs-card-kicker">Explore</span>
       <span class="docs-card-title">Run an interactive example</span>
       <span class="docs-card-description">Work through the eight executable notebooks with molecular viewers, tables, plots, and exercises.</span>
     </a>
     <a class="docs-card" href="workflows/index.html">
       <span class="docs-card-kicker">Do</span>
       <span class="docs-card-title">Run a workflow</span>
       <span class="docs-card-description">Use concise recipes for recurring modeling tasks and jump directly to their APIs and deep tutorials.</span>
     </a>
     <a class="docs-card" href="tutorial/recipe_index.html">
       <span class="docs-card-kicker">Find</span>
       <span class="docs-card-title">Look up a task</span>
       <span class="docs-card-description">Find the maintained tutorial, workflow recipe, and public API for a specific operation.</span>
     </a>
   </nav>

.. toctree::
   :maxdepth: 2
   :hidden:

   Quickstart <quickstart>
   Workflows <workflows/index>
   Examples <examples_index>
   API <api_reference>
   Contributing <contributor_guide>
