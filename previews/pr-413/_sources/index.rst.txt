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

Start with :doc:`quickstart` for a first score calculation. Then use the
:doc:`learning paths <learning_paths>` to follow the numbered curriculum or
look up a specific operation in the :doc:`task index
<tutorial/recipe_index>`.

.. raw:: html

   <nav class="docs-card-grid docs-card-grid--three" aria-label="Choose a TMol documentation path">
     <a class="docs-card" href="learning_paths.html">
       <span class="docs-card-kicker">Learn</span>
       <span class="docs-card-title">Follow a learning path</span>
       <span class="docs-card-description">Move from structure input and batching through scoring, packing, minimization, and specialized chemistry.</span>
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
   :caption: Get started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Learn
   :hidden:

   Learning paths <learning_paths>
   Tutorials <examples_index>
   Workflows <workflows/index>

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   API reference <api_reference>
   Concepts <concepts>
   Task index <tutorial/recipe_index>
   Rosetta-to-TMol crosswalk <tutorial/rosetta_crosswalk>

.. toctree::
   :maxdepth: 2
   :caption: Contribute
   :hidden:

   Contributing <contributor_guide>
