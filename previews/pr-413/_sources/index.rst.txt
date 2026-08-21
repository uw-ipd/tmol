TMol documentation
==================

TMol is a PyTorch library with Rosetta-inspired all-atom score terms. It
scores structures on CPU or GPU, supports autograd, side-chain packing,
minimization, and ligand preparation. Weighted outputs are TMol score units,
not kcal/mol or calibrated Rosetta score units, and are not guaranteed to match
Rosetta numerically.

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

   installation
   quickstart
   learning_paths

.. toctree::
   :maxdepth: 2
   :caption: Learn

   workflows/index
   Tutorials <examples_index>
   Core architecture <architecture>
   Data model and conventions <datatypes>
   Terminology and modeling choices <terminology>

.. toctree::
   :maxdepth: 2
   :caption: Reference

   Task index <tutorial/recipe_index>
   Rosetta-to-TMol crosswalk <tutorial/rosetta_crosswalk>
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Contribute

   Contributing <contributor_guide>
