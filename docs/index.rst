tmol documentation
==================

``tmol`` is a GPU-accelerated PyTorch implementation of Rosetta's molecular
modeling energy function. It scores protein structures, propagates derivatives
through coordinates, packs side chains, minimizes structures, and prepares
small-molecule ligands for protein-ligand scoring workflows.

Start with :doc:`quickstart` for a first score calculation. The user guide
then covers installation, workflows, runnable examples, API details, and
contributor workflows.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   quickstart
   installation
   tutorial/index
   workflows/index

.. toctree::
   :maxdepth: 2
   :caption: Learn

   user_guide/scoring
   user_guide/ligands
   user_guide/optimization
   user_guide/integrations
   user_guide/benchmarking
   examples_index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
   contributor_guide
